"""
LangChain RAG Pipeline
Production-grade RAG using LangChain for orchestration, FAISS for vector search,
and configurable LLM backends for generation.
"""

import logging
from typing import List, Dict, Optional
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS as LangChainFAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import Document

logger = logging.getLogger(__name__)

FINANCIAL_QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a senior financial analyst. Answer the question using ONLY the "
        "provided context from SEC filings and earnings calls. Cite sources as [Source N]. "
        "If the context is insufficient, state that clearly.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Provide a detailed, data-driven answer with specific numbers where available:"
    ),
)

DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 128


class LangChainRAGPipeline:
    """
    Production RAG pipeline using LangChain for orchestration.
    Supports configurable chunking, multiple embedding models, and hybrid retrieval.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        top_k: int = 5,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.embedding_model_name = embedding_model

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        self.vectorstore: Optional[LangChainFAISS] = None
        self.documents: List[Document] = []

        logger.info(
            "LangChain RAG pipeline initialized: model=%s, chunk_size=%d, overlap=%d",
            embedding_model, chunk_size, chunk_overlap,
        )

    def ingest_documents(self, documents: List[Dict]) -> int:
        """
        Ingest raw documents: chunk, embed, and index.

        Args:
            documents: List of dicts with 'text' and optional metadata keys

        Returns:
            Number of chunks indexed
        """
        langchain_docs = []
        for doc in documents:
            metadata = {k: v for k, v in doc.items() if k != "text"}
            langchain_docs.append(Document(page_content=doc["text"], metadata=metadata))

        chunks = self.text_splitter.split_documents(langchain_docs)
        self.documents.extend(chunks)

        if self.vectorstore is None:
            self.vectorstore = LangChainFAISS.from_documents(chunks, self.embeddings)
        else:
            self.vectorstore.add_documents(chunks)

        logger.info(
            "Ingested %d documents -> %d chunks (total: %d)",
            len(documents), len(chunks), len(self.documents),
        )
        return len(chunks)

    def retrieve(
        self, query: str, top_k: Optional[int] = None, company_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: User question
            top_k: Override default top_k
            company_filter: Optional company name to filter results

        Returns:
            List of chunks with scores and metadata
        """
        if self.vectorstore is None:
            logger.warning("No documents indexed yet")
            return []

        k = top_k or self.top_k
        results = self.vectorstore.similarity_search_with_score(
            query, k=k * 2 if company_filter else k
        )

        retrieved = []
        for doc, score in results:
            if company_filter and doc.metadata.get("company", "").lower() != company_filter.lower():
                continue
            retrieved.append({
                "text": doc.page_content,
                "similarity_score": float(1 - score) if score > 1 else float(score),
                "rank": len(retrieved) + 1,
                **doc.metadata,
            })
            if len(retrieved) >= k:
                break

        return retrieved

    def query(
        self, question: str, top_k: Optional[int] = None, company_filter: Optional[str] = None
    ) -> Dict:
        """
        Full RAG pipeline: retrieve + generate.

        Returns:
            Dict with answer, sources, and metadata
        """
        sources = self.retrieve(question, top_k=top_k, company_filter=company_filter)

        if not sources:
            return {
                "question": question,
                "answer": "No relevant documents found. Please ingest financial data first.",
                "sources": [],
                "num_sources": 0,
                "pipeline": "langchain",
            }

        answer = self._generate_answer(question, sources)

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "num_sources": len(sources),
            "pipeline": "langchain",
        }

    def _generate_answer(self, question: str, sources: List[Dict]) -> str:
        """Generate answer from retrieved context using LLM or fallback."""
        context = self._format_context(sources)

        try:
            from langchain_openai import ChatOpenAI
            from config import OPENAI_API_KEY

            if OPENAI_API_KEY and OPENAI_API_KEY.strip() not in ("", "your_openai_api_key_here"):
                llm = ChatOpenAI(
                    model="gpt-3.5-turbo",
                    temperature=0.3,
                    max_tokens=500,
                    api_key=OPENAI_API_KEY,
                )
                prompt = FINANCIAL_QA_PROMPT.format(context=context, question=question)
                response = llm.invoke(prompt)
                return response.content
        except Exception as e:
            logger.warning("OpenAI generation failed: %s, using fallback", e)

        return self._fallback_answer(question, sources)

    def _format_context(self, sources: List[Dict]) -> str:
        parts = []
        for i, src in enumerate(sources, 1):
            meta = []
            if src.get("company"):
                meta.append(f"Company: {src['company']}")
            if src.get("quarter"):
                meta.append(f"Quarter: {src['quarter']}")
            if src.get("source"):
                meta.append(f"Source: {src['source']}")
            header = f"[Source {i}] {' | '.join(meta)}" if meta else f"[Source {i}]"
            parts.append(f"{header}\n{src['text']}")
        return "\n\n".join(parts)

    def _fallback_answer(self, question: str, sources: List[Dict]) -> str:
        """Keyword-based fallback when no LLM is available."""
        query_words = set(question.lower().split())
        scored = []

        for src in sources:
            for sentence in src["text"].split(". "):
                sentence = sentence.strip()
                if not sentence:
                    continue
                score = len(query_words & set(sentence.lower().split()))
                if score > 0:
                    scored.append((sentence, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        if scored:
            answer = "Based on the available financial data:\n\n"
            for sent, _ in scored[:3]:
                if not sent.endswith("."):
                    sent += "."
                answer += f"- {sent}\n"
            return answer

        return "Relevant documents were found but a specific answer could not be generated without an LLM."

    def save_index(self, path: str):
        """Persist FAISS index to disk."""
        if self.vectorstore:
            self.vectorstore.save_local(path)
            logger.info("Index saved to %s", path)

    def load_index(self, path: str):
        """Load FAISS index from disk."""
        if Path(path).exists():
            self.vectorstore = LangChainFAISS.load_local(
                path, self.embeddings, allow_dangerous_deserialization=True
            )
            logger.info("Index loaded from %s", path)

    @property
    def chunk_count(self) -> int:
        if self.vectorstore is None:
            return 0
        return self.vectorstore.index.ntotal

"""
Streamlit UI Module
Web interface for the AI Financial Analyst RAG system.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Dict
from datetime import datetime
import sys
import os
import json
import logging
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.rag_pipeline import FinancialRAGPipeline
from sentiment.sentiment_analyzer import FinancialSentimentAnalyzer
from data_ingestion.news_scraper import fetch_financial_news
from data_ingestion.reports_parser import parse_pdf_report, chunk_report
from data_ingestion.sec_downloader import get_sample_sec_data

logger = logging.getLogger(__name__)

ACCENT = "#00D4AA"
ACCENT_DIM = "#00D4AA30"
BG_DARK = "#0A0E14"
BG_CARD = "#111827"
BG_ELEVATED = "#1A1F2E"
BORDER = "#1E2D3D"
TEXT_PRIMARY = "#E8EDF2"
TEXT_SECONDARY = "#8899AA"
TEXT_MUTED = "#556677"
POSITIVE = "#00D4AA"
NEGATIVE = "#FF5F6D"
NEUTRAL = "#6C7A8D"
WARNING = "#F0C040"


CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}}

#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
header {{ visibility: hidden; }}

.stApp {{
    background: {BG_DARK};
}}

@keyframes fadeInUp {{
    from {{ opacity: 0; transform: translateY(12px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes pulse-glow {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0.5; }}
}}
@keyframes slideInLeft {{
    from {{ opacity: 0; transform: translateX(-16px); }}
    to {{ opacity: 1; transform: translateX(0); }}
}}

section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0D1220 0%, #111827 100%);
    border-right: 1px solid {BORDER};
}}

.stTabs [data-baseweb="tab-list"] {{
    background: transparent;
    border-bottom: 1px solid {BORDER};
    gap: 0;
    padding: 0 4px;
}}
.stTabs [data-baseweb="tab"] {{
    color: {TEXT_SECONDARY};
    border-radius: 6px 6px 0 0;
    padding: 10px 24px;
    font-size: 0.82rem;
    font-weight: 500;
    letter-spacing: 0.02em;
    transition: all 0.2s ease;
    border-bottom: 2px solid transparent;
}}
.stTabs [aria-selected="true"] {{
    color: {ACCENT} !important;
    border-bottom: 2px solid {ACCENT} !important;
    background: {ACCENT}08 !important;
}}
.stTabs [data-baseweb="tab"]:hover {{
    color: {TEXT_PRIMARY};
    background: {BG_ELEVATED}80;
}}

[data-testid="stMetric"] {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-left: 3px solid {ACCENT};
    border-radius: 8px;
    padding: 16px 20px;
    animation: fadeInUp 0.4s ease-out;
}}
[data-testid="stMetricLabel"] {{
    color: {TEXT_SECONDARY} !important;
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 500;
}}
[data-testid="stMetricValue"] {{
    color: {TEXT_PRIMARY} !important;
    font-size: 1.5rem !important;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace !important;
}}

.glass-panel {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 24px;
    animation: fadeInUp 0.5s ease-out;
}}

.answer-box {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-left: 4px solid {ACCENT};
    border-radius: 0 10px 10px 0;
    padding: 20px 24px;
    margin: 12px 0 24px 0;
    color: {TEXT_PRIMARY};
    line-height: 1.8;
    font-size: 0.92rem;
    animation: slideInLeft 0.5s ease-out;
}}

.source-chip {{
    display: inline-block;
    background: {BG_DARK};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    color: {ACCENT};
    margin: 2px 4px 2px 0;
}}

.verdict-positive {{
    background: linear-gradient(90deg, {POSITIVE}15 0%, transparent 100%);
    border-left: 4px solid {POSITIVE};
    border-radius: 0 10px 10px 0;
    padding: 16px 24px;
    display: flex; align-items: center; gap: 12px;
    margin-bottom: 16px;
    animation: slideInLeft 0.4s ease-out;
}}
.verdict-positive .verdict-text {{
    color: {POSITIVE}; font-size: 1.1rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.06em;
}}
.verdict-negative {{
    background: linear-gradient(90deg, {NEGATIVE}15 0%, transparent 100%);
    border-left: 4px solid {NEGATIVE};
    border-radius: 0 10px 10px 0;
    padding: 16px 24px;
    display: flex; align-items: center; gap: 12px;
    margin-bottom: 16px;
}}
.verdict-negative .verdict-text {{
    color: {NEGATIVE}; font-size: 1.1rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.06em;
}}
.verdict-neutral {{
    background: linear-gradient(90deg, {NEUTRAL}15 0%, transparent 100%);
    border-left: 4px solid {NEUTRAL};
    border-radius: 0 10px 10px 0;
    padding: 16px 24px;
    display: flex; align-items: center; gap: 12px;
    margin-bottom: 16px;
}}
.verdict-neutral .verdict-text {{
    color: {NEUTRAL}; font-size: 1.1rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.06em;
}}

.news-card {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 12px;
    transition: border-color 0.2s ease;
    animation: fadeInUp 0.5s ease-out;
}}
.news-card:hover {{ border-color: {ACCENT}60; }}
.news-card-title {{ font-size: 0.95rem; font-weight: 600; color: {TEXT_PRIMARY}; margin-bottom: 8px; }}
.news-card-meta {{ font-size: 0.72rem; color: {TEXT_SECONDARY}; margin-bottom: 10px; display: flex; align-items: center; gap: 8px; }}
.news-card-desc {{ font-size: 0.84rem; color: #BCC8D4; line-height: 1.65; }}
.source-badge {{
    display: inline-block; background: {ACCENT}12; border: 1px solid {ACCENT}40;
    border-radius: 4px; padding: 2px 8px; font-size: 0.65rem; color: {ACCENT};
    text-transform: uppercase; letter-spacing: 0.06em; font-weight: 500;
}}

.status-row {{
    display: flex; align-items: center; gap: 10px; padding: 10px 12px;
    border-bottom: 1px solid {BORDER}80; font-size: 0.82rem; color: {TEXT_PRIMARY};
}}
.status-row:last-child {{ border-bottom: none; }}
.dot-green  {{ width: 8px; height: 8px; border-radius: 50%; background: {POSITIVE}; flex-shrink: 0; animation: pulse-glow 2s ease-in-out infinite; }}
.dot-yellow {{ width: 8px; height: 8px; border-radius: 50%; background: {WARNING}; flex-shrink: 0; }}
.status-label {{ color: {TEXT_SECONDARY}; font-size: 0.72rem; font-family: 'JetBrains Mono', monospace; }}

.section-label {{
    font-size: 0.68rem; font-weight: 600; color: {TEXT_SECONDARY};
    text-transform: uppercase; letter-spacing: 0.14em;
    margin: 24px 0 10px 0; padding-bottom: 6px; border-bottom: 1px solid {BORDER};
}}

.hero-stat {{
    background: {BG_CARD}; border: 1px solid {BORDER}; border-radius: 12px;
    padding: 20px; text-align: center; animation: fadeInUp 0.5s ease-out;
}}
.hero-stat:hover {{ border-color: {ACCENT}50; }}
.hero-stat-value {{
    font-size: 2rem; font-weight: 800; color: {ACCENT};
    font-family: 'JetBrains Mono', monospace; line-height: 1; margin-bottom: 4px;
}}
.hero-stat-label {{ font-size: 0.68rem; color: {TEXT_SECONDARY}; text-transform: uppercase; letter-spacing: 0.1em; }}
.hero-stat-detail {{ font-size: 0.72rem; color: {TEXT_MUTED}; margin-top: 4px; }}

.pipeline {{
    display: flex; align-items: stretch; gap: 0; margin: 16px 0;
}}
.pipeline-step {{
    flex: 1; text-align: center; padding: 16px 12px;
    background: {BG_CARD}; border: 1px solid {BORDER}; position: relative;
}}
.pipeline-step:first-child {{ border-radius: 10px 0 0 10px; }}
.pipeline-step:last-child {{ border-radius: 0 10px 10px 0; }}
.pipeline-step:not(:last-child)::after {{
    content: ''; position: absolute; right: -1px; top: 50%; transform: translateY(-50%);
    width: 0; height: 0; border-top: 8px solid transparent;
    border-bottom: 8px solid transparent; border-left: 8px solid {ACCENT}40; z-index: 1;
}}
.pipeline-step-num {{ font-size: 0.6rem; color: {ACCENT}; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 6px; }}
.pipeline-step-title {{ font-size: 0.85rem; font-weight: 600; color: {TEXT_PRIMARY}; margin-bottom: 4px; }}
.pipeline-step-desc {{ font-size: 0.72rem; color: {TEXT_SECONDARY}; line-height: 1.4; }}

details {{
    border: 1px solid {BORDER} !important;
    border-left: 3px solid {BORDER} !important;
    border-radius: 0 8px 8px 0 !important;
    background: {BG_CARD} !important;
    margin-bottom: 8px !important;
}}
details[open] {{ border-left-color: {ACCENT} !important; }}
summary {{ font-size: 0.84rem !important; color: #BCC8D4 !important; font-weight: 500 !important; }}

.stButton > button {{
    border-radius: 8px; font-weight: 600; letter-spacing: 0.03em; font-size: 0.84rem;
}}
.stButton > button[kind="primary"] {{
    background: {ACCENT}; color: {BG_DARK}; border: none;
}}

.stTextInput > div > div > input,
.stTextArea > div > div > textarea {{
    background: {BG_CARD} !important; border: 1px solid {BORDER} !important;
    border-radius: 8px !important; color: {TEXT_PRIMARY} !important; font-size: 0.88rem !important;
}}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {{
    border-color: {ACCENT} !important;
}}

[data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 8px; overflow: hidden; }}

[data-testid="stFileUploader"] {{
    border: 2px dashed {BORDER} !important; border-radius: 10px !important; background: {BG_CARD} !important;
}}
[data-testid="stFileUploader"]:hover {{ border-color: {ACCENT}60 !important; }}

.app-footer {{
    margin-top: 48px; padding: 16px 0; border-top: 1px solid {BORDER}; text-align: center;
}}
.app-footer-text {{ font-size: 0.72rem; color: {TEXT_MUTED}; letter-spacing: 0.03em; }}
.app-footer-text a {{ color: {ACCENT}; text-decoration: none; }}

.app-header {{ padding: 8px 0 20px 0; animation: fadeInUp 0.4s ease-out; }}
.app-header-title {{
    font-size: 1.8rem; font-weight: 800; color: {TEXT_PRIMARY}; letter-spacing: -0.03em;
    line-height: 1.15; display: flex; align-items: center; gap: 12px;
}}
.app-header-accent {{ color: {ACCENT}; }}
.app-header-sub {{ font-size: 0.82rem; color: {TEXT_SECONDARY}; margin-top: 6px; font-weight: 400; }}
.api-badge {{
    display: inline-flex; align-items: center; gap: 6px; border-radius: 20px;
    padding: 5px 14px; font-size: 0.7rem; font-weight: 500; letter-spacing: 0.03em;
}}
.api-badge-ok {{ background: {POSITIVE}12; border: 1px solid {POSITIVE}30; color: {POSITIVE}; }}
.api-badge-warn {{ background: {WARNING}12; border: 1px solid {WARNING}30; color: {WARNING}; }}

.sidebar-brand {{ padding: 12px 0 20px 0; border-bottom: 1px solid {BORDER}; margin-bottom: 16px; }}
.sidebar-brand-name {{ font-size: 0.95rem; font-weight: 700; color: {TEXT_PRIMARY}; display: flex; align-items: center; gap: 8px; }}
.sidebar-brand-version {{ font-size: 0.62rem; color: {TEXT_MUTED}; font-family: 'JetBrains Mono', monospace; margin-top: 2px; }}
.sidebar-section-title {{
    font-size: 0.62rem; font-weight: 600; color: {ACCENT}; text-transform: uppercase;
    letter-spacing: 0.14em; margin: 20px 0 8px 0;
}}

.prob-bar-container {{ display: flex; align-items: center; gap: 10px; margin-bottom: 8px; animation: slideInLeft 0.5s ease-out; }}
.prob-bar-label {{ font-size: 0.75rem; color: {TEXT_SECONDARY}; text-transform: capitalize; width: 70px; text-align: right; font-weight: 500; }}
.prob-bar-track {{ flex: 1; height: 10px; background: {BG_DARK}; border-radius: 5px; overflow: hidden; border: 1px solid {BORDER}; }}
.prob-bar-fill {{ height: 100%; border-radius: 5px; }}
.prob-bar-value {{ font-size: 0.72rem; color: {TEXT_PRIMARY}; font-family: 'JetBrains Mono', monospace; width: 48px; font-weight: 500; }}

.explainability-card {{
    background: {BG_ELEVATED}; border: 1px solid {BORDER}; border-radius: 8px;
    padding: 12px 16px; margin-bottom: 8px;
}}
.explainability-score {{
    font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: {ACCENT}; font-weight: 600;
}}
</style>
"""


def initialize_session_state():
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = None
    if 'sentiment_analyzer' not in st.session_state:
        st.session_state.sentiment_analyzer = None
    if 'sample_data_loaded' not in st.session_state:
        st.session_state.sample_data_loaded = False
    if 'hybrid_retriever' not in st.session_state:
        st.session_state.hybrid_retriever = None
    if 'reranker' not in st.session_state:
        st.session_state.reranker = None


def load_sample_data():
    if st.session_state.sample_data_loaded:
        return
    st.session_state.rag_pipeline = FinancialRAGPipeline()
    st.session_state.sentiment_analyzer = FinancialSentimentAnalyzer()

    sample_chunks = get_sample_sec_data()
    embedded = st.session_state.rag_pipeline.embedder.embed_document_chunks(sample_chunks)
    st.session_state.rag_pipeline.build_index(embedded)

    try:
        from retrieval.hybrid_retriever import HybridRetriever
        st.session_state.hybrid_retriever = HybridRetriever(alpha=0.7)
        st.session_state.hybrid_retriever.index_documents(sample_chunks)
    except Exception as e:
        logger.warning("Hybrid retriever init failed: %s", e)

    try:
        from retrieval.reranker import CrossEncoderReranker
        st.session_state.reranker = CrossEncoderReranker()
    except Exception as e:
        logger.warning("Reranker init failed: %s", e)

    st.session_state.sample_data_loaded = True


def _get_index_stats():
    rag = st.session_state.rag_pipeline
    if not rag or not rag.chunks:
        return 0, [], [], []
    chunks = rag.chunks
    companies = sorted(set(c.get('company', '?') for c in chunks))
    quarters = sorted(set(c.get('quarter', '?') for c in chunks))
    sources = sorted(set(c.get('source', '?') for c in chunks))
    return len(chunks), companies, quarters, sources


def _sentiment_verdict(sentiment: str) -> str:
    s = sentiment.lower()
    icon_map = {'positive': ('verdict-positive', '+'), 'negative': ('verdict-negative', '-'), 'neutral': ('verdict-neutral', '~')}
    css_class, icon = icon_map.get(s, ('verdict-neutral', '~'))
    return f'<div class="{css_class}"><div class="verdict-text">{s}</div></div>'


def _prob_bar_html(label: str, value: float, color: str) -> str:
    pct = max(0, min(100, value * 100))
    return f'''<div class="prob-bar-container">
  <div class="prob-bar-label">{label}</div>
  <div class="prob-bar-track"><div class="prob-bar-fill" style="width:{pct}%;background:{color};"></div></div>
  <div class="prob-bar-value">{pct:.1f}%</div>
</div>'''


def _make_gauge_chart(value: float, title: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        number={'suffix': '%', 'font': {'size': 28, 'color': TEXT_PRIMARY, 'family': 'JetBrains Mono'}},
        title={'text': title, 'font': {'size': 13, 'color': TEXT_SECONDARY}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 0, 'dtick': 25, 'tickfont': {'size': 9, 'color': TEXT_MUTED}},
            'bar': {'color': ACCENT, 'thickness': 0.3},
            'bgcolor': BG_CARD,
            'borderwidth': 1, 'bordercolor': BORDER,
            'steps': [
                {'range': [0, 33], 'color': f'{NEGATIVE}18'},
                {'range': [33, 66], 'color': f'{NEUTRAL}18'},
                {'range': [66, 100], 'color': f'{POSITIVE}18'},
            ],
        }
    ))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': TEXT_PRIMARY})
    return fig


def _make_prob_chart(probs: dict) -> go.Figure:
    labels = [k.title() for k in probs.keys()]
    values = list(probs.values())
    colors = [POSITIVE if 'pos' in k else NEGATIVE if 'neg' in k else NEUTRAL for k in probs.keys()]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation='h',
        marker=dict(color=colors, line=dict(width=0)),
        text=[f'{v:.1%}' for v in values], textposition='outside',
        textfont=dict(size=12, color=TEXT_PRIMARY, family='JetBrains Mono'),
    ))
    fig.update_layout(height=160, margin=dict(l=0, r=40, t=8, b=8), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      xaxis=dict(range=[0, 1.15], showgrid=False, showticklabels=False, zeroline=False),
                      yaxis=dict(showgrid=False, tickfont=dict(size=12, color=TEXT_SECONDARY)), bargap=0.35)
    return fig


def _render_news_card(article: dict, idx: int):
    source = article.get('source', 'Unknown')
    published = article.get('publishedAt', '')
    if published and 'T' in published:
        published = published.split('T')[0]
    title = article.get('title', f'Article {idx}')
    desc = article.get('description', '')
    st.markdown(f"""<div class="news-card">
  <div class="news-card-title">{title}</div>
  <div class="news-card-meta"><span class="source-badge">{source}</span><span>{published}</span></div>
  <div class="news-card-desc">{desc}</div>
</div>""", unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="FinRAG AI - Financial Analyst", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    initialize_session_state()
    load_sample_data()

    from config import OPENAI_API_KEY, NEWS_API_KEY
    openai_ok = bool(OPENAI_API_KEY and OPENAI_API_KEY.strip() not in ("", "your_openai_api_key_here", "OPENAI_API_KEY"))
    newsapi_ok = bool(NEWS_API_KEY and NEWS_API_KEY.strip() not in ("", "your_newsapi_key_here", "NEWS_API_KEY"))

    chunk_count, companies, quarters, sources = _get_index_stats()

    # Header
    h1, h2 = st.columns([3, 1])
    with h1:
        st.markdown(f'''<div class="app-header">
  <div class="app-header-title">
    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="{ACCENT}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
    <span>FinRAG <span class="app-header-accent">AI</span></span>
  </div>
  <div class="app-header-sub">LangChain RAG &middot; Hybrid Retrieval &middot; FinBERT Sentiment &middot; SEC Filings</div>
</div>''', unsafe_allow_html=True)
    with h2:
        badges = []
        for label, ok in [("OpenAI", openai_ok), ("NewsAPI", newsapi_ok)]:
            cls = "api-badge api-badge-ok" if ok else "api-badge api-badge-warn"
            badges.append(f'<span class="{cls}">{label}</span>')
        st.markdown(f'<div style="display:flex;justify-content:flex-end;gap:8px;padding-top:16px;">{"".join(badges)}</div>', unsafe_allow_html=True)

    # Hero stats
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="hero-stat"><div class="hero-stat-value">{chunk_count}</div><div class="hero-stat-label">Indexed Chunks</div><div class="hero-stat-detail">SEC filings + earnings</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="hero-stat"><div class="hero-stat-value">{len(companies)}</div><div class="hero-stat-label">Companies</div><div class="hero-stat-detail">{", ".join(companies[:3])}</div></div>', unsafe_allow_html=True)
    with c3:
        mode = "Hybrid" if st.session_state.hybrid_retriever else "Dense"
        st.markdown(f'<div class="hero-stat"><div class="hero-stat-value">{mode}</div><div class="hero-stat-label">Retrieval</div><div class="hero-stat-detail">BM25 + FAISS</div></div>', unsafe_allow_html=True)
    with c4:
        rerank = "On" if st.session_state.reranker and st.session_state.reranker.is_available else "Off"
        st.markdown(f'<div class="hero-stat"><div class="hero-stat-value">{rerank}</div><div class="hero-stat-label">Reranking</div><div class="hero-stat-detail">Cross-encoder</div></div>', unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown(f'''<div class="sidebar-brand">
  <div class="sidebar-brand-name">
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="{ACCENT}" stroke-width="2.5"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
    FinRAG AI
  </div>
  <div class="sidebar-brand-version">v2.0 &middot; LangChain + Hybrid RAG</div>
</div>''', unsafe_allow_html=True)

        st.markdown(f'<div class="sidebar-section-title">Pipeline Config</div>', unsafe_allow_html=True)
        retrieval_mode = st.selectbox("Retrieval Mode", ["hybrid", "dense", "sparse"], index=0, key="retrieval_mode")
        top_k = st.slider("Top-K Results", 1, 10, 5, key="top_k")
        rerank_enabled = st.checkbox("Enable Reranking", value=True, key="rerank_enabled")

        st.markdown(f'<div class="sidebar-section-title">System Status</div>', unsafe_allow_html=True)
        idx_dot = 'dot-green' if chunk_count > 0 else 'dot-yellow'
        oai_dot = 'dot-green' if openai_ok else 'dot-yellow'
        hybrid_dot = 'dot-green' if st.session_state.hybrid_retriever else 'dot-yellow'
        rerank_dot = 'dot-green' if (st.session_state.reranker and st.session_state.reranker.is_available) else 'dot-yellow'

        st.markdown(f'''
<div class="status-row"><div class="{idx_dot}"></div><span>FAISS Index</span><span class="status-label" style="margin-left:auto">{chunk_count} chunks</span></div>
<div class="status-row"><div class="{hybrid_dot}"></div><span>BM25 Index</span><span class="status-label" style="margin-left:auto">{"active" if st.session_state.hybrid_retriever else "off"}</span></div>
<div class="status-row"><div class="{rerank_dot}"></div><span>Cross-Encoder</span><span class="status-label" style="margin-left:auto">{"loaded" if (st.session_state.reranker and st.session_state.reranker.is_available) else "off"}</span></div>
<div class="status-row"><div class="{oai_dot}"></div><span>OpenAI</span><span class="status-label" style="margin-left:auto">{"active" if openai_ok else "fallback"}</span></div>
''', unsafe_allow_html=True)

        st.markdown(f'<div class="sidebar-section-title">Coverage</div>', unsafe_allow_html=True)
        if companies:
            st.markdown(" ".join(f'<span class="source-chip">{c}</span>' for c in companies), unsafe_allow_html=True)

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["  Query  ", "  News  ", "  Sentiment  ", "  Data Sources  ", "  Evaluation  "])

    # TAB 1 — QUERY
    with tab1:
        st.markdown('<div class="section-label">Ask anything about financial data (SEC filings, earnings)</div>', unsafe_allow_html=True)

        q_col1, q_col2 = st.columns([6, 1])
        with q_col1:
            query = st.text_input("", placeholder="What was Apple's total revenue in fiscal 2023?", label_visibility="collapsed")
        with q_col2:
            search_clicked = st.button("Search", type="primary", use_container_width=True, key="search_btn")

        if search_clicked and query:
            import time
            start_time = time.time()

            with st.spinner("Retrieving and generating answer..."):
                rag = st.session_state.rag_pipeline
                retrieval_mode = st.session_state.get("retrieval_mode", "hybrid")
                top_k_val = st.session_state.get("top_k", 5)
                rerank_on = st.session_state.get("rerank_enabled", True)

                # Step 1: Dense retrieval
                dense_results = rag.retrieve_relevant_chunks(query, top_k=top_k_val * 4)

                # Step 2: Hybrid retrieval
                if retrieval_mode == "hybrid" and st.session_state.hybrid_retriever:
                    results = st.session_state.hybrid_retriever.hybrid_search(query, dense_results, top_k=top_k_val * 2)
                elif retrieval_mode == "sparse" and st.session_state.hybrid_retriever:
                    results = st.session_state.hybrid_retriever.retrieve_bm25(query, top_k=top_k_val * 2)
                else:
                    results = dense_results[:top_k_val * 2]

                # Step 3: Reranking
                if rerank_on and st.session_state.reranker and st.session_state.reranker.is_available:
                    results = st.session_state.reranker.rerank(query, results, top_k=top_k_val)
                else:
                    results = results[:top_k_val]

                # Step 4: Generate answer
                answer = rag.generate_answer(query, results)
                latency = time.time() - start_time

                # Display answer
                st.markdown('<div class="section-label" style="margin-top:16px;">Answer</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

                # Retrieval explainability
                if results:
                    st.markdown('<div class="section-label">Retrieval Explainability</div>', unsafe_allow_html=True)
                    st.caption(f"Mode: **{retrieval_mode}** | Top-K: **{top_k_val}** | Rerank: **{'on' if rerank_on else 'off'}** | Latency: **{latency*1000:.0f}ms**")

                    rows = []
                    for src in results:
                        row = {
                            'Rank': src.get('rank', '-'),
                            'Company': src.get('company', '?'),
                            'Source': src.get('source', '?'),
                        }
                        if 'similarity_score' in src:
                            row['Dense Score'] = round(src['similarity_score'], 3)
                        if 'bm25_rrf' in src:
                            row['BM25 RRF'] = round(src['bm25_rrf'], 3)
                        if 'hybrid_score' in src:
                            row['Hybrid Score'] = round(src['hybrid_score'], 3)
                        if 'rerank_score' in src:
                            row['Rerank Score'] = round(src['rerank_score'], 3)
                        row['Excerpt'] = src['text'][:120] + '...'
                        rows.append(row)

                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                # Answer sentiment
                st.markdown('<div class="section-label" style="margin-top:16px;">Answer Sentiment</div>', unsafe_allow_html=True)
                sent = st.session_state.sentiment_analyzer.analyze_sentiment(answer)
                st.markdown(_sentiment_verdict(sent['sentiment']), unsafe_allow_html=True)
                sc1, sc2 = st.columns([1, 2])
                with sc1:
                    st.plotly_chart(_make_gauge_chart(sent['confidence'], 'Confidence'), use_container_width=True, config={'displayModeBar': False})
                with sc2:
                    st.plotly_chart(_make_prob_chart(sent['class_probabilities']), use_container_width=True, config={'displayModeBar': False})

        elif search_clicked:
            st.warning("Please enter a question.")

    # TAB 2 — NEWS
    with tab2:
        st.markdown('<div class="section-label">Live Financial News</div>', unsafe_allow_html=True)
        nc1, nc2 = st.columns([5, 1])
        with nc1:
            news_query = st.text_input("", placeholder="e.g., Apple earnings, Tesla deliveries...", label_visibility="collapsed", key="news_q")
        with nc2:
            fetch_clicked = st.button("Fetch", type="primary", use_container_width=True, key="fetch_btn")

        if fetch_clicked and news_query:
            with st.spinner("Fetching articles..."):
                articles = fetch_financial_news(news_query, page_size=10)
                if articles:
                    st.markdown(f'<div style="font-size:0.78rem;color:{ACCENT};margin:8px 0 16px 0;font-weight:500;">{len(articles)} articles found</div>', unsafe_allow_html=True)
                    left, right = st.columns(2)
                    for i, art in enumerate(articles):
                        with (left if i % 2 == 0 else right):
                            _render_news_card(art, i + 1)
                else:
                    st.warning("No articles found. Check your NewsAPI key.")

        st.markdown('<div class="section-label" style="margin-top:32px;">Sample Articles</div>', unsafe_allow_html=True)
        sample_news = [
            {'title': 'Apple Reports FY2023 Results: Services Growth Offsets iPhone Decline', 'source': 'SEC Filing', 'publishedAt': '2023-11-03', 'description': 'Apple reported total revenue of $383.3B for FY2023. Services segment grew 9% to $85.2B while iPhone revenue declined.'},
            {'title': 'Tesla 2023 Annual Report: 1.85M Vehicles Produced', 'source': 'SEC Filing', 'publishedAt': '2024-01-29', 'description': 'Tesla produced 1.85M vehicles in 2023 (+35% YoY) with automotive revenue of $82.4B. Gross margin compressed to 18.2%.'},
            {'title': 'Microsoft Cloud Surpasses $118B in Annual Revenue', 'source': 'SEC Filing', 'publishedAt': '2023-07-27', 'description': 'Microsoft Cloud revenue reached $118.5B (+22% YoY) with Azure growing 29%. Intelligent Cloud segment at $87.9B.'},
            {'title': 'Amazon 2023: AWS at $90.8B, Free Cash Flow Rebounds', 'source': 'SEC Filing', 'publishedAt': '2024-02-01', 'description': 'Amazon net sales grew 12% to $574.8B. AWS revenue $90.8B (+13%). Free cash flow improved from -$11.6B to +$36.8B.'},
        ]
        sl, sr = st.columns(2)
        for i, art in enumerate(sample_news):
            with (sl if i % 2 == 0 else sr):
                _render_news_card(art, i + 1)

    # TAB 3 — SENTIMENT
    with tab3:
        st.markdown('<div class="section-label">FinBERT Sentiment Analysis</div>', unsafe_allow_html=True)
        sentiment_text = st.text_area("", placeholder="Paste any financial text here...", height=120, label_visibility="collapsed")

        if st.button("Analyze", type="primary", key="analyze_btn"):
            if sentiment_text:
                with st.spinner("Running FinBERT..."):
                    result = st.session_state.sentiment_analyzer.analyze_sentiment(sentiment_text)
                    st.markdown(_sentiment_verdict(result['sentiment']), unsafe_allow_html=True)
                    gc, pc = st.columns([1, 2])
                    with gc:
                        st.plotly_chart(_make_gauge_chart(result['confidence'], 'Confidence'), use_container_width=True, config={'displayModeBar': False})
                    with pc:
                        st.markdown(f'<div class="section-label" style="margin-top:0;">Class Probabilities</div>', unsafe_allow_html=True)
                        probs = result['class_probabilities']
                        color_map = {'positive': POSITIVE, 'negative': NEGATIVE, 'neutral': NEUTRAL}
                        for cls, prob in probs.items():
                            st.markdown(_prob_bar_html(cls, prob, color_map.get(cls, NEUTRAL)), unsafe_allow_html=True)
                    mc1, mc2 = st.columns(2)
                    with mc1:
                        st.metric("Model", result['model_used'])
                    with mc2:
                        st.metric("Sentiment", result['sentiment'].title())
            else:
                st.warning("Please enter text to analyze.")

        st.markdown('<div class="section-label" style="margin-top:36px;">Quick Samples</div>', unsafe_allow_html=True)
        samples = [
            ("Apple's Services revenue grew 9% to $85.2 billion, driven by App Store and cloud growth.", "positive"),
            ("Tesla's gross margin declined from 25.6% to 18.2% due to aggressive price cuts.", "negative"),
            ("The Federal Reserve maintained rates at 5.25-5.50% while monitoring inflation data.", "neutral"),
        ]
        for text, expected in samples:
            with st.expander(f'{text[:75]}...'):
                if st.button("Analyze", key=f"sa_{hash(text)}"):
                    r = st.session_state.sentiment_analyzer.analyze_sentiment(text)
                    st.markdown(_sentiment_verdict(r['sentiment']), unsafe_allow_html=True)
                    st.plotly_chart(_make_prob_chart(r['class_probabilities']), use_container_width=True, config={'displayModeBar': False})

    # TAB 4 — DATA SOURCES
    with tab4:
        st.markdown('<div class="section-label">RAG Pipeline Architecture</div>', unsafe_allow_html=True)
        st.markdown(f'''<div class="pipeline">
  <div class="pipeline-step"><div class="pipeline-step-num">Step 1</div><div class="pipeline-step-title">Ingest</div><div class="pipeline-step-desc">SEC EDGAR, PDFs, earnings calls</div></div>
  <div class="pipeline-step"><div class="pipeline-step-num">Step 2</div><div class="pipeline-step-title">Chunk</div><div class="pipeline-step-desc">RecursiveCharacterTextSplitter (512/128)</div></div>
  <div class="pipeline-step"><div class="pipeline-step-num">Step 3</div><div class="pipeline-step-title">Embed</div><div class="pipeline-step-desc">Sentence-transformers + BM25</div></div>
  <div class="pipeline-step"><div class="pipeline-step-num">Step 4</div><div class="pipeline-step-title">Retrieve</div><div class="pipeline-step-desc">Hybrid search + reranking</div></div>
  <div class="pipeline-step"><div class="pipeline-step-num">Step 5</div><div class="pipeline-step-title">Generate</div><div class="pipeline-step-desc">LLM + FinBERT sentiment</div></div>
</div>''', unsafe_allow_html=True)

        st.markdown('<div class="section-label" style="margin-top:28px;">Index Statistics</div>', unsafe_allow_html=True)
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.metric("Total Chunks", chunk_count)
        with d2:
            st.metric("Companies", len(companies))
        with d3:
            st.metric("Quarters", len(quarters))
        with d4:
            st.metric("Source Types", len(sources))

        st.markdown('<div class="section-label" style="margin-top:28px;">Data Sources</div>', unsafe_allow_html=True)
        st.markdown(f'''
<div class="status-row"><div class="dot-green"></div><span>SEC Filings (10-K)</span><span class="status-label" style="margin-left:auto">Apple, Tesla, Microsoft, Amazon</span></div>
<div class="status-row"><div class="dot-green"></div><span>FOMC Statements</span><span class="status-label" style="margin-left:auto">Q4 2023</span></div>
<div class="status-row"><div class="dot-yellow"></div><span>Financial Reports (PDF)</span><span class="status-label" style="margin-left:auto">Upload below</span></div>
<div class="status-row"><div class="{"dot-green" if newsapi_ok else "dot-yellow"}"></div><span>News Articles</span><span class="status-label" style="margin-left:auto">{"Active" if newsapi_ok else "API key required"}</span></div>
''', unsafe_allow_html=True)

        st.markdown('<div class="section-label" style="margin-top:28px;">Upload Financial Reports</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Choose PDF files (10-K, 10-Q, annual reports)", type=['pdf'], accept_multiple_files=True)

        if uploaded:
            if st.button("Parse & Index PDFs", type="primary", key="parse_btn"):
                import tempfile
                new_chunks = []
                for f in uploaded:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(f.read())
                        tmp_path = tmp.name
                    try:
                        data = parse_pdf_report(tmp_path)
                        if data:
                            data['file_path'] = f.name
                            chunks = chunk_report(data)
                            new_chunks.extend(chunks)
                            st.success(f"Parsed **{f.name}**: {len(chunks)} chunks")
                    finally:
                        os.remove(tmp_path)
                if new_chunks:
                    rag = st.session_state.rag_pipeline
                    emb = rag.embedder.embed_document_chunks(new_chunks)
                    all_emb = list(rag.chunks) + emb
                    rag.build_index(all_emb)
                    if st.session_state.hybrid_retriever:
                        st.session_state.hybrid_retriever.index_documents([c for c in all_emb])
                    st.success(f"Index rebuilt with **{len(all_emb)}** total chunks")

    # TAB 5 — EVALUATION
    with tab5:
        st.markdown('<div class="section-label">RAG Evaluation Dashboard</div>', unsafe_allow_html=True)

        st.markdown(f"""
This tab shows evaluation results from running the RAG pipeline against a golden Q&A set of 25 financial questions.
Metrics follow the RAGAS framework: **Context Recall**, **Context Precision**, **Faithfulness**, and **Answer Relevance**.
""")

        results_dir = Path(__file__).parent.parent / "evaluation" / "results"
        results_files = sorted(results_dir.glob("eval_*.json"), reverse=True) if results_dir.exists() else []

        if results_files:
            latest = results_files[0]
            with open(latest, "r") as f:
                eval_data = json.load(f)

            st.markdown(f'<div class="section-label">Latest Results ({latest.stem})</div>', unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Context Recall", f"{eval_data.get('avg_context_recall', 0):.1%}")
            with m2:
                st.metric("Context Precision", f"{eval_data.get('avg_context_precision', 0):.1%}")
            with m3:
                st.metric("Faithfulness", f"{eval_data.get('avg_faithfulness', 0):.1%}")
            with m4:
                st.metric("Answer Relevance", f"{eval_data.get('avg_answer_relevance', 0):.1%}")

            if "per_question" in eval_data:
                st.markdown('<div class="section-label" style="margin-top:20px;">Per-Question Breakdown</div>', unsafe_allow_html=True)
                df = pd.DataFrame(eval_data["per_question"])
                display_cols = [c for c in ["question", "context_recall", "context_precision", "faithfulness", "answer_relevance", "latency_ms"] if c in df.columns]
                st.dataframe(df[display_cols], use_container_width=True, hide_index=True)

        else:
            st.info("No evaluation results yet. Run `make evaluate` to generate metrics.")

        if st.button("Run Evaluation Now", type="primary", key="run_eval"):
            with st.spinner("Running evaluation (this may take a minute)..."):
                try:
                    from evaluation.rag_evaluator import run_evaluation
                    results = run_evaluation()
                    st.success(f"Evaluation complete! Context Recall: {results['avg_context_recall']:.1%}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Evaluation failed: {e}")

    # Footer
    st.markdown(f'''<div class="app-footer">
  <div class="app-footer-text">
    FinRAG AI v2.0 &nbsp;&middot;&nbsp; LangChain &nbsp;&middot;&nbsp; FAISS &nbsp;&middot;&nbsp; FinBERT &nbsp;&middot;&nbsp; Hybrid Retrieval
    &nbsp;&middot;&nbsp; <a href="https://github.com/krishi-shah/FinRAG-AI">GitHub</a>
  </div>
</div>''', unsafe_allow_html=True)


if __name__ == "__main__":
    main()

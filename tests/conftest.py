<<<<<<< HEAD
"""Pytest configuration."""
=======
"""Shared fixtures for all tests."""
>>>>>>> 7d7bc625fee4bf9d4c70c4ee0ef89f65a02aa30c

import sys
from pathlib import Path

<<<<<<< HEAD
=======
# Ensure the project root is on sys.path so `from embeddings…` etc. work.
>>>>>>> 7d7bc625fee4bf9d4c70c4ee0ef89f65a02aa30c
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

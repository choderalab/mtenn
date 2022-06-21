"""
Unit and regression test for the mtenn package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import mtenn


def test_mtenn_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "mtenn" in sys.modules

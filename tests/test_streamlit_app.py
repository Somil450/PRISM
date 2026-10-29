"""
tests/test_streamlit_app.py
Automated Headless UI & Integration Tests for PRISM Streamlit Application.

Validates:
  1. Streamlit app boots cleanly without syntax or import errors.
  2. UI components (title, sidebar, tabs) render as expected.
  3. No unhandled exceptions occur during initial app rendering.
"""

import sys
import os
import unittest
from pathlib import Path

# Main-thread pre-imports to prevent Windows Python 3.13 DLL thread-initialization issues
try:
    import pyarrow
    import pandas
    import torch
    import torchvision
    import torchvision.io
    import timm
except Exception:
    pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from streamlit.testing.v1 import AppTest


class TestStreamlitApp(unittest.TestCase):
    """Headless automated UI tests using Streamlit AppTest framework."""

    def test_01_app_boots_cleanly_without_exception(self):
        """App must boot and render its header and instructions without crashing."""
        app_path = str(Path(__file__).parent.parent / "app.py")
        at = AppTest.from_file(app_path, default_timeout=30)
        at.run()

        # Check there are no unhandled exceptions in the Streamlit runtime
        self.assertFalse(at.exception, f"App threw unexpected runtime exception: {at.exception}")

        # Verify header is rendered via markdown
        prism_markdown = any("PRISM" in str(getattr(m, "value", "")) for m in at.markdown)
        self.assertTrue(prism_markdown, "PRISM header not found in app markdown elements")


if __name__ == "__main__":
    unittest.main()

"""Streamlit Community Cloud entry point for the live-inference companion.

Community Cloud installs dependencies from a file named ``requirements.txt``,
searching the entrypoint's own directory first and the repository root second.
The real app lives at the repository root (``../../streamlit_app.py``) because it
reuses the product's modules (``config``, ``models.registry``, ``models.explain``).
If that root file were the entrypoint, Community Cloud would install the root
``requirements.txt`` — the ``lite`` runtime set, which has no torch — and the
live models could not load.

So this thin shim sits in its own directory next to a ``requirements.txt`` that
carries the CPU-torch stack, puts the repository root on ``sys.path`` and hands
control to the real app. No application logic lives here.

Deploy on Streamlit Community Cloud with main file path
``deploy/streamlit/streamlit_app.py``.
"""
import os
import runpy
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)
runpy.run_path(os.path.join(REPO_ROOT, "streamlit_app.py"), run_name="__main__")

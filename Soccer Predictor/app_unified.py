"""
Thin launcher so ``streamlit run app_unified.py`` boots the unified app.

Streamlit runs THIS file directly, so all we do is put ``src`` on the import path
and hand control to :func:`soccer_predictor.apps.streamlit_app.main`. The real
UI lives in the package; this file deliberately contains no app logic. (The
legacy ``app.py`` is left untouched.)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Importing the module runs its bottom guard (`if _running_under_streamlit(): main()`)
# exactly once under a Streamlit runtime. We must NOT also call main() here, or the
# app renders twice and Streamlit raises DuplicateElementId on the widgets.
import soccer_predictor.apps.streamlit_app  # noqa: E402,F401

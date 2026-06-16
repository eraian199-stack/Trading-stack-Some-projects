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

# This file is the Streamlit entry point: it is re-executed on every rerun, so it
# calls main() explicitly each time. streamlit_app no longer auto-runs main() on
# import, so main() runs exactly once per render (no DuplicateElementId).
from soccer_predictor.apps.streamlit_app import main  # noqa: E402

main()

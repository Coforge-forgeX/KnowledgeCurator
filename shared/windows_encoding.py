"""
Windows Console UTF-8 Encoding Setup
=====================================

PURPOSE:
--------
Fixes Windows console encoding crashes when outputting Unicode characters.
Windows defaults to cp1252/cp437 encoding instead of UTF-8, causing crashes
when code outputs:
- Progress spinners (⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏)
- Box-drawing characters (│ ─ ┌ └ ├ ┤)
- Emoji or special Unicode characters
- LightRAG's rich console output

WITHOUT THIS FIX:
-----------------
You'll see errors like:
    UnicodeEncodeError: 'charmap' codec can't encode character '│'
    in position 42: character maps to <undefined>

WITH THIS FIX:
--------------
- Sets PYTHONIOENCODING=utf-8 environment variable
- Reconfigures stdout/stderr to use UTF-8 encoding
- Uses 'replace' error handling to avoid crashes on unsupported characters

USAGE:
------
Import and call at the top of your main.py (application entry point):

    from shared.windows_encoding import configure_windows_console_encoding

    configure_windows_console_encoding()

    # Rest of your application code...

IMPACT IF NOT USED:
-------------------
- ✅ On Linux/Mac: No impact (function returns immediately)
- ❌ On Windows without fix: Crashes when outputting Unicode characters
- ✅ On Windows with fix: Unicode output works correctly

BREAKING:
---------
- NO - Safe to call multiple times (idempotent)
- NO - Does nothing on non-Windows platforms
- NO - Won't break existing functionality
- YES - Removing this WILL break Unicode output on Windows
"""
import os
import sys


def configure_windows_console_encoding() -> None:
    """
    Configure Windows console for UTF-8 encoding to prevent Unicode crashes.

    Safe to call multiple times. Does nothing on non-Windows platforms.
    """
    # Skip if not Windows
    if os.name != "nt":
        return

    # Set environment variable for child processes
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    # Reconfigure stdout and stderr for UTF-8 encoding
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue

        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                # Keep startup resilient if reconfiguration fails
                # (e.g., redirected streams, older Python versions)
                pass

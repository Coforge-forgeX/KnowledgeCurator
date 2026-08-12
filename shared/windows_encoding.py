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


class _SafeEncodingStream:
    """
    Proxy around a text stream that can't be reconfigured to UTF-8.

    Some hosts (e.g. the Azure Functions Python worker) replace sys.stdout /
    sys.stderr with their own wrapper objects that have no ``reconfigure()``
    and still encode with the legacy console codepage (cp1252). Writing a
    spinner glyph to such a stream raises UnicodeEncodeError - and when the
    write happens on a library's background render thread (ascii_colors /
    rich Live), that kills the thread with a traceback in the logs.

    This proxy retries the write with unsupported characters replaced instead
    of letting the exception escape.
    """

    __slots__ = ("_stream",)

    def __init__(self, stream) -> None:
        self._stream = stream

    def write(self, text):
        try:
            return self._stream.write(text)
        except UnicodeEncodeError:
            encoding = getattr(self._stream, "encoding", None) or "ascii"
            safe = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
            return self._stream.write(safe)

    def writelines(self, lines) -> None:
        for line in lines:
            self.write(line)

    def __getattr__(self, name):
        # Delegate everything else (flush, isatty, fileno, encoding, buffer, ...)
        return getattr(self._stream, name)


def _needs_wrapping(stream) -> bool:
    encoding = (getattr(stream, "encoding", None) or "").lower().replace("-", "")
    return encoding not in ("utf8", "utf8sig")


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
        if stream is None or isinstance(stream, _SafeEncodingStream):
            continue

        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                # Keep startup resilient if reconfiguration fails
                # (e.g., redirected streams, older Python versions)
                pass

        # Streams that don't support reconfigure() (or where it silently had no
        # effect) still encode with the legacy codepage - wrap them so a stray
        # Unicode glyph degrades to '?' instead of raising.
        if _needs_wrapping(stream):
            try:
                setattr(sys, stream_name, _SafeEncodingStream(stream))
            except Exception:
                pass

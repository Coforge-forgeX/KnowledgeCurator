"""Progress reporting abstraction (ports + adapters).

This package keeps long-running API progress transport-agnostic so the same
business flow can publish progress to WebSockets, event buses, or no-op sinks.
"""

from .factory import create_progress_reporter
from .models import ProgressEvent
from .reporter import ProgressReporter

__all__ = ["ProgressEvent", "ProgressReporter", "create_progress_reporter"]

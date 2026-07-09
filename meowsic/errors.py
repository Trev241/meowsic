class MeowsicError(Exception):
    """Base exception for actionable Meowsic failures."""


class OptionalDependencyError(MeowsicError):
    """Raised when an optional backend is requested but unavailable."""


class SourceFetchError(MeowsicError):
    """Raised when source ingestion fails."""


class StemSeparationError(MeowsicError):
    """Raised when stem preparation fails."""


class SampleFetchError(MeowsicError):
    """Raised when meow sample fetching or validation fails."""


class AudioFormatError(MeowsicError):
    """Raised for unsupported or invalid audio files."""

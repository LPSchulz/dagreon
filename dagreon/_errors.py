class ValidationError(Exception):
    """Base exception for workflow validation errors."""

    pass


class DuplicateOutputError(ValidationError):
    """Raised when multiple tasks produce the same output type."""

    pass


class MissingProducerError(ValidationError):
    """Raised when a required input type has no producer."""

    pass


class CycleError(ValidationError):
    """Raised when the workflow contains a cycle."""

    pass

"""Domain-specific exceptions for AquaTrade MLOps."""


class AquaTradeException(Exception):
    """Base exception for all AquaTrade domain errors."""
    pass


class DomainValidationError(AquaTradeException):
    """Raised when domain entities or values violate business invariants."""
    pass


class DataQualityError(AquaTradeException):
    """Raised when ingested or processed dataset fails quality thresholds."""
    pass


class InsufficientDataError(AquaTradeException):
    """Raised when a time-series does not have enough observations for forecasting."""
    pass


class ModelTrainingError(AquaTradeException):
    """Raised when model training or evaluation fails."""
    pass


class ModelInferenceError(AquaTradeException):
    """Raised when model prediction fails."""
    pass


class RepositoryError(AquaTradeException):
    """Raised when persistence or analytical queries fail."""
    pass

class SimpleTrackException(Exception):
    """
    Error arising from incorrect code or data in Simple-Track
    """


class ConfigError(SimpleTrackException):
    """
    Error thrown when one or more config input parameters are not valid
    """


class IDError(SimpleTrackException):
    """Exception raised when input is not a valid ID"""


class ZeroIDError(IDError):
    """Exception raised when input is not a valid ID because value is 0"""


class NegativeIDError(IDError):
    """Exception raised when input is not a valid ID because of negative value"""


class FloatIDError(IDError):
    """Exception raised when input is not a valid ID because of float value"""


class ArrayError(SimpleTrackException):
    """
    Exception raised when input is not a valid array or cannot be converted to a valid
    array with the required constraints
    """


class ArrayShapeError(ArrayError):
    """
    Exception raised when input shape is not expected
    """


class ArrayTypeError(ArrayError):
    """
    Exception raised when contents of array are not expected
    """


class FeaturesNotFoundError(Exception):
    """
    Exception raised when no features are found in a Frame
    """

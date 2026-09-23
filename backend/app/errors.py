"""Domain error type.

Any *expected* invalid request (bad parameter values, unknown window names,
out-of-range filter bands, ...) is raised as :class:`BadRequest`.
The API layer turns it into an HTTP 400 with a clear, human-readable message.
"""


class BadRequest(ValueError):
    """Raised when an API call is rejected with an explanatory message."""

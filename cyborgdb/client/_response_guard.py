"""Guard mixed into the generated ``ApiClient`` classes.

A 2xx body the response models reject is the service breaking its contract;
it must surface as ``ServiceError``, not as the argument-side
``ValidationError`` that ``translate_api_error`` would produce. Non-2xx
statuses raise ``ApiException`` from a ``finally`` and never reach this guard.
"""

import pydantic

from cyborgdb.exceptions import malformed_response_error


class ResponseValidationGuard:
    """Mixin over either generated ``ApiClient``; must precede it in the MRO."""

    def response_deserialize(self, response_data, response_types_map=None):
        try:
            return super().response_deserialize(response_data, response_types_map)
        except (pydantic.ValidationError, ValueError) as exc:
            # ValueError: the generated anyOf wrappers reject mismatched
            # bodies with one. Scrubbed and unchained — a 2xx body can carry
            # decrypted contents.
            error = malformed_response_error(exc, response_data.status)
        raise error

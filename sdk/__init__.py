"""ReturnShield client SDK.

Import the client from the module, not the package root::

    from sdk.returnshield import ReturnShieldClient
"""

from sdk.returnshield import (  # noqa: F401
    ABUSIVE_OUTCOMES,
    BENIGN_OUTCOMES,
    DecisionScore,
    ReturnShieldClient,
    ReturnShieldError,
    default_payload,
)

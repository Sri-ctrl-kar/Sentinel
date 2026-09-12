"""Environment and credential handling for the AI layer.

One place reads the environment, so no API key is ever picked up ad hoc in the
middle of a provider. Nothing here logs, prints or serialises a secret: the
rest of the system can ask :attr:`ReasonerSettings.has_credentials`, which is
a boolean, and never needs the key itself.

``.env`` is read if present, and never overrides a variable already set in the
process — an explicit export always wins over a file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

#: Which provider to build when nothing says otherwise. The mock is the
#: default on purpose: Sentinel must run, and be demonstrable, with no
#: credentials and no network.
DEFAULT_REASONER = "mock"
DEFAULT_ANTHROPIC_MODEL = "claude-opus-5"
#: Non-streaming requests keep a generous ceiling: adaptive thinking spends
#: from the same budget as the answer, and a truncated explanation is worse
#: than a slower one. The explanation itself is a few hundred tokens.
DEFAULT_MAX_TOKENS = 16000
DEFAULT_TIMEOUT_SECONDS = 60.0

ENV_REASONER = "SENTINEL_REASONER"
ENV_MODEL = "SENTINEL_REASONER_MODEL"
ENV_MAX_TOKENS = "SENTINEL_REASONER_MAX_TOKENS"
ENV_TIMEOUT = "SENTINEL_REASONER_TIMEOUT_SECONDS"
ENV_ANTHROPIC_KEY = "ANTHROPIC_API_KEY"
ENV_ANTHROPIC_TOKEN = "ANTHROPIC_AUTH_TOKEN"
ENV_ANTHROPIC_PROFILE = "ANTHROPIC_PROFILE"

#: Every environment variable the Anthropic SDK will accept as a credential,
#: in the order it resolves them. An unset ``ANTHROPIC_API_KEY`` does **not**
#: mean there is nothing to authenticate with — a token, a workload-identity
#: federation setup or an ``ant auth login`` profile on disk all work — so the
#: provider must not refuse to start merely because the key variable is empty.
CREDENTIAL_ENV_VARS = (
    ENV_ANTHROPIC_KEY,
    ENV_ANTHROPIC_TOKEN,
    ENV_ANTHROPIC_PROFILE,
    "ANTHROPIC_IDENTITY_TOKEN",
    "ANTHROPIC_IDENTITY_TOKEN_FILE",
)

#: Where ``ant auth login`` stores its profiles.
CREDENTIAL_PROFILE_DIR = "~/.config/anthropic"


def load_dotenv(path: str = ".env", environ: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Load ``KEY=value`` lines from ``path`` without overriding the process.

    Returns the variables that were actually applied. A missing file is not an
    error — running without one is the normal case.
    """
    target = environ if environ is not None else os.environ
    applied: Dict[str, str] = {}
    file_path = Path(path)
    if not file_path.is_file():
        return applied

    for raw in file_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key or key in target:
            continue
        target[key] = value
        applied[key] = value
    return applied


@dataclass(frozen=True)
class ReasonerSettings:
    """Resolved configuration for building a reasoner."""

    provider: str = DEFAULT_REASONER
    model: str = DEFAULT_ANTHROPIC_MODEL
    max_tokens: int = DEFAULT_MAX_TOKENS
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    #: Whether *some* credential source was detected. The value itself is
    #: never stored here, and this is a hint, not a guarantee: only the API can
    #: say whether a credential is valid.
    has_credentials: bool = False
    #: Which source was detected, for a message a user can act on.
    credential_source: Optional[str] = None

    @classmethod
    def from_env(
        cls, environ: Optional[Dict[str, str]] = None, dotenv: Optional[str] = ".env"
    ) -> "ReasonerSettings":
        if environ is None:
            if dotenv:
                load_dotenv(dotenv)
            environ = dict(os.environ)

        source = detect_credential_source(environ)
        return cls(
            provider=environ.get(ENV_REASONER, DEFAULT_REASONER),
            model=environ.get(ENV_MODEL, DEFAULT_ANTHROPIC_MODEL),
            max_tokens=_int(environ.get(ENV_MAX_TOKENS), DEFAULT_MAX_TOKENS),
            timeout_seconds=_float(environ.get(ENV_TIMEOUT), DEFAULT_TIMEOUT_SECONDS),
            has_credentials=source is not None,
            credential_source=source,
        )

    def describe(self) -> str:
        credentials = self.credential_source or "absent"
        return (
            f"provider={self.provider} model={self.model} "
            f"max_tokens={self.max_tokens} credentials={credentials}"
        )


def detect_credential_source(
    environ: Optional[Dict[str, str]] = None, home: Optional[str] = None
) -> Optional[str]:
    """Name the credential source the SDK would use, or ``None``.

    Reports the *name* of the source — never the value — so a caller can print
    it. Detection is best-effort by design: it says a credential is available
    to try, not that it is valid. Only the API can say that.
    """
    target = os.environ if environ is None else environ
    for variable in CREDENTIAL_ENV_VARS:
        if str(target.get(variable, "")).strip():
            return variable

    base = Path(home).expanduser() if home else Path(CREDENTIAL_PROFILE_DIR).expanduser()
    profile = base if home is None else base / ".config" / "anthropic"
    if profile.is_dir() and any(profile.iterdir()):
        return "ant auth profile"
    return None


def _int(value: Optional[str], default: int) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return default


def _float(value: Optional[str], default: float) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default

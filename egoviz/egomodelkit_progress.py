""" Progress helpers for EgoModelKit container instrumentation. """

from __future__ import annotations

import json
from typing import Any, Final

PREFIX: Final[str] = "EGOMODELKIT_PROGRESS "


def emit_progress(kind: str, **payload: Any) -> None:
    """ Emit one machine-readable EgoModelKit progress line. """
    print(
        PREFIX + json.dumps(
            {
                "kind": kind,
                **payload,
            },
            sort_keys = True,
        ),
        flush = True,
    )

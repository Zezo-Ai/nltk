# Natural Language Toolkit: Improve the security when loading pickled data.
#
# Copyright (C) 2001-2025 NLTK Project
# Author: Eric Kafe <kafe.eric@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT
#

from __future__ import annotations

import pickle
import warnings
from typing import Any, BinaryIO


class RestrictedUnpickler(pickle.Unpickler):
    """
    Unpickler that prevents any class or function from being used during loading.
    """

    def find_class(self, module, name):
        # Forbid every function
        raise pickle.UnpicklingError(f"global '{module}.{name}' is forbidden")


_WARNING = (
    "Security warning: loading pickles can execute arbitrary code. "
    "Only load pickle files from trusted sources and never from untrusted "
    "or unauthenticated locations."
)


class WarningUnpickler(pickle.Unpickler):
    """
    Unpickler that emits a warning before loading data.

    This does NOT make unpickling safe; it only makes the risk explicit.
    """

    def __init__(self, file: BinaryIO, *, context: str | None = None, **kwargs: Any):
        super().__init__(file, **kwargs)
        self._context = context
        self._warned = False

    def load(self) -> Any:
        if not self._warned:
            msg = _WARNING if self._context is None else f"{_WARNING} ({self._context})"
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
            self._warned = True
        return super().load()


def load_with_warning(file: BinaryIO, *, context: str | None = None) -> Any:
    """
    Convenience wrapper mirroring pickle.load(file), but with a warning.
    """
    return WarningUnpickler(file, context=context).load()

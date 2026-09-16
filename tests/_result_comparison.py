"""Compare every public result field, including patch locations and statistics."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any

import numpy as np


def assert_results_equal(actual: Any, expected: Any) -> None:
    if isinstance(actual, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif is_dataclass(actual):
        assert type(actual) is type(expected)
        for field in fields(actual):
            assert_results_equal(getattr(actual, field.name), getattr(expected, field.name))
    elif isinstance(actual, list):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            assert_results_equal(left, right)
    else:
        assert actual == expected

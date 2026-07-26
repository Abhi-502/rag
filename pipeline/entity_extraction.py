"""
entity_extraction.py
=====================
Diagram step:  OCR text  ->  Medical Entity Extraction

Job of this file: turn a wall of raw OCR text into a clean list of
(test_name, value, unit) triples, e.g. "Hemoglobin, 10.2, g/dL".

Blood reports are messy (spacing, OCR typos, different lab formats), so we
use a small dictionary of known test names + flexible regex rather than
trying to hand-parse every possible layout.
"""

import re
from dataclasses import dataclass
from typing import List, Optional

from pipeline.reference_ranges import KNOWN_TESTS


@dataclass
class BloodTestResult:
    """One extracted lab value."""
    test_name: str          # normalized name, e.g. "Hemoglobin"
    raw_label: str           # exactly what the OCR text said, for transparency
    value: float
    unit: Optional[str]


# Matches things like: "Hemoglobin  10.2 g/dL", "WBC: 11,200 /uL", "Glucose - 105 mg/dL"
_VALUE_PATTERN = re.compile(
    r"(?P<label>[A-Za-z()/ .%-]+?)"      # the test name (letters/symbols)
    r"[\s:=-]+"                          # separator (colon, dash, spaces...)
    r"(?P<value>[\d,]+\.?\d*)"           # the number (handles "11,200")
    r"\s*(?P<unit>[A-Za-z/%μ]+)?",       # optional unit
    re.UNICODE,
)


def extract_entities(ocr_text: str) -> List[BloodTestResult]:
    """
    Scan OCR text line-by-line and pull out any recognizable test result.

    We only keep matches whose label fuzzy-matches one of our KNOWN_TESTS,
    so we don't accidentally treat a page number or a phone number as a
    lab value.
    """
    results: List[BloodTestResult] = []

    for line in ocr_text.splitlines():
        match = _VALUE_PATTERN.search(line)
        if not match:
            continue

        raw_label = match.group("label").strip(" .:-")
        value_str = match.group("value").replace(",", "")
        unit = match.group("unit")

        normalized_name = _match_known_test(raw_label)
        if not normalized_name:
            continue  # skip anything we can't confidently identify

        try:
            value = float(value_str)
        except ValueError:
            continue

        results.append(
            BloodTestResult(
                test_name=normalized_name,
                raw_label=raw_label,
                value=value,
                unit=unit,
            )
        )

    return results


def _match_known_test(raw_label: str) -> Optional[str]:
    """
    Fuzzy-match an OCR'd label (e.g. 'Hemogiobin', 'HB') against our known
    test dictionary's canonical name + aliases.
    """
    cleaned = raw_label.lower().strip()

    for canonical_name, info in KNOWN_TESTS.items():
        aliases = [canonical_name.lower()] + [a.lower() for a in info.aliases]
        for alias in aliases:
            if alias in cleaned or cleaned in alias:
                return canonical_name
    return None

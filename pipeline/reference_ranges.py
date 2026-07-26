"""
reference_ranges.py
====================
Diagram step:  ... ->  Reference-Range Verification

Job of this file:
1. Hold a small, well-known dictionary of common blood tests, their normal
   adult reference ranges, and the aliases OCR might produce for them.
2. Provide a deterministic function that flags each extracted value as
   LOW / NORMAL / HIGH.

This is intentionally NOT done by the language model. "Is 10.2 g/dL low
for hemoglobin?" is a lookup-and-compare problem, not something we want an
LLM guessing at — determinism here is what makes the final report
"clinically reliable" rather than merely plausible-sounding.

NOTE: These are general adult reference ranges for illustration. Real
reference ranges vary by lab, sex, age, and units — a production system
should pull ranges from the lab report itself when printed, or from a
maintained clinical database.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TestInfo:
    unit: str
    low: float
    high: float
    aliases: List[str] = field(default_factory=list)


# A small, illustrative set of common complete-blood-count / metabolic tests.
KNOWN_TESTS = {
    "Hemoglobin": TestInfo(unit="g/dL", low=13.0, high=17.0,
                            aliases=["hb", "hgb", "haemoglobin"]),
    "WBC Count": TestInfo(unit="/uL", low=4000, high=11000,
                           aliases=["wbc", "white blood cell", "leukocyte count"]),
    "Platelet Count": TestInfo(unit="/uL", low=150000, high=450000,
                                aliases=["platelets", "plt"]),
    "Fasting Glucose": TestInfo(unit="mg/dL", low=70, high=100,
                                 aliases=["glucose", "fbs", "blood sugar"]),
    "Total Cholesterol": TestInfo(unit="mg/dL", low=0, high=200,
                                   aliases=["cholesterol", "chol"]),
    "Creatinine": TestInfo(unit="mg/dL", low=0.6, high=1.3,
                            aliases=["creat", "s. creatinine"]),
    "RBC Count": TestInfo(unit="million/uL", low=4.5, high=5.9,
                           aliases=["rbc", "red blood cell"]),
    "Hematocrit": TestInfo(unit="%", low=38.0, high=50.0,
                            aliases=["hct", "pcv"]),
}


@dataclass
class VerifiedResult:
    test_name: str
    value: float
    unit: Optional[str]
    normal_low: float
    normal_high: float
    status: str  # "LOW", "NORMAL", "HIGH", or "UNKNOWN_RANGE"


def verify_results(extracted_results) -> List[VerifiedResult]:
    """
    Compare each extracted BloodTestResult against its known normal range.

    `extracted_results` is a list of entity_extraction.BloodTestResult.
    Returns a list of VerifiedResult with a LOW/NORMAL/HIGH flag attached.
    """
    verified: List[VerifiedResult] = []

    for result in extracted_results:
        info = KNOWN_TESTS.get(result.test_name)
        if info is None:
            verified.append(
                VerifiedResult(
                    test_name=result.test_name,
                    value=result.value,
                    unit=result.unit,
                    normal_low=float("nan"),
                    normal_high=float("nan"),
                    status="UNKNOWN_RANGE",
                )
            )
            continue

        if result.value < info.low:
            status = "LOW"
        elif result.value > info.high:
            status = "HIGH"
        else:
            status = "NORMAL"

        verified.append(
            VerifiedResult(
                test_name=result.test_name,
                value=result.value,
                unit=result.unit or info.unit,
                normal_low=info.low,
                normal_high=info.high,
                status=status,
            )
        )

    return verified

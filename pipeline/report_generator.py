"""
report_generator.py
====================
Diagram step:  Reference-Range Verification  ->  Clinically Reliable, Evidence-Grounded Report

Job of this file: assemble the final output. Notice the final report always
includes a deterministic "Verified Values" table computed directly from
reference_ranges.py — never from the language model — so a reader can trust
the numbers even if they read no further than that table.
"""

from typing import List

import config
from pipeline.reference_ranges import VerifiedResult


def build_final_report(
    verified_results: List[VerifiedResult],
    vlm_explanation: str,
) -> str:
    """Combine deterministic verification + the model's explanation into Markdown."""

    lines = ["# Blood Report Analysis\n"]

    lines.append("## Verified Values (computed directly, not by the AI model)\n")
    lines.append("| Test | Value | Normal Range | Status |")
    lines.append("|------|-------|--------------|--------|")
    for r in verified_results:
        if r.status == "UNKNOWN_RANGE":
            range_str = "n/a (no reference on file)"
        else:
            range_str = f"{r.normal_low} - {r.normal_high} {r.unit or ''}".strip()
        lines.append(f"| {r.test_name} | {r.value} {r.unit or ''} | {range_str} | **{r.status}** |")

    lines.append("\n## Explanation (evidence-grounded, AI-generated)\n")
    lines.append(vlm_explanation.strip())

    lines.append("\n---")
    lines.append(f"\n> {config.MEDICAL_DISCLAIMER}")

    return "\n".join(lines)

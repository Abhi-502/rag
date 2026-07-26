"""
retriever.py
============
Diagram steps:  Retriever  ->  Retrieved Evidence

Job of this file: for each abnormal (or all) extracted test result, ask the
VectorIndex for the most relevant knowledge snippets, and collect them into
one bundle of "Retrieved Evidence" to hand to the vision-language model.

Keeping this as its own file (separate from knowledge_base.py) mirrors the
diagram's separate "Retriever" and "Vector Index" boxes: the index just
knows how to do similarity search, the retriever knows *what* to search for
and *how much* evidence to gather for this specific report.
"""

from dataclasses import dataclass
from typing import List

from pipeline.knowledge_base import VectorIndex, KnowledgeDoc
from pipeline.reference_ranges import VerifiedResult


@dataclass
class EvidenceBundle:
    test_name: str
    status: str  # LOW / NORMAL / HIGH / UNKNOWN_RANGE
    snippets: List[KnowledgeDoc]


def retrieve_evidence(
    verified_results: List[VerifiedResult],
    index: VectorIndex,
    only_abnormal: bool = False,
) -> List[EvidenceBundle]:
    """
    For each verified test result, retrieve supporting knowledge snippets.

    only_abnormal=True limits retrieval to LOW/HIGH results, which keeps the
    prompt to the vision-language model shorter — useful once a report has
    many normal values that don't need explaining.
    """
    bundles: List[EvidenceBundle] = []

    for result in verified_results:
        if only_abnormal and result.status == "NORMAL":
            continue

        # A natural-language query works better with TF-IDF/embeddings than
        # just the bare test name, e.g. "Hemoglobin low anemia causes".
        query = f"{result.test_name} {result.status.lower()}"
        snippets = index.search(query)

        bundles.append(
            EvidenceBundle(test_name=result.test_name, status=result.status, snippets=snippets)
        )

    return bundles


def format_evidence_for_prompt(bundles: List[EvidenceBundle]) -> str:
    """Render the retrieved evidence as plain text to paste into the VLM prompt."""
    lines = []
    for bundle in bundles:
        lines.append(f"### {bundle.test_name} ({bundle.status})")
        for snippet in bundle.snippets:
            lines.append(f"- [{snippet.source_file}] {snippet.text.strip()}")
        lines.append("")
    return "\n".join(lines)

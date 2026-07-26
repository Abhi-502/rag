"""
main.py
=======
Runs the full pipeline shown in the diagram, one step at a time:

  Blood Report Image
        -> OCR
        -> Medical Entity Extraction
        -> (Vector Index built from Knowledge Base, once)
        -> Retriever -> Retrieved Evidence
        -> Vision-Language Model (sees image + query + evidence)
        -> Reference-Range Verification
        -> Final Report

Usage:
    python3 main.py --image path/to/report.png
"""

import argparse

import config
from pipeline import ocr_module
from pipeline import entity_extraction
from pipeline.knowledge_base import VectorIndex
from pipeline import retriever
from pipeline import reference_ranges
from pipeline import vlm_module
from pipeline import report_generator


def run_pipeline(image_path: str, output_path: str = "report_output.md") -> str:
    print(f"[1/6] Running OCR on {image_path} ...")
    ocr_lines = ocr_module.run_ocr(image_path)
    ocr_text = ocr_module.ocr_lines_to_text(ocr_lines)
    print(f"       -> extracted {len(ocr_lines)} lines of text")

    print("[2/6] Extracting medical entities ...")
    extracted_results = entity_extraction.extract_entities(ocr_text)
    print(f"       -> found {len(extracted_results)} recognizable test values")
    if not extracted_results:
        print("       ! No known test values found. Check the image quality "
              "or extend KNOWN_TESTS in reference_ranges.py.")

    print("[3/6] Verifying values against reference ranges ...")
    verified_results = reference_ranges.verify_results(extracted_results)
    for r in verified_results:
        print(f"       - {r.test_name}: {r.value} {r.unit} -> {r.status}")

    print("[4/6] Building knowledge base vector index ...")
    index = VectorIndex(docs_dir=config.KNOWLEDGE_DOCS_DIR)

    print("[5/6] Retrieving relevant evidence ...")
    evidence_bundles = retriever.retrieve_evidence(verified_results, index)
    evidence_text = retriever.format_evidence_for_prompt(evidence_bundles)

    print("[6/6] Asking the vision-language model to write the explanation ...")
    structured_data_text = "\n".join(
        f"- {r.test_name}: {r.value} {r.unit} (normal range "
        f"{r.normal_low}-{r.normal_high}, status: {r.status})"
        for r in verified_results
    )
    try:
        vlm_explanation = vlm_module.generate_explanation(
            image_path=image_path,
            structured_data_text=structured_data_text,
            evidence_text=evidence_text,
        )
    except Exception as e:
        print(f"       ! Vision-language model call failed ({e}).")
        print("       -> Falling back to a placeholder explanation so the "
              "pipeline still produces a report. Set your API key to get a "
              "real AI-written explanation.")
        vlm_explanation = (
            "_(Vision-language model unavailable — set OPENAI_API_KEY, or "
            "configure a Qwen2.5-VL endpoint in config.py, to generate a "
            "real explanation here.)_"
        )

    final_report = report_generator.build_final_report(verified_results, vlm_explanation)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_report)

    print(f"\nDone. Report saved to {output_path}\n")
    return final_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blood Report RAG Pipeline")
    parser.add_argument("--image", required=True, help="Path to the blood report image")
    parser.add_argument("--output", default="report_output.md", help="Where to save the report")
    args = parser.parse_args()

    report_text = run_pipeline(args.image, args.output)
    print(report_text)


def _parse_request_payload(request):
    try:
        if hasattr(request, "json") and request.json is not None:
            return request.json
    except Exception:
        pass

    body = getattr(request, "body", None)
    if isinstance(body, (bytes, bytearray)):
        try:
            body = body.decode("utf-8")
        except Exception:
            body = None

    if isinstance(body, str):
        try:
            import json
            return json.loads(body)
        except Exception:
            pass

    return {}


def handler(request):
    payload = _parse_request_payload(request)
    image_path = payload.get("image")
    output_path = payload.get("output", "report_output.md")

    if not image_path:
        args = getattr(request, "args", None)
        if args is not None:
            image_path = args.get("image")

    if not image_path:
        return {
            "statusCode": 400,
            "body": "Missing required field: image",
        }

    try:
        report_text = run_pipeline(image_path, output_path)
    except Exception as exc:
        return {
            "statusCode": 500,
            "body": f"Pipeline failed: {exc}",
        }

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "text/markdown"},
        "body": report_text,
    }

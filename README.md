# Blood Report RAG Pipeline

A plain-Python project that turns a photo of a blood test report into a
clinically-grounded, easy-to-read explanation. It follows this pipeline:

```
Blood Report Image
      |
      v
OCR (extract raw text from the image)
      |
      v
Medical Entity Extraction (pull out test name, value, unit)
      |
      v
Vector Index  <---  Knowledge Base (WHO / CDC / ICMR / PubMed notes)
      |
      v
Retriever (find the knowledge snippets relevant to THIS report)
      |
      v
Retrieved Evidence
      |
      v
Vision-Language Model  <-- also receives the raw image + a query directly
(reads image + evidence, writes a draft explanation)
      |
      v
Reference-Range Verification (checks every number against known normal ranges)
      |
      v
Final Report (clinically reliable, evidence-grounded)
```

## Why it's built this way

- **OCR + entity extraction** turn a messy photo into clean structured data
  (e.g. `Hemoglobin = 10.2 g/dL`), so later steps don't have to re-read the image
  from scratch for numbers.
- **Knowledge base + retriever** make sure the model doesn't "hallucinate"
  medical advice — it's given real reference text about each test before it
  writes anything.
- **The vision-language model still sees the original image** (the dashed
  "image + query" arrow in the diagram) because OCR can miss things like table
  layout, highlighted/flagged values, or handwritten doctor notes.
- **Reference-range verification is a separate, deterministic step** — not
  left to the language model — because "is this number normal?" should be
  computed with real thresholds, not guessed by an LLM.

## Folder layout

```
blood_report_rag/
├── main.py                        # runs the whole pipeline end-to-end
├── config.py                      # settings: API keys, model names, paths
├── requirements.txt
├── data/
│   ├── knowledge_docs/            # sample WHO/CDC/ICMR-style reference notes
│   └── sample_reports/            # put a test image here
└── pipeline/
    ├── ocr_module.py              # Step: OCR
    ├── entity_extraction.py       # Step: Medical Entity Extraction
    ├── knowledge_base.py          # Step: Vector Index + Knowledge Base
    ├── retriever.py               # Step: Retriever
    ├── vlm_module.py              # Step: Vision-Language Model
    ├── reference_ranges.py        # Step: Reference-Range Verification
    └── report_generator.py        # Step: final report assembly
```

## Setup

```bash
cd blood_report_rag
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

You also need an OCR engine installed on your system:
- **PaddleOCR** (matches the diagram): `pip install paddleocr paddlepaddle`
- or, as a lighter fallback, **Tesseract**: `pip install pytesseract` and
  install the `tesseract-ocr` system package.

The code auto-detects whichever one is available (see `ocr_module.py`).

For the vision-language model step, set an API key as an environment
variable, e.g.:

```bash
export OPENAI_API_KEY="sk-..."      # for GPT-4.1 / GPT-4.1 Vision
# or configure a Qwen2.5-VL endpoint — see config.py
```

## Running it

```bash
python3 main.py --image data/sample_reports/your_report.png
```

This will:
1. Build (or load a cached) vector index from `data/knowledge_docs/`.
2. OCR the image and extract test values.
3. Retrieve the relevant knowledge snippets for the tests found.
4. Send the image + extracted data + retrieved evidence to the VLM.
5. Check every value against known reference ranges.
6. Print a final Markdown report and save it to `report_output.md`.

## Notes on "understandability"

Every module is intentionally small (under ~150 lines), has a single
responsibility that matches one box in your diagram, and is commented as
"what this does / why it exists" rather than just "what the code says" —
so you can read the project top-to-bottom the same way you'd read the
flowchart.

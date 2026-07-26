"""
vlm_module.py
=============
Diagram step:  Vision-Language Model (Qwen2.5-VL / GPT-4.1 Pro / GPT-4.1 Vision)

Job of this file: send the model everything it needs to write a good draft
explanation:
  1. The original report IMAGE (the dashed "image + query" arrow in your
     diagram) — so it can see table structure, flags, or handwriting OCR
     missed.
  2. The extracted+verified test values (structured data).
  3. The retrieved evidence snippets (from retriever.py).

The model's job is ONLY to explain things in plain language and connect the
dots — it is explicitly instructed not to invent numbers or ranges, since
those are already computed deterministically upstream.
"""

import base64

import config


_SYSTEM_PROMPT = """You are a medical report explainer assistant.
You will be given:
- A photo of a blood test report.
- A structured list of test values that were already extracted and checked
  against reference ranges.
- Evidence snippets retrieved from general health-authority reference notes.

Your job:
- Write a clear, plain-language explanation of what the abnormal (LOW/HIGH)
  values might mean, grounded ONLY in the evidence snippets provided.
- Do NOT invent new numbers, reference ranges, or diagnoses.
- Do NOT state a definitive diagnosis — describe possible explanations and
  recommend follow-up with a clinician.
- If the image shows something the extracted data doesn't capture (e.g. a
  flagged/highlighted value, doctor's handwritten note), mention it.
- Keep it organized by test name.
"""


def generate_explanation(image_path: str, structured_data_text: str, evidence_text: str) -> str:
    """
    Call the configured vision-language model and return its draft
    explanation as plain text (Markdown-friendly).
    """
    from openai import OpenAI

    if config.VLM_PROVIDER == "qwen":
        client = OpenAI(base_url=config.QWEN_BASE_URL, api_key=config.QWEN_API_KEY)
        model_name = config.QWEN_MODEL_NAME
    else:
        client = OpenAI(api_key=config.OPENAI_API_KEY)
        model_name = config.OPENAI_MODEL_NAME

    image_data_url = _encode_image_as_data_url(image_path)

    user_content = [
        {
            "type": "text",
            "text": (
                "Structured, verified test values:\n"
                f"{structured_data_text}\n\n"
                "Retrieved evidence:\n"
                f"{evidence_text}\n\n"
                "Using the image plus the data above, write the explanation."
            ),
        },
        {"type": "image_url", "image_url": {"url": image_data_url}},
    ]

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        max_tokens=1200,
    )

    return response.choices[0].message.content


def _encode_image_as_data_url(image_path: str) -> str:
    """Vision APIs generally accept a base64 data URL for local images."""
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    extension = image_path.rsplit(".", 1)[-1].lower()
    mime_type = "image/png" if extension == "png" else "image/jpeg"
    return f"data:{mime_type};base64,{encoded}"

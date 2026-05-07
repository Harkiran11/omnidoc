"""
OmniDoc Test Suite
Run: python tests/test_omnidoc.py
All tests should pass before submission.
"""

import sys, os, base64, json, io, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import httpx
from PIL import Image, ImageDraw, ImageFont
import fitz

LLAMA_URL = os.getenv("LLAMA_VISION_URL", "http://localhost:8001/v1")
QWEN_URL  = os.getenv("QWEN_VL_URL",      "http://localhost:8002/v1")

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
results = []


def test(name, fn):
    try:
        fn()
        print(f"  {PASS} {name}")
        results.append((name, True, None))
    except Exception as e:
        print(f"  {FAIL} {name}: {e}")
        results.append((name, False, str(e)))


def make_test_pdf(path="tests/test_doc.pdf"):
    """Create a simple PDF with text and a fake chart description."""
    os.makedirs("tests", exist_ok=True)
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "OmniDoc Test Document", fontsize=18)
    page.insert_text((72, 110), "This document contains test content for OmniDoc.", fontsize=11)
    page.insert_text((72, 140), "Revenue: $1.2M in Q1, $1.5M in Q2, $1.8M in Q3", fontsize=11)
    page.insert_text((72, 170), "See Figure 1 for the quarterly revenue chart.", fontsize=11)
    page.insert_text((72, 220), "Table 1: Key Metrics", fontsize=12)
    page.insert_text((72, 245), "| Metric     | Value  |", fontsize=10)
    page.insert_text((72, 260), "| Revenue    | $4.5M  |", fontsize=10)
    page.insert_text((72, 275), "| Profit     | $1.2M  |", fontsize=10)
    page.insert_text((72, 290), "| Growth     | 24%    |", fontsize=10)
    doc.save(path)
    doc.close()
    return path


print("\n" + "="*50)
print("  OmniDoc Test Suite")
print("="*50 + "\n")

# ── Test 1: PDF creation ─────────────────────────────
print("[ Document Processing ]")

def test_pdf_creation():
    path = make_test_pdf()
    assert os.path.exists(path), "Test PDF not created"
    assert os.path.getsize(path) > 0, "Test PDF is empty"

test("Create test PDF", test_pdf_creation)


def test_pdf_load():
    from app import DocumentProcessor
    proc = DocumentProcessor()
    n = proc.load_pdf("tests/test_doc.pdf")
    assert n >= 1, f"Expected at least 1 page, got {n}"
    assert len(proc.pages_data) == n
    assert proc.pages_data[0]["image_b64"], "No image data for page 1"
    assert len(proc.pages_data[0]["raw_text"]) > 0, "No text extracted"

test("Load PDF and extract pages", test_pdf_load)


def test_page_image_valid():
    from app import DocumentProcessor
    proc = DocumentProcessor()
    proc.load_pdf("tests/test_doc.pdf")
    b64 = proc.pages_data[0]["image_b64"]
    img_bytes = base64.b64decode(b64)
    img = Image.open(io.BytesIO(img_bytes))
    assert img.size[0] > 0 and img.size[1] > 0


test("Page renders to valid image", test_page_image_valid)


# ── Test 2: Model endpoints ──────────────────────────
print("\n[ Model Endpoints ]")

def test_llama_health():
    r = httpx.get(f"{LLAMA_URL}/models", timeout=10)
    assert r.status_code == 200, f"Llama endpoint returned {r.status_code}"
    models = r.json().get("data", [])
    assert len(models) > 0, "No models loaded in Llama endpoint"

test("Llama Vision endpoint is up", test_llama_health)


def test_qwen_health():
    r = httpx.get(f"{QWEN_URL}/models", timeout=10)
    assert r.status_code == 200, f"Qwen endpoint returned {r.status_code}"

test("Qwen-VL endpoint is up", test_qwen_health)


def test_llama_text_response():
    """Test Llama gives a response to a simple text message."""
    payload = {
        "model"   : "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "messages": [{"role": "user", "content": "Say hello in one word."}],
        "max_tokens": 10
    }
    r = httpx.post(f"{LLAMA_URL}/chat/completions", json=payload, timeout=30)
    assert r.status_code == 200
    text = r.json()["choices"][0]["message"]["content"]
    assert len(text) > 0

test("Llama returns text response", test_llama_text_response)


def test_llama_vision_response():
    """Test Llama can process an image."""
    from app import DocumentProcessor
    proc = DocumentProcessor()
    proc.load_pdf("tests/test_doc.pdf")
    b64 = proc.pages_data[0]["image_b64"]

    payload = {
        "model": "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What text do you see in this image? Reply in one sentence."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            ]
        }],
        "max_tokens": 100
    }
    r = httpx.post(f"{LLAMA_URL}/chat/completions", json=payload, timeout=60)
    assert r.status_code == 200
    response = r.json()["choices"][0]["message"]["content"]
    assert len(response) > 10, "Response too short"
    # Check it actually read the document
    assert any(word in response.lower() for word in ["document", "test", "omnidoc", "revenue", "text"]), \
        f"Vision model didn't seem to read the image. Got: {response}"

test("Llama Vision reads page image correctly", test_llama_vision_response)


# ── Test 3: End-to-end chat ──────────────────────────
print("\n[ End-to-End Chat ]")

def test_full_pipeline():
    from app import STATE, process_document, chat_with_doc
    import types

    # Simulate file upload object
    class FakeFile:
        name = "tests/test_doc.pdf"

    # Process document (simplified — call processor directly)
    STATE.processor.load_pdf("tests/test_doc.pdf")
    STATE.doc_loaded = True

    # Manually set summaries so we don't need vision models for this unit test
    for page in STATE.processor.pages_data:
        page["summary"] = page["raw_text"]

    # Ask a question
    _, history = chat_with_doc("What is the revenue mentioned in this document?", [])
    assert len(history) == 1, "No response in history"
    question, answer = history[0]
    assert "revenue" in question.lower()
    assert len(answer) > 20, f"Answer too short: {answer}"

test("Full pipeline: load → question → answer", test_full_pipeline)


def test_relevant_pages():
    from app import STATE
    STATE.processor.load_pdf("tests/test_doc.pdf")
    for i, page in enumerate(STATE.processor.pages_data):
        page["summary"] = page["raw_text"]
    STATE.doc_loaded = True

    pages = STATE.find_relevant_pages("revenue chart", top_k=2)
    assert len(pages) > 0, "No relevant pages found"
    assert pages[0]["page_num"] >= 1

test("Relevant page retrieval works", test_relevant_pages)


# ── Summary ──────────────────────────────────────────
print("\n" + "="*50)
passed = sum(1 for _, ok, _ in results if ok)
total  = len(results)
print(f"  Results: {passed}/{total} tests passed")
if passed == total:
    print("  \033[92mAll tests passed! OmniDoc is ready.\033[0m")
else:
    print("  \033[91mSome tests failed. Fix issues before submitting.\033[0m")
    for name, ok, err in results:
        if not ok:
            print(f"    ✗ {name}: {err}")
print("="*50 + "\n")

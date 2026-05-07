"""
OmniDoc — Multimodal Document Intelligence
Fixed for Gradio 6.x + vLLM 0.17.1 on AMD MI300X
"""

import gradio as gr
import os, base64, io, httpx
from pathlib import Path
import fitz
from PIL import Image

LLAMA_URL   = os.getenv("LLAMA_VISION_URL", "http://localhost:8001/v1")
QWEN_URL    = os.getenv("QWEN_VL_URL",      "http://localhost:8002/v1")
LLAMA_MODEL = "meta-llama/Llama-3.2-11B-Vision-Instruct"
QWEN_MODEL  = "Qwen/Qwen2-VL-7B-Instruct"
DPI = 120

def detect_available_model():
    for url, model in [(LLAMA_URL, LLAMA_MODEL), (QWEN_URL, QWEN_MODEL)]:
        try:
            r = httpx.get(f"{url}/models", timeout=3)
            if r.status_code == 200:
                data = r.json().get("data", [])
                mid  = data[0]["id"] if data else model
                print(f"Using: {mid} at {url}")
                return url, mid
        except Exception:
            pass
    print("No model server found.")
    return None, None

ACTIVE_URL, ACTIVE_MODEL = detect_available_model()

def call_vision(prompt_text, b64_images):
    if ACTIVE_URL is None:
        return "No model server running. Start vLLM first."
    content = [{"type": "text", "text": prompt_text}]
    for b64 in b64_images[:2]:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    payload = {
        "model": ACTIVE_MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 1024,
        "temperature": 0.1,
    }
    try:
        resp = httpx.post(f"{ACTIVE_URL}/chat/completions", json=payload, timeout=90.0)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except httpx.TimeoutException:
        return "Request timed out. Model may still be loading — try again in 1 minute."
    except httpx.ConnectError:
        return f"Cannot reach model at {ACTIVE_URL}. Is vLLM running?"
    except Exception as e:
        return f"Error: {str(e)}"

class DocState:
    def __init__(self):
        self.pages, self.name, self.loaded = [], "", False
    def reset(self):
        self.__init__()

DOC = DocState()

def pdf_to_pages(pdf_path):
    DOC.reset()
    DOC.name = Path(pdf_path).name
    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc):
        mat = fitz.Matrix(DPI/72, DPI/72)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        if img.width > 800:
            img = img.resize((800, int(img.height * 800/img.width)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=75)
        DOC.pages.append({
            "page_num": i+1,
            "image_b64": base64.b64encode(buf.getvalue()).decode(),
            "raw_text": page.get_text("text").strip(),
            "summary": "",
        })
    doc.close()
    DOC.loaded = True
    return len(DOC.pages)

def find_relevant_pages(question, top_k=3):
    q_words = set(question.lower().split())
    visual_kw = {"chart","graph","table","figure","diagram","image","visual","plot","show"}
    wants_visual = bool(q_words & visual_kw)
    scored = []
    for page in DOC.pages:
        text  = (page.get("summary","") + " " + page.get("raw_text","")).lower()
        score = len(q_words & set(text.split()))
        if wants_visual and any(k in text for k in visual_kw):
            score += 5
        scored.append((score, page))
    scored.sort(key=lambda x: -x[0])
    return [p for _, p in scored[:top_k]]

def process_pdf(pdf_file, progress=gr.Progress()):
    if pdf_file is None:
        return "Please upload a PDF file.", []
    progress(0.1, desc="Reading PDF...")
    try:
        total = pdf_to_pages(pdf_file.name)
    except Exception as e:
        DOC.reset()
        return f"Could not open PDF: {e}", []
    progress(1.0, desc="Done!")
    m = f"Model: `{ACTIVE_MODEL}`" if ACTIVE_MODEL else "No model loaded — start vLLM"
    return (f"**{DOC.name}** — {total} pages ready.\n\n{m}\n\nAsk anything below."), []

def chat(user_msg, history):
    if not user_msg.strip():
        yield history
        return
    if not DOC.loaded:
        yield history + [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": "Please upload and process a PDF first."},
        ]
        return
    history = history + [{"role": "user", "content": user_msg}]
    yield history
    relevant = find_relevant_pages(user_msg, top_k=3)
    for page in relevant:
        if not page["summary"] and page["image_b64"]:
            page["summary"] = call_vision(
                f"Describe page {page['page_num']} of this document in detail: text, charts, tables, figures.",
                [page["image_b64"]]
            )
    context = ""
    for p in relevant:
        context += f"\n\n=== Page {p['page_num']} ===\n" + (p["summary"] or p["raw_text"][:600])
    prompt = (
        "You are OmniDoc, an expert document analyst. "
        "Use the page content below to answer the question precisely. "
        "Cite specific page numbers.\n\n"
        f"Document: {DOC.name}\nRelevant pages:{context}\n\nQuestion: {user_msg}"
    )
    imgs   = [p["image_b64"] for p in relevant[:2] if p.get("image_b64")]
    answer = call_vision(prompt, imgs)
    answer += f"\n\n*Pages consulted: {', '.join(str(p['page_num']) for p in relevant)}*"
    yield history + [{"role": "assistant", "content": answer}]

def clear_all():
    DOC.reset()
    return None, "Upload a new PDF to begin.", []

with gr.Blocks(title="OmniDoc") as demo:
    gr.HTML("""
    <div style="text-align:center; padding: 24px 0 8px;">
      <h1 style="font-size:2rem; font-weight:600; margin:0;">
        OmniDoc
      </h1>
      <p style="color:#6b7280; margin-top:8px; font-size:1rem;">
        Multimodal Document Intelligence · Powered by AMD Instinct MI300X
      </p>
      <div style="margin-top:8px;">
        <span style="background:#fef3c7; color:#92400e; padding:4px 12px; border-radius:20px; font-size:12px; margin:0 4px;">
          Llama 3.2 Vision
        </span>
        <span style="background:#dbeafe; color:#1e40af; padding:4px 12px; border-radius:20px; font-size:12px; margin:0 4px;">
          Qwen3-VL
        </span>
        <span style="background:#fee2e2; color:#991b1b; padding:4px 12px; border-radius:20px; font-size:12px; margin:0 4px;">
          ROCm 6.x
        </span>
      </div>
    </div>
    """)
    with gr.Row():
        with gr.Column(scale=1):
            pdf_input   = gr.File(label="Upload PDF", file_types=[".pdf"])
            process_btn = gr.Button("Analyse Document", variant="primary")
            clear_btn   = gr.Button("Clear", size="sm")
            status_md   = gr.Markdown("*Upload a PDF to begin.*")
            
            # ✅ Display model info with "Llama" label
            gr.Markdown("""
            **Model:** `Llama-3.2-11B-Vision`  
            **Backend:** `Qwen/Qwen2.5-VL-7B-Instruct`
            """)

            # ?~\~E Try asking section moved HERE (below "Ask anything below")
            gr.Markdown("""
            <div class="try-asking">

            ### Try asking:
            - *"Summarize the key findings"*
            - *"What does the chart on page 3 show?"*
            - *"List all tables and their contents"*
            - *"What are the main conclusions?"*
            - *"Find any financial figures mentioned"*
            
            </div>
            """)        
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="Ask anything about your document",
                height=500,
                
                
                
            )
            with gr.Row():
                msg_box  = gr.Textbox(placeholder="e.g. What does the chart on page 5 show?", label="", scale=5, lines=2, container=False)
                send_btn = gr.Button("Send", variant="primary", scale=1)

    process_btn.click(process_pdf, [pdf_input], [status_md, chatbot], show_progress=True)
    send_btn.click(chat, [msg_box, chatbot], [chatbot]).then(lambda: "", outputs=[msg_box])
    msg_box.submit(chat, [msg_box, chatbot], [chatbot]).then(lambda: "", outputs=[msg_box])
    clear_btn.click(clear_all, outputs=[pdf_input, status_md, chatbot])

if __name__ == "__main__":
    print(f"Model: {ACTIVE_MODEL or 'NONE'} | URL: {ACTIVE_URL or 'N/A'}")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True, show_error=True)

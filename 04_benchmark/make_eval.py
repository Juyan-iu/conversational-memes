#!/usr/bin/env python3
"""
Human evaluation set generator for meme benchmark

- 3 options per question: (Original OR Text distractor, randomly chosen) + Visual distractor + Easy distractor
- Options are shuffled and re-labeled A/B/C so the correct answer position is hidden
- Context info (original post, parent reply, meme text, OCR, labels) shown alongside
- Answer key saved separately as answers.txt

Usage:
  python make_eval.py --input ./benchmark_data --n 100
  python make_eval.py --input ./benchmark_data --n 100 --output ./eval_100
"""

import json
import random
import argparse
import base64
from pathlib import Path
from datetime import datetime


# ────────────────────────────────────────────────────────────────
def img_to_b64(path: Path) -> str | None:
    if path and path.exists():
        try:
            return base64.b64encode(path.read_bytes()).decode()
        except Exception:
            pass
    return None


def build_question(meta: dict, item_dir: Path, q_idx: int) -> dict | None:
    """
    Build a 3-option question from a meta dict.

    Logic:
      - Always 3 image options (A/B/C) + "None of the above" button
      - Randomly choose to show Original OR Text distractor as one of the 3 images
        - If Original is chosen → it IS in the options → answer is that label (A/B/C)
        - If Text distractor is chosen → it is shown BUT it's NOT the correct answer
          → all 3 images are wrong → answer is "None of the above" (N)
      - The other 2 slots are always Visual distractor + Easy distractor

    This means ~50% of questions have answer N, which is intentional.
    """
    uid     = meta.get("uid", "unknown")
    options = meta.get("options", {})

    orig_file = options.get("A")   # A_original.jpg  (the real meme)
    text_file = options.get("B")   # B_text_distractor.jpg
    vis_file  = options.get("C")   # C_visual_distractor.jpg
    easy_file = options.get("D")   # D_easy_distractor.jpg

    # All four must exist
    if not all([
        orig_file and (item_dir / orig_file).exists(),
        text_file and (item_dir / text_file).exists(),
        vis_file  and (item_dir / vis_file ).exists(),
        easy_file and (item_dir / easy_file).exists(),
    ]):
        return None

    # Randomly decide: show original (answer in options) or text_distractor (answer = None)
    show_original = random.choice([True, False])

    if show_original:
        # Original is one of the 3 images → it's the correct answer
        slot_file    = orig_file
        slot_type    = "original"
        none_of_above = False
    else:
        # Text distractor takes original's slot → all 3 images are wrong
        slot_file    = text_file
        slot_type    = "text_distractor"
        none_of_above = True

    # Build 3 choices: [slot, visual_distractor, easy_distractor]
    choices = [
        {"file": slot_file,  "type": slot_type,          "is_answer": not none_of_above},
        {"file": vis_file,   "type": "visual_distractor", "is_answer": False},
        {"file": easy_file,  "type": "easy_distractor",   "is_answer": False},
    ]
    random.shuffle(choices)

    labels       = ["A", "B", "C"]
    answer_label = "N" if none_of_above else None
    options_out  = []
    for label, choice in zip(labels, choices):
        b64 = img_to_b64(item_dir / choice["file"])
        if not b64:
            return None
        if choice["is_answer"]:
            answer_label = label
        options_out.append({
            "label":     label,
            "file":      choice["file"],
            "type":      choice["type"],
            "b64":       b64,
            "is_answer": choice["is_answer"],
        })

    return {
        "q_idx":         q_idx,
        "uid":           uid,
        "options":       options_out,
        "answer_label":  answer_label,   # "A"/"B"/"C" or "N" (none of the above)
        "none_of_above": none_of_above,
        "meta":          meta,
    }


# ────────────────────────────────────────────────────────────────
def render_question(q: dict) -> str:
    meta        = q["meta"]
    q_idx       = q["q_idx"]
    uid         = q["uid"]
    context     = meta.get("context", {})
    labels      = meta.get("labels", {})
    captions    = meta.get("captions", [])
    month       = meta.get("month", "")

    # ── context 파싱 (신/구 포맷 모두 지원) ──────────────────────
    orig_text     = (context.get("original_post_text") or context.get("original_post") or "").strip()
    orig_uri      = (context.get("original_post_uri") or "").strip()
    orig_images   = context.get("original_post_images") or []
    orig_ext_title = (context.get("original_post_external_title") or "").strip()
    orig_ext_url   = (context.get("original_post_external_url") or "").strip()

    quoted_text    = (context.get("quoted_post_text") or "").strip()
    quoted_images  = context.get("quoted_post_images") or []
    quoted_ext_title = (context.get("quoted_post_external_title") or "").strip()
    quoted_ext_url   = (context.get("quoted_post_external_url") or "").strip()

    ancestor_chain = context.get("ancestor_chain") or []

    parent_text   = (context.get("parent_reply_text") or context.get("parent_reply") or "").strip()
    parent_images = context.get("parent_reply_images") or []
    parent_ext_title = (context.get("parent_reply_external_title") or "").strip()

    meme_text = (context.get("meme_text") or "").strip()

    meme_label = (labels.get("meme_reply") or {}).get("discourse_function", "—")
    stance     = (labels.get("meme_reply") or {}).get("stance") or {}

    # Bluesky URI → URL
    def uri_to_url(uri):
        if not uri:
            return ""
        try:
            parts = uri.replace("at://", "").split("/")
            if len(parts) >= 3:
                return f"https://bsky.app/profile/{parts[0]}/post/{parts[2]}"
        except Exception:
            pass
        return uri

    orig_url = uri_to_url(orig_uri)

    # Stance tags
    stance_html = ""
    for k, color in [("sarcastic","#FF6B6B"), ("humorous","#FFD93D"), ("offensive","#6C5CE7")]:
        v   = stance.get(k, False)
        op  = "1" if v else "0.25"
        ico = "✓" if v else "✗"
        stance_html += (
            f'<span class="stance-tag" style="background:{color};opacity:{op}">'
            f'{ico} {k.capitalize()}</span>'
        )

    # 이미지 썸네일 (CDN URL)
    def img_thumbs(urls):
        if not urls:
            return ""
        imgs = "".join(
            f'<img src="{u}" onclick="openImg(\'{u}\')" class="ctx-img" title="Click to enlarge">'
            for u in urls[:4] if u
        )
        return f'<div class="ctx-imgs">{imgs}</div>' if imgs else ""

    # 외부 링크 뱃지
    def ext_link_html(title, url):
        if not title and not url:
            return ""
        label = title or url
        if url:
            return f'<div class="ext-link">🔗 <a href="{url}" target="_blank" class="post-link">{label[:80]}</a></div>'
        return f'<div class="ext-link">🔗 {label[:80]}</div>'

    # ── Context column 구성 ────────────────────────────────────
    ctx_parts = []

    # 1. 원 포스트
    if orig_text or orig_url or orig_images or orig_ext_title:
        link_html = (f' <a href="{orig_url}" target="_blank" class="post-link">🔗 View post</a>' if orig_url else "")
        ctx_parts.append(
            f'<div class="sec-label">📌 Original Post{link_html}</div>'
            + (f'<div class="ctx-text">{orig_text[:300]}{"…" if len(orig_text)>300 else ""}</div>' if orig_text else "")
            + img_thumbs(orig_images)
            + ext_link_html(orig_ext_title, orig_ext_url)
        )

    # 2. 원 포스트가 인용한 포스트
    if quoted_text or quoted_images or quoted_ext_title:
        ctx_parts.append(
            f'<div class="sec-label">💬 Quoted Post</div>'
            + (f'<div class="ctx-text muted">{quoted_text[:300]}{"…" if len(quoted_text)>300 else ""}</div>' if quoted_text else "")
            + img_thumbs(quoted_images)
            + ext_link_html(quoted_ext_title, quoted_ext_url)
        )

    # 3. 상위 댓글 체인 (ancestor_chain, 오래된 순)
    for i, node in enumerate(ancestor_chain):
        n_text   = (node.get("text") or "").strip()
        n_images = node.get("images") or []
        n_ext_title = (node.get("external_title") or "").strip()
        n_ext_url   = (node.get("external_url") or "").strip()
        n_quoted_text   = (node.get("quoted_post_text") or "").strip()
        n_quoted_images = node.get("quoted_post_images") or []
        if n_text or n_images or n_ext_title or n_quoted_text:
            ctx_parts.append(
                f'<div class="sec-label">↑ Thread [{i+1}]</div>'
                + (f'<div class="ctx-text">{n_text[:200]}{"…" if len(n_text)>200 else ""}</div>' if n_text else "")
                + img_thumbs(n_images)
                + ext_link_html(n_ext_title, n_ext_url)
                + ((f'<div class="ctx-text muted" style="margin-top:4px">↳ {n_quoted_text[:150]}</div>') if n_quoted_text else "")
                + img_thumbs(n_quoted_images)
            )

    # 4. 부모 댓글
    if parent_text or parent_images or parent_ext_title:
        ctx_parts.append(
            f'<div class="sec-label">↩ Parent Reply</div>'
            + (f'<div class="ctx-text">{parent_text[:200]}{"…" if len(parent_text)>200 else ""}</div>' if parent_text else "")
            + img_thumbs(parent_images)
            + ext_link_html(parent_ext_title, "")
        )

    # 5. 밈 댓글 텍스트
    if meme_text:
        ctx_parts.append(
            f'<div class="sec-label">🎭 Meme Reply Text</div>'
            f'<div class="ctx-text muted">{meme_text[:120]}</div>'
        )

    ctx_html = "\n".join(ctx_parts) or '<div class="ctx-text muted">(No context available)</div>'

    # Option images
    opts_html = ""
    for opt in q["options"]:
        opts_html += f"""
        <div class="option" data-qidx="{q_idx}" data-label="{opt['label']}">
            <img src="data:image/jpeg;base64,{opt['b64']}"
                 onclick="openImg(this.src)"
                 title="Click to enlarge">
            <button class="sel-btn" onclick="selectAnswer({q_idx}, '{opt['label']}', this)">
                {opt['label']}
            </button>
        </div>"""

    none_of_above = q.get("none_of_above", False)

    return f"""
    <div class="item" id="q{q_idx}" data-answered="0" data-noa="{'1' if none_of_above else '0'}">
        <div class="item-header">
            <span class="q-num">Q{q_idx}</span>
            <span class="uid">{uid[:28]}…</span>
            <span class="month-badge">📅 {month}</span>
            <span class="disc-chip">{meme_label}</span>
            {stance_html}
            <span class="answered-badge" id="badge-{q_idx}" style="display:none">✅ Answered</span>
        </div>
        <div class="task-desc">
            💬 <strong>Select the image that was actually used as a meme reply in this conversation context.</strong>
        </div>
        <div class="item-body">
            <div class="context-col">
                {ctx_html}
            </div>
            <div class="options-col">
                <div class="options-grid">
                    {opts_html}
                </div>
                <div class="noa-row">
                    <button class="noa-btn" data-qidx="{q_idx}" onclick="selectAnswer({q_idx}, 'N', this)">
                        ❌ None of the above
                    </button>
                </div>
            </div>
        </div>
    </div>"""


# ────────────────────────────────────────────────────────────────
def generate_html(questions: list, output_path: Path):
    total = len(questions)
    now   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows  = "\n".join(render_question(q) for q in questions)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Meme Benchmark Evaluation</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{{
  --bg:#0D0F14; --surface:#161920; --card:#1E2230;
  --border:#2A2F42; --accent:#7C6AFF; --accent2:#4ECDC4;
  --text:#E8EAF0; --muted:#6B7280; --correct:#00b894;
  --sel:#7C6AFF;
}}
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{background:var(--bg);color:var(--text);font-family:'Space Grotesk',sans-serif;padding-bottom:100px;}}

.topbar{{
  background:linear-gradient(135deg,#161920,#1a1230);
  border-bottom:1px solid var(--border);
  padding:14px 28px;display:flex;align-items:center;justify-content:space-between;
  position:sticky;top:0;z-index:100;
}}
.topbar-title{{
  font-size:1.05rem;font-weight:700;
  background:linear-gradient(90deg,#7C6AFF,#4ECDC4);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
}}
.topbar-right{{display:flex;align-items:center;gap:16px;}}
.progress-text{{color:var(--muted);font-family:'JetBrains Mono';font-size:0.78rem;}}
.submit-btn{{
  background:linear-gradient(135deg,#7C6AFF,#4ECDC4);
  color:#fff;border:none;border-radius:10px;
  padding:8px 20px;font-weight:700;font-size:0.85rem;cursor:pointer;
  opacity:0.4;transition:opacity .3s;
}}
.submit-btn.ready{{opacity:1;}}
.submit-btn.ready:hover{{filter:brightness(1.1);}}

.progress-bar-wrap{{height:3px;background:var(--border);position:sticky;top:53px;z-index:99;}}
.progress-bar{{height:100%;background:linear-gradient(90deg,#7C6AFF,#4ECDC4);width:0%;transition:width .3s;}}

.main{{max-width:1200px;margin:0 auto;padding:24px 18px;}}

.item{{
  background:var(--card);border:1px solid var(--border);
  border-radius:16px;margin-bottom:32px;overflow:hidden;
  transition:border-color .3s;
}}
.item.answered{{border-color:var(--accent2);}}

.item-header{{
  padding:11px 16px;background:var(--surface);
  border-bottom:1px solid var(--border);
  display:flex;align-items:center;gap:8px;flex-wrap:wrap;
}}
.q-num{{
  background:var(--accent);color:#fff;font-weight:700;
  font-size:0.78rem;padding:3px 10px;border-radius:8px;
  font-family:'JetBrains Mono';min-width:42px;text-align:center;
}}
.uid{{font-family:'JetBrains Mono';font-size:0.68rem;color:var(--muted);}}
.month-badge{{font-size:0.7rem;background:#2d3a4a;color:#74b9ff;padding:2px 7px;border-radius:8px;font-family:'JetBrains Mono';}}
.disc-chip{{font-family:'JetBrains Mono';font-size:0.72rem;background:rgba(124,106,255,0.15);color:#a78bfa;padding:2px 8px;border-radius:10px;}}
.stance-tag{{font-size:0.7rem;font-weight:600;padding:2px 8px;border-radius:10px;color:#fff;}}
.answered-badge{{font-size:0.72rem;color:var(--correct);font-weight:600;margin-left:auto;}}

.task-desc{{
  padding:10px 16px;font-size:0.82rem;color:#b0b8d0;
  border-bottom:1px solid var(--border);background:rgba(124,106,255,0.05);
}}
.task-desc strong{{color:var(--text);}}

.item-body{{display:grid;grid-template-columns:260px 1fr;}}
.context-col{{
  padding:14px;border-right:1px solid var(--border);
  background:var(--surface);overflow-y:auto;max-height:480px;
}}
.sec-label{{font-size:0.68rem;font-weight:600;color:var(--accent2);margin:10px 0 3px;font-family:'JetBrains Mono';}}
.sec-label:first-child{{margin-top:0;}}
.ctx-text{{font-size:0.8rem;line-height:1.55;color:var(--text);}}
.ctx-text.muted{{color:var(--muted);font-style:italic;}}
.ocr-row{{margin-top:8px;font-size:0.72rem;color:var(--muted);}}
.ocr-row code{{background:rgba(255,255,255,0.06);padding:1px 4px;border-radius:4px;font-family:'JetBrains Mono';}}
.ctx-imgs{{display:flex;gap:6px;flex-wrap:wrap;margin-top:6px;}}
.ctx-img{{width:72px;height:72px;object-fit:cover;border-radius:6px;cursor:zoom-in;border:1px solid var(--border);transition:transform .15s;}}
.ctx-img:hover{{transform:scale(1.06);border-color:var(--accent);}}
.post-link{{color:var(--accent2);font-size:0.68rem;text-decoration:none;font-weight:400;font-family:'JetBrains Mono';margin-left:4px;}}
.post-link:hover{{text-decoration:underline;}}
.ext-link{{margin-top:5px;font-size:0.72rem;color:var(--accent2);}}
.ext-link a{{color:var(--accent2);text-decoration:none;}}
.ext-link a:hover{{text-decoration:underline;}}

.options-col{{display:flex;flex-direction:column;}}
.options-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;padding:16px;align-items:start;}}
.noa-row{{padding:0 16px 14px;}}
.noa-btn{{
  width:100%;padding:9px;border-radius:10px;
  border:1.5px solid var(--border);background:var(--surface);
  color:var(--muted);font-family:'Space Grotesk';font-weight:600;
  font-size:0.84rem;cursor:pointer;transition:all .2s;
}}
.noa-btn:hover{{border-color:#FF6B6B;color:#FF6B6B;}}
.noa-btn.selected{{background:#FF6B6B;border-color:#FF6B6B;color:#fff;}}
.option{{display:flex;flex-direction:column;align-items:center;gap:8px;}}
.option img{{
  width:100%;border-radius:10px;cursor:zoom-in;
  border:2px solid var(--border);transition:border-color .2s,transform .1s;
}}
.option img:hover{{transform:scale(1.01);border-color:var(--accent);}}
.option.selected img{{border-color:var(--sel);box-shadow:0 0 0 3px rgba(124,106,255,0.35);}}

.sel-btn{{
  width:100%;padding:7px;border-radius:8px;
  border:1.5px solid var(--border);background:var(--surface);
  color:var(--muted);font-family:'Space Grotesk';font-weight:600;
  font-size:0.82rem;cursor:pointer;transition:all .2s;
}}
.sel-btn:hover{{border-color:var(--accent);color:var(--text);}}
.option.selected .sel-btn{{background:var(--accent);border-color:var(--accent);color:#fff;}}

/* Result modal */
#result-modal{{
  display:none;position:fixed;inset:0;background:rgba(0,0,0,0.85);
  z-index:200;justify-content:center;align-items:center;
}}
#result-modal.show{{display:flex;}}
.result-box{{
  background:var(--card);border:1px solid var(--border);border-radius:20px;
  padding:36px 40px;max-width:560px;width:95%;text-align:center;
}}
.result-box h2{{
  font-size:1.4rem;font-weight:700;margin-bottom:16px;
  background:linear-gradient(90deg,#7C6AFF,#4ECDC4);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
}}
.result-score{{font-size:3rem;font-weight:700;color:var(--correct);margin:12px 0;}}
.result-sub{{color:var(--muted);font-size:0.85rem;margin-bottom:20px;}}
.result-breakdown{{
  max-height:260px;overflow-y:auto;margin:16px 0;
  border:1px solid var(--border);border-radius:10px;text-align:left;
}}
.rb-row{{
  display:flex;align-items:center;gap:10px;padding:7px 12px;
  border-bottom:1px solid var(--border);font-size:0.78rem;
  font-family:'JetBrains Mono';
}}
.rb-row:last-child{{border-bottom:none;}}
.rb-row.correct{{background:rgba(0,184,148,0.07);}}
.rb-row.wrong{{background:rgba(255,107,107,0.07);}}
.rb-qnum{{width:34px;color:var(--muted);}}
.rb-icon{{width:18px;text-align:center;}}
.rb-detail{{flex:1;color:var(--text);}}
.rb-detail span{{color:var(--muted);}}
.result-btns{{display:flex;gap:10px;justify-content:center;margin-top:8px;}}
.result-close,.dl-btn{{
  border:none;border-radius:10px;
  padding:10px 22px;font-weight:700;font-size:0.85rem;cursor:pointer;
}}
.result-close{{background:linear-gradient(135deg,#7C6AFF,#4ECDC4);color:#fff;}}
.dl-btn{{background:var(--surface);border:1px solid var(--border);color:var(--text);}}
.dl-btn:hover{{border-color:var(--accent);}}

/* Image lightbox */
#img-modal{{
  display:none;position:fixed;inset:0;background:rgba(0,0,0,0.93);
  z-index:300;justify-content:center;align-items:center;cursor:zoom-out;
}}
#img-modal.show{{display:flex;}}
#img-modal img{{max-width:90vw;max-height:90vh;border-radius:12px;}}

@media(max-width:800px){{
  .item-body{{grid-template-columns:1fr;}}
  .context-col{{border-right:none;border-bottom:1px solid var(--border);max-height:none;}}
  .options-grid{{grid-template-columns:repeat(3,1fr);gap:8px;padding:10px;}}
}}
</style>
</head>
<body>

<div class="topbar">
  <div class="topbar-title">🎭 Meme Benchmark · Human Evaluation</div>
  <div class="topbar-right">
    <span class="progress-text" id="prog-text">0 / {total} completed</span>
    <button class="submit-btn" id="submit-btn" onclick="submitAll()">Submit</button>
  </div>
</div>
<div class="progress-bar-wrap"><div class="progress-bar" id="prog-bar"></div></div>

<div class="main">
  <div style="color:var(--muted);font-size:0.8rem;margin-bottom:20px;font-family:'JetBrains Mono';">
    Generated: {now} · {total} questions · For each question, select the image actually used as the meme reply.
  </div>
  {rows}
</div>

<div id="result-modal">
  <div class="result-box">
    <h2>Evaluation Complete 🎉</h2>
    <div class="result-score" id="res-score">—</div>
    <div class="result-sub" id="res-sub"></div>
    <div class="result-breakdown" id="res-breakdown"></div>
    <div class="result-btns">
      <button class="dl-btn" onclick="downloadJSON()">⬇ JSON</button>
      <button class="dl-btn" onclick="downloadCSV()">⬇ CSV</button>
      <button class="result-close" onclick="document.getElementById('result-modal').classList.remove('show')">Close</button>
    </div>
  </div>
</div>

<div id="img-modal" onclick="closeImg()">
  <img id="modal-img" src="" alt="">
</div>

<script>
const ANSWERS = {json.dumps({str(q["q_idx"]): q["answer_label"] for q in questions})};
const TOTAL   = {total};
const selected = {{}};

function selectAnswer(qIdx, label, btn) {{
  // Deselect all options and noa button for this question
  document.querySelectorAll(`[data-qidx="${{qIdx}}"]`).forEach(el => el.classList.remove('selected'));
  // Select current (option div or noa-btn)
  const target = btn.classList.contains('noa-btn') ? btn : btn.closest('.option');
  target.classList.add('selected');
  selected[qIdx] = label;

  // Mark as answered
  const item = document.getElementById('q' + qIdx);
  if (item.dataset.answered === '0') {{
    item.dataset.answered = '1';
    item.classList.add('answered');
    document.getElementById('badge-' + qIdx).style.display = 'inline';
  }}
  updateProgress();
}}

function updateProgress() {{
  const done = Object.keys(selected).length;
  document.getElementById('prog-text').textContent = done + ' / ' + TOTAL + ' completed';
  document.getElementById('prog-bar').style.width = (done / TOTAL * 100) + '%';
  const btn = document.getElementById('submit-btn');
  if (done === TOTAL) btn.classList.add('ready');
  else btn.classList.remove('ready');
}}

const META = {json.dumps({str(q["q_idx"]): {"uid": q["uid"], "none_of_above": q["none_of_above"]} for q in questions})};
let lastResults = [];

function submitAll() {{
  const done = Object.keys(selected).length;
  if (done < TOTAL) {{
    alert(`${{TOTAL - done}} question(s) remaining.`);
    return;
  }}

  let correct = 0;
  lastResults = [];
  let breakdownHTML = '';

  for (let qIdx = 1; qIdx <= TOTAL; qIdx++) {{
    const userLabel  = selected[qIdx];
    const rightLabel = ANSWERS[qIdx];
    const isCorrect  = userLabel === rightLabel;
    if (isCorrect) correct++;

    const meta = META[qIdx] || {{}};
    lastResults.push({{
      question:     qIdx,
      uid:          meta.uid || '',
      answer_type:  meta.answer_type || '',
      correct_label: rightLabel,
      selected_label: userLabel,
      is_correct:   isCorrect,
    }});

    const rightDisplay = rightLabel === 'N' ? 'None of the above' : rightLabel;
    const userDisplay  = userLabel  === 'N' ? 'None of the above' : userLabel;
    breakdownHTML += `
      <div class="rb-row ${{isCorrect ? 'correct' : 'wrong'}}">
        <span class="rb-qnum">Q${{qIdx}}</span>
        <span class="rb-icon">${{isCorrect ? '✅' : '❌'}}</span>
        <span class="rb-detail">
          Selected: <b>${{userDisplay}}</b>
          <span>· Correct: ${{rightDisplay}}</span>
          <span>· ${{meta.none_of_above ? "none_of_above" : "original"}}</span>
        </span>
      </div>`;
  }}

  document.getElementById('res-score').textContent = correct + ' / ' + TOTAL;
  document.getElementById('res-sub').textContent =
    `Accuracy ${{(correct/TOTAL*100).toFixed(1)}}% · ${{TOTAL-correct}} incorrect`;
  document.getElementById('res-breakdown').innerHTML = breakdownHTML;
  document.getElementById('result-modal').classList.add('show');
}}

function downloadJSON() {{
  const blob = new Blob([JSON.stringify(lastResults, null, 2)], {{type: 'application/json'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'eval_results.json';
  a.click();
}}

function downloadCSV() {{
  const header = 'question,uid,answer_type,correct_label,selected_label,is_correct';
  const rows = lastResults.map(r =>
    `${{r.question}},${{r.uid}},${{r.answer_type}},${{r.correct_label}},${{r.selected_label}},${{r.is_correct}}`
  );
  const blob = new Blob([[header, ...rows].join('\n')], {{type: 'text/csv'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'eval_results.csv';
  a.click();
}}

function openImg(src) {{
  document.getElementById('modal-img').src = src;
  document.getElementById('img-modal').classList.add('show');
}}
function closeImg() {{
  document.getElementById('img-modal').classList.remove('show');
}}
document.addEventListener('keydown', e => {{ if (e.key === 'Escape') closeImg(); }});
</script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[HTML] {output_path} ({total} questions)")


# ────────────────────────────────────────────────────────────────
def generate_answer_key(questions: list, output_path: Path):
    lines = [
        "# Meme Benchmark Answer Key",
        f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"# Total: {len(questions)} questions",
        "# Format: Q<num> TAB answer_label TAB answer_type TAB uid",
        "",
    ]
    for q in questions:
        noa = "none_of_above" if q["none_of_above"] else "original"
        lines.append(
            f"Q{q['q_idx']}\t{q['answer_label']}\t{noa}\t{q['uid']}"
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[KEY]  {output_path}")


# ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Generate human evaluation set for meme benchmark")
    parser.add_argument("--input",  default="./benchmark_data", help="benchmark_data directory")
    parser.add_argument("--n",      type=int, default=100,      help="number of samples (default: 100)")
    parser.add_argument("--output", default=None,               help="output directory (default: ./eval_N)")
    parser.add_argument("--seed",   type=int, default=42,       help="random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    input_dir  = Path(args.input)
    output_dir = Path(args.output) if args.output else Path(f"./eval_{args.n}")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = input_dir / "benchmark_summary.jsonl"
    if not summary.exists():
        print(f"[ERROR] {summary} not found")
        return

    # Load all metas
    all_metas = []
    with open(summary, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    all_metas.append(json.loads(line))
                except Exception:
                    pass
    print(f"[LOAD] {len(all_metas)} items loaded")

    # Sample N
    pool = all_metas if len(all_metas) <= args.n else random.sample(all_metas, args.n)
    random.shuffle(pool)
    print(f"[SAMPLE] {len(pool)} selected")

    # Build questions
    questions = []
    skipped   = 0
    for meta in pool:
        uid      = meta.get("uid", "")
        item_dir = input_dir / uid
        q = build_question(meta, item_dir, len(questions) + 1)
        if q:
            questions.append(q)
        else:
            skipped += 1

    print(f"[BUILD] {len(questions)} questions built (skipped: {skipped})")

    if not questions:
        print("[ERROR] No questions built — check image files.")
        return

    # Output
    html_path   = output_dir / "eval.html"
    answer_path = output_dir / "answers.txt"

    generate_html(questions, html_path)
    generate_answer_key(questions, answer_path)

    print(f"\n{'='*50}")
    print(f"  Questions: {len(questions)}")
    print(f"  HTML:      {html_path.resolve()}")
    print(f"  Answers:   {answer_path.resolve()}")
    print(f"{'='*50}")
    print(f"\nOpen with: python -m http.server 8080 --directory {output_dir.resolve()}")
    print(f"           → http://localhost:8080/eval.html")


if __name__ == "__main__":
    main()

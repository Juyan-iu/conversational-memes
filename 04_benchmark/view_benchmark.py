#!/usr/bin/env python3
"""
벤치마크 결과 HTML 뷰어

사용법:
  python view_benchmark.py
  python view_benchmark.py --input ./benchmark_data --output ./benchmark_viewer.html
"""

import json
import argparse
import base64
from pathlib import Path
from datetime import datetime


def img_to_b64(path: Path) -> str | None:
    if path and path.exists():
        try:
            return base64.b64encode(path.read_bytes()).decode()
        except Exception:
            pass
    return None


def render_item(meta: dict, item_dir: Path, idx: int) -> str:
    uid      = meta.get("uid", "unknown")
    month    = meta.get("month", "")
    answer   = meta.get("answer", "A")
    context  = meta.get("context", {})
    labels   = meta.get("labels", {})
    options  = meta.get("options", {})
    captions = meta.get("captions", [])

    orig_text   = context.get("original_post", "")
    parent_text = context.get("parent_reply", "")
    meme_text   = context.get("meme_text", "")

    # 담화 라벨
    meme_label = (labels.get("meme_reply") or {}).get("discourse_function", "—")
    stance     = (labels.get("meme_reply") or {}).get("stance") or {}
    visual_desc= ((labels.get("meme_reply") or {}).get("visual") or {}).get("visual_description", "")

    stance_html = ""
    for k, color in [("sarcastic","#FF6B6B"), ("humorous","#FFD93D"), ("offensive","#6C5CE7")]:
        v = stance.get(k, False)
        op = "1" if v else "0.3"
        stance_html += f'<span class="stance-tag" style="background:{color};opacity:{op}">{"✓" if v else "✗"} {k.capitalize()}</span>'

    # 보기 이미지
    option_labels = {"A": "✅ Original (정답)", "B": "📝 Text Distractor",
                     "C": "🖼 Visual Distractor", "D": "🎲 Easy Distractor"}
    options_html = ""
    shuffled = list(options.items())
    for key, fname in shuffled:
        if not fname:
            continue
        b64 = img_to_b64(item_dir / fname)
        if not b64:
            continue
        is_answer = key == answer
        border = "3px solid #00b894" if is_answer else "1px solid #2A2F42"
        options_html += f"""
        <div class="option {'correct' if is_answer else ''}">
            <div class="option-label">{option_labels.get(key, key)}</div>
            <img src="data:image/jpeg;base64,{b64}"
                 style="width:100%;border-radius:8px;cursor:pointer;border:{border};"
                 onclick="openImg(this.src)">
        </div>"""

    # OCR 캡션
    ocr_html = ""
    if captions:
        ocr_items = " · ".join(f'<code>{c["text"]}</code>' for c in captions[:5])
        ocr_html = f'<div class="ocr-row">🔤 OCR: {ocr_items}</div>'

    return f"""
    <div class="item" id="item-{idx}">
        <div class="item-header">
            <span class="item-idx">#{idx}</span>
            <span class="uid">{uid[:24]}...</span>
            <span class="month-badge">📅 {month}</span>
            <span class="disc-chip">{meme_label}</span>
            {stance_html}
        </div>

        <div class="item-body">
            <div class="context-col">
                <div class="sec-label">📌 원 포스트</div>
                <div class="ctx-text">{orig_text[:200]}{"..." if len(orig_text)>200 else ""}</div>
                {f'<div class="sec-label">↩ 부모 댓글</div><div class="ctx-text">{parent_text[:120]}</div>' if parent_text else ""}
                {f'<div class="sec-label">🎭 밈 텍스트</div><div class="ctx-text muted">{meme_text[:80]}</div>' if meme_text else ""}
                {f'<div class="sec-label">🎨 Visual</div><div class="ctx-text muted">{visual_desc[:150]}</div>' if visual_desc else ""}
                {ocr_html}
            </div>
            <div class="options-grid">
                {options_html}
            </div>
        </div>
    </div>"""


def generate_html(items: list, input_dir: Path, output_path: str):
    total = len(items)
    now   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows  = "".join(render_item(meta, input_dir / meta["uid"], i+1)
                    for i, meta in enumerate(items))

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Benchmark Viewer</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{{
  --bg:#0D0F14;--surface:#161920;--card:#1E2230;
  --border:#2A2F42;--accent:#7C6AFF;--accent2:#4ECDC4;
  --text:#E8EAF0;--muted:#6B7280;--correct:#00b894;
}}
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{background:var(--bg);color:var(--text);font-family:'Space Grotesk',sans-serif;padding-bottom:80px;}}

.topbar{{background:linear-gradient(135deg,#161920,#1a1230);border-bottom:1px solid var(--border);
  padding:16px 28px;display:flex;align-items:center;justify-content:space-between;
  position:sticky;top:0;z-index:100;}}
.topbar-title{{font-size:1.05rem;font-weight:700;
  background:linear-gradient(90deg,#7C6AFF,#4ECDC4);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}}
.topbar-meta{{color:var(--muted);font-size:0.78rem;font-family:'JetBrains Mono';}}

.filterbar{{padding:10px 24px;display:flex;gap:8px;align-items:center;flex-wrap:wrap;
  border-bottom:1px solid var(--border);background:var(--surface);}}
.filter-input{{background:var(--card);border:1px solid var(--border);color:var(--text);
  border-radius:8px;padding:6px 12px;font-family:'JetBrains Mono';font-size:0.78rem;width:220px;outline:none;}}
.filter-input:focus{{border-color:var(--accent);}}
.filter-btn{{background:var(--card);border:1px solid var(--border);color:var(--muted);
  border-radius:8px;padding:6px 10px;font-size:0.74rem;cursor:pointer;transition:all .2s;}}
.filter-btn:hover,.filter-btn.active{{background:var(--accent);color:#fff;border-color:var(--accent);}}
.stat-chip{{margin-left:auto;color:var(--muted);font-family:'JetBrains Mono';font-size:0.74rem;}}

.main{{max-width:1200px;margin:0 auto;padding:24px 18px;}}

.item{{background:var(--card);border:1px solid var(--border);border-radius:16px;
  margin-bottom:28px;overflow:hidden;}}
.item-header{{padding:12px 16px;background:var(--surface);border-bottom:1px solid var(--border);
  display:flex;align-items:center;gap:8px;flex-wrap:wrap;}}
.item-idx{{background:var(--accent);color:#fff;font-weight:700;font-size:0.75rem;
  padding:2px 8px;border-radius:6px;font-family:'JetBrains Mono';}}
.uid{{font-family:'JetBrains Mono';font-size:0.7rem;color:var(--muted);}}
.month-badge{{font-size:0.7rem;background:#2d3a4a;color:#74b9ff;padding:2px 7px;border-radius:8px;font-family:'JetBrains Mono';}}
.disc-chip{{font-family:'JetBrains Mono';font-size:0.72rem;background:rgba(124,106,255,0.15);
  color:#a78bfa;padding:2px 8px;border-radius:10px;}}
.stance-tag{{font-size:0.7rem;font-weight:600;padding:2px 8px;border-radius:10px;color:#fff;}}

.item-body{{display:grid;grid-template-columns:280px 1fr;gap:0;}}
.context-col{{padding:14px;border-right:1px solid var(--border);background:var(--surface);}}
.sec-label{{font-size:0.68rem;font-weight:600;color:var(--accent2);margin:8px 0 3px;font-family:'JetBrains Mono';}}
.ctx-text{{font-size:0.8rem;line-height:1.5;color:var(--text);}}
.ctx-text.muted{{color:var(--muted);font-style:italic;}}
.ocr-row{{margin-top:8px;font-size:0.72rem;color:var(--muted);}}
.ocr-row code{{background:rgba(255,255,255,0.06);padding:1px 4px;border-radius:4px;font-family:'JetBrains Mono';}}

.options-grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:14px;}}
.option{{display:flex;flex-direction:column;gap:6px;}}
.option-label{{font-size:0.72rem;font-weight:600;color:var(--muted);font-family:'JetBrains Mono';}}
.option.correct .option-label{{color:var(--correct);}}

#img-modal{{display:none;position:fixed;inset:0;background:rgba(0,0,0,0.92);
  z-index:999;justify-content:center;align-items:center;cursor:zoom-out;}}
#img-modal.show{{display:flex;}}
#img-modal img{{max-width:90vw;max-height:90vh;border-radius:12px;}}

.hidden{{display:none!important;}}

@media(max-width:700px){{
  .item-body{{grid-template-columns:1fr;}}
  .options-grid{{grid-template-columns:1fr 1fr;}}
  .context-col{{border-right:none;border-bottom:1px solid var(--border);}}
}}
</style>
</head>
<body>

<div class="topbar">
  <div class="topbar-title">🎭 Benchmark Viewer</div>
  <div class="topbar-meta">{total}개 · {now}</div>
</div>

<div class="filterbar">
  <input class="filter-input" id="search" placeholder="🔍 uid, 텍스트, 라벨 검색..." oninput="applyFilter()">
  <button class="filter-btn" onclick="filterStance('sarcastic')">😏 Sarcastic</button>
  <button class="filter-btn" onclick="filterStance('humorous')">😄 Humorous</button>
  <button class="filter-btn" onclick="filterStance('offensive')">⚠️ Offensive</button>
  <button class="filter-btn" onclick="clearFilter()">✕ 초기화</button>
  <span class="stat-chip" id="stat-chip">{total}개 표시 중</span>
</div>

<div class="main">
  {rows}
</div>

<div id="img-modal" onclick="closeImg()">
  <img id="modal-img" src="" alt="">
</div>

<script>
  let activeStance = null;
  function openImg(src){{document.getElementById('modal-img').src=src;document.getElementById('img-modal').classList.add('show');}}
  function closeImg(){{document.getElementById('img-modal').classList.remove('show');}}
  document.addEventListener('keydown',e=>{{if(e.key==='Escape')closeImg();}});
  function filterStance(s){{
    activeStance=(activeStance===s)?null:s;
    document.querySelectorAll('.filter-btn').forEach(b=>b.classList.remove('active'));
    if(activeStance)event.target.classList.add('active');
    applyFilter();
  }}
  function clearFilter(){{
    activeStance=null;
    document.getElementById('search').value='';
    document.querySelectorAll('.filter-btn').forEach(b=>b.classList.remove('active'));
    applyFilter();
  }}
  function applyFilter(){{
    const q=document.getElementById('search').value.toLowerCase();
    let v=0;
    document.querySelectorAll('.item').forEach(item=>{{
      const t=item.innerText.toLowerCase();
      const ms=!q||t.includes(q);
      let mf=true;
      if(activeStance)mf=t.includes(activeStance);
      if(ms&&mf){{item.classList.remove('hidden');v++;}}
      else item.classList.add('hidden');
    }});
    document.getElementById('stat-chip').textContent=v+'개 표시 중';
  }}
</script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[DONE] {output_path} ({total}개)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="./benchmark_data")
    parser.add_argument("--output", default="./benchmark_viewer.html")
    args = parser.parse_args()

    input_dir = Path(args.input)
    summary   = input_dir / "benchmark_summary.jsonl"

    if not summary.exists():
        print(f"[ERROR] {summary} 없음")
        return

    items = []
    with open(summary, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    items.append(json.loads(line))
                except Exception:
                    pass
    print(f"[LOAD] {len(items)}개 로드")
    generate_html(items, input_dir, args.output)


if __name__ == "__main__":
    main()

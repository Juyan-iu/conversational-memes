#!/usr/bin/env python3
"""
여러 모델 결과를 나란히 비교하는 HTML 뷰어

사용법:
  python compare_results.py \
    --inputs gpt4o=./results_4o/labeled_memes.jsonl \
             mini=./results_54mini/labeled_memes.jsonl \
             nano=./results_54nano/labeled_memes.jsonl \
    --output ./compare.html
"""

import json
import re
import argparse
from datetime import datetime


def load_records(jsonl_path: str) -> dict:
    records = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                    records[r.get("uid", "")] = r
                except Exception:
                    pass
    return records


def get_discourse_label(record: dict, key: str) -> str:
    labels = record.get("discourse_labels", {}) or {}
    label = (labels.get(key) or {}).get("discourse_function", "")
    return label or "—"


def get_stance(record: dict) -> str:
    labels = record.get("discourse_labels", {}) or {}
    stance = (labels.get("meme_reply") or {}).get("stance") or {}
    if not stance:
        return "—"
    parts = []
    if stance.get("sarcastic"): parts.append("😏 Sarcastic")
    if stance.get("humorous"):  parts.append("😄 Humorous")
    if stance.get("offensive"): parts.append("⚠️ Offensive")
    return " · ".join(parts) if parts else "None"


def get_visual(record: dict) -> str:
    labels = record.get("discourse_labels", {}) or {}
    visual = (labels.get("meme_reply") or {}).get("visual") or {}
    return visual.get("visual_description", "") or "—"


def get_valid(record: dict) -> tuple:
    v = record.get("meme_validation", {})
    passed = v.get("passed", False)
    ratio  = v.get("valid_ratio", 0)
    tmpl   = ""
    for val in v.get("validations", []):
        t = val.get("template_name")
        if t and t not in ("null", "None", None):
            tmpl = t
            break
    return passed, ratio, tmpl


def get_meme_img_url(record: dict) -> str:
    meme = record.get("meme_reply", {}) or {}
    for img in (meme.get("images", []) or []):
        url = img.get("url") or img.get("source_url", "")
        if url:
            return url
    return ""


def post_imgs_html(post: dict, h: int = 100) -> str:
    if not post:
        return ""
    imgs = post.get("images", []) or []
    tags = []
    for img in imgs:
        url = img.get("url") or img.get("source_url", "")
        alt = (img.get("alt") or "").strip()
        if not url:
            continue
        alt_h = '<div class="img-alt">💬 ' + alt + '</div>' if alt else ""
        tags.append(
            '<div class="img-wrap">'
            '<img src="' + url + '" style="max-height:' + str(h) + 'px;'
            'border-radius:6px;object-fit:contain;cursor:pointer;background:#1a1f2e;" '
            'onclick="openImg(\'' + url + '\')" onerror="this.style.display=\'none\'">'
            + alt_h + '</div>'
        )
    return '<div class="img-row">' + "".join(tags) + '</div>' if tags else ""


# ── 2레벨 밈 댓글 라벨 전체 설명 테이블 ──────────────────────────
MEME_LABELS = [
    ("Open.Attend",    "Greeting or response to greeting at the start of a conversation"),
    ("Open.Command",   "Request, invitation, or command to initiate a dialog"),
    ("Open.Demand",    "Question demanding information or opinion at conversation start"),
    ("Open.Give",      "Providing factual or evaluative information to open a conversation"),
    ("React.Rejoinder","Detailed comment, follow-up question, or emotional reaction to a previous utterance"),
    ("React.Respond",  "Direct positive or negative response to a previous utterance"),
    ("Sustain.Continue","Same speaker continues their own preceding statement (rare in meme replies)"),
]

LABEL_REF_HTML = """
<div class="label-ref">
  <div class="label-ref-title">📖 Meme Reply Discourse Function Labels (2-level)</div>
  <table class="label-ref-table">
    <thead><tr><th>Label</th><th>Description</th></tr></thead>
    <tbody>
""" + "".join(
    f'<tr><td><span class="disc-chip">{lbl}</span></td><td class="ref-desc">{desc}</td></tr>'
    for lbl, desc in MEME_LABELS
) + """
    </tbody>
  </table>
  <div class="label-ref-note">
    <b>Stance</b> (independent binary labels per meme reply):<br>
    😏 <b>Sarcastic</b> — ironic or mocking tone &nbsp;|&nbsp;
    😄 <b>Humorous</b> — comedic or playful &nbsp;|&nbsp;
    ⚠️ <b>Offensive</b> — aggressive or harmful content
  </div>
</div>"""


def render_row(uid: str, models: dict, model_names: list) -> str:
    first = next((models[n].get(uid) for n in model_names if uid in models[n]), None)
    if not first:
        return ""

    orig_post   = first.get("original_post") or {}
    orig_text   = orig_post.get("text") or ""
    orig_imgs   = post_imgs_html(orig_post, h=85)
    parent      = first.get("parent_reply")
    parent_text = (parent or {}).get("text") or ""
    parent_imgs = post_imgs_html(parent, h=75) if parent else ""
    meme_reply  = first.get("meme_reply") or {}
    meme_text   = meme_reply.get("text") or ""
    meme_img_url= get_meme_img_url(first)
    thread      = (first.get("thread_structure") or {}).get("label", "")

    # bsky 링크
    bsky_link = ""
    post_uri = orig_post.get("uri", "")
    m = re.match(r"at://([^/]+)/[^/]+/([^/]+)", post_uri)
    if m:
        did, rkey = m.group(1), m.group(2)
        bsky_link = f'<a href="https://bsky.app/profile/{did}/post/{rkey}" target="_blank" class="bsky-link">🔗 원본</a>'

    meme_img_html = (
        '<img src="' + meme_img_url + '" '
        'style="max-height:140px;border-radius:8px;object-fit:contain;cursor:pointer;background:#1a1f2e;" '
        'onclick="openImg(\'' + meme_img_url + '\')" onerror="this.style.display=\'none\'">'
        if meme_img_url else '<span class="no-img">이미지 없음</span>'
    )

    # 모델별 컬럼 생성
    model_cols = ""
    for name in model_names:
        rec = models[name].get(uid)
        if not rec:
            model_cols += '<td class="model-col missing">데이터 없음</td>'
            continue

        passed, ratio, tmpl = get_valid(rec)
        valid_icon  = "✅" if passed else "❌"
        meme_lbl    = get_discourse_label(rec, "meme_reply")
        stance      = get_stance(rec)
        visual      = get_visual(rec)

        tmpl_html = f'<span class="tmpl-badge">🃏 {tmpl}</span>' if tmpl else ""

        model_cols += f"""
        <td class="model-col {'passed' if passed else 'failed'}">
            <div class="val-row">
                <span class="val-icon">{valid_icon}</span>
                <span class="val-ratio">{ratio:.0%}</span>
                {tmpl_html}
            </div>
            <div class="disc-row">
                <span class="disc-label">Discourse</span>
                <span class="disc-chip {'meme-chip' if meme_lbl != '—' else ''}">{meme_lbl}</span>
            </div>
            <div class="stance-row">
                <span class="disc-label">Stance</span>
                <span class="stance-val">{stance}</span>
            </div>
            <div class="visual-row">
                <span class="disc-label">Visual</span>
                <span class="visual-val">{visual[:130]}{"..." if len(visual)>130 else ""}</span>
            </div>
        </td>"""

    parent_section = ""
    if parent:
        parent_section = (
            '<div class="sec-label">↩ 부모 댓글</div>'
            '<div class="post-text">' + parent_text[:100] + ("..." if len(parent_text)>100 else "") + '</div>'
            + parent_imgs
        )

    return f"""
    <tr class="rec-row">
        <td class="info-col">
            <div class="uid-row">
                <span class="uid-text">{uid[:22]}...</span>
                <span class="thread-badge">{thread}</span>
                {bsky_link}
            </div>
            <div class="sec-label">📌 원 포스트</div>
            <div class="post-text">{orig_text[:120]}{"..." if len(orig_text)>120 else ""}</div>
            {orig_imgs}
            {parent_section}
            <div class="sec-label">🎭 밈 댓글</div>
            <div class="meme-text">{meme_text[:80]}{"..." if len(meme_text)>80 else ""}</div>
            {meme_img_html}
        </td>
        {model_cols}
    </tr>"""


def generate_html(models: dict, model_names: list, output_path: str):
    all_uids = set()
    for n in model_names:
        all_uids.update(models[n].keys())
    uids = sorted(all_uids)

    rows = "".join(render_row(uid, models, model_names) for uid in uids)

    stats = {}
    for n in model_names:
        recs   = list(models[n].values())
        total  = len(recs)
        passed = sum(1 for r in recs if (r.get("meme_validation") or {}).get("passed", False))
        stats[n] = (total, passed)

    stat_cols = "".join(
        f'<th>{n}<br><span class="stat-sub">{stats[n][1]}/{stats[n][0]} passed</span></th>'
        for n in model_names
    )

    total = len(uids)
    now   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Model Comparison – Meme Discourse Labels</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{{
  --bg:#0D0F14;--surface:#161920;--card:#1E2230;
  --border:#2A2F42;--accent:#7C6AFF;--accent2:#4ECDC4;
  --text:#E8EAF0;--muted:#6B7280;--pass:#00b894;--fail:#e17055;
}}
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{background:var(--bg);color:var(--text);font-family:'Space Grotesk',sans-serif;min-height:100vh;}}

/* topbar */
.topbar{{background:linear-gradient(135deg,#161920,#1a1230);border-bottom:1px solid var(--border);
  padding:16px 28px;display:flex;align-items:center;justify-content:space-between;
  position:sticky;top:0;z-index:200;}}
.topbar-title{{font-size:1.05rem;font-weight:700;
  background:linear-gradient(90deg,#7C6AFF,#4ECDC4);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}}
.topbar-meta{{color:var(--muted);font-size:0.76rem;font-family:'JetBrains Mono';}}

/* label ref */
.label-ref{{margin:16px 24px;background:var(--card);border:1px solid var(--border);
  border-radius:12px;padding:14px 18px;}}
.label-ref-title{{font-size:0.82rem;font-weight:600;color:var(--accent2);margin-bottom:10px;}}
.label-ref-table{{width:100%;border-collapse:collapse;font-size:0.78rem;}}
.label-ref-table th{{background:var(--surface);color:var(--muted);padding:5px 10px;
  text-align:left;border-bottom:1px solid var(--border);font-weight:500;}}
.label-ref-table td{{padding:5px 10px;border-bottom:1px solid rgba(255,255,255,0.04);vertical-align:middle;}}
.ref-desc{{color:#9CA3AF;}}
.disc-chip{{font-family:'JetBrains Mono';font-size:0.72rem;background:rgba(124,106,255,0.15);
  color:#a78bfa;padding:2px 8px;border-radius:10px;white-space:nowrap;}}
.label-ref-note{{margin-top:10px;font-size:0.76rem;color:var(--muted);line-height:1.6;
  border-top:1px solid var(--border);padding-top:8px;}}

/* filterbar */
.filterbar{{padding:10px 24px;display:flex;gap:8px;align-items:center;flex-wrap:wrap;
  border-bottom:1px solid var(--border);background:var(--surface);}}
.filter-input{{background:var(--card);border:1px solid var(--border);color:var(--text);
  border-radius:8px;padding:6px 12px;font-family:'JetBrains Mono';font-size:0.78rem;width:220px;outline:none;}}
.filter-input:focus{{border-color:var(--accent);}}
.filter-btn{{background:var(--card);border:1px solid var(--border);color:var(--muted);
  border-radius:8px;padding:6px 10px;font-size:0.74rem;cursor:pointer;transition:all .2s;}}
.filter-btn:hover,.filter-btn.active{{background:var(--accent);color:#fff;border-color:var(--accent);}}
.stat-chip{{margin-left:auto;color:var(--muted);font-family:'JetBrains Mono';font-size:0.74rem;}}

/* table */
.main{{padding:16px 24px;overflow-x:auto;}}
table{{border-collapse:collapse;width:100%;min-width:900px;}}
thead th{{background:var(--surface);border:1px solid var(--border);padding:10px 14px;
  text-align:left;font-size:0.84rem;font-weight:600;white-space:nowrap;
  position:sticky;top:57px;z-index:100;}}
thead th:first-child{{min-width:240px;}}
thead th:not(:first-child){{min-width:240px;background:rgba(124,106,255,0.08);border-top:3px solid var(--accent);}}
.stat-sub{{font-size:0.68rem;color:var(--accent2);font-family:'JetBrains Mono';font-weight:400;}}

.rec-row{{border-bottom:1px solid var(--border);}}
.rec-row:hover td{{background:rgba(255,255,255,0.015);}}

/* info col */
.info-col{{background:var(--surface);border:1px solid var(--border);padding:12px 14px;vertical-align:top;}}
.uid-row{{display:flex;align-items:center;gap:8px;margin-bottom:6px;flex-wrap:wrap;}}
.uid-text{{font-family:'JetBrains Mono';font-size:0.68rem;color:var(--muted);}}
.thread-badge{{font-size:0.68rem;background:#2d3748;color:#a0aec0;padding:2px 6px;border-radius:8px;}}
.bsky-link{{font-size:0.68rem;color:var(--accent2);text-decoration:none;
  padding:2px 7px;border:1px solid rgba(78,205,196,0.3);border-radius:6px;}}
.bsky-link:hover{{background:rgba(78,205,196,0.1);}}
.sec-label{{font-size:0.68rem;font-weight:600;color:var(--accent2);
  margin:7px 0 3px;font-family:'JetBrains Mono';}}
.post-text{{font-size:0.8rem;color:var(--text);line-height:1.5;margin-bottom:4px;}}
.meme-text{{font-size:0.76rem;color:var(--muted);font-style:italic;margin-bottom:6px;}}
.img-row{{display:flex;gap:6px;flex-wrap:wrap;margin:4px 0 6px;}}
.img-wrap{{display:flex;flex-direction:column;gap:2px;}}
.img-alt{{font-size:0.66rem;color:var(--muted);font-style:italic;}}
.no-img{{font-size:0.72rem;color:var(--muted);}}

/* model col */
.model-col{{border:1px solid var(--border);padding:12px 14px;vertical-align:top;}}
.model-col.passed{{border-left:3px solid var(--pass);}}
.model-col.failed{{border-left:3px solid var(--fail);opacity:0.7;}}
.model-col.missing{{color:var(--muted);font-size:0.76rem;font-style:italic;text-align:center;padding-top:28px;}}
.val-row{{display:flex;align-items:center;gap:6px;margin-bottom:8px;}}
.val-icon{{font-size:1rem;}}
.val-ratio{{font-family:'JetBrains Mono';font-size:0.78rem;color:var(--muted);}}
.tmpl-badge{{font-size:0.68rem;background:#2d4a3e;color:#68d391;padding:2px 6px;border-radius:8px;}}
.disc-row,.stance-row,.visual-row{{display:flex;gap:6px;align-items:flex-start;margin-bottom:6px;}}
.disc-label{{font-size:0.68rem;color:var(--muted);font-family:'JetBrains Mono';min-width:54px;padding-top:2px;}}
.meme-chip{{background:rgba(124,106,255,0.15);color:#a78bfa;padding:2px 9px;
  border-radius:10px;font-family:'JetBrains Mono';font-size:0.72rem;}}
.stance-val{{font-size:0.76rem;color:var(--text);}}
.visual-val{{font-size:0.74rem;color:#9CA3AF;line-height:1.5;font-style:italic;}}

/* modal */
#img-modal{{display:none;position:fixed;inset:0;background:rgba(0,0,0,0.9);
  z-index:999;justify-content:center;align-items:center;cursor:zoom-out;}}
#img-modal.show{{display:flex;}}
#img-modal img{{max-width:90vw;max-height:90vh;border-radius:12px;}}

.hidden{{display:none!important;}}
</style>
</head>
<body>

<div class="topbar">
  <div class="topbar-title">🔬 Model Comparison — Meme Discourse Labels</div>
  <div class="topbar-meta">{total}개 · {now}</div>
</div>

{LABEL_REF_HTML}

<div class="filterbar">
  <input class="filter-input" id="search" placeholder="🔍 uid, 텍스트, 라벨 검색..." oninput="applyFilter()">
  <button class="filter-btn" onclick="filterPassed()">✅ passed only</button>
  <button class="filter-btn" onclick="filterSarcastic()">😏 Sarcastic</button>
  <button class="filter-btn" onclick="filterHumorous()">😄 Humorous</button>
  <button class="filter-btn" onclick="filterOffensive()">⚠️ Offensive</button>
  <button class="filter-btn" onclick="clearFilter()">✕ 초기화</button>
  <span class="stat-chip" id="stat-chip">{total}개 표시 중</span>
</div>

<div class="main">
  <table id="cmp-table">
    <thead>
      <tr>
        <th>공통 정보</th>
        {stat_cols}
      </tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
</div>

<div id="img-modal" onclick="closeImg()">
  <img id="modal-img" src="" alt="">
</div>

<script>
  let activeFilter = null;

  function openImg(url) {{
    document.getElementById('modal-img').src = url;
    document.getElementById('img-modal').classList.add('show');
  }}
  function closeImg() {{ document.getElementById('img-modal').classList.remove('show'); }}
  document.addEventListener('keydown', e => {{ if(e.key==='Escape') closeImg(); }});

  function setFilter(key) {{
    activeFilter = (activeFilter===key) ? null : key;
    document.querySelectorAll('.filter-btn').forEach(b=>b.classList.remove('active'));
    if(activeFilter) event.target.classList.add('active');
    applyFilter();
  }}
  function filterPassed()   {{ setFilter('passed'); }}
  function filterSarcastic(){{ setFilter('sarcastic'); }}
  function filterHumorous() {{ setFilter('humorous'); }}
  function filterOffensive(){{ setFilter('offensive'); }}
  function clearFilter() {{
    activeFilter=null;
    document.getElementById('search').value='';
    document.querySelectorAll('.filter-btn').forEach(b=>b.classList.remove('active'));
    applyFilter();
  }}
  function applyFilter() {{
    const q = document.getElementById('search').value.toLowerCase();
    let visible=0;
    document.querySelectorAll('.rec-row').forEach(row=>{{
      const text=row.innerText.toLowerCase();
      const matchSearch=!q||text.includes(q);
      let matchFilter=true;
      if(activeFilter==='passed') matchFilter=text.includes('✅');
      else if(activeFilter) matchFilter=text.toLowerCase().includes(activeFilter);
      if(matchSearch&&matchFilter){{row.classList.remove('hidden');visible++;}}
      else row.classList.add('hidden');
    }});
    document.getElementById('stat-chip').textContent=visible+'개 표시 중';
  }}
</script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[DONE] {output_path} ({total}개 레코드)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="모델명=경로. 예: gpt4o=./r1/labeled.jsonl")
    parser.add_argument("--output", default="./compare.html")
    args = parser.parse_args()

    model_names = []
    models = {}
    for item in args.inputs:
        name, path = item.split("=", 1)
        print(f"[LOAD] {name}: {path}")
        models[name] = load_records(path)
        model_names.append(name)
        print(f"  {len(models[name])}개 레코드")

    generate_html(models, model_names, args.output)


if __name__ == "__main__":
    main()

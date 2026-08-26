#!/usr/bin/env python3
"""
labeled_memes.jsonl → HTML 뷰어 생성

사용법:
  python view_results.py
  python view_results.py --input ./labeled_dataset/labeled_memes.jsonl
  python view_results.py --output ./labeled_viewer.html
"""

import json
import argparse
from datetime import datetime


def load_records(jsonl_path: str) -> list:
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception as e:
                    print(f"[WARN] 파싱 실패: {e}")
    return records


def stance_badges(stance: dict) -> str:
    if not stance:
        return ""
    badges = []
    colors = {
        "sarcastic": ("#FF6B6B", "#fff"),
        "humorous":  ("#FFD93D", "#333"),
        "offensive": ("#6C5CE7", "#fff"),
    }
    for key, (bg, fg) in colors.items():
        val = stance.get(key, False)
        opacity = "1" if val else "0.25"
        badges.append(
            f'<span class="stance-badge" style="background:{bg};color:{fg};opacity:{opacity}">'
            f'{"✓" if val else "✗"} {key.capitalize()}</span>'
        )
    return "".join(badges)


def discourse_chip(label: str, label_type: str) -> str:
    if not label:
        return ""
    color = "#7C6AFF" if "2level" in (label_type or "") else "#45B7D1"
    tag = "2L+Stance" if "2level" in (label_type or "") else "3L"
    return (
        f'<span class="discourse-chip" style="background:{color}">{label}</span>'
        f'<span class="label-type">{tag}</span>'
    )


def get_images_html(post: dict, downloaded: dict, key: str) -> str:
    """이미지 태그 생성 - 로컬 경로 우선, alt 텍스트 표시"""
    if not post:
        return ""
    images = post.get("images", []) or []
    if not images:
        return ""

    # downloaded_images에서 local_path 매핑
    local_map = {}
    if downloaded and key in downloaded:
        for d in (downloaded.get(key) or []):
            cid = d.get("cid", "")
            if cid:
                local_map[cid] = d.get("local_path", "")

    tags = []
    for img in images:
        url = img.get("url") or img.get("source_url", "")
        cid = img.get("cid", "")
        alt = (img.get("alt") or "").strip()
        local = local_map.get(cid, "")
        src = local if local else url
        if not src:
            continue

        alt_html = f'<div class="img-alt">💬 {alt}</div>' if alt else ""
        # 로컬 없으면 URL 직접 사용, URL도 없으면 broken placeholder
        src_local = local if local else ""
        src_url   = url if url else ""
        tags.append(f"""
        <div class="img-wrap">
            <img src="{src_local or src_url}" alt="{alt}"
                style="max-height:160px;max-width:260px;border-radius:8px;object-fit:contain;cursor:pointer;display:block;background:#1a1f2e;"
                onclick="openImg(this.src, '{alt}')"
                onerror="if(this.dataset.tried!=='1'){{this.dataset.tried='1';this.src='{src_url}';}}else{{this.style.display='none';this.nextElementSibling&&(this.nextElementSibling.style.display='block');}}">
            <div class="img-err" style="display:none">⚠️ 이미지 로드 실패</div>
            {alt_html}
        </div>""")

    return f'<div class="img-row">{"".join(tags)}</div>' if tags else ""


def render_unit(key: str, post: dict, label: dict, title: str,
                downloaded: dict, is_meme: bool = False) -> str:
    if not post and not label:
        return ""

    text        = (post or {}).get("text") or ""
    like_count  = (post or {}).get("like_count", 0)
    reply_count = (post or {}).get("reply_count", 0)
    imgs_html   = get_images_html(post, downloaded, key)

    disc_func  = (label or {}).get("discourse_function", "")
    disc_type  = (label or {}).get("label_type", "")
    disc_path  = (label or {}).get("discourse_path", [])
    stance     = (label or {}).get("stance")
    visual     = (label or {}).get("visual")

    meme_class = " meme-unit" if is_meme else ""
    icon = "🎭" if is_meme else "💬"

    # 분기 경로 - 답변만 표시 (질문 제외)
    path_html = ""
    if disc_path:
        arrows = []
        for step in disc_path:
            a = (step.get("answer") or "")[:80]
            if a:
                arrows.append(f'<span class="path-arrow">→ {a}</span>')
        if arrows:
            path_html = f'<div class="path-block">{" ".join(arrows)}</div>'

    # visual description
    visual_html = ""
    if visual:
        desc = visual.get("visual_description", "")
        if desc:
            visual_html = f'''
        <div class="visual-block">
            <span class="visual-label">🎨 Visual</span>
            <span class="visual-text">{desc}</span>
        </div>'''

    # selected_by 배지
    selected_by = (post or {}).get("selected_by", "") if key == "comparison_reply" else ""
    selected_badge = f'<span class="selected-badge">{selected_by}</span>' if selected_by else ""

    # 원포스트 링크 생성
    post_uri = (post or {}).get("uri", "")
    bsky_link = ""
    if key == "original_post" and post_uri:
        # at://did:.../app.bsky.feed.post/rkey → https://bsky.app/profile/did/post/rkey
        import re as _re
        m = _re.match(r"at://([^/]+)/[^/]+/([^/]+)", post_uri)
        if m:
            did, rkey = m.group(1), m.group(2)
            bsky_link = f'<a href="https://bsky.app/profile/{did}/post/{rkey}" target="_blank" class="bsky-link">🔗 원본 보기</a>'

    return f"""
    <div class="unit{meme_class}">
        <div class="unit-header">
            <span class="unit-icon">{icon}</span>
            <span class="unit-title">{title}</span>
            {selected_badge}
            {bsky_link}
            <span class="unit-meta">❤️ {like_count} &nbsp; 💬 {reply_count}</span>
        </div>
        {f'<div class="unit-text">{text}</div>' if text else '<div class="unit-text no-text">텍스트 없음</div>'}
        {imgs_html}
        <div class="label-row">
            {discourse_chip(disc_func, disc_type)}
            {stance_badges(stance) if stance else ""}
        </div>
        {path_html}
        {visual_html}
    </div>"""


def render_record(record: dict, idx: int) -> str:
    uid          = record.get("uid", "unknown")
    meme_prob    = record.get("meme_prob", 0)
    validation   = record.get("meme_validation", {})
    valid_ratio  = validation.get("valid_ratio", 0)
    valid_reason = validation.get("reason", "")
    labeled_at   = record.get("labeled_at", "")
    labels       = record.get("discourse_labels", {})
    downloaded   = record.get("downloaded_images", {})
    thread       = record.get("thread_structure", {})
    depth_label  = thread.get("label", "")

    # 밈 템플릿 이름 (30자 이상이면 텍스트 읽은 것으로 판단, 숨김)
    template = ""
    for v in validation.get("validations", []):
        t = v.get("template_name")
        if t and t not in ("null", "None", None) and len(t) <= 30:
            template = t
            break

    # 날짜 정보
    created_at = ""
    for field in ["createdAt", "created_at"]:
        val = record.get(field, "")
        if not val:
            val = (record.get("original_post") or {}).get(field, "")
        if val:
            created_at = str(val)[:10]
            break

    unit_defs = [
        ("original_post",    record.get("original_post"),          "원 포스트",   False, True),   # no_label=True
        ("parent_reply",     record.get("parent_reply"),           "부모 댓글",   False, False),
        ("meme_reply",       record.get("meme_reply"),             "밈 댓글 🎭",  True,  False),
    ]

    units_html = ""
    for key, post, title, is_meme, no_label in unit_defs:
        label = None if no_label else (labels or {}).get(key)
        if post or label:
            units_html += render_unit(key, post, label, title, downloaded, is_meme)

    valid_color = "#00b894" if valid_ratio >= 0.8 else "#e17055"

    return f"""
    <div class="record" id="rec-{idx}">
        <div class="record-header">
            <div class="record-meta-left">
                <span class="rec-idx">#{idx+1}</span>
                <span class="rec-uid" title="{uid}">{uid[:28]}...</span>
                <span class="depth-badge">{depth_label}</span>
                {f'<span class="date-badge">📅 {created_at}</span>' if created_at else ""}
                {f'<span class="template-badge">🃏 {template}</span>' if template else ""}
            </div>
            <div class="record-meta-right">
                <span class="prob-badge">🤖 {meme_prob:.2%}</span>
                <span class="valid-badge" style="color:{valid_color}" title="{valid_reason}">
                    ✓ {valid_ratio:.0%}
                </span>
                <span class="time-badge">🕐 {labeled_at[:16] if labeled_at else ""}</span>
            </div>
        </div>
        <div class="units-container">
            {units_html}
        </div>
    </div>"""


def generate_html(records: list, output_path: str):
    records_html = "\n".join(render_record(r, i) for i, r in enumerate(records))
    total = len(records)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Meme Discourse Labels</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg:#0D0F14; --surface:#161920; --card:#1E2230; --meme-card:#1a1f35;
    --border:#2A2F42; --accent:#7C6AFF; --accent2:#4ECDC4;
    --text:#E8EAF0; --muted:#6B7280; --meme-glow:rgba(124,106,255,0.15);
  }}
  *{{ box-sizing:border-box; margin:0; padding:0; }}
  body{{ background:var(--bg); color:var(--text); font-family:'Space Grotesk',sans-serif; min-height:100vh; padding-bottom:80px; }}

  .topbar{{ background:linear-gradient(135deg,#161920,#1a1230); border-bottom:1px solid var(--border); padding:18px 36px; position:sticky; top:0; z-index:100; backdrop-filter:blur(12px); display:flex; align-items:center; justify-content:space-between; }}
  .topbar-title{{ font-size:1.15rem; font-weight:700; background:linear-gradient(90deg,#7C6AFF,#4ECDC4); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
  .topbar-meta{{ color:var(--muted); font-size:0.8rem; font-family:'JetBrains Mono'; }}

  .filterbar{{ padding:12px 36px; display:flex; gap:8px; align-items:center; flex-wrap:wrap; border-bottom:1px solid var(--border); background:var(--surface); }}
  .filter-input{{ background:var(--card); border:1px solid var(--border); color:var(--text); border-radius:8px; padding:6px 12px; font-family:'JetBrains Mono'; font-size:0.8rem; width:240px; outline:none; }}
  .filter-input:focus{{ border-color:var(--accent); }}
  .filter-btn{{ background:var(--card); border:1px solid var(--border); color:var(--muted); border-radius:8px; padding:6px 11px; font-size:0.76rem; cursor:pointer; transition:all .2s; }}
  .filter-btn:hover,.filter-btn.active{{ background:var(--accent); color:#fff; border-color:var(--accent); }}
  .stat-chip{{ margin-left:auto; color:var(--muted); font-family:'JetBrains Mono'; font-size:0.76rem; }}

  .main{{ max-width:1100px; margin:0 auto; padding:24px 18px; }}

  .record{{ background:var(--card); border:1px solid var(--border); border-radius:16px; margin-bottom:22px; overflow:hidden; transition:box-shadow .3s; }}
  .record:hover{{ box-shadow:0 4px 32px rgba(124,106,255,0.1); }}
  .record-header{{ padding:11px 16px; background:var(--surface); border-bottom:1px solid var(--border); display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:6px; }}
  .record-meta-left,.record-meta-right{{ display:flex; align-items:center; gap:8px; flex-wrap:wrap; }}
  .rec-idx{{ background:var(--accent); color:#fff; font-weight:700; font-size:0.75rem; padding:2px 8px; border-radius:6px; font-family:'JetBrains Mono'; }}
  .rec-uid{{ font-family:'JetBrains Mono'; font-size:0.72rem; color:var(--muted); }}
  .depth-badge{{ font-size:0.7rem; background:#2d3748; color:#a0aec0; padding:2px 7px; border-radius:10px; }}
  .template-badge{{ font-size:0.7rem; background:#2d4a3e; color:#68d391; padding:2px 7px; border-radius:10px; }}
  .date-badge{{ font-size:0.7rem; background:#2d3a4a; color:#74b9ff; padding:2px 7px; border-radius:10px; font-family:'JetBrains Mono'; }}
  .prob-badge{{ font-size:0.72rem; color:var(--accent2); font-family:'JetBrains Mono'; }}
  .valid-badge,.time-badge{{ font-size:0.72rem; font-family:'JetBrains Mono'; color:var(--muted); }}

  .units-container{{ padding:12px; display:flex; flex-direction:column; gap:10px; }}

  .unit{{ background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:12px 14px; }}
  .unit.meme-unit{{ background:var(--meme-card); border-color:var(--accent); box-shadow:0 0 20px var(--meme-glow); }}
  .unit-header{{ display:flex; align-items:center; gap:8px; margin-bottom:8px; }}
  .unit-icon{{ font-size:1rem; }}
  .unit-title{{ font-weight:600; font-size:0.86rem; }}
  .unit-meta{{ margin-left:auto; font-size:0.7rem; color:var(--muted); }}
  .selected-badge{{ font-size:0.68rem; background:#2d4a3e; color:#68d391; padding:2px 6px; border-radius:8px; font-family:'JetBrains Mono'; }}

  .unit-text{{ font-size:0.86rem; line-height:1.6; margin-bottom:8px; padding:7px 10px; background:rgba(255,255,255,0.03); border-radius:8px; border-left:3px solid var(--border); white-space:pre-wrap; }}
  .unit-text.no-text{{ color:var(--muted); font-style:italic; }}

  .img-row{{ display:flex; gap:10px; flex-wrap:wrap; margin:8px 0; }}
  .img-wrap{{ display:flex; flex-direction:column; gap:4px; max-width:280px; }}
  .img-alt{{ font-size:0.72rem; color:var(--muted); font-style:italic; padding:2px 4px; }}
  .img-err{{ color:var(--muted); font-size:0.75rem; padding:8px; background:rgba(255,255,255,0.03); border-radius:6px; }}

  .label-row{{ display:flex; align-items:center; gap:8px; flex-wrap:wrap; margin-top:8px; }}
  .discourse-chip{{ font-size:0.73rem; font-weight:600; color:#fff; padding:3px 10px; border-radius:20px; font-family:'JetBrains Mono'; }}
  .label-type{{ font-size:0.66rem; color:var(--muted); font-family:'JetBrains Mono'; }}
  .stance-badge{{ font-size:0.7rem; font-weight:600; padding:3px 9px; border-radius:20px; }}

  .path-block{{ margin-top:8px; display:flex; flex-wrap:wrap; gap:6px; background:rgba(255,255,255,0.02); border-radius:8px; padding:7px 10px; }}
  .path-arrow{{ font-size:0.7rem; color:var(--accent2); font-family:'JetBrains Mono'; background:rgba(78,205,196,0.08); padding:2px 8px; border-radius:10px; }}
  .bsky-link{{ font-size:0.72rem; color:var(--accent2); text-decoration:none; padding:2px 8px; border:1px solid rgba(78,205,196,0.3); border-radius:8px; transition:all .2s; }}
  .bsky-link:hover{{ background:rgba(78,205,196,0.1); }}

  .visual-block{{ margin-top:8px; background:rgba(124,106,255,0.06); border:1px solid rgba(124,106,255,0.2); border-radius:8px; padding:8px 12px; display:flex; gap:8px; align-items:flex-start; }}
  .visual-label{{ font-size:0.7rem; font-weight:600; color:var(--accent); min-width:55px; font-family:'JetBrains Mono'; padding-top:2px; }}
  .visual-text{{ font-size:0.8rem; color:var(--text); line-height:1.5; }}

  #img-modal{{ display:none; position:fixed; inset:0; background:rgba(0,0,0,0.9); z-index:999; justify-content:center; align-items:center; cursor:zoom-out; flex-direction:column; gap:12px; }}
  #img-modal.show{{ display:flex; }}
  #img-modal img{{ max-width:90vw; max-height:85vh; border-radius:12px; object-fit:contain; }}
  #modal-caption{{ color:#ccc; font-size:0.85rem; max-width:600px; text-align:center; }}

  .hidden{{ display:none !important; }}
</style>
</head>
<body>

<div class="topbar">
  <div class="topbar-title">🎭 Meme Discourse Labels</div>
  <div class="topbar-meta">총 {total}개 · {generated_at}</div>
</div>

<div class="filterbar">
  <input class="filter-input" id="search" placeholder="🔍 uid, 텍스트, 라벨 검색..." oninput="applyFilter()">
  <button class="filter-btn" onclick="filterStance('sarcastic')">😏 Sarcastic</button>
  <button class="filter-btn" onclick="filterStance('humorous')">😄 Humorous</button>
  <button class="filter-btn" onclick="filterStance('offensive')">⚠️ Offensive</button>
  <button class="filter-btn" onclick="filterDepth('reply')">↩ reply</button>
  <button class="filter-btn" onclick="filterDepth('re-reply')">↩↩ re-reply</button>
  <button class="filter-btn" onclick="clearFilter()">✕ 초기화</button>
  <span class="stat-chip" id="stat-chip">{total}개 표시 중</span>
</div>

<div class="main" id="main-list">
  {records_html}
</div>

<div id="img-modal" onclick="closeImg()">
  <img id="modal-img" src="" alt="">
  <div id="modal-caption"></div>
</div>

<script>
  let activeStance = null, activeDepth = null;

  function openImg(src, alt) {{
    document.getElementById('modal-img').src = src;
    document.getElementById('modal-caption').textContent = alt || '';
    document.getElementById('img-modal').classList.add('show');
  }}
  function closeImg() {{ document.getElementById('img-modal').classList.remove('show'); }}
  document.addEventListener('keydown', e => {{ if(e.key==='Escape') closeImg(); }});

  function filterStance(s) {{
    activeStance = (activeStance===s) ? null : s;
    document.querySelectorAll('.filter-btn').forEach(b=>b.classList.remove('active'));
    if(activeStance) event.target.classList.add('active');
    applyFilter();
  }}
  function filterDepth(d) {{
    activeDepth = (activeDepth===d) ? null : d;
    document.querySelectorAll('.filter-btn').forEach(b=>b.classList.remove('active'));
    if(activeDepth) event.target.classList.add('active');
    applyFilter();
  }}
  function clearFilter() {{
    activeStance=null; activeDepth=null;
    document.getElementById('search').value='';
    document.querySelectorAll('.filter-btn').forEach(b=>b.classList.remove('active'));
    applyFilter();
  }}
  function applyFilter() {{
    const q = document.getElementById('search').value.toLowerCase();
    let visible=0;
    document.querySelectorAll('.record').forEach(rec=>{{
      const text=rec.innerText.toLowerCase();
      const matchSearch=!q||text.includes(q);
      let matchStance=true;
      if(activeStance){{
        matchStance=false;
        rec.querySelectorAll('.stance-badge').forEach(b=>{{
          if(b.textContent.toLowerCase().includes(activeStance)&&b.style.opacity==='1') matchStance=true;
        }});
      }}
      let matchDepth=!activeDepth||text.includes(activeDepth);
      if(matchSearch&&matchStance&&matchDepth){{rec.classList.remove('hidden');visible++;}}
      else rec.classList.add('hidden');
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
    parser.add_argument("--input",  default="./labeled_dataset/labeled_memes.jsonl")
    parser.add_argument("--output", default="./labeled_viewer.html")
    args = parser.parse_args()

    print(f"[LOAD] {args.input}")
    records = load_records(args.input)
    print(f"  {len(records)}개 레코드 로드")
    generate_html(records, args.output)

if __name__ == "__main__":
    main()

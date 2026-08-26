#!/usr/bin/env python3
"""
Discourse Function 라벨링 모델 비교 스크립트

react_tree.json 기반 6개 라벨로 GPT-4o, GPT-5, GPT-5-mini 세 모델 결과 비교.
샘플 10개 처리 후 결과를 HTML 뷰어로 출력 (이미지 + 대화맥락 포함).

사용법:
  python compare_discourse_models.py
  python compare_discourse_models.py --sample 10
  python compare_discourse_models.py --tree ./react_tree.json
"""

import os
import re
import json
import random
import argparse
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CONFIG = {
    "input_jsonl":  "./labeled_final/labeled_memes.jsonl",
    "tree_path":    "./new/react_tree.json",
    "output_dir":   "./model_comparison",
    "models": ["gpt-4o", "gpt-5", "gpt-5-mini"],
    "sample_size":  10,
    "random_seed":  42,
}

LABEL_COLORS = {
    "React.Support.Reply.Agree":      "#22c55e",
    "React.Support.Develop":          "#3b82f6",
    "React.Support.Track":            "#06b6d4",
    "React.Confront.Reply.Disavow":   "#f97316",
    "React.Confront.Reply.Disagree":  "#ef4444",
    "React.Confront.Challenge":       "#a855f7",
    "ERROR":                          "#6b7280",
    "N/A":                            "#d1d5db",
}

LABEL_DESC = {
    "React.Support.Reply.Agree":     "동의 — 이전 발화에 동의하거나 긍정적으로 확인",
    "React.Support.Develop":         "확장 — 이전 발화를 연장하거나 부연 설명",
    "React.Support.Track":           "명확화 요청 — 이전 발화에 대해 추가 정보나 확인 요청",
    "React.Confront.Reply.Disavow":  "부인 — 이전 발화에 대한 이해/지식을 부정",
    "React.Confront.Reply.Disagree": "반대 — 이전 발화를 거부하거나 반박",
    "React.Confront.Challenge":      "도전 — 이전 발화의 타당성/신뢰성/관련성에 의문 제기",
    "ERROR":                         "API 오류",
    "N/A":                           "원본 라벨 없음",
}

# ════════════════════════════════════════════════════════════════
#  GPT 호출
# ════════════════════════════════════════════════════════════════

def gpt_call(messages, model, max_tokens=10):
    response = client.chat.completions.create(
        model=model, messages=messages,
        max_completion_tokens=max_tokens, temperature=0 if "gpt-5" not in model else 1,
    )
    return response.choices[0].message.content.strip()


def traverse_tree(tree_node, utterance, context, model, images_content=None):
    path, node = [], tree_node
    while True:
        question = node.get("question_to_define_groups", "")
        groups   = node.get("groups", [])
        if not groups:
            data = node.get("data", [])
            return {"label": data[0] if data else "Unknown", "path": path}

        possible_answers = "\n".join(
            [f"Answer {i+1}: {g['label']}" for i, g in enumerate(groups)]
        )
        content = [{
            "type": "text",
            "text": (
                f"TASK: You will see the part of the dialog between speakers. "
                f"Answer the Question about Current Utterance. "
                f"You must analyze the relations between the Current Utterance and the Previous Context.\n\n"
                f"Previous Context: {context}\n\n"
                f"Current Utterance: {utterance}\n\n"
                f"Question: {question}\n"
                f"Possible Answers:\n{possible_answers}\n\n"
                f"Remember that the Question is about the Current Utterance. "
                f"You must select one answer from Possible Answers and reply ONLY with "
                f"\"Answer 1\", \"Answer 2\", etc., without any additional explanation."
            )
        }]
        if images_content:
            content.extend(images_content)

        try:
            resp = gpt_call([{"role": "user", "content": content}], model=model, max_tokens=10)
            m   = re.search(r"Answer\s+(\d+)", resp)
            idx = int(m.group(1)) - 1 if m else 0
            idx = max(0, min(idx, len(groups) - 1))
        except Exception as e:
            print(f"    [WARN] {model}: {e}")
            idx = 0

        chosen = groups[idx]
        path.append({"question": question, "answer": chosen["label"], "answer_idx": idx+1})

        if "next_split" in chosen:
            node = chosen["next_split"]
        else:
            data  = chosen.get("data", [])
            label = data[0] if len(data) == 1 else chosen["label"]
            return {"label": label, "path": path}


# ════════════════════════════════════════════════════════════════
#  컨텍스트 구성
# ════════════════════════════════════════════════════════════════

def get_post_text(post):
    if not post:
        return ""
    text     = post.get("text") or ""
    images   = post.get("images") or []
    captions = [img.get("alt","") for img in images if img.get("alt")]
    parts = []
    if text and text != "None":
        parts.append(text)
    if captions:
        parts.append("[Image: " + " | ".join(captions) + "]")
    return " ".join(parts).strip()


def get_post_images(post):
    if not post:
        return []
    images = post.get("images") or []
    return [img.get("source_url") or img.get("url","") for img in images if img.get("source_url") or img.get("url")]


def build_context(record):
    original_post  = record.get("original_post") or {}
    parent_reply   = record.get("parent_reply")
    ancestor_chain = record.get("ancestor_chain") or []
    meme_reply     = record.get("meme_reply") or {}

    ctx_parts = []
    orig_text = get_post_text(original_post)
    if orig_text:
        ctx_parts.append(f"[Original Post] {orig_text}")

    quoted = record.get("quoted_post")
    if quoted:
        qt = get_post_text(quoted)
        if qt:
            ctx_parts.append(f"[Quoted Post] {qt}")

    for i, anc in enumerate(ancestor_chain):
        at = get_post_text(anc)
        if at:
            ctx_parts.append(f"[Reply {i+1}] {at}")

    parent_text = get_post_text(parent_reply) if parent_reply else ""
    if parent_text:
        ctx_parts.append(f"[Parent Reply] {parent_text}")

    return "\n".join(ctx_parts), get_post_text(meme_reply)


def build_images_content(record):
    meme   = record.get("meme_reply") or {}
    images = meme.get("images") or []
    content = []
    for img in images:
        url = img.get("source_url") or img.get("url")
        if url:
            content.append({"type": "image_url", "image_url": {"url": url, "detail": "low"}})
    return content


def build_thread_data(record):
    """HTML 렌더링용 스레드 데이터"""
    thread = []

    orig = record.get("original_post") or {}
    if orig:
        thread.append({
            "role":   "Original Post",
            "text":   get_post_text(orig),
            "images": get_post_images(orig),
            "color":  "#1d4ed8",
        })

    quoted = record.get("quoted_post")
    if quoted and get_post_text(quoted):
        thread.append({
            "role":   "Quoted Post",
            "text":   get_post_text(quoted),
            "images": get_post_images(quoted),
            "color":  "#7c3aed",
        })

    for i, anc in enumerate(record.get("ancestor_chain") or []):
        text = get_post_text(anc)
        if text:
            thread.append({
                "role":   f"Reply {i+1}",
                "text":   text,
                "images": get_post_images(anc),
                "color":  "#0f766e",
            })

    parent = record.get("parent_reply")
    if parent and get_post_text(parent):
        thread.append({
            "role":   "Parent Reply",
            "text":   get_post_text(parent),
            "images": get_post_images(parent),
            "color":  "#b45309",
        })

    meme = record.get("meme_reply") or {}
    thread.append({
        "role":   "Meme Reply ★",
        "text":   get_post_text(meme),
        "images": get_post_images(meme),
        "color":  "#be123c",
        "is_meme": True,
    })

    return thread


# ════════════════════════════════════════════════════════════════
#  HTML 생성
# ════════════════════════════════════════════════════════════════

def generate_html(results, models):
    label_colors_js = json.dumps(LABEL_COLORS)

    cards_html = ""
    for i, r in enumerate(results):
        uid          = r.get("uid","")
        orig_label   = r.get("original_label","N/A")
        thread       = r.get("thread", [])
        model_results = r.get("models", {})

        # 스레드 HTML
        thread_html = ""
        for msg in thread:
            role    = msg["role"]
            text    = msg["text"] or ""
            imgs    = msg["images"] or []
            color   = msg["color"]
            is_meme = msg.get("is_meme", False)

            imgs_html = ""
            for url in imgs:
                imgs_html += f'<img src="{url}" alt="image" onerror="this.style.display=\'none\'">'

            thread_html += f"""
            <div class="msg {'meme-msg' if is_meme else ''}">
              <div class="role-badge" style="background:{color}">{role}</div>
              {f'<p class="msg-text">{text}</p>' if text else ''}
              {f'<div class="msg-images">{imgs_html}</div>' if imgs_html else ''}
            </div>"""

        # 모델 결과 HTML
        model_html = ""
        all_labels = [model_results.get(m,{}).get("label","ERROR") for m in models]
        all_agree  = len(set(all_labels)) == 1

        for m in models:
            res   = model_results.get(m, {})
            label = res.get("label","ERROR")
            color = LABEL_COLORS.get(label, "#6b7280")
            path  = res.get("path", [])

            path_html = ""
            for step in path:
                path_html += f"""
                <div class="path-step">
                  <div class="path-q">Q: {step['question']}</div>
                  <div class="path-a">→ {step['answer']}</div>
                </div>"""

            desc = LABEL_DESC.get(label, "")
            model_html += f"""
            <div class="model-result">
              <div class="model-name">{m}</div>
              <div class="label-badge" style="background:{color}">{label}</div>
              {f'<div class="label-desc">{desc}</div>' if desc else ''}
              <details class="path-details">
                <summary>Decision path</summary>
                <div class="path-content">{path_html}</div>
              </details>
            </div>"""

        agree_badge = '<span class="agree-badge">✓ All agree</span>' if all_agree else '<span class="disagree-badge">✗ Disagree</span>'

        cards_html += f"""
        <div class="card" id="card-{i}">
          <div class="card-header">
            <span class="card-num">#{i+1}</span>
            <span class="card-uid">{uid}</span>
            {agree_badge}
            <span class="orig-label">Original: <b>{orig_label}</b></span>
          </div>
          <div class="card-body">
            <div class="thread-col">
              <div class="col-title">Thread Context</div>
              <div class="thread">{thread_html}</div>
            </div>
            <div class="results-col">
              <div class="col-title">Model Results</div>
              <div class="model-results">{model_html}</div>
            </div>
          </div>
        </div>"""

    # 일치율 계산
    agreement_html = ""
    for i in range(len(models)):
        for j in range(i+1, len(models)):
            m1, m2 = models[i], models[j]
            agree = sum(
                1 for r in results
                if r["models"].get(m1,{}).get("label") == r["models"].get(m2,{}).get("label")
            )
            pct = agree / len(results) * 100 if results else 0
            agreement_html += f'<div class="stat-item"><span>{m1} vs {m2}</span><b>{agree}/{len(results)} ({pct:.0f}%)</b></div>'

    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Discourse Model Comparison</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;700;800&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg: #0d0d0d;
    --surface: #161616;
    --surface2: #1e1e1e;
    --border: #2a2a2a;
    --text: #e8e8e8;
    --text-dim: #888;
    --accent: #e8ff00;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'Syne', sans-serif; min-height: 100vh; }}

  header {{
    border-bottom: 1px solid var(--border);
    padding: 24px 40px;
    display: flex; align-items: center; justify-content: space-between;
    position: sticky; top: 0; background: var(--bg); z-index: 100;
  }}
  header h1 {{ font-size: 1.2rem; font-weight: 800; letter-spacing: -0.02em; }}
  header h1 span {{ color: var(--accent); }}
  .meta {{ font-size: 0.75rem; color: var(--text-dim); font-family: 'JetBrains Mono', monospace; }}

  .stats-bar {{
    display: flex; gap: 32px; padding: 16px 40px;
    border-bottom: 1px solid var(--border);
    background: var(--surface);
    flex-wrap: wrap;
  }}
  .stat-item {{ display: flex; flex-direction: column; gap: 2px; font-size: 0.8rem; }}
  .stat-item span {{ color: var(--text-dim); }}
  .stat-item b {{ color: var(--accent); font-family: 'JetBrains Mono', monospace; }}

  .nav-pills {{
    display: flex; gap: 8px; padding: 16px 40px; flex-wrap: wrap;
    border-bottom: 1px solid var(--border);
  }}
  .nav-pill {{
    padding: 4px 12px; border-radius: 20px; font-size: 0.75rem;
    cursor: pointer; border: 1px solid var(--border);
    background: var(--surface2); color: var(--text-dim);
    transition: all 0.15s; font-family: 'JetBrains Mono', monospace;
  }}
  .nav-pill:hover, .nav-pill.active {{ background: var(--accent); color: #000; border-color: var(--accent); }}

  main {{ padding: 32px 40px; display: flex; flex-direction: column; gap: 24px; }}

  .card {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; overflow: hidden;
    transition: border-color 0.2s;
  }}
  .card:hover {{ border-color: #444; }}

  .card-header {{
    display: flex; align-items: center; gap: 12px;
    padding: 14px 20px; border-bottom: 1px solid var(--border);
    background: var(--surface2); flex-wrap: wrap;
  }}
  .card-num {{
    font-weight: 800; font-size: 1.1rem; color: var(--accent);
    font-family: 'JetBrains Mono', monospace; min-width: 32px;
  }}
  .card-uid {{ font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: var(--text-dim); flex: 1; }}
  .orig-label {{ font-size: 0.78rem; color: var(--text-dim); margin-left: auto; }}
  .orig-label b {{ color: var(--text); }}

  .agree-badge {{
    padding: 3px 10px; border-radius: 20px; font-size: 0.72rem;
    background: #166534; color: #4ade80; font-weight: 700;
  }}
  .disagree-badge {{
    padding: 3px 10px; border-radius: 20px; font-size: 0.72rem;
    background: #7f1d1d; color: #f87171; font-weight: 700;
  }}

  .card-body {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0; }}

  .thread-col, .results-col {{
    padding: 20px;
  }}
  .thread-col {{ border-right: 1px solid var(--border); }}

  .col-title {{
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: var(--text-dim); margin-bottom: 14px;
  }}

  .thread {{ display: flex; flex-direction: column; gap: 10px; }}

  .msg {{
    border-left: 3px solid var(--border);
    padding: 10px 12px;
    border-radius: 0 6px 6px 0;
    background: var(--surface2);
  }}
  .meme-msg {{
    border-left-width: 4px;
    background: #1a0a0a;
  }}
  .role-badge {{
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-size: 0.65rem; font-weight: 700; color: #fff;
    margin-bottom: 6px; letter-spacing: 0.05em;
  }}
  .msg-text {{
    font-size: 0.82rem; line-height: 1.5; color: var(--text);
    white-space: pre-wrap; word-break: break-word;
  }}
  .msg-images {{ display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px; }}
  .msg-images img {{
    max-width: 180px; max-height: 180px; object-fit: cover;
    border-radius: 6px; border: 1px solid var(--border);
  }}

  .model-results {{ display: flex; flex-direction: column; gap: 12px; }}
  .model-result {{
    background: var(--surface2); border-radius: 8px;
    padding: 12px 14px; border: 1px solid var(--border);
  }}
  .model-name {{
    font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
    color: var(--text-dim); margin-bottom: 8px;
  }}
  .label-badge {{
    display: inline-block; padding: 4px 12px; border-radius: 6px;
    font-size: 0.78rem; font-weight: 700; color: #fff;
    margin-bottom: 8px; font-family: 'JetBrains Mono', monospace;
  }}
  .path-details summary {{
    font-size: 0.72rem; color: var(--text-dim); cursor: pointer;
    user-select: none; margin-top: 4px;
  }}
  .path-details summary:hover {{ color: var(--text); }}
  .path-content {{ margin-top: 8px; display: flex; flex-direction: column; gap: 6px; }}
  .path-step {{
    background: var(--surface); border-radius: 4px;
    padding: 8px 10px; border: 1px solid var(--border);
  }}
  .path-q {{ font-size: 0.7rem; color: var(--text-dim); margin-bottom: 3px; }}
  .path-a {{ font-size: 0.72rem; color: var(--text); font-weight: 600; }}

  @media (max-width: 900px) {{
    .card-body {{ grid-template-columns: 1fr; }}
    .thread-col {{ border-right: none; border-bottom: 1px solid var(--border); }}
    main {{ padding: 16px; }}
    header, .stats-bar, .nav-pills {{ padding-left: 16px; padding-right: 16px; }}
  }}
</style>
</head>
<body>

<header>
  <h1>Discourse <span>Model Comparison</span></h1>
  <span class="meta">{timestamp} · {len(results)} samples · {len(models)} models</span>
</header>

<div class="stats-bar">
  <div class="stat-item"><span>Models</span><b>{" · ".join(models)}</b></div>
  {agreement_html}
</div>

<div class="nav-pills">
  <div class="nav-pill active" onclick="filterCards('all')">All ({len(results)})</div>
  <div class="nav-pill" onclick="filterCards('agree')">✓ Agree</div>
  <div class="nav-pill" onclick="filterCards('disagree')">✗ Disagree</div>
</div>

<main>
{cards_html}
</main>

<script>
function filterCards(mode) {{
  document.querySelectorAll('.nav-pill').forEach(p => p.classList.remove('active'));
  event.target.classList.add('active');
  document.querySelectorAll('.card').forEach(card => {{
    const hasAgree  = card.querySelector('.agree-badge');
    const hasDis    = card.querySelector('.disagree-badge');
    if (mode === 'all') card.style.display = '';
    else if (mode === 'agree') card.style.display = hasAgree ? '' : 'none';
    else if (mode === 'disagree') card.style.display = hasDis ? '' : 'none';
  }});
}}
</script>

</body>
</html>"""


# ════════════════════════════════════════════════════════════════
#  메인
# ════════════════════════════════════════════════════════════════

def compare_record(record, tree, models):
    context_text, utterance_text = build_context(record)
    images_content = build_images_content(record)

    result = {
        "uid":            record.get("uid"),
        "original_label": (record.get("discourse_labels") or {}).get("meme_reply", {}).get("discourse_function","N/A"),
        "thread":         build_thread_data(record),
        "models":         {},
    }

    for model in models:
        print(f"    [{model}]", end=" ", flush=True)
        try:
            out = traverse_tree(tree, utterance_text, context_text, model, images_content)
            result["models"][model] = {"label": out["label"], "path": out["path"]}
            print(f"→ {out['label']}")
        except Exception as e:
            result["models"][model] = {"label": "ERROR", "error": str(e)}
            print(f"→ ERROR: {e}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',   default=CONFIG['input_jsonl'])
    parser.add_argument('--tree',    default=CONFIG['tree_path'])
    parser.add_argument('--output',  default=CONFIG['output_dir'])
    parser.add_argument('--sample',  type=int, default=CONFIG['sample_size'])
    parser.add_argument('--seed',    type=int, default=CONFIG['random_seed'])
    parser.add_argument('--models',  nargs='+', default=CONFIG['models'])
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    tree = json.loads(Path(args.tree).read_text(encoding='utf-8'))
    print(f"[TREE] {args.tree}")

    records = []
    with open(args.input, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    print(f"[LOAD] {len(records):,}개")

    random.seed(args.seed)
    samples = random.sample(records, min(args.sample, len(records)))
    print(f"[SAMPLE] {len(samples)}개 (seed={args.seed}) | models: {args.models}\n")

    results = []
    for i, record in enumerate(samples):
        print(f"[{i+1}/{len(samples)}] {record.get('uid','')}")
        results.append(compare_record(record, tree, args.models))
        print()

    ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')

    # JSON 저장
    json_path = output_dir / f"comparison_{ts}.json"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')

    # HTML 저장
    html_path = output_dir / f"comparison_{ts}.html"
    html_path.write_text(generate_html(results, args.models), encoding='utf-8')

    # 요약
    print(f"\n{'='*60}")
    for i in range(len(args.models)):
        for j in range(i+1, len(args.models)):
            m1, m2 = args.models[i], args.models[j]
            agree = sum(1 for r in results if r["models"].get(m1,{}).get("label") == r["models"].get(m2,{}).get("label"))
            print(f"  {m1} vs {m2}: {agree}/{len(results)} ({agree/len(results)*100:.0f}%)")
    print(f"\n  JSON: {json_path.resolve()}")
    print(f"  HTML: {html_path.resolve()}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

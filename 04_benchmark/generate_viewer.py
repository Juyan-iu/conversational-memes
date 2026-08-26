"""
generate_viewer.py
──────────────────
benchmark_data/ 를 읽어서 standalone HTML 파일을 생성합니다.
이미지는 base64로 인라인 embed → 인터넷/서버 없이 그냥 열 수 있습니다.

실행:
    python generate_viewer.py              # 1~30위
    python generate_viewer.py --skip 30   # 31~60위
    python generate_viewer.py --top 50    # 50개
"""

import argparse, base64, json, mimetypes, sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="benchmark_data")
    p.add_argument("--out",  default="viewer.html")
    p.add_argument("--top",  type=int, default=30)
    p.add_argument("--skip", type=int, default=0)
    return p.parse_args()


def img_to_datauri(path: Path):
    if not path.exists():
        return None
    mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def extract_likes(obj):
    return int(
        obj.get("like_count")
        or obj.get("likes")
        or (obj.get("labels") or {}).get("like_count")
        or (obj.get("context") or {}).get("like_count")
        or (obj.get("meme_reply") or {}).get("like_count")
        or (obj.get("metadata") or {}).get("like_count")
        or 0
    )


def load_items(data_dir: Path):
    jsonl = data_dir / "benchmark_summary.jsonl"
    if not jsonl.exists():
        sys.exit(f"[오류] {jsonl} 없음")
    items = []
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        obj["_lc"] = extract_likes(obj)
        items.append(obj)
    has_likes = any(i["_lc"] > 0 for i in items)
    if has_likes:
        items.sort(key=lambda o: o["_lc"], reverse=True)
    return items, has_likes


def embed_images(item, data_dir: Path):
    uid  = item.get("uid", "")
    opts = item.get("options", {
        "A": "A_original.jpg", "B": "B_text_distractor.jpg",
        "C": "C_visual_distractor.jpg", "D": "D_easy_distractor.jpg"
    })
    item["_imgs"] = {}
    for key, fname in opts.items():
        uri = img_to_datauri(data_dir / uid / fname)
        if uri:
            item["_imgs"][key] = uri
    return item


HTML = r"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Benchmark Viewer</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Noto+Sans+KR:wght@400;500&display=swap');
  :root {
    --bg: #ffffff;
    --surface: #f8f8f8;
    --border: #e5e5e5;
    --border2: #d0d0d0;
    --text: #111111;
    --muted: #888888;
    --dim: #555555;
    --accent: #2563eb;
    --green: #16a34a;
    --orange: #ea580c;
    --purple: #7c3aed;
    --red: #dc2626;
    --tag-bg: #f0f0f0;
    --mono: 'IBM Plex Mono', 'Courier New', monospace;
    --sans: 'Inter', 'Noto Sans KR', sans-serif;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: var(--sans); font-size: 14px; line-height: 1.6; }

  /* HEADER */
  .hdr {
    position: sticky; top: 0; z-index: 100;
    background: #fff; border-bottom: 1px solid var(--border);
    padding: 10px 28px; display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
  }
  .hdr-title { font-size: 13px; font-weight: 600; color: var(--text); letter-spacing: -0.01em; }
  .hdr-stat  { font-size: 12px; color: var(--muted); }
  .hdr-flag  { font-size: 12px; color: var(--accent); font-weight: 600; margin-left: auto; }

  /* CONTROLS */
  .ctrl { display: flex; align-items: center; gap: 10px; padding: 10px 28px; border-bottom: 1px solid var(--border); flex-wrap: wrap; }
  .pinfo { font-size: 12px; color: var(--muted); margin-left: auto; }
  .btn { font-size: 12px; font-weight: 500; padding: 6px 16px; border: 1px solid var(--border2); background: #fff; color: var(--dim); cursor: pointer; border-radius: 6px; transition: all .15s; }
  .btn:hover:not(:disabled) { border-color: var(--accent); color: var(--accent); }
  .btn:disabled { opacity: .35; cursor: not-allowed; }
  .btn-p { background: var(--accent); border-color: var(--accent); color: #fff; }
  .btn-p:hover:not(:disabled) { background: #1d4ed8; }

  /* ITEMS */
  .items { padding: 24px 28px; display: flex; flex-direction: column; gap: 32px; }

  /* CARD */
  .card { background: #fff; border: 1px solid var(--border); border-radius: 10px; overflow: hidden; transition: box-shadow .2s; }
  .card:hover { box-shadow: 0 2px 12px rgba(0,0,0,.07); }
  .card.flagged { border-color: var(--accent); box-shadow: 0 0 0 2px rgba(37,99,235,.15); }

  /* card header */
  .card-hd {
    display: flex; align-items: center; gap: 10px;
    padding: 10px 16px; background: var(--surface);
    border-bottom: 1px solid var(--border);
  }
  .c-num   { font-size: 12px; font-weight: 600; color: var(--muted); min-width: 24px; }
  .c-month { font-size: 11px; color: var(--muted); }
  .c-likes { font-size: 12px; font-weight: 600; color: var(--accent); margin-left: auto; }
  .flg-btn { background: none; border: none; cursor: pointer; font-size: 16px; opacity: .25; padding: 0 2px; transition: opacity .15s; color: var(--accent); }
  .flg-btn:hover, .flg-btn.on { opacity: 1; }

  /* card body: context on top, images below */
  .card-bd { display: flex; flex-direction: column; }

  /* ── CONTEXT SECTION (top) ── */
  .ctx-wrap {
    padding: 14px 16px; display: flex; flex-direction: column; gap: 10px;
    border-bottom: 1px solid var(--border);
  }
  .csec { border-left: 3px solid var(--border2); padding: 6px 10px; display: flex; flex-direction: column; gap: 5px; }
  .s-orig   { border-color: #3b82f6; }
  .s-quoted { border-color: var(--red); }
  .s-anc    { border-color: var(--purple); }
  .s-par    { border-color: var(--orange); }
  .s-meme   { border-color: var(--green); }
  .clbl { font-size: 10px; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }
  .ctxt { font-size: 13px; color: var(--text); line-height: 1.65; word-break: break-word; white-space: pre-wrap; }
  .ctxt.empty { color: #bbb; font-style: italic; font-size: 12px; }
  .cimgs { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 2px; }
  .cthumb { height: 72px; width: auto; max-width: 120px; object-fit: cover; border-radius: 4px; border: 1px solid var(--border); cursor: zoom-in; }
  .cthumb:hover { border-color: var(--border2); }
  .cextlink { display: inline-flex; align-items: center; gap: 5px; background: var(--tag-bg); border: 1px solid var(--border); border-radius: 4px; padding: 4px 8px; text-decoration: none; max-width: 100%; overflow: hidden; }
  .cextlink:hover { border-color: var(--border2); }
  .cextlink-t { font-size: 11px; color: var(--accent); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .anc-nd { background: #faf5ff; border-radius: 4px; padding: 6px 8px; display: flex; flex-direction: column; gap: 4px; }
  .anc-i  { font-size: 10px; color: var(--muted); }

  /* ── IMAGES SECTION (bottom) ── */
  .imgs-wrap {
    padding: 16px;
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 12px;
    max-width: 720px;
    margin: 0 auto;
  }

  .islot { display: flex; flex-direction: column; gap: 6px; }
  .islot-img-wrap { position: relative; border-radius: 6px; overflow: hidden; border: 1px solid var(--border); background: var(--surface); }
  .islot-img-wrap img { width: 100%; height: auto; display: block; }
  .ibadge {
    position: absolute; top: 6px; left: 6px;
    font-size: 10px; font-weight: 700; padding: 2px 7px;
    border-radius: 4px; pointer-events: none; letter-spacing: .03em;
  }
  .ba  { background: var(--green);  color: #fff; }
  .bb  { background: var(--orange); color: #fff; }
  .bc  { background: var(--purple); color: #fff; }
  .bdc { background: var(--red);    color: #fff; }
  .idesc { font-size: 13px; color: var(--muted); text-align: center; font-weight: 500; }
  .ierr  { display: flex; align-items: center; justify-content: center; aspect-ratio: 1; font-size: 11px; color: #ccc; text-align: center; padding: 8px; }

  /* LIGHTBOX */
  #lb { display: none; position: fixed; inset: 0; background: rgba(0,0,0,.85); z-index: 9999; align-items: center; justify-content: center; cursor: zoom-out; }
  #lb.open { display: flex; }
  #lb img { max-width: 92vw; max-height: 92vh; object-fit: contain; border-radius: 4px; }

  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-thumb { background: #ddd; border-radius: 3px; }
</style>
</head>
<body>

<div class="hdr">
  <div class="hdr-title">Benchmark Viewer</div>
  <div class="hdr-stat" id="h-stat"></div>
  <div class="hdr-flag" id="h-flag" style="display:none"></div>
</div>

<div id="main">
  <div class="ctrl">
    <button class="btn"   id="bp1" onclick="go(-1)">← Prev</button>
    <button class="btn btn-p" id="bn1" onclick="go(1)">Next →</button>
    <div class="pinfo" id="pi1"></div>
  </div>
  <div class="items" id="items"></div>
  <div class="ctrl" style="border-top:1px solid var(--border);border-bottom:none">
    <button class="btn"   id="bp2" onclick="go(-1)">← Prev</button>
    <button class="btn btn-p" id="bn2" onclick="go(1)">Next →</button>
    <div class="pinfo" id="pi2"></div>
  </div>
</div>

<div id="lb" onclick="lbClose()"><img id="lb-img" src="" alt=""></div>

<script>
const ALL = __ALL_DATA__;
const SZ  = 30;
let page = 0, flagged = new Set();

const BADGES = {A:'ba',B:'bb',C:'bc',D:'bdc'};
const DESCS  = {A:'(A) Correct Answer',B:'(B) Text Distractor',C:'(C) Visual Distractor',D:'(D) Easy Distractor'};

(()=>{
  const wl = ALL.filter(i=>i._lc>0).length;
  const sorted = wl > 0 ? '좋아요 순' : '등록 순';
  document.getElementById('h-stat').innerHTML =
    `총 <b>${ALL.length}</b>개 · ${sorted}`;
  renderPage(0);
})();

function go(dir){
  const max=Math.ceil(ALL.length/SZ)-1, np=page+dir;
  if(np<0||np>max) return;
  renderPage(np);
  window.scrollTo({top:0,behavior:'smooth'});
}
function renderPage(p){
  page=p;
  const max=Math.ceil(ALL.length/SZ)-1;
  const start=p*SZ, end=Math.min(start+SZ,ALL.length);
  const info=`${start+1}–${end} / ${ALL.length} (페이지 ${p+1}/${max+1})`;
  ['pi1','pi2'].forEach(id=>document.getElementById(id).textContent=info);
  ['bp1','bp2'].forEach(id=>document.getElementById(id).disabled=p===0);
  ['bn1','bn2'].forEach(id=>document.getElementById(id).disabled=p===max);
  const wrap=document.getElementById('items');
  wrap.innerHTML='';
  ALL.slice(start,end).forEach((item,i)=>wrap.appendChild(buildCard(item,start+i)));
}

function buildCard(item,gi){
  const ctx=item.context||{}, isFl=flagged.has(item.uid);
  const card=mk('div','card'+(isFl?' flagged':''));

  // ── header ──
  const hd=mk('div','card-hd');
  const num=mk('span','c-num'); num.textContent=`#${gi+1}`;
  const mon=mk('span','c-month'); mon.textContent=item.month||'';
  const lik=mk('span','c-likes');
  lik.textContent = item._lc>0 ? `♥ ${item._lc.toLocaleString()}` : '';
  const fb=mk('button','flg-btn'+(isFl?' on':''));
  fb.textContent='★'; fb.title='Figure 후보';
  fb.onclick=()=>toggleFlag(item.uid,fb,card);
  hd.append(num,mon,lik,fb);
  card.appendChild(hd);

  const bd=mk('div','card-bd');

  // ── context (top) ──
  const ctxWrap=mk('div','ctx-wrap');

  // Original post
  const oSec=sec('s-orig','Original Post');
  addText(oSec,ctx.original_post_text,600);
  addImgs(oSec,ctx.original_post_images);
  addExt(oSec,ctx.original_post_external_title,ctx.original_post_external_url);
  ctxWrap.appendChild(oSec);

  // Quoted post
  if(ctx.quoted_post_text||(ctx.quoted_post_images||[]).length||ctx.quoted_post_external_title){
    const qSec=sec('s-quoted','Quoted Post');
    addText(qSec,ctx.quoted_post_text,400);
    addImgs(qSec,ctx.quoted_post_images);
    addExt(qSec,ctx.quoted_post_external_title,ctx.quoted_post_external_url);
    ctxWrap.appendChild(qSec);
  }

  // Ancestor chain
  const ancs=ctx.ancestor_chain||[];
  if(ancs.length){
    const aSec=sec('s-anc',`Thread (${ancs.length})`);
    ancs.forEach((node,ni)=>{
      const nd=mk('div','anc-nd');
      const idx=mk('div','anc-i'); idx.textContent=`[${ni+1}/${ancs.length}]`; nd.appendChild(idx);
      if(node.text){const t=mk('div','ctxt');t.style.fontSize='12px';t.textContent=node.text.slice(0,300);nd.appendChild(t);}
      addImgs(nd,node.images);
      if(node.quoted_post_text||(node.quoted_post_images||[]).length){
        const qd=mk('div');
        qd.style.cssText='border-left:2px solid #dc2626;padding-left:6px;margin-top:4px;display:flex;flex-direction:column;gap:3px;';
        const ql=mk('div','clbl');ql.textContent='quoted';qd.appendChild(ql);
        if(node.quoted_post_text){const qt=mk('div','ctxt');qt.style.fontSize='12px';qt.textContent=node.quoted_post_text.slice(0,200);qd.appendChild(qt);}
        addImgs(qd,node.quoted_post_images);
        nd.appendChild(qd);
      }
      addExt(nd,node.external_title,node.external_url);
      aSec.appendChild(nd);
    });
    ctxWrap.appendChild(aSec);
  }

  // Parent reply
  if(ctx.parent_reply_text||(ctx.parent_reply_images||[]).length||ctx.parent_reply_external_title){
    const pSec=sec('s-par','Parent Reply');
    addText(pSec,ctx.parent_reply_text,400);
    addImgs(pSec,ctx.parent_reply_images);
    addExt(pSec,ctx.parent_reply_external_title);
    ctxWrap.appendChild(pSec);
  }

  // Meme text
  if(ctx.meme_text){
    const mSec=sec('s-meme','Meme Reply');
    addText(mSec,ctx.meme_text,300);
    ctxWrap.appendChild(mSec);
  }

  bd.appendChild(ctxWrap);

  // ── images (bottom) ──
  const imgsWrap=mk('div','imgs-wrap');
  const imgs=item._imgs||{};
  const opts=item.options||{A:'A_original.jpg',B:'B_text_distractor.jpg',C:'C_visual_distractor.jpg',D:'D_easy_distractor.jpg'};
  Object.keys(opts).forEach(key=>{
    const slot=mk('div','islot');
    const wrap2=mk('div','islot-img-wrap');
    const src=imgs[key];
    if(src){
      const img=document.createElement('img'); img.src=src; img.alt=key;
      img.onclick=()=>lbOpen(src);
      img.style.cursor='zoom-in';
      const badge=mk('span',`ibadge ${BADGES[key]||''}`); badge.textContent=key;
      wrap2.append(img,badge);
    } else {
      const e=mk('div','ierr'); e.textContent=`${key} 없음`; wrap2.appendChild(e);
    }
    const desc=mk('div','idesc'); desc.textContent=DESCS[key]||'';
    slot.append(wrap2,desc);
    imgsWrap.appendChild(slot);
  });
  bd.appendChild(imgsWrap);

  card.appendChild(bd);
  return card;
}

function sec(cls,label){
  const s=mk('div',`csec ${cls}`);
  const l=mk('div','clbl'); l.textContent=label; s.appendChild(l); return s;
}
function addText(p,text,max){
  if(!text||!text.trim()) return;
  const t=mk('div','ctxt');
  t.textContent=text.slice(0,max)+(text.length>max?'…':'');
  p.appendChild(t);
}
function addImgs(p,urls){
  const valid=(urls||[]).filter(Boolean);
  if(!valid.length) return;
  const row=mk('div','cimgs');
  valid.slice(0,6).forEach(url=>{
    const img=document.createElement('img');
    img.className='cthumb'; img.src=url; img.loading='lazy'; img.alt='';
    img.onclick=()=>lbOpen(url);
    img.onerror=()=>img.style.display='none';
    row.appendChild(img);
  });
  p.appendChild(row);
}
function addExt(p,title,url){
  if(!title) return;
  const a=mk('a','cextlink'); a.href=url||'#'; a.target='_blank'; a.rel='noopener';
  a.innerHTML=`<span>🔗</span><span class="cextlink-t">${x(title.slice(0,120))}</span>`;
  p.appendChild(a);
}
function toggleFlag(uid,btn,card){
  if(flagged.has(uid)){flagged.delete(uid);btn.classList.remove('on');card.classList.remove('flagged');}
  else{flagged.add(uid);btn.classList.add('on');card.classList.add('flagged');}
  const hf=document.getElementById('h-flag');
  if(flagged.size>0){hf.style.display='';hf.textContent=`★ ${flagged.size}개 선택됨`;}
  else hf.style.display='none';
}

function lbOpen(src){document.getElementById('lb-img').src=src;document.getElementById('lb').classList.add('open');}
function lbClose(){document.getElementById('lb').classList.remove('open');document.getElementById('lb-img').src='';}
document.addEventListener('keydown',e=>{if(e.key==='Escape')lbClose();});
function mk(tag,cls){const e=document.createElement(tag);if(cls)e.className=cls;return e;}
function x(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');}
</script>
</body>
</html>
"""


def main():
    args = parse_args()
    data_dir = Path(args.data)

    print(f"[1/3] JSONL 로딩: {data_dir}/benchmark_summary.jsonl")
    items, has_likes = load_items(data_dir)
    print(f"      → {len(items)}개 로드 ({'좋아요 순 정렬' if has_likes else '순서 그대로 (like_count 없음)'})")

    skip, top = args.skip, args.top
    items = items[skip:skip + top]
    rank_start, rank_end = skip + 1, skip + len(items)
    print(f"      → {rank_start}~{rank_end}번 {len(items)}개 사용")

    if args.out == "viewer.html" and skip > 0:
        args.out = f"viewer_{rank_start}-{rank_end}.html"

    print(f"[2/3] 이미지 base64 변환 중... ({len(items)}개)")
    for i, item in enumerate(items):
        embed_images(item, data_dir)
        print(f"      {i+1}/{len(items)}", end="\r")
    print()

    print(f"[3/3] HTML 생성 중...")
    html = HTML.replace("__ALL_DATA__", json.dumps(items, ensure_ascii=False))

    out = Path(args.out)
    out.write_text(html, encoding="utf-8")
    size_mb = out.stat().st_size / 1024 / 1024
    print(f"\n✓ {out}  ({size_mb:.1f} MB)")

if __name__ == "__main__":
    main()

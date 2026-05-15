#!/usr/bin/env python3
"""
Meme Discourse Labeling Pipeline

수집된 밈 데이터에 대해:
1. 밈 이미지 유효성 검증 (캡션 있고 패러디 가능한 형태인지)
   → 80% 이상 밈이 맞을 때만 다음 단계 진행
2. 관련 이미지 다운로드 (포스트/댓글/베스트댓글 이미지)
3. 담화기능 라벨링 (트리 기반 분기 질문)
   - 밈 아닌 발화: 3레벨 분류
   - 밈 발화: 2레벨 분류 + Stance (Sarcastic/Humorous/Offensive)
4. 짧은 Summary 라벨 (긴 텍스트 또는 이미지 있는 발화에만)
5. 결과를 기존 jsonl에 덧붙여 새 폴더에 저장

사용법:
  python label_pipeline.py                  # 샘플 10개 처리
  python label_pipeline.py --sample 50      # 샘플 50개 처리
  python label_pipeline.py --all            # 전체 처리
"""

import os
import json
import time
import base64
import argparse
import requests
from pathlib import Path
from datetime import datetime, timezone
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ════════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════════

CONFIG = {
    # 입력 데이터 폴더 (코드 위치 기준)
    "input_dirs": [
        "../01_collection/meme_dataset_24_06",
        "../01_collection/meme_dataset_25_02",
        "../01_collection/meme_dataset",
    ],

    # 출력 폴더
    "output_dir": "./labeled_dataset",

    # 트리 JSON 경로
    "tree_3level": "./three_levels_tree.json",
    "tree_2level": "./two_levels_tree.json",

    # 모델 설정 (고정: gpt-5.4-mini)
    "model_main":   "gpt-5.4-mini",
    "model_visual": "gpt-5.4-mini",

    # 밈 유효성 검증 임계값 (80% 이상이어야 통과)
    "meme_valid_threshold": 0.8,

    # 샘플 크기 (--all 옵션 없으면 이 개수만 처리)
    "sample_size": 10,

    # Summary 생략 기준 (텍스트가 이 길이 이하이고 이미지 없으면 Summary 생략)
    "summary_skip_length": 50,

    # API 요청 간 딜레이 (초)
    "api_delay": 0.1,

    # GPT-5.4-mini 가격 (per 1M tokens)
    "price_input":  0.75,   # $0.75/1M input tokens
    "price_output": 4.50,   # $4.50/1M output tokens
}

# ── 전역 비용 추적기 ──────────────────────────────────────────
class CostTracker:
    def __init__(self):
        self.input_tokens  = 0
        self.output_tokens = 0
        self.calls         = 0

    def add(self, response):
        usage = response.usage
        self.input_tokens  += usage.prompt_tokens
        self.output_tokens += usage.completion_tokens
        self.calls         += 1

    @property
    def cost(self) -> float:
        return (
            self.input_tokens  / 1_000_000 * CONFIG["price_input"] +
            self.output_tokens / 1_000_000 * CONFIG["price_output"]
        )

    def summary(self) -> str:
        return (
            f"  API 호출:      {self.calls}회\n"
            f"  입력 토큰:     {self.input_tokens:,}\n"
            f"  출력 토큰:     {self.output_tokens:,}\n"
            f"  예상 비용:     ${self.cost:.4f} (≈ ₩{self.cost * 1380:.0f})"
        )

COST = CostTracker()

# ════════════════════════════════════════════════════════════════
#  OpenAI 클라이언트
# ════════════════════════════════════════════════════════════════

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ════════════════════════════════════════════════════════════════
#  유틸리티
# ════════════════════════════════════════════════════════════════

def load_tree(path: str) -> dict:
    """트리 JSON 로드"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_records(input_dirs: list) -> list:
    """세 폴더에서 records/*.json 로드"""
    records = []
    for dir_str in input_dirs:
        d = Path(dir_str)
        if not d.exists():
            print(f"[SKIP] 폴더 없음: {d}")
            continue
        records_dir = d / "records"
        if not records_dir.exists():
            print(f"[SKIP] records/ 없음: {d}")
            continue
        files = sorted(records_dir.glob("*.json"))
        for fp in files:
            try:
                rec = json.loads(fp.read_text("utf-8"))
                rec["_source_dir"] = str(d)
                records.append(rec)
            except Exception as e:
                print(f"  [WARN] {fp.name} 읽기 실패: {e}")
        print(f"[LOAD] {d} → {len(files)}개")
    return records


def load_records_by_month(input_dirs: list) -> dict:
    """
    index_YYYY-MM-DD.jsonl 파일명에서 달 추출 → uid 매핑
    → records/{uid}.json 로드
    반환: {YYYY-MM: [record, ...]}
    """
    import re
    from collections import defaultdict
    by_month = defaultdict(list)
    uid_loaded = set()

    for dir_str in input_dirs:
        d = Path(dir_str)
        if not d.exists():
            continue
        records_dir = d / "records"

        # index_YYYY-MM-DD.jsonl 파일 찾기
        index_files = sorted(d.glob("index_????-??-??.jsonl"))
        if not index_files:
            # index 없으면 records/*.json에서 그냥 로드
            print(f"[WARN] {d}: index 파일 없음, records/ 직접 로드")
            for fp in sorted(records_dir.glob("*.json")):
                try:
                    rec = json.loads(fp.read_text("utf-8"))
                    uid = rec.get("uid", "")
                    if uid and uid not in uid_loaded:
                        rec["_source_dir"] = str(d)
                        by_month["unknown"].append(rec)
                        uid_loaded.add(uid)
                except Exception:
                    pass
            continue

        for idx_f in index_files:
            # 파일명에서 날짜 추출: index_2024-06-15.jsonl → 2024-06
            m = re.search(r"index_(\d{4}-\d{2})-\d{2}\.jsonl", idx_f.name)
            if not m:
                continue
            month = m.group(1)

            # index jsonl에서 uid 읽기
            try:
                with open(idx_f, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        entry = json.loads(line)
                        uid = entry.get("uid", "")
                        if not uid or uid in uid_loaded:
                            continue
                        # records/{uid}.json 로드
                        rec_path = records_dir / f"{uid}.json"
                        if rec_path.exists():
                            try:
                                rec = json.loads(rec_path.read_text("utf-8"))
                                rec["_source_dir"] = str(d)
                                by_month[month].append(rec)
                                uid_loaded.add(uid)
                            except Exception:
                                pass
            except Exception as e:
                print(f"  [WARN] {idx_f.name}: {e}")

        months_found = sorted(set(
            re.search(r"index_(\d{4}-\d{2})", f.name).group(1)
            for f in index_files
            if re.search(r"index_(\d{4}-\d{2})", f.name)
        ))
        print(f"[LOAD] {d} → {sum(len(by_month[m]) for m in months_found)}개 ({len(months_found)}개월)")

    return by_month


def download_image_to_base64(url: str, retries: int = 3, timeout: int = 15) -> str | None:
    """이미지 URL → base64 문자열"""
    session = requests.Session()
    session.headers.update({"User-Agent": "MemeResearchBot/1.0 (academic research)"})
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=(5, timeout), stream=True)
            if r.status_code == 404:
                return None
            if r.status_code == 200:
                return base64.b64encode(r.content).decode("utf-8")
        except Exception:
            if attempt < retries - 1:
                time.sleep(1)
    return None


def download_and_save_image(url: str, save_path: Path, retries: int = 3) -> bool:
    """이미지 URL → 파일 저장"""
    session = requests.Session()
    session.headers.update({"User-Agent": "MemeResearchBot/1.0 (academic research)"})
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=(5, 15), stream=True)
            if r.status_code == 404:
                return False
            if r.status_code == 200:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
        except Exception:
            if attempt < retries - 1:
                time.sleep(1)
    return False


def build_image_content(images: list, output_dir: Path, uid: str, subfolder: str) -> list:
    """
    이미지 리스트 → GPT-4o에 넘길 content 항목 리스트 + 로컬 저장
    반환: [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}]
    """
    content = []
    for img in (images or []):
        url = img.get("url") or img.get("source_url")
        cid = img.get("cid", "")
        if not url:
            continue

        # 로컬 저장
        save_path = output_dir / "images" / subfolder / uid / f"{cid}.jpg"
        if not save_path.exists():
            download_and_save_image(url, save_path)

        # base64 인코딩
        b64 = download_image_to_base64(url)
        if b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })
    return content


def gpt4o_call(messages: list, max_tokens: int = 200, use_visual_model: bool = False) -> str:
    """GPT-4o API 호출"""
    model = CONFIG["model_visual"] if use_visual_model else CONFIG["model_main"]
    time.sleep(CONFIG["api_delay"])
    # gpt-5.x 계열은 max_completion_tokens, 구버전은 max_tokens
    use_new_param = any(m in model for m in ["gpt-5", "o1", "o3"])
    token_param = "max_completion_tokens" if use_new_param else "max_tokens"
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        **{token_param: max_tokens},
        temperature=0,
    )
    COST.add(response)
    return response.choices[0].message.content.strip()


# ════════════════════════════════════════════════════════════════
#  1단계: 밈 유효성 검증
# ════════════════════════════════════════════════════════════════

def validate_meme_image(image_b64: str) -> dict:
    """
    이미지가 진짜 밈인지 검증
    - 캡션이 있는가?
    - 패러디 가능한 형태인가?
    반환: {"is_valid_meme": bool, "confidence": float, "reason": str}
    """
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Analyze this image and determine if it is an internet meme.\n\n"
                        "A meme must meet ALL of the following criteria:\n"
                        "1. Has visible text/caption overlaid ON the image itself.\n"
                        "   Images with NO text at all are NOT memes.\n"
                        "2. The text is part of the meme format "
                        "(not just a watermark, logo, or news subtitle).\n"
                        "3. Uses a recognizable meme format or template "
                        "that is designed to be remixed or parodied.\n\n"
                        "Answer false if:\n"
                        "- No text is visible on the image\n"
                        "- It is a regular photo or screenshot with no meme format\n"
                        "- It is an advertisement or infographic\n\n"
                        "Text can be in any language. When in doubt, answer false.\n\n"
                        "Respond ONLY with a valid JSON object (no markdown):\n"
                        '{"is_valid_meme": true/false, "confidence": 0.0-1.0, '
                        '"template_name": "name of meme template or null", '
                        '"reason": "one sentence explanation"}'
                    )
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
                }
            ]
        }
    ]
    try:
        result = gpt4o_call(messages, max_tokens=150, use_visual_model=True)  # GPT-4o 사용 (이미지 이해 중요)
        # JSON 파싱
        result = result.strip().strip("```json").strip("```").strip()
        return json.loads(result)
    except Exception as e:
        return {"is_valid_meme": False, "confidence": 0.0, "reason": f"파싱 실패: {e}"}


def validate_record_memes(record: dict, output_dir: Path = None) -> dict:
    """
    레코드의 밈 이미지들을 검증
    80% 이상 유효한 밈이어야 통과
    반환: {"passed": bool, "valid_ratio": float, "validations": [...]}
    """
    uid = record.get("uid", "unknown")
    meme_images = record.get("meme_reply", {}).get("images", [])
    if not meme_images:
        return {"passed": False, "valid_ratio": 0.0, "validations": [], "reason": "밈 이미지 없음"}

    validations = []
    for img in meme_images:
        url = img.get("url") or img.get("source_url")
        cid = img.get("cid", "")
        if not url:
            continue

        b64 = None

        # output_dir 캐시 우선
        if output_dir and cid:
            cached = output_dir / "images" / "meme_reply" / uid / f"{cid}.jpg"
            if cached.exists():
                try:
                    b64 = base64.b64encode(cached.read_bytes()).decode("utf-8")
                except Exception:
                    b64 = None

        # 없으면 URL 다운로드
        if not b64:
            b64 = download_image_to_base64(url)

        if not b64:
            validations.append({"url": url, "is_valid_meme": False, "confidence": 0.0, "reason": "다운로드 실패"})
            continue
        result = validate_meme_image(b64)
        result["url"] = url
        validations.append(result)

    if not validations:
        return {"passed": False, "valid_ratio": 0.0, "validations": [], "reason": "검증 가능한 이미지 없음"}

    valid_count = sum(1 for v in validations if v.get("is_valid_meme", False))
    valid_ratio = valid_count / len(validations)
    passed = valid_ratio >= CONFIG["meme_valid_threshold"]

    return {
        "passed": passed,
        "valid_ratio": round(valid_ratio, 4),
        "validations": validations,
        "reason": f"{valid_count}/{len(validations)} 이미지가 유효한 밈"
    }


# ════════════════════════════════════════════════════════════════
#  2단계: 이미지 다운로드
# ════════════════════════════════════════════════════════════════

def download_context_images(record: dict, output_dir: Path) -> dict:
    """
    포스트/댓글/베스트댓글 등 컨텍스트 이미지 다운로드
    반환: {"original_post": [...], "parent_reply": [...], "best_reply": [...], ...}
    """
    uid = record.get("uid", "unknown")
    downloaded = {}

    targets = {
        "original_post":          record.get("original_post", {}),
        "parent_reply":           record.get("parent_reply"),
        "best_reply_before_meme": record.get("best_reply_before_meme"),
        "comparison_reply":       record.get("comparison_reply"),
        "meme_reply":             record.get("meme_reply", {}),
    }

    for key, post in targets.items():
        if not post:
            downloaded[key] = []
            continue
        images = post.get("images", [])
        saved = []
        for img in (images or []):
            url = img.get("url") or img.get("source_url")
            cid = img.get("cid", "")
            if not url:
                continue
            save_path = output_dir / "images" / key / uid / f"{cid}.jpg"
            ok = False
            if save_path.exists():
                ok = True
            else:
                ok = download_and_save_image(url, save_path)
            if ok:
                saved.append({
                    "local_path": str(save_path.relative_to(output_dir)),
                    "cid": cid,
                    "url": url,
                    "alt": img.get("alt", "")
                })
        downloaded[key] = saved

    return downloaded


# ════════════════════════════════════════════════════════════════
#  3단계: 밈 시각적 요소 라벨링 (MemeCap 방식)
# ════════════════════════════════════════════════════════════════

def label_meme_visual(images_content: list, utterance_text: str = "") -> dict:
    """
    밈 이미지의 시각적 요소 라벨링
    - visual_description: 시각적 요소 + 상징하는 바를 하나의 문장으로
      (캡션 텍스트 제외, 맥락 독립적, 이 문장만 보고 밈 생성 가능할 정도로)
    반환: {"visual_description": str}
    """
    if not images_content:
        return {"visual_description": None}

    content = [
        {
            "type": "text",
            "text": (
                "Describe this meme image in ONE concise sentence that:\n"
                "1. Describes the visual elements (characters, objects, expressions, setting)\n"
                "2. Includes what those visual elements symbolize or represent as a metaphor\n"
                "3. Does NOT mention or quote any text/caption visible in the image\n"
                "4. Is self-contained enough that someone could recreate the meme "
                "just from reading your sentence (without seeing the image)\n\n"
                "Example: \"A dog sitting calmly in a burning room, "
                "symbolizing willful ignorance of an obvious crisis.\"\n\n"
                "Reply with ONE sentence only. No quotes around it."
            )
        }
    ] + images_content

    try:
        description = gpt4o_call(
            [{"role": "user", "content": content}],
            max_tokens=120,
            use_visual_model=True   # GPT-4o 사용
        )
        return {"visual_description": description}
    except Exception:
        return {"visual_description": None}


# ════════════════════════════════════════════════════════════════
#  4단계: 담화기능 라벨링 (트리 기반 분기)
# ════════════════════════════════════════════════════════════════

def traverse_tree(tree_node: dict, utterance: str, context: str,
                  images_content: list = None, is_meme: bool = False) -> dict:
    """
    트리를 순회하며 담화기능 라벨 분류
    반환: {"label": str, "path": [...질문/답변 경로...]}
    """
    path = []
    node = tree_node

    while True:
        question = node.get("question_to_define_groups", "")
        groups = node.get("groups", [])

        if not groups:
            # 리프 노드
            data = node.get("data", [])
            label = data[0] if data else "Unknown"
            return {"label": label, "path": path}

        # 선택지 구성
        possible_answers = "\n".join(
            [f"Answer {i+1}: {g['label']}" for i, g in enumerate(groups)]
        )

        # 프롬프트 구성
        content = [
            {
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
            }
        ]

        # 이미지 있으면 추가
        if images_content:
            content.extend(images_content)

        messages = [{"role": "user", "content": content}]

        try:
            response = gpt4o_call(messages, max_tokens=10)
            # "Answer 1" → 인덱스 0
            import re
            m = re.search(r"Answer\s+(\d+)", response)
            idx = int(m.group(1)) - 1 if m else 0
            idx = max(0, min(idx, len(groups) - 1))
        except Exception:
            idx = 0

        chosen_group = groups[idx]
        path.append({
            "question": question,
            "answer": chosen_group["label"],
            "answer_idx": idx + 1
        })

        # 다음 노드
        if "next_split" in chosen_group:
            node = chosen_group["next_split"]
        else:
            # 리프
            data = chosen_group.get("data", [])
            label = data[0] if len(data) == 1 else chosen_group["label"]
            return {"label": label, "path": path}


def classify_stance(utterance: str, images_content: list = None) -> dict:
    """
    Stance 분류 (밈 발화 전용)
    Sarcastic / Humorous / Offensive 각각 Yes/No
    """
    stance_questions = [
        ("sarcastic", "Is this utterance sarcastic or ironic? "
                      "Does it express the opposite of what it literally means, or mock someone/something?"),
        ("humorous",  "Is this utterance humorous or funny? "
                      "Does it use comedy, jokes, or playful language?"),
        ("offensive", "Is this utterance offensive or aggressive? "
                      "Does it attack, demean, or use hateful language toward someone or something?"),
    ]

    stance_result = {}
    for key, question in stance_questions:
        content = [
            {
                "type": "text",
                "text": (
                    f"Analyze the following meme utterance.\n\n"
                    f"Utterance: {utterance}\n\n"
                    f"Question: {question}\n\n"
                    f"Reply ONLY with \"Yes\" or \"No\"."
                )
            }
        ]
        if images_content:
            content.extend(images_content)

        messages = [{"role": "user", "content": content}]
        try:
            response = gpt4o_call(messages, max_tokens=5)
            stance_result[key] = "yes" in response.lower()
        except Exception:
            stance_result[key] = False

    return stance_result


def label_utterance(utterance_text: str, context_text: str,
                    is_meme: bool, tree_3level: dict, tree_2level: dict,
                    images_content: list = None) -> dict:
    """
    발화 하나에 대한 전체 라벨링
    - is_meme=True: 2레벨 + Stance
    - is_meme=False: 3레벨
    """
    if is_meme:
        discourse = traverse_tree(tree_2level, utterance_text, context_text, images_content, is_meme=True)
        stance = classify_stance(utterance_text, images_content)
        return {
            "discourse_function": discourse["label"],
            "discourse_path": discourse["path"],
            "stance": stance,
            "label_type": "2level+stance"
        }
    else:
        discourse = traverse_tree(tree_3level, utterance_text, context_text, images_content)
        return {
            "discourse_function": discourse["label"],
            "discourse_path": discourse["path"],
            "label_type": "3level"
        }


# ════════════════════════════════════════════════════════════════
#  4단계: Summary 라벨
# ════════════════════════════════════════════════════════════════

def generate_summary(utterance_text: str, images_content: list = None) -> str | None:
    """
    발화의 시각적/표면적 내용을 한 문장으로 요약
    - 짧은 텍스트(50자 이하)이고 이미지 없으면 None 반환
    - 맥락 정보 제외, 눈에 보이는 것만
    """
    text_len = len(utterance_text.strip())
    has_image = bool(images_content)

    # 짧고 이미지 없으면 스킵
    if text_len <= CONFIG["summary_skip_length"] and not has_image:
        return None

    content = [
        {
            "type": "text",
            "text": (
                "Describe what this utterance literally shows or says in ONE short sentence. "
                "Focus ONLY on what is visually or textually present - "
                "do NOT interpret context, intent, or meaning. "
                "Do NOT include any caption text from meme images.\n\n"
                f"Utterance text: {utterance_text}\n\n"
                "Reply with ONE sentence only."
            )
        }
    ]
    if images_content:
        content.extend(images_content)

    messages = [{"role": "user", "content": content}]
    try:
        return gpt4o_call(messages, max_tokens=80)
    except Exception:
        return None


# ════════════════════════════════════════════════════════════════
#  5단계: 전체 레코드 처리
# ════════════════════════════════════════════════════════════════

def get_post_text(post: dict) -> str:
    """포스트/댓글 딕셔너리에서 텍스트 추출"""
    if not post:
        return ""
    return post.get("text", "") or ""


def get_post_images_content(post: dict, output_dir: Path = None, uid: str = None, key: str = None) -> list:
    """
    포스트/댓글 딕셔너리에서 이미지 content 리스트 추출
    로컬 파일이 있으면 재사용, 없으면 URL에서 다운로드
    """
    if not post:
        return []
    images = post.get("images", []) or []
    content = []
    for img in images:
        url = img.get("url") or img.get("source_url")
        cid = img.get("cid", "")
        if not url:
            continue

        b64 = None

        # output_dir 캐시 우선
        if output_dir and uid and key and cid:
            cached = output_dir / "images" / key / uid / f"{cid}.jpg"
            if cached.exists():
                try:
                    b64 = base64.b64encode(cached.read_bytes()).decode("utf-8")
                except Exception:
                    b64 = None

        # 없으면 URL 다운로드
        if not b64:
            b64 = download_image_to_base64(url)

        if b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })
    return content


def process_record(record: dict, tree_3level: dict, tree_2level: dict,
                   output_dir: Path) -> dict | None:
    """
    레코드 하나를 처리:
    1. 밈 유효성 검증
    2. 이미지 다운로드
    3. 각 발화 라벨링
    4. Summary 생성
    """
    uid = record.get("uid", "unknown")
    print(f"\n[PROCESS] {uid}")

    # ── 1. 이미지 다운로드 (먼저 해서 로컬 캐시 확보) ───────────
    print(f"  [1] 이미지 다운로드...")
    downloaded_images = download_context_images(record, output_dir)

    # ── 2. 밈 유효성 검증 (로컬 캐시 사용) ─────────────────────
    print(f"  [2] 밈 유효성 검증...")
    validation = validate_record_memes(record, output_dir=output_dir)
    print(f"      결과: {validation['reason']} (valid_ratio={validation['valid_ratio']:.1%})")

    if not validation["passed"]:
        print(f"  [SKIP] 유효성 검증 실패")
        return None

    # ── 3. 각 발화 라벨링 ─────────────────────────────────────
    print(f"  [3] 담화기능 라벨링...")
    labels = {}

    # 처리할 발화 정의
    # (key, post_dict, is_meme, context_builder)
    original_post  = record.get("original_post", {})
    parent_reply   = record.get("parent_reply")
    best_reply     = record.get("best_reply_before_meme")
    meme_reply     = record.get("meme_reply", {})
    comp_reply     = record.get("comparison_reply")
    ancestor_chain = record.get("ancestor_chain") or []
    quoted_post    = record.get("quoted_post")

    # ── 컨텍스트 구성 (카드에 보이는 것과 동일하게) ──────────────
    orig_text = get_post_text(original_post)
    ctx_parts = []
    if orig_text:
        ctx_parts.append(f"[Original Post] {orig_text}")

    # 원 포스트에 인용된 quoted post
    if quoted_post:
        quoted_text = get_post_text(quoted_post)
        if quoted_text:
            ctx_parts.append(f"[Quoted Post] {quoted_text}")

    # ancestor chain (타래 중간 댓글들)
    for i, anc in enumerate(ancestor_chain):
        anc_text = get_post_text(anc)
        if anc_text:
            ctx_parts.append(f"[Reply {i+1}] {anc_text}")

    # parent reply (대댓글인 경우)
    parent_text = get_post_text(parent_reply) if parent_reply else ""
    if parent_text:
        ctx_parts.append(f"[Parent Reply] {parent_text}")

    meme_context_text = "\n".join(ctx_parts)

    # 라벨링 대상:
    # - original_post: 라벨링 제외 (항상 Open 계열이므로)
    # - parent_reply: 밈이 대댓글인 경우만 (3레벨), 컨텍스트는 orig_text만
    # - meme_reply: 항상 (2레벨 + Stance), 컨텍스트는 전체 타래
    is_re_reply = parent_reply is not None

    units = [
        ("parent_reply", parent_reply, False, orig_text) if is_re_reply else ("parent_reply", None, False, ""),
        ("meme_reply",   meme_reply,   True,  meme_context_text),
    ]

    for key, post, is_meme, ctx_text in units:
        if not post:
            labels[key] = None
            continue

        text = get_post_text(post)
        imgs = get_post_images_content(post, output_dir=output_dir, uid=uid, key=key)

        is_meme_flag = "🎭MEME" if is_meme else "💬TEXT"
        print(f"      [{key}] {is_meme_flag} text={text[:40]!r} images={len(imgs)}")

        label_result = label_utterance(
            utterance_text=text,
            context_text=ctx_text,
            is_meme=is_meme,
            tree_3level=tree_3level,
            tree_2level=tree_2level,
            images_content=imgs if imgs else None
        )

        # 밈 시각적 요소 라벨링 (밈 발화 + 이미지 있을 때만)
        if is_meme and imgs:
            print(f"        [시각적 요소 라벨링 중...]")
            visual_labels = label_meme_visual(imgs, text)
            label_result["visual"] = visual_labels
            print(f"        → visual: {str(visual_labels.get('visual_description', ''))[:60]}")
        else:
            label_result["visual"] = None

        labels[key] = label_result
        print(f"        → {label_result['discourse_function']} "
              f"| visual={'있음' if label_result.get('visual') else '없음'}")

    # ── 후처리: 라벨 품질 검증 ───────────────────────────────────
    meme_label = (labels.get("meme_reply") or {}).get("discourse_function", "")

    # 1. Sustain 라벨 → 밈 댓글은 항상 다른 화자이므로 제외
    if "Sustain" in meme_label:
        print(f"  [SKIP] 밈 댓글 Sustain 라벨 감지: {meme_label}")
        return None

    # 2. 빈 라벨 / 분류 실패 케이스 제외
    # 유효한 라벨: Open.* / React.* 로 시작해야 함
    valid_prefixes = ("Open.", "React.", "Sustain.")
    if not meme_label or not any(meme_label.startswith(p) for p in valid_prefixes):
        print(f"  [SKIP] 밈 댓글 라벨 불명확: {meme_label!r}")
        return None

    # ── 결과 조합 ──────────────────────────────────────────────
    result = {
        **record,
        "meme_validation":  validation,
        "downloaded_images": downloaded_images,
        "discourse_labels": labels,
        "labeled_at": datetime.now(timezone.utc).isoformat(),
    }

    return result


# ════════════════════════════════════════════════════════════════
#  메인
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Meme Discourse Labeling Pipeline")
    parser.add_argument("--sample", type=int, default=CONFIG["sample_size"],
                        help=f"처리할 샘플 수 (기본: {CONFIG['sample_size']})")
    parser.add_argument("--all", action="store_true",
                        help="전체 처리 (샘플링 없음)")
    parser.add_argument("--output", default=CONFIG["output_dir"],
                        help=f"출력 폴더 (기본: {CONFIG['output_dir']})")
    parser.add_argument("--input", default=None, nargs="+",
                        help="처리할 데이터 폴더 (기본: CONFIG의 input_dirs 전체)")
    parser.add_argument("--uid-file", default=None,
                        help="처리할 uid 목록 파일 (줄당 uid 하나, 모델 비교 시 사용)")
    parser.add_argument("--save-uids", default=None,
                        help="샘플링된 uid를 파일로 저장 (첫 실행 시 사용)")
    parser.add_argument("--model", default=None,
                        help="사용할 모델 (model_main + model_visual 동시 변경). 예: gpt-5.4-mini")
    parser.add_argument("--model-visual", default=None,
                        help="visual description에만 다른 모델 사용. 예: gpt-4o")
    parser.add_argument("--monthly-total", type=int, default=None,
                        help="달별 균등 수집 목표 (밈 통과 기준). 예: 20000 → 달별 20000/24개월 할당")
    args = parser.parse_args()

    # 모델 오버라이드
    if args.model:
        CONFIG["model_main"]   = args.model
        CONFIG["model_visual"] = args.model
        print(f"  모델 오버라이드: {args.model}")
    if args.model_visual:
        CONFIG["model_visual"] = args.model_visual
        print(f"  visual 모델 오버라이드: {args.model_visual}")

    # 출력 폴더 설정
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "records").mkdir(exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)

    # 트리 로드
    print("[LOAD] 트리 로드...")
    tree_3level = load_tree(CONFIG["tree_3level"])
    tree_2level = load_tree(CONFIG["tree_2level"])
    print(f"  3레벨 트리: {CONFIG['tree_3level']}")
    print(f"  2레벨 트리: {CONFIG['tree_2level']}")

    # 레코드 로드
    input_dirs = args.input if args.input else CONFIG["input_dirs"]
    print("\n[LOAD] 레코드 로드...")
    print(f"  대상 폴더: {input_dirs}")
    records = load_records(input_dirs)
    print(f"  총 {len(records)}개 레코드")

    # 멘션 도배 포스트 필터링 (@ 3개 이상이면 제외)
    import re
    from collections import defaultdict

    def has_mass_mention(record: dict, threshold: int = 3) -> bool:
        text = (record.get("original_post") or {}).get("text") or ""
        mentions = re.findall(r"@[\w.\-]+", text)
        return len(mentions) >= threshold

    before = len(records)
    records = [r for r in records if not has_mass_mention(r)]
    print(f"  멘션 도배 필터: {before - len(records)}개 제외 → {len(records)}개 남음")

    # 영어 필터링 (원포스트 + 부모댓글, 영문자 비율 70% 미만이면 제외)
    def is_english(record: dict, threshold: float = 0.7) -> bool:
        texts = []
        orig_text = (record.get("original_post") or {}).get("text") or ""
        if orig_text.strip():
            texts.append(orig_text)
        parent_text = (record.get("parent_reply") or {}).get("text") or ""
        if parent_text.strip():
            texts.append(parent_text)

        for text in texts:
            alpha = sum(1 for c in text if c.isascii() and c.isalpha())
            total = sum(1 for c in text if c.isalpha())
            if total == 0:
                continue
            if alpha / total < threshold:
                return False
        return True

    before = len(records)
    records = [r for r in records if is_english(r)]
    print(f"  영어 필터: {before - len(records)}개 제외 → {len(records)}개 남음")



    # uid-file로 고정 샘플 사용
    if args.uid_file:
        with open(args.uid_file, encoding="utf-8") as f:
            target_uids = set(line.strip() for line in f if line.strip())
        records = [r for r in records if r.get("uid", "") in target_uids]
        print(f"  uid-file 기준 필터: {len(records)}개")
    elif not args.all and not args.monthly_total:
        import random
        sample_size = args.sample
        random.shuffle(records)
        records = records[:sample_size]
        print(f"  랜덤 샘플 {sample_size}개만 처리")

    # 샘플 uid 저장
    if args.save_uids:
        with open(args.save_uids, "w", encoding="utf-8") as f:
            for r in records:
                f.write(r.get("uid", "") + "\n")
        print(f"  uid 저장: {args.save_uids} ({len(records)}개)")

    # 출력 jsonl (기존 파일에 덧붙이기)
    output_jsonl = output_dir / "labeled_memes.jsonl"
    print(f"\n  중복 체크 없이 처음부터 처리")

    # 처리
    success = 0
    skipped_validation = 0
    skipped_duplicate = 0
    skipped_label = 0

    # ── 달별 할당량 모드 ──────────────────────────────────────────
    if args.monthly_total:
        import random

        # index 파일 기반으로 달별 로드
        print("\n[LOAD] 달별 index 파일 기반 로드...")
        by_month_raw = load_records_by_month(input_dirs)

        # 멘션 도배 + 영어 필터 적용
        by_month = {}
        total_filtered = 0
        for m, recs in by_month_raw.items():
            filtered = [r for r in recs if not has_mass_mention(r) and is_english(r)]
            total_filtered += len(recs) - len(filtered)
            by_month[m] = filtered
        print(f"  필터 적용: {total_filtered:,}개 제외")

        months = sorted(k for k in by_month.keys() if k != "unknown")
        n_months = len(months)
        per_month = args.monthly_total // max(n_months, 1)

        print(f"\n  달별 할당량 모드")
        print(f"  목표: {args.monthly_total:,}개 / {n_months}개월 = 월 {per_month:,}개 (밈 통과 기준)")
        print(f"\n{'='*60}")

        # 달별로 셔플
        for m in months:
            random.shuffle(by_month[m])
        if "unknown" in by_month:
            random.shuffle(by_month["unknown"])

        month_passed = {m: 0 for m in months}
        month_passed["unknown"] = 0

        with open(output_jsonl, "a", encoding="utf-8") as out_f:
            for current_month in months + (["unknown"] if by_month.get("unknown") else []):
                quota = per_month if current_month != "unknown" else (args.monthly_total - sum(month_passed.values()))
                if quota <= 0:
                    continue
                month_records = by_month[current_month]
                print(f"\n  [{current_month}] 처리 시작 (할당: {quota:,}개, 전체: {len(month_records):,}개)")
                i = 0
                for record in month_records:
                    if month_passed[current_month] >= quota:
                        print(f"  [{current_month}] 할당량 {quota:,}개 달성 → 다음 달로")
                        break
                    i += 1
                    uid = record.get("uid", "unknown")
                    print(f"\n  [{current_month} {month_passed[current_month]+1}/{quota}] uid={uid}")
                    try:
                        result = process_record(record, tree_3level, tree_2level, output_dir)
                        if result is None:
                            skipped_validation += 1
                            continue
                        out_f.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
                        out_f.flush()
                        record_path = output_dir / "records" / f"{uid}.json"
                        record_path.write_text(
                            json.dumps(result, ensure_ascii=False, indent=2, default=str),
                            encoding="utf-8"
                        )
                        month_passed[current_month] += 1
                        success += 1
                    except Exception as e:
                        import traceback
                        print(f"  [ERROR] {e}\n{traceback.format_exc()}")

        print(f"\n  달별 수집 결과:")
        for m in months:
            print(f"    {m}: {month_passed[m]:,}개")

    else:
        # ── 일반 모드 ─────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"  처리 시작: {len(records)}개")
        print(f"{'='*60}")

        with open(output_jsonl, "a", encoding="utf-8") as out_f:
            for i, record in enumerate(records):
                uid = record.get("uid", "unknown")
                print(f"\n[{i+1}/{len(records)}] uid={uid}")

                try:
                    result = process_record(record, tree_3level, tree_2level, output_dir)

                    if result is None:
                        skipped_validation += 1
                        continue

                    # jsonl에 추가
                    out_f.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
                    out_f.flush()

                    # records/ 폴더에도 개별 저장
                    record_path = output_dir / "records" / f"{uid}.json"
                    record_path.write_text(
                        json.dumps(result, ensure_ascii=False, indent=2, default=str),
                        encoding="utf-8"
                    )
                    success += 1
                    print(f"  [DONE] 저장 완료")

                except Exception as e:
                    import traceback
                    print(f"  [ERROR] {e}\n{traceback.format_exc()}")

    # 결과 요약
    print(f"\n{'='*60}")
    print(f"  처리 완료")
    print(f"  성공:               {success}개")
    print(f"  유효성 검증 실패:   {skipped_validation}개")
    print(f"  라벨 품질 제외:     {skipped_label}개 (Sustain/불명확)")
    print(f"  출력: {output_dir.resolve()}")
    print(f"  ├─ labeled_memes.jsonl   (전체 병합)")
    print(f"  ├─ records/{{uid}}.json  (개별)")
    print(f"  └─ images/               (다운로드 이미지)")
    print(f"{'='*60}")
    print(f"\n  [API 비용]")
    print(COST.summary())
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

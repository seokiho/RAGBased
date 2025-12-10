import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List
 

from dotenv import load_dotenv
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "DATA"
PRECEDENTS_DIR = DATA_DIR / "precedents"
OUTPUT_PATH = DATA_DIR / "generated_questions.json"
TARGET_COUNT = 297
MODEL = "gpt-4o-mini"
SAVE_EVERY = 10  # 몇 건마다 중간 저장
CALL_DELAY = 0.3  # 초 단위, 호출 간 딜레이
TIMEOUT = 30  # OpenAI 호출 타임아웃(초)


def load_json_list(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def load_precedents() -> List[Dict[str, Any]]:
    files = sorted(PRECEDENTS_DIR.glob("*.json"))
    precedents: List[Dict[str, Any]] = []
    for fp in files:
        try:
            with fp.open(encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                precedents.append(data)
        except Exception:
            continue
    return precedents


def extract_fields(item: Dict[str, Any]) -> Dict[str, str]:
    # 일부 파일은 {"PrecService": {...}} 형태이므로 풀어준다.
    if "PrecService" in item and isinstance(item.get("PrecService"), dict):
        item = item["PrecService"]

    case_name = str(
        item.get("사건명")
        or item.get("case_name")
        or item.get("사건명_한글")
        or ""
    ).strip()
    case_no = str(item.get("사건번호") or "").strip()
    prec_id = str(item.get("판례정보일련번호") or "").strip()
    body = str(item.get("판례내용") or item.get("본문") or "").strip()
    stc_day = str(item.get("선고일자") or "").strip()

    statute = str(item.get("참조조문") or "").strip()
    return {
        "사건명": case_name,
        "사건번호": case_no,
        "판례정보일련번호": prec_id,
        "본문": body,
        "참조조문": statute,
        "선고일자": stc_day,
    }

def build_prompt(fields: Dict[str, str]) -> str:
    return f"""당신은 법률 전문가 어시스턴트입니다. 아래 판례 정보를 바탕으로 질문 1개와 답변 1개를 생성하세요.
- 참조판례/사건번호/판례정보일련번호는 수정하지 마십시오.
- 참조조문도 그대로 유지하십시오.
- 질문은 일반인의 입장에서 판례를 바탕으로 궁금해하거나 쟁점사항에 대해 묻는 내용으로 작성하되, 판례를 읽어보지 않았더라도 이해할 수 있는 질문("이 판례의 내용"과 같은 질문은 금지)으로 작성하세요. 
- 답변은 구조와 형식을 반드시 따르세요.
- 답변 길이는 최소 300자 이상으로 충분히 상세하게 작성하세요.

[판례 정보]
사건명: {fields['사건명']}
사건번호: {fields['사건번호']}
판례정보일련번호: {fields['판례정보일련번호']}
판례내용(참고용 발췌): {fields['본문'][:1200]}
참조조문: {fields['참조조문']}
선고일자: {fields['선고일자']}

[출력 형식 지침]
질문: <핵심 쟁점을 묻는 질문 1개>
답변은 아래 형식을 엄격히 따르세요.
[답변]
상세한 법률 설명을 여기에 작성... (300자 이상)

[관련 법령]
- 법령명 제X조: 조문 내용
- 추가 관련 법령...

[관련 사례/판례]
구체적인 사례나 예시...

[주의사항]
추가적인 고려사항이나 권고사항
"""


def generate_qna(client: OpenAI, fields: Dict[str, str]) -> Dict[str, str]:
    prompt = build_prompt(fields)
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=800,
        timeout=TIMEOUT,
    )
    content = completion.choices[0].message.content or ""
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if lines:
        question = lines[0]
        answer = "\n".join(lines[1:]) if len(lines) > 1 else ""
    else:
        question, answer = "", ""

    return {
        "질문": question,
        "참조판례": fields["사건명"],
        "사건번호": fields["사건번호"],
        "판례정보일련번호": fields["판례정보일련번호"],
        "참조조문": fields["참조조문"],
        "선고일자": fields["선고일자"],
        "답변": answer,
    }


def main(target: int = TARGET_COUNT) -> None:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY 환경변수가 필요합니다. (.env 또는 환경변수 설정)")

    client = OpenAI(api_key=api_key)
    precedents = load_precedents()
    if not precedents:
        raise RuntimeError(f"precedents 폴더에 JSON이 없습니다: {PRECEDENTS_DIR}")

    # 기존 결과를 로드하여 이어 붙이기
    existing = load_json_list(OUTPUT_PATH)
    results: List[Dict[str, str]] = list(existing)
    seen_ids = {r.get("판례정보일련번호", "") for r in results}

    print(f"총 판례 파일 수: {len(precedents)}, 기존 결과: {len(results)}개")

    for idx, item in enumerate(precedents, start=1):
        fields = extract_fields(item)
        pid = fields["판례정보일련번호"]
        if not pid:
            print(f"[스킵] 판례정보일련번호 없음 (index={idx}, 사건번호={fields['사건번호']})")
            continue
        if pid in seen_ids:
            continue
        try:
            qna = generate_qna(client, fields)
            results.append(qna)
            seen_ids.add(pid)
            print(f"[{len(results)}/{target}] 생성: 사건번호={fields['사건번호']} 판례ID={pid}")
            if len(results) % SAVE_EVERY == 0:
                with OUTPUT_PATH.open("w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
            if len(results) >= target:
                break
        except Exception as exc:  # noqa: BLE001
            print(f"[스킵] 생성 실패 (index={idx}, 사건번호={fields['사건번호']}): {exc}")
            continue
        time.sleep(CALL_DELAY)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"생성 완료: {len(results)}개 저장 -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

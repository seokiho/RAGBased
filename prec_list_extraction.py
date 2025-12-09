import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import requests

# 설정
OC = "applekiho"  # applekiho@gmail.com 의 ID 부분
LIST_URL = (
    "http://www.law.go.kr/DRF/lawSearch.do?"
    "OC=applekiho&target=prec&type=JSON&search=2&prncYd=20240101~20251209&org=400201&display=1679"
)
DETAIL_URL = "http://www.law.go.kr/DRF/lawService.do"
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "DATA"
DATA_DIR.mkdir(exist_ok=True)
PRECEDENTS_DIR = DATA_DIR / "precedents"
PRECEDENTS_DIR.mkdir(exist_ok=True)


def parse_prec_list_response(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """검색 응답에서 판례 목록을 리스트로 추출."""
    prec_search = data.get("PrecSearch", {})
    if prec_search:
        items = prec_search.get("prec", [])
    else:
        items = data.get("prec", [])

    if isinstance(items, dict):
        return [items]
    if isinstance(items, list):
        return items
    return []


def fetch_prec_list(url: str, timeout: float = 15.0) -> List[Dict[str, Any]]:
    """목록 API 호출 후 리스트 반환."""
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    items = parse_prec_list_response(data)
    print(f"목록 총 {len(items)}건 수신")
    return items


def load_prec_list(path: Path) -> List[Dict[str, Any]]:
    """로컬 JSON에서 목록 로드."""
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("목록 JSON은 리스트 형태여야 합니다")
    print(f"로컬 목록 {len(data)}건 로드")
    return data


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def filter_supreme_criminal(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    대법원 형사만 필터링.
    - 법원종류코드가 비어 있는 케이스가 많아, 아래 중 하나라도 만족하면 대법원으로 간주:
      * 법원종류코드 == "400201"
      * 법원명에 "대법원" 포함
      * 데이터출처명 == "대법원"
    - 사건종류명에 '형사' 포함
    """
    filtered: List[Dict[str, Any]] = []
    for item in items:
        court_code = str(item.get("법원종류코드", "")).strip()
        court_name = str(item.get("법원명", "")).strip()
        src_name = str(item.get("데이터출처명", "")).strip()
        case_type = str(item.get("사건종류명", "")).strip()

        is_supreme = (
            court_code == "400201"
            or "대법원" in court_name
            or src_name == "대법원"
        )
        is_criminal = "형사" in case_type

        if is_supreme and is_criminal:
            filtered.append(item)

    print(f"형사(대법원) 필터링 후 {len(filtered)}건")
    return filtered


def fetch_prec_detail(prec_id: str, timeout: float = 15.0) -> Dict[str, Any]:
    """판례 본문 조회."""
    params = {"OC": OC, "target": "prec", "ID": prec_id, "type": "JSON"}
    resp = requests.get(DETAIL_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def sanitize_filename(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", " ", "(", ")") else "_" for ch in name)
    return safe.strip() or "unknown"


def main(source: str | None = None) -> None:
    # 1) 목록 불러오기 (파일 경로면 로드, 아니면 URL로 호출)
    if source:
        candidate = Path(source)
        if not candidate.is_absolute():
            candidate = BASE_DIR / candidate
        if candidate.exists():
            items = load_prec_list(candidate)
        else:
            print("목록 호출 중...")
            items = fetch_prec_list(source)
            save_json(DATA_DIR / "prec_list.json", items)
            print(f"목록 저장 완료: {DATA_DIR / 'prec_list.json'}")
    else:
        print("목록 호출 중...")
        items = fetch_prec_list(LIST_URL)
        save_json(DATA_DIR / "prec_list.json", items)
        print(f"목록 저장 완료: {DATA_DIR / 'prec_list.json'}")

    # 2) 대법원 형사만 필터 후 저장
    filtered = filter_supreme_criminal(items)
    filtered_path = DATA_DIR / "prec_list_criminal.json"
    save_json(filtered_path, filtered)
    print(f"형사만 저장: {filtered_path}")

    # 3) 본문 조회 저장
    for idx, item in enumerate(filtered, start=1):
        prec_id = str(item.get("판례일련번호") or item.get("판례정보일련번호") or "").strip()
        case_no = str(item.get("사건번호") or "").strip()
        case_name = item.get("사건명", "")
        if not prec_id:
            print(f"[스킵] 판례일련번호 없음: {case_name}")
            continue

        filename = sanitize_filename(case_no or prec_id) + ".json"
        out_path = PRECEDENTS_DIR / filename
        if out_path.exists():
            print(f"[스킵] 이미 존재: {out_path.name}")
            continue

        try:
            detail = fetch_prec_detail(prec_id)
            save_json(out_path, detail)
            print(f"[{idx}/{len(filtered)}] 저장: {out_path.name} ({case_name})")
        except Exception as exc:  # noqa: BLE001
            print(f"[에러] {prec_id} ({case_name}): {exc}")

        time.sleep(0.2)


if __name__ == "__main__":
    # 사용법: python prec_list_extraction.py "<목록 URL 또는 prec_list.json 경로>"
    url_or_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(url_or_path)

import json
import sys
from typing import Any, Dict, List, Optional

import requests


BASE_URL = "http://www.law.go.kr/DRF/lawSearch.do"
OC = "applekiho"  # applekiho@gmail.com 의 ID 부분


def fetch_supreme_court_criminal_prec(
    query: str = "",
    display: int = 20,
    page: int = 1,
    timeout: float = 10.0,
) -> List[Dict[str, Any]]:
    """
    대법원 형사 판례 목록을 조회한다.
    - curt=대법원 으로 대법원만 조회
    - type=JSON 으로 JSON 응답 요청
    - 사건종류명에 '형사' 가 포함된 결과만 필터링
    """
    params = {
        "OC": OC,
        "target": "prec",
        "type": "JSON",
        "curt": "대법원",
        "display": display,
        "page": page,
        "query": query,
        "sort": "ddes",  # 선고일자 내림차순
    }

    try:
        resp = requests.get(BASE_URL, params=params, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"판례 API 호출 실패: {exc}") from exc

    try:
        data = resp.json()
    except json.JSONDecodeError as exc:
        raise ValueError("JSON 파싱 실패") from exc

    items = data.get("prec", [])
    if not isinstance(items, list):
        raise ValueError("예상치 못한 응답 형식")

    # 사건종류명에 '형사' 가 포함된 것만 필터
    filtered = [item for item in items if "형사" in str(item.get("사건종류명", ""))]
    return filtered


def main(query: str = "", display: int = 20) -> None:
    results = fetch_supreme_court_criminal_prec(query=query, display=display)
    for item in results:
        사건명 = item.get("사건명")
        사건번호 = item.get("사건번호")
        선고일자 = item.get("선고일자")
        법원명 = item.get("법원명")
        링크 = item.get("판례상세링크")
        print(f"[{선고일자}] {법원명} / {사건번호} / {사건명}")
        if 링크:
            print(f"  상세: {링크}")
        print()


if __name__ == "__main__":
    # 터미널에서 인자: python Untitled-1.py "검색어" 10
    q = sys.argv[1] if len(sys.argv) > 1 else ""
    d = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    main(query=q, display=d)
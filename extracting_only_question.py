import json
import os

def extract_questions():
    # 경로 설정
    json_path = os.path.join("DATA", "generated_questions.json")
    output_path = os.path.join("DATA", "questions_only.txt")

    # (1) 파일 존재 여부 확인
    if not os.path.exists(json_path):
        print(f"[에러] JSON 파일을 찾을 수 없습니다: {json_path}")
        return

    # (2) JSON 읽기 예외 처리
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

            # 빈 파일 처리
            if not content:
                print("[에러] JSON 파일이 비어 있습니다.")
                return

            # JSON 파싱
            data = json.loads(content)

    except json.JSONDecodeError as e:
        print(f"[에러] JSON 형식 오류: {e}")
        return
    except Exception as e:
        print(f"[에러] 파일 읽기 실패: {e}")
        return

    # (3) 질문 추출
    questions = []
    for item in data:
        q = item.get("질문", "")

        # "질문:" 제거
        if q.startswith("질문:"):
            q = q.replace("질문:", "", 1).strip()

        questions.append(q)

    # (4) txt 저장
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            for q in questions:
                f.write(q + "\n")
        print(f"총 {len(questions)}개의 질문을 추출했습니다.")
        print(f"저장 파일: {output_path}")

    except Exception as e:
        print(f"[에러] txt 파일 저장 실패: {e}")



if __name__ == "__main__":
    extract_questions()

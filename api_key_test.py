import os
from pathlib import Path

# 1. .env 파일 경로 확인
env_path = Path(r"C:\Users\seoki\Desktop\Prj_RAG\.env")
print(f"1. .env 파일 존재: {env_path.exists()}")
print(f"   경로: {env_path}")

# 2. .env 파일 직접 읽기
if env_path.exists():
    with open(env_path, 'r', encoding='utf-8') as f:
        content = f.read()
    print(f"\n2. .env 파일 내용 (raw):")
    print(repr(content[:200]))  # 처음 200자의 raw 형태
    print(f"\n   전체 길이: {len(content)}자")

# 3. dotenv로 로드
from dotenv import load_dotenv
load_dotenv(env_path)

api_key = os.getenv("OPENAI_API_KEY")
print(f"\n3. dotenv로 읽은 결과:")
print(f"   API Key: {api_key}")
print(f"   길이: {len(api_key) if api_key else 0}자")
print(f"   repr: {repr(api_key)}")

# 4. 직접 파싱
print(f"\n4. 수동 파싱:")
if env_path.exists():
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip().startswith('OPENAI_API_KEY'):
                print(f"   라인: {repr(line)}")
                if '=' in line:
                    key, value = line.split('=', 1)
                    value = value.strip()
                    print(f"   파싱된 값 길이: {len(value)}자")
                    print(f"   시작: {value[:30]}...")
import json
import os
import time
import pickle
import numpy as np
import faiss
import openai
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

import logging
from datetime import datetime
from typing import List, Dict
from tqdm import tqdm
import pandas as pd


from langchain.text_splitter import RecursiveCharacterTextSplitter
 
 
openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = openai_api_key

# 기본 디렉토리 경로 설정
index_dir = r'C:\Users\seoki\Desktop\Studying_ML\db'
index_file = os.path.join(index_dir, 'index.faiss')
metadata_file = os.path.join(index_dir, 'index.pkl')

# 경로 출력
print(f"Index file path: {index_file}")
print(f"Metadata file path: {metadata_file}")

# OpenAI Embeddings 초기화
embedding = OpenAIEmbeddings()

# FAISS 벡터 스토어 로드 함수
def load_faiss_store(index_file, metadata_file, embedding):
    if os.path.exists(index_file) and os.path.exists(metadata_file):
        try:
            # FAISS 인덱스 로드 (allow_dangerous_deserialization=True 추가)
            faiss_store = FAISS.load_local(index_dir, embedding, allow_dangerous_deserialization=True)
            
            # 메타데이터 로드
            with open(metadata_file, 'rb') as f:
                faiss_store.metadata = pickle.load(f)

            print("FAISS store and metadata loaded successfully.")
            return faiss_store
        except Exception as e:
            print(f"Error loading FAISS store: {e}")
    else:
        print("FAISS store or metadata file not found.")
        return None

# 저장된 FAISS 객체와 메타데이터 불러오기
faiss_store = load_faiss_store(index_file, metadata_file, embedding)

# faiss_store가 정상적으로 로드되었는지 확인
if faiss_store is not None:
    print("FAISS store is ready for use.")
else:
    print("Failed to load FAISS store.")

    
    
# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_few_shot_examples(few_shot_json: str) -> List[Dict[str, str]]:
    """Few-shot 학습용 JSON 파일을 로드"""
    try:
        with open(few_shot_json, "r", encoding="utf-8-sig") as f:
            few_shot_data = json.load(f)
        return few_shot_data
    except Exception as e:
        logger.error(f"Few-shot 데이터 로드 실패: {str(e)}")
        raise

def expand_query(query: str) -> str:
    """Self-querying 방식으로 질문 확장"""
    expanded_queries = [
        query,
        f"{query}의 법적 근거는 무엇인가요?",
        f"{query}와 관련된 주요 판례나 사례는 무엇인가요?"
    ]
    return " ".join(expanded_queries)



def create_few_shot_prompt(few_shot_examples: List[Dict], context: str, question: str) -> str:
    """Few-shot 프롬프트 템플릿 생성"""
    few_shot_template = """다음은 법률 질문과 답변의 예시입니다:

{few_shot_examples}

주어진 맥락:
{context}

질문: {question}
답변:"""
    
    # Few-shot 예제 포매팅
    formatted_examples = "\n\n".join([
        f"질문: {ex['question']}\n답변: {ex['answer']}\n참조법령: {ex['reference_law']}"
        for ex in few_shot_examples[:3]
    ])
    
    return PromptTemplate(
        template=few_shot_template,
        input_variables=["few_shot_examples", "context", "question"]
    ).format(
        few_shot_examples=formatted_examples,
        context=context,
        question=question
    )



def get_answer_with_few_shot(
    qa_chain,
    llm,
    query: str,
    few_shot_examples: List[Dict[str, str]],
    retriever,
    use_self_querying: bool = False
) -> str:
    """Few-shot 예제와 RAG 기반으로 답변 생성"""
    try:
        # 질문 확장
        if use_self_querying:
            query = expand_query(query)
        
        # 관련 문서 검색
        docs = retriever.get_relevant_documents(query)
        context = "\n".join([doc.page_content for doc in docs])
        
        # Few-shot 프롬프트 생성
        system_prompt = """당신은 법률 전문가 어시스턴트입니다. 다음 지침을 반드시 따라주세요:

1. 답변 구조:
   - 질문에 대한 상세한 설명 (최소 300자 이상)
   - 관련 법률 조항 명시적 인용
   - 실제 사례나 예시 포함
   - 추가 주의사항이나 권고사항

2. 형식:
   [답변]
   상세한 법률 설명을 여기에 작성...

   [관련 법령]
   - 법령명 제X조: 조문 내용
   - 추가 관련 법령...

   [관련 사례/예시]
   구체적인 사례나 예시...

   [주의사항]
   추가적인 고려사항이나 권고사항...
"""
        
        few_shot_examples_text = "\n\n".join([
            f"""질문: {ex['question']}

답변: {ex['answer']}

관련 법령: {ex['reference_law']}

---"""
            for ex in few_shot_examples[:3]
        ])
        
        combined_prompt = f"""
{system_prompt}

=== 참고할 Few-shot 예시 ===
{few_shot_examples_text}

=== 관련 문서 ===
{context}

=== 질문 ===
{query}

위 질문에 대해 시스템 프롬프트에서 제시한 구조를 정확히 따라 답변해주세요. 
반드시 [답변], [관련 법령], [관련 사례/예시], [주의사항] 섹션을 모두 포함해야 합니다.
각 섹션은 충분히 상세하게 작성해주세요.
"""
        # 답변 생성
        response = llm.predict(combined_prompt)
        return response
        
    except Exception as e:
        logger.error(f"답변 생성 실패: {str(e)}")
        raise


def setup_qa_system(faiss_store):
    """QA 시스템 설정 - MMR 검색 방식 사용"""
    try:
        retriever = faiss_store.as_retriever(
            search_type="mmr",  
            search_kwargs={
                "k": 5,   
                "fetch_k": 10,   
                "lambda_mult": 0.9   
            }
        )
        
        llm = ChatOpenAI(
            temperature=0.1,
            model_name="gpt-4o",
            max_tokens=2000
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        
        return qa_chain, retriever, llm
    except Exception as e:
        logger.error(f"QA 시스템 설정 실패: {str(e)}")
        raise

 
def save_results(results: List[Dict], output_dir: str, file_name: str = "results") -> Dict:
    """결과 저장"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        df = pd.DataFrame(results)
        paths = {}
        
        # TSV 저장
        tsv_path = os.path.join(output_dir, f"{file_name}.tsv")
        df.to_csv(tsv_path, sep="\t", index=False, encoding="utf-8")
        paths["tsv"] = tsv_path
        
        # JSON 저장
        json_path = os.path.join(output_dir, f"{file_name}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        paths["json"] = json_path
        
        return paths
    except Exception as e:
        logger.error(f"결과 저장 실패: {str(e)}")
        raise

queries = [
"[관련 법령 특정]: 위 문서에서 언급된 피고인이 피해자로부터 인건비를 편취한 행위는 어떤 법령 조항에 해당할 수 있으며, 해당 법령의 의미는 무엇인가요?",
"[판례 검색]: 피고인이 피해자로부터 인건비를 편취한 것과 유사한 사안에서 법원이 어떤 법리를 적용했는지 확인할 수 있는 주요 판례가 있습니까? 판례의 고유 식별자나 문서 ID를 제시해 주세요. 만약 문서 내에 판례 ID가 없다면 '참조 판례 없음'으로 표시하세요.",
"[요건 사실 정리]: 피고인이 피해자로부터 인건비를 편취했다는 공소사실이 성립되기 위해 법적으로 충족되어야 하는 구체적인 요건은 무엇인가요?",
]

    
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
    
    
def process_queries_batch_with_few_shot(
    faiss_store,
    queries: List[str],
    few_shot_json: str,
    output_dir: str,
    use_self_querying: bool = False
) -> Dict:
    """배치 처리 메인 함수"""
    try:
        # Few-shot 예제 로드
        few_shot_examples = load_few_shot_examples(few_shot_json)
        
        # QA 시스템 설정
        qa_chain, retriever, llm = setup_qa_system(faiss_store)
        
        results = []
        for idx, query in enumerate(tqdm(queries, desc="쿼리 처리 중"), 1):
            try:
                answer = get_answer_with_few_shot(
                    qa_chain,
                    llm,
                    query,
                    few_shot_examples,
                    retriever,
                    use_self_querying
                )
                
                results.append({
                    "query_id": idx,
                    "query": query,
                    "answer": answer,
                    "timestamp": datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"쿼리 {idx} 처리 실패: {str(e)}")
                results.append({
                    "query_id": idx,
                    "query": query,
                    "answer": f"오류 발생: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                })
        
        # 결과 저장
        file_paths = save_results(results, output_dir)
        
        return {
            "total_queries": len(queries),
            "processed_queries": len(results),
            "output_paths": file_paths
        }
        
    except Exception as e:
        logger.error(f"배치 처리 실패: {str(e)}")
        raise
        
        

def create_prompt(few_shot_examples: List[Dict], context: str, query: str) -> str:
    """프롬프트 생성 함수"""
    system_prompt = """당신은 법률 전문가 어시스턴트입니다. 다음 지침을 반드시 따라주세요:

1. 답변 구조:
   - 질문에 대한 상세한 설명 (최소 300자 이상)
   - 관련 법률 조항 명시적 인용
   - 실제 사례나 예시 포함
   - 추가 주의사항이나 권고사항

2. 형식:
   [답변]
   상세한 법률 설명을 여기에 작성...

   [관련 법령]
   - 법령명 제X조: 조문 내용
   - 추가 관련 법령...

   [관련 사례/예시]
   구체적인 사례나 예시...

   [주의사항]
   추가적인 고려사항이나 권고사항...
"""
    
    few_shot_examples_text = "\n\n".join([
        f"""질문: {ex['question']}

답변: {ex['answer']}

관련 법령: {ex['reference_law']}

---"""
        for ex in few_shot_examples[:3]
    ])
    
    combined_prompt = f"""
{system_prompt}

=== 참고할 Few-shot 예시 ===
{few_shot_examples_text}

=== 관련 문서 ===
{context}

=== 질문 ===
{query}

위 질문에 대해 시스템 프롬프트에서 제시한 구조를 정확히 따라 답변해주세요. 
반드시 [답변], [관련 법령], [관련 사례/예시], [주의사항] 섹션을 모두 포함해야 합니다.
"""
    return combined_prompt

@retry(
    retry=retry_if_exception_type((Exception)),
    wait=wait_exponential(multiplier=1, min=4, max=20),  # 최대 대기 시간 증가
    stop=stop_after_attempt(5)  # 재시도 횟수 증가
)
def get_embeddings_with_retry(query: str, embeddings: OpenAIEmbeddings):
    """임베딩 생성 함수 with 재시도 로직"""
    time.sleep(random.uniform(1.0, 2.0))  # 요청 간 딜레이 증가
    return embeddings.embed_query(query)


@retry(
    retry=retry_if_exception_type(),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    stop=stop_after_attempt(3)
)
def get_answer_with_rate_limit(
    qa_chain,
    llm,
    query: str,
    few_shot_examples: List[Dict[str, str]],
    retriever,
    use_self_querying: bool = False
) -> str:
    """재시도 로직이 포함된 답변 생성 함수"""
    try:
        # 고정 딜레이 사용
        time.sleep(3.0)  # 3초 고정 딜레이
        
        if use_self_querying:
            query = expand_query(query)
        
        # 임베딩/검색 전 딜레이
        time.sleep(3.0)
        docs = retriever.get_relevant_documents(query)
        context = "\n".join([doc.page_content for doc in docs])
        
        # GPT 호출 전 딜레이
        time.sleep(3.0)
        combined_prompt = create_prompt(few_shot_examples, context, query)
        response = llm.predict(combined_prompt)
        
        return response
        
    except Exception as e:
        logger.error(f"답변 생성 실패: {str(e)}")
        raise

@retry(
    retry=retry_if_exception_type((Exception)),
    wait=wait_exponential(multiplier=1, min=4, max=20),
    stop=stop_after_attempt(5)
)
def get_embeddings_with_retry(query: str, embeddings: OpenAIEmbeddings):
    """임베딩 생성 함수 with 재시도 로직"""
    time.sleep(3.0)  # 3초 고정 딜레이
    return embeddings.embed_query(query)

        


def process_queries_batch_with_retry(
    faiss_store,
    queries: List[str],
    few_shot_json: str,
    output_dir: str,
    batch_size: int = 5,  # 배치 크기 감소
    delay_between_batches: float = 3.0,  # 딜레이 증가
    use_self_querying: bool = False
) -> Dict:
    """개선된 배치 처리 함수"""
    try:
        few_shot_examples = load_few_shot_examples(few_shot_json)
        qa_chain, retriever, llm = setup_qa_system(faiss_store)
        
        results = []
        processed = 0
        
        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i:i + batch_size]
            
            for query in tqdm(batch_queries, desc=f"Batch {i//batch_size + 1} 처리 중"):
                try:
                    answer = get_answer_with_rate_limit(
                        qa_chain,
                        llm,
                        query,
                        few_shot_examples,
                        retriever,
                        use_self_querying
                    )
                    
                    results.append({
                        "query_id": processed + 1,
                        "query": query,
                        "answer": answer,
                        "timestamp": datetime.now().isoformat(),
                        "status": "success"
                    })
                    
                except Exception as e:
                    logger.error(f"쿼리 {processed + 1} 처리 실패: {str(e)}")
                    results.append({
                        "query_id": processed + 1,
                        "query": query,
                        "answer": f"오류 발생: {str(e)}",
                        "timestamp": datetime.now().isoformat(),
                        "status": "error"
                    })
                
                processed += 1
                
            # 중간 결과 저장 및 배치 간 딜레이
            if results:
                save_results(results, output_dir, f"results_batch_{i//batch_size + 1}")
                if i + batch_size < len(queries):
                    time.sleep(delay_between_batches)
        
        final_paths = save_results(results, output_dir, "final_results")
        
        return {
            "total_queries": len(queries),
            "processed_queries": processed,
            "successful_queries": sum(1 for r in results if r["status"] == "success"),
            "failed_queries": sum(1 for r in results if r["status"] == "error"),
            "output_paths": final_paths
        }
        
    except Exception as e:
        logger.error(f"배치 처리 실패: {str(e)}")
        raise
        
        
 

import random

def process_queries_one_by_one(
    faiss_store,
    queries: List[str],
    few_shot_json: str,
    output_dir: str,
    delay_between_queries: float = 3.0,
    use_self_querying: bool = False,
    start_from: int = 0  # 시작 인덱스 추가
) -> Dict:
    """중단된 지점부터 이어서 처리할 수 있는 함수"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        few_shot_examples = load_few_shot_examples(few_shot_json)
        qa_chain, retriever, llm = setup_qa_system(faiss_store)
        
        json_path = os.path.join(output_dir, "final_results.json")
        tsv_path = os.path.join(output_dir, "final_results.tsv")
        
        # 이전 결과 불러오기
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                all_results = json.load(f)
            # 마지막으로 처리된 query_id 확인
            if all_results:
                start_from = max(r['query_id'] for r in all_results)
                print(f"이전 결과에서 {start_from}번째 쿼리까지 처리됨")
        else:
            all_results = []
        
        successful = sum(1 for r in all_results if r['status'] == 'success')
        failed = sum(1 for r in all_results if r['status'] == 'error')
        
        # start_from 이후의 쿼리만 처리
        remaining_queries = queries[start_from:]
        
        for idx, query in enumerate(tqdm(remaining_queries, desc="전체 진행상황", total=len(remaining_queries)), start_from + 1):
            try:
                answer = get_answer_with_rate_limit(
                    qa_chain, llm, query, few_shot_examples, retriever, use_self_querying
                )
                
                result = {
                    "query_id": idx,
                    "query": query,
                    "answer": answer,
                    "timestamp": datetime.now().isoformat(),
                    "status": "success"
                }
                successful += 1
                
            except Exception as e:
                logger.error(f"쿼리 {idx}/{len(queries)} 처리 실패: {str(e)}")
                result = {
                    "query_id": idx,
                    "query": query,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                    "status": "error"
                }
                failed += 1
            
            all_results.append(result)
            
            # 즉시 저장
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            
            df = pd.DataFrame(all_results)
            df.to_csv(tsv_path, sep="\t", index=False, encoding="utf-8")
            
            if idx < len(queries):
                time.sleep(delay_between_queries)
        
        return {
            "total_queries": len(queries),
            "successful_queries": successful,
            "failed_queries": failed,
            "json_path": json_path,
            "tsv_path": tsv_path
        }
        
    except Exception as e:
        logger.error(f"처리 실패: {str(e)}")
        raise

# 실행 예시
try: 
        results = process_queries_one_by_one(
        faiss_store=faiss_store,
        queries=queries,
        few_shot_json="./DATA/few_shot.json",  # few-shot 파일 경로
        output_dir="./DATA/generated_answers/few_shot/",  # 결과 저장 경로
        delay_between_queries=3.0,  # 쿼리 간 딜레이 (초)
        use_self_querying=True  # 쿼리 확장 여부
    )
    
        print("\n처리 완료:")
        print(f"총 쿼리: {results['total_queries']}")
        print(f"성공: {results['successful_queries']}")
        print(f"실패: {results['failed_queries']}")
    
except Exception as e:
    print(f"실행 중 오류 발생: {str(e)}")
  
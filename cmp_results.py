import json
import re
from typing import Dict, List, Tuple
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher

class LegalQAComparator:
    """법률 QA 시스템의 답변을 비교 분석하는 클래스"""
    
    def __init__(self, generated_file: str, parsed_file: str):
        """
        Args:
            generated_file: generated_questions.json 파일 경로
            parsed_file: parsed_results.json 파일 경로
        """
        with open(generated_file, 'r', encoding='utf-8') as f:
            self.generated_data = json.load(f)
        
        with open(parsed_file, 'r', encoding='utf-8') as f:
            self.parsed_data = json.load(f)
        
        # query_id 기반으로 매칭
        self.parsed_dict = {item['query_id']: item for item in self.parsed_data}
        
        self.results = []
    
    def normalize_law_name(self, law: str) -> str:
        """법령명 정규화"""
        # 「」 제거
        law = law.replace('「', '').replace('」', '')
        # 공백 제거
        law = law.replace(' ', '')
        # 영문자 소문자 변환
        law = law.lower()
        return law
    
    def extract_law_references(self, text: str) -> List[Dict]:
        """텍스트에서 법령 참조를 추출하고 상세 정보 반환"""
        # 다양한 법령 패턴 매칭
        patterns = [
            r'「([^」]+)」\s*제(\d+)조(?:의(\d+))?(?:\s*제(\d+)항)?',
            r'([가-힣]+법)\s*제(\d+)조(?:의(\d+))?(?:\s*제(\d+)항)?',
        ]
        
        laws = []
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                law_info = {
                    'full': match.group(0),
                    'law_name': match.group(1),
                    'article': match.group(2),
                    'sub_article': match.group(3) if len(match.groups()) > 2 else None,
                    'paragraph': match.group(4) if len(match.groups()) > 3 else None,
                }
                laws.append(law_info)
        
        return laws
    
    def calculate_text_similarity(self, text1: str, text2: str) -> Dict[str, float]:
        """두 텍스트 간의 유사도를 여러 방법으로 계산"""
        
        # 1. TF-IDF 코사인 유사도
        vectorizer = TfidfVectorizer()
        try:
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except:
            cosine_sim = 0.0
        
        # 2. 문자 수준 유사도
        seq_sim = SequenceMatcher(None, text1, text2).ratio()
        
        # 3. 단어 집합 기반 Jaccard 유사도
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if len(words1.union(words2)) > 0:
            jaccard_sim = len(words1.intersection(words2)) / len(words1.union(words2))
        else:
            jaccard_sim = 0.0
        
        return {
            'tfidf_cosine': round(cosine_sim, 4),
            'sequence_matcher': round(seq_sim, 4),
            'jaccard': round(jaccard_sim, 4),
            'average': round((cosine_sim + seq_sim + jaccard_sim) / 3, 4)
        }
    
    def compare_law_references(self, gen_laws: str, parsed_laws: str) -> Dict:
        """법령 참조의 일치도 비교 - 개선된 버전"""
        laws1 = self.extract_law_references(gen_laws)
        laws2 = self.extract_law_references(parsed_laws)
        
        # 법령명만 추출 (정규화)
        law_names1 = set([self.normalize_law_name(l['law_name']) for l in laws1])
        law_names2 = set([self.normalize_law_name(l['law_name']) for l in laws2])
        
        # 법령명 + 조항 (정규화)
        law_articles1 = set([
            f"{self.normalize_law_name(l['law_name'])}제{l['article']}조" 
            for l in laws1
        ])
        law_articles2 = set([
            f"{self.normalize_law_name(l['law_name'])}제{l['article']}조" 
            for l in laws2
        ])
        
        # 매칭 계산
        law_name_match = law_names1.intersection(law_names2)
        article_match = law_articles1.intersection(law_articles2)
        
        # Precision, Recall 계산
        precision_name = len(law_name_match) / len(law_names1) if law_names1 else 0
        recall_name = len(law_name_match) / len(law_names2) if law_names2 else 0
        
        precision_article = len(article_match) / len(law_articles1) if law_articles1 else 0
        recall_article = len(article_match) / len(law_articles2) if law_articles2 else 0
        
        return {
            'generated_laws': [l['full'] for l in laws1],
            'parsed_laws': [l['full'] for l in laws2],
            'generated_law_names': list(law_names1),
            'parsed_law_names': list(law_names2),
            'law_name_matches': list(law_name_match),
            'article_matches': list(article_match),
            'metrics': {
                'law_name_precision': round(precision_name, 4),
                'law_name_recall': round(recall_name, 4),
                'article_precision': round(precision_article, 4),
                'article_recall': round(recall_article, 4)
            }
        }
    
    def extract_case_numbers(self, text: str) -> List[str]:
        """판례 번호 추출"""
        pattern = r'\d{4}[가-힣]+\d+'
        return list(set(re.findall(pattern, text)))
    
    def compare_cases(self, gen_data: Dict, parsed_data: Dict) -> Dict:
        """판례 참조 비교"""
        gen_case_num = gen_data.get('사건번호', '')
        gen_case_ref = gen_data.get('참조판례', '')
        parsed_case_ref = parsed_data.get('relevant_cases', '')
        
        # 사건번호가 parsed_data의 답변에 포함되는지 확인
        case_number_mentioned = gen_case_num in parsed_case_ref if gen_case_num else False
        
        # parsed_data에서 사건번호 추출
        parsed_case_numbers = self.extract_case_numbers(parsed_case_ref)
        
        # 판례 내용의 유사도
        similarity = self.calculate_text_similarity(gen_case_ref, parsed_case_ref)
        
        return {
            'reference_case_number': gen_case_num,
            'parsed_case_numbers': parsed_case_numbers,
            'case_number_mentioned': case_number_mentioned,
            'has_case_reference': len(parsed_case_ref) > 10,  # 판례 설명이 있는지
            'case_similarity': similarity
        }
    
    def calculate_f1_score(self, precision: float, recall: float) -> float:
        """F1 Score 계산"""
        if precision + recall > 0:
            return 2 * (precision * recall) / (precision + recall)
        return 0.0
    
    def compare_single_pair(self, gen_item: Dict, parsed_item: Dict) -> Dict:
        """개별 질문-답변 쌍 비교"""
        
        # 1. 질문 일치 확인
        gen_question = gen_item.get('질문', '').replace('질문: ', '')
        parsed_question = parsed_item.get('query', '')
        question_match = gen_question.strip() == parsed_question.strip()
        
        # 2. 답변 내용 유사도
        gen_answer = gen_item.get('답변', '')
        parsed_answer = parsed_item.get('answer_content', '')
        answer_similarity = self.calculate_text_similarity(gen_answer, parsed_answer)
        
        # 3. 법령 비교
        gen_laws = gen_item.get('참조조문', '') + '\n' + gen_answer
        parsed_laws = parsed_item.get('relevant_laws', '') + '\n' + parsed_answer
        law_comparison = self.compare_law_references(gen_laws, parsed_laws)
        
        # 4. 판례 비교
        case_comparison = self.compare_cases(gen_item, parsed_item)
        
        # 5. 종합 메트릭
        law_metrics = law_comparison['metrics']
        f1_law_name = self.calculate_f1_score(
            law_metrics['law_name_precision'], 
            law_metrics['law_name_recall']
        )
        f1_article = self.calculate_f1_score(
            law_metrics['article_precision'], 
            law_metrics['article_recall']
        )
        
        return {
            'question': gen_question[:100] + '...' if len(gen_question) > 100 else gen_question,
            'question_match': question_match,
            'answer_similarity': answer_similarity,
            'law_comparison': law_comparison,
            'case_comparison': case_comparison,
            'summary_metrics': {
                'answer_similarity_avg': answer_similarity['average'],
                'law_name_f1': round(f1_law_name, 4),
                'article_f1': round(f1_article, 4),
                'has_case_reference': case_comparison['has_case_reference']
            }
        }
    
    def compare_all(self) -> List[Dict]:
        """모든 질문-답변 쌍 비교"""
        for idx, gen_item in enumerate(self.generated_data, 1):
            query_id = str(idx)
            
            if query_id in self.parsed_dict:
                parsed_item = self.parsed_dict[query_id]
                comparison = self.compare_single_pair(gen_item, parsed_item)
                comparison['query_id'] = query_id
                self.results.append(comparison)
            else:
                print(f"Warning: query_id {query_id}가 parsed_results에 없습니다.")
        
        return self.results
    
    def generate_summary_statistics(self) -> Dict:
        """전체 통계 요약"""
        if not self.results:
            return {}
        
        # 답변 유사도 평균
        avg_answer_sim = {
            'tfidf_cosine': np.mean([r['answer_similarity']['tfidf_cosine'] for r in self.results]),
            'sequence_matcher': np.mean([r['answer_similarity']['sequence_matcher'] for r in self.results]),
            'jaccard': np.mean([r['answer_similarity']['jaccard'] for r in self.results]),
            'average': np.mean([r['answer_similarity']['average'] for r in self.results])
        }
        
        # 법령 매칭 통계
        law_name_f1_scores = [r['summary_metrics']['law_name_f1'] for r in self.results]
        article_f1_scores = [r['summary_metrics']['article_f1'] for r in self.results]
        
        # 판례 참조 통계
        has_case_count = sum([r['summary_metrics']['has_case_reference'] for r in self.results])
        case_number_match_count = sum([r['case_comparison']['case_number_mentioned'] for r in self.results])
        
        # 법령명이 일치하는 비율
        law_name_match_rate = np.mean([
            1 if len(r['law_comparison']['law_name_matches']) > 0 else 0 
            for r in self.results
        ])
        
        return {
            'total_comparisons': len(self.results),
            'answer_similarity': {
                'tfidf_cosine': round(avg_answer_sim['tfidf_cosine'], 4),
                'sequence_matcher': round(avg_answer_sim['sequence_matcher'], 4),
                'jaccard': round(avg_answer_sim['jaccard'], 4),
                'average': round(avg_answer_sim['average'], 4)
            },
            'law_matching': {
                'avg_law_name_f1': round(np.mean(law_name_f1_scores), 4),
                'avg_article_f1': round(np.mean(article_f1_scores), 4),
                'law_name_match_rate': round(law_name_match_rate, 4),
                'perfect_match_count': sum([1 for score in law_name_f1_scores if score == 1.0])
            },
            'case_reference': {
                'has_case_reference_rate': round(has_case_count / len(self.results), 4),
                'case_number_match_count': case_number_match_count,
                'case_number_match_rate': round(case_number_match_count / len(self.results), 4)
            }
        }
    
    def save_results(self, output_file: str = 'comparison_results.json'):
        """결과를 JSON 파일로 저장"""
        summary = self.generate_summary_statistics()
        
        output = {
            'summary_statistics': summary,
            'detailed_results': self.results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ 결과가 {output_file}에 저장되었습니다.")
    
    def print_summary(self):
        """요약 통계 출력"""
        summary = self.generate_summary_statistics()
        
        print("\n" + "=" * 70)
        print("📊 법률 QA 답변 비교 분석 결과")
        print("=" * 70)
        
        print(f"\n📋 총 비교 건수: {summary['total_comparisons']}")
        
        print("\n📝 [답변 유사도]")
        print(f"  • TF-IDF 코사인 유사도: {summary['answer_similarity']['tfidf_cosine']:.1%}")
        print(f"  • Sequence Matcher: {summary['answer_similarity']['sequence_matcher']:.1%}")
        print(f"  • Jaccard 유사도: {summary['answer_similarity']['jaccard']:.1%}")
        print(f"  • 평균 유사도: {summary['answer_similarity']['average']:.1%}")
        
        print("\n⚖️  [법령 참조 정확도]")
        print(f"  • 법령명 일치율: {summary['law_matching']['law_name_match_rate']:.1%}")
        print(f"  • 법령명 F1 Score: {summary['law_matching']['avg_law_name_f1']:.1%}")
        print(f"  • 조항 F1 Score: {summary['law_matching']['avg_article_f1']:.1%}")
        print(f"  • 완벽 매칭 건수: {summary['law_matching']['perfect_match_count']}건")
        
        print(f"\n📖 [판례 참조]")
        print(f"  • 판례 설명 포함률: {summary['case_reference']['has_case_reference_rate']:.1%}")
        print(f"  • 사건번호 일치 건수: {summary['case_reference']['case_number_match_count']}건")
        print(f"  • 사건번호 일치율: {summary['case_reference']['case_number_match_rate']:.1%}")
        
        print("\n" + "=" * 70)
    
    def print_detailed_examples(self, n: int = 3):
        """상세 비교 예시 출력"""
        print("\n" + "=" * 70)
        print(f"🔍 상세 비교 예시 (처음 {n}개)")
        print("=" * 70)
        
        for i, result in enumerate(self.results[:n], 1):
            print(f"\n[예시 {i}] Query ID: {result['query_id']}")
            print(f"질문: {result['question']}")
            print(f"\n답변 유사도: {result['answer_similarity']['average']:.1%}")
            
            print(f"\n참조 법령 비교:")
            print(f"  Generated: {result['law_comparison']['generated_law_names']}")
            print(f"  Parsed: {result['law_comparison']['parsed_law_names']}")
            print(f"  일치: {result['law_comparison']['law_name_matches']}")
            
            print(f"\n판례 정보:")
            print(f"  참조 사건번호: {result['case_comparison']['reference_case_number']}")
            print(f"  Parsed 사건번호: {result['case_comparison']['parsed_case_numbers']}")
            print(f"  판례 설명 포함: {result['case_comparison']['has_case_reference']}")
            print("-" * 70)


# 사용 예시
if __name__ == "__main__":
    print("🚀 법률 QA 답변 비교 분석을 시작합니다...\n")
    
    # 비교 수행
    comparator = LegalQAComparator(
        './DATA/generated_questions.json',
        'parsed_results.json'
    )
    
    # 비교 실행
    results = comparator.compare_all()
    
    # 요약 통계 출력
    comparator.print_summary()
    
    # 상세 예시 출력
    comparator.print_detailed_examples(n=3)
    
    # 결과 저장
    comparator.save_results('comparison_results.json')
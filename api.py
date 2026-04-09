import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# LangChain 관련 모듈
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import DocArrayInMemorySearch

# 환경변수 로드 (.env 파일에 OPENAI_API_KEY 설정 필요)
load_dotenv()

app = FastAPI(title="Hansei Chatbot API", description="한세대학교 학사 챗봇 백엔드 API 서버")

# 모바일 UI 프론트엔드가 다른 포트나 도메인에서 호출할 수 있도록 CORS 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 배포 환경에서는 실제 프론트엔드 도메인으로 제한하세요
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 글로벌 변수로 리트리버 선언
global_retriever = None

@app.on_event("startup")
def load_data():
    """서버가 켜질 때 '학부생 학칙'과 '장학금 텍스트' 데이터를 로드합니다."""
    global global_retriever
    
    if "OPENAI_API_KEY" not in os.environ:
        print("경고: OPENAI_API_KEY가 설정되지 않았습니다.")
        
    docs = []
        
    target_file = "학부학칙.pdf"
    print(f"{target_file} 데이터를 로드 중입니다...")
    if os.path.exists(target_file):
        loader = PyPDFLoader(target_file)
        docs.extend(loader.load())
        print(f"{target_file} 로드 완료!")
    else:
        print(f"오류: {target_file} 파일이 실행 폴더에 없습니다. 반드시 파일을 준비해주세요.")
        
    scholarship_file = "scholarship.txt"
    print(f"{scholarship_file} 데이터를 로드 중입니다...")
    if os.path.exists(scholarship_file):
        loader2 = TextLoader(scholarship_file, encoding='utf-8')
        docs.extend(loader2.load())
        print(f"{scholarship_file} 로드 완료!")
    else:
        print(f"경고: {scholarship_file} 파일이 실행 폴더에 없습니다.")

    phone_file = "phone_directory.txt"
    print(f"{phone_file} 데이터를 로드 중입니다...")
    if os.path.exists(phone_file):
        loader3 = TextLoader(phone_file, encoding='utf-8')
        docs.extend(loader3.load())
        print(f"{phone_file} 로드 완료!")
    else:
        print(f"경고: {phone_file} 파일이 실행 폴더에 없습니다.")

    if docs:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        temp_db = DocArrayInMemorySearch.from_documents(docs, embeddings)
        global_retriever = temp_db.as_retriever()
        print("모든 지식 베이스 데이터 결합 및 로딩 완료!")
    else:
        print("오류: 지식 베이스(RAG)로 사용할 데이터가 하나도 없습니다.")

from typing import List, Optional

class Message(BaseModel):
    role: str
    content: str

# 프론트엔드에서 보낼 질문 데이터 구조
class QueryRequest(BaseModel):
    query: str
    history: Optional[List[Message]] = []

class QueryResponse(BaseModel):
    answer: str

def scrape_academic_schedule():
    """사용자 요청에 따라 실시간 크롤링 대신 고정된 1~12월 전체 학사일정을 바로 반환합니다."""
    return """📅 **[2026학년도 학사일정 (1월~12월)]**

**01월 January**
• [01-06 ~ 01-06] 2학기 성적 확정
• [01-14 ~ 01-16] 동계 계절학기 성적입력
• [01-17 ~ 01-19] 동계 계절학기 성적정정 및 확정
• [01-26 ~ 02-06] 1학기 복학 신청(2차)

**02월 February**
• [02-10 ~ 02-12] 2026학년도 1학기 예비수강신청
• [02-12 ~ 02-12] 2025학년도 전기 학위수여식
• [02-19 ~ 02-27] 1학기 등록금 수납기간(예정) / 1학기 분납신청기간(예정)
• [02-23 ~ 02-25] 1학기 재·복학생 수강신청
• [02-24 ~ 02-25] 신입생 오리엔테이션
• [02-26 ~ 02-27] 1학기 신·편입생 수강신청

**03월 March**
• [03-03 ~ 03-03] 1학기 개강 / 입학식 및 1학기 개강예배
• [03-03 ~ 03-09] 추가등록금 및 분납 1차 수납기간(예정) / 수강신청 정정
• [03-11 ~ 03-17] 연한초과자 등록금 수납기간(예정)
• [03-17 ~ 03-20] 수강신청 철회신청 및 출력
• [03-23 ~ 03-25] 수강신청 철회신청서 제출
• [03-23 ~ 03-27] 분납2차 수납기간(예정) / 다전공 철회 신청
• [03-30 ~ 03-30] 학기 개시일로부터 30일 / 수업일수 1/4

**04월 April**
• [04-08 ~ 04-08] 수업일수 1/3
• [04-13 ~ 04-17] 분납3차 수납기간(예정)
• [04-21 ~ 04-27] 중간고사
• [04-27 ~ 04-27] 수업일수 1/2
• [04-27 ~ 05-01] 다전공 신청
• [04-28 ~ 04-28] 전공진로박람회
• [04-29 ~ 04-29] 학기 개시일로부터 60일 / 한세 JOB FAIR

**05월 May**
• [05-11 ~ 05-15] 분납4차 수납기간(예정) / 재입학, 전부(과) 신청
• [05-12 ~ 05-13] 오순절 축제
• [05-14 ~ 05-14] 수업일수 2/3
• [05-22 ~ 05-29] 하계 계절학기 수강신청
• [05-25 ~ 05-25] 수업일수 3/4
• [05-29 ~ 05-29] 학기 개시일로부터 90일

**06월 June**
• [06-01 ~ 06-08] 다전공 이수인정서 제출
• [06-01 ~ 06-05] 하계 계절학기 등록(예정)
• [06-02 ~ 06-02] 종강예배
• [06-09 ~ 06-15] 보강주
• [06-16 ~ 06-22] 기말고사
• [06-16 ~ 06-26] 1학기 성적입력 및 조회
• [06-22 ~ 06-22] 1학기 종강
• [06-22 ~ 06-26] 2학기 복학 신청(1차)
• [06-23 ~ 07-13] 하계 계절학기 수업
• [06-29 ~ 07-01] 1학기 성적 정정

**07월 July**
• [07-03 ~ 07-03] 1학기 성적 확정
• [07-14 ~ 07-15] 하계 계절학기 성적입력
• [07-16 ~ 07-17] 하계 계절학기 성적정정 및 확정
• [07-20 ~ 07-31] 2학기 복학 신청(2차)

**08월 August**
• [08-18 ~ 08-18] 2025학년도 후기 졸업
• [08-18 ~ 08-31] 2학기 휴학 신청
• [08-19 ~ 08-21] 2학기 수강신청
• [08-24 ~ 08-31] 2학기 등록금 수납기간(예정) / 2학기 분납신청기간(예정)

**09월 September**
• [09-01 ~ 09-01] 2학기 개강 / 2학기 개강예배
• [09-01 ~ 09-07] 추가등록금 및 분납1차 수납기간(예정) / 수강신청 정정
• [09-09 ~ 09-15] 연한초과자 등록금 수납기간(예정)
• [09-15 ~ 09-18] 수강신청 철회신청 및 출력
• [09-21 ~ 09-29] 수강신청 철회신청서 제출
• [09-21 ~ 09-28] 분납2차 수납기간(예정) / 다전공 철회 신청
• [09-28 ~ 09-28] 수업일수 1/4
• [09-30 ~ 09-30] 학기 개시일로부터 30일

**10월 October**
• [10-06 ~ 10-06] 한세체육대회(예정)
• [10-07 ~ 10-07] 수업일수 1/3
• [10-12 ~ 10-16] 분납3차 수납기간(예정)
• [10-20 ~ 10-26] 중간고사
• [10-26 ~ 10-26] 수업일수 1/2
• [10-26 ~ 10-30] 다전공 신청
• [10-30 ~ 10-30] 학기 개시일로부터 60일

**11월 Nobember**
• [11-09 ~ 11-13] 분납4차 수납기간(예정) / 재입학, 전부(과) 신청
• [11-12 ~ 11-12] 수업일수 2/3
• [11-23 ~ 11-23] 수업일수 3/4
• [11-23 ~ 11-27] 동계 계절학기 신청
• [11-29 ~ 11-29] 학기 개시일로부터 90일
• [11-30 ~ 12-04] 다전공 이수인정서 제출 / 동계 계절학기 등록(예정)

**12월 December**
• [12-01 ~ 12-01] 2학기 종강예배
• [12-08 ~ 12-14] 보강주
• [12-15 ~ 12-21] 기말고사
• [12-15 ~ 12-24] 2학기 성적입력 및 조회
• [12-21 ~ 12-21] 2학기 종강
• [12-22 ~ 01-13] 동계 계절학기 수업
• [12-22 ~ 12-31] 1학기 복학 신청(1차)
• [12-28 ~ 12-31] 2학기 성적 정정"""

@app.post("/chat", response_model=QueryResponse)
def chat_endpoint(request: QueryRequest):
    """실제 프론트엔드 앱이 질문을 던지는 API 주소입니다."""
    prompt = request.query
    
    # 1. 가로채기: 학사일정 크롤링 (띄어쓰기 무시하고 검사)
    if "학사일정" in prompt.replace(" ", ""):
        schedule_text = scrape_academic_schedule()
        return QueryResponse(answer=schedule_text)
        
    if global_retriever is None:
        raise HTTPException(status_code=500, detail="서버에 학칙 데이터가 로드되지 않았습니다.")
    
    try:
        # 2. 다중 대화의 문맥 파악
        history_text = ""
        if hasattr(request, 'history') and request.history:
            for msg in request.history[-4:]: # 최대 최근 4개 문답만 컨텍스트로 사용
                role_name = "학생" if msg.role == "user" else "상담원"
                history_text += f"[{role_name}] {msg.content}\n"
                
        # 검색용 쿼리는 현재 문맥(이전 질문)을 포함시킬 수도 있으나 단순하게 유지
        relevant_docs = global_retriever.invoke(prompt)
        context = "\n".join([d.page_content for d in relevant_docs])
        
        # 3. LLM 호출 시 프롬프트에 문맥 포함
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        
        full_prompt = "당신은 한세대학교 학부생 상담원입니다. 아래 학칙 및 지침을 바탕으로 당당하고 친절하게 답하세요.\n"
        if history_text:
            full_prompt += f"\n[이전 대화 기록]\n{history_text}\n(위의 이전 대화의 문맥을 이어서 학생의 질문에 답변하세요.)\n"
            
        full_prompt += f"\n[관련 학칙 및 정보]\n{context}\n\n[학생 질문]\n{prompt}"
        
        response = llm.invoke(full_prompt)
        
        return QueryResponse(answer=response.content)
    except Exception as e:
        print(f"AI 통신 에러: {e}")
        raise HTTPException(status_code=500, detail="AI가 응답을 생성하는 중 오류가 발생했습니다.")

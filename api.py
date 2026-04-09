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
from langchain_community.document_loaders import PyPDFLoader
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
    """서버가 켜질 때 '학부생' 기준으로 학칙 데이터를 한 번만 로드합니다."""
    global global_retriever
    
    if "OPENAI_API_KEY" not in os.environ:
        print("경고: OPENAI_API_KEY가 설정되지 않았습니다.")
        
    target_file = "학부학칙.pdf"
    print(f"{target_file} 데이터를 로드 중입니다...")
    
    if os.path.exists(target_file):
        loader = PyPDFLoader(target_file)
        docs = loader.load()
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        temp_db = DocArrayInMemorySearch.from_documents(docs, embeddings)
        global_retriever = temp_db.as_retriever()
        print("데이터 로딩 완료!")
    else:
        print(f"오류: {target_file} 파일이 실행 폴더에 없습니다. 반드시 파일을 준비해주세요.")

# 프론트엔드에서 보낼 질문 데이터 구조
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

def scrape_academic_schedule():
    url = "https://www.hansei.ac.kr/kor/302/subview.do"
    try:
        response = requests.get(url, verify=False, timeout=10)
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser')
        
        items = soup.select('.calendar_list dl')
        if not items:
            return "현재 한세대학교 홈페이지에서 학사일정을 불러올 수 없습니다."
            
        res = "📅 **[현재 학사일정 (1월~12월)]**\n\n"
        for item in items:
            month = item.find('dt')
            if month:
                res += f"**{month.get_text(strip=True)}**\n"
            
            events = item.find_all('li')
            if not events:
                res += "- 일정이 없습니다.\n"
            for event in events:
                date_elem = event.find('span', class_='date')
                date_str = f"[{date_elem.text.strip()}]" if date_elem else ""
                
                title_elem = event.find('a')
                if not title_elem:
                    title_elem = event.find('div')
                
                if title_elem:
                    title_str = title_elem.text.strip()
                else:
                    # 필요없는 날짜 텍스트를 중복으로 가져오지 않도록 span.date 제거 후 텍스트 추출
                    if date_elem:
                        date_elem.extract()
                    title_str = event.get_text(strip=True)
                
                res += f"• {date_str} {title_str}\n"
            res += "\n"
        return res.strip()
    except Exception as e:
        print(f"Scraping error: {e}")
        return "홈페이지 연결 지연으로 학사일정을 불러오지 못했습니다."

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
        # 관련 문서 검색
        relevant_docs = global_retriever.invoke(prompt)
        context = "\n".join([d.page_content for d in relevant_docs])
        
        # LLM 호출
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        full_prompt = f"당신은 한세대학교 학부생 상담원입니다. 아래 학칙을 바탕으로 친절하고 자연스럽게 답하세요.\n\n[관련 학칙]\n{context}\n\n[학생 질문]\n{prompt}"
        
        response = llm.invoke(full_prompt)
        
        return QueryResponse(answer=response.content)
    except Exception as e:
        print(f"AI 통신 에러: {e}")
        raise HTTPException(status_code=500, detail="AI가 응답을 생성하는 중 오류가 발생했습니다.")

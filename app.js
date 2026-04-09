// 챗봇 이전 대화 기억(Context Memory) (최대 8개)
let chatHistory = [];

function handleEnter(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
}

function sendQuickMessage(text) {
    appendUserMessage(text);
    fetchChatResponse(text);
}

function sendMessage() {
    const input = document.getElementById('chatInput');
    const text = input.value.trim();
    
    if (text === '') return;
    
    appendUserMessage(text);
    input.value = '';
    
    fetchChatResponse(text);
}

async function fetchChatResponse(text) {
    // 로딩 말풍선 표시
    const chatbox = document.getElementById('chatbox');
    const row = document.createElement('div');
    row.className = 'message-row bot-row';
    const avatar = document.createElement('div');
    avatar.className = 'bot-avatar';
    const icon = document.createElement('i');
    icon.setAttribute('data-lucide', 'bot');
    avatar.appendChild(icon);
    
    const bubble = document.createElement('div');
    bubble.className = 'message-bubble bot-bubble';
    bubble.textContent = '자료를 검토 중입니다...잠시만 기다려주세요 👀';
    
    row.appendChild(avatar);
    row.appendChild(bubble);
    chatbox.appendChild(row);
    lucide.createIcons({ root: row });
    scrollToBottom();

    // 서버에 요청 전송
    try {
        const response = await fetch('https://hansei-chatbot3.onrender.com/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: text, history: chatHistory })
        });
        
        if (!response.ok) throw new Error('서버 에러');
        const data = await response.json();
        
        // 대화 기록 저장 (어시스턴트 응답과 함께)
        chatHistory.push({ role: 'user', content: text });
        chatHistory.push({ role: 'assistant', content: data.answer });
        // 최근 8개 대화만 유지
        if (chatHistory.length > 8) {
            chatHistory = chatHistory.slice(-8);
        }
        
        // 줄바꿈 문자를 <br>로 치환하여 말풍선 갱신
        bubble.innerHTML = data.answer.replace(/\n/g, '<br>');
    } catch (error) {
        bubble.innerHTML = '⚠️ 서버 연결 실패<br><span style="font-size:0.8rem; color:#666;">로컬에서 FastAPI(api.py)를 실행했는지 확인해주세요.</span>';
        console.error(error);
    }
    scrollToBottom();
}

function appendUserMessage(text) {
    const chatbox = document.getElementById('chatbox');
    
    const row = document.createElement('div');
    row.className = 'message-row user-row';
    
    const bubble = document.createElement('div');
    bubble.className = 'message-bubble user-bubble';
    bubble.textContent = text;
    
    row.appendChild(bubble);
    chatbox.appendChild(row);
    
    scrollToBottom();
}

function appendBotMessage(text) {
    const chatbox = document.getElementById('chatbox');
    
    const row = document.createElement('div');
    row.className = 'message-row bot-row';
    
    const avatar = document.createElement('div');
    avatar.className = 'bot-avatar';
    
    // Create Lucide icon element
    const icon = document.createElement('i');
    icon.setAttribute('data-lucide', 'bot');
    avatar.appendChild(icon);
    
    const bubble = document.createElement('div');
    bubble.className = 'message-bubble bot-bubble';
    bubble.textContent = text;
    
    row.appendChild(avatar);
    row.appendChild(bubble);
    chatbox.appendChild(row);
    
    // Re-render lucide icons inside the new row
    lucide.createIcons({ root: row });
    
    scrollToBottom();
}

function scrollToBottom() {
    const chatbox = document.getElementById('chatbox');
    chatbox.scrollTop = chatbox.scrollHeight;
}

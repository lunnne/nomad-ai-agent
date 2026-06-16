import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import Agent, Runner, WebSearchTool, SQLiteSession, set_default_openai_client, set_tracing_disabled
import httpx
import streamlit as st


load_dotenv()

set_tracing_disabled(True)

set_default_openai_client(
    AsyncOpenAI(
        http_client=httpx.AsyncClient(verify=False) 
    )
)

# Streamlit 기본 설정
st.set_page_config(
    page_title="Life Coach Agent",
    page_icon="🌱",
)

st.title("🌱 Life Coach Agent")
st.write("동기부여, 습관 형성, 자기 개발 팁을 도와주는 AI 라이프 코치입니다.")

# Agent 생성
if "agent" not in st.session_state:
    st.session_state["agent"] = Agent(
        name="Life Coach Agent",
        instructions="""
        You are a warm, encouraging, and practical life coach.

        Your job:
        - Help users with motivation, self-development, habit building, and personal growth.
        - Give kind, supportive, and realistic advice.
        - When the user asks about current methods, research-backed tips, trends, or something you are unsure about,
          use the Web Search Tool first.
        - Explain advice in a simple and actionable way.
        - Do not sound judgmental.
        - Always encourage the user gently.
        """,
        tools=[
            WebSearchTool(),
        ],
    )

agent = st.session_state["agent"]

# 대화 기억용 SQLite 세션
if "session" not in st.session_state:
    # 같은 session_id를 쓰면 대화가 이어짐
    st.session_state["session"] = SQLiteSession("life_coach_user")

session = st.session_state["session"]

# 화면에 보여줄 채팅 기록
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 이전 대화 출력
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# Agent 실행 함수
async def run_agent(user_message):
    response_text = ""

    with st.chat_message("assistant"):
        status_container = st.status("⏳ Thinking...", expanded=False)
        text_placeholder = st.empty()

        # Runner가 Agent를 실행함
        result = Runner.run_streamed(
            agent,
            input=user_message,
            session=session,
        )

        async for event in result.stream_events():
            # 웹 검색 상태 표시
            if event.type == "raw_response_event":
                event_type = event.data.type

                if event_type == "response.web_search_call.in_progress":
                    status_container.update(
                        label="🔍 Searching the web...",
                        state="running",
                    )

                elif event_type == "response.web_search_call.completed":
                    status_container.update(
                        label="✅ Web search completed.",
                        state="complete",
                    )

                # 답변이 스트리밍으로 들어오는 부분
                elif event_type == "response.output_text.delta":
                    response_text += event.data.delta
                    text_placeholder.write(response_text.replace("$", "\\$"))

        status_container.update(label="✅ Done", state="complete")

    return response_text


# 사용자 입력
prompt = st.chat_input("Ask your life coach anything!")

if prompt:
    # 유저 메시지 화면 출력
    with st.chat_message("user"):
        st.write(prompt)

    # 화면용 기록 저장
    st.session_state["messages"].append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    # Agent 실행
    answer = asyncio.run(run_agent(prompt))

    # assistant 답변 기록 저장
    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": answer,
        }
    )
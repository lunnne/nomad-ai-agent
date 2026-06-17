import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI,OpenAI
from agents import  Agent, Runner, WebSearchTool, SQLiteSession, set_default_openai_client, set_tracing_disabled
from agents import FileSearchTool
import httpx
import streamlit as st


load_dotenv()

set_tracing_disabled(True)

set_default_openai_client(
    AsyncOpenAI(
        http_client=httpx.AsyncClient(
            verify=False,
            trust_env=False,
        ) 
    )
)

client = OpenAI(
    http_client=httpx.Client(
        verify=False,
        trust_env=False,
    )   
)

VECTOR_STORE_ID = "vs_6a329239f8b4819187b23f7bf6649225"

st.title("🌱 Life Coach Agent")
st.write("목표와 일기를 기억하고, 웹 검색까지 활용하는 개인 라이프 코치입니다.")

# Agent 생성
if "agent" not in st.session_state:
    st.session_state["agent"] = Agent(
        name="Life Coach Agent",
        instructions="""
        You are a warm, encouraging, and practical life coach.

        Your job:
        - Help the user with motivation, self-development, habit building, and personal growth.
        - When the user asks about their personal goals, diary entries, progress, routines, or past plans,
          use the File Search Tool first.
        - When the user asks for current or research-based advice, use the Web Search Tool.
        - Combine personal information from files with web search results to give personalized advice.
        - Track the user's progress over time based on uploaded goal documents and diary entries.
        - Be kind, supportive, realistic, and encouraging.
        """,
        tools=[
            WebSearchTool(),
            FileSearchTool(
                vector_store_ids=[VECTOR_STORE_ID],
                max_num_results=3,
            ),
        ],
    )

agent = st.session_state["agent"]
# 대화 기억 Session

if "session" not in st.session_state:
    st.session_state["session"] = SQLiteSession(
        "life-coach-history",
        "life-coach-memory.db",
    )

session = st.session_state["session"]
# 이전 대화 출력
async def paint_history():
    messages = await session.get_items()

    for message in messages:
        if "role" in message:
            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    st.write(message["content"])
                else:
                    if message["type"] == "message":
                        st.write(message["content"][0]["text"].replace("$", "\\$"))

        if "type" in message:
            if message["type"] == "web_search_call":
                with st.chat_message("ai"):
                    st.write("🔍 Searched the web...")
            elif message["type"] == "file_search_call":
                with st.chat_message("ai"):
                    st.write("🗂️ Searched your goal files...")


asyncio.run(paint_history())

# Tool 상태 표시
def update_status(status_container, event_type):
    status_messages = {
        "response.web_search_call.in_progress": (
            "🔍 Starting web search...",
            "running",
        ),
        "response.web_search_call.searching": (
            "🔍 Web search in progress...",
            "running",
        ),
        "response.web_search_call.completed": (
            "✅ Web search completed.",
            "complete",
        ),
        "response.file_search_call.in_progress": (
            "🗂️ Starting file search...",
            "running",
        ),
        "response.file_search_call.searching": (
            "🗂️ Searching your goal files...",
            "running",
        ),
        "response.file_search_call.completed": (
            "✅ File search completed.",
            "complete",
        ),
        "response.completed": (
            "✅ Done",
            "complete",
        ),
    }

    if event_type in status_messages:
        label, state = status_messages[event_type]
        status_container.update(label=label, state=state)


# Agent 실행

async def run_agent(message):
    with st.chat_message("ai"):
        status_container = st.status("⏳ Thinking...", expanded=False)
        text_placeholder = st.empty()
        response = ""

        stream = Runner.run_streamed(
            agent,
            message,
            session=session,
        )

        async for event in stream.stream_events():
            if event.type == "raw_response_event":
                update_status(status_container, event.data.type)

                if event.data.type == "response.output_text.delta":
                    response += event.data.delta
                    text_placeholder.write(response.replace("$", "\\$"))


# 사용자 입력 + 파일 업로드

prompt = st.chat_input(
    "Ask your life coach anything or upload your goal/diary file!",
    accept_file=True,
    file_type=["txt", "pdf"],
)

if prompt:
    # 파일 업로드 처리
    for file in prompt.files:
        with st.chat_message("ai"):
            with st.status("⏳ Uploading your goal file...") as status:
                uploaded_file = client.files.create(
                    file=(file.name, file.getvalue()),
                    purpose="user_data",
                )

                status.update(label="⏳ Attaching file to vector store...")

                client.vector_stores.files.create(
                    vector_store_id=VECTOR_STORE_ID,
                    file_id=uploaded_file.id,
                )

                status.update(
                    label=f"✅ {file.name} uploaded and attached!",
                    state="complete",
                )
 # 텍스트 질문 처리
    if prompt.text:
        with st.chat_message("human"):
            st.write(prompt.text)

        asyncio.run(run_agent(prompt.text))



# 사이드바
with st.sidebar:
    reset = st.button("Reset memory")

    if reset:
        asyncio.run(session.clear_session())
        st.rerun()

    st.write("### Memory")
    st.write(asyncio.run(session.get_items()))

import asyncio
import httpx
import streamlit as st
from dotenv import load_dotenv
from openai import AsyncOpenAI,OpenAI
from agents import  Agent, Runner, handoff, set_default_openai_client, set_tracing_disabled
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX


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


# 1. 전문 에이전트 만들기
menu_agent = Agent(
    name="menu_agent",
    handoff_description="메뉴, 재료, 알레르기, 채식 메뉴 질문을 담당합니다.",
    instructions=f"""
You are a restaurant menu expert.
""",
)

# 2. 일반 에이전트 만들기
menu_agent = Agent(
    name="menu_agent",
    handoff_description="메뉴, 재료, 알레르기, 채식 메뉴 질문을 담당합니다.",
    instructions=f"""
{RECOMMENDED_PROMPT_PREFIX}

너는 레스토랑의 메뉴 전문가입니다.

역할:
- 메뉴 추천
- 재료 설명
- 알레르기 정보 안내
- 채식/비건 메뉴 안내

규칙:
- 모르는 정보는 지어내지 말고 확인이 필요하다고 말하세요.
- 고객에게 친절하고 짧게 답하세요.

예시 메뉴:
- Bibimbap: vegetables, rice, egg, gochujang
- Vegan Bibimbap: vegetables, rice, tofu, gochujang
- Bulgogi: beef, soy sauce marinade, rice
- Kimchi Stew: kimchi, pork, tofu
- Japchae: glass noodles, vegetables, soy sauce
""",
)

order_agent = Agent(
    name="order_agent",
    handoff_description="음식 주문을 받고, 주문 내용을 확인합니다.",
    instructions=f"""
{RECOMMENDED_PROMPT_PREFIX}

너는 레스토랑의 주문 담당자입니다.

역할:
- 고객의 주문 받기
- 메뉴 이름과 수량 확인하기
- 주문이 애매하면 다시 질문하기
- 마지막에 주문 내용을 요약해서 확인받기

규칙:
- 결제는 처리하지 않습니다.
- 배달 주소나 픽업 시간 정보가 필요하면 물어보세요.
""",
)

reservation_agent = Agent(
    name="reservation_agent",
    handoff_description="테이블 예약, 인원수, 날짜, 시간 확인을 담당합니다.",
    instructions=f"""
{RECOMMENDED_PROMPT_PREFIX}

너는 레스토랑의 예약 담당자입니다.

역할:
- 예약 날짜 확인
- 예약 시간 확인
- 인원수 확인
- 고객 이름 확인
- 필요한 정보가 다 모이면 예약 내용을 요약해서 확인받기

규칙:
- 실제 예약 시스템에 저장했다고 말하지 마세요.
- 대신 '예약 요청 내용을 확인했습니다'라고 말하세요.
""",
)

# 2. Handoff 발생 시 UI에 보여줄 메시지
def show_menu_handoff(ctx):
    st.session_state.handoff_messages.append("🍽️ 메뉴 전문가에게 연결합니다...")


def show_order_handoff(ctx):
    st.session_state.handoff_messages.append("🛒 주문 담당에게 연결합니다...")


def show_reservation_handoff(ctx):
    st.session_state.handoff_messages.append("📅 예약 담당에게 연결합니다...")

# 3. Triage Agent 만들기
triage_agent = Agent(
    name="Triage Agent",
    instructions=f"""
{RECOMMENDED_PROMPT_PREFIX}

너는 레스토랑의 첫 응대 직원입니다.

너의 가장 중요한 역할:
- 고객의 요청을 파악합니다.
- 직접 길게 답하지 않습니다.
- 아래 전문 에이전트 중 가장 적절한 에이전트에게 handoff합니다.

라우팅 규칙:
1. 메뉴, 재료, 알레르기, 채식/비건 질문이면 Menu Agent로 handoff
2. 음식 주문, 포장 주문, 픽업 주문이면 Order Agent로 handoff
3. 테이블 예약, 날짜, 시간, 인원수 관련이면 Reservation Agent로 handoff

고객이 중간에 주제를 바꾸면 새로운 요청에 맞는 에이전트로 다시 handoff하세요.
""",
    handoffs=[
        handoff(agent=menu_agent, on_handoff=show_menu_handoff),
        handoff(agent=order_agent, on_handoff=show_order_handoff),
        handoff(agent=reservation_agent, on_handoff=show_reservation_handoff),
    ],
)

# 4. Streamlit UI
st.title("🍜 Restaurant Bot with Handoffs")

st.write("메뉴 질문, 주문, 예약 요청을 해보세요.")

# 대화 기록 저장
if "messages" not in st.session_state:
    st.session_state.messages = []

# handoff 표시 메시지 저장
if "handoff_messages" not in st.session_state:
    st.session_state.handoff_messages = []

# 이전 대화 화면에 보여주기
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 사용자 입력창
user_input = st.chat_input("무엇을 도와드릴까요?")

if user_input:
    # 사용자 메시지 저장
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
    })

    with st.chat_message("user"):
        st.write(user_input)

    # 이번 턴의 handoff 메시지만 보여주기 위해 초기화
    st.session_state.handoff_messages = []

    async def run_bot():
        # Runner.run()은 agent와 input을 받아 실행합니다.
        result = await Runner.run(
            triage_agent,
            user_input,
        )
        return result.final_output

    # Agents SDK 실행
    final_answer = asyncio.run(run_bot())

    # handoff가 있었다면 먼저 표시
    for handoff_message in st.session_state.handoff_messages:
        st.info(handoff_message)

    # 봇 답변 표시
    with st.chat_message("assistant"):
        st.write(final_answer)

    # 봇 답변 저장
    st.session_state.messages.append({
        "role": "assistant",
        "content": final_answer,
    })
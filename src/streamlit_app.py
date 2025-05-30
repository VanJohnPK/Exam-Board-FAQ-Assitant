import asyncio
import os
import urllib.parse
from collections.abc import AsyncGenerator

import streamlit as st
from dotenv import load_dotenv
from pydantic import ValidationError
from streamlit.runtime.scriptrunner import get_script_run_ctx

from client import AgentClient, AgentClientError
from schema import ChatHistory, ChatMessage
from schema.task_data import TaskData, TaskDataStatus

# A Streamlit app for interacting with the langgraph agent via a simple chat interface.
# The app has three main functions which are all run async:

# - main() - sets up the streamlit app and high level structure
# - draw_messages() - draws a set of chat messages - either replaying existing messages
#   or streaming new ones.
# - handle_feedback() - Draws a feedback widget and records feedback from the user.

# The app heavily uses AgentClient to interact with the agent's FastAPI endpoints.


APP_TITLE = "Exam Board Assistant"
APP_ICON = "🐹"


async def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        menu_items={},
    )

    # Hide the streamlit upper-right chrome
    # st.html(
    #     """
    #     <style>
    #     [data-testid="stStatusWidget"] {
    #             visibility: hidden;
    #             height: 0%;
    #             position: fixed;
    #         }
    #     </style>
    #     """,
    # )
    if st.get_option("client.toolbarMode") != "minimal":
        st.set_option("client.toolbarMode", "minimal")
        await asyncio.sleep(0.1)
        st.rerun()

    if "agent_client" not in st.session_state:
        load_dotenv()
        agent_url = os.getenv("AGENT_URL")
        if not agent_url:
            host = os.getenv("HOST", "127.0.0.1")
            port = os.getenv("PORT", 8080)
            agent_url = f"http://{host}:{port}"
        try:
            with st.spinner("Connecting to agent service..."):
                st.session_state.agent_client = AgentClient(base_url=agent_url)
        except AgentClientError as e:
            st.error(f"Error connecting to agent service at {agent_url}: {e}")
            st.markdown("The service might be booting up. Try again in a few seconds.")
            st.stop()
    agent_client: AgentClient = st.session_state.agent_client

    if "thread_id" not in st.session_state:
        thread_id = st.query_params.get("thread_id")
        if not thread_id:
            thread_id = get_script_run_ctx().session_id
            messages = []
        else:
            try:
                messages: ChatHistory = agent_client.get_history(thread_id=thread_id).messages
            except AgentClientError:
                st.error("No message history found for this Thread ID.")
                messages = []
        st.session_state.messages = messages
        st.session_state.thread_id = thread_id

    # Config options
    with st.sidebar:
        st.header(f"{APP_ICON} {APP_TITLE}")
        ""
        "Agentic RAG service built with LangGraph, FastAPI and Streamlit"
        with st.popover(":material/settings: Settings", use_container_width=True):
            model_idx = agent_client.info.models.index(agent_client.info.default_model)
            model = st.selectbox("LLM to use", options=agent_client.info.models, index=model_idx)
            agent_list = [a.key for a in agent_client.info.agents]
            agent_idx = agent_list.index(agent_client.info.default_agent)
            agent_client.agent = st.selectbox(
                "Agent to use",
                options=agent_list,
                index=agent_idx,
            )
            use_streaming = st.toggle("Stream results", value=True)

        @st.dialog("Architecture")
        def architecture_dialog() -> None:
            st.image(
                "https://github.com/JoshuaC215/agent-service-toolkit/blob/main/media/agent_architecture.png?raw=true"
            )
            "[View full size on Github](https://github.com/JoshuaC215/agent-service-toolkit/blob/main/media/agent_architecture.png)"
            st.caption(
                "App hosted on [Streamlit Cloud](https://share.streamlit.io/) with FastAPI service running in [Azure](https://learn.microsoft.com/en-us/azure/app-service/)"
            )

        if st.button(":material/schema: Architecture", use_container_width=True):
            architecture_dialog()

        with st.popover(":material/policy: Privacy", use_container_width=True):
            st.write(
                "Prompts, responses and feedback in this app are anonymously recorded and saved to LangSmith for product evaluation and improvement purposes only."
            )

        @st.dialog("Share/resume chat")
        def share_chat_dialog() -> None:
            session = st.runtime.get_instance()._session_mgr.list_active_sessions()[0]
            st_base_url = urllib.parse.urlunparse(
                [session.client.request.protocol, session.client.request.host, "", "", "", ""]
            )
            # if it's not localhost, switch to https by default
            if not st_base_url.startswith("https") and "localhost" not in st_base_url:
                st_base_url = st_base_url.replace("http", "https")
            chat_url = f"{st_base_url}?thread_id={st.session_state.thread_id}"
            st.markdown(f"**Chat URL:**\n```text\n{chat_url}\n```")
            st.info("Copy the above URL to share or revisit this chat")

        if st.button(":material/upload: Share/resume chat", use_container_width=True):
            share_chat_dialog()

        "[View the reference](https://github.com/JoshuaC215/agent-service-toolkit)"
        st.caption(
            "Made with :material/favorite: by [darkVinci](https://github.com/VanJohnPK) in Shanghai"
        )

    # Draw existing messages
    messages: list[ChatMessage] = st.session_state.messages

    if len(messages) == 0:
        WELCOME = "你好，我是上海考试院小助手，擅长解答高考学考和中考中招相关的问题。"
        # WELCOME = "你好，我是上海考试院小助手，擅长解答高考学考、中考中招、研考成考、自学考试和证书考试相关的问题。"
        with st.chat_message("ai"):
            st.write(WELCOME)

    # draw_messages() expects an async iterator over messages
    async def amessage_iter() -> AsyncGenerator[ChatMessage, None]:
        for m in messages:
            yield m

    await draw_messages(amessage_iter())

    # Generate new message if the user provided new input
    if user_input := st.chat_input():
        messages.append(ChatMessage(type="human", content=user_input))
        st.chat_message("human").write(user_input)
        try:
            if use_streaming:
                stream = agent_client.astream(
                    message=user_input,
                    model=model,
                    thread_id=st.session_state.thread_id,
                )
                await draw_messages(stream, is_new=True)
            else:
                response = await agent_client.ainvoke(
                    message=user_input,
                    model=model,
                    thread_id=st.session_state.thread_id,
                )
                messages.append(response)
                st.chat_message("ai").write(response.content)
            st.rerun()  # Clear stale containers
        except AgentClientError as e:
            st.error(f"Error generating response: {e}")
            st.stop()

    # # If messages have been generated, show feedback widget
    # if len(messages) > 0 and st.session_state.last_message:
    #     with st.session_state.last_message:
    #         await handle_feedback()


import re

async def draw_messages(
    messages_agen: AsyncGenerator[ChatMessage | str, None],
    is_new: bool = False,
) -> None:
    last_message_type = None
    st.session_state.last_message = None

    streaming_content = ""
    streaming_placeholder = None

    while msg := await anext(messages_agen, None):
        if isinstance(msg, str):
            if not streaming_placeholder:
                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai")
                with st.session_state.last_message:
                    streaming_placeholder = st.empty()

            streaming_content += msg
            streaming_placeholder.write(streaming_content)
            continue

        if not isinstance(msg, ChatMessage):
            st.error(f"Unexpected message type: {type(msg)}")
            st.write(msg)
            st.stop()

        match msg.type:
            case "human":
                last_message_type = "human"
                st.chat_message("human").write(msg.content)

            case "ai":
                if is_new:
                    st.session_state.messages.append(msg)

                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai")

                tool_outputs = []

                # 所有内容统一在这一个 chat_message 容器下
                with st.session_state.last_message:
                    # 1️⃣ 先渲染 LLM 主回答内容
                    if msg.content:
                        if streaming_placeholder:
                            streaming_placeholder.write(msg.content)
                            streaming_content = ""
                            streaming_placeholder = None
                        else:
                            st.write(msg.content)

                    # 2️⃣ 如果有 Tool Call，我们现在不立刻显示，而是延后处理
                    if msg.tool_calls:
                        call_results = {}

                        for tool_call in msg.tool_calls:
                            status = st.status(
                                f"""Tool Call: {tool_call["name"]}""",
                                state="running" if is_new else "complete",
                            )
                            call_results[tool_call["id"]] = status
                            status.write("调用中...")

                        for _ in range(len(call_results)):
                            tool_result: ChatMessage = await anext(messages_agen)
                            if tool_result.type != "tool":
                                st.error(f"Unexpected ChatMessage type: {tool_result.type}")
                                st.write(tool_result)
                                st.stop()

                            if is_new:
                                st.session_state.messages.append(tool_result)

                            # 清理问答格式
                            cleaned_items = []
                            for line in tool_result.content.split("\n"):
                                line = line.strip()
                                if line.startswith("问"):
                                    question = line
                                elif line.startswith("答"):
                                    answer = line
                                    cleaned_items.append((question, answer))
                                elif "问：" in line and "答：" in line:
                                    match = re.findall(r"(问[:：].*?)(答[:：].*)", line)
                                    if match:
                                        cleaned_items.append(match[0])
                                elif "问：" in line:
                                    question = line
                                elif "答：" in line:
                                    answer = line
                                    cleaned_items.append((question, answer))

                            if cleaned_items:
                                tool_outputs.extend(cleaned_items)

                            status = call_results[tool_result.tool_call_id]
                            status.update(label="调用完成", state="complete")

                    # 3️⃣ 所有 tool call 处理完之后，在同一 chat_message 中展示它们
                    if tool_outputs:
                        st.markdown("---")
                        st.markdown("📂 **相关问答参考：**")
                        for i, (q, a) in enumerate(tool_outputs[:2]):
                            st.markdown(
                                f'<div style="color: gray; font-size: 0.95em;">'
                                f'<strong>{q.strip()}</strong><br>{a.strip()}'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                        st.markdown("---")
                        st.markdown("📂 **智能助手回答：**")


            case "custom":
                try:
                    task_data: TaskData = TaskData.model_validate(msg.custom_data)
                except ValidationError:
                    st.error("Unexpected CustomData message received from agent")
                    st.write(msg.custom_data)
                    st.stop()

                if is_new:
                    st.session_state.messages.append(msg)

                if last_message_type != "task":
                    last_message_type = "task"
                    st.session_state.last_message = st.chat_message(
                        name="task", avatar=":material/manufacturing:"
                    )
                    with st.session_state.last_message:
                        status = TaskDataStatus()

                status.add_and_draw_task_data(task_data)

            case _:
                st.error(f"Unexpected ChatMessage type: {msg.type}")
                st.write(msg)
                st.stop()


# async def handle_feedback() -> None:
#     """Draws a feedback widget and records feedback from the user."""

#     # Keep track of last feedback sent to avoid sending duplicates
#     if "last_feedback" not in st.session_state:
#         st.session_state.last_feedback = (None, None)

#     latest_run_id = st.session_state.messages[-1].run_id
#     feedback = st.feedback("stars", key=latest_run_id)

#     # If the feedback value or run ID has changed, send a new feedback record
#     if feedback is not None and (latest_run_id, feedback) != st.session_state.last_feedback:
#         # Normalize the feedback value (an index) to a score between 0 and 1
#         normalized_score = (feedback + 1) / 5.0

#         agent_client: AgentClient = st.session_state.agent_client
#         try:
#             await agent_client.acreate_feedback(
#                 run_id=latest_run_id,
#                 key="human-feedback-stars",
#                 score=normalized_score,
#                 kwargs={"comment": "In-line human feedback"},
#             )
#         except AgentClientError as e:
#             st.error(f"Error recording feedback: {e}")
#             st.stop()
#         st.session_state.last_feedback = (latest_run_id, feedback)
#         st.toast("Feedback recorded", icon=":material/reviews:")


if __name__ == "__main__":
    asyncio.run(main())

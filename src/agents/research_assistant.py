from typing import Literal, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph, START
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import ToolNode, tools_condition
# from agents.tools import calculator
from agents.rag_tool import retriever_tools
from core import get_model, settings
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
class AgentState(MessagesState, total=False):
    """`total=False` is PEP589 specs.

    documentation: https://typing.readthedocs.io/en/latest/spec/typeddict.html#totality
    """

    remaining_steps: RemainingSteps


tools = retriever_tools
instructions = f"""
    你是一个有用的上海考试院问答助手，擅长通过检索准确回答高考学考、中考中招、研考成考、自学考试和证书考试相关问题。
    """


def wrap_model(model: BaseChatModel, tools: Optional[list] = None, instructions: Optional[str] = None) -> RunnableSerializable[AgentState, AIMessage]:
    # 如果传入了 tools 参数，则将工具绑定到模型上
    if tools is not None:
        model = model.bind_tools(tools)
    # 如果传入了 instructions 参数，则创建预处理器
    if instructions is not None:
        preprocessor = RunnableLambda(lambda state: [SystemMessage(content=instructions)] + state["messages"],)
    else:
        preprocessor = RunnableLambda(lambda state: state["messages"])
    # 如果没有传入 instructions 参数，则直接返回模型
    return preprocessor | model

async def agent(state: AgentState, config: RunnableConfig) -> AgentState:
    print("---CALL AGENT---")
    # print(state["messages"][-1])
    print(state["remaining_steps"])
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(m, tools, instructions)
    if state["remaining_steps"] < 15:
        model_runnable = wrap_model(m, tools=None, instructions=instructions)
    response = await model_runnable.ainvoke(state, config)

    # We return a list, because this will get added to the existing list
    print("---AGENT FINISH---")
    return {"messages": [response]}

async def rewrite(state: AgentState, config: RunnableConfig) -> AgentState:
    print("---REWRITE QUERY---")
    # 获取最后一个 HumanMessage 的内容
    messages = state["messages"]
    last_human_message = next((msg for msg in reversed(messages) if isinstance(msg, HumanMessage)), None)
    if last_human_message is None:
        raise ValueError("No HumanMessage found in the state messages.")
    question = last_human_message.content

    msg = [
        HumanMessage(
            content=f""" \n 
                观察输入内容，并尝试推断其潜在的语义意图或含义。 \n
                以下是最初的问题：
                \n ------- \n
                {question} 
                \n ------- \n
                构思一个更清晰明确的问题：""",
        )
    ]
    # Grader
    model = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    response = model.invoke(msg)
    print(f"rewrite message: {response}")
    print("---FINISH REWRITE---")
    return {"messages": [response]}

async def generate(state: AgentState, config: RunnableConfig) -> AgentState:
    print("---GENERATE---")
    # 获取最后一个 HumanMessage 的内容
    messages = state["messages"]
    last_human_message = next((msg for msg in reversed(messages) if isinstance(msg, HumanMessage)), None)
    if last_human_message is None:
        raise ValueError("No HumanMessage found in the state messages.")
    question = last_human_message.content
    last_message = state["messages"][-1]

    docs = last_message.content

    # Prompt
    prompt = PromptTemplate(
        template="""你是一个用于问答任务的助手。使用以下检索到的上下文片段来回答问题。如果你不知道答案，就直接说不知道。最多使用三个句子，并且回答内容以Context为主。\n
        Question: \n\n {question} \n\n
        Context: {context} \n
        Answer:""",
        input_variables=["context", "question"],
    )

    # LLM
    llm = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))


    # Chain
    rag_chain = prompt | llm

    # Run
    response = rag_chain.invoke({"context": docs, "question": question})
    print("---FINISH GENERATE---")
    return {"messages": [response]}

async def grade_documents(state: AgentState, config: RunnableConfig) -> Literal["generate", "rewrite"]:
    print("---CHECK RELEVANCE---")
    # Data model
    class grade(BaseModel):
        """Binary score for relevance check."""

        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

     # LLM
    model = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))

    # LLM with tool and validation
    llm_with_tool = model.with_structured_output(grade)

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )

    # Chain
    chain = prompt | llm_with_tool

    messages = state["messages"]
    last_message = messages[-1]

    question = messages[0].content
    docs = last_message.content

    scored_result = chain.invoke({"question": question, "context": docs})

    score = scored_result.binary_score

    if score == "yes":
        print("---DECISION: RELEVANT -> generate---")
        return "generate"

    else:
        print("---DECISION: IRRELEVANT -> rewrite---")
        print(score)
        return "rewrite"



# Define the graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent)
retrieve = ToolNode(tools)
workflow.add_node("retrieve", retrieve)  # retrieval
workflow.add_node("rewrite", rewrite)  # Re-writing the question
workflow.add_node("generate", generate)  # Generating a response after we know the documents are relevant
# Call agent node to decide to retrieve or not
workflow.add_edge(START, "agent")

# Decide whether to retrieve
workflow.add_conditional_edges(
    "agent",
    # Assess agent decision
    tools_condition,
    {
        # Translate the condition outputs to nodes in our graph
        "tools": "retrieve",
        END: END,
    },
)

# Edges taken after the `action` node is called.
workflow.add_conditional_edges(
    "retrieve",
    # Assess agent decision
    grade_documents,
)
workflow.add_edge("generate", END)
workflow.add_edge("rewrite", "agent")

research_assistant = workflow.compile(checkpointer=MemorySaver())

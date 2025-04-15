import os
from dotenv import load_dotenv

import chainlit as cl
from chainlit.types import ThreadDict
from typing import cast, Dict, Optional

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from chainlit.input_widget import Select,Switch,Slider

# Load environment variables
load_dotenv()

# Optional GitHub OAuth handler
# @cl.oauth_callback
# def oauth_callback(
#     provider_id: str,
#     token: str,
#     raw_user_data: Dict[str, str],
#     default_user: cl.User,
# ) -> Optional[cl.User]:
#     print(f"Provider: {provider_id}")
#     print(f"User data: {raw_user_data}")
#     return default_user


@cl.on_chat_start
async def on_chat_start():
    # Initialize model
    model = ChatOpenAI(
        base_url=os.getenv("OPENROUTER_BASE_URL"),
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model="openai/gpt-4.1-mini",
        temperature=0.6,
        max_tokens=4096,
    )

    # Set up prompt
    prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an advanced, intelligent, and articulate assistant designed to deliver responses that are **thorough, well-structured, and insightful**. 
When answering user queries:
- Provide **clear definitions**, **background context**, and **step-by-step explanations** where relevant.
- Offer **real-world examples**, **analogies**, or **case studies** to reinforce understanding.
- If the topic has **multiple perspectives**, include them and explain their differences.
- Present information in a **logical flow**, optionally using headings, bullet points, or numbered steps to enhance readability.
- Maintain a **friendly and respectful tone** while delivering **in-depth analysis**, **rationale**, and **recommendations**.
- Use markdown-style formatting if appropriate (e.g., `**bold**`, bullet points, etc.) to make long responses easier to scan.

Assume the user values depth, nuance, and clarity over brevity. Do not oversimplify unless explicitly asked. You are free to expand your answers as needed to ensure deep understanding.
""",
        ),
        ("human", "{question}"),
    ]
)


    # Combine prompt, model, and output parser
    runnable = prompt | model | StrOutputParser()

    # Store runnable in session
    cl.user_session.set("runnable", runnable)

    # Welcome message
    await cl.Message(content="Hey there! 👋 How can I assist you today?").send()


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cast(Runnable, cl.user_session.get("runnable"))
    msg = cl.Message(content="")

    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()

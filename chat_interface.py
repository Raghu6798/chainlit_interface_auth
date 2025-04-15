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
                "You're a helpful day-to-day assistant. Be concise, friendly, and informative.",
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

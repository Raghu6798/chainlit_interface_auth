import os 

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from chainlit.types import ThreadDict
from dotenv import load_dotenv
from langchain.schema.runnable.config import RunnableConfig
from typing import cast,Dict,Optional
import chainlit as cl

load_dotenv()
    model = ChatOpenAI(
        base_url=os.getenv("OPENROUTER_BASE_URL"),
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model="meta-llama/llama-4-maverick:free",
        temperature=0.6,
        max_tokens=4096
    )



@cl.on_chat_start
async def on_chat_start():
    model = ChatOpenAI(
        base_url=os.getenv("OPENROUTER_BASE_URL"),
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model="meta-llama/llama-4-maverick:free",
        temperature=0.6,
        max_tokens=4096
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You're a very knowledgeable AI assistant who provides accurate and eloquent answers to questions related to Large Language Models",
            ),
            ("human", "{question}"),
        ]
    )
    runnable = prompt | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cast(Runnable, cl.user_session.get("runnable"))  # type: Runnable

    msg = cl.Message(content="")

    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()

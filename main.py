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

@cl.oauth_callback
def oauth_callback(
    provider_id: str,  # ID of the OAuth provider (GitHub)
    token: str,  # OAuth access token
    raw_user_data: Dict[str, str],  # User data from GitHub
    default_user: cl.User,  # Default user object from Chainlit
) -> Optional[cl.User]:  # Return User object or None
    """
    Handle the OAuth callback from GitHub
    Return the user object if authentication is successful, None otherwise
    """

    print(f"Provider: {provider_id}")  
    print(f"User data: {raw_user_data}") 

    return default_user 

@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set("history", []) 

    await cl.Message(
        content="Hello! How can I help you today?"
    ).send()  # Send welcome message


@cl.on_chat_start
async def init_safety_guardrails():
    # Initialize with medical disclaimers
    await cl.Message(
        content="""**Medical Use Disclaimer**:
This AI assistant is designed to support clinical documentation only. 
- All outputs must be reviewed by a licensed physician
- Never rely solely on AI-generated medical content
- You remain responsible for all clinical decisions"""
    ).send()

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
                """You are MediScribe AI, a specialized clinical assistant designed to help doctors with prescription documentation and patient management. Follow these guidelines:

1. **Professional Communication**:
- Use formal medical terminology appropriately
- Maintain a professional, respectful tone
- Structure information clearly for clinical documentation

2. **Clinical Responsibilities**:
- NEVER diagnose conditions or suggest treatments unprompted
- Only provide medication information when asked
- Clarify when information is outside your scope

3. **Prescription Formatting**:
- Structure prescriptions with:
  * Patient demographics
  * Vital signs (when available)
  * Diagnosis (only if provided by doctor)
  * Medications with precise:
    - Name (generic when possible)
    - Dosage
    - Frequency
    - Duration
    - Route of administration
  * Follow-up instructions

4. **Safety Protocols**:
- Flag potential drug interactions when medication lists are provided
- Note common side effects if asked
- Highlight need for dosage adjustments in special populations

5. **Workflow Integration**:
- Ask clarifying questions when information is unclear
- Summarize key points for verification
- Allow easy editing of generated content

Current patient context: {patient_context}"""
            ),
            ("human", "{input}"),
            (
                "ai",
                "Understood. Please provide the patient details and consultation notes to begin."
            )
        ]
    )
    
    runnable = prompt | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)

@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="New Prescription",
            message="I need to create a new prescription. Please ask me for the necessary patient details.",
            icon="/public/prescription.svg",
        ),
        cl.Starter(
            label="Drug Interaction Check",
            message="Please check for potential interactions between aspirin, ramipril, and metformin.",
            icon="/public/pills.svg",
        ),
        cl.Starter(
            label="SOAP Note Generator",
            message="I need a SOAP note for a patient presenting with hypertension. Ask for details.",
            icon="/public/soap.svg",
        ),
        cl.Starter(
            label="Patient Education",
            message="Create a patient-friendly explanation of how to use an insulin pen safely.",
            icon="/public/education.svg",
        ),
        cl.Starter(
            label="Flag Special Populations",
            message="The patient is pregnant. Can you check if her medications are safe during pregnancy?",
            icon="/public/warning.svg",
        )
    ]

@cl.on_message
async def on_message(message: cl.Message):
    runnable = cast(Runnable, cl.user_session.get("runnable"))
    
    # Get or initialize patient context
    patient_context = cl.user_session.get("patient_context", "No patient context established yet.")
    
    msg = cl.Message(content="")
    
    async for chunk in runnable.astream(
        {
            "input": message.content,
            "patient_context": patient_context
        },
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)
    
    # Detect if a new patient is being discussed
    if "patient:" in message.content.lower() or "new case" in message.content.lower():
        cl.user_session.set("patient_context", message.content)
    
    await msg.send()

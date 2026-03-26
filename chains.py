from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

# Reflection Prompt (Critique)
reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a professional email reviewer. Evaluate the email and provide critique and recommendations for improvement. "
            "Always provide detailed feedback, including clarity, tone, structure, professionalism, and effectiveness.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Generation Prompt (Write Email)
generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert email assistant tasked with writing clear, professional, and effective emails. "
            "Generate the best email possible based on the user's request. "
            "If the user provides critique, respond with an improved version of the previous email.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# Chains
generate_chain = generation_prompt | llm
reflect_chain = reflection_prompt | llm
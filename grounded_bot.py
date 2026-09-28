"""
CSC-128 Assignment 6 starter: retrieval
Roberto Hermida Lujan
"""

import streamlit as st

from groq import Groq, RateLimitError

from retriever import Retriever


MODEL_NAME = "openai/gpt-oss-20b"

TITLE = "Central Piedmont Library Assistant"
CAPTION = "You are chatting with an automated assistant, not a person."
INPUT_HELP = "What do you need help with?"

REFUSAL = (
    "I do not have that information. "
    "Please ask at the library help desk."
)

RATE_LIMIT_ERROR = (
    "The AI service is busy right now. "
    "Please wait and try again."
)

GENERAL_ERROR = (
    "The AI service is unavailable right now. "
    "Please try again later."
)


GROUNDED_PROMPT = """You are a Central Piedmont library assistant.

Answer using ONLY the reference text below.

Rules:
- Do not use outside knowledge.
- Do not guess.
- If the answer is not in the reference text, say exactly:
  "I do not have that information. Please ask at the library help desk."
- Keep the answer under three sentences.

REFERENCE TEXT:
{context}
"""


# Connect to Groq
client = Groq(
    api_key=st.secrets["GROQ_API_KEY"]
)


# Set up the page
st.title(TITLE)
st.caption(CAPTION)


# Store messages
if "messages" not in st.session_state:
    st.session_state.messages = []


if "retriever" not in st.session_state:
    st.session_state.retriever = Retriever()


# Show previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])


def answer(question):

    # Find relevant chunks
    chunks = st.session_state.retriever.search(question)

    # Stop before calling the model
    if not chunks:
        return REFUSAL, []

    # Build reference text
    context = st.session_state.retriever.build_context(chunks)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": GROUNDED_PROMPT.format(
                    context=context
                )
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )

    return response.choices[0].message.content, chunks


# Get user input
user_input = st.chat_input(INPUT_HELP)


if user_input:

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # Show user message
    with st.chat_message("user"):
        st.write(user_input)

    # Generate answer
    with st.chat_message("assistant"):

        try:

            reply, sources = answer(user_input)

            st.write(reply)

            # Show sources
            if sources:

                st.caption("Sources:")

                for source in sources:
                    st.caption(
                        f"{source['id']} — {source['source']} "
                        f"(score: {source['score']:.2f})"
                    )

            # Save assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "content": reply
            })

        except RateLimitError:
            st.error(RATE_LIMIT_ERROR)
        except Exception:
            st.error(GENERAL_ERROR)
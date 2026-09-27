# the Streamlit interface

from retriever import Retriever
from groq import Groq
from groq import RateLimitError
import os
import streamlit as st;

#MODEL_NAME = "llama-3.1-8b-instant"

MODEL_NAME = "openai/gpt-oss-20b"
GREETING_TITLE = "IT Help Desk Bot"
GREETING_CAPTION = "You are chatting with an automated assistant, not a person."
INPUT_HELP = "What do you need help with?"
ERROR_MESSAGE_RATELIMIT = "The AI service is busy right now. Please wait and try again."
ERROR_MESSAGE_EXCEPTION = "The AI service is unavailable right now. Please try again later."
SYSTEM_PROMPT = ""
NO_CONTEXT_FOUND = "Sorry I cannot help you with that question."

# grounded_bot.py

# Connect to Groq
client = Groq(api_key=os.environ["GROQ_API_KEY"])

GROUNDED_PROMPT = """You are the Central Piedmont library assistant.

Answer the student's question using ONLY the reference text below.

Rules:
- If the reference text does not contain the answer, say exactly:
  "I do not have that information. Please ask at the library help desk."
- Do not use any knowledge outside the reference text.
- Do not guess at hours, dates, or numbers.
- Keep the answer under three sentences.

REFERENCE TEXT:
{context}
"""


# Set up the page
st.title(GREETING_TITLE)
st.caption(GREETING_CAPTION)



# Store messages
if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = Retriever();

# Show old messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])



def answer(question):
    chunks = st.session_state.retriever.search(question)

    if not chunks:
        return ("I do not have that information. " "Please ask at the library help desk.")

    #context = "\n\n".join(chunks)
    context = st.session_state.retriever.build_context(chunks)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system",
             "content": GROUNDED_PROMPT.format(context=context)},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content


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


    # Keep the last 10 messages
    recent_messages = st.session_state.messages[-10:]


    # Get AI response
    with st.chat_message("assistant"):
        try:

            reply = answer(user_input);

            # Save AI response
            st.session_state.messages.append({
                "role": "assistant",
                "content": reply
            })


            # Save AI response
            st.session_state.messages.append({
                "role": "assistant",
                "content": reply
            })

        # Handle rate limits
        except RateLimitError:
            st.error(ERROR_MESSAGE_RATELIMIT)
        # Handle other errors
        except Exception:
            st.error(ERROR_MESSAGE_EXCEPTION)
        

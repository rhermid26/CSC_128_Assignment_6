# the Streamlit interface

from retriever import retrieve
from groq import Groq
from groq import RateLimitError
import os
import streamlit as st;



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


def answer(question):
    chunks = retrieve(question)

    if not chunks:
        return ("I do not have that information. "  "Please ask at the library help desk.")

    context = "\n\n".join(chunks)
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system",
             "content": GROUNDED_PROMPT.format(context=context)},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content
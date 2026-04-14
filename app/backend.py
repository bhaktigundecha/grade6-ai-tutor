from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rag.generate import generate_answer

app = FastAPI(title="Grade 6 AI Tutor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"message": "Grade 6 AI Tutor backend is running"}


@app.get("/ask")
def ask(query: str, subject: str = ""):
    return generate_answer(query, subject)
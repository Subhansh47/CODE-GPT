from fastapi import FastAPI, Form, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import json
from typing import Dict

app = FastAPI()
templates = Jinja2Templates(directory="templates")

db_faiss_path = 'vectorstore/db_faiss'

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=['context', 'question']
    )
    return prompt

def load_llm():
    llm = CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens=512,
        temperature=0,
        top_p=95
    )
    return llm

def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain

def qa_bot():
    embeddings = HuggingFaceEmbeddings(
        model_name='microsoft/unixcoder-base',
        model_kwargs={'device': 'cpu'}
    )
    db = FAISS.load_local(db_faiss_path, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa

def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/get_response")
async def get_response(query: str = Form(...)):
    resp = final_result(query)
    result = resp['result']
    source_doc = None
    if resp['source_documents']:
        source_doc = resp['source_documents'][0].get('metadata', {}).get('source', None)
    response_data = jsonable_encoder({"result": result, "source_doc": source_doc})
    return Response(content=json.dumps(response_data), media_type="application/json")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host='0.0.0.0', port=8000, reload=True)

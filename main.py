import os

import faiss
from dotenv import load_dotenv
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

if __name__ == "__main__":
    llm = OpenAI()

    pdf_path = "2210.03629v3.pdf"
    loader = PyPDFLoader(file_path=pdf_path)
    documens = loader.load()
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=30, separator="\n"
    )

    docs = text_splitter.split_documents(documents=documens)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_index_react")

    new_vectorstore = FAISS.load_local(
        "faiss_index_react", embeddings, allow_dangerous_deserialization=True
    )

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

    retrival_chain = create_retrieval_chain(
        retriever=new_vectorstore.as_retriever(),
        combine_docs_chain=combine_docs_chain,
    )

    query = "Give me the gist of ReAct in 3 sentences"

    result = retrival_chain.invoke(input={"input": query})

    print(result["answer"])

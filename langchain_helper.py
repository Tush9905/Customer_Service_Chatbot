import os
import dotenv
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import GooglePalm

dotenv.load_dotenv()
google_api_key = os.environ["GOOGLE_API_KEY"]

llm = GooglePalm(google_api_key=google_api_key, temperature=0.1)

file_path = "faqs.csv"
source_column = "prompt"
instructor_embeddings = HuggingFaceBgeEmbeddings()
vectordb_file_path = "faiss_index"

def get_qa_chain():
        # Load the vector database from the local folder
    vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings)

    # Create a retriever for querying the vector database
    retriever = vectordb.as_retriever(score_threshold=0.7)

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": PROMPT})

    return chain

def create_vectordb():
    loader = CSVLoader(file_path=file_path, source_column=source_column)
    docs = loader.load()
    vectordb = FAISS.from_documents(documents=docs, embedding=instructor_embeddings)
    vectordb.save_local(vectordb_file_path)

#if __name__ == "__main__":
#    create_vectordb()
#    chain = get_qa_chain()






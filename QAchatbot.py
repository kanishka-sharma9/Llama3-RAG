import streamlit as st 
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import LanceDB,FAISS

api=os.getenv("GROQ_API_KEY")

st.title('chat groq demo')
llm=ChatGroq(api_key=api,model_name="Llama3-8b-8192")

prompt=ChatPromptTemplate.from_template(
    """
answer the following question using only the provided context.
Please provide the most appropriate response.
<context>
{context}
</context>

Question : {input}
""")




def vector_embeddings():
    if "vector" not in st.session_state:
        # st.session_state.embeddings=OllamaEmbeddings()
        st.session_state.loader=PyPDFDirectoryLoader(path="us_census")
        st.session_state.docs=st.session_state.loader.load()
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.docs=st.session_state.text_splitter.split_documents(documents=st.session_state.docs[:10])
        # st.session_state.vector_store=LanceDB.from_documents(documents=st.session_state.docs,embedding=OllamaEmbeddings())
        st.session_state.vector_store=FAISS.from_documents(documents=st.session_state.docs,embedding=OllamaEmbeddings())




prompt1=st.text_input("enter question from docs")

if st.button('docs embedding'):
    vector_embeddings()
    st.write("document has been processed.")

import time
# document_chain=create_stuff_documents_chain(llm,prompt)
# retriever=st.session_state.vector_store.as_retriever()
# ret_chain=create_retrieval_chain(retriever,document_chain)



if prompt1:
    #### uncomment to fix initial error
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vector_store.as_retriever()
    ret_chain=create_retrieval_chain(retriever,document_chain)


    start=time.process_time()
    response=ret_chain.invoke({'input':prompt1})
    print("response time : ",time.process_time()-start)
    st.write(response["answer"])
    

    with st.expander("doc similarity search"):
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("---------------")

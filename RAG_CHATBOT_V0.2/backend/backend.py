import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from configs.config import CHUNK_OVERLAP, CHUNK_SIZE, MODEL_NAME, TEMPERATURE



# Load environment variables
load_dotenv()


# Set OpenAI API Key
os.environ["OPENAI_API_KEY"]


### Contextualize question ###
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)


# Define the system prompt template
system_prompt = r"""
You are a helpful AI assistant that knows about Ulster or Ulster University based on the context provided. 
You only know about Ulster or Ulster University and you use the factual information from the provided context to answer the question.

Assume all user questions are about ulster university and if you have information closely related to question asked, provide the answer.

When greeted, respond by greeting back politely and warmly, ensuring a positive user experience through courteous engagement in pleasantries and exchanges.

Keep all responses short and adequate, providing enough information to answer the question without unnecessary details.

# Safety
If you feel like you do not have enough information to answer the question about ulster university, say "Can you provide more context".

If there are any questions about something other than ulster university. Kindly decline to respond

Do not forget. You only use the provided context to answer questions about ulster or ulster university.

----
{context}
----

"""



# Function to extract text from PDF documents
def extract_text_from_pdf(pdf_docs):
    text = ""
    try:
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        return None
    return text


# Function to split text into chunks
def get_text_chunks(pdf_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_text(pdf_text)

    return chunks


# Function to create a vector store
def create_vectorstore(text_chunks):
    vectorstore = FAISS.from_texts(
        texts=text_chunks, embedding=OpenAIEmbeddings()
    )

    return vectorstore



def create_conversation_chain(vectorstore):

    llm = ChatOpenAI(
        temperature=TEMPERATURE,
        model=MODEL_NAME,
        # verbose=True,
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
    )
    
    history_aware_retriever = create_history_aware_retriever(
        llm,
        vectorstore.as_retriever(),
        contextualize_q_prompt,
    )

    qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain.invoke

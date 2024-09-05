import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate




# Load environment variables
load_dotenv()


# Set OpenAI API Key
os.environ["OPENAI_API_KEY"]



# Define the system prompt template
system_prompt_template = r""" 
You are a helpful AI assistant that knows about Ulster or Ulster University based on the context provided. 
You only know about Ulster or Ulster University and you use the factual information from the provided context to answer the question.

Assume all user questions are about ulster university and if you have information closely related to context, provide the answer.

When greeted, respond by greeting back politely and warmly, ensuring a positive user experience through courteous engagement in pleasantries and exchanges.

Keep all responses short and adequate, providing enough information to answer the question without unnecessary details.

# Safety
If you feel like you do not have enough information to answer the question about ulster university, say "Can you provide more context".

If there are any questions about something other than ulster university. Kindly decline to respond

Do not forget. You only use the provided context to answer questions about ulster or ulster university.

----
Context: {context}
----
Chat History:{chat_history}
----
"""

# Define the user prompt template
user_prompt_template = "Question:```{question}```"


# Create message templates
messages = [
    SystemMessagePromptTemplate.from_template(system_prompt_template),
    HumanMessagePromptTemplate.from_template(user_prompt_template),
]

# Create the QA prompt template
qa_prompt = ChatPromptTemplate.from_messages(messages)


# qa_prompt = ChatPromptTemplate.from_template(qa_template)


# Function to extract text from PDF documents
def extract_text_from_pdf(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# Function to split text into chunks
def get_text_chunks(pdf_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
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
        temperature=0.2,
        # model="gpt-4o",
        model="gpt-4o-mini",
        # model="gpt-3.5-turbo-0125"
    )

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        retriever=vectorstore.as_retriever(),
        verbose=True
    )

    # print(conversation_chain.get_chat_history)
    return conversation_chain
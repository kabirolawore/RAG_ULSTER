import streamlit as st
from streamlit_chat import message
from langchain_core.messages import AIMessage, HumanMessage
# from backend_test import get_text_chunks, create_vectorstore, create_conversation_chain, extract_text_from_pdf
from backend.backend import get_text_chunks, create_conversation_chain, create_vectorstore, extract_text_from_pdf




def handle_user_question(user_question):
    response = st.session_state.conversation({"input": user_question, "chat_history": []})


    st.session_state.chat_history.extend(
        [
            HumanMessage(content=response['input']),
            AIMessage(content=response['answer']),
        ]
    )



    print('st.session_state.chat_history', st.session_state.chat_history)

    # Display the chat messages
    for i, msg in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            message(msg.content, is_user=True, key=str(i) + '_user', avatar_style="initials", seed="Kavita")
        else:
            message(msg.content, is_user=False, key=str(i), avatar_style="initials", seed="AI",)




def main():

    st.set_page_config(page_title='ChatBot', page_icon=':books:')

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header('Ask questions about Ulster University :books:')

    user_question = st.chat_input(placeholder='Ask me about ulster university')
    if user_question and user_question != "":
        with st.spinner('Thinking...'):
            handle_user_question(user_question)

    if st.session_state.chat_history:
        clear_history = st.button('Clear Chat History') # clear chat history button
        if clear_history:
            st.session_state.chat_history = []
            # st.session_state.conversation = None
            st.success('Chat history cleared')

    with st.sidebar:
        st.subheader('Your documents')
        pdf_docs = st.file_uploader('Upload a PDF and click process', accept_multiple_files=True)
        if st.button('Process'):
            with st.spinner('Processing document... This may take a moment.'):
            
                # get pdf text
                pdf_text = extract_text_from_pdf(pdf_docs) 

                # get the text chunks
                text_chunks = get_text_chunks(pdf_text)
                # st.write(text_chunks)

                # create vector store
                vectorstore = create_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = create_conversation_chain(vectorstore)


            st.success('Processed successfully')



if __name__ == '__main__':
    main()
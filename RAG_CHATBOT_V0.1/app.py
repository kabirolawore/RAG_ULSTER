import streamlit as st
from streamlit_chat import message
from backend import get_text_chunks, create_vectorstore, create_conversation_chain, extract_text_from_pdf



def handle_user_question(user_question):
    response = st.session_state.conversation({"question": user_question})
    # st.write(response)

    # print('chat history', response['chat_history'])
    st.session_state.chat_history = response['chat_history']

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
        st.session_state.chat_history = None

    st.header('Ask questions from Knowledge Base :books:')

    user_question = st.chat_input(placeholder='Ask me')
    if user_question:
        with st.spinner('Thinking'):
            handle_user_question(user_question)


    with st.sidebar:
        st.subheader('Your documents')
        pdf_docs = st.file_uploader('Upload a PDF and click process', accept_multiple_files=True)
        if st.button('Process'):
            with st.spinner('Processing'):
            
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
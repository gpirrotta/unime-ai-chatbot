from dotenv import load_dotenv
import pandas as pd
from langchain.embeddings import OpenAIEmbeddings 
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    PromptTemplate,
)
import streamlit as st

load_dotenv()


def get_conversation_chain(vector_store:FAISS, system_message:str, human_message:str) -> ConversationalRetrievalChain:
    
     # questo è il cuore del sistema. Qui viene invocato il modello di linguaggio GPT
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    # Qui è dove LangChain svolge il ruolo di direttore d'orchestra, collegando tutte le strumenti del sistema
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, # questo è il modello GPT
        retriever=st.session_state.vector_store.as_retriever(), # quà viene definito il database vettoriale
        return_source_documents=True,
        verbose=True,
        memory=st.session_state.memory,
        get_chat_history=lambda h : h,
        condense_question_prompt=PromptTemplate.from_template(
            ('Questa è la precedente domanda {question}), questa è la precedente risposta ({chat_history}).')
        ),
        combine_docs_chain_kwargs={ # qui viene specificato il prompt come unione di più parti tenute insieme tramite un template
            "prompt": ChatPromptTemplate.from_messages(
                [
                    system_message,
                    human_message,
                ]
                ),
            },
        )
    
    return conversation_chain



def setup_chat_header():
    st.set_page_config(
        page_title="Documentation Chatbot",
        page_icon=":school:",
    )

    st.image("https://archivio.unime.it/sites/default/files/7f7ebabb21b60312954b1d1ecc3e1eeddab2926e/409x205xloco,P20unime,P20orizzontale.jpg.pagespeed.ic.nIu0lfMiM1.webp") 
    st.title("UNIME Chatbot")
    st.subheader("Quando l'AI entra in Ateneo")
    st.markdown(
        """
        Questo chatbot è stato creato per rispondere alle domande riguardanti l'immatricolazione
        e l'iscrizione all'Università di Messina.
        Poni una domanda e il chatbot ti risponderà nel migliore dei modi.
        """
    )

def main():
    if 'vector_store' not in st.session_state:
        vector_store = FAISS.load_local("./db", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
        st.session_state.vector_store = vector_store
    if "memory" not in st.session_state:
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')
        st.session_state.memory = memory
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
        
        
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None  

    # ISTRUZIONI PROMPT
    system_message_prompt = SystemMessagePromptTemplate.from_template(
        """
        Sei un chatbot e il tuo compito è rispondere a domande sull'immatricolazione e iscrizione
        dei corsi di laurea all'Università di Messina.
        Ad ogni domanda devi sempre rispondere con la risposta più rilevante e più accurata.
        Non rispondere mai con una nuova domanda.
        Non rispondere a domande non inerenti l'Università di Messina. 
        Se la domanda è in italiano rispondi in italiano. Se la domanda è in inglese rispondi in inglese.
        Data una domanda dovresti rispondere con le informazioni più rilevanti in base al seguente contesto: \n
        {context}
        """
    )
    
    human_message_prompt = HumanMessagePromptTemplate.from_template("{question}")
       
    
    # SETUP STREAMLIT
    setup_chat_header()     
    user_question = st.text_input("Cosa vuoi chiedere?")
    with st.spinner("Sto elaborando la risposta..."):
        if user_question:
            response = st.session_state.conversation({"question": user_question })
            st.session_state.chat_history = response["chat_history"]

            human_style = "background-color: #e6f7ff; border-radius: 10px; padding: 10px;"
            chatbot_style = "background-color: #f9f9f9; border-radius: 10px; padding: 10px;"
            
            for i, message in enumerate(st.session_state.chat_history):
                if i % 2 == 0:
                    st.markdown(
                        f"<p style='text-align: right;'><b>Utente</b></p><p style='text-align: right;{human_style}'> <i>{message.content}</i> </p>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"<p style='text-align: left;'><b>Chatbot</b></p> <p style='text-align: left;{chatbot_style}'> <i>{message.content}</i> </p>",
                        unsafe_allow_html=True,
                    )
            st.session_state.input = ""  
    
    st.session_state.conversation = get_conversation_chain(st.session_state.vector_store, system_message_prompt, human_message_prompt)
    
    
if __name__ == "__main__":
    main()
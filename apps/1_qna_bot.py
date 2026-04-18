from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st

llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")
st.title("AskBuddy - AI QnA Bot")
st.markdown("My QnA Bot with LangChain and Google Gemini!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)

query = st.chat_input("Ask a Question")
if query:
    st.session_state.messages.append({"role":"user", "content":query})
    st.chat_message("user").markdown(query)
    res = llm.invoke(query)
    st.chat_message("ai").markdown(res.content)
    st.session_state.messages.append({"role":"ai", "content":res.content})

# command to run this app
# py -m streamlit run 1_qna_bot.py   


# while True:
#     query = input("User: ")

#     if query.lower() in ["quit", "exit", "bye"]:
#         print("GoodBye!")
#         break
    
#     result = llm.invoke(query) 
#     print("AI: ", result.content, "\n")
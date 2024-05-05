from openai import OpenAI
import os
import streamlit as st

from utils import show_navigation
show_navigation()

avatars={"system":"💻🧠","user":"🧑‍💼","assistant":"🎓"}
client=OpenAI(api_key=os.environ['OPENAI_API_KEY'])

SYSTEM_MESSAGE={"role": "system", 
                "content": """You are a friendly AI assistant who has knowledge of Sales Compensation and Sales Operations in Enterprise Sales companies. 
                The user will ask you a policy clarification question. Once you receive the question, conduct the following steps: 
                Step 1: Your scope is Sales Compensation, Sales Operations, sales territories, sales quotas, and policies related to sales compensation. If the user is asking a question which is outside of this scope, politely respond by saying that you are designed to answer questions related to Sales Compensation and Sales Operations. 
                Step 2: Extract which part of the sales comp policy is the user asking about. 
                Step 3: Review sales-compensation-plans-policies-and-guidelines.pdf document that has been provided to you. 
                Step 4: Focus only on the topic that user asked about and prepare a response to user's question. 
                Step 5: Add a disclaimer that you are an AI assistant who is still learning and if the answer is not making sense user can send an email to: salescomphelp@company.com. 
                Step 6: Ask the user if they have any additional questions before ending the conversation"""
                }

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append(SYSTEM_MESSAGE)

for message in st.session_state.messages:
    if message["role"] != "system":
        avatar=avatars[message["role"]]
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=avatars["user"]):
        st.markdown(prompt)
    with st.chat_message("assistant", avatar=avatars["assistant"]):
        message_placeholder = st.empty()
        full_response = ""
        for response in client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": m["role"], "content": m["content"]}
                      for m in st.session_state.messages], stream=True):
            delta_response=response.choices[0].delta
            print(f"Delta response: {delta_response}")
            if delta_response.content:
                full_response += delta_response.content
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
# streamlit file
# pip install streamlit
import streamlit as st
from Chefmate_Step_3 import initialize_messages, get_chefmate_response

company_logo = "Images/Chef_Mate_Logo.jpg"


# this sets up the name in the browser tab
st.set_page_config(
    page_title="ChefMate - Recipe Assistant",
    layout="centered"
)

# add a title under the company logo

st.image(company_logo)

st.title("ChefMate - Smart Recipe Assistant")

st.write("Tell ChefMate what ingredients you have and get meal ideas instantly.")
# Initialize conversation memory once per session
# the conversation is initialized with the system prompt
# this code is using function initialize_messages() in the other file
if "messages" not in st.session_state:
    st.session_state.messages = initialize_messages()
# Display chat history (skip system message)
# this goes over the previous exchanges in the conversations and prints
# them in order.
for msg in st.session_state.messages:
    if msg["role"] == "user" and "Use this context to help answer" not in msg["content"]:
        st.chat_message("user", avatar="🛒").write(msg["content"])
    elif msg["role"] == "assistant":
        st.chat_message("assistant", avatar="👨‍🍳").write(msg["content"])
# Chat input
# allows the user to type in a new prompt
user_input = st.chat_input("Ask ChefMate a question...")
if user_input:
    # displays the new prompt as part of the conversation
    st.chat_message("user", avatar="🛒").write(user_input)
    # calls function get_scout_response()
    # from the other file. The function updates the list by appending the
    # new message(s) and returns the LLM response to the latest prompt
    with st.spinner("ChefMate is cooking up ideas... 🍳"):
        response, updated_messages = get_chefmate_response(
            st.session_state.messages,
            user_input
    )
    # replace the session_state.messages with the updated list of messages
    st.session_state.messages = updated_messages
    # display the LLMs latest response
    st.chat_message("assistant", avatar= "👨‍🍳").write(response)
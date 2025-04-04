# app_ui.py
import streamlit as st
import requests

st.set_page_config(page_title="PDF Q&A with LLaMA 2", layout="centered")
st.title("Ask Questions from your PDF using LLaMA 2")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
question = st.text_input("Enter your question")

if st.button("Get Answer"):
    if not uploaded_file:
        st.warning("Please upload a PDF file.")
    elif not question:
        st.warning("Please enter your question.")
    else:
        # Show spinner while waiting for backend response
        with st.spinner("Processing... Please wait."):
            try:
                # Send request to FastAPI backend
                files = {"file": uploaded_file.getvalue()}
                data = {"question": question}

                response = requests.post("http://127.0.0.1:8000/ask", files={"file": uploaded_file}, data=data)
                
                result = response.json()

                if response.status_code == 200:
                    st.success("Answer received!")
                    st.markdown(f"**Answer:** {result['answer']}")
                    st.markdown("**Context from PDF:**")
                    st.code(result['context'])
                else:
                    st.error(f"Error: {result.get('error', 'Unknown error')}")
            except Exception as e:
                st.error(f"Failed to connect to FastAPI backend: {str(e)}")

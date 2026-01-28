import streamlit as st
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import configparser

def load_config():
    config = configparser.ConfigParser()
    config.read('config/properties.cfg')
    return config

# Load configuration
config = load_config()

# Azure API Configuration (Set your Azure credentials)
os.environ["OPENAI_API_VERSION"] = config['LLM']['version']
os.environ["AZURE_OPENAI_ENDPOINT"] = config['LLM']['endpoint']
os.environ["AZURE_OPENAI_API_KEY"] = config['LLM']['api_key'] 

model = config['LLM']['model']
# Initialize the AzureChatOpenAI model from langchain
llm = AzureChatOpenAI(deployment_name=model, temperature=0)

parser = StrOutputParser()

not_expainedable_msg = "Please provide valid code for explanation."
# Define a function to generate an explanation for the code
def generate_code_explanation(code: str):
    # Define the prompt template for code explanation
    prompt = """
    You are an expert in all the programming languages. Your role is to explain the following code in a way that is easy to understand, especially for someone who is learning to code.

    Please break your explanation into clear sections, focusing on clarity and educational value.

    1. **Code Overview**:
       Provide a high-level summary of what the code does and what its overall purpose is.

    2. **Line-by-Line Breakdown**:
       Explain what each line of the code does. Break it down in simple terms for someone who may be new to coding.

    3. **Key Concepts**:
       Identify and explain any key programming concepts used in the code, such as loops, conditionals, functions, etc.

    4. **Algorithm Explanation**:
       If the code implements a specific algorithm, explain the steps involved in that algorithm.

    5. **Algorithm Flowchart**:
        Please provide the clean and complete flow chart of the algorithm with pipe symbols.

    6. **Data Structures Used**:
       Identify any data structures used in the code (e.g., lists, dictionaries, arrays) and explain how they are utilized.

    7. **Potential Edge Cases**:
       Mention any edge cases that this code may account for or overlook (e.g., handling empty inputs, large numbers, etc.).

    8. **Conclusion**:
       Summarize the key takeaways from the code and its functionality.

    Now, here is the code you need to explain:
    {code}

    If the provided input is not programming code, respond with: Please provide valid code for explanation.
    """
    
    # Run the model to get the explanation
    template = PromptTemplate.from_template(prompt)

    chain = template | llm | parser
    explanation = chain.invoke({"code": code})
    
    return explanation

# Streamlit UI
def main():
    st.title("CodeClarify - AI-Powered Code Explanation and Learning Assistant")
    
    # Initialize session state variables
    if 'code_input' not in st.session_state:
        st.session_state.code_input = ""
    if 'explanation' not in st.session_state:
        st.session_state.explanation = ""
    
    # Radio button for input type
    input_type = st.radio(
        "Select the input method:",
        ('Upload a file', 'Free Text')
    )
    
    # Input for uploading file
    if input_type == 'Upload a file':
        uploaded_file = st.file_uploader("Choose a code file")
        if uploaded_file is not None:
            # Read and display the uploaded file's contents
            st.session_state.code_input = uploaded_file.getvalue().decode("utf-8")
            #st.text_area("Uploaded Code", code_input, height=300)
            # Generate explanation
            if st.button("Explain Code"):
                with st.spinner("Processing the code..."):
                    st.session_state.explanation = generate_code_explanation(st.session_state.code_input)
        else:
            # Clear code and explanation when no file is uploaded
            st.session_state.code_input = ""
            st.session_state.explanation = ""
                
        # Display explanation if it exists
        if st.session_state.explanation:
            st.subheader("Explanation")
            st.write(st.session_state.explanation)
    
    # Input for free-text code
    elif input_type == 'Free Text':
        code_input = st.text_area("Enter your code here", height=300, value=st.session_state.code_input)
        st.session_state.code_input = code_input
        if st.button("Explain Code"):
            if st.session_state.code_input.strip():
                with st.spinner("Processing the code..."):
                    st.session_state.explanation = generate_code_explanation(st.session_state.code_input)
            else:
                st.error("Please enter some code to explain.")
                
        # Display explanation if it exists
        if st.session_state.explanation:
            st.subheader("Explanation")
            st.write(st.session_state.explanation)
    
    # Q&A feature - only show if there's code to ask about
    if st.session_state.code_input.strip() and st.session_state.explanation != not_expainedable_msg and st.session_state.explanation:
        st.subheader("Ask specific questions about the code:")
        question = st.text_input("Ask a question about the code")
        if question:
            # Generate Q&A response from AI
            prompt = """
            You are an expert programming mentor. Answer the user's question based on the provided code.

            Here is the code:
            {code}

            Here is the user's question:
            {question}
            """
            template = PromptTemplate.from_template(prompt)
            chain = template | llm | parser
            response = chain.invoke({"code": st.session_state.code_input, "question": question})
            st.write(response)

# Run the app
if __name__ == "__main__":
    main()

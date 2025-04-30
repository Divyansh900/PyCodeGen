import streamlit as st
from Model import PyCode

# App title
st.title(":green[Py]CodeGen")

# Initialize the model only once using st.session_state
if 'model' not in st.session_state:
    with st.spinner('Initializing model... Please wait.'):
        st.session_state.model = PyCode()
    st.success('Model initialized successfully!')

# Input box for user prompts
user_prompt = st.text_area("Enter your prompt",
                           placeholder="Example: Write a function to calculate factorial",
                           height=100)

# Button to generate code
if st.button('Generate Code'):
    if user_prompt:
        with st.spinner('Generating code...'):
            generated_code = st.session_state.model.generate(user_prompt)

        # Display the generated code
        st.subheader("Generated Code:")
        st.code(generated_code)
    else:
        st.warning('Please enter a prompt first.')

# Add some helpful instructions
st.sidebar.header("Instructions")
st.sidebar.markdown("""
1. Enter your prompt describing the code you want to generate
2. Click 'Generate Code' button
3. View the generated code in the main panel
""")

# Add information about the model
st.sidebar.header("About")
st.sidebar.markdown("""
This app uses a PyCode model to generate code based on your prompts.
The model is initialized once when the app starts and will continue running until the app is terminated.
""")
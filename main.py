import streamlit as st              # to build UI and other functionalities
import os                           # to access the secret variables from .env file
from dotenv import load_dotenv      # used to load/create/update the env file

# LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI   # to access the gemini chatmodel
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace   # Endpoint -> to access the model by its endpoint through unique_id 
                                                                         # ChatHuggingFace -> if the model you are accessing comes out to be a regular model, then we wrap that as ChatModel by using this
                                                                         
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate  # to create System and Human/User Messages and wrap them in a wrapper called ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough   # Passes the user input forward as-is so it can be mapped into the 'question' field
from langchain_core.output_parsers import StrOutputParser  # Convert model output into clean text

# Loads environment variables
load_dotenv()

# ---------- 1. Custom Memory ---------
class CustomConversationMemory:
    def __init__(self):
        if 'conversation_memory' not in st.session_state:
            st.session_state.conversation_memory = []
            
    def load_memory_variables(self, _=None):
        # We join the history list into a single string to pass to the LLM
        history_text = '\n'.join(st.session_state.conversation_memory)
        return {'history': history_text}
    
    def save_context(self, user_input, ai_output):
        st.session_state.conversation_memory.append(f"User: {user_input}")
        st.session_state.conversation_memory.append(f"AI: {ai_output}")

# ---------- 2. LLM Selection ---------
def get_llm_for_module(module):
    # GEMINI: for Python, ML, DL
    if module in ["Python", "Machine Learning", "Deep Learning"]:
        return ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0.3)
    
    # HUGGING FACE: Specialized models
    repo_ids = {
        "SQL": 'deepseek-ai/DeepSeek-R1',
        "Exploratory Data Analysis (EDA)": 'deepseek-ai/DeepSeek-R1',
        "Power BI": "meta-llama/Meta-Llama-3-8B-Instruct", 
        "Generative AI": "meta-llama/Meta-Llama-3-8B-Instruct",
        "Agentic AI": "Qwen/Qwen2.5-Coder-7B-Instruct"
    }
    
    repo_id = repo_ids.get(module)
    
    if repo_id:
        try: 
            llm_raw = HuggingFaceEndpoint(
                repo_id=repo_id, 
                task = 'text_generation', # Explicit task helps prevent 410 errors
                max_new_tokens = 512,
                temperature=0.3, 
                huggingfacehub_api_token=os.getenv('HF_TOKEN')
            )
            return ChatHuggingFace(llm=llm_raw)
        except Exception as e:
            st.error(f"Error loading model '{repo_id}': {e}")
            return None
        
    return None

# ---------- 3. Prompt Engineering ---------
def build_prompt(module):
    # We rely on this instruction to keep the bot on topic.
    system_template = f"""
    You are an expert AI Mentor specialized in {module}.
    
    Your Rules:
    1. Answer ONLY questions related to {module}.
    2. If a user asks about {module} concepts (like RAG for GenAI, or Pandas for Python), explain them clearly.
    3. If the question is completely unrelated (like "How to cook pasta" or "What is the capital of France"), apologize and refuse.
    4. Keep answers concise and educational.
    """
    
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        SystemMessagePromptTemplate.from_template("Conversation History:\n{history}"),
        HumanMessagePromptTemplate.from_template("{question}")
    ])

# ---------- 4. Streamlit UI Setup ---------
st.set_page_config(page_title='AI Chatbot Mentor', page_icon="🤖", layout='centered')

if 'module' not in st.session_state:
    st.session_state.module = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [] 
if 'memory' not in st.session_state:
    st.session_state.memory = CustomConversationMemory()

# --- Page 1: Module Selection ---
if not st.session_state.module:
    st.title("🤖 AI Chatbot Mentor")
    st.write("Select a specialization to begin your learning journey.")
    
    modules = ["Python", "SQL", "Power BI", "Exploratory Data Analysis (EDA)", 
               "Machine Learning", "Deep Learning", "Generative AI", "Agentic AI"]
    
    selected_module = st.selectbox('Choose a Module', modules)
    
    if st.button('Start Mentoring Session'):
        st.session_state.module = selected_module
        st.session_state.chat_history = [] # Reset chat for new module
        st.session_state.conversation_memory = [] # Reset memory
        st.rerun()

# --- Page 2: Chat Interface ---
else:
    module = st.session_state.module
    st.title(f" 🧑‍🏫 {module} Mentor")
    
    # 1. Display Chat History using Streamlit Native Chat Elements
    # This automatically adds the 'User' and 'AI' icons
    for message in st.session_state.chat_history:
        role = message["role"]
        content = message["content"]
        with st.chat_message(role):
            st.markdown(content)

    # 2. Chat Input Area
    # st.chat_input is better than st.text_input for chatbots
    if user_input := st.chat_input("Ask your question here..."):
        
        # Displays User Message immediately
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Saves to UI history
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Process the response
        try:
            llm = get_llm_for_module(module)
            prompt = build_prompt(module)
            
            # Create the chain
            # We use StrOutputParser to get clean text out
            chain = prompt | llm | StrOutputParser()
            
            # Load current history for the context
            current_history = st.session_state.memory.load_memory_variables()['history']
            
            with st.spinner("Thinking..."):
                response = chain.invoke({
                    "history": current_history,
                    "question": user_input
                })
            
            # Displays AI Message
            with st.chat_message("assistant"):
                st.markdown(response)
            
            # Save to UI history & Internal Memory
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.session_state.memory.save_context(user_input, response)
            
        except Exception as e:
            st.error(f"An error occurred: {e}")

    # 3. Sidebar Options
    with st.sidebar:
        st.header("Session Settings")
        
        # Download Button
        if st.session_state.chat_history:
            chat_str = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.chat_history])
            st.download_button(
                label="📥 Download Chat",
                data=chat_str,
                file_name=f"{module}_Session.txt",
                mime="text/plain"
            )
            
        # End Session Button
        if st.button("🔁 End Session"):
            st.session_state.module = None
            st.session_state.chat_history = []
            st.session_state.conversation_memory = []
            st.rerun()


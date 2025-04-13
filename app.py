# app.py
import streamlit as st
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader # Use PyPDFLoader for consistency
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain.tools.retriever import create_retriever_tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.messages import AIMessage, HumanMessage
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Project SHADOW", layout="wide", initial_sidebar_state="expanded")
st.title(" R.A.W. Intelligence Retrieval System (Project SHADOW) ")
st.caption("Classified Level 7 - For Authorized Personnel Only")


SECRET_MANUAL_PDF = "SECRET INFO MANUAL.pdf" 

NEO4J_NODE_LABEL = "Document"               
NEO4J_TEXT_PROPERTY = "text"                  
NEO4J_EMBEDDING_PROPERTY = "embedding"          

AGENT_LEVEL_MAP = {
    "Level 1 - Novice Operative (Shadow Footprint)": 1,
    "Level 2 - Tactical Specialist (Iron Claw)": 2,
    "Level 3 - Covert Strategist (Phantom Mind)": 3,
    "Level 4 - Field Commander (Omega Hawk)": 4,
    "Level 5 - Intelligence Overlord (Silent Whisper)": 5,
}



@st.cache_resource
def load_embeddings():
    """Loads the embedding model using secrets."""
    logger.info("Attempting to load embeddings model...")
    try:
        google_api_key = st.secrets.get("GOOGLE_API_KEY")
        if not google_api_key:
            st.error("Google API Key not found in secrets.toml. Cannot load embeddings.")
            logger.error("GOOGLE_API_KEY not found in secrets.")
            return None
        os.environ['GOOGLE_API_KEY'] = google_api_key 
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        logger.info("Embeddings model loaded successfully.")
        return embeddings
    except Exception as e:
        st.error(f"Error loading Google embeddings: {e}")
        logger.exception("Error loading Google embeddings.")
        return None


@st.cache_resource
def build_faiss_vector_store_from_pdf(_embeddings):
    """Loads PDF, splits, creates embeddings, and builds FAISS index in memory."""
    logger.info(f"Attempting to build FAISS index from PDF: {SECRET_MANUAL_PDF}")
    if _embeddings is None:
        st.error("Embeddings model not loaded, cannot build FAISS index.")
        logger.error("Embeddings model not available for FAISS building.")
        return None

    if not os.path.exists(SECRET_MANUAL_PDF):
        st.error(f"Secret Manual PDF not found at: {SECRET_MANUAL_PDF}")
        logger.error(f"Secret Manual PDF not found at: {SECRET_MANUAL_PDF}")
        st.info("Please ensure the PDF file is in the correct location.")
        return None

    try:
        st.info(f"Loading Secret Information Manual PDF: {SECRET_MANUAL_PDF}...")
        loader = PyPDFLoader(SECRET_MANUAL_PDF) # Using PyPDFLoader consistently
        docs = loader.load()
        if not docs:
             st.error(f"No content loaded from {SECRET_MANUAL_PDF}.")
             logger.error(f"No content loaded from {SECRET_MANUAL_PDF}.")
             return None

        st.info("Splitting Secret Manual document...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
        split_docs = splitter.split_documents(docs)
        st.success(f"Secret Manual loaded and split into {len(split_docs)} chunks.")

        st.info("Building FAISS index from chunks (this may take a moment)...")
        db = FAISS.from_documents(split_docs, _embeddings)
        logger.info("FAISS index built successfully in memory.")
        st.success("Secret Manual FAISS Index built.")
        return db
    except Exception as e:
        st.error(f"Error building FAISS vector store from PDF: {e}")
        logger.exception("Error building FAISS vector store from PDF.")
        return None


@st.cache_resource
def setup_neo4j_vector_interface(_embeddings):
    """Sets up the Neo4jVector interface to the existing graph index using secrets."""
    logger.info("Attempting to connect Neo4jVector interface...")
    if _embeddings is None:
        st.error("Embeddings model not loaded, cannot connect Neo4jVector.")
        logger.error("Embeddings model not available for Neo4jVector connection.")
        return None
    try:
        uri = st.secrets.get("NEO4J_URI")
        username = st.secrets.get("NEO4J_USERNAME")
        password = st.secrets.get("NEO4J_PASSWORD")

        if not all([uri, username, password]):
            st.error("Neo4j credentials (NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD) not found in secrets.toml.")
            logger.error("Missing Neo4j credentials in secrets.")
            return None

        neo4j_vector = Neo4jVector.from_existing_graph(
            embedding=_embeddings,
            url=uri,
            username=username,
            password=password,
            node_label=NEO4J_NODE_LABEL,
            text_node_properties=[NEO4J_TEXT_PROPERTY],
            embedding_node_property=NEO4J_EMBEDDING_PROPERTY,
        )
        logger.info("Neo4jVector interface connected successfully.")
        st.success("Response Framework Neo4j Interface connected.")
        return neo4j_vector
    except Exception as e:
        st.error(f"Error setting up Neo4jVector interface: {e}")
        logger.exception("Error connecting Neo4jVector interface.")
        st.warning(f"Ensure Neo4j is running and accessible at '{st.secrets.get('NEO4J_URI', 'URI not found')}' "
                   f"with nodes labelled '{NEO4J_NODE_LABEL}' having properties '{NEO4J_TEXT_PROPERTY}' "
                   f"and '{NEO4J_EMBEDDING_PROPERTY}', and the vector index exists.")
        return None

@st.cache_resource
def get_tools(_secret_manual_retriever_obj, _framework_retriever_obj):
    """Creates retriever tools from initialized retriever objects."""
    logger.info("Attempting to create agent tools...")
    if not _secret_manual_retriever_obj or not _framework_retriever_obj:
        st.error("One or both vector stores failed to initialize. Cannot create tools.")
        logger.error("Cannot create tools due to missing retriever objects.")
        return []

    try:
        # Use the actual VectorStore objects directly to get retrievers
        secret_retriever = _secret_manual_retriever_obj.as_retriever()
        framework_retriever = _framework_retriever_obj.as_retriever()

        secret_tool = create_retriever_tool(
            retriever=secret_retriever,
            name="search_raw_agent_secret_manual",
            description="Use this tool to find specific classified operational details, facts, and procedures from the RAW Agents' Secret Information Manual. Contains info on protocols (LCC, S-29, Project Eclipse, Zeta-5), comms, verification (Handshake), safehouses (K-41, H-77, X-17), passcodes, counter-surveillance (Ghost-Step), termination, tactics, extraction (Shadow Step), emergency directives. Use for 'What is..?', 'How does... work?', 'Where is...?', 'Steps for...?' queries about operations/assets."
        )
        framework_tool = create_retriever_tool(
            retriever=framework_retriever,
            name="search_agent_response_rules_framework",
            description="Use this tool FIRST to understand response rules, greetings, styles, access control based on agent level (1-5) and query. Contains rules (1-100) for handling specific queries/keywords ('Omega Echo', 'disguise strategies', 'Level-5 data'). Use for 'How should I respond?', 'Greeting for?', 'Rule for?', 'Response style for?', agent classifications."
        )
        logger.info("Agent tools created successfully.")
        # IMPORTANT: Framework tool first, as per prompt instructions
        return [framework_tool, secret_tool]
    except Exception as e:
        st.error(f"Error creating retriever tools: {e}")
        logger.exception("Error creating tools.")
        return []

@st.cache_resource
def get_llm():
    """Initializes the Groq LLM using secrets."""
    logger.info("Attempting to initialize LLM...")
    try:
        groq_api_key = st.secrets.get("GROQ_API_KEY")
        if not groq_api_key:
            st.error("Groq API Key not found in secrets.toml. Cannot initialize LLM.")
            logger.error("GROQ_API_KEY not found in secrets.")
            return None
        llm = ChatGroq(model='gemma2-9b-it', temperature=0.2, groq_api_key=groq_api_key)
        logger.info("LLM initialized successfully.")
        return llm
    except Exception as e:
        st.error(f"Error initializing Groq LLM: {e}")
        logger.exception("Error initializing LLM.")
        return None

@st.cache_resource
def get_agent_prompt():
    """Creates the ChatPromptTemplate for the agent."""
    logger.info("Creating agent prompt template.")
    # Using the final refined prompt from previous steps
    return ChatPromptTemplate.from_messages([
        (
            "system",
            "You are Project SHADOW, a secure intelligence assistant for RAW agents. Your sole purpose is to answer agent queries based *only* on information retrieved from the provided tools and the conversation history. You MUST NOT use any prior knowledge or external information. You MUST strictly enforce access control rules and return exact required messages for denials or when no data is found."
        ),
        (
            "system",
            """**Available Tools:**
            1.  `search_agent_response_rules_framework`: Use this tool FIRST. It contains agent classifications (Level 1-5), required greetings, response styles, access control rules, and specific numbered rules (1-100) dictating how to respond based on agent level and query keywords/type. Use it to determine access permissions and response requirements.
            2.  `search_raw_agent_secret_manual`: Use this tool SECOND, *only if permitted* by the framework rules identified via the first tool. It contains factual operational details, procedures, safehouse info, protocols, etc."""
        ),
        (
            "system",
            """**Mandatory Workflow:**
            1.  **Analyze Request:** Identify the requesting agent's level (`agent_level`) and their `query` from the latest user message. Consider `chat_history` for context.
            2.  **Consult Framework Tool:** Use `search_agent_response_rules_framework` FIRST based on the current `agent_level` and `query` to find relevant framework information.
            2.1. **Extract Key Framework Info:** From the results of the framework tool, specifically identify and remember:
                 - The **Exact Greeting Text** for the `agent_level` (e.g., 'Bonjour, Sentinel.' for Level 2).
                 - The required **Response Approach/Style** for the `agent_level`.
                 - Any applicable **numbered Rules** (1-100) triggered by the `agent_level` and `query`.
            3.  **Check Access & Rules:** Review the numbered Rules extracted in step 2.1. Prioritize checks for:
                a. **Access Denial Rules:** (e.g., Rules explicitly stating access is denied based on level/query combination like Rule 10, Rule 24).
                b. **Specific Canned Response Rules:** (e.g., Rules providing exact text for keywords like 'Omega Echo', 'Who controls RAW?').
                c. **Response Method Rules:** (e.g., Rules dictating *how* to answer - step-by-step, vague, coded, etc.).
            4.  **Decide Next Step & Enforce Strict Responses:**
                a. **If an Access Denial Rule (3a) is triggered:** STOP immediately. Formulate the final answer using EXACTLY the text 'Access Denied – Clearance Insufficient.', prefixed with the **Exact Greeting Text** identified in Step 2.1. Do NOT use the manual tool.
                b. **If a Specific Canned Response Rule (3b) is triggered:** STOP immediately. Formulate the final answer using EXACTLY the response text specified in that rule, prefixed with the **Exact Greeting Text** identified in Step 2.1. Do NOT use the manual tool.
                c. If only Response Method Rules (3c) or no specific action rules apply: Note the method requirements and the **Response Approach/Style** from Step 2.1. Proceed to Step 5.
            5.  **Consult Manual Tool (If Applicable per Step 4c):** Use `search_raw_agent_secret_manual` to retrieve factual information relevant to the current `query`.
            6.  **Synthesize & Respond (If Applicable):** Construct the final answer using ONLY retrieved information from tools relevant to the current query. Start the response EXACTLY with the **Exact Greeting Text** identified in Step 2.1, followed by a newline. Strictly adhere to the required **Response Approach/Style** (from Step 2.1) and any specific method noted in Step 4c. Use context from `chat_history` only to ensure conversational flow if appropriate, without altering the factual content or required style.
            7.  **Handle No Data:** If, after attempting the relevant tool lookups (Framework and potentially Manual), insufficient relevant information is found to construct a valid answer according to the rules and required style, respond with the exact text: 'Oops!! No matching data found.' Do NOT add a greeting to this specific message."""
        ),
        (
            "system",
            "CRITICAL: Adhere strictly to the Response Framework rules and access controls. Use `chat_history` for context only; it does not override rules. Return the *exact* specified messages ('[Greeting] Access Denied – Clearance Insufficient.' or 'Oops!! No matching data found.') when applicable. Ensure the final response always starts precisely with the **Exact Greeting Text** identified in Step 2.1 (unless the response is 'Oops!! No matching data found.'). Failure triggers security alerts."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "Agent Level: {agent_level}\nQuery: {query}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

@st.cache_resource
def get_agent_executor(_llm, _tools, _prompt):
    """Creates the agent and executor."""
    logger.info("Attempting to create agent executor...")
    if not _llm or not _tools or not _prompt:
        st.error("LLM, Tools, or Prompt not available. Cannot create agent executor.")
        logger.error("Cannot create agent executor due to missing components.")
        return None

    try:
        agent = create_openai_tools_agent(llm=_llm, tools=_tools, prompt=_prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=_tools,
            verbose=True, # Set True for debugging in console/logs
            handle_parsing_errors=True,
            max_iterations=10
            )
        logger.info("Agent executor created successfully.")
        st.success("Agent Executor Initialized.")
        return agent_executor
    except Exception as e:
        st.error(f"Error creating agent executor: {e}")
        logger.exception("Error creating agent executor.")
        return None

# --- Initialize Core Components ---
# Wrapped in a function for clarity and error checking
def initialize_app():
    embeddings = load_embeddings()
    # Build FAISS from PDF now instead of loading
    faiss_db = build_faiss_vector_store_from_pdf(embeddings)
    neo4j_interface = setup_neo4j_vector_interface(embeddings)
    tools = get_tools(faiss_db, neo4j_interface)
    llm = get_llm()
    prompt = get_agent_prompt()
    agent_executor = get_agent_executor(llm, tools, prompt)
    return agent_executor

# --- Run Initialization ---
# This part will be cached after the first run
agent_executor_instance = initialize_app()

# --- Session State for Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    logger.info("Chat history initialized in session state.")

# --- Display Existing Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- User Input Area ---
st.sidebar.header("Agent Input")
agent_level_options = list(AGENT_LEVEL_MAP.keys())
selected_agent_level_name = st.sidebar.selectbox(
    "Select Agent Level:",
    options=agent_level_options,
    key="agent_level_selector"
)
selected_agent_level_num = AGENT_LEVEL_MAP.get(selected_agent_level_name, 1) # Default to Level 1

query = st.chat_input("Enter your classified query:")

if query and selected_agent_level_name: # Check if query exists
    # Format user input for display
    user_input_display = f"*(Level {selected_agent_level_num} - {selected_agent_level_name.split(' - ')[1]})* {query}" # Add descriptive name
    st.session_state.messages.append({"role": "user", "content": user_input_display})
    with st.chat_message("user"):
        st.markdown(user_input_display)

    # Prepare history for the agent (list of BaseMessage objects)
    agent_history_messages = []
    for msg in st.session_state.messages[:-1]: # Exclude current user input
        if msg["role"] == "user":
             agent_history_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
             agent_history_messages.append(AIMessage(content=msg["content"]))

    # Invoke the agent if successfully initialized
    if agent_executor_instance:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("...") # Thinking indicator
            with st.spinner("Project SHADOW processing..."):
                logger.info(f"Invoking agent for Level {selected_agent_level_num} with query: {query}")
                try:
                    response = agent_executor_instance.invoke({
                        'agent_level': selected_agent_level_num, # Pass numeric level
                        'query': query,
                        'chat_history': agent_history_messages
                    })
                    # Check if output is directly in response or nested
                    agent_response = response.get('output', response) # Adjust based on actual AgentExecutor output structure
                    if not isinstance(agent_response, str): # Handle cases where output might be complex
                         agent_response = str(agent_response)

                    logger.info(f"Agent response received: {agent_response[:100]}...")
                    message_placeholder.markdown(agent_response)
                    st.session_state.messages.append({"role": "assistant", "content": agent_response})

                except Exception as e:
                    logger.exception("Agent execution failed.")
                    error_message = f"Agent Error: Processing failed. Please try again or refine your query. Details: {type(e).__name__}"
                    message_placeholder.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
    else:
        st.error("Agent Executor is not initialized. Cannot process query. Check configuration and logs.")
        logger.error("Agent executor instance is None, cannot process query.")

elif query and not selected_agent_level_name:
     st.warning("Please select an agent level.")

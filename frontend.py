import streamlit as st
import httpx
import asyncio

API_URL = "http://localhost:8000"

st.set_page_config(page_title="DevBot", page_icon="🤖", layout="wide")

if "session_id" not in st.session_state:
    st.session_state.session_id = None
    st.session_state.messages = []
    st.session_state.indexed = False

async def check_backend():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{API_URL}/", timeout=2.0)
            return response.status_code == 200
    except:
        return False

async def create_session():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{API_URL}/session/new", timeout=10.0)
            if response.status_code == 200:
                return response.json()["session_id"]
    except Exception as e:
        st.error(f"Failed to create session: {str(e)}")
    return None

async def index_code(folder_path: str):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{API_URL}/index",
                json={"folder_path": folder_path, "session_id": st.session_state.session_id},
                timeout=120.0
            )
            if response.status_code == 200:
                return response.json()["message"]
    except Exception as e:
        st.error(f"Indexing error: {str(e)}")
    return None

async def send_message(message: str):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{API_URL}/chat",
                json={"message": message, "session_id": st.session_state.session_id},
                timeout=60.0
            )
            if response.status_code == 200:
                return response.json()
    except Exception as e:
        st.error(f"Error: {str(e)}")
    return None

async def clear_history():
    if not st.session_state.session_id:
        return
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{API_URL}/session/{st.session_state.session_id}/clear",
                timeout=10.0
            )
            if response.status_code == 200:
                st.session_state.messages = []
                st.success("History cleared!")
    except Exception as e:
        st.error(f"Failed to clear history: {str(e)}")

backend_status = asyncio.run(check_backend())

st.title("Developer Bot")

col1, col2 = st.columns([4, 1])
with col2:
    if st.button("🗑️ Clear History"):
        asyncio.run(clear_history())
        st.rerun()

if not backend_status:
    st.error("⚠️ Backend not running! Start: `uvicorn backend:app --reload`")
    st.stop()
else:
    st.success("✅ Backend connected")

if not st.session_state.session_id:
    session_id = asyncio.run(create_session())
    if session_id:
        st.session_state.session_id = session_id
    else:
        st.error("Failed to create session")
        st.stop()

with st.sidebar:
    st.header("⚙️ RAG Setup")
    folder = st.text_input("Code folder path", value=".", help="Path to your code")
    
    if st.button("📚 Index Code", type="primary"):
        with st.spinner("Indexing..."):
            result = asyncio.run(index_code(folder))
            if result:
                st.session_state.indexed = True
                st.success(result)
    
    st.divider()
    st.markdown("### 📊 Status")
    st.write(f"**RAG**: {'✅ Indexed' if st.session_state.indexed else '❌ Not indexed'}")
    st.write(f"**Context7**: ✅ Ready")
    
    st.divider()
    st.markdown("### ℹ️ How to use")
    st.markdown("""
    1. Index your local code with RAG
    2. Ask questions
    3. Get answers from:
       - Your local code (RAG)
       - Official docs (Context7)
    """)

st.markdown("---")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask about your code or documentation..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Searching RAG + Context7..."):
            result = asyncio.run(send_message(prompt))
            if result:
                response = result["response"]
                sources = result.get("sources", {})
                
                st.write(response)
                
                if sources:
                    st.caption(f"📚 Sources: RAG: {'✅' if sources.get('rag') else '❌'} | Context7: {'✅' if sources.get('context7') else '❌'}")
                
                st.session_state.messages.append({"role": "assistant", "content": response})
    
    st.rerun()
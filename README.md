This is a simple demo app that lets you **ask questions about YouTube videos** by analyzing the video’s transcript using a locally running LLM

User types query
↓
Transcript fetched from YouTube
↓
Transcript split into chunks
↓
Chunks stored in FAISS vector database
↓
Relevant transcript chunks retrieved via vector search
↓
Prompt + retrieved transcript fed to local Phi-3 model
↓
Answer displayed in Streamlit




![image](https://github.com/user-attachments/assets/39e9ec87-50e5-4f63-b226-5072f01b4aed)

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import tempfile
import os

# === Load API Keys ===
load_dotenv()

# === Streamlit UI ===
st.set_page_config(page_title="AI Resume Matcher", layout="centered")
st.title("📄 AI Resume Matcher with Gemini")

uploaded_file = st.file_uploader("📤 Upload your Resume (PDF)", type=["pdf"])
job_description_text = st.text_area("📝 Paste the Job Description", height=200)

# === Utility: Format points nicely ===
def format_points(text):
    return text.replace("1.", "\n1.").replace("2.", "\n2.").replace("3.", "\n3.").replace("4.", "\n4.")

# === Define Output Model ===
class ResumeMatchFeedback(BaseModel):
    Rating: str = Field(
        description="Rating like '85/100'. Also include section-wise breakdown like:\n1. Skills: ...\n2. Projects: ..."
    )
    Strength: str = Field(description="Key strengths: in points like 1. ..., 2. ...")
    Weakness: str = Field(description="Key weaknesses: 1. ..., 2. ... etc.")
    Recommendations: str = Field(description="Improvement suggestions with examples")
    NotImpactful: str = Field(description="Lines in the resume that are not effective")
    Recommended_projects: str = Field(description="Project ideas relevant to the job")
    Key_changes: str = Field(description="Suggested line rewrites: Before → After format rember to add arrow and next line")

# === Process Resume & Match ===
if uploaded_file and job_description_text and st.button("🔍 Analyze Resume"):

    # Save PDF to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_path = tmp_file.name

    # Load and split documents
    loader = PyPDFLoader(temp_path)
    resume_docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    resume_chunks = splitter.split_documents(resume_docs)

    job_doc = [Document(page_content=job_description_text)]
    job_chunks = splitter.split_documents(job_doc)

    # Embeddings and FAISS store
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    resume_store = FAISS.from_documents(resume_chunks, embedding_model)

    # Retrieve relevant resume content
    retriever = resume_store.as_retriever(search_type="similarity", search_kwargs={"k": 12})
    relevant_chunks = retriever.invoke("Find content in the resume related to this job description.")
    relevant_resume_text = "\n".join([doc.page_content for doc in relevant_chunks])
    job_text = "\n".join([doc.page_content for doc in job_chunks])

    # Prompt Template
    prompt = PromptTemplate(
        template = """
Rate how well this resume fits the following job.

### Job Description:
{job}

### Resume:
{resume}

Be strict and domain-aware:
- Penalize heavily if job field (e.g., mechanical engineering vs software development) doesn't match.
- Penalize missing core job-specific skills.
- Be honest. A mismatch should score very low.

Respond in the following sections:
- Rating: score out of 100 with section-wise analysis (Skills, Experience, Projects, etc).
- Strength: bullet points.
- Weakness: bullet points.
- Recommendations: bullet points.
- NotImpactful: bullet points.
- Recommended_projects: bullet points.
- Key_changes: show 'Before:' and 'After:' for lines that should change and add arrow '->' example : before : .... \t -> \t after: ... next before after pair in next line.
""",
        input_variables=["job", "resume"]
    )

    # LLM and chain
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    structured_model = llm.with_structured_output(ResumeMatchFeedback)
    chain = prompt | structured_model

    # Invoke model
    with st.spinner("Analyzing resume..."):
        result = chain.invoke({
            "job": job_text,
            "resume": relevant_resume_text
        })

    # Clean up
    os.remove(temp_path)

    # === Display Results ===
    st.subheader("📊 Matching Results")
    st.markdown(f"**🎯 Rating:** {result.Rating}")
    st.markdown("**✅ Strengths:**")
    st.markdown(format_points(result.Strength))

    st.markdown("**⚠️ Weaknesses:**")
    st.markdown(format_points(result.Weakness))

    st.markdown("**💡 Recommendations:**")
    st.markdown(format_points(result.Recommendations))

    st.markdown("**🚫 Not Impactful Lines:**")
    st.markdown(format_points(result.NotImpactful) if result.NotImpactful != "N/A" else "_No non-impactful lines found._")

    st.markdown("**🧪 Recommended Projects:**")
    st.markdown(format_points(result.Recommended_projects))

    st.markdown("**✏️ Key Changes (Before → After):**")
    st.markdown(format_points(result.Key_changes) if result.Key_changes != "N/A" else "_No specific changes recommended._")

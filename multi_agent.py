import os
# load API key from .env file
from dotenv import load_dotenv
load_dotenv()
# we can aslo add in the code
# os.environ["GOOGLE_API_KEY"] = "your-api-key"

from typing import Literal, TypedDict
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# 1. Initialize Vector DB (Run this once or move to a helper function)
def initialize_vector_db(pdf_paths: list[str]):
    all_docs = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        all_docs.extend(loader.load())

    # Split text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(all_docs)

    # Create Chroma DB (persist_directory makes it reusable)
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    return vectorstore.as_retriever()

# make sure to create ./research_papers dir and run download_papers.py
pdf_files = [
"./research_papers/2410.05258v2.Differential_Transformer.pdf",
"./research_papers/2501.13353v2.Contrast__A_Hybrid_Architecture_of_Transformers_and_State_Space_Models_for_Low_Level_Vision.pdf",
"./research_papers/2503.01124v1.ViKANformer__Embedding_Kolmogorov_Arnold_Networks_in_Vision_Transformers_for_Pattern_Based_Learning.pdf",
"./research_papers/2503.04112v2.The_Spinning_Blimp__Design_and_Control_of_a_Novel_Minimalist_Aerial_Vehicle_Leveraging_Rotational_Dynamics_and_Locomotion.pdf",
"./research_papers/2505.23735v1.ATLAS__Learning_to_Optimally_Memorize_the_Context_at_Test_Time.pdf",
"./research_papers/2508.09834v1.Speed_Always_Wins__A_Survey_on_Efficient_Architectures_for_Large_Language_Models.pdf",
"./research_papers/2510.03989v1.A_Mathematical_Explanation_of_Transformers_for_Large_Language_Models_and_GPTs.pdf",
"./research_papers/2510.05364v1.The_End_of_Transformers__On_Challenging_Attention_and_the_Rise_of_Sub_Quadratic_Architectures.pdf",
"./research_papers/2512.24880v2.mHC__Manifold_Constrained_Hyper_Connections.pdf",
"./research_papers/2601.02360v1.Heterogeneous_Low_Bandwidth_Pre_Training_of_LLMs.pdf",
]

retriever = initialize_vector_db(pdf_files)

# --- STATE DEFINITION ---
class AgentState(TypedDict):
    research_query : str
    retrieved_docs : str
    analysis       : str
    final_paper    : str
    review_feedback: str
    revision_count : int  # To prevent infinite loops

# --- AGENT DEFINITIONS ---

def librarian_agent(state: AgentState):
    """Searches Chroma for technical data related to the research query."""
    # Find top 5 most relevant segments across all PDFs
    docs = retriever.invoke(state["research_query"])
    context = "\n\n".join([d.page_content for d in docs])
    return {"retrieved_docs": context}

def reviewer_agent(state: AgentState):
    """Evaluates the paper for academic rigor and clarity."""
    prompt = f"""
    Review this research paper draft for:
    1. Scientific novelty
    2. Use of evidence from the provided data: {state['retrieved_docs']}
    3. Logical flow

    Paper Draft: {state['final_paper']}

    If the paper is excellent, start your response with 'APPROVED'.
    Otherwise, provide specific constructive feedback on what to improve.
    """
    response = llm.invoke(prompt)
    return {"review_feedback": response.content}

# Logic to decide: Finish or Revise?
def should_continue(state: AgentState) -> Literal["revise", "end"]:
    if "APPROVED" in state["review_feedback"] or state.get("revision_count", 0) >= 2:
        return "end"
    return "revise"

def analyst_agent(state: AgentState):
    """Proposes research, now considering previous feedback."""
    feedback_context = f"\nPrevious Review Feedback: {state.get('review_feedback', 'None')}"
    prompt = f"Analyze these snippets: {state['retrieved_docs']} and suggest a hypothesis. {feedback_context}"
    response = llm.invoke(prompt)
    return {
        "analysis": response.content,
        "revision_count": state.get("revision_count", 0) + 1
    }

def writer_agent(state: AgentState):
    """Drafts the paper based on the analysis."""
    prompt = f"Write a research paper draft. Focus: {state['research_query']}. Evidence: {state['analysis']}"
    response = llm.invoke(prompt)
    return {"final_paper": response.content}


# --- GRAPH ORCHESTRATION ---
workflow = StateGraph(AgentState)
workflow.add_node("librarian", librarian_agent)
workflow.add_node("analyst"  , analyst_agent)
workflow.add_node("writer"   , writer_agent)
workflow.add_node("reviewer" , reviewer_agent)

# Define the flow
workflow.set_entry_point("librarian")
workflow.add_edge("librarian", "analyst")
workflow.add_edge("analyst"  , "writer")
workflow.add_edge("writer"   , "reviewer")

# Add the Conditional Loop
workflow.add_conditional_edges(
    "reviewer",
    should_continue,
    {
        "revise": "analyst", # Loop back to Analyst to refine the idea
        "end": END
    }
)

app = workflow.compile()

# Run it
inputs = {
     "research_query": "Identify remaining limitations in sub-quadratic Transformers that current normalization-free methods have yet to solve"
     }
final_state = app.invoke(inputs)
print(final_state["final_paper"])
# also save to file
with open("final_paper.txt",'w') as f:
    f.write(final_state["final_paper"])

# Example of research_query
#############################
# Query Type    Example research_query                                                                                                          Goal for the Agents
# Comparative   "Compare the noise-reduction mechanisms of the Differential Transformer (DIFF) against traditional RMSNorm stability."          Force the Librarian to find specific math/architecture sections in the DIFF paper.
# Evolutionary  "How do 2025 hybrid architectures (Mamba-Transformer) address the 'associative recall' failures of 2024 sub-quadratic models?"  The Analyst must identify a timeline of improvements across multiple papers.
# Efficiency    "Analyze the trade-offs between KV-cache compression and dynamic normalization (Dynamic Tanh) for 10M+ context lengths."        Focuses the synthesis on hardware efficiency and deployment.
# Gap Finding   "Identify remaining limitations in sub-quadratic Transformers that current normalization-free methods have yet to solve."       Specifically targets the 'Future Work' sections of the papers to propose a new hypothesis.

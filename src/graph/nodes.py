import os
import json
from typing import List, Dict, Any
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import PromptTemplate
from src.config import config
from src.graph.state import CRAGState
from src.retrieval import Retriever

# Initialize retriever globally so we don't reload embeddings/models on every run
retriever = Retriever()

def get_llm():
    if config.llm_provider.lower() == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model_name=config.groq_model,
            api_key=config.groq_api_key if config.groq_api_key else None,
            temperature=0.0,
            max_tokens=1024,
            model_kwargs={"reasoning_effort": "low"},
        )
    elif config.llm_provider.lower() == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=config.gemini_model,
            google_api_key=config.gemini_api_key if config.gemini_api_key else None,
            temperature=0.0,
        )
    else:
        raise ValueError(f"Unsupported llm_provider: {config.llm_provider}")


def retrieve_documents(state: CRAGState) -> Dict:
    query = state["query"]
    # We use the existing reranked hybrid search method
    docs = retriever.reranked_hybrid_search(query, top_k=5)
    return {
        "documents": docs,
        "execution_trace": ["retrieve_documents: Executed local hybrid retrieval"]
    }


def grade_documents(state: CRAGState) -> Dict:
    docs = state["documents"]
    th_high = config.reranker_threshold_high
    th_low = config.reranker_threshold_low

    correct_docs = []
    ambiguous_docs = []

    for doc in docs:
        score = doc.get("score", -999.0)
        if score >= th_high:
            correct_docs.append(doc)
        elif score >= th_low:
            ambiguous_docs.append(doc)
        # below th_low: dropped (knowledge refinement)

    has_correct = len(correct_docs) > 0
    web_search_required = not has_correct

    # FIX: if local retrieval is being judged insufficient overall, don't carry
    # forward ambiguous local noise into the web-augmented context — start
    # clean so web results aren't diluted by irrelevant local chunks.
    filtered_docs = correct_docs if has_correct else []

    return {
        "documents": filtered_docs,
        "web_search_required": web_search_required,
        "execution_trace": [
            f"grade_documents: {len(correct_docs)} CORRECT, {len(ambiguous_docs)} AMBIGUOUS (dropped), "
            f"{len(docs) - len(correct_docs) - len(ambiguous_docs)} INCORRECT (dropped). "
            f"Web search required: {web_search_required}"
        ]
    }


def rewrite_query(state: CRAGState) -> Dict:
    query = state["query"]
    llm = get_llm()
    
    prompt = PromptTemplate(
        template="""You are an expert technical assistant. Rewrite the following query 
to be a highly effective search engine query.
Extract the core intent and keywords. DO NOT arbitrarily append generic terms like 
"industrial equipment" or "energy efficiency" unless they are vital to the specific query.
Output ONLY the rewritten query text.

Original Query: {query}""",
        input_variables=["query"],
    )
    
    chain = prompt | llm
    rewritten_query = chain.invoke({"query": query}).content.strip()
    
    return {
        "query": rewritten_query,
        "execution_trace": [f"rewrite_query: Rewrote query to -> '{rewritten_query}'"]
    }

def web_search(state: CRAGState) -> Dict:
    query = state["query"]
    docs = state["documents"]
    iters = state.get("web_search_iterations", 0)
    
    if "TAVILY_API_KEY" not in os.environ and config.tavily_api_key:
        os.environ["TAVILY_API_KEY"] = config.tavily_api_key

    search_tool = TavilySearchResults(
        max_results=3, 
        include_domains=config.allowed_web_domains
    )
    
    try:
        web_results = search_tool.invoke({"query": query})
        
        # Defensive parsing against unexpected string returns
        if isinstance(web_results, str):
            trace_msg = f"web_search: Search failed/returned error message: {web_results}"
        elif isinstance(web_results, list):
            valid_results = 0
            for res in web_results:
                if isinstance(res, dict) and "url" in res and "content" in res:
                    docs.append({
                        "id": f"web_{hash(res['url'])}",
                        "text": res["content"],
                        "doc_name": res["url"],
                        "page": 0,
                        "score": 1.0
                    })
                    valid_results += 1
            trace_msg = f"web_search: Retrieved {valid_results} valid documents from allowed domains."
        else:
            trace_msg = f"web_search: Unexpected response format from Tavily: {type(web_results)}"
            
    except Exception as e:
        trace_msg = f"web_search: Search failed with exception: {e}"

    return {
        "documents": docs,
        "web_search_iterations": iters + 1,
        "execution_trace": [trace_msg]
    }


def generate_answer(state: CRAGState) -> Dict:
    query = state["query"]
    docs = state["documents"]
    iters = state.get("generation_iterations", 0)
    llm = get_llm()
    
    context = "\n\n".join([f"[Source: {d.get('doc_name', 'Unknown')}, Page: {d.get('page', 0)}] {d['text']}" for d in docs])
    
    prompt = PromptTemplate(
        template="""You are a technical assistant for industrial equipment. 
Answer the user query based ONLY on the provided context. 
If the context does not contain the answer, say "I cannot answer this based on the provided documents."
Include inline citations to the [Source] used.

Context:
{context}

Query: {query}
Answer:""",
        input_variables=["context", "query"],
    )
    
    chain = prompt | llm
    generation = chain.invoke({"context": context, "query": query}).content.strip()
    
    return {
        "generation": generation,
        "generation_iterations": iters + 1,
        "execution_trace": [f"generate_answer: Generated answer (iteration {iters + 1})"]
    }


def check_groundedness(state: CRAGState) -> Dict:
    query = state["query"]
    docs = state["documents"]
    generation = state["generation"]
    iters = state["generation_iterations"]
    llm = get_llm()

    context = "\n\n".join([d['text'] for d in docs])

    prompt = PromptTemplate(
        template="""You are a reasonable evaluator checking for hallucinations.
Does the ANSWER contain any factual claims that are clearly NOT supported by
and CANNOT be reasonably inferred or paraphrased from the CONTEXT?
Minor paraphrasing, rewording, or summarizing of context content is NOT a hallucination.
Only flag claims that introduce information absent from the context or contradict it.

CRITICAL EXCEPTION: If the ANSWER is a refusal or admission of ignorance (e.g., "I cannot answer this based on the provided documents."), you MUST respond with "YES".

CONTEXT: {context}
ANSWER: {generation}

Respond with EXACTLY ONE WORD: "YES" (if all claims are supported/reasonably inferable) or "NO" (if there are unsupported or contradicted claims).""",
        input_variables=["context", "generation"],
    )

    chain = prompt | llm
    raw_response = chain.invoke({"context": context, "generation": generation})
    
    # print(f"[DEBUG] finish_reason: {raw_response.response_metadata.get('finish_reason')}")
    # print(f"[DEBUG] token_usage: {raw_response.response_metadata.get('token_usage')}")
    
    result = raw_response.content.strip().upper()
    cleaned = ''.join(c for c in result if c.isalpha())

    if cleaned == "YES":
        grounded = True
        verdict_note = "YES"
    elif cleaned == "NO":
        grounded = False
        verdict_note = "NO"
    else:
        # FIX: malformed/empty response is NOT a confirmed hallucination finding —
        # log it distinctly rather than silently treating it as a real "NO".
        grounded = False
        verdict_note = f"MALFORMED_RESPONSE(raw={raw_response!r})"
        # print(f"[WARNING] check_groundedness got an unparseable response: {raw_response!r}")

    final_generation = generation
    if not grounded and iters >= 2:
        final_generation = "[WARNING: The system was unable to verify this response against the retrieved sources.]\n\n" + generation

    return {
        "grounded": grounded,
        "generation": final_generation,
        "execution_trace": [f"check_groundedness: Grounded={grounded} (LLM Output: {verdict_note})"]
    }
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
    query = state["query"]
    docs = state["documents"]
    th_high = config.reranker_threshold_high
    th_low = config.reranker_threshold_low
    llm = get_llm()
    trace_logs = []

    # 1. INTENT GUARD: Check for Greetings / Small Talk first
    # This prevents the system from doing web searches for "hello"
    try:
        intent_prompt = PromptTemplate(
            template="""Classify the following user input. 
If it is a simple greeting, pleasantry, or conversational small talk (e.g., "hi", "hello", "who are you", "thanks", "good morning"), respond with EXACTLY ONE WORD: GREETING
If it is a factual question, technical query, or request for information, respond with EXACTLY ONE WORD: FACTUAL

Input: {query}
Response:""",
            input_variables=["query"],
        )
        intent_chain = intent_prompt | llm
        intent_response = intent_chain.invoke({"query": query}).content.strip().upper()
        
        if "GREETING" in intent_response:
            # Inject a synthetic document so the Generator knows how to answer
            synthetic_doc = {
                "id": "sys_greeting",
                "text": "System Directive: The user is greeting you or making conversational small talk. Respond warmly, professionally, and briefly. Do not include standard citations for this response.",
                "doc_name": "System_Logic",
                "page": 0,
                "score": 1.0
            }
            return {
                "documents": [synthetic_doc],
                "web_search_required": False,
                "execution_trace": ["grade_documents: Query classified as GREETING. Bypassing document retrieval and web search."]
            }
    except Exception as e:
        trace_logs.append(f"Intent guard check failed (falling back to standard RAG): {e}")

    # 2. MATH FILTER: Cross-Encoder Thresholds
    correct_docs = []
    ambiguous_docs = []
    for doc in docs:
        score = doc.get("score", -999.0)
        if score >= th_high:
            correct_docs.append(doc)
        elif score >= th_low:
            ambiguous_docs.append(doc)

    # 3. SEMANTIC GRADER: The "Smart" Check to prevent False Positives
    # We only spend tokens checking docs that already passed the math filter
    final_docs = []
    
    if correct_docs:
        grade_prompt = PromptTemplate(
            template="""You are a strict grading assistant evaluating document relevance.

If the provided document explicitly contains the factual information required to answer the user's query, respond with EXACTLY ONE WORD: YES
If the document does NOT contain the specific facts to answer the query (even if it shares broad keywords), respond with EXACTLY ONE WORD: NO

Query: {query}
Document: {document_text}

Response:""",
            input_variables=["query", "document_text"],
        )
        grade_chain = grade_prompt | llm
        
        for i, doc in enumerate(correct_docs):
            try:
                grade_response = grade_chain.invoke({"query": query, "document_text": doc["text"]}).content.strip().upper()
                if "YES" in grade_response:
                    final_docs.append(doc)
                    trace_logs.append(f"Doc {i+1} Semantic Check: YES")
                else:
                    trace_logs.append(f"Doc {i+1} Semantic Check: NO (Dropped as false positive)")
            except Exception as e:
                # Failsafe: if LLM fails, trust the math score and keep the doc so the pipeline survives
                final_docs.append(doc)
                trace_logs.append(f"Doc {i+1} Semantic Check: ERROR (Kept via failsafe)")

    # 4. ROUTING LOGIC
    has_correct = len(final_docs) > 0
    web_search_required = not has_correct

    # We intentionally drop ambiguous docs completely. If local docs fail the semantic check,
    # we want a completely clean slate for the web search to prevent pollution.
    
    summary_trace = (
        f"grade_documents: Math filter -> {len(correct_docs)} pass, {len(ambiguous_docs)} ambig, {len(docs) - len(correct_docs) - len(ambiguous_docs)} drop. "
        f"Semantic LLM filter -> {len(final_docs)} verified. Web search required: {web_search_required}"
    )
    
    return {
        "documents": final_docs,
        "web_search_required": web_search_required,
        "execution_trace": [summary_trace] + trace_logs
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
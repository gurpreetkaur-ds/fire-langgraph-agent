"""LangGraph workflow: Question Decomposer -> Research/Answer Agent -> Synthesizer.

START -> decompose -> answer -> synthesize -> END

Every LLM call in every node goes through FireChatModel (app/fire_llm.py), which
is the sole gateway to Fire's /v1/chat endpoint.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import List, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from evaluation.tracing import traced_invoke

from .fire_llm import FireAPIError, FireChatModel

logger = logging.getLogger("fire_agent.graph")

MAX_SUBQUESTIONS = 4


class AgentState(TypedDict, total=False):
    question: str
    sub_questions: List[str]
    qa_pairs: List[dict]
    final_answer: str
    error: Optional[str]


# Two distinct "bots": a fast model for decomposition, a stronger model for
# answering and synthesis. Both are Fire species_name values from /v1/models.
decomposer_llm = FireChatModel(species_name="claude-haiku-4-5", temperature=0.2, max_tokens=512)
answer_llm = FireChatModel(species_name="claude-sonnet-4-5", temperature=0.4, max_tokens=700)
synth_llm = FireChatModel(species_name="claude-sonnet-4-5", temperature=0.3, max_tokens=900)

DECOMPOSER_SYSTEM_PROMPT = (
    "You are a Question Decomposer. Break the user's question into 2 to "
    f"{MAX_SUBQUESTIONS} focused, independently-answerable sub-questions that together "
    "cover what's needed for a complete answer. Respond with ONLY a JSON array of "
    "strings, no prose, no markdown fences. Example: [\"...\", \"...\"]"
)

ANSWER_SYSTEM_PROMPT = (
    "You are a Research Agent. Answer the given sub-question clearly and concisely "
    "(3-5 sentences). Be factual and specific; do not pad with filler."
)

SYNTH_SYSTEM_PROMPT = (
    "You are a Synthesis Agent. You are given an original question and a set of "
    "sub-question/answer pairs researched by another agent. Combine them into one "
    "clear, well-organized final answer to the ORIGINAL question. Use short "
    "paragraphs or bullet points where helpful. Do not mention the sub-questions "
    "process explicitly -- just answer well."
)

# Changes whenever any agent prompt changes, so evaluation results can be compared across prompt versions.
PROMPT_VERSION = hashlib.sha256(
    "\n".join([DECOMPOSER_SYSTEM_PROMPT, ANSWER_SYSTEM_PROMPT, SYNTH_SYSTEM_PROMPT]).encode()
).hexdigest()[:8]


def _extract_json_array(text: str) -> Optional[list]:
    text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
    return None


async def decompose_node(state: AgentState) -> dict:
    question = state["question"]
    try:
        resp = await traced_invoke(
            "decomposer", "decompose", decomposer_llm,
            [SystemMessage(content=DECOMPOSER_SYSTEM_PROMPT), HumanMessage(content=question)],
        )
        parsed = _extract_json_array(str(resp.content))
        sub_questions = [str(q).strip() for q in parsed if str(q).strip()] if parsed else []
        if not sub_questions:
            logger.warning("decompose_fallback_to_original_question")
            sub_questions = [question]
        sub_questions = sub_questions[:MAX_SUBQUESTIONS]
        return {"sub_questions": sub_questions, "error": None}
    except FireAPIError as exc:
        logger.error("decompose_node_error=%s", exc)
        return {"sub_questions": [], "error": str(exc)}


async def answer_node(state: AgentState) -> dict:
    if state.get("error"):
        return {}
    qa_pairs: list[dict] = []
    for sub_q in state.get("sub_questions", []):
        try:
            resp = await traced_invoke(
                "answer_agent", "answer", answer_llm,
                [SystemMessage(content=ANSWER_SYSTEM_PROMPT), HumanMessage(content=sub_q)],
            )
            qa_pairs.append({"question": sub_q, "answer": str(resp.content)})
        except FireAPIError as exc:
            logger.error("answer_node_error=%s", exc)
            return {"qa_pairs": qa_pairs, "error": str(exc)}
    return {"qa_pairs": qa_pairs}


async def synthesize_node(state: AgentState) -> dict:
    if state.get("error"):
        return {}
    qa_pairs = state.get("qa_pairs", [])
    notes = "\n\n".join(f"Q: {p['question']}\nA: {p['answer']}" for p in qa_pairs)
    prompt = f"Original question: {state['question']}\n\nResearch notes:\n{notes}"
    try:
        resp = await traced_invoke(
            "synthesizer", "synthesize", synth_llm,
            [SystemMessage(content=SYNTH_SYSTEM_PROMPT), HumanMessage(content=prompt)],
        )
        return {"final_answer": str(resp.content)}
    except FireAPIError as exc:
        logger.error("synthesize_node_error=%s", exc)
        return {"error": str(exc)}


def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("decompose", decompose_node)
    builder.add_node("answer", answer_node)
    builder.add_node("synthesize", synthesize_node)
    builder.add_edge(START, "decompose")
    builder.add_edge("decompose", "answer")
    builder.add_edge("answer", "synthesize")
    builder.add_edge("synthesize", END)
    return builder.compile()


graph = build_graph()

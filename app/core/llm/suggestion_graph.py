from __future__ import annotations

from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from app.core.llm.llm_utils import chat_complete, parse_json_response
from app.core.llm.prompt_builder import (
    format_document_context,
    format_transcript_sections,
)
from app.core.stt.transcript_buffer import TranscriptEntry

SAFE_FALLBACK_REPLY = (
    "I'd want to double-check the specifics before answering that — "
    "let me follow up on that point."
)

_RELEVANCE_SYSTEM = """You filter retrieved document snippets for a live meeting copilot.

Given the latest thing the other person said, decide which numbered snippets are actually relevant.

Rules:
- Return JSON only: {"relevant_indices": [1, 3]} using 1-based indices from the input.
- Include a snippet ONLY if it directly helps answer or respond to the latest [Them] message.
- If none are relevant, return {"relevant_indices": []}.
- Do not explain your reasoning outside the JSON."""

_VERIFY_SYSTEM = """You are a strict grounding verifier for meeting reply suggestions.

Check whether the draft reply contains ANY unsupported factual claim.

Supported sources ONLY:
1. The conversation transcript (including [You] and [Them] lines)
2. The provided document context snippets
3. Universal pleasantries and non-factual conversational glue

Unsupported claims include: specific numbers, SLAs, deadlines, policy details, locations, names, or product facts NOT explicitly present in those sources.

Return JSON only:
{
  "passed": true,
  "issues": "brief explanation if failed, else empty string"
}

Set passed=true only when every factual statement in the draft is grounded."""

_REVISE_SYSTEM = """You revise a meeting reply to remove all unsupported factual claims.

Rules:
- Keep the same intent and conversational tone.
- Remove or soften any specific fact not explicitly in the transcript or document context.
- Prefer hedging ("I'd want to confirm…") over inventing details.
- Output ONLY the revised reply text, nothing else."""

_DRAFT_SYSTEM = """You are helping the user respond in real time during a live meeting or call.

You receive a recent transcript labeled [Them] / [You], a conversation summary, and optional document snippets.

Rules:
- Reply to the MOST RECENT [Them] line only; [You] lines are context, never the addressee.
- 2-3 short, natural sentences the user could say out loud.
- NEVER invent specific facts, numbers, policies, dates, or names not in the transcript or document context.
- If a specific fact is needed but missing, hedge instead of guessing.
- Ignore irrelevant document snippets entirely.
- Conversational tone, not formal writing."""


class SuggestionGraphState(TypedDict):
    segment_index: int
    recent_transcript: list[TranscriptEntry]
    conversation_summary: str
    latest_them_text: str
    transcript_context: str
    candidates: list[dict]
    retrieved_chunks: list[dict]
    draft_suggestion: str
    final_suggestion: str
    verification_passed: bool
    verification_feedback: str
    revision_count: int
    max_revisions: int


def _filter_relevance(state: SuggestionGraphState) -> dict:
    candidates = state["candidates"]
    if not candidates:
        return {"retrieved_chunks": []}

    latest = state["latest_them_text"]
    numbered = "\n\n".join(
        f"[{index}] (source: {chunk.get('source', 'unknown')}, "
        f"score: {chunk.get('score', 0.0):.3f})\n{chunk.get('text', '').strip()}"
        for index, chunk in enumerate(candidates, start=1)
    )
    try:
        raw = chat_complete(
            [
                {"role": "system", "content": _RELEVANCE_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f'Latest [Them] message:\n"{latest}"\n\n'
                        f"Candidate snippets:\n{numbered}"
                    ),
                },
            ],
            temperature=0.0,
            json_mode=True,
        )
        payload = parse_json_response(raw)
        indices = payload.get("relevant_indices", [])
        if not isinstance(indices, list):
            indices = []
        selected = [
            candidates[int(i) - 1]
            for i in indices
            if isinstance(i, int) and 1 <= i <= len(candidates)
        ]
    except Exception:
        selected = list(candidates)

    return {"retrieved_chunks": selected}


def _generate_draft(state: SuggestionGraphState) -> dict:
    context_section = format_document_context(state["retrieved_chunks"])
    user_content = (
        f"{state['transcript_context']}\n\n"
        f"{context_section}\n\n"
        "Suggest what the user could say next."
    )
    draft = chat_complete(
        [
            {"role": "system", "content": _DRAFT_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        temperature=0.3,
    )
    return {"draft_suggestion": draft}


def _verify_grounding(state: SuggestionGraphState) -> dict:
    context_section = format_document_context(state["retrieved_chunks"])
    try:
        raw = chat_complete(
            [
                {"role": "system", "content": _VERIFY_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"{state['transcript_context']}\n\n"
                        f"{context_section}\n\n"
                        f'Draft reply:\n"{state["draft_suggestion"]}"'
                    ),
                },
            ],
            temperature=0.0,
            json_mode=True,
        )
        payload = parse_json_response(raw)
        passed = bool(payload.get("passed", False))
        feedback = str(payload.get("issues", "") or "")
    except Exception:
        passed = False
        feedback = "Verification step failed; using safe fallback."

    return {
        "verification_passed": passed,
        "verification_feedback": feedback,
    }


def _revise_draft(state: SuggestionGraphState) -> dict:
    context_section = format_document_context(state["retrieved_chunks"])
    revised = chat_complete(
        [
            {"role": "system", "content": _REVISE_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"{state['transcript_context']}\n\n"
                    f"{context_section}\n\n"
                    f'Original draft:\n"{state["draft_suggestion"]}"\n\n'
                    f"Verifier feedback:\n{state['verification_feedback']}"
                ),
            },
        ],
        temperature=0.2,
    )
    return {
        "draft_suggestion": revised.strip(),
        "revision_count": state["revision_count"] + 1,
    }


def _finalize_passed(state: SuggestionGraphState) -> dict:
    return {"final_suggestion": state["draft_suggestion"].strip()}


def _finalize_safe(state: SuggestionGraphState) -> dict:
    return {"final_suggestion": SAFE_FALLBACK_REPLY}


def _route_after_verify(state: SuggestionGraphState) -> Literal["finalize", "revise", "safe"]:
    if state["verification_passed"]:
        return "finalize"
    if state["revision_count"] < state["max_revisions"]:
        return "revise"
    return "safe"


def build_suggestion_graph():
    graph = StateGraph(SuggestionGraphState)
    graph.add_node("filter_relevance", _filter_relevance)
    graph.add_node("generate_draft", _generate_draft)
    graph.add_node("verify_grounding", _verify_grounding)
    graph.add_node("revise_draft", _revise_draft)
    graph.add_node("finalize", _finalize_passed)
    graph.add_node("safe_fallback", _finalize_safe)

    graph.add_edge(START, "filter_relevance")
    graph.add_edge("filter_relevance", "generate_draft")
    graph.add_edge("generate_draft", "verify_grounding")
    graph.add_conditional_edges(
        "verify_grounding",
        _route_after_verify,
        {
            "finalize": "finalize",
            "revise": "revise_draft",
            "safe": "safe_fallback",
        },
    )
    graph.add_edge("revise_draft", "verify_grounding")
    graph.add_edge("finalize", END)
    graph.add_edge("safe_fallback", END)
    return graph.compile()


_GRAPH = build_suggestion_graph()


def run_suggestion_graph(
    *,
    segment_index: int,
    recent_transcript: list[TranscriptEntry],
    conversation_summary: str,
    candidates: list[dict],
    max_revisions: int = 1,
) -> SuggestionGraphState:
    latest_them_text, transcript_context = format_transcript_sections(
        recent_transcript,
        conversation_summary=conversation_summary,
    )
    initial: SuggestionGraphState = {
        "segment_index": segment_index,
        "recent_transcript": recent_transcript,
        "conversation_summary": conversation_summary,
        "latest_them_text": latest_them_text,
        "transcript_context": transcript_context,
        "candidates": candidates,
        "retrieved_chunks": [],
        "draft_suggestion": "",
        "final_suggestion": "",
        "verification_passed": False,
        "verification_feedback": "",
        "revision_count": 0,
        "max_revisions": max_revisions,
    }
    return _GRAPH.invoke(initial)

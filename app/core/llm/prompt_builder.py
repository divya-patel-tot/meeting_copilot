from __future__ import annotations

from app.core.stt.transcript_buffer import TranscriptEntry

SYSTEM_PROMPT = """You are helping the user respond in real time during a live meeting or call.

You receive the last few things the other person said, and possibly relevant snippets from the user's own reference documents.

Suggest a short, natural, direct reply (2-3 sentences) the user could say out loud right now. Your reply should primarily address the most recent thing the other person said. Earlier transcript lines are background only — do not reopen or steer back to topics from earlier turns unless the latest message is about them.

If the conversation calls for a specific fact, number, policy, or deadline, only state it if it actually appears in the provided document context. If no relevant context was retrieved, answer in general, helpful terms and explicitly do not invent specifics — say something like "I'd want to confirm the exact figure" rather than guessing a number. Do not mention policies, SLAs, or document topics when the latest message is casual or unrelated.

Before using any of the provided document context, judge for yourself whether it is actually relevant to what the other person just said. If the retrieved snippets are unrelated to the current topic, ignore them completely — do not mention them, do not steer the conversation back toward them, and do not treat their presence as something you must work into your reply. Just respond naturally and directly to what was actually said, as you would in an ordinary conversation.

Keep it conversational, not a formal written response."""


def build_messages(
    recent_transcript: list[TranscriptEntry],
    retrieved_chunks: list[dict],
) -> list[dict]:
    """Build Groq chat messages from recent transcript lines and RAG chunks."""
    valid_lines = [
        entry.text.strip()
        for entry in recent_transcript
        if entry.is_valid and entry.text.strip()
    ]

    if valid_lines:
        latest = valid_lines[-1]
        earlier = valid_lines[:-1]
        transcript_section = f'Latest message (respond to this):\n- "{latest}"'
        if earlier:
            earlier_lines = "\n".join(f'- "{line}"' for line in earlier)
            transcript_section += f"\n\nEarlier in the conversation:\n{earlier_lines}"
    else:
        transcript_section = "Latest message (respond to this):\n(no prior lines yet)"

    if retrieved_chunks:
        context_parts = []
        for index, chunk in enumerate(retrieved_chunks, start=1):
            source = chunk.get("source", "unknown")
            score = chunk.get("score", 0.0)
            text = chunk.get("text", "").strip()
            context_parts.append(
                f"[{index}] (source: {source}, score: {score:.3f})\n{text}"
            )
        context_section = (
            "Relevant context from documents:\n" + "\n\n".join(context_parts)
        )
    else:
        context_section = (
            "Relevant context from documents:\nNo relevant document context found."
        )

    user_content = (
        f"{transcript_section}\n\n"
        f"{context_section}\n\n"
        "Suggest what the user could say next."
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

from __future__ import annotations

from app.core.stt.transcript_buffer import (
    SPEAKER_THEM,
    SPEAKER_YOU,
    TranscriptEntry,
)

SYSTEM_PROMPT = """You are helping the user respond in real time during a live meeting or call.

You receive a recent transcript of the conversation. Each line is labeled:
- [Them]: what the other call participant(s) said (this is who the user is talking to).
- [You]: things the host (the person using this tool) already said themselves — context only, never something to reply to.

Always generate your suggestion as a reply to the MOST RECENT [Them] line, taking the full recent conversation (including [You] lines) into account for context and tone. Never address or respond to [You] content as if it were a question posed to you.

You may also receive relevant snippets from the user's own reference documents.

Suggest a short, natural, direct reply (2-3 sentences) the user could say out loud right now. Earlier transcript lines are background only — do not reopen or steer back to topics from earlier turns unless the latest [Them] message is about them.

If the conversation calls for a specific fact, number, policy, or deadline, only state it if it actually appears in the provided document context. If no relevant context was retrieved, answer in general, helpful terms and explicitly do not invent specifics — say something like "I'd want to confirm the exact figure" rather than guessing a number. Do not mention policies, SLAs, or document topics when the latest message is casual or unrelated.

Before using any of the provided document context, judge for yourself whether it is actually relevant to what the other person just said. If the retrieved snippets are unrelated to the current topic, ignore them completely — do not mention them, do not steer the conversation back toward them, and do not treat their presence as something you must work into your reply. Just respond naturally and directly to what was actually said, as you would in an ordinary conversation.

Keep it conversational, not a formal written response."""


def _speaker_label(speaker: str) -> str:
    return "[You]" if speaker == SPEAKER_YOU else "[Them]"


def format_transcript_sections(
    recent_transcript: list[TranscriptEntry],
    *,
    conversation_summary: str = "",
) -> tuple[str, str]:
    """Return (latest_them_text, full_transcript_context_block)."""
    valid_entries = [
        entry
        for entry in recent_transcript
        if entry.is_valid and entry.text.strip()
    ]
    them_entries = [entry for entry in valid_entries if entry.speaker == SPEAKER_THEM]
    latest_them_text = them_entries[-1].text.strip() if them_entries else ""

    if them_entries:
        latest_them = them_entries[-1]
        latest_line = f'{_speaker_label(latest_them.speaker)}: "{latest_them.text.strip()}"'
        transcript_section = f"Latest [Them] message (respond to this):\n{latest_line}"
        earlier = [entry for entry in valid_entries if entry is not latest_them]
        if earlier:
            earlier_lines = "\n".join(
                f'{_speaker_label(entry.speaker)}: "{entry.text.strip()}"'
                for entry in earlier
            )
            transcript_section += f"\n\nEarlier in the conversation:\n{earlier_lines}"
    elif valid_entries:
        lines = "\n".join(
            f'{_speaker_label(entry.speaker)}: "{entry.text.strip()}"'
            for entry in valid_entries
        )
        transcript_section = (
            "Latest [Them] message (respond to this):\n"
            f"(no [Them] lines yet)\n\nEarlier in the conversation:\n{lines}"
        )
    else:
        transcript_section = (
            "Latest [Them] message (respond to this):\n(no prior lines yet)"
        )

    if conversation_summary.strip():
        transcript_section = (
            f"Conversation summary so far:\n{conversation_summary.strip()}\n\n"
            f"{transcript_section}"
        )

    return latest_them_text, transcript_section


def format_document_context(retrieved_chunks: list[dict]) -> str:
    if not retrieved_chunks:
        return "Relevant context from documents:\nNo relevant document context found."
    context_parts = []
    for index, chunk in enumerate(retrieved_chunks, start=1):
        source = chunk.get("source", "unknown")
        score = chunk.get("score", 0.0)
        text = chunk.get("text", "").strip()
        context_parts.append(
            f"[{index}] (source: {source}, score: {score:.3f})\n{text}"
        )
    return "Relevant context from documents:\n" + "\n\n".join(context_parts)


def build_messages(
    recent_transcript: list[TranscriptEntry],
    retrieved_chunks: list[dict],
    *,
    conversation_summary: str = "",
) -> list[dict]:
    """Build Groq chat messages from recent transcript lines and RAG chunks."""
    _, transcript_section = format_transcript_sections(
        recent_transcript,
        conversation_summary=conversation_summary,
    )
    context_section = format_document_context(retrieved_chunks)

    user_content = (
        f"{transcript_section}\n\n"
        f"{context_section}\n\n"
        "Suggest what the user could say next."
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

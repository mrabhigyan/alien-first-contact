"""The two players (Speaker, Listener) and the "brain" they think with.

A brain turns a prompt into a structured answer. GeminiBrain calls the real model;
MockBrain answers at random so the whole game can be tested for free.

Each agent keeps its own private notebook. The only thing that ever passes from
speaker to listener is the message itself.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Protocol, TypeVar

from pydantic import BaseModel, Field, ValidationError

from world import Obj

# ---------------------------------------------------------------- structured outputs


class SpeakerOut(BaseModel):
    reasoning: str = Field(description="Brief private reasoning, written before the message.")
    message: list[str] = Field(
        description="The message: a list of symbols. Each item is exactly ONE symbol."
    )


class ListenerOut(BaseModel):
    reasoning: str = Field(description="Brief private reasoning, written before the choice.")
    choice: int = Field(description="The number of the chosen object, as listed.")


class NotesOut(BaseModel):
    reasoning: str = Field(description="Brief private reasoning, written before the notes.")
    notes: str = Field(description="Your updated private notes.")


T = TypeVar("T", bound=BaseModel)

# ---------------------------------------------------------------- brains


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class Brain(Protocol):
    usage: Usage

    def ask(self, system: str, prompt: str, schema: type[T]) -> T: ...


def _network_errors() -> tuple[type[BaseException], ...]:
    """Connection/timeout errors worth retrying. The HTTP library google-genai uses
    has changed between versions (httpx, httpx2), so look it up instead of hard-coding it."""
    found: list[type[BaseException]] = [ConnectionError, TimeoutError]
    for module_name in ("httpx2", "httpx"):
        try:
            module = __import__(module_name)
            found.append(module.TransportError)
        except (ImportError, AttributeError):
            pass
    return tuple(found)


class GeminiBrain:
    """Calls Gemini with a Pydantic schema and retries on rate limits and server errors."""

    RETRYABLE_CODES = {429, 500, 502, 503, 504}

    def __init__(
        self, api_key: str, model: str, max_retries: int = 6, base_delay: float = 2.0
    ) -> None:
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.usage = Usage()

    def ask(self, system: str, prompt: str, schema: type[T]) -> T:
        from google.genai import errors, types

        # No temperature/top_p/top_k: newer Gemini models have deprecated them.
        config = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=schema,
        )
        for attempt in range(self.max_retries + 1):
            last_try = attempt == self.max_retries
            try:
                response = self._client.models.generate_content(
                    model=self.model, contents=prompt, config=config
                )
            except errors.APIError as exc:
                if exc.code in self.RETRYABLE_CODES and not last_try:
                    self._wait(attempt, f"API error {exc.code}")
                    continue
                raise
            except _network_errors() as exc:
                if not last_try:
                    self._wait(attempt, f"network error {type(exc).__name__}")
                    continue
                raise

            self._record_usage(response)
            if isinstance(response.parsed, schema):
                return response.parsed
            try:
                return schema.model_validate_json(response.text or "")
            except ValidationError:
                if last_try:
                    raise
                self._wait(attempt, "unparseable response")
        raise RuntimeError("unreachable")

    def _wait(self, attempt: int, why: str) -> None:
        delay = min(60.0, self.base_delay * 2**attempt)
        print(f"    [retry] {why}; waiting {delay:.0f}s")
        time.sleep(delay)

    def _record_usage(self, response: object) -> None:
        self.usage.calls += 1
        meta = getattr(response, "usage_metadata", None)
        if meta is None:
            return
        self.usage.input_tokens += meta.prompt_token_count or 0
        # Thinking tokens are billed as output, so count them too.
        self.usage.output_tokens += (meta.candidates_token_count or 0) + (
            getattr(meta, "thoughts_token_count", None) or 0
        )


class MockBrain:
    """A free stand-in that answers at random. Expect ~1/candidates success."""

    def __init__(
        self, rng: random.Random, alphabet: list[str], n_candidates: int, max_len: int
    ) -> None:
        self.rng = rng
        self.alphabet = alphabet
        self.n_candidates = n_candidates
        self.max_len = max_len
        self.usage = Usage()

    def ask(self, system: str, prompt: str, schema: type[T]) -> T:
        self.usage.calls += 1
        if schema is SpeakerOut:
            length = self.rng.randint(1, self.max_len)
            message = [self.rng.choice(self.alphabet) for _ in range(length)]
            return SpeakerOut(reasoning="(mock)", message=message)  # type: ignore[return-value]
        if schema is ListenerOut:
            choice = self.rng.randint(1, self.n_candidates)
            return ListenerOut(reasoning="(mock)", choice=choice)  # type: ignore[return-value]
        if schema is NotesOut:
            return NotesOut(reasoning="(mock)", notes="(mock notes)")  # type: ignore[return-value]
        raise TypeError(f"MockBrain cannot answer schema {schema.__name__}")


# ---------------------------------------------------------------- prompts
# Rule: these prompts must never say what any symbol means, or hint at a scheme.

SPEAKER_SYSTEM = """You are playing a communication game with a partner you cannot see.
You and your partner share no language. The only way to communicate is a message made
of symbols from this alphabet:

{alphabet}

Each round you are shown one target object. Send a message of 1 to {max_len} symbols.
Your partner sees several objects and must pick the target using only your message.

Rules:
- Use only symbols from the alphabet above. Each list item must be exactly one symbol.
- No letters, words, digits, punctuation, or spaces. Breaking a rule fails the round.
- No one has told your partner what any symbol means. Meaning can only come from how
  the two of you use the symbols over many rounds.

After each round you learn whether your partner picked correctly.
Your private notebook of past rounds (and your own notes, once you write them)
is shown with each round. Use it."""

LISTENER_SYSTEM = """You are playing a communication game with a partner you cannot see.
You and your partner share no language. Your partner was shown one target object and
sent you a message made of symbols from this alphabet:

{alphabet}

Each round you see the message and a numbered list of objects. Exactly one of them is
the target. Pick it by its number.

No one has told you what any symbol means. Meaning can only come from how the two of
you use the symbols over many rounds.

After each round you learn whether you were right, and which object was the target.
Your private notebook of past rounds (and your own notes, once you write them)
is shown with each round. Use it."""


def _notebook_text(notebook: list[str], window: int) -> str:
    if not notebook:
        return "(empty: this is the first round)"
    shown = notebook[-window:]
    header = f"(showing the last {len(shown)} of {len(notebook)} rounds)\n" if len(
        notebook
    ) > window else ""
    return header + "\n".join(shown)


NOTES_REQUEST = """Time to update your private notes. Only you will ever read them.
Rewrite them so they keep everything useful for future rounds, using your current notes
and the rounds since you last wrote them. Keep them under {words} words.

Your current notes:
{notes}

Rounds since your last update:
{recent}

Reason briefly, then give your new notes."""

NOTES_WORDS = 150


# ---------------------------------------------------------------- agents


@dataclass
class Agent:
    """Shared memory behaviour. Each instance has its own notebook and notes;
    nothing here is ever shared between the two agents."""

    brain: Brain
    alphabet: list[str]
    memory_window: int = 40
    notebook: list[str] = field(default_factory=list)
    notes: str = ""
    notes_through: int = 0  # notebook entries already folded into `notes`

    def system_prompt(self) -> str:
        raise NotImplementedError

    def memory_text(self) -> str:
        """What the agent sees of its past: its own notes (if any) + recent rounds."""
        recent = _notebook_text(self.notebook, self.memory_window)
        if not self.notes:
            return f"Your notebook:\n{recent}"
        return (f"Your private notes (written by you after round {self.notes_through}):\n"
                f"{self.notes}\n\nYour notebook, recent rounds:\n{recent}")

    def update_notes(self) -> None:
        """Fold every round since the last update into the agent's own notes."""
        new_entries = self.notebook[self.notes_through:]
        if not new_entries:
            return
        prompt = NOTES_REQUEST.format(
            words=NOTES_WORDS,
            notes=self.notes or "(none yet)",
            recent="\n".join(new_entries),
        )
        out = self.brain.ask(self.system_prompt(), prompt, NotesOut)
        self.notes = out.notes.strip()
        self.notes_through = len(self.notebook)


@dataclass
class Speaker(Agent):
    max_len: int = 3

    def system_prompt(self) -> str:
        return SPEAKER_SYSTEM.format(alphabet=" ".join(self.alphabet), max_len=self.max_len)

    def speak(self, round_no: int, target: Obj) -> SpeakerOut:
        prompt = (
            f"{self.memory_text()}\n\n"
            f"Round {round_no}. Target object: {target.name}\n"
            "Reason briefly, then give your message."
        )
        return self.brain.ask(self.system_prompt(), prompt, SpeakerOut)

    def remember(self, round_no: int, target: Obj, message: list[str], outcome: str) -> None:
        sent = " ".join(message) if message else "(nothing)"
        self.notebook.append(f"Round {round_no}: target {target.name} | I sent {sent} | {outcome}")


@dataclass
class Listener(Agent):
    def system_prompt(self) -> str:
        return LISTENER_SYSTEM.format(alphabet=" ".join(self.alphabet))

    def listen(self, round_no: int, message: list[str], candidates: list[Obj]) -> ListenerOut:
        options = "\n".join(f"{i}. {obj.name}" for i, obj in enumerate(candidates, start=1))
        prompt = (
            f"{self.memory_text()}\n\n"
            f"Round {round_no}. Message received: {' '.join(message)}\n"
            f"Objects:\n{options}\n"
            "Reason briefly, then give the number of your choice."
        )
        return self.brain.ask(self.system_prompt(), prompt, ListenerOut)

    def remember(
        self,
        round_no: int,
        message: list[str],
        candidates: list[Obj],
        picked: Obj | None,
        target: Obj,
    ) -> None:
        options = ", ".join(obj.name for obj in candidates)
        picked_name = picked.name if picked else "(invalid choice)"
        result = "RIGHT" if picked == target else f"WRONG, the target was {target.name}"
        self.notebook.append(
            f"Round {round_no}: message {' '.join(message)} | options: {options} | "
            f"I picked {picked_name} | {result}"
        )

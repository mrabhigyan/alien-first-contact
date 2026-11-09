# Alien First Contact

**Can two AI agents with no shared language invent one from scratch?**

Two LLM agents play a repeated guessing game using only a small set of randomly generated
alien symbols. They never share memory or notes — the only channel between them is the
message. Over many rounds, a shared language emerges.

> As AI agents built by different companies start working together, how do they learn
> to understand each other?

## How it works

1. The **speaker** sees a target object (e.g. "red circle") and sends 1–3 alien symbols.
2. The **listener** sees several candidates and picks the one it thinks was meant.
3. Both learn whether it succeeded and record the round in their **private memory**.

Random guessing with 4 candidates = **25%**. Anything above that is communication.

## Results

_Coming soon: success curve, emerged dictionary, compositionality score, held-out accuracy._

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
cp .env.example .env            # then add your Gemini API key
python main.py --mock --rounds 20
```

## Experiment rules

- No human language in messages — only symbols from the run's alphabet.
- The validator is the source of truth; invalid messages count as failures.
- The alphabet is randomized every run.
- Agents never share memory, and are never told what any symbol means.

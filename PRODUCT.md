# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Primary: a college student presenting their own final-year/academic
project (DeRAG) to a viva panel of professors/judges, on the student's
own laptop, panel viewing over-shoulder / close range (not projected).
The panel is technical — they understand blockchain, RAG, and embeddings
at a working level — so the UI does not need to explain fundamentals,
only make the project's specific claim legible fast.

Secondary: the same panel (or other graders) may independently clone the
repo and run the UI themselves afterward, without the student narrating
live. The interface must hold up unguided, not only as a backdrop to a
live walkthrough.

## Product Purpose

DeRAG is an academic MVP demonstrating that retrieval-augmented
generation can be made more resistant to data poisoning by
cross-referencing retrieved documents against an on-chain, crowd-sourced
reputation ledger before trusting them — instead of ranking purely by
embedding similarity. The Streamlit page in scope for this design pass
(`app/main.py`) is the live proof of that idea: a user asks a question,
picks a retrieval mode, and sees both the generated answer and a
per-source audit trail showing whether each source is cryptographically
verifiable and how much it's trusted.

## Positioning

Unlike a conventional RAG demo, which can only show "here's an answer and
here are the sources it came from," DeRAG can show *why each source is or
isn't trustworthy* — a live, per-query verification against a real
(if local) blockchain and content-addressed storage layer, not a
simulated or hard-coded trust score. That verifiable audit trail is the
one thing a neighboring RAG project's UI could not truthfully replicate
without the same underlying registry/IPFS machinery.

## Operating Context

Entirely local, run from the student's own machine during the viva:
a local Hardhat blockchain node, a local IPFS daemon, a local Ollama
instance running `gemma4:12b`, and a local persistent ChromaDB store —
no cloud APIs. Any of the three network-backed services (blockchain,
IPFS, LLM) can be independently up or down at any moment (a judge might
ask "what if the blockchain node crashes?" as a direct test), so the UI
must visibly and legibly degrade per-service rather than fail silently
or crash. The demo is a single continuous session: one query box, a
retrieval-mode toggle, and results in the same view — there is no
multi-page flow to navigate.

## Capabilities and Constraints

- Query input + toggle between "Standard RAG" (pure embedding similarity)
  and "DeRAG" (trust-weighted re-ranking against the reputation ledger).
- Streamed LLM answer with inline source citations (`[1]`, `[2]`, ...).
- A sidebar audit trail, up to 3 sources per query, each showing: a
  truncated content hash, the IPFS CID, the publisher's blockchain
  address, an on-chain reputation number, and a verification badge.
- The verification badge has four distinct states, and distinguishing
  them at a glance is this design pass's central job:
  - **verified** — the registry and IPFS both responded, and the fetched
    content's hash matches what was registered on-chain.
  - **unverified** — both services responded, but the hash did not
    match (a genuine integrity failure, not a service outage).
  - **unknown** — the registry responded but IPFS could not be reached,
    so verification could not be attempted; the other fields
    (publisher, reputation, CID) are still shown, since they came from
    the registry, not IPFS.
  - **error** — the registry itself is unreachable, or the document was
    never registered on-chain at all; nothing else in that row can be
    trusted, so no other fields are shown.
- Scope boundary for this design pass: only `app/main.py`'s presentation
  layer (Streamlit rendering calls, layout, and any custom CSS/HTML) is
  in play. `app/ui_helpers.py` is pure, unit-tested business logic (the
  state machine that produces which of the four badge states applies,
  and the retrieval-mode dispatch) with 100+ passing tests behind it —
  its logic, signatures, and return shapes are out of scope and must not
  change.
- No authentication, no multi-user concerns, no persistence of past
  queries between sessions — each query is a fresh, single-shot
  interaction.

## Evidence on Hand

- `README.md` and `DECISIONS.md` — the project's own running technical
  narrative across all 8 milestones so far, including this UI's design
  rationale.
- `data/sample.txt` — the one real ingested document used in live
  testing/demos; not a fabricated example.
- No customer testimonials, benchmarks, press, or external case studies
  exist or should be implied — this is an academic MVP, not a shipped
  product.

## Product Principles

- The audit trail is the thesis, not a supporting data table. Every
  design decision on this surface should ask whether it makes the
  verified/unverified/unknown/error distinction *more* immediately
  legible, not just tidier.
- Read as a credible research/engineering tool, not a consumer app or
  generic SaaS dashboard — the audience is technical and will notice
  (and discount) generic AI-app styling.
- Degrade visibly, never silently. A stopped service should be obvious
  in the specific section it affects, matching the project's existing
  engineering discipline around per-service error handling.
- Must hold up both narrated (live, over-shoulder) and unnarrated
  (independently run later) — don't rely on a presenter's voice to
  explain what a badge means.

## Accessibility & Inclusion

No project-specific accessibility requirement has been established;
default to reasonable contrast and legibility (the badge distinction
requirement above already implies this) rather than a formal standard.

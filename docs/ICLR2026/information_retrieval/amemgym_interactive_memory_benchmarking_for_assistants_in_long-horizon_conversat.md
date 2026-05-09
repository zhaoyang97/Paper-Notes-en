---
title: >-
  [Paper Note] AMemGym: Interactive Memory Benchmarking for Assistants in Long-Horizon Conversations
description: >-
  [ICLR 2026][conversational memory evaluation] This paper proposes AMemGym — the first long-horizon conversational memory benchmark environment supporting on-policy interactive evaluation. It drives LLM-simulated users via structured data sampling (user profile → state evolution → personalized QA), reveals ranking biases inherent in off-policy evaluation, and systematically diagnoses write/read/utilization failure modes across RAG, long-context, and agent-based memory systems.
tags:
  - ICLR 2026
  - conversational memory evaluation
  - on-policy evaluation
  - user state tracking
  - memory diagnostics
  - simulated user
date: 2026-05-08
content_hash: a013b1dfdc385443
---

# AMemGym: Interactive Memory Benchmarking for Assistants in Long-Horizon Conversations

**Conference**: ICLR 2026
**arXiv**: [2603.01966](https://arxiv.org/abs/2603.01966)
**Code**: [https://agi-eval-official.github.io/amemgym/](https://agi-eval-official.github.io/amemgym/)
**Area**: Information Retrieval
**Keywords**: conversational memory evaluation, on-policy evaluation, user state tracking, memory diagnostics, simulated user

## TL;DR
This paper proposes AMemGym — the first long-horizon conversational memory benchmark environment supporting on-policy interactive evaluation. It drives LLM-simulated users via structured data sampling (user profile → state evolution → personalized QA), reveals ranking biases inherent in off-policy evaluation, and systematically diagnoses write/read/utilization failure modes across RAG, long-context, and agent-based memory systems.

## Background & Motivation

**Background**: LLM assistants must manage memory across long-horizon conversations to deliver personalized service, yet all existing memory benchmarks (MSC, LongMemEval, PersonaMem, etc.) rely on static off-policy data for evaluation.

**Limitations of Prior Work**: (a) Off-policy evaluation cannot capture the consequences of an assistant's own conversational behavior, as memory operations are tightly coupled with interaction patterns; (b) manually curated test scenarios are costly and do not scale; (c) evaluation metrics are predominantly end-to-end QA accuracy, offering no diagnosis of why memory failures occur.

**Key Challenge**: The effectiveness of an agent's memory is highly dependent on its own interaction patterns (on-policy), yet evaluation is conducted over conversation histories produced by other models (off-policy), causing a systematic mismatch between evaluation conclusions and real-world deployment performance.

**Goal**: Construct a scalable interactive environment that supports on-policy evaluation, multi-granularity diagnostics, and memory strategy optimization.

**Key Insight**: Structured data serves as the "skeleton" (user state schema → state evolution trajectory → state-dependent QA), while LLM role-playing generates natural dialogue as the "flesh" — structure ensures evaluability, interaction ensures authenticity.

**Core Idea**: Reverse-engineer user state variables and evolution trajectories from evaluation objectives, then use structured data to drive simulated users for on-policy interaction — yielding an approach that is both reliable and scalable.

## Method

### Overall Architecture
AMemGym proceeds in four steps: (1) structured data generation (user profile → state schema → state evolution → personalized answers); (2) on-policy interaction (LLM-simulated user role-plays to expose state → assistant responds → memory updated); (3) periodic evaluation (predefined questions probe the assistant after each state-evolution period); (4) diagnostic analysis (failure attribution across write/read/utilization stages).

### Key Designs

1. **Structured Data Sampling (Reverse Engineering Strategy)**

    - Function: Reverse-constructs complete simulation data from evaluation objectives.
    - Mechanism: Sample user profile $p$ → generate evaluation questions $\mathcal{Q}_p$ → extract required state variables $\Sigma = \{(s_j, V_j)\}_{j=1}^M$ → generate state evolution trajectory $\mathcal{T}_\sigma = (\sigma_0, ..., \sigma_{N_p})$ (each transition triggered by a narrative event $e_t$) → generate personalized answers $r_{i,\nu}$ for each (question, state) pair and validate via reflection.
    - Design Motivation: Guarantees evaluation reliability at the ground level — every evaluation question has an explicit ground-truth state dependency, enabling automatic scoring.

2. **On-Policy Interaction Generation**

    - Function: Uses LLMs to simulate users naturally exposing state information during real interactions.
    - Mechanism: Pre-generated state-bearing utterances serve as conversation openers for each session, ensuring critical state information enters the dialogue. The LLM user role-plays based on the user profile, current state, and dialogue history.
    - Design Motivation: Fixed openers ensure benchmark consistency; free role-playing ensures conversational naturalness. The key distinction is that the assistant's responses genuinely influence subsequent dialogue — something off-policy methods cannot capture.

3. **Three-Stage Diagnostic Metrics (Write/Read/Utilization)**

    - Function: Decomposes end-to-end QA failures into three stages of memory operation.
    - Mechanism: For each failed question, the framework first checks whether the assistant "knows" all relevant state values — if so, the failure is classified as a **utilization failure** (knows but cannot leverage the knowledge); otherwise, the most recent write point is queried — if the assistant did not know at write time, the failure is classified as a **write failure** (information was never stored); if it knew at write time but not at evaluation time, the failure is a **read failure** (stored but unrecoverable).
    - Design Motivation: End-to-end metrics can only indicate that a failure occurred, not where. Three-stage diagnostics allow memory system designers to precisely locate bottlenecks.

### Meta-Evaluation
- State exposure quality: 99.1% (200 samples)
- Conversational state consistency: 99.2% (748 annotations)
- Ground-truth judgment reliability: LLM–human annotator agreement 0.94–0.96

## Key Experimental Results

### Main Results — On-Policy vs. Off-Policy

| Memory Configuration | On-policy↑ | Off-policy↑ | Rank Change |
|---------|-----------|------------|---------|
| AWE-(2,4,30) | .291 | .253 | ▼3 |
| AWE-(2,4,10) | .275 | .273 | ▲2 |
| RAG-(2,4,30) | .227 | .241 | ▲2 |
| LLM (vanilla) | .203 | .198 | ▼1 |

**Rankings differ substantially** — off-policy evaluation leads to incorrect configuration selection recommendations.

### Native LLMs Evaluation

| Model | Memory Score (Period 1→10) |
|------|--------------------------|
| claude-sonnet-4 | .336 (best) |
| gemini-2.5-flash | .327 |
| gpt-4.1 | .244 |
| deepseek-v3 | .152 |

All models have an upper bound >0.8, but performance drops sharply as interactions accumulate, with most eventually falling to near-random levels.

### Ablation Study

| Strategy | Write↓ | Read↓ | Util.↓ |
|------|--------|-------|--------|
| LLM | .301 | .087 | .244 |
| RAG | .377 | .172 | .067 |
| AWE | .338 | .159 | .074 |
| AWI | .286 | .245 | .122 |

### Key Findings
- **RAG/AWE reduce utilization failures but increase write/read failures** — external storage mitigates long-context issues but introduces information loss.
- **AWI achieves the lowest write failure but the highest read failure** — compressing information into context facilitates writing but hinders retrieval.
- **Lower update frequency → higher read failure** — retaining too many local messages causes confusion across multiple memory sources.
- **Retrieval count has minimal impact on read/utilization but has a non-monotonic effect on write** — excessive retrieval introduces noise that impairs write-time judgment.

## Highlights & Insights
- The **empirical comparison of on-policy vs. off-policy evaluation** is the paper's most valuable contribution — it directly demonstrates the ranking bias of off-policy evaluation, serving as a warning for all memory research.
- The **three-stage diagnostic framework** (write/read/utilization) provides a highly actionable analysis structure — making the strengths and weaknesses of each memory strategy precise enough to guide engineering decisions.
- The dual-layer design combining structured data with free dialogue is elegant — it simultaneously ensures evaluation reliability (ground truth exists) and interaction authenticity (on-policy).
- The paper presents a proof-of-concept for agent self-evolution — autonomously improving memory strategies using environment feedback.

## Limitations & Future Work
- Simulated users remain LLM-based role-players; the gap from real user behavior is not thoroughly discussed.
- State variables are represented as discrete value sets, which cannot express continuous evolution or fuzzy preferences.
- Evaluation primarily focuses on personalized QA — other memory demands in dialogue (e.g., task tracking, knowledge accumulation) are not covered.
- Data generation costs remain high (gpt-4.1 is used for both data generation and user simulation).
- The base configuration requires 128K context and the extra configuration requires 512K, restricting the range of applicable models.

## Related Work & Insights
- **vs. LongMemEval/PersonaMem**: These are static off-policy benchmarks; AMemGym is the first to enable on-policy interactive evaluation.
- **vs. CollabLLM**: CollabLLM uses simulated users to train collaborative models; AMemGym focuses on evaluation rather than training.
- **vs. Mem0/A-Mem**: These are memory systems being evaluated; AMemGym provides the environment in which they are assessed.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ On-policy memory evaluation combined with a structured-data-driven interactive environment represents a fully original design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers multiple memory systems, multiple LLMs, configuration ablations, diagnostic analysis, and meta-evaluation — exceptionally comprehensive.
- Writing Quality: ⭐⭐⭐⭐⭐ Problem motivation is clearly articulated, framework diagrams are intuitive, and diagnostic analysis is substantive.
- Value: ⭐⭐⭐⭐⭐ Defines a new evaluation paradigm for LLM memory system research.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Mem-PAL: Towards Memory-based Personalized Dialogue Assistants for Long-term User-Agent Interaction](../../AAAI2026/information_retrieval/mem-pal_towards_memory-based_personalized_dialogue_assistants_for_long-term_user.md)
- [\[ICLR 2026\] TokMem: One-Token Procedural Memory for Large Language Models](tokmem_one-token_procedural_memory_for_large_language_models.md)
- [\[AAAI 2026\] ComoRAG: A Cognitive-Inspired Memory-Organized RAG for Stateful Long Narrative Reasoning](../../AAAI2026/information_retrieval/comorag_a_cognitive-inspired_memory-organized_rag_for_stateful_long_narrative_re.md)
- [\[ICLR 2026\] Leveraging Data to Say No: Memory Augmented Plug-and-Play Selective Prediction](leveraging_data_to_say_no_memory_augmented_plug-and-play_selective_prediction.md)
- [\[ICLR 2026\] Beyond RAG vs. Long-Context: Learning Distraction-Aware Retrieval for Efficient Knowledge Grounding](beyond_rag_vs_long-context_learning_distraction-aware_retrieval_for_efficient_kn.md)

<!-- RELATED:END -->

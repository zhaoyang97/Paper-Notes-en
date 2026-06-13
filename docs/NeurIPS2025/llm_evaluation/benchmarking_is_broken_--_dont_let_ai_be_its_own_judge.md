---
title: >-
  [Paper Note] Benchmarking is Broken — Don't Let AI be its Own Judge
description: >-
  [NeurIPS 2025][LLM Evaluation][benchmark evaluation] This paper systematically critiques the fundamental flaws of current AI benchmark evaluation—data contamination (45%+ overlap in MMLU), selective reporting…
tags:
  - "NeurIPS 2025"
  - "LLM Evaluation"
  - "benchmark evaluation"
  - "data contamination"
  - "PeerBench"
  - "peer review"
  - "reputation system"
date: 2026-05-08
content_hash: a894193d97cf9d21
---

# Benchmarking is Broken — Don't Let AI be its Own Judge

**Conference**: NeurIPS 2025
**arXiv**: [2510.07575](https://arxiv.org/abs/2510.07575)  
**Code**: [https://peerbench.ai](https://peerbench.ai)  
**Area**: AI Safety / Evaluation Methodology
**Keywords**: benchmark evaluation, data contamination, PeerBench, peer review, reputation system

## TL;DR
This paper systematically critiques the fundamental flaws of current AI benchmark evaluation—data contamination (45%+ overlap in MMLU), selective reporting, and lack of proctoring—and proposes PeerBench: drawing on the proctoring paradigm of high-stakes exams (e.g., SAT/GRE), it constructs a next-generation AI evaluation infrastructure via a rolling confidential question bank, peer-review quality control, reputation-weighted scoring, and cryptographic commitment mechanisms.

## Background & Motivation

**Background**: Static benchmarks such as MMLU and SuperGLUE serve as the primary metrics for measuring AI progress. LLM developers compete on these benchmarks and publish leaderboard results.

**Limitations of Prior Work**: (a) **Data contamination**—retrieval audits reveal 45%+ overlap between QA benchmarks and training data, and GPT-4 can infer 57% of masked MMLU answers; (b) **Selective reporting**—curators select favorable task subsets while developers cherry-pick favorable benchmarks; (c) **Benchmark saturation**—SuperGLUE was saturated within months, and performance gains reflect memorization rather than capability; (d) **Lack of proctoring**—no identity verification, unlimited submissions, and unaddressed cultural/demographic biases.

**Key Challenge**: Benchmarks are intended to objectively measure capability, yet the current system contains too many exploitable loopholes—LLM developers have both the incentive and the ability to optimize for benchmark scores rather than genuine capability.

**Goal**: To propose an alternative evaluation architecture that equips benchmark assessment with anti-cheating mechanisms and continuous update capabilities, analogous to SAT/GRE.

**Key Insight**: High-stakes human examinations (bar exams, medical licensing) already have mature mechanisms for question confidentiality, proctoring, and reputation management. AI evaluation should draw on these institutional designs.

**Core Idea**: Replace the current open-benchmark paradigm with an examination-governance paradigm—confidential question bank + peer review + reputation weighting + cryptographic commitments + timed retirement and release.

## Method

### Overall Architecture
**PeerBench Architecture**: Data contributors submit confidential questions → Reviewers conduct peer review for quality → Model developers register inference endpoints → A coordination server manages the active question pool, schedules reviews, updates reputations, and publishes leaderboards. Three leaderboards are maintained: contributor scores, reviewer scores, and model scores.

### Key Designs

1. **Continuous Evaluation Workflow (T1–T6)**:

    - **Function**: Implements lifecycle management for questions.
    - **Mechanism**: T1 submission + hash commitment $h = \text{Com}(T, F)$ → T2 single evaluation pass over all registered models → T3 ≥3 reviewers assess quality $q \in \{-1,0,1,2\}$ → T4 reputation-weighted score computation $w = 0.7 \cdot \text{quality} + 0.3 \cdot \min(2, \rho/100)$ → T5 questions enter/exit the active pool → T6 reputations of all participants updated. Retired questions are fully released to the public.
    - **Design Motivation**: Rolling updates ensure question freshness; public release upon retirement ensures transparency and auditability.

2. **Reputation System (Three-Party Game)**:

    - **Function**: Incentivizes high-quality contributions, honest reviewing, and fair participation.
    - **Mechanism**: Contributor $\text{Score}(c) = \sum_i \text{quality}(T_i) + \text{bonuses}$; Reviewer $\text{Score}(r) = \text{Pearson}(\{q_r\}, \{\bar{q}\})$ (correlation with consensus); Model $\text{Score}(m) = \frac{\sum_i w(T_i) s_i^{(m)}}{\sum_i w(T_i)}$ (quality-weighted average). A slashing mechanism penalizes malicious behavior.
    - **Design Motivation**: The reputation system constitutes the economic foundation of the entire mechanism—dishonest participation degrades reputation and results in exclusion.

3. **Security and Audit Mechanisms**:

    - **Function**: Prevents data leakage, tampering, and collusion.
    - **Mechanism**: Reviewers see only a random image-formatted subset of questions (to prevent copying); all submissions and evaluations are cryptographically signed against tampering; upon retirement, all questions are publicly released for community verification of hash commitments; participants whose reputation falls below a threshold are removed.
    - **Design Motivation**: Current systems have zero security mechanisms—anyone can submit an unlimited number of times without identity verification, and evaluation data is fully public.

### Loss & Training
- This is a framework design paper; no model training is involved.
- A temporal fairness dilemma is discussed: immediate scoring (fast response but cross-period incomparability) vs. synchronized evaluation windows (fair but inflexible) → a hybrid solution is proposed.

## Key Experimental Results

### Comparison of Existing Platforms

| Platform | Dynamic Updates | Contamination Resistance | Quality Control | Transparency |
|----------|----------------|--------------------------|-----------------|--------------|
| MMLU | ✗ | ✗ | ✗ | ✓ |
| SuperGLUE | ✗ | ✗ | ✗ | ✓ |
| LiveBench | ✓ (monthly) | Partial | Opaque | — |
| Chatbot Arena | ✓ | ✓ | Limited (Elo) | — |
| **PeerBench** | **✓** | **✓** | **✓ (peer review)** | **✓** |

### Summary of Contamination Evidence

| Type | Evidence |
|------|----------|
| Retrieval overlap | 45%+ in QA benchmarks |
| Inference capability | GPT-4 recovers 57% of masked MMLU answers |
| Curation bias | Humanity's Last Exam targets only failures of 5 models |
| Saturation | SuperGLUE saturated within months |

### Key Findings
- Data contamination is not a marginal issue—45% overlap suggests that a substantial portion of benchmark performance may reflect memorization rather than reasoning.
- Private benchmarks transfer epistemic authority—from community consensus to curator fiat.
- The lack of standardized evaluation makes cross-model comparisons unreliable, as different evaluations employ different few-shot settings and prompt templates.

## Highlights & Insights
- **The insights from psychometrics for AI evaluation are profound**: humanity has centuries of experience designing cheat-resistant examinations, and AI evaluation should borrow from this tradition rather than reinvent it.
- **The game-theoretic design of the reputation system** exhibits the rigor of mechanism design—incentive compatibility and collusion deterrence.
- **"Don't let AI be its own judge"** cuts to the core—self-evaluation is inherently subject to conflicts of interest.

## Limitations & Future Work
- The temporal fairness dilemma has no perfect solution—models evaluated at different times face different questions.
- A sustained supply of high-quality questions is required (a non-trivial burden), and long-term sustainability remains uncertain.
- The formalization of economic incentives and reputation mechanisms is incomplete and requires rigorous game-theoretic analysis.
- A neutral operating organization (e.g., NIST or MLCommons) is necessary, but such organizations may themselves harbor preferences.
- A trade-off exists between responsiveness and certification-grade rigor—the hybrid solution increases system complexity.
- Designing unified standards for cross-modal evaluation (text/image/audio/code) is challenging.
- Standardizing inference environments for API-only models remains unresolved.

## Related Work & Insights
- **vs. MMLU/SuperGLUE**: Static, one-time benchmarks; PeerBench employs continuous updates.
- **vs. Chatbot Arena**: Crowdsourced comparisons with limited quality control; PeerBench incorporates peer review.
- **vs. LiveBench**: Monthly updates but opaque quality control.
- **vs. Kaggle competition format**: Kaggle uses private test sets but lacks rolling updates and peer review; PeerBench is more complete.
- **Insight**: Any evaluation employing LLM-as-Judge should be alert to self-evaluation bias; cross-validation is essential.
- **vs. SWE-bench/METR**: Capability evaluations with fixed questions; PeerBench's rolling alternation ensures freshness.
- **Analogy to blockchain**: PeerBench's hash commitments, slashing penalties, and decentralized auditing draw on blockchain trust mechanisms.
- **Analogy to academic peer review**: The reviewer reputation system parallels academic refereeing but with a higher degree of automation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Importing psychometric principles into AI evaluation represents a profound paradigm shift.
- Experimental Thoroughness: ⭐⭐⭐ Primarily a design proposal; empirical validation is limited.
- Writing Quality: ⭐⭐⭐⭐⭐ Critique is sharp yet constructive; institutional design is detailed.
- Value: ⭐⭐⭐⭐⭐ May fundamentally reshape AI evaluation paradigms; prototype peerbench.ai is already live.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] AstaBench: Rigorous Benchmarking of AI Agents with a Scientific Research Suite](../../ICLR2026/llm_evaluation/astabench_benchmarking_ai_agents.md)
- [\[ICML 2026\] From Human-Level AI Tales to AI Leveling Human Scales](../../ICML2026/llm_evaluation/from_human-level_ai_tales_to_ai_leveling_human_scales.md)
- [\[NeurIPS 2025\] Beyond the Surface: Enhancing LLM-as-a-Judge Alignment with Human via Internal Representations](beyond_the_surface_enhancing_llm-as-a-judge_alignment_with_human_via_internal_re.md)
- [\[ACL 2026\] IF-RewardBench: Benchmarking Judge Models for Instruction-Following Evaluation](../../ACL2026/llm_evaluation/if-rewardbench_benchmarking_judge_models_for_instruction-following_evaluation.md)
- [\[ACL 2026\] AJ-Bench: Benchmarking Agent-as-a-Judge for Environment-Aware Evaluation](../../ACL2026/llm_evaluation/aj-bench_benchmarking_agent-as-a-judge_for_environment-aware_evaluation.md)

</div>

<!-- RELATED:END -->

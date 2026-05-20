---
title: >-
  [Paper Note] FregeLogic at SemEval 2026 Task 11: A Hybrid Neuro-Symbolic Architecture for Content-Robust Syllogistic Validity Prediction
description: >-
  [ACL 2026][LLM Agent][syllogistic reasoning] FregeLogic is a hybrid neuro-symbolic system that combines a five-member LLM ensemble with a Z3 SMT solver as a tiebreaker…
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "syllogistic reasoning"
  - "belief bias"
  - "neuro-symbolic"
  - "LLM ensemble"
  - "Z3 solver"
date: 2026-05-08
content_hash: 9c8f901ae0fe186f
---

# FregeLogic at SemEval 2026 Task 11: A Hybrid Neuro-Symbolic Architecture for Content-Robust Syllogistic Validity Prediction

**Conference**: ACL 2026
**arXiv**: [2604.18328](https://arxiv.org/abs/2604.18328)  
**Code**: None  
**Area**: LLM Agent / Neuro-Symbolic Reasoning
**Keywords**: syllogistic reasoning, belief bias, neuro-symbolic, LLM ensemble, Z3 solver

## TL;DR

FregeLogic is a hybrid neuro-symbolic system that combines a five-member LLM ensemble with a Z3 SMT solver as a tiebreaker, achieving a 16% reduction in belief bias alongside a 0.9% accuracy improvement on syllogistic validity prediction.

## Background & Motivation

**Background**: Syllogistic reasoning is a fundamental form of deductive inference. SemEval-2026 Task 11 requires systems to judge the logical validity of syllogisms while also measuring the degree to which predictions are influenced by the believability of the content (belief bias). The scoring formula $\text{Score} = \text{Accuracy} / (1 + \ln(1 + \text{CE}))$ simultaneously rewards high accuracy and low belief bias.

**Limitations of Prior Work**: LLMs exhibit human-like belief bias — they tend to judge syllogisms as valid when the content is believable in the real world, and invalid otherwise. Mechanistic analyses suggest that reasoning circuits developed during LLM pretraining are susceptible to contamination by world knowledge.

**Key Challenge**: How can the powerful reasoning capabilities of LLMs be leveraged while overcoming their systematic sensitivity to content believability?

**Goal**: To design a reasoning system that minimizes belief bias while maintaining high accuracy.

**Key Insight**: The degree of disagreement within an LLM ensemble vote is used as a signal for belief-biased cases, which are then delegated to a content-agnostic formal logic solver.

**Core Idea**: Narrow margins in ensemble voting (3–2 splits) disproportionately correspond to belief-biased errors — precisely the cases where a formal verifier can add value.

## Method

### Overall Architecture

The system consists of three components: (1) a five-member LLM ensemble providing high-accuracy predictions; (2) a Z3 formal verification pipeline for structured logical judgments; and (3) a tiebreaker decision module that delegates to Z3 only when the ensemble vote results in a narrow margin (3–2 split).

### Key Designs

1. **Diversified LLM Ensemble**:

    - Function: Provides a high-accuracy baseline by combining uncorrelated errors across members.
    - Mechanism: Three open-source models (Llama 4 Maverick, Llama 4 Scout, Qwen3-32B) × four prompting strategies (zero-shot, few-shot, few-shot CoT, simple CoT) yield 12 combinations in total; the top-5 configurations by combined score are selected per fold.
    - Design Motivation: Architectural diversity (MoE vs. dense, two distinct model families) and prompting diversity maximize error decorrelation among ensemble members.

2. **Z3 Formal Verification Pipeline**:

    - Function: Provides content-neutral logical validity judgments.
    - Mechanism: (a) An LLM with a structured output API extracts the logical structure of the syllogism as JSON; (b) the structure is encoded in first-order logic (adopting Aristotelian existential import); (c) a two-step satisfiability check first verifies premise consistency, then tests whether $P_1 \wedge P_2 \wedge \neg C$ is unsatisfiable.
    - Design Motivation: The Z3 encoding strips all semantic content, making it structurally content-neutral. Using a structured output API reduces extraction failure rates from approximately 22% to near zero.

3. **Selective Tiebreaker Mechanism**:

    - Function: Pinpoints belief-biased ensemble cases and corrects them via formal logic.
    - Mechanism: The vote margin $m = |2 \sum v_i - 5|$ is computed; the Z3 result overrides the ensemble majority only when $m \leq 1$ (a 3–2 split) and Z3 returns a valid judgment.
    - Design Motivation: Empirical observation shows that 3–2 splits disproportionately correspond to belief-biased cases. Extending Z3 authority to higher-consensus cases degrades performance, as Z3 achieves only 48.6% accuracy on valid syllogisms.

### Loss & Training

The system involves no parametric training. Model and prompt selection, as well as fusion strategy selection, are performed via nested 5-fold cross-validation. In each fold, all 12 combinations are evaluated on a 200-sample internal subset, and the top-5 configurations are selected.

## Key Experimental Results

### Main Results (Nested 5-Fold Cross-Validation, N=960)

| Strategy | Accuracy | Belief Bias | Combined Score |
|----------|----------|-------------|----------------|
| Ensemble only | 93.4% | 3.39 | 39.12 |
| **+ Z3 tiebreaker** | **94.3%** | **2.85** | **41.88** |
| Z3 only | 74.7% | 26.28 | 17.39 |
| Confidence + Z3 | 91.7% | 6.15 | 31.77 |

### Subgroup Accuracy Analysis

| Strategy | Valid-Believable | Valid-Unbelievable | Invalid-Believable | Invalid-Unbelievable |
|----------|-----------------|-------------------|-------------------|---------------------|
| Ensemble only | 95.9% | 96.0% | 90.2% | 91.9% |
| + Z3 tiebreaker | 95.6% | 93.8% | **94.5%** | 93.5% |

### Key Findings
- The tiebreaker mechanism yields the largest gains on the Invalid-Believable subgroup (90.2% → 94.5%), which represents the most strongly belief-biased cases.
- 3–2 splits account for only 7.9% of cases, yet across 30 Z3 override decisions, Z3 achieved a net gain of 8 correct flips.
- Z3 exhibits a pronounced "invalidity bias" — 97.6% accuracy on invalid syllogisms versus 52.2% on valid ones — rooted in structural extraction errors.
- All 11 incorrect overrides are unidirectional: Z3 erroneously rejects valid syllogisms, primarily due to extraction errors involving double negations and compound term boundaries.
- The Scout model appears most frequently in the minority coalition (53.9%), suggesting it is more susceptible to belief bias.

## Highlights & Insights
- The system design is elegant: rather than simply replacing LLMs with formal logic, ensemble consensus degree is used as a bias signal to precisely identify cases warranting formal verification.
- The in-depth analysis of Z3's invalidity bias reveals that the bottleneck lies in extraction rather than encoding, and that the directional asymmetry is consistent (valid → invalid).
- The engineering insight that structured output APIs substantially reduce extraction failure rates has practical value.
- The choice of Aristotelian existential import is validated by the annotation of Felapton-type syllogisms in the dataset.

## Limitations & Future Work
- Each sample requires 6 LLM calls plus one Z3 solve, resulting in relatively high inference cost.
- Model and prompt selection requires nested cross-validation, introducing considerable setup complexity.
- The Z3 pipeline relies on an LLM for structure extraction, making extraction errors the primary system bottleneck.
- No comparison is made against larger monolithic models (70B+), leaving open the question of whether architectural diversity outperforms a single large model.
- Adaptive tuning of the tiebreaker threshold $\tau=1$ has not been explored.

## Related Work & Insights
- **vs. pure LLM approaches**: FregeLogic compensates for LLM belief bias via formal verification rather than relying on improved prompting.
- **vs. purely formal methods such as LINC**: FregeLogic applies formal verification only to low-consensus cases, preventing extraction errors from contaminating high-confidence predictions.
- **vs. activation steering approaches (Valentino et al., 2025)**: FregeLogic requires no access to model internals and operates as a black-box solution.
- **Insight**: The use of ensemble disagreement as a bias signal is generalizable to other reasoning tasks.

## Rating
- Novelty: ⭐⭐⭐⭐ The selective hybrid strategy — using ensemble disagreement to trigger formal verification — is cleverly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐ Nested cross-validation is rigorous; subgroup analysis and error attribution are thorough.
- Writing Quality: ⭐⭐⭐⭐⭐ System description is clear and well-analyzed; every design choice is fully justified.
- Value: ⭐⭐⭐ A shared-task system paper; the methodological ideas are inspiring but direct generalizability is limited.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] PerfGuard: A Performance-Aware Agent for Visual Content Generation](../../ICLR2026/llm_agent/radiometrically_consistent_gaussian_surfels_for_inverse_rendering.md)
- [\[ICLR 2026\] Exploratory Memory-Augmented LLM Agent via Hybrid On- and Off-Policy Optimization](../../ICLR2026/llm_agent/exploratory_memory-augmented_llm_agent_via_hybrid_on-_and_off-policy_optimizatio.md)
- [\[NeurIPS 2025\] Generative AI Agents for Controllable and Protected Content Creation](../../NeurIPS2025/llm_agent/generative_ai_agents_for_controllable_and_protected_content_creation.md)
- [\[ICLR 2026\] AgentSynth: Scalable Task Generation for Generalist Computer-Use Agents](../../ICLR2026/llm_agent/agentsynth_scalable_task_generation_for_generalist_computer-use_agents.md)
- [\[AAAI 2026\] Cook and Clean Together: Teaching Embodied Agents for Parallel Task Execution](../../AAAI2026/llm_agent/cook_and_clean_together_teaching_embodied_agents_for_paralle.md)

</div>

<!-- RELATED:END -->

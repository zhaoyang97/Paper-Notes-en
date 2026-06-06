---
title: >-
  [Paper Note] FregeLogic at SemEval 2026 Task 11: A Hybrid Neuro-Symbolic Architecture for Content-Robust Syllogistic Validity Prediction
description: >-
  [ACL 2026][LLM Agent][Syllogistic reasoning] This paper proposes FregeLogic, a hybrid neuro-symbolic system that combines a five-member LLM ensemble with a Z3 SMT solver acting as a tie-breaking judge. It reduces the con…
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "Syllogistic reasoning"
  - "content effect"
  - "neuro-symbolic"
  - "LLM ensemble"
  - "Z3 solver"
date: 2026-05-08
content_hash: 5c67d43e1870964a
---

# FregeLogic at SemEval 2026 Task 11: A Hybrid Neuro-Symbolic Architecture for Content-Robust Syllogistic Validity Prediction

**Conference**: ACL 2026  
**arXiv**: [2604.18328](https://arxiv.org/abs/2604.18328)  
**Code**: None  
**Area**: LLM Agent / Neuro-Symbolic Reasoning  
**Keywords**: Syllogistic reasoning, content effect, neuro-symbolic, LLM ensemble, Z3 solver

## TL;DR

This paper proposes FregeLogic, a hybrid neuro-symbolic system that combines a five-member LLM ensemble with a Z3 SMT solver acting as a tie-breaking judge. It reduces the content effect by 16% while improving accuracy by 0.9% in syllogistic validity judgment.

## Background & Motivation

**Background**: Syllogistic reasoning is a fundamental form of deductive reasoning. SemEval-2026 Task 11 requires systems to judge the logical validity of syllogisms while assessing the degree to which the system is influenced by content believability (the content effect). The scoring formula $\text{Score} = \text{Accuracy} / (1 + \ln(1 + \text{CE}))$ rewards both high accuracy and low content effect simultaneously.

**Limitations of Prior Work**: LLMs exhibit human-like content effects—they tend to judge a syllogism as valid when the content is believable in reality, and vice versa. Mechanistic analysis suggests that the reasoning circuits developed by LLMs during pre-training are easily contaminated by world knowledge.

**Key Challenge**: How can the powerful reasoning capabilities of LLMs be leveraged while overcoming their systematic sensitivity to content believability?

**Goal**: To design a reasoning system capable of minimizing the content effect while maintaining high accuracy.

**Key Insight**: The degree of disagreement in LLM ensemble voting can be used to signal cases of content bias, which are then delegated to a content-agnostic formal logic solver.

**Core Idea**: Narrow voting margins (3-2 splits) in an ensemble disproportionately correspond to content-biased errors—precisely the cases where a formal verifier can add value.

## Method

### Overall Architecture

The system consists of three components: (1) A five-member LLM ensemble providing high-accuracy predictions; (2) A Z3 formal verification pipeline for structured logical judgment; (3) A selective tie-breaking module that delegates decisions to Z3 only when the ensemble vote results in a narrow split (3-2).

### Key Designs

1.  **Diversified LLM Ensemble**:
    - **Function**: Provides a high-accuracy baseline through the combination of uncorrelated errors.
    - **Mechanism**: Utilizes three open-source models (Llama 4 Maverick, Llama 4 Scout, Qwen3-32B) across four prompting strategies (Zero-shot, Few-shot, Few-shot CoT, Simple CoT) for a total of 12 combinations. The top-5 configurations with the highest combined scores are selected for each fold.
    - **Design Motivation**: Architectural diversity (MoE vs. Dense, two different model families) and prompting diversity maximize error uncorrelation among ensemble members.

2.  **Z3 Formal Verification Pipeline**:
    - **Function**: Provides content-neutral logical validity judgments.
    - **Mechanism**: (a) Uses LLM + Structured Output API to extract the logical structure of the syllogism into JSON; (b) Encodes the structure into First-Order Logic (adopting Aristotelian existential import); (c) Performs a two-step satisfiability check—first verifying premise consistency, then checking if $P_1 \wedge P_2 \wedge \neg C$ is unsatisfiable.
    - **Design Motivation**: Z3 encoding strips away all semantic content and is content-neutral by construction. The Structured Output API reduced extraction failure rates from approximately 22% to near zero.

3.  **Selective Tie-breaking Mechanism**:
    - **Function**: Precisely identifies biased ensemble cases and corrects them using formal logic.
    - **Mechanism**: Calculates the voting margin $m = |2 \sum v_i - 5|$; the Z3 result replaces the ensemble majority vote only when $m \leq 1$ (a 3-2 split) and Z3 returns a valid judgment.
    - **Design Motivation**: Empirical observations show that 3-2 splits disproportionately correspond to content-biased cases. Expanding Z3's authority to cases with higher consensus would decrease performance because Z3 has lower accuracy on valid syllogisms (48.6%).

### Loss & Training

The system utilizes parameter-free training. Model and prompt selection, as well as fusion strategy selection, are completed via nested 5-fold cross-validation. For each fold, all 12 combinations are evaluated on an internal subset of 200 samples to select the top-5 configurations.

## Key Experimental Results

### Main Results (Nested 5-fold CV, N=960)

| Strategy | Accuracy | Content Effect | Combined Score |
|----------|----------|----------------|----------------|
| Pure Ensemble | 93.4% | 3.39 | 39.12 |
| **+ Z3 Tie-break** | **94.3%** | **2.85** | **41.88** |
| Z3 Only | 74.7% | 26.28 | 17.39 |
| Confidence + Z3 | 91.7% | 6.15 | 31.77 |

### Subgroup Accuracy Analysis

| Strategy | Valid-Believable | Valid-Unbelievable | Invalid-Believable | Invalid-Unbelievable |
|----------|------------------|--------------------|--------------------|----------------------|
| Pure Ensemble | 95.9% | 96.0% | 90.2% | 91.9% |
| + Z3 Tie-break | 95.6% | 93.8% | **94.5%** | 93.5% |

### Key Findings
- The tie-breaking mechanism primarily gains performance in the "Invalid-Believable" subgroup (90.2% → 94.5%), which contains the cases with the strongest content bias.
- 3-2 splits account for only 7.9% of cases, but Z3 achieved a net gain of 8 correct decisions out of 30 overrides.
- Z3 exhibits a significant "invalid bias"—97.6% accuracy on invalid syllogisms but only 52.2% on valid ones, rooted in structural extraction errors.
- All 11 erroneous flips occurred in the same direction: Z3 incorrectly rejected valid syllogisms, primarily due to extraction errors involving double negatives or complex term boundaries.
- The Scout model appeared most frequently in minority alliances (53.9%), indicating it is more susceptible to content bias.

## Highlights & Insights
- Sophisticated system design: Rather than simply replacing the LLM with formal logic, the system uses ensemble consensus as a bias signal to precisely target cases requiring formal verification.
- In-depth analysis of Z3's invalid bias reveals that the bottleneck lies in extraction rather than encoding, showing directional asymmetry (Valid → Invalid).
- Engineering insights regarding Structured Output APIs significantly lowering extraction failure rates provide practical value.
- The choice of Aristotelian existential import was validated by the labeling of Felapton-type syllogisms in the dataset.

## Limitations & Future Work
- Each sample requires 6 LLM calls + 1 Z3 solver execution, resulting in high inference costs.
- Model and prompt selection require nested cross-validation, leading to high setup complexity.
- The Z3 pipeline relies on the LLM for structure extraction; extraction errors remain the primary system bottleneck.
- No comparison was made with larger monolithic models (70B+); whether architectural diversity is superior to a single large model remains an open question.
- No exploration was conducted regarding adaptive adjustment of the tie-breaking threshold $\tau=1$.

## Related Work & Insights
- **vs. Pure LLM Methods**: FregeLogic compensates for LLM content bias through formal verification rather than relying solely on better prompting.
- **vs. Pure Formal Methods (e.g., LINC)**: FregeLogic uses formal verification only in low-consensus cases, preventing extraction errors from contaminating high-confidence cases.
- **vs. Activation Guidance (Valentino et al., 2025)**: FregeLogic is a black-box solution that does not require access to internal model states.
- **Insight**: The idea of using ensemble disagreement as a bias signal could be generalized to other reasoning tasks.

## Rating
- Novelty: ⭐⭐⭐⭐ The selective hybrid strategy using ensemble disagreement for formal verification intervention is cleverly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐ Rigorous nested cross-validation with deep subgroup and error attribution analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Clearly described system with thorough analysis and well-justified design choices.
- Value: ⭐⭐⭐ As a shared-task system paper, the methodological approach is inspiring but has limited direct generalizability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Lifting Traces to Logic: Programmatic Skill Induction with Neuro-Symbolic Learning for Long-Horizon Agentic Tasks](../../ICML2026/llm_agent/lifting_traces_to_logic_programmatic_skill_induction_with_neuro-symbolic_learnin.md)
- [\[ACL 2026\] MAGMA: A Multi-Graph based Agentic Memory Architecture for AI Agents](magma_a_multi-graph_based_agentic_memory_architecture_for_ai_agents.md)
- [\[ACL 2026\] Robust Tool Use via Fission-GRPO: Learning to Recover from Execution Errors](robust_tool_use_via_fission-grpo_learning_to_recover_from_execution_errors.md)
- [\[ACL 2026\] Supplement Generation Training for Enhancing Agentic Task Performance](supplement_generation_training_for_enhancing_agentic_task_performance.md)
- [\[ACL 2026\] IntrAgent: An LLM Agent for Content-Grounded Information Retrieval through Literature Review](intragent_an_llm_agent_for_content-grounded_information_retrieval_through_litera.md)

</div>

<!-- RELATED:END -->

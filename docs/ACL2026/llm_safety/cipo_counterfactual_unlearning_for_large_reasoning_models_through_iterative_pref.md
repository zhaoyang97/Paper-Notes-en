---
title: >-
  [Paper Note] CiPO: Counterfactual Unlearning for Large Reasoning Models through Iterative Preference Optimization
description: >-
  [ACL 2026][LLM Safety][Reasoning Model Unlearning] Addressing the challenge of unlearning in Large Reasoning Models (LRMs)—which require removing sensitive knowledge from both the Chain-of-Thought (CoT) and the final ans…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Reasoning Model Unlearning"
  - "Counterfactual Reasoning"
  - "Preference Optimization"
  - "Chain-of-Thought"
  - "Privacy Protection"
date: 2026-05-08
content_hash: 2be765deb2fd837c
---

# CiPO: Counterfactual Unlearning for Large Reasoning Models through Iterative Preference Optimization

**Conference**: ACL 2026  
**arXiv**: [2604.15847](https://arxiv.org/abs/2604.15847)  
**Code**: [https://github.com/TerryLee77/CiPO](https://github.com/TerryLee77/CiPO)  
**Area**: LLM Safety / Reasoning Model Unlearning  
**Keywords**: Reasoning Model Unlearning, Counterfactual Reasoning, Preference Optimization, Chain-of-Thought, Privacy Protection

## TL;DR

Addressing the challenge of unlearning in Large Reasoning Models (LRMs)—which require removing sensitive knowledge from both the Chain-of-Thought (CoT) and the final answer—the CiPO framework is proposed. By instructing the model to generate logically valid counterfactual reasoning trajectories and guiding model preferences toward these counterfactual paths through iterative preference optimization, effective unlearning is achieved while maintaining reasoning capabilities.

## Background & Motivation

**Background**: LRMs (e.g., DeepSeek-R1, o1) solve complex problems through long Chain-of-Thought reasoning. However, CoT itself becomes a carrier for data leakage, as sensitive information cited during the reasoning process is explicitly recorded and exposed.

**Limitations of Prior Work**: (1) Representation perturbation methods (e.g., R2MU) map the hidden representations of the forget set to random vectors; while they can erase target trajectories, excessive suppression destroys CoT interpretability and reasoning ability, resulting in incoherent output. (2) Refusal-based methods (e.g., ReasonedIDK) train models to generate "I don't know" style responses, which introduces a large distribution shift leading to unstable optimization, and the consistent refusal pattern itself becomes a channel for information leakage (attackers can infer what was forgotten). (3) Traditional LLM unlearning methods (GA/NPO) do not handle multi-step reasoning structures and cannot resolve information leakage within the CoT.

**Key Challenge**: Existing methods choose between "erasing" or "avoiding"—either forcibly destroying the reasoning chain (damaging capability) or training the model to refuse (introducing new risks). Neither provides a "constructive" alternative.

**Goal**: Redefine unlearning as a "constructive intervention" in CoT reasoning—replacing the original reasoning chain with safe, task-consistent counterfactual trajectories rather than destruction or refusal.

**Key Insight**: Model LRM unlearning as an intervention from a causal perspective—severing the causal influence of the forget set on the CoT and the answer, provided by alternative paths through counterfactual reasoning.

**Core Idea**: Given a target to forget, instruct the LRM to generate logically valid counterfactual reasoning trajectories (consistent CoT logic but a conclusion different from the original). These are used as positive samples for preference optimization, while the current model's output containing sensitive information serves as negative samples. Preference data is updated iteratively to track the evolution of the model distribution.

## Method

### Overall Architecture

CiPO consists of two core components: (1) A counterfactual generator—instructing the model to construct logically valid but conclusion-divergent counterfactual reasoning trajectories for the forget target; (2) Iterative preference optimization—sampling dynamic preference pairs from the current model in each round (counterfactual trajectories as chosen, current model output as rejected) and optimizing with a DPO-style objective. Multi-round iteration ensures unlearning remains aligned with the shifting model distribution.

### Key Designs

1.  **Counterfactual Trajectory Generation**:
    - **Function**: Provides safe, logically valid alternative reasoning paths.
    - **Mechanism**: Given a forget target $(q, c, a)$, the LRM is instructed to generate a counterfactual trajectory $(c^*, a^*)$. Requirements: the reasoning process $c^*$ must be logically coherent and structurally complete (maintaining the `<think>...</think>` format), but the final conclusion $a^*$ must differ from the original answer $a$. The counterfactual is not a simple negation or random replacement, but a "reasonable yet incorrect" reasoning chain—mimicking how someone unaware of the correct answer would reason.
    - **Design Motivation**: Refusal methods ("I don't know") introduce large distribution shifts and instability. Counterfactuals maintain the naturalness of the reasoning structure—the model continues "normal reasoning," only with a different conclusion.

2.  **Iterative Online Preference Optimization**:
    - **Function**: Keeps the unlearning signal aligned with the model distribution.
    - **Mechanism**: In each iteration, outputs for forget prompts are sampled from the current model $\pi_t$ as rejected samples, while counterfactual trajectories serve as chosen samples to construct dynamic preference pairs. Using a DPO objective optimizes the model to prefer counterfactual paths. Iterative updates ensure preference data reflects the model's real-time distribution, avoiding distribution mismatch issues found in fixed offline data.
    - **Design Motivation**: Standard DPO uses fixed, pre-collected preference pairs, which is off-policy relative to the current model. As the model changes during unlearning, fixed data deviates from its distribution. Iterative online updates resolve this.

3.  **Theoretical Support via Causal Graph Modeling**:
    - **Function**: Provides a formal definition for the unlearning objective.
    - **Mechanism**: A causal graph $Q \to C \to A$ is constructed, where the forget set $F$ influences the output via $F \to C$ and $F \to A$. The unlearning goal is defined as an intervention $\text{do}(F \to \{C, A\})$—severing the causal influence of $F$ on the CoT and the answer. Counterfactual trajectories are the concrete implementation of this intervention—providing alternative paths when $F$ does not influence reasoning.
    - **Design Motivation**: The causal framework provides a theoretical basis for why counterfactuals are needed instead of simple erasure.

### Loss & Training

DPO-style preference optimization loss with iterative updates to preference data. Evaluated on the R-TOFU benchmark (an extension for LRM unlearning) and real-world benchmarks. Based on LRMs such as DeepSeek-R1-Distill.

## Key Experimental Results

### Main Results

| Method | CoT Unlearning Effect | Answer Unlearning Effect | Reasoning Preservation |
| :--- | :--- | :--- | :--- |
| R2MU | Moderate | Moderate | Poor (Reasoning degradation) |
| ReasonedIDK | Poor (CoT leakage) | Good | Moderate (Over-refusal) |
| NPO/GA | Poor | Moderate | Poor |
| **Ours (CiPO)** | **Good** | **Good** | **Good** |

### Ablation Study

| Configuration | Effect | Explanation |
| :--- | :--- | :--- |
| Single-round DPO (No iteration) | Moderate | Distribution mismatch |
| Multi-round Iterative DPO | Optimal | Continuous alignment |
| No Counterfactual (Direct refusal) | Poor | Large distribution shift |
| Random Replacement (Non-counterfactual) | Poor | Incoherent |

### Key Findings

- CiPO is the only method capable of effectively removing sensitive information from both the CoT and the final answer simultaneously.
- While R2MU erases information, it severely damages reasoning capabilities (producing gibberish output).
- The consistent refusal patterns in ReasonedIDK can be exploited by Membership Inference Attacks (MIA).
- Iterative updates perform significantly better than single-round fixed-data training.
- CiPO maintains performance levels close to the original model on the retain set and reasoning benchmarks.

## Highlights & Insights

- **Paradigm shift of "Constructive Substitution" vs. "Destructive Erasure"**: Instead of teaching the model "not to think" or "refuse to answer," it teaches the model to "think in another way." This preserves the naturalness of the reasoning structure and avoids distribution shifts.
- **Causal theoretical support for counterfactuals as unlearning targets**: It proves the rationality of counterfactual substitution from the perspective of the do-operation in a causal graph.
- **Necessity of iterative online updates**: The model distribution continuously changes during unlearning; preference optimization using fixed data gradually loses efficacy. This insight is valuable for all unlearning methods using DPO.

## Limitations & Future Work

- The quality of counterfactual trajectory generation depends on the model's own capabilities—weak models may generate low-quality counterfactuals.
- The computational cost of the iterative process is higher than single-round methods.
- Counterfactual reasoning might preserve certain reasoning patterns (as opposed to the information itself), which high-level attacks might still use to infer forgotten knowledge.
- Systematically verified primarily on R-TOFU; evaluation across more real-world privacy scenarios needs expansion.

## Related Work & Insights

- **vs. R2MU (Representation Perturbation)**: R2MU "destroys" reasoning by mapping representations to random vectors, while CiPO "substitutes" reasoning with counterfactuals. The former harms capability, while the latter preserves it.
- **vs. ReasonedIDK (Refusal-based)**: Refusal introduces large distribution shifts and risks from membership inference attacks. Counterfactuals maintain a natural reasoning structure and do not expose what has been forgotten.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The counterfactual unlearning idea is original and theoretically deep; causal graph modeling provides a solid foundation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple baseline comparisons + ablations + CoT-level evaluation, though benchmarks are limited.
- Writing Quality: ⭐⭐⭐⭐⭐ Thorough problem analysis; the arguments regarding the limitations of existing methods are persuasive.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Reasoning Hijacking: The Fragility of Reasoning Alignment in Large Language Models](reasoning_hijacking_the_fragility_of_reasoning_alignment_in_large_language_model.md)
- [\[ACL 2026\] AutoRAN: Automated Hijacking of Safety Reasoning in Large Reasoning Models](autoran_automated_hijacking_of_safety_reasoning_in_large_reasoning_models.md)
- [\[ICML 2026\] COFT: Counterfactual-Conformal Decoding for Fair Chain-of-Thought Reasoning in Large Language Models](../../ICML2026/llm_safety/coft_counterfactual-conformal_decoding_for_fair_chain-of-thought_reasoning_in_la.md)
- [\[ACL 2026\] CarO: Chain-of-Analogy Reasoning Optimization for Robust Content Moderation](caro_chain-of-analogy_reasoning_optimization_for_robust_content_moderation.md)
- [\[ACL 2026\] How Should We Enhance the Safety of Large Reasoning Models: An Empirical Study](how_should_we_enhance_the_safety_of_large_reasoning_models_an_empirical_study.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Old Habits Die Hard: How Conversational History Geometrically Traps LLMs
description: >-
  [ICML 2026][LLM Safety][Conversational History] The History-Echoes framework analyzes the carryover effects of LLM conversational history through two lenses: "Markov chain state consistency" and "latent space geometric a…
tags:
  - "ICML 2026"
  - "LLM Safety"
  - "Conversational History"
  - "Behavioral Persistence"
  - "Markov Chains"
  - "Geometric Trap"
  - "Refusal / Sycophancy / Hallucination"
date: 2026-05-08
content_hash: 3979104db4f65fd7
---

# Old Habits Die Hard: How Conversational History Geometrically Traps LLMs

**Conference**: ICML 2026  
**arXiv**: [2603.03308](https://arxiv.org/abs/2603.03308)  
**Code**: https://github.com/technion-cs-nlp/OldHabitsDieHard  
**Area**: LLM Safety / Mechanistic Interpretability / Conversational Behavior Analysis  
**Keywords**: Conversational History, Behavioral Persistence, Markov Chains, Geometric Trap, Refusal / Sycophancy / Hallucination

## TL;DR
The History-Echoes framework analyzes the carryover effects of LLM conversational history through two lenses: "Markov chain state consistency" and "latent space geometric angles." It identifies a Spearman correlation of 0.78 between them. Once a specific behavior (hallucination, sycophancy, or refusal) emerges, the model becomes confined within the corresponding region of the latent space, making it difficult to escape; the "refusal" trap is the strongest, while "hallucination" is the weakest, and the trap dissolves when topic coherence is disrupted.

## Background & Motivation

**Background**: LLMs exhibit various state-dependent behaviors—both undesirable (hallucination, sycophancy) and desirable (refusal). Prior work has documented these phenomena, but how they **persist across multi-turn dialogues and how they are represented internally** lacks a unified framework. Existing studies on safety trajectories or generation difficulty focus on isolated phenomena without linking "persistence probability" to "internal geometry."

**Limitations of Prior Work**: Analyzing strictly from a black-box perspective (output layer) or a white-box perspective (hidden states) is insufficient. Black-box methods fail to reveal the underlying mechanism (why persistence occurs), while white-box methods lack behavioral validation (whether a geometric pattern actually corresponds to external behavior).

**Key Challenge**: To explain why "a model that has refused before is more likely to refuse again," it is necessary to concurrently prove that "behavior persists at the output level" and "internal geometry has a structural correspondence," while demonstrating a correlation between the two. Otherwise, the findings may be statistical illusions or cherry-picked geometry.

**Goal**: (1) Quantify behavioral carryover; (2) Reveal the mechanism through latent space geometry; (3) Prove a strong correlation between these two perspectives to provide dual evidence for "behavioral persistence $\approx$ geometric trap."

**Key Insight**: Conversational states are binarized (behavior present/absent) and modeled using first-order Markov chains. Simultaneously, in the latent space, orthogonal bases for $\mathcal{H}_{\phi^+}$ and $\mathcal{H}_{\phi^-}$ are constructed via Gram-Schmidt to measure angular separation between activation sets. It is predicted that these two metrics (black-box persistence rate vs. white-box geometric angle) are positively correlated.

**Core Idea**: Behavioral persistence is not an isolated output-layer phenomenon; rather, it occurs because two phase regions in the latent space are separated by large angles. Transitioning between states requires significant rotation, and this rotation is often incomplete, causing the model to be geometrically trapped in its original state.

## Method

### Overall Architecture

The History-Echoes Dual-Perspective:
1.  **Probabilistic Perspective** (Black-box): Models the dialogue state sequence as a 2-state Markov chain $\mathbf{T}$. Persistence is quantified using $\text{Tr}(\mathbf{T}) = P(s_{\phi^+}|s_{\phi^+}) + P(s_{\phi^-}|s_{\phi^-})$. When no carryover exists, $\text{Tr}=1$; $\text{Tr}>1$ indicates a preference for state self-loops.
2.  **Geometric Perspective** (White-box): For each phenomenon, sets of hidden states $\mathcal{H}_{\phi^\pm}$ for states $\phi^+$ and $\phi^-$ are collected. Orthogonal bases for two-dimensional subspaces are constructed via Gram-Schmidt. Two signatures are used:
    -   **$\theta_{\text{ref}}$**: The angular separation between the two state subspaces (larger indicates greater distance).
    -   **Rotation Incompleteness**: The actual rotation angle during cross-state transitions (less completeness suggests a stronger pull from the original state).

Data Construction: For each dataset (TriviaQA, NaturalQA, SORRY-Bench, Do-Not-Answer, SycophancyEval), QA pairs are embedded using Qwen3-Embedding and sorted by nearest-neighbor to create $D_{\text{consistent}}$ (coherent topics) or randomized to create $D_{\text{inconsistent}}$. 100 dialogues of 20 turns each are sampled.

### Key Designs

1.  **2-State Markov Chain + Trace to Measure Persistence**:
    -   **Function**: Quantifies conversational persistence effects for any phenomenon via a black-box approach.
    -   **Mechanism**: Classifies whether phenomenon $\phi$ occurs in each turn (string matching + manual verification showing a 6.5% error rate). Estimates the transition matrix $T_{ij} = P(s_j|s_i)$. $\text{Tr}(\mathbf{T}) = 1 + \lambda_2$ (where $\lambda_2$ is the second eigenvalue); a larger $\lambda_2$ implies a longer mixing time (stronger persistence).
    -   **Design Motivation**: The trace is an intuitive scalar summary that does not require internal model access, making it suitable for closed-source models. It extends directly to higher-order Markov chains.

2.  **Gram-Schmidt Orthogonal Bases + $\theta_{\text{ref}}$ Geometric Angle**:
    -   **Function**: Measures the distance between two phase states in latent space.
    -   **Mechanism**: Obtains several $\phi^+$ and $\phi^-$ hidden states per dataset and applies Gram-Schmidt to derive orthogonal bases for their respective 2D subspaces. $\theta_{\text{ref}}$ is defined as the principal angle between these subspaces. "Rotation incompleteness" is defined as the difference between the actual latent rotation and $\theta_{\text{ref}}$ during state transitions.
    -   **Design Motivation**: Individual hidden state directions are noisy; 2D subspaces are more robust. Orthogonal bases provide a clear definition of "angle." Large $\theta_{\text{ref}}$ combined with incomplete rotation implies the model is stuck between states.

3.  **Cross-Perspective Correlation Validation**:
    -   **Function**: Correlates black-box and white-box metrics across multiple models and datasets to prove they reflect the same underlying mechanism.
    -   **Mechanism**: Calculates $\text{Tr}(\mathbf{T})$ and $\theta_{\text{ref}}$ concurrently across 18 combinations (3 models × 6 datasets) using Spearman correlation.
    -   **Design Motivation**: Correlation provides dual evidence, ruling out trace values as statistical noise and $\theta_{\text{ref}}$ values as geometric coincidences.

## Key Experimental Results

### Behavioral Persistence (trace, average across three models)

| Phenomenon | NaturalQA | TriviaQA | Sorry | DoNotAns | S-pos | S-neg | Mean |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Tr(T) | 1.13 | 1.12 | **1.57** | **1.59** | 1.33 | 1.14 | 1.31 |

All phenomena show $\text{Tr} > 1$; refusal datasets show the highest trace ($\approx 1.6$), indicating the strongest carryover.

### Geometric Angular Separation $\theta_{\text{ref}}$ (Degrees)

| Model | NaturalQA | TriviaQA | Sorry | DoNotAns | S-pos | S-neg |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| LLaMA-3.1-8B | 11.3 | 13.1 | **66.5** | **54.3** | 14.6 | 28.2 |
| Qwen-8B | 11.7 | 6.4 | **46.4** | **38.6** | 22.5 | 22.6 |
| GPT-OSS-20B | 9.6 | 13.9 | **42.7** | **34.0** | 27.8 | 23.6 |

$\theta_{\text{ref}}$ for refusal datasets ($30-66^{\circ}$) is significantly larger than for hallucinations ($6-14^{\circ}$), showing refusal states are geometrically distinct.

### Cross-Perspective Correlation

Across 18 pairs of ($\text{trace}, \theta_{\text{ref}}$) data points, the Spearman correlation is **0.78**, confirming a strong positive correlation where high trace values correspond to large geometric angles.

### Topic Coherence Dissolving the Trap

| Dataset | $D_{\text{consistent}}$ trace | $D_{\text{inconsistent}}$ trace | Difference |
| :--- | :--- | :--- | :--- |
| Sorry | 1.57 | 1.18 | −0.39 |
| Do-not-answer | 1.59 | 1.20 | −0.39 |
| S-neg | 1.14 | 1.05 | −0.09 |

Randomizing topics significantly reduces the trace and $\theta_{\text{ref}}$, verifying that the "geometric trap" relies on topic coherence. This aligns with adversarial jailbreak strategies that inject irrelevant tokens to break context.

### Closed-Source Model Validation

Running black-box trace analysis on GPT-5 and Claude-Opus-4.5 shows patterns consistent with open-source models. This suggests the trace is a universal diagnostic metric for inferring internal carryover in closed-source models.

### Key Findings
-   **Hierarchical Carryover Strength**: The order is refusal > sycophancy > hallucination, and this hierarchy is consistent across both trace and $\theta_{\text{ref}}$ perspectives.
-   **Refusal Strength Stems from "Single Directionality"**: This aligns with findings by Arditi et al. (2024) that refusal is controlled by a single representation direction. Clearly defined phenomena are more geometrically separated, resulting in "deeper" traps.
-   **Hallucination is Weakest**: This likely occurs because hallucination is a broad collection of failure modes (factual errors, fabrications, inconsistencies) without a unified latent subspace.
-   **Incoherent Dialogue Breaks Traps**: Practically, this suggests that "switching topics" is a simple method to de-trap a model.

## Highlights & Insights
-   **Strong Correlation Between Black-box and White-box Perspectives**: For the first time, behavioral statistics are systematically linked to latent space geometry, providing dual evidence that "behavioral persistence = geometric trap." This "dual-end validation" methodology can be transferred to other LLM behavioral studies.
-   **Unified Treatment of Three Phenomena**: By analyzing hallucination, sycophancy, and refusal (one failure and two conservative behaviors) under one framework, the study finds that carryover strength corresponds to "phenomenon clarity." This suggests that "clearly identifiable = geometrically separated = hard to escape."
-   **Diagnosability for Closed-Source Models**: Since the trace does not require internal access, it offers an indirect diagnostic tool for behavioral persistence in models like GPT-5 or Claude, which has practical value for LLM governance.
-   **Geometric Explanation for Jailbreaks**: Jailbreaks often work by injecting irrelevant tokens to break continuity. This study finds that such actions reduce carryover, providing a potential geometric mechanism for why jailbreaks are effective.

## Limitations & Future Work
-   Phenomenon detection relies on string matching (6.5% error rate); finer granularity for hallucinations (factual vs. logic vs. fabrication) might be needed to avoid diluting signals.
-   The first-order Markov assumption may be oversimplified, potentially under-modeling long-range dependencies (though higher-order validation is provided in the appendix).
-   Model sizes are relatively small (4–20B); larger models might exhibit different geometric trap patterns.
-   Geometric angles ($\theta_{\text{ref}}$) are aggregated across layers; differences across individual layers were not extensively explored.
-   The study focuses on the "once-trapped-stay-trapped" observation rather than active "de-trapping" strategies beyond topic randomization.

## Related Work & Insights
-   **vs. Arditi et al. 2024 (Refusal Directions)**: This work generalizes the finding that refusal not only has a single direction but also strong carryover and a geometric mechanism.
-   **vs. Carryover Effects Studies (Simhi 2024, Zhang 2024)**: Previous studies focused on the output layer; this work adds the white-box perspective and proves the correlation.
-   **vs. Jailbreak via Adversarial Tokens (Zou 2023)**: This work provides a latent space geometric explanation: adversarial tokens break topic coherence, thus dissolving the geometric trap.
-   **Insights**: The "behavioral persistence + geometric trap" framework can be extended to other state-dependent phenomena such as format-locking in in-context learning, persona drift, or code style locking. It can also inform the design of active "de-trapping" mechanisms, such as periodic topic refreshes as prompt-side safety patches.

## Rating
-   Novelty: ⭐⭐⭐⭐ The unified dual-perspective framework is new, though Markov chains and geometric separation are independently known.
-   Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across 3 models, 6 datasets, 3 phenomena, consistent/inconsistent controls, closed-source validation, and higher-order Markov analysis.
-   Writing Quality: ⭐⭐⭐⭐ Concepts are introduced clearly; Figure 1 is intuitive. The geometric derivation could be more detailed.
-   Value: ⭐⭐⭐⭐ Provides practical insights for multi-turn safety, jailbreak mechanisms, and dialogue deployment. Serves as a template for combining mechanistic interpretability with behavioral analysis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Hard to Read, Easy to Jailbreak: How Visual Degradation Bypasses MLLM Safety Alignment](../../ACL2026/llm_safety/hard_to_read_easy_to_jailbreak_how_visual_degradation_bypasses_mllm_safety_align.md)
- [\[ICML 2026\] Deep Sequence Models Tend to Memorize Geometrically; It Is Unclear Why](deep_sequence_models_tend_to_memorize_geometrically_it_is_unclear_why.md)
- [\[ICLR 2026\] Revisiting the Past: Data Unlearning with Model State History](../../ICLR2026/llm_safety/revisiting_the_past_data_unlearning_with_model_state_history.md)
- [\[AAAI 2026\] An LLM-Based Simulation Framework for Embodied Conversational Agents in Psychological Counseling](../../AAAI2026/llm_safety/an_llm-based_simulation_framework_for_embodied_conversationa.md)
- [\[ICML 2026\] Gradient Transformer: Learning to Generate Updates for LLMs](gradient_transformer_learning_to_generate_updates_for_llms.md)

</div>

<!-- RELATED:END -->

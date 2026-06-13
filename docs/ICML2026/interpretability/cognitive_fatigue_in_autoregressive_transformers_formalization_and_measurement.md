---
title: >-
  [Paper Note] Cognitive Fatigue in Autoregressive Transformers: Formalization and Measurement
description: >-
  [ICML2026][Interpretability][Cognitive fatigue] This paper formalizes the degradation of autoregressive language models during long-sequence generation as "cognitive fatigue." It proposes the Fatigue Index (FI)…
tags:
  - "ICML2026"
  - "Interpretability"
  - "Cognitive fatigue"
  - "Autoregressive Transformer"
  - "Runtime monitoring"
  - "Fatigue Index"
  - "Long-sequence degradation"
date: 2026-05-08
content_hash: 35f1ea60a30dd6b2
---

# Cognitive Fatigue in Autoregressive Transformers: Formalization and Measurement

**Conference**: ICML2026  
**arXiv**: [2605.30981](https://arxiv.org/abs/2605.30981)  
**Code**: None  
**Area**: Interpretability/Model Diagnosis  
**Keywords**: Cognitive fatigue, Autoregressive Transformer, Runtime monitoring, Fatigue Index, Long-sequence degradation  

## TL;DR
This paper formalizes the degradation of autoregressive language models during long-sequence generation as "cognitive fatigue." It proposes the Fatigue Index (FI), a lightweight, model-agnostic online diagnostic metric that aggregates signals from prompt attention decay, representation drift, and entropy misalignment. The predictive power of FI for degradation (AUROC=0.976) is validated across 9 models, revealing non-monotonic scaling behaviors.

## Background & Motivation

**Background**: Large language models perform excellently with short prompts but undergo systematic degradation in long-sequence generation scenarios (e.g., multi-step reasoning, tool calling, multi-turn dialogues)—manifesting as repetitive text, loss of instruction-following capability, and unstable entropy.

**Limitations of Prior Work**: Current mitigation strategies primarily operate during training (e.g., unlikelihood training) or offline evaluation, lacking online diagnostic signals for real-time detection during inference. Practitioners often discover unreliable outputs only after generation is complete, precluding timely intervention.

**Key Challenge**: Autoregressive decoding in Transformers inherently faces structural pressures such as attention dilution, residual accumulation, and overfitting. These pressures compound as sequences grow, yet models lack an internal "reliability dashboard" to reflect the health of the current generation.

**Goal**: (1) Formalize long-sequence degradation as the measurable concept of "cognitive fatigue"; (2) Design an online metric, FI, that satisfies axiomatic constraints; (3) Validate the predictive power and stability of FI across multiple models and tasks.

**Key Insight**: The authors leverage three observable internal signals from Transformers—attention distribution, latent state trajectories, and output entropy—each corresponding to a specific degradation mode. This approach requires no weight modification or retraining.

**Core Idea**: Normalize and linearly aggregate three orthogonal signals (attention decay, representation drift, and entropy misalignment) into a bounded Fatigue Index, transforming long-sequence degradation from "post-hoc observation" into "real-time monitoring."

## Method

### Overall Architecture
The input prompt and context are processed by a decoder-only model for autoregressive generation. At each decoding step $t$, probes extract three lightweight signals: (1) average attention weights of the current token toward the prompt region; (2) the Euclidean distance between the current latent state and the state at the end of the prompt; (3) the Shannon entropy of the softmax distribution for the next token. These three signals are normalized and linearly aggregated into the Fatigue Index (FI), serving as a risk score for degradation at that step. FI satisfies five axioms ensuring interpretability and stability, utilizing a hysteresis mechanism for online alerting to avoid false triggers.

### Key Designs

1.  **Three-Signal Extraction and Normalization**:
    - **Function**: Maps heterogeneous raw signals to a unified penalty scale within $[0,1]$.
    - **Mechanism**: **Prompt Attention** $A_t$ calculates the mean weight of all heads in the final layer for the prompt slice, normalized as $\phi_A(A_t) = 1 - \text{clip}(A_t, 0, 1)$; lower attention leads to higher penalties. **Output Entropy** $E_t$ defines a "health band" $[H_\ell, H_u]$, where penalties are zero inside the band and linear outside (penalizing over-confidence/repetition or excessive uncertainty). **Embedding Drift** $D_t = \|h_t - h_0\|_2$ is divided by a fixed upper bound $\kappa$ and clipped to $[0,1]$.
    - **Design Motivation**: These signals respectively capture instruction-following decline, calibration errors, and internal representation drift. Normalization ensures comparability and additivity.

2.  **Axiomatic Fatigue Index Aggregation**:
    - **Function**: Synthesizes normalized signals into a single interpretable risk score.
    - **Mechanism**: $FI_t = w_A \phi_A(A_t) + w_E \phi_E(E_t) + w_D \phi_D(D_t)$, with weights $w_A=0.40, w_E=0.35, w_D=0.25$. This satisfies five axioms: Monotonicity (FI must increase if signals worsen), Scale Invariance (order-preserving transformations do not change ranking), Boundedness ($FI \in [0,1]$), Temporal Stability (Lipschitz continuity), and Decomposability (attribution to individual signal contributions).
    - **Design Motivation**: Linear aggregation ensures transparency. The weight ranking $w_A \geq w_E \geq w_D$ encodes domain priors, as attention most directly reflects instruction following.

3.  **Hysteresis Alerting Mechanism**:
    - **Function**: Converts FI into stable online alerts, avoiding frequent toggling near thresholds.
    - **Mechanism**: Activation threshold $\theta = 0.50$ and deactivation threshold $\theta_{\text{low}} = 0.40$ are established. FI must continuously exceed the activation threshold to trigger an alert and must drop below the deactivation threshold to recover. Short-window smoothing further suppresses instantaneous jitter.
    - **Design Motivation**: Single-thresholding in production leads to excessive false alerts; the hysteresis mechanism reduces alert flipping by $>91\%$ across datasets.

## Key Experimental Results

### Main Results
Evaluation on OPT-2.7B across three QA datasets (27,405 generated sequences):

| Dataset | Samples | Avg FI | Repetition Rate | Spearman ρ (Full) | Spearman ρ (First 20) |
|---------|---------|--------|-----------------|-------------------|-----------------------|
| HotpotQA| 7,405   | 0.815  | 0.404           | 0.848             | 0.425                 |
| SQuAD   | 10,000  | 0.812  | 0.423           | 0.856             | 0.375                 |
| TriviaQA| 10,000  | 0.833  | 0.467           | 0.820             | 0.404                 |

AUROC comparison for severe degradation detection (HotpotQA):

| Method | AUROC |
|--------|-------|
| **Fatigue Index (Ours)** | **0.976** |
| Entropy Only | 0.954 |
| Drift Only | 0.929 |
| Attention Only (Inverse) | 0.307 |

### Ablation Study

| Configuration | Key Metric | Description |
|---------------|------------|-------------|
| Full FI + Hysteresis | 91-93% Flip Reduction | Toggling reduced from 18-21 to 1.4-1.7 per sequence |
| FP16 Precision | Stable Entropy | Normal attention and drift trajectories |
| 4-bit NF4 Quantization | Unstable Entropy Collapse | Quantization mainly disrupts calibration over attention/drift |
| Short Context (len=192) | High Prompt Attention | Attention remains significantly non-zero |
| Long Context (len=1446) | Near-zero Attention | Faster, total attention collapse observed |

### Key Findings
- **Aggregation outperforms single signals**: FI's AUROC (0.976) significantly exceeds the strongest single signal (Drift 0.929), validating multi-signal aggregation. While attention alone performs poorly in AUROC (0.307), it receives high weight because it directly reflects instruction following.
- **Non-monotonic scaling**: Among 9 models (1B to 13B), instruct-tuned models below 3B collapse faster than base models. This trend reverses at 7B, where instruct-tuned models begin to outperform. Llama-2-13B-Chat exhibits "safety fatigue," collapsing into low-entropy refusal templates.
- **Drift slope is size-independent**: Latent drift slopes cluster between 0.08 and 0.14 across different scales, suggesting larger models are not "drifting less" but rather "drifting more coherently."
- **Positional bias exacerbates fatigue**: Evidence at the beginning of a prompt receives 5-10x more attention than middle or end positions; primacy bias leads to systematic neglect of late-context information.

## Highlights & Insights
- **From post-hoc observation to real-time diagnosis**: FI requires only logits, attention, and latent states, allowing for online calculation without retraining. This "model endoscope" approach is applicable to any LLM deployment requiring runtime reliability monitoring.
- **Axiomatic design ensures metric quality**: The five axioms (Monotonicity/Scale Invariance/Boundedness/Stability/Decomposability) constrain FI's form and provide a general framework for evaluating online diagnostic metrics.
- **"Safety Fatigue" discovery**: The collapse of 13B aligned models into refusal templates reveals side effects of over-alignment on output diversity, offering a warning for RLHF and alignment research.

## Limitations & Future Work
- FI requires access to internal components (logits, attention, latent states), precluding use with closed-source APIs (e.g., GPT-4).
- Experiments were restricted to QA tasks with a 120-token generation limit; longer scenarios such as dialogue, coding, or tool-calling remain to be validated.
- The assumptions of linear aggregation and fixed weights are strong; weight transferability across model families remains unverified.
- Currently only validates prediction of repetitive degradation; non-repetitive failures like hallucinations or factual errors are not covered.
- Future work could explore closed-loop intervention (e.g., switching strategies when FI triggers), mechanistic interpretability (locating circuit-level causes of fatigue), and adaptive weight learning.

## Related Work & Insights
- Refines the task-level findings of "Lost in the Middle" (Liu et al., 2023) into token-level online signals.
- Addresses the same text degradation problem as nucleus sampling (Holtzman et al., 2020) but from a monitoring perspective.
- Contrasts with offline semantic entropy methods (Farquhar et al., 2024) by emphasizing real-time, multi-signal fusion.
- The FI approach could be extended to multimodal models (attention decay of visual tokens) or Agent systems (monitoring cumulative degradation in multi-step calls).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] How Intrinsic Motivation Shapes Learned Representations in Decision Transformers: A Cognitive Interpretability Analysis](../../NeurIPS2025/interpretability/toward_explainable_offline_rl_analyzing_representations_in_intrinsically_motivat.md)
- [\[ICLR 2026\] Decoupling Dynamical Richness from Representation Learning: Towards Practical Measurement](../../ICLR2026/interpretability/decoupling_dynamical_richness_from_representation_learning_towards_practical_mea.md)
- [\[ICLR 2026\] AdAEM: An Adaptively and Automated Extensible Measurement of LLMs' Value Difference](../../ICLR2026/interpretability/adaem_an_adaptively_and_automated_extensible_measurement_of_llms_value_differenc.md)
- [\[ACL 2026\] Interpreto: An Explainability Library for Transformers](../../ACL2026/interpretability/interpreto_an_explainability_library_for_transformers.md)
- [\[ICML 2026\] Dissecting Multimodal In-Context Learning: Modality Asymmetries and Circuit Dynamics in modern Transformers](dissecting_multimodal_in-context_learning_modality_asymmetries_and_circuit_dynam.md)

</div>

<!-- RELATED:END -->

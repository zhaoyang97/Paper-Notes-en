---
title: >-
  [Paper Note] Transformer Copilot: Learning from The Mistake Log in LLM Fine-tuning
description: >-
  [NeurIPS 2025][Recommender Systems][Mistake Log] This paper proposes the Transformer Copilot framework, which systematically records a "Mistake Log" during LLM fine-tuning…
tags:
  - "NeurIPS 2025"
  - "Recommender Systems"
  - "Mistake Log"
  - "Pilot-Copilot"
  - "logits rectification"
  - "fine-tuning"
  - "error-aware"
date: 2026-05-08
content_hash: 865d0f036510b0c3
---

# Transformer Copilot: Learning from The Mistake Log in LLM Fine-tuning

**Conference**: NeurIPS 2025
**arXiv**: [2505.16270](https://arxiv.org/abs/2505.16270)  
**Code**: [GitHub](https://github.com/jiaruzouu/TransformerCopilot)  
**Area**: Recommender Systems
**Keywords**: Mistake Log, Pilot-Copilot, logits rectification, fine-tuning, error-aware

## TL;DR
This paper proposes the Transformer Copilot framework, which systematically records a "Mistake Log" during LLM fine-tuning, trains an auxiliary Copilot model to learn the Pilot's error patterns, and rectifies logits at inference time to improve generation quality, achieving up to 34.5% improvement across 12 benchmarks.

## Background & Motivation

**Background**: Supervised fine-tuning (SFT) is the standard approach for adapting LLMs to specific domains. However, fine-tuned models still suffer from train-test misalignment at inference time — they cannot fully capture task-specific nuances or may overfit certain patterns in the training data.

**Limitations of Prior Work**:
- Standard SFT updates parameters via loss gradients, discarding each error immediately after consumption; the model retains no explicit memory of *where*, *how*, or *why* it erred.
- Data-side interventions (self-refinement) and external feedback (RLHF, Reflexion) require additional data or human annotation.
- The final fine-tuned parameters $\theta_T^P$ contain no error information from the training trajectory — these valuable learning signals are wasted.

**Key Challenge**: SFT optimizes parameters to minimize loss, but parameter updates are "use-and-discard" — the model may repeatedly make similar errors on the same class of problems because it lacks an explicit error-reflection mechanism.

**Goal**: Rather than modifying the Pilot model's training process, leverage intermediate signals recorded during training (the Mistake Log) to assist correction at inference time.

**Key Insight**: Analogous to the "mistake notebook" reflection mechanism in human learning — recording errors, analyzing their causes, and using them as reminders during evaluation.

**Core Idea**: Systematically record the model's error patterns (inputs, internal states, token-level errors) during fine-tuning, train an auxiliary Copilot to learn these patterns, and rectify the Pilot's logits at inference time.

## Method

### Overall Architecture
Three components: (1) Mistake Log definition and collection; (2) Copilot model design and joint training; (3) logits fusion at inference time. The Pilot is fine-tuned normally while the Copilot learns the Pilot's error patterns in parallel; at inference time, both collaborate for generation.

### Key Designs

1. **Mistake Log**:

    - Function: Systematically records three types of information throughout the entire fine-tuning process.
    - Three components:
        - **Question** $\tilde{X}_t$: Input representation (encoder output or embedding layer output).
        - **Rationale** $h_t$: Hidden states of each token across all decoder layers $\{h_{t,i,l}\}_{l=1}^{L^P}$, reflecting the model's internal "reasoning process."
        - **Mistake** $\ell_t$: Token-level prediction error $\ell_t(p_{t,i}, \hat{p}_{t,i}) = p_{t,i} - \hat{p}_{t,i}$, precisely quantifying the direction and magnitude of each token's error.
    - Complete Mistake Log: $M_T = \{(\tilde{X}_t, h_t, \ell_t)\}_{t=1}^T$

2. **Copilot Model Design**:

    - Function: Initialized from the Pilot's decoder, trained to predict the Pilot's token-level errors.
    - **Encoder-Decoder Copilot**: Input is the token-level error sequence $\ell_{t,<i}$ (projected to hidden dim). Uses modified cross-attention where Queries come from the Copilot's own hidden states and Keys/Values come from the concatenation of the Pilot's input representations and pooled hidden states.
    - **Decoder-only Copilot**: Odd-numbered layers use standard self-attention; even-numbered layers use modified cross-attention attending to Pilot information.
    - Loss function: $\mathcal{L}_t^C = \sqrt{\sum_i \|f_{t,i}^C - \ell_t(p_{t,i}, \hat{p}_{t,i})\|^2}$ (RMSE to avoid excessive gradient smoothing).

3. **Joint Training Paradigm**:

    - Each iteration: (a) Pilot forward pass and parameter update; (b) collect Mistake Log entry; (c) sample from Mistake Log to train Copilot.
    - The Copilot continuously tracks the Pilot's evolution, learning the most recent error patterns.

4. **Inference-time Logits Rectification**:

    - Core formula: $\tilde{p}_{t,i} = \hat{p}_{t,i} + \lambda f_{t,i}^C$
    - The Copilot autoregressively generates error predictions, which are added back to the Pilot's logits.
    - $\lambda$ is a rectification strength hyperparameter (default 1); theoretical guarantees ensure the existence of $\lambda_0 > 0$ such that the rectified output is closer to the true distribution.

### Theoretical Guarantee (Theorem 4.1)
Under the mild condition that Copilot error $\epsilon_C < \sqrt{\epsilon_P^2 + \sigma_P^2}$, the rectified $\tilde{p}_{t,i}$ is strictly closer to the true distribution $p_{t,i}$ than $\hat{p}_{t,i}$. Notably, the Copilot may have larger bias than the Pilot ($\epsilon_C > \epsilon_P$) and still be effective — explaining why a smaller Copilot is sufficient.

## Key Experimental Results

### Main Results
12 benchmarks spanning commonsense reasoning, arithmetic reasoning, and recommendation tasks.

| Pilot Model | Task Type | w/o Copilot | + Copilot | Gain |
|---|---|---|---|---|
| T5 series | Commonsense reasoning | baseline | +2–15% | Significant |
| LLaMA-3.2-3B | Commonsense reasoning | baseline | +5–34.5% | Up to 34.5% |
| Qwen2.5-7B + 1B Copilot | General | Below Qwen2.5-14B | **Surpasses Qwen2.5-14B** | 4B fewer params |
| Various Pilots | Arithmetic reasoning | baseline | +2–20% | Consistent gains |

### Ablation Study

| Analysis Dimension | Finding |
|---|---|
| Copilot size | A 1B Copilot is sufficient to effectively assist a 3B–7B Pilot |
| Computational overhead | Marginal increment — Copilot inference cost is far lower than scaling up the Pilot |
| Transferability | A trained Copilot can be directly transferred to a new Pilot without retraining |
| Scalability | Remains consistently effective as Pilot scale increases |
| Logits rectification visualization | Copilot correction directions align with correct answers |

### Key Findings
- **Copilot as "error corrector," not "independent reasoner"**: It learns the Pilot's error patterns rather than independent task knowledge.
- **Small Copilot suffices**: A 1B Copilot assisting a 7B Pilot can outperform a standalone 14B model — more parameter-efficient than simply scaling up.
- **Cross-model transferability**: A Copilot trained on one Pilot transfers to other Pilots in the same family, suggesting error patterns are shared across models.
- **Precise token-level rectification**: Visualizations show that the Copilot applies corrections at exactly the positions where the Pilot makes formatting or factual errors.

## Highlights & Insights
- **The "mistake notebook" metaphor is intuitive and effective**: The reflection mechanism from human learning is formalized as a Mistake Log — a natural and practical concept.
- **Exploiting discarded training signals**: In standard SFT, intermediate hidden states and token-level errors are discarded after parameter updates; Copilot transforms this "waste" into valuable supervision signals.
- **Lenient theoretical conditions**: The Copilot need not be more accurate than the Pilot — it only needs to satisfy mild conditions to guarantee improvement, making a small Copilot viable.
- **High parameter efficiency**: No part of the Pilot is modified; significant gains are achieved solely by attaching a small auxiliary model.

## Limitations & Future Work
- **Training-time memory overhead**: Storing the Mistake Log (hidden states and errors across all training steps) may require strategies such as retaining only the most recent $N$ steps for large-scale training.
- **Inference latency**: Although the Copilot is small, the additional forward pass still increases latency.
- **Copilot error accumulation**: During autoregressive inference, the Copilot conditions on its own generated error predictions rather than ground-truth errors, potentially accumulating bias.
- **Dependence on the Pilot's training trajectory**: The Copilot learns error patterns from a specific training run; changes to training data or hyperparameters may necessitate retraining.
- **No comparison against alignment methods**: Methods such as RLHF and DPO also address the train-inference misalignment problem but are not evaluated against.

## Related Work & Insights
- **vs. Self-Refinement / Reflexion**: These methods prompt the model to self-reflect at inference time, requiring multiple inference passes. Copilot requires only a single forward pass and is thus more efficient.
- **vs. Knowledge Distillation**: Distillation transfers knowledge from a large model to a small one; Copilot uses a small model to assist a large one — the direction is reversed.
- **vs. Speculative Decoding**: Both use a small model to assist large-model inference, but Speculative Decoding accelerates decoding while Copilot improves output quality.
- **vs. Logits Calibration**: Post-processing methods such as temperature scaling apply global adjustments; Copilot performs token-level conditional rectification, offering finer granularity.

## Rating
- Novelty: ⭐⭐⭐⭐ The Mistake Log concept and Pilot-Copilot framework are novel, though logits rectification itself is not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 12 benchmarks × 3 task types × multiple Pilots + theoretical analysis + visualization + transferability/scalability analysis — highly comprehensive.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Overcoming Sparsity Artifacts in Crosscoders to Interpret Chat-Tuning](overcoming_sparsity_artifacts_in_crosscoders_to_interpret_chat-tuning.md)
- [\[ACL 2026\] ReRec: Reasoning-Augmented LLM-based Recommendation Assistant via Reinforcement Fine-tuning](../../ACL2026/recommender/rerec_reasoning-augmented_llm-based_recommendation_assistant_via_reinforcement_f.md)
- [\[NeurIPS 2025\] Semantic Retrieval Augmented Contrastive Learning for Sequential Recommendation](semantic_retrieval_augmented_contrastive_learning_for_sequential_recommendation.md)
- [\[NeurIPS 2025\] Validating LLM-as-a-Judge Systems under Rating Indeterminacy](validating_llm-as-a-judge_systems_under_rating_indeterminacy.md)
- [\[NeurIPS 2025\] Who You Are Matters: Bridging Topics and Social Roles via LLM-Enhanced Logical Recommendation](who_you_are_matters_bridging_topics_and_social_roles_via_llm-enhanced_logical_re.md)

</div>

<!-- RELATED:END -->

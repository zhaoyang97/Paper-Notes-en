---
title: >-
  [Paper Note] Self-Refining Language Model Anonymizers via Adversarial Distillation
description: >-
  [NeurIPS 2025][LLM Safety][Privacy Protection] This paper proposes SEAL, a framework that distills GPT-4-level text anonymization capabilities into an 8B model via adversarial distillation…
tags:
  - "NeurIPS 2025"
  - "LLM Safety"
  - "Privacy Protection"
  - "Text Anonymization"
  - "Knowledge Distillation"
  - "Adversarial Learning"
  - "Self-Refinement"
date: 2026-05-08
content_hash: ed854974e214658b
---

# Self-Refining Language Model Anonymizers via Adversarial Distillation

**Conference**: NeurIPS 2025
**arXiv**: [2506.01420](https://arxiv.org/abs/2506.01420)
**Code**: [GitHub](https://github.com/kykim0/SEAL)
**Area**: AI Safety
**Keywords**: Privacy Protection, Text Anonymization, Knowledge Distillation, Adversarial Learning, Self-Refinement

## TL;DR

This paper proposes SEAL, a framework that distills GPT-4-level text anonymization capabilities into an 8B model via adversarial distillation, combining SFT + DPO training with a self-refinement mechanism. The resulting small model achieves privacy–utility trade-offs on par with or superior to GPT-4-based anonymizers while enabling fully local deployment.

## Background & Motivation

Large language models are widely deployed in sensitive domains such as healthcare, finance, and dialogue systems. However, their ability to **infer personal information from seemingly innocuous text** poses serious emerging privacy risks. LLMs can infer sensitive attributes such as location, identity, and demographic characteristics from user utterances with surprisingly high accuracy, often without the user's awareness.

Traditional anonymization approaches (e.g., named entity recognition, pattern matching) address only surface-level identifiers (names, ID numbers, etc.) and cannot handle inference that LLMs perform by exploiting semantic context. For instance, the phrase "Debugging life like it's faulty code!" contains no explicit PII, yet an LLM can infer that the author is likely a software developer.

Recent LLM-based anonymization frameworks (e.g., adversarial anonymization) have shown progress but suffer from two major issues:

**Reliance on commercial large models** (e.g., GPT-4), incurring high costs

**Data security risks**: sensitive text must be sent to untrusted external systems

Existing distillation attempts still rely on GPT-4 for adversarial feedback, failing to fundamentally resolve the problem.

## Method

### Overall Architecture

SEAL (Self-refining Anonymization with Language model) is a three-stage framework: (1) generating adversarial anonymization trajectories using LLMs; (2) distilling these into a small model via SFT and DPO; and (3) applying self-refinement at inference time. The core innovation is the simultaneous distillation of both anonymization capability and **judging capability** (privacy inference + utility evaluation) into a single small model, enabling self-improvement without reliance on external feedback.

### Key Designs

1. **Adversarial Data Synthesis**:

    - Three LLM roles are employed: an anonymizer $\mathcal{M}_{\text{anon}}$, a privacy inference model $\mathcal{M}_{\text{priv}}$, and a utility evaluator $\mathcal{M}_{\text{util}}$
    - Iterative pipeline: the inference model identifies recoverable attributes $\mathcal{P}_t$ from the current text $x_t$ → the anonymizer refines the text to $x_{t+1}$ → the utility evaluator assesses $\mathcal{U}_{t+1}$
    - Each document produces a trajectory $\tau = (s_0, s_1, \ldots, s_T)$, where each step contains $(x_i, \mathcal{P}_i, \mathcal{U}_i)$
    - GPT-4o is used to generate data over 275 synthetic persona profiles with up to 3 refinement iterations each
    - Design motivation: training on synthetic profiles allows the distilled small model to run locally on real sensitive data without external transmission

2. **Multi-Task SFT Training**:

    - Three tasks are jointly trained:
        - **Anonymization task**: text pairs where both privacy and utility improve are extracted from trajectories, $\mathcal{D}_{\text{anon}} = \{(x_i, x_j) \mid p(s_j) > p(s_i), u(s_j) \geq u(s_i)\}$
        - **Privacy inference task**: $\mathcal{D}_{\text{priv}} = \{(x_i, \mathcal{P}_i)\}$, training the model to identify inferable attributes
        - **Utility evaluation task**: $\mathcal{D}_{\text{util}} = \{(x_i, \mathcal{U}_i)\}$, training the model to assess anonymization quality
    - Privacy scoring function: $p(s_i) = (-|\mathcal{P}_i|, -\sum_{m \in \mathcal{P}_i} \text{conf}(m)/|\mathcal{P}_i|)$, jointly considering the number of inferable attributes and inference confidence
    - Total loss: $\mathcal{L}_{\text{SFT}} = \lambda_{\text{anon}} \cdot \mathcal{L}_{\text{anon}} + \lambda_{\text{priv}} \cdot \mathcal{L}_{\text{priv}} + \lambda_{\text{util}} \cdot \mathcal{L}_{\text{util}}$
    - Design motivation: enabling the model to simultaneously learn to "act" (anonymize) and "judge" (infer + evaluate) lays the foundation for self-refinement

3. **DPO Preference Learning**:

    - Preference pairs are constructed from the same trajectory: anonymizations with better privacy and no utility degradation are treated as preferred outputs
    - $\mathcal{D}_{\text{pref}} = \{(x_i, x_w, x_l) \mid p(s_w) > p(s_l), u(s_w) \geq u(s_l)\}$
    - DPO loss is minimized: $\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(x_w|x_i)}{\pi_{\text{ref}}(x_w|x_i)} - \beta\log\frac{\pi_\theta(x_l|x_i)}{\pi_{\text{ref}}(x_l|x_i)}\right)\right]$
    - Design motivation: while SFT enables the model to generate diverse anonymizations, DPO teaches the model to prefer stronger privacy–utility trade-offs

4. **Inference-Time Self-Refinement**:

    - The model alternates between: inferring attributes $\mathcal{P}_t^\pi$ → evaluating utility $\mathcal{U}_t^\pi$ → conditionally generating an improved anonymization $x_{t+1} \sim \pi(\cdot | x_t, \mathcal{P}_t^\pi, \mathcal{U}_t^\pi)$
    - No external model feedback is required; a single model completes the entire loop
    - Despite training on 3-step trajectories, the model generalizes to more refinement iterations

### Loss & Training

- Stage 1: Multi-task SFT (anonymization + inference + evaluation) using standard next-token prediction loss
- Stage 2: DPO preference learning, using the SFT model as the reference model
- Inference: iterative self-refinement, allowing users to interactively control the degree of anonymization according to their privacy–utility preferences

## Key Experimental Results

### Main Results (Main Dataset)

| Method | Privacy↓ | Utility↑ | Combined↑ |
|------|-------|-------|-------|
| Original Text | 0.625 | 1.0 | - |
| Azure PII | 0.587 | 0.962 | 0.023 |
| Dipper (11B) | 0.555 | 0.868 | -0.020 |
| Adversarial Anonymization (GPT-4o) | 0.434 | 0.947 | 0.253 |
| **SEAL (8B, iter 1)** | 0.391 | 0.931 | 0.305 |
| **SEAL (8B, iter 2)** | 0.302 | 0.893 | 0.410 |
| **SEAL (8B, iter 3)** | **0.263** | 0.862 | **0.441** |

The SEAL 8B model surpasses all baselines in privacy protection at the first iteration, with a substantially higher combined score.

### Ablation Study

| Configuration | Privacy↓ (Main) | Utility↑ (Main) | Privacy↓ (Hard) |
|------|-------------|-------------|-------------|
| SFT only, anonymization only | 0.513 | 0.963 | 0.672 |
| SFT only, anon + judging | 0.498 | 0.968 | 0.679 |
| SFT only, + adversarial feedback | 0.460 | 0.958 | 0.675 |
| SFT only, + confidence scoring | 0.458 | 0.952 | 0.671 |
| **SFT+DPO, full** | **0.379** | 0.931 | **0.614** |

Each component contributes incrementally: DPO > multi-task > adversarial feedback > confidence scoring.

### Key Findings

- **An 8B model can match or exceed GPT-4o**: on the Main dataset, it surpasses GPT-4o at iteration 1; on the Hard dataset, after iteration 2
- **Model scale effects**: the 8B model performs best; 3B–4B models are competent but show earlier saturation in refinement gains; 1B models still substantially outperform traditional methods
- **Cross-judge consistency**: evaluation results are consistent across GPT-4.1, Claude Sonnet 4, and Gemini 2.5 Flash
- **Human evaluation validation**: GPT-4.1 shows high agreement with humans on readability ($r=0.717$), semantic preservation ($r=0.814$), and hallucination detection ($\text{acc}=0.775$)
- Azure PII and Dipper are largely ineffective — traditional approaches entirely fail to handle context-embedded private information
- Inference latency: the 8B model in anonymization-only mode (0.94s) is even faster than the GPT-4o API (1.09s)

## Highlights & Insights

- Key insight: **"Self-judging" capability is a prerequisite for self-refinement** — only by jointly training the model to anonymize and evaluate can it form a closed improvement loop at inference time
- The distillation strategy is elegant: training on synthetic data and deploying locally on real data fundamentally eliminates the need to transmit sensitive information externally
- DPO contributes beyond SFT: while SFT enables diverse anonymization generation, DPO instills the model's ability to judge "which is better"
- The model generalizes from 3-step training trajectories to 5 or more self-refinement steps, demonstrating that it has learned a general privacy-preserving strategy rather than rote memorization

## Limitations & Future Work

- On the Hard dataset, which contains more context-embedded private information, more refinement iterations are required to achieve satisfactory results
- Utility loss gradually accumulates with more refinement iterations (utility drops to 0.862 at iter 3)
- The stability and convergence of self-refinement lack theoretical guarantees
- Future directions include leveraging the judging capability as a generative reward model for training-time self-improvement

## Related Work & Insights

- This work aligns with the direction of Staab et al. (2025) "Language models are advanced anonymizers" but addresses the core limitation of dependence on commercial models
- SEAL's "generation + judging" distillation paradigm is general and can be extended to tasks such as summarization and code generation
- The approach has direct applicability to privacy-sensitive scenarios such as medical NLP and legal document processing

## Rating

- Novelty: ⭐⭐⭐⭐ The adversarial distillation + self-refinement paradigm is novel, and the joint "generation + judging" distillation concept is clearly motivated
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers multiple datasets, model scales, judges, human evaluation, and latency analysis comprehensively
- Writing Quality: ⭐⭐⭐⭐ Well-structured with clear algorithmic diagrams
- Value: ⭐⭐⭐⭐⭐ Addresses a core practical problem in LLM privacy protection with open-sourced code and data

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] AgentStealth: Reinforcing Large Language Model for Anonymizing User-generated Text](agentstealth_reinforcing_large_language_model_for_anonymizing_user-generated_tex.md)
- [\[ICLR 2026\] Self-Destructive Language Model](../../ICLR2026/llm_safety/self-destructive_language_model.md)
- [\[NeurIPS 2025\] Demystifying Language Model Forgetting with Low-Rank Example Associations](demystifying_language_model_forgetting_with_low-rank_example_associations.md)
- [\[NeurIPS 2025\] PULSE: Practical Evaluation Scenarios for Large Multimodal Model Unlearning](pulse_practical_evaluation_scenarios_for_large_multimodal_model_unlearning.md)
- [\[NeurIPS 2025\] MixAT: Combining Continuous and Discrete Adversarial Training for LLMs](mixat_combining_continuous_and_discrete_adversarial_training_for_llms.md)

</div>

<!-- RELATED:END -->

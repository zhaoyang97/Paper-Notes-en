---
title: >-
  [Paper Note] Leave My Images Alone: Preventing Multi-Modal Large Language Models from Analyzing Unauthorized Images
description: >-
  [ACL 2026][Multimodal VLM][Visual prompt injection] This paper proposes ImageProtector, which embeds nearly imperceptible adversarial perturbations into images as a visual prompt injection attack…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "Visual prompt injection"
  - "image privacy protection"
  - "multimodal large language models"
  - "adversarial perturbation"
  - "refusal response"
date: 2026-05-08
content_hash: 03e6a59fbd9b4911
---

# Leave My Images Alone: Preventing Multi-Modal Large Language Models from Analyzing Unauthorized Images

**Conference**: ACL 2026
**arXiv**: [2604.09024](https://arxiv.org/abs/2604.09024)
**Code**: None
**Area**: AI Safety / Multimodal Privacy Protection
**Keywords**: Visual prompt injection, image privacy protection, multimodal large language models, adversarial perturbation, refusal response

## TL;DR

This paper proposes ImageProtector, which embeds nearly imperceptible adversarial perturbations into images as a visual prompt injection attack, causing MLLMs to generate refusal responses when analyzing protected images. This prevents malicious actors from exploiting open-weight MLLMs to extract private information from images at scale.

## Background & Motivation

**Background**: Multimodal large language models (MLLMs) such as LLaVA, MiniGPT-4, and Qwen-VL can be used to analyze images on the internet at scale, extracting sensitive information such as identity and location. The proliferation of open-weight models has further lowered the barrier to malicious exploitation.

**Limitations of Prior Work**: Existing privacy protection measures (e.g., face blurring, metadata removal) are insufficient against the deep comprehension capabilities of MLLMs. Traditional adversarial attacks (e.g., jailbreak attacks, visual prompt injection) are primarily designed for offensive purposes and have not been repurposed for privacy defense.

**Key Challenge**: Users wish to share images on social media while maintaining usability, yet simultaneously need to prevent automated MLLM analysis from extracting private information—a fundamental utility–privacy conflict.

**Goal**: Design a user-side proactive defense mechanism that adds imperceptible perturbations to images before sharing, causing any MLLM to output a refusal response upon analysis.

**Key Insight**: Repurpose visual prompt injection from an offensive technique into a defensive mechanism—the embedded perturbation acts as an "invisible instruction" that causes the model to respond with "Sorry, I can't help you" regardless of the query received.

**Core Idea**: Formalize privacy protection as a constrained optimization problem that maximizes the probability of refusal responses from MLLMs on perturbed images under an $\ell_\infty$ norm constraint.

## Method

### Overall Architecture

The core pipeline of ImageProtector consists of: (1) constructing a shadow question set using an LLM; (2) generating perturbations via gradient-based optimization on target MLLMs; and (3) publishing images with the embedded perturbations. The optimization objective simultaneously satisfies effectiveness (high refusal rate) and practicality (imperceptible perturbations).

### Key Designs

1. **Shadow Question Construction**: Three categories of shadow questions are designed—precise probe questions (directly matching anticipated attack queries), similar probe questions (LLM-generated thematic variants), and universal probe questions (covering arbitrary probe scenarios). The underlying assumption is that optimizing over diverse shadow questions enables the perturbation to generalize to unseen real-world malicious queries.

2. **Constrained Optimization Objective**: The objective is formalized as cross-entropy loss minimization: $\delta^*_R = \arg\min_{\delta_R} \sum_{M \in \mathcal{M}} \sum_{q \in Q_S} \mathcal{L}_{CE}(M, R, x_I + \delta_R, q)$, subject to $\|\delta_R\|_\infty \leq \epsilon$. Here $R$ is a target refusal response sampled randomly from 10 refusal templates to enhance diversity and stealthiness.

3. **BIM Gradient Optimization**: The Basic Iterative Method (BIM) is used to solve the optimization, with per-step update $\delta_R = \text{proj}(\delta_R - \alpha \cdot \text{sign}(\nabla_{\delta_R} \mathcal{L}), \epsilon)$. The method supports simultaneous optimization of universal perturbations across multiple MLLMs for cross-model protection. **Design Motivation**: BIM is more efficient than PGD, reducing GPU time from 61.2 minutes to 45.6 minutes with comparable performance.

### Loss & Training

The loss function is based on the cross-entropy over the target refusal sequence $R = (t_1, \ldots, t_r)$: $\mathcal{L}_{CE} = -\sum_{k=1}^{r} \log p_M(t_k | [x_I + \delta_R, q, t_{<k}])$. At each iteration, a mini-batch is sampled from the shadow question set to compute gradients. An early stopping criterion is applied (terminating when the loss falls below 0.001 for 30 consecutive steps) to prevent overfitting.

## Key Experimental Results

### Main Results

| Target MLLM | VQAv2 | GQA | CelebA | TextVQA | Avg. |
|---|---|---|---|---|---|
| LLaVA-1.5 | 0.94 | 0.94 | 1.00 | 0.91 | 0.95 |
| MiniGPT-4 | 0.86 | 0.93 | 0.97 | 0.81 | 0.89 |
| Qwen-VL-Chat | 0.94 | 0.95 | 0.99 | 0.88 | 0.94 |
| InstructBLIP | 0.91 | 0.94 | 0.93 | 0.92 | 0.93 |
| Phi-4-multimodal | 1.00 | 1.00 | 1.00 | 0.98 | 1.00 |
| Qwen2.5-VL | 0.96 | 1.00 | 1.00 | 0.97 | 0.98 |

*Refusal rates under precise shadow questions (image-relevant questions)*

### Ablation Study

| Method | Precise Questions | Similar Questions | Universal Questions |
|---|---|---|---|
| No Perturbation | 0.00 | 0.00 | 0.00 |
| Qi et al. | 0.02 | 0.02 | 0.02 |
| Bagdasaryan et al. | 0.65 | 0.62 | 0.51 |
| ImageProtector+PGD | 0.94 | 0.91 | 0.91 |
| ImageProtector (BIM) | 0.94 | 0.88 | 0.88 |

*Refusal rate comparison of different methods on LLaVA-1.5 / VQAv2*

### Key Findings

- ImageProtector achieves an average refusal rate of 0.86–0.95 across 6 MLLMs and 4 datasets.
- Refusal rates for image-relevant questions (0.95) are slightly higher than for image-irrelevant questions (0.94).
- InstructBLIP is the most resistant model due to its Q-Former architecture.
- Three countermeasures (Gaussian noise, DiffPure, adversarial training) can partially mitigate the perturbations, but simultaneously cause significant degradation in model accuracy.

## Highlights & Insights

- **Attack-to-Defense Perspective**: This work is the first to repurpose visual prompt injection from an offensive technique into a user-side privacy protection tool.
- **Generalization of Universal Refusal**: Perturbations optimized on universal shadow questions also effectively elicit refusals for domain-specific questions, suggesting that the perturbations learn a general "refusal pattern" rather than question-specific patterns.
- **Defense–Countermeasure Dilemma**: All three countermeasures require a trade-off between protection effectiveness and model performance, establishing a new equilibrium in the adversarial arms race.

## Limitations & Future Work

- The method assumes white-box access to the target MLLM; transferability to closed-source commercial models (e.g., GPT-4V) is limited.
- Although perturbations with $\epsilon=8/255$ are nearly imperceptible, they remain detectable under extreme magnification.
- The impact of JPEG compression and social platform image processing pipelines on perturbation robustness is not considered.
- Future work may explore black-box transfer attacks and adaptive perturbation generation (eliminating the need for per-image optimization).

## Related Work & Insights

- Shares the proactive defense philosophy of face recognition adversarial methods (Fawkes, LowKey), but extends the target from classifiers to generative MLLMs.
- Constitutes a defensive application of visual prompt injection (Bagdasaryan et al., 2023).
- May inspire the development of more general "AI-analysis immunity" techniques.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The perspective of repurposing adversarial attacks for privacy defense is novel, with a well-formalized problem definition.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive coverage across 6 models × 4 datasets × 3 shadow question types × 3 countermeasures.
- **Writing Quality**: ⭐⭐⭐⭐ Problem motivation, threat model, and method formalization are all clearly articulated.
- **Value**: ⭐⭐⭐⭐ Proposes a new defensive paradigm in the AI privacy protection domain with practical application potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Modal Aphasia: Can Unified Multimodal Models Describe Images From Memory?](../../ICLR2026/multimodal_vlm/modal_aphasia_can_unified_multimodal_models_describe_images_from_memory.md)
- [\[ICCV 2025\] Large Multi-modal Models Can Interpret Features in Large Multi-modal Models](../../ICCV2025/multimodal_vlm/large_multi-modal_models_can_interpret_features_in_large_multi-modal_models.md)
- [\[ACL 2026\] LaMI: Augmenting Large Language Models via Late Multi-Image Fusion](lami_augmenting_large_language_models_via_late_multi-image_fusion.md)
- [\[ICLR 2026\] VTool-R1: VLMs Learn to Think with Images via Reinforcement Learning on Multimodal Tool Use](../../ICLR2026/multimodal_vlm/vtool-r1_vlms_learn_to_think_with_images_via_reinforcement_learning_on_multimoda.md)
- [\[CVPR 2026\] TIGeR: A Unified Framework for Time, Images and Geo-location Retrieval](../../CVPR2026/multimodal_vlm/tiger_a_unified_framework_for_time_images_and_geo-location_retrieval.md)

</div>

<!-- RELATED:END -->

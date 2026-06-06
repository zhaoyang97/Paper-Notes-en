---
title: >-
  [Paper Note] Leave My Images Alone: Preventing Multi-Modal Large Language Models from Analyzing Unauthorized Images
description: >-
  [ACL 2026][Multimodal VLM][Visual Prompt Injection] Ours proposes ImageProtector, which embeds nearly imperceptible adversarial perturbations as visual prompt injection attacks into images. This triggers MLLMs to generat…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "Visual Prompt Injection"
  - "Image Privacy Protection"
  - "Multimodal Large Language Models (MLLMs)"
  - "Adversarial Perturbations"
  - "Refusal Response"
date: 2026-05-08
content_hash: dc00844054f0ff69
---

# Leave My Images Alone: Preventing Multi-Modal Large Language Models from Analyzing Unauthorized Images

**Conference**: ACL 2026  
**arXiv**: [2604.09024](https://arxiv.org/abs/2604.09024)  
**Code**: None  
**Area**: AI Safety / Multimodal Privacy Protection  
**Keywords**: Visual Prompt Injection, Image Privacy Protection, Multimodal Large Language Models (MLLMs), Adversarial Perturbations, Refusal Response

## TL;DR

Ours proposes ImageProtector, which embeds nearly imperceptible adversarial perturbations as visual prompt injection attacks into images. This triggers MLLMs to generate refusal responses for protected images, preventing malicious actors from using open-weight MLLMs to extract private information at scale.

## Background & Motivation

**Background**: Multimodal Large Language Models (MLLMs) such as LLaVA, MiniGPT-4, and Qwen-VL can be used for large-scale analysis of internet images to extract sensitive information like identity and location. The popularity of open-weight models further lowers the threshold for malicious use.

**Limitations of Prior Work**: Existing privacy protection methods (e.g., face blurring, metadata removal) cannot handle the deep understanding capabilities of MLLMs. Traditional adversarial attacks (e.g., jailbreaking, visual prompt injection) are primarily used for offensive purposes and have not been utilized for privacy defense.

**Key Challenge**: Users want to maintain the usability of images shared on social media while needing to prevent automated MLLM analysis from extracting private information, creating a utility-privacy conflict.

**Goal**: Design a user-side proactive defense method that adds imperceptible perturbations before sharing images, causing any MLLM to output a refusal response during analysis.

**Key Insight**: Transform visual prompt injection attacks from an offensive technique into a defensive mechanism—embedded perturbations act as "invisible instructions" that compel the model to answer "Sorry, I can't help you" regardless of the query.

**Core Idea**: Formalize privacy protection as a constrained optimization problem, maximizing the MLLM's refusal probability for perturbed images under $\ell_\infty$ norm constraints.

## Method

### Overall Architecture

The core workflow of ImageProtector: (1) Use an LLM to construct a shadow question set; (2) Perform gradient optimization on target MLLMs to generate perturbations; (3) Embed perturbations into images before publishing. The optimization goal satisfies both effectiveness (high refusal rate) and utility (imperceptibility).

### Key Designs

1. **Construction of Shadow Questions**: Design three types of shadow questions—exact probing (direct match to expected attack), similar probing (LLM-generated subject-related variants), and general probing (covering arbitrary scenarios). Optimizing on diverse shadow questions is hypothesized to generalize the perturbations to unseen real-world malicious queries.

2. **Constrained Optimization Objective**: Formalize the goal as cross-entropy loss minimization: $$\delta^*_R = \arg\min_{\delta_R} \sum_{M \in \mathcal{M}} \sum_{q \in Q_S} \mathcal{L}_{CE}(M, R, x_I + \delta_R, q)$$ subject to $\|\delta_R\|_\infty \leq \epsilon$. Here, $R$ is a target refusal response randomly sampled from 10 templates to enhance diversity and stealth.

3. **BIM Gradient Optimization**: Solve using the Basic Iterative Method (BIM). Update $\delta_R = \text{proj}(\delta_R - \alpha \cdot \text{sign}(\nabla_{\delta_R} \mathcal{L}), \epsilon)$ at each step. Supports simultaneous optimization across multiple MLLMs for universal protection. **Design Motivation**: BIM is more efficient than PGD, reducing GPU time from 61.2 to 45.6 minutes with comparable performance.

### Loss & Training

The loss function is based on the cross-entropy of the target refusal sequence $R = (t_1, \ldots, t_r)$: $$\mathcal{L}_{CE} = -\sum_{k=1}^{r} \log p_M(t_k | [x_I + \delta_R, q, t_{<k}])$$. Gradient computation uses mini-batches sampled from the shadow question set, with an early stopping mechanism (terminates if loss is below 0.001 for 30 consecutive iterations) to prevent overfitting.

## Key Experimental Results

### Main Results

| Target MLLM | VQAv2 | GQA | CelebA | TextVQA | Average |
|---|---|---|---|---|---|
| LLaVA-1.5 | 0.94 | 0.94 | 1.00 | 0.91 | 0.95 |
| MiniGPT-4 | 0.86 | 0.93 | 0.97 | 0.81 | 0.89 |
| Qwen-VL-Chat | 0.94 | 0.95 | 0.99 | 0.88 | 0.94 |
| InstructBLIP | 0.91 | 0.94 | 0.93 | 0.92 | 0.93 |
| Phi-4-multimodal | 1.00 | 1.00 | 1.00 | 0.98 | 1.00 |
| Qwen2.5-VL | 0.96 | 1.00 | 1.00 | 0.97 | 0.98 |

*Refusal rates under exact shadow questions (image-relevant questions)*

### Ablation Study

| Method | Exact Questions | Similar Questions | General Questions |
|---|---|---|---|
| No Perturbation | 0.00 | 0.00 | 0.00 |
| Qi et al. | 0.02 | 0.02 | 0.02 |
| Bagdasaryan et al. | 0.65 | 0.62 | 0.51 |
| ImageProtector+PGD | 0.94 | 0.91 | 0.91 |
| ImageProtector (BIM) | 0.94 | 0.88 | 0.88 |

*Comparison of refusal rates for different methods using LLaVA-1.5 on VQAv2*

### Key Findings

- ImageProtector achieves average refusal rates of 0.86-0.95 across 6 MLLMs and 4 datasets.
- Refusal rate for image-relevant questions (0.95) is slightly higher than for irrelevant ones (0.94).
- InstructBLIP is the most difficult model to break due to its Q-Former structure.
- Three countermeasures (Gaussian noise, DiffPure, adversarial training) mitigate perturbations but severely degrade model accuracy.

## Highlights & Insights

- **Perspective Innovation**: Reframe visual prompt injection from an attack technique to a user-side privacy tool for the first time.
- **Universal Refusal Generalization**: Perturbations trained on general shadow questions generalize to specific domains, indicating that the perturbations learn "refusal patterns" rather than specific question patterns.
- **Defense-Countermeasure Dilemma**: Highlights a defense-countermeasure dilemma where trade-offs between protection and performance create a new equilibrium in adversarial games.

## Limitations & Future Work

- Assumes white-box access to target MLLMs; limited transferability to closed-source models (e.g., GPT-4V).
- Perturbations are nearly invisible at $\epsilon=8/255$ but detectable under extreme magnification.
- Impact of JPEG compression and social platform pipelines not considered.
- Future work could explore black-box transfer attacks and adaptive perturbation generation without per-image optimization.

## Related Work & Insights

- Proactive defense similar to facial recognition adversarial work (Fawkes, LowKey), but expanded from classifiers to generative MLLMs.
- Defensive application of visual prompt injection (Bagdasaryan et al., 2023).
- Inspires development of more general "AI analysis immunity" technologies.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Novel perspective of using adversarial attacks for privacy defense with clear formalization.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive coverage with 6 models, 4 datasets, 3 shadow question types, and 3 countermeasures.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation, threat model, and technical formalization are clearly articulated.
- **Value**: ⭐⭐⭐⭐ Proposes a new defense paradigm in AI privacy protection with practical application potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Structured and Abstractive Reasoning on Multi-modal Relational Knowledge Images](structured_and_abstractive_reasoning_on_multi-modal_relational_knowledge_images.md)
- [\[ACL 2026\] AdaTooler-V: Adaptive Tool-Use for Images and Videos](adatooler-v_adaptive_tool-use_for_images_and_videos.md)
- [\[ACL 2026\] Decoding Scientific Experimental Images: The SPUR Benchmark for Perception, Understanding, and Reasoning](decoding_scientific_experimental_images_the_spur_benchmark_for_perception_unders.md)
- [\[ICML 2026\] Debate with Images: Detecting Deceptive Behaviors in Multimodal Large Language Models](../../ICML2026/multimodal_vlm/debate_with_images_detecting_deceptive_behaviors_in_multimodal_large_language_mo.md)
- [\[ICLR 2026\] Modal Aphasia: Can Unified Multimodal Models Describe Images From Memory?](../../ICLR2026/multimodal_vlm/modal_aphasia_can_unified_multimodal_models_describe_images_from_memory.md)

</div>

<!-- RELATED:END -->

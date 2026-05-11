---
title: >-
  [Paper Note] Knowledge Distillation Detection for Open-weights Models
description: >-
  [NeurIPS 2025][Image Generation][knowledge distillation detection] This paper introduces the task of knowledge distillation detection, proposing a data-free input synthesis and statistical scoring framework to determine…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "knowledge distillation detection"
  - "model provenance"
  - "data-free synthesis"
  - "statistical detection"
  - "text-to-image generation"
date: 2026-05-08
content_hash: 789f2abe5d859bbf
---

# Knowledge Distillation Detection for Open-weights Models

**Conference**: NeurIPS 2025
**arXiv**: [2510.02302](https://arxiv.org/abs/2510.02302)
**Code**: [GitHub](https://github.com/shqii1j/distillation_detection)
**Area**: Image Generation
**Keywords**: knowledge distillation detection, model provenance, data-free synthesis, statistical detection, text-to-image generation

## TL;DR

This paper introduces the task of knowledge distillation detection, proposing a data-free input synthesis and statistical scoring framework to determine whether an open-weights student model has been distilled from a specific teacher model.

## Background & Motivation

**Background**: Knowledge distillation is widely used for model compression, transferring knowledge from large teacher models to smaller student models, with demonstrated success in image classification, LLMs, and text-to-image generation.

**Limitations of Prior Work**: Distillation techniques may be misused to clone proprietary models without authorization, infringing intellectual property rights; yet no effective method currently exists to detect whether a model has been distilled.

**Key Challenge**: Existing approaches (e.g., membership inference attacks, OOD detection) primarily focus on training data detection and cannot directly determine distillation relationships between models.

**Goal**: To detect whether a student model has been distilled from a specific teacher, given only the student model weights and access to a teacher model API.

**Key Insight**: The problem is formulated as a multiple-choice task, selecting the most likely distillation source from a set of candidate teacher models.

**Core Idea**: A general framework based on data-free input synthesis and statistical scoring compares output alignment between the student and candidate teachers to detect distillation.

## Method

### Overall Architecture

A three-stage detection pipeline: **Input Construction** → **Score Computation** → **Decision Prediction**. Given an open-weights student model and APIs for $K$ candidate teacher models, synthetic inputs are generated to probe model behavior, alignment scores between the student and each teacher are computed, and the teacher with the highest score is selected as the distillation source.

### Key Designs

1. **Prediction Decision (Score Maximization)**:

    - Function: Select the distillation source from $K$ candidate teachers.
    - Design Motivation: The multiple-choice formulation avoids threshold calibration.
    - Mechanism: $k^* = \arg\max_{k \in \{1,...,K\}} S(g_\theta, f^{(k)}, \mathcal{P})$
    - Novelty: Can be naturally extended to binary detection via a threshold.

2. **Point-wise Score**:

    - Function: Compute per-sample discrepancy between student and teacher outputs.
    - Design Motivation: Simple and effective, functioning even with a single input.
    - Mechanism: For each input $x_n$, compute $s_n^{(k)} = \frac{1}{\delta(g_\theta(x_n), f^{(k)}(x_n)) + \epsilon}$ and average over samples. KL divergence is used for classification; LPIPS is used for text-to-image generation.

3. **Set-level Score**:

    - Function: Measure overall distributional alignment.
    - Design Motivation: Captures global distribution-level alignment patterns.
    - Mechanism: Aligned Cosine Similarity (ACS) is used for classification; CKA with RBF kernels is used for text-to-image generation.
    - Novelty: Requires multiple input samples to compute.

4. **Data-free Input Synthesis**:

    - Function: Generate synthetic queries without access to training data.
    - Design Motivation: Training data is unavailable in practical scenarios.
    - Mechanism (classification): Train a generator $G_\phi$ using a mixup strategy with BNS loss to align batch normalization statistics.
    - Mechanism (text-to-image): Use an empty string as the prompt, exploiting the unconditional generation learned during CFG training.
    - Key formula: $\min_\phi \sum_{i=1}^C w_i \cdot \mathcal{L}_{hard}(g_\theta(\hat{x}(\phi)), y_i) + \mathcal{L}_{BNS}(g_\theta, \hat{x}(\phi))$

### Loss & Training

- Generator training: cross-entropy loss $\mathcal{L}_{hard}$ + BNS alignment loss $\mathcal{L}_{BNS}$
- BNS loss: $\mathcal{L}_{BNS} = \sum_{l=1}^{L} \|\mu_l^r - \mu_l\|_2^2 + \|\sigma_l^r - \sigma_l\|_2^2$
- For text-to-image scenarios, an empty string is used as input, leveraging the unconditional generation property of CFG training.

## Key Experimental Results

### Main Results

**CIFAR-10 Distillation Detection** ($N=100$ synthetic inputs, Acc./AUC):

| Method | N=1 | N=50 | N=100 | Avg. |
|--------|-----|------|-------|------|
| MIA Filter + KL | 0.43/0.66 | 0.54/0.79 | 0.55/0.80 | 0.51/0.75 |
| OOD Filter + KL | 0.42/0.68 | 0.55/0.79 | 0.54/0.80 | 0.50/0.75 |
| **Ours (KL)** | **0.62/0.75** | **0.87/0.94** | **0.87/0.94** | **0.81/0.89** |
| Oracle | 0.45/0.64 | 0.87/0.96 | 0.95/0.99 | 0.70/0.84 |

**Text-to-Image Model Distillation Detection** (Acc./AUC):

| Method | N=1 | N=10 | N=100 | Avg. |
|--------|-----|------|-------|------|
| GPT-2 + DINO | 0.81/0.87 | 0.80/0.94 | 0.80/0.95 | 0.80/0.92 |
| Blip-Base + CLIP | 0.71/0.78 | 0.81/0.93 | 0.83/0.96 | 0.79/0.91 |
| **Ours (LPIPS)** | **0.89/1.00** | **0.97/1.00** | **1.00/0.99** | **0.96/1.00** |

### Ablation Study

| Setting | CIFAR-10 | ImageNet | Avg. |
|---------|----------|----------|------|
| OOD filter + ACS | 0.56/0.77 | 0.37/0.52 | 0.47/0.65 |
| Synthetic Data + CKA | 0.82/0.77 | — | — |
| Ours (full) | **0.87/0.94** | **0.75/0.92** | — |

### Key Findings

- Using only a single synthetic input, the proposed method already substantially outperforms all baselines.
- On CIFAR-10, the method even surpasses the Oracle that uses real training data.
- For text-to-image detection, accuracy reaches 0.97 with $N=10$; the empty-string prompt leveraging CFG unconditional generation proves highly effective.
- Vanilla KD is more detectable than RKD and OFAKD.

## Highlights & Insights

- **Problem Novelty**: The paper is the first to systematically define the knowledge distillation detection task, with significant implications for intellectual property protection.
- **General Framework**: The same framework applies to both classification and generative models and is model-agnostic.
- **Data-Free**: No training data is required; only model weights and API access are needed.
- **Elegant Use of Empty-String Prompts**: The unconditional modeling learned during CFG training makes empty strings a natural in-distribution probe input.

## Limitations & Future Work

- The current formulation only addresses the multiple-choice setting; binary detection requires threshold calibration.
- The generator training for classification relies on BatchNorm statistics and may be limited for architectures without BN (e.g., ViT).
- Adversarial scenarios are not considered: a distiller may deliberately obfuscate model behavior to evade detection.
- The framework could be extended to LLM distillation detection.

## Related Work & Insights

- Complements model watermarking: watermarking is an active defense, whereas this work performs passive detection.
- Data-free synthesis techniques from data-free quantization and distillation are creatively repurposed for detection.
- From an AI-for-Good perspective, this work contributes to the security and traceability of open-source models.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Pioneers the distillation detection task with a distinctive problem formulation.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers both classification and generation tasks with multiple distillation methods.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured with rigorous mathematical derivations.
- Value: ⭐⭐⭐⭐ — Practically meaningful for AI security and intellectual property protection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Why Knowledge Distillation Works in Generative Models: A Minimal Working Explanation](why_knowledge_distillation_works_in_generative_models_a_minimal_working_explanat.md)
- [\[NeurIPS 2025\] Learnable Sampler Distillation for Discrete Diffusion Models](learnable_sampler_distillation_for_discrete_diffusion_models.md)
- [\[NeurIPS 2025\] Epistemic Uncertainty for Generated Image Detection](epistemic_uncertainty_for_generated_image_detection.md)
- [\[NeurIPS 2025\] Distilled Decoding 2: One-step Sampling of Image Auto-regressive Models with Conditional Score Distillation](distilled_decoding_2_onestep_sampling_of_image_autoregressiv.md)
- [\[NeurIPS 2025\] Can Knowledge-Graph-based Retrieval Augmented Generation Really Retrieve What You Need?](can_knowledge-graph-based_retrieval_augmented_generation_really_retrieve_what_yo.md)

</div>

<!-- RELATED:END -->

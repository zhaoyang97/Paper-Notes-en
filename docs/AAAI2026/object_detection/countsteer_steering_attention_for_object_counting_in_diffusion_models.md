---
title: >-
  [Paper Note] CountSteer: Steering Attention for Object Counting in Diffusion Models
description: >-
  [AAAI2026][Object Detection][diffusion models] This paper proposes CountSteer, a training-free inference-time method that injects adaptive steering vectors into the cross-attention hidden states of diffusion models…
tags:
  - "AAAI2026"
  - "Object Detection"
  - "diffusion models"
  - "object counting"
  - "steering vector"
  - "cross-attention"
  - "training-free"
date: 2026-05-08
content_hash: 8784f437261e0480
---

# CountSteer: Steering Attention for Object Counting in Diffusion Models

**Conference**: AAAI2026
**arXiv**: [2511.11253](https://arxiv.org/abs/2511.11253)  
**Code**: To be confirmed  
**Area**: Image Generation
**Keywords**: diffusion models, object counting, steering vector, cross-attention, training-free

## TL;DR

This paper proposes CountSteer, a training-free inference-time method that injects adaptive steering vectors into the cross-attention hidden states of diffusion models, improving object counting accuracy by approximately 4% without degrading image quality.

## Background & Motivation

- Text-to-image (T2I) diffusion models (e.g., Stable Diffusion) can generate highly photorealistic images, yet consistently struggle to follow numerical instructions in text prompts, frequently producing images with incorrect object counts.
- Object counting serves as a clear and quantifiable indicator of generation fidelity, but models find it difficult to maintain numerical information stably throughout the stochastic denoising process.
- Existing solutions primarily rely on fine-tuning (LoRA, DreamBooth) or structural modifications (Composer, T2I-Adapter), incurring additional training and deployment costs.
- Through kernel density estimation (KDE) analysis, the authors identify a key insight: **diffusion models are not entirely "count-blind"—the internal cross-attention hidden state distributions exhibit linearly separable directional differences between correctly and incorrectly counted samples**, suggesting that quantity-related information is already implicitly encoded but not stably expressed.

## Core Problem

How can one leverage the quantity-aware signals already present within a diffusion model's internals to guide the generation process at inference time—without retraining or modifying the model architecture—so that the number of objects in the output image matches the count specified in the text prompt?

## Method

### 1. Mechanism: Steering Vector

Drawing inspiration from Inference-Time Intervention (ITI) in large language models, which identifies linearly separable directions between "truthful" and "untruthful" representations and controls outputs by injecting directional vectors, CountSteer transfers this concept to the UNet architecture of diffusion models:

- For each denoising timestep $t$ and UNet block $b$, the mean hidden states of correctly counted samples (Class 1) and incorrectly counted samples (Class 0) are computed as $\mu_{t,b}^1$ and $\mu_{t,b}^0$, respectively.
- The base steering vector is defined as their difference: $s_{t,b} = \mu_{t,b}^1 - \mu_{t,b}^0$

### 2. Dataset Construction

- GPT-4o is used to automatically generate 600 prompts in the format "{count} {object}" (e.g., "three cats"), with counts restricted to 1–4 (as models almost completely fail beyond 4).
- 400 prompts are used to construct steering vectors; 200 are reserved for evaluation with no overlap.
- One image is generated per prompt and manually labeled as Class 0 or Class 1; class balance is ensured by regenerating images with different random seeds.
- The cross-attention query vectors from the first $k$ denoising steps are extracted from labeled images as hidden states.

### 3. Adaptive Scaling Mechanism

A fixed steering vector performs inconsistently across different prompts—sometimes under-correcting, sometimes over-correcting. CountSteer introduces an adaptive scaling factor $\alpha_{t,b}$ comprising two key metrics:

**Distance scaling**: The ratio of the distance from the current hidden state $h_{t,b}$ to the target distribution mean:

$$d_{t,b} = \frac{\|\delta_{t,b}\|_2}{\|s_{t,b}\|_2}, \quad \delta_{t,b} = \mu_{t,b}^1 - h_{t,b}$$

**Directional alignment**: Cosine similarity is used to determine whether the steering vector points in the correct direction.

**Combined scaling factor**:

$$\alpha_{t,b} = \cos(s_{t,b}, \delta_{t,b}) \cdot (1 - e^{-d_{t,b}}) \cdot c$$

- When the hidden state is far from the target ($d_{t,b} > 1$), the exponential term approaches 1, permitting stronger correction.
- When the hidden state is close to the target ($d_{t,b} < 1$), the correction weakens automatically, preventing over-correction.
- $c = 100$ is a global amplification constant (since the product of the cosine and exponential terms is inherently small).

### 4. Inference-Time Injection

At each denoising timestep $t$, the adaptive steering vector is injected into the hidden states of each UNet block:

$$h_{t,b}' = h_{t,b} + \alpha_{t,b} \cdot s_{t,b}$$

Injection is applied only during the first 10 denoising steps ($k=10$), consistent with prior findings that global layout and coarse structure are primarily determined in the early denoising stages.

## Key Experimental Results

**Backbone**: Stable Diffusion v1.5, 50 denoising steps, guidance scale 7.5

**Evaluation**: LLaVA-OneVision is used to automatically count objects in generated images.

| Method | ACC ↑ | MAE ↓ | CLIP-Score ↑ |
|--------|-------|-------|-------------|
| SD v1.5 (Baseline) | 50.0% | 1.125 | 30.99 |
| SD v1.5 + CountSteer | **54.0%** | **0.940** | 30.39 |

- Counting accuracy improves by 4.0 percentage points.
- MAE decreases by 0.185, significantly reducing extreme deviations.
- CLIP-Score remains comparable, indicating that semantic alignment and image quality are preserved.

## Highlights & Insights

1. **Discovery of quantity-aware signals within diffusion models**: KDE analysis reveals that cross-attention hidden states are linearly separable between correctly and incorrectly counted samples—an observation that is analytically valuable in its own right.
2. **Fully training-free**: No fine-tuning or architectural modifications are required; steering vectors are injected solely at inference time, resulting in minimal deployment overhead.
3. **Well-motivated adaptive mechanism**: The combination of distance scaling, directional alignment, and exponential decay allows correction strength to be dynamically adjusted based on the current hidden state, avoiding the instability of fixed steering vectors.
4. **Conceptual clarity and methodological simplicity**: Transferring the ITI paradigm from LLMs to diffusion models represents an inspiring cross-domain approach.

## Limitations & Future Work

1. **Limited count range**: The method only supports counts of 1–4; beyond 4, the base model itself almost entirely fails, and steering cannot compensate.
2. **Modest improvement magnitude**: While the 4% accuracy gain is consistent, its absolute value is small—50%→54% still implies that nearly half of generated images have incorrect counts.
3. **Validation limited to SD v1.5**: The absence of experiments on more recent models such as SDXL or Flux leaves generalizability in question.
4. **Clear failure modes**: Three categories of failures are observed—over-generation, rendering failures, and cases where steering corrupts originally correct outputs.
5. **Manual annotation required for dataset construction**: The 400 samples used to build steering vectors require human annotation of counting correctness, limiting automation.
6. **Lack of comparison with concurrent work**: No direct comparison is made against other training-free object counting methods such as Attend-and-Excite or Divide-and-Bind.

## Related Work & Insights

| Method Type | Representative Work | Distinction from CountSteer |
|-------------|--------------------|-----------------------------|
| Fine-tuning | LoRA, DreamBooth | Require additional training; CountSteer is training-free |
| Structural modification | Composer, T2I-Adapter | Require architectural changes; CountSteer leaves the model intact |
| LLM Steering | ITI (Li et al., 2023) | Operates on language model attention heads; CountSteer transfers this to the diffusion UNet |
| Attention manipulation | Attend-and-Excite | Manipulates attention maps to reinforce neglected tokens; CountSteer steers hidden state directions |

**Broader implications**:

- **Generality of the steering paradigm**: From truthfulness control in LLMs to counting control in diffusion models, steering vectors demonstrate cross-modal transferability. The same framework could potentially be extended to compositional attributes such as color consistency, spatial layout, and multi-object interactions.
- **Mining implicit knowledge in diffusion models**: The finding that models "know" what is correct but fail to produce it suggests that activating existing capabilities may be a more efficient research direction than training new ones.
- **Relationship to Classifier-Free Guidance**: CFG guides generation along the conditional/unconditional direction; CountSteer guides counting along the correct/incorrect direction. The two share geometric structural similarities that merit further unification.
- **Practical considerations**: The current 4% improvement has limited standalone application value, but combining CountSteer with layout guidance methods may yield synergistic effects.

## Rating

- Novelty: 3/5 (Transferring LLM steering to diffusion models is novel, though the method itself is relatively intuitive)
- Experimental Thoroughness: 2/5 (Limited to SD v1.5, lacks baseline comparisons, narrow count range)
- Writing Quality: 3/5 (Motivation is clear and structure is complete, but the experimental section is thin)
- Value: 3/5 (Provides valuable analytical findings, but practical improvement is limited)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] FSOD-VFM: Few-Shot Object Detection with Vision Foundation Models and Graph Diffusion](../../ICLR2026/object_detection/fsod-vfm_few-shot_object_detection_with_vision_foundation_models_and_graph_diffu.md)
- [\[CVPR 2026\] InvAD: Inversion-based Reconstruction-Free Anomaly Detection with Diffusion Models](../../CVPR2026/object_detection/invad_inversion-based_reconstruction-free_anomaly_detection_with_diffusion_model.md)
- [\[AAAI 2026\] CountVid: Open-World Object Counting in Videos](open-world_object_counting_in_videos.md)
- [\[ICLR 2026\] Bootstrapping MLLM for Weakly-Supervised Class-Agnostic Object Counting (WS-COC)](../../ICLR2026/object_detection/bootstrapping_mllm_for_weakly-supervised_class-agnostic_object_counting.md)
- [\[AAAI 2026\] Harnessing Vision-Language Models for Time Series Anomaly Detection](harnessing_vision-language_models_for_time_series_anomaly_detection.md)

</div>

<!-- RELATED:END -->

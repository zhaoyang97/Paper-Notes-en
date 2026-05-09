---
title: >-
  [Paper Note] Iris: Breaking GUI Complexity with Adaptive Focus and Self-Refining
description: >-
  [ICCV 2025][Multimodal VLM][GUI understanding] Iris introduces two core innovations — Information-Sensitive Cropping (ISC) and Self-Refining Dual Learning (SRDL) — achieving SOTA on multiple GUI understanding benchmarks with only 850K annotated samples, matching methods that use over 10× more data, while reducing inference time from 3 seconds to 1 second.
tags:
  - ICCV 2025
  - Multimodal VLM
  - GUI understanding
  - visual agent
  - information-sensitive cropping
  - self-refining dual learning
  - element grounding
date: 2026-05-08
content_hash: f3cb5b2f37323798
---

# Iris: Breaking GUI Complexity with Adaptive Focus and Self-Refining

**Conference**: ICCV 2025
**arXiv**: [2412.10342](https://arxiv.org/abs/2412.10342)
**Code**: None
**Area**: Multimodal VLM
**Keywords**: GUI understanding, visual agent, information-sensitive cropping, self-refining dual learning, element grounding

## TL;DR

Iris introduces two core innovations — Information-Sensitive Cropping (ISC) and Self-Refining Dual Learning (SRDL) — achieving SOTA on multiple GUI understanding benchmarks with only 850K annotated samples, matching methods that use over 10× more data, while reducing inference time from 3 seconds to 1 second.

## Background & Motivation

Digital agents are required to autonomously execute tasks in interactive environments such as web pages, software applications, and operating systems. Text-based agents rely on platform-specific APIs and incur high maintenance costs, whereas vision-based agents interact directly with GUIs to achieve cross-platform adaptability and greater scalability. However, vision-based agents face two core challenges:

**Architectural limitations**: GUI interfaces are typically high-resolution (e.g., 1920×1080) with highly uneven information distribution — dense UI elements coexist with large blank regions. Existing methods partition images uniformly, allocating equal tokens to each sub-image, and are therefore unable to adaptively distribute computational resources according to information density. This leads to insufficient perception of fine-grained regions or wasted computation on blank areas.

**Training data bias**: Annotated data is biased toward large, prominent UI elements (e.g., input fields, "OK" buttons), neglecting small but interaction-critical components (e.g., sidebar buttons). This causes models to underperform on complex layouts and fine-grained elements. The high cost of obtaining comprehensive annotations severely limits agent scalability.

The root cause of both problems is that existing methods neither efficiently handle heterogeneous visual information in GUI interfaces nor possess the ability to autonomously learn from difficult samples. Iris addresses these two bottlenecks with ISC and SRDL respectively.

## Method

### Overall Architecture

Iris is built on Qwen-VL as the base model and focuses on two complementary tasks: **Referring** (generating UI element descriptions given a location) and **Grounding** (localizing a UI element given a description). Training proceeds in two stages: first, ISC-enhanced visual training, followed by self-refining training via SRDL.

### Key Designs

1. **Information-Sensitive Cropping (ISC)**

   The core idea of ISC is to perform adaptive cropping based on the distribution of visual information in GUI screenshots, such that each sub-image contains a balanced amount of information. The process consists of three steps:

   - **Information detection**: Canny edge detection is applied to generate a binary information matrix $M \in \{0,1\}^{n \times m}$, where $M_{i,j}=1$ indicates the presence of meaningful visual information at that location (GUI elements typically exhibit clear boundaries).
   - **Adaptive cropping**: A multi-scale sliding window approach is employed, starting from the minimum window size $k_{\min}$ with stride $\text{step}=\max(k/4, 32)$. The edge density of each window is computed, and the density threshold decreases as the window size grows: $\rho_k = \rho_{\min} / (k/k_{\min})^2$. When the density exceeds the threshold, the region is extracted and the processed area is zeroed out to avoid overlap. Window sizes increase geometrically by a factor $\alpha$.
   - **Uniform rescaling**: All cropped sub-images are resized to a uniform resolution (e.g., $224 \times 224$), ensuring that each visual token carries meaningful information.

   ISC requires less than 0.1 seconds on CPU and can be executed in parallel with GPU inference, introducing no additional latency. Compared to uniform partitioning, ISC uses far fewer tokens on simple interfaces and automatically increases tokens for complex interfaces, achieving a **300% efficiency improvement**.

2. **Self-Refining Dual Learning (SRDL)**

   SRDL leverages the complementary relationship between Referring and Grounding to construct a self-reinforcing learning loop. The core pipeline is as follows:

   - **Dual learning loop**: For each UI element in a GUI image, Grounding is first applied to obtain position $\mathbf{p}$; Referring then generates a description $D'$ from that position; Grounding is subsequently applied to $D'$ to obtain a new position. When consecutive positions stabilize (IoU exceeds threshold $\tau$), the sample is considered converged and added to the training set. Formally: $\text{Sim}(G(R(\mathbf{p})), \mathbf{p}) > \tau$.

   - **Visual hard sample mining**: The information matrix $M$ from ISC is used to compute spectral entropy $H = -\sum_k p_k \log(p_k)$, where $p_k$ denotes the normalized energy of frequency components. High spectral entropy corresponds to visually complex regions; such images are prioritized and fed into the dual learning loop for additional training.

   - **Functional hard sample mining**: Based on historical model performance, samples on which the model performs poorly in functional description understanding are collected as $\mathcal{D}_{\text{hard}}$. An LLM is then used to generate description variants $\{D_i^{(1)}, D_i^{(2)}, \ldots, D_i^{(n)}\}$, which serve as synthetic functional hard samples fed into the dual learning loop.

   SRDL ultimately generates approximately **3M self-annotated samples**, yielding a **10% accuracy improvement** without requiring additional human annotation.

### Loss & Training

Training follows the SeeClick pipeline, initialized from Qwen-VL. Initial training uses 850K GUI data combined with 150K LLaVA general visual-language instructions with ISC-enhanced visual perception, followed by the SRDL stage trained on ~3M self-annotated samples.

## Key Experimental Results

### Main Results

**ScreenSpot Benchmark** (GUI element grounding accuracy):

| Model | GUI Annotations | Mobile Text | Mobile Icon | Desktop Text | Desktop Icon | Web Text | Web Icon | **Avg.** |
|-------|----------------|-------------|-------------|--------------|--------------|----------|----------|---------|
| SeeClick | 850K | 78.0 | 52.0 | 72.2 | 30.0 | 55.7 | 32.5 | 53.4 |
| UGround | 10M | 82.8 | 60.3 | 82.5 | 63.6 | 80.4 | 70.4 | 73.3 |
| **Iris** | **850K** | **85.3** | **64.2** | **86.7** | 57.5 | **82.6** | **71.2** | **74.6** |

**GroundUI-1K Benchmark**:

| Model | Web | Desktop | Mobile | Overall |
|-------|-----|---------|--------|---------|
| SeeClick | 64.3 | 44.3 | 73.7 | 61.1 |
| **Iris** | **72.2** | **61.3** | **80.2** | **71.3** |

**Downstream Agent Tasks** (Mind2Web / AITW): Iris achieves the best performance on 11 out of 12 evaluation categories. AITW overall score: 63.6 (SeeClick: 59.3, GPT-4V: 50.5).

### Ablation Study

| Configuration | Accuracy | Inference Time | Notes |
|---------------|----------|----------------|-------|
| Baseline (SeeClick) | ~53% | 0.5s | Baseline |
| + ISC only | ~64% | 1.0s | ISC brings efficiency gains |
| + SRDL w/o Visual Mining | 71.4% | - | Functional mining only |
| + SRDL w/o Functional Mining | 72.1% | - | Visual mining only |
| **Full Iris (ISC + SRDL)** | **74.6%** | **1.0s** | Complementary; optimal |

### Key Findings

- ISC achieves high accuracy with fewer tokens on low-complexity interfaces and automatically increases token allocation on high-complexity interfaces, consistently outperforming AnyRes across all settings.
- Both visual and functional hard sample mining in SRDL are indispensable; their combination outperforms either alone by 2.5%–3.2%.
- The most significant gains are observed on Web and Desktop platforms, where higher resolution and more complex layouts better demonstrate the advantages of ISC.

## Highlights & Insights

- **Exceptional data efficiency**: Iris matches UGround (10M annotations) using only 850K annotations, primarily because SRDL autonomously identifies and learns from difficult samples, compensating for annotation bias.
- **Elegant ISC design**: Edge detection-based information density estimation is simple and efficient (<0.1s on CPU), enabling adaptive cropping without increasing inference latency.
- **Self-consistency of dual learning**: Referring and Grounding serve as mutual validators, forming a natural data quality filtering mechanism — only converged samples are admitted to training.

## Limitations & Future Work

- Discrimination between visually highly similar UI elements (e.g., button groups with nearly identical colors and shapes) may remain limited.
- The quality of self-annotated data generated by SRDL depends on the capacity of the initial model; a weak initial model may lead to a low-quality training loop.
- Cross-resolution and cross-device transfer capabilities have not been explored.
- Handling of dynamic GUIs (e.g., pop-ups, animations) is not discussed.

## Related Work & Insights

- SeeClick first established GUI Grounding as a foundational capability for visual agents; Iris builds upon this foundation to achieve a qualitative breakthrough with substantially higher data efficiency.
- The adaptive token allocation concept underlying ISC can be generalized to other high-resolution visual tasks, such as remote sensing and medical imaging.
- The dual learning loop in SRDL resembles self-training in semi-supervised learning, but achieves more natural quality control through task complementarity.

## Rating

- **Novelty**: ⭐⭐⭐⭐ ISC and SRDL are not entirely novel concepts individually, but their combination in the GUI setting is particularly elegant
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers both GUI Grounding and downstream agent tasks with clear ablations
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured paper with effective visualizations
- **Value**: ⭐⭐⭐⭐⭐ Significant contribution to the GUI visual agent field; the improvement in data efficiency is of considerable importance

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] NegRefine: Refining Negative Label-Based Zero-Shot OOD Detection](negrefine_refining_negative_label-based_zero-shot_ood_detection.md)
- [\[ICCV 2025\] Acknowledging Focus Ambiguity in Visual Questions](acknowledging_focus_ambiguity_in_visual_questions.md)
- [\[ICLR 2026\] Breaking the Limits of Open-Weight CLIP: An Optimization Framework for Self-supervised Fine-tuning of CLIP](../../ICLR2026/multimodal_vlm/breaking_the_limits_of_open-weight_clip_an_optimization_framework_for_self-super.md)
- [\[ICCV 2025\] Mastering Collaborative Multi-modal Data Selection: A Focus on Informativeness, Uniqueness, and Representativeness](mastering_collaborative_multi-modal_data_selection_a_focus_on_informativeness_un.md)
- [\[ICCV 2025\] SC-Captioner: Improving Image Captioning with Self-Correction by Reinforcement Learning](sc-captioner_improving_image_captioning_with_self-correction_by_reinforcement_le.md)

<!-- RELATED:END -->

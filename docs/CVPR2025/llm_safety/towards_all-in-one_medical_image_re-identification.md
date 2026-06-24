---
title: >-
  [Paper Note] Towards All-in-One Medical Image Re-Identification
description: >-
  [CVPR 2025][LLM Safety][Medical Image Re-Identification] This paper proposes MaMI, the first all-in-one unified model for medical image re-identification. It dynamically generates modality-specific parameters via a Continuous-modality Parameter Adapter (ComPA) and transfers medical priors using difference feature alignment from medical foundation models. MaMI outperforms 25 foundation models and 8 large language models across 11 datasets.
tags:
  - "CVPR 2025"
  - "LLM Safety"
  - "Medical Image Re-Identification"
  - "Modality Adaptation"
  - "Parameter Adapter"
  - "Medical Priors"
  - "Privacy Protection"
date: 2026-05-08
content_hash: 4b0a71f1b8b1ef92
---

# Towards All-in-One Medical Image Re-Identification

**Conference**: CVPR 2025  
**arXiv**: [2503.08173](https://arxiv.org/abs/2503.08173)  
**Code**: [GitHub](https://github.com/tianyuan168326/All-in-One-MedReID-Pytorch)  
**Area**: Medical Images  
**Keywords**: Medical Image Re-Identification, Modality Adaptation, Parameter Adapter, Medical Priors, Privacy Protection

## TL;DR

This paper proposes MaMI, the first all-in-one unified model for medical image re-identification. It dynamically generates modality-specific parameters via a Continuous-modality Parameter Adapter (ComPA) and transfers medical priors using difference feature alignment from medical foundation models. MaMI outperforms 25 foundation models and 8 large language models across 11 datasets.

## Background & Motivation

Medical Image Re-Identification (MedReID) is crucial for personalized healthcare and privacy protection, yet remains heavily under-explored:

1. **Limitations of prior work to a single modality**: For example, Packhäuser et al. only process chest X-rays, failing to share knowledge across modalities.
2. **Challenges of a unified model**: Naive joint multimodal training benefits some modalities (e.g., fundus images 76.88% $\rightarrow$ 82.48%) but harms others (e.g., X-ray 94.21% $\rightarrow$ 92.30%) due to neglecting modality-specific knowledge.
3. **Lack of medical priors**: Models tend to overfit to shallow textures such as machine noise rather than learning anatomical features.
4. **Real-world applications**: Both historical medical record retrieval and medical image privacy protection demand robust MedReID capabilities.

This study defines and constructs a comprehensive benchmark for MedReID.

## Method

### Overall Architecture

MaMI is based on ViT-Base (CLIP pre-trained) and consists of two core modules: (1) Continuous-modality Parameter Adapter (ComPA) dynamically adjusts network parameters according to input images; (2) Medical Prior Transfer module inherits anatomical knowledge from medical foundation models via difference feature alignment.

### Key Designs

**1. Continuous-modality Parameter Adapter (ComPA)**

- **Function**: Dynamically adapts the modality-agnostic model into a modality-specific model at runtime.
- **Mechanism**: Projects each $16 \times 16$ patch into a local modality context through an MLP, which is averaged to obtain a global modality context. Another MLP generates a modality probability vector $\mathbf{w} \in \mathbb{R}^L$ ($L=32$ pseudo-modalities), facilitating the weighted summation of learnable modality bases $\Omega \in \mathbb{R}^{L \times 768}$ to yield continuous modality features $\mathcal{M}_i$. Att-PNet and FFN-PNet generate low-rank parameters (LoRA format) from $\mathcal{M}_i$, which are integrated into the ViT layers.
- **Design Motivation**: $L=32$ exceeds the number of actual modalities because different devices and parameters within the same modality produce various imaging styles. Continuous representations offer more flexibility than discrete labels and exhibit superior generalization to out-of-domain images.

**2. Difference Feature-Based Medical Prior Alignment**

- **Function**: Transfers anatomical knowledge from pre-trained medical foundation models (MFMs) to the ReID task.
- **Mechanism**: Generates $N$ modality-specific query tokens from the modality feature $\mathcal{M}_i$, extracting critical features $\mathbf{P}_i^n$ from the feature map via attention. For any image pair $(x_i, x_j)$, the difference $\mathbf{u}^n = \mathbf{P}_i^n - \mathbf{P}_j^n$ and MFM difference $\mathbf{v}^n = \mathbf{Q}_i^n - \mathbf{Q}_j^n$ are computed and aligned using a contrastive loss.
- **Design Motivation**: Single-image feature alignment is inconsistent with ReID tasks (which require distinguishing multiple images), whereas difference feature alignment aligns better with the nature of the task. Query tokens depend on modality features to focus on key structures across different modalities.

**3. Modality-specific Query Tokens**

- **Function**: Automatically focuses on various critical anatomical structures according to the modality.
- **Mechanism**: Generates $N$ query tokens $\mathbf{O}_i$ from $\mathcal{M}_i$ via a three-layer MLP, which are pooled to obtain key features through attention $\mathbf{A}_i^n = \text{Softmax}(\frac{\mathbf{O}_i^n \cdot \text{Linear}(\mathbf{f}_i)}{\sqrt{d}})$.
- **Design Motivation**: Identity clues in X-rays reside in rib/heart shapes, whereas clues in fundus images lie in the optic disc/vessels. Thus, distinct modalities require focusing on different regions.

### Loss & Training

$$\mathcal{L} = \mathcal{L}_{id\text{-}classify} + \mathcal{L}_{tri} + \lambda \mathcal{L}_{med\text{-}align}$$

$\mathcal{L}_{id\text{-}classify}$ is the identity classification cross-entropy loss, $\mathcal{L}_{tri}$ is the soft-margin triplet loss, $\mathcal{L}_{med\text{-}align}$ is the difference feature contrastive loss, and $\lambda = 0.01$.

## Key Experimental Results

### Comparison with Foundation Models and LLMs (CMC-R1 %)

| Method | MIMIC-X | CCII-CT | EyePACS | Chest-X | OASIS-MRI | Average |
|------|---------|---------|---------|---------|-----------|------|
| CLIP | 33.10 | 58.82 | 41.14 | - | - | - |
| BiomedCLIP | 42.30 | 68.91 | 53.48 | - | - | - |
| GPT-4V | 12.50 | 15.00 | 10.20 | - | - | - |
| Single-modality | 94.21 | - | 76.88 | - | - | - |
| **MaMI (Ours)** | **96.89** | **95.59** | **85.71** | **91.23** | **80.00** | - |

### Ablation Study

| Method | X-ray (%) | Fundus (%) |
|------|-----------|-----------|
| CLIP baseline | 33.10 | 41.14 |
| Single-modality | 94.21 | 76.88 |
| Multiple-modality | 92.30 | 82.48 |
| **Continuous-modality (Ours)** | **96.89** | **85.71** |

### Key Findings

- The unified MaMI model simultaneously outperforms single-modality specialized models (X-ray +2.68%, Fundus +8.83%).
- The ComPA continuous-modality strategy outperforms naive joint multimodal training across both modalities.
- Difference feature alignment improves CMC-R1 by ~2% compared to single-image feature alignment.
- Among 25 foundation models, the best-performing BiomedCLIP achieves only 42.30% (X-ray), indicating that existing general foundation models are highly inadequate for MedReID.

## Highlights & Insights

1. **Significant contribution to problem definition**: First to systematically define the MedReID problem and establish a comprehensive benchmark.
2. **Elegant ComPA design**: Combines continuous modality representation with dynamic LoRA generation, achieving both flexibility and efficiency.
3. **Ingenious difference feature alignment**: The methodology for adapting MFM priors to the ReID task serves as a valuable reference.
4. **Practical value**: Demonstrates real-world deployment value in personalized diagnosis and privacy protection.

## Limitations & Future Work

- The setting of $L=32$ pseudo-modalities might be insufficient when the actual number of modalities scales up significantly.
- Multi-slice scans are processed using simple averaging, which discards spatial information across slices.
- Incorporating 3D convolutions or more advanced inter-slice modeling remains unexplored.
- The implications of identity removal for privacy protection require thorough security validation.

## Related Work & Insights

- **FisherRF**: The first work to apply ReID to chest X-rays.
- **CLIP/BiomedCLIP**: General vision-language models yield suboptimal performance on MedReID.
- **LoRA/MOE-LoRA**: Parameter-efficient fine-tuning methods. ComPA surpasses static LoRA configurations via runtime dynamic generation.

## Rating

⭐⭐⭐⭐⭐ — Highly novel and practical problem definition, robust method design, and extensive evaluation (11 datasets, 25+ baselines). Both ComPA and difference feature alignment represent high-quality contributions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Federated CLIP for Resource-Efficient Heterogeneous Medical Image Classification](../../AAAI2026/llm_safety/federated_clip_for_resource-efficient_heterogeneous_medical_image_classification.md)
- [\[ICLR 2026\] Train Once, Answer All: Many Pretraining Experiments for the Cost of One](../../ICLR2026/llm_safety/train_once_answer_all_many_pretraining_experiments_for_the_cost_of_one.md)
- [\[NeurIPS 2025\] Watermarking Autoregressive Image Generation](../../NeurIPS2025/llm_safety/watermarking_autoregressive_image_generation.md)
- [\[ICLR 2026\] Attention Smoothing Is All You Need For Unlearning](../../ICLR2026/llm_safety/attention_smoothing_is_all_you_need_for_unlearning.md)
- [\[ACL 2025\] Core: Robust Factual Precision with Informative Sub-Claim Identification](../../ACL2025/llm_safety/core_robust_factual_precision_with_informative_sub-claim_identification.md)

</div>

<!-- RELATED:END -->

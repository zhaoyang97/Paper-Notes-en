---
title: >-
  [Paper Note] ASAP: Advancing Semantic Alignment Promotes Multi-Modal Manipulation Detecting and Grounding
description: >-
  [CVPR 2025][Multimodal VLM][Multi-Modal Manipulation Detection] This paper proposes the ASAP framework, which systematically advances image-text semantic alignment to improve multi-modal manipulation detection and grounding performance through three core modules: Large Model-Assisted Alignment (LMA), Manipulation-Guided Cross-Attention (MGCA), and Patch Manipulation Modeling (PMM). It achieves a 94.38% AUC and 76.52% text grounding F1 on the DGM4 benchmark…
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Multi-Modal Manipulation Detection"
  - "Semantic Alignment"
  - "Cross Attention"
  - "DGM4"
  - "Hard Negative Mining"
date: 2026-05-08
content_hash: 2aa511447d3c45cb
---

# ASAP: Advancing Semantic Alignment Promotes Multi-Modal Manipulation Detecting and Grounding

**Conference**: CVPR 2025  
**arXiv**: [2412.12718](https://arxiv.org/abs/2412.12718)  
**Code**: [https://github.com/CriliasMiller/ASAP](https://github.com/CriliasMiller/ASAP)  
**Area**: Robotics  
**Keywords**: Multi-Modal Manipulation Detection, Semantic Alignment, Cross Attention, DGM4, Hard Negative Mining  
**Authors**: Zhenxing Zhang, Yaxiong Wang et al. (Hefei University of Technology)

## TL;DR
This paper proposes the ASAP framework, which systematically advances image-text semantic alignment to improve multi-modal manipulation detection and grounding performance through three core modules: Large Model-Assisted Alignment (LMA), Manipulation-Guided Cross-Attention (MGCA), and Patch Manipulation Modeling (PMM). It achieves a 94.38% AUC and 76.52% text grounding F1 on the DGM4 benchmark, significantly outperforming existing methods.

## Background & Motivation
With the rapid development of AIGC technologies (such as diffusion models and LLMs), high-quality manipulated image-text content is becoming increasingly easy to generate, posing a serious threat to the credibility of social media information. The multi-modal manipulation detection task requires simultaneously detecting manipulated regions in both images and text. It not only demands determining whether the overall content is manipulated (classification) but also requires precisely locating the manipulated image regions and text segments (grounding).

Although existing methods (such as HAMMER) have made some progress, they still face two core bottlenecks:
1. **Insufficient Image-Text Semantic Alignment**: The key to manipulation detection lies in discovering semantic inconsistencies between images and text. However, prior alignment learning is insufficient, making it difficult to capture fine-grained semantic discrepancies.
2. **Limited Region-Level Grounding Accuracy**: Image manipulation grounding typically requires precise patch-level judgment, and existing methods lack effective region-level supervision and hard negative mining mechanisms.

## Core Problem
How to systematically enhance image-text semantic alignment, enabling the model to detect and ground manipulated regions in multi-modal content more accurately?

## Method

### Overall Architecture
ASAP is built upon the CLIP dual-encoder architecture and consists of three core modules. The total loss function is:

35714L = L_{DGM} + L_{LMA} + \alpha \cdot L_{MGCA} + \lambda \cdot L_{PMM}35714

where $\alpha=0.1$ and $\lambda=0.01$. $L_{DGM}$ is the basic DGM4 multi-task loss.

### Module 1: Large Model-Assisted Alignment (LMA)
The LMA module leverages pre-trained large models to generate rich textual descriptions, thereby enhancing vision-language alignment learning:

1. **Image Caption Generation**: Uses a Multi-modal Large Language Model (MLLM, e.g., InstructBLIP) to generate detailed visual descriptions (captions) for each image.
2. **Manipulation Explanation Generation**: Uses a Large Language Model (LLM, e.g., ChatGLM) to generate a manipulation explanation based on the original and manipulated text pairs, describing where and what kind of manipulation occurred in the text.
3. **Contrastive Learning Alignment**: Constructs vision-caption pairs and text-explanation pairs for contrastive learning to pull matching pairs closer and push mismatched pairs further apart. The loss is:

35714L_{LMA} = L_{cap} + L_{exp}35714

| Text Type | Generative Model | Purpose | Example |
|---------|---------|-----|------|
| Caption | InstructBLIP (MLLM) | Image-to-text semantic bridging | A man standing next to a red car |
| Explanation | ChatGLM (LLM) | Describing specific manipulation changes | Replacing the red car with a blue truck |
| Original Text | Provided by Dataset | Benchmark text | A man standing next to a red car |
| Manipulated Text | Provided by Dataset | Detection target | A man standing next to a blue truck |

### Module 2: Manipulation-Guided Cross-Attention (MGCA)
The MGCA module enhances cross-modal attention towards manipulated regions through explicit guidance from manipulated areas:

1. **Guided Mask Generation**: Generates a binary guidance mask $M_g$ based on the image manipulation annotations, marking which patches are manipulated.
2. **Mask-Enhanced Attention**: Modulates attention weights using the guidance mask on top of standard cross-attention, causing the model to focus more on manipulation-related regions:

35714Attn_{MGCA} = \text{softmax}(\frac{QK^T}{\sqrt{d}} + \beta \cdot M_g)35714

3. **Auxiliary Loss**: An additional cross-attention alignment loss $L_{MGCA}$ encourages attention weights to concentrate on the actual manipulated areas.

### Module 3: Patch Manipulation Modeling (PMM)
The PMM module improves region-level manipulation grounding accuracy through a hard negative patch selection strategy:

1. **Hard Negative Patch Selection (HNP)**: In each batch, patches that are visually most similar to the manipulated patches but themselves unmanipulated are selected as hard negative samples.
2. **Contrastive Learning**: Constructs a contrastive learning objective at the patch level to bring patch representations within the same manipulated region closer while pushing them away from hard negative representations.
3. **Region Grounding Enhancement**: Through the HNP strategy, the model learns to distinguish visually similar but semantically different patches, thereby improving grounding accuracy.

35714L_{PMM} = -\log \frac{\exp(sim(z_i^+, z_i) / \tau)}{\exp(sim(z_i^+, z_i) / \tau) + \sum_j \exp(sim(z_j^-, z_i) / \tau)}35714

## Key Experimental Results

### Main Results
A comprehensive comparison with major methods on the DGM4 benchmark dataset:

| Method | AUC (%) | ACC (%) | mAP (%) | Text F1 (%) | Image IoU (%) |
|-----|---------|---------|---------|-----------|------------|
| HAMMER | 93.09 | 86.42 | 87.20 | 72.22 | 76.10 |
| DGM4-baseline | 91.56 | 84.90 | 85.03 | 70.15 | 74.82 |
| MFCLIP | 92.47 | 85.88 | 86.51 | 71.68 | 75.63 |
| **ASAP (Ours)** | **94.38** | **87.71** | **88.53** | **76.52** | **77.35** |
| ASAP vs HAMMER | +1.29 | +1.29 | +1.33 | +4.30 | +1.25 |

Key Findings:
- Outperforms existing state-of-the-art methods consistently across all five metrics.
- The improvement in Text F1 is the most significant (+4.30%), indicating that semantic alignment is particularly critical for text manipulation grounding.
- Synchronous improvements in AUC and ACC demonstrate the enhancement of overall detection capabilities.

### Performance on Different Manipulation Types

| Manipulation Type | AUC (%) | Text F1 (%) | Image IoU (%) |
|---------|---------|-----------|------------|
| Text Manipulation Only | 95.12 | 79.83 | - |
| Image Manipulation Only | 93.67 | - | 78.91 |
| Joint Image-Text Manipulation | 94.45 | 73.72 | 75.23 |

Joint image-text manipulation is the most challenging, but ASAP still maintains high performance.

## Ablation Study

### Contribution of Each Module
Performance shifts when progressively adding each module:

| Setting | AUC (%) | ACC (%) | Text F1 (%) | Image IoU (%) |
|------|---------|---------|-----------|------------|
| Baseline (DGM4) | 93.16 | 86.01 | 72.05 | 75.88 |
| + LMA | 94.28 | 87.30 | 75.41 | 76.72 |
| + LMA + MGCA | 94.40 | 87.55 | 76.18 | 77.10 |
| + LMA + MGCA + PMM (Full ASAP) | 94.38 | 87.71 | 76.52 | 77.35 |

Key observations:
- LMA contributes the most (AUC +1.12%), proving that large model-assisted semantic alignment learning is the core.
- MGCA further improves grounding accuracy (Text F1 +0.77%).
- PMM primarily improves image grounding (IoU +0.25%) and text grounding (F1 +0.34%), with a slight fluctuation in AUC (-0.02%), indicating that PMM focuses more on grounding rather than classification.

### Influence of Different Text Types in LMA

| Setting | AUC (%) | Text F1 (%) | Image IoU (%) |
|------|---------|-----------|------------|
| Without LMA | 93.16 | 72.05 | 75.88 |
| Caption Only | 93.85 | 74.12 | 76.30 |
| Explanation Only | 93.72 | 73.88 | 76.15 |
| Caption + Explanation | 94.28 | 75.41 | 76.72 |

Caption and Explanation provide mutually complementary information, and their joint use yields the best performance. Caption offers visual semantic descriptions, while Explanation provides explicit descriptions of manipulation operations.

### Influence of HNP Strategy
- With HNP: Image IoU 77.35%
- Without HNP: Image IoU 76.92% (-0.43%)
- By mining visually similar but semantically distinct hard negatives, HNP effectively improves grounding accuracy in boundary regions.

## Summary & Review

### Strengths
1. **Systematic Design**: Three modules enhance semantic alignment from different perspectives, complementing each other.
2. **Large Model Empowerment**: Ingeniously utilizes MLLMs and LLMs to generate auxiliary texts, eliminating the need for additional manual annotation.
3. **Comprehensive Superiority**: Achieves state-of-the-art results across all metrics in both detection and grounding levels.

### Limitations & Future Work
1. **Inference Efficiency**: Although the large-model generation in the LMA module can be completed offline during training, it increases data preprocessing costs.
2. **Dependence on Guidance Mask**: MGCA requires manipulation region annotations during training, which limits its applicability to unlabeled scenarios.
3. **Single Dataset**: The experiments are only validated on the DGM4 benchmark, and generalization to other multi-modal manipulation detection datasets remains to be verified.

### Insights
- Utilizing large models to generate auxiliary signals for alignment learning is a low-cost yet highly efficient enhancement strategy.
- The importance of hard negative mining in region-level tasks is re-verified.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] ASAP: Advancing Semantic Alignment for Multi-Modal Manipulation Detection](asap_advancing_semantic_alignment_promotes_multi-modal_manipulation_detecting_an.md)
- [\[CVPR 2025\] Can Large Vision-Language Models Correct Semantic Grounding Errors By Themselves?](can_large_vision-language_models_correct_semantic_grounding_errors_by_themselves.md)
- [\[CVPR 2025\] Topo-R1: Detecting Topological Anomalies via Vision-Language Models](topo-r1_detecting_topological_anomalies_via_vision-language_models.md)
- [\[CVPR 2025\] GeoMM: On Geodesic Perspective for Multi-Modal Learning](geomm_on_geodesic_perspective_for_multi-modal_learning.md)
- [\[CVPR 2025\] EgoLM: Multi-Modal Language Model of Egocentric Motions](egolm_multi-modal_language_model_of_egocentric_motions.md)

</div>

<!-- RELATED:END -->

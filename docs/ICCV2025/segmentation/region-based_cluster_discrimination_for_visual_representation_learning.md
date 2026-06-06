---
title: >-
  [Paper Note] Region-based Cluster Discrimination for Visual Representation Learning
description: >-
  [ICCV 2025][Segmentation][Region representation learning] This paper proposes RICE (Region-Aware Cluster Discrimination), which constructs a billion-scale region dataset, designs a Region Transformer layer…
tags:
  - "ICCV 2025"
  - "Segmentation"
  - "Region representation learning"
  - "cluster discrimination"
  - "OCR-awareness"
  - "visual encoder"
  - "multimodal large language models"
date: 2026-05-08
content_hash: a0649822d15c6cfe
---

# Region-based Cluster Discrimination for Visual Representation Learning

**Conference**: ICCV 2025
**arXiv**: [2507.20025](https://arxiv.org/abs/2507.20025)  
**Code**: [GitHub](https://github.com/deepglint/MVT)  
**Area**: Image Segmentation
**Keywords**: Region representation learning, cluster discrimination, OCR-awareness, visual encoder, multimodal large language models

## TL;DR

This paper proposes RICE (Region-Aware Cluster Discrimination), which constructs a billion-scale region dataset, designs a Region Transformer layer, and introduces a unified region cluster discrimination loss to jointly optimize object-aware and OCR capabilities, significantly improving visual encoder performance across segmentation, detection, and MLLM multi-task benchmarks.

## Background & Motivation

Vision-language contrastive models such as CLIP and SigLIP have learned powerful global visual representations through large-scale image-text alignment. However, these models exhibit limited performance on **dense prediction tasks** (segmentation, localization, OCR) due to the following reasons:

**Semantic insufficiency in instance discrimination**: The partitioning of positive and negative sample pairs ignores semantic similarity, treating semantically similar samples from different instances as negative pairs.

**OCR interference with high-level semantics**: When OCR pairs are included in vision-language contrastive training, the visual encoder tends to focus on text recognition at the expense of object semantics.

**Limitations of global representations for local modeling**: Existing cluster discrimination methods (e.g., UNICOM, MLCD) assign one or more pseudo-labels per image, precluding the learning of local region-level representations.

Existing region-level methods such as RegionCLIP and CLIM rely on region-text descriptions, limiting their scalability. The **core motivation** is to replace text descriptions with cluster centers as region supervision signals, enabling large-scale and scalable region-level representation learning.

## Method

### Overall Architecture

RICE adopts a ViT backbone, appending Region Transformer layers after the standard Transformer layers to extract both global and region-level semantics in a single forward pass. Training supervision comes from two branches:

- **Object Region Loss**: Single-label classification based on cluster centers
- **OCR Region Loss**: Multi-label classification based on token embeddings

### Key Design 1: Region Data Construction

**Object region data**: Sampled from LAION-2B, COYO-700M, and SAM-1B. SAM is applied to LAION and COYO to generate fine-grained masked regions; candidate boxes with shortest side ≥128px are retained, yielding 400 million images and 2 billion candidate regions. Region features are extracted via CLIP and clustered into 1 million semantic centers using k-means:

$$\mathbf{y}_{i,j}^{object} = \arg\min_{k \in [1,K]} \|\mathbf{f}_{i,j} - \mathbf{c}_k\|_2$$

Clustering employs hierarchical k-means with Faiss GPU, completing in approximately 10 hours on 64 GPUs.

**OCR region data**: PaddleOCR is applied to LAION-2B and COYO-700M to extract text (confidence >0.7), yielding 50 million images and 400 million candidate regions. Extracted text is tokenized to produce OCR labels.

### Key Design 2: Region Transformer Layer

**Region sampling**: The number of regions per image is normalized to $N$. If the actual count exceeds $N$, regions are randomly subsampled; otherwise, they are sampled with replacement.

**Region attention**:
- The number of tokens per region varies with spatial size, making direct batch processing difficult.
- A **region visibility mask** $\mathcal{M}$ is introduced: tokens inside the region are set to 0 and tokens outside are set to $-\infty$.
- Region attention is computed as:

$$\mathcal{R}_{\text{batch}} = \sigma\left(\frac{\mathbf{Q}_{\text{batch}} \mathbf{K}_{\text{batch}}^\top}{\sqrt{d_k}} + \mathcal{M}\right) \mathbf{V}_{\text{batch}}$$

- Fixed-length embeddings for all regions are extracted in a single forward pass.

### Key Design 3: Region Cluster Discrimination Loss

**Object region loss** (single-label classification):

$$\mathcal{L}_{object} = \log(1 + \exp(-sim(\hat{\mathbf{y}}_{i,j}, \mathbf{y}_{i,j}^{object}))) + \log(1 + \sum_{j \in \Omega_n^{object}} \exp(sim(\hat{\mathbf{y}}_{i,j}, \mathbf{y}_{i,j}^{object})))$$

**OCR region loss** (multi-label classification): Each OCR region has multiple token embeddings as positives:

$$\mathcal{L}_{ocr} = \log(1 + \sum_{j \in \Omega_p^{ocr}} \exp(-sim)) + \log(1 + \sum_{j \in \Omega_n^{ocr}} \exp(sim))$$

**Negative sampling strategy**: Negative cluster centers are uniformly sampled from the full category set at rate $\rho=0.1$, reducing semantically conflicting gradients and improving training stability.

## Experiments

### MLLM Multimodal Understanding (LLaVA-NeXT Framework)

| Vision Tower | LLM | DocVQA | OCRBench | InfoVQA | MM-Bench | Other Avg |
|-------------|-----|:------:|:--------:|:-------:|:--------:|:-------:|
| CLIP ViT-L-336px | Qwen2.5-7B | 75.21 | 525 | 38.88 | 74.57 | 69.83 |
| SigLIP SO400M-384px | Qwen2.5-7B | 76.71 | 554 | 41.38 | 76.98 | 70.62 |
| AIMv2 ViT-L-336px | Qwen2.5-7B | 77.19 | 572 | 35.44 | 78.61 | 70.58 |
| **RICE ViT-L-336px** | Qwen2.5-7B | **79.19** | **575** | **45.23** | 76.55 | **73.03** |

RICE-336px consistently leads on OCR tasks: +50 points on OCRBench and +3.98% on DocVQA over CLIP.

### Referring Image Segmentation (LLaVA-NeXT + LISA)

| Vision Tower | LLM | RefCOCO val | RefCOCO+ val | RefCOCOg val |
|-------------|-----|:----------:|:----------:|:-----------:|
| CLIP | Qwen2.5-7B | 81.8 | 76.6 | 77.3 |
| MLCD | Qwen2.5-7B | 82.8 | 77.4 | 78.5 |
| **RICE** | Qwen2.5-7B | **83.5** | **79.4** | **79.8** |

RICE surpasses both CLIP and MLCD on all RefCOCO benchmarks, with mean IoU improvements of +2.45 (vs. CLIP) and +1.30 (vs. MLCD).

### Key Findings

1. Region cluster discrimination yields a pronounced advantage on dense OCR tasks (InfoVQA +9.79 vs. AIMv2), as the joint training objective avoids conflicts between object semantics and OCR.
2. t-SNE visualizations demonstrate that RICE produces substantially better object feature clustering than DINOv2, MLCD, and SigLIP.
3. A random negative sampling rate of $\rho=0.1$ is optimal; higher rates introduce semantically conflicting gradients.
4. Placing the Region Transformer layers in the last few layers is optimal, balancing global context and region-level precision.

## Highlights & Insights

- **Data engineering at scale**: The construction of 2 billion regions with 1 million cluster centers is ambitious; replacing text descriptions with cluster centers enables text-free region supervision.
- **Unified framework**: Object recognition and OCR tasks are jointly trained within a single classification framework, avoiding multi-task conflicts.
- **Plug-and-play compatibility**: The RICE visual encoder can directly replace CLIP in frameworks such as LLaVA without architectural modifications.

## Limitations & Future Work

- Data construction depends on the quality of SAM and PaddleOCR; annotation noise may propagate into cluster centers.
- Storage and update costs for 1 million cluster centers are non-trivial.
- Evaluation is limited to ViT-L and ViT-B scales; performance at larger scales (e.g., ViT-G) remains unexplored.

## Related Work & Insights

- Instance discrimination: Visual representation learning methods such as CLIP, SigLIP, and DINOv2.
- Cluster discrimination: Clustering-based self-supervised methods such as DeepCluster, SwAV, UNICOM, and MLCD.
- Region representation: Region-language alignment methods such as RegionCLIP, CLIM, and GLIP.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The region cluster discrimination approach is original; the unified Object+OCR design is meaningful.
- **Technical Depth**: ⭐⭐⭐⭐ — The Region Transformer and loss function designs are complete; data engineering is rigorous.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers MLLM, segmentation, detection, and OCR with thorough ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is clear and the architecture diagram is intuitive.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Implicit Counterfactual Learning for Audio-Visual Segmentation](implicit_counterfactual_learning_for_audio-visual_segmentation.md)
- [\[ICCV 2025\] Hierarchical Visual Prompt Learning for Continual Video Instance Segmentation](hierarchical_visual_prompt_learning_for_continual_video_instance_segmentation.md)
- [\[ICML 2026\] SEMIR: Semantic Minor-Induced Representation Learning on Graphs for Visual Segmentation](../../ICML2026/segmentation/semir_semantic_minor-induced_representation_learning_on_graphs_for_visual_segmen.md)
- [\[ICCV 2025\] On the Generalization of Representation Uncertainty in Earth Observation](on_the_generalization_of_representation_uncertainty_in_earth_observation.md)
- [\[CVPR 2026\] AFRO: Bootstrap Dynamic-Aware 3D Visual Representation for Scalable Robot Learning](../../CVPR2026/segmentation/bootstrap_dynamic-aware_3d_visual_representation_for_scalable_robot_learning.md)

</div>

<!-- RELATED:END -->

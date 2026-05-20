---
title: >-
  [Paper Note] NaiLIA: Multimodal Nail Design Retrieval Based on Dense Intent Descriptions and Palette Queries
description: >-
  [CVPR2026][multimodal retrieval] This paper proposes NaiLIA, a multimodal retrieval method for nail design images that achieves fine-grained matching via dense intent descriptions and palette queries. A confidence-based…
tags:
  - "CVPR2026"
  - "multimodal retrieval"
  - "dense intent description"
  - "palette query"
  - "contrastive learning"
  - "unlabeled positive"
  - "fashion AI"
  - "nail design"
date: 2026-05-08
content_hash: a84be19d1c116b5e
---

# NaiLIA: Multimodal Nail Design Retrieval Based on Dense Intent Descriptions and Palette Queries

**Conference**: CVPR2026
**arXiv**: [2603.05446](https://arxiv.org/abs/2603.05446)  
**Code**: [Project Page](https://nailia-94dpr.kinsta.page/)  
**Area**: Others (Multimodal Retrieval / Fashion AI)
**Keywords**: multimodal retrieval, dense intent description, palette query, contrastive learning, unlabeled positive, fashion AI, nail design

## TL;DR

This paper proposes NaiLIA, a multimodal retrieval method for nail design images that achieves fine-grained matching via dense intent descriptions and palette queries. A confidence-based relaxed contrastive (CRC) loss is introduced to handle unlabeled positives. NaiLIA substantially outperforms existing methods on the authors' newly constructed NAIL-STAR benchmark and on Marqo Fashion200K.

## Background & Motivation

**Market Demand**: The global nail salon market is valued at approximately $11 billion. Users have a strong demand for searching nail design images based on personal preferences, yet existing retrieval systems struggle to handle multi-level user intent.

**Retrieval over Generation**: Nail technicians report that AI-generated images frequently violate physical constraints (e.g., unrealizable decorative accessories), and multiple beauty platforms have restricted their use. Retrieval from real images is thus more practical.

**Challenges of Dense Intent Descriptions**: User descriptions typically encompass drawing elements (patterns), decorative elements (accessories), themes (e.g., "mermaid style"), and overall impressions (e.g., "dreamlike"), posing significant challenges to existing vision-language models due to their multi-level abstraction.

**Inadequacy of Color Expression**: Subtle color differences are critical in fashion, yet existing methods ignore continuous color inputs (e.g., RGB values) and rely solely on text descriptions, which cannot precisely convey color preferences.

**Inherent Limitations of InfoNCE Loss**: Existing vision-language foundation models (CLIP, SigLIP, etc.) rely on InfoNCE loss, treating all non-positive samples as negatives. However, nail design images frequently contain numerous similar unlabeled positives, causing the similarity of similar samples to be erroneously minimized.

**Abstraction Level Bias**: Existing models tend to retrieve results at a specific abstraction level (typically biased toward realism); for example, interpreting "shell-inspired design" as a real shell decoration rather than an artistically stylized shell-themed design.

## Method

### Overall Architecture

NaiLIA consists of three core modules:

- **Intent-Palette Fusion Module (IPFM)**: Fuses dense intent descriptions with palette queries.
- **Visual Design Fusion Module (VDFM)**: Fuses three visual representations for comprehensive understanding of nail design images.
- **Confidence-based Relaxed Alignment Module (CRAM)**: Estimates confidence scores for unlabeled positives and incorporates them into a relaxed loss.

The input is defined as $\bm{x} = \{\bm{x}_{\text{txt}}, \bm{x}_{\text{pal}}, X_{\text{img}}\}$, where $\bm{x}_{\text{txt}}$ is the dense intent description, $\bm{x}_{\text{pal}} \in \mathbb{R}^{3 \times N_{\text{pal}}}$ is the palette query (zero or more RGB colors), and $X_{\text{img}}$ is the set of nail design images to be ranked.

### Key Designs

**IPFM — Intent-Palette Fusion Module**:

- An LLM (GPT-4o) generates **Multi-level Design Descriptions (MDD)** and **Normalized Noun Phrases (NNP)** from raw descriptions, structuring intent and distilling key design elements, respectively.
- Multiple text encoders (BEiT-3, SigLIP) extract language representations $(\bm{l}_{\text{txt}}, \bm{l}_{\text{MDD}}, \bm{l}_{\text{NNP}})$.
- A palette encoder converts RGB to CIELAB color space and passes it through Transformer layers to obtain the palette representation $\bm{p}$.
- A cross-attention mechanism uses $\bm{p}$ as the query over the language representations to selectively emphasize color-relevant elements: $\bm{l}_{+} = \text{CrossAttn}(\bm{p}, \text{TFLayers}([\bm{l}_{\text{txt}}; \bm{l}_{\text{MDD}}; \bm{l}_{\text{NNP}}]))$

**VDFM — Visual Design Fusion Module**:

- **Unimodal visual representation** $\bm{v}_s$: DINOv2 captures color, shape, and texture features.
- **Multimodal aligned representation** $\bm{v}_a$: Image encoders from BEiT-3 and SigLIP extract language-aligned representations.
- **Img2txt intent-structured representation** $\bm{v}_n$: Multiple MLLMs (GPT-4o, Qwen2-VL) generate textual descriptions of design elements, decorations, themes, and impressions for each image; these are then encoded by text encoders to capture abstract design concepts and spatial relationships.
- The three representations are fused via Transformer layers: $\bm{v}^{(i)} = \text{TFLayers}([\bm{v}_s^{(i)}; \bm{v}_a^{(i)}; \bm{v}_n^{(i)}])$

**CRAM — Confidence-based Relaxed Alignment Module**:

- An MLLM (Qwen2-VL) estimates a confidence score $c_{ij} \in [0,1]$ for each pair $(i,j)$, taking the query NNP, the candidate image, and its NNP as input.
- If $c_{ij} \geq \theta$, the pair is added to the unlabeled positive set $\mathcal{Z}$.

### Loss & Training

The paper proposes the **Confidence-based Relaxed Contrastive (CRC) loss**:

$$\mathcal{L}_{\text{CRC}} = \mathcal{L}_P + \lambda_{UP} \mathcal{L}_{UP} + \lambda_N \mathcal{L}_N$$

- $\mathcal{L}_P = \sum_i (1 - S_{ii})^2$: Similarity of positive pairs should approach 1.
- $\mathcal{L}_{UP} = \sum_{(i,j) \in \mathcal{Z}} (\max(c_{ij} - S_{ij}, 0))^2$: Similarity of unlabeled positives should be no less than their confidence score.
- $\mathcal{L}_N = \sum_{(i,j) \notin \mathcal{Z}} (\max(S_{ij}, 0))^2$: Similarity of negative pairs should approach 0.

## Key Experimental Results

### Benchmark Dataset

**NAIL-STAR** (authors' construction): 10,625 nail design images with dense intent descriptions provided by 208 annotators. Average sentence length is 21.5 words, vocabulary size is 7,014, and palette queries contain an average of 2.0 colors. Images are sourced from users of 42 language backgrounds (Pinterest), covering diverse cultural contexts. Train/Val/Test = 8,625/400/1,600.

### Main Results

| Method | NAIL-STAR R@1 | NAIL-STAR MRR | Fashion200K R@1 | Fashion200K MRR |
|--------|:---:|:---:|:---:|:---:|
| CLIP | 15.5 | 25.2 | 47.6 | 61.7 |
| SigLIP | 47.5 | 58.8 | 60.3 | 71.9 |
| BEiT-3 | 40.6 | 53.9 | 52.8 | 66.2 |
| BLIP-2 | 20.8 | 33.3 | 65.2 | 75.3 |
| **NaiLIA (desc-only)** | **49.5** | **61.0** | **73.8** | **82.0** |
| **NaiLIA (full)** | **56.4** | **67.6** | **74.6** | **82.7** |

- NaiLIA (full) achieves R@1 of 56.4% on NAIL-STAR, surpassing the best baseline SigLIP by **8.9 pp**.
- On Marqo Fashion200K, R@1 reaches 74.6%, surpassing the best baseline BLIP-2 by **9.4 pp**.
- All differences are statistically significant at $p < 0.01$.

### Ablation Study

| Variant | R@1 | vs. Full Model |
|---------|:---:|:---:|
| Full model (a) | 56.4 | — |
| w/o MDD (b) | 54.9 | -1.5 |
| w/o NNP (c) | 54.5 | -1.9 |
| w/o MDD+NNP (d) | 51.6 | -4.8 |
| w/o multimodal aligned repr. (f) | 42.1 | **-14.3** |
| w/o img2txt repr. (g) | 54.0 | -2.4 |
| InfoNCE instead of CRC (i) | 52.7 | -3.7 |
| $\lambda_{UP}=0$ (j) | 54.5 | -1.9 |
| Fixed $c_{ij}=0.7$ (k) | 55.1 | -1.3 |

**Key Findings**:

1. **Multimodal aligned representation is most critical**: Removing $\bm{v}_a$ causes a 14.3 pp drop in R@1, making it the most important visual component.
2. **Generalizability of CRC loss**: Replacing InfoNCE with CRC loss on CLIP also yields a 1.0 pp improvement, demonstrating that CRC loss can serve as a general-purpose retrieval loss.
3. **Broad applicability of palette queries**: Adding palette inputs to CLIP and SigLIP improves R@1 by 5.8 pp and 5.9 pp, respectively.
4. **MLLM-estimated confidence outperforms fixed values**: Dynamic estimation of $c_{ij}$ outperforms a fixed value of 0.7 by 1.3 pp, validating the effectiveness of MLLMs as confidence estimators.

## Highlights & Insights

- This work is the first to systematically define the semantic retrieval task for nail design (NAIL-STAR), combining dense intent descriptions with continuous color palette queries.
- The CRC loss elegantly addresses the unlabeled positive problem in contrastive learning by leveraging MLLMs as confidence estimators — an approach transferable to other retrieval tasks with dense visually similar images.
- The img2txt intent-structured representation is an elegant design — converting images to design semantic descriptions via MLLMs before encoding compensates for the limitations of visual encoders in understanding abstract concepts.
- A high-quality, cross-cultural NAIL-STAR benchmark (208 annotators, 42 language backgrounds) is constructed and will be publicly released.

## Limitations & Future Work

- The method relies on multiple large models (GPT-4o, Qwen2-VL, BEiT-3, SigLIP, DINOv2) for preprocessing and inference, incurring high computational costs and latency that may limit practical deployment.
- The dataset scale of 10,625 images is relatively small, potentially limiting generalization to more diverse designs.
- The retrieval setting is user-agnostic; personalized user preference modeling is not considered.
- MLLM confidence scores are precomputed before training and cannot be dynamically updated during model training, potentially introducing estimation bias.
- The application domain is narrow (nail design); while the authors claim CRC loss is generalizable, it is only validated on fashion datasets.

## Related Work & Insights

- **Multimodal Retrieval in Fashion AI**: EI-CLIP extends CLIP with fashion terminology; CoSMo handles composed queries of reference images and modification text; FashionViL/FAME-ViL are representative vision-language models for fashion. This paper is distinguished by introducing continuous color inputs and dense intent descriptions.
- **Vision-Language Foundation Models**: CLIP, SigLIP, BEiT-3, BLIP-2, etc. achieve cross-modal alignment via contrastive learning; AlphaCLIP introduces an alpha channel to focus on regions of interest. This paper goes further by fusing multi-encoder features and incorporating img2txt conversion to capture abstract concepts.
- **Noisy Labels in Contrastive Learning**: The single-label supervision of InfoNCE is inherently susceptible to noise. This paper relaxes the contrastive loss by estimating confidence scores for unlabeled positives using MLLMs.

## Rating

- Novelty: ⭐⭐⭐⭐ (The combination of CRC loss, palette fusion, and img2txt representation is innovative)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Two datasets, comprehensive ablations, qualitative analysis, statistical significance testing)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, intuitive figures, well-defined task formulation)
- Value: ⭐⭐⭐ (Narrow application domain limits broader impact, though the CRC loss idea has certain general value)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SldprtNet: A Large-Scale Multimodal Dataset for CAD Generation in Language-Driven 3D Design](sldprtnet_a_large-scale_multimodal_dataset_for_cad_generation_in_language-driven.md)
- [\[CVPR 2026\] Shoe Style-Invariant and Ground-Aware Learning for Dense Foot Contact Estimation](shoe_style-invariant_and_ground-aware_learning_for_dense_foot_contact_estimation.md)
- [\[ICLR 2026\] Beyond Linearity in Attention Projections: The Case for Nonlinear Queries](../../ICLR2026/others/beyond_linearity_in_attention_projections_the_case_for_nonlinear_queries.md)
- [\[ACL 2026\] Reliable Evaluation Protocol for Low-Precision Retrieval](../../ACL2026/others/reliable_evaluation_protocol_for_low-precision_retrieval.md)
- [\[ICLR 2026\] Exchangeability of GNN Representations with Applications to Graph Retrieval](../../ICLR2026/others/exchangeability_gnn_representations.md)

</div>

<!-- RELATED:END -->

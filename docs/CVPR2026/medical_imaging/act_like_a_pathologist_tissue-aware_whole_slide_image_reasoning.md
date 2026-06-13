---
title: >-
  [Paper Note] Act Like a Pathologist: Tissue-Aware Whole Slide Image Reasoning
description: >-
  [CVPR 2026][Medical Imaging][Whole Slide Image] This paper proposes HistoSelect, a framework that emulates the coarse-to-fine reasoning process of pathologists through a three-stage filtering mechanism — tissue segmentat…
tags:
  - "CVPR 2026"
  - "Medical Imaging"
  - "Whole Slide Image"
  - "Visual Question Answering"
  - "Information Bottleneck"
  - "Patch Selection"
  - "Tissue-Aware Reasoning"
date: 2026-05-08
content_hash: da64ac122053a6de
---

# Act Like a Pathologist: Tissue-Aware Whole Slide Image Reasoning

**Conference**: CVPR 2026
**arXiv**: [2603.00667](https://arxiv.org/abs/2603.00667)  
**Authors**: Wentao Huang et al. (Stony Brook University, Mayo Clinic, Harvard/MGH, Stanford)
**Area**: Medical Imaging / Pathology VQA
**Keywords**: Whole Slide Image, Visual Question Answering, Information Bottleneck, Patch Selection, Tissue-Aware Reasoning

## TL;DR

This paper proposes HistoSelect, a framework that emulates the coarse-to-fine reasoning process of pathologists through a three-stage filtering mechanism — tissue segmentation → Group Sampler → Patch Selector — grounded in Information Bottleneck (IB) theory. By compressing task-irrelevant visual tokens, the method achieves state-of-the-art performance across three datasets while reducing computational cost by approximately 70%.

## Background & Motivation

Whole slide images (WSIs) are the gold standard for cancer diagnosis, yet a single WSI may contain tens of thousands of patches. Directly feeding these into large language models presents two fundamental bottlenecks:

**Computational bottleneck**: WSIs can reach resolutions of 100,000×100,000 pixels, yielding tens of thousands of patches after tiling. Encoding each patch as a visual token far exceeds the context window of LLMs.

**Information redundancy**: Pathologists do not examine every patch sequentially; instead, they first identify tissue types and then focus on regions relevant to the diagnostic question — the majority of patches are irrelevant to any given query.

Existing methods such as Q-Instruct and PathChat either apply uniform sampling (discarding critical information) or process all patches (computationally intractable). The root cause is: **how can the number of tokens be drastically reduced while preserving diagnostically relevant information?**

The natural workflow of pathologists provides direct inspiration: low-magnification overview to assess tissue architecture, followed by high-magnification examination of suspicious regions. HistoSelect formalizes this coarse-to-fine reasoning process as a learnable pipeline.

## Method

### Overall Architecture

HistoSelect consists of three core stages that mirror the cognitive workflow of pathologists:

1. **Tissue Segmentation**: Patches from the WSI are grouped by tissue type.
2. **Group Sampler**: The sampling ratio for each group is determined adaptively.
3. **Patch Selector**: The most relevant patches within each group are selected.

The selected patches are then fed into a VLM for question answering.

### Key Designs

**Stage 1: Tissue-Aware Grouping**

- A pathologist pre-defines $M$ tissue-type text prompts (e.g., "tumor tissue," "stroma," "necrosis").
- CONCH, a pathology-domain CLIP model, computes cosine similarity between each patch feature and the tissue prompts.
- Each patch is assigned to the highest-similarity tissue group, yielding $M$ groups $\{G_1, G_2, \ldots, G_M\}$.

**Stage 2: Group Sampler (IB-based Group-Level Sampling)**

- A group prototype vector $g_j$ is computed as the mean of patch features within each group.
- $g_j$ is concatenated with the question encoding $q$ and passed through a two-layer MLP with sigmoid activation to produce a sampling rate $r_j \in (0,1)$.
- $r_j$ determines the proportion of patches to retain from group $j$: $k_j = \lceil r_j \cdot N_j \rceil$.
- IB objective: maximize mutual information between $r_j$ and the answer while minimizing the complexity of $r_j$.

**Stage 3: Patch Selector (Hard Patch-Level Selection)**

- For each patch, a selection probability is computed as $s_i = \sigma(F_{\text{patch}}([x_i; q]))$, where $F_{\text{patch}}$ is a lightweight MLP.
- Within group $G_j$, patches are ranked by $s_i$ and the top-$k_j$ are selected.
- The Straight-Through Estimator (STE) is used to enable gradient flow through the non-differentiable hard selection operation.

### Loss & Training

The total loss consists of three terms, reflecting a two-level IB compression design:

$$L = L_{\text{VQA}} + \lambda_1 L_{\text{group}} + \lambda_2 L_{\text{patch}}$$

- **$L_{\text{VQA}}$**: Standard cross-entropy loss for VQA.
- **$L_{\text{group}}$** (group-level IB regularization): Bernoulli KL divergence between $r_j$ and a prior derived from cosine similarity.
- **$L_{\text{patch}}$** (patch-level IB regularization): Bernoulli KL divergence between $s_i$ and a patch-question cosine similarity prior.

Training strategy:
- The Group Sampler, Patch Selector, and VLM are trained end-to-end jointly.
- STE ensures gradient propagation through hard selection.
- Cosine similarity priors serve as unsupervised weak signals to guide selection.

## Key Experimental Results

### Main Results

| Method | SlideBench-VQA (Acc) | WSI-Bench (Acc) | In-house Ovarian (Acc) | Visual Token Reduction |
|---|---|---|---|---|
| Random Sampling | 52.3 | 48.7 | 61.2 | 70% |
| Q-Instruct | 56.1 | 51.3 | 64.8 | 0% |
| PathChat | 58.4 | 53.9 | 67.3 | 0% |
| **HistoSelect** | **63.7** | **58.2** | **73.6** | **~70%** |

Trained on 356K QA pairs; achieves consistent state-of-the-art results across all three datasets.

### Ablation Study

| Configuration | SlideBench-VQA | Change |
|---|---|---|
| Full HistoSelect | 63.7 | — |
| w/o Group Sampler | 59.8 | -3.9 |
| w/o Patch Selector | 60.5 | -3.2 |
| w/o IB Loss (group) | 61.2 | -2.5 |
| w/o IB Loss (patch) | 61.8 | -1.9 |
| Random patch selection | 55.1 | -8.6 |

### Key Findings

1. **Both filtering stages are necessary**: Removing either the Group Sampler or the Patch Selector leads to significant performance degradation, confirming that the two-level coarse-to-fine filtering is mutually complementary.
2. **IB regularization is effective**: Performance drops without IB losses, demonstrating that prior-guided information compression not only reduces computation but also improves accuracy.
3. **Strong interpretability**: Selected patches are highly consistent with diagnostically critical regions annotated by senior pathologists, validating the clinical plausibility of the approach.
4. **Lossless 70% compression**: Substantially reducing the number of tokens while surpassing full-input methods indicates that removing noisy patches is intrinsically beneficial.

## Highlights & Insights

1. **Cognitively inspired design**: By encoding the pathologist's overview-then-focus workflow into the model architecture, domain knowledge is leveraged more efficiently than purely data-driven approaches.
2. **Elegant application of IB theory**: The information bottleneck framework is instantiated at two levels (group and patch), with Bernoulli KL divergence and cosine similarity priors forming a concise and effective design.
3. **Hard selection via STE**: Hard patch selection reduces actual computation more faithfully than soft attention; STE ensures the model remains trainable.
4. **Clinical interpretability**: Beyond benchmark scores, the selected patches align with pathologist cognition, enhancing both the credibility and practical utility of the method.

## Limitations & Future Work

1. **Tissue types must be predefined**: The $M$ tissue prompts are manually specified by domain experts; reconfiguration is required when transferring across diseases or organs.
2. **Dependence on CONCH**: Grouping quality is bounded by CONCH's encoding capability in specific pathological domains; rare or tail tissue types may be grouped inaccurately.
3. **Information loss from hard selection**: Although STE enables training, discarded patches may still contain weak but potentially useful contextual information.
4. **Single-magnification processing**: The current method operates at a single magnification level and does not exploit the multi-scale pyramidal structure of WSIs.
5. **Non-negligible preprocessing overhead**: Encoding all patches with CONCH plus MLP inference may incur considerable upfront cost on extremely large WSIs.

## Related Work & Insights

- **CONCH / PLIP**: Vision-language alignment models for pathology, providing high-quality patch feature spaces.
- **Information Bottleneck**: The classical framework of Tishby et al., with established applications in video understanding (AdaFocus) and NLP (VIB).
- **PathChat / LLaVA-Med**: Representative pathology VQA methods; HistoSelect can serve as a plug-and-play frontend for these systems.
- **Insight**: The two-level IB compression framework generalizes naturally to other long-sequence tasks with hierarchical structure, such as long-video understanding and multi-document QA.

## Rating

| Dimension | Score (1–5) | Notes |
|---|---|---|
| Novelty | 4 | Novel combination of two-level IB compression and pathologist cognitive workflow |
| Technical Depth | 4 | Solid information-theoretic foundations; well-motivated STE and prior designs |
| Experimental Thoroughness | 4 | Three datasets, ablation studies, and interpretability analysis |
| Value | 5 | 70% token reduction with performance gains; clinically deployable |
| Writing Quality | 4 | Clear structure and intuitive figures |
| **Overall** | **4.2** | An elegant integration of cognitive inspiration and information theory |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Towards Effective and Efficient Context-aware Nucleus Detection in Histopathology Whole Slide Images](../../AAAI2026/medical_imaging/towards_effective_and_efficient_context-aware_nucleus_detection_in_histopatholog.md)
- [\[CVPR 2026\] Parameter-efficient Prompt Tuning and Hierarchical Textual Guidance for Few-shot Whole Slide Image Classification](parameter-efficient_prompt_tuning_and_hierarchical_textual_guidance_for_few-shot.md)
- [\[CVPR 2026\] MUSE: Harnessing Precise and Diverse Semantics for Few-Shot Whole Slide Image Classification](muse_harnessing_precise_and_diverse_semantics_for_few-shot_whole_slide_image_cla.md)
- [\[ICML 2026\] Federated Distillation for Whole Slide Image via Gaussian-Mixture Feature Alignment and Curriculum Integration](../../ICML2026/medical_imaging/federated_distillation_for_whole_slide_image_via_gaussian-mixture_feature_alignm.md)
- [\[CVPR 2026\] Ultrasound-CLIP: Semantic-Aware Contrastive Pre-training for Ultrasound Image-Text Understanding](ultrasound-clip_semantic-aware_contrastive_pre-training_for_ultrasound_image-tex.md)

</div>

<!-- RELATED:END -->

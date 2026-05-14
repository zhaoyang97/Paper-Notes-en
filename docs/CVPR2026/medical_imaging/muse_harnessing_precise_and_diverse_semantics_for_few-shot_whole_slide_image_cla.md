---
title: >-
  [Paper Note] MUSE: Harnessing Precise and Diverse Semantics for Few-Shot Whole Slide Image Classification
description: >-
  [CVPR2026][Medical Imaging][Whole Slide Image Classification] This paper proposes the MUSE framework, which significantly improves generalization in few-shot whole slide image (WSI) classification through MoE-driven samp…
tags:
  - "CVPR2026"
  - "Medical Imaging"
  - "Whole Slide Image Classification"
  - "Few-Shot Learning"
  - "Multiple Instance Learning"
  - "Vision-Language Models"
  - "Semantic Enhancement"
  - "MoE"
  - "Knowledge Base Retrieval"
date: 2026-05-08
content_hash: 445222225bd39f1d
---

# MUSE: Harnessing Precise and Diverse Semantics for Few-Shot Whole Slide Image Classification

**Conference**: CVPR2026
**arXiv**: [2602.20873](https://arxiv.org/abs/2602.20873)
**Code**: [JiahaoXu-god/CVPR2026_MUSE](https://github.com/JiahaoXu-god/CVPR2026_MUSE)
**Area**: Medical Imaging
**Keywords**: Whole Slide Image Classification, Few-Shot Learning, Multiple Instance Learning, Vision-Language Models, Semantic Enhancement, MoE, Knowledge Base Retrieval

## TL;DR

This paper proposes the MUSE framework, which significantly improves generalization in few-shot whole slide image (WSI) classification through MoE-driven sample-wise fine-grained semantic enhancement (SFSE) and LLM knowledge base-based stochastic multi-view model optimization (SMMO).

## Background & Motivation

1. **Annotations for WSIs are extremely scarce**: WSI annotation requires pathology experts, and privacy regulations further constrain data availability, leaving only a handful of labeled samples per diagnostic category — making few-shot learning essential.
2. **MIL is the standard framework for WSI analysis**: WSIs are divided into patches, and features are extracted and aggregated at the bag level. However, in few-shot settings, purely visual features are insufficient to capture the discriminative information needed to distinguish pathological subtypes.
3. **VLMs introduce textual semantics to aid classification**: Pathology-domain pretrained models such as PLIP, CONCH, and MUSK provide cross-modal encoders, and prior work has leveraged LLM-generated textual descriptions to assist classification.
4. **Existing methods employ semantics too coarsely**: LLMs serve merely as description generators, producing static, class-level priors that are shared across all samples, lacking sample-level adaptation.
5. **A single global query cannot disentangle fine-grained diagnostic attributes**: Complex concepts such as tumor grading and immune infiltration are collapsed into a single global query, resulting in coarse vision–semantic alignment and an inability to precisely localize diagnostically relevant regions.
6. **Insufficient textual diversity leads to overfitting**: Reliance on unoptimized, fixed prompts ignores the structural diversity of clinical language across abstraction levels, contextual nuances, and syntactic formulations, making models highly prone to overfitting to specific phrasings under few-shot conditions.

## Method

### Overall Architecture

MUSE comprises two core modules: **SFSE** (Sample-wise Fine-grained Semantic Enhancement) and **SMMO** (Stochastic Multi-view Model Optimization). SFSE is responsible for precise semantic awareness, while SMMO provides rich semantic diversity; together they improve few-shot generalization.

### SFSE: Sample-wise Fine-grained Semantic Enhancement

**(1) Decompositional Semantic Refinement (DSR)**

- $M$ learnable prompt vectors are appended to each class name, and class-level textual features $D$ are obtained via a text encoder.
- Following the MoE paradigm, $R$ expert query matrices are constructed; a routing network selects the top-$k$ experts for each class.
- Input-dependent Gaussian noise is injected into the routing scores to encourage expert diversity and prevent collapse.
- Each class produces $k$ fine-grained semantic cues, each encoding a distinct diagnostic sub-concept.

**(2) Sample-wise Vision-Text Interaction (SVTI)**

- The semantic cues produced by DSR serve as queries for multi-head cross-attention over the WSI patch features.
- Only the top-$r\%$ patches with the highest attention scores are retained, focusing on regions highly relevant to each semantic cue.
- The interaction results from $k$ semantic cues are fused with routing-score weighting to obtain a sample-wise semantic prior $f$.

### SMMO: Stochastic Multi-view Model Optimization

**(1) LLM Knowledge Base Construction**

- GPT-4 is used to decompose each class name into four clinical dimensions: cellular morphology, tissue architecture, staining characteristics, and spatial texture patterns.
- Ten concrete example descriptions are generated per dimension; these are randomly combined, and a locally deployed Qwen2-7B generates 300 multi-view descriptions per class, which are encoded to form the knowledge base.

**(2) Semantic Retrieval and Stochastic Optimization**

- The sample-wise prior $f$ produced by SFSE is used to retrieve the top-$m$ semantically matching texts from the knowledge base via cosine similarity.
- At each training iteration, one text is randomly sampled from the retrieved queue and passed through DSR+SVTI to produce an auxiliary prior $f_{\text{aux}}$.
- The main prior $f$ and auxiliary prior $f_{\text{aux}}$ are each mapped to logits via an MLP, and their mean is used to compute the cross-entropy loss.
- The stochastic sampling mechanism exposes the model to diverse semantic viewpoints, mitigating overfitting.

### Loss & Training

Standard cross-entropy loss: $\mathcal{L}^{t} = \text{CE}(z^{t}_{\text{final}}, GT)$, where $z^{t}_{\text{final}} = (z + z^{t}_{\text{aux}}) / 2$.

## Key Experimental Results

### Main Results: Three Datasets × Three Few-Shot Settings

Evaluation on CAMELYON, TCGA-NSCLC, and TCGA-BRCA under 4/8/16-shot settings (mean ± std over 10 runs):

| Dataset | Setting | ACC | AUC | F1 |
|---------|---------|-----|-----|-----|
| CAMELYON | 4-shot | **74.86** (vs. FOCUS 68.13, +6.73) | **76.65** | **68.66** |
| CAMELYON | 8-shot | **84.01** (vs. FOCUS 80.33, +3.68) | **88.32** | **82.42** |
| CAMELYON | 16-shot | **89.70** (vs. FOCUS 88.62, +1.08) | **92.52** | **88.59** |
| NSCLC | 4-shot | **79.90** (vs. FOCUS 79.14, +0.76) | **87.57** | **79.85** |
| NSCLC | 8-shot | **87.27** (vs. FOCUS 86.04, +1.23) | **94.27** | **87.20** |
| NSCLC | 16-shot | **89.74** | **96.82** | **89.70** |
| BRCA | 4-shot | **84.14** (vs. ViLa 81.80, +2.34) | **86.66** | **73.81** |
| BRCA | 8-shot | **84.37** | **88.39** | **76.29** |

Key finding: gains are larger with fewer annotations; under the 4-shot setting, CAMELYON ACC improves by 6.73%.

### Ablation Study (CAMELYON)

| Configuration | 4-shot ACC | 8-shot ACC | 16-shot ACC |
|---------------|-----------|-----------|------------|
| Base MIL | 62.04 | 71.52 | 82.37 |
| +TI (conventional interaction) | 65.13 | 79.10 | 86.61 |
| +TI+SFSE | 65.16 | 80.85 | 86.72 |
| +TI+SMMO | 71.18 | 83.68 | 87.99 |
| +TI+SFSE+SMMO (full) | **74.86** | **84.01** | **89.70** |

- SMMO contributes most in low-data settings (~6% gain at 4-shot); SFSE and SMMO yield synergistic gains when combined.
- Cosine similarity retrieval > L2 norm > random retrieval.
- Stochastic optimization > multi-text mean optimization (4-shot ACC: 74.86 vs. 70.33).
- Knowledge base LLM selection: Qwen2-7B > Deepseek-R1-7B > Llama-3.2-1B.

## Highlights & Insights

1. **First work to improve few-shot WSI classification from a semantic optimization perspective**: rather than merely generating semantics, MUSE actively optimizes both the precision and diversity of semantics.
2. **MoE-driven fine-grained semantic decomposition**: global class-level semantics are decomposed into multiple expert sub-concepts, and the most relevant sub-concepts are selected per sample, enabling sample-level adaptation.
3. **Elegant stochastic multi-view training strategy**: knowledge base retrieval combined with random sampling exposes the model to different semantic viewpoints at each iteration — analogous to data augmentation but operating in the semantic space — effectively combating overfitting.
4. **Comprehensive experiments with substantial gains**: 3 datasets × 3 shot settings, with extensive ablation studies covering retrieval strategies, optimization strategies, and LLM selection.

## Limitations & Future Work

1. **Knowledge base construction depends on GPT-4**: the concept decomposition and example generation stages require GPT-4 API calls, which are costly and difficult to run fully offline.
2. **Validation limited to binary classification**: CAMELYON (normal/metastasis), NSCLC (LUAD/LUSC), and BRCA (IDC/ILC) are all binary tasks; scalability to multi-class scenarios remains unverified.
3. **Knowledge base quality is tightly coupled to the choice of LLM**: ablation results show that different LLMs produce knowledge bases with notably different effects on performance, yet clear guidelines for selecting an appropriate LLM are lacking.
4. **Potentially high inference overhead**: MoE routing, multi-head cross-attention in SFSE, and SMMO retrieval all add computational cost; training is reported on a single RTX 3090, but inference speed is not reported.
5. **The text knowledge base is built offline in a one-shot manner**: it does not update dynamically during training, fixing the upper bound on textual diversity at 300 descriptions per class.

## Related Work & Insights

| Method | Semantic Source | Semantic Granularity | Semantic Diversity | Sample Adaptation |
|--------|----------------|---------------------|-------------------|-------------------|
| Top | Class-name text | Class-level | None | None |
| ViLa-MIL | LLM descriptions | Class-level | Limited | None |
| FOCUS | LLM descriptions + knowledge-guided compression | Class-level | Limited | Partial (visual compression) |
| **MUSE** | LLM knowledge base + MoE decomposition | **Sub-concept level** | **300 descriptions/class + random sampling** | **MoE routing + sample-wise cross-attention** |

The key distinction of MUSE lies in elevating semantics from "static class-level descriptions" to "dynamic, sample-adaptive multi-view semantic optimization."

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of MoE fine-grained semantic decomposition and stochastic multi-view optimization is novel; the first work to introduce a semantic optimization perspective into few-shot WSI classification.
- Experimental Thoroughness: ⭐⭐⭐⭐ — 3 datasets, multiple shot settings, and rich ablations covering modules, retrieval strategies, optimization strategies, and LLM selection; multi-class evaluation and inference efficiency analysis are missing.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, intuitive figures, and complete derivations.
- Value: ⭐⭐⭐⭐ — Strong practical value for few-shot pathological image classification; the semantic optimization paradigm is transferable to other medical VLM tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Parameter-efficient Prompt Tuning and Hierarchical Textual Guidance for Few-shot Whole Slide Image Classification](parameter-efficient_prompt_tuning_and_hierarchical_textual_guidance_for_few-shot.md)
- [\[ICLR 2026\] Exploiting Low-Dimensional Manifold of Features for Few-Shot Whole Slide Image Classification](../../ICLR2026/medical_imaging/exploiting_low-dimensional_manifold_of_features_for_few-shot_whole_slide_image_c.md)
- [\[CVPR 2026\] CARE: A Molecular-Guided Foundation Model with Adaptive Region Modeling for Whole Slide Image Analysis](care_a_molecular-guided_foundation_model_with_adaptive_region_modeling_for_whole.md)
- [\[CVPR 2026\] Act Like a Pathologist: Tissue-Aware Whole Slide Image Reasoning](act_like_a_pathologist_tissue-aware_whole_slide_image_reasoning.md)
- [\[CVPR 2026\] Mind the Discriminability Trap in Source-Free Cross-domain Few-shot Learning](mind_the_discriminability_trap_in_source-free_cross-domain_few-shot_learning.md)

</div>

<!-- RELATED:END -->

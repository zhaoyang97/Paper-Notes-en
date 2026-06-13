---
title: >-
  [Paper Note] Head Pursuit: Probing Attention Specialization in Multimodal Transformers
description: >-
  [NeurIPS2025][Image Generation][attention head specialization] This paper reinterprets the classical sparse signal recovery algorithm (SOMP) as a multi-sample interpretability tool…
tags:
  - "NeurIPS2025"
  - "Image Generation"
  - "attention head specialization"
  - "Matching Pursuit"
  - "Logit Lens"
  - "model editing"
  - "vision-language model"
date: 2026-05-08
content_hash: fcc156bad6aa7515
---

# Head Pursuit: Probing Attention Specialization in Multimodal Transformers

**Conference**: NeurIPS2025 Spotlight  
**arXiv**: [2510.21518](https://arxiv.org/abs/2510.21518)  
**Code**: [GitHub](https://github.com/lorenzobasile/HeadPursuit)  
**Area**: Image Generation
**Keywords**: [attention head specialization, Matching Pursuit, Logit Lens, model editing, vision-language model]

## TL;DR

This paper reinterprets the classical sparse signal recovery algorithm (SOMP) as a multi-sample interpretability tool, revealing fine-grained semantic specialization of attention heads in LLMs and VLMs. By flipping approximately 1% of heads, specific concepts (e.g., country names, toxic content, colors) can be reliably suppressed or amplified during generation.

## Background & Motivation

**Background**: Large-scale generative models (LLMs, VLMs) achieve strong performance across diverse tasks, yet their internal mechanisms remain incompletely understood. Prior work has identified functional roles for attention heads (syntactic tracking, copying behavior, factual recall), but these findings are typically based on heuristic methods and are difficult to generalize across samples.

**Limitations of Prior Work**: (1) Interpretability tools such as Logit Lens analyze one sample at a time and cannot stably quantify head importance; (2) Attention Lens requires training a separate linear probe per head, incurring high computational cost; (3) Existing head-editing methods lack a mathematical foundation and rely on trial and error.

**Key Challenge**: A method capable of aggregating multi-sample analysis without additional training is needed to systematically discover semantic specialization in attention heads.

**Goal**: To provide a mathematically principled method for identifying, quantifying, and exploiting the specialization of attention heads in specific semantic domains.

**Key Insight**: Generalizing Logit Lens as a sparse signal recovery problem, using SOMP (Simultaneous Orthogonal Matching Pursuit) to perform multi-sample sparse decomposition over an unembedding dictionary.

**Core Idea**: The output of an attention head can be approximated as a sparse linear combination of a small number of semantic directions in the unembedding matrix, and the explained variance ratio can quantify a head's degree of specialization for a target concept.

## Method

### Overall Architecture

The method proceeds in three steps: (1) given a dataset, compute the contribution matrix $\mathbf{H}_{h,l} \in \mathbb{R}^{n \times d}$ of each attention head to the residual stream; (2) for a target concept (e.g., colors, countries), restrict the unembedding matrix to the corresponding token rows and perform sparse decomposition via SOMP, ranking heads by explained variance ratio; (3) intervene on top-$k$ heads (sign flipping / scaling) and observe the effect on target concept generation.

### Key Designs

1. **SOMP Sparse Decomposition as Multi-Sample Logit Lens**:

    - Function: Apply the SOMP algorithm to sparsely decompose head activations $\mathbf{H} \in \mathbb{R}^{n \times d}$ over the unembedding matrix $\mathbf{D} \in \mathbb{R}^{v \times d}$, identifying the small number of semantic directions that best explain head behavior.
    - Mechanism: At each step, SOMP selects the dictionary atom most correlated with residuals across all samples, $p^t = \arg\max_j \|\mathbf{D}[j]\mathbf{R}^{tT}\|_1$, then refits via least squares after updating the support set: $\mathbf{W}^t = \arg\min_{\mathbf{W}} \|\mathbf{H} - \mathbf{W}\mathbf{D}[\mathbb{S}^{t+1}]\|_F$. Connection to Logit Lens: LL is equivalent to a single-step Matching Pursuit on a single sample; SOMP is its natural multi-sample, multi-step generalization.
    - Design Motivation: Single-sample LL results are noisy and redundant (Table 6); multi-sample SOMP obtains stable semantic head signatures by aggregating across the dataset.

2. **Variance-Explained Head Selection and Intervention**:

    - Function: Given a target concept (e.g., "color"), restrict the dictionary to unembedding rows of color-related tokens, decompose each head via SOMP, rank heads by explained variance ratio $\|\mathbf{H}_r\|_F^2 / \|\mathbf{H}\|_F^2$, and intervene on the top-$k$ heads.
    - Mechanism: Interventions scale each head's contribution to the residual stream—suppression uses $\alpha = -1$ (sign flip), and amplification uses $\alpha = 5$ (5× scaling). A key finding is that a very small number of heads (8–32, approximately 0.8%–3%) suffices to significantly affect target concept generation.
    - Design Motivation: If a head's variance is well explained by SOMP under a concept-restricted dictionary, the head predominantly outputs signals related to that concept, and intervening on it should selectively affect concept generation.

### Loss & Training

The method is entirely training-free—no model weights are modified; interventions are applied only at inference time by scaling the outputs of selected heads. In all experiments, head selection is performed on training data and evaluated on disjoint test data. Control experiments use randomly selected heads with the same quantity and layer distribution as the baseline.

## Key Experimental Results

### Main Results

**Question Answering (Mistral-7B on TriviaQA, F1 score)**:

| Heads Intervened | Target (Country) Performance ↓ | Non-Target Performance ↓ | Random Head Effect |
|---------|-------------------|-----------|----------|
| 8 heads (0.8%) | Significant drop | Minor drop | No significant effect |
| 16 heads | Large drop | Moderate drop | No significant effect |
| 32 heads | Severe drop | Moderate drop | No significant effect |

**Toxicity Mitigation (Normalized toxic generation count ↓)**:

| Dataset | 8-head SOMP | 8-head LL | 8-head Random | 32-head SOMP | 32-head LL | 32-head Random |
|--------|---------|--------|-----------|----------|---------|------------|
| RTP | **0.83** | 0.91 | 1.02 | **0.66** | 0.71 | 1.13 |
| TET | 0.83 | **0.81** | 0.97 | **0.49** | 0.68 | 0.95 |

### Ablation Study

**LLaVA Image Classification (Normalized accuracy after head flipping)**:

| Dataset | 16-head SOMP ↓ | 16-head Random | 32-head SOMP ↓ | 32-head Random |
|--------|-----------|------------|-----------|------------|
| MNIST | Large drop | No change | Severe drop | No change |
| SVHN | Large drop | No change | Severe drop | No change |
| GTSRB | Large drop | No change | Severe drop | No change |
| EuroSAT | Large drop | No change | Severe drop | No change |

**Flickr30k Image Captioning (Color suppression/amplification, 16 heads)**:

| Intervention | Color Keyword Frequency | CIDEr Retention |
|------|-------------|-------------|
| Suppression ($\alpha=-1$) | Near zero | >80% |
| Amplification ($\alpha=5$) | +60% or more | >80% |

### Key Findings

- Heads selected by SOMP exhibit high concept specificity: intervening on target concepts causes a far greater performance drop than on non-target concepts.
- Heads selected by Logit Lens are relevant but not specific—they degrade both target and non-target performance equally.
- Jaccard similarity analysis reveals that semantically similar tasks share heads (high MNIST/SVHN overlap; high EuroSAT/RESISC45 overlap).
- Amplification interventions are equally effective: $\alpha=5$ increases color, sentiment, and quantity word frequencies by 60%+.
- Consistent trends are confirmed across multiple VLMs (LLaVA-13B, Gemma3-12B, Qwen2.5-VL-7B).

## Highlights & Insights

- **Elegant theoretical connection**: Logit Lens is reinterpreted as a single-step, single-sample special case of Matching Pursuit; SOMP is its natural generalization.
- **High intervention efficiency**: Only 0.8%–3% of heads are sufficient to significantly control generation, suggesting that attention layers contain highly structured linear semantic subspaces.
- **Bidirectional controllability**: Both suppression and amplification are effective without severely degrading overall generation quality (CIDEr >80%).
- **Cross-modal consistency**: Consistent head specialization patterns are observed across both text and vision-language tasks, supporting the hypothesis that concepts are encoded linearly in the residual stream.

## Limitations & Future Work

- SOMP assumes linear decomposition and may fail to capture nonlinear structure in head representations.
- The quality and coverage of the semantic dictionary directly affect discovery quality—incomplete keyword lists may introduce bias.
- The intervention mechanism is coarse (global scaling) and does not distinguish intervention effects across different positions or modality tokens.
- Potential applications for controlling image generation (e.g., the image decoding stage of VLMs) remain unexplored.

## Related Work & Insights

- **Logit Lens / Tuned Lens**: The former requires no training but only examines a single step and single sample; the latter requires training a probe per layer. The proposed method strikes a balance between the two.
- **Head decomposition in CLIP**: Gandelsman et al. (2024) similarly decompose CLIP's attention heads, but the present work applies to generative models.
- **Factual editing (ROME/MEMIT)**: These methods target knowledge editing in MLP layers; this paper shifts focus to semantic control at the attention head level.
- **Implications**: The specialized structure of attention heads may offer a new avenue for lightweight model alignment and safety control—without fine-tuning, by simply identifying and scaling a small number of heads.

## Rating

- Novelty: ⭐⭐⭐⭐ — Introducing SOMP into LLM interpretability is a novel perspective with clear theoretical connections.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Covers four task types (QA, toxicity mitigation, image classification, image captioning) and five models with thorough comparisons.
- Writing Quality: ⭐⭐⭐⭐ — Method and experiments are presented in a well-structured, progressive manner with sufficient justification.
- Value: ⭐⭐⭐⭐ — Provides a practical tool for model understanding and lightweight control.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] DiTFastAttnV2: Head-wise Attention Compression for Multi-Modality Diffusion Transformers](../../ICCV2025/image_generation/ditfastattnv2_head-wise_attention_compression_for_multi-modality_diffusion_trans.md)
- [\[ICCV 2025\] EDiT: Efficient Diffusion Transformers with Linear Compressed Attention](../../ICCV2025/image_generation/edit_efficient_diffusion_transformers_with_linear_compressed_attention.md)
- [\[ICCV 2025\] Rethinking Cross-Modal Interaction in Multimodal Diffusion Transformers](../../ICCV2025/image_generation/rethinking_cross-modal_interaction_in_multimodal_diffusion_transformers.md)
- [\[ICCV 2025\] Exploring Multimodal Diffusion Transformers for Enhanced Prompt-based Image Editing](../../ICCV2025/image_generation/exploring_multimodal_diffusion_transformers_for_enhanced_prompt-based_image_edit.md)
- [\[AAAI 2026\] Melodia: Training-Free Music Editing Guided by Attention Probing in Diffusion Models](../../AAAI2026/image_generation/melodia_training-free_music_editing_guided_by_attention_probing_in_diffusion_mod.md)

</div>

<!-- RELATED:END -->

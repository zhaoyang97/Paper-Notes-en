---
title: >-
  [Paper Note] Beyond the Global Scores: Fine-Grained Token Grounding as a Robust Detector of LVLM Hallucinations
description: >-
  [CVPR 2026][LLM Safety][hallucination detection] A patch-level LVLM hallucination detection framework is proposed. Hallucinated tokens are found to exhibit two characteristic signatures—dispersed attention patterns and l…
tags:
  - "CVPR 2026"
  - "LLM Safety"
  - "hallucination detection"
  - "LVLM"
  - "attention dispersion"
  - "patch-level grounding"
  - "token-level"
date: 2026-05-08
content_hash: d4998961598bfb96
---

# Beyond the Global Scores: Fine-Grained Token Grounding as a Robust Detector of LVLM Hallucinations

**Conference**: CVPR 2026
**arXiv**: [2604.04863](https://arxiv.org/abs/2604.04863)  
**Code**: Available  
**Area**: LLM Safety
**Keywords**: hallucination detection, LVLM, attention dispersion, patch-level grounding, token-level

## TL;DR

A patch-level LVLM hallucination detection framework is proposed. Hallucinated tokens are found to exhibit two characteristic signatures—dispersed attention patterns and low semantic alignment—based on which two lightweight metrics are designed: Attention Dispersion Score (ADS) and Cross-modal Grounding Consistency (CGC), achieving 90% detection accuracy.

## Background & Motivation

LVLMs are prone to visual hallucinations—describing objects or attributes absent from the image. Existing detection methods rely on coarse-grained global statistics (e.g., aggregated overall attention, output probabilities, global embedding similarity). This global strategy has a fundamental limitation: hallucinated tokens may have weak but dispersed correlations with multiple local regions, which aggregate into a deceptively high overall correlation that evades global detectors.

Core insight: A faithful object token must be strongly grounded to a specific image region. Hallucination detection must therefore shift from whole-image analysis to patch-level grounding analysis.

## Method

### Overall Architecture

Fine-grained interactions between generated tokens and image patches are analyzed to extract two types of structural features, which are then used to train a lightweight classifier for token-level hallucination detection.

### Key Designs

1. **Attention Dispersion Score (ADS)**: Quantifies the spatial compactness of a target token's attention distribution. The top-$k$% attention activations are retained, and attention sinks (small blobs with area $< \tau_{ADS}$) are filtered using 8-connected components. The foreground blob mass $m_t^{(n)} = \sum_{c \in \mathcal{C}_t^{(n)*}} \sum_{p \in c} \bar{\mathbf{A}}_t^{(n)}(p)$ and background normalized entropy $\hat{H}_t^{(n)} = -\sum_{p \in \mathcal{B}} \mathbf{E}(p) \log \mathbf{E}(p) / \log|\mathcal{P}|$ are computed, yielding $ADS_t^{(n)} = (1-m_t^{(n)}) \cdot \hat{H}_t^{(n)}$. Low ADS indicates compact focus (faithful object); high ADS indicates diffuse dispersion (hallucination). Middle layers exhibit the strongest separation; the gap converges in deeper layers where language priors dominate.

2. **Cross-modal Grounding Consistency (CGC)**: Per-layer cosine similarity between the token embedding and each image patch embedding is computed, with the mean similarity of the top-$k$ patches taken as the score. Faithful tokens exhibit sharp similarity peaks with corresponding regions, while hallucinated tokens show low and diffuse similarity across all regions.

3. **Layer-wise Feature Concatenation for Classification**: ADS and CGC values from all layers are concatenated into a feature vector, and an XGB/MLP/random forest classifier is trained. Middle layers provide the strongest discriminability; differences converge in deeper layers due to increasing language prior dominance.

### Loss & Training

Classifiers are trained with cross-entropy loss. Labels are derived from GPT-4o semantic verification combined with the CHAIR metric and human-written descriptions. Experiments are conducted on 4,000 images from the MS-COCO 2014 validation set with a 90/10 split.

## Key Experimental Results

### Main Results (Image Captioning Task)

| Method | LLaVA-1.5-7B F1 | Qwen2.5-VL-7B F1 | InternVL2.5-8B F1 |
|--------|----------------|------------------|-------------------|
| MetaToken | 0.51 | 0.54 | — |
| SVAR | — | — | — |
| Ours (XGB) | **0.90** | **0.88** | **0.88** |

### Key Findings

- Faithful tokens exhibit compact, well-localized attention in early and middle layers, while hallucinated tokens show dispersed attention
- Faithful tokens have high semantic similarity with corresponding patches, while hallucinated tokens show low similarity with all patches
- Both findings jointly indicate that hallucinations primarily stem from over-reliance on language priors rather than deficiencies in the visual encoder
- Middle-layer features are most discriminative; the gap converges in deeper layers where language priors dominate
- ADS alone as a classifier achieves F1 = 0.73–0.77; combined with CGC it reaches F1 = 0.88–0.90
- Experiments are conducted on 4,000 images from the MS-COCO 2014 validation set with a 90/10 split
- Labels are determined by GPT-4o semantic verification combined with the CHAIR metric

## Highlights & Insights

- The shift from "global to local" perspective captures the key to hallucination detection
- The connected-component filtering and background entropy design in ADS elegantly address the attention sink phenomenon
- Strong interpretability—attention heatmaps can be visualized to explain detection results
- A lightweight classifier suffices to achieve 90% accuracy

## Limitations & Future Work

- Requires access to internal attention weights, necessitating white-box model access
- The classifier must be trained separately for each LVLM
- Only validated on object-level hallucinations; attribute hallucinations are not addressed
- Lightweight classifiers such as XGB/MLP/random forest offer strong interpretability but limited capacity for capturing complex patterns
- CGC computes per-layer cosine similarity between token embeddings and patch embeddings, taking the top-$k$ patch mean; faithful tokens exhibit sharp peaks while hallucinated tokens are diffuse and low
- The analysis of the fundamental limitations of existing global statistical methods such as SVAR is insightful: weak but widespread dispersed correlations aggregate into deceptively high global scores

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Pioneering work on patch-level hallucination analysis
- Technical Depth: ⭐⭐⭐⭐ — ADS and CGC are elegantly designed
- Experimental Thoroughness: ⭐⭐⭐⭐ — Validated across multiple models and benchmarks
- Practical Value: ⭐⭐⭐⭐ — Lightweight and interpretable hallucination detector

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] SHIELD: Suppressing Hallucinations In LVLM Encoders via Bias and Vulnerability Defense](../../ICLR2026/llm_safety/shield_suppressing_hallucinations_in_lvlm_encoders_via_bias_and_vulnerability_de.md)
- [\[ACL 2026\] TPA: Next Token Probability Attribution for Detecting Hallucinations in RAG](../../ACL2026/llm_safety/tpa_next_token_probability_attribution_for_detecting_hallucinations_in_rag.md)
- [\[CVPR 2026\] IAG: Input-aware Backdoor Attack on VLM-based Visual Grounding](iag_input-aware_backdoor_attack_on_vlm-based_visual_grounding.md)
- [\[CVPR 2026\] HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in LVLMs](hulluedit_subspace_editing_hallucination.md)
- [\[NeurIPS 2025\] FALCON: Fine-grained Activation Manipulation by Contrastive Orthogonal Unalignment for Large Language Model](../../NeurIPS2025/llm_safety/falcon_fine-grained_activation_manipulation_by_contrastive_orthogonal_unalignmen.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Hyper-ICL: Attention Calibration with Hyperbolic Anchor Distillation for Multimodal ICL
description: >-
  [ICML 2026][Multimodal VLM][Multimodal ICL] Hyper-ICL provides a structural prior for multimodal LVLM in-context learning by lifting **CLIP embeddings into hyperbolic space** to form structured "hyperspherical anchors" c…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Multimodal ICL"
  - "Hyperbolic Embedding"
  - "Attention Calibration"
  - "Cross-modal Hierarchical Structure"
date: 2026-05-08
content_hash: e1d4edcd5db72521
---

# Hyper-ICL: Attention Calibration with Hyperbolic Anchor Distillation for Multimodal ICL

**Conference**: ICML 2026  
**arXiv**: [2605.29103](https://arxiv.org/abs/2605.29103)  
**Code**: To be confirmed  
**Area**: Multimodal / Vision-Language Models / In-Context Learning  
**Keywords**: Multimodal ICL, Hyperbolic Embedding, Attention Calibration, Cross-modal Hierarchical Structure

## TL;DR
Hyper-ICL provides a structural prior for multimodal LVLM in-context learning by lifting **CLIP embeddings into hyperbolic space** to form structured "hyperspherical anchors" combined with **hierarchy-aware distillation attention**. It consistently outperforms traditional demo selection strategies on tasks such as VQA, Captioning, and Caption Editing.

## Background & Motivation

**Background**: Multimodal ICL requires models to learn from few-shot demonstrations and apply this knowledge to new queries. However, LVLMs face two major challenges when selecting and combining multimodal demos: **attention mismatch** and **structural blind spots**.

**Limitations of Prior Work**: (1) Existing methods select demos based on Euclidean similarity, ignoring the **hierarchical structure** between images, text, and categories; (2) LVLM attention struggles to focus on the most relevant information within demos, especially when modality information is inconsistent; (3) Traditional high-dimensional Euclidean space fails to capture heterogeneous semantic hierarchies efficiently.

**Key Challenge**: Multimodal semantics inherently possess a hierarchical structure (Image → Local Region → Semantic Concept → Category Label), but Euclidean space suffers from exponential volume explosion, making it difficult to represent such hierarchies efficiently.

**Goal**: Inject structural priors into multimodal ICL to guide attention toward hierarchically relevant demos.

**Key Insight**: Hyperbolic geometric space naturally balances **hyperbolic embedding radius vs. node depth**, where the volume grows exponentially with the radius—making it highly suitable for hierarchical representation.

**Core Idea**: Map CLIP multimodal embeddings into hyperbolic space to form "hyperspherical anchors," which serve as distillation targets to guide LVLM attention. This retains the strong semantics of pre-trained CLIP while enhancing hierarchical understanding.

## Method

### Overall Architecture
The framework consists of two stages—(1) **Offline**: Constructing a hyperbolic anchor bank (CLIP embedding → Hyperbolic projection → Hierarchical distillation anchors); (2) **Online**: Computing the hyperbolic embedding of a query, performing a hyperbolic hierarchical attention mechanism for demo selection, and distilling LVLM attention.

### Key Designs

1.  **Hyperspherical Anchors + Hyperbolic Projection**:
    - **Function**: Map CLIP embeddings into hyperbolic space to create hierarchy-aware anchors.
    - **Mechanism**: Use the exponential map $\exp_o(x) = \tanh(c\|x\|) \frac{x}{c\|x\|}$ to project the CLIP embedding $x$ onto the Poincaré ball $\mathbb{B}^d_c$ (with curvature $c$). A hyperspherical manifold of semantically related demos is formed by minimizing a contrastive loss under the hyperbolic metric: $\mathcal{L}_{\text{anchor}} = \sum_i \log \frac{\exp(-d_{\mathbb{B}}(x_i, x_i^+) / \tau)}{\sum_j \exp(-d_{\mathbb{B}}(x_i, x_j) / \tau)}$, where $d_{\mathbb{B}}(u, v) = \frac{2}{\sqrt{c}} \text{arctanh}(\sqrt{c} \|-u \oplus v\|)$.
    - **Design Motivation**: CLIP Euclidean embeddings compress heterogeneous semantics into a shared sphere, failing to express hierarchy; hyperbolic space naturally supports hierarchical representation, where root concepts are at the center and leaf nodes are at the boundary.

2.  **Hierarchy-aware Distillation Attention**:
    - **Function**: Use hyperbolic anchors to guide LVLM attention distribution, strengthening the model's focus on hierarchically relevant demos.
    - **Mechanism**: Introduce a calibration term after the LVLM attention calculation: $\alpha_{i, j}^* = \text{softmax}(QK^T / \sqrt{d} + \lambda \mathcal{H}(x_i, x_j))$, where $\mathcal{H}(\cdot, \cdot)$ is the inverse function of hyperbolic hierarchical distance. The LVLM learns the hierarchical prior through a distillation loss: $\mathcal{L}_{\text{distill}} = \text{KL}(\alpha_{\text{teacher}} \| \alpha_{\text{student}})$.
    - **Design Motivation**: Directly modifying LVLM attention weights risks damaging pre-trained knowledge; distillation allows the LVLM to internalize the hierarchical structure naturally.

3.  **Hyperbolic Demo Selection Algorithm**:
    - **Function**: Select the set of demos most hierarchically relevant to the query from a candidate pool.
    - **Mechanism**: Project the query into hyperbolic space to obtain $\hat{x}_q$. Calculate the hyperbolic distance and hierarchical depth between candidate demos and the query, then rank them according to $\text{score} = -d_{\mathbb{B}}(x_i, \hat{x}_q) + \mu \cdot \text{depth}_{\mathbb{B}}(x_i)$. Select the top-K as in-context demos.
    - **Design Motivation**: Traditional strategies based on cosine similarity only consider semantic distance; this method considers hierarchical depth, avoiding the selection of demos that are too generic or too specific.

## Key Experimental Results

### Main Results

| Task | Model | Random | TopK-CLIP | RICES | **Hyper-ICL** | Gain over RICES |
|------|------|--------|---------|-------|----------|---------------|
| VQA v2 | IDEFICS-9B | 28.4 | 31.2 | 33.7 | **37.9** | **+4.2** |
| OK-VQA | IDEFICS-9B | 19.8 | 22.3 | 24.1 | **28.5** | **+4.4** |
| COCO Caption | IDEFICS-9B | 67.5 | 71.8 | 74.2 | **78.6** | **+4.4** |
| Caption Editing | IDEFICS-9B | 31.2 | 35.7 | 38.4 | **42.1** | **+3.7** |
| Image-Text Match | Otter-9B | 52.3 | 56.8 | 59.4 | **64.7** | **+5.3** |
| Visual Reasoning | Otter-9B | 42.8 | 46.3 | 48.9 | **54.2** | **+5.3** |

### Ablation Study

| Configuration | VQA v2 | COCO Caption | Description |
|------|--------|-------------|------|
| Hyperbolic Anchors Only (No Attention Calib.) | 35.1 | 76.2 | Contribution of anchors |
| Attention Calib. Only (Euclidean Anchors) | 33.8 | 75.5 | Contribution of attention mechanism |
| Attention Calib. + Hyperbolic Anchors | **37.9** | **78.6** | Full Hyper-ICL |
| Hyperbolic Metric ↔ Euclidean Metric | 33.7 | 74.2 | Metric degradation comparison |
| Demo Count K=2 → K=4 → K=8 | 35.2 / 37.9 / 38.4 | 76.8 / 78.6 / 79.2 | K=8 is optimal but with diminishing returns |

### Curvature Sensitivity

| Curvature c | VQA v2 ACC | COCO BLEU-4 |
|-------|-----------|-------------|
| 0.5 | 35.6 | 76.4 |
| **1.0** | **37.9** | **78.6** |
| 2.0 | 36.4 | 77.2 |

$c = 1.0$ is optimal; too low a curvature degrades to Euclidean, while too high causes numerical instability.

### Key Findings
- Hyperbolic anchors and attention calibration produce a synergistic effect—individual use yields +2-3 points, while the combination yields +4-5 points.
- The hierarchical representation advantage of the hyperbolic metric is more significant in hierarchical tasks (reasoning, image-text match).
- More robust to demo count—traditional methods show sharply diminishing returns as K increases from 4 to 8, while Hyper-ICL maintains steady growth.

## Highlights & Insights
- **First Application of Hyperbolic Geometry to Multimodal ICL**: Breaks through the representational limits of Euclidean space and introduces new geometric tools.
-  **Elegant Balance in Distillation Mechanism**: Uses soft target distillation rather than hard constraint modification, injecting hierarchical priors while preserving LVLM pre-trained knowledge.
- **Complete Closed-loop Design**: Forms a unified hyperbolic framework from anchor construction and demo selection to attention calibration, with components reinforcing each other.

## Limitations & Future Work
- Numerical stability of hyperbolic computation: High curvature or points near the boundary can lead to gradient explosion/vanishing and require careful handling.
- Scale and coverage of the anchor bank: Since it is constructed based on CLIP embeddings, hierarchical mismatch may occur for concepts outside the CLIP training distribution.
- Inference overhead: Additional hyperbolic metric calculations are required for each ICL instance (lightweight but with cumulative overhead).
- Improvements: Explore more stable hyperbolic optimization algorithms; extend to other modalities like video and audio; investigate applicability to larger LVLMs (e.g., LLaVA-1.6, GPT-4V).

## Related Work & Insights
- **vs RICES**: RICES selects demos based on CLIP similarity; Hyper-ICL introduces hyperbolic hierarchical structures to improve both selection and attention mechanisms.
- **vs Poincaré Embedding**: Classic Poincaré embeddings target tree-structured corpora; this work extends them to the dynamic scenario of LVLM in-context learning.
- **vs Attention Calibration (in NLP)**: Prior work focuses only on textual attention; this work expands the concept to multimodal scenarios for the first time.

## Rating
- Novelty: ⭐⭐⭐⭐⭐  First attempt to apply hyperbolic geometry to multimodal ICL, showing clear interdisciplinary innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐  Covers 6 tasks and 2 LVLMs with detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐  Mathematical formulas are clear, though some hyperbolic geometry concepts may require reader background.
- Value: ⭐⭐⭐⭐⭐  Provides a new paradigm in the frontier field of multimodal ICL, with consistent multi-task improvements indicating high potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Explaining Is Harder than Predicting Alone: Evaluating Concept-Based Explanations of MLLMs as ICL Visual Classifiers](explaining_is_harder_than_predicting_alone_evaluating_concept-based_explanations.md)
- [\[ICML 2026\] Seeing is Understanding: Unlocking Causal Attention into Modality-Mutual Attention for Multimodal LLMs](seeing_is_understanding_unlocking_causal_attention_into_modality-mutual_attentio.md)
- [\[ICML 2026\] Smoothing Slot Attention Iterations and Recurrences](smoothing_slot_attention_iterations_and_recurrences.md)
- [\[ICML 2026\] Gated Relational Alignment via Confidence-based Distillation for Efficient VLMs](gated_relational_alignment_via_confidence-based_distillation_for_efficient_vlms.md)
- [\[ACL 2026\] VL-Calibration: Decoupled Confidence Calibration for Large Vision-Language Models Reasoning](../../ACL2026/multimodal_vlm/vl-calibration_decoupled_confidence_calibration_for_large_vision-language_models.md)

</div>

<!-- RELATED:END -->

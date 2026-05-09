---
title: >-
  [Paper Note] AutoPP: Towards Automated Product Poster Generation and Optimization
description: >-
  [AAAI2026][Recommender Systems][product poster generation] This paper proposes AutoPP, the first pipeline to unify automated product poster generation with CTR-feedback-driven optimization in a single framework. It employs a unified design module to jointly design background, text, and layout; an element rendering module for efficient and controllable poster generation; and Isolated DPO (IDPO) to achieve element-level click-through rate optimization.
tags:
  - AAAI2026
  - Recommender Systems
  - product poster generation
  - CTR optimization
  - diffusion model
  - DPO
  - multimodal generation
date: 2026-05-08
content_hash: 86a9007ae24b2c1c
---

# AutoPP: Towards Automated Product Poster Generation and Optimization

**Conference**: AAAI2026  
**arXiv**: [2512.21921](https://arxiv.org/abs/2512.21921)  
**Code**: [JD-GenX/AutoPP](https://github.com/JD-GenX/AutoPP)  
**Area**: Recommender Systems  
**Keywords**: product poster generation, CTR optimization, diffusion model, DPO, multimodal generation

## TL;DR

This paper proposes AutoPP, the first pipeline to unify automated product poster generation with CTR-feedback-driven optimization in a single framework. It employs a unified design module to jointly design background, text, and layout; an element rendering module for efficient and controllable poster generation; and Isolated DPO (IDPO) to achieve element-level click-through rate optimization.

## Background & Motivation

- Product posters require the artful combination of product images, text, and backgrounds to attract user clicks, yet manual creation and iterative refinement are highly time- and labor-intensive.
- Existing methods exhibit clear automation bottlenecks:
    - PAID adopts a four-stage pipeline (prompt → layout → background → text rendering), in which text attributes (font, color) rely on hand-crafted rules, limiting automation and compromising visual coherence.
    - PosterMaker leverages SD3 + ControlNet for simultaneous background and text rendering, but still requires users to manually design the layout and advertising copy for each poster, making large-scale production inefficient.
- On the optimization side, CG4CTR and CAIG only optimize the CTR of background elements, neglecting the impact of text and layout on click-through rates; moreover, their coarse-grained joint optimization cannot attribute improvements to specific elements.

## Core Problem

1. **Generation side**: How can high-quality posters be generated automatically from only basic product information (product image + candidate copy), without manual layout design, copywriting, or text attribute specification?
2. **Optimization side**: How can CTR improvements be precisely attributed to specific poster elements (background / text / layout) to enable fine-grained, element-level optimization rather than coarse holistic adjustment?

## Method

### Overall Architecture

AutoPP consists of two major components: a **generator** and an **optimizer**.

### 1. Unified Design Module

- A multimodal large language model (MLLM, initialized from LLaVA) jointly generates three key elements: background prompt $b$, selected copy $T^*$, and layout $l$.
- Input: product image $I_{\text{product}}$ + candidate copy set $T$.
- The joint distribution is modeled autoregressively: $\pi(y|I_{\text{product}}, X_{\text{instr}}) = \prod_i p(y_i | I_{\text{product}}, X_{\text{instr}}, y_{<i})$
- Compared to distributed multi-model pipelines, single-model joint inference ensures design coherence.

### 2. Element Rendering Module

- Built upon FLUX.1 dev; product images and glyph images are encoded as condition tokens.
- Key innovation: **Decomposed Attention (DA)** mechanism replacing full attention in MM-DiT:
    - **Condition Self-Attention**: Glyph and product tokens each perform self-attention independently, capturing intra-element dependencies.
    - **Image-Condition Cross-Attention**: Query = [prompt tokens; noise tokens], Key/Value = [all token types concatenated], enabling cross-modal information exchange.
- Advantage of the token-based mechanism: no pixel-level alignment required, robust to spatial misalignment between glyph images and target images.
- Training loss: flow matching loss + OCR perceptual loss (intermediate features from PaddleOCRv4 backbone enforce text region clarity, $\lambda=0.1$).

### 3. Systematic Element Replacement

- Starting from generated posters, one element is replaced at a time (keeping the rest fixed) to create variants:
    - Background replacement: GPT-4o generates alternative background descriptions based on the original prompt.
    - Text replacement: a candidate copy of equal length is selected from the candidate set.
    - Layout replacement: the unified design module regenerates the layout.
- Variant posters are randomly displayed on the JD.com platform to collect CTR feedback.

### 4. Isolated Direct Preference Optimization (IDPO)

- Standard DPO performs coarse-grained alignment over the entire output, unable to distinguish the contribution of individual elements.
- IDPO introduces **element-aware weights**: $w_i = \sum_{c \in \{b, T^*, l\}} \alpha_c \cdot \mathbb{I}(y_i \in c)$
- The replaced element receives weight $\alpha=5$; unchanged elements receive $\alpha=1$, so CTR feedback precisely guides the most influential elements.
- Weighted log-likelihood normalization: $\log \pi^w(y|I,X) = \frac{\sum_i w_i \log p(y_i|I,X,y_{<i})}{\sum_i w_i}$

### AutoPP1M Dataset

- **Generation subset**: 1 million high-quality product posters (1:1 aspect ratio, ≥800×800), sourced from JD.com, with aesthetic filtering, blur detection, and watermark removal.
- **Optimization subset**: 50,000 pairwise comparisons collected via a 10-day random display experiment; each poster viewed by at least 50 users, with over 1.118 million users participating; pairwise CTR difference ≥1%.

## Key Experimental Results

### Poster Generation (Offline Evaluation, 500 Posters)

| Method | FID↓ | CLIP-T↑ | Alignment↓ | Overlap↓ | MIoU↑ |
|--------|------|---------|------------|----------|-------|
| P&R | 104.05 | 27.21 | 0.014 | 0.024 | 0.203 |
| PAID | 83.55 | 28.92 | 0.013 | 0.041 | 0.215 |
| GPT-4o | 63.47 | 29.58 | 0.009 | 0.018 | 0.140 |
| **AutoPP** | **60.71** | **29.75** | **0.007** | **0.011** | **0.256** |

### Text Rendering Quality

| Method | Sen. Acc↑ | NED↓ | FID↓ | CLIP-T↑ |
|--------|-----------|------|------|---------|
| PosterMaker | 57.87 | 21.93 | 49.76 | 30.43 |
| **AutoPP** | **65.19** | **12.94** | **43.19** | **30.49** |

### Online CTR Optimization (JD.com, 1-Week Experiment, 10,000 Products)

- AutoPP (IDPO): relative CTR improvement **+4.49%**
- AutoPP (standard DPO): +3.10%
- CG4CTR / CAIG: negative CTR gain (due to optimizing background only, neglecting text and layout)

### Efficiency

- DA mechanism reduces MM-DiT block GFLOPs by 18% at 800×800 resolution and 24% at 1024×1024.
- No additional parameters introduced (PosterMaker +1.6B, Flux-ControlNet +4.2B).

### Effect of Data Scale

- Reward Accuracy improves with data scale: 10K→51.20%, 30K→67.19%, 50K→75.99%.

## Highlights & Insights

1. **End-to-end full automation**: From basic product information to final optimized poster, with no manual input of layout, copy attributes, or manual adjustments required.
2. **Fine-grained attribution via IDPO**: Through systematic single-element replacement combined with element-aware weighted DPO, this work is the first to precisely attribute CTR improvements to isolated elements.
3. **Decomposed Attention**: Without adding parameters, decomposing full attention into condition SA + image-condition CA reduces computational overhead for long sequences.
4. **Large-scale industrial validation**: AutoPP1M is the largest product poster dataset to date; the online experiment involved over one million real users.
5. **Cross-lingual generalization**: Although trained primarily on Chinese data, the model exhibits emergent cross-lingual generation capability in English, Japanese, and Korean.

## Limitations & Future Work

- CTR optimization uses aggregated data from all users, potentially overlooking minority preferences; **personalized preference learning** could be explored in future work.
- The design module and rendering module remain two separate stages; future work could integrate them into a **single autoregressive model** with unified RLHF-based optimization.
- The element replacement strategy relies on GPT-4o for generating background variants, introducing a dependency on an external model.
- Validation is limited to the JD.com platform; cross-platform generalizability remains unknown.

## Related Work & Insights

| Method | Fully Automated | Text Rendering | Layout Design | CTR Optimization | Element Attribution |
|--------|:---:|:---:|:---:|:---:|:---:|
| PAID | ✗ (manual text rules) | Rule-based | Automatic | ✗ | ✗ |
| PosterMaker | ✗ (user-provided layout & copy) | SD3+ControlNet | Manual | ✗ | ✗ |
| CG4CTR | - | - | - | ✓ (background only) | ✗ |
| CAIG | - | - | - | ✓ (background only) | ✗ |
| **AutoPP** | **✓** | **Token+DA** | **Automatic** | **✓ (all elements)** | **✓ (IDPO)** |

- The element-isolation optimization paradigm of IDPO generalizes to other **multi-element compositional optimization** scenarios (e.g., ad creatives, web design, recommendation feed cards).
- The condition SA + cross-attention decomposition strategy of Decomposed Attention applies to any multi-condition controlled generation task.
- The paradigm of systematic element replacement combined with preference optimization can be applied to other settings requiring online A/B testing feedback.
- The use of MLLMs for joint design (simultaneously outputting layout, copy selection, and background description) is worth emulating in other multi-step design tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The element-level attribution optimization of IDPO and the fully automated pipeline design are novel, though individual sub-modules (MLLM design, FLUX rendering) build on mature architectures.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Dual offline and online validation, million-user-scale experiments, and complete ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, rich illustrations, and complete method descriptions.
- **Value**: ⭐⭐⭐⭐⭐ — Strong industrial deployment value; actually deployed on JD.com, where even a 0.5% CTR improvement yields significant commercial returns.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] SlideTailor: Personalized Presentation Slide Generation for Scientific Papers](slidetailor_personalized_presentation_slide_generation_for_scientific_papers.md)
- [\[ICLR 2026\] GoalRank: Group-Relative Optimization for a Large Ranking Model](../../ICLR2026/recommender/goalrank_group-relative_optimization_for_a_large_ranking_model.md)
- [\[AAAI 2026\] Inference-Aware Prompt Optimization for Aligning Black-Box Large Language Models](inference-aware_prompt_optimization_for_aligning_black-box_large_language_models.md)
- [\[AAAI 2026\] Semi-Supervised Synthetic Data Generation with Fine-Grained Relevance Control for Short Video Search Relevance Modeling](semi-supervised_synthetic_data_generation_with_fine-grained_relevance_control_fo.md)
- [\[AAAI 2026\] Behavior Tokens Speak Louder: Disentangled Explainable Recommendation with Behavior Vocabulary](behavior_tokens_speak_louder_disentangled_explainable_recommendation_with_behavi.md)

</div>

<!-- RELATED:END -->

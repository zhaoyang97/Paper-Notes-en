---
title: >-
  [Paper Note] Verify Claimed Text-to-Image Models via Boundary-Aware Prompt Optimization
description: >-
  [CVPR 2026][Image Generation][Model verification] BPO proposes a reference-free white-box T2I model verification method that employs a three-stage pipeline (adversarial anchor identification → binary search boundary exploration → target optimization) to locate model-specific semantic boundary regions. The generated verification prompts achieve an average accuracy of 96% and F1 of 0.93 across 5 T2I models, while being 2× faster than the TVN baseline.
tags:
  - CVPR 2026
  - Image Generation
  - Model verification
  - semantic boundary
  - adversarial prompt optimization
  - T2I model fingerprinting
  - intellectual property
date: 2026-05-08
content_hash: e851c85cfc8ab150
---

# Verify Claimed Text-to-Image Models via Boundary-Aware Prompt Optimization

**Conference**: CVPR 2026
**arXiv**: [2603.26328](https://arxiv.org/abs/2603.26328)
**Code**: None
**Area**: Image Generation
**Keywords**: Model verification, semantic boundary, adversarial prompt optimization, T2I model fingerprinting, intellectual property

## TL;DR

BPO proposes a reference-free white-box T2I model verification method that employs a three-stage pipeline (adversarial anchor identification → binary search boundary exploration → target optimization) to locate model-specific semantic boundary regions. The generated verification prompts achieve an average accuracy of 96% and F1 of 0.93 across 5 T2I models, while being 2× faster than the TVN baseline.

## Background & Motivation

1. **Background**: The commercial value of T2I models (e.g., the Stable Diffusion series) has made model attribution verification an important requirement. There is a need to verify whether a publicly deployed T2I model is indeed the claimed model (e.g., to prevent model reskinning or theft).
2. **Limitations of Prior Work**: (1) TVN relies on multiple reference models for comparison, requiring maintenance of a reference model set and offering poor scalability; (2) random/greedy prompt methods achieve only 17–23% accuracy, as generic prompts cannot distinguish between similar models; (3) existing methods suffer from low computational efficiency.
3. **Key Challenge**: Although the text encoders and generators of different T2I models are similar (most being fine-tuned from the same architecture), their semantic boundaries—regions in the embedding space where output semantics shift abruptly—are model-specific.
4. **Goal**: To generate verification prompts by directly exploiting the semantic boundary characteristics of the target model itself, without requiring any reference models.
5. **Key Insight**: Drawing an analogy to classifier decision boundaries—each model's semantic boundary lies at a distinct location—precisely localizing the boundary and generating prompts near it enables differentiation between models.
6. **Core Idea**: A three-stage pipeline: adversarial attack to locate semantic flip points → binary search for precise boundary localization → GCG optimization to generate boundary-oriented verification prompts.

## Method

### Overall Architecture

Given an input prompt $I$ → **Stage 1**: GCG adversarial attack appends a suffix $s$ to induce semantic flipping in the generated output, yielding anchor points on both sides of the boundary $(P_{pis}, P_{adv})$ → **Stage 2**: Linear interpolation in the embedding space $e_\alpha = (1-\alpha)e_{pis} + \alpha e_{adv}$ combined with binary search to locate the precise boundary $e_{\alpha^*}$ → **Stage 3**: Optimize a new suffix $s'$ on $P_{adv}$ to drive the embedding toward $e_{\alpha^*}$ → Output verification prompt $P_v$.

### Key Designs

1. **Adversarial Anchor Identification (Stage 1)**

   - **Function**: Identify two anchor points in the embedding space at which semantic flipping occurs.
   - **Mechanism**: GCG optimizes an 8-token suffix $s$ with objective $\min_s \cos(E_t(I+s), E_t(I))$. During iteration, the step $k^*$ at which semantic flipping first occurs is identified via a VLM judge; $P_{adv} = P_{k^*}$ and $P_{pis} = P_{k^*-1}$ are taken as the two boundary-side anchors.
   - **Design Motivation**: Directly searching the embedding space for the boundary is infeasible due to its high dimensionality. However, adversarial attacks naturally traverse directions away from the original semantics, and their trajectory must cross the semantic boundary.

2. **Binary Search Boundary Exploration (Stage 2)**

   - **Function**: Precisely localize the semantic boundary from coarse anchor points.
   - **Mechanism**: Linearly interpolate between $e_{pis}$ and $e_{adv}$, and apply binary search to find $\alpha^*$ such that $S(G_t(e_{\alpha^*})) \neq S(M_t(I))$, with precision threshold $\epsilon = 0.001$.
   - **Design Motivation**: The linear interpolation assumes local linearity of the embedding space near the boundary (empirically validated). Binary search achieves $O(\log(1/\epsilon))$ complexity, far superior to grid search.

3. **Target Optimization (Stage 3)**

   - **Function**: Generate high-discriminability prompts suitable for verification.
   - **Mechanism**: Optimize a new suffix $s'$ on $P_{adv}$ with objective $\max_{s'} \cos(E_t(I+s'), e_{\alpha^*})$, using 100 GCG iterations with batch size 256. The resulting $P_v$ has an embedding located precisely near the semantic boundary of the target model.
   - **Design Motivation**: $P_v$ lies on the semantic boundary of the target model but is unlikely to reside near the boundary of other models—thus the same $P_v$ produces semantically different outputs on different models, enabling discrimination.

### Loss & Training

No training is involved; all optimization is performed at inference time. GCG is used for suffix optimization, and a VLM (qwen-vl-max) is used for semantic judgment. For each verification task, 10 generated images are used to evaluate a consistency score $C = |2r - 1|$.

## Key Experimental Results

### Main Results

| Method | SD v1.4 | SD v2.1 | SDXL | Dreamlike | Openjourney | Avg. Acc |
|--------|---------|---------|------|-----------|-------------|----------|
| Normal | 0.17 | 0.17 | 0.17 | 0.17 | 0.17 | 0.17 |
| Random | 0.33 | 0.20 | 0.17 | 0.33 | 0.17 | 0.23 |
| TVN | 0.50 | 1.00 | 0.83 | 0.50 | 0.17 | 0.60 |
| **BPO** | **1.00** | **0.80** | **1.00** | **1.00** | **1.00** | **0.96** |

### Ablation Study

| Prompt Variant | Avg. Acc | Avg. F1 | Note |
|----------------|---------|---------|------|
| $P_{pis}$ (pre-boundary) | 0.80 | 0.78 | Insufficient proximity to boundary |
| $P_{adv}$ (post-boundary) | 0.84 | 0.80 | Already crosses the boundary |
| **$P_v$ (optimized)** | **0.96** | **0.93** | Precise boundary localization |

### Key Findings

- BPO achieves an average accuracy of 96%, outperforming TVN's 60% by 36 percentage points, without requiring any reference models.
- 2× efficiency improvement: BPO averages 159s vs. TVN's 321s (5× speedup on SD v1.4: 108s vs. 553s).
- Accuracy plateaus with only 10 generated images (0.96); additional images yield no significant gain.
- Suffix length of 8–9 tokens is optimal; shorter suffixes carry insufficient information while longer ones may overfit.
- VLM choice has limited impact: qwen-vl-max = 0.96, gemini-2.5-flash = 0.92, gpt-5 = 0.92.

## Highlights & Insights

- **Semantic Boundaries as Model Fingerprints**: The analogy to classifier decision boundaries is elegant—semantic boundaries are intrinsic and non-replicable model characteristics, making them harder to forge than model watermarks.
- **Progressive Refinement via Three Stages**: Adversarial attack → binary search → target optimization; each stage has a clear mathematical foundation and experimental validation.
- **Completely Reference-Free**: Eliminating the need to maintain a reference model set reduces overhead and renders the method scalable to arbitrary new models.

## Limitations & Future Work

- Requires white-box access to the target model's text encoder (for gradient computation), making it inapplicable to pure API-based services.
- Only 5 open-source models are evaluated; generalizability to recent proprietary models (e.g., DALL-E 3, Midjourney) remains unknown.
- Models with stronger adversarial robustness may have less well-defined semantic boundaries, making localization more difficult.
- Regularization techniques could render boundaries less distinctive, potentially degrading verification accuracy.
- Future work may explore a black-box variant via boundary probing through API queries.

## Related Work & Insights

- **vs. TVN**: TVN requires a reference model set to compare inconsistency rates; BPO directly exploits intrinsic model properties, yielding a conceptually simpler and empirically superior approach.
- **vs. Model Watermarking**: Watermarking requires embedding during training, whereas BPO is a post-hoc verification method applicable to already-deployed models.
- **vs. GCG Adversarial Attack**: BPO repurposes GCG from an "attack" tool into a "diagnostic" tool, serving an entirely different objective.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The concept of semantic boundaries as model fingerprints is highly innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 5 models with ablations and efficiency analysis, though the evaluation scale is relatively small.
- Writing Quality: ⭐⭐⭐⭐ The three-stage description is clear and the formalization is rigorous.
- Value: ⭐⭐⭐⭐ Addresses a practical need for model IP protection and introduces a novel methodological perspective.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Neighbor-Aware Localized Concept Erasure in Text-to-Image Diffusion Models](neighbor-aware_localized_concept_erasure_in_text-to-image_diffusion_models.md)
- [\[ICLR 2026\] Exposing Hidden Biases in Text-to-Image Models via Automated Prompt Search](../../ICLR2026/image_generation/exposing_hidden_biases_in_text-to-image_models_via_automated_prompt_search.md)
- [\[ICLR 2026\] Diverse Text-to-Image Generation via Contrastive Noise Optimization](../../ICLR2026/image_generation/diverse_text-to-image_generation_via_contrastive_noise_optimization.md)
- [\[CVPR 2026\] Smoothing the Score Function for Generalization in Diffusion Models: An Optimization-based Explanation Framework](smoothing_score_function_generalization_diffusion_models.md)
- [\[CVPR 2026\] Neighbor GRPO: Contrastive ODE Policy Optimization Aligns Flow Models](neighbor_grpo_contrastive_ode_policy_optimization_aligns_flow_models.md)

<!-- RELATED:END -->

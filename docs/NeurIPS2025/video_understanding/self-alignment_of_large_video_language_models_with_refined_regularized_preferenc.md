---
title: >-
  [Paper Note] Self-alignment of Large Video Language Models with Refined Regularized Preference Optimization
description: >-
  [NeurIPS 2025][Video Understanding][video LLM] This paper proposes RRPO (Refined Regularized Preference Optimization), which replaces DPO's response-level rewards with subsequence-level fine-grained rewards and token-wis…
tags:
  - "NeurIPS 2025"
  - "Video Understanding"
  - "video LLM"
  - "preference optimization"
  - "self-alignment"
  - "hallucination"
  - "temporal understanding"
date: 2026-05-08
content_hash: b80e43130e00155e
---

# Self-alignment of Large Video Language Models with Refined Regularized Preference Optimization

**Conference**: NeurIPS 2025
**arXiv**: [2504.12083](https://arxiv.org/abs/2504.12083)  
**Code**: [GitHub](https://pritamsarkar.com/RRPO)  
**Area**: LLM Alignment
**Keywords**: video LLM, preference optimization, self-alignment, hallucination, temporal understanding

## TL;DR
This paper proposes RRPO (Refined Regularized Preference Optimization), which replaces DPO's response-level rewards with subsequence-level fine-grained rewards and token-wise KL regularization. Combined with a self-alignment data generation framework, RRPO reduces hallucinations and improves temporal reasoning on video understanding tasks.

## Background & Motivation

**Core Problem of LVLMs**: Large video language models frequently make errors in fine-grained temporal understanding, hallucination, and simple QA tasks.

**Key Challenge**: Insufficient spatiotemporal understanding, misalignment between visual and linguistic representations, spurious correlations from co-occurring concepts, and over-reliance on language cues while ignoring visual information.

**Limitations of Prior Work — DPO**:
   - Response-level rewards are too coarse-grained, penalizing all tokens rather than the critical differing tokens.
   - Large gradients from long responses cause the model to deviate significantly from its initial state, losing original capabilities.
   - Weak regularization fails to effectively control this deviation.

## Method

### Overall Architecture: Self-Alignment Pipeline

1. Sample video-question pairs from open-source benchmarks.
2. Apply spatiotemporal perturbations to videos (frame masking 25%–50% + temporal shuffling).
3. Use perturbed videos for inference; incorrect responses serve as non-preferred, correct responses as preferred.
4. Use an LLM to identify key differing concepts between preferred and non-preferred responses.
5. Optimize model preferences with RRPO.

### Key Designs: RRPO

**Subsequence-Level Fine-Grained Rewards**: Rewards are computed only on **the key differing concept subsequences** between preferred and non-preferred responses, rather than over entire responses:

$$u = \sum_{i=1}^N (r_\theta(x, y_i^+) - r_\theta(x, y_i^-))$$

where $y_i^+$ and $y_i^-$ denote the $i$-th differing subsequence.

**Token-wise KL Regularization**: Token-level KL divergence is computed on the preferred response to prevent model deviation:

$$\mathbb{D}_{\text{TKL}}(x, y; \pi_{\text{ref}} \| \pi_\theta) = \sum_{t=1}^{|y|} \mathbb{D}_{\text{KL}}(\pi_{\text{ref}}(\cdot|[x,y_{<t}]) \| \pi_\theta(\cdot|[x,y_{<t}]))$$

**Final Loss**:

$$\mathcal{L}_{\text{RRPO}} = -\mathbb{E}[\log\sigma(u) + \alpha \cdot \mathbb{D}_{\text{TKL}}(x, y^+)]$$

### Gradient Analysis

The gradient upper bound of RRPO is $\|\nabla_\theta \mathcal{L}_{\text{RRPO}}^{(\text{rank})}\| \leq \beta M(2NL)$, while that of DPO is $\|\nabla_\theta \mathcal{L}_{\text{DPO}}\| \leq \beta M(|y^+|+|y^-|)$. Since $2NL \ll |y^+|+|y^-|$, RRPO produces smaller gradients and more stable updates. The negative gradient contributed by the TKL term further reduces the total gradient magnitude.

### Loss & Training

- LoRA is used to train only the LLM component; all other parameters are frozen.
- Training uses 16 frames; more frames can be used at inference.
- Training is conducted on 4×A100 80GB GPUs for 1–10 hours.
- Three base models are evaluated: VideoChat2, LLaVA-Video, and LongVU.

## Key Experimental Results

### Main Results: RRPO vs. Other Alignment Methods

| Method | TVBench | VideoHallucer | VideoMME | MLVU | Δ/%Δ |
|--------|---------|---------------|----------|------|------|
| LongVU (base) | 53.7 | 39.2 | 56.2 | 63.6 | - |
| + DPO | 54.3 | 40.9 | 56.6 | 63.6 | 0.7/1.5 |
| + DPA | 54.6 | 40.3 | 56.9 | 63.9 | 0.7/1.5 |
| + TDPO | 53.9 | 41.4 | 57.0 | 63.8 | 0.8/1.9 |
| + **RRPO** | **56.5** | **44.0** | **57.7** | **64.5** | **2.5/5.4** |

### Comparison with Existing Aligned LVLMs

| Model | TVBench | VideoHallucer | VideoMME | MLVU |
|-------|---------|---------------|----------|------|
| LLaVA-Video-TPO | 51.1 | 50.6 | 65.6/71.5 | 68.7 |
| **LLaVA-Video-RRPO** | **52.2** | **55.8** | 65.5/**71.8** | **69.4** |

RRPO outperforms TPO across all setups, with a gain of 5.2% on VideoHallucer.

### Ablation Study

| Variant | TVBench | VideoHallucer | Δ |
|---------|---------|---------------|---|
| RRPO w/o fine-grained reward | 54.3 | 43.0 | −1.5 |
| RRPO w/o TKL | 54.9 | 39.1 | −2.6 |
| Full RRPO | 56.5 | 44.0 | baseline |

Both components contribute positively, with TKL regularization having the larger impact.

### Model Deviation Analysis

- RRPO KL divergence ≈ 1 (using a 10× higher learning rate)
- DPO KL divergence ≈ 20
- TDPO/DPA KL divergence ≈ 1, but with significantly worse performance
- RRPO achieves the optimal performance–deviation trade-off

### Key Findings

1. Temporal understanding improves by up to 2.8% (TVBench).
2. Hallucinations are reduced by 4.8%–8.8% (VideoHallucer).
3. Consistent improvements are observed on both short- and long-video understanding.
4. Among perturbation strategies, Mask + Local Shuffle performs best.

## Highlights & Insights

1. **Self-Alignment Data Generation**: Spatiotemporal perturbations are used to elicit model errors, eliminating the need for manual annotation.
2. **Theoretically Grounded Gradient Analysis**: Mathematical proofs demonstrate that RRPO produces smaller and more stable gradients.
3. **Concept-Level Precise Alignment**: Only differing concepts are penalized rather than entire responses, avoiding over-penalization.
4. **TKL as a Trust Region Constraint**: Prevents large model deviation while permitting higher learning rates.

## Limitations & Future Work

1. The perturbation strategy remains relatively simple (frame masking + shuffling); more complex visual perturbations may be more effective.
2. The method relies on GPT-4o-mini for concept comparison and correctness verification.
3. Experiments are conducted only on 7B models; generalization to larger models remains to be verified.
4. A discrepancy exists between training frames (16) and inference frames (64–100), potentially introducing distribution shift.

## Related Work & Insights

- **DPO**: The starting point for RRPO; this work addresses its response-level reward granularity and weak regularization.
- **TDPO**: Enhances DPO regularization but yields limited performance gains; RRPO simultaneously improves both reward design and regularization.
- **DDPO**: Provides fine-grained rewards but lacks strong regularization; RRPO combines the advantages of both.
- **Insight**: The subsequence-level reward paradigm is generalizable to any preference optimization scenario.

## Rating

- Novelty: ⭐⭐⭐⭐ The combined design of subsequence-level rewards and TKL regularization is innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three base models × eight benchmarks, with comprehensive comparisons and ablations.
- Writing Quality: ⭐⭐⭐⭐ Gradient analysis is clearly presented and experiments are extensive.
- Value: ⭐⭐⭐⭐ Offers practical reference value for video LLM alignment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] LongVPO: From Anchored Cues to Self-Reasoning for Long-Form Video Preference Optimization](longvpo_from_anchored_cues_to_selfreasoning_for_longform_vid.md)
- [\[NeurIPS 2025\] MoniTor: Exploiting Large Language Models with Instruction for Online Video Anomaly Detection](monitor_exploiting_large_language_models_with_instruction_for_online_video_anoma.md)
- [\[NeurIPS 2025\] SAMA: Towards Multi-Turn Referential Grounded Video Chat with Large Language Models](sama_towards_multi-turn_referential_grounded_video_chat_with_large_language_mode.md)
- [\[NeurIPS 2025\] FastVID: Dynamic Density Pruning for Fast Video Large Language Models](fastvid_dynamic_density_pruning_for_fast_video_large_languag.md)
- [\[CVPR 2026\] Token Reduction via Local and Global Contexts Optimization for Efficient Video Large Language Models](../../CVPR2026/video_understanding/token_reduction_via_local_and_global_contexts_optimization_for_efficient_video_l.md)

</div>

<!-- RELATED:END -->

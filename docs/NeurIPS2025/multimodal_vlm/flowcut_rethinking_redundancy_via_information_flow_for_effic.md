---
title: >-
  [Paper Note] FlowCut: Rethinking Redundancy via Information Flow for Efficient Vision-Language Models
description: >-
  [NeurIPS 2025][Multimodal VLM][visual token pruning] Reexamining visual token redundancy in VLMs through the lens of Information Flow: the CLS token acts as an information relay, redundancy emerges progressively…
tags:
  - "NeurIPS 2025"
  - "Multimodal VLM"
  - "visual token pruning"
  - "information flow"
  - "VLM efficiency"
  - "attention analysis"
  - "training-free"
date: 2026-05-08
content_hash: 2c75120c99bef26a
---

# FlowCut: Rethinking Redundancy via Information Flow for Efficient Vision-Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2505.19536](https://arxiv.org/abs/2505.19536)  
**Code**: [https://github.com/TungChintao/FlowCut](https://github.com/TungChintao/FlowCut)  
**Area**: Model Compression / Multimodal VLM
**Keywords**: visual token pruning, information flow, VLM efficiency, attention analysis, training-free

## TL;DR
Reexamining visual token redundancy in VLMs through the lens of Information Flow: the CLS token acts as an information relay, redundancy emerges progressively, and single-layer single-criterion scoring is unreliable. FlowCut—an information-flow-aware multi-criteria cumulative importance pruning framework—surpasses SOTA by 1.6% on LLaVA-1.5-7B at an 88.9% token reduction rate, and by 4.3% on LLaVA-NeXT-7B.

## Background & Motivation
VLMs incur high computational cost due to large numbers of visual tokens. Existing methods rank and prune tokens using single-layer attention scores (e.g., FastV, SparseVLM). The authors raise a fundamental challenge: given the complexity of token–token and layer–layer interactions, is a single-layer attention score truly sufficient to identify redundant tokens?

## Core Problem
What is the fundamental nature of visual token redundancy from an information flow perspective? How can pruning strategies be designed to align with the model's intrinsic information propagation behavior?

## Method

### Overall Architecture
Three key modules: (1) layer-adaptive pruning ratios based on attention entropy; (2) multi-criteria scoring (attention strength + information density + semantic relevance); and (3) cumulative flow importance tracking.

### Key Designs (Information Flow Analysis)

1. **Insight 1 — CLS token as information relay**: Analysis of per-layer information flow-in and flow-out in ViT reveals that patch tokens primarily attend to neighboring tokens and the CLS token in shallow layers, and to distant "hub tokens" in deep layers. The CLS token acts as a global information broadcaster—first aggregating information from all patch tokens, then redistributing it. **The CLS token's attention therefore serves as a proxy for global information flow.**

2. **Insight 2 — Progressive emergence of redundancy**: The attention distribution of the CLS token becomes increasingly concentrated with depth (entropy decreases), with a sharp drop at layers 11–15. Redundancy is not a static property but emerges layer by layer during encoding.

3. **Insight 3 — Unreliability of single-criterion scoring**: Tokens that receive high CLS attention may nonetheless have low information density (small Value L1 norm) or low semantic relevance (low cosine similarity with CLS)—causing different criteria to produce contradictory importance rankings.

### Key Designs

1. **Layer-Adaptive Pruning Ratio**: Attention entropy guides pruning intensity at each layer—layers with low entropy (concentrated attention → more redundancy) are pruned more aggressively; layers with high entropy are pruned conservatively. Fixed per-layer ratios are replaced entirely.

2. **Multi-Criteria Fusion Scoring**: Token importance is assessed along three dimensions:

    - **Attention strength**: Attention score from the CLS token to the target token
    - **Information density**: L1 norm of the Value vector (signal strength)
    - **Semantic relevance**: Cosine similarity with the CLS token (global semantic relevance)

3. **Cumulative Flow Importance Tracking**: Rather than relying solely on the current layer, importance is accumulated across layers: `S_cum^(l) = 0.5 × I_cur^(l) + 0.5 × S_cum^(l-1)`. Pruning is applied every two layers to ensure both historical and current information are considered.

### Loss & Training
- Entirely training-free; plug-and-play at inference time.
- Pruning is applied at intermediate layers of the ViT visual encoder.

## Key Experimental Results

**LLaVA-1.5-7B (retaining 64 tokens from 576, ↓88.9%)**:

| Method | Avg. Accuracy (Relative %) |
|--------|---------------------------|
| Vanilla (576 tokens) | 100% |
| FastV | 77.5% |
| SparseVLM | 84.6% |
| PDrop | 86.1% |
| VisionZip | 94.4% |
| **FlowCut** | **96.0% (+1.6%)** |

**LLaVA-NeXT-7B (retaining 32 tokens from 2880, ↓94.4%)**: FlowCut 91.9% vs. VisionZip 87.6% (+4.3%)

- Retaining 192 tokens (↓66.7%): FlowCut achieves 100.2%—**performance even exceeds the original model**.
- 3.2× speedup in the prefilling stage.
- Generalization demonstrated on InternVL2 and Qwen2-VL.

### Ablation Study Highlights
- **Multi-criteria vs. single-criterion**: Multi-criteria scoring substantially outperforms attention-only scoring.
- **Cumulative vs. single-layer**: Cumulative importance tracking improves performance by 2–3%.
- **Adaptive vs. fixed ratio**: Adaptive pruning outperforms a fixed ratio by 1–2%.
- **Contribution of each criterion**: Attention strength is the foundation; information density and semantic relevance each contribute approximately 0.5–1%.

## Highlights & Insights
- **Theory-driven**: Starting from the fundamental perspective of information flow, the three insights form a coherent progression; the analysis leads naturally to each design decision.
- **96% retained at 64 tokens**: Maintaining 96% relative performance with an 88.9% token reduction rate substantially outperforms all competing methods at the same compression level.
- **Possibility of exceeding the original model**: 100.2% at 192 tokens—post-pruning performance marginally surpasses the original, suggesting redundant tokens may be actively harmful.
- **CLS as information relay**: This finding has independent academic value for understanding ViT's internal mechanisms.

## Limitations & Future Work
- Information flow analysis is validated only on ViT-based visual encoders; other architectures (e.g., SigLIP without CLS) have not been verified and require global average token substitution.
- Multi-criteria weights (0.5/0.5) are manually set; learnable adaptive weights are a natural extension.
- Not yet combined with dimension-level compression methods such as KV-Latent.
- A 3.2× prefilling speedup is reported, but decoding-stage speedup (and KV cache size reduction) is not reported.
- Evaluation focuses primarily on understanding tasks; the degree of impact on generation tasks (e.g., image captioning) is not sufficiently tested.

## Related Work & Insights
- **vs. FastV**: FastV performs one-shot pruning using layer-2 attention scores; FlowCut's cross-layer accumulation and multi-criteria scoring yield large gains.
- **vs. VisionZip**: VisionZip is a strong baseline but uses only attention with a small fixed retention budget; FlowCut's adaptive multi-criteria approach is comprehensively superior.
- **vs. mPLUG-DocOwl2 (cross-attention compression)**: DocOwl2 requires additional compression modules and training; FlowCut requires no training whatsoever.

## Related Work & Insights
- The information flow analysis framework can be generalized to token-level analysis in LLMs—similar attention concentration and redundancy emergence phenomena may exist in text tokens.
- The layer-adaptive pruning ratio is complementary to TrimLLM (layer-level compression)—FlowCut prunes tokens within layers, while TrimLLM removes entire layers.
- The "CLS as information relay" finding may be applicable to KV-Latent—the CLS token's KV representations could be maintained at higher dimensionality after dimensional down-sampling.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The information flow perspective and three derived insights are important theoretical contributions to the field.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four VLM architectures, 12 benchmarks, and detailed ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Information flow visualizations in Figures 2–4 are exceptionally elegant and intuitive.
- Value: ⭐⭐⭐⭐⭐ State-of-the-art in VLM token pruning with both strong theoretical grounding and practical applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Aligning What Vision-Language Models See and Perceive with Adaptive Information Flow](../../CVPR2026/multimodal_vlm/aif_adaptive_information_flow_vlm.md)
- [\[NeurIPS 2025\] SCOPE: Saliency-Coverage Oriented Token Pruning for Efficient Multimodal LLMs](scope_saliency-coverage_oriented_token_pruning_for_efficient_multimodel_llms.md)
- [\[NeurIPS 2025\] NaViL: Rethinking Scaling Properties of Native Multimodal Large Language Models under Data Constraints](navil_rethinking_scaling_properties_of_native_multimodal_large_language_models_u.md)
- [\[NeurIPS 2025\] Balanced Token Pruning: Accelerating Vision Language Models Beyond Local Optimization](balanced_token_pruning_accelerating_vision_language_models_b.md)
- [\[ICML 2026\] VEENA: Interpreting and Enhancing Emotional Circuits in Large Vision-Language Models via Cross-Modal Information Flow](../../ICML2026/multimodal_vlm/interpreting_and_enhancing_emotional_circuits_in_large_vision-language_models_vi.md)

</div>

<!-- RELATED:END -->

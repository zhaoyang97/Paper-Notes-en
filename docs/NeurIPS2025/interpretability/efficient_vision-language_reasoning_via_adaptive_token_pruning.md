---
title: >-
  [Paper Note] Efficient Vision-Language Reasoning via Adaptive Token Pruning
description: >-
  [NeurIPS 2025 (Workshop on VLM4RWD)][Interpretability][visual token pruning] This paper proposes Adaptive Token Pruning (ATP), a training-free plug-and-play module that selects the most informative visual tokens by fusin…
tags:
  - "NeurIPS 2025 (Workshop on VLM4RWD)"
  - "Interpretability"
  - "visual token pruning"
  - "inference acceleration"
  - "multimodal efficiency"
  - "training-free compression"
  - "edge deployment"
date: 2026-05-08
content_hash: e422a8a24d3ac377
---

# Efficient Vision-Language Reasoning via Adaptive Token Pruning

**Conference**: NeurIPS 2025 (Workshop on VLM4RWD)
**arXiv**: [2512.12701](https://arxiv.org/abs/2512.12701)  
**Code**: None  
**Area**: Interpretability
**Keywords**: visual token pruning, inference acceleration, multimodal efficiency, training-free compression, edge deployment

## TL;DR

This paper proposes Adaptive Token Pruning (ATP), a training-free plug-and-play module that selects the most informative visual tokens by fusing ViT CLS attention (intra-modal saliency) and CLIP text-image similarity (inter-modal relevance). ATP achieves less than 1% accuracy degradation on VQA/GQA/COCO Captioning in exchange for approximately 40% FLOPs reduction and 1.5× speedup.

## Background & Motivation

VLMs such as BLIP-2, LLaVA, and Flamingo typically pass all visual patch tokens produced by a ViT to the LLM, yet a large proportion of these tokens correspond to background regions or repetitive structures (e.g., blank walls, similar box surfaces), introducing redundant computation and memory overhead. In real-time scenarios such as robotics, autonomous driving, and assistive technologies, high latency and large memory requirements severely constrain edge deployment of VLMs.

Existing token reduction methods (e.g., Token Merging, Token Dropping) generally require retraining or modification of internal model structures, limiting their practical applicability. The core motivation of ATP is to design a lightweight, training-free, architecture-agnostic gating module inserted between the ViT and the LLM that forwards only the semantically most important visual tokens.

## Method

### Overall Architecture

ATP is positioned in the VLM inference pipeline between the final-layer output of the ViT and the vision-to-language projector. The ViT, projector, and LLM remain fully frozen; ATP executes once prior to visual-language interaction, maximizing computational savings during the subsequent LLM prefill stage.

ATP takes input from two sources:

- **Visual tokens**: $V = \{v_1, \ldots, v_N\}$, patch embeddings from the last layer of the ViT
- **Text embeddings**: $T$, encodings of the user prompt from a frozen CLIP text encoder

### Key Designs

#### Intra-Modal Saliency $S_{\text{intra}}(i)$

The CLS attention map from the last layer of the ViT is used to estimate the importance of each patch token within the visual modality. Tokens with high CLS attention typically correspond to salient regions in the image (objects, key structures), consistent with findings from interpretability research.

$$S_{\text{intra}}(i) = \frac{1}{Z} \sum_j \text{Attention}_{ij}$$

where $Z$ is a normalization constant. This serves as a query-agnostic measure of "objectness."

#### Inter-Modal Relevance $S_{\text{inter}}(i)$

This score evaluates the alignment between visual token $v_i$ and the textual prompt. The authors specifically emphasize using a CLIP text encoder (e.g., CLIP-ViT-L/14) matched to the frozen visual backbone of the VLM, ensuring that dot-product operations are performed within a unified embedding space.

$$S_{\text{inter}}(i) = \frac{E_{\text{ViT}}(v_i) \cdot E_{\text{CLIP}}(T)}{\|E_{\text{ViT}}(v_i)\| \; \|E_{\text{CLIP}}(T)\|}$$

Tokens semantically related to the prompt (e.g., "dog," "robot arm") receive higher scores.

#### Score Fusion and Top-K Selection

The two saliency terms are normalized and fused via weighted combination:

$$S(i) = \alpha \cdot N(S_{\text{inter}}(i)) + (1-\alpha) \cdot N(S_{\text{intra}}(i))$$

- High $\alpha$: ATP emphasizes query-focused relevance
- Low $\alpha$: ATP emphasizes general objectness-driven saliency

Tokens are ranked by $S(i)$ and the top-$K$ are retained: $V_{\text{pruned}} = \text{TopK}(V, K)$. Background patches (grass, sky, blank walls) are pruned.

### Inference Efficiency Gains

ATP token pruning yields two primary efficiency benefits:

1. **LLM FLOPs reduction**: The LLM processes a shorter visual prefix sequence, significantly reducing computation during the prefill stage
2. **KV-cache memory reduction**: Fewer visual tokens result in slower growth of the attention KV cache

Since ATP reuses the ViT CLS attention map and CLIP text embeddings, its own computational overhead is negligible.

### System Integration

ATP requires no retraining of the ViT or LLM, no modification of internal LLM attention layers, and no custom architectural changes. It is fully plug-and-play, making it suitable for edge scenarios such as real-time robotics and mobile deployment.

## Key Experimental Results

### Main Results

**Table 1: Preliminary Efficiency Analysis (LLaVA-7B backbone)**

| Method | Visual Token Count | Relative FLOPs | Estimated Accuracy Change |
|---|---|---|---|
| Baseline (Full) | 256 (100%) | 1.0× | — |
| ATP (Ours) | ~150 (60%) | 0.6× | <1% degradation |

**Table 2: Preliminary Cross-Task Results**

| Benchmark | Task Type | ATP Effect |
|---|---|---|
| VQAv2 | Visual Question Answering | ~40% FLOPs reduction, <1% accuracy loss |
| GQA | Compositional Reasoning | Similar efficiency gains |
| COCO Captioning | Image Captioning | Generation quality maintained |

### Ablation Study

The paper currently provides only preliminary observations:

- **Improved robustness**: Under visual corruptions such as Gaussian noise, blur, and occlusion, ATP prunes noisy background patches and retains stable object regions, improving the model's focus
- **Robustness to text perturbations**: When faced with paraphrased questions or distracting phrases, ATP prunes irrelevant patches and reduces hallucinated responses in small-scale tests
- The **$\alpha$ hyperparameter** and pruning schedule have not yet been thoroughly optimized

### Key Findings

1. ATP substantially reduces computational cost while preserving multimodal reasoning quality
2. Pruning not only improves efficiency but may also suppress spurious correlations and hallucination-inducing features, suggesting that resource-constrained inference and model reliability are not mutually exclusive
3. ATP can serve as a model interpretability tool — retained and pruned patches can be visualized to understand what the model attends to

## Highlights & Insights

- The approach is remarkably simple: fusing two readily available signals (CLS attention + CLIP similarity) for ranking, with no training required and full plug-and-play compatibility
- The finding that token pruning simultaneously improves efficiency and robustness is insightful — suggesting that redundant tokens not only waste computation but may also introduce noise
- Target application scenarios are clearly defined (robot vision, edge computing, warehouse monitoring)

## Limitations & Future Work

- This is a workshop paper with very limited experimental scale — only preliminary small-scale tests are conducted, lacking systematic benchmark comparisons
- The selection of core hyperparameters $\alpha$ and $K$ has not been sufficiently investigated
- No direct comparison is made with other token compression methods (Token Merging, SparseVLM, LV-Prune)
- Only single-image scenarios are validated; multi-image, video, and multi-turn dialogue settings are not addressed
- When the CLIP text encoder and the ViT do not share the same embedding space, the validity of the inter-modal relevance score is questionable

## Related Work & Insights

- Token Merging (ToMe) reduces sequence length by merging similar tokens but requires modification of model internals
- SparseVLM dynamically sparsifies visual tokens during LLM inference
- ATP's advantage lies in being fully external and training-free, at the cost of potentially less flexibility compared to deeply integrated methods
- The idea of using CLIP similarity as a cross-modal importance measure is generalizable to other multimodal architectures

## Rating

- Novelty: ⭐⭐⭐ — The idea is intuitive yet effective; the dual-signal fusion design is well-motivated
- Technical Depth: ⭐⭐ — A workshop paper with detailed method description but insufficient experimental depth
- Experimental Thoroughness: ⭐⭐ — Only preliminary testing; lacks systematic evaluation and comparison
- Practical Value: ⭐⭐⭐ — The plug-and-play design is highly attractive from an engineering perspective, but scaled validation is needed

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Learning to Focus: Causal Attention Distillation via Gradient-Guided Token Pruning](learning_to_focus_causal_attention_distillation_via_gradient-guided_token_prunin.md)
- [\[ICML 2026\] Towards Long-Horizon Interpretability: Efficient and Faithful Multi-Token Attribution for Reasoning LLMs](../../ICML2026/interpretability/towards_long-horizon_interpretability_efficient_and_faithful_multi-token_attribu.md)
- [\[NeurIPS 2025\] Probabilistic Token Alignment for Large Language Model Fusion](probabilistic_token_alignment_for_large_language_model_fusion.md)
- [\[NeurIPS 2025\] Explaining Similarity in Vision-Language Encoders with Weighted Banzhaf Interactions](explaining_similarity_in_vision-language_encoders_with_weighted_banzhaf_interact.md)
- [\[NeurIPS 2025\] AdaptGrad: Adaptive Sampling to Reduce Noise](adaptgrad_adaptive_sampling_to_reduce_noise.md)

</div>

<!-- RELATED:END -->

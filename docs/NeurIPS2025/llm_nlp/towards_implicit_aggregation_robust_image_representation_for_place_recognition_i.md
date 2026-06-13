---
title: >-
  [Paper Note] Towards Implicit Aggregation: Robust Image Representation for Place Recognition in the Transformer Era
description: >-
  [NeurIPS 2025][LLM/NLP][Visual Place Recognition] This paper proposes ImAge (Implicit Aggregation), which inserts learnable aggregation tokens at a specific layer of a Transformer backbone and leverages the intrinsic sel…
tags:
  - "NeurIPS 2025"
  - "LLM/NLP"
  - "Visual Place Recognition"
  - "Implicit Aggregation"
  - "ViT"
  - "Aggregation Token"
  - "DINOv2"
date: 2026-05-08
content_hash: 92abf2a9511d9ef0
---

# Towards Implicit Aggregation: Robust Image Representation for Place Recognition in the Transformer Era

**Conference**: NeurIPS 2025
**arXiv**: [2511.06024](https://arxiv.org/abs/2511.06024)  
**Code**: [GitHub](https://github.com/lu-feng/image)  
**Area**: LLM Evaluation
**Keywords**: Visual Place Recognition, Implicit Aggregation, ViT, Aggregation Token, DINOv2

## TL;DR

This paper proposes ImAge (Implicit Aggregation), which inserts learnable aggregation tokens at a specific layer of a Transformer backbone and leverages the intrinsic self-attention mechanism to implicitly aggregate patch features into a global descriptor, completely eliminating the need for an external aggregator. With the smallest descriptor dimensionality (6144) and fastest inference speed, ImAge surpasses SOTA methods such as SALAD and BoQ across multiple VPR benchmarks, and ranks 1st on the MSLS Challenge leaderboard.

## Background & Motivation

Visual Place Recognition (VPR) is a specialized form of image retrieval whose core challenge lies in encoding images into robust global descriptors. Over the past decade, the field has converged on a standard "backbone + aggregator" paradigm — patch features are first extracted by a CNN or ViT, then compressed into a global descriptor by aggregators such as NetVLAD, GeM, SALAD, or BoQ. This paradigm, however, suffers from the following issues:

**Structural redundancy**: The two-stage pipeline (feature extraction + aggregation) introduces unnecessary architectural complexity. The aggregator itself requires a large number of parameters (e.g., BoQ incurs 8.6M additional parameters).

**No opportunity for correction**: Traditional aggregators perform one-shot aggregation and immediately produce an output, with no chance for iterative refinement or correction.

**Aggregator design difficulty**: NetVLAD discards positional information; SALAD requires Sinkhorn iterations; BoQ introduces additional encoder blocks and cross-attention layers.

The core insight is that in the Transformer era, the self-attention mechanism inherently possesses global information aggregation capability. Inspired by the DINOv2-register work — where register tokens cache global information into auxiliary tokens — the authors find that inserting a small number of aggregation tokens into the backbone and allowing self-attention to naturally "transfer" patch information onto these tokens yields high-quality global descriptors, **without modifying the backbone or introducing any external aggregator**.

## Method

### Overall Architecture

The ImAge pipeline is remarkably concise. A pretrained ViT (e.g., DINOv2-base-register) is used as the backbone. The first $L_1$ layers process patch tokens normally. After layer $L_1$, $M$ learnable aggregation tokens are prepended to the patch token sequence to form $[a, z]$. In the subsequent $L_2$ layers, all tokens interact via self-attention. Only the aggregation token outputs are retained at the end, then flattened and L2-normalized to produce the global descriptor.

### Key Designs

1. **Implicit aggregation via self-attention**: When aggregation tokens $a$ and patch tokens $z$ enter MHSA together, the attention output naturally decomposes as:

$$\text{Attn}(Q,K,V) = [\underbrace{Q_aK_a^\top V_a}_{\text{Agg-Agg}} + \underbrace{Q_aK_z^\top V_z}_{\text{Agg-Patch}}, \; Q_zK_a^\top V_a + Q_zK_z^\top V_z]$$

The **Agg-Agg** term enables aggregation tokens to interact with each other to enhance their own representations, while the **Agg-Patch** term enables aggregation tokens to capture global contextual information from patch tokens. Design motivation: unlike one-shot aggregation, aggregation tokens are progressively refined across multiple subsequent Transformer blocks, achieving **progressive aggregation**.

2. **Aggregation token insertion strategy**: The authors propose inserting tokens **at the boundary between frozen and trainable layers** (e.g., the 4th-to-last layer in DINOv2), rather than at layer 1 as in prompt tuning. Two reasons are given:

    - Early shallow-layer features lack sufficient representational capacity, so inserting tokens too early prevents them from learning meaningful information.
    - If tokens are inserted before frozen layers, although frozen layer parameters are not updated, gradients must still be computed through them for training the aggregation tokens, wasting GPU memory.
    - Four insertion strategies are compared experimentally: (a) insertion at all layers, (b) insertion at the frozen-trainable boundary (optimal), (c) insertion at deeper layers, and (d) progressive layer-by-layer insertion.

3. **Aggregation token initialization**: Tokens are initialized using **k-means clustering followed by L2 normalization**, analogous to the role of cluster centers in NetVLAD — each aggregation token represents a semantically meaningful VPR-relevant category. L2 normalization reduces the influence of extreme values and is shown experimentally to outperform both raw cluster centers and random initialization. Concretely, k-means (k=M) is applied to patch tokens extracted from training images using the pretrained backbone, and the L2-normalized cluster centers are used as initial values for the aggregation tokens.

### Loss & Training

Training uses multi-similarity loss with 120 places per batch and 4 images per place (480 images total). The Adam optimizer is used with an initial learning rate of 5e-5, halved every 3 epochs, for a maximum of 20 epochs. Only the last 4 layers of the backbone are fine-tuned; earlier layers are frozen. Training resolution is 224×224 and inference resolution is 322×322. The primary training dataset is GSV-Cities; for the comprehensive comparison, Pitts30k-train, MSLS-train, SF-XL, and GSV-Cities are combined.

## Key Experimental Results

### Main Results (fair comparison, identical setting: DINOv2-base-reg, GSV-Cities)

| Method | Descriptor Dim | Aggregator Params | Inference (ms) | Pitts30k R@1 | MSLS-val R@1 | Tokyo24/7 R@1 | Nordland R@1 |
|--------|---------------|-------------------|----------------|-------------|-------------|--------------|-------------|
| NetVLAD | 6144 | 0.012M | 15.0 | 92.8 | 91.8 | 95.6 | 90.5 |
| SALAD | 8448 | 1.411M | 16.3 | 92.5 | 92.6 | 95.6 | 86.5 |
| BoQ | 12288 | 8.626M | 16.4 | 93.1 | 92.8 | 95.2 | 87.0 |
| **ImAge** | **6144** | **0 M** | **14.8** | **94.0** | **93.0** | **96.2** | **93.2** |

### Comprehensive Comparison (best setting per method)

| Method | Pitts30k R@1 | MSLS-val R@1 | MSLS-chall R@5 | Tokyo24/7 R@1 | Nordland R@1 |
|--------|-------------|-------------|----------------|--------------|-------------|
| SALAD-CM | 92.7 | 94.2 | 91.2 | 96.8 | 96.0 |
| BoQ | 93.7 | 93.8 | 90.3 | 96.5 | 90.6 |
| EDTformer | 93.4 | 92.0 | 89.8 | 97.1 | 88.3 |
| **ImAge** | **94.1** | **94.5** | **93.8** | **97.1** | **97.7** |

### Ablation Study

| Configuration | MSLS-val R@1 | Pitts30k R@1 | Note |
|---------------|-------------|-------------|------|
| Full ImAge (strategy b + k-means init) | **93.0** | **94.0** | Optimal |
| Strategy a (all-layer insertion) | 91.5 | 93.1 | Insufficient shallow features |
| Strategy c (deeper insertion) | 92.4 | 93.5 | Fewer refinement steps |
| Strategy d (progressive insertion) | 92.1 | 93.3 | Inferior to single-point insertion |
| Random initialization | 91.8 | 93.2 | k-means init is effective |
| Raw cluster centers (no L2 norm) | 92.3 | 93.6 | L2 normalization is beneficial |
| Token count M=4 | 92.2 | 93.4 | 8 tokens sufficient |
| Token count M=16 | 92.8 | 93.8 | Diminishing returns |

### Key Findings

- **Zero-parameter aggregator**: ImAge's aggregator has 0 parameters (only 8 aggregation tokens, ~0.006M, representing 0.07% of BoQ), yet consistently outperforms all baselines.
- **MSLS Challenge Rank 1**: ImAge achieves R@5 of 93.8% on the most challenging MSLS test set, surpassing all publicly available methods.
- **Cross-season recognition on Nordland**: R@1 reaches 97.7% and R@5 is nearly perfect (99.3%), significantly outperforming methods based on explicit aggregation, demonstrating that progressive implicit aggregation is more robust under extreme appearance changes.
- **Fastest inference**: At 14.8ms, ImAge is faster than NetVLAD (15.0ms), SALAD (16.3ms), and BoQ (16.4ms), since no external aggregator inference is required.

## Highlights & Insights

1. **Paradigm shift: "aggregators are unnecessary"**: This is a highly compelling conclusion — in the Transformer era, self-attention is itself the best aggregator. ImAge demonstrates this with maximal simplicity and elegance.
2. **Progressive refinement outperforms one-shot aggregation**: Aggregation tokens are continuously refined across multiple blocks, yielding superior results compared to the one-shot output of traditional aggregators — an approach conceptually akin to iterative refinement.
3. **Inheritance and innovation in k-means initialization**: The paper directly borrows the clustering idea from NetVLAD but realizes it more elegantly — no soft-assignment layers are needed; self-attention naturally performs the "assignment."
4. **Minimum dimensionality, maximum performance**: The 6144-dimensional descriptor (8 tokens × 768 dimensions) simultaneously achieves the lowest dimensionality and highest performance, indicating that implicit aggregation learns a more compact and efficient representation.

## Limitations & Future Work

- Strong dependence on Transformer backbones (requiring self-attention); the method is not applicable to CNN-based backbones.
- The number of aggregation tokens (M=8) is chosen empirically; while different values are explored experimentally, theoretical guidance is lacking.
- Validation is currently limited to VPR; whether implicit aggregation generalizes to other retrieval and representation tasks (e.g., person re-identification, fine-grained geo-localization regression) remains to be explored.
- Training depends on the quality of DINOv2 pretraining; if the backbone is insufficiently pretrained, the aggregation tokens may not function effectively.

## Related Work & Insights

- **Distinction from DINOv2-register**: Register tokens are discarded at inference time, whereas ImAge's aggregation tokens constitute the final output. Their functional roles differ — register tokens serve as a "garbage bin" buffering redundant global information, while aggregation tokens act as "collectors" actively gathering useful information.
- **Comparison with BoQ is most instructive**: BoQ also employs learnable queries but requires additional encoder blocks and cross-attention layers (8.6M parameters). ImAge demonstrates that these additional structures are unnecessary.
- The paper's taxonomy of three roles for auxiliary tokens in Transformers (output-oriented / prompt / memory) provides a useful reference framework for future work.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ A paradigm shift in VPR; aggregators are elegantly eliminated in a concise and principled manner.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive multi-dataset evaluation + MSLS leaderboard Rank 1 + fair same-setting comparison + detailed ablation study.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear logical flow from the limitations of the existing paradigm to the motivation for implicit aggregation; intuitive figures.
- **Value**: ⭐⭐⭐⭐⭐ Redefines VPR engineering practice with a method that is maximally simple yet maximally effective.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Writing in Symbiosis: Mapping Human Creative Agency in the AI Era](writing_in_symbiosis_mapping_human_creative_agency_in_the_ai_era.md)
- [\[NeurIPS 2025\] Spectral Conditioning of Attention Improves Transformer Performance](spectral_conditioning_of_attention_improves_transformer_performance.md)
- [\[NeurIPS 2025\] Characterizing the Expressivity of Fixed-Precision Transformer Language Models](characterizing_the_expressivity_of_fixed-precision_transformer_language_models.md)
- [\[NeurIPS 2025\] On the Role of Hidden States of Modern Hopfield Network in Transformer](on_the_role_of_hidden_states_of_modern_hopfield_network_in_transformer.md)
- [\[ICML 2026\] The Cylindrical Representation Hypothesis for Language Model Steering](../../ICML2026/llm_nlp/the_cylindrical_representation_hypothesis_for_language_model_steering.md)

</div>

<!-- RELATED:END -->

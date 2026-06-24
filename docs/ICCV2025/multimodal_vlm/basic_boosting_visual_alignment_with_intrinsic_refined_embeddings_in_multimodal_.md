---
title: >-
  [Paper Note] BASIC: Boosting Visual Alignment with Intrinsic Refined Embeddings in Multimodal Large Language Models
description: >-
  [ICCV 2025][Multimodal VLM][MLLM] By analyzing the semantic refinement of visual embeddings in the shallow layers of LLMs, this paper proposes BASIC, a method that leverages intrinsically refined visual embeddings from within the LLM as supervision signals to directly guide the visual projector in generating better initial visual embeddings along two dimensions: directional alignment and semantic distribution.
tags:
  - "ICCV 2025"
  - "Multimodal VLM"
  - "MLLM"
  - "Visual Embedding"
  - "Self-Distillation"
  - "Modality Alignment"
  - "LLM Interpretability"
date: 2026-05-08
content_hash: 2bf1d60b315f9c9f
---

# BASIC: Boosting Visual Alignment with Intrinsic Refined Embeddings in Multimodal Large Language Models

**Conference**: ICCV 2025
**arXiv**: [2508.06895](https://arxiv.org/abs/2508.06895)  
**Code**: None  
**Area**: Multimodal VLM / Vision-Language Alignment
**Keywords**: MLLM, Visual Embedding, Self-Distillation, Modality Alignment, LLM Interpretability

## TL;DR

By analyzing the semantic refinement of visual embeddings in the shallow layers of LLMs, this paper proposes BASIC, a method that leverages intrinsically refined visual embeddings from within the LLM as supervision signals to directly guide the visual projector in generating better initial visual embeddings along two dimensions: directional alignment and semantic distribution.

## Background & Motivation

Mainstream MLLMs (e.g., LLaVA, InternVL, Qwen2-VL) adopt a "visual encoder – visual projector – LLM" architecture. A critical bottleneck is that **visual embeddings are treated merely as contextual cues**: autoregressive supervision during training is applied only to text tokens, while **direct supervision over visual embeddings is entirely absent**.

This asymmetric supervision introduces two problems:
1. Rich information in visual data is not fully exploited.
2. Fine-grained alignment between visual and linguistic representations is limited.

Existing alternatives each have shortcomings:
- Discrete visual tokens (Chameleon/SEED-LLaMA): discretization causes information loss.
- L2 regression on next-position embeddings (Emu1/2): good for image generation, but visual understanding lags behind.
- Additional VM-head (VW-LMM): requires a complex four-stage training pipeline.

A key finding of this paper is that **the shallow layers of LLMs automatically refine visual embeddings**—visual embeddings that initially match semantically irrelevant text tokens progressively align to meaningful tokens in shallow layers. These refined embeddings can serve as high-quality supervision signals.

## Method

### Overall Architecture

BASIC augments the standard MLLM training pipeline with two direct visual supervision losses, using refined visual embeddings from the shallow layers of the LLM to supervise the initial visual embeddings produced by the projector. No additional models or annotated data are required.

### Key Designs

1. **Analysis of the Visual Perception Process**

    - For each initial visual embedding $\boldsymbol{v}_i$, the cosine similarity with all token embeddings in the LLM vocabulary is computed to identify the best-matching token.
    - Finding: some initial embeddings already match semantically relevant tokens (e.g., "clock", "white").
    - When visual embeddings are replaced by their matched token embeddings, the LLM still generates descriptions highly consistent with the original image → indicating that the LLM understands images by interpreting textual concepts encoded in visual embeddings.
    - Cross-layer tracking reveals a key pattern:
        - **Shallow layers** (1–16/32): meaningless matches progressively transition to meaningful ones.
        - **Deep layers** (16–32/32): embeddings tend to match the special end token `</s>`.

2. **Construction of Supervision Visual Embeddings**

    - Refined visual embeddings from the shallow layers (first $k$ layers) of the LLM are aggregated with learned weights:
    $\hat{V} = \sum_{i=1}^{k} w_i \tilde{V}_i, \quad w_i = \frac{i^2}{\sum_{i=1}^{k} i^2}$
    - Weights increase quadratically with layer depth: deeper shallow-layer representations are better refined and thus receive higher weights.
    - Attention-based supervision strength accounts for varying importance across image patches:
    $a_i = \frac{1}{kn} \sum_{h=1}^{k} \sum_{j=1}^{n} a_{h,j,i}$
    - $a_i$ denotes the average attention score of text tokens toward the $i$-th image patch.

3. **Directional Alignment Supervision (DAS)**

    - Initial and supervision embeddings are projected onto the unit hypersphere, and their angular distance is minimized:
    $\mathcal{L}_{das} = \sum_{i=1}^{m} a_i \left\| \frac{\boldsymbol{v}_i}{\|\boldsymbol{v}_i\|_2} - \frac{\hat{\boldsymbol{v}}_i}{\|\hat{\boldsymbol{v}}_i\|_2} \right\|_2^2$
    - This eliminates the influence of magnitude and focuses exclusively on semantic directional alignment.

4. **Semantic Distribution Supervision (SDS)**

    - Logit vectors are obtained by computing the inner product between visual embeddings and the full vocabulary, reflecting global semantic associations.
    - The KL divergence between the logit distributions of the initial and supervision embeddings is minimized:
    $P = \text{softmax}(\hat{V}E^\top), \quad Q = \text{softmax}(VE^\top)$
    $\mathcal{L}_{sds} = \sum_{i=1}^{m} a_i \text{KL}(\boldsymbol{p}_i \| \boldsymbol{q}_i)$
    - This ensures consistency of semantic distributions of visual embeddings across the entire vocabulary space.

### Loss & Training

Total loss: $\mathcal{L} = \mathcal{L}_{lm} + \lambda_1 \mathcal{L}_{das} + \lambda_2 \mathcal{L}_{sds}$

- $\lambda_1 = 1$, $\lambda_2 = 0.01$
- Pre-training stage: only the projector is trainable; LLaVA-558k data.
- Instruction tuning stage: projector and LLM are both trainable; LLaVA-665k data.
- For 7B models, the first 16/32 layers are used; for 13B models, the first 20/40 layers.

## Key Experimental Results

### Main Results

| Method | LLM | VQAv2 | GQA | SQA-I | MMB-EN | MMB-CN | MM-Vet | VizWiz |
|--------|-----|-------|-----|-------|--------|--------|--------|--------|
| LLaVA-1.5 | Vicuna-7B | 78.5 | 62.0 | 66.8 | 64.3 | 58.3 | 31.1 | 50.0 |
| **BASIC** | Vicuna-7B | **79.2** | **63.5** | **70.6** | **68.8** | **62.1** | **33.8** | **52.5** |
| LLaVA-1.5 | Vicuna-13B | 80.0 | 63.3 | 71.6 | 67.7 | 63.6 | 36.1 | 53.6 |
| **BASIC** | Vicuna-13B | **80.6** | **64.6** | **73.1** | **69.6** | **64.9** | **37.2** | **55.8** |

The 7B model achieves gains of **+3.8** on SQA-I, **+4.5** on MMB-EN, and **+3.8** on MMB-CN.

### Ablation Study

| $\mathcal{L}_{das}$ | $\mathcal{L}_{sds}$ | VQAv2 | GQA | SQA-I | MMB-EN | MM-Vet |
|---|---|-------|-----|-------|--------|--------|
| ✗ | ✗ | 78.5 | 62.0 | 66.8 | 64.3 | 31.1 |
| ✓ | ✗ | 78.9 | 63.0 | 68.5 | 68.6 | 33.1 |
| ✗ | ✓ | 79.1 | 63.3 | 68.1 | 68.0 | 32.5 |
| ✓ | ✓ | **79.2** | **63.5** | **70.6** | **68.8** | **33.8** |

Cross-model generalization (all compared under the LLaVA architecture):

| VE + LLM | LLaVA MMB-EN | BASIC MMB-EN | Gain |
|----------|-------------|-------------|------|
| CLIP-L + Gemma-2B | 54.0 | 55.8 | +1.8 |
| CLIP-L + Phi3-3.8B | 68.7 | 70.2 | +1.5 |
| CLIP-L + Mistral-7B | 70.0 | 72.1 | +2.1 |
| SigLIP + Vicuna-7B | 68.0 | 69.6 | +1.6 |
| SigLIP + Vicuna-13B | 69.5 | 70.8 | +1.3 |

Ablation on layer weights and supervision strength:

| Setting | VQAv2 | MMB-EN | MM-Vet |
|---------|-------|--------|--------|
| $w_i$ decreasing | 72.9 | 57.4 | 27.8 |
| $w_i$ constant | 73.2 | 57.9 | 28.3 |
| **$w_i$ increasing** | **73.4** | **58.4** | **28.7** |
| Uniform supervision | 73.0 | 57.6 | 27.6 |
| **Attention-weighted** | **73.4** | **58.4** | **28.7** |

### Key Findings

- Both supervision losses are individually effective; their combination yields the best results.
- Using refined embeddings from the lower half of the LLM (layers 1–$l/2$) is optimal; incorporating deep-layer embeddings is detrimental, as deep layers tend to predict `</s>`.
- Quadratically increasing layer weights outperform decreasing and constant variants, confirming that the refinement process strengthens progressively across layers.
- Attention-weighted supervision outperforms uniform weighting, indicating the importance of accounting for varying patch relevance.
- BASIC exhibits a slight decline on TextVQA (−0.2), as the supervision signal emphasizes "semantic concepts" and may obscure fine-grained textual details.
- The proportion of semantically meaningful initial embeddings increases from 74/576 to 217/576 (BASIC vs. LLaVA, statistics over 30 images).

## Highlights & Insights

- **A distinctive application of self-distillation**: the LLM's shallow layers serve as a "teacher" to guide the projector at the input level, without requiring any external model.
- **Interpretability-driven method design**: the paper first analyzes how the LLM processes visual embeddings across layers, then designs the method based on these findings.
- **Zero additional overhead**: no extra supervision models or manual annotations are needed, and the approach is applicable to any "VE–Projector–LLM" architecture.
- **Complementary design from directional and distributional perspectives**: directional alignment adjusts the overall semantic orientation, while distribution supervision ensures consistency of vocabulary-space associations.

## Limitations & Future Work

- A slight performance drop on TextVQA suggests that semantic-level supervision may be disadvantageous for tasks requiring precise recognition of fine-grained text.
- Supervision signals are derived from the shallow-layer outputs of a frozen LLM; changes in LLM weights during instruction tuning may affect supervision quality.
- Validation is conducted only within the LLaVA-1.5 framework; integration with more recent architectures such as InternVL and Qwen2-VL has not been explored.
- Dynamic adjustment strategies for $k$ (the number of supervised layers) remain unexplored.

## Related Work & Insights

- **Comparison with Emu1/2**: Emu employs L2 regression on next-position embeddings for unified modeling, whereas BASIC uses shallow-layer refined embeddings for directional and distributional supervision.
- **Comparison with VW-LMM**: VW-LMM requires an additional VM-head and a four-stage training pipeline; BASIC is considerably simpler.
- **Logit Lens inspiration**: the LLM head is used to project hidden states for interpreting intermediate representations, an idea cleverly leveraged here.
- The cross-layer self-distillation paradigm generalizes to alignment problems in other modular architectures.

## Rating

- Novelty: ⭐⭐⭐⭐ The idea of deriving supervision signals from the LLM's internal refinement process is highly inspiring.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 8 benchmarks, 5 LLMs, 2 visual encoders, and detailed ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Thorough analysis, polished figures, and clear logical flow.
- Value: ⭐⭐⭐⭐ A plug-and-play MLLM enhancement method that is both practical and general.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Visual-Oriented Fine-Grained Knowledge Editing for MultiModal Large Language Models](visual-oriented_fine-grained_knowledge_editing_for_multimodal_large_language_mod.md)
- [\[CVPR 2026\] Unleashing the Intrinsic Visual Representation Capability of Multimodal Large Language Models](../../CVPR2026/multimodal_vlm/unleashing_the_intrinsic_visual_representation_capability_of_multimodal_large_la.md)
- [\[ICCV 2025\] CapeLLM: Support-Free Category-Agnostic Pose Estimation with Multimodal Large Language Models](capellm_support-free_category-agnostic_pose_estimation_with_multimodal_large_lan.md)
- [\[ICCV 2025\] AIGI-Holmes: Towards Explainable and Generalizable AI-Generated Image Detection via Multimodal Large Language Models](aigi_holmes_towards_explainable_and_generalizable_ai_generated_image_detection_via_mllm.md)
- [\[ICCV 2025\] FALCON: Resolving Visual Redundancy and Fragmentation in High-resolution Multimodal Large Language Models via Visual Registers](falcon_resolving_visual_redundancy_and_fragmentation_in_high.md)

</div>

<!-- RELATED:END -->

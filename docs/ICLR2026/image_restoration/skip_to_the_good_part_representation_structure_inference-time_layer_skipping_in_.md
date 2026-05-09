---
title: >-
  [Paper Note] Skip to the Good Part: Representation Structure & Inference-Time Layer Skipping in Diffusion vs. Autoregressive LLMs
description: >-
  [ICLR 2026][Image Restoration][Diffusion Language Models] This work presents the first systematic comparison of layer-wise representation structure between diffusion large language models (dLLMs) and autoregressive (AR) LLMs. It finds that natively trained dLLMs exhibit stronger hierarchical abstraction and greater early-layer redundancy. Based on this finding, a static, task-agnostic inference-time layer skipping strategy is proposed, achieving 90%+ performance retention on LLaDA while skipping 6 layers (18.75% FLOPs reduction).
tags:
  - ICLR 2026
  - Image Restoration
  - Diffusion Language Models
  - Layer Skipping
  - Representation Redundancy
  - Inference Acceleration
  - LLaDA
date: 2026-05-08
content_hash: 6ab55d4356f102af
---

# Skip to the Good Part: Representation Structure & Inference-Time Layer Skipping in Diffusion vs. Autoregressive LLMs

**Conference**: ICLR 2026
**arXiv**: [2603.07475](https://arxiv.org/abs/2603.07475)
**Code**: None
**Area**: Image Restoration
**Keywords**: Diffusion Language Models, Layer Skipping, Representation Redundancy, Inference Acceleration, LLaDA

## TL;DR
This work presents the first systematic comparison of layer-wise representation structure between diffusion large language models (dLLMs) and autoregressive (AR) LLMs. It finds that natively trained dLLMs exhibit stronger hierarchical abstraction and greater early-layer redundancy. Based on this finding, a static, task-agnostic inference-time layer skipping strategy is proposed, achieving 90%+ performance retention on LLaDA while skipping 6 layers (18.75% FLOPs reduction).

## Background & Motivation

**Background**: Autoregressive (AR) language models build representations incrementally via left-to-right next-token prediction, while diffusion language models (dLLMs) such as LLaDA and Dream are trained through full-sequence denoising. Recent dLLMs have approached AR model performance on reasoning and code generation tasks.

**Limitations of Prior Work**: Despite converging performance, the internal representation structure differences between dLLMs and AR models remain unsystematically studied. Existing efficiency optimization work (e.g., YOCO) focuses on architecture-level modifications such as KV-cache sharing, which require structural changes to the model.

**Key Challenge**: Does the training objective (diffusion vs. autoregressive) fundamentally reshape the model's internal representations? If systematic differences exist, can they be directly exploited to accelerate inference?

**Goal**: (a) Quantify differences in layer-wise and token-wise representation similarity between dLLMs and AR LLMs; (b) investigate the persistent influence of AR initialization on dLLM representations; (c) leverage identified representational redundancy to accelerate inference without architectural modifications.

**Key Insight**: Layer-wise cosine similarity analysis reveals that natively trained dLLMs exhibit highly similar early-layer representations (>0.95), indicating that computation in these layers is redundant and can be safely skipped.

**Core Idea**: The diffusion training objective induces a hierarchical "coarse-to-fine" abstraction: early layers establish rough representations (high redundancy) while later layers perform refinement. This redundancy can be directly exploited for static layer skipping without KV-cache sharing or architectural modifications.

## Method

### Overall Architecture

Three model families are compared: LLaDA-8B (native dLLM), Qwen2.5-7B (native AR), and Dream-7B (AR-initialized dLLM). Layer-wise and token-wise representation similarity analyses are first conducted, after which a layer skipping strategy is designed based on the findings and applied at inference time to directly skip high-similarity layers.

### Key Designs

1. **Layer-wise Cosine Similarity Analysis**:

    - **Function**: Tracks cosine similarity between consecutive layer representations $\mathbf{h}_\ell$ and $\mathbf{h}_{\ell+1}$.
    - **Core Finding**: LLaDA exhibits similarity >0.95 in early layers (the first 60–70%), forming a high-redundancy plateau, followed by a drop in similarity in later layers (refinement region). Qwen2.5 shows lower and more uniformly distributed similarity across all depths (no salient redundancy). Dream-7B, despite diffusion fine-tuning, exhibits a representation pattern highly consistent with Qwen2.5, indicating a strong initialization bias.
    - **Design Motivation**: Quantifies the effect of training objectives on representation structure and provides a theoretical basis for layer skipping.

2. **Token-wise Similarity and Recency Bias Analysis**:

    - **Function**: Analyzes token-wise representation variation patterns across different layers.
    - **Core Finding**: LLaDA exhibits minimal recency bias (globally smooth, high similarity), whereas both Qwen2.5 and Dream-7B show pronounced recency bias, with each new token inducing cross-layer representation changes.
    - **Design Motivation**: Validates that the full-sequence feedback training of dLLMs leads to more globally distributed representational abstraction.

3. **Static Layer Skipping Strategy (Algorithm 1)**:

    - **Function**: Skips redundant high-similarity layers at inference time.
    - **Mechanism**: Based on similarity analysis from training, layers with cosine similarity > θ (default 0.95) are selected. Layers are ranked in descending order of similarity, and the top-$k$ layers are chosen for skipping, with the constraint that consecutive layers may not be skipped simultaneously (to maintain representational continuity). During skipping, $\mathbf{h}_{\ell-1}$ is passed directly to layer $\ell+1$.
    - **Design Motivation**: A purely inference-time operation requiring no retraining, no KV-cache sharing, and no architectural modifications; orthogonal to and composable with KV-cache optimizations.

### Loss & Training

This work does not modify the training process; layer skipping is applied solely at inference time. Analysis is conducted on pre-trained LLaDA-8B, Qwen2.5-7B, and Dream-7B models.

## Key Experimental Results

### Main Results

| Model | Layers Skipped | FLOPs Reduction | GSM8K Retention | HumanEval Retention | MATH500 Retention |
|-------|---------------|-----------------|-----------------|---------------------|-------------------|
| LLaDA-8B | 0 | 0% | 100% (0.83) | 100% (0.51) | 100% (0.37) |
| LLaDA-8B | 2 | 6.25% | 101.3% | 100% | 108.5% |
| LLaDA-8B | 4 | 12.5% | 102.5% | 92.2% | 89.4% |
| LLaDA-8B | 6 | 18.75% | 91.8% | 88.2% | 102.1% |
| LLaDA-8B | 8 | 25% | 91.8% | 62.7% | 70.2% |
| Qwen2.5-7B | 2 | 7.14% | 34.9% | 64.7% | - |
| Dream-7B | 2 | 7.14% | 76.8% | 66.2% | 81.4% |

### Ablation Study

| Configuration | GSM8K Retention | HumanEval Retention | Notes |
|---------------|-----------------|---------------------|-------|
| LLaDA 6 layers, non-consecutive skipping | 91.8% | 88.2% | Full method |
| LLaDA 6 layers, consecutive skipping allowed | 75.3% | 64.7% | Consecutive skipping causes severe degradation |
| Dream-7B 2 layers, non-consecutive skipping | 76.8% | 66.2% | AR initialization limits skipability |
| Dream-7B 2 layers, consecutive skipping | 14.1% | - | Collapse |

### Key Findings
- **Native dLLMs support aggressive skipping**: LLaDA retains 88–102% performance when skipping 6 layers, whereas Qwen2.5 collapses at just 2 skipped layers (34.9%).
- **AR initialization bias persists**: After diffusion fine-tuning, Dream-7B's representation patterns remain consistent with Qwen2.5, and its layer-skipping robustness similarly resembles that of AR models.
- **Prohibiting consecutive skipping is critical**: Consecutive skipping breaks representational continuity and leads to severe performance degradation.
- **Skipped layers are concentrated in the first 40–60% of the network**: Consistent with the analysis that early layers establish coarse representations.
- dLLMs achieve 2.6× greater FLOPs reduction and 1.4× higher quality retention compared to AR models.

## Highlights & Insights
- **Training objective determines representation structure**: A key finding is that it is the training objective—not the architecture—that governs representational redundancy patterns. This implies that desired representation properties could in principle be "designed" by selecting appropriate training objectives.
- **Depth of initialization bias**: The Dream-7B case demonstrates that diffusion fine-tuning is insufficient to override the representational structure instilled by AR pretraining; initialization matters more than the fine-tuning objective.
- **Orthogonality to KV-cache**: Layer skipping reduces depth-wise computation while KV-cache reduces token-wise redundant computation; the two approaches can be multiplicatively combined.

## Limitations & Future Work
- Only 7–8B scale models are evaluated; redundancy patterns in larger models may differ.
- The skipping strategy is static; dynamic or adaptive skipping may yield further improvements.
- The possibility of training dLLMs specifically optimized for layer skipping (e.g., via inter-layer regularization) is not explored.
- Benchmark coverage is limited (GSM8K, MATH500, HumanEval, MBPP).

## Related Work & Insights
- **vs. YOCO**: YOCO requires a cache-once architectural design; the proposed method requires no architectural modifications.
- **vs. Early Exit**: Early Exit terminates computation during the prefill stage, whereas this work applies layer skipping at every step of full-sequence denoising.
- **vs. DiffuCoder (Gong et al., 2025)**: DiffuCoder analyzes the AR-ness of dLLMs from a behavioral perspective; this work provides a complementary view at the representational level.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic analysis of dLLM vs. AR representation differences; findings are valuable though the technical contribution is primarily analytical.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three model families + four benchmarks + ablations, though scale is limited.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure, excellent visualizations, coherent narrative.
- Value: ⭐⭐⭐⭐ Important contribution to understanding dLLM internal mechanisms; the layer skipping method is practical though straightforward.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] AdaBlock-dLLM: Semantic-Aware Diffusion LLM Inference via Adaptive Block Size](adablock-dllm_semantic-aware_diffusion_llm_inference_via_adaptive_block_size.md)
- [\[ICLR 2026\] Breaking Scale Anchoring: Frequency Representation Learning for Accurate High-Resolution Inference from Low-Resolution Training](breaking_scale_anchoring_frequency_representation_learning_for_accurate_high-res.md)
- [\[ICLR 2026\] Beyond Scattered Acceptance: Fast and Coherent Inference for DLMs via Longest Stable Prefixes](beyond_scattered_acceptance_fast_and_coherent_inference_for_dlms_via_longest_sta.md)
- [\[ICCV 2025\] Learning Pixel-adaptive Multi-layer Perceptrons for Real-time Image Enhancement](../../ICCV2025/image_restoration/learning_pixel-adaptive_multi-layer_perceptrons_for_real-time_image_enhancement.md)
- [\[ICLR 2026\] Activation Steering for Masked Diffusion Language Models](activation_steering_for_masked_diffusion_language_models.md)

</div>

<!-- RELATED:END -->

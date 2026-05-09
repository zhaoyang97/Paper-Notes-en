---
title: >-
  [Paper Note] FOCUS: Internal MLLM Representations for Efficient Fine-Grained Visual Question Answering
description: >-
  [NeurIPS 2025][Multimodal VLM][Fine-grained VQA] This paper proposes FOCUS, a training-free visual cropping method that constructs object relevance maps via cosine similarity of value features in the MLLM's internal KV-cache, enabling efficient localization of question-relevant image regions. FOCUS achieves accuracy comparable to state-of-the-art methods on fine-grained VQA benchmarks while improving computational efficiency by 3–6.5×.
tags:
  - NeurIPS 2025
  - Multimodal VLM
  - Fine-grained VQA
  - Visual Cropping
  - KV-Cache
  - Object Localization
  - MLLM
date: 2026-05-08
content_hash: 7dd844e3ea6dc43f
---

# FOCUS: Internal MLLM Representations for Efficient Fine-Grained Visual Question Answering

**Conference**: NeurIPS 2025
**arXiv**: [2506.21710](https://arxiv.org/abs/2506.21710)
**Code**: [https://focus-mllm-vqa.github.io](https://focus-mllm-vqa.github.io) (project page)
**Area**: Multimodal VLM
**Keywords**: Fine-grained VQA, Visual Cropping, KV-Cache, Object Localization, MLLM

## TL;DR

This paper proposes FOCUS, a training-free visual cropping method that constructs object relevance maps via cosine similarity of value features in the MLLM's internal KV-cache, enabling efficient localization of question-relevant image regions. FOCUS achieves accuracy comparable to state-of-the-art methods on fine-grained VQA benchmarks while improving computational efficiency by 3–6.5×.

## Background & Motivation

**Background**: MLLMs demonstrate strong performance on VQA tasks, but struggle with small objects in high-resolution images. Global-view MLLMs (e.g., LLaVA-1.5, limited to 336×336) suffer from information loss due to downsampling; global-local MLLMs (e.g., LLaVA-OneVision) retain local crops but face difficulty identifying the small subset of visual tokens relevant to a given question among a large token pool.

**Limitations of Prior Work**: Existing visual cropping methods each have notable drawbacks — SEAL requires task-specific fine-tuning; DC2 and ZoomEye perform exhaustive hierarchical search with very low efficiency (ZoomEye requires 3 forward passes per candidate region); ViCrop relies on full Q-K attention weights, which are incompatible with FlashAttention.

**Key Challenge**: The core challenge is how to accurately localize small, question-relevant image regions without additional training, exhaustive search, or dependency on explicit attention matrices that are unavailable under efficient attention implementations.

**Key Insight**: The KV-cache computed during MLLM inference already encodes implicit semantic correspondences between visual and textual tokens. Target object tokens and their corresponding image tokens should exhibit high cosine similarity in the value feature space, from which spatial localization information can be extracted at zero additional computational cost.

**Core Idea**: Replace conventional attention weights with cosine similarity between value features in the KV-cache to construct object relevance maps, enabling training-free, efficient, and FlashAttention-compatible fine-grained object localization.

## Method

### Overall Architecture

FOCUS operates in four steps: (1) extract target object names from the VQA question using in-context learning (ICL); (2) compute an object relevance map from value features in the KV-cache; (3) propose and rank candidate ROIs based on the relevance map; (4) perform final VQA inference on the highest-confidence region. The entire pipeline requires no additional training or fine-tuning, relying solely on standard MLLM inference.

### Key Designs

1. **Target Object Extraction**:

    - Leverages the MLLM's in-context learning capability with few-shot prompting to extract the names of objects of interest from the VQA question.
    - Supports extraction of single or multiple target objects; a separate relevance map is constructed for each.

2. **V-V Pseudo-Attention and Object Relevance Maps**:

    - For each target token, cosine similarity with all visual tokens is computed at layer $l$ and reshaped into an $a \times a$ spatial map.
    - Cross-layer aggregation: attention rollout with residual connections is applied to aggregate information from layer $l$ to layer $L$.
    - Multi-token intersection: element-wise multiplication is applied across relevance maps of different target tokens, ensuring that only regions matching all tokens simultaneously are retained (e.g., "red car" retains only regions that are jointly red and car-like).
    - Design motivation: conventional Q-K attention weights are unavailable under FlashAttention, whereas value features are already present in the KV-cache required for inference, incurring zero additional overhead.
    - For global-local MLLMs, visual tokens from local crops are used to compute pseudo-attention, which empirically yields better fine-grained detail capture.

3. **Candidate Region Proposal and Ranking**:

    - Top-$k$ highest-scoring positions in the relevance map are selected as anchors (with a minimum spacing constraint).
    - Each anchor initializes an ROI at minimum size, which is expanded outward until reaching maximum size or until the average relevance falls below a threshold.
    - After non-maximum suppression (NMS) deduplication, the top-$n_\text{steps}$ ROIs are verified by querying the MLLM on whether the target object is present, and confidence scores are used for re-ranking.
    - Design motivation: relevance maps may contain spurious high-activation tokens, necessitating a secondary verification step to confirm target presence within the ROI.

4. **Final VQA Inference**:

    - Type-1 questions (single target): the highest-confidence ROI is used for VQA; for questions involving multiple objects, the best ROI for each target is merged.
    - Type-2 questions (multiple instances): all ROIs with confidence above a threshold are selected.
    - For global-local MLLMs, text-image-interleaved prompting provides the model with both a globally annotated image indicating target locations and the best ROI for each target.

### Loss & Training

No training is required. All operations are performed at inference time. Computational budget is controlled via $n_\text{steps}$ (range: 1–8). Layers 14–32 are used for LLaVA-1.5 and layers 14–28 for LLaVA-OneVision.

## Key Experimental Results

### Main Results (LLaVA-1.5-7B)

| Dataset | FOCUS Acc | ZoomEye Acc | Efficiency Gain |
|--------|-----------|-------------|---------|
| V*Bench | 72.77% | 77.48% | 3.43× |
| HRBench-4K | 51.75% | 49.75% | 4.39× |
| HRBench-8K | 45.00% | 49.00% | 4.72× |

### Main Results (LLaVA-OneVision-7B)

| Dataset | FOCUS Acc | ZoomEye Acc | Vanilla |
|--------|-----------|-------------|---------|
| V*Bench | 92.15% | 89.53% | 74.46% |
| HRBench-4K | 72.00% | 68.50% | 58.00% |
| HRBench-8K | 66.50% | 64.75% | 56.25% |

### MME-RealWorld-Lite (LLaVA-OV-7B)

| Method | Perception Acc | Reasoning Acc | Perception FP | Reasoning FP |
|------|---------|---------|---------|---------|
| Vanilla | 52.01% | 40.93% | - | - |
| ZoomEye | 56.29% | 43.20% | 41.60 | 45.95 |
| FOCUS | 54.15% | 44.53% | 7.71 | 8.21 |

FOCUS outperforms ZoomEye on reasoning tasks and is slightly weaker on perception, while being 5.47× more efficient.

### Generalization to Qwen-2.5-VL-7B

| Dataset | Vanilla | FOCUS |
|--------|---------|-------|
| V*Bench | 79.06% | 90.58% |
| HRBench-4K | 71.62% | 79.25% |
| HRBench-8K | 68.62% | 76.25% |

These results validate the generalizability of FOCUS across different MLLM architectures.

### Ablation Study

| Configuration | V*Bench Acc | V*Bench Recall | HRBench-4K Acc |
|------|------------|----------------|----------------|
| Full FOCUS | 72.77% | - | 51.75% |
| Random relevance map + ranking | 48.68% | 18.37% | 36.13% |
| Relevance map + no ranking | 51.30% | 38.48% | 41.13% |
| K-K pseudo-attention (w/o RoPE) | 69.10% | 63.47% | 45.63% |
| Layers 0–14 | 66.49% | 76.17% | 47.38% |
| Layers 0–32 | 71.20% | 75.56% | 49.38% |
| Layers 14–32 (default) | 72.77% | - | 51.75% |

### Key Findings

- Both the relevance map and the ROI ranking modules are indispensable: removing the relevance map reduces accuracy by ~24 pp, and removing ranking reduces it by ~21 pp.
- Even with a random relevance map, the ranking mechanism substantially outperforms random guessing (48.68% vs. 35.99%), indicating that the ranking module is independently robust.
- V-V features outperform RoPE-free K-K features: RoPE introduces positional rotations into key features that inflate cosine similarity between adjacent tokens, while removing RoPE compromises semantic integrity.
- Later-layer representations (14–32) outperform early layers (0–14) and all layers (0–32), consistent with Logit Lens findings that later layers encode more semantically discriminative information.
- Performance degradation on large-object datasets is modest: A-OKVQA drops by only 3.23 pp and GQA by 1.63 pp.
- The method demonstrates strong hyperparameter robustness: maximum variation of 4.71 pp for LLaVA-1.5 and only 2.62 pp for LLaVA-OV.

## Highlights & Insights

- **Novel use of KV-cache**: Value features already present in the KV-cache are repurposed for object localization at zero additional storage cost, with native compatibility with FlashAttention — a quintessential "free lunch" design.
- **V-V pseudo-attention**: Using value-value cosine similarity as a substitute for Q-K attention weights circumvents the unavailability of explicit attention matrices under efficient attention implementations. This also reveals that value features are more suitable than key features for semantic similarity measurement, as they are not subject to RoPE interference — an insight with broader implications for understanding attention mechanisms.
- **Multi-token intersection filtering**: Element-wise multiplication across relevance maps of multiple target tokens applies AND semantics to ensure that only regions satisfying all textual conditions simultaneously are retained, which is both conceptually simple and empirically effective.
- **Clear efficiency advantage**: ZoomEye requires 3 forward passes per candidate region and exhaustive hierarchical search; FOCUS requires only 1 forward pass to construct a global relevance map, making the search informed rather than exhaustive.

## Limitations & Future Work

- Constrained by the spatial resolution of MLLM internal representations: LLaVA-1.5 produces only 24×24 relevance maps, which may be insufficient for detecting extremely small objects in 8K images.
- Inherits the base MLLM's limitations in understanding spatial relations (e.g., "left/right of image"), a fundamental weakness that cannot be corrected without training.
- On large-object datasets (e.g., GQA with LLaVA-OV), performance drops by up to 10.99 pp, as cropping may discard global context necessary for large objects.
- Target object extraction relies on ICL and may be inaccurate in complex multi-object scenarios.

## Related Work & Insights

- **vs. ZoomEye**: The core difference lies in the search strategy — ZoomEye performs exhaustive hierarchical tree search with 3 forward passes per candidate, while FOCUS uses KV-cache-based relevance maps for informed search requiring only 1 forward pass for localization, yielding 3–6.5× efficiency gains.
- **vs. ViCrop**: ViCrop's rel-attn and attn-grad variants depend on full Q-K attention weights or gradients, which are incompatible with FlashAttention; FOCUS is fully compatible with modern efficient inference frameworks via V-V pseudo-attention.
- **vs. DC2**: DC2 queries the MLLM to generate captions for each candidate region to determine target presence, incurring substantial computational overhead; FOCUS extracts spatial information directly from internal representations without additional text generation.
- **vs. SEAL**: SEAL requires an auxiliary decoder and task-specific fine-tuning to predict heatmaps; FOCUS is entirely training-free and plug-and-play.

The finding that KV-cache implicitly encodes spatial information is transferable to a variety of tasks, including training-free open-vocabulary detection, image editing region localization, and spatiotemporal localization in video. V-V pseudo-attention can serve as a general-purpose substitute for attention visualization and interpretability analysis under FlashAttention. The AND-semantic multi-token intersection filtering paradigm is a clean and extensible approach for compositional attribute queries, applicable to multi-attribute retrieval and combinatorial reasoning.

## Rating

- Novelty: ⭐⭐⭐⭐ — Using KV-cache value features for object localization is a novel perspective, though the overall "localize-then-answer" paradigm remains conventional.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Four datasets, three model architectures, detailed ablation studies, and hyperparameter sensitivity analysis provide comprehensive coverage.
- Writing Quality: ⭐⭐⭐⭐⭐ — Motivation is clearly derived, method description is precise, figures and tables are well-designed, and comparisons with competing methods are fair.
- Value: ⭐⭐⭐⭐ — Addresses a practical efficiency bottleneck with an elegant, plug-and-play method of clear relevance to industrial deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] WearVQA: A Visual Question Answering Benchmark for Wearables in Egocentric Authentic Real-world scenarios](wearvqa_a_visual_question_answering_benchmark_for_wearables_in_egocentric_authen.md)
- [\[NeurIPS 2025\] Guiding Cross-Modal Representations with MLLM Priors via Preference Alignment](guiding_cross-modal_representations_with_mllm_priors_via_preference_alignment.md)
- [\[NeurIPS 2025\] AffordBot: 3D Fine-grained Embodied Reasoning via Multimodal Large Language Models](affordbot_3d_fine-grained_embodied_reasoning_via_multimodal_large_language_model.md)
- [\[ICCV 2025\] ReasonVQA: A Multi-hop Reasoning Benchmark with Structural Knowledge for Visual Question Answering](../../ICCV2025/multimodal_vlm/reasonvqa_a_multi-hop_reasoning_benchmark_with_structural_knowledge_for_visual_q.md)
- [\[NeurIPS 2025\] Are Vision Language Models Ready for Clinical Diagnosis? A 3D Medical Benchmark for Tumor-centric Visual Question Answering](are_vision_language_models_ready_for_clinical_diagnosis_a_3d_medical_benchmark_f.md)

</div>

<!-- RELATED:END -->

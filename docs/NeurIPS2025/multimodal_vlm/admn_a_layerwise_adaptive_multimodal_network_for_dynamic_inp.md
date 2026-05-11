---
title: >-
  [Paper Note] ADMN: A Layer-Wise Adaptive Multimodal Network for Dynamic Input Noise and Compute Resources
description: >-
  [NeurIPS 2025][Multimodal VLM][multimodal] This paper proposes ADMN (Adaptive Depth Multimodal Network), a two-stage training framework: (1) Multimodal LayerDrop fine-tuning to make the backbone robust to arbitrary layer…
tags:
  - "NeurIPS 2025"
  - "Multimodal VLM"
  - "multimodal"
  - "adaptive depth"
  - "LayerDrop"
  - "quality-of-information"
  - "dynamic compute budget"
  - "layer allocation"
  - "sensor corruption"
date: 2026-05-08
content_hash: 6215f0e9a5d48ffd
---

# ADMN: A Layer-Wise Adaptive Multimodal Network for Dynamic Input Noise and Compute Resources

**Conference**: NeurIPS 2025
**arXiv**: [2502.07862](https://arxiv.org/abs/2502.07862)
**Code**: [https://github.com/nesl/ADMN](https://github.com/nesl/ADMN)
**Area**: Multimodal VLM / Model Compression
**Keywords**: multimodal, adaptive depth, LayerDrop, quality-of-information, dynamic compute budget, layer allocation, sensor corruption

## TL;DR
This paper proposes ADMN (Adaptive Depth Multimodal Network), a two-stage training framework: (1) Multimodal LayerDrop fine-tuning to make the backbone robust to arbitrary layer configurations, and (2) a QoI-aware controller that dynamically allocates layer budgets across modalities. ADMN adaptively assigns layers based on per-modality quality-of-information (QoI) under strict compute constraints, matching full-model accuracy while reducing FLOPs by 75% and latency by 60%.

## Background & Motivation
Deploying multimodal systems in dynamic environments presents two concurrent challenges:
1. **Varying compute resources**: Multi-tenancy, device heterogeneity, and thermal throttling cause available compute budgets to fluctuate over time and require strict adherence (no budget overrun).
2. **Varying input quality**: Sensor corruption, weather changes, and similar factors cause per-modality signal quality to fluctuate dynamically—heavily corrupted modalities should not consume the same compute as clean ones.

**Limitations of Prior Work**: Static multimodal networks cannot adapt to changing compute budgets; existing dynamic networks (Early Exit, DynMM, etc.) optimize average-case efficiency but cannot handle strict budget constraints; and nearly all methods ignore the effect of per-modality QoI.

## Core Problem
How to build a multimodal network that **simultaneously adapts to dynamic compute constraints and dynamic input quality**—controlling the total number of layers according to the compute budget while allocating layers on demand based on per-modality QoI.

## Method

### Overall Architecture
Two-stage training: Stage 1 builds a layer-adaptive multimodal backbone; Stage 2 trains a QoI-aware controller to allocate layer budgets.

### Key Designs
1. **Multimodal LayerDrop (Stage 1)**:
   - LayerDrop (rate 0.2) is introduced during MAE pre-training to make the ViT backbone robust to missing layers.
   - LayerDrop is retained during multimodal task fine-tuning so that fusion and output layers adapt to varied backbone layer configurations.
   - **Full-backbone Dropout**: With 10% probability, all layers of a modality's backbone are dropped—simulating the extreme case where that modality is entirely unavailable.
   - Result: A single set of weights operates under any layer budget.

2. **QoI-Aware Controller (Stage 2)**:
   - Lightweight architecture: downsampled inputs → modality-specific convolutions → Transformer fusion → MLP output layer allocation logits.
   - **Corruption-aware supervision (ADMN)**: An auxiliary corruption prediction loss $\mathcal{L}_{corr}$ explicitly teaches the controller to attend to per-modality QoI.
   - **Autoencoder initialization (ADMN_AE)**: When corruption labels are unavailable, the controller's perception layers are pre-trained with an AE—the reconstruction objective forces the latent space to cluster by QoI (verified via t-SNE visualization).
   - Ablations confirm that task loss alone is insufficient for learning QoI-aware allocation.

3. **Differentiable Layer Selection**:
   - Gumbel-Softmax sampling (temperature 1) + Top-$L$ discretization + straight-through estimator.
   - Enables differentiable selection of $C$ backbone layers under a total budget constraint of $L$ layers.

### Loss & Training
Stage 1: Task loss + LayerDrop (0.2) + Full-backbone dropout (10%). Stage 2: $\mathcal{L}_{total} = \mathcal{L}_{model} + \mathcal{L}_{corr}$ (or AE initialization + $\mathcal{L}_{model}$).

## Key Experimental Results

| Dataset | Task | Method | 6 layers | 8 layers | 12 layers | 16 layers | Upper Bound (24 layers) |
|--------|------|------|-----|-----|------|------|----------|
| GDTM (Gaussian noise) | Localization (cm↓) | Naive Alloc | 112.5 | 97.6 | 46.9 | 31.0 | 29.6 |
| | | **ADMN** | **51.4** | **39.0** | **33.1** | **30.3** | |
| | | **ADMN_AE** | **53.6** | **38.4** | **33.5** | **29.4** | |
| GDTM (low light) | Localization (cm↓) | Naive Alloc | 90.3 | 67.3 | 27.1 | 17.7 | 18.8 |
| | | **ADMN** | **49.5** | **23.9** | **18.0** | **17.3** | |
| MM-Fi | Classification (↑) | Naive Alloc | 5.56% | 12.96% | 29.01% | 42.90% | 44.44% |
| | | **ADMN** | **35.03%** | **39.25%** | **41.92%** | **43.31%** | |
| AVE | Classification (↑) | Naive Alloc | 36.16% | 46.89% | 65.71% | 67.95% | 71.19% |
| | | **ADMN** | **57.07%** | **62.95%** | **67.48%** | **66.60%** | |

GDTM (Blur, 8 layers): ADMN localization error ~11 cm, approaching the upper bound (9.4 cm), while reducing FLOPs by 75% and latency by 60%.

### Ablation Study
- **LayerDrop stages**: Applying LayerDrop in both MAE pre-training and fine-tuning yields the best results (Fig. 6).
- **Necessity of QoI supervision**: A task-loss-only controller fails to learn QoI-aware allocation (Table 5: task loss 46.2% vs. ADMN 57.07%).
- **AE latent space**: t-SNE confirms that the AE automatically clusters different corruption levels (Fig. 5).
- **Full-backbone dropout**: Essential; without it, the model cannot handle the extreme case of a fully missing modality.
- **Three-modality generalization**: RGB + Depth + mmWave experiments validate generality (Fig. 7).
- **Unequal-compute modalities**: In a scenario where the visual backbone has 3× the FLOPs of the audio backbone, ADMN allocates layers correctly (Table 4).
- **Stability across 6 seeds**: Standard deviation < 5%; larger budgets yield more stable results.

## Highlights & Insights
- **Dual adaptivity** is the core contribution—simultaneously adapting to varying compute budgets and varying input quality, a previously unaddressed combination.
- Extending LayerDrop from single-modality text Transformers to multimodal ViTs is a non-trivial engineering contribution, requiring special handling such as full-backbone dropout.
- ADMN_AE learns QoI-aware allocation without any QoI annotations, offering high practical value.
- The controller accounts for only ~1% of total FLOPs, imposing negligible overhead.
- The ablation study spans 6 seeds × 3 datasets × 3–4 layer budgets × 3–4 corruption types, constituting a large-scale empirical evaluation.

## Limitations & Future Work
- A separate controller must be trained for each layer budget (a universal controller shows preliminary feasibility but requires further development).
- Batch inference is incompatible, as different samples may require different layer configurations.
- The approach could be combined with Early Exit for further efficiency gains.
- Validation is limited to embedding-level fusion architectures; data-level and late-fusion settings remain unexplored.

## Related Work & Insights
- **vs. DynMM / AdaMML (model selection)**: These methods select from predefined expert models; ADMN performs layer-level allocation within a single model, offering greater flexibility.
- **vs. PrefixKV**: PrefixKV allocates KV cache budgets across layers; ADMN allocates layer budgets across modalities—both exemplify "adaptive allocation across dimensions."
- **vs. ASF**: ASF fuses sensors in a unified canonical space and estimates availability; ADMN achieves finer-grained resource control through layer allocation.

**Transfer potential**: The QoI-aware layer allocation idea is transferable to VLMs—e.g., allocating fewer layers to the visual encoder for low-quality images and more to the LLM for high-quality text. The AE-based QoI clustering could also serve as an automatic sensor degradation detector without explicit annotations.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — Dual adaptivity (compute + QoI) constitutes an entirely new problem formulation and solution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 3 datasets, 3–4 corruption types, 6 seeds, extensive ablations, and qualitative analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear problem definition, detailed method description, and thorough ablation study.
- Value: ⭐⭐⭐⭐⭐ — Addresses the practical dual-constraint problem in multimodal deployment; ADMN_AE requires no QoI annotations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Vision Function Layer in Multimodal LLMs](vision_function_layer_in_multimodal_llms.md)
- [\[NeurIPS 2025\] Learning to Steer: Input-dependent Steering for Multimodal LLMs](learning_to_steer_input-dependent_steering_for_multimodal_llms.md)
- [\[NeurIPS 2025\] SITCOM: Scaling Inference-Time COMpute for VLAs](sitcom_scaling_inference-time_compute_for_vlas.md)
- [\[NeurIPS 2025\] DynamicVL: Benchmarking MLLMs for Dynamic City Understanding](dynamicvl_benchmarking_multimodal_large_language_models_for_dynamic_city_underst.md)
- [\[ICCV 2025\] Dynamic-VLM: Simple Dynamic Visual Token Compression for VideoLLM](../../ICCV2025/multimodal_vlm/dynamic-vlm_simple_dynamic_visual_token_compression_for_videollm.md)

</div>

<!-- RELATED:END -->

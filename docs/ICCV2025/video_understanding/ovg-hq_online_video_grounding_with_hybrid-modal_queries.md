---
title: >-
  [Paper Note] OVG-HQ: Online Video Grounding with Hybrid-modal Queries
description: >-
  [ICCV 2025][Video Understanding][Online video grounding] This paper proposes OVG-HQ, a new online video grounding task supporting hybrid-modal queries (text, image, and video clip)…
tags:
  - "ICCV 2025"
  - "Video Understanding"
  - "Online video grounding"
  - "hybrid-modal queries"
  - "parametric memory block"
  - "cross-modal distillation"
  - "streaming video"
date: 2026-05-08
content_hash: 2db7670fc9ce5df7
---

# OVG-HQ: Online Video Grounding with Hybrid-modal Queries

**Conference**: ICCV 2025
**arXiv**: [2508.11903](https://arxiv.org/abs/2508.11903)
**Code**: [GitHub](https://github.com/maojiaqi2324/OVG-HQ)
**Area**: Video Understanding
**Keywords**: Online video grounding, hybrid-modal queries, parametric memory block, cross-modal distillation, streaming video

## TL;DR

This paper proposes OVG-HQ, a new online video grounding task supporting hybrid-modal queries (text, image, and video clip), and introduces a Parametric Memory Block (PMB) to retain historical context alongside a hybrid distillation strategy to mitigate modality imbalance, enabling real-time moment localization in streaming video.

## Background & Motivation

Traditional Video Grounding methods suffer from two critical limitations:

**Offline settings are incompatible with streaming scenarios**: Existing methods require access to the complete video for prediction, making them unsuitable for real-time detection in settings such as surveillance. For instance, security systems require immediate localization of queries like "a group of people gathering near the front gate" from live footage, rather than waiting to process a complete offline recording.

**Single-modality text queries limit multimodal applicability**: Current methods support only natural language queries, whereas practical applications may require images or video clips as queries. For example, a security operator may prefer to upload a past surveillance clip to retrieve similar behaviors, rather than composing a detailed textual description.

These limitations motivate a new task: **Online Video Grounding with Hybrid-modal Queries (OVG-HQ)**—online moment localization in streaming video using text, image, video clip, or their combinations as queries.

The task introduces two new challenges:
- **Limited context in the online setting**: The model can only access frames within a sliding window and must efficiently model and leverage historical information.
- **Modality imbalance during training**: Different modalities contribute unevenly to the task; dominant modalities suppress the optimization of weaker ones, making it difficult for a single unified model to handle all modalities effectively.

## Method

### Overall Architecture

OVG-HQ-Unify is a unified and flexible model that supports hybrid-modal query inputs for online moment localization. It consists of three core components:

1. **Memory-guided multimodal fusion module**: Extracts query-aware video features and enhances long-term dependencies via PMB.
2. **Memory-guided moment prediction module**: Generates predictions based on predefined anchors and refines results through PMB.
3. **Hybrid distillation strategy**: Uses teacher models to guide learning for non-dominant modalities.

### Key Designs

1. **Parametric Memory Block (PMB)**

   Conventional memory bank approaches integrate historical information via self-attention, incurring additional storage overhead with computational costs that grow with history length. LSTMs use fixed-size hidden states but have limited expressive capacity.

   PMB is built upon Test-Time Training (TTT) layers, compressing historical information into **network parameters** and leveraging the greater capacity of neural networks for stronger expressiveness. It operates in two steps:

   **Step 1: Memorize the current input.** The core component $f_{\text{PML}}(\cdot; W^m)$ compresses the current input $r_t$ into parameters $W^m$ via a self-supervised reconstruction loss:

    $\mathcal{L}_{\text{PML}}(r_t; W^m) = \|f_{\text{PML}}(W_K r_t; W^m) - W_V r_t\|^2$

   Parameters $W^m$ are then updated via gradient descent, with an adaptive learning rate $\eta_{\text{PML}} = \sigma(W_{lr} \cdot r_t)$.

   **Step 2: Generate memory-augmented output.** The current input is projected and passed through the updated $f_{\text{PML}}$; after layer normalization and projection, the memory-augmented feature is obtained:

    $\hat{r}_t = W_O \cdot \text{LN}(f_{\text{PML}}(W_Q r_t; W^m))$

   **Key advantage**: Parameters are dynamically updated at inference time, enabling the model to "memorize" historical information and adapt to new scenes—a sharp contrast to conventional fixed-parameter inference.

2. **Memory-guided Multimodal Fusion**

    - **Query feature extraction**: CLIP's text/image encoders are used to extract features; video clip queries are sampled at fixed intervals.
    - **Video feature extraction**: Streaming video is processed via a sliding window of step size $M$ seconds; overlapping frame features are computed once and cached for reuse.
    - **Cross-modal fusion**: A Transformer decoder performs cross-attention, with video features as queries and modality-specific features as keys/values. Modality-specific tokens $\mathbf{m}_*$ are prepended to distinguish different input modalities.
    - **PMB augmentation**: The fused query-aware features $\mathbf{F}_{qv}$ are passed through the PMB module, where the self-attention layer is replaced by $f_{\text{PML}}$, yielding memory-guided features $\hat{\mathbf{F}}_{qv}$ enriched with historical context.

3. **Memory-guided Moment Prediction and Refinement**

   Based on predefined anchors $A_n = (t - L_n, t)$ where $L_n = L_q / 2^{n-1}$, a Transformer decoder processes anchor queries and fused features to predict foreground/background classification scores and boundary regression offsets.

   **Prediction Refinement Module (PRM)**: Since past predictions cannot be revised in the online setting, PRM uses PMB to compress current anchor features and predictions into parameters, allowing refined predictions to incorporate historical prediction information. Only anchors with foreground scores exceeding threshold $\theta$ are retained.

### Loss & Training

**Hybrid distillation strategy**: Directly training a unified model on mixed-modality data leads to poor performance on non-text queries due to modality imbalance. To address this:

- Three query types (text, visual, visual+text) are used in alternating training batches.
- Expert teacher models trained on text+clip queries guide the unified student model via distillation:

$$\mathcal{L}_d = \frac{1}{N}\sum_{i=1}^{N}\left(\mathcal{L}_{\text{KL}}(\mathbf{F}_{a,i}^s, \mathbf{F}_{a,i}^t) + \mathcal{L}_2(\mathbf{r}_i^s, \mathbf{r}_i^t) + \mathcal{L}_2(\mathbf{c}_i^s, \mathbf{c}_i^t)\right)$$

Total loss: $\mathcal{L} = \mathcal{L}_d + \lambda \mathcal{L}_{cls} + \mathcal{L}_{reg}$, where $\lambda = 10$. The classification head uses Focal Loss and the regression head uses L1 Loss.

## Key Experimental Results

### Main Results

The authors construct the QVHighlights-Unify dataset by extending QVHighlights with image and video clip queries.

| Method | Setting | oR¹₀.₅ | omAP₀.₅ |
|------|------|---------|----------|
| TaskWeave | Offline→Online | 7.02 | 5.96 |
| TR-DETR | Offline→Online | 7.37 | 6.06 |
| R2-Tuning | Offline→Online | 9.30 | 8.17 |
| TwinNet | Online VG | 20.78 | 19.73 |
| **OVG-HQ-Unify** | **Online VG** | **23.26** | **23.09** |

On ANet-Captions, TACoS, and MAD datasets (text query):

| Dataset | Metric | TwinNet | Ours | Gain |
|--------|------|---------|------|------|
| ANet-Captions | R¹₀.₇ | 12.56 | 14.36 | +1.80 |
| TACoS | R¹₀.₇ | 19.07 | 21.17 | +2.10 |
| MAD | R⁵₀.₅ | 2.00 | 3.27 | +1.27 |

### Ablation Study

| Configuration | oR¹₀.₅ (Text) | omAP₀.₅ (Text) | Note |
|------|---------------|-----------------|------|
| Ours-ATT (replaced with self-attention) | 13.93 | 16.41 | PMB significantly outperforms self-attention |
| Ours-LSTM | 22.37 | 21.66 | PMB also outperforms LSTM |
| **Ours (PMB)** | **23.37** | **22.51** | — |
| w/o Refine | 17.64 | 17.43 | PRM yields substantial gains |
| Pred only | 18.99 | 21.07 | Prediction alone is insufficient |
| Pred+AF (full) | 23.37 | 22.51 | Anchor features + predictions jointly optimal |

Runtime analysis: Overall FPS = 45.95; PMB latency is only 2.20 ms with dynamic update overhead of 0.30 ms, satisfying real-time requirements.

### Key Findings

- Hybrid distillation improves image-only query oR¹₀.₅ from 11.43% to 20.41% (+8.98%), effectively mitigating modality imbalance.
- Video clip queries outperform image queries (20.33% vs. 16.14%), as video clips are better suited for describing dynamic content.
- Multimodal queries consistently outperform single-modality queries.

## Highlights & Insights

- **Practical significance of the task formulation**: OVG-HQ combines online inference with hybrid-modal queries, making it significantly closer to real-world applications than the conventional offline + text-only setting.
- **Parameters as memory**: Using TTT layer parameters as dynamic memory—continuously updated during inference—is more efficient and expressive than traditional memory banks.
- **Online evaluation metric design**: The proposed oR and omAP metrics incorporate a timeliness decay factor $\beta$ that penalizes delayed predictions, providing a more faithful evaluation of online scenarios.

## Limitations & Future Work

- The multimodal dataset is currently constructed only from QVHighlights, covering relatively simple scenarios; validation on more complex settings (e.g., surveillance) is needed.
- The sliding window size is fixed; adaptive window strategies warrant future exploration.
- The hybrid distillation strategy requires pre-training teacher models, increasing overall training cost.

## Related Work & Insights

- The parametric memory paradigm from TTT is generalizable to other online video understanding tasks.
- The cross-modal distillation strategy offers broad reference value for training multimodal unified models.
- The online evaluation metric design methodology is applicable to other streaming tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ Novel task formulation and elegant PMB design
- Experimental Thoroughness: ⭐⭐⭐⭐ Four datasets, multiple query types, comprehensive ablations
- Writing Quality: ⭐⭐⭐⭐ Clear structure with well-defined problem formulation
- Value: ⭐⭐⭐⭐ Advances online multimodal video understanding

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Hierarchical Event Memory for Accurate and Low-latency Online Video Temporal Grounding](hierarchical_event_memory_for_accurate_and_low-latency_online_video_temporal_gro.md)
- [\[ICCV 2025\] Vamba: Understanding Hour-Long Videos with Hybrid Mamba-Transformers](vamba_understanding_hour-long_videos_with_hybrid_mamba-transformers.md)
- [\[ICCV 2025\] Moment Quantization for Video Temporal Grounding](moment_quantization_for_video_temporal_grounding.md)
- [\[NeurIPS 2025\] PixFoundation 2.0: Do Video Multi-Modal LLMs Use Motion in Visual Grounding?](../../NeurIPS2025/video_understanding/pixfoundation_20_do_video_multi-modal_llms_use_motion_in_visual_grounding.md)
- [\[ICCV 2025\] Online Dense Point Tracking with Streaming Memory](online_dense_point_tracking_with_streaming_memory.md)

</div>

<!-- RELATED:END -->

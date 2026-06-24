---
title: >-
  [Paper Note] MambaVLT: Time-Evolving Multimodal State Space Model for Vision-Language Tracking
description: >-
  [CVPR 2025][Video Understanding][Vision-Language Tracking] MambaVLT is the first Mamba-based vision-language tracker. Leveraging the temporal evolution of state spaces, it maintains long-term target memory and adaptively updates multimodal reference features, achieving state-of-the-art (SOTA) performance on multiple vision-language tracking benchmarks.
tags:
  - "CVPR 2025"
  - "Video Understanding"
  - "Vision-Language Tracking"
  - "State Space Models"
  - "Mamba"
  - "Multimodal Fusion"
  - "Temporal Modeling"
date: 2026-05-08
content_hash: 796d9c1731c88a7e
---

# MambaVLT: Time-Evolving Multimodal State Space Model for Vision-Language Tracking

**Conference**: CVPR 2025  
**arXiv**: [2411.15459](https://arxiv.org/abs/2411.15459)  
**Code**: None  
**Area**: Video Understanding  
**Keywords**: Vision-Language Tracking, State Space Models, Mamba, Multimodal Fusion, Temporal Modeling

## TL;DR

MambaVLT is the first Mamba-based vision-language tracker. Leveraging the temporal evolution of state spaces, it maintains long-term target memory and adaptively updates multimodal reference features, achieving state-of-the-art (SOTA) performance on multiple vision-language tracking benchmarks.

## Background & Motivation

Vision-Language Tracking (VLT) aims to continuously track a target in a video based on multimodal references (initial bounding boxes, natural language descriptions, or a combination of both). Existing Transformer-based methods face two key challenges:

1. **Underutilization of Temporal Information**: The target's appearance and motion patterns continuously change during the video. Existing methods primarily extract context in a discrete manner—generating a context prompt based on the predicted bounding box and then decoding it. This discrete update mechanism lacks explicit cross-frame correlation, relies heavily on tracking precision, and is highly prone to error accumulation.

2. **Difficulty in Reference Feature Updating**: Most methods only update visual references, lacking an effective mechanism to jointly update both language and visual information. When target appearance changes drastically, fixed reference features inevitably become outdated.

**Mechanism**: The autoregressive state space evolution process of Mamba inherently possesses sequence memory capabilities, where the final state space implicitly encapsulates global information. Leveraging this characteristic, a continuously evolving state space memory can be designed to retain long-term target information and adaptively update reference features accordingly.

## Method

### Overall Architecture

MambaVLT supports three reference configurations: bounding box only, natural language only, and their combination. The architecture includes: (1) decoupled visual (Vmamba-tiny) and language (first 4 layers of Mamba-130m) encoders to extract features; (2) a Time-Evolving Multimodal Fusion (TEMF) module for cross-frame feature modeling and reference updating; (3) a modality selection module to dynamically weight different modality references; and (4) a localization head to output target coordinates. Notably, a template video clip (multiple frames instead of a single-frame template) is employed to explicitly capture target appearance changes.

### Key Designs

1. **Hybrid Multimodal State Space (HMSS) Block**: The core innovative module, embracing two crucial mechanisms:
    - **Temporal State Space Evolution**: A multi-level state space memory $SS = \{\{\mathbf{H}_{t-1}^{fin_i, \alpha}, \mathbf{H}_{t-1}^{fin_i, \beta}\}\}$ is constructed to store the final state space of each TEMF module. Since the final state during Mamba's autoregressive sequence processing implicitly encapsulates global information, the memory naturally evolves and accumulates long-term target features as the video is processed frame-by-frame. The initial state of each HMSS block is a weighted fusion of a learnable state and the historical memory: $\mathbf{H}_t^{ini} = a\mathbf{H}^l + (1-a)\mathbf{H}_{t-1}^{fin}$
    - **Modality-Guided Bidirectional Scanning**: Two scanning sequences are designed—text-first $\alpha$ (Language $\rightarrow$ Template $\rightarrow$ Search Region) and template-first $\beta$ (Template $\rightarrow$ Language $\rightarrow$ Search Region), where the search region is always placed at the end of the sequence to aggregate reference information. The two directions share $\bar{B}, C, D$ parameters to reduce redundancy and use different $\bar{A}^\alpha, \bar{A}^\beta$ as state update gates. The bidirectional outputs are then averaged.

2. **Selective Local Enhancement (SLE) Block**: After the HMSS performs global cross-frame modeling, the SLE enhances intra-modality dependencies and inter-modality relationships for the current frame. The core idea is to extract a global selective map $A_l$ from the HMSS output via convolution, which serves as a prior for linear attention scanning. This allows SLE to maintain a global receptive field while keeping linear complexity. The formulation is $h_t = A_l + B_l G$, $G' = \gamma(h_l) + D_l G$, where $\gamma$ represents sliding-window linear attention.

3. **Modality Selection Module**: This module dynamically evaluates the reliability of both visual and language references in the current frame. It first extracts invariant language information through language-template feature similarity, then aggregates invariant objective clues for language and vision ($P_l, P_z$) respectively using a query decoder, and finally refines search region features via a Mamba selective block that weights and fuses the two.

### Loss & Training

The total training objective consists of five loss terms:
$$\mathcal{L} = \lambda_{bbox}\mathcal{L}_{bbox} + \lambda_{tgt}\mathcal{L}_{tgt} + \lambda_{cls}\mathcal{L}_{cls} + \lambda_{c_w}\mathcal{L}_{c_w} + \lambda_{c_o}\mathcal{L}_{c_o}$$

- $\mathcal{L}_{bbox}$: Bounding box regression ($L_1$ + GIoU)
- $\mathcal{L}_{tgt}$: Target score map (Binary Cross-Entropy)
- $\mathcal{L}_{cls}$: Center score map
- $\mathcal{L}_{c_w}$: Intra-video contrastive loss (positive sample = target-center token, negative sample = most similar token in search region background)
- $\mathcal{L}_{c_o}$: Inter-video contrastive loss (negative sample = target-center token of other videos)

Training data: OTB99, LaSOT, TNL2K, MGIT, RefCOCOg, GOT-10k. Adam optimizer, lr=0.0005, 300 epochs.

## Key Experimental Results

### Main Results

| Dataset | Modality | Metrics (AUC/Prec) | Ours | Prev. SOTA (UVLTrack-B) | Gain |
|--------|----------|-------|------|----------|------|
| TNL2K | BBOX | AUC | 63.3 | 62.7 | +0.6 |
| TNL2K | NL | AUC/Prec | 58.4/58.9 | 55.7/57.2 | +2.7/+1.7 |
| TNL2K | NL&BBOX | AUC/Prec | 66.5/69.9 | 63.1/66.7 | +3.4/+3.2 |
| OTB99 | NL&BBOX | AUC/Prec | 72.2/94.4 | 69.3/89.9 | +2.9/+4.5 |
| MGIT | NL&BBOX | Prec | 58.9 | - (JointNLT: 44.5) | +14.4 |

The performance gain is most evident under the joint NL&BBOX modality setting, demonstrating the effectiveness of multimodal fusion.

### Ablation Study

| Configuration | TNL2K AUC (BBOX/NL/NL&BBOX) | Description |
|------|---------|------|
| Baseline | 60.9 / 55.3 / 62.6 | Without temporal evolution and modality selection |
| +THSS | 62.1 / 56.8 / 64.5 | Temporal state space adds +1.2/+1.5/+1.9 |
| +MgB | 62.5 / 57.3 / 65.3 | Modality-guided bidirectional scanning yields further improvements |
| +MS | 63.0 / 57.8 / 65.8 | Modality selection for dynamic weighting |
| +SLE | 63.3 / 58.4 / 66.5 | Local enhancement for final refining |

### Key Findings

- **State space memory possesses strong target preservation capabilities**: In semi-reference-free (SRF) tracking experiments—where reference information is provided only in the first frame, and subsequent frames rely entirely on state space memory—MambaVLT still outperforms the normal tracking setup of UVLTrack, proving that the state space can efficiently extract and preserve target traits.
- **Highest improvement in NL&BBOX**: The temporal state space boosts the AUC by 1.9% in the joint-modality task (vs. 1.2% in BBOX), showing that multimodal scenarios are more reliant on temporal cross-frame modeling.
- **Modality selection visualization**: After passing through the modality selection module, the similarity map between the search region and reference tokens becomes more focused on the target area, effectively suppressing distractors.

## Highlights & Insights

- **SSM temporal evolution is not just a sequence-modeling tool but a native target memory mechanism**: Utilizing the final state space of Mamba as a cross-frame memory is an elegant design that avoids extra network components.
- **Modality-Guided Bidirectional Scanning** cleverly exploits Mamba's sensitivity to scanning order. Efficient bidirectional modeling is achieved via shared parameters coupled with distinct state transition gates.
- The **SRF experimental paradigm** provides a fresh perspective to evaluate the temporal memory capacities of trackers.

## Limitations & Future Work

- Performance on the LaSOT dataset is inferior to UVLTrack, potentially due to attention decay of Mamba over exceptionally long sequences.
- The visual encoder employs Vmamba-tiny, limiting model capacity; utilizing a larger visual backbone might yield further performance gains.
- The state space memory trade-off parameter $a$ is fixed; adaptive adjustment could be more optimal.
- The language encoder uses only 4 layers of Mamba-130m, which might limit the comprehension of complex language descriptions.

## Related Work & Insights

- Transformer-based approaches like JointNLT/QueryNLT implement temporal updates via discrete context prompts; the continuous state space approach introduced here presents a more natural alternative.
- While VideoMamba introduces Mamba for sequence modeling in video classification, this work further explores the unique value of state spaces in tracking tasks (memory + updating).
- The concept of modality selection can be extended to other multimodal tracking or detection tasks.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The first Mamba-based VLT. The concept of utilizing state space evolution for target memory is highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated across 4 datasets and 3 modality configurations, though comparisons with more recent Transformer-based trackers are lacking.
- Writing Quality: ⭐⭐⭐⭐ The methodology is clearly defined, the SRF experiment layout is well-reasoned, and the figures are highly intuitive.
- Value: ⭐⭐⭐⭐ Paves a new direction for the application of SSMs in the field of visual tracking.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] LLAVIDAL: A Large Language Vision Model for Daily Activities of Living](llavidal_a_large_language_vision_model_for_daily_activities_of_living.md)
- [\[CVPR 2025\] GG-SSMs: Graph-Generating State Space Models](gg-ssms_graph-generating_state_space_models.md)
- [\[NeurIPS 2025\] PASS: Path-Selective State Space Model for Event-Based Recognition](../../NeurIPS2025/video_understanding/pass_path-selective_state_space_model_for_event-based_recognition.md)
- [\[ECCV 2024\] VideoMamba: State Space Model for Efficient Video Understanding](../../ECCV2024/video_understanding/videomamba_state_space_model_for_efficient_video_understanding.md)
- [\[ECCV 2024\] VideoMamba: Spatio-Temporal Selective State Space Model](../../ECCV2024/video_understanding/videomamba_spatio-temporal_selective_state_space_model.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Tracking Meets LoRA: Faster Training, Larger Model, Stronger Performance
description: >-
  [ECCV 2024][Video Understanding][Visual Object Tracking] LoRAT introduces LoRA to visual object tracking for the first time. Through two LoRA-friendly designs—decoupled position encoding (shared spatial components + independent type embeddings) and a pure MLP detection head—it enables training a tracker with a ViT-g backbone using laboratory-level resources. It achieves a SUC of 0.762 on LaSOT (new SOTA), while the lightest variant, LoRAT-B-224, runs at 209 FPS.
tags:
  - "ECCV 2024"
  - "Video Understanding"
  - "Visual Object Tracking"
  - "LoRA"
  - "Parameter-Efficient Fine-Tuning"
  - "Vision Transformer"
  - "Position Encoding"
date: 2026-05-08
content_hash: 3124c87e9f58e71f
---

# Tracking Meets LoRA: Faster Training, Larger Model, Stronger Performance

**Conference**: ECCV 2024  
**arXiv**: [2403.05231](https://arxiv.org/abs/2403.05231)  
**Code**: [https://github.com/LitingLin/LoRAT](https://github.com/LitingLin/LoRAT)  
**Area**: Video Understanding  
**Keywords**: Visual Object Tracking, LoRA, Parameter-Efficient Fine-Tuning, Vision Transformer, Position Encoding

## TL;DR

LoRAT introduces LoRA to visual object tracking for the first time. Through two LoRA-friendly designs—decoupled position encoding (shared spatial components + independent type embeddings) and a pure MLP detection head—it enables training a tracker with a ViT-g backbone using laboratory-level resources. It achieves a SUC of 0.762 on LaSOT (new SOTA), while the lightest variant, LoRAT-B-224, runs at 209 FPS.

## Background & Motivation

Visual object tracking has recently made significant progress owing to Transformer architectures, especially one-stream frameworks like OSTrack. However, the training resource requirements for Transformer trackers are escalating. The current largest tracking model, SeqTrack-L384, utilizes a ViT-L backbone and requires multiple high-end GPUs and extremely long training times. Larger pre-trained ViT models (e.g., ViT-g with 1.1B parameters) theoretically yield stronger performance, but the cost of full fine-tuning deters most researchers.

**Key Challenge**: Parameter-efficient fine-tuning (PEFT) methods in NLP (such as LoRA) have demonstrated efficient fine-tuning while freezing most parameters. However, direct transfer to visual tracking faces two unique challenges:

**Incompatible Position Encodings**: Existing trackers use independent position encodings for the template (small image) and the search area (large image), which disrupts the original structure of the pre-trained ViT and leads to sub-optimal results under PEFT methods like LoRA.

**Inductive Bias of Convolutional Heads**: The convolutional prediction heads of OSTrack struggle to converge under LoRA fine-tuning. The local assumption of convolutions on data structures hinders the parameter-efficient adaptation of LoRA.

**Core Idea**: Design a LoRA-friendly tracker architecture. Preserving the structural integrity of pre-trained models is key to the success of PEFT.

## Method

### Overall Architecture

LoRAT is built upon the one-stream tracking framework (OSTrack):
1. Template and search area → patch embedding → added with shared position encodings + token type embeddings.
2. Concatenated and fed into the Transformer encoder (original weights frozen, LoRA added to all linear layers).
3. Search area features → MLP-only head → classification scores + anchor-free bounding box regression.

Frozen during training: all original weights of the ViT backbone. Trainable: LoRA matrices (occupying a tiny fraction of the total parameters), token type embeddings, and the MLP prediction head.

### Key Designs

1. **Decoupled Input Embedding**:

    - Function: Decouples spatial position information from token source identification, preserving the integrity of pre-trained position encodings.
    - **Token type embedding**: Inspired by BERT's segment embeddings, independent type embedding vectors are assigned to three classes of tokens: template foreground $\mathbf{E}_{type}^{T_o}$, template background $\mathbf{E}_{type}^{T_b}$, and search area $\mathbf{E}_{type}^{S}$.
    - $\mathbf{E}_T^{(i,j)} = \mathbf{E}_{pos}^{(i,j)} + \mathbf{E}_{type}^{T_o/T_b}$ (Template, with type embedding selected based on whether it is the target foreground)
    - $\mathbf{E}_S^{(i,j)} = \mathbf{E}_{pos}^{(i,j)} + \mathbf{E}_{type}^{S}$ (Search area)
    - **Shared Position Encoding Adaptation**: Two candidate strategies: interpolation-like (interpolating 2D position encodings to the template resolution) and cropping-like (cropping a sub-matrix of template size from the top-left of the search area's position encodings). Experiments show that the cropping-like strategy is superior.
    - **Foreground Indicator Embedding**: Further distinguishes between target foreground and background tokens in the template, helping the model localize the tracking target within the template.
    - Design Motivation: OSTrack's use of independent position encodings is equivalent to learning two sets of unrelated spatial information from scratch, which fails to effectively inherit pre-trained spatial knowledge under the PEFT setting where LoRA freezes the pre-trained parameters.

2. **MLP-only Head**:

    - Function: Replaces the convolutional prediction head of OSTrack with a 3-layer MLP for classification and regression.
    - Split into two branches: classification branch (3-layer MLP → classification scores per token) and regression branch (3-layer MLP → center-based anchor-free bounding boxes).
    - Design Motivation: Convolutional networks have a strong local inductive bias on data structures, which hinders convergence under the LoRA setup where only a small number of parameters are fine-tuned. MLPs have no such limitations and are better aligned with the global low-rank updates of LoRA.

3. **LoRA Configuration**:

    - Applied positions: All linear layers in the ViT backbone (Q/K/V/O projections in MSA + two projection layers in FFN, totaling 6 locations per layer).
    - Unified rank r = 64 for all variants.
    - During inference, the LoRA weights are merged back into the original weight matrices, yielding zero additional inference latency.
    - Initialization: Truncated normal distribution with std=0.02.

### Loss & Training

- Training data: LaSOT + TrackingNet + GOT-10k (removing 1k overlapping sequences) + COCO 2017.
- 170 epochs, with 131,072 image pairs per epoch; GOT-10k specific variants reduced to 100 epochs.
- 8 × V100 GPUs, batch size of 128 (16 per card).
- LoRAT-B-224 can be trained on a single RTX 4090 within 11 hours.
- Inference: Standard Siamese tracking pipeline + Hanning window to suppress large displacements.

## Key Experimental Results

### Main Results

Comparison across five large-scale benchmarks:

| Tracker | LaSOT SUC | LaSOT_ext SUC | TrackingNet SUC | GOT-10k AO | TNL2K SUC |
|--------|-----------|---------------|----------------|------------|-----------|
| OSTrack-384 | 71.1 | 50.5 | 83.9 | 73.7 | 55.9 |
| SeqTrack-L384 | 72.5 | 50.7 | 85.5 | 74.8 | 57.8 |
| ARTrack | 73.1 | 52.8 | 85.6 | 78.5 | 60.3 |
| LoRAT-B-224 | 71.7 | 50.3 | 83.5 | 72.1 | 58.8 |
| LoRAT-L-378 | 75.1 | 56.6 | 85.6 | 77.5 | 62.3 |
| **LoRAT-g-378** | **76.2** | 56.5 | **86.0** | **78.9** | **62.7** |

Efficiency comparison:

| Tracker | FPS | MACs (G) | Total Params (M) | Trainable Params |
|--------|-----|----------|-----------|-----------|
| SeqTrack-L384 | 6 | 524 | 309 | All |
| LoRAT-B-224 | **209** | 30 | 99 | 13M (LoRA:11 + Head:2) |
| LoRAT-L-224 | 119 | 103 | 336 | 32M (LoRA:28 + Head:4) |
| LoRAT-g-378 | 20 | 1161 | 1216 | **80M** (LoRA:71 + Head:9) |

### Ablation Study

LoRA vs Full Fine-Tuning (LaSOT SUC / P):

| Variant | LoRA SUC | Full FT SUC | Δ |
|------|---------|-----------|---|
| B-224 | **71.7** | 70.9 | +0.8 |
| L-224 | **74.2** | 73.0 | +1.2 |
| L-378 | **75.1** | 74.9 | +0.2 |

Ablation of Input Embedding Configurations (ViT-L-224, LaSOT SUC):

| Freeze Pos. Enc. | Shared Pos. Enc. | Type Emb. | Foreground Ind. | SUC |
|---------|---------|---------|---------|-----|
| ✗ | ✗ | ✗ | ✗ | 73.9 |
| ✗ | ✓ | ✓ | ✗ | **74.2** |
| ✓ | ✓ | ✓ | ✗ | 74.0 |
| ✓ | ✓ | ✓ | ✓ | **74.2** |

### Key Findings

1. **LoRA Fine-Tuning Outperforms Full Fine-Tuning**: LoRA performs better than full parameter fine-tuning on nearly all variants, indicating that LoRA effectively mitigates catastrophic forgetting, thereby better preserving the rich pre-trained visual knowledge.
2. **Larger Models and Greater LoRA Advantages**: On L-224, LoRA yields a +1.2 SUC gain (with full fine-tuning achieving only 73.0).
3. **First Use of ViT-g for Tracking**: Performance escalates from 0.731 (ARTrack with ViT-B) to 0.762 (LoRAT with ViT-g) on LaSOT, demonstrating a positive correlation between model scale and tracking performance.
4. **Shared Position Encoding + Type Embedding Brings Consistent Gains**: The ablation study proves that independent position encoding is inferior to the shared scheme under the PEFT setting.
5. **Foreground Indicator Embedding is More Effective at Higher Resolutions**: A +0.7 SUC gain is observed on L-378 (75.1 vs 74.4), as high-resolution templates contain more background tokens.
6. **Substantial Boost in Training Efficiency**: The training time of L-224 drops from 35.0 to 10.8 GPU hours; the ViT-g variant requires only 25.8GB of GPU memory.

## Highlights & Insights

- **Proposing the "LoRA-friendly" Design Principle**: Key Insight—PEFT requires preserving the structural integrity of pre-trained models as much as possible; designs that disrupt the structure perform poorly under PEFT.
- **Cross-Domain Transfer of BERT Expertise**: Inspired by segment embeddings in NLP's BERT, it elegantly addresses the input compatibility issue in visual tracking.
- **High Practical Value**: A highly competitive tracker can be trained on a single consumer-grade GPU (B-224, 209 FPS, 71.7 SUC), significantly lowering the research barrier.
- **Systematic Exploration of LoRA in Vision Tasks**: Beyond validating feasibility, it systematically identifies typical obstacles to its application (position encoding, convolutional heads) and provides corresponding solutions.
- **First Application of ViT-g in Tracking**: Opens up new research directions for large-model tracking.

## Limitations & Future Work

- The LoRA rank r = 64 is unified for all variants without tailored tuning/search.
- Only DINOv2 pre-trained weights are validated; other pre-training schemes like MAE and CLIP are left unexplored.
- The simple design of the MLP head may limit the upper bound of localization precision.
- Advanced tracking strategies such as dynamic template updates have not been explored.
- The speed of ViT-g-378 is still only 20 FPS, which is far from real-time application requirements.
- The improvement on GOT-10k is less significant compared to LaSOT (the one-shot setting limits the efficacy of LoRA).

## Related Work & Insights

- **OSTrack**: The one-stream framework serves as the optimal PEFT baseline due to its minimal modifications to the pre-trained ViT.
- **SeqTrack / ARTrack**: Autoregressive schemes incorporating decoders, which are heavier but more flexible.
- **LoRA / PEFT**: Parameter-efficient methods in NLP that approximate weight updates using low-rank matrix decomposition.
- **BERT**: The inspiration for the token type embedding.
- Insights: (1) The LoRA-friendly principle can be extended to other downstream vision tasks; (2) the advantages of LoRA become increasingly pronounced as the pre-trained model grows larger.

## Rating

- Novelty: ⭐⭐⭐⭐ (First work to apply LoRA to tracking, with insightful problem analysis and solutions)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (5 benchmarks, 6 variants, exhaustive ablation studies, and efficiency analysis)
- Writing Quality: ⭐⭐⭐⭐⭐ (Rigorous logical flow from motivation to problem identification and solution, clear charts and tables)
- Value: ⭐⭐⭐⭐⭐ (Significantly lowers the barrier to large-model tracking research, strongly driving open community development)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Boosting 3D Single Object Tracking with 2D Matching Distillation and 3D Pre-training](boosting_3d_single_object_tracking_with_2d_matching_distillation_and_3d_pre-trai.md)
- [\[ICLR 2026\] FARTrack: Fast Autoregressive Visual Tracking with High Performance](../../ICLR2026/video_understanding/fartrack_fast_autoregressive_visual_tracking_with_high_performance.md)
- [\[ECCV 2024\] Towards Model-Agnostic Dataset Condensation by Heterogeneous Models](towards_model-agnostic_dataset_condensation_by_heterogeneous_models.md)
- [\[ECCV 2024\] VideoMamba: State Space Model for Efficient Video Understanding](videomamba_state_space_model_for_efficient_video_understanding.md)
- [\[ECCV 2024\] VideoMamba: Spatio-Temporal Selective State Space Model](videomamba_spatio-temporal_selective_state_space_model.md)

</div>

<!-- RELATED:END -->

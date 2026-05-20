---
title: >-
  [Paper Note] Token Bottleneck: One Token to Remember Dynamics
description: >-
  [NeurIPS 2025][Video Understanding][Self-supervised learning] This paper proposes Token Bottleneck (ToBo), a self-supervised visual representation learning pipeline that compresses a reference scene into a single bottlen…
tags:
  - "NeurIPS 2025"
  - "Video Understanding"
  - "Self-supervised learning"
  - "visual representation"
  - "robot manipulation"
  - "token bottleneck"
  - "sequential scene understanding"
date: 2026-05-08
content_hash: a6cbb63d0b986536
---

# Token Bottleneck: One Token to Remember Dynamics

**Conference**: NeurIPS 2025
**arXiv**: [2507.06543](https://arxiv.org/abs/2507.06543)  
**Code**: [GitHub](https://github.com/naver-ai/tobo)  
**Area**: Video Understanding
**Keywords**: Self-supervised learning, visual representation, robot manipulation, token bottleneck, sequential scene understanding

## TL;DR

This paper proposes Token Bottleneck (ToBo), a self-supervised visual representation learning pipeline that compresses a reference scene into a single bottleneck token and uses this token together with a minimal number of target scene patches to reconstruct the subsequent scene, thereby training visual backbone networks to simultaneously encode scene information conservatively and capture temporal dynamics.

## Background & Motivation

### Core Requirements for Sequential Scene Understanding

In sequential scene understanding tasks such as visual tracking and robot manipulation, visual backbone networks require two capabilities: (1) conservatively encoding the current observed scene state, and (2) capturing temporal dynamic transitions between consecutive scenes. Existing self-supervised methods exhibit notable deficiencies in both respects.

### Limitations of Static-Scene SSL

Self-supervised methods based on static images, such as MAE, acquire strong localization ability through masked prediction but are never optimized to compare consecutive frames and thus cannot model temporal dynamics. Recent studies further reveal that such methods struggle to learn broader context, limiting global scene understanding.

### Limitations of Dynamic-Scene SSL

Methods such as SiamMAE attempt to introduce cross-frame correspondence learning within the MAE framework by propagating reference-frame patches to their corresponding positions in the target frame. However, the authors find that the improvements offered by such approaches are quite limited and, on certain robot manipulation tasks, they even underperform MAE.

**Root Cause Analysis**: Although the training objective of SiamMAE establishes patch-level correspondences, it neglects the holistic understanding of what these matches signify. In other words, **identifying temporal change alone is insufficient; the task additionally requires the ability to summarize the observed scene without loss of information**.

### Efficiency Issues of Compositional Architectures

RSP achieves comprehensive capability by combining masked autoencoding, global representation alignment, and target-scene reconstruction, but its computational overhead exceeds that of competing methods by more than a factor of two (32.5 vs. 13.0–15.9 GFLOPs), yielding an unfavorable performance-to-cost ratio.

## Method

### Overall Architecture

ToBo consists of two steps: a **Squeeze** step that compresses the reference scene into a single bottleneck token, and a **Reconstruction** step that uses the bottleneck token together with a minimal number of target scene patches to predict the target scene.

### Key Designs

#### 1. Compression Mechanism of the Bottleneck Token

Given a reference scene $\mathbf{x}^t$ and a target scene $\mathbf{x}^{t+k}$ (with temporal interval $k$), both are patchified into $N$ non-overlapping patches.

All patches from the reference scene are fed into encoder $f_\theta$, producing spatial representations $\{\mathbf{u}_i^t\}$. The CLS token output during encoding serves as the **bottleneck token** $\mathbf{u}_{tobo}$, which is guided to compactly summarize the entire reference scene.

**Design Essence**: By imposing an information bottleneck—compressing the entire scene into a single token—the encoder is forced to retain only the most essential information rather than distributing it across multiple tokens. Ablation experiments confirm that performance is optimal with a single bottleneck token (1 token mean: 61.1% vs. 2 tokens: 41.8% vs. 4 tokens: 36.1%).

#### 2. Target-Scene Reconstruction with an Extremely High Masking Ratio

The target scene $\mathbf{x}^{t+k}$ is masked at an **extremely high masking ratio of $r=0.9$**, retaining only approximately 10% of patches as hints. These sparse target patches are processed by the same encoder, then concatenated with the bottleneck token; mask tokens are appended and the sequence is passed to decoder $d_\phi$ to predict the missing patches.

**Key Constraint**: Because target-scene information is extremely scarce, the decoder is forced to rely heavily on the bottleneck token to complete reconstruction. This yields two learning effects:
- The bottleneck token must retain critical information from the reference scene (otherwise reconstruction fails).
- This information must be encoded in a manner that recognizes temporal dynamics (understanding scene changes is required when interleaved with target hints).

The training loss is the cosine distance:

$$\mathcal{L}_{\text{ToBo}} = \sum_{i \in \mathcal{M}} d(\hat{\mathbf{x}}_i^{t+k}, \mathbf{x}_i^{t+k})$$

#### 3. Pure Self-Attention Decoder

Unlike SiamMAE and related methods that employ cross-attention layers to learn temporal correspondences, ToBo's decoder uses only self-attention layers and MLP layers. This ensures the decoder reasons solely from the provided information (bottleneck token + sparse target patches) without introducing additional cross-attention mechanisms.

The decoder comprises 8 Transformer blocks and incurs a computational cost of only 15.9 GFLOPs, far below RSP's 32.5 GFLOPs.

### Loss & Training

- Dataset: Kinetics-400, trained for 400 epochs (with 2× repeated sampling, effectively 200 epochs).
- Architecture: ViT-S/16 (21.7M parameters).
- Frame sampling: 30 FPS, temporal interval of 4–96 frames.
- Optimizer: AdamW, batch size 1536.
- Auxiliary objective: An additional Siamese masked autoencoding loss to reinforce patch-level correspondence learning.

## Key Experimental Results

### Main Results on Robot Manipulation (Franka Kitchen)

| Task | MAE | SiamMAE | RSP | CropMAE | **ToBo** | Gain |
|------|-----|---------|-----|---------|---------|------|
| Knob1 on | 12.0 | 16.8 | 31.0 | 31.5 | **57.0** | +25.5 |
| Light on | 24.3 | 36.5 | 44.5 | 54.0 | **82.0** | +28.0 |
| Sdoor open | 71.5 | 68.0 | 82.5 | 77.0 | **95.0** | +12.5 |
| Ldoor open | 12.8 | 17.3 | 28.8 | 25.5 | **51.0** | +22.2 |
| Micro open | 10.0 | 13.5 | 30.3 | 32.5 | **55.0** | +22.5 |

### Real-Robot Manipulation

| Method | Cabinet Opening | Drawer Closing | Cup Stacking |
|--------|----------------|----------------|-------------|
| SiamMAE | 20.0 | 55.0 | 50.0 |
| RSP | 25.0 | 65.0 | 55.0 |
| CropMAE | 0.0 | 25.0 | 20.0 |
| **ToBo** | **65.0** | **75.0** | **80.0** |

### Ablation Study

| Configuration | Knob1 on | Light on | Sdoor open | Ldoor open | Micro open | Mean |
|---------------|----------|----------|------------|------------|------------|------|
| 1 bottleneck token | **46.7** | **78.7** | **95.3** | **47.3** | **37.3** | **61.1** |
| 2 tokens | 31.0 | 54.0 | 74.0 | 26.0 | 24.0 | 41.8 |
| 4 tokens | 28.0 | 24.3 | 78.0 | 28.0 | 22.0 | 36.1 |
| 8 tokens | 10.0 | 20.0 | 56.0 | 26.0 | 9.3 | 24.3 |

| Masking Ratio | Knob1 on | Light on | Sdoor open | Ldoor open | Micro open |
|---------------|----------|----------|------------|------------|------------|
| 0.50 | 14.0 | 24.0 | 70.0 | 16.0 | 14.0 |
| 0.75 | 26.0 | 60.0 | 79.0 | 28.0 | 22.0 |
| **0.90** | **46.7** | **78.7** | **95.3** | **47.3** | **37.3** |
| 0.95 | 34.0 | 66.0 | 86.0 | 38.0 | 26.0 |

### Key Findings

1. **Single token is optimal**: As the number of bottleneck tokens increases from 1 to 8, performance degrades continuously (61.1% → 24.3%), confirming that extreme compression is necessary to force retention of critical information.
2. **Extremely high masking ratio is critical**: 0.9 is the optimal point; performance drops substantially at 0.5 and 0.75, and over-pruning at 0.95 also causes degradation, validating the design logic of "extreme scarcity → forced reliance on the bottleneck."
3. **Generalization to real environments**: ToBo achieves a 65% success rate on the Cabinet Opening task, whereas the best competing method reaches only 25%.
4. **Comparison with vision–language models**: ToBo (21.7M parameters, 0.2B frames) outperforms CLIP (149M parameters, 12.8B frames), DINOv2, SigLIP, and others, leading by at least 13% on every Kitchen task.
5. **Scalability**: ToBo maintains a substantial lead over baselines on both ViT-B/16 and ViT-L/16.

## Highlights & Insights

1. **Minimalist yet highly effective information bottleneck design**: Encoding an entire scene into a single token is counterintuitive, yet precisely this extreme constraint compels the representation to retain only the most essential information.
2. **Thorough motivation analysis**: The paper clearly articulates a progressive rationale from MAE (no temporal modeling) → SiamMAE (temporal correspondence but no holistic summary) → ToBo (compression and temporal reasoning unified).
3. **Excellent performance-to-cost ratio**: Training compute of 15.9 GFLOPs is far below RSP (32.5 GFLOPs), yet performance is substantially superior.
4. **Real-world deployment validation**: Evaluation on physical robots provides strong evidence of practical utility.
5. **No additional inference overhead**: All models use the same backbone and input resolution at inference time, resulting in identical FLOPs.

## Limitations & Future Work

- Naïve extension to multi-frame source inputs performs poorly (46.9% vs. 61.1%), necessitating dedicated multi-frame bottleneck designs.
- The method is currently validated only on robot manipulation and video label propagation; evaluation on broader video understanding tasks remains to be conducted.
- Downstream tasks employ a frozen backbone with a simple MLP policy head; whether more complex policy networks benefit equally is an open question.
- The method exhibits some sensitivity to video quality in training data and to the choice of temporal interval (optimal interval: 96 frames).
- The interpretability of the bottleneck token has not been analyzed in depth.

## Related Work & Insights

- **Static-image SSL**: MAE, SimMIM, DINO, SimCLR, MoCo v3
- **Dynamic-scene SSL**: SiamMAE, RSP, CropMAE
- **Robot representation learning**: VC-1 (MAE+Ego4D), MVP, R3M, Voltron, Theia
- **Insights**: The information bottleneck concept is broadly applicable to other sequential decision-making tasks requiring compact state representations; the extremely high masking ratio strategy can be extended to cross-modal correspondence learning (e.g., speech–vision).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — The single-token bottleneck combined with an extremely high masking ratio is a novel and counterintuitive design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive validation across simulation, real robots, video propagation, multiple scales, and extensive ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ — Motivation is clearly derived, and the progressive baseline analysis is convincing.
- Value: ⭐⭐⭐⭐⭐ — A small model trained on limited data surpasses large-scale VLMs, offering important guidance for robot visual representation learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] AIM: Adaptive Inference of Multi-Modal LLMs via Token Merging and Pruning](../../ICCV2025/video_understanding/aim_adaptive_inference_of_multi_modal_llms_via_token_merging_and_pruning.md)
- [\[CVPR 2026\] StreamingTOM: Streaming Token Compression for Efficient Video Understanding](../../CVPR2026/video_understanding/streamingtom_streaming_token_compression_for_efficient_video_understanding.md)
- [\[CVPR 2026\] UTPTrack: Towards Simple and Unified Token Pruning for Visual Tracking](../../CVPR2026/video_understanding/utptrack_towards_simple_and_unified_token_pruning_for_visual_tracking.md)
- [\[NeurIPS 2025\] When One Moment Isn't Enough: Multi-Moment Retrieval with Cross-Moment Interactions](when_one_moment_isnt_enough_multi-moment_retrieval_with_cross-moment_interaction.md)
- [\[ICLR 2026\] Video-KTR: Enhancing Video Reasoning via Key Token Attribution](../../ICLR2026/video_understanding/video-ktr_reinforcing_video_reasoning_via_key_token_attribution.md)

</div>

<!-- RELATED:END -->

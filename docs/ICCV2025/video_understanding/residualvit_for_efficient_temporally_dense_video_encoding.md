---
title: >-
  [Paper Note] ResidualViT for Efficient Temporally Dense Video Encoding
description: >-
  [ICCV 2025][Video Understanding][Video encoding efficiency] This paper proposes ResidualViT, which draws an analogy to I-frame/P-frame strategies in video compression by alternating between a full ViT and a lightweight residual ViT for encoding video frames. The approach achieves up to 60% reduction in computational cost and 2.5× inference speedup while maintaining accuracy close to the original CLIP.
tags:
  - "ICCV 2025"
  - "Video Understanding"
  - "Video encoding efficiency"
  - "ViT"
  - "temporal redundancy"
  - "knowledge distillation"
  - "temporally dense features"
date: 2026-05-08
content_hash: fd51fe0673083d18
---

# ResidualViT for Efficient Temporally Dense Video Encoding

**Conference**: ICCV 2025
**arXiv**: [2509.13255](https://arxiv.org/abs/2509.13255)  
**Code**: None  
**Area**: Video Understanding
**Keywords**: Video encoding efficiency, ViT, temporal redundancy, knowledge distillation, temporally dense features

## TL;DR
This paper proposes ResidualViT, which draws an analogy to I-frame/P-frame strategies in video compression by alternating between a full ViT and a lightweight residual ViT for encoding video frames. The approach achieves up to 60% reduction in computational cost and 2.5× inference speedup while maintaining accuracy close to the original CLIP.

## Background & Motivation
Many video understanding tasks—such as natural language temporal video grounding (NLTVG), temporal action localization (TAL), and audio description generation (AD)—require temporally dense, frame-level inference, typically demanding high temporal resolution sampling at 1–5 FPS. However, increasing the frame rate from 0.1–0.5 FPS to 1–5 FPS raises computational resource requirements by 2–50×, posing a significant challenge for large-scale deployment.

Existing approaches to reducing computational cost primarily rely on distilling large models into smaller ones, which often incurs accuracy degradation. More critically, these methods treat video frames independently, **without exploiting the inherent temporal redundancy in video**—adjacent frames are typically highly similar visually.

The core insight of this paper is that video compression has long exploited this temporal redundancy (I-frames store complete information; P-frames store only differences), and the same strategy can be applied to visual feature encoding. By running a full ViT on a subset of frames (I-features) and an efficient approximate encoder on adjacent frames (P-features), the method substantially reduces computational cost with negligible accuracy loss.

## Method

### Overall Architecture
ResidualViT adopts an interleaved encoding strategy: for every $N+1$ frames in a video sequence, the full CLIP ViT encoder $\mathcal{E}_\mathcal{V}$ computes I-features, while the subsequent $N$ frames are encoded by the lightweight ResidualViT encoder $\mathcal{E}_\mathcal{S}$ to produce P-features. The computation of P-features leverages temporal context provided by the preceding I-feature.

### Key Designs
1. **Token Reduction Module $\mathcal{R}$**:

    - Function: Substantially reduces the number of input tokens when ResidualViT encodes P-features.
    - Mechanism: Applies a PatchDropout strategy, discarding patch tokens with probability $p$ while retaining the most informative ones. Random, uniform, center-based, and motion-based dropping strategies are explored.
    - Design Motivation: The computational complexity of ViT scales quadratically with the number of tokens; reducing token count significantly lowers encoding cost. Ablation studies show that token dropping outperforms token merging and resolution reduction in the efficiency–accuracy trade-off.

2. **Residual Tokenizer Module $\mathcal{A}$**:

    - Function: Transforms I-features $f_t$ into residual tokens injected into the P-feature computation.
    - Mechanism: A learnable linear projection $\mathcal{A}: \mathbb{R}^b \rightarrow \mathbb{R}^d$ maps I-features into tokens compatible with the ViT input space, which are then concatenated with the [CLS] token and sparse frame tokens before being fed into the ViT.
    - Design Motivation: Token reduction inevitably discards some visual information, yet the temporal continuity between adjacent frames implies that the preceding frame's features contain substantial reusable semantic content. The residual token adds only approximately 0.1 GFLOPs (0.1% of the frame encoding cost), making its overhead negligible.

3. **Interleaved Encoding Strategy**:

    - Function: Determines the alternation frequency between I-features and P-features.
    - Mechanism: The average encoding cost is $C = C_{\mathcal{E}_\mathcal{V}} \frac{1+(1-p)N}{1+N}$, which is strictly less than $C_{\mathcal{E}_\mathcal{V}}$ when $N>0$ and $p>0$.
    - Design Motivation: $N=2$ represents the optimal trade-off, achieving 56% cost savings with virtually no accuracy loss. For larger $N$, the temporal distance between I-features and P-features increases, weakening visual relevance.

### Loss & Training
The method employs visual-language feature distillation for training: the teacher is the original CLIP ViT encoder $\mathcal{E}_\mathcal{V}$, and the student is ResidualViT $\mathcal{E}_\mathcal{S}$. The loss function is a bidirectional soft-target cross-entropy:

$$\mathcal{J}_{L \rightarrow V} = -\sum_{i=1}^{B}\sum_{k=1}^{N}\sum_{j=1}^{B} \sigma_j(g^\top f_{i,t+k}^{(\mathcal{V})}) \log(\sigma_j(g^\top f_{i,t+k}^{(\mathcal{S})}))$$

The final loss is $\min_\mathcal{A}(\mathcal{J}_{L \rightarrow V} + \mathcal{J}_{V \rightarrow L})$. Key characteristics:
- Only the residual tokenizer $\mathcal{A}$ (a single linear transformation) is trained; ViT weights are frozen.
- Training is conducted on WebVid-2.5M for 5 epochs using 4 V100 GPUs.
- The objective encourages not only visual feature alignment but also preserves CLIP's joint visual-language embedding space.

## Key Experimental Results

### Main Results

| Dataset | Metric | ResidualViT (L/14) | CLIP (L/14) | Cost Savings |
|--------|------|---------------------|-------------|---------|
| Charades-STA | R@1, IoU=0.5 | 41.5 | 42.9 | 56% |
| Charades-STA | R@1, IoU=0.7 | 23.8 | 24.1 | 56% |
| ActivityNet-Captions | R@1, IoU=0.5 | 28.3 | 29.1 | 56% |
| MAD (long video) | R@1, IoU=0.5 | 4.3 | 5.0 | 56% |
| MAD (long video) | R@1, IoU=0.3 | 7.3 | 8.6 | 56% |

### Ablation Study

| Configuration | R@1 IoU=0.5 | R@1 IoU=0.7 | Avg. Cost (GFLOPs) | Notes |
|------|------------|------------|-------------------|------|
| CLIP baseline | 42.9 | 24.1 | 233.4 | Upper bound |
| Token Reduction only | 28.5 | 14.5 | 35.7 (−85%) | Large accuracy drop |
| + Interleave (N=2) | 38.9 | 22.8 | 102.0 (−56%) | Most accuracy recovered |
| + Residual Tokenizer (distillation) | 41.5 | 23.8 | 102.6 (−56%) | Near-original accuracy |

### Key Findings
- Token reduction alone causes a 34–40% relative accuracy drop, but combined with interleaved encoding, the accuracy loss shrinks to only 5–9%.
- The residual tokenizer further improves accuracy by approximately 2.6 percentage points (IoU=0.5) with negligible additional computational cost.
- $N=2$ is the optimal interleaving factor, balancing 56% cost savings with near-lossless accuracy.
- The method performs well in both zero-shot and supervised settings, and generalizes to both short and long videos.
- ResidualViT combined with a zero-shot grounding algorithm advances the state of the art on the MAD long-video dataset to 3.1 R@1 (IoU=0.5) while halving computation.

## Highlights & Insights
- **Elegant and effective analogy**: Transporting the I-frame/P-frame concept from video compression into feature encoding yields clear intuition and strong empirical results.
- **Minimal training cost**: Only a single linear layer (the residual tokenizer) requires training; large-scale training data are unnecessary, and the entire training process completes in 5 epochs on 4 V100 GPUs.
- **Preservation of visual-language alignment**: The distillation objective approximates not only visual features but also preserves CLIP's multimodal embedding space, which is key to enabling zero-shot generalization.
- **Broad task generalization**: The method is effective across four tasks and five datasets, covering both short and long videos (NLTVG, AD, TAL, AR).
- **Practical speedup**: Beyond GFLOPs reduction, actual wall-clock inference time improves by 2.5×, demonstrating clear engineering value.

## Limitations & Future Work
- Accuracy degrades noticeably when $N > 3$; for videos with rapid scene changes or large motion, the temporal redundancy assumption may not hold.
- The method has only been validated on CLIP; generalizability to other vision foundation models (e.g., DINOv2, SigLIP, InternVL) remains to be explored.
- The motion-based token dropping strategy relies on optical flow estimation, which may introduce additional computational overhead requiring careful trade-off analysis in practical deployment.
- In long videos with scene transitions, the reference value of I-features for subsequent P-features may degrade sharply, necessitating adaptive detection mechanisms.
- Training exclusively on WebVid-2.5M leaves generalization to larger scales or different domains (e.g., medical, remote sensing) unverified.
- Dynamic combination or adaptive selection among different token reduction strategies has not been explored.

## Related Work & Insights
- The connection to video compression suggests that **temporal redundancy in video is a valuable resource for reducing computational cost**, an insight transferable to a broader range of video understanding scenarios.
- Token reduction techniques (PatchDropout, ToMe) are emerging as important tools for efficient ViT design and merit exploration across more architectures.
- The lightweight distillation strategy (training only a handful of parameters) demonstrates an efficient paradigm for adapting large models, with practical implications for resource-constrained settings.
- The I/P interleaved encoding design can be extended to storage optimization for video embeddings—storing only I-features and computing P-features on-the-fly as needed.
- Joint visual-language distillation (rather than visual-only distillation) is a critical strategy for preserving multimodal capabilities and can be applied to other VLM compression scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐ The analogy from video compression to feature encoding is clever, though token reduction and distillation are both established techniques.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four tasks, five datasets, both zero-shot and supervised settings covered, with comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure with a complete motivation–method–experiment logical chain and well-crafted figures.
- Value: ⭐⭐⭐⭐ Practically valuable for temporally dense video tasks; broader generalizability still requires validation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Sparse-Dense Side-Tuner for Efficient Video Temporal Grounding](sparse-dense_side-tuner_for_efficient_video_temporal_grounding.md)
- [\[ICCV 2025\] AllTracker: Efficient Dense Point Tracking at High Resolution](alltracker_efficient_dense_point_tracking_at_high_resolution.md)
- [\[CVPR 2025\] EDCFlow: Exploring Temporally Dense Difference Maps for Event-based Optical Flow Estimation](../../CVPR2025/video_understanding/edcflow_exploring_temporally_dense_difference_maps_for_event-based_optical_flow_.md)
- [\[ICCV 2025\] Factorized Learning for Temporally Grounded Video-Language Models](factorized_learning_for_temporally_grounded_video-language_models.md)
- [\[ICCV 2025\] TOGA: Temporally Grounded Open-Ended Video QA with Weak Supervision](toga_temporally_grounded_open-ended_video_qa_with_weak_supervision.md)

</div>

<!-- RELATED:END -->

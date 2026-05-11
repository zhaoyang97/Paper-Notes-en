---
title: >-
  [Paper Note] Aligning Moments in Time using Video Queries
description: >-
  [ICCV 2025][Video Generation][Video Moment Retrieval] This paper proposes MATR (Moment Alignment TRansformer), which conditions target video representations on query video features via dual-stage sequence alignment (soft…
tags:
  - "ICCV 2025"
  - "Video Generation"
  - "Video Moment Retrieval"
  - "Video Query"
  - "Sequence Alignment"
  - "Self-Supervised Pretraining"
  - "Transformer"
date: 2026-05-08
content_hash: 9e1a56827f0960b7
---

# Aligning Moments in Time using Video Queries

**Conference**: ICCV 2025
**arXiv**: [2508.15439](https://arxiv.org/abs/2508.15439)
**Code**: [GitHub](https://github.com/vl2g/MATR)
**Area**: Video Generation
**Keywords**: Video Moment Retrieval, Video Query, Sequence Alignment, Self-Supervised Pretraining, Transformer

## TL;DR
This paper proposes MATR (Moment Alignment TRansformer), which conditions target video representations on query video features via dual-stage sequence alignment (soft-DTW), enabling video-to-video moment retrieval (Vid2VidMR). A self-supervised pretraining strategy is designed accordingly, achieving +13.1% R@1 and +8.1% mIoU on ActivityNet-VRL.

## Background & Motivation

1. **Background**: Video Moment Retrieval (VMR) has primarily focused on text query settings, as exemplified by Moment-DETR and QD-DETR. Vid2VidMR, which uses video queries, is a relatively nascent paradigm formally introduced by Feng et al.
2. **Limitations of Prior Work**: Text queries often fail to precisely describe complex actions (e.g., bicycle kicks), and users find it difficult to convey what they have seen in natural language. Existing Vid2VidMR methods lack explicit semantic frame-level alignment and the capacity to model complex query-target dependencies.
3. **Key Challenge**: Query and target videos vary greatly in length, speed, and context, requiring simultaneous capture of high-level semantic relationships and fine-grained frame-level dependencies. Existing methods either naively substitute text encoders with C3D features or lack precise temporal alignment.
4. **Goal**: (1) How to precisely align the semantic sequences of query and target videos? (2) How to improve model generalization using unannotated data?
5. **Key Insight**: Differentiable soft-DTW is applied in a dual-stage alignment scheme before and after the Transformer encoder, combined with self-supervised pretraining (randomly sampling clips from a video as queries for self-localization).
6. **Core Idea**: Explicit dual-stage sequence alignment within a Transformer framework transforms target video representations into query-aligned representations, enabling precise video moment localization.

## Method

### Overall Architecture
Given a target video $V_t$ ($M$ frames) and a query video $V_q$ ($N$ frames), both are encoded by CLIP ViT-B/32 and linearly projected into $d$-dimensional embeddings. The concatenated embeddings are fed into a Transformer Encoder for joint understanding. The resulting query-aligned representations are passed to foreground classification and boundary prediction heads.

### Key Designs

1. **Dual-stage Sequence Alignment**:
    - **Function**: Sequence alignment is performed twice — before and after the encoder — to capture both global semantics and fine-grained temporal dependencies.
    - **Mechanism**: Soft-DTW (differentiable dynamic time warping) is used for alignment. **Pre-fusion alignment**: Before the encoder, a cosine-similarity-based alignment cost matrix is computed between target embeddings $E_t$ and query embeddings $E_q$ as $C_{i,j} = 1 - \frac{\langle e_i^t, e_j^q \rangle}{\|e_i^t\| \|e_j^q\|}$, and the loss $\mathcal{L}_{\text{align}}^{\text{pre}} = \text{soft-DTW}_\gamma(A^{\text{pre}}, C^{\text{pre}})$ is minimized. **Post-fusion alignment**: Soft-DTW is applied again on the encoder output representations $E_t^g, E_q^g$. The aligned subsequence $E_t^g[s:e]$ identified by post-fusion alignment is forwarded to the decoder for further refinement.
    - **Design Motivation**: Pre-fusion alignment enhances semantic representations, while post-fusion alignment performs finer matching on fused features. The two stages are complementary. Soft-DTW enables non-linear alignment of sequences with different lengths, is differentiable, and is robust to speed variations.

2. **Self-Supervised Pretraining Strategy**:
    - **Function**: Initializes the model's moment localization capability without requiring annotated data.
    - **Mechanism**: A random clip is sampled from the target video $V_t$ and used as the query $V_q$. The model is trained to localize the start and end times of this clip within $V_t$. Random augmentations (frame reversal, Gaussian noise, speed perturbation) are applied to the query clip, doubling the number of pretraining samples. The pretraining loss shares the same structure as the main training loss (foreground classification + boundary prediction + dual alignment).
    - **Design Motivation**: The pretraining objective is highly consistent with the Vid2VidMR task and leverages large-scale unlabeled video data (e.g., Kinetics700) to learn temporal localization skills, thereby improving generalization.

3. **Prediction Heads (Classification + Boundary Prediction)**:
    - **Function**: Outputs foreground labels and moment boundaries from the query-aligned representations.
    - **Mechanism**: The final representation $E_f = [E_t^g; E_t^l]$ combines global semantics from the encoder and fine-grained features from the decoder. The classification head uses 3-layer 1×3 convolutions followed by Sigmoid to produce foreground probabilities $\hat{f}_i$, trained with binary cross-entropy loss $\mathcal{L}_{fg}$. The boundary head shares the initial structure but outputs left and right offsets $(d_i^L, d_i^R)$, trained with smooth L1 and gIoU losses $\mathcal{L}_{seg}$ at foreground positions. 1D NMS (threshold 0.7) is applied at inference to suppress overlapping boundaries.
    - **Design Motivation**: Convolutional layers operating along the temporal axis preserve temporal continuity, while the decoupled foreground/boundary design simplifies the localization problem in dense prediction.

### Loss & Training
Total loss: $\mathcal{L} = \frac{1}{S}\sum(\lambda_{fg}\mathcal{L}_{fg} + \lambda_{seg}\mathcal{L}_{seg} + \lambda_{\text{align}}^{\text{pre}}\mathcal{L}_{\text{align}}^{\text{pre}} + \lambda_{\text{align}}^{\text{post}}\mathcal{L}_{\text{align}}^{\text{post}})$, with all $\lambda$ set to 1. AdamW optimizer with learning rate 1e-4 and weight decay 1e-4. Trained for 200 epochs on ActivityNet-VRL with batch size 1200.

## Key Experimental Results

### Main Results

| Method | ActivityNet-VRL mIoU | ActivityNet-VRL R@1 | SportsMoments mIoU | SportsMoments R@1 |
|--------|------|------|----------|------|
| FFI+SRM (Prev. SOTA) | 48.7 | 40.6 | — | — |
| MATR (Ours) | **61.8** | **53.7** | **Best** | **Best** |
| Gain | +13.1 | +13.1 | +14.4 | +14.7 |

Comparison with VLM-based methods:

| Method | mIoU | R@1 |
|------|------|------|
| TimeChat | 26.4 | 23.8 |
| Video-LLaMA2 | 17.6 | 15.2 |
| MATR | **61.8** | **53.7** |

### Ablation Study

| Configuration | R@1 | Note |
|------|---------|------|
| Full MATR | Best | Complete model |
| w/o Pre-fusion alignment | Notable drop | Remove pre-fusion alignment |
| w/o Post-fusion alignment | Drop | Remove post-fusion alignment |
| w/o Self-supervised pretraining | Drop | Remove self-supervised pretraining |
| w/o Dual-stage alignment | Largest drop | Remove both alignment stages |

### Key Findings
- Dual-stage alignment is the most critical design component; removing either stage leads to significant performance degradation.
- Self-supervised pretraining yields larger gains on SportsMoments (a smaller dataset).
- MATR performs particularly well in "re-occurrence" scenarios where subjects appear repeatedly across multiple shots.
- Video queries substantially outperform text queries, especially for complex action descriptions.

## Highlights & Insights
- The paper introduces the SportsMoments dataset (770K pairs, 176.6 hours of full match footage), filling a gap in fine-grained Vid2VidMR for the sports domain.
- A key insight of the self-supervised pretraining is that randomly cropping clips from the video itself for self-localization is highly consistent with the downstream task objective.
- The flexibility of soft-DTW enables alignment between video pairs of different lengths and speeds.
- The comparative experimental design — adapting text-query VMR methods (e.g., Moment-DETR) to video queries — is comprehensive and informative.

## Limitations & Future Work
- Features from CLIP ViT-B/32 may limit fine-grained dynamic understanding.
- Computational complexity scales with video length: soft-DTW is $O(MN)$ and the full-attention encoder is $O((M+N)^2)$.
- Retrieval with multiple simultaneous query videos has not been explored.
- SportsMoments covers only football and cricket; extension to additional sports remains future work.

## Related Work & Insights
- **vs. GDP/SRL**: These earlier Vid2VidMR methods lack explicit sequence alignment and Transformer-based architectures.
- **vs. Moment-DETR/QD-DETR**: Originally designed for text queries, these methods perform far worse than MATR when adapted to video queries.
- **vs. VLMs (Video-LLaVA, etc.)**: Zero-shot VLMs perform poorly on Vid2VidMR, indicating that the task requires specialized architectures.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of dual-stage sequence alignment and self-supervised pretraining is well-motivated and elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers four categories of baselines (fully supervised / VLM / text VMR / image VMR) with detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with well-articulated motivation.
- Value: ⭐⭐⭐⭐ Advances the Vid2VidMR field; the new dataset is expected to have lasting impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] VPO: Aligning Text-to-Video Generation Models with Prompt Optimization](vpo_aligning_text-to-video_generation_models_with_prompt_optimization.md)
- [\[NeurIPS 2025\] Autoregressive Adversarial Post-Training for Real-Time Interactive Video Generation](../../NeurIPS2025/video_generation/autoregressive_adversarial_posttraining_for_realtime_interac.md)
- [\[CVPR 2026\] StreamDiT: Real-Time Streaming Text-to-Video Generation](../../CVPR2026/video_generation/streamdit_real-time_streaming_text-to-video_generation.md)
- [\[ICLR 2026\] TTOM: Test-Time Optimization and Memorization for Compositional Video Generation](../../ICLR2026/video_generation/ttom_test-time_optimization_and_memorization_for_compositional_video_generation.md)
- [\[ICLR 2026\] MotionStream: Real-Time Video Generation with Interactive Motion Controls](../../ICLR2026/video_generation/motionstream_real-time_video_generation_with_interactive_motion_controls.md)

</div>

<!-- RELATED:END -->

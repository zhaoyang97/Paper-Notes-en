---
title: >-
  [Paper Note] An Empirical Study of Autoregressive Pre-training from Videos
description: >-
  [ICCV 2025][Video Understanding][Autoregressive Pre-training] This paper systematically investigates autoregressive pre-training from videos (termed Toto), training a causal Transformer on over one trillion visual tokens. Despite minimal inductive biases, the approach achieves competitive performance across image recognition, video classification, object tracking, and robot manipulation, while exhibiting scaling laws analogous to those of language models, albeit at a slower rate.
tags:
  - ICCV 2025
  - Video Understanding
  - Autoregressive Pre-training
  - Video Models
  - Visual Tokens
  - Scaling Laws
  - Representation Learning
date: 2026-05-08
content_hash: 3015fd21cb9b8a49
---

# An Empirical Study of Autoregressive Pre-training from Videos

**Conference**: ICCV 2025
**arXiv**: [2501.05453](https://arxiv.org/abs/2501.05453)
**Code**: None
**Area**: Visual Representation Learning / Video Understanding
**Keywords**: Autoregressive Pre-training, Video Models, Visual Tokens, Scaling Laws, Representation Learning

## TL;DR
This paper systematically investigates autoregressive pre-training from videos (termed Toto), training a causal Transformer on over one trillion visual tokens. Despite minimal inductive biases, the approach achieves competitive performance across image recognition, video classification, object tracking, and robot manipulation, while exhibiting scaling laws analogous to those of language models, albeit at a slower rate.

## Background & Motivation
Autoregressive pre-training has achieved remarkable success in NLP (e.g., the GPT series), with the central idea of modeling data distributions via next-token prediction. In the visual domain, particularly for video, this paradigm remains underexplored.

Existing visual representation learning methods fall into two major paradigms:

**Discriminative methods** (e.g., SimCLR, DINO): obtain strong recognition features via instance discrimination or contrastive learning, but do not directly model the data distribution.

**Generative methods** (e.g., MAE, BEiT): employ masked autoencoding pre-training, but are not autoregressive in nature.

Video constitutes the largest source of big data on the internet, and its temporal structure is naturally suited to autoregressive modeling. However, prior visual autoregressive works (e.g., iGPT) operate at the pixel level, incurring prohibitive computational costs and limited scalability.

The **Key Insight** of Toto is to discretize video frames into visual token sequences via a tokenizer, then perform causal next-token prediction analogously to language model training. This enables images and videos to be jointly trained under a unified format, leveraging mature training techniques and scaling insights from the language modeling community.

## Method

### Overall Architecture
Toto follows a straightforward pipeline: video/image frames are converted to discrete token sequences via a dVAE tokenizer → arranged into a 1D sequence in raster-scan order → modeled with a causal Transformer for next-token prediction → intermediate-layer features are extracted for downstream evaluation. After pre-training, representations are extracted from intermediate layers via attention pooling for downstream transfer.

### Key Designs

1. **Tokenizer Selection and Evaluation**:

    - **Function**: Convert image/video frames into discrete token sequences.
    - **Mechanism**: By default, a dVAE (vocabulary size 8k) is used, generating 256 tokens per frame (16×16 grid). Videos use 16 frames, yielding a context length of 4,096 tokens.
    - **Comparative Experiments**: dVAE, VQGAN, and continuous patch-normalized tokens achieve similar ImageNet linear probe accuracy (~61%), indicating limited sensitivity to tokenizer choice. However, VQGAN indirectly introduces ImageNet label information through its perceptual loss (via VGG-net), constituting a form of data contamination.
    - **Advantage of dVAE**: Its 1-gram distribution covers nearly all tokens, whereas VQGAN covers fewer than 50%.
    - **Design Motivation**: An image-level tokenizer enables joint processing of images and videos while avoiding leakage of supervisory signals.

2. **Architecture Design (LLaMA-style)**:

    - **Function**: Provide high-quality causal sequence modeling.
    - **Mechanism**: The LLaMA architecture is adopted—a causal attention Transformer incorporating RMSNorm (pre-normalization), SwiGLU activations, and RoPE positional encodings.
    - **Model Scales**: Base (120M, 12 layers), Large (280M, 16 layers), 1B (1.1B, 22 layers).
    - **Comparison**: LLaMA outperforms GPT-2 and Mamba by 4.7% and 12.5%, respectively, on ImageNet linear probing.
    - **Training Configuration**: Batch size of 1M tokens, AdamW optimizer, peak learning rate $3\times10^{-4}$, $\beta_1=0.9$, $\beta_2=0.95$.
    - **Design Motivation**: Incorporates state-of-the-art architectural improvements from the language modeling literature.

3. **Resolution Strategy and RoPE Adaptation**:

    - **Function**: Reduce pre-training cost while improving performance.
    - **Mechanism**: Pre-training is first conducted at 128×128 (16×16 tokens), followed by fine-tuning at 256×256 (32×32 tokens). Only 1 epoch of fine-tuning suffices, and the resulting model surpasses one trained at 256×256 throughout (64.4% vs. 61.2%).
    - **RoPE Base Adjustment**: The RoPE base is increased from 10,000 to 50,000 during fine-tuning, further improving high-resolution adaptation.
    - **Design Motivation**: High-resolution dVAE tokens are 4× more numerous than low-resolution ones, resulting in substantial differences in pre-training compute.

4. **Downstream Feature Extraction Strategy**:

    - **Function**: Extract high-quality visual representations from a decoder-only model.
    - **Attention Pooling vs. Average Pooling**: Attention pooling outperforms average pooling by 7.9% on ImageNet (61.1% vs. 53.2%). Causal attention causes later tokens to attend to more context, making simple averaging susceptible to distortion from the asymmetric structure.
    - **Optimal Probing Layer**: Across all model sizes and tasks, the best representations consistently emerge at approximately 50% network depth, consistent with findings from iGPT. This suggests that the first half of a decoder-only model compresses information like an encoder, while the second half projects compressed semantics back into the input space.
    - **Design Motivation**: Representation extraction from decoder-only models differs fundamentally from encoder–decoder architectures and warrants dedicated investigation.

### Dataset Construction
The training corpus totals approximately 2.5 trillion visual tokens from over 100,000 hours of video:
- ImageNet (13.9M images), sampling ratio 20%
- Kinetics-600 (530K videos, 1,496 hours), sampling ratio 10%
- Ego4D (52K videos, 3,750 hours), sampling ratio 10%
- HowTo100M (1.172M videos, 92K hours), sampling ratio 60%

Approximately 1 trillion tokens are used in practice.

### Loss & Training
- 16 frames are sampled per video (one frame every 4), with a context length of 4,096 tokens.
- For images, 16 images are randomly sampled to form a sequence, simulating the video format.
- Special tokens: videos begin with [1], images with [3], and all sequences end with [2].
- Loss function: standard negative log-likelihood $\mathcal{L}_{\text{pre-train}} = \mathbb{E}_{x^j \sim X} -\log p(x^j)$

## Key Experimental Results

### Main Results

| Task | Dataset | Toto-Base | Toto-Large | Toto-1B | Comparable SOTA |
|------|---------|-----------|------------|---------|-----------------|
| Image Recognition | ImageNet | 64.7% | 71.1% | 75.3% | iGPT-XL: 72.0% (6.8B) |
| Action Recognition | K400 | 59.3% | 65.3% | 74.4% | VideoMAE: 79.8% |
| Action Anticipation | Ego4D Overall | — | 2.70 | — | MAE-ST: 2.60 |
| Video Tracking | DAVIS J&F | 42.0 | 44.8 / 62.4 (512) | 46.1 | DINO-B/8: 54.3 |
| Object Permanence | CATER | — | 62.8 / 72.9 | — | TFC-V3D: 54.6 / 70.2 |
| Robot Manipulation | Real Franka | 63% | — | — | MVP: 75% |

### Ablation Study

| Design Choice | Best Config | Key Metric | Notes |
|--------------|-------------|------------|-------|
| Tokenizer | dVAE 32×32 | 61.2% Top-1 | dVAE and VQGAN comparable; dVAE avoids data contamination |
| Pooling | Attention pooling | 61.1% vs. 53.2% | Substantially superior to average pooling |
| Resolution Strategy | 16→32 + RoPE 50k | 64.4% Top-1 | Low-res pre-training + high-res fine-tuning is more effective and efficient |
| Architecture | LLaMA | 53.2% Top-1 | Outperforms GPT-2 (48.5%) and Mamba (40.7%) |
| Probing Layer | ~50% depth | — | Consistent across all models and tasks |

### Key Findings
- Toto-1B (1.1B parameters) matches the performance of iGPT-XL (6.8B parameters), demonstrating substantially improved scaling efficiency among autoregressive generative models.
- Scaling follows a power-law relationship $L(C) = 7.32 \cdot C^{-0.0378}$, which is slower than language models (GPT-3: $C^{-0.048}$).
- Temporal redundancy in video frames is identified as a likely contributor to reduced scaling efficiency.
- This work provides the first competitive K400 action recognition results within the autoregressive pre-training paradigm.
- The phenomenon of optimal representations at intermediate layers is consistent across all model sizes and tasks.

## Highlights & Insights
- A unified training format for images and videos enables a single model to cover diverse downstream tasks.
- RoPE's resolution adaptability makes low-resolution pre-training followed by high-resolution fine-tuning feasible, substantially reducing training costs.
- This is the first systematic study of compute-optimal scaling behavior for visual autoregressive models.
- Surpassing task-specific methods on object permanence (CATER) suggests that autoregressive pre-training implicitly learns long-horizon temporal reasoning.
- Robot manipulation experiments demonstrate the potential of generative pre-training for embodied intelligence.

## Limitations & Future Work
- Discriminative methods still substantially outperform Toto on most recognition tasks (e.g., DINO 80.1% vs. Toto 75.3%).
- Tokenizer quality becomes a performance bottleneck, as dVAE reconstruction quality is limited.
- Inter-frame redundancy in videos reduces the information density of training data, potentially necessitating more intelligent frame sampling strategies.
- Design choices are validated only on ImageNet classification and may not generalize to dense prediction tasks.
- Scaling efficiency is slower than language models, requiring substantially more compute to achieve comparable gains.

## Related Work & Insights
- **vs. iGPT**: Toto replaces pixels with tokens, achieving better scaling efficiency (1.1B vs. 6.8B parameters for similar performance).
- **vs. MAE**: MAE employs masked autoencoding rather than autoregressive modeling; its encoder–decoder structure places optimal representations at the top of the encoder, whereas Toto finds optimal representations at intermediate layers of a decoder-only model.
- **vs. AIM**: AIM uses CLIP-filtered data, indirectly introducing supervisory signals; Toto is fully unsupervised.
- **vs. DINO**: DINO achieves stronger recognition performance, but Toto's generative nature confers broader generality.

## Rating
- **Novelty**: ⭐⭐⭐ The methodological framework is relatively straightforward (video tokens + causal Transformer); the primary contribution lies in the systematic empirical investigation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers six task categories—image recognition, video classification, action anticipation, tracking, object permanence, and robot manipulation—with thorough ablation of design choices.
- **Writing Quality**: ⭐⭐⭐⭐ Logically structured with rigorous experimental design, though the presentation of methodological novelty is limited.
- **Value**: ⭐⭐⭐⭐ Provides important empirical reference for visual autoregressive models; the scaling law analysis offers meaningful guidance to the research community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Beyond the Frame: Generating 360° Panoramic Videos from Perspective Videos](beyond_the_frame_generating_360deg_panoramic_videos_from_perspective_videos.md)
- [\[ICCV 2025\] XTrack: Multimodal Training Boosts RGB-X Video Object Trackers](xtrack_multimodal_training_boosts_rgb-x_video_object_trackers.md)
- [\[ICCV 2025\] Fine-grained Spatiotemporal Grounding on Egocentric Videos](fine-grained_spatiotemporal_grounding_on_egocentric_videos.md)
- [\[ICCV 2025\] MEMFOF: High-Resolution Training for Memory-Efficient Multi-Frame Optical Flow Estimation](memfof_high-resolution_training_for_memory-efficient_multi-frame_optical_flow_es.md)
- [\[ICCV 2025\] Vamba: Understanding Hour-Long Videos with Hybrid Mamba-Transformers](vamba_understanding_hour-long_videos_with_hybrid_mamba-transformers.md)

</div>

<!-- RELATED:END -->

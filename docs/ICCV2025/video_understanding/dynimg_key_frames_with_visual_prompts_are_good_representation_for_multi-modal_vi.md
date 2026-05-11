---
title: >-
  [Paper Note] DynImg: Key Frames with Visual Prompts are Good Representation for Multi-Modal Video Understanding
description: >-
  [ICCV 2025][Video Understanding][Video Representation] DynImg proposes a novel video representation method that appends non-key frames as "temporal visual prompts" below key frames to form dynamic images, enabling fine-grained spatiotemporal interaction inside the visual encoder (rather than at the high-level token stage). Combined with a 4D rotary positional encoding to maintain correct spatiotemporal ordering, DynImg surpasses SOTA by approximately 2% on multiple video understanding benchmarks while using fewer visual tokens.
tags:
  - ICCV 2025
  - Video Understanding
  - Video Representation
  - Spatiotemporal Interaction
  - Visual Prompts
  - Positional Encoding
  - Multimodal LLM
date: 2026-05-08
content_hash: 960db990058695b9
---

# DynImg: Key Frames with Visual Prompts are Good Representation for Multi-Modal Video Understanding

**Conference**: ICCV 2025
**arXiv**: [2507.15569](https://arxiv.org/abs/2507.15569)
**Code**: [https://dynimg.github.io/](https://dynimg.github.io/)
**Area**: Video Understanding / Multimodal Large Language Models
**Keywords**: Video Representation, Spatiotemporal Interaction, Visual Prompts, Positional Encoding, Multimodal LLM

## TL;DR

DynImg proposes a novel video representation method that appends non-key frames as "temporal visual prompts" below key frames to form dynamic images, enabling fine-grained spatiotemporal interaction inside the visual encoder (rather than at the high-level token stage). Combined with a 4D rotary positional encoding to maintain correct spatiotemporal ordering, DynImg surpasses SOTA by approximately 2% on multiple video understanding benchmarks while using fewer visual tokens.

## Background & Motivation

Multimodal large language models (MLLMs) are increasingly applied to video understanding, yet effectively incorporating temporal information remains a central challenge:

- **Spatiotemporal decoupling in existing methods**: Conventional approaches process spatial and temporal information separately — pretrained image encoders first extract per-frame spatial features, and spatiotemporal interaction is then performed at the high-level query/token stage (e.g., parallel spatiotemporal pooling in Video-ChatGPT, Q-former with a temporal module in Video-LLaMA).
- **Loss of fine-grained information**: During visual feature extraction and spatial merging (k-means, pooling, Q-former, convolution), spatial information is progressively abstracted and compressed, causing many fine-grained details to be blurred and lost through averaging.
- **Neglect of fast-motion regions**: Due to motion blur and related factors, regions containing fast-moving objects fail to obtain accurate fine-grained representations during the spatial feature extraction stage. If these temporally important regions are overlooked at this early stage, subsequent token-level interaction becomes substantially less effective.

The root cause is that **spatiotemporal interaction occurs too late** — by the time interaction takes place on high-level abstract tokens, critical fine-grained motion information has already been lost. The paper's core idea is to **move spatiotemporal interaction earlier, to the pixel level during visual feature extraction**, using non-key frames as "temporal prompts" to guide the encoder's attention toward fast-motion regions.

## Method

### Overall Architecture

A video is decomposed into key frames and non-key frames. Each DynImg consists of one high-resolution key frame (top) and four downscaled non-key frames arranged chronologically (bottom row). DynImg is fed together with 4D positional encodings into the visual encoder, where fine-grained spatiotemporal interaction between the key frame and temporal prompts is achieved via self-attention. The output features are projected and passed to the LLM.

### Key Designs

1. **Temporal Prompts**:

    - **Function**: Incorporate temporal information into the spatial feature extraction process at the pixel level.
    - **Mechanism**: Four I-frames are uniformly sampled using the MPEG-4 method as key frames $K$. For each key frame $K_i$, two I-frames are randomly selected from the preceding and two from the following interval as non-key frames $N$. The non-key frames are downscaled and concatenated below the key frame to form a DynImg. Within ViT self-attention, patches from the key frame can attend to highly similar patches in the temporal prompts, enabling the model to identify motion trends.
    - **Design Motivation**: Inspired by visual prompting in multimodal image understanding (e.g., drawing bounding boxes on images), this approach extends the concept to video — non-key frames supply motion change information, and the ViT's long-range modeling capability facilitates spatiotemporal interaction within the encoder.

2. **4D Video Rotary Position Embedding (4D-RoPE)**:

    - **Function**: Maintain correct four-dimensional spatiotemporal ordering for visual tokens within a DynImg.
    - **Mechanism**: Standard 1D RoPE is extended to four dimensions — height (H), width (W), time (T), and sequence (S). The rotation angle is computed as a weighted sum: $x\cdot\theta = x_h\cdot\theta_h + x_w\cdot\theta_w + x_t\cdot\theta_t + x_s\cdot\theta_s$. Here $\theta_s$ retains the sinusoidal encoding from LLM pretraining, while $\theta_h, \theta_w, \theta_t$ are initialized to 0 (learnable) to avoid disrupting the pretrained LLM at the start of training.
    - **Design Motivation**: The composite structure of DynImg is unfamiliar to the LLM, and using it directly leads to disordered spatiotemporal relationships. Explicit temporal coordinate dimensions are needed to establish associations across per-frame spatial features. The key frame is assigned a temporal coordinate of 0, while preceding and following non-key frames receive symmetrically increasing or decreasing values.

3. **DynImg Composition Details**:

    - **Function**: Ensure patch-level feature extraction does not cross frame boundaries.
    - **Mechanism**: The downscaled size of non-key frames is controlled to be an integer multiple of the patch size, preventing any patch after concatenation from spanning two frames.
    - **Design Motivation**: Patches that cross frame boundaries mix pixel information from different frames, introducing noise into the features.

### Loss & Training

- Visual encoder: SigLip-so400m-384
- LLM: Qwen2.5-7B-Instruct, all parameters trainable
- Projection layer: feed-forward layer + PLLaVA adaptive average pooling module (pooling shape 16×12×12)
- Each video produces 4 DynImgs as input
- Training data: approximately 737K video–text pairs (TextVR 39K, YouCook2 8K, VideoChat 7K, WebVid 400K, Kinetics-710 40K, SSv2 40K, etc.)
- Training recipe follows PLLaVA

## Key Experimental Results

### Main Results

**Open-ended Video QA (average over 5 benchmarks):**

| Method | Encoder | LLM | MSVD Acc | MSRVTT Acc | ActivityNet Acc | TGIF Acc | Video-ChatGPT Avg |
|--------|---------|-----|----------|-----------|-----------------|----------|-------------------|
| PLLaVA | ViT-L | 7B | 76.6 | 62.0 | 56.3 | 77.5 | 3.12 |
| IG-VLM | Unk | GPT-4V | 76.3 | 63.8 | 57.0 | 65.3 | 3.17 |
| **DynImg** | **SigLip** | **7B** | **78.6** | **64.1** | **57.9** | **77.5** | **3.25** |

**MVBench multiple-choice QA (average over 20 categories): DynImg 55.8 vs. PLLaVA 46.6 vs. ST-LLM 54.9**

Notable gains on motion-related tasks: Moving Direction +21.0%, Moving Count +15.0%, Moving Attribute +26.5%.

### Ablation Study

**Ablation of key DynImg components (MSVD):**

| Temporal Prompts | Fusion Stage | 4D-RoPE | Acc | Score |
|-----------------|-------------|---------|-----|-------|
| ✗ | - | - | 74.9 | 3.8 |
| ✓ | After encoder | - | 75.2 | 3.9 |
| ✓ | Before encoder | ✗ | 77.3 | 4.0 |
| ✓ | Before encoder | ✓ | **78.6** | **4.2** |

**Ablation of number of non-key frames:**

| N-frames | 1 | 2 | 4 | 6 | 12 |
|----------|---|---|---|---|-----|
| Acc | 71.9 | 72.3 | **78.6** | 78.1 | 77.5 |

Four frames is optimal; too few fails to provide sufficient temporal variation, while too many reduces the resolution of each non-key frame to the point of information loss.

**Ablation of number of DynImgs: 4 DynImgs is optimal (78.6), vs. 77.0 for 1 and 77.7 for 16.**

### Key Findings

- **Before-encoder vs. after-encoder fusion**: Fusing prompts before the encoder (+2.4%) substantially outperforms after-encoder fusion, confirming the importance of pixel-level spatiotemporal interaction — prompts fused after the encoder merely supply additional information to the LLM without enabling genuine spatiotemporal interaction.
- **4D-RoPE is critical**: Accuracy improves from 77.3 to 78.6 (+1.3%), helping the LLM understand the compositional structure of DynImg.
- **Token efficiency**: DynImg requires only 576 visual tokens (4 frames), compared to 2,304 tokens (16 frames) for PLLaVA, achieving higher accuracy with a 75% reduction in token count.

## Highlights & Insights

- **Core insight of "earlier spatiotemporal interaction"**: Pushing interaction from high-level tokens down to low-level pixels avoids the loss of motion information during feature compression.
- **Elegant extension of visual prompting**: The progression from image-level visual prompts (bounding box annotations) to video-level temporal prompts (non-key frame concatenation) is both natural and novel.
- **Elegant 4D positional encoding design**: Learnable dimensions initialized to 0 enable gradual training without disrupting pretrained weights.
- **Efficiency and effectiveness combined**: Higher accuracy is achieved with fewer tokens, yielding significant practical deployment value.

## Limitations & Future Work

- MPEG-4 decoding increases data loading time (0.06s → 0.32s); while acceptable relative to training time, further optimization is needed.
- Only 4 DynImg inputs are explored; scalability to long-video scenarios remains unverified.
- The resolution of downscaled non-key frames is limited, and very fine-grained motion details may still be lost.
- Validation is conducted only on a 7B LLM; the effect of larger models is unknown.
- Key frame selection relies on MPEG-4 I-frames, which may not be the optimal content-aware selection strategy.

## Related Work & Insights

- **Video-ChatGPT**: Parallel spatiotemporal pooling architecture; spatiotemporal interaction occurs at the high-level token stage.
- **Video-LLaVA**: Unifies image and video inputs using an aligned projection strategy.
- **PLLaVA**: Proposes adaptive pooling to reduce token redundancy; serves as the primary baseline for this work.
- **IG-VLM**: Employs a comic-strip-style image grid and achieves improvements leveraging GPT-4V.
- **Follow-your-pose / AniPortrait**: Successful applications of visual prompts in image-level tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The concept of temporal visual prompts is novel and the 4D-RoPE design is elegant, though the underlying approach is an extension of visual prompting.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Five open-ended QA benchmarks, MVBench, and multi-dimensional ablations constitute a comprehensive evaluation.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is clearly articulated, figures are intuitive, and both quantitative and qualitative analyses are complete.
- **Value**: ⭐⭐⭐⭐ — Offers a new perspective on spatiotemporal interaction for video understanding and provides meaningful insights for MLLM video architecture design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] VideoMiner: Iteratively Grounding Key Frames of Hour-Long Videos via Tree-based Group Relative Policy Optimization](videominer_iteratively_grounding_key_frames_of_hour-long_videos_via_tree-based_g.md)
- [\[ICCV 2025\] AIM: Adaptive Inference of Multi-Modal LLMs via Token Merging and Pruning](aim_adaptive_inference_multimodal_llms_token_merging_pruning.md)
- [\[NeurIPS 2025\] MUVR: A Multi-Modal Untrimmed Video Retrieval Benchmark with Multi-Level Visual Correspondence](../../NeurIPS2025/video_understanding/muvr_a_multi-modal_untrimmed_video_retrieval_benchmark_with_multi-level_visual_c.md)
- [\[ICCV 2025\] 4D-Bench: Benchmarking Multi-Modal Large Language Models for 4D Object Understanding](4dbench_benchmarking_multimodal_large_language_models_for_4d.md)
- [\[ICCV 2025\] Multi-modal Multi-platform Person Re-Identification: Benchmark and Method](multi-modal_multi-platform_person_re-identification_benchmark_and_method.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Efficient Motion-Aware Video MLLM
description: >-
  [CVPR 2025][Multimodal VLM][Video Multimodal Large Language Model] This paper proposes EMA (Efficient Motion-Aware video MLLM), which utilizes the GOP structure in compressed videos to fuse spatial and motion information. By leveraging a native slow-fast architecture, it reduces redundancy while enhancing motion representation. Additionally, it introduces MotionBench as a motion understanding benchmark, achieving SOTA on various video QA and motion understanding tasks.
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Video Multimodal Large Language Model"
  - "Motion-Aware"
  - "Compressed Video"
  - "GOP Encoder"
  - "MotionBench"
date: 2026-05-08
content_hash: 9047dcb0986c3183
---

# Efficient Motion-Aware Video MLLM

**Conference**: CVPR 2025  
**arXiv**: [2503.13016](https://arxiv.org/abs/2503.13016)  
**Code**: None  
**Area**: Video Understanding / Multimodal VLM  
**Keywords**: Video Multimodal Large Language Model, Motion-Aware, Compressed Video, GOP Encoder, MotionBench

## TL;DR
This paper proposes EMA (Efficient Motion-Aware video MLLM), which utilizes the GOP structure in compressed videos to fuse spatial and motion information. By leveraging a native slow-fast architecture, it reduces redundancy while enhancing motion representation. Additionally, it introduces MotionBench as a motion understanding benchmark, achieving SOTA on various video QA and motion understanding tasks.

## Background & Motivation

**Background**: Most current video multimodal large language models (Video MLLMs) adopt a paradigm of uniform frame sampling combined with image-level encoders. Typical methods extract a set of frames at equal intervals from a video, extract frame features using an image encoder (such as CLIP-ViT), and feed them into the LLM. This strategy has achieved decent results in tasks such as video question answering and video description.

**Limitations of Prior Work**: The uniform frame sampling strategy suffers from two fundamental issues. The first is **low efficiency**: substantial visual redundancy exists between adjacent frames, and uniform sampling wastes computational resources on processing repetitive content. Second is **insufficient motion awareness**: image-level encoders cannot explicitly capture motion information between frames, leading to poor model performance in tasks requiring fine-grained understanding of object motion direction, speed, trajectory, etc.

**Key Challenge**: Intuitively, to improve motion understanding, denser frame sampling is required to capture motion details, but this dramatically increases computational overhead. How to enhance motion awareness while reducing computational cost is a key trade-off between efficiency and performance.

**Goal**: (1) Design an efficient video representation scheme that reduces redundancy while preserving motion information; (2) Establish a dedicated benchmark for evaluating motion understanding.

**Key Insight**: Compressed videos (such as H.264/H.265) naturally contain a GOP (Group of Pictures) structure, where I-frames are complete RGB frames, and P/B-frames store motion vectors and residuals. This inherently represents a "slow-fast" architecture—where a few RGB keyframes provide spatial information and many motion vectors provide temporal motion information, and motion vectors can be acquired with almost zero additional computational cost.

**Core Idea**: Leverage the GOP structure of compressed videos to design a motion-aware GOP encoder that fuses spatial features of sparse RGB frames with motion features of dense motion vectors within the GOP unit, creating compact and information-rich visual tokens.

## Method

### Overall Architecture
Input is a compressed video stream from which I-frames (RGB keyframes) and motion vectors are decoded. The spatial branch processes sparse RGB frames using an image encoder, while the motion branch processes dense sequences of motion vectors. The two are fused at the GOP level into unified visual tokens, which are then passed through a projection layer to the LLM for video understanding.

### Key Designs

1. **Motion-Aware GOP Encoder**:

    - **Function**: Fuses spatial and motion information within a GOP unit to generate compact visual tokens.
    - **Mechanism**: Each GOP consists of one I-frame and several P/B-frames. The I-frame's spatial features are extracted via an image encoder (such as ViT) to serve as the "slow" pathway, providing rich semantic and appearance information. The motion vectors in the P/B-frames are processed by a lightweight motion encoder to serve as the "fast" pathway, providing dense motion information. The two pathways are integrated at the GOP level through a fusion module. This design naturally borrows the concept of SlowFast networks, but the input comes directly from the compressed video stream without requiring additional optical flow computation.
    - **Design Motivation**: Motion vectors in compressed videos are "free" motion information already calculated during encoding. Compared to re-calculating optical flow from RGB frames (e.g., using RAFT), directly utilizing motion vectors saves tremendous computation. Meanwhile, the GOP structure provides natural temporal grouping units.

2. **Slow-Fast Native Input Architecture**:

    - **Function**: Uses fewer but spatially denser RGB frames + more but spatially sparser motion vectors to enhance motion representation while reducing redundancy.
    - **Mechanism**: The spatial path (Slow) samples fewer keyframes (I-frames), with each frame providing high-resolution spatial semantic information. The motion path (Fast) utilizes more motion vector frames, with each frame containing only pixel-level displacement information but at a higher temporal resolution. This asymmetric design makes the total number of tokens significantly smaller than uniform frame sampling methods, while paradoxically enriching the motion representation.
    - **Design Motivation**: Spatial appearance in videos changes slowly (requiring a low frame rate), whereas motion changes rapidly (requiring a high frame rate); thus, using different temporal resolutions for the two is the optimal strategy.

3. **MotionBench Motion Understanding Benchmark**:

    - **Function**: Evaluates the model's ability to understand different types of motion.
    - **Mechanism**: A video QA benchmark covering four motion types: linear motion, curved motion, rotational motion, and contact-based motion. Targeted questions are designed for each motion category, such as "Which direction is the ball moving?" or "How many degrees did the object rotate?".
    - **Design Motivation**: Existing video QA benchmarks (e.g., VideoQA, NExT-QA) focus more on scene understanding and causal reasoning, lacking fine-grained evaluation of motion itself. MotionBench fills this gap.

### Loss & Training
Standard video-language alignment training is adopted: first, pre-train the vision-language projection layer on large-scale video-text data, then perform instruction tuning on downstream video QA datasets. During training, the image encoder and LLM are frozen, and only the GOP encoder and projection layer are updated.

## Key Experimental Results

### Main Results

| Benchmark | Metric | EMA | Prev. SOTA | Gain |
|------|------|------|----------|------|
| MotionBench | Overall Accuracy | SOTA | — | First-of-its-kind benchmark |
| Common Video QA | Accuracy | SOTA | Runner-up | Significant Gain |
| Long Video Understanding | Accuracy | Competitive | — | Scalability Verified |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Full EMA | Best | Complete model (RGB + motion vector fusion) |
| w/o Motion Vectors | Dropped | Using RGB frames only; motion understanding significantly degrades |
| w/o GOP Fusion | Dropped | Simple concatenation instead of in-GOP fusion; performance degrades |
| Uniform Sampling Baseline | Lowest | Traditional equal-interval sampling; high redundancy and lacks motion information |

### Key Findings
- The introduction of motion vectors improves performance across all four motion types on MotionBench, especially for rotational and contact-based motion.
- The inference cost of EMA is lower than methods that uniformly sample more frames, as the overhead of processing motion vectors is much smaller than that of RGB frames.
- EMA also performs excellently on long video understanding benchmarks, indicating good scalability of the GOP structure.
- The ratio of RGB frames to motion vector frames in the Slow-Fast architecture affects performance, requiring a careful balance.

## Highlights & Insights
- **Compressed video is a natural Slow-Fast architecture**: The structural design of I/P/B frames in GOP was originally intended to trade off bandwidth and quality, which happens to be the optimal balance between efficiency and information density in video understanding. Utilizing compressed-domain information directly instead of decoding and then re-encoding is a clever and practical idea.
- **"Free" motion information**: Motion vectors are already calculated during video encoding and decoding, eliminating the need for extra optical flow networks. This approach of leveraging existing information can be transferred to other video tasks requiring motion information.
- **Contribution of MotionBench**: The systematic evaluation of motion understanding fills a gap in existing benchmarks and helps drive progress in motion understanding for video MLLMs.

## Limitations & Future Work
- **Dependency on compression formats**: The method relies on the GOP structure of specific compression formats like H.264/H.265, requiring adaptation for uncompressed videos or other encoding formats.
- **Motion vector quality**: Motion vectors in compressed videos are optimized for encoding efficiency, not for accurate motion estimation; they may be inaccurate in scenarios with fast motion or occlusions.
- **Scale of MotionBench**: As a new benchmark, its scale and diversity may not be large enough yet, requiring collaborative development from the community.
- **Unexplored audio**: Many aspects of motion understanding (such as contact judgment) can leverage audio information, which is not covered in this work's multimodal fusion.

## Related Work & Insights
- **vs VideoChat / Video-LLaVA**: Traditional Video MLLMs rely on uniform frame sampling and image encoders, lacking explicit motion modeling; EMA explicitly introduces motion information through compressed-domain motion vectors.
- **vs SlowFast Networks**: SlowFast uses a dual-pathway design with different frame sampling rates in the RGB domain; EMA naturally achieves slow-fast in the compressed domain, using I-frames and motion vectors to handle spatial and motion roles, respectively.
- **vs CoDeF / MotionFormer**: These methods extract motion representations by learning from videos; EMA directly utilizes motion vectors from the compressed stream, avoiding additional motion estimation computation.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of utilizing compressed video GOP structure as native slow-fast input is creative, although using compressed-domain motion vectors for video understanding is not entirely unprecedented.
- Experimental Thoroughness: ⭐⭐⭐⭐ Compared with various methods across multiple benchmarks, while introducing a new motion understanding benchmark, MotionBench.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, intuitive method description, and well-organized design of MotionBench.
- Value: ⭐⭐⭐⭐ Provides a practical solution and evaluation tool for motion understanding in video MLLMs, offering high practical value.
- Overall: ⭐⭐⭐⭐ Cleverly connects low-level compressed video structures with high-level video understanding, balancing engineering practicality with academic contribution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] V-Stylist: Video Stylization via Collaboration and Reflection of MLLM Agents](v-stylist_video_stylization_via_collaboration_and_reflection_of_mllm_agents.md)
- [\[CVPR 2026\] MSJoE: Jointly Evolving MLLM and Sampler for Efficient Long-Form Video Understanding](../../CVPR2026/multimodal_vlm/msjoe_jointly_evolving_mllm_and_sampler_for_efficient_long-form_video_understand.md)
- [\[NeurIPS 2025\] In the Eye of MLLM: Benchmarking Egocentric Video Intent Understanding with Gaze-Guided Prompting](../../NeurIPS2025/multimodal_vlm/in_the_eye_of_mllm_benchmarking_egocentric_video_intent_understanding_with_gaze-.md)
- [\[CVPR 2025\] Video-XL: Extra-Long Vision Language Model for Hour-Scale Video Understanding](video-xl_extra-long_vision_language_model_for_hour-scale_video_understanding.md)
- [\[CVPR 2026\] ReMoRa: Multimodal Large Language Model based on Refined Motion Representation for Long-Video Understanding](../../CVPR2026/multimodal_vlm/remora_multimodal_large_language_model_based_on_refined_motion_representation_fo.md)

</div>

<!-- RELATED:END -->

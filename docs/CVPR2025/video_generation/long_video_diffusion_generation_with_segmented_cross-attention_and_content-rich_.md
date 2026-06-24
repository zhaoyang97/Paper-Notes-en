---
title: >-
  [Paper Note] Presto: Long Video Diffusion Generation with Segmented Cross-Attention and Content-Rich Video Data Curation
description: >-
  [CVPR 2025][Video Generation][Long Video Generation] Presto proposes a Segmented Cross-Attention (SCA) strategy, which segments latent states along the temporal dimension and performs cross-attention with corresponding sub-captions respectively. Combined with a meticulously curated 261K high-quality long video dataset, LongTake-HD, it enables the generation of 15-second long-range coherent videos with rich content, achieving a Semantic Score of 78.5% and a Dynamic Degree of 1…
tags:
  - "CVPR 2025"
  - "Video Generation"
  - "Long Video Generation"
  - "Segmented Cross-Attention"
  - "Video Data Curation"
  - "DiT"
date: 2026-05-08
content_hash: 037592c2e9d220fb
---

# Presto: Long Video Diffusion Generation with Segmented Cross-Attention and Content-Rich Video Data Curation

**Conference**: CVPR 2025  
**arXiv**: [2412.01316](https://arxiv.org/abs/2412.01316)  
**Code**: [https://presto-video.github.io](https://presto-video.github.io) (Project Page)  
**Area**: Diffusion Models / Video Generation  
**Keywords**: Long Video Generation, Segmented Cross-Attention, Video Data Curation, DiT

## TL;DR

Presto proposes a Segmented Cross-Attention (SCA) strategy, which segments latent states along the temporal dimension and performs cross-attention with corresponding sub-captions respectively. Combined with a meticulously curated 261K high-quality long video dataset, LongTake-HD, it enables the generation of 15-second long-range coherent videos with rich content, achieving a Semantic Score of 78.5% and a Dynamic Degree of 100% on VBench.

## Background & Motivation

**Background**: Current video diffusion models mainly focus on generating short video clips of 3-8 seconds, which severely restricts content representation and richness. Early methods for extending video duration expanded short clips via interpolation/extrapolation, but the content diversity remained constrained by the capacity of the original short clips.

**Limitations of Prior Work**: Autoregressive methods (e.g., using extra modules to extend videos step-by-step) suffer from error propagation; although concatenating multiple texts can improve content diversity, the transitions between different scenes are abrupt. Existing long-video approaches overlook the importance of high-quality data, leading to low consistency and limited content diversity in the generated videos.

**Key Challenge**: Long video generation requires a balance between content diversity and long-range coherence—a single text input lacks sufficient information to describe rich scene changes in long videos, while existing text encoders suffer from information loss due to length truncation.

**Goal**: (1) How to simultaneously handle multiple progressive text conditions within the DiT architecture? (2) How to construct a high-quality long video training dataset?

**Key Insight**: Splitting the textual description of a long video into multiple progressive sub-captions allows different temporal segments of the model to attend to their corresponding text conditions, thus ensuring content richness while maintaining temporal consistency.

**Core Idea**: Segmenting the latent states along the temporal dimension and applying segmented cross-attention with progressive sub-captions, paired with a meticulously curated long video dataset, to achieve rich and coherent long video generation.

## Method

### Overall Architecture

Presto is built upon Allegro (a 2.8B parameter DiT model). On the input side, each video is equipped with a global caption and 5 progressive sub-captions, which are encoded using a T5 encoder to obtain 5 sets of text embeddings. In the cross-attention layers of DiT, the latent state is divided into 5 segments along the temporal dimension, and each segment computes cross-attention with its corresponding sub-caption. The training process consists of two stages: pre-training (261K data, 1500 steps) and fine-tuning (47K curated data, 500 steps). During inference, the user inputs a single prompt, and GPT-4o acts as a "director" to generate 5 progressive sub-captions.

### Key Designs

1. **Segmented Cross-Attention (SCA)**:

    - **Function**: To make different temporal segments of the latent states attend to their corresponding text conditions.
    - **Mechanism**: The $N$ sub-captions are encoded into $\{c_i\}_{i=1}^N$, and the latent state $z$ is equally divided into $N$ segments $\{z_i\}_{i=1}^N$ along the temporal dimension. Each segment $z_i$ only performs cross-attention with the corresponding $c_i$. The authors explore three variants: ISCA (independently segmented), SSCA (sequentially segmented), and OSCA (overlappingly segmented). Ultimately, OSCA is adopted, which introduces a $\delta$-frame overlap at the boundaries of adjacent segments, averaging the attention outputs of the overlapping region to facilitate smooth transitions between segments. SCA does not introduce any additional parameters.
    - **Design Motivation**: A single long text embedding is prone to truncation, leading to information loss, while letting all frames attend to the entire text blurs details. The segmentation strategy is conceptually similar to window attention; it maintains local textual precision while enabling global information exchange through self-attention.

2. **LongTake-HD Data Curation**:

    - **Function**: To provide high-quality long video-multitext paired training data.
    - **Mechanism**: Starting from 8.9 million public videos, the dataset is curated through duration/fps/resolution filtering $\to$ scene detection (PySceneDetect) $\to$ low-level attribute filtering (brightness, watermark) $\to$ aesthetic and motion content filtering (LAION Aesthetics + optical flow), resulting in 261K single-shot videos. Aria is used to generate captions for the videos and keyframes, which are further refined causally by GPT-4o to generate 5 progressive sub-captions (including camera movement information).
    - **Design Motivation**: Existing video datasets contain significant noise and low-quality content, and lack multi-segment text descriptions tailored for long videos. High-quality data is crucial for long video generation.

3. **Progressive Sub-caption Generation Strategy**:

    - **Function**: To generate coherent, non-redundant multi-segment narrative descriptions for each long video.
    - **Mechanism**: The video is divided into $N$ segments to generate independent descriptions, which are then refined causally by an LLM segment-by-segment—the description for the $i$-th segment is generated by referencing all prior sub-captions and the global caption, ensuring each segment represents an independent event in the storyline and explicitly incorporates camera movement descriptions. This "narrative-style" annotation eliminates redundant descriptions between segments.
    - **Design Motivation**: Traditional multi-text methods violently combine multiple unrelated descriptions (e.g., TALC), leading to redundancy and a lack of narrative coherence.

### Loss & Training

The standard diffusion model training loss is adopted. The pre-training stage is conducted on 64 H100 GPUs with a batch size of 256 and a learning rate of 1e-4 for 1500 steps (processing 384K videos); the fine-tuning stage is trained on 47K curated videos for 500 steps. Post-processing utilizes EMA-VFI for frame interpolation to further extend video length and normalize speed.

## Key Experimental Results

### Main Results

| Method | Semantic Score | Dynamic Degree | Overall Score |
|------|---------------|----------------|---------------|
| Gen-3 (Commercial) | 75.2 | 60.1 | 82.3 |
| Allegro (Open Source SOTA) | 73.0 | 55.0 | 81.1 |
| TALC (MT2V) | 44.4 | 98.6 | 58.9 |
| **Presto (Ours)** | **78.5** | **100.0** | 80.2 |

User study (win rate %):

| Baseline | Overall Win | Diversity Win | Coherence Win | Text-Video Win |
|----------|-------------|---------------|---------------|----------------|
| vs Gen-3 | 45.0 | 59.1 | 35.1 | 40.9 |
| vs Allegro | 54.9 | 68.0 | 45.1 | 51.4 |
| vs TALC | 91.8 | 95.3 | 89.5 | — |

### Ablation Study

| Configuration | Overall Score | Dynamic Degree | Consistency |
|------|--------------|----------------|-------------|
| OSCA (Full) | 74.7 | 100.0 | 25.29 |
| SSCA | 73.7 ↓ | 100.0 | 25.06 ↓ |
| ISCA | 73.1 ↓ | 100.0 | 24.88 ↓ |
| w/o Fine Data Filtering | 72.0 ↓ | 97.2 ↓ | 24.06 ↓ |
| Single Long Text Condition | 71.8 ↓ | 100.0 | 24.06 ↓ |

### Key Findings

- OSCA is the optimal among the three SCA strategies; the overlap design facilitates smooth transitions between segments, balancing both content richness and coherence.
- Data quality has a significant impact on long video generation—removing fine-grained filtering drops the Overall Score by 2.7%.
- Progressive sub-captions vs. single long text concatenation: the latter results in a 2.9% drop in the Overall Score, proving that segmented text modeling is more effective than simple concatenation.
- Reaching 100% in Dynamic Degree demonstrates that SCA is exceptionally strong at capturing motion dynamics.

## Highlights & Insights

- **Zero extra parameters for SCA**: Segmented Cross-Attention introduces no new parameters or modules and can be seamlessly integrated into any DiT-based architecture, which minimizes transfer costs. The core concept is similar to window attention but applied to the temporal dimension of cross-attention, making it simple and effective.
- **Progressive Narrative Descriptions**: Traditional "multi-text concatenation" is upgraded to causal progressive descriptions, eliminating redundancy while maintaining narrative coherence. This concept can be transferred to any generation tasks that require segmented control (e.g., long document generation, long audio synthesis, etc.).
- **Systematized Data Curation Pipeline**: Starting from 8.9 million videos and filtering down to 261K high-quality samples, the authors establish a complete multi-level filtering and multi-modal annotation pipeline, which holds significant reference value for the community.

## Limitations & Future Work

- The Quality Score drops under highly dynamic and complex scenes, indicating an ongoing trade-off between rich content and visual quality.
- Relying on GPT-4o to generate sub-captions during inference increases latency and costs.
- Only 15-second videos (88 frames + interpolation) were evaluated; the performance on even longer videos (e.g., minute-level) remains unknown.
- The fixed 5-segment partition may not suit all video content, making adaptive segmenting strategies worth exploring.

## Related Work & Insights

- **vs TALC**: TALC also uses multi-text inputs but adopts a rigid combination of multiple scenes, with substantial redundancy between sub-captions. Presto's progressive descriptions eliminate redundancy and reinforce narrative coherence, outperforming it by 34.1% in Semantic Score.
- **vs Gen-L-Video / FreeNoise**: These methods extend video length via noise scheduling or sliding-window attention but are constrained by the content capacity of the original short clips. Presto addresses the problem from both the model architecture and data aspects simultaneously.
- **vs Allegro**: Presto is built directly on Allegro. It achieves significant improvements solely by modifying the cross-attention and utilizing curated data, demonstrating great compatibility.

## Rating

- Novelty: ⭐⭐⭐⭐ The SCA concept is simple but not revolutionary, while the progressive descriptions are novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ VBench quantitative evaluations + large-scale user studies + detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with sound method descriptions.
- Value: ⭐⭐⭐⭐ The dataset and methodology hold practical value for the long video generation community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Radial Attention: O(n log n) Sparse Attention with Energy Decay for Long Video Generation](../../NeurIPS2025/video_generation/radial_attention_onlog_n_sparse_attention_with_energy_decay_for_long_video_gener.md)
- [\[CVPR 2025\] VideoGigaGAN: Towards Detail-rich Video Super-Resolution](videogigagan_towards_detail-rich_video_super-resolution.md)
- [\[CVPR 2025\] MovieBench: A Hierarchical Movie Level Dataset for Long Video Generation](moviebench_a_hierarchical_movie_level_dataset_for_long_video_generation.md)
- [\[CVPR 2025\] LongDiff: Training-Free Long Video Generation in One Go](longdiff_training-free_long_video_generation_in_one_go.md)
- [\[CVPR 2025\] StreamingT2V: Consistent, Dynamic, and Extendable Long Video Generation from Text](streamingt2v_consistent_dynamic_and_extendable_long_video_generation_from_text.md)

</div>

<!-- RELATED:END -->

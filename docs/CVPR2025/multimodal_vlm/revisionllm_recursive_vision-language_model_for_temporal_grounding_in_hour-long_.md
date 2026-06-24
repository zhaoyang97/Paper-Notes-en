---
title: >-
  [Paper Note] ReVisionLLM: Recursive Vision-Language Model for Temporal Grounding in Hour-Long Videos
description: >-
  [CVPR 2025][Multimodal VLM][Long-form Video Temporal Grounding] This paper proposes ReVisionLLM, the first vision-language model capable of temporal grounding in hour-long videos. It mimics human search strategies to recursively process videos by first coarsely localizing relevant segments and progressively refining them to precise temporal boundaries, outperforming the state-of-the-art on the MAD dataset by +2.6% in R1@0.1.
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Long-form Video Temporal Grounding"
  - "Recursive Vision-Language Model"
  - "Hierarchical Adapter"
  - "Progressive Training"
  - "Video Understanding"
date: 2026-05-08
content_hash: 2ace5c69ace3e042
---

# ReVisionLLM: Recursive Vision-Language Model for Temporal Grounding in Hour-Long Videos

**Conference**: CVPR 2025  
**arXiv**: [2411.14901](https://arxiv.org/abs/2411.14901)  
**Code**: [https://github.com/Tanveer81/ReVisionLLM](https://github.com/Tanveer81/ReVisionLLM)  
**Area**: Multimodal VLM  
**Keywords**: Long-form Video Temporal Grounding, Recursive Vision-Language Model, Hierarchical Adapter, Progressive Training, Video Understanding

## TL;DR

This paper proposes ReVisionLLM, the first vision-language model capable of temporal grounding in hour-long videos. It mimics human search strategies to recursively process videos by first coarsely localizing relevant segments and progressively refining them to precise temporal boundaries, outperforming the state-of-the-art on the MAD dataset by +2.6% in R1@0.1.

## Background & Motivation

Long-form video temporal grounding aims to localize the start and end times of an event in a long video based on a textual query. Existing methods face three major challenges:

1. **Limited Frame Capacity of VLMs**: Existing VLMs (e.g., VTimeLLM) process videos by uniformly sampling a fixed number of frames. For hour-long videos (e.g., a 2-hour movie), uniformly sampling 100 frames means an interval of 72 seconds between frames, leading to the complete loss of temporal details. In experiments, VTimeLLM scores **0** across all metrics on the MAD dataset.
2. **Training Resource Bottlenecks**: Directly training VLMs on hour-long videos requires massive GPU memory and computational power.
3. **Confidence Calibration Issue**: Visual predictions from VLMs are often overconfident (high-confidence false positives), a problem that is amplified in long videos. There is a critical need to distinguish the actual target event from a vast number of irrelevant segments.
4. **Limitations of Non-LLM Methods**: Non-LLM approaches such as CONE, SOONet, and RGNet rely on multiple networks and complex post-processing. They lack flexibility and cannot handle free-form textual queries.

## Method

### Overall Architecture

ReVisionLLM consists of three core components: (1) a multimodal encoder (CLIP ViT-L/14 to extract frame-level CLS features); (2) a Hierarchical Adapter (generating both dense and sparse temporal features); and (3) an LLM (Vicuna-7B acting as a temporal grounding decoder to recursively predict event boundaries). The model recursively processes the video: the top layer uses sparse features to scan the entire video and identify candidate segments, and the model progressively moves down to the bottom layer using dense features to precisely ground the event boundaries.

### Key Designs

1. **Hierarchical Adapter**:
    - **Function**: Transforms video frame features into multi-granularity temporal representations—bottom-layer dense features for precise localization, and top-layer sparse features for efficient scanning.
    - **Mechanism**: First, frame features $\mathcal{F}$ are partitioned into sliding window segments $C = [C^i]_{i=1,...,|C|}$, where each segment $C^i \in \mathbb{R}^{L_w \times D}$. **Dense features** $\mathcal{D}^i = h_d(C^i)$ preserve the original temporal resolution via linear projection. The generation of **sparse features** involves two steps: (a) cross-attention using segment features as queries and text features as keys to produce text-aligned features $\tilde{C}^i = \text{Cross-Attention}(C^i, Q)$; (b) self-attention compressing $\tilde{C}^i$ into a single vector $\mathcal{S}^i = A_0$, where $A = \text{Self-Attention}([\mathcal{S}^i; \tilde{C}^i])$.
    - **Design Motivation**: For hour-long videos containing tens of thousands of frames, feeding all dense features into the LLM would far exceed its context length limit. Sparse features compress a segment of several minutes into a single 768-dimensional vector, substantially reducing active input sequence length. Cross-attention aligns sparse features with textual queries, making the top-layer scanning more targeted.

2. **Recursive Video Grounding**:
    - **Function**: Mimics human search strategies to narrow down the search space progressively from global to local levels.
    - **Mechanism**: Constructs $L$-level hierarchical video inputs $[I^{(\ell)}]_{\ell=1,...,L}$. The lowest layer $I^{(1)}$ uses dense features $\mathcal{D}$, while upper layers $[I^{(\ell)}]_{\ell=2,...,L}$ utilize sparse features $\mathcal{S}$. At each layer $l$, the LLM receives the input $P^{(l)} = [I^{(l)}, w_1,...,w_M]$ and predicts the temporal boundary $\tau^{(l)}$ for that layer, outputting in the format of "From $s$ to $e$." or "Not Present.". Lower layers use the boundaries $\tau^{(<l)}$ predicted by upper layers as priors to progressively refine the predictions. The training objective is: $p(T^{(l)}|P^{(l)}) = \prod_{k=1}^{K} p(T_k^{(\ell)} | T_{<k}^{(\ell)}, P^{(l)})$.
    - **Design Motivation**: Directly localizing a fine-grained event (e.g., a 4-second clip in a 2-hour movie) across the entire video is like looking for a needle in a haystack. The recursive "zoom-in" strategy ensures that each layer only needs to make decisions within a restricted range, dramatically simplifying the task. Meanwhile, controlling the input token budget per layer addresses GPU memory bottlenecks for long-form videos.

3. **Progressive Training**:
    - **Function**: Train the model in stages to handle data scale and efficiency issues associated with long videos.
    - **Mechanism**: Divided into two phases—**Phase 1 (Short-segment training)**: (a) First, train the LLM (using LoRA) with dense features to learn precise boundary predictions ("From $s$ to $e$."); a key innovation is introducing **contrastive segments**—video segments without the target event, where the model is required to output "Not Present." to calibrate confidence. (b) Freeze the LLM and train the Hierarchical Adapter using a simplified binary classification ("Yes/No") to generate sparse features. **Phase 2 (Long-form video training)**: Utilize the sparse features learned in Phase 1 to train LoRA on hour-long videos to locate segments containing the event.
    - **Design Motivation**: (a) Contrastive segments solve the overconfidence problem in VLMs—traditional VLMs are only trained on relevant clips containing target events and cannot identify event absence, leading to overconfident predictions across all segments; (b) the coarse-to-fine sequence allows the model to first master basic temporal boundary identification before transferring the capability to long-form video settings.

### Loss & Training

- Standard autoregressive language modeling training loss (next-token prediction).
- Inference uses **Calibrated Confidence Ranking**: Calculates the entropy of the probability distribution for each predicted word from the LLM: $H_k^{(i)} = -\sum_w p(w|T_{<k}, \mathcal{D}^{(i)}) \log p(w|T_{<k}, \mathcal{D}^{(i)})$. The confidence score is computed as the reciprocal of the average entropy: $R^i = \frac{1}{\frac{1}{K}\sum_{k=1}^{K} H_k^i}$. Low entropy (high confidence) predictions are ranked higher.
- Optimizer: AdamW + cosine decay; LoRA parameters $r=64, \alpha=128$.
- Phase 1 is trained for 1 epoch ($lr=1e-3$), and Phase 2 is trained for 5 epochs (MAD) or 1 epoch (VidChapters-7M) with $lr=1e-4$.

## Key Experimental Results

### Main Results

MAD Dataset (Hour-long movies, average 110 minutes, event average 4.1 seconds):

| Method | R1@0.1 | R5@0.1 | R1@0.3 | R5@0.3 | Avg. |
|------|--------|--------|--------|--------|------|
| RGNet | 12.4 | 25.1 | 9.5 | 18.7 | 13.7 |
| SnAG | 10.3 | 24.4 | 8.5 | 20.6 | 13.8 |
| VTimeLLM+CONE | 1.4 | 3.1 | 1.3 | 2.5 | 1.7 |
| **ReVisionLLM** | **15.0** | **25.1** | **11.0** | **18.8** | **14.4** |
| **ReVisionLLM-I** | **17.3** | **31.4** | **12.7** | **23.5** | **17.5** |

VidChapters-7M Dataset (YouTube videos, up to 12 hours long):

| Method | R1@0.3 | R1@0.5 | R1@0.7 | R1@0.9 | Avg. |
|------|--------|--------|--------|--------|------|
| M-DETR | 37.4 | 27.3 | 17.6 | 6.4 | 22.1 |
| **ReVisionLLM** | **33.8** | **27.4** | **21.8** | **15.2** | **24.6** |

### Ablation Study

Cumulative Ablation (MAD Dataset):

| Module | R1@0.1 | R5@0.1 | Description |
|------|--------|--------|------|
| Baseline (VTimeLLM) | 0.0 | 0.0 | Uniform sampling of 100 frames, completely failed |
| +CONE Ranking | 1.4 | 2.4 | Short segments + CLIP ranking |
| +Contrastive Segments | 4.8 | 6.7 | Learned to identify event absence |
| +Calibrated Confidence | 8.4 | 12.7 | LLM internal confidence instead of CLIP |
| +Recursive Processing | 15.0 | 25.1 | The most significant source of improvement |

Number of Layers Ablation:

| Number of Layers | R1@0.1 | R5@0.1 | R1@0.3 | R5@0.3 |
|--------|--------|--------|--------|--------|
| 0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 1 | 8.4 | 12.7 | 6.6 | 8.9 |
| 2 | 11.9 | 17.5 | 8.7 | 13.2 |
| 3 | 15.0 | 25.1 | 11.0 | 18.8 |

### Key Findings

- **Recursive processing is the most critical module**: It contributes to an improvement from 8.4% to 15.0% (R1@0.1), almost doubling performance.
- **Contrastive segments effectively calibrate confidence**: Their introduction boosts R1@0.1 from 1.4% to 4.8%, delivering the first meaningful performance of a VLM on long-form video tasks.
- **ReVisionLLM outperforms the baseline (VTimeLLM+CONE) while processing only 57% of the frames**, demonstrating the efficiency advantage of the recursive strategy.
- **Robust to video length**: Performance drops only marginally when scaling up to 10-hour videos, whereas non-recursive methods completely fail.
- Achieves competitive performance on text-to-video retrieval (MSRVTT) as well, demonstrating that the model learns generalized video-text correspondence.

## Highlights & Insights

- **Biomimetic design mimicking human search**: Recursive zoom-in perfectly corresponds to cognitive science findings on human visual search—scanning coarsely to build a target representation before progressively focusing on local targets.
- **Contrastive segments address a fundamental flaw of VLMs**: The overconfidence issue is less obvious in short video tasks but becomes drastically amplified in long-form videos. Contrastive training is a simple yet crucial design.
- **Highly efficient sparse feature compression**: Compresses minutes of video into a single vector, preventing hour-long videos from exceeding the LLM's context window limit.
- **Confidence ranking using LLM entropy** is more effective than CLIP similarity-based ranking because the LLM confidence has been calibrated via contrastive training.

## Limitations & Future Work

- Sparse feature compression inevitably discards fine-grained detail, which may lead to missing subtle events.
- The number of layers is fixed at 3; adaptive layer selection could be more optimal.
- Currently, only frame-level CLS global features are used, ignoring spatial information within frames.
- The LLM backbone is Vicuna-7B; upgrading to a stronger LLM may yield further improvements.
- The ReVisionLLM-I variant is more accurate but requires processing all frames, resulting in lower efficiency.

## Related Work & Insights

- Complementary to non-LLM methods like CONE and SnAG: ReVisionLLM excels in natural language interaction and confidence calibration.
- The recursive processing approach can be extended to other long-form sequence understanding tasks (such as long document retrieval and long audio event detection).
- The hierarchical design of sparse-dense features echoes the coarse-to-fine strategy utilized in image processing.
- The contrastive segment training strategy can inspire other VLMs to address the overconfidence problem.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The first VLM to address temporal grounding in hour-long videos with an elegantly designed recursive architecture.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Evaluated on MAD and VidChapters-7M with cumulative ablations, variant comparisons, length robustness analysis, and generalization experiments.
- **Writing Quality**: ⭐⭐⭐⭐ Logical flow is clear throughout from problem formulation to methodology and experiments.
- **Value**: ⭐⭐⭐⭐⭐ Initiates a new direction for VLMs in long-form temporal video grounding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Video-XL: Extra-Long Vision Language Model for Hour-Scale Video Understanding](video-xl_extra-long_vision_language_model_for_hour-scale_video_understanding.md)
- [\[CVPR 2025\] VideoGLaMM: A Large Multimodal Model for Pixel-Level Visual Grounding in Videos](videoglamm_a_large_multimodal_model_for_pixel-level_visual_grounding_in_videos.md)
- [\[CVPR 2025\] VidComposition: Can MLLMs Analyze Compositions in Compiled Videos?](vidcomposition_can_mllms_analyze_compositions_in_compiled_videos.md)
- [\[CVPR 2025\] Your Large Vision-Language Model Only Needs a Few Attention Heads for Visual Grounding](your_large_vision-language_model_only_needs_a_few_attention_heads_for_visual_gro.md)
- [\[CVPR 2025\] MIMO: A Medical Vision Language Model with Visual Referring Multimodal Input and Pixel Grounding Multimodal Output](mimo_a_medical_vision_language_model_with_visual_referring_multimodal_input_and_.md)

</div>

<!-- RELATED:END -->

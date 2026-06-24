---
title: >-
  [Paper Note] ReWind: Understanding Long Videos with Instructed Learnable Memory
description: >-
  [CVPR 2025][Video Understanding][Long Video Understanding] This paper proposes ReWind, a vision-language model architecture based on a learnable memory module. Through a novel read-perceive-write loop mechanism and instruction-guided dynamic frame selection, it significantly outperforms previous methods on long video VQA and temporal localization tasks while using fewer tokens and frames.
tags:
  - "CVPR 2025"
  - "Video Understanding"
  - "Long Video Understanding"
  - "Learnable Memory"
  - "Visual-Language Models"
  - "Dynamic Frame Selection"
  - "Video Question Answering"
date: 2026-05-08
content_hash: 882e5a2310ef2880
---

# ReWind: Understanding Long Videos with Instructed Learnable Memory

**Conference**: CVPR 2025  
**arXiv**: [2411.15556](https://arxiv.org/abs/2411.15556)  
**Code**: None  
**Area**: Video Understanding / Multimodal VLM  
**Keywords**: Long Video Understanding, Learnable Memory, Visual-Language Models, Dynamic Frame Selection, Video Question Answering

## TL;DR

This paper proposes ReWind, a vision-language model architecture based on a learnable memory module. Through a novel read-perceive-write loop mechanism and instruction-guided dynamic frame selection, it significantly outperforms previous methods on long video VQA and temporal localization tasks while using fewer tokens and frames.

## Background & Motivation

**Background**: Vision-Language Models (VLMs) have made significant progress in the field of multimodal understanding, demonstrating the capacity to integrate textual and visual information for tasks such as question-answering and description generation. However, existing VLMs face severe challenges when processing long videos (over 10 minutes), primarily due to the quadratic computational complexity of self-attention mechanisms, memory limitations, and the difficulty of maintaining coherent temporal understanding over long sequences.

**Limitations of Prior Work**: (1) **Excessive Information Compression**—Methods like MovieChat and MA-LMM use FIFO queues or hierarchical memory modules, which severely compress temporal information, sacrificing accurate understanding of event dynamics; (2) **Isolated Frame Processing**—Methods such as LLaMA-VID and VTimeLLM process each frame independently, failing to capture coherent temporal representations across frames; (3) **Fixed-Density Representation**—Existing methods maintain the same spatial representation density for all frames, storing unnecessary details for non-key frames and wasting memory resources.

**Key Challenge**: Long video understanding requires resolving two competing demands simultaneously: (a) efficiently compressing and storing temporal information of the entire video, and (b) retaining sufficient spatial details for key frames. However, existing methods either over-compress and lose details, or uniformly retain data leading to memory explosion.

**Goal**: Design a memory-driven long video understanding framework capable of selectively storing and retrieving video information based on user instructions, achieving high-quality video question answering and temporal localization with low memory overhead.

**Key Insight**: The authors propose a "coarse-to-fine" two-stage processing strategy: the first stage progressively compresses video information via a memory module (representing each frame with a small number of tokens), and the second stage "rewinds" the video to select key frames relevant to the instructions and retain high-resolution spatial details.

**Core Idea**: Progressively construct instruction-aware compressed video representations using a memory module with a read-perceive-write loop, then "rewind" to key moments to supply spatial details via instruction-guided dynamic frame selection.

## Method

ReWind divides long video understanding into two stages. In Stage-1, the video is split into sub-clips and processed segment-by-segment. After visual features are extracted by a vision encoder, they are encoded and stored in an instruction-guided manner via a Perceiver layer equipped with a memory module. In Stage-2, based on the memory content and the user instruction, key frames are dynamically selected to preserve high-resolution spatial details. Finally, the memory content and the selected frames are fed together into an LLM to generate the answer.

### Overall Architecture

The input is an entire long video V (e.g., 10 minutes, sampled at 1fps to yield ~600 frames) and a user's text instruction. The video is first split into N sub-clips, each containing F frames. In Stage-1, ViT features are extracted frame-by-frame for each sub-clip, updating the memory bank M through a read-perceive-write loop (storing only 2 tokens per frame). In Stage-2, based on the content of M and the instruction encoding, a two-stage selection mechanism (instruction relevance ranking + KNN density peak clustering) selects 8 key frames, retrieving their high-resolution representations (32 tokens per frame) from a feature buffer. Finally, the content of M, key frame features, and the instruction are concatenated and fed into LLaMA-2 7B (fine-tuned with LoRA) to generate the answer.

### Key Designs

1. **Read-Perceive-Write memory loop**:

    - Function: Progressively construct instruction-aware compressed video representations
    - Mechanism: A three-step loop—(a) **Read**: Use $N_R=32$ learnable read queries $Q_R$ via cross-attention to retrieve historical context from the memory bank M, obtaining a summary of the current memory; (b) **Perceive**: Treat the read queries (carrying historical information) as the initial queries for the Perceiver, performing cross-attention with the current frame's ViT features and the instruction text encoding to produce an instruction-aware frame-level representation $\hat{Q}_{ij}$, followed by self-attention along the temporal dimension to capture intra-clip temporal relationships; (c) **Write**: Use 2 learnable write queries $Q_W$ via cross-attention to distill the Perceiver output into an ultra-compact representation of 2 tokens per frame, which is stored in M chronologically.
    - Design Motivation: Unlike prior Q-Former methods (e.g., Video-LLaMA), ReWind's Perceiver processes frames independently at the frame level before applying temporal attention, preserving temporal fidelity rather than producing clip-level compressed representations. The read operation ensures each encoding step "knows" what was previously stored, and the write operation enables highly efficient storage using minimal tokens.

2. **Dynamic Frame Selection (DFS)**:

    - Function: Select key frames most relevant to the instruction from the entire video to supply high-resolution spatial details
    - Mechanism: Two-stage selection—(a) **Instruction Relevance Selection**: Calculate attention scores between the instruction encoding $\bar{I}$ and each frame representation in memory bank M, selecting the top $L=64$ frames; (b) **Density Peak Clustering**: Apply KNN-based Density Peak Clustering (DPC-KNN) to these L frames to select the $K_c=8$ most representative frames. For each frame, local density $\sigma_l$ and distance indicator $\rho_l$ are computed, and the $K_c$ frames with the largest $\sigma_l \times \rho_l$ are chosen as the final selection. The selected frames retrieve their original ViT features from the feature buffer and are pooled into 32 tokens per frame.
    - Design Motivation: Uniform sampling wastes a massive amount of tokens on irrelevant frames. The two-stage selection first narrows the candidates down (from hundreds of frames to 64), and then eliminates redundancy via clustering, ensuring that the selected frames are both relevant to the instruction and diverse, covering key moments of the video.

3. **LLM Input Construction**:

    - Function: Combine compressed memory contents with detailed key frame representations to construct the LLM input
    - Mechanism: Concatenate the contents of M (progressive temporal information) and the frame representations $\hat{Z}$ selected by DFS (spatial details of key moments), separated by a special token $\tau$: $\langle m_0, m_1, \dots, \tau, \hat{Z} \rangle$. These two parts provide complementary information: M provides global temporal context, and $\hat{Z}$ provides local spatial details.
    - Design Motivation: Using only compressed memory loses spatial details, while using only selected frames lacks temporal context. Combining both enables the LLM to understand both "what happened" and "exactly how it happened."

### Loss & Training

Two-stage training: (1) **Multimodal Pre-training**: Freeze all components except the Perceiver, using SigLIP contrastive loss to align visual and textual features (100K video-caption pairs); (2) **Instruction Fine-tuning**: Enable the memory module, DFS, and LLM (LoRA rank=64, alpha=32), training on video instruction data for 100K steps. For the temporal localization task, an additional 15K steps of fine-tuning are performed on DiDemo and ActivityNet. Only 8×V100 GPUs are required.

## Key Experimental Results

### Main Results (Long Video VQA - MovieChat-1K)

| Model | #Frames | #Tokens | Global Acc | Global Score | Breakpoint Acc |
|------|---------|---------|------------|-------------|----------------|
| Video-LLaMA | 32 | 32 | 51.4 | 3.10 | 38.2 |
| MovieChat | 2048 | 8192 | 67.8 | 3.81 | 50.4 |
| **ReWind** | **548*** | **1184*** | **80.6** | **4.46** | **57.2** |

### Temporal Localization (Charades-STA)

| Model | R@0.3 | R@0.5 | R@0.7 | mIoU |
|------|-------|-------|-------|------|
| VTimeLLM | 51.0 | 27.5 | 11.4 | 31.2 |
| **ReWind** | **59.0** | **41.6** | **20.5** | **39.3** |

### Ablation Study

| Configuration | Global Acc | Global Score |
|------|-----------|-------------|
| Baseline (64-frame uniform sampling, no memory) | 61.5 | 3.21 |
| + Memory | Significant improvement | Significant improvement |
| + Memory + DFS | **80.6** | **4.46** |

### Key Findings

- While using approximately 1/8 of the token budget and 1/4 of the frame count of MovieChat, ReWind improves VQA accuracy by +13% (67.8% → 80.6%).
- Temporal localization mIoU increases by +8% compared to VTimeLLM (31.2% → 39.3%).
- ReWind also achieves the top average score on the short video benchmark (VideoChatGPT), indicating that the method is not only applicable to long videos.
- Both the memory module and DFS yield independent improvements, with their combination achieving the best performance.

## Highlights & Insights

- **Exquisite Design of the Read-Perceive-Write Loop**: The read operation makes new frame encoding history-aware; the perceive step preserves temporal fidelity at the frame level; and the write operation achieves efficient storage using very few tokens. The synergy of these three components realizes progressive information accumulation.
- **"Coarse-to-Fine" Two-Stage Strategy**: Building a global temporal understanding with compressed memory, and then "rewinding" to key frames to obtain details aligns with human cognitive patterns when watching long videos.
- **Extremely High Token Efficiency**: Storing only 2 tokens per frame into memory, yet supplementing key frames with 32-token spatial details through DFS, achieves an ideal balance between efficiency and accuracy.
- **Instruction Guidance Throughout**: From Perceiver encoding to frame selection, user instructions constantly participate in information filtering, ensuring both storage and retrieval are highly task-relevant.

## Limitations & Future Work

- Currently sampling at 1fps, which may miss fast-occurring events.
- The number of frames in DFS (8 frames) is fixed; future work can explore adaptive adjustment based on video complexity.
- Only LLaMA-2 7B is utilized, leaving the performance of larger-scale LLMs unexplored.
- The number of read/write queries in the memory module must be pre-set, lacking an adaptive mechanism.
- Integrating the audio modality could be considered to further enhance long video understanding.

## Related Work & Insights

- **MovieChat**: Uses a FIFO short-term memory + merged long-term memory, but suffers from over-compression and loses temporal dynamics.
- **MA-LMM**: Features a hierarchical memory module but similarly suffers from excessive compression.
- **LLaMA-VID**: Utilizes only 2 tokens per frame, but processes frames independently, lacking temporal modeling.
- **Q-Former (BLIP-2)**: Compresses information at the clip level, which is less granular than ReWind's frame-level processing.
- **Insight**: The paradigm of a memory module coupled with dynamic selection can be generalized to other multimodal tasks requiring long-sequence processing (e.g., long document understanding, multi-turn video dialogue).

## Rating

- Novelty: ⭐⭐⭐⭐ — The read-perceive-write loop and the two-stage framework feature novel designs.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive evaluations across long/short video VQA, temporal localization, and thorough ablation studies.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured with detailed descriptions of methodologies.
- Value: ⭐⭐⭐⭐⭐ — A significant breakthrough in long video understanding, achieving an optimal balance between efficiency and accuracy.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] VideoLucy: Deep Memory Backtracking for Long Video Understanding](../../NeurIPS2025/video_understanding/videolucy_deep_memory_backtracking_for_long_video_understanding.md)
- [\[ECCV 2024\] AMEGO: Active Memory from Long EGOcentric Videos](../../ECCV2024/video_understanding/amego_active_memory_from_long_egocentric_videos.md)
- [\[CVPR 2025\] SEAL: SEmantic Attention Learning for Long Video Representation](seal_semantic_attention_learning_for_long_video_representation.md)
- [\[CVPR 2025\] DrVideo: Document Retrieval Based Long Video Understanding](drvideo_document_retrieval_based_long_video_understanding.md)
- [\[ICCV 2025\] VideoLLaMB: Long Streaming Video Understanding with Recurrent Memory Bridges](../../ICCV2025/video_understanding/videollamb_long_streaming_video_understanding_with_recurrent_memory_bridges.md)

</div>

<!-- RELATED:END -->

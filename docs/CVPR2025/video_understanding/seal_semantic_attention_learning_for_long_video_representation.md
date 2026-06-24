---
title: >-
  [Paper Note] SEAL: SEmantic Attention Learning for Long Video Representation
description: >-
  [CVPR 2025][Video Understanding][Long Video Understanding] This paper proposes SEAL, a unified long video representation method that decomposes video into three semantic tokens (scene, object, and action). It uses a query-aware subset selection optimization to balance relevance and diversity, achieving a score of 45.9% on LVBench and outperforming Qwen2-VL-72B (41.3%).
tags:
  - "CVPR 2025"
  - "Video Understanding"
  - "Long Video Understanding"
  - "Semantic Decomposition"
  - "Attention Learning"
  - "Video Question Answering"
  - "Temporal Localization"
date: 2026-05-08
content_hash: a5445815036dd969
---

# SEAL: SEmantic Attention Learning for Long Video Representation

**Conference**: CVPR 2025  
**arXiv**: [2412.01798](https://arxiv.org/abs/2412.01798)  
**Code**: None  
**Area**: Video Understanding  
**Keywords**: Long Video Understanding, Semantic Decomposition, Attention Learning, Video Question Answering, Temporal Localization

## TL;DR

This paper proposes SEAL, a unified long video representation method that decomposes video into three semantic tokens (scene, object, and action). It uses a query-aware subset selection optimization to balance relevance and diversity, achieving a score of 45.9% on LVBench and outperforming Qwen2-VL-72B (41.3%).

## Background & Motivation

Long video understanding faces three major challenges:
- **High Computational Complexity**: The number of frames and pixels in hour-long videos far exceeds the capacity of current hardware.
- **Severe Temporal Redundancy**: Scenes and objects change slowly, meaning a large number of frames carry duplicate information.
- **Cross-Task Generalization**: Effective representation must simultaneously support fine-grained factual retrieval and high-level reasoning.

Limitations of Prior Work:
- Uniform sampling loses key information and generates redundancy.
- Memory bank methods merge similar frames but still rely on task-specific designs.
- Models focusing only on a single task (such as QA or temporal localization) struggle to generalize.

Brain Inspiration: Selectively attending to new information, continuously updating memory online, and dynamically adjusting focus based on the task. Inspired by this, SEAL designs a unified framework of semantic decomposition and attention learning.

## Method

### Overall Architecture

SEAL consists of two core steps:
1. **Semantic Decomposition**: Decomposing long videos from raw frames into three compressed semantic representations: scene tokens $\mathbf{T}_{\text{scene}}$, object tokens $\mathbf{T}_{\text{object}}$, and action tokens $\mathbf{T}_{\text{action}}$.
2. **Attention Learning**: Query-based subset selection optimization to choose a fixed-size subset from all semantic tokens, which is then fed into a vision head or an MLLM head to complete downstream tasks.

### Key Design 1: Three Types of Semantic Token Decomposition

- **Function**: Compresses high-dimensional dense videos into compact sets of semantic entities, significantly reducing computational overhead.
- **Mechanism**:
    - Scene tokens: Uniformly sample $N_{\text{scene}}$ frames to capture background environment information.
    - Action tokens: Extract dynamic trajectories (tracklets) using class-agnostic trackers such as SAM-2. Tracklets shorter than $L_{\min}$ are discarded, while those longer than $L_{\max}$ are segmented. The spatial union of bounding boxes across frames is taken for each tracklet.
    - Object tokens: Perform class-agnostic segmentation with SAM on keyframes to obtain static object masks.
- **Design Motivation**: The three types of tokens respectively answer the questions "where" (scene), "what" (object), and "how" (action), covering the core dimensions of video understanding. This decomposition is more information-efficient than brute-force sampling and is task-agnostic.

### Key Design 2: Subset Selection Attention Learning

- **Function**: Selects the optimal subset from a large number of candidate tokens, balancing query relevance and token diversity.
- **Mechanism**: Formulated as a combinatorial optimization problem: $T_s^* = \arg\max_{T_s \subset T_G} \alpha \sum_{t_s \in T_s} R(t_s, q) + (1-\alpha) \sum_{t_i, t_j \in T_s, i \neq j} \frac{1}{S(t_i, t_j)}$, where $R(\cdot)$ is the token-query cosine similarity calculated by BLIP-2, and $S(\cdot)$ is the cosine similarity between tokens.
- **Design Motivation**: Selecting tokens purely by relevance would lead to highly redundant token sets (concentrated in a small region). Adding the diversity term ensures that the selected tokens cover different aspects of the video. The hyperparameter $\alpha=0.9$ balances these two objectives.

### Key Design 3: Streaming and Global Dual Modes

- **Function**: Supports online processing of videos of arbitrary lengths.
- **Mechanism**: The global mode processes all tokens at once to output a unified representation. The streaming mode uses a fixed-size sliding window, performing attention learning at each step on the union of the current window's tokens and the previously selected subset: $T_{\text{sub}}^t = \text{Attention\_Learning}(T_t \cup T_{\text{sub}}^{t-1})$.
- **Design Motivation**: The global mode is suitable for offline analysis, while the streaming mode supports real-time scenarios (e.g., answering questions while watching a movie), making the representation independent of video length.

### Loss & Training

Task-specific downstream objectives: temporal localization uses IoU distance + focal loss to train classification + regression heads; video QA uses the negative log-likelihood loss of autoregressive next-token prediction in MLLMs.

## Key Experimental Results

### Main Results 1: LVBench Video QA (Hour-Long Videos)

| Model | LLM Size | Overall | KIR | EU | Sum | ER | Rea | TG |
|------|---------|---------|-----|-----|-----|-----|-----|-----|
| Qwen2-VL | 72B | 41.3 | 38.3 | 41.1 | 46.6 | 38.0 | 46.5 | 41.4 |
| InternVL2 | 34B | 39.6 | 43.4 | 39.7 | 41.4 | 37.4 | 42.5 | 31.4 |
| **SEAL** | **34B** | **45.9** | **51.5** | **41.3** | 39.7 | **47.9** | 43.3 | 32.3 |

SEAL, using a 34B model, outperforms the 72B Qwen2-VL by 4.6%, leading by 8.1% on KIR and 5.1% on ER, respectively.

### Main Results 2: Ego4D-NLQ Temporal Localization (Limited Token Constraints)

| Model | #Tokens | R@1 IoU=0.3 | R@1 IoU=0.5 | R@5 IoU=0.3 | R@5 IoU=0.5 |
|------|--------|-------------|-------------|-------------|-------------|
| SnAG | 450 | 13.44 | 9.23 | 34.02 | 23.04 |
| **SEAL** | **450** | **13.78** | **9.26** | **34.79** | **23.10** |
| SnAG | 200 | 10.03 | 6.35 | 26.56 | 16.90 |
| **SEAL** | **200** | **10.83** | **7.06** | **27.39** | **17.41** |

### Key Findings

- Semantic decomposition effectively reduces redundancy: models with fewer parameters can outperform larger ones.
- Action and object tokens contribute the most to KIR and ER tasks.
- Streaming mode performance is only slightly lower than that of the global mode, validating the feasibility of online representation updates.
- It does not rely on a specific LLM architecture; the unified representation can be interfaces with different prediction heads.

## Highlights & Insights

1. **Cognitive Science-Inspired Design**: The three types of semantic tokens highly align with the attention allocation mechanism of the human brain for videos.
2. **Unified Representation Cross-Task Generalization**: The same representation can be integrated with different heads to perform QA and temporal localization without requiring task-specific encoding.
3. **Small Models Outperforming Large Models**: This suggests that in video understanding, "what to look at" is more critical than "how large the model is." Efficient information selection can compensate for discrepancies in parameter sizes.

## Limitations & Future Work

- Semantic decomposition relies on the quality of external models like SAM-2 and may fail in unconventional scenes (e.g., severe motion blur, heavy occlusion).
- The subset selection optimization is NP-hard. Grasping it with a greedy approximation might yield sub-optimal results.
- Performance on causal reasoning questions is relatively weak (e.g., "why" type questions).
- Video dialogue and multi-turn interaction scenarios have not yet been explored.

## Related Work & Insights

- **MovieChat**: A long video method based on the Atkinson-Shiffrin memory model. SEAL also shows advantages on its dataset.
- **SnAG**: A baseline for temporal localization. SEAL consistently outperforms it under limited token constraints.
- **TimeSformer**: The backbone architecture for space-time attention blocks in SEAL.

## Rating

⭐⭐⭐⭐ — The problem definition is clear, and the design combining semantic decomposition and attention learning is both elegant and practical. Outstanding performance of the 34B model surpassing the 72B model on LVBench is highly convincing. The greedy approximation for subset selection and reliance on external models remain the primary weaknesses.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Learning Audio-Guided Video Representation with Gated Attention for Video-Text Retrieval](learning_audio-guided_video_representation_with_gated_attention_for_video-text_r.md)
- [\[CVPR 2025\] Heterogeneous Skeleton-Based Action Representation Learning](heterogeneous_skeleton-based_action_representation_learning.md)
- [\[CVPR 2025\] ReWind: Understanding Long Videos with Instructed Learnable Memory](rewind_understanding_long_videos_with_instructed_learnable_memory.md)
- [\[CVPR 2025\] H-MoRe: Learning Human-centric Motion Representation for Action Analysis](h-more_learning_human-centric_motion_representation_for_action_analysis.md)
- [\[CVPR 2026\] CLCR: Cross-Level Semantic Collaborative Representation for Multimodal Learning](../../CVPR2026/video_understanding/clcr_cross-level_semantic_collaborative_representation_for_multimodal_learning.md)

</div>

<!-- RELATED:END -->

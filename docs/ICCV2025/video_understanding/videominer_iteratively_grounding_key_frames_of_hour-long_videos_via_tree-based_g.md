---
title: >-
  [Paper Note] VideoMiner: Iteratively Grounding Key Frames of Hour-Long Videos via Tree-based Group Relative Policy Optimization
description: >-
  [ICCV 2025][Video Understanding][Long-form video understanding] This paper proposes VideoMiner, a tree-structured reinforcement learning framework for long-form video understanding. It iteratively applies segmentation–captioning–clustering to construct a hierarchical video tree, and introduces T-GRPO (Tree-based Group Relative Policy Optimization) to guide a policy model in adaptively exploring key frames. VideoMiner achieves state-of-the-art performance on four long-video be…
tags:
  - "ICCV 2025"
  - "Video Understanding"
  - "Long-form video understanding"
  - "key frame extraction"
  - "reinforcement learning"
  - "GRPO"
  - "tree structure"
  - "hierarchical video representation"
date: 2026-05-08
content_hash: eef05a4d7056de07
---

# VideoMiner: Iteratively Grounding Key Frames of Hour-Long Videos via Tree-based Group Relative Policy Optimization

**Conference**: ICCV 2025
**arXiv**: [2510.06040](https://arxiv.org/abs/2510.06040)  
**Code**: [GitHub](https://github.com/caoxinye/VideoMiner)  
**Area**: Video Understanding / Long-form Video QA
**Keywords**: Long-form video understanding, key frame extraction, reinforcement learning, GRPO, tree structure, hierarchical video representation

## TL;DR

This paper proposes VideoMiner, a tree-structured reinforcement learning framework for long-form video understanding. It iteratively applies segmentation–captioning–clustering to construct a hierarchical video tree, and introduces T-GRPO (Tree-based Group Relative Policy Optimization) to guide a policy model in adaptively exploring key frames. VideoMiner achieves state-of-the-art performance on four long-video benchmarks, and it is observed that T-GRPO spontaneously elicits chain-of-thought reasoning.

## Background & Motivation

Understanding hour-long videos is a frontier challenge for multimodal large language models (MM-LLMs), with applications spanning sports highlight detection, movie narrative summarization, and surveillance anomaly detection. Compared to static images or short clips, long videos contain **thousands of frames and complex temporal dynamics**, giving rise to two core challenges:

### Challenge 1: How to eliminate massive irrelevant and redundant information?

- **End-to-end methods** (e.g., LLaVA-Video, Qwen2-VL): Reduce videos to uniformly sampled frame lists, but as video length grows, **irrelevant information increases exponentially**, overwhelming the LLM.
- **Hierarchical methods** (e.g., VideoTree): Introduce structure to reduce complexity, but may **disrupt the original video structure** and lose temporal information.

### Challenge 2: How to precisely localize key frames within a complex hierarchical structure?

- VideoTree's visual clustering and relevance scoring are limited for hour-long videos.
- Key frame extraction must simultaneously satisfy three principles: (1) integrating event-level spatiotemporal information, (2) query-guided exploration, and (3) compatibility with the hierarchical tree structure.
- Existing methods lack **adaptive decision-making** capabilities — when to stop exploring and when to go deeper.

VideoMiner's core mechanism: **coarse-to-fine hierarchical decomposition** (video → events → frames) preserves temporal coherence, while T-GRPO trains the policy model to learn **when to accept, continue, or discard** tree nodes.

## Method

### Overall Architecture

VideoMiner consists of three components in sequence:

1. **Scene segmentation + captioning + clustering**: Iteratively decomposes a long video into a hierarchical tree structure.
2. **T-GRPO tree exploration**: A policy model decides the fate of each node (accept/continue/delete).
3. **LLM inference**: Selected key frames and the question are fed into a VLM to generate the final answer.

### Key Designs

#### Scene Segmentation

Parameter-free segmentation is achieved via grayscale histogram change detection:

1. Compute a normalized grayscale histogram $H_t(k)$ for each frame.
2. Quantify inter-frame differences using the **Bhattacharyya distance**:

$$D_i = -\ln \sum_{k=0}^{255} \sqrt{H_i(k) \cdot H_{i+1}(k)}$$

3. Select the top $K-1$ change points from the distance sequence as segmentation boundaries.
4. Obtain $K$ event segments $E = \{E_1, \ldots, E_K\}$.

**Highlight**: Segmentation is performed at the event level rather than on discrete frames, preserving temporal coherence.

#### Caption Generation and Clustering

1. **Caption generation**: For each event $E_m$, a VLM generates a caption conditioned on the user query $Q$:

$$\text{Caption}_m = \text{VLM}(E_m, Q)$$

Query-oriented captions ensure that the extracted information is aligned with user intent.

2. **Clustering to build the tree**: Captions are encoded into vectors $v_m$ via an embedding model and clustered using **DBSCAN**:

$$\{v_1, \ldots, v_K\} \xrightarrow{\text{DBSCAN}} \{l_1, \ldots, l_C\}$$

Each cluster forms a tree node; $C \leq K$ ensures semantically related scenes are merged.

#### Tree Exploration: Policy Model

The policy model $\text{PM}$ takes three inputs to make a decision:

$$\text{State}(N_i) = \text{PM}(\text{Caption}_m, Q, \text{depth}(N_i))$$

- **Event caption**: Provides spatiotemporal information.
- **User query**: Ensures exploration is aligned with the query.
- **Node depth**: Provides hierarchical position information.

Three decision states:
- **Accept**: The node contains sufficient key frames; no further exploration needed.
- **Continue**: The node may be relevant; expand into new child nodes (re-segment–caption–cluster).
- **Delete**: The node is irrelevant to the query; discard.

### Loss & Training

#### T-GRPO: Tree-based Group Relative Policy Optimization

T-GRPO extends DeepSeek's GRPO to accommodate tree structures and video understanding tasks.

**Rollout**: The VideoMiner pipeline is executed to generate $n$ distinct trees $T = \{\vec{T_1}, \ldots, \vec{T_n}\}$.

**Reward design** operates at two levels:

**Node-level reward $R_{\text{node}}$** comprises three components:

1. **Format reward** $r_{\text{format}}$: Full format compliance yields $\delta_{\max}$; partial compliance yields $\delta_{\text{corr}}$.

2. **Length reward** $r_{\text{length}}$: Modeled as a Gaussian distribution to control the output token length of the policy model:

$$r_{\text{length}}(l_o) = \rho \exp\left(-\frac{(l_o - l_t)^2}{2\sigma^2}\right)$$

Longer outputs correspond to more detailed reasoning and higher accuracy.

3. **Action reward** $r_{\text{action}}$: Different actions receive different rewards ($\delta_d > \delta_a > \delta_c$). A **tree auxin** factor is defined as:

$$\lambda_{\text{auxin}} = \frac{\delta_d + \delta_a}{2\delta_c}$$

The intuition draws from plant auxin: moderately discouraging continued exploration (continue) while encouraging timely termination decisions (accept/delete) improves localization efficiency.

**Tree-level reward $R_{\text{tree}}$**: Based on the accuracy of the final answer.

**Total reward**:

$$R_{\text{total}} = r_{\text{format}} + (r_{\text{length}} + r_{\text{action}}) \cdot R_{\text{tree}}$$

**Loss function**: Group advantages are computed and the policy model is optimized with a PPO-clip style objective:

$$A_{ij} = \frac{r_{ij} - \text{mean}(\{r_{11}, \ldots, r_{nG_n}\})}{\text{std}(\{r_{11}, \ldots, r_{nG_n}\})}$$

## Key Experimental Results

### Main Results: Long-Video Understanding Benchmarks (Table 1)

| Method | Base Model | EgoSchema | Video-MME Long | LongVideoBench (900–3600s) | MLVU M-Avg |
|--------|-----------|-----------|----------------|--------------------------|------------|
| LLaVA-Video | Qwen2-7B | 60.2 | 49.3 | 45.5 | 62.1 |
| InternVL2.5 | InternVL-2-8B | 60.0 | 50.6 | 46.4 | 59.2 |
| VideoTree | Qwen-plus | 59.8 | 39.3 | 44.6 | 51.6 |
| LLoVi | Qwen-plus | 62.8 | 50.6 | 39.5 | 54.9 |
| **VideoMiner** | **Qwen2-VL-7B** | **66.2** | **52.2** | **49.3** | **65.1** |

VideoMiner achieves state-of-the-art results across all long-video sub-tasks. Its relative advantage grows with video length (surpassing the best baseline by +1.6 pp on Video-MME Long and +2.9 pp on LongVideoBench).

### Ablation Study

**Figure 3a: Clustering method comparison**

| Method | Performance | Efficiency |
|--------|-------------|------------|
| No clustering | Low | Slowest (exponential node growth) |
| Frame clustering | Medium | Slower |
| **Event clustering** | **Highest** | **Fastest** |

Event clustering retains more temporal information, enabling the policy model to make accurate decisions earlier.

**Figure 3b: Reinforcement learning method comparison**

| Method | Performance |
|--------|-------------|
| No RL (base model) | Worst; degrades severely with video length |
| RF (no tree-level reward) | Significantly outperforms baseline |
| **T-GRPO (with tree-level reward)** | **Best** |

The tree-level reward enables the policy model to account for the downstream impact of current decisions.

**Figure 5b: Effect of auxin $\lambda_{\text{auxin}}$**

- $\lambda < 1$: The model favors continue; exploration is thorough but inefficient, and aimless exploration may degrade performance.
- $\lambda \approx 1$: Optimal balance — sufficient exploration with timely termination.
- $\lambda > 1$: Premature termination; key frames may be missed.

## Highlights & Insights

1. **T-GRPO spontaneously elicits chain-of-thought reasoning**: After training, the policy model autonomously generates CoT-style reasoning (e.g., "This node shows a sports match… relevant to the question… decision: continue"), substantially deepening its reasoning capability.
2. **The biological analogy of tree auxin is elegant**: The concept of auxin regulating plant growth is borrowed to control the depth of tree exploration via reward ratios.
3. **Adaptiveness of DBSCAN**: No pre-specified number of clusters is required; the number of nodes is determined automatically based on the semantic distribution of captions.
4. Only a lightweight policy model is trained — without modifying the VLM itself — yet long-video understanding is substantially improved.

## Limitations & Future Work

1. The **cascaded architecture** that repeatedly invokes the VLM for captioning and final inference incurs high latency, making real-time deployment difficult.
2. Scene segmentation relies on grayscale histograms — simple but coarse — making it sensitive to abrupt illumination changes and prone to spurious segmentation points.
3. The reward design for the policy model involves six hyperparameters that require careful tuning.
4. On short-video tasks, the method underperforms end-to-end approaches, introducing unnecessary complexity for scenarios that do not require key frame selection.

## Related Work & Insights

- **Long-form video understanding**: LLoVi, VideoTree, VideoAgent, LLaVA-Video
- **Video RL**: GRPO, PPO, RWM-RL
- **Key frame extraction**: VideoTree, VideoAgent

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Extending GRPO to tree structures is a novel direction for RL + video understanding; the spontaneous emergence of chain-of-thought reasoning is a compelling finding.
- **Value**: ⭐⭐⭐⭐ — Directly enhances the long-video capability of existing VLMs, though latency is a concern.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Four benchmarks, ten baselines, and complete ablations on clustering and RL design.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear pipeline diagrams and illustrative case studies.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Vamba: Understanding Hour-Long Videos with Hybrid Mamba-Transformers](vamba_understanding_hour-long_videos_with_hybrid_mamba-transformers.md)
- [\[AAAI 2026\] TSPO: Temporal Sampling Policy Optimization for Long-form Video Language Understanding](../../AAAI2026/video_understanding/tspo_temporal_sampling_policy_optimization_for_long-form_video_language_understa.md)
- [\[ICCV 2025\] DynImg: Key Frames with Visual Prompts are Good Representation for Multi-Modal Video Understanding](dynimg_key_frames_with_visual_prompts_are_good_representation_for_multi-modal_vi.md)
- [\[ICCV 2025\] Fine-grained Spatiotemporal Grounding on Egocentric Videos](fine-grained_spatiotemporal_grounding_on_egocentric_videos.md)
- [\[CVPR 2025\] DeCafNet: Delegate and Conquer for Efficient Temporal Grounding in Long Videos](../../CVPR2025/video_understanding/decafnet_delegate_and_conquer_for_efficient_temporal_grounding_in_long_videos.md)

</div>

<!-- RELATED:END -->

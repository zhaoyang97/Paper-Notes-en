---
title: >-
  [Paper Note] FOCUS: Efficient Keyframe Selection for Long Video Understanding
description: >-
  [ICLR 2026][Video Understanding][Keyframe Selection] FOCUS reformulates the task of "selecting the most query-relevant video frames under a strict token budget" as a Combinatorial Pure Exploration (CPE) problem in multi-armed bandits. By treating short video segments as arms and adaptively allocating the scoring budget using empirical means and Bernstein confidence radii, it significantly improves long video QA accuracy while viewing less than 2% of total frames.
tags:
  - "ICLR 2026"
  - "Video Understanding"
  - "Keyframe Selection"
  - "Long Video Understanding"
  - "Multi-Armed Bandits"
  - "Pure Exploration"
  - "Training-free"
date: 2026-05-08
content_hash: 7697bfdbda8551c0
---

# FOCUS: Efficient Keyframe Selection for Long Video Understanding

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=1OQKqLFcbB](https://openreview.net/forum?id=1OQKqLFcbB)  
**Code**: To be confirmed  
**Area**: Video Understanding / Multimodal Large Language Models  
**Keywords**: Keyframe Selection, Long Video Understanding, Multi-Armed Bandits, Pure Exploration, Training-free  

## TL;DR
FOCUS reformulates the task of "selecting the most query-relevant video frames under a strict token budget" as a Combinatorial Pure Exploration (CPE) problem in multi-armed bandits. By treating short video segments as arms and adaptively allocating the scoring budget using empirical means and Bernstein confidence radii, it significantly improves long video QA accuracy while viewing less than 2% of total frames.

## Background & Motivation
**Background**: Multimodal Large Language Models (MLLMs) encode images/video frames into visual tokens and feed them into the LLM alongside text. However, when extending from a single image to a one-hour video, the explosion in frame count (over $10^5$ frames at 30fps) causes visual tokens to far exceed computational limits. Common practices involve either uniform downsampling (e.g., sampling 64 frames from an hour) or using vision-language encoders like CLIP/BLIP to score every frame's relevance to the query and taking the Top-K.

**Limitations of Prior Work**: Uniform sampling often misses critical moments. Meanwhile, relevance-scoring-based keyframe selection methods, while seemingly training-free and plug-and-play, carry a hidden cost: **they must downsample the video to 1fps for pre-filtering before scoring.** The reason is practical: scoring every frame of a one-hour video with BLIP requires $10^{11}\sim10^{12}$ FLOPs (empirical tests show AKS requires 255 GPU hours without pre-filtering). This pre-filtering contradicts the goal of "finding the most informative frames from the entire video," as the most valuable moments might be discarded during downsampling.

**Key Challenge**: Exhaustive scoring is too expensive vs. pre-filtering for cost-saving loses key frames—this is essentially a resource allocation problem of "where to spend computational power given a limited scoring budget."

**Goal**: Design a training-free, model-agnostic, plug-and-play keyframe selection module that identifies query-relevant frames under a strict token budget **without requiring pre-filtering**, enabling direct processing of ultra-long videos.

**Core Idea**: The authors observe that natural videos exhibit **strong temporal locality**—adjacent frames are highly correlated in appearance/motion, and this smoothness carries over to "frame-query relevance scores." Using the Autocorrelation Function (ACF) on LongVideoBench and Video-MME, the paper finds that median correlation remains above 0.5 within the first 5 seconds. **This implies that exhaustive scoring is wasteful: adjacent frames can be grouped into segments, and bandit-style adaptive exploration can be used to quickly exclude irrelevant regions and concentrate scoring on promising segments.**

## Method

### Overall Architecture
FOCUS (Frame-Optimistic Confidence Upper-bound Selection) partitions the video along the temporal axis into $M$ equal-length, non-overlapping segments, each treated as an "arm" of a bandit. Pulling an arm involves randomly sampling a frame from that segment and calculating its relevance to the query using BLIP as the reward. The goal is to identify the "top $m$ most valuable segments" (CPE) under a limited pull budget, then pick the top relevant frames from these selected segments to form a $k$-frame keyframe set for the MLLM. The pipeline follows two stages: a coarse exploration to obtain reliable statistics for each arm, followed by fine-grained scoring for the most promising arms, and finally selecting based on unbiased means.

```mermaid
flowchart LR
    A[Long Video] --> B[Partition into M segments<br/>Each segment = An arm]
    B --> C[Stage I: Coarse Exploration<br/>Pull each arm q times<br/>Compute empirical mean + Bernstein radius]
    C --> D[Select top-αm arms using optimistic UCB]
    D --> E[Stage II: Fine-grained Scoring<br/>Perform z additional pulls for selected arms]
    E --> F[Select top-m arms by unbiased empirical mean]
    F --> G[Intra-arm interpolation + sampling<br/>Sample ka frames per arm]
    G --> H[k keyframes → MLLM for Answer]
```

### Key Designs

**1. Modeling keyframe selection as a Combinatorial Pure Exploration bandit: Replacing "exhaustive scoring" with "budget allocation."** The paper decomposes the task-level utility $R_\Phi(K\mid V,q)$ into a sum of frame-level utilities $K^\star=\arg\max_{|K|=k}\mathbb{E}[\sum_{t\in K}y_t]$. It assumes frame-level utility $y_t$ can be indirectly observed via a vision-language encoder: $r_t=\psi(x_t,q;\theta)=y_t+\epsilon_\psi$, where noise $\epsilon_\psi$ has zero mean and variance $\sigma_\psi^2$, making the relevance score $r_t$ an unbiased estimate of $y_t$. By treating each segment $A_a$ as an arm with frame-level utilities following a distribution $\nu_a$ with mean $\mu_a$, "selecting the optimal subset of segments" becomes the classic CPE objective $S^\star=\arg\max_{S\in\mathcal{S}}\sum_{a\in S}\mu_a$ (where $\mathcal{S}$ is the set of all subsets of size $m$). This step is the pivot of the entire work—it transforms "which frames to score" from heuristic pre-filtering into a theoretically grounded exploration-exploitation decision.

**2. Bernstein confidence radius driven "optimism in the face of uncertainty" exploration.** Each arm maintains an empirical mean $\hat\mu_a(n)$ and a variance-adaptive empirical Bernstein confidence radius:
$$\beta_a(n)=\sqrt{\frac{2\hat\sigma_a^2\ln n}{\max(1,N_a(n))}}+\frac{3\ln n}{\max(1,N_a(n))},$$
which guarantees $P[|\hat\mu_a(n)-\mu_a|\le\beta_a(n)]\ge 1-6/n$. Computational power is adaptively allocated to arms that are either "promising" (high mean) or "uncertain" (large radius). Specifically, for arms currently in the top-$m$, the lower confidence bound is considered; for those outside, the upper confidence bound is considered. As long as an external arm's UCB could exceed an internal arm's LCB, the "most uncertain" arm $a=\arg\max_{a\in(\tilde A_n\setminus A_n)\cup(A_n\setminus\tilde A_n)}\beta_a(n)$ is pulled iteratively until the top-$m$ set stabilizes. The iterative version (Algorithm 1) has high probability guarantees for converging to the optimal top-$m$ set. Compared to standard variance-agnostic UCB, the empirical Bernstein radius allows segments with low variance to converge faster, saving pulls for truly ambiguous regions.

**3. Two-stage coarse-to-fine reduction: Parallelizing serial iterations.** While the iterative version has guarantees, it pull arms one by one (batch size=1), resulting in low GPU utilization. The paper specializes this into a two-stage batch process (Algorithm 2): **Stage I: Coarse Exploration**—parallelly pull each arm $q$ times to obtain global empirical means and confidence radii; **Stage II: Fine-grained Exploitation**—select top-$\alpha m$ arms $A_{\text{coarse}}$ using optimistic means $\tilde\mu_a(n)=\hat\mu_a(n)+\beta_a(n)$, and perform $z$ additional batch pulls for each. The hyperparameter $\alpha$ controls the budget ratio between coarse and fine exploration. This step uses "optimistic means to guide exploration but unbiased empirical means for final selection" ($A_{\text{fine}}=\text{TopM}(\hat\mu,m)$), retaining the core of "optimism" while eliminating step-by-step scheduling overhead to allow one-time batch BLIP inference.

**4. Intra-arm interpolation sampling for keyframes.** After selecting the arm set $A_{\text{fine}}$, the $k$-frame budget is divided among arms ($k_a\approx k/|A_{\text{fine}}|$). Since only a few frames per arm were sampled, the paper uses nearest-neighbor interpolation to propagate observed rewards to all frames in the arm $\hat r_{a,t}$, constructing an intra-arm sampling distribution $p_a$. Then, $k_a$ frames are sampled without replacement. The final keyframe set $K=\bigcup_{a\in A_{\text{fine}}}K_a$ falls within high-relevance segments while following the relevance distribution internally rather than simply taking the maximum, balancing relevance and coverage.

## Key Experimental Results

### Main Results
QA accuracy (%) for "FOCUS Selection vs. Uniform Sampling" across four MLLMs:

| Model | Frames | LongVideoBench | Video-MME |
|------|------|----------------|-----------|
| GPT-4o | 32 | 51.6 → 54.8 (↑3.2) | 61.8 → 62.5 (↑0.7) |
| Qwen2-VL-7B | 32 | 55.6 → 62.3 (↑6.7) | 57.4 → 59.7 (↑2.3) |
| LLaVA-OV-7B | 32 | 54.8 → 60.7 (↑5.9) | 56.5 → 58.3 (↑1.8) |
| LLaVA-Video-7B | 64 | 58.9 → 63.5 (↑4.6) | 64.4 → 65.4 (↑1.0) |

Highlight: Qwen2-VL-7B with FOCUS outperforms Gemini-1.5-Flash on LongVideoBench using only 1/8 of the input frames.

### Ablation Study
Comparison with SOTA training-free keyframe selection methods on LLaVA-Video-7B (k=64) by video length:

| Method | LVB-Short | LVB-Medium | LVB-Long | LVB-Overall | VMME-Overall |
|------|-----------|------------|----------|-------------|--------------|
| Uniform | 67.5 | 57.4 | 51.8 | 58.9 | 64.4 |
| Top-K | 72.3 | 58.0 | 60.5 | 62.3 | 62.9 |
| AKS | 72.3 | 59.2 | 56.1 | 62.1 | 64.6 |
| **FOCUS** | 72.3 | 59.0 | **63.7** | **63.5** | **65.4** |

Efficiency comparison (LongVideoBench, single H100):

| Method | Frames Seen (%) | GPU Hours |
|------|-----------------|----------|
| AKS (No pre-filtering) | 100 | 255 |
| AKS (With pre-filtering) | 3.7 | 9.3 |
| **FOCUS** | **1.6** | **5.5** |

$\alpha$ hyperparameter trade-off: $\alpha=0.1$ scans 1.1% frames / 3.5 GPU hours / 62.9%; $\alpha=0.25$ scans 1.6% / 5.5h / 63.5%; $\alpha=0.5$ scans 2.5% / 9.2h / 63.6% (diminishing returns).

### Key Findings
- **Gains are largest for long videos**: On LongVideoBench videos exceeding 20 minutes, FOCUS outperforms uniform sampling by 11.9% and Top-K by 7.6%, indicating that adaptive selection is more critical as frame count increases and information becomes sparser.
- **Minimal difference for short videos**: Performance of selection methods is similar on short videos and all exceed uniform sampling, likely because MLLM reasoning capacity saturates on shorter contexts, reducing the impact of input selection.
- **Efficiency and accuracy win-win**: FOCUS achieves the highest overall accuracy while viewing only 1.6% of frames and requiring 5.5 GPU hours, eliminating the need for the 1fps pre-filtering required by AKS.

## Highlights & Insights
- **Elegant problem reformulation**: Elevating "keyframe selection" from heuristic scoring to a theoretically grounded CPE bandit problem makes the decision of "where to spend compute" optimizable and analyzable rather than arbitrary.
- **Balance between theory and engineering**: While the iterative version has guarantees but is not parallelizable, the authors' reduction to a two-stage batch version preserves the essence of "optimistic exploration + unbiased selection," which is key to making bandit theory practical on GPUs.
- **Empirical support for temporal locality**: Quantifying "adjacent frame relevance score correlation" using ACF provides direct evidence for the "segment-as-arm" and "intra-arm interpolation" design choices.
- **Truly plug-and-play**: Being training-free and model-agnostic, it directly improves both open-source and closed-source MLLMs, with fair comparisons ensured by using the same BLIP scorer.

## Limitations & Future Work
- **i.i.d. assumption ignores temporal dependencies**: FOCUS assumes frame-level relevance scores within a segment are i.i.d., lacking explicit modeling of temporal dependencies between segments. This explains why its gains on Video-MME (more global questions, dispersed keyframes) are smaller than on LongVideoBench (more specific questions, concentrated keyframes). Integrating Lipschitz/metric bandits or contextual bandits to characterize temporal structure is a clear future direction.
- **Dependency on scorer quality**: Relevance scores are provided by BLIP; thus, overall performance is limited by the vision-language encoder's alignment capability, and encoder noise $\epsilon_\psi$ directly impacts estimation.
- **Uniform frame allocation to arms**: Equally distributing keyframe counts across selected arms might not be optimal for cases where information is highly concentrated in a single segment.

## Related Work & Insights
FOCUS aligns with the "visual token compression" trajectory for long-video MLLMs, joining uniform sampling (VideoLLaVA), relevance Top-K, and methods balancing relevance/coverage like AKS or multi-resolution adaptive methods like Q-Frame. Its uniqueness lies in integrating these into a bandit pure exploration framework, providing the first scalable solution that "does not require pre-filtering." A broader insight: any "subset selection under budget" problem in multimodal domains (frame selection, token pruning, chunk selection in RAG) can be remodeled using pure exploration or contextual bandits to replace heuristics with theoretically grounded, adaptive compute allocation.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Reformulating keyframe selection as CPE bandits and implementing it via a parallelizable two-stage reduction is novel and theoretically sound.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 4 MLLMs, 2 main + 2 generalization benchmarks, including accuracy/efficiency trade-offs, length binning, and $\alpha$ ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is empirically supported by ACF, and the method is logically derived from oracle objectives to executable algorithms.
- **Value**: ⭐⭐⭐⭐ Training-free, plug-and-play, and significantly reduces costs (viewing only 1.6% frames), offering direct practical value for long-video MLLM deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Efficient Frame Selection for Long Video Understanding via Reinforcement Learning](../../CVPR2026/video_understanding/efficient_frame_selection_for_long_video_understanding_via_reinforcement_learnin.md)
- [\[ICLR 2026\] FLoC: Facility Location-Based Efficient Visual Token Compression for Long Video Understanding](floc_facility_location-based_efficient_visual_token_compression_for_long_video_u.md)
- [\[CVPR 2026\] Wavelet-based Frame Selection by Detecting Semantic Boundary for Long Video Understanding](../../CVPR2026/video_understanding/wavelet-based_frame_selection_by_detecting_semantic_boundary_for_long_video_unde.md)
- [\[CVPR 2026\] GIFT: Global Irreplaceability Frame Targeting for Efficient Video Understanding](../../CVPR2026/video_understanding/gift_global_irreplaceability_frame_targeting_for_efficient_video_understanding.md)
- [\[ICLR 2026\] ScaleLong: A Multi-Timescale Benchmark for Long Video Understanding](scalelong_a_multi-timescale_benchmark_for_long_video_understanding.md)

</div>

<!-- RELATED:END -->

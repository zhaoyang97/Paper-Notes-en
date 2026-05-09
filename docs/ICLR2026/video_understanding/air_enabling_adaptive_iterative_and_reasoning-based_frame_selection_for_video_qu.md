---
title: >-
  [Paper Note] A.I.R.: Adaptive, Iterative, and Reasoning-based Frame Selection For Video Question Answering
description: >-
  [ICLR 2026][Video Understanding][video QA] This paper proposes A.I.R., a training-free adaptive-iterative-reasoning-driven frame selection framework that addresses two fundamental challenges in VideoQA—inaccurate similarity estimation by lightweight models (CLIP) and the prohibitive computational cost of VLM-based analysis—via a two-stage strategy: GMM-based adaptive initial sampling followed by iterative VLM-guided refinement. In the worst case, A.I.R. analyzes only 72 frames (vs. 128 for baselines), while consistently improving performance across multiple long-video benchmarks.
tags:
  - ICLR 2026
  - Video Understanding
  - video QA
  - frame selection
  - VLM
  - iterative search
  - computational efficiency
date: 2026-05-08
content_hash: 9dcd9afe8d637e79
---

# A.I.R.: Adaptive, Iterative, and Reasoning-based Frame Selection For Video Question Answering

**Conference**: ICLR 2026
**arXiv**: [2510.04428](https://arxiv.org/abs/2510.04428)
**Code**: [https://ucf-air.github.io/](https://ucf-air.github.io/)
**Area**: Video Understanding
**Keywords**: video QA, frame selection, VLM, iterative search, computational efficiency

## TL;DR
This paper proposes A.I.R., a training-free adaptive-iterative-reasoning-driven frame selection framework that addresses two fundamental challenges in VideoQA—inaccurate similarity estimation by lightweight models (CLIP) and the prohibitive computational cost of VLM-based analysis—via a two-stage strategy: GMM-based adaptive initial sampling followed by iterative VLM-guided refinement. In the worst case, A.I.R. analyzes only 72 frames (vs. 128 for baselines), while consistently improving performance across multiple long-video benchmarks.

## Background & Motivation
**Background**: Frame selection is critical in VideoQA, as full videos are too long to process entirely. Existing approaches fall into two categories: (1) lightweight models (e.g., CLIP) that compute query-frame similarity for selection—fast but inaccurate for complex queries; and (2) VLMs that analyze frames individually—accurate but computationally prohibitive (128 frames ≈ 162 seconds).

**Limitations of Prior Work**: CLIP treats queries as bags of keywords, failing to capture temporal reasoning (e.g., "after introducing tofu") and complex semantics, resulting in similarity scores that do not reflect true relevance. Exhaustive VLM analysis of all frames is infeasible.

**Key Challenge**: Accurate frame selection requires deep semantic understanding (VLM), yet the per-frame cost of VLM analysis scales linearly with the number of frames. The core challenge is reducing VLM invocations without sacrificing selection quality.

**Goal**: To make deep VLM analysis computationally feasible by restricting it to a small set of high-potential frames rather than the entire video.

**Key Insight**: A two-stage pipeline—coarse filtering via CLIP (fast but imprecise), followed by fine-grained VLM analysis on a small set of high-potential frames (precise but expensive), with iterative discovery of additional relevant frames.

**Core Idea**: An iterative loop in which the VLM analyzes only small batches of the most promising frames, complemented by localized dense sampling to discover temporally adjacent relevant frames.

## Method

### Overall Architecture
The framework consists of three stages: (1) **Adaptive Initial Sampling**—fitting a GMM to the similarity score distribution to identify event intervals and allocating a sampling budget $K$ proportional to event duration; (2) **Iterative Frame Selection**—a four-step loop (ranking → VLM analysis → early stopping check → localized dense sampling) that progressively refines the frame set; and (3) **QA Stage**—feeding the final selected frames to an answering VLM.

### Key Designs

1. **Adaptive Initial Sampling**:

    - **Function**: Adaptively selects $K$ highly relevant frames from $n$ uniformly sampled frames.
    - **Mechanism**: A two-component GMM is fitted to the CLIP similarity score distribution. An adaptive threshold $T = \max(\mu_1, \mu_2) - \gamma \cdot \max(\sigma_1, \sigma_2)$ separates high- and low-relevance frames. Consecutive high-relevance frames are grouped into "events," with short gaps merged and overly short events pruned; the sampling budget is then allocated proportionally to event duration.
    - **Design Motivation**: Concentrates coverage on query-relevant regions more effectively than uniform sampling, with the threshold adapting to each video's score distribution.

2. **Iterative Frame Selection**:

    - **Step 1 — Interval Potential Ranking**: The video is partitioned into intervals by the currently selected frames; each interval's "potential" is computed as Relevance × Complexity × Length, and the top $C$ frames are selected.
    - **Step 2 — Reasoning-Based VLM Analysis**: The VLM performs reasoning-based analysis on the $C$ frames (generating rationales and scores on a 1–5 scale); frames scoring above threshold $\theta$ are retained as Positive frames.
    - **Step 3 — Early Stop**: If the number of selected frames reaches the adaptive budget $B$, the loop terminates immediately.
    - **Step 4 — Localized Density Sampling (LDS)**: Within the temporal neighborhood of VLM-verified frames, new candidate frames are discovered using exponentially growing step sizes and added to the next iteration's candidate pool.

3. **Efficiency Guarantee**:

    - Best case: VLM analyzes $C$ frames (1 iteration); worst case: $C \times I_{\max}$ frames.
    - Default settings ($C=12$, $I_{\max}=6$) yield a worst-case of 72 frames, fewer than the 128-frame baseline.
    - Early stopping typically triggers within 2–3 iterations in practice.

### Loss & Training
A.I.R. is entirely training-free and plug-and-play, compatible with VLMs including VILA, Qwen-VL, InternVL-3, and LLaVA-OneVision. The same VLM is used for both frame analysis and question answering.

## Key Experimental Results

### Main Results (Multi-benchmark comparison with frame selection methods)

| VLM | Method | #Frames | Video-MME | MLVU | LVB |
|-----|--------|---------|-----------|------|-----|
| VILA-1.5-8B | Uniform | 8 | 48.9 | 44.7 | 47.9 |
| | MDP3 | 8 | 53.3 | 52.3 | 52.3 |
| | **A.I.R.** | 8 | **53.7** | **54.2** | **52.9** |
| QwenVL-2.5-7B | Uniform | 32 | 60.8 | 59.3 | 58.1 |
| | MDP3 | 32 | 63.8 | 66.2 | 60.0 |
| | **A.I.R.** | 32 | **High** | **High** | **High** |

### Efficiency Comparison

| Method | VLM Frames Analyzed | Adaptive |
|--------|-------------------|---------|
| Frame-Voyager | 128 (fixed) | No |
| VideoTree | 128 (fixed) | No |
| **A.I.R.** | **12–72 (adaptive)** | **Yes** |

### Key Findings
- A.I.R. consistently outperforms all frame selection baselines on Video-MME, MLVU, and LVB, and is compatible with four distinct VLM backbones.
- The advantage is most pronounced on long videos (MLVU: +10% vs. uniform sampling), where more redundant frames need to be skipped.
- The LDS step is critical: it recovers frames with low CLIP scores that are nonetheless judged relevant by the VLM (e.g., a "Buddhist temple" frame with CLIP score 0.4 that corresponds to the correct answer).
- Performance improvements also hold on short-video benchmarks (EgoSchema, NextQA), demonstrating the generality of the approach.

## Highlights & Insights
- **Two-stage "coarse filtering + refinement" philosophy**: Using cheap CLIP for first-pass filtering and expensive VLM for precise verification is a classical yet effective multi-stage filtering paradigm.
- **Signal-processing perspective of Interval Potential Ranking**: Evaluating intervals rather than individual frames via the Relevance × Complexity × Length triplet captures the information density of temporal regions.
- **Exponential step size in LDS**: Denser sampling near verified frames and sparser sampling farther away aligns naturally with the temporal locality assumption.

## Limitations & Future Work
- The initial CLIP similarity remains the foundation; if CLIP entirely misses an event region, LDS is unlikely to recover it.
- Multiple hyperparameters ($\gamma$, $d_{\min}$, $l_{\min}$, $C$, $I_{\max}$, $\alpha$, $\beta$, $D$, $c_{\text{len}}$) require tuning.
- Using the same VLM for both analysis and answering may be suboptimal, as the two tasks have different requirements.
- Evaluation is limited to multiple-choice QA; applicability to open-ended generation tasks remains unverified.

## Related Work & Insights
- **vs. MDP3/Q-Frame (CLIP-based methods)**: A.I.R. compensates for CLIP's inaccuracy through VLM-based fine-grained analysis, with a pronounced advantage on complex queries.
- **vs. Frame-Voyager/VideoTree (VLM-based methods)**: These methods analyze 128 frames in a single pass; A.I.R. analyzes at most 72 frames and typically terminates earlier via early stopping.
- **vs. training-based methods (SeViLA)**: A.I.R. requires no training and can be directly integrated with any VLM.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of iterative frame selection and LDS is creative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated with 4 VLMs, 5 benchmarks, multiple baselines, and rigorous efficiency analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation is clearly articulated, the pipeline diagram is intuitive, and the efficiency analysis is rigorous.
- Value: ⭐⭐⭐⭐ A practically useful frame selection framework, though frame selection itself may be superseded by future VLMs with longer context windows.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Wavelet-based Frame Selection by Detecting Semantic Boundary for Long Video Understanding](../../CVPR2026/video_understanding/wavelet-based_frame_selection_by_detecting_semantic_boundary_for_long_video_unde.md)
- [\[CVPR 2026\] HERBench: A Benchmark for Multi-Evidence Integration in Video Question Answering](../../CVPR2026/video_understanding/herbench_a_benchmark_for_multi-evidence_integration_in_video_question_answering.md)
- [\[CVPR 2026\] DIvide, then Ground: Adapting Frame Selection to Query Types for Long-Form Video Understanding](../../CVPR2026/video_understanding/divide_then_ground_adapting_frame_selection_to_query_types_for_long-form_video_u.md)
- [\[CVPR 2026\] EgoPointVQA: Gesture-Based Egocentric Video Question Answering](../../CVPR2026/video_understanding/egopointvqa_gesture_based_egocentric_video_qa.md)
- [\[NeurIPS 2025\] Tool-Augmented Spatiotemporal Reasoning for Streamlining Video Question Answering Task](../../NeurIPS2025/video_understanding/toolaugmented_spatiotemporal_reasoning_for_streamlining_vide.md)

</div>

<!-- RELATED:END -->

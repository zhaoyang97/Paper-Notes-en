---
title: >-
  [Paper Note] Wavelet-based Frame Selection by Detecting Semantic Boundary for Long Video Understanding
description: >-
  [CVPR2026][Video Understanding][frame selection] This paper proposes WFS-SB, a training-free frame selection framework that applies wavelet transforms to query-frame similarity signals for semantic boundary detection. Th…
tags:
  - "CVPR2026"
  - "Video Understanding"
  - "frame selection"
  - "long video understanding"
  - "wavelet transform"
  - "semantic boundary detection"
  - "large vision-language models"
  - "training-free"
date: 2026-05-08
content_hash: 2067352271e28a9e
---

# Wavelet-based Frame Selection by Detecting Semantic Boundary for Long Video Understanding

**Conference**: CVPR2026
**arXiv**: [2603.00512](https://arxiv.org/abs/2603.00512)  
**Code**: [MAC-AutoML/WFS-SB](https://github.com/MAC-AutoML/WFS-SB)  
**Area**: Video Understanding
**Keywords**: frame selection, long video understanding, wavelet transform, semantic boundary detection, large vision-language models, training-free

## TL;DR

This paper proposes WFS-SB, a training-free frame selection framework that applies wavelet transforms to query-frame similarity signals for semantic boundary detection. The video is segmented into semantically coherent segments, over which frame budgets are adaptively allocated and diversity-aware sampling is performed. WFS-SB substantially surpasses state-of-the-art methods on VideoMME, MLVU, and LongVideoBench.

## Background & Motivation

1. **Severe frame redundancy in long videos**: Long videos typically contain thousands of frames, yet LVLMs are constrained by limited context windows and computational resources. Processing all frames is infeasible, making frame selection a critical preprocessing step for deploying LVLMs on long videos.
2. **Existing methods neglect narrative structure**: Mainstream frame selection approaches retrieve only frames with the highest query relevance, producing discrete, unordered frame sets that fail to capture causal relationships and procedural progression (e.g., in a makeup tutorial, selecting scattered frames containing "eyes" loses the sequential flow of drawing eyeliner before shaping eyebrows).
3. **Semantic transitions are the key**: Effective video understanding requires not only knowing *which frames are relevant* but also capturing *when the narrative shifts*—i.e., the critical transition moments at semantic boundaries.
4. **High noise in similarity signals**: Query-frame ITM scores are corrupted by model uncertainty, cross-modal ambiguity, and visual artifacts (illumination changes, occlusion, camera motion), making high-frequency noise a serious obstacle for direct semantic boundary detection.
5. **Non-stationary and multi-scale signals**: The dynamic nature of video content causes the statistical properties of similarity signals to vary dramatically over time. Semantic segments span anywhere from a few frames to hundreds of frames, and global analysis tools such as the Fourier transform cannot simultaneously capture temporal location and frequency information.
6. **Shortcomings of alternative approaches**: Extending context windows incurs heavy computational overhead; video-to-text summarization loses critical visual details; training-based frame selection requires large annotated datasets and generalizes poorly.

## Method

### Overall Architecture

WFS-SB is a training-free, three-stage frame selection framework: (1) wavelet-based semantic boundary detection → (2) adaptive frame budget allocation → (3) diversity-aware intra-segment frame selection.

### Stage 1: Wavelet-based Semantic Boundary Detection

- **ITM signal construction**: The video is uniformly sampled at 1 FPS to obtain $N$ frames. The ITM head of BLIP-2 computes a query-frame matching score $s_t = \mathcal{M}(q, f_t)$ for each frame, forming a temporal similarity signal.
- **Adaptive multi-level wavelet decomposition**: Discrete Wavelet Transform (DWT) with the Daubechies-4 (db4) wavelet is applied to the ITM signal. The decomposition depth is set to $J = \max(1, \lfloor \log_2 N \rfloor - l)$ (with $l=3$), so that longer videos use deeper decompositions to focus on coarse-grained trends while shorter videos retain temporal detail.
- **Semantic change feature extraction**: Only the detail coefficients $d_J$ at the coarsest scale are retained; all remaining coefficients are zeroed before applying the Inverse DWT (IDWT) to reconstruct a clean semantic change signal $\tilde{s}_t$, naturally suppressing high-frequency noise from fine-scale coefficients.
- **Boundary detection**: The change intensity $c_t = |\tilde{s}_t|$ is computed, and peak detection with adaptive height and prominence thresholds is applied. The detected peak indices $\mathcal{B} = \{b_1, \ldots, b_M\}$ partition the video into $M+1$ semantically coherent segments.

### Stage 2: Adaptive Budget Allocation

- **Segment importance scoring**: A composite importance score is computed for each segment: $\text{Imp}(\mathcal{G}_i) = w_d \cdot \frac{|\mathcal{G}_i|}{N} + w_a \cdot \bar{s}_i + w_m \cdot s_i^{\max} + w_v \cdot \frac{\sigma_i^2}{\sigma_{\text{global}}^2}$, jointly considering segment duration, mean relevance, peak relevance, and internal diversity (weights 0.4/0.2/0.3/0.1).
- **Segment filtering**: Segments whose importance falls below $\tau = \text{mean}(\text{Imp}) - 1.2 \cdot \text{std}(\text{Imp})$ are discarded, concentrating the budget on salient content.
- **Softmax-weighted allocation**: A softmax over the remaining segments' importance scores yields allocation proportions for the total frame budget $K$; remainders are greedily assigned to segments with the largest fractional parts.

### Stage 3: Diversity-aware Intra-segment Selection

- For each segment, the frame with the highest relevance score is first selected as an anchor.
- Remaining frames are iteratively selected via localized Maximal Marginal Relevance (MMR): $t^* = \arg\max_{t \in \mathcal{G}_i \setminus \mathcal{T}_i} [\lambda \cdot s_t - (1-\lambda) \cdot \max_{t' \in \mathcal{T}_i} \text{sim}(f_t, f_{t'})]$ (with $\lambda = 0.5$), balancing query relevance against visual diversity.
- Localizing MMR within each segment prevents visually similar frames across segments from suppressing one another, ensuring representative frames are obtained independently per segment.

### Loss & Training

The method is training-free and involves no loss function. All hyperparameters use unified default values across all LVLM backbones, all benchmarks, and all frame budgets, requiring no task-level tuning. Core hyperparameter settings: db4 wavelet, drift factor $l=3$, importance weights $(w_d, w_a, w_m, w_v)=(0.4, 0.2, 0.3, 0.1)$, filtering factor $\eta=1.2$, MMR parameter $\lambda=0.5$. Hyperparameter sensitivity experiments show that VideoMME fluctuates by only 0.4% for $\lambda \in [0.3, 0.7]$, demonstrating strong robustness.

## Key Experimental Results

### Main Results: Multi-benchmark, Multi-model Comparison

| Model | Method | Frames | VideoMME (Δ) | MLVU (Δ) | LVB (Δ) |
|-------|--------|--------|--------------|----------|---------|
| LLaVA-Video-7B | AKS | 8 | 60.1 (+3.9) | 64.2 (+6.8) | 59.6 (+4.7) |
| LLaVA-Video-7B | **WFS-SB** | 8 | **61.7 (+5.5)** | **66.9 (+9.5)** | **61.1 (+6.2)** |
| Qwen2.5-VL-7B | A.I.R. | ≤32 | 65.0 (+4.2) | 67.5 (+8.2) | 61.4 (+3.3) |
| Qwen2.5-VL-7B | **WFS-SB** | 32 | **64.4 (+3.2)** | **70.4 (+10.7)** | **64.4 (+5.5)** |
| InternVL3-8B | A.I.R. | ≤32 | 68.2 (+2.6) | 74.5 (+6.1) | 62.8 (+4.5) |
| InternVL3-8B | **WFS-SB** | 32 | 67.4 (+1.8) | **74.8 (+6.4)** | **62.9 (+4.4)** |

Average gains across 4 LVLMs: VideoMME +3.9%, MLVU +8.8%, LVB +5.4%.

### Ablation Study

| Configuration | VideoMME | MLVU |
|---------------|----------|------|
| Uniform Sampling | 57.7 | 56.2 |
| **WFS-SB (full)** | **61.9** | **67.9** |
| w/o DWT (local minima) | 60.8 | 64.6 (−3.3) |
| w/o DWT (gradient) | 61.2 | 66.8 |
| w/o adaptive budget (uniform allocation) | 61.6 | 67.4 |
| w/o MMR (top-K) | 60.9 | 66.7 |
| w/o MMR (uniform sampling) | 59.2 | 62.7 |

- The wavelet transform is the most critical component for MLVU gains (replacing it with local minima reduces MLVU by 3.3%), validating the core hypothesis that multi-scale decomposition suppresses noise.
- Model scale ablation (Qwen2.5-VL 3B→72B): consistent gains are observed across all scales (+2.2%~+4.3%), indicating that the benefits are orthogonal to model size.
- Wavelet family robustness: Db4/Db8/Sym4/Bior3.3 all achieve 61.9% on VideoMME; only Haar is slightly lower (61.3%), demonstrating that multi-scale decomposition itself—rather than a specific basis function—is the key factor.
- Computational overhead: ITM extraction takes 19.4s; wavelet processing and MMR together add only 0.7s, making the total additional overhead negligible.

### Frame Budget Sensitivity

Testing $K \in \{8, 16, 32, 64\}$ on 4 backbones on VideoMME, WFS-SB consistently outperforms uniform sampling at all budgets (+1.2%~+5.5%), with larger gains at smaller budgets ($K=8, 16$), demonstrating that semantic boundary awareness is most valuable under tight frame budgets.

## Highlights & Insights

- **Novel perspective**: The paper reframes frame selection from "select the most relevant frames" to "detect semantic turning points," redefining the problem from a signal processing perspective. The application of wavelet transforms to video frame selection is innovative.
- **Precise three-property analysis**: The ITM signal is analyzed along three dimensions—non-stationarity, multi-scale structure, and low signal-to-noise ratio—from which the necessity of wavelet transforms is naturally derived, forming a complete and convincing argumentative chain.
- **Training-free, plug-and-play**: No fine-tuning or architectural modifications are required; WFS-SB serves directly as a preprocessing module for LVLMs and is compatible with diverse backbones.
- **Unified hyperparameters**: All experiments use the same default parameters with no per-model, per-dataset, or per-budget tuning, offering strong practical utility.
- **Significant gains**: Improvements are most pronounced under tight frame budgets ($K=8$, MLVU +9.5%), confirming that semantic boundary awareness is most valuable under resource constraints.

## Limitations & Future Work

- **ITM computation bottleneck**: BLIP-2 ITM signal extraction accounts for 79% of total runtime (19.4s), which remains a burden for real-time applications; acceleration via batching, quantization, or distillation is warranted.
- **Dependence on an external VLM**: The quality of semantic signals is bounded by the ITM model's capability; different VLM scorers perform unevenly (CLIP-ViT-B is weaker than BLIP-based variants).
- **Fixed 1 FPS pre-sampling**: The fixed pre-sampling rate may generate an excessive number of candidate frames for very long videos (hour-scale), increasing ITM computation, or may miss critical frames in fast-action videos.
- **Single-query-driven**: Boundary detection is conditioned on a single query, limiting applicability to multi-turn dialogue or query-free generic video summarization; future work could explore query-free adaptive semantic segmentation.
- **Manually set wavelet parameters**: Although experiments show insensitivity to the drift factor $l=3$ and the weight combination, no automatic selection mechanism is provided.
- **Audio modality not considered**: The method relies solely on visual-text matching scores and does not exploit audio information (e.g., dialogue transitions, background music changes) to assist semantic boundary detection.
- **Segment filtering may discard relevant content**: The statistical-threshold-based filtering ($\tau$) may erroneously remove segments that have low relevance scores yet are narratively necessary in certain edge cases.

## Related Work & Insights

- **vs. KFC/BOLT/AKS**: These methods focus on frame-level relevance sampling and neglect video narrative structure. WFS-SB consistently outperforms them across all settings (e.g., vs. AKS: VideoMME +1.6%, MLVU +2.7%, LVB +1.5%).
- **vs. Frame-Voyager/FrameOracle (training-based)**: WFS-SB requires no training data or annotations, achieves superior performance, and offers better generalizability at lower deployment cost.
- **vs. A.I.R. (iterative inference-based)**: A.I.R. requires multiple rounds of LVLM inference with greater computational overhead; WFS-SB completes selection in a single forward pass and achieves better results on MLVU and LVB.
- **vs. MDP3 (Markov decision-based)**: WFS-SB surpasses MDP3 on Qwen2.5-VL by 4.2% on MLVU and 4.4% on LVB, demonstrating that semantic structure segmentation outperforms sequential decision modeling.
- **vs. uniform sampling**: WFS-SB comprehensively outperforms the uniform baseline (VideoMME +4.2%, MLVU +11.7%), with larger margins at tighter frame budgets.

## Rating

- Novelty: ⭐⭐⭐⭐ — Applying wavelet transforms to detect semantic boundaries for frame selection is a genuinely novel perspective; the intersection of signal processing and video understanding is highly inspiring.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 4 LVLMs, 3 benchmarks, multiple frame budgets, multiple model scales, multiple wavelet families, multiple VLM scorers, and component-wise ablations; extremely comprehensive.
- Writing Quality: ⭐⭐⭐⭐ — Problem motivation is clear; the necessity of wavelet transforms is derived persuasively from three signal properties (non-stationarity, multi-scale structure, low SNR).
- Value: ⭐⭐⭐⭐ — Training-free, plug-and-play, unified hyperparameters, and significant gains directly advance the practical deployment of long video understanding.
- Overall: ⭐⭐⭐⭐ — The method is elegant and concise, the experiments are rigorous, and the core insight (semantic boundaries > frame-level relevance) is thoroughly validated across 3 benchmarks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] DIvide, then Ground: Adapting Frame Selection to Query Types for Long-Form Video Understanding](divide_then_ground_adapting_frame_selection_to_query_types_for_long-form_video_u.md)
- [\[CVPR 2026\] VSI: Visual-Subtitle Integration for Keyframe Selection to Enhance Long Video Understanding](vsi_visual-subtitle_integration_for_keyframe_selection_to_enhance_long_video_un.md)
- [\[ICLR 2026\] A.I.R.: Adaptive, Iterative, and Reasoning-based Frame Selection For Video Question Answering](../../ICLR2026/video_understanding/air_enabling_adaptive_iterative_and_reasoning-based_frame_selection_for_video_qu.md)
- [\[CVPR 2026\] Question-guided Visual Compression with Memory Feedback for Long-Term Video Understanding](question-guided_visual_compression_with_memory_feedback_for_long-term_video_unde.md)
- [\[CVPR 2026\] LongVideo-R1: Smart Navigation for Low-cost Long Video Understanding](longvideo-r1_smart_navigation_for_low-cost_long_video_understanding.md)

</div>

<!-- RELATED:END -->

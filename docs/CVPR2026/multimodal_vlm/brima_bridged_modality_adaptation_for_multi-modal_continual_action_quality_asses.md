---
title: >-
  [Paper Note] BriMA: Bridged Modality Adaptation for Multi-Modal Continual Action Quality Assessment
description: >-
  [CVPR2026][Multimodal VLM][Action Quality Assessment] BriMA is proposed to address the non-stationary modality imbalance problem in multi-modal continual action quality assessment (AQA) via memory-guided bridging imputat…
tags:
  - "CVPR2026"
  - "Multimodal VLM"
  - "Action Quality Assessment"
  - "Continual Learning"
  - "Missing Modality"
  - "Multi-Modal Fusion"
  - "Memory Replay"
date: 2026-05-08
content_hash: 55432a237d1371af
---

# BriMA: Bridged Modality Adaptation for Multi-Modal Continual Action Quality Assessment

**Conference**: CVPR2026  
**arXiv**: [2602.19170](https://arxiv.org/abs/2602.19170)  
**Code**: [github.com/ZhouKanglei/BriMA](https://github.com/ZhouKanglei/BriMA)  
**Area**: Multi-Modal VLM  
**Keywords**: Action Quality Assessment, Continual Learning, Missing Modality, Multi-Modal Fusion, Memory Replay

## TL;DR
BriMA is proposed to address the non-stationary modality imbalance problem in multi-modal continual action quality assessment (AQA) via memory-guided bridging imputation and modality-aware replay optimization, achieving an average improvement of 6–8% in correlation coefficient and a 12–15% reduction in error across three benchmarks.

## Background & Motivation
1. Action Quality Assessment (AQA) is widely applied in sports analysis, rehabilitation evaluation, and skill assessment; multi-modal methods leveraging visual and kinematic cues have achieved notable progress.
2. In real-world deployments, sensor failures and missing annotations cause non-stationary modality imbalance—modality availability varies over time.
3. Existing multi-modal AQA methods assume complete and stable input modalities; any modality absence leads to significant performance degradation.
4. Existing continual AQA methods focus solely on task-level forgetting and do not handle modality-level dynamic changes.
5. Simple imputation, retrieval-based completion, and generative synthesis all fail to preserve the geometric structure critical for AQA scoring, disrupting ranking consistency.
6. The fine-grained score sensitivity of AQA makes it fundamentally different from conventional missing modality reconstruction problems.

## Method

### Overall Architecture
At each training session, BriMA: (1) completes missing modality features via the MBI module; (2) fuses all modality features for score prediction; (3) selects informative samples for replay using the MRO module to counteract distribution drift.

### Key Designs

**MBI (Memory-Guided Bridging Imputation)**:
1. **Candidate Retrieval**: For a missing modality $m$, cosine similarity is used to retrieve $K$ structurally aligned exemplar features from the memory buffer $\mathcal{B}_{t-1}$:
    $s_{j,t'} = \frac{\langle \mathbf{z}_{i,t}^{\mathcal{O}}, \mathbf{z}_{j,t'}^{\mathcal{O}} \rangle}{\|\mathbf{z}_{i,t}^{\mathcal{O}}\| \|\mathbf{z}_{j,t'}^{\mathcal{O}}\|}$
2. **Task Indicator**: A binary mask $\mathbf{r}_{i,t}$ identifies missing modalities, coupled with a learnable task embedding $\mathbf{p}_t^m$ to provide task-specific conditioning.
3. **Bridging Residual**: A residual correction is learned rather than synthesizing complete features:
    $\tilde{\mathbf{z}}_{i,t}^m = \bar{\mathbf{z}}_{i,t}^m + \Delta\mathbf{z}_{i,t}^m = \bar{\mathbf{z}}_{i,t}^m + B_\Theta(\mathbf{z}_{i,t}^{\mathcal{O}}, \bar{\mathbf{z}}_{i,t}^m, \mathbf{c}_t^m)$

**MRO (Modality-Aware Replay Optimization)**:
- Dynamically prioritizes replay samples based on modality distortion and score drift.
- Maintains a representative sample buffer with reliable modalities and balanced score coverage.
- Counteracts cross-task distribution drift through replay.

### Loss & Training
$$\min_{\theta_f, \theta_g} \mathcal{L}_{score} + \lambda_{mem}\mathcal{L}_{mem} + \lambda_{rec}\mathcal{L}_{rec}$$
where $\mathcal{L}_{score}$ is the MSE scoring loss, $\mathcal{L}_{mem}$ is the memory replay regularization loss, and $\mathcal{L}_{rec} = \|\tilde{\mathbf{z}} - \mathbf{z}\|_2^2$ is the feature reconstruction loss.

## Key Experimental Results

### Main Results: RG Dataset Comparison ($\beta=10\%$ Missing Rate)

| Method | Publication | SRCC↑ Avg | MSE↓ Avg | RL2↓ Avg |
|------|------|-----------|----------|----------|
| ST-MLAVL | CVPR'25 | 0.599 | 9.94 | 3.558 |
| EWC | PNAS'17 | 0.605 | 10.26 | 3.709 |
| MER | ICLR'19 | 0.722 | 6.77 | — |
| **BriMA** | **Ours** | **Best (~0.76+)** | **Lowest** | **Lowest** |

### Ablation Study

| Component | SRCC Change | MSE Change |
|------|----------|---------|
| w/o MBI (zero-fill) | Significant drop | Significant rise |
| w/o MRO (random replay) | Drop | Rise |
| w/o residual (direct generation) | Drop | Rise |
| Full BriMA | Best | Best |

### Cross-Dataset Performance
Across three datasets—RG, Fis-V, and FS1000—BriMA achieves average improvements of:
- Rank correlation: +6.1%, +8.3%, +1.4%
- Error reduction: −12.7%, −15.3%, −6.4%
- Relative error reduction: −13.9%, −14.1%, −5.2%

### Key Findings
- The residual learning strategy is more stable than direct feature generation, particularly under limited supervision signals.
- Modality-aware replay selection is substantially more effective than random replay.
- Both MBI and MRO contribute meaningfully to overall performance gains.

## Highlights & Insights
- This work is the first to systematically define and address the non-stationary modality imbalance problem in multi-modal continual AQA.
- Residual bridging is more conservative and safer than complete reconstruction—particularly important in score-sensitive tasks.
- The memory-guided retrieval combined with residual correction demonstrates strong capability in preserving the structure of the scoring manifold.

## Limitations & Future Work
- The framework assumes that the missing modality pattern is known (i.e., $\mathcal{M}_{i,t}$ is observable during training); automatic detection of missing modalities is not explored.
- Experiments are limited to two-modality scenarios; scalability to three or more modalities remains to be verified.
- The impact of memory buffer size on performance is not sufficiently discussed.

## Related Work & Insights
- **Distinction from general missing modality learning**: BriMA is specifically designed for AQA score sensitivity, avoiding the scoring manifold corruption introduced by general-purpose methods.
- **Distinction from continual AQA methods** (e.g., Fs-Aug, MAGR): These methods only address task-level non-stationarity and do not resolve modality-level dynamics.
- **Inspiration**: The residual bridging idea is transferable to other tasks requiring modality completion under high output precision requirements.

## Rating
- Novelty: ⭐⭐⭐⭐ (novel problem formulation; MBI design is well-motivated)
- Experimental Thoroughness: ⭐⭐⭐⭐ (3 datasets, multiple missing rates, comprehensive ablations)
- Writing Quality: ⭐⭐⭐⭐ (clear problem formalization; consistent notation)
- Value: ⭐⭐⭐ (relatively niche application scenario, though the methodology has broader generalizability)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] MCMoE: Completing Missing Modalities with Mixture of Experts for Incomplete Multimodal Action Quality Assessment](../../AAAI2026/multimodal_vlm/mcmoe_completing_missing_modalities_with_mixture_of_experts_for_incomplete_multi.md)
- [\[ICLR 2026\] VisJudge-Bench: Aesthetics and Quality Assessment of Visualizations](../../ICLR2026/multimodal_vlm/visjudge-bench_aesthetics_and_quality_assessment_of_visualizations.md)
- [\[CVPR 2026\] Decoupling Stability and Plasticity for Multi-Modal Test-Time Adaptation](decoupling_stability_and_plasticity_for_multi-modal_test-time_adaptation.md)
- [\[CVPR 2026\] FluoCLIP: Stain-Aware Focus Quality Assessment in Fluorescence Microscopy](fluoclip_stain-aware_focus_quality_assessment_in_fluorescence_microscopy.md)
- [\[ICLR 2026\] KeepLoRA: Continual Learning with Residual Gradient Adaptation](../../ICLR2026/multimodal_vlm/keeplora_continual_learning_with_residual_gradient_adaptation.md)

</div>

<!-- RELATED:END -->

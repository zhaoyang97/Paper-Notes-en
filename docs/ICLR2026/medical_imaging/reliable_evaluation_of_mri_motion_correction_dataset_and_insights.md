---
title: >-
  [Paper Note] Reliable Evaluation of MRI Motion Correction: Dataset and Insights
description: >-
  [ICLR 2026][Medical Imaging][Paper Note] Addressing the fundamental dilemma that "3D MRI motion correction methods cannot be reliably evaluated," this paper releases PMoC3D, a paired real-motion dataset, and proposes MoMRISim, a feature-space metric trained via self-supervision. By systematically auditing three evaluation paradigms—real-paired, simulated-moti
tags:
  - ICLR 2026
  - Medical Imaging
date: 2026-05-08
content_hash: 67e1bb4cff076ff9
---
# Reliable Evaluation of MRI Motion Correction: Dataset and Insights

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=5PY8HR2Zz6](https://openreview.net/forum?id=5PY8HR2Zz6)  
**Code**: https://github.com/MLI-lab/PMoC3D (Dataset: https://huggingface.co/datasets/mli-lab/PMoC3D)  
**Area**: Medical Imaging  
**Keywords**: MRI Motion Correction, Evaluation Benchmark, Paired Dataset, Perceptual Metrics, Self-supervised

## TL;DR
Addressing the fundamental dilemma that "3D MRI motion correction methods cannot be reliably evaluated," this paper releases PMoC3D, a paired real-motion dataset, and proposes MoMRISim, a feature-space metric trained via self-supervision. By systematically auditing three evaluation paradigms—real-paired, simulated-motion, and no-reference—the study concludes that "Real-Paired + MoMRISim" is the most reliable despite being imperfect, whereas simulated motion systematically overestimates algorithms, and no-reference metrics favor over-smoothed deep learning outputs.

## Background & Motivation

**Background**: MRI scans are time-consuming, and even slight head movement by subjects introduces inconsistencies in $k$-space, leading to motion artifacts such as blurring, ringing, and ghosting in reconstructed images, which can compromise clinical utility. Recent works focus on "retrospective" motion correction—estimating and correcting motion directly from acquired $k$-space or reconstructed images without external hardware. Since subject motion is inherently 3D, 3D motion correction holds the highest clinical value, with methods evolving from classical Alternating Optimization (AltOpt) to hybrid Deep Learning + Physics (MotionTTT), and end-to-end U-Nets (E2E Stacked U-Net).

**Limitations of Prior Work**: This field lacks a standard evaluation pipeline that "truly reflects performance," primarily because **motion-free ground truth is inherently unavailable**. The three existing routes have significant flaws: real-motion data captures authentic artifacts but lacks quantifiable ground truth; acquiring a motion-free reference scan requires complex registration and preprocessing, and the reference itself may contain residual motion. Simulated rigid motion is convenient for calculating reference metrics but fails to capture the complexity of non-rigid motion and requires fully sampled data (whereas 3D MRI is almost always undersampled). No-reference gradient-based metrics (e.g., Tenengrad, AES) correlate poorly with human-perceived image quality.

**Key Challenge**: Quantitative evaluation requires ground truth, which does not exist in perfect form for real-world scenarios. Replacing real data with simulation provides ground truth but sacrifices authenticity, leading to distorted conclusions (e.g., methods that perform perfectly in simulation may leave obvious artifacts under real-world severe motion).

**Goal**: The paper addresses three sub-problems: (1) providing a paired dataset for real-world evaluation; (2) developing a metric that aligns with human judgment to distinguish artifact severity; and (3) systematically comparing evaluation routes to determine when they are credible or misleading.

**Key Insight**: Instead of developing a "stronger correction algorithm," the authors treat "how to evaluate" as the primary research object. Since the progress of the entire sub-field relies on evaluation, unreliable evaluation makes "progress" an illusion.

**Core Idea**: Anchor the evaluation using a real-paired dataset (PMoC3D) combined with a feature metric (MoMRISim) trained using "motion severity" as a self-supervised signal. Use human expert scores (PMAS) as the gold standard to audit the reliability of the three evaluation paradigms.

## Method

### Overall Architecture

This is a methodological work on "how to reliably evaluate motion correction" rather than a paper proposing a new correction algorithm. It consists of three components: **a real paired dataset**, **two new metrics**, and **a systematic audit process**.

On the data side, the authors collected PMoC3D: 8 subjects, each with 1 motion-free scan and 3 motion scans of varying severity, preserving raw $k$-space data. To label the severity of each motion scan, the authors used L1-wavelet reconstruction without correction for pairwise comparisons, fitting a Bradley–Terry model to obtain the continuous Perceptual Motion Artifact Score (PMAS).

For evaluation, three routes are analyzed: **Real-Paired Evaluation** (using PMoC3D motion-free scans as reference), **Simulated Motion Evaluation** (injecting motion into fully sampled Calgary-Campinas data), and **No-Reference Evaluation** (scoring without a reference). Each route includes a new metric: the reference-based MoMRISim and the no-reference VLM score.

For auditing, the authors used expert PMAS as the gold standard, calculating Spearman rank correlation to measure consistency between metrics and human perception across representative algorithms (AltOpt, MotionTTT, E2E).

### Key Designs

**1. PMoC3D Paired Dataset: Releasing Real Motion + Raw $k$-space**

Real evaluation lacks data, and previous public datasets often provide only magnitude images. Algorithms with explicit motion estimation require raw $k$-space. The authors acquired 4 scans (1 motion-free + 3 motion) from 8 healthy subjects on a 3.0T scanner. Motion was instruction-guided (turning, nodding, shaking, chin-to-chest) with varying types and timing to create diverse artifacts. Parameters included 1mm isotropic resolution, undersampling factor $\mu=4.94$, and quasi-random trajectory. Data includes coil sensitivity maps, trajectories, and motion timestamps, making real-paired evaluation possible for $k$-space-based methods.

**2. PMAS: Turning Pairwise Comparisons into Continuous Severity**

To audit "human-like" metrics, a gold standard is needed. Direct absolute scoring is unstable, so the authors used pairwise comparisons. Experts compared L1 reconstructions to determine which had heavier artifacts, yielding preference probabilities $p(i>j)$. These were fitted to a Bradley–Terry model to estimate the latent severity parameter $\beta_i$:

$$\mathrm{PMAS} = \arg\max_{\beta} \sum_{i \neq j} p(i>j)\,\log\!\left(\frac{\exp(\beta_i)}{\exp(\beta_i)+\exp(\beta_j)}\right)$$

Higher $\beta_i$ indicates more severe artifacts. This score serves as both a "difficulty label" and a gold standard for auditing other metrics.

**3. MoMRISim Metric: Self-supervised Feature Space via "Motion Severity"**

Pixel-level PSNR/SSIM correlate only moderately with perception, and generic perceptual metrics like DreamSim are not tailored for MRI artifacts. The key observation is that motion severity is a free, precise supervisory signal. An encoder $f(\cdot)$ is trained such that the distance to the reference in feature space reflects severity: mildly corrupted reconstructions should be closer to the reference than severely corrupted ones. Triplets were constructed by injecting synthetic rigid motion into the Calgary-Campinas data. The model learns artifact features invariant to reconstruction style by randomly using L1 or U-Net reconstructions during training. At inference, it calculates the cosine distance:

$$\mathrm{MoMRISim}(R,X) = 1 - \mathrm{CosineSimilarity}\big(f(R), f(X)\big)$$

**4. VLM Score and Systematic Audit: Locating Failure Boundaries**

Standard no-reference metrics correlate poorly with human judgment. The authors proposed the VLM score, where vision-language models (GPT-4o, Qwen2.5-VL-Max) prompt-score artifacts from 0 to 3. While better than gradient metrics, auditing reveals they still favor "over-smoothed" E2E U-Net outputs. By auditing all three routes, the authors defined reliability boundaries: real-paired evaluation is reliable for medium-to-heavy motion but distorted for light motion (due to residual motion in the reference); simulation systematically overestimates performance; no-reference evaluation is generally unreliable.

### Loss & Training
MoMRISim's objective is triplet ranking: for (Reference $R$, Mild $X_{\text{mild}}$, Severe $X_{\text{severe}}$), $f$ is constrained so $X_{\text{mild}}$ is closer to $R$ than $X_{\text{severe}}$. Supervision comes from the known severity order of synthetic motion. VLM score requires no training and relies on zero-shot inference.

## Key Experimental Results

### Main Results: Rank Correlation with Human PMAS

| Metric | Type | Spearman $\rho$ with PMAS | Conclusion |
|------|------|------|------|
| MoMRISim | Reference (Ours) | **0.92** | Highly consistent with humans |
| PSNR | Reference (Pixel) | 0.64 | Moderately consistent |
| VLM score (GPT-4o) | No-Reference (Ours) | 0.44 | Low alignment with humans |

MoMRISim maintains high correlation across different algorithm types (AltOpt / MotionTTT / E2E), proving it is a reliable proxy for expert evaluation.

### Comparison of Evaluation Paradigms (Table 1 Overview)

| Paradigm | Data Source | Reference Type | Human Alignment | Main Defect | Conclusion |
|----------|--------|----------|----------|----------|------|
| Real Paired | PMoC3D | Paired Scan | High | Imperfect Reference | Most reliable |
| Simulated Motion | Calgary-Campinas | Ground Truth | — | Fails to capture complexity | Overestimates performance |
| No-Reference | PMoC3D | None | Low | Favors over-smoothing | Unreliable |

### Key Findings
- **MoMRISim is the primary contribution**: It increases correlation with human judgment from PSNR’s 0.64 to 0.92 without requiring human labels.
- **Simulation Overestimates Performance**: Under equivalent severity, simulated reconstructions appear cleaner than real ones. Some methods yield "zero error" in simulation but leave clear ringing in real severe motion.
- **Light Motion is a Weakness for Real Evaluation**: In mild motion cases, the corrected image may appear cleaner than the "motion-free" reference (which carries its own slight artifacts), leading to a paradox where reference-based evaluation fails.
- **No-Reference Metrics Favor Over-smoothing**: VLM scores reward E2E U-Nets that erase anatomical details to suppress artifacts, whereas PMAS correctly identifies these as severe artifacts.

## Highlights & Insights
- **Severity as Free Supervision**: MoMRISim bypasses massive human labeling by using the synthetic severity order as relative labels—a concept transferable to any task with controllable degradation levels.
- **Auditing the Foundation**: Instead of pursuing a new network architecture, the paper proves that common simulation-based evaluations are misleading, correcting a methodological bias in the field.
- **Bradley–Terry for Gold Standards**: Using pairwise comparisons + BT models is more robust than absolute scoring and allows for inter-rater reliability checks.
- **Raw k-space Accessibility**: Releasing raw measurements allows researchers to reproduce and evaluate explicit motion modeling methods on real paired data for the first time.

## Limitations & Future Work
- **Small Sample Size**: Only 8 healthy subjects (Brain T1). Generality needs validation across larger cohorts and different anatomical sites.
- **Imperfect Reference Scans**: The "motion-free" reference may contain residual motion, leading to distortion in mild motion evaluation.
- **Closed-source VLM Dependency**: Version drift in models like GPT-4o impacts reproducibility.
- **Future Direction**: Extending MoMRISim training signals from purely synthetic motion to include real motion severity (PMAS) via hybrid supervision.

## Related Work & Insights
- **vs DreamSim/DISTS**: MoMRISim uses motion severity for self-supervision rather than generic semantic similarity, achieving higher alignment for MRI artifacts (0.92 vs 0.64).
- **vs Gradient-based No-ref Metrics**: While VLM score improves alignment over Tenengrad, it introduces a bias toward over-smoothing.
- **vs Magnitude-only Datasets**: PMoC3D provides raw measurements, enabling evaluation of methods like AltOpt and MotionTTT on real data rather than just simulation.

## Rating
- Novelty: ⭐⭐⭐⭐ Treating "evaluation methodology" as a research object is timely; MoMRISim is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Audits three paradigms across three algorithm types; sample size is slightly limited.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear problem definition and honest analysis of paradigm trade-offs.
- Value: ⭐⭐⭐⭐⭐ The dataset-metric-audit triplet provides a reproducible foundation for 3D MRI motion correction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Unsupervised Motion-Compensated Decomposition for Cardiac MRI Reconstruction via Neural Representation](../../AAAI2026/medical_imaging/unsupervised_motion-compensated_decomposition_for_cardiac_mri_reconstruction_via.md)
- [\[CVPR 2026\] Building Robust Vision Encoders for Cross-Dataset Evaluation in Immunofluorescent Microscopy](../../CVPR2026/medical_imaging/building_robust_vision_encoders_for_cross-dataset_evaluation_in_immunofluorescen.md)
- [\[CVPR 2026\] Prospective Dynamic 3D MRI Reconstruction via Latent-Space Motion Tracking from Single Measurement](../../CVPR2026/medical_imaging/prospective_dynamic_3d_mri_reconstruction_via_latent-space_motion_tracking_from_.md)
- [\[ICLR 2026\] SEED: Towards More Accurate Semantic Evaluation for Visual Brain Decoding](seed_towards_more_accurate_semantic_evaluation_for_visual_brain_decoding.md)
- [\[ICLR 2026\] OpenPros: A Large-Scale Dataset for Limited View Prostate Ultrasound Computed Tomography](openpros_a_large-scale_dataset_for_limited_view_prostate_ultrasound_computed_tom.md)

</div>

<!-- RELATED:END -->

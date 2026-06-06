---
title: >-
  [Paper Note] Let EEG Models Learn EEG
description: >-
  [ICML 2026][Image Generation][EEG generation] JET redefines multi-channel EEG generation as a "continuous trajectory on the neural manifold…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "EEG generation"
  - "Conditional Flow Matching"
  - "Transformer"
  - "Spectral Fidelity"
  - "Structured Constraints"
date: 2026-05-08
content_hash: 51e0aebb332a7b03
---

# Let EEG Models Learn EEG

**Conference**: ICML 2026  
**arXiv**: [2605.21280](https://arxiv.org/abs/2605.21280)  
**Code**: https://y-research-sbu.github.io/JET/ (Project Page)  
**Area**: Medical Imaging / Neural Signal Generation / Flow Matching  
**Keywords**: EEG generation, Conditional Flow Matching, Transformer, Spectral Fidelity, Structured Constraints

## TL;DR
JET redefines multi-channel EEG generation as a "continuous trajectory on the neural manifold," utilizing Conditional Flow Matching (CFM) with a standard Transformer to directly model raw waveforms. By incorporating three structured constraints specifically designed for EEG spectrum, stationarity, and statistics, JET reduces the TS-FID by over 40% across three clinical TUH benchmarks compared to strong baselines.

## Background & Motivation

**Background**: EEG foundation models have developed rapidly in recent years (BrainBERT/Brant/Neuro-GPT/EEGPT/CbraMod, etc.). However, high-quality clinical EEG data is limited by privacy and annotation costs, being several orders of magnitude smaller than text or image data. Therefore, reliable "native EEG generation" is required as a prerequisite for large-scale neural modeling.

**Limitations of Prior Work**: Existing EEG generators typically rely on GANs (EEG-GAN), discrete denoising diffusion, or autoregressive modeling after signal tokenization (MEG-GPT, GPT2MEG). The training objectives of these methods facilitate local reconstruction under isotropic Gaussian noise assumptions, leading to severe spectral bias, monotonous waveform repetition in long sequences, and an inability to cover pathological large-amplitude events.

**Key Challenge**: EEG signals are fundamentally continuous biological time series characterized by $1/f^{\chi}$ power-law spectra, non-stationarity, and heavy tails. Mainstream generation paradigms (discrete denoising + Gaussian priors) excel at minimizing local mean squared error, resulting in a systemic mismatch across frequency, temporal, and statistical dimensions. Small errors accumulate over sampling steps, destroying global structure.

**Goal**: (1) Formalize EEG generation as a continuous dynamical process rather than discrete denoising steps; (2) Design a backbone capable of capturing long-range dependencies and dynamic inter-channel interactions; (3) Incorporate "EEG-aware" structural constraints into the training objective to maintain EEG invariants in terms of geometry and statistics within the flow field.

**Key Insight**: Brain activity evolves smoothly in a high-dimensional state space (the neural manifold hypothesis). Thus, generation should follow this continuous trajectory rather than repeated noisy/denoisy cycles. Conditional Flow Matching (CFM) provides a continuous alternative by "learning a vector field that transports a prior to the data."

**Core Idea**: Directly perform Conditional Flow Matching on raw multi-channel EEG data. A pure DiT/JiT-style Transformer is used to learn the time-varying vector field $\mathbf{v}_\theta(\mathbf{x}_t,t,c)$, while explicitly incorporating physical EEG characteristics (robust reconstruction, statistical consistency, and time-frequency structure) into the loss function.

## Method

### Overall Architecture
JET processes multi-channel EEG segments of the form $\mathbf{X}\in\mathbb{R}^{C\times T}$. During training, $\mathbf{x}_1$ is sampled from the data and $\mathbf{x}_0$ from $\mathcal{N}(\mathbf 0,\mathbf I)$. A linear interpolation path is defined as $\mathbf{x}_t = t\mathbf{x}_1 + (1-t)\mathbf{x}_0$, with the target vector field being $\mathbf{u}_t = \mathbf{x}_1 - \mathbf{x}_0$. The Transformer $f_\theta$ takes $(\mathbf{x}_t, t, c)$ as input and outputs the predicted vector field $\mathbf{v}_\theta$, trained alongside three structured constraints. During inference, starting from Gaussian noise, the ODE $\mathrm{d}\mathbf{x}_t/\mathrm{d}t = \mathbf{v}_\theta(\mathbf{x}_t,t,c)$ is solved until $t=1$ to obtain synthetic EEG. This is used for data augmentation or filling rare categories downstream.

### Key Designs

1. **Generation Paradigm of Raw Waveform + Conditional Flow Matching**:
    - **Function**: Defines EEG synthesis as a "continuous vector field from a noise distribution to a data distribution," bypassing discrete denoising steps.
    - **Mechanism**: Utilizes Conditional Flow Matching (Lipman et al.) to perform vector field regression on a linear interpolation path, where the loss simplifies to $\ell_{\text{CFM}} = \mathbb{E}_t \|\mathbf{v}_\theta(\mathbf{x}_t,t,c) - (\mathbf{x}_1 - \mathbf{x}_0)\|$. Inference involves a single ODE integration, which is more efficient than the multi-step denoising in diffusion. Adaptive class-balanced sampling $p_i \propto 1/N_c^\alpha$ is introduced to mitigate the severe imbalance between normal background and rare epileptic events in TUH.
    - **Design Motivation**: Brain activity is a smoothly evolving continuous process; discrete noise schedules mismatch neural dynamics. A continuous flow field is more natural at the "global trajectory" level and faster than token autoregression (4.78s vs. 7.01s for Diffusion under equal conditions).

2. **Channel-Identity Preserving Transformer Backbone (JET)**:
    - **Function**: Learns long-range temporal dependencies and cross-channel interactions directly from raw multi-channel waveforms without time-frequency transformations or predefined adjacency graphs.
    - **Mechanism**: $\mathbf{X}$ is sliced along the time axis into non-overlapping patches of length $P$ to obtain $\mathbf{X}_p\in\mathbb{R}^{C\times N\times P}$. Unlike ViT, patch projection to a $D$-dimensional token preserves the channel dimension, resulting in a sequence of length $C\cdot N$. DiT/JiT-style Transformer blocks are stacked, where embeddings of time $t$ and class $c$ are injected into each block's scale/shift via adaLN. Ablations show $P=200$ is the optimal compromise for efficiency and fidelity.
    - **Design Motivation**: EEG is influenced by volume conduction and functional connectivity; electrodes exhibit long-range synchronization that drifts over time, violating the local assumptions of CNNs and the fixed topology of static graph models. Self-attention’s global receptive field combined with "channel-identity preserving" tokenization models both temporal and spatial structures.

3. **Three "EEG-aware" Structured Constraints**:
    - **Function**: Ensures the dynamics learned by the vector field align with real EEG invariants across frequency, statistical, and spatio-temporal dimensions.
    - **Mechanism**: The current state is first extrapolated to the destination estimate via $\hat{\mathbf{x}}_1 = \mathbf{x}_t + (1-t)\,\mathbf{v}_\theta$, followed by: (i) Laplacian prior reconstruction $\mathcal{L}_{\text{recon}} = \mathbb{E}_t \|\mathbf{x}_1 - \hat{\mathbf{x}}_1\|_1$ to resist EMG/electrode artifacts; (ii) First/Second moment consistency $\mathcal{L}_{\text{cons}} = \lambda_{\text{cons}} (\|\mu(\mathbf{x}_1) - \mu(\hat{\mathbf{x}}_1)\|_1 + \|\sigma(\mathbf{x}_1) - \sigma(\hat{\mathbf{x}}_1)\|_1)$ to prevent amplitude drift; (iii) Spatio-temporal structural term $\mathcal{L}_{\text{geo}} = \lambda_{\text{tv}}\frac{1}{T}\sum_t \|\nabla_t \hat{\mathbf{x}}_1\|_1 + \lambda_{\text{corr}} (1 - \rho(\mathbf{x}_1, \hat{\mathbf{x}}_1))$, where TV suppresses spurious high-frequency jitter and Pearson correlation $\rho$ preserves waveform morphology. The total loss is $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{cons}} + \mathcal{L}_{\text{geo}}$.
    - **Design Motivation**: Euclidean regression in standard flow matching corresponds to Gaussian likelihood, which is biased by sharp artifacts in EEG, underfits the power-law spectrum, and fails to constrain mean/variance drift in long sequences. These constraints address "robustness—statistical manifold—time-frequency structure" failure modes respectively.

### Loss & Training
The total objective is $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{cons}} + \mathcal{L}_{\text{geo}}$, where $\ell_1$ is used for reconstruction, $\ell_1$ moment matching for statistics, and TV + Pearson for geometry. The base distribution is fixed as $\mathcal{N}(\mathbf 0, \mathbf I)$. Sample weights are re-weighted by the inverse class frequency $1/N_c^\alpha$ to cover rare pathological events.

## Key Experimental Results

### Main Results
JET was compared against EEG-GAN and Vanilla Diffusion on three subsets of the TUH Corpus (TUAB Abnormal, TUEV Events, TUSZ Seizures, totaling 10k+ clinical sessions). Metrics include distributional fidelity (TS-FID), conditional consistency (Silhouette), and downstream augmentation gain ($\Delta$ Acc using a CbraMod classifier).

| Dataset | Metric | EEG-GAN | Vanilla Diffusion | JET (Ours) |
|---------|--------|---------|-------------------|------------|
| TUAB | TS-FID $\downarrow$ | 324.18 | 342.91 | **188.27** |
| TUAB | Silhouette $\uparrow$ | 0.786 | 0.710 | **0.995** |
| TUAB | $\Delta$ Acc $\uparrow$ | +0.000 | -0.002 | **+0.029** |
| TUEV | TS-FID $\downarrow$ | 448.65 | 415.82 | **235.86** |
| TUEV | Silhouette $\uparrow$ | 0.667 | 0.703 | **0.983** |
| TUEV | $\Delta$ Acc $\uparrow$ | -0.004 | -0.000 | **+0.032** |
| TUSZ | TS-FID $\downarrow$ | 274.37 | 300.47 | **151.27** |
| TUSZ | Silhouette $\uparrow$ | 0.891 | 0.746 | **0.987** |
| TUSZ | $\Delta$ Acc $\uparrow$ | +0.001 | +0.000 | **+0.017** |

JET achieved at least a 40% reduction in TS-FID across all datasets. A Silhouette score near 1 indicates nearly perfect intra-class consistency. Crucially, only JET's synthetic samples provided positive gains for the downstream CbraMod classifier, whereas baselines often degraded accuracy.

### Ablation Study
**Constraint Ablation (Table 4, TS-FID)**:

| Configuration | TUAB | TUEV | TUSZ | Note |
|---------------|------|------|------|------|
| $\mathcal{L}_{\text{recon}}$ only | 231.19 | 287.81 | 221.74 | Pure $\ell_1$, worst; validates Euclidean regression is insufficient |
| +$\mathcal{L}_{\text{cons}}$ | 228.87 | 281.70 | 209.99 | Moment matching prevents drift; small gain |
| +$\mathcal{L}_{\text{tv}}$ | 219.45 | 266.61 | 210.00 | Suppresses spurious high frequencies |
| +$\mathcal{L}_{\text{corr}}$ | 221.26 | 278.01 | 200.87 | Preserves waveform morphology |
| Full (All 4) | **188.27** | **235.86** | **151.27** | Complementary; best performance |

**Base Distribution Ablation**: Replacing the Gaussian prior with a degenerate $\delta(\mathbf 0)$ caused TS-FID to skyrocket from ~200 to over 1600+, validating the necessity of a non-degenerate base distribution for covering multi-modal EEG.

**Drift Analysis**: Using the linear slope of the RMS envelope and moment differences ($D_\mu, D_\sigma$) between start/end segments to measure spurious drift, JET's Wasserstein distance (0.015 / 0.021 / 0.018) was within 2× of the real-vs-real baseline. Baselines were 5–8× higher.

### Key Findings
- The three structured constraints are complementary: TV eliminates high-frequency noise, Pearson preserves morphology, and moment consistency prevents drift. Combining them halved the error across six physical diagnostic metrics (PSD slope, temporal envelope, Hjorth parameters).
- The base distribution must be non-degenerate: starting from a single point $\delta(\mathbf 0)$ causes the flow field to collapse into an ill-posed one-to-many mapping.
- Spectral analysis shows JET preserves the $\alpha$ peak (8–13Hz) while actively suppressing EMG noise above 15Hz, indicating selective "EEG-aware" modeling rather than simple marginal spectrum approximation.

## Highlights & Insights
- **Paradigm Shift**: Transitioning EEG generation from "discrete denoising" to "flow matching ODE" incorporates continuous trajectories—a physical reality—into the training objective, resulting in a cleaner technical path and faster inference.
- **Pedagogical Breakdown of Constraints**: Table 5 validates how $\mathcal{L}_{\text{cons}}/\mathcal{L}_{\text{tv}}/\mathcal{L}_{\text{corr}}$ each address specific failure modes using physical metrics not directly optimized, providing a clear template for future constraint design.
- **Transferability**: The combination of CFM, channel-identity preserving Transformers, and physical constraints is applicable to other biological time series like ECG and MEG by replacing constraints with signal-specific invariants (rhythm, HRV, stationarity, etc.).

## Limitations & Future Work
- Utilization was limited to the TUH corpus family; generalization across devices, sampling rates, and electrode standards has not been verified. Few-shot cross-dataset generation is a natural next step.
- TS-FID uses Fréchet distance of spectral features, which partially aligns with the model's frequency constraints, posing a slight risk of metric inflation. Independent subjective blinded reviews by clinicians would be beneficial.
- Current conditions $c$ are one-hot labels; subject metadata (age, placement, medication) are not utilized. Future work could introduce fine-grained control for personalized synthesis.

## Related Work & Insights
- **vs. EEG-GAN (Hartmann 2018)**: Early GAN approach; unstable adversarial training and poor mode coverage failed to preserve EEG spectra. JET bypasses adversarial objectives using continuous flow, reducing TS-FID by ~40%.
- **vs. Vanilla Diffusion (Song 2021)**: Discrete denoising has severe spectral bias and drift in long sequences. JET replaces discrete steps with CFM and physical constraints, turning downstream transfer gains from zero to positive.
- **vs. MEG-GPT / GPT2MEG (2024–2025)**: Autoregressive tokenization is fundamentally mismatched with continuous neural dynamics. JET maintains raw waveforms, skipping quantization loss.
- **vs. DiT / JiT (Peebles 2023; Li & He 2025)**: Methodological lineage—proves that plain Transformer + adaLN from vision successfuly transfers to EEG via "minimal inductive bias + high scalability."

## Rating
- Novelty: ⭐⭐⭐⭐ Systematically introduces CFM to EEG generation with "physical invariant alignment" constraints.
- Experimental Thoroughness: ⭐⭐⭐⭐ Solid results across three clinical benchmarks, physical diagnostics, and multiple ablations; restricted to the TUH family.
- Writing Quality: ⭐⭐⭐⭐ Clear chain of logic from motivation to method and constraints.
- Value: ⭐⭐⭐⭐ Provides a high-fidelity synthesis baseline for the data-scarce era of EEG foundation models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Step-Aware Residual-Guided Diffusion for EEG Spatial Super-Resolution](../../ICLR2026/image_generation/step-aware_residual-guided_diffusion_for_eeg_spatial_super-resolution.md)
- [\[CVPR 2026\] CognitionCapturerPro: Towards High-Fidelity Visual Decoding from EEG/MEG via Multi-modal Information and Asymmetric Alignment](../../CVPR2026/image_generation/cognitioncapturerpro_towards_highfidelity_visual_d.md)
- [\[ICLR 2026\] Concept-TRAK: Understanding how diffusion models learn concepts through concept-level attribution](../../ICLR2026/image_generation/concept-trak_understanding_how_diffusion_models_learn_concepts_through_concept-l.md)
- [\[ICLR 2026\] When Scores Learn Geometry: Rate Separations under the Manifold Hypothesis](../../ICLR2026/image_generation/when_scores_learn_geometry_rate_separations_under_the_manifold_hypothesis.md)
- [\[ICML 2026\] Adversarial Flow Models](adversarial_flow_models.md)

</div>

<!-- RELATED:END -->

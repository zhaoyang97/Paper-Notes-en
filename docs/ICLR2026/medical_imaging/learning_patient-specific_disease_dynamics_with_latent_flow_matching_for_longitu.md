---
title: >-
  [Paper Note] Learning Patient-Specific Disease Dynamics with Latent Flow Matching for Longitudinal Imaging Generation
description: >-
  [ICLR 2026][Medical Imaging][disease progression] This paper proposes the Δ-LFM framework, which employs an ArcRank loss to construct patient-specific…
tags:
  - "ICLR 2026"
  - "Medical Imaging"
  - "disease progression"
  - "flow matching"
  - "patient-specific"
  - "longitudinal MRI"
  - "ArcRank loss"
date: 2026-05-08
content_hash: 55d27d13ffdd13c8
---

# Learning Patient-Specific Disease Dynamics with Latent Flow Matching for Longitudinal Imaging Generation

**Conference**: ICLR 2026
**arXiv**: [2512.09185](https://arxiv.org/abs/2512.09185)  
**Code**: Unavailable  
**Area**: Medical Imaging / Disease Progression Modeling
**Keywords**: disease progression, flow matching, patient-specific, longitudinal MRI, ArcRank loss

## TL;DR
This paper proposes the Δ-LFM framework, which employs an ArcRank loss to construct patient-specific, temporally aligned trajectories in latent space (directionally consistent and monotonically increasing in magnitude). The framework extends the flow matching time range from $[0,1]$ to $[0,T]$ (actual time intervals) to enable prediction at arbitrary time points. Δ-LFM comprehensively outperforms eight baseline methods across three Alzheimer's longitudinal MRI benchmarks and introduces a progression-specific evaluation metric, Δ-RMAE.

## Background & Motivation

**Background**: Disease progression modeling is critical for early diagnosis and personalized treatment. The evolution from GANs to diffusion models has enabled higher-fidelity longitudinal medical image generation, yet most approaches capture only population-level trends.

**Limitations of Prior Work**: (1) Most models neglect individual heterogeneity—progression rates vary substantially across patients with the same disease; (2) the stochastic denoising process of diffusion models disrupts temporal continuity; (3) autoencoder latent spaces are misaligned across patients and uncorrelated with clinical severity indices; (4) conventional image quality metrics (PSNR/SSIM) are inflated in longitudinal settings—scans from the same patient are naturally highly similar, causing subtle disease-related changes to be masked by normal anatomy.

**Key Challenge**: Longitudinal image generation must simultaneously satisfy high fidelity (image quality) and high accuracy (correct progression direction); existing methods prioritize the former at the expense of the latter.

**Goal**: To construct a patient-specific generative framework in which the latent space is semantically meaningful, predictions at arbitrary time points are supported, and the direction of progression is correctly captured.

**Key Insight**: Disease progression in latent space can be modeled as a velocity field—flow matching naturally learns velocity fields from source to target, making it conceptually aligned with disease dynamics.

**Core Idea**: ArcRank constraints enforce a "straight-line" latent trajectory per patient (constant direction, increasing magnitude), and Δ-LFM advances along this line using real-valued time steps.

## Method

### Overall Architecture
The framework consists of two stages. Stage 1: a VAE with ArcRank loss constructs the patient-specific latent space. Stage 2: a 3D U-Net learns the velocity field within the latent space via flow matching, with conditional signals injected through AdaLN.

### Key Designs

1. **ArcRank Loss — Patient Trajectory Latent Alignment**
    - **Function**: Forces latent representations of the same patient at different time points to align along a specific direction, with magnitude monotonically increasing over time.
    - **Mechanism**: SVD decomposition $U\Sigma V^\top = \text{SVD}(\mathbf{z})$ is applied to the latent vector $\mathbf{z}$, where $U$ encodes direction (angle) and $\Sigma$ encodes magnitude (severity). The ArcRank loss is defined as:
     $$\mathcal{L}_{\text{ArcRank}} = \lambda_{\text{arc}} \sum_{i<j} |U_i - U_j| + \lambda_{\text{rank}} \sum_{i<j} \max(0, m - (\Sigma_j - \Sigma_i)), \quad t_i < t_j$$
     A pull term $\mathcal{L}_{\text{Pull}} = |\Sigma_j - \Sigma_i|$ is added to prevent excessive separation between adjacent time points.
    - **Design Motivation**: SVD jointly handles direction and magnitude in a unified manner, providing greater stability than separately applying cosine similarity and absolute value; stop-gradient is used to stabilize training.

2. **Δ-LFM — Temporally Semantic Flow Matching**
    - **Function**: Learns a patient-specific continuous-time velocity field in latent space, supporting prediction at arbitrary future time points.
    - **Mechanism**: The standard flow matching time range $[0,1]$ is extended to $[0,T]$, where $T = t_j - t_i$ denotes the actual interval in years. The target velocity is $v^*(i,j) = (\mathbf{z}_j - \mathbf{z}_i)/(t_j - t_i)$; at inference, integration proceeds with step size $\text{d}t = 0.01$: $\mathbf{z}_{i+\text{d}t} = \mathbf{z}_i + \text{d}t \cdot v_\theta(\mathbf{z}_i, t_i)$.
    - **Design Motivation**: Normalizing to $[0,1]$ discards actual temporal semantics; the $[0,T]$ parameterization directly supports queries such as "predict MRI three years from now."

3. **Δ-RMAE Evaluation Metric**
    - **Function**: Assesses the accuracy of the progression direction in generated images rather than absolute image quality.
    - **Mechanism**: A residual metric defined as $\Delta\text{-RMAE} = \frac{|\Delta_{\text{gt}} - \Delta_{\text{gen}}|}{(\frac{1}{2}(|\Delta_{\text{gt}}| + |\Delta_{\text{gen}}|))} \in [0, 2]$, where $\Delta = \mathbf{x}_T - \mathbf{x}_0$.
    - **Design Motivation**: Conventional PSNR/SSIM are inflated in longitudinal settings (a model that simply copies the baseline scan scores well); Δ-RMAE focuses exclusively on disease-induced change.

### Loss & Training
**Stage 1 (AE)**: Reconstruction loss + ArcRank, with $\lambda_{\text{arc}}=0.005$, $\lambda_{\text{rank}}=0.01$, and margin $m$. AdamW optimizer, lr=$10^{-3}$, batch size=2, 300 epochs.
**Stage 2 (FM)**: $\mathcal{L}_{\text{LFM}} = \sum_{i<j} |v_\theta(i,j) - v^*(i,j)|^2$. 3D U-Net, AdamW, lr=$3 \times 10^{-5}$, batch size=4, 200 epochs. Conditional signals (age, sex, clinical status) are injected via AdaLN.

## Key Experimental Results

### Main Results — Image Quality (3 Longitudinal MRI Benchmarks, mean±std)

| Method | ADNI PSNR↑ | ADNI SSIM↑ | AIBL PSNR↑ | OASIS PSNR↑ |
|--------|:---:|:---:|:---:|:---:|
| CardiacAging | 27.78±1.49 | 92.04 | 28.41 | 26.23 |
| DiffuseMorph | 29.56±1.63 | 93.57 | 29.17 | 28.13 |
| SADM | 26.94±2.28 | 85.15 | 27.97 | 26.74 |
| BrLP | 28.51±1.77 | 91.52 | 28.96 | 27.98 |
| MambaControl | 29.72±1.04 | 93.60 | 29.86 | 28.24 |
| **Δ-LFM** | **30.59±0.89** | **94.62** | **30.52** | **29.01** |

### Main Results — Progression Accuracy (Region MAE + Δ-RMAE)

| Method | ADNI Δ-RMAE↓ | AIBL Δ-RMAE↓ | OASIS Δ-RMAE↓ |
|--------|:---:|:---:|:---:|
| DiffuseMorph | 0.516 | 0.482 | 0.503 |
| BrLP | 0.630 | 0.594 | 0.622 |
| MambaControl | 0.554 | 0.525 | 0.561 |
| **Δ-LFM** | **0.436** | **0.417** | **0.473** |

Δ-RMAE is reduced by approximately 21%/21%/16% relative to MambaControl.

### Ablation Study (Average over 3 Datasets)

| Configuration | PSNR↑ | Δ-RMAE↓ | Notes |
|---------------|:---:|:---:|------|
| LFM Baseline (unconditional, [0,1]) | 27.59 | 0.552 | Worst |
| + Conditional information | 28.46 | 0.486 | Conditioning signals matter |
| + [0,T] time sampling | 28.78 | 0.472 | Temporal semantics are effective |
| + Arc Loss only | 29.52 | 0.457 | Directional constraint is most important |
| + Rank Loss only | 28.36 | 0.474 | Ranking alone is weaker |
| + ArcRank + [0,T] (full) | **30.04** | **0.442** | Components are synergistic |

### Key Findings
- t-SNE visualization of the ArcRank latent space reveals: (1) scans from the same patient cluster together; (2) diagnostic status (CN/MCI/AD) naturally separates into distinct groups—**despite no diagnostic labels being used during training**.
- Long-term prediction performance degrades gracefully: PSNR of 31–32 dB at 1–5 years, ~28.6 dB at 10 years, and ~27 dB at 13 years.
- The SVD computation in ArcRank introduces ~40% training time overhead; using `full_matrices=False` reduces per-call time from 0.055 s to 0.009 s (6× speedup).

## Highlights & Insights
- **"Disease as a velocity field" modeling perspective**: Rather than generating future snapshots, the model learns continuous dynamics of the change process—the velocity field in flow matching is conceptually aligned with disease progression.
- **Dual design of ArcRank**: SVD unifies direction (patient identity) and magnitude (disease severity) along two fundamentally distinct axes—an elegant and parsimonious formulation.
- **Δ-RMAE fills an evaluation blind spot**: Conventional metrics fail in longitudinal settings (a model that "copies the baseline" still scores high); Δ-RMAE forces models to genuinely capture change rather than remain static.
- **Unsupervised emergence of diagnostic states**: ArcRank constrains only temporal ordering and directional consistency, yet naturally learns the CN→MCI→AD severity gradient—demonstrating the power of well-chosen inductive biases.

## Limitations & Future Work
- Validation is limited to Alzheimer's disease; rapidly progressing conditions or diseases involving treatment intervention (e.g., brain tumors) may require different modeling assumptions.
- The linear trajectory assumption (straight-line progression in latent space) may fail to capture nonlinear patterns such as sudden deterioration or stable plateaus.
- Irregular scan intervals are only partially addressed through conditioning signals; changes in progression rate are not explicitly modeled.
- Dataset heterogeneity (multi-scanner/protocol variation) is mitigated only through preprocessing, without dedicated harmonization techniques.
- AE capacity is constrained by GPU memory (48 GB A6000); larger crops or deeper architectures may yield further improvements.

## Related Work & Insights
- **vs. BrLP (Puglisi et al. 2024)**: BrLP achieves partial personalization via ControlNet conditioned on volumetric ratios, but the conditioning is coarse; Δ-LFM enables finer individual trajectory modeling through ArcRank in latent space.
- **vs. TADM (Litrico et al. 2024)**: TADM predicts residual images but relies on diffusion-based denoising, which disrupts temporal continuity; Δ-LFM preserves continuity naturally through flow matching.
- **vs. ImageFlowNet (Liu et al. 2025)**: ImageFlowNet also employs flow fields but operates in image space; Δ-LFM is more efficient in latent space and additionally supports ArcRank trajectory alignment.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — Flow matching for disease progression, ArcRank latent alignment, and the Δ-RMAE evaluation metric constitute three distinct contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Three benchmarks (ADNI/AIBL/OASIS), eight comparison methods, detailed ablations, and long-term prediction analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is clearly articulated, derivations are concise, and visualizations are convincing.
- **Value**: ⭐⭐⭐⭐⭐ — Significant contribution to medical image generation and disease progression modeling; Δ-RMAE has potential to become a standard metric in the field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] EvoFlows: Evolutionary Edit-Based Flow-Matching for Protein Engineering](evoflows_evolutionary_edit-based_flow-matching_for_protein_engineering.md)
- [\[NeurIPS 2025\] Energy Matching: Unifying Flow Matching and Energy-Based Models for Generative Modeling](../../NeurIPS2025/medical_imaging/energy_matching_unifying_flow_matching_and_energy-based_models_for_generative_mo.md)
- [\[AAAI 2026\] Ambiguity-aware Truncated Flow Matching for Ambiguous Medical Image Segmentation](../../AAAI2026/medical_imaging/ambiguity-aware_truncated_flow_matching_for_ambiguous_medica.md)
- [\[ICLR 2026\] Resp-Agent: An Agent-Based System for Multimodal Respiratory Sound Generation and Disease Diagnosis](resp-agent_an_agent-based_system_for_multimodal_respiratory_sound_generation_and.md)
- [\[CVPR 2026\] EI: Early Intervention for Multimodal Imaging based Disease Recognition](../../CVPR2026/medical_imaging/ei_early_intervention_for_multimodal_imaging_based_disease_recognition.md)

</div>

<!-- RELATED:END -->

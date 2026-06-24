---
title: >-
  [Paper Note] Prospective Dynamic 3D MRI Reconstruction via Latent-Space Motion Tracking from Single Measurement
description: >-
  [CVPR 2026][Medical Imaging][Prospective Reconstruction] PDMR learns a low-dimensional non-linear manifold of dynamic 3D MRI motion (Deformation Vector Field, DVF) offline. During online inference, it optimizes only a 12-dimensional latent vector using a single instantaneous k-space measurement. This enables real-time reconstruction of high-fidelity 3D images under ultra-sparse sampling for prospective applications such as MR-guided radiotherapy.
tags:
  - "CVPR 2026"
  - "Medical Imaging"
  - "Prospective Reconstruction"
  - "Dynamic MRI"
  - "Manifold Learning"
  - "Deformation Vector Field"
  - "MR-guided Radiotherapy"
date: 2026-05-08
content_hash: fcf08f79e4360f52
---

# Prospective Dynamic 3D MRI Reconstruction via Latent-Space Motion Tracking from Single Measurement

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_Prospective_Dynamic_3D_MRI_Reconstruction_via_Latent-Space_Motion_Tracking_from_CVPR_2026_paper.html)  
**Code**: TBD  
**Area**: Medical Imaging  
**Keywords**: Prospective Reconstruction, Dynamic MRI, Manifold Learning, Deformation Vector Field, MR-guided Radiotherapy

## TL;DR
PDMR learns a low-dimensional non-linear manifold of dynamic 3D MRI motion (Deformation Vector Field, DVF) offline. During online inference, it optimizes only a 12-dimensional latent vector using a single instantaneous k-space measurement. This enables real-time reconstruction of high-fidelity 3D images under ultra-sparse sampling for prospective applications such as MR-guided radiotherapy.

## Background & Motivation
**Background**: MR-guided radiotherapy and interventional procedures require "prospective reconstruction"—reconstructing the patient's current 3D anatomy and motion state using only a **single instantaneous spoke** (one k-space measurement) acquired within the current latency window to guide treatment in real-time. Conversely, most existing methods focus on "retrospective reconstruction," which aggregates all time frames after the full sequence is acquired, utilizing spatio-temporal redundancy to fill undersampling gaps.

**Limitations of Prior Work**: Retrospective methods fail the two hard constraints of prospective scenarios: ultra-sparse measurement ($n \ll m$, single spoke) and instantaneous runtime requirements (sub-second latency). Directly applying them results in either severe blurring (compressed sensing like GRASP loses anatomical details) or extrapolation failure (INR methods like SPINER tend to extrapolate past motion trends, outputting nearly static images at new time points).

**Key Challenge**: Speed requires a highly compressed motion representation with minimal online optimization parameters. However, compact linear representations (e.g., MR-MOTUS, DREME-MR, which decompose DVF into a few spatial bases; Prior-INR, which uses a hand-crafted discrete respiratory state manifold) fail to capture the **non-linear and continuous** nature of real physiological motion, leading to a collapse in accuracy and robustness under ultra-sparse sampling. A direct trade-off exists between the compactness of linear/discrete models and the expressiveness of non-linear motion.

**Goal**: To decompose the problem into two sub-tasks: (a) learning a robust motion prior from undersampled retrospective data, and (b) **rapidly** adapting this prior to low-latency prospective reconstruction at new time points.

**Key Insight**: The authors adopt a motion-compensated (MoCo) decomposition—splitting the dynamic image $x_t = W(m, u_t)$ into a time-varying deformation field $u_t$ and a static template image $m$ (provided by a pre-scan as a patient-specific anatomical prior). Consequently, prospective reconstruction simplifies to "estimating only the current DVF" rather than reconstructing the full 3D volume from scratch. The core observation is that DVF is driven by a few physiological signals (e.g., respiration) and resides on a low-dimensional manifold.

**Core Idea**: Offline, a **non-linear, geometry-aware DVF manifold + mapping network** is learned (using a tri-plane representation to map latent vectors back to detailed 3D deformation fields). Online, the mapping network is frozen, and only a low-dimensional latent vector is optimized over a few steps to recover the current motion state from a single measurement—replacing "high-dimensional deformation fitting" with "latent vector search on a low-dimensional manifold."

## Method

### Overall Architecture
PDMR consists of two phases: **Offline Manifold Learning** uses time-continuous sparse measurements $\{y_t\}_{t=0}^T$ and a template $m$ from a patient's pre-scan to learn the non-linear mapping $f_{\psi,\theta}$ (latent vector $z \to$ 3D DVF $u$) alongside the latent codes, resulting in a compact and generalizable motion manifold. **Online Prospective Reconstruction** handles a new instantaneous measurement $y_{t'}$. By freezing the mapping network parameters $(\psi^*, \theta^*)$ and **optimizing only the current latent vector $z_{t'}$**, the current DVF is obtained in a few iterations. The template is then warped to produce the 3D image. The entire pipeline replaces high-dimensional fitting with a search in a 12D latent space, ensuring both speed and stability.

```mermaid
graph TD
    A["Input<br/>Pre-scan time-continuous sparse measurements + Template image m"] --> B["Manifold DVF Representation<br/>Latent vector z ∈ R¹² → Deformation field u"]
    B --> C["Geometry-Aware Mapping Network<br/>Tri-plane generator + MLP decoder, z→3D DVF"]
    C --> D["Offline Manifold Learning<br/>Auto-decoder joint optimization of latents and network, consistency + smoothness"]
    D -->|Freeze (ψ*,θ*)| E["Online Prospective Reconstruction<br/>Single measurement y_t' optimizes only latent vector z_t'"]
    E --> F["Warp Template<br/>x_t' = W(m, f(z_t')) outputs current 3D image"]
```

### Key Designs

**1. Manifold DVF Representation: Transforming High-Dimensional Deformation into Low-Dimensional Latent Search**

The fundamental issue with linear bases (MR-MOTUS) and hand-crafted discrete manifolds (Prior-INR) is insufficient expressiveness to capture non-linear continuous organ motion. PDMR parameterizes the deformation field as a **non-linear function** of a low-dimensional latent vector $f: z \in \mathbb{R}^r \mapsto u \in \mathbb{R}^{m\times 3}$ (where $r=12$). This encodes the current motion state into just 12 numbers. During prospective reconstruction, instead of optimizing millions of voxel displacements, only the 12D latent code is optimized. This extreme compression of the search space provides speed, while the non-linear mapping ensures accuracy and prevents collapse at non-rigid interfaces.

**2. Geometry-Aware Tri-plane Mapping Network: Mapping Latents to Precise and Coherent 3D DVFs**

Directly mapping a latent code to a full 3D DVF via an MLP is computationally heavy and unstable. Inspired by tri-plane representations, the generator $G_\psi$ first maps $z$ to three orthogonal feature planes $\{F_{xy}, F_{xz}, F_{yz}\}$. For any spatial coordinate $p=(x,y,z)$, its projected features from the three planes are concatenated: $F(p) = F_{xy}(x,y) \oplus F_{xz}(x,z) \oplus F_{yz}(y,z)$. A lightweight decoder $M_\theta$ then predicts the displacement $\Delta p = M_\theta(F(p))$. Iterating through all coordinates $\Omega$ yields the full field $u = [f_{\psi,\theta}(z,p)]_{p\in\Omega}$. Tri-planes provide high-resolution, structurally coherent feature embeddings that preserve both global anatomy and local deformation details, which is critical for stable adaptation under ultra-sparse sampling.

**3. Auto-decoder Style Offline Manifold Learning: Jointly Learning Manifolds and Mapping via Consistency and Regularization**

To learn the manifold from undersampled retrospective data, the authors use an auto-decoder framework to jointly optimize the set of latent codes $Z=\{z_t\}$ and network parameters $(\psi, \theta)$. For each time point, the latent vector $z_t$ (with a Gaussian prior) yields DVF $\hat u_t$, which warps the template to $\hat x_t = W(\hat u_t, m)$. This is passed through the dynamic MRI forward model $\hat Y = \{A_t \hat x_t\}$ ($A_t \triangleq P_t T$, where $P_t$ is time-varying sampling and $T$ is the Fourier operator). The objective is measurement consistency plus DVF regularization:
$$Z^*,\psi^*,\theta^* = \arg\min_{Z,\psi,\theta} \|\hat Y - Y\|_2^2 + \lambda R(U)$$
where $R(\cdot)$ enforces temporal smoothness and $\lambda$ is a weight. This step embeds the patient-specific continuous motion manifold into the network.

**4. Online Single-Measurement Adaptation: Freezing the Network and Optimizing Only Latents**

Post-learning, given an instantaneous measurement $y_{t'}$, $(\psi^*, \theta^*)$ are fixed. The optimal latent code is searched within the manifold:
$$z_{t'} = \arg\min_z \|A_{t'} x_{t'} - y_{t'}\|_2^2,\quad x_{t'} = W(m, f_{\psi^*,\theta^*}(z))$$
Once $\hat z_{t'}$ is found, $\hat u_{t'} = f_{\psi,\theta}(\hat z_{t'})$ is used to warp the template to get the current frame $\hat x_{t'}$. Because only a 12D vector is optimized and the solution is constrained to the learned manifold, it requires very few iterations and adapts rapidly to unseen motion states while maintaining physical plausibility.

### Loss & Training
Offline training uses Adam with a learning rate of $1\times 10^{-2}$ for the mapping network and $5\times 10^{-3}$ for latent vectors over 50 iterations. Latent dimension $r=12$ and each tri-plane has 32 channels. Implemented in PyTorch on an A100. Sampling uses a golden-angle stack-of-stars radial trajectory (448 readout samples, $k_z=96$ partitions). Spokes 0–150 are used for offline learning, and spokes 150–300 (Immediate) and 1000–1150 (After-2min, approx. 2 minutes after initial acquisition) are used for prospective evaluation.

## Key Experimental Results

### Main Results
The method was evaluated on XCAT digital phantoms and 6 in-house abdominal MRI cases against six baselines (Analytical: NUFFT/GRASP; Retrospective: TDDIP/SPINER; Prospective: Prior-INR/MR-MOTUS). Metrics used are PSNR(dB)/SSIM across "Immediate" and "After-2min" settings:

| Category | Method | XCAT-Immediate | XCAT-After2min | In-house-Immediate | In-house-After2min |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Analytical | NUFFT | 7.80/0.252 | 7.79/0.252 | 10.89/0.364 | 10.89/0.365 |
| Analytical | GRASP | 8.47/0.158 | 8.47/0.158 | 10.89/0.120 | 11.05/0.126 |
| Retrospective | TDDIP | 17.73/0.498 | 18.05/0.552 | 25.38/0.661 | 25.70/0.687 |
| Retrospective | SPINER | 20.25/0.873 | 20.10/0.869 | 35.43/0.942 | 36.36/0.946 |
| Prospective | Prior-INR | 15.05/0.444 | 15.27/0.473 | 26.72/0.810 | 27.00/0.811 |
| Prospective | MR-MOTUS | 24.39/0.931 | 24.22/0.929 | 41.04/0.981 | 41.11/0.976 |
| Prospective | **PDMR (Ours)** | **26.28/0.958** | **25.52/0.950** | **46.32/0.994** | **43.39/0.978** |

PDMR leads across all settings: the PSNR for in-house Immediate is ~5 dB higher than the second-best MR-MOTUS (46.32 vs 41.04), with a near-perfect SSIM of 0.994.

### Ablation Study
The main paper does not provide an independent ablation table but uses qualitative analysis (z–t profiles and error maps in Fig. 3) to explain failure modes:

| Comparison | Observation | Explanation |
| :--- | :--- | :--- |
| GRASP (Analytical) | Severe blurring, lost anatomy | Traditional methods fail under ultra-sparse measurements. |
| SPINER (Retro. INR) | Extrapolates past trends | Retrospective models fail to generalize to unseen time points. |
| Prior-INR (Discrete) | Discontinuous z–t trajectory | Hand-crafted discrete manifolds cannot reflect continuous motion. |
| MR-MOTUS (Linear) | Tracks large motion but misses small details | Linear limits; fails to capture small-scale motion (red arrows). |
| **PDMR (Non-linear)** | Nearly perfect alignment with GT | Captures both large-scale and fine local dynamics. |

### Key Findings
- **Non-linear manifold + tri-plane are critical**: Compared to the linear MR-MOTUS, PDMR excels in capturing **small-scale motion** and maintaining continuous z–t trajectories.
- **Prospective vs. Retrospective**: While retrospective SPINER achieves 0.94+ SSIM on in-house data, it effectively fails at extrapolation for "new moments"; PDMR is specifically designed for prospective use.
- **Drift Impact**: Performance drops slightly in the After-2min setting (in-house 46.32 $\to$ 43.39) due to motion drift over time, yet PDMR remains the top performer.

## Highlights & Insights
- **Reformulating "Reconstruction" as "Low-Dimensional Latent Search"**: By using MoCo decomposition and manifold priors, the reconstruction task is reduced from millions of voxels to a 12D vector optimization. This strategy is transferable to any inverse problem requiring low-latency adaptation.
- **Tri-planes for motion fields rather than appearance**: Tri-planes, typically used for 3D generation/NeRF, are repurposed as geometry-aware decoders for DVFs, balancing global anatomical consistency with local detail.
- **Auto-decoder latents as motion states**: Each time point has a latent code, and the network is shared, inherently parameterizing the patient-specific continuous motion manifold. This opens possibilities for motion prediction via latent space interpolation.

## Limitations & Future Work
- The paper lacks module-level ablation (e.g., impact of removing tri-planes, changing $r$, or removing DVF regularization). Quantitative evidence for individual components is missing. ⚠️
- Strong dependency on high-quality patient-specific templates $m$. If the template differs significantly from the current anatomy (e.g., tumor changes or posture shifts), the warp paradigm may fail.
- Evaluation is limited to abdominal respiratory motion (XCAT + 6 in-house cases). Generalization to cardiac or more complex motion with sliding interfaces is not fully verified.
- Detailed inference time analysis is relegated to the supplementary material; specific latency numbers are missing from the main text.

## Related Work & Insights
- **vs. MR-MOTUS / DREME-MR**: These methods represent DVF as a linear combination of spatial bases and update temporal coefficients. PDMR utilizes a non-linear manifold + tri-plane mapping, primarily gaining an advantage in capturing non-linear/subtle motion at the cost of offline training.
- **vs. Prior-INR**: Prior-INR uses a hand-crafted discrete respiratory state manifold for online search. PDMR’s manifold is data-driven, continuous, and generalizable, avoiding the discontinuous trajectory issues of discrete manifolds.
- **vs. SPINER / TDDIP (Retrospective INR/DIP)**: These fit full time series and struggle to extrapolate. PDMR is explicitly designed for prospective use, freezing the network to search the latent space for instantaneous state recovery.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to use non-linear manifolds for prospective dynamic MRI; clever tri-plane adaptation for motion fields.
- Experimental Thoroughness: ⭐⭐⭐⭐ Strong baselines and multiple settings, though lacks module-level ablation and large sample sizes.
- Writing Quality: ⭐⭐⭐⭐ Clear problem decomposition and complete formulations.
- Value: ⭐⭐⭐⭐ High potential for clinical scenarios like MR-guided radiotherapy.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Unsupervised Motion-Compensated Decomposition for Cardiac MRI Reconstruction via Neural Representation](../../AAAI2026/medical_imaging/unsupervised_motion-compensated_decomposition_for_cardiac_mri_reconstruction_via.md)
- [\[CVPR 2026\] Diffusion MRI Transformer with a Diffusion Space Rotary Positional Embedding (D-RoPE)](diffusion_mri_transformer_with_a_diffusion_space_rotary_positional_embedding_d-r.md)
- [\[CVPR 2026\] SIMSPINE: A Biomechanics-Aware Simulation Framework for 3D Spine Motion Annotation and Benchmarking](simspine_a_biomechanics-aware_simulation_framework_for_3d_spine_motion_annotatio.md)
- [\[CVPR 2026\] Breaking the Continuum: Discrete Distribution Learning for Structural MRI Reconstruction](breaking_the_continuum_discrete_distribution_learning_for_structural_mri_reconst.md)
- [\[CVPR 2026\] Decoding 3D Perception via BrainSSD: Synergistic Fusion of EEG Representations from Static and Dynamic Visual Streams](decoding_3d_perception_via_brainssd_synergistic_fusion_of_eeg_representations_fr.md)

</div>

<!-- RELATED:END -->

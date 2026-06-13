---
title: >-
  [Paper Note] Plug-and-Play Diffusion Meets ADMM: Dual-Variable Coupling for Robust Medical Image Reconstruction
description: >-
  [ICML 2026][Medical Imaging][PnP Diffusion Prior] This paper reintroduces the dual variable of ADMM into the PnP diffusion prior loop, utilizing "duality" to provide integral feedback that eliminates steady-state bias. A…
tags:
  - "ICML 2026"
  - "Medical Imaging"
  - "PnP Diffusion Prior"
  - "ADMM Dual Variable"
  - "Spectral Whitening"
  - "CT/MRI Reconstruction"
  - "Steady-state Bias"
date: 2026-05-08
content_hash: 71ec73634d52acf7
---

# Plug-and-Play Diffusion Meets ADMM: Dual-Variable Coupling for Robust Medical Image Reconstruction

**Conference**: ICML 2026  
**arXiv**: [2602.23214](https://arxiv.org/abs/2602.23214)  
**Code**: https://github.com/duchenhe/DC-PnPDP (Available)  
**Area**: Medical Image Reconstruction / Diffusion Models / Inverse Problems  
**Keywords**: PnP Diffusion Prior, ADMM Dual Variable, Spectral Whitening, CT/MRI Reconstruction, Steady-state Bias

## TL;DR
This paper reintroduces the dual variable of ADMM into the PnP diffusion prior loop, utilizing "duality" to provide integral feedback that eliminates steady-state bias. A frequency-domain Spectral Homogenization module is proposed to whiten structured dual residuals into pseudo-AWGN, preventing the triggering of OOD hallucinations in the diffusion denoiser. It achieves SOTA fidelity and approximately 3× inference acceleration on sparse-view/limited-angle CT and accelerated MRI.

## Background & Motivation

**Background**: The mainstream approach for solving medical inverse problems ($y=Ax+n$) involves PnP Diffusion Priors (PnPDP), which alternate between a data consistency sub-problem and a diffusion denoising prior sub-problem. Common implementations are based on Half-Quadratic Splitting (HQS) or proximal gradients, such as DiffPIR, DDS, DDNM, DAPS, and SITCOM.

**Limitations of Prior Work**: From a cybernetic perspective, the authors point out that HQS/PG-type solvers are "memoryless" operators—each iteration only considers the instantaneous data fidelity gradient, equivalent to a Proportional (P) controller. P-controllers cannot eliminate steady-state errors when the system encounters "high resistance" (heavy undersampling, strong noise), resulting in reconstruction results trapped at a **biased equilibrium point** that neither strictly satisfies physical measurements nor lies on the prior manifold. In medical scenarios, this bias directly compromises clinical reliability.

**Key Challenge**: Classical optimization theory provides a solution: adding a dual variable (Lagrange multiplier), which integrates the primal residuals and is equivalent to an Integral (I) controller, driving $x \to z$ to strictly satisfy constraints. However, inserting the dual $u^{(k)}$ directly back into the diffusion PnP loop triggers a second conflict: $u$ accumulates "structured" residuals (directional streaks in CT, coherent aliasing in MRI) with colored spectra. Since diffusion denoisers are trained only on AWGN, the input $v^{(k+1)}=x^{(k+1)}+u^{(k)}$ immediately becomes OOD (Out-of-Distribution), and the denoiser "hallucinates" these artifacts as semantic content.

**Goal**: (1) Reconnect the dual variable to PnP diffusion; (2) Ensure the input seen by the diffusion denoiser remains equivalent to AWGN.

**Key Insight**: Decouple the "geometric role" and the "statistical role"—the dual variable manages geometric convergence, while an additional frequency-domain whitening module "bleaches" the colored residuals accumulated by the dual variable into pseudo-AWGN.

**Core Idea**: Use ADMM duality to provide integral feedback for eliminating steady-state bias, then use Spectral Homogenization in the frequency domain to fill "spectral dips," fitting the power spectrum of the denoiser input to white noise. This reconciles the conflict between "geometric strictness" and "statistical compatibility."

## Method

### Overall Architecture
The DC-PnPDP framework strictly follows the three-step ADMM structure but inserts a frequency-domain adaptation module $T_{\text{SH}}$ before the second step. One iteration (Algorithm 1) is as follows:

1.  **Data Fidelity Update**: $x^{(k+1)}=\arg\min_x \|Ax-y\|_2^2+\rho\|x-z^{(k)}+u^{(k)}\|_2^2$ (Closed-form / CG solver);
2.  **Dual Shift**: $v^{(k+1)} = x^{(k+1)} + u^{(k)}$;
3.  **Spectral Homogenization**: $\tilde v^{(k+1)} = T_{\text{SH}}(v^{(k+1)}; z^{(k)}, \sigma_t)$;
4.  **Diffusion Denoising**: $z^{(k+1)} = D_\sigma(\tilde v^{(k+1)}, t)$;
5.  **Dual Update**: $u^{(k+1)} = u^{(k)} + (x^{(k+1)} - z^{(k+1)})$.

The noise schedule $\sigma_t$ follows linear annealing from the EDM framework. Compared to DiffPIR, the structural differences are **strictly limited** to "whether the dual $u$ is maintained" and "whether $T_{\text{SH}}$ is inserted," allowing the ablation study to clearly attribute gains to these modules.

### Key Designs

1.  **Dual-Coupled Iteration**:
    - **Function**: Explicitly maintains the ADMM dual variable $u^{(k)}$ in the PnP diffusion loop as an "integral memory" to drive $x \to z$ convergence to a strictly consistent point.
    - **Mechanism**: After each iteration, $u^{(k+1)}=u^{(k)}+(x^{(k+1)}-z^{(k+1)})$ is performed, accumulating historical consensus errors into a corrective force, upgrading from a P-controller to a PI-controller. In the next $x$ update, $u$ enters the center of the quadratic term in the data fidelity step, acting as a feedback loop that "repeatedly applies pressure to bring the two variables closer."
    - **Design Motivation**: Existing HQS/PG-based PnP diffusion solvers treat $u \equiv 0$ by default, effectively removing the integral action of ADMM, which inevitably leaves steady-state bias under heavy undersampling. Empirically, enabling the dual variable alone contributes +4.55 dB on LACT-90.

2.  **Spectral Homogenization**:
    - **Function**: Transforms the structured, colored residuals in the dual-shifted $v^{(k+1)}=x^{(k+1)}+u^{(k)}$ into pseudo-AWGN with an approximately flat power spectrum to avoid OOD hallucinations.
    - **Mechanism**: A three-step process: (1) **Diagnosis**: Use $r^{(k+1)}=v^{(k+1)}-z^{(k)}$ as a residual proxy and perform kernel-smoothed PSD estimation $\hat S_r(\omega) = (|\mathcal F(r)(\omega)|^2)*K_\delta$; (2) **Synthesis**: Define the spectral gap $\Delta S(\omega)=\max(\epsilon, \sigma_t^2(HW) - \hat S_r(\omega))$ and construct complementary noise $\xi^{(k+1)} = \mathcal F^{-1}(\sqrt{\Delta S(\omega)} \odot e^{i\angle\mathcal F(n)})$ using the random phase of white noise $n$; (3) **Fusion**: $\tilde v^{(k+1)} = v^{(k+1)} + \xi^{(k+1)}$. Proposition 4.1 provides a theoretical guarantee for second-order spectral consistency: $\mathbb E_\xi[S_{n_{\text{eff}}}(\omega)] \approx \sigma_t^2(HW)$, equivalent to $\text{Cov}(n_{\text{eff}}) \approx \sigma_t^2 I$.
    - **Design Motivation**: Physical artifacts (CT streaks, MRI aliasing) are inherently concentrated in specific frequency bands and are "colored." Spatial noise addition would blur the entire image, whereas the frequency-domain approach only adds energy to "spectral dips," preserving the semantic information carried by "spectral peaks"—whitening the noise while preserving structure. This process is likened to "Coherence Breaking": using random phases + complementary amplitudes to drown out the coherence of structured artifacts.

3.  **DiffPIR-aligned Data-Consistency**:
    - **Function**: Ensures the data fidelity step of DC-PnPDP is strictly equivalent to the DiffPIR implementation for fair ablation and comparison.
    - **Mechanism**: CT uses torch-radon for projection (parallel-beam, 20 views for SVCT, 90 views in $[0,90]^\circ$ for LACT); MRI uses 1D equidistant Cartesian undersampling (AF=6/10). The fidelity sub-problem is solved using CG. For complex-domain MRI, Spectral Homogenization is applied independently to real and imaginary parts.
    - **Design Motivation**: The authors intended to decouple the gains of "Dual + SH" from the choice of the solver in the baseline—only when the data consistency steps are identical can performance gains be cleanly attributed to the new modules.

### Loss & Training
The diffusion prior is pre-trained using the EDM framework (CT from scratch on AbdomenCT-1K, MRI using public weights from Zheng et al. 2025). **Diffusion weights are not updated during inference**—all modifications occur on the PnP solver side. This is standard for plug-and-play settings and means the SH module is plug-and-play for any pre-trained diffusion prior.

## Key Experimental Results

### Main Results
Comparing 5 SOTA PnPDP solvers on AbdomenCT-1K (CT) and fastMRI brain (MRI), PSNR/SSIM are as follows (selected from Table 1):

| Task | Metric | DiffPIR (Strongest baseline) | SITCOM | DAPS | DC-PnPDP (100 NFE) | Gain vs. Prev. SOTA |
|------|------|------|------|------|------|------|
| LACT-90 | PSNR / SSIM | 34.70 / 0.926 | 32.07 / 0.911 | 30.02 / 0.891 | **39.46 / 0.955** | **+4.76 dB** |
| SVCT-20 | PSNR / SSIM | 37.86 / 0.947 | 37.76 / 0.945 | 37.05 / 0.939 | **40.55 / 0.963** | **+2.69 dB** |
| Brain MRI AF=6 | PSNR / SSIM | 34.88 / 0.965 | 35.58 / 0.969 | 34.89 / 0.967 | **36.43 / 0.972** | +0.85 dB |
| Brain MRI AF=10 | PSNR / SSIM | 27.92 / 0.918 | 28.67 / 0.927 | 27.04 / 0.910 | **30.91 / 0.943** | **+2.24 dB** |

The +4.76 dB jump in "missing angle" tasks like LACT significantly validates the value of "dual-variable elimination of steady-state bias"—baselines almost always produce false structures in the direction of the missing wedge.

### Ablation Study
Toggle DC (Dual Variable) and SH (Spectral Homogenization) on LACT-90; the first row is DiffPIR:

| DC | SH | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Insight |
|----|----|--------|--------|---------|------|
| ✗ | ✗ | 31.36 | 0.894 | 0.023 | DiffPIR baseline, maximum steady-state bias |
| ✗ | ✓ | 31.51 | 0.898 | 0.022 | Whitening alone does little, showing colored residuals are mostly caused by the dual variable |
| ✓ | ✗ | 35.91 | 0.934 | 0.012 | Adding dual gains +4.55 dB, but 1.1 dB below the full version (OOD leak in denoiser) |
| ✓ | ✓ | **37.02** | **0.943** | **0.011** | Strong synergy between the two modules |

### Key Findings
- **DC is the primary driver, SH is the "safety valve"**: Opening SH alone only adds +0.15 dB, but when SH is off, DC is 1.1 dB behind the full model—the value of SH is not its independent score contribution, but the elimination of the "OOD risk introduced by the dual variable."
- **Inference Efficiency ~3.3× Acceleration**: DC-PnPDP at 30 NFE already exceeds DiffPIR at 100 NFE; on SVCT-20, DiffPIR requires 1000 NFE to approach the quality of DC-PnPDP at 50 NFE.
- **Spectral Mechanism Visualization (Fig. 3)** confirms: (a) Ideal AWGN input vs. (b) the proposed SH output PSDs almost overlap; (c) naive noise addition $x+u+\sigma_t n$ leads to over-energy and denoiser under-correction; (d) $x+u$ without noise has high-frequency spikes, triggering hallucinations.

## Highlights & Insights
- **Control Theory Perspective for PnP Solvers**: Analogizing HQS/PG solvers to P-controllers and dual variables to I-terms not only explains why baselines converge to biased points but also predicts the "de-biasing" effect of the dual—theory and experimental data align perfectly.
- **Handling Structured Residuals as an OOD Problem**: Explicitly addressing "dual accumulation = OOD," acknowledging the utility of the dual while admitting it breaks the AWGN assumption, and solving it in the frequency domain—this is a rare "acknowledge then remediate" design in medical inverse problems.
- **Transferable Trick**: Spectral Homogenization is essentially a lightweight wrapper for "spectral whitening of the input for a pre-trained denoiser." It can theoretically be applied to any scenario where solver residuals are colored but the denoiser expects AWGN, such as natural image super-resolution or deblurring PnP pipelines.
- **Strict Data Fidelity Alignment with DiffPIR**: This "controlled variable" experimental design is remarkably disciplined for the PnP literature—many papers change both the solver and the diffusion prior simultaneously, obscuring the source of gains.

## Limitations & Future Work
- **Acknowledged Limitations**: The method is primarily validated under single-coil + Cartesian equidistant undersampling for CT/MRI, not covering multi-coil parallel imaging, 3D volumetric reconstruction, or non-linear forward operators (e.g., phase retrieval).
- **SH is an Approximate Whitening based on Second-order Moments**: Proposition 4.1 only guarantees the expected PSD approaches $\sigma_t^2 I$, not necessarily that high-order statistics match AWGN. If the denoiser is sensitive to high-order moments (due to highly non-linear scores), residual OOD may persist.
- **Bootstrap of PSD Estimation using $z^{(k)}$ as a Clean Proxy**: In early iterations, $z^{(k)}$ is noisy, leading to biased residual estimates. EMA or score-based uncertainty estimates could be alternatives.
- **Hyperparameter $\rho$ Sensitivity**: ADMM convergence depends heavily on $\rho$. The paper uses a fixed $\rho$ and does not discuss adaptive penalties (e.g., ADMM's classic $\rho$ scheduling).
- **Future Directions**: Extending SH to colored measurement noise (e.g., Poisson-Gaussian in CT), replacing the dual with modern primal-dual splitting (Chambolle-Pock), or learning $T_{\text{SH}}$ as a conditional module rather than manual estimation.

## Related Work & Insights
- **vs. DiffPIR (PnP-HQS)**: This method is perfectly aligned with DiffPIR in the data fidelity step. The +4.76 dB gain on LACT-90 is entirely attributable to the new modules, providing a clean "additive control."
- **vs. DAPS / SITCOM / DDS / DDNM**: These methods focus on "designing the likelihood sub-problem" or "modifying the reverse SDE" but retain the memoryless HQS-like structure. This paper uses control theory to point out their common weakness.
- **vs. Shrestha & Fu 2026 (AC-DC)**: AC-DC also identifies the mismatch between ADMM iterates and score model training distributions, but they solve it by inserting a conditional Langevin inner loop (multiple steps, higher computation). SH is a single frequency-domain operation.
- **vs. Bendel et al. 2025 (iterative colored re-noising)**: They also aim to pull the diffusion input back to AWGN but use spatial re-noising; the frequency-domain route in this paper more precisely "fills dips without harming peaks."

## Rating
- Novelty: ⭐⭐⭐⭐ Reintroducing dual variables is not a new concept, but the combination of P/I control analysis, explicit OOD acknowledgment, and frequency-domain adaptation is highly complete and well-motivated.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covering three CT/MRI tasks, 5 SOTA baselines, clean ablation via "data fidelity alignment," and efficiency curves. The main gap is multi-coil/3D coverage.
- Writing Quality: ⭐⭐⭐⭐⭐ Strong internal consistency between motivation, method, and experiments. The cybernetic metaphor is used effectively throughout.
- Value: ⭐⭐⭐⭐ A plug-in friendly solver upgrade for the medical inverse problem community; the SH module could serve as a "general patch" for other PnP works.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] DM4CT: Benchmarking Diffusion Models for Computed Tomography Reconstruction](../../ICLR2026/medical_imaging/dm4ct_benchmarking_diffusion_models_for_computed_tomography_reconstruction.md)
- [\[CVPR 2026\] BiCLIP: Bidirectional and Consistent Language-Image Processing for Robust Medical Image Segmentation](../../CVPR2026/medical_imaging/biclip_bidirectional_and_consistent_language-image_processing_for_robust_medical.md)
- [\[CVPR 2026\] Diffusion-Based Feature Denoising and Using NNMF for Robust Brain Tumor Classification](../../CVPR2026/medical_imaging/diffusion-based_feature_denoising_and_using_nnmf_for_robust_brain_tumor_classifi.md)
- [\[CVPR 2026\] SD-FSMIS: Adapting Stable Diffusion for Few-Shot Medical Image Segmentation](../../CVPR2026/medical_imaging/sd_fsmis_adapting_stable_diffusion_for_few_shot_medical_image_segmentation.md)
- [\[ICML 2026\] Foundation VAEs for 3D CT Reconstruction, Augmentation, and Generation](foundation_vaes_for_3d_ct_reconstruction_augmentation_and_generation.md)

</div>

<!-- RELATED:END -->

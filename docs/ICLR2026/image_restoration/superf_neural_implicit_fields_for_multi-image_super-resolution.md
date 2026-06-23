---
title: >-
  [Paper Note] SuperF: Neural Implicit Fields for Multi-Image Super-Resolution
description: >-
  [ICLR 2026][Image Restoration][Paper Note] SuperF treats multi-frame low-resolution (LR) images as "reconstruction targets" rather than network inputs. It uses a cross-frame shared coordinate MLP (Implicit Neural Representation) to fit the scene on a high-resolution (HR) continuous grid while simultaneously optimizing affine alignment parameters for each frame.
tags:
  - ICLR 2026
  - Image Restoration
date: 2026-05-08
content_hash: 4e90112ae190d10e
---
# SuperF: Neural Implicit Fields for Multi-Image Super-Resolution

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=FiiItlSqqL](https://openreview.net/forum?id=FiiItlSqqL)  
**Code**: https://sjyhne.github.io/superf (Project page + demo + dataset)  
**Area**: Image Restoration / Super-Resolution  
**Keywords**: Multi-image super-resolution, Implicit Neural Representations, Test-time optimization, Sub-pixel alignment, Neural fields

## TL;DR
SuperF treats multi-frame low-resolution (LR) images as "reconstruction targets" rather than network inputs. It uses a cross-frame shared coordinate MLP (Implicit Neural Representation) to fit the scene on a high-resolution (HR) continuous grid while simultaneously optimizing affine alignment parameters for each frame. This enables multi-image super-resolution (MISR) for satellite and handheld camera bursts under a Test-Time Optimization (TTO) framework **without requiring any high-resolution training data**, achieving magnification factors up to ×8.

## Background & Motivation
**Background**: Super-resolution follows two paths. Single-image super-resolution (SISR) is essentially an ill-posed inverse problem that must rely on strong priors—learned either from large amounts of high-resolution data or from high-resolution guidance in another modality. Multi-image super-resolution (MISR) takes a different approach: capturing multiple LR frames of the same scene with **sub-pixel shifts**. Each frame exhibits different aliasing artifacts due to different sampling grids; these "noise-like" differences actually carry complementary high-frequency information. Fusing them allows for the reconstruction of a shared high-resolution image.

**Limitations of Prior Work**: SISR relies on learned priors that easily "hallucinate" structures non-existent in reality—acceptable for mobile photography but fatal for scientific applications like medicine or remote sensing. For MISR, supervised methods require paired LR-HR training sets, where HR data collection is expensive and pairing is non-trivial. TTO methods (e.g., Wronski's steerable kernel regression) do not require training data but treat LR frames as model inputs to directly regress HR images, which limits representation. Existing works using Implicit Neural Representations (INR) for burst fusion (e.g., Nam’s NIR, originally designed for layer separation) share similar ideas but are **not designed for precise sub-pixel alignment in MISR**, which the authors prove is the key to MISR.

**Key Challenge**: The success of MISR hinges on whether the sub-pixel shifts of multiple frames can be **precisely aligned** to the same continuous coordinate system. If shifts are poorly estimated, multi-frame information not only fails to help but also blurs each other (in experiments, "multi-frame without alignment" yields a lower PSNR than a single frame). The traditional paradigm of treating LR frames as inputs naturally struggles to refine such sub-pixel shifts within a continuous coordinate space.

**Goal**: Within a TTO framework independent of high-resolution training data: (1) use a continuous field to represent the underlying HR signal; (2) achieve sufficiently precise sub-pixel alignment for each frame; (3) maintain robustness against anomalous pixels like clouds and noise in real data.

**Key Insight**: Drawing inspiration from De Lutio's guided super-resolution, the problem is **inverted**—LR frames are not fed into the model but are treated as supervision **targets**. The model itself is an INR defined on continuous high-resolution coordinates. The continuity of INR serves two purposes: performing sub-pixel alignment in coordinate space and representing the underlying HR signal.

**Core Idea**: Use a cross-frame shared coordinate MLP to represent the HR field and treat the affine transformation parameters of each frame directly as optimizable variables. **Jointly optimizing "alignment + representation"** is simple, but this joint optimization is precisely what makes INR truly suitable for MISR tasks.

## Method

### Overall Architecture
The core of SuperF is a shared implicit neural representation $f_\theta$ (coordinate MLP): it takes continuous coordinates $v\in[0,1)^d$ on an HR grid as input and outputs the RGB intensity (and optionally uncertainty) at that location. To have this single field serve $T$ shifted LR frames, the key is to align each frame to the same reference frame.

The pipeline works as follows: start with an HR coordinate grid $v$ corresponding to the **output resolution**. For the $t$-th frame, the coordinates are transformed to that frame's perspective via an affine transform $\hat A^{(t)}$ (the reference frame is fixed at $\hat A^{(1)}=I$). After Fourier positional encoding, these are fed into the shared MLP, followed by a per-frame spectral projection $\hat\rho^{(t)}$ (correcting brightness/contrast) to obtain the HR estimate under that frame's perspective $\hat y^{(t)}_\theta(v)=\hat\rho^{(t)}(f_\theta(\hat A^{(t)}v))$. Since the supervision signal consists only of LR frames, a boxcar filter (average pooling) is used to downsample the HR output to the LR resolution $\hat y^{(t)}_{LR,\theta}$, which is then compared against the ground truth LR frame $y^{(t)}_{LR}$ to calculate the loss. Training involves joint gradient descent on $\theta$ and all $\hat A^{(t)}$, iteratively optimizing alignment and the shared representation. During inference, $f_\theta$ is queried directly on the HR grid to obtain the super-resolution result.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["HR Coordinate Grid v"] --> B["Affine Alignment<br/>Per-frame Â(t)·v<br/>Reference Frame Fixed at I"]
    B --> C["Fourier Positional Encoding<br/>Countering Spectral Bias"]
    C --> D["Shared INR + Inverted Problem<br/>Coordinate MLP fθ + Spectral Projection ρ(t)<br/>Output HR RGB"]
    D --> E["Supersampling Optimization<br/>Avg Pool Downsampling to LR"]
    E -->|Compared with LR frames| F["GNLL Uncertainty Loss<br/>Per-frame Variance · Ignoring Noise Pixels"]
    F -->|Backprop Jointly Update θ and Â(t)| B
    D -.Inference: Direct Query.-> G["HR RGB Output"]
```

### Key Designs

**1. Shared INR + Inverted Problem: Treating LR Frames as Reconstruction Targets**

Traditional TTO-MISR takes a sequence of LR frames as model **inputs** to directly regress the HR image, with representation power limited by grid discreteness. SuperF inverts the problem definition: the model is a coordinate MLP $f_\theta$ defined on continuous HR coordinates. All $T$ frames **share** the same $f_\theta$, while LR frames serve as supervision **targets**. Formally, the authors assume the imaging process satisfies $y^{(t)}_{LR}(v)\approx \phi * y_{HR}(A^{(t)}v)$, meaning the LR frame is the HR signal subjected to an affine transform $A^{(t)}$ and then downsampled by a boxcar filter $\phi$ (spatial average pooling). Thus, the optimization objective is the average per-frame reconstruction loss over $T$ frames:

$$\arg\min L(\theta,\hat A^{(1)},\dots,\hat A^{(T)})=\frac{1}{T}\sum_{t=1}^{T}\sum_{v\in W}\ell\big(\hat y^{(t)}_{LR,\theta}(v),\,y^{(t)}_{LR}(v)\big)$$

Since $f_\theta$ integrates information from all frames into a single set of weights, it is forced to fit an **underlying continuous HR signal** that explains all frames simultaneously—this is where complementary multi-frame information is fused. Additionally, each frame is assigned a spectral projection $\rho^{(t)}$ (scaling + translation parameters per spectral band) to absorb inter-frame brightness/contrast differences, with the reference frame $\rho^{(1)}$ fixed to scale 1 and translation 0.

**2. Direct Parameterization of Affine Transformations: Enabling Precise Sub-pixel Alignment**

This is the most critical improvement of SuperF compared to the nearest neighbor baseline (NIR). NIR uses **another ReLU MLP** $g(t)$ conditioned on the frame index to estimate the transformation matrix $T_t=g(t)$. Such indirect parameterization is not friendly to sub-pixel level refinement. SuperF directly treats the transformation matrix $\hat A^{(t)}$ for each frame as part of the model parameters, explicitly described by only three scalars: two translations $\Delta x^{(t)}, \Delta y^{(t)}$ and one rotation angle $\alpha^{(t)}$, which are optimized via gradient descent alongside $\theta$. Following MISR convention, the **reference frame is fixed** ($\hat A^{(1)}=I$), allowing all other frames to align relative to it. This reduces degrees of freedom and, since the reference frame is naturally aligned with the HR reference, facilitates evaluation without introducing global misalignment. Ablations (Table 3) show this "direct parameterization" is the most crucial component: adding it alone cuts alignment error from 0.650 to 0.012 and increases PSNR from 24.63 to 26.14.

**3. Supersampling Optimization: Computing on HR Grid, Downsampling for Supervision**

Coordinate MLPs exhibit spectral bias—prioritizing low frequencies while high-frequency details converge extremely slowly. Since SuperF only has LR "views" for supervision, the MLP is prone to outputting only low frequencies. To address this, the authors employ a supersampling strategy: during optimization, the INR runs directly on the **HR grid $v$ corresponding to the super-resolution output** to obtain HR estimates. These are then downsampled to LR resolution using average pooling to be compared with the LR targets (i.e., $\hat y^{(t)}_{LR,\theta}(v)=\phi*\hat\rho^{(t)}(f_\theta(\hat A^{(t)}v))$ in Eq. 2). Compared to computing directly on the LR grid, this ensures backpropagated gradients carry fine-grained positional information from the HR grid, significantly improving sub-pixel alignment and high-frequency recovery. In ablations, this is the second most critical component and pushes PSNR to 31.30 when combined with direct parameterization. This also requires Fourier features positional encoding: $\gamma(v)=[\cos(2\pi b_1^Tv),\dots,\sin(2\pi b_m^Tv)]^T$, where $b_i\sim N(0,\sigma^2 I)$. The scale $\sigma$ controls the sampling frequency range and is a key hyperparameter for countering spectral bias ($\sigma=10$ for satellite domains, $\sigma=3$ for terrestrial).

**4. GNLL Heteroscedastic Uncertainty: Allowing the Model to "Ignore" Cloud and Noise Pixels**

In real satellite time-series, some frames are obscured by clouds. Forcing the minimization of reconstruction error for these pixels would contaminate the representation. SuperF allows the MLP decoder to **additionally output an uncertainty estimate** $\hat s^{(t)}_{LR}(v)$ for each frame and band (increasing the final layer output count from $n_c$ to $(T+1)\times n_c$). This is interpreted as the log-variance of a Gaussian distribution with the predicted HR signal as the mean. Gaussian Negative Log-Likelihood (GNLL) replaces MSE:

$$L_{GNLL}=\frac{1}{2}\sum_{t=1}^{T}\Big(\hat s^{(t)}_{LR}(v)+\frac{\big(\hat y^{(t)}_{LR,\theta}(v)-y^{(t)}_{LR}(v)\big)^2}{\exp(\hat s^{(t)}_{LR}(v))}\Big)$$

For positions that the model cannot fit well (e.g., clouds), GNLL can reduce the loss by **increasing the variance** rather than strictly minimizing the reconstruction error; the first term $\hat s$ acts as a regularizer to prevent variance from being increased indefinitely. MSE is a special case of GNLL with constant variance. Experiments show GNLL significantly outperforms MSE on the cloud-heavy WorldStrat-bitter dataset and performs equally on clean sequences.

### Loss & Training
The average per-frame reconstruction loss is calculated over $T$ frames (Eq. 3), where $\ell$ can be MSE or GNLL. The AdamW optimizer is used with a base learning rate of $2\times10^{-3}$, decaying to $1\times10^{-6}$ over 2000 iterations using cosine annealing. Batch size = 1 frame, ReLU MLP, trained on a single H100. During evaluation, a 16-pixel boundary is cropped, and color matching is performed following Bhat's method to correct global color shifts.

## Key Experimental Results

### Main Results
Two domains of datasets: the self-constructed **SatSynthBurst** (synthesized from 20 high-res WorldStrat satellite images, each generating 16 LR frames with ×2/×4/×8 factors, sub-pixel shifts, MTF simulation, and Gaussian noise) and **SyntheticBurst** (handheld terrestrial bursts provided by Bhat, 50 structured scenes, 14 frames each). Metrics: PSNR / SSIM / LPIPS.

PSNR Comparison (higher is better, standard deviation in parentheses, iterations in brackets):

| Method | SatSynth ×2 | ×4 | ×8 | Synth ×2 | ×4 | ×8 |
|------|------|------|------|------|------|------|
| Bilinear | 34.69 | 29.71 | 26.62 | 27.66 | 26.12 | 25.44 |
| Lafenetre 2023 (Kernel Reg. TTO) | 33.46 | 27.70 | 24.88 | 27.02 | 26.46 | 25.19 |
| NIR (Nam 2022) [5k] | 25.65 | 24.99 | 23.61 | 24.46 | 23.39 | 22.93 |
| **SuperF MSE (Ours)** [2k] | 36.73 | 32.94 | 28.87 | 29.38 | 27.90 | 27.08 |
| **SuperF GNLL (Ours)** [2k] | **37.26** | **34.03** | **29.28** | **29.48** | 27.47 | 26.58 |

Two existing methods fail to beat bilinear (in PSNR), while SuperF outperforms them comprehensively with both MSE and GNLL. GNLL is consistently better in the satellite domain (more robust to spectral changes) and performs equally in the terrestrial domain. SuperF exceeds NIR's 5k-iteration results in just 2k iterations.

### Ablation Study
Stepwise addition of components (×4, 16 frames, SatSynthBurst / SyntheticBurst):

| FF | Multi-frame | Alignment | SatSynth PSNR | Synth PSNR | Note |
|----|------|------|------|------|------|
| ✗ | ✗ | ✗ | 20.33 | 16.63 | Vanilla INR Single Frame |
| ✓ | ✗ | ✗ | 30.42 | 24.69 | + Fourier Encoding (single frame, ≈bilinear) |
| ✓ | ✓ | ✗ | 28.11 | 22.83 | Multi-frame w/o Alignment → Performance Drop |
| ✓ | ✓ | ✓ | **32.94** | **27.87** | Joint Alignment enables multi-frame utility |

Component-level breakdown (added sequentially to NIR, SatSynthBurst ×4):

| Configuration | PSNR | Alignment Error ↓ | Note |
|------|------|------|------|
| NIR baseline | 24.63 | 0.650 | No components from this work |
| + Direct Param. T | 26.14 | 0.012 | Most critical; alignment error plummets |
| + Supersampling SS | 24.76 | 0.079 | Limited effect in isolation |
| + Fixed Ref Frame FBF | 26.39 | 0.319 | Beneficial but relies on direct param |
| Direct T + SS | 31.30 | 0.012 | Significant gains when combined |
| **SuperF (All)** | **32.94** | **0.012** | — |

### Key Findings
- **Alignment is the Bottleneck of MISR**: Multi-frame without alignment (PSNR 28.11) is worse than single-frame (30.42) because sub-pixel shifts blur the signal; only joint optimization of alignment turns multi-frame info into gains.
- **Direct Parameterization > Indirect MLP Estimation**: Treating the three affine parameters directly as model parameters, rather than estimating them via an auxiliary MLP as in NIR, is the primary reason alignment error dropped from 0.65 to 0.012.
- **Value of GNLL in Dirty Data**: On cloud-contaminated time-series (WorldStrat-bitter), GNLL actively increases the variance for cloud pixels to ignore them, significantly better than MSE; on clean data, there is no difference.
- **Fourier Scale $\sigma$ Is Domain-Dependent**: Optimal values are 10 for satellite and 3 for terrestrial bursts, but $\sigma$ is consistent across samples within the same domain and independent of the chosen loss.

## Highlights & Insights
- **The "Input-Target Reversal" is a Transferable Paradigm**: Instead of feeding LR into the network, letting a continuous field explain a set of LR observations allows sub-pixel alignment to be refined via gradients in continuous coordinate space. This approach of "jointly optimizing representation and geometric transforms" is applicable to any multi-observation fusion task (multi-view, temporal, multi-modal).
- **Simplicity is Effective**: The core innovation—changing the transformation matrix from "MLP estimation" to "direct optimization of three scalars"—yielded the largest gains, suggesting that inductive bias selection is more important than model capacity for INR-MISR.
- **Supersampling Solves Spectral Bias**: Computing on the HR grid and downsampling for supervision injects high-frequency positional gradients into a low-frequency biased MLP. This trick is generalizable to all INR tasks where only LR supervision is available but high-frequency recovery is desired.
- **Zero HR Training Data = Zero Hallucination**: Pure test-time optimization fundamentally avoids the imaginary structures introduced by SISR learned priors, making it ideal for remote sensing and medical imaging where hallucinations are intolerable.

## Limitations & Future Work
- Authors acknowledge that in **noise-dominant** real-world scenarios (e.g., drastic lighting changes, surface changes, seasonal snow), the assumption of "repeated observations of the same scene" fails, requiring further handling of extreme noise.
- Key hyperparameters like $\sigma$ are **domain-dependent** and must be tuned; while they generalize well within a domain, an automatic selection mechanism for $\sigma$ is lacking.
- The TTO paradigm requires separate optimization for each scene (2000 iterations/sample), making it slower for large-scale deployment compared to supervised methods (inference time/VRAM/FLOPs on H100 are reported in the appendix).
- SyntheticBurst lacks an aligned LR reference frame, requiring heavy post-processing alignment for metrics; while synthetic satellite data simulates MTF and noise, gaps still exist compared to real Sentinel-2 data.

## Related Work & Insights
- **vs. Lafenetre 2023 / Wronski 2019 (Steerable Kernel Regression TTO)**: They use kernel regression to reconstruct RGB directly from LR frames, designed for satellite bursts ≤×2. SuperF uses continuous INR and joint alignment optimization, showing significant advantages at ×4/×8 and generalizing across satellite and terrestrial domains.
- **vs. NIR (Nam 2022) (INR Burst Fusion)**: Originally for layer separation, NIR uses auxiliary MLPs for transform estimation, does not fix a reference frame, and lacks supersampling. SuperF’s three modifications (direct affine parameterization, supersampling, fixed target frame) are specialized for sub-pixel alignment in MISR, raising PSNR from 24.63 to 32.94.
- **vs. LIIF / Thera (INR for SISR)**: Those are supervised, embedding-learned SISR methods for arbitrary scales. SuperF is an unsupervised TTO MISR method that does not learn priors but relies on multi-frame complementary constraints for reconstruction, effectively avoiding hallucinations.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of "problem inversion + direct affine parameterization + supersampling" is concise and addresses the core of MISR, despite being an elegant recombination of existing ideas.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers two domains, multiple scales, component-level ablations, real Sentinel-2 data, and cloud robustness. The self-constructed dataset fixes the lack of alignment benchmarks.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, with rigorous yet accessible mathematical descriptions of the imaging model and GNLL.
- Value: ⭐⭐⭐⭐ No HR training data needed, zero hallucinations, practical for remote sensing/scientific imaging. Includes demo and open-source datasets.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] The Surprising Effectiveness of Noise Pretraining for Implicit Neural Representations](../../CVPR2026/image_restoration/the_surprising_effectiveness_of_noise_pretraining_for_implicit_neural_representa.md)
- [\[ICLR 2026\] Continuous Space-Time Video Super-Resolution with 3D Fourier Fields](continuous_space-time_video_super-resolution_with_3d_fourier_fields.md)
- [\[NeurIPS 2025\] Implicit Augmentation from Distributional Symmetry in Turbulence Super-Resolution](../../NeurIPS2025/image_restoration/implicit_augmentation_from_distributional_symmetry_in_turbulence_super-resolutio.md)
- [\[ICML 2026\] Semi-Supervised Neural Super-Resolution for Mesh-Based Simulations](../../ICML2026/image_restoration/semi-supervised_neural_super-resolution_for_mesh-based_simulations.md)
- [\[ICLR 2026\] Test-Time Domain Generalization for Image Super-Resolution](test-time_domain_generalization_for_image_super-resolution.md)

</div>

<!-- RELATED:END -->

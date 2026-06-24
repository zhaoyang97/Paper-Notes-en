---
title: >-
  [Paper Note] Computational Speckle Pattern Interferometry
description: >-
  [CVPR 2026][3D Vision][Speckle Interferometry] Reformulates classic Electronic Speckle Pattern Interferometry (ESPI) into an inner product model of "speckle appearance vector $\times$ displacement phasor vector". It calibrates the system without precise phase-shifting hardware via a single matrix factorization, and then recovers sub-micron per-pixel displacements from a **single** speckle pattern using a Horn-Schunck-style energy functional, while additionally reading out the…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "Speckle Interferometry"
  - "Single-Frame Displacement Recovery"
  - "Matrix Factorization Calibration"
  - "Phasor"
  - "Optical Flow Reconstruction"
date: 2026-05-08
content_hash: e1bf0181b3cd3ccc
---

# Computational Speckle Pattern Interferometry

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Wu_Computational_Speckle_Pattern_Interferometry_CVPR_2026_paper.html)  
**Code**: None  
**Area**: Computational Imaging / Interferometry / Low-Level Vision  
**Keywords**: Speckle Interferometry, Single-Frame Displacement Recovery, Matrix Factorization Calibration, Phasor, Optical Flow Reconstruction

## TL;DR
Reformulates classic Electronic Speckle Pattern Interferometry (ESPI) into an inner product model of "speckle appearance vector $\times$ displacement phasor vector". It calibrates the system without precise phase-shifting hardware via a single matrix factorization, and then recovers sub-micron per-pixel displacements from a **single** speckle pattern using a Horn-Schunck-style energy functional, while additionally reading out the amount of in-exposure motion using the phasor amplitude.

## Background & Motivation
**Background**: Measuring tiny surface deformations (vibrations, strain) can reveal a wealth of "invisible" information such as material properties, acoustics, and structural health. Among optical methods, Electronic Speckle Pattern Interferometry (ESPI) is one of the few classic techniques capable of **full-field dense** measurements—illuminating the target with a laser and comparing two speckle patterns before and after deformation, where intensity variations indicate deformation.

**Limitations of Prior Work**: Traditional ESPI suffers from two major pain points. First, to accurately recover phase from speckle patterns, it usually requires **temporal phase shifting**—applying multiple controlled phase offsets to a coherent beam and capturing multiple frames. This relies on precision piezoelectric/phase-shifting hardware and complex calibration, and requiring multiple frames per measurement limits its application strictly to **static** scenes. Second, limited laser power leads to relatively long exposure times. In dynamic scenes, objects move during a single exposure, whereas the classic model assumes an "instantaneous snapshot" and cannot handle in-exposure motion. If one settles for directly comparing intensities instead (dual-exposure), the measurements are heavily contaminated by speckle parameters and noise when the phase is not an integer multiple of $2\pi$.

**Key Challenge**: High-sensitivity dense phase measurement $\Longleftrightarrow$ single-frame, without precision hardware, and capable of handling dynamic motion. In traditional frameworks, these requirements are mutually exclusive—accuracy comes from multi-frame controlled phase shifting, which in turn sacrifices dynamic capability and practicality.

**Goal**: (1) Eliminate dependency on precision phase-shifting/wavefront modulation hardware; (2) Recover per-pixel displacement from a **single** post-deformation pattern; (3) Simultaneously estimate the amount of motion during the exposure.

**Key Insight**: The authors discover that the speckle interferometry imaging model can be **factorized** into a spatially varying term (per-pixel) and a temporally varying term (displacement), both of which are 3D vectors whose inner product yields the image intensity. This observation transforms the "phase recovery" problem into an uncalibrated-photometric-stereo-like matrix factorization problem.

**Core Idea**: By expressing the intensity as $l_{jk}=\mathbf v_j\,\mathbf u_k^\top$, a rank-3 matrix factorization is first performed to calibrate the per-pixel speckle vector $\mathbf v_j$ (replacing manual phase-shifting). Then, borrowing the concept of optical flow (Horn-Schunck), the displacement phasor $\mathbf u$ is solved from a single frame. The **angle** of the phasor yields the displacement, while its **magnitude** yields the in-exposure motion.

## Method

### Overall Architecture
The input to CSPI is a speckle video (calibration phase) plus a post-deformation speckle pattern (measurement phase), and the output is the per-pixel displacement map and in-exposure motion map. The entire pipeline proceeds in three steps: First, the imaging model is factorized into an inner product formulation of "spatial speckle vector $\times$ temporal displacement phasor" (Section 3), which serves as the mathematical foundation for everything that follows. Next, in the **calibration phase**, a rank-3 matrix decomposition is applied to a set of speckle patterns with unknown global phase shifts to recover the per-pixel speckle appearance vector $\mathbf v_j$, thereby completely eliminating the need for precise phase-shifting hardware (Section 4.1). Finally, in the **measurement phase**, the calibrated $\mathbf v_j$ is combined with a Horn-Schunck-style global energy functional to solve for the per-pixel phasor $\mathbf u_j$ from a single image. The angle of the phasor gives the displacement, and its magnitude gives the motion (Section 4.2).

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Speckle Video<br/>(Unknown Global Phase Shifts)"] --> B["Factorized Imaging Model<br/>l = v·uᵀ Inner Product"]
    B --> C["Rank-3 Decomposition Calibration<br/>SVD yields V,U + Ellipse-to-Circle Transform Q"]
    C --> D["Per-pixel Speckle Vector vⱼ"]
    E["Single Post-deformation Speckle Pattern"] --> F["Horn-Schunck Single-frame Reconstruction<br/>Energy Functional Eq.12 + Smoothness Prior"]
    D --> F
    F --> G["Per-pixel Phasor uⱼ"]
    G -->|Take Angle ∠u| H["Displacement Map"]
    G -->|Take Magnitude |u|| I["In-exposure Motion Map"]
```

### Key Designs

**1. Inner Product Factorization of the Speckle Imaging Model: Demoting "Phase Recovery" to Matrix Factorization**

In the traditional ESPI intensity model $l = a\cos(b-\phi)+c$, the speckle parameters $a,b$ (spatially varying) and the displacement phase $\phi$ (temporally varying) are entangled within a cosine term, which is precisely why phase-shifting hardware is required to decouple them individually. The authors expand it trigonometrically and reorganize it into a clean inner product:

$$l_{jk} = a_j\cos b_j\cos\phi_k + a_j\sin b_j\sin\phi_k + c_j = \mathbf v_j\,\mathbf u_k^\top,$$

where $\mathbf v_j=[\,a_j\cos b_j,\;a_j\sin b_j,\;c_j\,]$ only varies with pixel $j$ (local speckle appearance), and $\mathbf u_k=[\,\cos\phi_k,\;\sin\phi_k,\;1\,]$ only varies with frame $k$ (phase due to displacement). The displacement phase is given by $\phi_k=\frac{2\pi}{\lambda}\mathbf k\cdot\Delta\mathbf x_k$, where the sensitivity direction $\mathbf k$ is determined by the optical geometry (for in-plane, $\mathbf k=\hat{\mathbf l}^{s_1}-\hat{\mathbf l}^{s_2}$; increasing the angle between the two light sources increases $\|\mathbf k\|$ to improve sensitivity). The beauty of this formulation is that once the spatial and temporal components are completely decoupled, "simultaneously recovering $\mathbf v_j$ and $\phi_k$ from a set of patterns with unknown phase shifts" becomes a standard low-rank factorization problem, isomorphic to uncalibrated photometric stereo—which is the root of eliminating the phase-shifting hardware.

**2. Rank-3 Matrix Factorization Calibration: Replacing Precision Phase-shifting Hardware with a Single SVD**

Since each frame is simply the inner product of the same set of $\mathbf v_j$ with a different $\mathbf u_k$, stacking the intensity of $J$ pixels $\times$ $K$ frames into a matrix $\mathbf L\in\mathbb R^{J\times K}$ gives a matrix $\mathbf L$ that is **exactly rank-3** under noise-free, globally-shared-phase-shift assumptions. Applying SVD to $\mathbf L$ yields the optimal rank-3 approximation $\mathbf L=\mathbf V\mathbf U^\top$, where the rows of $\mathbf V$ correspond to $\mathbf v_j$ and the rows of $\mathbf U$ to $\mathbf u_k$. Calibration simply requires manually translating one light source on a translation stage by a small amount (producing an **unknown** global phase shift) and capturing a few frames—no feedback-controlled precision phase-shifter is needed.

However, the factorization has a generalized bas-relief ambiguity: for any invertible $3\times3$ matrix $\mathbf Q$, $\tilde{\mathbf V}=\mathbf V\mathbf Q$ and $\tilde{\mathbf U}=\mathbf U\mathbf Q^{-\top}$ are also valid decompositions. The authors resolve $\mathbf Q$ using two physical constraints: $u_{k3}=1$ and $u_{k1}^2+u_{k2}^2=1$ (phasors lie on the unit circle, equivalent to assuming no in-exposure motion during calibration). Geometrically, the rows of the initially decomposed $\tilde{\mathbf U}$ lie on an ellipse in 3D, and $\mathbf Q$ is the transformation mapping this ellipse back to the unit circle. The factorization requires $J\ge2$ and $K\ge5$ (which is exactly the minimum number of 2D observations to fit an ellipse), while larger $J,K$ improve noise robustness. The recovered phase is unique up to an unknown offset and sign, which is usually trivial for deformation measurement.

**3. Horn-Schunck-Style Single-Frame Reconstruction: Decoding Per-Pixel Displacement from a Single Image**

Once $\mathbf v_j$ is obtained from calibration, the measurement phase provides only **one** intensity constraint per pixel, $\mathbf v_j\cdot\mathbf u_j=l_j$, to solve for two unknowns $u_{j1},u_{j2}$ (i.e., $\cos\phi,\sin\phi$). This single-constraint problem is underdetermined. Borrowing classic insights from optical flow, the authors introduce a spatial smoothness prior, feeding all pixels into a global energy functional:

$$\min_{\{u_{j1},u_{j2}\}}\sum_j(\mathbf v_j\cdot\mathbf u_j-l_j)^2+\alpha^2\|\nabla\mathbf u_j\|^2.$$

This formulation is highly analogous to Horn-Schunck optical flow—the data term enforces the phasor to satisfy the imaging constraint, while the regularization term $\alpha^2\|\nabla\mathbf u_j\|^2$ assumes that the displacement varies smoothly in space. Optimization is initialized by solving a small linear system under a "locally constant" phasor assumption in a small neighborhood, and then optimized using a Horn-Schunck-like iterative scheme to achieve a "locally smooth" solution. Compared to the pixel-wise phase estimation of Kao et al.'s (5,1) algorithm, which has ambiguities and poor phase map quality, this global solver significantly improves the single-frame phase map quality (as shown in the simulation comparison in the paper's Fig.4, where (b) traditional dual-exposure only yields fringes, (c)(d) Kao's method has ambiguities and high noise, and (e)(f) ours provides cleaner phase maps).

**4. Phasor Amplitude Encoding In-Exposure Motion: Transforming "Motion Blur" into a Readable Quantity**

The classic model treats each frame as an instantaneous snapshot, but under long exposures, the object may move from phase $\phi_a$ to $\phi_b$ within a single frame. The authors point out that this is equivalent to temporal averaging of unit-length phasors in the phasor domain: the recovered phasor $\vec u_k$ is no longer unit-length, but becomes

$$\vec u_k=\mathrm{sinc}\!\left(\tfrac{\phi_a-\phi_b}{2}\right)\exp\!\left(i\tfrac{\phi_a+\phi_b}{2}\right)\quad(\text{constant speed}),$$

whose **angle** $\angle\vec u_k=\tfrac12(\phi_a+\phi_b)$ gives the average displacement within the exposure, whereas its **magnitude** $|\vec u_k|=|\mathrm{sinc}(\cdot)|$ monotonically decays with the phase variation range $|\phi_a-\phi_b|$—the magnitude is 1 when the object is stationary, and as motion increases, the averaged unit phasors cancel each other out, driving the magnitude towards 0 (reaching exactly zero when the motion spans exactly $0\to2\pi$ at a constant speed). If the in-exposure motion is an integer number of sinusoidal oscillations, the average phasor takes the form of a zeroth-order Bessel function of the first kind $J_0(\tfrac{\phi_a-\phi_b}{2})$. This leads to an elegant conclusion: **displacement is encoded in the phasor angle, while in-exposure motion is encoded in the phasor magnitude**, allowing both to be read out from a single image without any additional hardware.

## Key Experimental Results

> Since this is a computational imaging/systems paper without a standard benchmark SOTA table, validation primarily relies on "quantitative measurements against known ground truth + qualitative demonstrations in multiple scenes". The paper's validation scenarios are summarized in the tables below.

### Main Results (Quantitative Verification + Scene Demonstration)

| Scene | Setup | Key Results |
|------|------|----------|
| Stepping Translation Stage | 4 µm/step, 600 fps, baseline 5.2/9.2 cm | Unwrapped phase matches reference stage position; larger baseline yields higher sensitivity but more severe phase wrapping (precision vs. unwrapping trade-off) |
| Continuous Translation | Trapezoidal velocity profile, max 10 µm/s, acceleration 3.8 µm/s² | Recovered displacement curve matches the expected motion profile |
| Reciprocating Motion | 20 µm forward, stop, 20 µm backward, 20 fps | Unwrapped phase + phasor magnitude fluctuate with stage velocity; magnitude decays at large displacements due to speckle decorrelation |
| Acoustic Recovery | Potato chip bag + speaker playing C3→C4, 1801 fps | Reconstructed spectrum matches microphone recording, containing clear second harmonics (replicating Davis et al.'s visual microphone) |
| Deformation Measurement | Q-tip pressing tuning fork tines | Visualizes sub-micron displacement and motion; heavier pressure leads to more hue wrapping, and large motion areas exhibit color desaturation |
| Vibration Modes | 288 Hz tuning fork, Chladni plates (134/150 Hz resonance) | Recovers out-of-phase motion of the two tines at fundamental frequency, and Chladni pattern |

### Comparison with Traditional/Existing Single-Frame Methods (Fig.4 Simulation)

| Method | Phase Map Quality | Drawbacks |
|------|-----------|------|
| Butters & Leendert (Dual-exposure subtraction) [1] | Fringes only | Contaminated by speckle parameters when phase is not an integer multiple of $2\pi$ |
| Kao et al. (5,1) algorithm [10] | Low | Pixel-wise estimation is ambiguous, requires extra optical elements, does not account for in-exposure motion |
| Ours (locally constant) | High | Linear system solution assuming locally constant phasor |
| Ours (locally smooth, Eq.12) | Highest | Smooth phasor assumption + further refinement via Horn-Schunck optimization |

### Key Findings
- **Baseline as Sensitivity Dial**: Increasing the angle/baseline between the two light sources scales up $\|\mathbf k\|$, which yields larger phase changes per unit displacement and better robustness to sensor noise, but also more severe phase wrapping—representing a system-level trade-off between precision and unwrapping feasibility.
- **Phasor Magnitude is a Real Physical Quantity, Not Noise**: In the reciprocating experiment, the magnitude fluctuates with the stage's velocity ripples. The authors independently verified this relationship using finite differences of pure phase, confirming that the magnitude indeed encodes in-exposure motion.
- **Extremely Fast**: For a 1280$\times$1024 image, MATLAB takes $\sim$0.15 s, Python $\sim$1 s, and GPU parallelization achieves <5 ms/frame, indicating potential for real-time full-field vibration analysis.
- **Out-of-Plane Configuration Is Too Sensitive**: Experiments primarily use in-plane configurations because out-of-plane setups are overly sensitive to environmental vibrations, necessitating active vibration isolation.

## Highlights & Insights
- **"Translating" Interferometry into Familiar CV Tools**: Map uncalibrated photometric stereo to matrix factorization calibration and optical flow to Horn-Schunck reconstruction. This cross-domain migration of using mature algorithms from neighboring fields to solve hardware challenges in another is elegant and highly intuitive for the CV community.
- **Phasor Magnitude = Motion**: The insight transforms what was traditionally discarded as "exposure degradation/speckle decorrelation" into a readable motion signal channel—obtaining both displacement and motion from a single image is a true masterstroke for single-shot gains.
- **Rank-3 Is Physically Determined, Not Empirically Tuned**: The inner product model naturally dictates that the rank of $\mathbf L$ is 3. The lower bound of $J\ge2,K\ge5$ corresponds exactly to the minimum observations needed to fit an ellipse, providing a clear geometric explanation for calibration feasibility rather than empirical parameter tuning.
- **Transferability**: Any computational imaging task whose imaging model can be formulated as a spatial-temporal tensor product (e.g., structured light, some phase-retrieval setups) could potentially benefit from this "low-rank factorization calibration + optical-flow-style reconstruction" paradigm.

## Limitations & Future Work
- **Active Illumination Required**: It relies on coherent lasers, making it inapplicable to scenarios where active lighting is not feasible.
- **Failure Under Large Displacements**: Displacements exceeding the phase unwrapping limit cannot be reliably reconstructed (an inherent limitation of interferometry).
- **Susceptibility to Environmental Vibration / Speckle Decorrelation**: Optimal performance requires vibration isolation equipment. If motion is excessive or speckle decorrelation occurs, **re-calibration** is mandatory. The calibration itself assumes global phase shifts and no in-exposure motion, limiting its dynamic calibration capabilities.
- **Cost of the Smoothness Prior**: The reconstruction step enforces a spatial smoothness prior on the displacement field, which may oversmooth genuine displacement discontinuities (such as boundaries or cracks).
- **Impracticality of Out-of-Plane Sensing**: Although theoretically supported, out-of-plane configurations are too sensitive in practice; the paper focuses almost exclusively on in-plane setups, limiting the measurable deformation directions.
- **Future Improvements**: Replace Horn-Schunck's quadratic smoothness with edge-preserving/anisotropic regularization to handle discontinuous deformations; explore dynamic calibration to relax the assumption of "no motion during calibration"; introduce learned priors to replace handcrafted smoothness terms.

## Related Work & Insights
- **vs. Traditional ESPI / Temporal Phase Shifting [28]**: Traditional methods rely on precision phase-shifting hardware to resolve phase over multiple frames, mostly limiting measurements to static scenes. This work replaces the phase-shifting hardware with a one-time matrix factorization calibration and enables dynamic measurements from a single frame, drastically reducing hardware complexity.
- **vs. Kao et al. (5,1) Algorithm [10]**: While both target single-shot post-deformation measurements, Kao's method still requires 5 phase-shifted patterns for calibration, extra optical components, and pixel-wise estimations that are ambiguous and ignore in-exposure motion. In contrast, this approach's calibration requires no precision elements, yields higher-quality phase maps via global optimization, and provides additional motion estimations.
- **vs. Visual Microphone (Davis et al. [3])**: Passive methods extract sub-pixel vibrations from standard video intensity variations to recover sound. This work uses active coherent illumination to bring the sensitivity to the sub-micron scale with dense full-field coverage. It replicates the potato chip bag acoustic recovery experiment and captures distinct second harmonics (though acknowledging it does not match the audio quality of surface-slope-based methods [21]).
- **vs. Uncalibrated Photometric Stereo [29]**: The calibration step directly borrows the concepts of matrix factorization and ambiguity resolution from uncalibrated photometric stereo, representing a successful transfer of mathematical tools into speckle interferometry calibration.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Reformulates classic ESPI as an inner product low-rank model, solving the "single-frame + no precision hardware + dynamic measurement" bottleneck using two classic CV frameworks (photometric stereo + optical flow). Very refreshing perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Solid quantitative verification against translation stage ground truth and diverse demonstrations in vibration, acoustics, and deformation. However, it lacks quantitative comparison tables against standard benchmarks, relying mostly on qualitative visualizations.
- Writing Quality: ⭐⭐⭐⭐⭐ The mathematical transitions (factorization $\to$ phasor $\to$ motion) are step-by-step and logical, with clear physical intuition and well-crafted illustrations.
- Value: ⭐⭐⭐⭐ Provides a low-cost, single-frame solution for full-field and transient vibration analysis. It holds practical value for computational imaging, non-destructive testing, and acoustic recovery, though limited by requirements for active illumination and vibration isolation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Einstein Fields: A Neural Perspective To Computational General Relativity](../../ICLR2026/3d_vision/einstein_fields_a_neural_perspective_to_computational_general_relativity.md)
- [\[CVPR 2026\] OLATverse: A Large-scale Real-world Object Dataset with Precise Lighting Control](olatverse_a_large-scale_real-world_object_dataset_with_precise_lighting_control.md)
- [\[CVPR 2026\] QD-PCQA: Quality-Aware Domain Adaptation for Point Cloud Quality Assessment](qd-pcqa_quality-aware_domain_adaptation_for_point_cloud_quality_assessment.md)
- [\[CVPR 2026\] GaussianFluent: Gaussian Simulation for Dynamic Scenes with Mixed Materials](gaussianfluent_gaussian_simulation_for_dynamic_scenes_with_mixed_materials.md)
- [\[CVPR 2026\] Think-Then-Generate: Structural Chain-of-Thought Reasoning for Consistent 3D Generation](think-then-generate_structural_chain-of-thought_reasoning_for_consistent_3d_gene.md)

</div>

<!-- RELATED:END -->

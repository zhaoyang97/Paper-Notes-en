---
title: >-
  [Paper Note] Delta Rectified Flow Sampling for Text-to-Image Editing
description: >-
  [CVPR 2026][Image Generation][Rectified Flow] DRFS transfers the concept of "subtraction to cancel out mutual information" from DDS to the velocity field distillation sampling of Rectified Flow (RF). By introducing a time-decaying offset term that pulls the target latent back onto the correct trajectory, it resolves the over-smoothing issue of RFDS during editing without architectural modifications, inversion, or training. It achieves state-of-the-art editing fidelity and con…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Rectified Flow"
  - "Distillation Sampling"
  - "Text Editing"
  - "Inversion-Free"
  - "Energy Function"
date: 2026-05-08
content_hash: c74a75eaf4e26845
---

# Delta Rectified Flow Sampling for Text-to-Image Editing

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Beaudouin_Delta_Rectified_Flow_Sampling_for_Text-to-Image_Editing_CVPR_2026_paper.html)  
**Code**: https://github.com/Harvard-AI-and-Robotics-Lab/DeltaRectifiedFlowSampling  
**Area**: Image Generation / Text-to-Image Editing / Rectified Flow  
**Keywords**: Rectified Flow, Distillation Sampling, Text Editing, Inversion-Free, Energy Function

## TL;DR
DRFS transfers the concept of "subtraction to cancel out mutual information" from DDS to the velocity field distillation sampling of Rectified Flow (RF). By introducing a time-decaying offset term that pulls the target latent back onto the correct trajectory, it resolves the over-smoothing issue of RFDS during editing without architectural modifications, inversion, or training. It achieves state-of-the-art editing fidelity and controllability on the PIE benchmark.

## Background & Motivation
**Background**: Text-guided image editing (T2I editing) consists of two main avenues. One is **non-energy-based methods** (e.g., RF-Inversion, FlowEdit, FTEdit), which preserve the background by employing two velocity fields (one for inversion, one for generation) alongside heuristic techniques such as attention injection and latent averaging. Among these, FlowEdit dispenses with explicit inversion altogether, directly estimating target latents using the offset between source and target velocities. The other is **energy-based optimization methods**: SDS and DDS formulate editing as the minimization of an energy function derived from frozen diffusion priors. RFDS (Rectified Flow Distillation Sampling) extends this approach from the noise residuals of diffusion models to the velocity fields of Rectified Flow, enabling plug-and-play editing.

**Limitations of Prior Work**: When RFDS directly applies $\phi=\phi_{tgt}$ to editing, it suffers from **over-smoothing**—background and high-frequency details are unintentionally altered, degrading fidelity. The authors diagnose the root cause: the gradient of the RFDS energy, $E_{RFDS}=\mathbb{E}\,\|v_\theta(x_t,t,\phi)-\dot{x}_t\|^2$, treats the "edited region" and the "preserved region" identically, thereby generating non-zero gradients in regions that should remain unchanged and destroying their high-frequency details. RFDS's own remedy (iRFDS: optimizing noise first to invert the image) incurs additional computational overhead.

**Key Challenge**: Gradients in energy-based optimization methods cannot distinguish between edited and preserved regions, resulting in a trade-off between editing strength and background fidelity.

**Goal**: To design an energy function within the Rectified Flow framework that penalizes only the differences between the source and target, automatically preserving mutual information, while simultaneously resolving the model-data mismatch caused by the target latent evaluation point deviating from the correct trajectory.

**Key Insight**: The authors observe that DDS preserves the background because it minimizes the **difference between two denoising trajectories** (source and target), where shared information is canceled out via subtraction. Translating this "subtraction" principle to the RF velocity field can drive gradients in background regions to zero.

**Core Idea**: Replace the "target residual" with "target residual - source residual" as the energy (residual subtraction), and overlay a time-decaying offset term $c_t(x_0^{tgt}-x_0^{src})$ to pull the evaluation point back onto the target trajectory. This simultaneously eliminates over-smoothing and precisely aligns with the target distribution.

## Method

### Overall Architecture
DRFS is an **inversion-free, training-free, architecture-preserving** distillation sampling optimizer. It takes the source image $x_0^{src}$ and source/target prompts $(\phi_{src}, \phi_{tgt})$ as input, and outputs the edited image $x_0^{tgt}$. Treating the to-be-edited image itself as the optimizable parameter ($\Theta=x_0^{tgt}$, initialized as $x_0^{src}$), it iteratively updates this image using approximate gradients under an RF velocity field prior until it is semantically aligned with the target prompt while preserving the background.

The entire pipeline is an optimization loop (Algorithm 1): at each step, a time $t$ is drawn from a descending schedule $\{\tau_j\}$, and a Gaussian noise $\varepsilon$ is sampled to construct the source latent $x_t^{src}=(1-t)x_0^{src}+t\varepsilon$ and the **offset target latent** $\hat{x}_t^{tgt}=(1-t)x_0^{tgt}+t\varepsilon+c_t(x_0^{tgt}-x_0^{src})$. The gradient calculated from the difference between their velocities is then used to update $x_0^{tgt}$. The two core contributions—the "residual subtraction energy function" and the "time-varying offset term $c_t$"—co-determine the gradient direction. The theoretical analysis further proves that this design unifies DDS and FlowEdit. Since this is an improvement purely at the sampling/energy function level (without adding network modules or multi-stage pipelines), no extra architecture diagram is provided, as the formulas are self-explanatory.

### Key Designs

**1. Residual Subtraction Energy: Automatically Nullifying Background Gradients**

The energy in RFDS is essentially $\mathbb{E}\,\|r_{tgt}\|^2$, where the residual is $r=v_\theta(x_t,t,\phi)-\dot{x}_t$. The limitation is that it only considers the target residual, causing background regions to be modified as well. DRFS adopts the "subtraction" principle from DDS to reformulate the energy as the difference between the source and target residuals:

$$E=\mathbb{E}_{t,\varepsilon}\Big[\big\|v_\theta(x_t^{tgt})-v_\theta(x_t^{src})-(\dot{x}_t^{tgt}-\dot{x}_t^{src})\big\|^2\Big]=\mathbb{E}_{t,\varepsilon}\big[\|r_{tgt}-r_{src}\|^2\big]$$

Consequently, the optimization only penalizes **differences** between the source and target. Shared information between the source and target images (typically the background) is canceled out via subtraction, causing the gradients in corresponding regions to approach zero, which naturally prevents degradation. The authors provide a clean sanity check: when $\phi_{tgt}=\phi_{src}$, the "shared information" encompasses the entire image, and the optimization should alter nothing. In this case, with the initial $x_0^{tgt}=x_0^{src}$, we have $x_t^{tgt}=x_t^{src}$, leading to an energy of exactly $0$, aligned with expectations. This step is pivotal in shifting from the single residual of RFDS to the "delta residual". Crucially, unlike naive "delta" approaches that only subtract conditional predictions, DRFS subtracts the **entire residual** between the velocity and data dynamics, thereby canceling out components shared by the source and target, leaving an RF-specific drift term in the gradient.

**2. Time-Varying Offset Term $c_t$: Pulling the Target Latent Evaluation Point Back to the Correct Trajectory**

Applying only residual subtraction still poses a risk: when interpolating directly using $x_t^{tgt}=a_t x_0^{tgt}+b_t\varepsilon$, because $x_0^{tgt}$ is mid-way along the optimization trajectory from source to target, the resulting $x_t^{tgt}$ deviates from the forward posterior of the target distribution. This causes inaccurate estimates of target velocity, weakens editing performance, and slows down convergence. DRFS remedies this by adding a linear compensation to the target latent, obtaining the corrected evaluation point:

$$\hat{x}_t^{tgt}=a_t x_0^{tgt}+b_t\varepsilon+c_t\,(x_0^{tgt}-x_0^{src}),\quad c_t\ge 0$$

The offset $c_t(x_0^{tgt}-x_0^{src})$ pushes the sampling trajectory closer to the target distribution along the "source $\to$ target" direction, enabling more accurate estimation of $v_\theta(\hat{x}_t^{tgt})$ and mitigating model-data mismatch in early optimization stages. Substituting this yields the DRFS gradient (approximating the network Jacobian as an identity matrix and directly optimizing $\Theta=x_0^{tgt}$):

$$\nabla_\Theta E_{DRFS}=\mathbb{E}_{t,\varepsilon}\Big[w_{DRFS}(t)\big(v_\theta(\hat{x}_t^{tgt})-v_\theta(x_t^{src})\big)-(\dot{a}_t+\dot{c}_t)(x_0^{tgt}-x_0^{src})\Big]$$

where $w_{DRFS}(t)=2(a_t+c_t-\dot{a}_t-\dot{c}_t)$ is a time-varying weight. It is precisely this $c_t$ that makes DRFS a "path-aware" method—it explicitly utilizes the editing path rather than merely evaluating velocities at a fixed interpolation point.

**3. Form of $c_t$ and Descending Time Schedule: Progressive Offset to Prevent Early Error Amplification**

$c_t$ cannot be chosen arbitrarily. The authors constrain it based on two boundary conditions: in the late stages of editing as $t\to 0$, $x_0^{tgt}$ should already lie on the target distribution and should no longer be offset, requiring $c_t\propto t$ so that the offset decays with $t$; the other initial condition is $c_t \to 0$ as $t \to 1$. Consequently, they set $c_t=\frac{k}{T}\,t\approx(1-t)t$ (where $k$ is the current step and $T$ is the total steps), meaning the offset coefficient **progressively increases from 0 to 1** throughout the optimization process. This progressive offset ensures that only a tiny offset is applied in early, high-noise steps, preventing error amplification. As optimization progresses, the offset becomes stronger, steadily pushing the image toward the target. Correspondingly, DRFS employs a **descending time schedule** (coarse updates with larger $t$ first, followed by fine-tuning with smaller $t$) to achieve a coarse-to-fine transition. This allows major modifications like shape/pose changes in early high-noise steps, while refining color and texture in late low-noise steps. In contrast, random schedules mix coarse and fine updates, which easily introduces visible artifacts. Trajectory analysis further indicates that when setting $c_t = \eta t$, a larger $\eta$ results in a straighter editing path (measured by the path-to-chord ratio $SR=\sum_k\|x_{0,k+1}^{tgt}-x_{0,k}^{tgt}\|/\|x_{0,N}^{tgt}-x_{0,0}^{tgt}\|$, where $SR=1$ represents a perfectly straight line) and larger single-step updates. Thus, an intermediate value balances alignment strength and fidelity.

**4. Unified View: DDS and FlowEdit as Special Cases of DRFS**

The offset coefficient $c_t$ of DRFS acts as a knob that unifies existing methods into a spectrum. The authors demonstrate that when $c_t=0$, the DRFS energy $E_{DRFS}$ strictly reduces to the DDS energy $E_{DDS}$ (provable by translating between velocity fields and noise predictions via the equivalence relation $\varepsilon_\theta=\frac{a_t}{\dot{b}_t a_t-\dot{a}_t b_t}(v_\theta-\frac{\dot{a}_t}{a_t}x)$). When using the RF parameterization $(a_t,b_t)=(1-t,t)$ and setting $c_t=t$, the editing trajectory of DRFS strictly reduces to FlowEdit, an inversion-free method. This is because the term $x_0^{tgt}(t)+x_t^{src}-x_0^{src}$ used in FlowEdit to evaluate the target velocity can be precisely interpreted as $\hat{x}_t^{tgt}$ with $c_t=t$. Consequently, DRFS unifies "score-based diffusion optimization (DDS) $\leftrightarrow$ velocity-based rectified flow optimization $\leftrightarrow$ ODE-based editing (FlowEdit)" within a single energy framework, highlighting that the offset choices of DDS ($c_t=0$) and FlowEdit (implicit $c_t=t$) are suboptimal, while its own $c_t\approx(1-t)t$ serves as the sweet spot.

### Loss & Training
The method is essentially training-free latent optimization: it runs for several rounds using an SGD optimizer, a descending time schedule, and a batch size of 1 (a single sampled time step per optimization step). The source and target CFG scales are set to 6 and 16.5, respectively, with unit weights. The base models are the Rectified Flow models Stable Diffusion 3 / 3.5. The entire process requires no network parameter updates or inversion.

## Key Experimental Results

### Main Results
Evaluated on the PIE benchmark (700 multi-task editing images) to compare diffusion-based and rectified flow-based methods. Metrics include structural distance, background preservation (PSNR/LPIPS/MSE/SSIM), and CLIP similarity (whole/edited).

| Method | Model | Structural Dist. ×10³ ↓ | PSNR ↑ | LPIPS ×10³ ↓ | MSE ×10⁴ ↓ | SSIM ×10² ↑ | CLIP-Edited ↑ |
|------|------|------|------|------|------|------|------|
| FlowEdit | SD3 | 27.24 | 22.13 | 105.46 | 87.34 | 83.48 | 23.67 |
| iRFDS | SD3 | 62.72 | 19.61 | 186.39 | 179.76 | 74.59 | 21.67 |
| **DRFS** | **SD3** | **23.05** | **23.38** | **93.81** | **67.49** | **84.85** | **23.83** |
| FlowEdit | SD3.5 | 12.73 | 26.59 | 56.17 | 33.84 | 89.34 | 23.00 |
| DNAEdit | SD3.5 | 14.19 | 26.66 | 74.57 | 32.76 | 88.63 | 22.71 |
| **DRFS** | **SD3.5** | **12.00** | **26.97** | **55.83** | **30.76** | **89.41** | **23.17** |

DRFS achieves the highest editing area CLIP similarity (23.83) among all SD3/SD3.5 methods, indicating the best semantic alignment. Compared with iRFDS, which also follows the RF distillation path, the improvement in background preservation is substantial: LPIPS 93.81 vs. 186.39, MSE 67.49 vs. 179.76, and SSIM 84.85 vs. 74.59. This demonstrates that DRFS successfully mitigates the over-smoothing issue. On SD3.5, it also outperforms FlowEdit, FTEdit, and DNAEdit across almost all metrics.

### Ablation Study
Three configurations of the offset coefficient $c_t$ (on SD3, corresponding to the DDS / Ours / FlowEdit special cases):

| Configuration | Structural Dist. ×10³ ↓ | PSNR ↑ | LPIPS ×10³ ↓ | SSIM ×10² ↑ | CLIP-Edited ↑ | Explanation |
|------|------|------|------|------|------|------|
| $c_t=0$ (≡DDS) | 8.35 | 28.63 | 44.66 | 90.52 | 22.53 | Best background preservation, weakest editing |
| $c_t\approx(1-t)t$ (Ours) | 23.05 | 23.38 | 93.81 | 84.85 | 23.83 | Balanced editing strength and fidelity |
| $c_t=t$ (≡FlowEdit) | 37.28 | 20.71 | 143.06 | 80.27 | 23.21 | Over-editing, poorest fidelity |

### Key Findings
- **$c_t$ acts as a knob for editing strength and background preservation**: A larger $c_t$ yields larger gradient updates, a straighter trajectory, and more aggressive progression toward the target, which enhances editing but degrades background preservation. When $c_t=0$, background alteration is minimal, but the editing performance is weak. The proposed $c_t\approx(1-t)t$ achieves the optimal trade-off between these extremes (attaining the highest CLIP-Edited score of 23.83 while keeping SSIM at 84.85, significantly outperforming $c_t=t$).
- **Progressive offset outperforms fixed offset**: A $c_t$ that progressively increases from 0 prevents error amplification in early high-noise steps, which is why it outperforms the constant $c_t=t$ implied by FlowEdit.
- **Descending time schedules outperform random schedules**: A coarse-to-fine update order (reforming shape and pose first, then refining color and texture) yields fewer artifacts and higher consistency compared to randomly interleaved steps.

## Highlights & Insights
- **Transferring the "subtraction-based cancellation of mutual information" from diffusion to Rectified Flow**: While DDS subtracts noise residuals, DRFS subtracts velocity field residuals, causing background gradients to automatically drop to zero. This is a clean translation supported by a closed-loop sanity check proving that $\phi_{tgt}=\phi_{src}\Rightarrow E=0$.
- **A single offset coefficient organizing three methods into a spectrum**: $c_t=0\to$DDS, $c_t=t\to$FlowEdit, and $c_t\approx(1-t)t\to$Ours. This unifies "energy optimization" and "ODE editing" (which typically appear to be different paradigms) under a single framework, allowing the authors to argue that existing offset choices are suboptimal. This narrative of "first unifying, then finding the sweet spot" is highly compelling.
- **Transferability of the path-aware concept**: Generalizing the observation that "evaluation points must fall on the correct forward posterior," any distillation-sampling-based editing or optimization method can introduce a time-decaying trajectory compensation term to alleviate model-data mismatch.

## Limitations & Future Work
- **Trade-off between editing strength and background preservation remains**: Ablation results show that enhancing CLIP alignment inevitably sacrifices some source fidelity. $c_t$ merely pushes the frontier of this trade-off curve rather than eliminating it entirely; furthermore, the optimal form of $c_t$ is selected using a combination of analysis and empirical results.
- **Base model constrained to SD3/SD3.5 Rectified Flow**: The method relies on the RF velocity field prior; adapting it to other generative paradigms would require reforming the respective equivalence relationships.
- **Batch size of 1 and single-timestep sampling**: Estimating the gradient expectation with a single $(t,\varepsilon)$ may introduce high variance. The impact of increasing the batch size on performance and stability is not extensively detailed (⚠️ subject to the original paper).
- **Directions for improvement**: Making $c_t$ learnable or adaptive to different editing types (e.g., large-scale shape replacement vs. local texture edits may benefit from distinct offset curves) or adjusting $c_t$ spatially to further decouple edited and preserved regions.

## Related Work & Insights
- **vs. DDS**: DDS performs source/target subtraction on diffusion noise residuals to preserve background, and DRFS reveals DDS to be its special case where $c_t=0$. DRFS extends this concept to Rectified Flow velocity fields and adds an offset term to resolve the mismatch from "evaluation points deviating from the trajectory," enabling stronger editing.
- **vs. FlowEdit**: FlowEdit is the first inversion-free RF editing method, estimating target latents directly from source/target velocity offsets. DRFS demonstrates it is a special case equivalent to $c_t=t$ and points out that a constant offset amplifies early errors, whereas the progressive offset $c_t\approx(1-t)t$ is more stable.
- **vs. RFDS / iRFDS**: RFDS extends SDS to the RF velocity field but suffers from over-smoothing; iRFDS relies on pre-inverting noise to remedy this at the cost of additional computation. DRFS is inversion-free, preserves architecture, and directly resolves over-smoothing at the energy level (consistently outperforming iRFDS across LPIPS, MSE, and SSIM).

## Rating
- Novelty: ⭐⭐⭐⭐ residual subtraction + time-varying offset term unify DDS/FlowEdit into the RF distillation framework with clear theoretical justifications
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-metric evaluation on the PIE benchmark + ablation of three special cases of $c_t$ with self-consistent data; however, evaluation is primarily centered on PIE, with cross-dataset results relegated to the appendix.
- Writing Quality: ⭐⭐⭐⭐ Solid mathematical derivations, and a smooth narrative of "unification followed by identifying the sweet spot."
- Value: ⭐⭐⭐⭐ Inversion-free, training-free, architecture-preserving, and plug-and-play, offering high practicality for RF editing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] NAMI: Efficient Image Generation via Bridged Progressive Rectified Flow Transformers](nami_efficient_image_generation_via_bridged_progressive_rectified_flow_transform.md)
- [\[CVPR 2026\] CaReFlow: Cyclic Adaptive Rectified Flow for Multimodal Fusion](careflow_cyclic_adaptive_rectified_flow_for_multimodal_fusion.md)
- [\[CVPR 2026\] FlowDC: Flow-Based Decoupling-Decay for Complex Image Editing](flowdc_flow-based_decoupling-decay_for_complex_image_editing.md)
- [\[CVPR 2026\] VDE: Training-Free Accelerating Rectified Flow Model via Velocity Decomposition and Estimation](vde_training-free_accelerating_rectified_flow_model_via_velocity_decomposition_a.md)
- [\[ICML 2025\] Taming Rectified Flow for Inversion and Editing](../../ICML2025/image_generation/taming_rectified_flow_for_inversion_and_editing.md)

</div>

<!-- RELATED:END -->

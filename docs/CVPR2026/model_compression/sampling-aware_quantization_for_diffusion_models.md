---
title: >-
  [Paper Note] Sampling-Aware Quantization for Diffusion Models
description: >-
  [CVPR 2026][Model Compression][PTQ] This paper points out that the two acceleration paths for diffusion models—"fast samplers" and "network quantization"—conflict when used together: quantization noise perturbs the directional estimation of high-order samplers at each step, causing the smooth Probability Flow ODE to degrade into a variance-exploding SDE.
tags:
  - CVPR 2026
  - Model Compression
  - PTQ
  - QLoRA
date: 2026-05-08
content_hash: febde37636c253cb
---
# Sampling-Aware Quantization for Diffusion Models

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Zeng_Sampling-Aware_Quantization_for_Diffusion_Models_CVPR_2026_paper.html)  
**Code**: https://github.com/TaylorJocelyn/Sampling-aware-Quantization  
**Area**: Model Compression / Diffusion Models / Quantization  
**Keywords**: Diffusion Model Quantization, High-order Samplers, Probability Flow ODE, Trajectory Alignment, PTQ, QLoRA

## TL;DR
This paper points out that the two acceleration paths for diffusion models—"fast samplers" and "network quantization"—conflict when used together: quantization noise perturbs the directional estimation of high-order samplers at each step, causing the smooth Probability Flow ODE to degrade into a variance-exploding SDE. The authors propose "Sampling-Aware Quantization," which uses a Mixed-Order Trajectory Alignment objective to align quantized first-order directional trajectories with full-precision high-order directional trajectories. This linearizes the probability flow, allowing for dual acceleration of "sampling speedup + model compression" under sparse steps with almost no quality degradation.

## Background & Motivation

**Background**: Diffusion models provide high generation quality but are slow due to two bottlenecks: the long denoising chain and the heavy noise estimation network. Research accelerates this via two paths: 1) Designing high-order **fast samplers** (DPM-Solver, DDIM, PLMS, etc.) that use numerical methods to accurately approximate the reverse SDE/ODE with larger step sizes; 2) Using **quantization** to compress the noise estimation network by converting FP32 weights/activations into low-bit integers, reducing memory and per-step computation.

**Limitations of Prior Work**: these two paths have been treated as independent modules. However, when combined directly, the quantization noise $\Delta\epsilon_\theta$ perturbs the directional estimation of the sampler. The problem is particularly severe for high-order samplers: a $k$-order sampler must evaluate $k-1$ intermediate points $\{s_j\}$ within the interval $(t_{i-1}, t_i)$ to jointly estimate the direction. Quantization noise perturbs both the **position** and the **directional estimation** of each intermediate point, causing them to drift over time and eventually contaminating the joint direction.

**Key Challenge**: The "joint directional estimation at multiple intermediate steps" in high-order samplers was designed to lower truncation error and support sparse-step fast sampling. Quantization noise not only destroys this fast convergence potential but also risks transforming the deterministic, smooth Probability Flow ODE into a **variance-exploding SDE**, inducing trajectory diffusion. Under low-bit settings like W4A4, this can lead to complete generation failure.

**Goal**: Achieve "high-fidelity dual acceleration"—quantizing the network while preserving the sparse-step convergence capability of high-order fast samplers, ensuring the two methods do not cancel each other out.

**Key Insight**: The authors analyze quantization error through the lens of sampling acceleration principles. By decomposing the numerical integration error upper bound $L_\Delta=L_{\Delta_{\text{quant}}}+L_{\Delta_{\text{disc}}}$, they find that the total error is dominated by the directional deviation $\delta$ and is non-linearly amplified by the model, far exceeding the discrete truncation error. The conclusion is that $\delta$ must be reduced to the same order as the discrete step size $O(\lambda_t-\lambda_s)$ to constrain the error bound and recover convergence.

**Core Idea**: Redesign the quantization scheme to **learn a more linear probability flow**. Instead of simply minimizing the "MSE of tensors before and after quantization," the objective aligns the quantized low-order sampling directional trajectory to the full-precision high-order sampling directional trajectory (mixed-order trajectory alignment). This suppresses directional deviation $\delta$ and avoids rapid error accumulation leading to sampling diffusion.

## Method

### Overall Architecture
The input to the method is a pre-trained FP32 diffusion network $\epsilon_\theta$ and a target sampler; the output is a low-bit quantized network that remains faithful under sparse-step high-order sampling. The core idea is to change the quantization objective from "per-tensor MSE alignment" to "mixed-order trajectory alignment": aligning the quantized first-order directional trajectory $\hat\epsilon_\theta(x_{\lambda_s},\lambda_s)$ to the full-precision high-order directional trajectory at intermediate nodes $\epsilon_\theta(x_{\lambda_t},\lambda_t)$, thereby linearizing the probability flow. The authors instantiate two versions: SA-PTQ (dual-order trajectory sampling + block-level reconstruction calibration) for 8-bit, and SA-QLoRA (QLoRA fine-tuning with an additional directional cosine constraint) for 4-bit and lower. The alignment logic is generalized to samplers like DDIM and PLMS.

### Key Designs

**1. Mixed-Order Trajectory Alignment: Aligning Quantized First-Order Direction to Full-Precision High-Order Direction**

This is the core of the paper. To clarify what a "trajectory" aligns: a sampling trajectory $\{x_t\}$ is uniquely determined by the sampler, initial point $x_T$, and the schedule. Since each directional step $\epsilon_\theta(x_t,t)$ corresponds to a sample trajectory, **aligning the trajectory is essentially aligning the directional sequence** $\{\epsilon_\theta(x_t,t)\}$. Traditional quantization only minimizes $\mathbb{E}\lVert\epsilon_\theta-\hat\epsilon_\theta\rVert^2$. This work instead aligns the quantized first-order direction trajectory to the values of the full-precision high-order direction trajectory at intermediate node $s$:

$$\arg\min_{s,z}\ \mathbb{E}_{(x_t,t)\sim D,(x_s,s)\sim S}\lVert\hat\epsilon_\theta(x_{\lambda_s},\lambda_s)-\epsilon_\theta(x_{\lambda_t},\lambda_t)\rVert^2$$

The intuition comes from high-order samplers: a $k$-order sampler evaluates $k-1$ intermediate points, where each $\epsilon_\theta(x_{\lambda_{s_i}},\lambda_{s_i})$ encodes high-order derivative information. By forcing the quantized first-order direction to approximate these intermediate directions, the quantization process "internalizes" high-order precision. This compresses $\delta$ to $O(\lambda_t-\lambda_s)$, linearizing the probability flow. Using DPM-Solver-2 as an example, the quantized direction $\hat\epsilon_\theta(\tilde x_{t_{i-1}},t_{i-1})$ is aligned to the full-precision $\epsilon_\theta(u_i,s_i)$ to linearize the trajectory.

**2. SA-PTQ: Dual-Order Trajectory Sampling + Block-Level Reconstruction Calibration for 8-bit PTQ**

For W8A8/W4A8, alignment is applied via Post-Training Quantization (no large-scale retraining, only calibrating a few parameters). Using BRECQ as a baseline and Adaround for weight quantization, only the parameters $\alpha$ are trained. Calibration occurs in two steps: ① **Dual-Order Trajectory Sampling**: Using the same initial $x_T$, a first-order trajectory is collected using a first-order sampler, and a second-order trajectory is collected using a second-order sampler at intermediate points $s_i$. ② **Mixed-Order Alignment Calibration**: For each module $f_i$ and its quantized version $\hat f_i$, the cross-order alignment error is minimized independently:

$$\arg\min_\alpha\ \mathbb{E}_{(t_j,s_j)}\lVert f_i(x_{t_j},t_j,\text{cond})-\hat f_i(x_{s_j},s_j,\text{cond})\rVert^2$$

This aligns the output of the quantized module at intermediate step $s_j$ to the full-precision module at first-order step $t_j$.

**3. SA-QLoRA: Dual-Aware LoRA with Directional Cosine Constraint for 4-bit**

At low bits (e.g., W4A4), PTQ noise is too high. The authors combine mixed-order alignment with QLoRA. Standard QLoRA jointly optimizes LoRA weights $w$ and quantization parameters $s, z$. Since the direction $\epsilon_\theta$ determines the sampling path, a **Directional Cosine Constraint** is added alongside the L2 alignment loss to emphasize directional consistency:

$$L_{\text{COS}}=1-\frac{\langle\epsilon_\theta(x_{t_i},t_i),\hat\epsilon_\theta(x_{s_i},s_i)\rangle}{\lVert\epsilon_\theta(x_{t_i},t_i)\rVert\,\lVert\hat\epsilon_\theta(x_{t_i},t_i)\rVert}$$

$$L_{\text{MOTA}}=\mathbb{E}_{(t_i,s_i)}\lVert\hat\epsilon_\theta(x_{s_i},s_i)-\epsilon_\theta(x_{t_i},t_i)\rVert^2,\quad \arg\min_{w,s,z}\ L_{\text{COS}}+L_{\text{MOTA}}$$

The cosine term preserves direction while the L2 term preserves magnitude, ensuring stable convergence at very low bits.

**4. Generalization to General Samplers: DDIM and PLMS**

DDIM is equivalent to a DPM-Solver-1 update and is treated as a first-order trajectory instance. PLMS (based on PNDM) uses linear multi-step methods to estimate gradients $\epsilon_\theta^{(t)}$. Different orders of numerical methods correspond to different trajectories, allowing the "different orders of $\epsilon_\theta^{(t)}$" to be aligned.

### Loss & Training
SA-PTQ uses Adaround to train $\alpha$ only, minimizing per-block mixed-order alignment MSE. SA-QLoRA jointly trains LoRA weights $w$ and quantization parameters $s, z$ with the objective $L_{\text{COS}}+L_{\text{MOTA}}$. For class-conditional generation, DPM-Solver-1/2 are used as low/high-order samplers with steps=20. Text-to-image aligns PLMS with its lower-order version with steps=50.

## Key Experimental Results

### Main Results
Evaluated on ImageNet 256×256 (LDM-4, steps=20), LSUN, and MS-COCO 512×512 (SD-v1.4, steps=50). Metrics include FID/sFID/IS/Precision/Recall; efficiency is measured in BOPs.

| Config (W/A) | Method | BOPs(T) | IS↑ | FID↓ | sFID↓ |
|------|------|------|------|------|------|
| 32/32 | FP (Full Precision) | 102.20 | 174.33 | 9.45 | 8.08 |
| 8/8 | PTQD | 8.76 | 122.46 | 10.76 | 10.58 |
| 8/8 | **SA-PTQ (ours)** | 8.76 | 120.71 | **10.16** | **9.89** |
| 4/8 | EfficientDM | 4.38 | 132.70 | 9.91 | 8.76 |
| 4/8 | **SA-QLoRA (ours)** | 4.38 | **140.56** | **8.55** | **8.51** |
| 4/4 | EfficientDM | 2.19 | 225.20 | 17.28 | 13.78 |
| 4/4 | **SA-QLoRA (ours)** | 2.19 | **242.03** | **13.73** | **12.45** |

W8A8/W4A8/W4A4 reach bit compression of 3.99×/7.95×/7.95× and bit-op speedups of 11.47×/23.33×/46.67×. At W4A8, SA-QLoRA's FID 8.55 is even lower than the FP32 model. At W4A4, where other methods **fail to generate** due to ODE-to-SDE degradation (PTQ4DM/Q-diffusion/PTQD show no valid results), SA-QLoRA still converges.

### Ablation Study
| Setting | Phenomenon | Explanation |
|------|------|------|
| W8A8 (SA-PTQ) | FID 10.16 < PTQD 10.76 | PTQ variant exceeds previous SOTA. |
| W4A8 (SA-QLoRA) | FID 8.55, IS 140.56 | Better than FP32; quantization adds regularization. |
| W4A4 baseline | Most methods fail to generate | Quantization noise induces trajectory diffusion (ODE→VE-SDE). |
| W4A4 (SA-QLoRA) | FID 13.73, normal convergence | Mixed-order alignment suppresses error accumulation. |

### Key Findings
- **Mixed-order trajectory alignment linearizes probability flow**: Stability metrics across bit-widths confirm that it suppresses rapid error accumulation in high-order samplers.
- **Low-bit is the true test**: At W4A4, other methods collapse as the ODE degrades into a variance-exploding SDE. This method is the only one to converge normally, highlighting the advantage over "per-tensor MSE" quantization.
- **Quantization can provide slight regularization**: At W4A8, FID is lower than FP32, suggesting that aligning high-order trajectories smooths the sampling process.

## Highlights & Insights
- **Unified Perspective**: Connects "sampling acceleration" and "network quantization" in a single error analysis. It proves that quantization cumulative error is dominant and controlled by directional deviation $\delta$, reframing quantization as learning a more linear probability flow.
- **Smart Supervision**: Using the intermediate directions of high-order samplers as a supervisory signal effectively "distills" high-order precision into the quantized network.
- **Directional Cosine Constraint**: Separating directional consistency ($L_{\text{COS}}$) from magnitude is a correct inductive bias for diffusion models, where direction determines the generative path.
- **Deployment Flexibility**: The framework provides both SA-PTQ for 8-bit calibration-only scenarios and SA-QLoRA for 4-bit fine-tuning scenarios.

## Limitations & Future Work
- Analysis focused on DPM-Solver; systems validation on more diverse stochastic samplers (SDE-based) and larger-scale text-to-image models is limited.
- Requires full-precision high-order trajectories as alignment targets, meaning the calibration phase requires collecting intermediate points, which is costlier than per-tensor PTQ.
- While W4A4 converges, the sFID remains higher than full precision (~4.37 higher), indicating the gap in image quality is not fully closed at extreme low bits.

## Related Work & Insights
- **vs PTQ4DM / Q-Diffusion / PTQD**: These adapt traditional quantization to the multi-timestep framework but ignore the impact of noise on high-speed sampling. This work performs joint optimization based on sampling principles and survives at W4A4.
- **vs EfficientDM**: Both use QLoRA, but EfficientDM relies on step-by-step MSE. SA-QLoRA's mixed-order alignment and cosine constraint yield better FID/IS at W4A8/W4A4.
- **vs High-Order Samplers**: Previous works pursued fast sampling in full precision. This work reveals that quantization destroys this synergy and provides a solution to turn "sampler × quantization" into a collaborative process.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to examine quantization error from sampling principles and unify the two acceleration paths.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers conditional/unconditional/T2I and multiple bit-widths; W4A4 results are striking, though sampler diversity is limited.
- Writing Quality: ⭐⭐⭐⭐ Clear error analysis and motivation; formulas are dense.
- Value: ⭐⭐⭐⭐⭐ Solves the practical deployment pain point where "fast sampling + quantization" fails.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Quant Experts: Token-aware Adaptive Error Reconstruction with Mixture of Experts for Large Vision-Language Models Quantization](quant_experts_token_aware_vlm_quantization.md)
- [\[ECCV 2024\] Adaptive Compressed Sensing with Diffusion-Based Posterior Sampling](../../ECCV2024/model_compression/adaptive_compressed_sensing_with_diffusionbased_posterior_sa.md)
- [\[ICML 2025\] Diffusion Sampling Correction via Approximately 10 Parameters](../../ICML2025/model_compression/diffusion_sampling_correction_via_approximately_10_parameters.md)
- [\[CVPR 2026\] DMGD: Train-Free Dataset Distillation with Semantic-Distribution Matching in Diffusion Models](dmgd_train-free_dataset_distillation_with_semantic-distribution_matching_in_diff.md)
- [\[CVPR 2026\] LIFT and PLACE: A Simple, Stable, and Effective Knowledge Distillation Framework for Lightweight Diffusion Models](lift_and_place_a_simple_stable_and_effective_knowledge_distillation_framework_fo.md)

</div>

<!-- RELATED:END -->

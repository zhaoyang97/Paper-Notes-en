---
title: >-
  [Paper Note] LithoGRPO: Fast Inverse Lithography via GRPO Reinforced Flow Matching
description: >-
  [ICML 2026][Image Generation][Inverse Lithography Technology (ILT)] LithoGRPO models lithography mask generation as a rectified flow conditioned on target layouts and fine-tunes it using Group Relative Policy Optimizatio…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Inverse Lithography Technology (ILT)"
  - "Rectified Flow"
  - "GRPO"
  - "Non-differentiable Metrics"
  - "Shot Count"
date: 2026-05-08
content_hash: 8a3ef3996d42c360
---

# LithoGRPO: Fast Inverse Lithography via GRPO Reinforced Flow Matching

**Conference**: ICML 2026  
**arXiv**: [2606.00228](https://arxiv.org/abs/2606.00228)  
**Code**: https://github.com/laiyao1/LithoGRPO  
**Area**: Scientific Computing / Semiconductor Manufacturing / Flow Matching / Reinforcement Learning  
**Keywords**: Inverse Lithography Technology (ILT), Rectified Flow, GRPO, Non-differentiable Metrics, Shot Count

## TL;DR
LithoGRPO models lithography mask generation as a rectified flow conditioned on target layouts and fine-tunes it using Group Relative Policy Optimization (GRPO). This allows a single forward pass to simultaneously optimize four types of lithography metrics: L2/PVB (differentiable) and EPE/Shot (non-differentiable). Combined with a fast shot-count algorithm accelerated by 130×–490×, it improves the comprehensive ranking on LithoBench from 5.6 to 4.3, with a single-sample inference time of only 0.1 s.

## Background & Motivation

**Background**: In semiconductor manufacturing, lithography projects circuit layouts onto wafers via masks. When feature sizes shrink below the exposure wavelength, diffraction causes the printed image to deviate significantly from the target layout. Traditional Optical Proximity Correction (OPC) performs local displacements on existing edges, whereas Inverse Lithography Technology (ILT) treats the entire mask as a pixel-level inverse problem, representing the current strongest paradigm. ILT methods are generally divided into optimization-based (e.g., MOSAIC, LevelSet, using iterative gradient descent) and learning-based (e.g., GAN-OPC, Neural-ILT, DAMO, ILILT, diffusion models, using end-to-end image-to-image mapping).

**Limitations of Prior Work**: Optimization-based methods are slow and limited to differentiable objectives. Learning-based methods are hindered by two factors: first, supervised data originates from optimization results, capping the quality ceiling; second, training losses must be differentiable, leading to the neglect of two critical metrics for yield and cost—**Edge Placement Error (EPE, discrete counting)** and **Shot Count (the number of rectangular shots a mask is decomposed into, directly related to mask writing cost)**—during training. Diffusion-based ILT (e.g., DiffOPC, AdaOPC) provides high image quality but suffers from slow multi-step sampling inference.

**Key Challenge**: The ILT objective function is naturally a "differentiable + non-differentiable" hybrid. L2 and PVB support gradients, whereas EPE and Shot do not. Furthermore, these four metrics conflict (optimizing L2 often fragments mask geometry, causing Shot counts to surge). Pure generative models only learn the training data distribution without a "metric feedback" channel, while pure optimization can only navigate differentiable landscapes.

**Goal**: To simultaneously optimize four metrics within a unified framework while maintaining single-step inference speed, and to accelerate the Shot evaluation bottleneck to be feasible within a training loop.

**Key Insight**: The authors analogize ILT to "image synthesis with physical rewards." Lithography metrics are explicit, deterministic scalar functions naturally suited as RL rewards, corresponding to the recent trend of applying GRPO to flow models.

**Core Idea**: Use **rectified flow** to model the mask as a linear ODE from noise to mask (one-step inference). Then, use **GRPO** to generate multiple candidate masks under the same target via randomized SDE sampling, calculating advantages through group-normalized rewards across the four metrics to incorporate non-differentiable metrics into gradient updates. Simultaneously, a **minimum overlapping rectangle cover ILP** replaces the NP-hard traditional shot-count to enable the RL training loop.

## Method

### Overall Architecture
LithoGRPO is an ILT generation framework featuring three-stage training and single-step rectified flow inference:

- **Input**: Target layout $\mathbf{T}$ (channel-concatenated with noise $\mathbf{x}_t$ as a condition).
- **Model**: An 87M parameter U-Net, parameterizing the time-dependent velocity field $\mathbf{v}_\theta(\mathbf{x}_t, t; \mathbf{T})$.
- **Loss & Training**: (1) **Pretraining** — Training on (T, mask) pairs from datasets using rectified flow MSE loss to learn basic mask-target alignment; (2) **SFT** — Adding differentiable metrics $\mathrm{L2}(\mathbf{x}_1, \mathbf{T}) + \mathrm{PVB}(\mathbf{x}_1)$ to the flow loss to saturate differentiable metrics (at the cost of increased Shot counts); (3) **RLFT** — GRPO fine-tuning using the normalized negative sum of all four metrics as a reward to reduce Shot counts without degrading L2/PVB/EPE.
- **Inference**: Starting from Gaussian noise, a single Euler step $\mathbf{x}_1 = \mathbf{x}_0 + \mathbf{v}_\theta(\mathbf{x}_0, 0; \mathbf{T})$ is used to output the mask. A $512 \times 512$ mask is generated in 0.1 s.

From the perspective of lithography physics, the mask $\mathbf{x}$ passes through a Hopkins diffraction model $\mathbf{I} = \sum_k \mu_k |h_k \otimes \mathbf{x}|^2$ to produce the aerial image, followed by a sigmoid-softened threshold $\mathbf{Z} = 1/(1+\exp[-\alpha(\mathbf{I}-I_\mathrm{th})])$ to obtain the resist image. The entire chain $g(\mathbf{x}) = f(h(\mathbf{x}))$ is differentiable with respect to $\mathbf{x}$, providing a backpropagation path for L2 and PVB.

### Key Designs

1. **Three-stage Pretrain → SFT → RLFT Flow Matching Training**:
    - **Function**: Decouples generation from metric optimization, feeding metrics with different properties in stages to avoid being stuck by discrete objectives like Shot initially.
    - **Mechanism**: Pretraining uses standard rectified flow loss $\mathcal{L}_\mathrm{flow} = \mathbb{E}[\|\mathbf{v}_\theta(\mathbf{x}_t, t) - (\mathbf{x}_1 - \mathbf{x}_0)\|^2]$. In the SFT stage, the current velocity at any intermediate time $t$ is projected to the endpoint $\mathbf{x}_1 = \mathbf{x}_t + (1-t)\mathbf{v}_\theta$, and differentiable metrics are calculated on $\mathbf{x}_1$, with loss $\mathcal{L}_\mathrm{sft} = \lambda_\mathrm{flow}\mathcal{L}_\mathrm{flow} + \lambda_{\mathrm{L2}}\mathrm{L2} + \lambda_\mathrm{PVB}\mathrm{PVB}$. After freezing this initialization, RLFT uses GRPO to allow EPE/Shot to influence parameters.
    - **Design Motivation**: Training dynamics (Fig. 4) show that L2/EPE decrease monotonically during Pretrain+SFT while Shot increases—a physical trade-off of "fragmenting masks for fidelity." Separating the stages allows RLFT to specifically refine Shot counts once differentiable metrics are saturated.

2. **GRPO + Color Noise SDE Sampling**:
    - **Function**: Introduces randomness to generate $G=6$ candidate masks without changing the flow model's marginal distribution, enabling intra-group advantage normalization on non-differentiable metrics.
    - **Mechanism**: The deterministic ODE is rewritten as an equivalent SDE, discretized via Euler–Maruyama: $\mathbf{x}_{t+\Delta t} = \mathbf{x}_t + [\mathbf{v}_\theta + \frac{\sigma_t^2}{2t}(\mathbf{x}_t + (1-t)\mathbf{v}_\theta)]\Delta t + \sigma_t\sqrt{\Delta t}\boldsymbol{\varepsilon}$, where $\sigma_t = a\sqrt{(1-t)/t}$. The reward is the normalized negative sum $R = -\sum_{k \in \{\mathrm{L2,PVB,EPE,Shot}\}} k/k_0$. The advantage $A_i = (R_i - \mathrm{mean}) / (\mathrm{std} + \varepsilon)$ is updated using standard PPO/GRPO clipping. Crucially, $\boldsymbol{\varepsilon}$ uses **low-frequency color noise** (low-pass filtered white noise) to prevent high-frequency fragmentation that would spike shot counts.
    - **Design Motivation**: Flow models are deterministic; ILT requires "blocky" geometry. Color noise resolves the conflict between the need for RL exploration and mask manufacturability.

3. **Fast Shot Count via Minimum Overlapping Rectangle Cover ILP (130×–490× Speedup)**:
    - **Function**: Approximates the NP-hard "minimum non-overlapping rectangle partition" as an ILP to reduce evaluation time from ~60 s to 0.2 s per mask.
    - **Mechanism**: (i) Enumereate local maximal rectangles in $O(N^2)$ using histogram-based scanning; (ii) Prune redundant candidates in $O(K^2)$; (iii) Construct a set-cover ILP with constraints generated via row scanning in $O(NK^2)$, solved with PuLP.
    - **Design Motivation**: Traditional shot counting is too slow for RL loops. The authors prove that GRPO's intra-group normalization (Eq. 12) effectively cancels constant offsets, meaning **as long as the intra-group ranking is preserved, the policy gradient remains unaffected**. The fast shot count correlates with the traditional one at $R^2 = 0.994$.

### Loss & Training
- Total training = 50 epoch Pretrain + 25 epoch SFT + 1000 step RLFT.
- Standard clipped GRPO loss: $\mathcal{L}_\mathrm{grpo} = -\mathbb{E}_\mathbf{T}[\sum_i \min(r_i A_i, \mathrm{clip}(r_i, 1-\varepsilon, 1+\varepsilon) A_i)]$.
- Gaussian approximation $\mathcal{N}(\boldsymbol{\mu}_t, \sigma_t^2 \Delta t \mathbf{I})$ is used for log-probs.
- Hardware: 4 × RTX 3090, < 8 hours per stage.

## Key Experimental Results

### Main Results
On LithoBench (MetalSet / StdMetal / ViaSet / StdContact), evaluated on 4 metrics + inference time:

| Category | Method | MetalSet L2 | MetalSet Shot | ViaSet L2 | StdContact Shot | Time (s) | Avg. Rank |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Optimization | MOSAIC | 35860 | 361 | – | – | 0.940 | 9.8 |
| Optimization | LevelSet | 34712 | 263 | 9632 | 275 | 2.290 | 6.9 |
| Optimization | MultiLevel | 27893 | 1250 | 4268 | 1473 | 1.030 | 5.6 |
| Learning | GAN-OPC | 43414 | 574 | 14767 | 276 | 0.010 | 7.4 |
| Learning | Neural-ILT | 36670 | 476 | 12723 | 265 | 0.025 | 6.5 |
| Learning | DAMO | 32579 | 523 | 5081 | 458 | 0.028 | 5.7 |
| Hybrid | ILILT | 30353 | 433 | 4666 | 510 | 0.441 | 5.9 |
| **Ours** | **LithoGRPO (RLFT)** | **28933** | 444 | **4276** | **889** | 0.104 | **4.3** |

LithoGRPO (RLFT) achieved an average rank of 4.3, significantly outperforming the best baseline MultiLevel (5.6).

### Ablation Study

| Configuration | MetalSet Shot ↓ | Key Observation |
| :--- | ---: | :--- |
| Pretrain only | 487 | Flow baseline |
| + SFT | 803 | L2/PVB dropped significantly, but Shot surged 65% |
| + RLFT (Default) | **444** | Shot reduced by 45% vs SFT without losing L2 quality |
| RLFT + White Noise | ↑ | Mask fragmentation increased Shot significantly |
| RLFT + Color Noise ($a=0.1$) | **444** | Optimal exploration vs manufacturability |

### Key Findings
- **Multistage separation is essential**: Shot and L2 are physically conflicting. Separating SFT and RLFT allows the model to refine Shot on a fidelity-saturated initialization.
- **Color noise is a critical engineering trick**: It maintains mask continuity while providing the exploration required for RL.
- **GRPO's robustness to reward offsets** justifies the use of fast, approximate rewards for NP-hard metrics.

## Highlights & Insights
- **"Metric as Reward" Shift**: ILT tasks with explicit physical metrics are better suited for GRPO than text-to-image tasks because rewards are ground-truth physics rather than learned models.
- **Flow Matching + RL Integration**: This is the first application in the ILT domain, bypassing diffusion's speed bottlenecks while retaining SDE exploration.
- **Algorithm-Training Co-design**: The fast shot count is specifically designed for RL, prioritizing ranking consistency over absolute numerical precision.

## Limitations & Future Work
- **Computational Cost**: The three-stage training takes approx. 24 GPU·h, an order of magnitude higher than some pure learning methods.
- **Evaluation Scope**: High-end industrial processes (EUV, larger layouts) were not evaluated.
- **Hyperparameter Sensitivity**: $G$ and $a$ require manual tuning and lack adaptive mechanisms.

## Related Work & Insights
- Compared to **ILILT**, LithoGRPO is faster (0.1 s vs 0.44 s) and handles non-differentiable metrics, leading to a better rank (4.3 vs 5.9).
- Compared to **Diffusion-based ILT**, rectified flow allows for more computationally feasible RL fine-tuning due to single-step efficiency.
- This work validates the transferability of the **GRPO-on-flow** paradigm to scientific computing tasks with known physical constraints.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (Introduction of GRPO/Flow to ILT; ILP-based reward approximation)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Extensive benchmarks, though lacks industrial EUV nodes)
- Writing Quality: ⭐⭐⭐⭐ (Convincing visualization of dynamics and noise)
- Value: ⭐⭐⭐⭐⭐ (Direct benefit to semiconductor yield/cost; clean demonstration of RL in scientific computing)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Saving Foundation Flow-Matching Priors for Inverse Problems](saving_foundation_flow-matching_priors_for_inverse_problems.md)
- [\[ICML 2026\] (HB-ARFM) History-Bootstrapped Flow Matching for Inverse Boiling Reconstruction](hb-arfm_history-bootstrapped_flow_matching_for_inverse_boiling_reconstruction.md)
- [\[ICLR 2026\] SafeFlowMatcher: Safe and Fast Planning using Flow Matching with Control Barrier Functions](../../ICLR2026/image_generation/safeflowmatcher_safe_and_fast_planning_using_flow_matching_with_control_barrier_.md)
- [\[CVPR 2026\] Neighbor GRPO: Contrastive ODE Policy Optimization Aligns Flow Models](../../CVPR2026/image_generation/neighbor_grpo_contrastive_ode_policy_optimization_aligns_flow_models.md)
- [\[ICML 2026\] A Kinetic Energy Perspective of Flow Matching](a_kinetic_energy_perspective_of_flow_matching.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICML 2026\] Saving Foundation Flow-Matching Priors for Inverse Problems](saving_foundation_flow-matching_priors_for_inverse_problems.md)
- [\[ICML 2026\] (HB-ARFM) History-Bootstrapped Flow Matching for Inverse Boiling Reconstruction](hb-arfm_history-bootstrapped_flow_matching_for_inverse_boiling_reconstruction.md)
- [\[CVPR 2026\] Neighbor GRPO: Contrastive ODE Policy Optimization Aligns Flow Models](../../CVPR2026/image_generation/neighbor_grpo_contrastive_ode_policy_optimization_aligns_flow_models.md)
- [\[ICLR 2026\] SafeFlowMatcher: Safe and Fast Planning using Flow Matching with Control Barrier Functions](../../ICLR2026/image_generation/safeflowmatcher_safe_and_fast_planning_using_flow_matching_with_control_barrier_.md)
- [\[ICML 2026\] Exploring and Exploiting Stability in Latent Flow Matching](exploring_and_exploiting_stability_in_latent_flow_matching.md)

</div>

<!-- RELATED:END -->

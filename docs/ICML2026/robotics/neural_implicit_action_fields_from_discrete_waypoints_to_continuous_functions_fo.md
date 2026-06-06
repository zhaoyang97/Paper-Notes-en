---
title: >-
  [Paper Note] Neural Implicit Action Fields: From Discrete Waypoints to Continuous Functions for Vision-Language-Action Models
description: >-
  [ICML 2026][Robotics][VLA] NIAF transforms the VLA model's "action chunk" from a sequence of discrete waypoints into a continuous time function $\mathcal{A}(\tau)=\Phi(\tau…
tags:
  - "ICML 2026"
  - "Robotics"
  - "VLA"
  - "SIREN"
  - "Implicit Neural Representations"
  - "Impedance Control"
  - "Action Chunking"
date: 2026-05-08
content_hash: 31b484ea6ea94a64
---

# Neural Implicit Action Fields: From Discrete Waypoints to Continuous Functions for Vision-Language-Action Models

**Conference**: ICML 2026  
**arXiv**: [2603.01766](https://arxiv.org/abs/2603.01766)  
**Code**: TBD  
**Area**: Robotics / VLA / Embodied AI  
**Keywords**: VLA, SIREN, Implicit Neural Representations, Impedance Control, Action Chunking  

## TL;DR
NIAF transforms the VLA model's "action chunk" from a sequence of discrete waypoints into a continuous time function $\mathcal{A}(\tau)=\Phi(\tau;\theta)$. By employing an MLLM as a "layered spectral modulator" for SIREN to output parameters $\theta$, the method achieves $C^\infty$ smooth trajectories, arbitrary frequency queries, and analytically differentiable velocity/jerk signals. This results in SOTA performance on CALVIN/LIBERO and eliminates jitter in real-world impedance control.

## Background & Motivation

**Background**: Vision-Language-Action (VLA) models have evolved from single-step token autoregressive prediction (e.g., RT-2, OpenVLA) to "action chunking" (e.g., ACT, Diffusion Policy), and further to compressing action sequences into discrete tokens using B-spline control points or DCT coefficients (e.g., BEAST, FAST). A commonality among these is that the final actions are represented as a sequence of discrete waypoints, which are tied to the training data collection frequency.

**Limitations of Prior Work**: The forced discretization of continuous physical motion introduces three specific issues: (1) **Time Resolution Binding**: Models can only output points at the training frequency; higher frequency execution requires interpolation, which introduces artifacts. (2) **Lack of High-order Dynamics Supervision**: BEAST destroys the analytical continuity of splines by quantizing control points into a codebook, while other methods lack high-order constraints entirely. Discontinuous velocity curves lead to motor jitter. (3) **Inability to Analytically Differentiate**: Discrete representations rely on numerical differentiation to recover velocity, which amplifies quantization noise and fails to provide clean feedforward terms for impedance control—forcing robots to use rigid position control, which performs poorly in compliant operations (e.g., insertion, stacking).

**Key Challenge**: Physical control is inherently a continuous function $\mathcal{A}: t\to \mathbb{R}^{\dim}$, but the LLM-token paradigm naturally produces discrete sequences. There is a mismatch between these two mathematical structures—obtaining smooth velocity/jerk typically requires abandoning tokenization or accepting quantization loss.

**Goal**: Define a new representation where "action = parameterized continuous function," repurposing the MLLM as a "parameter predictor" rather than a "waypoint predictor." Simultaneously, ensure the representation itself is $C^\infty$ continuous and analytically differentiable, providing position and velocity feedforward terms required for impedance control in a single pass.

**Key Insight**: Implicit Neural Representations (INR) have proven capable of representing signals with high fidelity in continuous functions (as seen in NeRF). The $\sin$ activation in SIREN (sinusoidal representation network) allows all orders of derivatives to be expressed analytically. By representing action chunks as SIREN and using a hypernetwork mechanism to let the MLLM predict SIREN parameters, one gains both the semantic understanding of LLMs and the physical smoothness of continuous functions.

**Core Idea**: Represent action chunks as $\mathcal{A}(\tau)=\Phi(\tau;\theta)$, where $\theta$ is mapped by the MLLM through a set of learnable query embeddings. The latent output of the MLLM is not waypoints, but the "frequency modulation $\gamma$ + phase modulation $\beta$" for each layer of SIREN, which perturbs a shared set of motion-prior meta-parameters.

## Method

### Overall Architecture
A two-stage process:

1.  **Multimodal Context Encoding**: RGB observations $\mathcal{o}$ + instructions $\mathcal{t}$ + a set of learnable query embeddings $\mathbf{E}_{qry}\in\mathbb{R}^{Q\times d}$ are fed into a pretrained MLLM decoder. A **single parallel decoding pass** outputs $\mathbf{Z}=\text{MLLM}(\mathbf{E}_{qry};\mathcal{o},\mathcal{t})\in\mathbb{R}^{Q\times d}$.
2.  **Action Manifold Decoding**: $\mathbf{Z}$ is partitioned into $L$ segments, each projected into the $(\boldsymbol{\gamma}^{(\ell)}, \boldsymbol{\beta}^{(\ell)})$ for the $\ell$-th layer of SIREN. These are used to modulate a set of meta-parameters $(\mathbf{W},\mathbf{b})$ shared across all tasks, resulting in an instance-specific SIREN $\Phi(\tau;\theta)$. Position is obtained by querying at any time $\tau\in[-1,1]$, while velocity, acceleration, and jerk are obtained via automatic differentiation.

The input side requires only one forward pass. During inference, $K$ time points $\tau_k = -1 + \frac{2k}{K-1}$ are sampled as needed to obtain the action sequence, completely decoupling it from the training frequency.

### Key Designs

1.  **MLLM as a Layered Spectral Modulator (Hypernetwork)**:
    *   **Function**: Utilizes the semantic understanding of the language model to modulate a shared "motion prior" to suit the current task.
    *   **Mechanism**: Constraints $Q = L\times (G+1)$, partitioning $\mathbf{Z}$ by SIREN layers. The first $G$ tokens of each layer are projected into frequency modulation $\boldsymbol{\gamma}^{(\ell)} = \text{Concat}(\psi_{\gamma_1}(\mathbf{Z}_{(\ell,1)}),\dots,\psi_{\gamma_G}(\mathbf{Z}_{(\ell,G)}))$, and the last token is projected into phase modulation $\boldsymbol{\beta}^{(\ell)} = \psi_\beta(\mathbf{Z}_{(\ell,bias)})$. The final parameters are $\hat{\mathbf{W}}^{(\ell)} = \mathbf{W}^{(\ell)}\odot(\mathbf{1}+\boldsymbol{\gamma}^{(\ell)})$ and $\hat{\mathbf{b}}^{(\ell)} = \mathbf{b}^{(\ell)} + \boldsymbol{\beta}^{(\ell)}$.
    *   **Design Motivation**: (a) Instead of regressing the entire SIREN weights from scratch (which would lead to parameter explosion), the model predicts lightweight modulation coefficients. This embeds "general motion laws" in $(\mathbf{W},\mathbf{b})$ and "task differences" in $(\gamma,\beta)$—similar to the design philosophy of LoRA or modulated INRs. (b) Layer-wise modulation offers stronger expressive power than a single global embedding, avoiding information bottlenecks.

2.  **SIREN for $C^\infty$ Continuity and Analytical High-order Derivatives**:
    *   **Function**: Ensures the action representation is naturally smooth with closed-form derivatives for all orders, supporting joint position/velocity supervision and jitter suppression.
    *   **Mechanism**: SIREN forward pass $\mathbf{h}^{(\ell)} = \sin(\omega_0(\hat{\mathbf{W}}^{(\ell)}\mathbf{h}^{(\ell-1)} + \hat{\mathbf{b}}^{(\ell)}))$, where $\mathcal{A}(\tau) = \mathbf{W}_{out}\mathbf{h}^{(L)} + \mathbf{b}_{out}$. Velocity is computed via the chain rule recursion $\mathbf{v}(\tau) = \hat{\mathbf{W}}_{out}\dot{\mathbf{h}}^{(L)}$, where $\dot{\mathbf{h}}^{(\ell)} = \cos(\mathbf{u}^{(\ell)})\odot(\hat{\mathbf{W}}^{(\ell-1)}\dot{\mathbf{h}}^{(\ell-1)})$. Jerk $\mathbf{j}(\tau)$ is obtained by differentiating twice more.
    *   **Design Motivation**: ReLU-based INRs lack second-order derivatives, and quantizing B-spline control points destroys differentiability. Only SIREN's $\sin/\cos$ isomorphic derivative chain allows "position predictor = velocity predictor = jerk predictor," fundamentally avoiding numerical differentiation noise. The $\omega_0$ frequency factor ensures the network operates at an appropriate scale from the start.

3.  **Physical Consistency Supervision: Position + Analytical Velocity + Jerk Regularization**:
    *   **Function**: Forces the network to fit not just waypoints but also velocities while suppressing jerk, resulting in the "clean feedforward" required for impedance control.
    *   **Mechanism**: In simulation, $\mathcal{L}_{\text{pos}} = \frac{1}{K}\sum_k \|\Phi(\tau_k) - \mathbf{a}_{gt,k}\|_2^2$ is used. For real robots, $\mathcal{L}_{\text{vel}} = \frac{1}{K}\sum_k \|\frac{2}{T}\nabla_\tau \Phi(\tau_k) - \mathbf{v}_{gt,k}\|_2^2$ (ground truth velocity from FOC drivers) and jerk regularization $\mathcal{L}_{\text{jerk}} = \frac{1}{K}\sum_k \|(\frac{2}{T})^3 \nabla_\tau^3 \Phi(\tau_k)\|_2^2$ are added. The total loss is $\mathcal{L}_{\text{real}} = \lambda_p \mathcal{L}_{\text{pos}} + \lambda_v \mathcal{L}_{\text{vel}} + \lambda_j \mathcal{L}_{\text{jerk}}$. Inference uses the impedance law $\mathbf{u}_{cmd} = \mathbf{K}_p(\Phi(\tau)-\mathbf{a}_{curr}) + \mathbf{K}_d(\frac{2}{T}\nabla_\tau \Phi(\tau) - \mathbf{v}_{curr})$.
    *   **Design Motivation**: Since position and velocity supervision come from independent measurements (vision vs. FOC encoders), constraining the same function $\Phi$ by both creates cross-signal regularization—encouraging the model to discard inconsistent noise in either signal and produce physically self-consistent trajectories. Jerk regularization further suppresses motor vibration.

### Loss & Training
- Simulation (CALVIN / LIBERO): Only $\mathcal{L}_{\text{pos}}$ is used (no velocity feedback), with SIREN's $C^\infty$ bias acting as implicit regularization.
- Real Robot: All three terms of $\mathcal{L}_{\text{real}}$ are used.
- Employs single-pass parallel decoding (no iterative denoising like flow-matching), offering significant inference speed advantages.
- Experiments conducted across different backbones: Florence-2 Large / Qwen3-VL / $\pi_{0.5}$.

## Key Experimental Results

### Main Results

| Dataset | Metric | NIAF (Ours) | Prev. SOTA | Gain |
| :--- | :--- | :--- | :--- | :--- |
| CALVIN ABCD→D | Avg. Len | **4.66** | 4.62 (FLOWER) | +0.04 |
| CALVIN ABC→D | Avg. Len | **4.47** | 4.44 (FLOWER) | +0.03 |
| LIBERO-Object (Florence-2) | Success % | **100.0** | 98.8 ($\pi_0$) | +1.2 |
| LIBERO Mean (Florence-2) | Success % | **97.9** | 95.7 (FLOWER) | +2.2 |
| LIBERO Mean (Qwen3-VL) | Success % | **97.7** | 96.6 (OFT) | +1.1 |
| Real Item Placement | Success % | **90** | < (BEAST/OFT) | Significant |
| Real Cup Stacking | Success % | **80** | < (BEAST/OFT) | Significant |

Note: NIAF beats the 9B UniVLA on CALVIN with only 0.77B parameters and **no large-scale robot data pre-training**.

### Ablation Study

| Configuration | Key Metric | Description |
| :--- | :--- | :--- |
| Full NIAF (Florence-2) | LIBERO-Long 95.5 | Full model |
| BEAST-F (Discrete control points) | LIBERO-Long ~86 | Accuracy loss due to quantization |
| BEAST-CT (Continuous control points) | LIBERO-Long < NIAF | Refutes "non-discrete is enough" hypothesis |
| OFT (MLP direct waypoint) | LIBERO-Long < NIAF | Lacks analytical continuity |
| FAST (Autoregressive) | LIBERO-Long < NIAF | Token serializing is slow and not smooth |
| $\pi_{0.5}$-NIAF vs. $\pi_{0.5}$-BEAST | Shape Insertion (Better) | Continuous representation is decisive for precision |

### Key Findings
- **Continuous $\neq$ Non-discrete**: BEAST-CT also uses continuous control points but still lags behind NIAF. The real advantage stems from $C^\infty$ smoothness and analytical differentiability, not just "avoiding quantization."
- **Real-world Velocity Curve Comparison**: Measured velocities for BEAST/OFT show high-frequency jitter oscillating around zero (forcing rigid position control), while NIAF’s velocity curves follow the true motion trend with non-zero mean continuous lines—making impedance control viable.
- **Small Models + No Pre-training Can Win**: The 0.77B NIAF outperforming the 9B UniVLA on CALVIN suggests that structural improvements to action representation are more effective than simply stacking parameters or data.

## Highlights & Insights
- **Applying Hypernetwork to the Right Scenario**: While HyperVLA / Trans-INR have shown the feasibility of using MLLMs as hypernetworks, NIAF is the first to apply it to "outputting continuous action functions" where the token paradigm previously reigned. The layered modulation ($\gamma, \beta$ per layer) is well-implemented with clear physical meaning (frequency and phase).
- **Shared Motion-Prior is an Underrated Design**: Modulating rather than rewriting ensures that the shared $(\mathbf{W},\mathbf{b})$ captures "general robotic motion grammar," while tasks only learn the differences. This reduces the output burden on the MLLM and lowers the risk of overfitting to single tasks. This approach can transition to any generation scenario requiring "fast task adaptation."
- **Analytical Differentiation as a System-level Lever**: While many papers treat "acceleration/jerk terms in the training objective" as a nice-to-have, NIAF turns this into a binary switch for impedance control. Discrete $\to$ numerical differentiation $\to$ noise amplification $\to$ mandatory rigid position control; Continuous differentiable $\to$ analytical velocity $\to$ executable impedance law (14). This is a critical chain from representation to system performance.

## Limitations & Future Work
- The authors acknowledge that NIAF modifies the action head and does not improve the high-level reasoning or zero-shot generalization of the base VLM. Advantages on unseen objects/instructions still primarily depend on the backbone and data diversity.
- Real-world velocity supervision depends on high-frequency FOC feedback. Low-cost platforms can only numerically differentiate position, which might amplify demonstration noise and potentially degrade $\mathcal{L}_{\text{vel}}$ quality.
- Best suited for compliant operations, long-horizon tasks, or cross-frequency execution. For short-range, fixed-frequency, simple pick-and-place tasks not requiring smooth velocity, discrete waypoints remain simple and practical.
- $\pi_{0.5}$-NIAF performed worse than native $\pi_{0.5}$ on Towel Folding—replacing the action head "forgets" the pre-trained knowledge of the original flow-matching head. Safely inheriting pre-training remains an open question.

## Related Work & Insights
- **vs BEAST** (Zhou et al., 2026): BEAST compresses actions using B-spline control points but quantizes them into a codebook, breaking differentiability; NIAF uses SIREN to preserve both compression and continuity.
- **vs $\pi_0$ / GR00T N1** (Black et al., 2024; Bjorck et al., 2025): Flow-matching requires multi-step iterative denoising; NIAF obtains a full continuous function in a single-step parallel decoding pass, making it faster and naturally smooth.
- **vs ACT / Diffusion Policy** (Zhao et al., 2023; Chi et al., 2025): Early chunking solved trajectory coherence but remained waypoint-based; NIAF rewrites the chunk itself as a function.
- **vs OpenVLA-OFT** (Kim et al., 2025): OFT uses an MLP to map queries to single-step waypoints; NIAF maps queries to SIREN parameters, offering superior expressive power and differentiability.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The combination of "MLLM as SIREN hypernet" + "Layered spectral modulation" + "Analytical dynamics supervision" represents a genuine paradigm shift.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers CALVIN, LIBERO, four real-world tasks, and three backbones. The velocity curve visualizations are highly convincing.
- Writing Quality: ⭐⭐⭐⭐⭐ Three Definitions clarify the paradigm shift, and the derivation chain (6)-(14) is complete and self-consistent.
- Value: ⭐⭐⭐⭐⭐ Addresses the critical bottleneck in transitioning VLA from "capable of movement" to "capable of compliant movement." Expected to be widely adopted as a new default action head.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Discrete Diffusion VLA: Bringing Discrete Diffusion to Action Decoding in Vision-Language-Action Policies](discrete_diffusion_vla_bringing_discrete_diffusion_to_action_decoding_in_vision-.md)
- [\[ICML 2026\] LangForce: Bayesian Decomposition of Vision-Language-Action Models via Latent Action Queries](langforce_bayesian_decomposition_of_vision_language_action_models_via_latent_act.md)
- [\[ICML 2026\] StableVLA: Towards Robust Vision-Language-Action Models without Extra Data](stablevla_towards_robust_vision-language-action_models_without_extra_data.md)
- [\[ICML 2026\] Latent Reasoning VLA: Latent Thinking and Prediction for Vision-Language-Action Models](latent_reasoning_vla_latent_thinking_and_prediction_for_vision-language-action_m.md)
- [\[ICML 2026\] Spatial Memory for Out-of-Vision Manipulation in Vision-Language-Action](spatial_memory_for_out-of-vision_manipulation_in_vision-language-action.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Inference-Time Scaling for Flow Models via Stochastic Generation and Rollover Budget Forcing
description: >-
  [NeurIPS2025][Image Generation][Flow Models] This paper proposes inference-time scaling methods for flow models: stochasticity is introduced via ODE→SDE conversion to enable particle sampling; the search space is expanded through linear→VP interpolant conversion; and a Rollover Budget Forcing (RBF) strategy is designed to adaptively allocate the computational budget. The approach substantially outperforms all existing methods on compositional text-to-image generation and quan…
tags:
  - "NeurIPS2025"
  - "Image Generation"
  - "Flow Models"
  - "Inference-Time Scaling"
  - "Particle Sampling"
  - "SDE Conversion"
  - "Interpolant Conversion"
  - "Rollover Budget Forcing"
  - "FLUX"
date: 2026-05-08
content_hash: fd03b1c111b81176
---

# Inference-Time Scaling for Flow Models via Stochastic Generation and Rollover Budget Forcing

**Conference**: NeurIPS2025
**arXiv**: [2503.19385](https://arxiv.org/abs/2503.19385)  
**Code**: [flow-inference-time-scaling](https://flow-inference-time-scaling.github.io/)  
**Area**: Image Generation / Inference-Time Scaling
**Keywords**: Flow Models, Inference-Time Scaling, Particle Sampling, SDE Conversion, Interpolant Conversion, Rollover Budget Forcing, FLUX

## TL;DR

This paper proposes inference-time scaling methods for flow models: stochasticity is introduced via ODE→SDE conversion to enable particle sampling; the search space is expanded through linear→VP interpolant conversion; and a Rollover Budget Forcing (RBF) strategy is designed to adaptively allocate the computational budget. The approach substantially outperforms all existing methods on compositional text-to-image generation and quantity-aware generation tasks.

## Background & Motivation

- **Inference-Time Scaling**: Recent advances in LLMs (OpenAI o1, DeepSeek R1) demonstrate that investing more computation at inference time improves output quality; this paradigm has been extended to diffusion models.
- **Particle Sampling for Diffusion Models**: Because intermediate denoising steps are inherently stochastic, particle sampling methods (SVDD, CoDe, SMC) can maintain multiple candidate samples during denoising and select high-reward particles, far more efficiently than simple Best-of-N.
- **The Dilemma with Flow Models**: Flow models (e.g., FLUX) have become mainstream due to faster generation and higher output quality, but their ODE-based deterministic generation produces a unique sample for a given initial noise, making **particle sampling methods inapplicable directly**.
- **Limitations of Prior Work**: The only prior inference-time scaling method for flow models, Search over Paths (SoP), samples particles only through the forward kernel without exploring modifications to the backward kernel, limiting scaling efficiency.

## Core Problem

Given a pretrained flow model, how can one generate samples highly aligned with user preferences or a reward function through inference-time compute scaling, without additional training?

The formal objective is:

$$p_0^* = \arg\max_q \ \mathbb{E}_{\mathbf{x}_0 \sim q}[r(\mathbf{x}_0)] - \beta \mathcal{D}_{\text{KL}}[q \| p_0]$$

That is, maximize expected reward while using KL divergence regularization to prevent excessive deviation from the original distribution.

## Method

### 1. Inference-Time SDE Conversion

Flow models generate samples by solving the Probability Flow ODE:

$$\mathrm{d}\mathbf{x}_t = u_t(\mathbf{x}_t)\mathrm{d}t$$

This process is fully deterministic—the same noise produces a single sample, and all particles collapse to the same point.

**Key conversion**: The deterministic ODE is converted into a stochastic SDE sharing the same marginal distribution:

$$\mathrm{d}\mathbf{x}_t = \mathbf{f}_t(\mathbf{x}_t)\mathrm{d}t + g_t\mathrm{d}\mathbf{w}$$

where:
- Drift coefficient $\mathbf{f}_t(\mathbf{x}_t) = u_t(\mathbf{x}_t) - \frac{g_t^2}{2}\nabla\log p_t(\mathbf{x}_t)$
- $g_t$ is a freely chosen diffusion coefficient (set to $g_t = t^2$ with a scaling factor of 3 in this work)
- The score function is computed from the pretrained velocity field: $\nabla\log p_t(\mathbf{x}_t) = \frac{1}{\sigma_t}\frac{\alpha_t u_t(\mathbf{x}_t) - \dot{\alpha}_t \mathbf{x}_t}{\dot{\alpha}_t \sigma_t - \alpha_t \dot{\sigma}_t}$

The resulting proposal distribution is:

$$p_\theta(\mathbf{x}_{t-\Delta t}|\mathbf{x}_t) = \mathcal{N}(\mathbf{x}_t - \mathbf{f}_t(\mathbf{x}_t)\Delta t,\ g_t^2 \Delta t\ \mathbf{I})$$

This enables particle sampling—multiple distinct $\mathbf{x}_{t-\Delta t}$ can be drawn from the same $\mathbf{x}_t$.

### 2. Inference-Time Interpolant Conversion

Flow models use **linear interpolation** $(\alpha_t = 1-t, \sigma_t = t)$, while diffusion models commonly adopt **Variance Preserving (VP) interpolation**. The two interpolants are interconvertible via a scale-time transformation:

$$\bar{\mathbf{x}}_s = c_s \mathbf{x}_{t_s}, \quad t_s = \rho^{-1}(\bar{\rho}(s)), \quad c_s = \bar{\sigma}_s / \sigma_{t_s}$$

The velocity field under the new interpolant is:

$$\bar{u}_s(\bar{\mathbf{x}}_s) = \frac{\dot{c}_s}{c_s}\bar{\mathbf{x}}_s + c_s \dot{t}_s u_{t_s}\left(\frac{\bar{\mathbf{x}}_s}{c_s}\right)$$

**Advantages of VP-SDE**:
- VP interpolation maintains a lower log-SNR at each timestep, meaning larger noise components per step and more diverse samples.
- The interpolant conversion synergistically combines **timestep transformation** (sampling at lower SNR) and **diffusion coefficient scaling** (increasing variance); neither mechanism alone is sufficient, but their combination significantly improves diversity without degrading quality.

### 3. Rollover Budget Forcing (RBF)

Prior methods (SVDD, CoDe) uniformly distribute the computational budget across all denoising steps (fixed NFE per step), but experiments reveal that different steps require vastly different amounts of computation.

**RBF strategy**:
1. Distribute the total NFE budget uniformly as a per-step quota $Q$.
2. At each step, as soon as a particle $\mathbf{x}_{t-\Delta t}$ with higher reward than the current $\mathbf{x}_t$ is found, **immediately proceed to the next step**.
3. The remaining NFE quota **rolls over to subsequent steps**.
4. If the quota is exhausted without finding a better sample, the particle with the highest expected future reward in the current set is selected for the next step.

This adaptive allocation avoids wasting budget on "easy-to-improve" steps and concentrates computation on steps that require more exploration.

### 4. Future Reward Estimation

Particle selection is based on **expected future reward**, approximated via the posterior mean from the Tweedie formula:

$$v(\mathbf{x}_t) \approx r(\mathbf{x}_{0|t}), \quad \mathbf{x}_{0|t} \coloneq \mathbb{E}_{\mathbf{x}_0 \sim p_\theta(\mathbf{x}_0|\mathbf{x}_t)}[\mathbf{x}_0]$$

Flow models (especially those fine-tuned with rectification) produce cleaner posterior means at intermediate steps, enabling more accurate reward estimation—a distinctive advantage of flow models over diffusion models for inference-time scaling.

## Key Experimental Results

### Experimental Setup
- Pretrained model: **FLUX**
- Total NFE budget: 500; denoising steps: 10 (50 NFE per step)
- Baselines: BoN, SoP, SMC (DAS), CoDe, SVDD

### Compositional Text-to-Image Generation (GenAI-Bench, 121 prompts)
- **Given reward** (VQAScore): VP-SDE + RBF achieves the highest score, substantially outperforming all Linear-ODE methods.
- **Unseen reward** (InstructBLIP): VP-SDE also achieves the best performance, demonstrating generalization.
- **Image quality** (Aesthetic Score): VP-SDE is on par with baseline FLUX, with no degradation in generation quality.
- Performance improvement trajectory: Linear-ODE → Linear-SDE → VP-SDE, with consistent gains across all particle sampling methods.

### Quantity-Aware Image Generation (T2I-CompBench++, 100 prompts)
- VP-SDE + RBF achieves the highest accuracy, **improving 4–6× over baseline FLUX**.
- Linear-SDE already surpasses all Linear-ODE methods (BoN and SoP); VP interpolation provides additional gains.

### Ablation Study (LPIPS-MPD Diversity Metric)

| Method | LPIPS-MPD ↑ | VQAScore ↑ | Inst. BLIP ↑ |
|--------|------------|------------|-------------|
| Linear-ODE | – | 0.788 | 0.789 |
| Linear-SDE | 0.158 | 0.900 | 0.813 |
| + Adaptive timestep | 0.270 | 0.908 | 0.813 |
| + Adaptive diffusion coeff. | 0.429 | 0.702 | 0.571 |
| VP-SDE | **0.509** | **0.925** | **0.843** |

- Increasing the diffusion coefficient alone improves diversity but severely degrades quality.
- VP-SDE achieves both the highest diversity and the highest reward through the synergistic combination of timestep transformation and diffusion coefficient scaling.

## Theoretical Foundation: Correctness of SDE Conversion

The paper provides complete theoretical proofs in the appendix. The core proposition is:

**Proposition 1**: For a linear stochastic process $\mathbf{x}_t = \alpha_t \mathbf{x}_0 + \sigma_t \mathbf{x}_1$ and its Probability-Flow ODE $\mathrm{d}\mathbf{x}_t = u_t(\mathbf{x}_t)\mathrm{d}t$, the following forward and reverse SDEs (with arbitrary diffusion coefficient $g_t \geq 0$) share the same marginal densities:

- Forward SDE: $\mathrm{d}\mathbf{x}_t = [u_t(\mathbf{x}_t) + \frac{g_t^2}{2}\nabla\log p_t(\mathbf{x}_t)]\mathrm{d}t + g_t\mathrm{d}\mathbf{w}$
- Reverse SDE: $\mathrm{d}\mathbf{x}_t = [u_t(\mathbf{x}_t) - \frac{g_t^2}{2}\nabla\log p_t(\mathbf{x}_t)]\mathrm{d}t + g_t\mathrm{d}\mathbf{w}$

The key proof step aligns the Fokker-Planck equation with the continuity equation, requiring $p_t(\mathbf{x}_t)(\mathbf{f}_t(\mathbf{x}_t) - u_t(\mathbf{x}_t)) = \frac{g_t^2}{2}\nabla p_t(\mathbf{x}_t)$, which yields the correction term for the drift coefficient.

**Corollary 1**: When the diffusion coefficient is chosen as $g_t = \sqrt{2(\sigma_t\dot{\sigma}_t - \sigma_t^2 \dot{\alpha}_t/\alpha_t)}$, the score function term in the forward SDE vanishes, simplifying to $\mathrm{d}\mathbf{x}_t = \frac{\dot{\alpha}_t}{\alpha_t}\mathbf{x}_t\mathrm{d}t + g_t\mathrm{d}\mathbf{w}$. This implies that a special diffusion coefficient exists under which the forward process does not depend on the score function.

## Search Algorithm Implementation Details

The appendix provides detailed parameter configurations for each algorithm:

| Algorithm | Batch size N | Particles K | Interval L | Total NFE |
|-----------|-------------|------------|------------|-----------|
| **BoN** | 50 | — | — | 500 |
| **SoP** | 2 | 5 | — | 500 |
| **SMC (DAS)** | 50 | — | — | 500 |
| **CoDe** | 2 | 25 | 2 | 500 |
| **SVDD** | 2 | 25 | — | 500 |
| **RBF (Ours)** | 2 | Adaptive | — | 500 |

**Core logic of RBF pseudocode**:
1. Initialization: sample $\bar{\mathbf{x}}_1 \sim \mathcal{N}(0, \mathbf{I})$, compute initial reward $r^* \leftarrow r(\bar{\mathbf{x}}_{0|1})$.
2. For each denoising step $i$: allocate quota $q \leftarrow Q^{(i)}$.
3. Sample particles $\bar{\mathbf{x}}_{s-\Delta s}^{(j)}$ one by one; if $r(\bar{\mathbf{x}}_{0|s-\Delta s}^{(j)}) > r^*$, update the current best, roll over the remaining quota $Q^{(i)} - j$ to $Q^{(i+1)}$, and break immediately.
4. If the quota is exhausted without a better particle, select the one with the highest reward from the current set.

**Adaptive time scheduling**: VP-SDE sampling uses a non-uniform time schedule $t_{\text{new}} = \sqrt{1-(1-t)^2}$, taking smaller steps early (when variance is large) for thorough exploration and larger steps later. This configuration performs well under 10-step denoising, benefiting from the few-step generation capability of flow models.

**NFE analysis**: The paper analyzes the number of NFEs required to find a higher-reward sample at each timestep and finds high variance—some steps may require only 1–2 evaluations to find a better particle, while others may demand far more than the uniformly allocated budget. Uniform allocation wastes budget on "easy" steps while being insufficient for "hard" ones.

## Aesthetic Image Generation Experiment (Appendix)

When the reward function is **differentiable** (e.g., Aesthetic Score), RBF can be combined with gradient-based methods (DPS) for synergistic improvement:

| Method | Aesthetic Score† ↑ | ImageReward (held-out) ↑ |
|--------|--------------------|--------------------------|
| FLUX (baseline) | 5.795 | 0.991 |
| DPS | 6.438 | 0.605 |
| SVDD + DPS | 6.887 | 1.077 |
| **RBF + DPS** | **7.170** | **1.152** |

- DPS alone improves the aesthetic score but reduces the held-out reward (ImageReward), indicating reward over-optimization.
- Both SVDD + DPS and RBF + DPS achieve improvements on both metrics, with RBF + DPS reaching the optimum.
- This validates the complementarity of particle sampling and gradient-based methods: gradients provide local optimization direction, while particle sampling provides global exploration.

## Highlights & Insights

1. **First realization of particle sampling for flow models**: Through inference-time ODE→SDE conversion, efficient particle sampling methods from diffusion models are brought to flow models without retraining.
2. **Theoretical insight of VP interpolant conversion**: The paper analyzes from a log-SNR perspective how VP interpolation expands the search space via the synergy of timestep transformation and diffusion coefficient scaling, avoiding quality degradation from simply increasing noise.
3. **Rollover Budget Forcing is simple yet effective**: No additional hyperparameter tuning is required; the strategy adaptively allocates the computational budget and further improves performance on top of all particle sampling methods.
4. **Model-agnostic**: The method does not modify the pretrained model and can be directly applied to any flow model (e.g., FLUX), and can be combined with gradient-based methods (e.g., DPS) for synergistic gains.
5. **Unique advantages of flow models quantitatively verified**: The straighter trajectories of rectified flow produce cleaner intermediate posterior means, enabling more accurate reward estimation during inference-time scaling than diffusion models.
6. **Complete theoretical guarantee**: SDE conversion is rigorously proven to preserve marginal densities via the Fokker-Planck equation, providing a solid theoretical foundation for the method.

## Limitations & Future Work

1. **Inference overhead**: Additional inference-time computation is introduced, which may become a bottleneck when the base model's forward pass is computationally intensive.
2. **Safety risks**: Pretrained models may have been trained on unvetted datasets; inference-time scaling risks being exploited to generate inappropriate content.
3. **Diffusion coefficient selection**: The choice $g_t = t^2$ lacks theoretical optimality guarantees; systematic exploration of timestep scheduling and diffusion coefficient scaling is a future direction.
4. **Reward function dependency**: Method effectiveness depends on reward function quality; non-differentiable rewards limit integration with gradient-based methods.
5. **Video generation validation**: Although flow models are widely used for video generation (e.g., Goku), experiments are validated only on image tasks.
6. **Reward over-optimization risk**: The held-out metric degrades when DPS is used alone, indicating that excessive optimization of a given reward may harm generalization.

## Training / Inference Details

- **No training required**: The method operates entirely at inference time without modifying pretrained model weights. SDE conversion and interpolant transformation rely solely on the existing velocity field $u_t$.
- **Inference pipeline**: Sample initial noise from a standard Gaussian → perform 10-step denoising under VP-SDE → estimate $\mathbf{x}_{0|t}$ at each step via the Tweedie formula → compute reward $r(\mathbf{x}_{0|t})$ → RBF strategy decides whether to continue sampling or advance to the next step.
- **Computational cost**: Total NFE is fixed at 500 (equivalent to BoN with 50 samples), but because particle sampling filters candidates at intermediate steps, efficiency is far superior to BoN.
- **Reward functions**: Compatible with any reward function (VQAScore, RSS, Aesthetic Score, etc.); differentiable rewards can further be combined with gradient-based methods (DPS) for synergistic gains.
- **Diffusion coefficient setting**: $g_t = t^2$ multiplied by a scaling factor of 3 injects more noise at early steps (large $t$) for exploration, with noise naturally diminishing at later steps to ensure quality.
- **Adaptive time scheduling**: VP-SDE uses the non-uniform schedule $t_{\text{new}} = \sqrt{1-(1-t)^2}$, with small steps early for exploration and large steps later for convergence.
- **Compatibility**: Directly applicable to flow matching-based models such as FLUX and Stable Diffusion 3, and can be stacked on top of fine-tuned models for further improvement.

## Related Work & Insights

| Method | Type | Model Type | Requires Stochasticity | Budget Allocation | Key Feature |
|--------|------|-----------|----------------------|-------------------|-------------|
| **BoN** | Baseline | General | No | All upfront | Independently sample N, select best |
| **SoP** (Ma et al.) | Particle sampling | Flow | Forward kernel injection | Uniform | Only prior method for flow models; modifies forward kernel only |
| **SVDD** (Li et al.) | Particle sampling | Diffusion | Inherent | Uniform | Select highest-reward particle per step |
| **CoDe** (Singh et al.) | Particle sampling | Diffusion | Inherent | Uniform / interval | Particle selection at every few steps |
| **DAS/SMC** (Kim et al.) | SMC | Diffusion | Inherent | Uniform | Multinomial sampling by importance weights |
| **Ours (VP-SDE + RBF)** | Particle sampling | **Flow** | **Introduced at inference time** | **Adaptive (Rollover)** | ODE→SDE + linear→VP interpolant + adaptive budget |

**Key distinctions**:
- Compared to SoP: This work explores **modification of the backward kernel** (SDE conversion + interpolant transformation), yielding a larger search space, and RBF provides adaptive budget allocation.
- Compared to SVDD/CoDe/DAS: These methods rely on the inherent stochasticity of diffusion models and cannot be directly applied to flow models; this work bridges the gap through inference-time SDE conversion.
- Compared to BoN: Under the same total NFE budget, particle sampling filters low-quality samples at intermediate steps, avoiding the waste of fully generating and then discarding samples.

**Inspiration and connections**:
1. **Practical value of ODE↔SDE equivalence**: The theoretical property that the probability-flow ODE and its corresponding SDE share marginal distributions is cleverly leveraged here for a practical purpose (enabling particle sampling), demonstrating the engineering utility of theoretical tools.
2. **A universal paradigm for inference-time compute scaling**: From LLM test-time compute (e.g., o1's chain-of-thought) to particle sampling for diffusion models and now to flow model scaling in this work, "investing more computation at inference time for better results" is emerging as a universal cross-modal paradigm.
3. **Search space advantage of VP interpolation**: VP interpolation naturally maintains lower SNR at early steps, which benefits not only generation diversity but also other exploration-requiring tasks (e.g., RLHF sampling, diverse generation).
4. **Generalizability of the Rollover Budget strategy**: The principle of "advance to the next step upon finding a good result and carry over the remaining budget" is not limited to generative models; it generalizes to any step-by-step decision-making scenario with budget constraints (e.g., dynamic-width beam search).
5. **Unique advantages of flow models quantitatively verified**: The straighter trajectories of rectified flow yield more accurate intermediate predictions, providing theoretical support for flow models in tasks requiring intermediate-step quality assessment.
6. **Connection to video generation**: Video flow models such as Goku can theoretically adopt this method directly, but the design of reward functions for video (temporal consistency, motion quality) remains a key challenge.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — First systematic study of particle sampling for flow models; all three components (SDE conversion, VP interpolation, RBF) make independent contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Two main tasks + ablation studies + appendix experiments on aesthetic generation and flow vs. diffusion comparisons; video experiments are absent.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Theoretical derivations are clear; the logical chain from problem motivation to method to experiments is complete; figures and tables are of high quality.
- **Value**: ⭐⭐⭐⭐ — Introduces inference-time scaling capability to the flow model ecosystem with strong practical utility, though inference overhead limits large-scale deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Inference-Time Scaling of Diffusion Models Through Classical Search](../../ICLR2026/image_generation/inference-time_scaling_of_diffusion_models_through_classical_search.md)
- [\[ICML 2025\] Performance Plateaus in Inference-Time Scaling for Text-to-Image Diffusion Without External Models](../../ICML2025/image_generation/performance_plateaus_in_inference-time_scaling_for_text-to-image_diffusion_witho.md)
- [\[CVPR 2026\] Rethinking Prompt Design for Inference-time Scaling in Text-to-Visual Generation](../../CVPR2026/image_generation/rethinking_prompt_design_for_inference-time_scaling_in_text-to-visual_generation.md)
- [\[ICLR 2026\] Compositional Visual Planning via Inference-Time Diffusion Scaling](../../ICLR2026/image_generation/compositional_visual_planning_via_inference-time_diffusion_scaling.md)
- [\[NeurIPS 2025\] Progressive Inference-Time Annealing of Diffusion Models for Sampling from Boltzmann Densities](progressive_inference-time_annealing_of_diffusion_models_for_sampling_from_boltz.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Constrained Discrete Diffusion
description: >-
  [NeurIPS 2025][Image Generation][Discrete diffusion models] This paper proposes CDD (Constrained Discrete Diffusion), which embeds a differentiable constrained optimization projection operator into the denoising process…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Discrete diffusion models"
  - "constrained optimization"
  - "augmented Lagrangian"
  - "controllable text generation"
  - "molecular generation"
  - "toxicity mitigation"
date: 2026-05-08
content_hash: 5b9087cc3153dd54
---

# Constrained Discrete Diffusion

**Conference**: NeurIPS 2025
**arXiv**: [2503.09790](https://arxiv.org/abs/2503.09790)
**Code**: To be confirmed
**Area**: Image Generation
**Keywords**: Discrete diffusion models, constrained optimization, augmented Lagrangian, controllable text generation, molecular generation, toxicity mitigation

## TL;DR

This paper proposes CDD (Constrained Discrete Diffusion), which embeds a differentiable constrained optimization projection operator into the denoising process of discrete diffusion models. Without retraining, CDD enforces sequence-level constraints at sampling time, achieving zero constraint violations across three task categories: toxic text generation, molecular design, and instruction following.

## Background & Motivation

Discrete diffusion models (e.g., MDLM, UDLM) have demonstrated strong capabilities in text and molecular sequence generation. However, real-world applications require satisfying various constraints—toxicity thresholds, chemical synthesizability rules, instruction-following requirements, and so on.

**Fundamental difficulty with autoregressive models**: Token-by-token generation makes sequence-level constraints difficult to enforce. Existing approaches (RLHF, rejection sampling, post-processing) all impose **soft constraints** and cannot provide provable compliance guarantees.

**Unique opportunity with discrete diffusion**: Each denoising step exposes a **global view of the complete sequence**, which is naturally suited to imposing sequence-level structural constraints. However, directly applying Euclidean projection from continuous diffusion to the discrete probability simplex is inappropriate.

**Core innovation of CDD**: A projection operator based on **KL divergence** (rather than Euclidean distance) is designed. Via the **augmented Lagrangian method**, a constrained optimization subproblem is solved at each denoising step, projecting the generative distribution onto the feasible region.

## Method

### Overall Architecture

The CDD pipeline:
1. Standard discrete diffusion initialization (all [MASK] or uniform distribution)
2. After each denoising step, a **projection operation** is applied:
   - Input: probability distribution $\bm{x}_t'$ output by the denoiser
   - Output: projected distribution $\bm{x}_s$ satisfying the constraints
3. The projected distribution is used to continue to the next denoising step

Key property: **Training-free**—projection is applied only at sampling time and does not modify model weights.

### Key Designs

**KL projection operator**:

$$\bm{x}_s = \mathcal{P}_{\mathbf{C}}(\bm{x}_t') = \arg\min_{\bm{y}} D_{\text{KL}}(\bm{x}_t' \| \bm{y}) \quad \text{s.t.} \quad \arg\max(\bm{y}) \in \mathbf{C}$$

In the space of probability distributions, KL divergence is more natural than Euclidean distance—it guarantees that the projected distribution is the "closest" feasible distribution to the original in the constraint-satisfying region.

**Gumbel-Softmax differentiable relaxation**: Since $\arg\max$ is non-differentiable, a Gumbel-Softmax relaxation is used:

$$\tilde{\phi}(\bm{x}_t^i)(v) = \frac{\exp\left(\frac{\log \bm{x}_t^i(v) + \xi_v}{T_{\text{sample}}}\right)}{\sum_{v'=1}^{N} \exp\left(\frac{\log \bm{x}_t^i(v') + \xi_{v'}}{T_{\text{sample}}}\right)}$$

where $\xi_v$ is Gumbel(0,1) noise and $T_{\text{sample}}$ controls the degree of approximation to $\arg\max$.

**Augmented Lagrangian projection**: The constraint violation is defined as $\Delta g_i(\tilde{\phi}(\bm{x}_t)) = \max(0, g_i(\tilde{\phi}(\bm{x}_t)) - \tau_i)$. The augmented Lagrangian objective is:

$$\mathcal{L}_{\text{ALM}}(\bm{y}, \bm{x}_t'; \lambda, \mu) = D_{\text{KL}}(\bm{x}_t' \| \bm{y}) + \sum_{i=1}^{m} \lambda_i \Delta g_i(\tilde{\phi}(\bm{y})) + \frac{\mu_i}{2} \Delta g_i(\tilde{\phi}(\bm{y}))^2$$

Iterative updates:
- Gradient update: $\bm{y} \leftarrow \bm{y} - \eta \nabla_{\bm{y}} \mathcal{L}_{\text{ALM}}$
- Multiplier update: $\lambda \leftarrow \lambda + \mu \Delta g(\bm{y}^*)$
- Penalty increment: $\mu \leftarrow \min(\alpha \mu, \mu_{\max})$

### Loss & Training

CDD itself **introduces no training loss**—it is a purely sampling-time technique. The underlying diffusion model is trained with the standard MDLM/UDLM denoising objective. The augmented Lagrangian used for projection is an online optimization objective applied at sampling time.

**Convergence guarantee** (Theorem 4.1): Under the condition that the constraint set $\bm{C}$ satisfies $\beta$-prox-regularity, the KL distance from the projected distribution to the feasible region decays at rate $(1 - \bm{\alpha}_t)$:

$$D_{\text{KL}}(\bm{x}_s', \bm{C}) \leq (1 - \bm{\alpha}_t) D_{\text{KL}}(\bm{x}_t', \bm{C}) + \bm{\alpha}_{t+1}^2 G^2$$

$\epsilon$-feasibility is achieved after $\mathcal{O}(\bm{\alpha}_{\min}^{-1})$ steps.

## Key Experimental Results

### Main Results

**Toxic text generation** (RealToxicityPrompts, 1000 samples):

| Model | Params | PPL↓ | Coherence↑ | Violation Rate ($\tau$=0.25)↓ | Violation Rate ($\tau$=0.50)↓ | Violation Rate ($\tau$=0.75)↓ |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| GPT-2 | 124M | 18.78 | 42.68 | 33.2% | 21.6% | 13.1% |
| GPT-2+PPLM | 124M | 46.40 | 18.88 | 16.1% | 8.4% | 4.0% |
| Llama 3.2 | 1B | 15.66 | 57.10 | 34.9% | 27.8% | 23.1% |
| MDLM | 110M | 46.72 | 20.02 | 32.1% | 23.2% | 17.2% |
| **CDD ($\tau$=0.25)** | **110M** | **61.55** | **20.16** | **0.0%** | **0.0%** | **0.0%** |
| **CDD ($\tau$=0.50)** | **110M** | **59.44** | **20.30** | — | **0.0%** | **0.0%** |
| **CDD ($\tau$=0.75)** | **110M** | **54.87** | **20.88** | — | — | **0.0%** |

All CDD configurations achieve a **0% violation rate** at their respective thresholds, with PPL comparable to the MDLM baseline and a marginal improvement in coherence.

**Molecular generation (synthesizability constraints)**:

| Model | Valid Molecules↑ | Novel Molecules↑ | QED↑ | Violation Rate ($\tau$=3.0)↓ |
|:--|:--:|:--:|:--:|:--:|
| AR | 1023 | 0 | 0.46 | 91.6% |
| UDLM | 895 | 21 | 0.47 | 89.4% |
| UDLM+D-CFG | 850 | 18 | 0.47 | 80.6% |
| **CDD ($\tau$=3.0)** | **353** | **36** | **0.63** | **0.0%** |
| **CDD ($\tau$=4.5)** | **938** | **33** | **0.58** | **0.0%** |

### Ablation Study

**Molecular novelty constraint**:

| Model | Valid & Novel↑ | QED↑ | Violation Rate↓ |
|:--|:--:|:--:|:--:|
| MDLM | 271 | 0.45 | 54.53% |
| UDLM | 345 | 0.46 | 61.45% |
| **CDD** | **511** | **0.45** | **0.0%** |

CDD increases the number of valid and novel molecules from 345 to 511 (+48%) while reducing the violation rate from 61.45% to 0%.

Sensitivity analysis of augmented Lagrangian hyperparameters: across a full grid search, the Lagrangian relaxation converges to feasible solutions in all cases—demonstrating high robustness to hyperparameter selection.

### Key Findings

1. CDD achieves **zero constraint violations** across all configurations in all three task categories, a result that no baseline (including the 10× larger Llama 3.2) can match.
2. Constraint satisfaction does not require sacrificing generation quality: PPL remains largely unchanged, and coherence shows a slight improvement.
3. The projection operator preserves proximity to the original denoising distribution via KL divergence, retaining generation diversity.
4. The method is highly flexible with respect to the form of constraint functions—these may be classifiers (toxicity detection), black-box functions (synthesizability), or symbolic rules (instruction following).

## Highlights & Insights

- ⭐ First work to achieve provable hard constraint satisfaction within the discrete diffusion framework, filling an important gap in the literature.
- ⭐ The training-free design allows the method to be applied as a plug-and-play component to any pretrained discrete diffusion model.
- Replacing Euclidean distance with KL divergence for projection constitutes a principled discretization of constrained methods developed for continuous diffusion.
- The augmented Lagrangian approach avoids the inefficiency of rejection sampling and the unreliability of post-processing.

## Limitations & Future Work

- CDD runs a Lagrangian inner loop at each denoising step, resulting in slower inference (approximately 2–3× additional overhead).
- Gumbel-Softmax is only an approximation to $\arg\max$; the theoretical convergence guarantee relies on the prox-regularity assumption.
- The constraint functions $g_i$ must be differentiable or have differentiable surrogate models; purely symbolic constraints require additional modeling effort.
- Experiments use relatively small diffusion models (110M / 92M); behavior at larger model scales has not been validated.

## Related Work & Insights

CDD bridges constrained optimization and generative modeling. Compared to constrained methods for continuous diffusion (e.g., MPGD), CDD's KL projection is better suited to the discrete probability simplex. Compared to alignment methods such as RLHF, CDD provides **hard guarantees** rather than soft guidance. Future directions include extending the projection operator to autoregressive decoding (via mechanisms analogous to speculative decoding) and supporting more complex compositional constraints.

## Rating

⭐⭐⭐⭐⭐ (5/5)

| Dimension | Rating |
|:--|:--:|
| Novelty | ⭐⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐⭐ |

The problem is precisely formulated, the technical solution is elegant, and the experiments span three distinct domains, all achieving zero violations. This work represents an important milestone in controllable generation for discrete diffusion models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Information-Theoretic Discrete Diffusion](information-theoretic_discrete_diffusion.md)
- [\[NeurIPS 2025\] Composition and Alignment of Diffusion Models using Constrained Learning](composition_and_alignment_of_diffusion_models_using_constrai.md)
- [\[NeurIPS 2025\] Learnable Sampler Distillation for Discrete Diffusion Models](learnable_sampler_distillation_for_discrete_diffusion_models.md)
- [\[NeurIPS 2025\] Split Gibbs Discrete Diffusion Posterior Sampling](split_gibbs_discrete_diffusion_posterior_sampling.md)
- [\[NeurIPS 2025\] Remasking Discrete Diffusion Models with Inference-Time Scaling](remasking_discrete_diffusion_models_with_inference-time_scaling.md)

</div>

<!-- RELATED:END -->

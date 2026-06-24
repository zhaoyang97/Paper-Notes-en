---
title: >-
  [Paper Note] Erasing Undesirable Influence in Diffusion Models (EraseDiff)
description: >-
  [CVPR 2025][Image Generation][machine unlearning] This paper proposes EraseDiff, which formalizes the data unlearning problem in diffusion models as a constrained optimization problem based on a value function. It optimizes both model preservation and erasing effectiveness simultaneously using a natural first-order algorithm. It runs 11x faster than SA and 2x faster than SalUn on DDPM/Stable Diffusion, while achieving Pareto-optimality in the preservation-forgetting trade-off…
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "machine unlearning"
  - "diffusion models"
  - "constrained optimization"
  - "value function"
  - "NSFW content erasure"
date: 2026-05-08
content_hash: 7757117e285c2745
---

# Erasing Undesirable Influence in Diffusion Models (EraseDiff)

**Conference**: CVPR 2025  
**arXiv**: [2401.05779](https://arxiv.org/abs/2401.05779)  
**Code**: None  
**Area**: Image Generation  
**Keywords**: machine unlearning, diffusion models, constrained optimization, value function, NSFW content erasure

## TL;DR
This paper proposes EraseDiff, which formalizes the data unlearning problem in diffusion models as a constrained optimization problem based on a value function. It optimizes both model preservation and erasing effectiveness simultaneously using a natural first-order algorithm. It runs 11x faster than SA and 2x faster than SalUn on DDPM/Stable Diffusion, while achieving Pareto-optimality in the preservation-forgetting trade-off.

## Background & Motivation

1. **Background**: Diffusion models can generate high-quality images but pose serious risks—they may memorize and regenerate personal images from the training set, or generate NSFW content. Regulations like GDPR and CCPA grant users the "right to be forgotten," requiring the removal of specific data's influence from trained models.

2. **Limitations of Prior Work**: Retraining from scratch is extremely costly (SD training requires 256 A100 GPUs and 150K GPU hours). Existing unlearning methods face a core dilemma—the gradient directions of the erasing objective and the preservation objective conflict, and a simple weighted combination ($\mathcal{L}_r + \lambda \mathcal{L}_f$) fails to balance the two. SA introduces techniques like EWC but suffers from high FIM computation costs, while SalUn focuses on salient parameters but still employs simple weighting.

3. **Key Challenge**: The erasing loss drives the model away from the denoising trajectory of the forgotten data, while the preservation loss drives the model to maintain denoising capability on the remaining data. Their gradient directions are often opposed—when the update direction beneficial for erasing happens to harm preservation performance, simple multi-objective optimization (MOO) leads to oscillations.

4. **Goal**: How to efficiently erase undesirable data influence (class unlearning/concept erasure) in diffusion models while maximizing the preservation of the model's generation capability on the remaining data?

5. **Key Insight**: Inspired by meta-learning algorithms (such as MAML), this work first models the problem as a bilevel optimization (inner loop for erasing, outer loop for preservation), and then reformulates it as a single-level constrained optimization via a value function. This yields a first-order update rule with a closed-form solution, where the update direction simultaneously minimizes both objective functions.

6. **Core Idea**: Use a value function constraint to transform the bi-objective optimization into a constrained optimization, deriving the optimal update direction that satisfies both erasing and preservation.

## Method

### Overall Architecture
Given the pre-trained diffusion model parameters $\theta_0$, forgetting data $\mathcal{D}_f$, and remaining data $\mathcal{D}_r$. For the forgotten data, the ground-truth noise is replaced by noise prediction under a mismatched label ($\epsilon_f = \epsilon_\theta(x_t|c_m), c_m \neq c$), making it impossible for the model to generate meaningful images for the forgotten class. For the remaining data, the standard denoising objective is used to maintain performance. A constrained optimization is solved to find an update direction that accommodates both.

### Key Designs

1. **Value Function Formulation**:

    - Function: Unifies the conflicting bi-objective of erasing and preservation into a single-level optimization problem with an elegant solution.
    - Mechanism: First defines a bilevel optimization—the outer loop minimizes the preservation loss $\mathcal{L}_r(\theta; \mathcal{D}_r)$, while the inner loop constrains $\theta$ to be a minimizer of the forgetting loss $\mathcal{L}_f$. Then, it is rewritten as a constrained optimization using a value function: $\min_\theta \mathcal{L}_r$ s.t. $g(\theta) = \mathcal{L}_f(\theta) - \min_\phi \mathcal{L}_f(\phi) \le 0$. The constraint $g(\theta)$ measures the gap between the current parameters and the optimal solution for forgetting. Furthermore, solving for the update vector $\delta_t$ is formalized as a QP problem: finding the direction closest to the preservation gradient while ensuring a positive projection on the constraint gradient direction (guaranteeing a decrease in the constraint value).
    - Design Motivation: In simple weighting (MOO), $\lambda$ is fixed and cannot dynamically adapt to the relative states of the two objectives. Constrained optimization naturally introduces an adaptive mechanism for $\lambda_t$—when the two gradients are already consistent, $\lambda_t = 0$ (no extra adjustment needed); when they conflict, $\lambda_t > 0$ (projecting the preservation gradient to avoid violating the erasing constraint).

2. **Optimal Update Rule (Theorem 3.1)**:

    - Function: Provides a closed-form first-order update formula for the constrained optimization problem.
    - Mechanism: The optimal update direction is $\delta^* = \nabla_\theta \mathcal{L}_r + \lambda_t \nabla_\theta g$, where $\lambda_t = \max\{0, \frac{a_t - \nabla g^\top \nabla \mathcal{L}_r}{\|\nabla g\|^2}\}$. When the preservation and erasing gradients align ($\nabla g^\top \nabla \mathcal{L}_r > a_t$), $\lambda_t = 0$, directly updating along the preservation gradient; when they conflict, $\lambda_t > 0$, amending the update direction to guarantee that the erasing constraint value also decreases. In practice, $\min_\phi \mathcal{L}_f(\phi)$ is approximated via a $K$-step inner loop gradient descent.
    - Design Motivation: This is a direct derivation from the KKT conditions of the constrained optimization. The choice of $a_t = \eta \|\nabla g\|^2$ ensures that the rate of decrease of the constraint value is proportional to its gradient magnitude, preventing over- or under-erasing. Theorem 3.2 further proves that this algorithm converges to a Pareto-optimal solution.

3. **Forgetting Objective**:

    - Function: Guides the diffusion model to fail at generating meaningful results for forgotten data.
    - Mechanism: For the noise prediction objective of forgotten data, the ground-truth noise $\epsilon$ is replaced with the noise predicted under an incorrect condition $\epsilon_f = \epsilon_\theta(x_t | c_m)$, where $c_m \neq c$ is a random label (EraseDiff$_{\text{rl}}$) or uniform noise (EraseDiff$_{\text{noise}}$). This forces the denoising trajectory actually followed by the model to point to other classes or random noise when asked to generate forgotten class images, failing to produce any meaningful image of the forgotten class.
    - Design Motivation: Directly applying gradient ascent (NegGrad) leads to "over-forgetting"—not only forgetting the target class but also damaging the generation capability of other classes. Guiding with incorrect conditions is gentler—it simply "confuses" the denoising direction of the forgotten class without destroying the model wholesale.

### Loss & Training
Preservation loss: $\mathcal{L}_r = \mathbb{E}[\|\epsilon - \epsilon_\theta(x_t|c)\|^2], \, (x_0, c) \sim \mathcal{D}_r$. Forgetting loss: $\mathcal{L}_f = \mathbb{E}[\|\epsilon_f - \epsilon_\theta(x_t|c)\|^2], \, (x_0, c) \sim \mathcal{D}_f$. The outer loop updates $\theta$ for $T$ steps, with each step running a $K$-step inner loop to approximate the value function. In practice, a small value for $K$ (e.g., 5-10 steps) is typically sufficient.

## Key Experimental Results

### Main Results (CIFAR-10 DDPM forgetting "airplane" class)

| Method | FID↓ | Precision↑ | Recall↑ | $P_\psi(y=c_f|x_f)$↓ |
|------|------|-----------|---------|---------------------|
| Unscrubbed | 9.63 | 0.40 | 0.79 | 0.97 |
| Finetune | 8.21 | 0.43 | 0.77 | 0.96 |
| NegGrad | 76.73 | 0.08 | 0.61 | 0.61 |
| SA | 8.19 | 0.43 | 0.75 | **0.06** |
| SalUn | 9.16 | 0.41 | 0.76 | **0.07** |
| **EraseDiff$_\text{noise}$** | **7.61** | **0.43** | 0.72 | 0.22 |
| **EraseDiff$_\text{rl}$** | 8.66 | **0.43** | **0.77** | 0.24 |

### Stable Diffusion erasing nudity concept

| Method | FID↓ | CLIP↑ | NudeNet Detections↓ |
|------|------|-------|-------------|
| ESD | 15.76 | 30.33 | ~150 |
| SA | 25.58 | 31.03 | ~180 |
| SalUn | 25.06 | 28.91 | - |
| **EraseDiff** | **17.01** | **30.58** | **Lowest** |

### Key Findings
- NegGrad leads to over-forgetting (FID skyrockets to 76.73, and Precision drops to just 0.08), verifying that simple gradient reversal is infeasible.
- Finetune and BlindSpot fail to unlearn sufficiently ($P_\psi$ remains close to unscrubbed), showing that traditional classification unlearning methods are not suitable for generative models.
- EraseDiff outperforms or matches SA and SalUn across all preservation performance metrics (FID, Precision, and Recall).
- EraseDiff$_\text{noise}$ achieves the best preservation performance (FID 7.61) with slightly weaker unlearning effects ($P_\psi$ 0.22); EraseDiff$_\text{rl}$ achieves better diversity (Recall 0.77).
- In nudity erasure, EraseDiff's FID (17.01) is significantly better than SA (25.58) and SalUn (25.06), demonstrating superior preservation capability.
- It runs 11x faster than SA and 2x faster than SalUn.
- Gradient cosine similarity analysis verifies that EraseDiff effectively avoids gradient conflicts—the update vector is positively correlated with both preservation and erasing gradients simultaneously.

## Highlights & Insights
- **Constrained Optimization Perspective**: Elevates the unlearning problem from "weighting two losses" to "minimizing preservation loss under erasing constraints," which is mathematically more elegant and naturally allows $\lambda_t$ to adapt. This framework is transferable to any fine-tuning scenario requiring a balance between two conflicting objectives.
- **Proof of Pareto Optimality**: Theorem 3.2 proves that the algorithm converges to a Pareto optimum, which is a rare theoretical guarantee among machine unlearning methods.
- **Gradient Conflict Visualization**: The cosine similarity analysis in Figure 2 clearly illustrates the fundamental difference between MOO and EraseDiff—in MOO, the preservation and erasing gradients alternately dominate (leading to oscillations), while in EraseDiff, both remain positively correlated (working in coordination).

## Limitations & Future Work
- The unlearning effect ($P_\psi$ of 0.22-0.24) is not as low as SA (0.06) and SalUn (0.07), which may be insufficient for scenarios requiring strict "complete forgetting."
- The $K$-step inner loop gradient descent increases computational overhead, and the choice of $K$ requires tuning.
- It is only validated on SD v1.4, with larger models like SDXL remaining untested.
- The choice of alternative labels $c_m$ for the forgotten data (random label vs. noise) impacts the results and lacks systematic study.
- Reversibility of unlearning is not discussed—whether an attacker can recover the forgotten capabilities through fine-tuning.

## Related Work & Insights
- **vs SA (Selective Amnesia)**: SA utilizes EWC + generative replay to preserve performance, which incurs heavy FIM computation overhead and requires generating remaining data. EraseDiff directly resolves conflicts at the gradient update level, making it much more efficient.
- **vs SalUn**: SalUn selectively updates by identifying salient parameters of the forgotten data. Its methodology focuses on "which parameters to update," while EraseDiff focuses on "what the update direction is." The two are orthogonal and potentially complementary.
- **vs ESD**: ESD directly modifies cross-attention weights to erase concepts, which is only applicable to text-conditional models. EraseDiff is applicable to both conditional and unconditional diffusion models.

## Rating
- Novelty: ⭐⭐⭐⭐ The formulation of constrained optimization and the value function method are novel in diffusion unlearning, though the overall algorithmic framework originates from existing optimization theory.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive validation across both DDPM and SD, class unlearning and concept erasure, speed comparisons, and gradient analysis.
- Writing Quality: ⭐⭐⭐⭐ Rigorous theoretical derivations, but contains many symbols and equations, raising the entry barrier.
- Value: ⭐⭐⭐⭐ Provides an efficient method with theoretical guarantees for diffusion model unlearning, carrying practical significance for AI safety and privacy protection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] SAEmnesia: Erasing Concepts in Diffusion Models with Supervised Sparse Autoencoders](../../ICML2026/image_generation/saemnesia_erasing_concepts_in_diffusion_models_with_supervised_sparse_autoencode.md)
- [\[CVPR 2025\] Decentralized Diffusion Models](decentralized_diffusion_models.md)
- [\[CVPR 2025\] Diffusion-4K: Ultra-High-Resolution Image Synthesis with Latent Diffusion Models](diffusion-4k_ultra-high-resolution_image_synthesis_with_latent_diffusion_models.md)
- [\[CVPR 2025\] UIBDiffusion: Universal Imperceptible Backdoor Attack for Diffusion Models](uibdiffusion_universal_imperceptible_backdoor_attack_for_diffusion_models.md)
- [\[CVPR 2025\] The Art of Deception: Color Visual Illusions and Diffusion Models](the_art_of_deception_color_visual_illusions_and_diffusion_models.md)

</div>

<!-- RELATED:END -->

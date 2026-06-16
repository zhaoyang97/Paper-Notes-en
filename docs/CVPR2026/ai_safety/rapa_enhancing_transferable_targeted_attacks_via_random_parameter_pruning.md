---
title: >-
  [Paper Note] RaPA: Enhancing Transferable Targeted Attacks via Random Parameter Pruning
description: >-
  [CVPR 2026][AI Safety][Paper Note] RaPA discovers that adversarial perturbations in existing targeted transfer attacks over-rely on a few key parameters in the surrogate model. By applying random parameter pruning (DropConnect) at each optimization step, it implicitly adds an "importance equalization regularization" to the loss. This disperses the depen
tags:
  - CVPR 2026
  - AI Safety
date: 2026-05-08
content_hash: 8ab6c0fce7f9b843
---
# RaPA: Enhancing Transferable Targeted Attacks via Random Parameter Pruning

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Su_RaPA_Enhancing_Transferable_Targeted_Attacks_via_Random_Parameter_Pruning_CVPR_2026_paper.html)  
**Code**: [https://github.com/molarsu/RaPA](https://github.com/molarsu/RaPA)  
**Area**: AI Security / Adversarial Attack Transferability  
**Keywords**: Targeted Transferable Attacks, Adversarial Examples, Random Parameter Pruning, Self-Ensemble, Importance Regularization

## TL;DR
RaPA discovers that adversarial perturbations in existing targeted transfer attacks over-rely on a few key parameters in the surrogate model. By applying random parameter pruning (DropConnect) at each optimization step, it implicitly adds an "importance equalization regularization" to the loss. This disperses the dependency and significantly improves targeted attack success rates across architectures (especially CNN $\rightarrow$ Transformer).

## Background & Motivation

**Background**: Transfer-based black-box attacks generate adversarial examples on a white-box surrogate model to deceive invisible target models. Compared to non-targeted attacks, **targeted** transfer success rates (ASR) have remained significantly lower. Mainstream enhancement methods fall into three categories: input transformations (DI/RDI/Admix/CFM/FTM to suppress overfitting), gradient stabilization (MI/SI adding momentum or multi-scale), and self-ensembles (generating variants from a single surrogate, e.g., Ghost Network, MUP, SE-ViT).

**Limitations of Prior Work**: Although CFM and FTM have pushed ASR to new highs, targeted transfer success rates remain relatively low. Adversarial examples are "strong in white-box, weak in black-box"—effective on the surrogate but failing when switching architectures. These methods focus on the input or feature space, without questioning which model characteristics lead to overfitting.

**Key Challenge**: Through a pilot study, the authors identified an overlooked root cause—adversarial perturbations **over-rely on a tiny number of "key parameters"** in the surrogate model. These parameters arise from specific training schemes/data/architectures. When the target model has a different parameter configuration, the attack fails, as the perturbation effectively learned a set of "parameter shortcuts."

**Goal**: Disperse this strong reliance on a few parameters without significantly increasing computation, allowing perturbation importance to be distributed more evenly across all parameters for better transferability.

**Key Insight**: Instead of directly masking the "most important parameters" (which requires expensive second-order derivatives and severely degrades surrogate performance), **random** pruning is used. It is computationally cheap and, in an expectant sense, precisely serves to balance importance.

**Core Idea**: Apply a Bernoulli random mask (DropConnect) to the surrogate model parameters at each optimization step. Use the average gradient from multiple masked variants to update the adversarial example. Theoretically, this is proven equivalent to adding an importance penalty to the original loss, forcing the perturbation to no longer rely on a few parameters.

## Method

### Overall Architecture
RaPA addresses the root cause of overfitting in targeted transfer attacks—perturbation reliance on a few high-importance parameters in the surrogate. The logic follows a "diagnosis then prescription" approach: first, a pilot study quantifies that "pruning the most important 0.5% of parameters causes ASR to plummet, while pruning the least important 0.5% has almost no impact," confirming that over-reliance exists. Then, random parameter pruning is proposed, with a second-order Taylor expansion proving that random masking is equivalent to a regularization term that equalizes parameter contributions.

In practice, RaPA does not modify the main optimization framework for adversarial examples but embeds it into each iteration: at step $t$, Bernoulli random masks are independently sampled for selected linear and normalization layers to generate multiple pruned variants of the surrogate. Each variant performs a forward and backward pass to obtain a gradient. After averaging gradients from $S$ variants, iterative methods like MI-TI are used to update the adversarial example and project it back to the $\epsilon$-ball. Since masks are re-sampled per layer and per step, it is equivalent to finding a consensus direction across a set of "self-ensemble" models that are semantically consistent but parametrically diverse. It is training-free, cross-architecture, and plug-and-play.

### Key Designs

**1. Diagnosis: Adversarial Perturbations Over-rely on Few Key Parameters**

To justify the method, the authors first prove the existence of the issue. Using Optimal Brain Damage (OBD) sensitivity analysis, parameter importance is defined as $I(\theta_i)=\frac{\partial^2 L(f(x_{adv}))}{\partial \theta_i^2}\times \theta_i^2$ (second-order derivative times squared parameter, approximating the change in loss if the parameter is removed). Parameters are split into the "most important 0.5%" and "least important 0.5%". Results on ResNet-50 show that pruning the most important 0.5% causes ASR to drop by **over 46%** (RaPA drops from 98.2 to 64.5, while baselines like DI drop to 16–37), whereas pruning the least important 0.5% has negligible effect. This proves perturbations rely on a few "shortcut parameters."

**2. Random Parameter Pruning ≡ Importance Equalization Regularization**

Rather than calculating expensive second-order derivatives to mask important parameters, RaPA uses **random** pruning. Defining a random binary mask $M\in\{0,1\}^{|\theta|}$ where $M_i\sim\text{Bernoulli}(1-p)$, the Taylor expansion of the expected loss is:

$$\mathbb{E}_M[L(f(x_{adv};M\odot\theta))]\approx L(f(x_{adv};\theta))+\frac{p(1-p)}{2}\sum_i \frac{\partial^2 L(f(x_{adv};\theta))}{\partial\theta_i^2}\theta_i^2$$

The first term is the original loss, and the second is proportional to the sum of importance $I(\theta_i)$ of all parameters—an **importance penalty**. Minimizing this objective via re-sampled masks forces the adversarial example to spread importance across all parameters.

**3. Implementation via DropConnect and Multi-variant Self-Ensemble**

Random parameter pruning is formally equivalent to DropConnect. The authors apply it to weights and biases of **linear layers** and transformation parameters (scale/shift) of **normalization layers**, both prevalent in CNNs and Transformers. For a linear layer $W$, masks $M_w\sim\text{Bernoulli}(1-p)$ are sampled to get $W_M=M_w\odot W$. By performing $S$ inferences per step with re-sampled masks and averaging gradients, a self-ensemble effect is achieved. Compared to Ghost Network or MUP, RaPA's layer-wise and step-wise randomness introduces stronger diversity and more uniform parameter importance distribution (measured by Gini coefficient).

### Loss & Training
Ours is training-free. It only modifies gradient estimation during attack optimization under $\ell_\infty$ constraints with $\epsilon=16/255$ and step size $\alpha=2/255$ using Logit loss. DropConnect probability $p$ is fine-tuned per surrogate: ResNet-50 (0.05), Inception-v3 (0.02), DenseNet-121 (0.04), ViT (0.01), CLIP (0.03).

## Key Experimental Results

Evaluated on ImageNet-Compatible dataset with 16 target models (including cross-modal CLIP). Main experiments use $S=5$.

### Main Results

| Transfer Setting | Surrogate | Prev. SOTA (ASR%) | RaPA (ASR%) | Gain |
|------------------|-----------|-------------------|-------------|------|
| CNN $\rightarrow$ Transformer (5 targets) | ResNet-50 | 33.3 (FTM) | 45.0 | +11.7 |
| CNN $\rightarrow$ Transformer | DenseNet-121 | 22.8 (FTM) | 40.3 | +17.5 |
| $\rightarrow$ CNN (10 targets) | Inception-v3 | 54.9 (CFM) | 68.0 | +13.1 |
| $\rightarrow$ CNN (10 targets) | ViT | 40.1 (CFM) | 51.2 | +11.1 |

On the difficult CNN $\rightarrow$ Transformer transfer, RaPA improves average ASR from 33.3% to 45.0%. Against strong defenses like ensIR and HGD, RaPA outperforms the second-best by **29.4%** and **10.5%** respectively.

### Ablation Study

| Configuration | Avg ASR(%) (16 targets, RN50 surrogate) | Description |
|---------------|-----------------------------------------|-------------|
| BN + FC (Recommended) | 72.4 | Optimal, consistent with theory |
| BN layers only | 72.1 | Near optimal |
| Conv layers only | 65.1 | Conv weights are sparse; less affected by over-reliance |
| All layers | 69.2 | Still outperforms all baselines |
| FC layers only | 41.1 | RN50 has only one FC; insufficient diversity |

For probability $p$, ASR peaks at $p=0.05$ (72.4%) and remains stable within $p\in[0.03, 0.07]$.

### Key Findings
- **DropConnect Target**: Applying to BN+FC is best (72.4%). Pure Conv layers are less effective because Conv weights are naturally sparse and less prone to the "over-reliance" bottleneck.
- **Lowest Gini Coefficient**: RaPA's average Gini coefficient for parameter importance is 0.08 (CFM: 0.19, DI: 0.32), quantitatively proving it flattens the importance distribution.
- **High Scaling Returns**: Increasing iterations (300 $\rightarrow$ 500) and $S$ (1 $\rightarrow$ 5) yields a 15.9% ASR gain, the highest among all methods, showing random self-ensembles benefit most from increased computation.
- **Parameter Robustness**: Performance is stable across $p$ values from 0.03 to 0.07.

## Highlights & Insights
- **Theoretical Link**: The proof that "random pruning ≡ importance regularization" via second-order Taylor expansion provides a solid theoretical foundation and explains the mechanism's success.
- **Diagnosis-Driven**: Identifying the "over-reliance on few parameters" through OBD before proposing the solution makes the approach highly convincing.
- **Plug-and-play**: As a parameter-space randomization, it can be combined with any input-space or gradient-space attack.

## Limitations & Future Work
- DropConnect probability $p$ currently requires manual tuning per surrogate model; lacks an adaptive selection mechanism.
- Theoretical equivalence assumes small $p$ for the approximation; behavior with larger $p$ is not fully explored.
- Verification is primarily on classification; effectiveness on detection/segmentation needs further testing.

## Related Work & Insights
- **vs. CFM / FTM (Input/Feature Space)**: While they target diversity in inputs/features, RaPA targets the **parameter space**. The methods are orthogonal and can be stacked.
- **vs. Ghost Network / MUP (Self-Ensemble)**: These use deterministic or local perturbations. RaPA’s layer-wise stochastic pruning introduces higher diversity and lower Gini coefficients.
- **vs. SE-ViT (ViT-Specific)**: Even when using ViT as a surrogate, RaPA outperforms SE-ViT.

## Rating
- Novelty: ⭐⭐⭐⭐ New perspective on parameter dependency with theoretical equivalence proof.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 16 target models, cross-modal coverage, and detailed scaling analysis.
- Writing Quality: ⭐⭐⭐⭐ Logical flow from diagnosis to theory to implementation.
- Value: ⭐⭐⭐⭐ Training-free and effective for understanding model vulnerabilities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] VCP-Attack: Visual-Contrastive Projection for Transferable Black-Box Targeted Attacks on Large Vision-Language Models](vcp-attack_visual-contrastive_projection_for_transferable_black-box_targeted_att.md)
- [\[CVPR 2026\] AdvFM: Lookahead Flow-Matching Velocity-Field Attacks for Imperceptible and Transferable Adversarial Examples](advfm_lookahead_flow-matching_velocity-field_attacks_for_imperceptible_and_trans.md)
- [\[ECCV 2024\] CLIP-Guided Generative Networks for Transferable Targeted Adversarial Attacks](../../ECCV2024/ai_safety/clip-guided_generative_networks_for_transferable_targeted_adversarial_attacks.md)
- [\[CVPR 2026\] When Robots Obey the Patch: Universal Transferable Patch Attacks on Vision-Language-Action Models](when_robots_obey_the_patch_universal_transferable_patch_attacks_on_vision-langua.md)
- [\[CVPR 2026\] Roots Beneath the Cut: Uncovering the Risk of Concept Revival in Pruning-Based Unlearning for Diffusion Models](roots_beneath_the_cut_uncovering_the_risk_of_concept_revival_in_pruning-based_un.md)

</div>

<!-- RELATED:END -->

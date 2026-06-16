---
title: >-
  [Paper Note] From Navigation to Refinement: Revealing the Two-Stage Nature of Flow-based Diffusion Models through Oracle Velocity
description: >-
  [CVPR 2026][Image Generation][Flow Matching] This paper derives a **closed-form oracle solution** for the marginal velocity field of rectified flow under Gaussian priors and finite datasets. Using this, it reveals that the training objective of flow-based diffusion models naturally splits into a "Navigation Stage" (early, guided by a mixture of data modes, respon
tags:
  - CVPR 2026
  - Image Generation
  - Flow Matching
  - oracle velocity
  - timestep schedule
date: 2026-05-08
content_hash: b6dda5f7acb8b032
---
# From Navigation to Refinement: Revealing the Two-Stage Nature of Flow-based Diffusion Models through Oracle Velocity

**Conference**: CVPR 2026  
**arXiv**: [2512.02826](https://arxiv.org/abs/2512.02826)  
**Code**: Project Page https://maps-research.github.io/from-navigation-to-refinement/  
**Area**: Diffusion Models / Image Generation / Generative Model Theory  
**Keywords**: Flow Matching, oracle velocity, memorization and generalization, two-stage training, timestep schedule

## TL;DR
This paper derives a **closed-form oracle solution** for the marginal velocity field of rectified flow under Gaussian priors and finite datasets. Using this, it reveals that the training objective of flow-based diffusion models naturally splits into a "Navigation Stage" (early, guided by a mixture of data modes, responsible for global layout/generalization) and a "Refinement Stage" (late, dominated by a single nearest-neighbor sample, responsible for details/memorization). This two-stage perspective provides a unified explanation for why empirical techniques like timestep shifting, CFG intervals, and latent space designs are effective.

## Background & Motivation
**Background**: Flow Matching (FM) / rectified flow has become the de facto standard for training SOTA diffusion models—using linear interpolation $x_t = \alpha_t x_1 + \sigma_t x_0$, the model regresses the velocity field. However, the FM objective itself is generally considered "intractable" because the ground-truth velocity field $u_t(x_t)$ is unavailable. In practice, gradient-equivalent Conditional Flow Matching (CFM) is used, constructing conditional paths with individual samples $x_1$.

**Limitations of Prior Work**: The "memorization vs. generalization" behavior of models has remained unclear. Previous works had two limitations: (1) most studied the "sampling from pure noise" setting; (2) most were validated on small-scale/low-resolution datasets (FFHQ, CIFAR-10). When data scales to ImageNet levels, where memorizing training samples from scratch is nearly impossible, these conclusions no longer apply. Song et al. observed that ImageNet-scale models exhibit a divergence in memorization/generalization when "resampling from different time points": resampling early (near the prior) yields new samples, while resampling late (near the data) tends to reproduce training images—but a principled explanation for **why** this happens is missing.

**Key Challenge**: The FM objective is treated as a "black-box intractable regression task," making it impossible to explain model behavior from the training objective itself. If the exact form of the oracle objective could be written down, memorization/generalization, learning difficulty, and various sampling tricks could be derived from "what exactly the model is required to fit at each moment."

**Key Insight**: The authors point out that under three mild assumptions—Gaussian prior, finite dataset approximation of $p_{\text{data}}$, and rectified flow linear interpolation—the marginal velocity field actually **has a closed-form expression**. Thus, the oracle velocity can be calculated precisely at any point in the sample space, allowing FM training to be dissected as a supervised learning problem with ground-truth labels.

**Core Idea**: Using the closed-form oracle velocity field as a microscope, the authors found that the oracle objective naturally splits into two segments along time $t$ (Navigation / Refinement). This serves as a unified framework to explain model memorization-generalization behavior and a series of sampling practices.

## Method
Ours is not a presentation of a new model, but an **analytical framework**: first deriving the closed-form oracle velocity, then characterizing the two-stage training objective, and finally mapping model behaviors and empirical tricks back to these two stages. The following sections expand on "Core Mathematical Results → Two-Stage Characterization → Experimental Probes → Explaining Techniques."

### Key Designs

**1. Closed-form oracle velocity field under Gaussian prior: Calculating the "intractable" FM objective**

The pain point is that the ground truth $u_t(x_t)$ in the FM objective $\mathcal{L}_{\text{FM}} = \mathbb{E}_{t,p_t(x_t)} \| v_t(x_t;\theta) - u_t(x_t) \|^2$ is usually unavailable, requiring CFM as an approximation. The authors rewrite the data distribution as an empirical mixture distribution over a finite dataset. Applying Bayes' rule to the path marginal (a Gaussian mixture) and taking the conditional expectation $u_t^*(x_t,t) := \mathbb{E}[u_t(x_t\mid x_1)\mid x_t]$, the closed-form solution is obtained (Theorem 2.1):

$$u_t^*(x_t,t) = A_t \sum_{i=1}^{N} \gamma_i(x_t,t)\, x_1^{(i)} + B_t\, x_t$$

Where $A_t = \dot\alpha_t - \alpha_t \dot\sigma_t/\sigma_t$, $B_t = \dot\sigma_t/\sigma_t$, and the normalized posterior weights

$$\gamma_i(x_t,t) = \frac{\exp\!\big(-\|x_t - \alpha_t x_1^{(i)}\|^2 / 2\sigma_t^2\big)}{\sum_{j=1}^{N} \exp\!\big(-\|x_t - \alpha_t x_1^{(j)}\|^2 / 2\sigma_t^2\big)}$$

represent a softmax-form weight of "how close $x_t$ is to each data point $x_1^{(i)}$." This step is critical because it transforms the oracle objective into "a weighted average of $N$ data points + a term following the current position"—all subsequent conclusions are derived from how $\gamma_i$ changes with $t$. For class-conditional generation, $u_t^*(x_t,t\mid y)$ is simply calculated over the subset $\{x_1^{(i)}\}_{i\in I_y}$.

**2. Two-stage training objective: Navigation vs. Refinement, with the transition determined by $D$, $N$, and $\sigma_t$**

With the closed-form for $\gamma_i$, the structural change of the oracle objective along $t$ becomes clear. In the early stage $t\in[0,0.1]$ near the prior (**Navigation Stage**), $\gamma_i$ is relatively flat, and multiple data points contribute weight; the oracle velocity is a "mixture of multiple related data modes," pushing the model toward a global layout. After $t\approx 0.1$ (**Refinement Stage**), the top-1 posterior weight quickly saturates to 1, and the oracle velocity collapses to being dominated by a **single nearest-neighbor sample**, effectively degenerating into the CFM objective $x_1 - x_0$. The paper validates this with two curves: (a) the MSE between $u_t^*$ and the CFM target $(x_1-x_0)$ only diverges significantly at $t\in[0,0.1]$ and coincides thereafter; (b) the top-1 posterior weight rapidly approaches 1 after $t>0.1$.

Why the rapid saturation? In $D$-dimensional space, the squared distance in the exponent of $\gamma_i$ grows linearly with $D$ and is divided by $2\sigma_t^2$. As $D$ increases or $\sigma_t$ decreases, even tiny distance differences between samples are exponentially amplified into overwhelming weight differences—causing the posterior to "spike" at the nearest neighbor. This yields the dependency of the turning point: **higher data dimension $D$ leads to an earlier transition, while larger sample size $N$ leads to a later transition**. For ImageNet $256^2$ with latent $D\in\{4096,8192\}$ and $N\approx 1400$, the transition falls exactly around $t\approx 0.1$. This conclusion is powerful—it implies that for the same rectified flow objective, the effective training objective actually differs across different datasets.

**3. Mixed sampling probe: Mapping memorization/generalization to specific stages by using "oracle for the first half, model for the second"**

A two-stage training objective is not enough; the authors aim to prove that **model behavior** also splits into two stages and map this to memorization/generalization. They designed a mixed sampling: starting from a Gaussian prior, the oracle velocity $u_t^*$ is used until a switching point $t_{\text{switch}}$, after which the velocity switches to the trained model. The advantage of using the oracle for the first stage is ensuring the intermediate state falls strictly on the interpolation distribution, isolating the confounding factor of "imperfect model prediction." The results are clear: using the oracle throughout deterministically retrieves a training sample; for $t_{\text{switch}}\in(0.2,1.0]$ (switching during the Refinement Stage), the model nearly reproduces the training trajectory and produces approximate training images (**memorization**); for $t_{\text{switch}}\in[0,0.2]$ (switching during the Navigation Stage), the strong prior perturbation prevents the model from backtracking to the original trajectory, leading it to deviate from training instances and exhibit **generalization**. Visualizing intermediate predictions (one-step Euler to $t=1$) confirms this: at $t=0$, predictions collapse to class means (sharks are blueish, pandas are black and white); the early stage navigates the global layout (stabilizing at $t\approx0.2$), while the later stage refines local details. Notably, the model behavior transition at $t\approx0.2$ slightly lags behind the training objective transition at $t\approx0.1$, which the authors hypothesize is the extra margin needed for the model to correct accumulated errors after switching to a consistent objective.

**4. Unified re-explanation of empirical tricks: timestep shift / CFG interval / latent space**

Finally, the framework is applied to practice. The authors observe that the two stages have different "learning difficulties": the divergence in training MSE is concentrated in the Navigation Stage, while the curves for both objectives coincide in the Refinement Stage. Furthermore, **navigation performance is almost independent of model capacity, while refinement significantly benefits from larger models/longer training**—confirmed by the model swap experiments in Tab. 1 (using a small model for Stage 1 hardly affects performance, while using a small model for Stage 2 significantly worsens gFID). Based on this, three things are re-explained: (i) **timestep shifting** $t_m = s t_n / (1+(s-1)t_n)$ essentially adjusts the allocation of sampling steps between the two stages; $s<1$ allocates more steps to navigation, which yields better samples; (ii) **CFG interval** restriction to a sub-interval shows the optimal interval is concentrated in the **early-to-mid Refinement Stage** transitioning from navigation, and expanding the interval should exclude the initial $[0,0.1]$ (where amplifying guidance under extreme noise interferes with global layout formation); (iii) **latent space**: for the semantically aligned VA-VAE (DINO VF loss), the oracle loss converges into a smooth parabolic shape in the Refinement Stage, while for the purely reconstructive SD-VAE, it appears wavy, suggesting that latents with clearer mode organization benefit both navigation and refinement, leading to faster convergence.

### Loss & Training
Ours does not modify the training objective, still using standard rectified flow / CFM; the analysis targets LightningDiT-XL/1 (and B variants) trained on VA-VAE and SD-VAE latents, ImageNet $256^2$, mostly for 100 epochs (with some 800-epoch controls for 8× compute). All "oracle" quantities are calculated precisely offline using the closed-form $u_t^*$, requiring no additional training.

## Key Experimental Results

### Main Results: Capacity allocation under the two-stage framework (Tab. 1)
Fixing total NFE (uniform sampling with 25 steps per sub-interval, no CFG), swapping models for Stage 1 / Stage 2 to observe gFID@50K. Conclusion: swapping to a smaller model for Stage 1 (Navigation) hardly drops performance, but swapping to a smaller model for Stage 2 (Refinement) significantly degrades it—capacity is primarily consumed by refinement.

| Split Point | Stage 1 Model | Stage 2 Model | gFID@50K ↓ |
|----------|--------------|--------------|------------|
| [0,0.1]+[0.1,1.0] | XL | XL | 2.94 |
| [0,0.1]+[0.1,1.0] | Base | XL | 3.71 |
| [0,0.1]+[0.1,1.0] | XL | Base | 11.26 |
| [0,0.1]+[0.1,1.0] | Base | Base | 12.45 |
| [0,0.2]+[0.2,1.0] | XL | XL | 2.60 |
| [0,0.2]+[0.2,1.0] | Base | XL | 4.47 |
| [0,0.2]+[0.2,1.0] | XL | Base | 9.24 |
| [0,0.2]+[0.2,1.0] | Base | Base | 12.01 |

The Base→XL transition (swapping only Refinement) goes from 12.45→3.71, while XL→Base (downgrading only Refinement) results in 11.26, confirming that "Refinement is the capacity bottleneck, Navigation is not."

### Ablation Study: CFG interval scan (Tab. 3, LightningDiT-B/1, $\omega=2.5$)
| CFG Interval | gFID@50K ↓ | CFG Interval | gFID@50K ↓ |
|----------|-----------|----------|-----------|
| None | 12.99 | [0.0,1.0] | 10.79 |
| [0.0,0.1] | 6.33 | [0.1,0.2] | 5.21 |
| [0.1,0.3] | 3.54 | [0.1,0.5] | 2.82 |
| [0.1,0.6] | 2.80 | [0.1,0.7] | 2.86 |
| [0.4,0.5] | 10.39 | [0.9,1.0] | 12.20 |

The optimal interval is concentrated in the early-to-mid Refinement Stage ([0.1, 0.6] achieves the lowest 2.80); intervals that are too late or only in the initial Navigation Stage perform poorly.

### Key Findings
- **Turning point $t\approx 0.1$ is determined by $D$, $N$, and $\sigma_t$**: Larger $D$ means an earlier transition; larger $N$ means a later transition; ImageNet latents with $D\in\{4096,8192\}$ and $N\approx1400$ explain the observed $t\approx0.1$.
- **timestep shift (Tab. 2)**: Increasing the early-step proportion from ~22% (uniform) to 28%–34% ($s=0.7$/$0.5$) yields the best gFID (12.46 / 12.23); shifting too much ($s=0.1$, 72%) worsens it to 19.91—navigation needs more steps but not excessively.
- **Model behavior transition ($t\approx0.2$) lags behind the training objective transition ($t\approx0.1$)**: Models need a margin to correct accumulated errors after the switch.
- **Latent structure affects refinement convergence**: VA-VAE converges smoothly, while SD-VAE is wavy; semantically aligned latents converge faster.

## Highlights & Insights
- **Turning the "intractable" FM objective into a closed-form oracle**: This is the fulcrum of the paper—once $u_t^*$ can be calculated precisely, memorization/generalization, learning difficulty, and sampling tricks can be derived from "what the model is asked to fit at each $t$," rather than relying on empirical observations.
- **The softmax perspective of posterior weights $\gamma_i$ is highly intuitive**: It explains the "near prior = mode mixture, near data = nearest neighbor dominance" as a softmax spiking process as the temperature $\sigma_t$ shrinks, providing quantitative dependencies on $D$/$N$ that are transferable to any linear interpolation diffusion framework.
- **The oracle-early mixed sampling is a clean probe**: Using analytical velocity isolates the confounder of "model imperfection," allowing memorization/generalization to be replicated as a single-variable function of the switching point $t_{\text{switch}}$, a methodological approach worth emulating.
- **A single framework unifies three types of tricks**: Timestep shift allocates sampling steps, CFG interval targets the "most cost-effective early-to-mid Refinement Stage," and latent design highlights "mode organization benefiting both stages"—consolidating disparate tuning experiences into a single principle.

## Limitations & Future Work
- **Strong assumptions**: The closed-form oracle relies on three assumptions: Gaussian prior, finite dataset, and linear interpolation. The paper uses synthetic unit Gaussian samples for demonstrations; real data distributions might deviate slightly (though behavior remains similar). Whether conclusions hold under non-linear schedules or non-Gaussian priors requires further validation.
- **Validation mainly on ImageNet class-conditional latent diffusion**: While Flux.1 is mentioned in the appendix, whether the two-stage turning point and optimal trick intervals hold for large-scale text-to-image or video generation remains to be systematically tested.
- **Oracle loss absolute values are not directly comparable across latents**: As noted by the authors, only the curve shapes should be compared, making the VA-VAE vs. SD-VAE comparison more of a qualitative conclusion.
- **Future Directions**: Since navigation is insensitive to capacity while refinement consumes it, this naturally suggests "asymmetric capacity allocation / staged distillation"—using smaller models or fewer steps for the first stage and concentrating compute on the latter. Tab. 1 provides a feasibility signal for this, though a full training scheme was not presented.

## Related Work & Insights
- **vs. existing work on closed-form diffusion objectives**: Others have derived closed-form objectives under different formulas; this paper focuses on the marginal velocity field of rectified flow and specifically uses the closed-form solution to reveal the "two-stage" structure and attribute memorization/generalization, rather than just as a sampling tool.
- **vs. memorization/generalization analysis (small dataset route)**: Most previous studies were on FFHQ/CIFAR-10 under from-scratch sampling. Ours points out that in large-scale data, "memorization from scratch" is nearly impossible, and true memorization comes from continuing training trajectories during the refinement stage, moving conclusions to ImageNet scale.
- **vs. Song et al.’s resampling observations**: They empirically found the divergence in memorization-generalization between early/late resampling. Ours provides a principled explanation (navigation = generalization, refinement = memorization) using the two-stage oracle objective.
- **vs. original proposals of timestep shifting / CFG interval**: These tricks originally had various motivations (imbalance of high-resolution noise, avoiding over-guidance). Ours unifies them as "adjusting compute allocation between stages / adding guidance during the early-to-mid refinement stage."

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Expressing the intractable FM objective as a closed-form oracle and revealing the two-stage structure is a fresh perspective with strong explanatory power.
- Experimental Thoroughness: ⭐⭐⭐⭐ Quantitative validation on ImageNet (capacity/timestep/CFG/latent) is comprehensive, but coverage across tasks (text-to-image/video) is limited.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear derivations, distinct mapping between figures and conclusions, and well-summarized takeaways.
- Value: ⭐⭐⭐⭐⭐ Provides an actionable theoretical framework for diffusion training dynamics, offering direct guidance for sampling tuning and architectural design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] VeCoR — Velocity Contrastive Regularization for Flow Matching](vecor_--_velocity_contrastive_regularization_for_flow_matching.md)
- [\[CVPR 2026\] LeapAlign: Post-Training Flow Matching Models at Any Generation Step by Building Two-Step Trajectories](leapalign_post_training_flow_matching_models_at_any_generation_step.md)
- [\[ICML 2026\] Stable Velocity: A Variance Perspective on Flow Matching](../../ICML2026/image_generation/stable_velocity_a_variance_perspective_on_flow_matching.md)
- [\[CVPR 2026\] VDE: Training-Free Accelerating Rectified Flow Model via Velocity Decomposition and Estimation](vde_training-free_accelerating_rectified_flow_model_via_velocity_decomposition_a.md)
- [\[CVPR 2026\] Few-Step Diffusion Sampling Through Instance-Aware Discretizations](few-step_diffusion_sampling_through_instance-aware_discretizations.md)

</div>

<!-- RELATED:END -->

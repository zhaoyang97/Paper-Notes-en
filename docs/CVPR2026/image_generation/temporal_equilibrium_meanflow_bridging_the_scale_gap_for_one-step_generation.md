---
title: >-
  [Paper Note] Temporal Equilibrium MeanFlow: Bridging the Scale Gap for One-Step Generation
description: >-
  [CVPR 2026][Image Generation][MeanFlow] Addressing the training collapse of MeanFlow in one-step generation when increasing the proportion of "trajectory samples," this paper identifies the root cause as a severe imbalance in gradient variance across different temporal scales. It proposes two modifications with zero additional inference overhead: "Temporal E
tags:
  - CVPR 2026
  - Image Generation
  - MeanFlow
  - Flow Matching
date: 2026-05-08
content_hash: ef5faddd1ada6d9d
---
# Temporal Equilibrium MeanFlow: Bridging the Scale Gap for One-Step Generation

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Tu_Temporal_Equilibrium_MeanFlow_Bridging_the_Scale_Gap_for_One_Step_Generation_CVPR_2026_paper.html)  
**Code**: None (Project page: https://temf.github.io)  
**Area**: Diffusion Models / Image Generation  
**Keywords**: One-step generation, MeanFlow, Flow Matching, Gradient variance equilibrium, Boundary scheduling  

## TL;DR
Addressing the training collapse of MeanFlow in one-step generation when increasing the proportion of "trajectory samples," this paper identifies the root cause as a severe imbalance in gradient variance across different temporal scales. It proposes two modifications with zero additional inference overhead: "Temporal Equilibrium Weighting" and "Dynamic Boundary Scheduling," pushing the 1-NFE FID on ImageNet 256×256 to 2.62, outperforming all diffusion-based or flow-based one-step methods.

## Background & Motivation
**Background**: Mapping noise to images in a single step (one-step generation) is critical for making diffusion/flow models practical. Flow Matching directly regresses the "instantaneous velocity field" $v(x_t, t)$ to transport distributions, but sampling requires dozens or hundreds of Network Function Evaluations (NFE). MeanFlow is a representative recent method: it defines an "average velocity field" $u(x_t, r, t) = \frac{1}{t-r} \int_r^t v(x_\tau, \tau) \, d\tau$ and derives an identity connecting average and instantaneous velocities, enabling the **training of one-step generators from scratch without distillation or curriculum learning**.

**Limitations of Prior Work**: MeanFlow has a counter-intuitive practical limitation—when the proportion of "trajectory samples" (i.e., $r \neq t$, samples spanning long time intervals) exceeds an optimal threshold during training, performance degrades significantly, resulting in blurry samples and high FID. This essentially prevents the effective use of "long-range jumps," which are most beneficial for one-step generation.

**Key Challenge**: The authors attribute this to **temporal scale imbalance**. Denoting the interval length as $\Delta = t - r$, two opposing forces affect the training gradient variance: ① as $\Delta \to 0$ (near the starting point $r \approx t$), strong boundary constraints are needed for stability, but the variance of this term explodes at $O(1/\Delta)$ as a function of path curvature; ② at large $\Delta$ (long intervals $r \ll t$), the JVP (Jacobian-Vector Product) term from trajectory modeling dominates, with variance growing at $O(\Delta^2)$. MeanFlow uses a **fixed** sampling strategy that cannot simultaneously stabilize both ends.

**Goal**: To make gradient contributions "equalized" across all temporal scales during training without altering the network architecture or increasing inference overhead, thereby allowing the safe utilization of long-range trajectory samples.

**Key Insight / Core Idea**: Since variance is a function of $\Delta$ with a predictable shape ($\frac{C_1}{\Delta} + C_2 \Delta^2$), **a weight varying with $\Delta$ can be used to flatten the variance**. This is paired with a **boundary constraint schedule** that evolves with training progress—stabilizing boundaries first, then training trajectories. In short: using "temporal scale adaptive weighting + two-stage boundary scheduling" instead of fixed sampling to resolve the temporal scale imbalance in one-step generation.

## Method

### Overall Architecture
TEMF (Temporal Equilibrium MeanFlow) keeps the original MeanFlow model and identities intact, introducing changes only to the **training objective and sampling scheduling**, plus a detail to enable one-step CFG output. Given data $x_0 \sim p_{data}$, prior $x_1 \sim \mathcal{N}(0, I)$, and the noising process $x_t = \alpha_t x_0 + \beta_t x_1$, the original MeanFlow objective regresses a target velocity with stop-gradient:

$$L_{MF}(\theta) = \mathbb{E}_{x_t, r, t} \big[ \|u_\theta(x_t, r, t) - \mathrm{sg}(u_{tgt})\|_2^2 \big], \quad u_{tgt} = v_t - (t-r) \big(v_t \partial_x u_\theta + \partial_t u_\theta \big)$$

where $v_t = \alpha_t' x_0 + \beta_t' x_1$ is the conditional velocity and the term in parentheses is the JVP. The core of TEMF involves: first, multiplying the loss of each sample by a **weight $w(\Delta)$ depending only on the interval length $\Delta = t - r$** to equalize gradient variance (Design 1); second, allowing the "sampling ratio $p$ of $r \neq t$ samples" to **evolve from low to high** according to training progress $\tau$, achieving two-stage training (Design 2); finally, incorporating CFG guidance velocity into the training objective with a decreasing "velocity mixing coefficient" to suppress variance under high guidance scales (Design 3). Inference remains one-step: $x_0 = x_1 - u_\theta(x_1, 0, 1)$, with **zero extra overhead**.

The method essentially diagnoses and corrects training dynamics (weighting function + sampling schedule) rather than being a multi-module pipeline, making it clearer to explain via formulas.

### Key Designs

**1. Temporal Equilibrium Weighting: Flattening gradient variance with a weight varying with $\Delta$**

This directly addresses the conflict between the $O(1/\Delta)$ explosion at short intervals and the $O(\Delta^2)$ growth at long intervals. The authors formalize the gradient variance in Theorem 1: Under mild regularity/curvature assumptions,

$$\mathrm{Var}[\nabla_\theta L_{MF}] \le \frac{C_1}{\Delta} + C_2 \Delta^2 + O(\Delta^3), \qquad C_1, C_2 > 0$$

where $C_1/\Delta$ arises from boundary constraint variance as $\Delta \to 0$ (induced by path curvature), and $C_2 \Delta^2$ arises from the amplification of the JVP term at large $\Delta$. To ensure the weighted conditional variance is constant for all $\Delta$, $w(\Delta)^2 \big(\frac{C_1}{\Delta} + C_2 \Delta^2 \big) = \text{const}$ is required. The ideal solution is $w(\Delta) = \big(\frac{C_1}{\Delta} + C_2 \Delta^2 \big)^{-1/2}$. The paper uses a bounded, adjustable parametric approximation:

$$w(\Delta) = \frac{1}{(1 + \lambda_1 \Delta^{\beta_1}) \sqrt{1 + \lambda_2 \Delta^2}}$$

The first term $1/(1 + \lambda_1 \Delta^{\beta_1})$ adjusts boundary constraint strength (controlling the small $\Delta$ end), and the second term $1/\sqrt{1 + \lambda_2 \Delta^2}$ suppresses JVP amplification at large $\Delta$. The weighted loss becomes $L_{TEMF} = \mathbb{E} \big[ w(t-r) \|u_\theta - \mathrm{sg}(u_{tgt})\|_2^2 \big]$. Theorem 2 proves that under truncated sampling $\Delta \ge \Delta_{min} > 0$, the weighted conditional gradient variance is uniformly bounded by a finite constant $K$. Ablation studies confirm both terms are necessary: "No weighting 6.52 → JVP term only 5.13 / Boundary term only 4.86 → Full weighting 4.31."

**2. Dynamic Boundary Scheduling: Shifting the sample ratio from "stability" to "trajectories"**

While weighting solves variance across scales at a single moment, there is a temporal conflict: early in training, the model hasn't even learned the boundary condition $u(x_t, t, t) = v_t$. Forcing long-range trajectory samples at this stage causes instability. In later stages, once boundaries are stable, the focus should shift to trajectory accuracy. The authors define **boundary constraint strength** $\Gamma(\tau) = \mathbb{E} \big[ \|u_\theta(x_t, t, t) - v_t\|_2^2 \big]$ (where $\tau \in [0, 1]$ is normalized training progress) and heuristically derive an optimal trajectory for $\Gamma$ as an exponential decay $\Gamma^*(\tau) = \Gamma_0 e^{-\lambda \tau}$ (Theorem 3, noted as an intuition rather than precise characterization ⚠️ refer to original text).

In practice, they schedule the "$r \neq t$ sample ratio" $p$ using a smoothed sigmoid climb:

$$p(\tau) = p_{min} + (p_{max} - p_{min}) \, \sigma\big(\kappa(\tau - \tau_0)\big), \quad p_{min} = 0.1, \ p_{max} = 0.9, \ \kappa = 8, \ \tau_0 = 0.6$$

This divides training into two phases: $\tau < \tau_0$ is the **boundary stabilization phase** (focusing on boundary conditions with $r = t$ samples), and $\tau \ge \tau_0$ is the **trajectory optimization phase** (focusing on long-range trajectories with $r \neq t$ samples). Ablations show the schedule shape matters: Static 25% (MeanFlow approach) 5.72, Linear 4.86, Cosine 4.53, Exponential 4.31—the exponential-like schedule best aligns with the decay predicted by Theorem 3.

**3. CFG Velocity Mixing: One-step output for Classifier-Free Guidance without variance amplification**

One-step generation relies on CFG for quality, but using the guided velocity field $v_{cfg} = \omega v(x_t, t|c) + (1-\omega)v(x_t, t)$ directly as a target introduces massive variance at high guidance scales $\omega$. TEMF learns both conditional and unconditional velocities by dropping labels with a 0.1 probability and introduces **velocity mixing** to smooth the target: $v_{mix} = m \cdot v_t^{cfg} + (1-m) \cdot v_{pred}$, where the mixing coefficient decreases during training $m(\tau) = 0.3(1-\tau) + 0.1\tau$. In early stages, a large $m$ (closer to the true guidance target, but usable while model predictions are unreliable) reduces variance; in later stages, a reduced $m$ makes the model more independent. This preserves $x_0 = x_1 - u_\theta(x_1, 0, 1)$ one-step sampling while gaining CFG quality at zero inference cost. Ablation: No mixing 5.24 → Fixed $m=0.25$ 4.58 → Dynamic mixing 4.31.

### Loss & Training
The final training objective is the weighted $L_{TEMF}$. Training procedure (Algorithm 1) per step: calculate $p$ based on progress → sample $t$ (via sigmoid on Gaussian for $t \in (0,1)$) and $r \sim U(0,1)$ ensuring $t > r$ → with probability $p$, add a small perturbation to $r$ to create $r \neq t$ cases → calculate $x_t, v_t$, forward pass for $u_{pred}$, compute $u_{tgt}$ via JVP → calculate $\Delta$ and $w$ → backpropagate $w \|u_{pred} - \mathrm{sg}(u_{tgt})\|_2^2$. Training is from scratch without pre-training, distillation, or external initialization.

## Key Experimental Results

### Main Results: ImageNet 256×256 Class-Conditional Generation (FID-50K)
All TEMF models trained from scratch for 240 epochs (XL+ for an additional 60), compared against one-step methods with the same backbone.

| Method | Params | NFE | FID↓ |
|--------|--------|-----|------|
| MeanFlow-B | 131M | 1 | 6.17 |
| **Ours-B** | 131M | 1 | **4.31** |
| MeanFlow-L | 459M | 1 | 3.84 |
| **Ours-L** | 459M | 1 | **3.26** |
| MeanFlow-XL | 675M | 1 | 3.43 |
| SoFlow-XL | 675M | 1 | 3.35 |
| **Ours-XL** | 675M | 1 | **2.81** |
| **Ours-XL+** | 675M | 1 | **2.62** |
| **Ours-XL+** | 675M | 2 | **2.30** |
| DiT-XL (Multi-step) | 675M | 250×2 | 2.27 |
| SiT-XL (Multi-step) | 675M | 250×2 | 2.06 |

Ours consistently outperforms MeanFlow/SoFlow at every scale; 2.62 (1-NFE) is the new SOTA for diffusion/flow one-step methods, and 2.30 (2-NFE) approaches multi-step DiT-XL/SiT-XL requiring 500 NFE. On CIFAR-10 unconditional (55M U-Net, pixel space, no EDM preconditioning), it achieves a new SOTA of 2.81 (vs. 2.92 for MeanFlow/SoFlow).

### Ablation Study (ImageNet 256×256, DiT-B/4 131M, FID 1-NFE)

| Dimension | Config | FID | Note |
|-----------|--------|-----|------|
| Components | Baseline MeanFlow | 6.17 | Starting point |
| | + Temporal Weighting | 5.26 | Weighting provides initial drop |
| | + Dynamic Scheduling | 4.93 | Scheduling adds further reduction |
| | + CFG Integration | 4.61 | Guidance improves quality |
| | Full TEMF | **4.31** | Synergy between all three |
| Weighting | None / JVP only / Boundary only / Full | 6.52 / 5.13 / 4.86 / **4.31** | Both terms are essential |
| Scheduling | Static 25% / Linear / Cosine / Exponential | 5.72 / 4.86 / 4.53 / **4.31** | Exponential optimal; matches Thm 3 |
| Mixing | None / Fixed 0.25 / Dynamic | 5.24 / 4.58 / **4.31** | Early variance reduction is key |
| $r \neq t$ ratio | 0 / .25 / .5 / .75 / 1.0 | 5.95 / 4.87 / **4.31** / 4.45 / 4.68 | 0.5 is optimal under equilibrium |

### Key Findings
- **Weighting and scheduling are highly complementary**: Weighting provides the largest single gain (6.17 → 5.26), while scheduling and CFG push the FID down to 4.31. Removing any part causes regression.
- **Correction of the "more trajectory samples is better" bias**: Original MeanFlow is extremely sensitive to the $r \neq t$ ratio (collapsing if pushed too high); the temporal equilibrium framework pushes the optimal ratio from ~0.25 to 0.5, proving weighting truly "unlocks" long-range information.
- **Clear hyperparameter sweet spots**: FID is lowest at $\lambda_1=0.5, \lambda_2=1.0, \beta_1=1.0, m=0.3$. Deviations in either direction degrade results, indicating the weighting balances between "under-constrained" and "over-constrained" states.
- **Single-step FID improves monotonically with capacity**: The B → XL scaling law is consistent with DiT.

## Highlights & Insights
- **Translating "training collapse" into a variance formula**: Instead of just observing that long-range samples cause failure, the authors derive $\mathrm{Var} \le C_1/\Delta + C_2 \Delta^2$. The solution (finding $w(\Delta)$ to keep variance constant) follows directly from the formula—the diagnosis and medicine are perfectly aligned.
- **Zero inference overhead for both modifications**: Weighting affects only the loss, and scheduling affects only the sampling ratio. The model architecture and one-step sampling $x_0 = x_1 - u_\theta(x_1, 0, 1)$ remain unchanged, achieving cross-scale stability at zero deployment cost.
- **Boundary constraints as evolving, not static**: Scheduling boundary conditions as an exponentially decaying quantity quantized the intuition of "learning to stand before learning to run," a principle transferable to other from-scratch training scenarios (e.g., Consistency Models, Flow Map).
- **Reusable trick**: When the variance of a target term is a known function of an interval/scale, a general recipe is to "derive a weight to equalize variance and approximate it with a bounded parameterization."

## Limitations & Future Work
- **Heuristic theory**: Theorem 3's exponential decay, the error decomposition $E_{bias}(\Gamma) = a\Gamma^2$, and $E_{var}(p) = b/(1-p)$ are explicit heuristic approximations (⚠️ refer to original text) rather than precise characterizations of true training dynamics. The $O(1/\Delta)$ term is also empirically fitted.
- **Requirement for truncated sampling $\Delta \ge \Delta_{min}$**: Theorem 2's bounded variance depends on a positive minimum interval truncation; very small $\Delta$ might still be unstable, and the setting of this threshold is not fully discussed.
- **Significant number of hyperparameters**: $\lambda_1, \lambda_2, \beta_1$ plus the schedule's $p_{min}, p_{max}, \kappa, \tau_0$ and the mixing coefficient all influence performance. Tuning costs for new datasets/resolutions are unknown. Experiments were mainly on ImageNet-256 and CIFAR-10; higher resolutions/T2I are not yet verified.
- **Future directions**: Replacing heuristic schedules with learnable/adaptive variance estimation; extending equilibrium weighting to other "dual-time" objectives like Flow Map or Consistency Models.

## Related Work & Insights
- **vs. MeanFlow**: Uses the same identity and architecture. MeanFlow uses a fixed sampling ratio; TEMF identifies the resulting temporal scale variance imbalance and fixes it with weighting and scheduling, consistently outperforming it at all scales with no extra cost.
- **vs. Consistency Models (iCT / sCT / Shortcut / IMM)**: These rely on cross-time consistency/moment constraints for one-step generation, often requiring 1×2 NFE or showing lower performance (IMM-XL 7.77, Shortcut-XL 10.60). TEMF leads significantly with true 1-NFE FID 2.62.
- **vs. SoFlow / AlphaFlow**: These also modify MeanFlow loss (computational efficiency/coupling). TEMF focuses on temporal scale gradient variance equilibrium and slightly outperforms SoFlow per scale.
- **vs. Flow Map (Boffi et al.)**: Flow Map learns path integrals of velocity (total displacement). TEMF still uses average velocity, but its logic of "adaptive weighting based on interval length" is highly relevant to Flow Map objectives.

## Rating
- Novelty: ⭐⭐⭐⭐ Precisely attributes one-step training instability to temporal scale variance imbalance and provides a corresponding equilibrium weighting.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers four backbone scales, nine ablation tables, and CIFAR-10 cross-validation with self-consistent conclusions.
- Writing Quality: ⭐⭐⭐⭐ Clear connection between formulas and motivation, though several theoretical parts are acknowledged as heuristic.
- Value: ⭐⭐⭐⭐⭐ Achieves FID 2.62 for diffusion/flow one-step generation at zero extra cost, approaching multi-step models. High practical value.

<!-- RELATED:START -->
<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] OMP: One-step Meanflow Policy with Directional Alignment](../../ICML2026/image_generation/omp_one-step_meanflow_policy_with_directional_alignment.md)
- [\[CVPR 2026\] Extending One-Step Image Generation from Class Labels to Text via Discriminative Text Representation](emf_meanflow_text_to_image.md)
- [\[CVPR 2026\] BiFM: Bidirectional Flow Matching for Few-Step Image Editing and Generation](bifm_bidirectional_flow_matching_for_few-step_image_editing_and_generation.md)
- [\[CVPR 2026\] MeanFlow Transformers with Representation Autoencoders](meanflow_transformers_with_representation_autoencoders.md)
- [\[CVPR 2026\] WaDi: Weight Direction-aware Distillation for One-step Image Synthesis](wadi_weight_direction-aware_distillation_for_one-step_image_synthesis.md)

</div>

<!-- RELATED:END -->

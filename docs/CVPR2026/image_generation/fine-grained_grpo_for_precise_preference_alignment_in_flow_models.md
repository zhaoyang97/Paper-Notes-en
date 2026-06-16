---
title: >-
  [Paper Note] Fine-Grained GRPO for Precise Preference Alignment in Flow Models
description: >-
  [CVPR 2026][Image Generation][GRPO] G²RPO (Granular-GRPO) transforms the sparse reward paradigm in flow-based GRPO training—where SDE noise is injected at every step and terminal rewards are averaged across the trajectory—into a "Singular Stochastic Sampling" approach where randomness is injected only at one step while others follow deterministic ODEs. B
tags:
  - CVPR 2026
  - Image Generation
  - GRPO
date: 2026-05-08
content_hash: ecee00935493f1b9
---
# Fine-Grained GRPO for Precise Preference Alignment in Flow Models

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Zhou_Fine-Grained_GRPO_for_Precise_Preference_Alignment_in_Flow_Models_CVPR_2026_paper.html)  
**Code**: https://bujiazi.github.io/g2rpo.github.io/ (Project Page)  
**Area**: Diffusion Models / Alignment RLHF  
**Keywords**: GRPO, Flow Model Alignment, Sparse Reward, Credit Assignment, Multi-granularity Denoising

## TL;DR
G²RPO (Granular-GRPO) transforms the sparse reward paradigm in flow-based GRPO training—where SDE noise is injected at every step and terminal rewards are averaged across the trajectory—into a "Singular Stochastic Sampling" approach where randomness is injected only at one step while others follow deterministic ODEs. By calculating and fusing advantages across multiple denoising granularities for the same direction, it provides precise and comprehensive reward signals. On Flux.1-dev, it outperforms DanceGRPO and MixGRPO across various in-/out-domain metrics including HPS, ImageReward, and Unified Reward.

## Background & Motivation
**Background**: Fine-tuning diffusion/flow generative models using online RL (specifically GRPO) for human preference alignment is a prominent research direction. GRPO eliminates the need for a separate value model by updating the policy based on relative advantages within a group of samples. To enable deterministic flow matching models to generate diverse denoising directions for comparison, methods like Flow-GRPO and DanceGRPO replace ODE samplers with equivalent SDEs, injecting random noise $\sigma_t dw_t$ at **every** denoising step.

**Limitations of Prior Work**: This "full-sequence stochasticity" leads to two core issues. First, **Sparse Reward**: reward models only score the final image $x_0$, and this terminal advantage $A_0^i$ is **uniformly broadcasted** to every step $A_t^i$ along the trajectory. Since the reward appears only after the full chain of decisions, it cannot be attributed to specific noise injected at intermediate steps, breaking the credit assignment chain and leading to unstable training. Second, **Incomplete Evaluation**: existing methods bind each denoising direction to a **fixed number of steps**, evaluating samples at a single granularity. However, the same SDE direction can produce different details and fluctuating scores under different denoising intervals, making a single-granularity score insufficient to judge the true value of a direction.

**Key Challenge**: SDE exploration necessitates "breadth" (disturbing every step for exploration), while reward attribution requires "precision" (strong correlation between noise and reward). Full-sequence stochasticity sacrifices attribution precision for breadth, while fixed-granularity imaging makes the "true value" of a direction susceptible to the randomness of specific intervals.

**Core Idea**: Instead of diffusing randomness across the entire trajectory, **randomness is constrained to a single step** (with other steps following deterministic ODEs). This ensures the noise injected at that step is the sole source of intra-group reward variance, allowing terminal rewards to be precisely attributed to it. Furthermore, **multiple denoising granularities** are used to image and calculate advantages for that sampled direction, which are then fused to obtain a robust comprehensive evaluation.

## Method

### Overall Architecture
G²RPO maintains the skeleton of flow-based GRPO, modeling the denoising process as a multi-step MDP where the state is $s_t=(c,t,x_t)$ and the action $a_t$ is the single-step denoising direction. Policy $\pi_\theta$ is updated using intra-group relative advantages. It modifies sampling and advantage calculation in two ways:

1. **Sampling Phase**: Given a text prompt $c$ and an initial noise $x_T$, a deterministic ODE advances the noise to a selected step $k$ to obtain a **shared starting point** $x_k$. SDE sampling is performed only at step $x_k$ to branch out $G$ different directions $\{x_{k-1}^i\}_{i=1}^G$. Subsequently, all branches switch back to ODE to complete the process. This ensures reward differences within the group stem entirely from the randomness at step $k$.
2. **Advantage Phase**: For each direction $x_{k-1}^i$, instead of using only the standard granularity, **interval sampling** is performed using a set of scaling factors $\Lambda=\{\lambda_1,\dots,\lambda_J\}$. This generates multiple images at different granularities. Advantages are calculated for each granularity and summed into $A_k^{i,\mathrm{mix}}$ as the final evaluation for the GRPO loss.

The process iterates over $k$ within a selected set of timesteps $M$ (typically the first half of denoising), where each $k$ produces a group of samples with multi-granularity fused advantages for policy updating.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Text prompt c<br/>+ Initial noise x_T"] --> B["ODE advances to shared starting point x_k"]
    B --> C["Singular Stochastic Sampling<br/>Inject SDE at k only<br/>Branch into G directions"]
    C -->|Multi-granularity imaging for each direction| D["Multi-Granularity<br/>Advantage Integration<br/>Sum advantages across granularities"]
    D --> E["Fused Advantage A_k^mix<br/>→ GRPO Loss updates π_θ"]
    E -->|Iterate over k∈M (First half)| B
```

### Key Designs

**1. Singular Stochastic Sampling: Constraining randomness for clean credit assignment**

To address the "full-sequence stochasticity → non-attributable rewards" problem, this design uses deterministic ODEs ($dx_t=v_\theta(x_t,t)dt$) to push all samples in a group from $x_T$ to the **same starting point** $x_k$. SDE sampling (Euler–Maruyama discretization, injecting $\sigma_t\sqrt{\Delta t}\,\epsilon$ where $\sigma_t=\eta\sqrt{t/(1-t)}$) is only applied at step $k$ to generate $G$ different next states $\{x_{k-1}^i\}$. All subsequent steps return to deterministic ODE until $x_0$. Because all steps except $k$ are deterministic, the variance in rewards $\{R(x_{0\leftarrow k}^i,c)\}$ is **entirely determined by the randomness at step $k$**, allowing for a step-aligned, precise advantage:

$$A_k^i=\frac{R(x_{0\leftarrow k}^i,c)-\mathrm{mean}(\{R(x_{0\leftarrow k}^i,c)\})}{\mathrm{std}(\{R(x_{0\leftarrow k}^i,c)\})}$$

Unlike older methods that broadcast $A_0^i$ indiscriminately, the advantage at step $k$ here is a direct measure of whether "the noise at this step was good," establishing a dense causal correspondence. The optimization set $M=\{T,T-1,\dots,\lfloor T/2\rfloor\}$ focuses on the **first half** of denoising, where SDE exploration has a larger impact on the trajectory and GRPO gains are higher. Efficiency is also improved as importance ratios $r_k^i(\theta)$ can reuse the same ODE direction $v_k$ due to the shared starting point.

**2. Multi-Granularity Advantage Integration (MGAI): Cross-evaluating directions to remove granularity bias**

The authors observe that even with singular stochasticity, the same SDE direction can yield different imaging details and fluctuating reward scores depending on the subsequent **denoising interval**. MGAI addresses this by using a set of integer scaling factors $\Lambda=\{\lambda_1,\dots,\lambda_J\}$ for interval sampling. For each direction $x_{k-1}^i$, denoising sequences are defined as:

$$S_j=\Big\{1,\,1+\lambda_j,\,1+2\lambda_j,\,\dots,\,\big\lfloor\tfrac{k-1}{\lambda_j}\big\rfloor\lambda_j\Big\}$$

Larger $\lambda_j$ indicates coarser granularity. Each direction results in $J$ images at different granularities, each receiving a score and an intra-group advantage $A_k^{i,j}$. These are **summed** to form the multi-granularity advantage:

$$A_k^{i,\mathrm{mix}}=\sum_{j}^{J}A_k^{i,j}$$

This summation ensures that a truly superior direction remains leading across different granularities, making the evaluation robust and particularly improving out-of-domain metrics like the LLM-augmented Unified Reward. The final GRPO objective replaces the step advantage with $A_k^{i,\mathrm{mix}}$:

$$f=\frac{1}{G}\sum_{i=1}^{G}\frac{1}{K}\sum_{k\in M}\min\!\big(r_k^i(\theta)A_k^{i,\mathrm{mix}},\,\mathrm{clip}(r_k^i(\theta),1-\varepsilon,1+\varepsilon)A_k^{i,\mathrm{mix}}\big)$$

### Loss & Training
The clipped surrogate objective of GRPO is used with KL coefficient $\beta=0$ for stability. Backbone: Flux.1-dev. Dataset: HPSv2 (103.7k training / 400 test prompts). Sampling: $G=12$ images, total steps $T=16$, advantage clip $\varepsilon=5$, noise level $\eta=0.7$, granularity set $\Lambda=\{1,2,3\}$. Training used 16×H200, batch size 1, AdamW, learning rate $2\times10^{-6}$, weight decay $1\times10^{-4}$, and bf16 mixed precision.

## Key Experimental Results

### Main Results
Two setups: ① HPS-v2.1 only (in-domain upper bound, prone to hacking); ② Joint HPS-v2.1 + CLIP training (more robust). Evaluations use HPS, CLIP, PickScore (PS), ImageReward (IR), and Unified Reward (UR).

| Training Reward | Method | HPS | CLIP | PS | IR | UR |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| — | Flux.1-dev (base) | 0.305 | 0.388 | 0.226 | 1.040 | 3.621 |
| HPS | DanceGRPO | 0.353 | 0.375 | 0.228 | 1.233 | 3.548 |
| HPS | MixGRPO | 0.378 | 0.358 | 0.225 | 1.266 | 3.421 |
| HPS | G²RPO w/o MGAI | 0.376 | 0.351 | 0.228 | 1.286 | 3.469 |
| HPS | **G²RPO** | **0.385** | 0.355 | 0.229 | **1.313** | 3.487 |
| HPS&CLIP | DanceGRPO | 0.331 | 0.389 | 0.227 | 1.128 | 3.569 |
| HPS&CLIP | MixGRPO | 0.363 | 0.399 | 0.230 | 1.436 | 3.661 |
| HPS&CLIP | G²RPO w/o MGAI | 0.372 | 0.395 | 0.234 | 1.421 | 3.688 |
| HPS&CLIP | **G²RPO** | 0.376 | **0.406** | **0.235** | **1.483** | **3.783** |

When training with HPS only, Singular Stochastic Sampling (w/o MGAI) shows a 6.52% relative improvement in HPS over DanceGRPO, confirming that singular-step randomness provides cleaner signals. In the joint HPS & CLIP setup, the full G²RPO leads across in-/out-domain metrics, with UR significantly increasing from 3.661 (MixGRPO) to 3.783.

### Ablation Study: Denoising Granularity Set $\Lambda$ (HPS&CLIP Training)
| $\Lambda$ | HPS | CLIP | PS | IR | UR |
|:---|:---:|:---:|:---:|:---:|:---:|
| {1} (i.e., w/o MGAI) | 0.372 | 0.395 | 0.234 | 1.421 | 3.688 |
| {1,2} | 0.375 | 0.404 | 0.234 | 1.468 | 3.759 |
| {1,3} | 0.378 | 0.404 | 0.234 | 1.465 | 3.760 |
| {1,2,3} | 0.376 | **0.406** | **0.235** | **1.483** | **3.783** |

Increasing the number of granularities provides more comprehensive evaluations, steadily improving IR and UR metrics. {1,2,3} was chosen as the optimal balance for multiple metrics.

### Robustness across Inference Steps (HPS&CLIP Training)
| Steps | Method | HPS | CLIP | PS | IR | UR |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| 10 | MixGRPO | 0.358 | 0.401 | 0.230 | 1.431 | 3.641 |
| 10 | **G²RPO** | **0.378** | **0.408** | **0.235** | **1.519** | **3.805** |
| 20 | MixGRPO | 0.363 | 0.401 | 0.230 | 1.430 | 3.651 |
| 20 | **G²RPO** | **0.376** | **0.407** | **0.235** | **1.511** | **3.806** |

G²RPO maintains its lead even at lower inference budgets (10 steps), proving that MGAI makes the model more robust to denoising configurations.

### Key Findings
- **MGAI primarily aids out-of-domain generalization**: Comparing the version without MGAI to the full version, IR and UR (unseen during training) show the most significant gains, validating that cross-granularity evaluation removes accidental bias.
- **Singular Stochastic Sampling is effective on its own**: Achieving a +6.52% HPS gain over DanceGRPO proves that credit assignment is the primary bottleneck in flow-based GRPO.
- **Single-reward training leads to hacking**: Optimizing HPS alone can drop UR below the baseline (MixGRPO 3.421 vs. baseline 3.621), confirming the necessity of multi-reward training.

## Highlights & Insights
- **Decoupling "Exploration Breadth" and "Attribution Precision"**: Contrary to the assumption that exploration requires full-trajectory noise, this paper shows that a single step of randomness is enough to create intra-group variance while keeping the credit assignment causal and clean.
- **Multi-granularity Advantage Fusion**: By treating different granularities as different reward "views," it borrows the multi-reward fusion technique for a simpler, highly effective implementation.
- **Efficiency through Shared Start Points**: Reusing the same $v_k$ for across the group minimizes the overhead of step-wise calculation.

## Limitations & Future Work
- **Manual Heuristics for $\Lambda$**: {1,2,3} is empirically effective, but the optimal set for different backbones or step counts is unknown, and costs increase with more granularities.
- **Optimization of First Half Only**: Based on the assumption that early steps are more critical, it remains to be seen if ignoring late-stage refinements affects certain fine details.
- **Scope**: Verified only on T2I and Flux.1-dev; generalizability to video flow models or weaker backbones is unproven.
- **Exploration Trade-off**: Constraining randomness to one step might limit the diversity of reachable denoising trajectories compared to full SDE exploration.

## Related Work & Insights
- **vs DanceGRPO / Flow-GRPO**: These methods convert ODE to SDE globally and broadcast rewards. This work uses singular SDE steps for "dense" step-aligned alignment and adds multi-granularity evaluation.
- **vs MixGRPO**: MixGRPO focuses on efficiency with mixed ODE-SDE. G²RPO addresses evaluation precision/comprehensiveness, outperforming MixGRPO particularly at low-step budgets.
- **vs Diffusion-DPO**: DPO optimizes from paired preferences without explicit reward models but lacks the continuous exploration of online RL.

## Rating
- Novelty: ⭐⭐⭐⭐ "Singular SDE + Multi-granularity Advantage Integration" accurately targets credit assignment and evaluation bias in flow-GRPO.
- Experimental Thoroughness: ⭐⭐⭐⭐ Coverage of 5 reward models, in/out-domain analysis, and step robustness is strong, though cost analysis and backbone variety are limited.
- Writing Quality: ⭐⭐⭐⭐ Clear problem diagnosis and mapping of solutions; comprehensive math and algorithms.
- Value: ⭐⭐⭐⭐ Provide a plug-and-play improvement for RLHF in flow models with immediate utility for preference alignment research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Stepwise-Flow-GRPO：给流匹配模型的去噪步逐步分配信用](stepwise_credit_assignment_for_grpo_on_flow-matching_models.md)
- [\[CVPR 2026\] CogniEdit: Dense Gradient Flow Optimization for Fine-Grained Image Editing](cogniedit_dense_gradient_flow_optimization_for_fine-grained_image_editing.md)
- [\[CVPR 2026\] BeautyGRPO: Aesthetic Alignment for Face Retouching via Dynamic Path Guidance and Fine-Grained Preference Modeling](beautygrpo_aesthetic_alignment_for_face_retouching_via_dynamic_path_guidance_and.md)
- [\[CVPR 2026\] Towards Fine-Grained Attribution: Instance-Aware Preference Optimization for Aligning Diffusion Models](towards_fine-grained_attribution_instance-aware_preference_optimization_for_alig.md)
- [\[CVPR 2026\] Neighbor GRPO: Contrastive ODE Policy Optimization Aligns Flow Models](neighbor_grpo_contrastive_ode_policy_optimization_aligns_flow_models.md)

</div>

<!-- RELATED:END -->

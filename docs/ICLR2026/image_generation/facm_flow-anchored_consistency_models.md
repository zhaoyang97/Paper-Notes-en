---
title: >-
  [Paper Note] FACM: Flow-Anchored Consistency Models
description: >-
  [ICLR 2026][Image Generation][Consistency Models] FACM jointly trains Flow Matching (as an "anchor") and Consistency Models (as a "shortcut" goal) within a single model. By employing an "extended time interval" technique to decouple the two tasks into different time domains, it fundamentally resolves the training collapse issue in continuous-time consistency models. It achieves FID scores of 1.70/1.32 with NFE=1/2 on ImageNet $256 \times 256$, respectively…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "Consistency Models"
  - "Flow Matching"
  - "Few-step Generation"
  - "Distillation"
  - "Training Stability"
  - "JVP"
  - "FSDP"
date: 2026-05-08
content_hash: 8e9e9335b1d56742
---

# FACM: Flow-Anchored Consistency Models

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=k9BpW1c4in](https://openreview.net/forum?id=k9BpW1c4in)  
**Code**: [https://github.com/ali-vilab/FACM](https://github.com/ali-vilab/FACM)  
**Area**: Image Generation / Few-step Sampling / Consistency Models  
**Keywords**: Consistency Models, Flow Matching, Few-step Generation, Distillation, Training Stability, JVP, FSDP  

## TL;DR
FACM jointly trains Flow Matching (as an "anchor") and Consistency Models (as a "shortcut" goal) within a single model. By employing an "extended time interval" technique to decouple the two tasks into different time domains, it fundamentally resolves the training collapse issue in continuous-time consistency models. It achieves FID scores of 1.70/1.32 with NFE=1/2 on ImageNet $256 \times 256$, respectively, and scales effectively to 14B text-to-image models.

## Background & Motivation
**Background**: While diffusion and flow matching models yield high generation quality, they require dozens or hundreds of steps for inference, hindering real-time applications. Consistency Models (CM) attempt to "shortcut" the ODE trajectory by mapping any point $x_t$ directly to the endpoint $x_1$ in a single step. Discrete-time versions suffer from discretization errors, while continuous-time versions theoretically circumvent these errors but have long been plagued by severe **training instability**, leading to frequent collapse.

**Limitations of Prior Work**: Recent approaches address symptoms rather than the root cause. One lineage, sCM, relies on regularization and architectural modifications (e.g., pixel normalization) to stabilize continuous-time training, making it difficult to adapt to large-scale pre-trained models. Another lineage, Flow Mapping (e.g., MeanFlow, IMM), stabilizes training by redesigning the shortcut objective—such as modeling "average velocity" or adding multi-step self-consistency constraints. However, these rely on an **overly-coupled single objective** to learn both the "flow" and the "shortcut," failing to explicitly decouple the two tasks and sacrificing trajectory fidelity.

**Key Challenge**: The training target for continuous-time CM, $T = v + (1-t)\frac{dF_\theta}{dt}$, is **self-referential**, containing both the true instantaneous velocity $v$ and the model's own derivative estimate. Since the CM loss only supervises the final prediction $F_\theta$, there is no mechanism ensuring the model's internal dynamics remain faithful to the underlying instantaneous velocity field $v(x_t,t)$. Without this "anchor," the model output drifts, and the derivative term $(1-t)\frac{dF_\theta}{dt}$ overpowers the ground-truth velocity signal $v$. Satisfying the consistency identity no longer converges to the boundary conditions, causing the training target to become extremely noisy, creating a vicious cycle of error amplification until collapse.

**Goal**: Fundamentally eliminate instability in continuous CM while remaining architecture-agnostic and scalable to ultra-large models.

**Core Idea** (**Flow-Anchoring**): The authors argue that instability stems not from the shortcut objective itself, but from "training the shortcut in isolation," causing the model to lose its anchor in the underlying velocity field. The solution is straightforward: **explicitly re-introduce supervision from the instantaneous velocity field (i.e., a Flow Matching task) as a dynamic anchor for the main shortcut objective**, ensuring well-behaved gradient fields and stabilizing the derivative term in the CM objective.

## Method

### Overall Architecture
FACM requires no specialized architecture and utilizes a **mixed-objective** training strategy: the total loss is the simple summation of the Flow Matching loss (anchor for stability) and the Consistency Model loss (accelerator for the shortcut), $L_{\text{FACM}} = L_{\text{FM}} + L_{\text{CM}}$. To allow the model to distinguish between predicting instantaneous velocity versus average velocity, the authors use an **extended time interval** condition signal to decouple the tasks into $[0,1]$ and $[1,2]$ time domains, combined with a stable interpolative CM target and a scalable Chain-JVP implementation.

```mermaid
flowchart TD
    A[Sample x0, x1, t<br/>Construct xt=(1-t)x0+t·x1] --> B[Define two conditions by t<br/>cCM=t, cFM=2-t]
    B --> C[FM Branch: Fθ(xt, cFM)<br/>Regress instantaneous velocity v — Anchor]
    B --> D[CM Branch: Fθ(xt, cCM) + JVP<br/>Calc. consistency residual g and clamp]
    D --> E[Relaxed Interpolation Target<br/>vtar=(1-α)·sg(FCM)+α·T(F)]
    C --> F[L_FM]
    E --> G[L_CM]
    F --> H[L_total = L_FM + L_CM]
    G --> H
```

### Key Designs
**1. Flow-Anchoring: Using FM as an Anchor to Cure Instability.** The shortcut objective's physical meaning is clarified: using OT-FM parameterization $f_\theta(x_t,t)=x_t+(1-t)F_\theta(x_t,t)$, a one-step shortcut $f_\theta=x_1$ requires the network to learn $F_\theta(x_t,t)=\frac{x_1-x_t}{1-t}$, which is the **average velocity** $v(x_t,t)$ from $x_t$ to the endpoint. Its derivative yields the identity $v(x_t,t)=v(x_t,t)+(1-t)\frac{dv}{dt}$, identical in form to the continuous CM and MeanFlow identities. Instability occurs because the model lacks direct supervision of the instantaneous velocity $v$ when trained in isolation. FACM re-introduces the Flow Matching loss as an anchor:

$$L_{\text{FM}}(\theta) = \mathbb{E}\big[\|F_\theta(x_t, c_{\text{FM}}) - v\|_2^2 + L_{\cos}(F_\theta(x_t, c_{\text{FM}}), v)\big]$$

The anchor ensures the model's gradient field is well-behaved, stabilizing the derivative term in the CM objective. Stability is guaranteed by the Flow-Anchoring principle regardless of specific weighting or loss choices.

**2. Extended Time Interval: Decoupling Tasks Across Time Domains.** To enable a single model to predict both instantaneous and average velocities, the authors **double the conceptual time domain**: the CM task occupies $t\in[0,1]$ with $c_{\text{CM}}=t$, while the FM task at the same $x_t$ is mapped to $[1,2]$ with $c_{\text{FM}}=2-t$. This provides decoupled, symmetric, and easily distinguishable conditions without architecture changes. It also ensures boundary continuity: as $t\to1$, the CM target converges to the FM target:

$$\lim_{t\to1^-}\Big(v + (1-t)\frac{dF_\theta(x_t,t)}{dt}\Big) = v$$

**3. Relaxed Fixed-point Iteration CM Target.** The CM loss is interpreted as a fixed-point problem $F_\theta = T(F_\theta)$. To avoid gradient explosion when approaching $T(F) \triangleq v+(1-t)\frac{dF}{dt}$, the authors calculate the residual $g = F_{\theta^-} - T(F_{\theta^-})$ using a stop-gradient model, clamp it to $[-1,1]$, and construct an **interpolation target**:

$$v_{\text{tar}} = (1-\alpha(t))F_{\theta^-} + \alpha(t)\,T(F_{\theta^-})$$

This acts as a relaxed iteration between current output and the ideal target, used with a weighted norm L2 loss $L_{\text{CM}}=\mathbb{E}[\beta(t)\cdot L_{\text{norm}}(F_\theta(x_t,c_{\text{CM}}), v_{\text{tar}})]$.

**4. Chain-JVP: Making JVP Compatible with FSDP for 14B Models.** Total derivatives $\nabla_t F_\theta$ require Jacobian-Vector Products (JVP). Standard JVP requires materializing all parameters $\theta$ on-device, which triggers all-gather in FSDP, causing memory to explode for 10B+ models. The authors apply the chain rule to decompose JVP by module sequence $J_{F_\theta}(z)\cdot v = J_{f_L}\cdot(\cdots J_{f_1}(z_0)\cdot v)$. By calculating JVP for only one module at a time within FSDP logic, **only one module's parameters are materialized at any moment**. Peak memory depends on the largest module rather than the whole model, enabling the scaling of FACM to Wan 2.2 (14B) text-to-image models.

## Key Experimental Results

### Main Results (Few-step SOTA)

ImageNet $256 \times 256$ (class-conditional):

| Method | Params | NFE | FID (↓) |
|---|---|---|---|
| LightningDiT (Multi-step Baseline) | 675M | 250×2 | 1.35 |
| MeanFlow | 676M | 1 | 3.43 |
| **FACM (Ours)** | 675M | **1** | **1.70** |
| MeanFlow | 676M | 2 | 2.20 |
| IMM | 675M | 1×2 | 7.77 |
| **FACM (Ours)** | 675M | **2** | **1.32** |

CIFAR-10 (unconditional): FACM achieves FID 2.69 at NFE=1 (outperforming sCM 2.85, MeanFlow 2.92) and 1.87 at NFE=2 (outperforming IMM 1.98, sCM 2.06).

### Ablation Study

Different backbones (ImageNet $256 \times 256$, NFE=2, FID↓), demonstrating architecture-agnosticism:

| Backbone | Baseline | sCM† | MeanFlow† | **FACM** |
|---|---|---|---|---|
| SiT-XL/2 | 2.06 | 2.83 | 2.27 | **2.07** |
| REPA | 1.42 | 2.25 | 1.88 | **1.52** |
| DiT-XL/2 | 2.27 | 2.91 | 2.62 | **2.31** |
| LightningDiT | 1.35 | 1.94 | 1.74 | **1.32** |

Stabilization strategies (NFE=1): sCM collapses (✗) without pixel norm. FACM using auxiliary conditions achieves 1.97, while the **extended interval version achieves 1.81**.

Key components (Epoch 10, NFE=1): Pure MeanFlow (0% FM) collapses (FID 372-391). Adding 75% FM reduces FID to 43. Flow-Anchoring with extended time interval drops it to **4.31**, and adding interpolation $\alpha$ reaches 3.42.

### Key Findings
- **FM anchor is a sufficient condition for stability**: Removing FM leads to collapse; adding it ensures stability regardless of loss weighting.
- **Robustness to FM loss weight**: In finetuned settings, $\lambda_{\text{FM}}$ as low as $10^{-8}$ maintains stability. Direct summation ($\lambda_{\text{FM}}=1.0$) works best without tuning.
- **Better teachers yield better results**: FACM performance scales monotonically with teacher quality, indicating high-fidelity trajectory compression.

## Highlights & Insights
- **Deep Diagnosis**: Attributing continuous CM instability to "self-referential objectives + loss of flow anchor" provides a more fundamental explanation than empirical "regularization" fixes.
- **Simple yet Universal**: $FM + CM$ summation and time interval extension require no architecture changes, allowing seamless integration with any pre-trained backbone.
- **Decoupling vs. Coupling**: Unlike MeanFlow which treats instantaneous velocity as a boundary case, FACM keeps the anchor supervision "clean" and undiluted in separate time domains.
- **Scaling via Chain-JVP**: Solving the memory conflict between JVP and FSDP is the engineering key that moves the method from a toy problem to 14B-scale world-class models.

## Limitations & Future Work
- Primarily relies on a two-stage process (FM pre-training then distillation); the potential for end-to-end training from scratch remains to be fully explored.
- Evaluation focuses heavily on FID; systematical large-scale human preference and prompt consistency evaluations for the 14B model could be more comprehensive.
- Using $[1,2]$ for FM occupies condition space; potential side effects on models already utilizing full time-condition ranges are not fully discussed.

## Related Work & Insights
- **sCM**: Stabilizes via architecture changes (pixel norm). FACM notes this limits adaptability to large pre-trained models.
- **Flow Mapping / MeanFlow / IMM**: Stabilize via redesigned shortcut objectives. FACM argues these "over-couple and dilute" flow supervision.
- **Insight**: When a self-referential objective is unstable, instead of adding regularization, **explicitly re-introduce the quantity it implicitly depends on but lacks direct supervision for**—in this case, the instantaneous velocity field.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ High originality in diagnosing "anchor loss" and the minimalist extended-interval solution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers multiple backbones, scaling to 14B, and detailed stability ablations.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear derivation and logical progression from motivation to solution.
- **Value**: ⭐⭐⭐⭐⭐ Significant impact for few-step generation by providing a stable, scalable, and architecture-agnostic framework.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] CMT: Mid-Training for Efficient Learning of Consistency, Mean Flow, and Flow Map Models](cmt_mid-training_for_efficient_learning_of_consistency_mean_flow_and_flow_map_mo.md)
- [\[ICLR 2026\] UniEdit-Flow: Unleashing Inversion and Editing in the Era of Flow Models](uniedit-flow_unleashing_inversion_and_editing_in_the_era_of_flow_models.md)
- [\[ICLR 2026\] Large Scale Diffusion Distillation via Score-Regularized Continuous-Time Consistency](large_scale_diffusion_distillation_via_score-regularized_continuous-time_consist.md)
- [\[ICLR 2026\] Joint Distillation for Fast Likelihood Evaluation and Sampling in Flow-based Models](joint_distillation_for_fast_likelihood_evaluation_and_sampling_in_flow-based_mod.md)
- [\[ICLR 2026\] Generalised Flow Maps for Few-Step Generative Modelling on Riemannian Manifolds](generalised_flow_maps_for_few-step_generative_modelling_on_riemannian_manifolds.md)

</div>

<!-- RELATED:END -->

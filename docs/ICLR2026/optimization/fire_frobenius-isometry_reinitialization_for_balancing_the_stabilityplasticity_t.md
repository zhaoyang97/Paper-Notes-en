---
title: >-
  [Paper Note] FIRE: Frobenius-Isometry Reinitialization for Balancing the Stability–Plasticity Tradeoff
description: >-
  [ICLR 2026][Optimization & Theory][stability-plasticity tradeoff] FIRE reformulates the long-standing problem of "how much to reset weights" as a constrained optimization problem with a closed-form solution. By projecting weights onto an orthogonal (isometric) manifold to restore plasticity while minimizing Frobenius error relative to old weights (maintaining stability), FIRE uses Ne
tags:
  - ICLR 2026
  - Optimization & Theory
  - stability-plasticity tradeoff
  - loss of plasticity
  - reinitialization
  - dynamical isometry
  - Newton–Schulz
  - continual learning
date: 2026-05-08
content_hash: c7cb5e3b9de7f956
---
# FIRE: Frobenius-Isometry Reinitialization for Balancing the Stability–Plasticity Tradeoff

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=CfZLxT3zIZ](https://openreview.net/forum?id=CfZLxT3zIZ)  
**Code**: [https://isaac7778.github.io/fire/](https://isaac7778.github.io/fire/)  
**Area**: Continual Learning / Loss of Plasticity / Weight Reinitialization  
**Keywords**: stability-plasticity tradeoff, loss of plasticity, reinitialization, dynamical isometry, Newton–Schulz, continual learning  

## TL;DR
FIRE reformulates the long-standing problem of "how much to reset weights" as a constrained optimization problem with a closed-form solution. By projecting weights onto an orthogonal (isometric) manifold to restore plasticity while minimizing Frobenius error relative to old weights (maintaining stability), FIRE uses Newton-Schulz iterations for efficient approximation. It outperforms naive training and standard reinitialization methods across vision, language, and RL tasks with nearly zero hyperparameter tuning.

## Background & Motivation

**Background**: Neural networks trained on non-stationary data streams must balance two conflicting attributes: **stability** (retaining learned knowledge to avoid catastrophic forgetting) and **plasticity** (the ability to integrate information from new tasks). When past data remains accessible (as in foundation models or robotic agents), stability primarily serves to accelerate convergence on new tasks using old representations. In this context, the main bottleneck is the **loss of plasticity**, where the model becomes increasingly rigid and fails to fit new distributions.

**Limitations of Prior Work**: Existing methods to mitigate plasticity loss fall into two categories, each with critical flaws:
- **Regularization-based** (e.g., L2init, Parseval orthogonal constraints): These continuously pull parameters or features toward initialization or orthogonal geometry during training. Strong constraints slow down convergence and increase computational overhead, while weak constraints fail to prevent plasticity degradation.
- **Reinitialization-based** (e.g., S&P, DASH): These reset weights to an earlier checkpoint when new data arrives. While they do not interfere with the current optimization and have low overhead, they face a **tuning dilemma**: aggressive resets erase useful knowledge (collapsing stability), while conservative resets fail to restore plasticity.

**Key Challenge**: The "strength" of a reset is essentially a continuous tradeoff between stability and plasticity. Previous methods treat this as a hyperparameter requiring manual grid search, which is neither reliable nor transferable—optimal settings for vision often fail in LLMs or RL.

**Goal**: Upgrade reinitialization from "tuning a scalar strength" to "solving a principled constrained optimization problem," allowing the reset point to automatically land at the intersection of high stability and high plasticity without per-task tuning.

**Key Insight**: **Quantify stability and plasticity using two metrics, one of which is differentiable.** Stability is measured by the squared Frobenius error (SFE) between current and old weights. Plasticity is measured by the Deviation from Isometry (DfI), which is theoretically proven to correlate with loss curvature, dormant neurons, and effective rank. "Restoring plasticity while minimizing knowledge loss" is thus equivalent to **minimizing SFE under the constraint DfI=0**. This is the classic orthogonal Procrustes problem, which has a closed-form solution via polar decomposition.

## Method

### Overall Architecture

The mechanism of FIRE maps stability and plasticity to specific matrix quantities (SFE and DfI) and proves that these quantities bound representation similarity and loss curvature/rank/activity, respectively. The stability-plasticity tradeoff is translated into a clean constrained optimization: **find the closest solution to the original weights under the constraint of strict weight orthogonality (DfI=0)**. This problem is equivalent to orthogonal Procrustes, where the solution is the orthogonal factor of the polar decomposition $\tilde W^\star = W(W^\top W)^{-1/2}$. To avoid direct computation of matrix square root inverses in large networks, Newton–Schulz iterations are used to efficiently push singular values toward 1. This operation is applied only once at the transition between tasks, incurring $<1\%$ additional training time.

```mermaid
flowchart LR
    A[Weights W after training on current data] --> B[Stability metric SFE<br/>= ‖W − W̃‖²_F]
    A --> C[Plasticity metric DfI<br/>= ‖W̃ᵀW̃ − I‖²_F]
    B --> D[Constrained Optimization<br/>min SFE  s.t. DfI=0]
    C --> D
    D --> E[Closed-form solution = Orthogonal factor<br/>W̃⋆ = W(WᵀW)⁻¹/²]
    E --> F[Newton–Schulz Iterative Approximation<br/>Singular values → 1, cost <1%]
    F --> G[Continue training on new data]
```

### Key Designs

**1. SFE: Using Frobenius distance to old weights as a stability metric, proven to bound representation drift.** FIRE defines stability as the squared Frobenius error $\mathrm{SFE}(W,\tilde W)=\lVert W-\tilde W\rVert_F^2$ between current weights $W$ and reset weights $\tilde W$. Theorem 1 demonstrates that the difference in normalized covariance of output features between two $L$-layer networks is upper-bounded by SFE: $\lVert C_\Theta^\ell-C_{\tilde\Theta}^\ell\rVert_F\le \tfrac{4\lVert Z\rVert_F}{m_\ell}\sqrt{\ell}\,S^{\ell-1}\sqrt{\mathrm{SFE}}$ (assuming normalized input and 1-Lipschitz activations). This indicates that **minimizing SFE monotonically tightens the upper bound of representation drift**, making SFE a theoretically grounded proxy for stability.

**2. DfI: Translating "plasticity" into a differentiable metric of deviation from isometry, linking curvature, rank, and dormant neurons.** Traditional plasticity indicators (loss curvature, dormant neurons, effective rank) are data-dependent and non-differentiable. FIRE uses the Deviation from Isometry metric $\mathrm{DfI}(W)=\lVert W^\top W-I\rVert_F^2$, which is determined solely by weights and is differentiable. Three theorems prove that reducing DfI improves all traditional plasticity signals: Theorem 2 uses $\nu_k=1+\sqrt{\mathrm{DfI}(W_k)}$ to upper-bound the Hessian spectral norm (curvature) per layer; Theorem 3 provides a lower bound for the effective rank (srank) of features; and Theorem 4 proves that lowering DfI tightens the bounds on individual neuron activity $s_j$ ($\sqrt{\tfrac{1-\varepsilon}{1+\varepsilon}}\le s_j\le\sqrt{\tfrac{1+\varepsilon}{1-\varepsilon}}$), thereby reducing dormant neurons.

**3. Constrained Optimization + Closed-form Polar Decomposition: Replacing hyperparameter tuning with a "projection to the nearest point on the isometric manifold."** The tradeoff is formalized as: $\min_{\tilde W}\lVert W-\tilde W\rVert_F^2 \;\text{s.t.}\; \tilde W^\top\tilde W=I$. The solution is the orthogonal factor of the polar decomposition $\tilde W^\star=W(W^\top W)^{-1/2}$. This mechanism balances stability and plasticity by pushing the spectrum of $W$ toward isotropy (low DfI) while staying near the original parameter space (low SFE), avoiding both over-conservative and over-aggressive traps.

**4. Newton–Schulz Approximation: Replacing matrix square root inverse with matrix multiplications for <1% overhead.** To maintain efficiency, FIRE uses Newton–Schulz iterations: $X_0=W/\lVert W\rVert_F$, then $X_{k+1}=aX_k+bX_k(X_k^\top X_k)$ with $a=1.5, b=-0.5$. This pushes singular values toward 1. For convolutional layers, kernel-wise orthogonalization is performed. Experiments show FIRE is highly robust to the number of iterations, achieving major gains with **only 5 steps**, effectively eliminating the need for hyperparameter tuning.

## Key Experimental Results

### Main Results

| Domain | Setting / Architecture | Baselines | FIRE Findings |
|---|---|---|---|
| Continual Vision | CIFAR-10/ResNet-18, CIFAR-100/ViT-Tiny, Tiny-ImageNet/VGG-16; warm-start/continual/class-incremental | S&P, DASH, Parseval, L2init, CBP, SNR, ReDo, Muon | Achieves best performance on most benchmarks; comparable to DASH/S&P on ViT-Tiny; minimal performance drop after reset. |
| Continual LLM Pre-training | GPT-0.1B: WikiText-103 pre-training → OpenWebText+WikiText continual training | Base, full reset, S&P | **With zero tuning (fixed 5 steps)**, FIRE outperforms tuned S&P; full reset fails in this scenario. |
| Reinforcement Learning | DQN/Atari, SAC+SimBa/HumanoidBench, high Replay Ratio, mid-training reset | Full reset, S&P, Plasticity Injection | Consistently outperforms or matches S&P; outperforms full reset on Asterix. |

### Ablation Study

| Ablation | Conclusion |
|---|---|
| Newton–Schulz Iteration Steps | Highly robust; **5 steps** yield strong gains with diminishing returns for more steps. |
| Stability/Plasticity/Curvature Metrics | FIRE achieves **lowest DfI + lowest SFE** simultaneously; results in smoother loss curvature than S&P. |

### Key Findings
- **Consistency between theory and practice**: FIRE reduces DfI and SFE while smoothing curvature, validating the theorems.
- **Full reset and DASH cause sharp performance drops** in continual settings, whereas FIRE maintains stability without significant drops.
- **Node-level resets (CBP/SNR/ReDo) and Muon show poor performance**: Maintaining "trainability" alone is insufficient for generalization; applying Newton-Schulz to weight resets is more effective than applying it to gradients.

## Highlights & Insights
- **Principled mechanism**: Reformulates the "reset strength" hyperparameter as a constrained optimization problem with a closed-form solution.
- **DfI as the "Master Key"**: Unifies three disparate plasticity symptoms (curvature, rank, dormant neurons) into a single differentiable weight proxy.
- **Cross-domain zero-tuning**: The single hyperparameter (iteration steps) is extremely robust, allowing the same mechanism to work across vision, LLMs, and RL.
- **Low overhead**: The use of Newton–Schulz makes the implementation lightweight (<1% cost).

## Limitations & Future Work
- **Reliance on past data**: FIRE was not evaluated in restricted/memoryless scenarios where SFE might be a less effective stability proxy.
- **LLM Scale**: Evaluations were limited to small models (GPT-0.1B); scalability to larger parameters or different fine-tuning protocols remains to be verified.
- **Strict Isometry Constraint**: Forcing all layers to be strictly orthogonal may not be optimal for architectures that are naturally anisotropic (e.g., Transformers), as suggested by the comparative performance against DASH on ViT.

## Related Work & Insights
- **Reinitialization**: FIRE improves upon S&P and DASH by replacing heuristic reset targets with principled projections onto the isometric manifold.
- **Regularization**: Unlike Parseval or L2init which apply constraints throughout training, FIRE applies a one-time projection at data transition points, avoiding interference with convergence.
- **Plasticity Diagnosis**: Previous work identified dormant neurons (ReDo), rank collapse (Kumar), and curvature (Lyle) as signals; FIRE is the first to optimize these via the DfI proxy.
- **Newton–Schulz**: While Muon applies this to gradients, FIRE shows that application to weights during resets is significantly more effective for continual learning.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ (Principled reformulation of the stability-plasticity tradeoff).
- **Experimental Thoroughness**: ⭐⭐⭐⭐ (Broad coverage across domains; could use more numerical tables and larger LLM tests).
- **Writing Quality**: ⭐⭐⭐⭐ (Strong logical flow from theory to practice).
- **Value**: ⭐⭐⭐⭐⭐ (Practical, low-overhead, and tuning-free solution for loss of plasticity).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Activation Function Design Sustains Plasticity in Continual Learning](activation_function_design_sustains_plasticity_in_continual_learning.md)
- [\[ICLR 2026\] Seesaw: Accelerating Training by Balancing Learning Rate and Batch Size Scheduling](seesaw_accelerating_training_by_balancing_batch_size_and_learning_rate_schedulin.md)
- [\[ICLR 2026\] Generalization Below the Edge of Stability: The Role of Data Geometry](generalization_below_the_edge_of_stability_the_role_of_data_geometry.md)
- [\[ICML 2026\] Stability Analysis of Sharpness-Aware Minimization](../../ICML2026/optimization/stability_analysis_of_sharpness-aware_minimization.md)
- [\[ICML 2026\] AdaGC: Enhancing LLM Pretraining Stability via Adaptive Gradient Clipping](../../ICML2026/optimization/adagc_enhancing_llm_pretraining_stability_via_adaptive_gradient_clipping.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Adaptive Preconditioners Trigger Loss Spikes in Adam
description: >-
  [ICML 2026][Optimization & Theory][loss spike] This paper attributes loss spikes in Adam training to the lagged decoupling of the second moment preconditioner from current squared gradients and utilizes the gradient-direction curvature of the preconditioned Hessian to explain and predict spike occurrences.
tags:
  - ICML 2026
  - Optimization & Theory
  - loss spike
date: 2026-05-08
content_hash: d8c72f1974a1124d
---
# Adaptive Preconditioners Trigger Loss Spikes in Adam

**Conference**: ICML 2026  
**arXiv**: [2506.04805](https://arxiv.org/abs/2506.04805)  
**Code**: None  
**Area**: Optimization / Adam Training Stability  
**Keywords**: Adam optimizer, loss spike, preconditioned Hessian, second moment estimation, training stability  

## TL;DR
This paper attributes loss spikes in Adam training to the lagged decoupling of the second moment preconditioner from current squared gradients and utilizes the gradient-direction curvature of the preconditioned Hessian to explain and predict spike occurrences.

## Background & Motivation
**Background**: Loss spikes frequently occur during neural network training, particularly when using Adam for Transformers or large-scale models, where the loss suddenly surges before recovering. Existing explanations primarily focus on the sharpness of the loss landscape, such as the "lower-loss-as-sharper" and "Edge of Stability" phenomena, suggesting that instability is triggered when the model enters sharper regions.

**Limitations of Prior Work**: Geometric explanations of the landscape alone are insufficient for Adam's spikes. The paper demonstrates a direct counterexample: in a 1D quadratic function with constant curvature, standard GD converges smoothly under a stable learning rate, while Adam exhibits significant spikes even when the learning rate is far below the GD stability threshold. This indicates that spikes do not necessarily stem from "low-loss regions becoming sharper" but can arise from the dynamics of the optimizer's internal state variables.

**Key Challenge**: Adam's adaptive step size is intended to increase the second moment estimate $v_t$ when gradients enlarge, thereby reducing the effective step size. However, when $v_t$ is dominated by historical terms, it may continue to decay, failing to track the current squared gradient $g_t^2$ in time. Consequently, the effective preconditioned curvature is continuously amplified, pushing the training into a sustained unstable regime.

**Goal**: The authors aim to answer three questions: what quantity controls Adam's stability; why the second moment estimate fails before a spike; and whether a more accurate spike warning indicator can be constructed compared to the maximum Hessian eigenvalue.

**Key Insight**: Starting from a local quadratic approximation, the paper views Adam updates as applying spatial and momentum preconditioning to the Hessian. This perspective explicitly incorporates the "internal state of the optimizer" into the stability analysis, thereby explaining the spike mechanism observed in both 1D quadratic functions and real Transformers.

**Core Idea**: Use the gradient-direction curvature of the Adam-preconditioned Hessian, rather than the maximum eigenvalue of the original Hessian, to characterize the true trigger conditions for loss spikes.

## Method
This paper does not propose a new optimizer but establishes a mechanistic explanation, predictive indicators, and suppression suggestions for Adam's loss spikes. The logical flow is as follows: derive Adam's stability conditions using a local quadratic model, analyze how the decoupling of the second moment $v_t$ and squared gradient $g_t^2$ causes sustained stability failure, and finally validate this mechanism through multi-scale experiments.

### Overall Architecture
The input is an optimization trajectory generated using Adam. The authors observe changes in gradients, second moments, Hessians, and preconditioned Hessians along this trajectory. The analysis begins by reviewing the local stability threshold $\lambda_{\max}(H_t) < 2/\eta$ in GD as a reference, then progresses through three core mechanisms: first, folding Adam's momentum terms and adaptive denominators into the Hessian to obtain the preconditioned Hessian $\hat H_t$, changing the stability criterion to $\lambda_{\max}(\hat H_t) < 2/\eta$; second, identifying the decoupling between $v_t$ and current $g_t^2$—where gradients rise while the denominator continues to decay—which drives the eigenvalues of $\hat H_t$ higher and distinguishes spikes from ordinary Edge of Stability oscillations; third, replacing the maximum eigenvalue with the gradient-direction curvature $\lambda_{\mathrm{grad}}(\hat H_t)$ as a more precise warning metric. Finally, the mechanism is verified on 1D quadratic functions, FNNs, CNNs, and Transformers, with suppression methods such as decreasing $\beta_2$ or increasing $\epsilon$ provided.

In Adam, updates involve the first moment $m_t$ and second moment $v_t$. If momentum is temporarily ignored, Adam is approximately equivalent to multiplying the local Hessian $H_t$ by a diagonal matrix $\mathrm{diag}(1/(\sqrt{\hat v_t}+\epsilon))$. The paper further incorporates the momentum term to derive the comprehensive Adam-preconditioned Hessian:
$$\hat H_t = \frac{1}{1-\beta_1^t}\frac{1-\beta_1}{1+\beta_1}\mathrm{diag}(1/(\sqrt{\hat v_t}+\epsilon))H_t$$
When the effective curvature of this matrix exceeds $2/\eta$ for a sustained period, the training risks entering a spike.

### Key Designs

**1. Adam Preconditioned Hessian: Folding the Adaptive Denominator and Momentum into Local Curvature**

The local stability of GD is determined by the maximum eigenvalue of the original Hessian; as long as $\lambda_{\max}(H_t) < 2/\eta$, divergence does not occur. However, Adam applies diagonal scaling via the second moment denominator $\mathrm{diag}(1/(\sqrt{\hat v_t}+\epsilon))$ plus momentum corrections. Together, these form the preconditioned Hessian $\hat H_t$, changing the criterion to $\lambda_{\max}(\hat H_t) < 2/\eta$. This step explicitly embeds the "optimizer's internal state" into the stability condition: even if the original Hessian remains constant, a decreasing $\sqrt{\hat v_t}$ can amplify the preconditioned curvature beyond the stability boundary. This explains the counterexample where Adam spikes while GD converges smoothly on a constant-curvature function.

**2. Decoupling of Second Moment and Squared Gradient: Why Spikes Persist Rather Than Flashing By**

Normally, an increasing gradient should push $v_t = \beta_2 v_{1-t} + (1-\beta_2)g_t^2$ up, lowering the effective step size and creating negative feedback. The problem arises when the current gradient term is too small relative to historical terms: $v_t$ decays approximately by $\beta_2 v_{t-1}$. Thus, while the gradient is already rising, the denominator continues to shrink, pushing the eigenvalues of $\hat H_t$ higher from two sides. This lag is the watershed between a spike and an Edge of Stability oscillation—if $v_t$ responds in time, the system merely jitters near the threshold; if $v_t$ fails to keep up, instability accumulates, and the loss bulges into a peak. This directly suggests mitigation: decreasing $\beta_2$ to allow the second moment to track gradients faster.

**3. Gradient-Direction Curvature: Using Curvature in the Update Direction Instead of the Max Eigenvalue for Alerts**

Whether the loss rises in the next step depends on the second-order term in the actual update direction, not the steepest possible direction. In high-dimensional models, the direction of maximum curvature often does not align with the gradient direction; relying solely on $\lambda_{\max}$ leads to frequent false positives. The authors define the gradient-direction curvature $\lambda_{\mathrm{grad}}(H_t) = \nabla L(\theta_t)^T H_t \nabla L(\theta_t)/\|\nabla L(\theta_t)\|^2$ and replace it with the preconditioned version $\lambda_{\mathrm{grad}}(\hat H_t)$. A spike truly occurs only when this quantity crosses $2/\eta$. Because it directly corresponds to the loss change caused by the current update, it is closer to the spike onset than "omnidirectional maximum curvature," significantly reducing false alarms.

### Loss & Training
The paper uses the standard loss functions associated with each experimental task without introducing new ones. The primary training strategy involves computing Hessian-vector products along the training trajectory to estimate $\lambda_{\max}$, $\lambda_{\mathrm{grad}}$, and their preconditioned versions. Regarding suppression strategies, the authors validate two intuitive interventions: increasing Adam's $\epsilon$ to raise the lower bound of the denominator (reducing effective curvature) and decreasing $\beta_2$ to allow the second moment to respond faster to current gradients.

## Key Experimental Results

### Main Results
The paper primarily utilizes plots and trajectory analysis. The key findings across different scenarios are summarized below:

| Scene | Metrics | Key Findings (Ours) | Baseline Phenomenon | Conclusion |
|-------|---------|---------------------|----------------------|------------|
| 1D Quadratic | Loss & Effective LR | Adam spikes at small LRs when $\eta/\sqrt{\hat v_t}$ hits a threshold | GD converges smoothly in the same range | Spikes can be triggered by Adam's internal state |
| 2-layer FNN ($\sin x+\sin 4x$) | $\lambda_{\max}$ & $\lambda_{\mathrm{grad}}$ | 77 spikes; spikes only happen when $\lambda_{\mathrm{grad}}(\hat H_t) > 2/\eta$ | $\lambda_{\max}(\hat H_t)$ has 1010 crossings (high false positives) | Gradient-direction curvature is more accurate |
| 50D Function Approx FNN | Spike timing | $\lambda_{\max}(\hat H_t)$ crosses at epoch 179, but spike occurs at 184 | Original $\lambda_{\max}(H_t)$ stabilizes early | Alignment with gradient direction is necessary |
| 88-layer Transformer | Sustained predictor | 7 loss spikes correspond to sustained $\lambda_{\mathrm{grad}}(\hat H_t)$ violations | Single-step metrics are noisy due to mini-batches | Sustained violation criterion is needed for stochastic training |
| 187M LLaMA Structure | Spike frequency vs. $\beta_2$ | Multiple spikes at $\beta_2=0.999$; spikes reduced at lower $\beta_2$ | Curvature violation still observed in large models | Mechanism scales to real language models |

### Ablation Study
The "ablation" here refers to the analysis of predictive indicators and hyperparameter interventions.

| Configuration | Key Metric | Description |
|---------------|------------|-------------|
| $\lambda_{\max}(H_t)$ only | Crosses early in high-D | Max curvature direction might not participate in update; does not imply loss increase |
| $\lambda_{\max}(\hat H_t)$ | Reflects risk from Adam preconditioning | Captures effective curvature amplification but lacks directional info |
| $\lambda_{\mathrm{grad}}(\hat H_t)$ | Correlates with spikes in FNN/Adam | Directly corresponds to single-step loss growth; few false positives |
| Increase $\epsilon$ to 0.1 | Eliminates spikes in FNN | Limits denominator decay; prevents preconditioned curvature amplification |
| Decrease $\beta_2$ to 0.9 | Lower spike frequency in Transformers | Faster tracking of gradients; weakens $v_t$ and $g_t^2$ decoupling |

### Key Findings
- The most significant evidence is that "original Hessian is insufficient; preconditioned Hessian explains Adam." Quadratic functions, FNNs, CNNs, and Transformers all display the same pattern: $v_t$ decay leads to rising effective curvature.
- The maximum eigenvalue is a risk signal, but not a trigger signal. Loss only rises when the gradient direction enters the high-curvature unstable zone.
- The explanation for decreasing $\beta_2$ is clarified: it is not "magic tuning" but a way to ensure second moment estimates keep pace with gradient changes, preventing the denominator from falling while gradients rise.

## Highlights & Insights
- The paper clearly articulates that "loss spikes are a dynamic mismatch of optimizer states." It goes beyond empirical observation by incorporating Adam's second moment into the stability threshold, allowing spikes to be explained even on quadratic functions.
- Gradient-direction curvature provides a valuable diagnostic perspective. While many monitor loss, gradient norms, or max Hessian eigenvalues, this work reminds us that curvature in the *direction of update* determines if the loss will surge.
- For large model training, the insight is that a lower $\beta_2$ may not just affect convergence speed but also reduce loss spike risks. This provides a mechanistic explanation for why practices like $\beta_2=0.95$ are used in LLM training.

## Limitations & Future Work
- The most rigorous theoretical portions are built on 1D quadratic functions and local quadratic approximations; conclusions in high-dimensional non-convex networks rely heavily on empirical validation.
- Computing metrics at the Hessian-vector product level remains expensive for models exceeding 200M parameters, making it difficult to use as a routine monitoring tool. Cheaper proxies are needed.
- Spikes are not always detrimental; the appendix discusses neutral, benign, malignant, and catastrophic types. Distinguishing between spikes that should be suppressed and those that might facilitate basin transitions remains an open question.

## Related Work & Insights
- **vs Edge of Stability**: EoS explains the non-monotonic decrease of the max Hessian eigenvalue near $2/\eta$ in GD; this work generalizes that framework to Adam's preconditioned Hessian and emphasizes sustained violations.
- **vs Lower-Loss-as-Sharper**: LLAS explains spikes via landscape geometry; this paper notes that even if landscape curvature is constant, Adam's $v_t$ can change the effective curvature, making optimizer state an independent mechanism.
- **vs Adam Convergence Analysis**: Traditional theory focuses on convergence/non-convergence; this work acts as a training dynamics diagnostic, explaining the onset, duration, and recovery phases of spikes.
- **Practical Implications**: Monitoring second-moment decay, gradient-direction curvature, or their low-cost proxies may detect instability earlier than loss monitoring. Adjusting $\beta_2$ or $\epsilon$ serves as a theoretically grounded stabilization method.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Excellent perspective explaining loss spikes via internal optimizer dynamics.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers everything from quadratic functions to 187M Transformers, though metrics are mostly graphical rather than tabular for large scales.
- Writing Quality: ⭐⭐⭐⭐☆ Complete logical chain, though dense with formulas and plots requiring optimization background.
- Value: ⭐⭐⭐⭐⭐ Direct implications for large model stability and Adam hyperparameter selection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] The Implicit Bias of Adam and Muon on Smooth Homogeneous Neural Networks](the_implicit_bias_of_adam_and_muon_on_smooth_homogeneous_neural_networks.md)
- [\[ICML 2026\] Towards Understanding Adam Convergence on Highly Degenerate Polynomials](towards_understanding_adam_convergence_on_highly_degenerate_polynomials.md)
- [\[ICML 2026\] Multi-Objective Bayesian Optimization via Adaptive ε-Constraints Decomposition](multi-objective_bayesian_optimization_via_adaptive_varepsilon-constraints_decomp.md)
- [\[NeurIPS 2025\] Abstain Mask Retain Core: Time Series Prediction by Adaptive Masking Loss with Representation Consistency](../../NeurIPS2025/optimization/abstain_mask_retain_core_time_series_prediction_by_adaptive.md)
- [\[CVPR 2026\] Globscope: Toward a Global View of the Loss Landscape](../../CVPR2026/optimization/globscope_toward_a_global_view_of_the_loss_landscape.md)

</div>

<!-- RELATED:END -->

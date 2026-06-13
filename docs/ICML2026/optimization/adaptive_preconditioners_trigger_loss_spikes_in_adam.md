---
title: >-
  [Paper Note] Adaptive Preconditioners Trigger Loss Spikes in Adam
description: >-
  [ICML 2026][Optimization][Adam optimizer] This paper attributes loss spikes in Adam training to the lagged decoupling between the second-moment preconditioner and the current squared gradient…
tags:
  - "ICML 2026"
  - "Optimization"
  - "Adam optimizer"
  - "loss spike"
  - "preconditioned Hessian"
  - "second-order moment estimation"
  - "training stability"
date: 2026-05-08
content_hash: fc3b9ac55b760d85
---

# Adaptive Preconditioners Trigger Loss Spikes in Adam

**Conference**: ICML 2026  
**arXiv**: [2506.04805](https://arxiv.org/abs/2506.04805)  
**Code**: None  
**Area**: Optimization / Adam training stability  
**Keywords**: Adam optimizer, loss spike, preconditioned Hessian, second-order moment estimation, training stability  

## TL;DR
This paper attributes loss spikes in Adam training to the lagged decoupling between the second-moment preconditioner and the current squared gradient, while explaining and predicting the occurrence of spikes using the gradient-direction curvature of the preconditioned Hessian.

## Background & Motivation
**Background**: Loss spikes frequently occur during neural network training, especially when using Adam for Transformers or large-scale models, where the loss abruptly surges and then recovers. Existing explanations primarily focus on the sharpness of the loss landscape, such as "lower-loss-as-sharper" and the "Edge of Stability" phenomenon, suggesting that instability is triggered when the model enters sharper regions.

**Limitations of Prior Work**: Geometric landscape explanations alone are insufficient for Adam's spikes. The paper presents a direct counterexample: in a 1D quadratic scenario with constant curvature, standard GD converges smoothly under a stable learning rate, whereas Adam exhibits significant spikes even when the learning rate is far below the GD stability threshold. This indicates that spikes do not necessarily originate from "low-loss regions becoming sharper" but may stem from the dynamics of the optimizer’s internal state variables.

**Key Challenge**: Adam's adaptive step size is intended to increase the second-order moment estimate $v_t$ when gradients increase, thereby reducing the effective step size. However, when $v_t$ is dominated by historical terms, it may continue to decay and fail to track the current squared gradient $g_t^2$ in time. Consequently, the effective preconditioned curvature is continuously amplified, pushing the training into a sustained unstable regime.

**Goal**: The authors aim to answer three questions: what quantity controls Adam's stability; why the second-order moment estimate fails before a spike; and whether a more precise spike early-warning metric than the maximum Hessian eigenvalue can be constructed.

**Key Insight**: Starting from a local quadratic approximation, the paper views Adam's updates as applying both spatial and momentum preconditioning to the Hessian. This perspective explicitly incorporates "internal optimizer states" into the stability analysis, thereby explaining the shared spike mechanism across 1D quadratic functions and real-world Transformers.

**Core Idea**: To characterize the true trigger condition of loss spikes using the gradient-direction curvature of the Adam preconditioned Hessian rather than the maximum eigenvalue of the original Hessian.

## Method
This paper does not propose a new optimizer but establishes a mechanistic explanation, predictive metrics, and mitigation suggestions for Adam's loss spikes. The logic is: first, derive Adam's stability conditions using a local quadratic model; second, analyze how the decoupling of the second moment $v_t$ and the squared gradient $g_t^2$ causes sustained violation of stability conditions; finally, validate this mechanism through multi-scale experiments.

### Overall Architecture
The input is an optimization trajectory trained with Adam. The authors observe the evolution of gradients, second moments, Hessians, and preconditioned Hessians along this trajectory. The analysis proceeds in four steps: reviewing the local stability threshold for GD; formulating Adam's adaptive terms as a preconditioned Hessian; proposing gradient-direction curvature as a more precise criterion for spikes; and validating this criterion on simple functions, FNNs, CNNs, and Transformers.

In Adam, updates involve the first-order moment $m_t$ and second-order moment $v_t$. Temporarily ignoring momentum, Adam is approximately equivalent to multiplying the local Hessian $H_t$ by a diagonal matrix $\mathrm{diag}(1/(\sqrt{\hat v_t}+\epsilon))$. The paper further incorporates the momentum term to derive the comprehensive Adam preconditioned Hessian: $\hat H_t = \frac{1}{1-\beta_1^t}\frac{1-\beta_1}{1+\beta_1}\mathrm{diag}(1/(\sqrt{\hat v_t}+\epsilon))H_t$. When the effective curvature of this matrix exceeds $2/\eta$ for a sustained period, the training is at risk of entering a spike.

### Key Designs
1. **Adam Preconditioned Hessian Stability Viewpoint**:
    - **Function**: Converts Adam's adaptive denominator and momentum into curvature scaling terms within the local stability framework.
    - **Mechanism**: While GD's local stability is controlled by $\lambda_{\max}(H_t)<2/\eta$, Adam should be governed by $\lambda_{\max}(\hat H_t)<2/\eta$, where $\hat H_t$ includes the second-moment denominator and momentum scaling. Thus, even if the raw Hessian remains constant, the preconditioned curvature increases if $\sqrt{\hat v_t}$ decreases.
    - **Design Motivation**: This explains why even 1D quadratic functions with constant curvature exhibit spikes under Adam. The root cause is not the geometry suddenly becoming sharper, but Adam changing the coordinate scale, pushing the effective curvature past the stability boundary.

2. **Decoupling Mechanism of Second Moment and Squared Gradient**:
    - **Function**: Explains why spikes are not transient oscillations but develop into sustained loss surges.
    - **Mechanism**: Normally, an increased gradient should increase $v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^2$, reducing the effective step size. However, when the current gradient term is too small relative to historical terms, $v_t$ decays autonomously approximately at rate $\beta_2 v_{t-1}$. In this phase, the denominator continues to shrink even as gradients begin to grow, causing eigenvalues of $\hat H_t$ to rise further.
    - **Design Motivation**: This mechanism distinguishes spikes from regular Edge of Stability oscillations. If $v_t$ responds quickly, the system oscillates near the threshold; if $v_t$ lags, stability violations persist, forming a loss peak.

3. **Gradient-Direction Curvature Predictive Metric**:
    - **Function**: Reduces false alarms when predicting spikes compared to using only the maximum eigenvalue.
    - **Mechanism**: Whether the loss increases in the next step depends on the second-order term in the update direction, not the maximum curvature across all directions. The paper defines $\lambda_{\mathrm{grad}}(H_t)=\nabla L(\theta_t)^T H_t \nabla L(\theta_t)/\|\nabla L(\theta_t)\|^2$ and replaces it with $\lambda_{\mathrm{grad}}(\hat H_t)$ for Adam. A spike truly occurs only when this gradient-direction curvature also exceeds $2/\eta$.
    - **Design Motivation**: In high-dimensional models, the direction of maximum curvature may not align with the gradient direction. Monitoring $\lambda_{\max}$ alone might trigger premature warnings. Gradient-direction curvature directly corresponds to the loss change caused by the current update, making it closer to the spike onset.

### Loss & Training
The paper uses the standard losses associated with each task; no new loss functions are introduced. The primary experimental strategy involves calculating Hessian-vector products along the training trajectory to estimate $\lambda_{\max}$, $\lambda_{\mathrm{grad}}$, and their preconditioned versions. Regarding mitigation, the authors validate two intuitive interventions: increasing Adam's $\epsilon$ to raise the denominator floor and reduce effective curvature; and decreasing $\beta_2$ to allow the second moment to respond faster to current gradients, alleviating decoupling at the source.

## Key Experimental Results

### Main Results
The paper focuses on trajectory analysis and visualization rather than standard "Dataset-Metric-SOTA" tables. Key results across experimental settings are summarized below:

| Scenario | Metrics | Key Experimental Results (Ours) | Comparison/Baseline | Conclusion |
|--------|------|------|----------|------|
| 1D Quadratic Function | Loss and effective LR | Adam shows spikes even at small LR; triggered when $\eta/\sqrt{\hat v_t}$ nears a threshold | GD converges smoothly in the same stable region | Spikes can be triggered by Adam’s internal state |
| 2-layer FNN fitting $\sin x+\sin 4x$ | $\lambda_{\max}$ and $\lambda_{\mathrm{grad}}$ | 77 spikes observed; occur only when $\lambda_{\mathrm{grad}}(\hat H_t)>2/\eta$ | $\lambda_{\max}(\hat H_t)$ had 1010 boundary crossings (more false positives) | Gradient-direction curvature is more precise |
| 50D Function Approx FNN | Spike timing | $\lambda_{\max}(\hat H_t)$ crosses boundary at epoch 179, but loss spikes at epoch 184 | Raw $\lambda_{\max}(H_t)$ stabilizes quickly | Curvature aligned with gradient is necessary |
| 88-layer Transformer | Sustained predictor | All 7 loss spikes correspond to sustained $\lambda_{\mathrm{grad}}(\hat H_t)$ violations | Raw single-step metrics are noisy due to mini-batches | Sustained boundary violation is the criterion in stochastic training |
| 187M LLaMA Structure | Spike freq vs. $\beta_2$ | Multiple spikes at default $\beta_2=0.999$; spikes reduced by lowering $\beta_2$ | Gradient-direction curvature violations still observed in large models | Mechanism scales to real LLM training |

### Ablation Study
"Ablation" here refers to the analysis of predictive metrics and hyperparameter interventions.

| Configuration | Key Metrics | Description |
|------|---------|------|
| Monitoring $\lambda_{\max}(H_t)$ | Premature boundary crossing in high-dim | Maximum curvature direction may not participate in the update; does not directly imply loss increase |
| Monitoring $\lambda_{\max}(\hat H_t)$ | Reflects risks from Adam preconditioning | Captures effective curvature amplification from $v_t$ decay, but lacks directional information |
| Monitoring $\lambda_{\mathrm{grad}}(\hat H_t)$ | Spikes occur exactly when this exceeds $2/\eta$ | Directly corresponds to single-step loss growth conditions; fewer false positives |
| Increasing $\epsilon$ to 0.1 | Eliminates spikes in FNN experiments | Raises the denominator lower bound, preventing preconditioned curvature from exploding |
| Lowering $\beta_2$ to 0.9 | Reduced spike frequency in LLaMA | Second moment tracks current gradients faster, weakening $v_t$ and $g_t^2$ decoupling |

### Key Findings
- The most critical experimental evidence is that "raw Hessian is insufficient, preconditioned Hessian explains Adam." Quadratic functions, FNNs, CNNs, and Transformers all exhibit the same pattern where $v_t$ decay leads to rising effective curvature.
- The maximum eigenvalue is a risk signal, but not a trigger signal. Only when the gradient direction enters the high-curvature unstable zone does the loss truly surge.
- The explanation for lowering $\beta_2$ is clarified: it is not "magic tuning," but a way to ensure the second-moment estimate keeps pace with gradient changes, preventing the denominator from continuing to fall while gradients are rising.

## Highlights & Insights
- The paper clearly articulates that "loss spikes are dynamic mismatches of the optimizer state." It moves beyond empirical observation by incorporating Adam’s second moment into the stability threshold, making spikes explainable even on quadratic functions.
- Gradient-direction curvature serves as a valuable diagnostic perspective. While many monitors look at loss, gradient norms, or max Hessian eigenvalues, this paper highlights that the curvature along the update direction is what determines if the next step will surge.
- Practical insight for LLM training: a lower $\beta_2$ may not just affect convergence speed but also reduce loss spike risks. This provides a mechanistic explanation for why some LLM practices use $\beta_2=0.95$ or lower.

## Limitations & Future Work
- The most rigorous theoretical parts are built on 1D quadratic functions and local quadratic approximations; conclusions in high-dimensional non-convex networks rely heavily on experimental validation. The coupling between the preconditioner, real loss landscape, and stochastic mini-batch noise may be more complex.
- Metric computation at the Hessian-vector product level remains expensive for models exceeding 200M parameters, making it difficult to use as a routine training monitor. Cheaper proxy metrics are needed.
- Spikes are not always detrimental; the appendix discusses neutral, benign, malignant, and catastrophic types. Distinguishing between "spikes to suppress" and "spikes that favor basin transitions" remains an open question.

## Related Work & Insights
- **vs. Edge of Stability**: EoS explains non-monotonic descent in GD after the max Hessian eigenvalue nears $2/\eta$. This work generalizes that framework to Adam’s preconditioned Hessian and emphasizes that sustained violations form spikes.
- **vs. lower-loss-as-sharper**: LLAS explains spikes from the shape of the loss landscape. This paper points out that even if landscape curvature is constant, Adam's $v_t$ can change the effective curvature, identifying the optimizer state as an independent mechanism.
- **vs. Adam Convergence Analysis**: Traditional Adam theory focuses on convergence/non-convergence. This work functions as training dynamics diagnostics, explaining the phases of spike occurrence, duration, and recovery.
- **Practical Implications**: Monitoring second-moment decay, gradient-direction curvature, or their low-cost proxies may detect instability earlier than monitoring loss alone. Lowering $\beta_2$ or increasing $\epsilon$ acts as a theoretically-grounded stabilization method.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Insightfully explains loss spikes through internal preconditioner dynamics.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers range from quadratic functions to 187M Transformers, though numerical results are largely illustrative.
- Writing Quality: ⭐⭐⭐⭐☆ Complete mechanistic chain, though dense with formulas and figures requiring optimization background.
- Value: ⭐⭐⭐⭐⭐ Directly informs large model training stability and Adam hyperparameter selection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] The Implicit Bias of Adam and Muon on Smooth Homogeneous Neural Networks](the_implicit_bias_of_adam_and_muon_on_smooth_homogeneous_neural_networks.md)
- [\[ICML 2026\] Towards Understanding Adam Convergence on Highly Degenerate Polynomials](towards_understanding_adam_convergence_on_highly_degenerate_polynomials.md)
- [\[NeurIPS 2025\] Abstain Mask Retain Core: Time Series Prediction by Adaptive Masking Loss with Representation Consistency](../../NeurIPS2025/optimization/abstain_mask_retain_core_time_series_prediction_by_adaptive.md)
- [\[ICML 2026\] Multi-Objective Bayesian Optimization via Adaptive $\varepsilon$-Constraints Decomposition](multi-objective_bayesian_optimization_via_adaptive_varepsilon-constraints_decomp.md)
- [\[ICML 2026\] Taming the Loss Landscape of PINNs with Noisy Feynman-Kac Supervision: Operator Preconditioning and Non-Asymptotic Error Bounds](taming_the_loss_landscape_of_pinns_with_noisy_feynman-kac_supervision_operator_p.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICML 2026\] The Implicit Bias of Adam and Muon on Smooth Homogeneous Neural Networks](the_implicit_bias_of_adam_and_muon_on_smooth_homogeneous_neural_networks.md)
- [\[ACL 2025\] Better Embeddings with Coupled Adam](../../ACL2025/others/better_embeddings_with_coupled_adam.md)
- [\[ICML 2026\] Adaptive Multi-Round Allocation with Stochastic Arrivals](adaptive_multi-round_allocation_with_stochastic_arrivals.md)
- [\[ICML 2026\] Provably Data-driven Multiple Hyper-parameter Tuning with Structured Loss Function](provably_data-driven_multiple_hyper-parameter_tuning_with_structured_loss_functi.md)
- [\[ICML 2026\] Expectation Consistency Loss: Rethink Confidence Calibration under Covariate Shift](expectation_consistency_loss_rethink_confidence_calibration_under_covariate_shif.md)

</div>

<!-- RELATED:END -->

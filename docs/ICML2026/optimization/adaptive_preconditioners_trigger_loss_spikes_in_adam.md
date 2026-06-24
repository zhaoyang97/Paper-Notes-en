---
title: >-
  [Paper Note] Adaptive Preconditioners Trigger Loss Spikes in Adam
description: >-
  [ICML 2026][Optimization][Adam Optimizer] This paper attributes loss spikes in Adam training to the lag-induced decoupling between the second-moment preconditioner and the current squared gradients, and explains as well as predicts spike occurrences using the curvature of the preconditioned Hessian in the gradient direction.
tags:
  - "ICML 2026"
  - "Optimization"
  - "Adam Optimizer"
  - "loss spike"
  - "preconditioned Hessian"
  - "second-moment estimation"
  - "training stability"
date: 2026-05-08
content_hash: 2c85a1499dc7a331
---

# Adaptive Preconditioners Trigger Loss Spikes in Adam

**Conference**: ICML 2026  
**arXiv**: [2506.04805](https://arxiv.org/abs/2506.04805)  
**Code**: None  
**Area**: Optimization / Adam Training Stability  
**Keywords**: Adam Optimizer, loss spike, preconditioned Hessian, second-moment estimation, training stability  

## TL;DR
This paper attributes loss spikes in Adam training to the lag-induced decoupling between the second-moment preconditioner and the current squared gradients, and explains as well as predicts spike occurrences using the curvature of the preconditioned Hessian in the gradient direction.

## Background & Motivation
**Background**: Loss spikes frequently occur during neural network training, especially when using Adam to train Transformers or large models, where the loss suddenly surges and then recovers. Existing explanations primarily focus on the sharpness of the loss landscape, such as the "lower-loss-as-sharper" and "Edge of Stability" phenomena, suggesting that instability is triggered when the model enters sharper regions.

**Limitations of Prior Work**: Geometric explanations of the landscape alone are insufficient for Adam's spikes. The paper presents a direct counterexample: in a 1D quadratic function scenario with constant curvature, standard GD converges smoothly under a stable learning rate, while Adam exhibits significant spikes even when the learning rate is far below the GD stability threshold. This indicates that spikes do not necessarily arise from "low-loss regions becoming sharper" but can stem from the dynamics of the optimizer's internal state variables.

**Key Challenge**: Adam's adaptive step size is intended to increase the second-moment estimate $v_t$ when gradients increase, thereby reducing the effective step size. However, when $v_t$ is dominated by historical terms, it may continue to decay, failing to track the current squared gradient $g_t^2$ in time. Consequently, the effective curvature after preconditioning is continuously amplified, leading the training into a sustained unstable interval.

**Goal**: The authors aim to answer three questions: what quantity controls Adam's stability; why the second-moment estimate fails before a spike; and whether a more accurate spike warning indicator can be constructed than the maximum Hessian eigenvalue.

**Key Insight**: Starting from a local quadratic approximation, the paper views Adam's updates as applying spatial and momentum preconditioning to the Hessian. This perspective explicitly incorporates the "internal state of the optimizer" into stability analysis, thereby explaining the spike mechanism observed in both 1D quadratic functions and real-world Transformers.

**Core Idea**: Use the curvature of the Adam-preconditioned Hessian in the gradient direction, rather than the maximum eigenvalue of the original Hessian, to characterize the true trigger conditions of loss spikes.

## Method
This paper does not propose a new optimizer but establishes a mechanistic explanation, predictive indicators, and suppression suggestions for Adam's loss spikes. The overall logic is: first derive Adam's stability conditions using a local quadratic model, then analyze how the decoupling of the second moment $v_t$ and squared gradient $g_t^2$ causes sustained failure of stability conditions, and finally validate this mechanism through multi-scale experiments.

### Overall Architecture

```mermaid
graph TD
    A[Adam Training Trajectory] --> B{Stability Criterion}
    B --> C[Preconditioned Hessian: $\hat H_t$]
    B --> D[Curvature in Gradient Direction: $\lambda_{\mathrm{grad}}$]
    C & D --> E[Decoupling Analysis: $v_t$ vs $g_t^2$]
    E --> F[Spike Prediction & Mitigation]
    F --> G[Lower $\beta_2$ / Higher $\epsilon$]
```

The input is an optimization trajectory obtained using Adam. The authors observe changes in gradients, second moments, Hessians, and preconditioned Hessians along this trajectory. The analysis first revisits the local stability threshold $\lambda_{\max}(H_t) < 2/\eta$ in GD as a reference, then progresses through three core mechanisms: first, incorporating Adam's momentum terms and adaptive denominators into the Hessian to obtain the preconditioned Hessian $\hat H_t$, changing the stability criterion to $\lambda_{\max}(\hat H_t) < 2/\eta$; second, the decoupling of the second moment $v_t$ and the current squared gradient $g_t^2$—where the gradient rises while the denominator continues to decay—forcing the eigenvalues of $\hat H_t$ higher, which marks the watershed between a spike and ordinary Edge of Stability oscillations; third, replacing the maximum eigenvalue with the gradient direction curvature $\lambda_{\mathrm{grad}}(\hat H_t)$ as a more accurate warning sign. Finally, the mechanism is validated on 1D quadratic functions, FNNs, CNNs, and Transformers, with theoretically grounded suppression methods such as reducing $\beta_2$ or increasing $\epsilon$.

In Adam, updates involve the first moment $m_t$ and second moment $v_t$. Temporarily ignoring momentum, Adam is approximately equivalent to multiplying the local Hessian $H_t$ by a diagonal matrix $\mathrm{diag}(1/(\sqrt{\hat v_t}+\epsilon))$. The paper further incorporates the momentum term to derive the comprehensive Adam-preconditioned Hessian: $$\hat H_t = \frac{1}{1-\beta_1^t}\frac{1-\beta_1}{1+\beta_1}\mathrm{diag}(1/(\sqrt{\hat v_t}+\epsilon))H_t$$ When the effective curvature of this matrix exceeds $2/\eta$ for a sustained period, the training is at risk of a spike.

### Key Designs

**1. Adam Preconditioned Hessian: Folding denominators and momentum into local curvature**  
Standard GD stability is determined by the maximum eigenvalue of the original Hessian; as long as $\lambda_{\max}(H_t) < 2/\eta$, it will not diverge. However, Adam applies diagonal scaling via the second-moment denominator $\mathrm{diag}(1/(\sqrt{\hat v_t}+\epsilon))$ combined with momentum correction, forming the preconditioned Hessian $\hat H_t$. Thus, the criterion becomes $\lambda_{\max}(\hat H_t) < 2/\eta$. This step explicitly embeds the "internal state of the optimizer" into the stability condition: even if the original Hessian is static, as $\sqrt{\hat v_t}$ decreases, the preconditioned curvature is amplified and crosses the stability boundary. This explains why Adam spikes on a 1D quadratic function with constant curvature while GD converges smoothly.

**2. Decoupling of Second Moment and Squared Gradient: Why spikes persist**  
Normally, an increasing gradient should push up $v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$, lowering the effective step size and forming negative feedback. The problem occurs when the current gradient term is too small relative to the historical term: $v_t$ decays approximately as $\beta_2 v_{t-1}$. Consequently, while the gradient is already rising, the denominator is still shrinking, doubling the push on the eigenvalues of $\hat H_t$. This lag is the defining difference between a spike and ordinary Edge of Stability oscillations—if $v_t$ responds in time, the system merely jitters near the threshold; if $v_t$ fails to keep up, stability violations accumulate, and the loss bulges into a peak. This directly suggests mitigation: reducing $\beta_2$ to make the second moment track gradients faster.

**3. Gradient Direction Curvature: Warning via curvature in the update direction**  
Whether the loss rises depends on the second-order term in the actual update direction, not the steepest possible direction. In high-dimensional models, the direction of maximum curvature often does not align with the gradient direction; viewing only $\lambda_{\max}$ leads to frequent false alarms. The authors define the gradient direction curvature $\lambda_{\mathrm{grad}}(H_t) = \nabla L(\theta_t)^T H_t \nabla L(\theta_t) / \|\nabla L(\theta_t)\|^2$ and use its preconditioned version $\lambda_{\mathrm{grad}}(\hat H_t)$ in Adam. A spike only truly occurs when this specific value crosses $2/\eta$. Because it directly corresponds to the loss change of the current update, it is much closer to the spike onset than the "all-direction maximum curvature," significantly reducing false positives.

### Loss & Training
The paper uses the standard loss functions associated with each experimental task and does not introduce new ones. The primary experimental strategy involves calculating Hessian-vector products along training trajectories to estimate $\lambda_{\max}$, $\lambda_{\mathrm{grad}}$, and their preconditioned versions. Regarding suppression strategies, the authors validate two intuitive interventions: increasing Adam's $\epsilon$ to raise the denominator's lower bound (reducing effective curvature) and decreasing $\beta_2$ to allow the second moment to respond faster to current gradients (mitigating decoupling).

## Key Experimental Results

### Main Results
The paper focuses on figures and trajectory analysis rather than standard "dataset-metric-SOTA" tables. Key results across scenarios are summarized below:

| Scenario | Metric Observed | Key Result (Ours) | Comparison/Baseline | Conclusion |
|----------|-----------------|-------------------|---------------------|------------|
| 1D Quadratic | Loss & Effective LR | Adam spikes at small LR; triggered when $\eta/\sqrt{\hat v_t}$ hits threshold | GD converges smoothly in the same range | Spikes can be triggered by internal state |
| 2-layer FNN ($\sin x + \sin 4x$) | $\lambda_{\max}$ vs $\lambda_{\mathrm{grad}}$ | 77 spikes for Adam; spikes only occur when $\lambda_{\mathrm{grad}}(\hat H_t) > 2/\eta$ | $\lambda_{\max}(\hat H_t)$ has 1010 boundary crossings (high false alarms) | Gradient direction curvature is more precise |
| 50D Function Approx FNN | Spike timing | $\lambda_{\max}(\hat H_t)$ crosses at epoch 179, but loss spikes at epoch 184 | Original $\lambda_{\max}(H_t)$ stabilizes quickly | Curvature alignment with gradient matters |
| 88-layer Transformer | Sustained predictor | 7 loss spikes correspond directly to sustained $\lambda_{\mathrm{grad}}(\hat H_t)$ crossings | Single-step metrics are noisy due to mini-batches | Need sustained criteria in stochastic training |
| 187M LLaMA Transformer | Spike freq vs $\beta_2$ | Multiple spikes at default $\beta_2=0.999$; spikes reduce at lower $\beta_2$ | Curvature crossings still observable in large models | Mechanism scales to real language models |

### Ablation Study
The "ablations" here refer to the analysis of predictive indicators and hyperparameter interventions.

| Configuration | Key Metric | Description |
|---------------|------------|-------------|
| $\lambda_{\max}(H_t)$ only | Crosses threshold early in high-D | Max curvature direction might not participate in update; doesn't guarantee loss increase |
| $\lambda_{\max}(\hat H_t)$ | Reflects risk from preconditioning | Captures effective curvature amplification from $v_t$ decay, but still lacks directionality |
| $\lambda_{\mathrm{grad}}(\hat H_t)$ | Spikes occur only when this $> 2/\eta$ | Directly corresponds to single-step loss growth; fewer false positives |
| Increase $\epsilon$ to 0.1 | Eliminates spikes in FNN | Blocks the denominator from shrinking too far, capping effective curvature |
| Decrease $\beta_2$ to 0.9 | Spike frequency drops in Transformers | $v_t$ tracks $g_t^2$ faster, reducing the lag-induced decoupling |

### Key Findings
- The most critical experimental evidence is that "original Hessian is insufficient; preconditioned Hessian explains Adam's spikes." Quadratic functions, FNNs, CNNs, and Transformers all follow the same pattern: $v_t$ decay leads to rising effective curvature.
- The maximum eigenvalue is a risk signal, not a trigger. A spike only occurs when the gradient direction itself enters the high-curvature unstable zone.
- The explanation for lowering $\beta_2$ is clear: it is not "magic tuning" but a way to ensure second-moment estimates keep pace with gradient changes, preventing the denominator from continuing to fall while gradients rise.

## Highlights & Insights
- The paper clearly articulates that "loss spikes are a dynamic mismatch of optimizer states." It goes beyond empirical observation by incorporating Adam's second moment into the stability threshold, allowing spikes to be explained even on quadratic functions.
- Gradient direction curvature provides a highly useful diagnostic perspective. Many monitoring tools focus on loss, gradient norm, or max Hessian eigenvalues, but this work reminds us that curvature in the update direction is what fundamentally determines if the next step increases loss.
- For large model training, the practical takeaway is that lower $\beta_2$ does more than just affect convergence speed—it actively mitigates the risk of loss spikes. This provides a mechanistic justification for using $\beta_2=0.95$ or lower in LLM training practices.

## Limitations & Future Work
- The most rigorous theoretical parts are founded on 1D quadratic functions and local quadratic approximations; conclusions in high-dimensional non-convex networks rely primarily on experimental validation. There may be more complex couplings between preconditioners, the real landscape, and mini-batch noise.
- Computing metrics at the Hessian-vector product level remains expensive for models exceeding 200M parameters, making it difficult to use as a routine training monitor. Cheaper proxy metrics are needed.
- Spikes are not always detrimental; the appendix discusses neutral, benign, malignant, and catastrophic types. Distinguishing between "spikes to be suppressed" and "spikes facilitating basin transitions" remains an open question.

## Related Work & Insights
- **vs Edge of Stability**: EoS explains the non-monotonic decrease when the max Hessian eigenvalue approaches $2/\eta$ in GD; this paper extends that framework to Adam's preconditioned Hessian and emphasizes that *sustained* crossings create spikes.
- **vs lower-loss-as-sharper**: LLAS explains spikes via landscape geometry; this paper shows that even with constant curvature, Adam's $v_t$ can alter effective curvature, making the optimizer state an independent mechanism.
- **vs Adam Convergence Analysis**: Traditional theory focuses on convergence/non-convergence; this work acts as training dynamics diagnostics, explaining the phases of spike onset, duration, and recovery.
- **Insights for Training Practice**: Monitoring second-moment decay, gradient direction curvature, or their proxies may detect instability earlier than monitoring loss alone; adjusting $\beta_2$ or $\epsilon$ serves as a theoretically grounded stabilization technique.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Clear perspective explaining spikes via internal preconditioner dynamics.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers 1D to 187M Transformers, though numerical results are mostly graphical rather than tabular for large scales.
- Writing Quality: ⭐⭐⭐⭐☆ Complete logic chain, though dense math/figures require an optimization background for quick digestion.
- Value: ⭐⭐⭐⭐⭐ Direct implications for large model stability and Adam hyperparameter selection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] The Implicit Bias of Adam and Muon on Smooth Homogeneous Neural Networks](the_implicit_bias_of_adam_and_muon_on_smooth_homogeneous_neural_networks.md)
- [\[ICML 2026\] AdaGC: Enhancing LLM Pretraining Stability via Adaptive Gradient Clipping](adagc_enhancing_llm_pretraining_stability_via_adaptive_gradient_clipping.md)
- [\[ICML 2026\] Towards Understanding Adam Convergence on Highly Degenerate Polynomials](towards_understanding_adam_convergence_on_highly_degenerate_polynomials.md)
- [\[ICML 2026\] CLoVE: Personalized Federated Learning through Clustering of Loss Vector Embeddings](clove_personalized_federated_learning_through_clustering_of_loss_vector_embeddin.md)
- [\[NeurIPS 2025\] Abstain Mask Retain Core: Time Series Prediction by Adaptive Masking Loss with Representation Consistency](../../NeurIPS2025/optimization/abstain_mask_retain_core_time_series_prediction_by_adaptive.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICML 2026\] The Implicit Bias of Adam and Muon on Smooth Homogeneous Neural Networks](the_implicit_bias_of_adam_and_muon_on_smooth_homogeneous_neural_networks.md)
- [\[ICML 2026\] Towards Understanding Adam Convergence on Highly Degenerate Polynomials](towards_understanding_adam_convergence_on_highly_degenerate_polynomials.md)
- [\[NeurIPS 2025\] Abstain Mask Retain Core: Time Series Prediction by Adaptive Masking Loss with Representation Consistency](../../NeurIPS2025/optimization/abstain_mask_retain_core_time_series_prediction_by_adaptive.md)
- [\[ICML 2026\] Multi-Objective Bayesian Optimization via Adaptive ε-Constraints Decomposition](multi-objective_bayesian_optimization_via_adaptive_varepsilon-constraints_decomp.md)
- [\[CVPR 2026\] Globscope: Toward a Global View of the Loss Landscape](../../CVPR2026/optimization/globscope_toward_a_global_view_of_the_loss_landscape.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] From Sequential to Recursive: Enhancing Decision-Focused Learning with Bidirectional Feedback
description: >-
  [AAAI 2026][Decision-Focused Learning] This paper proposes the first Recursive Decision-Focused Learning (R-DFL) framework, which introduces a bidirectional feedback loop between the prediction module and the optimizatio…
tags:
  - "AAAI 2026"
  - "Decision-Focused Learning"
  - "Predict-then-Optimize"
  - "Recursive Learning"
  - "Implicit Differentiation"
  - "Bidirectional Feedback"
date: 2026-05-08
content_hash: d2efa7d33fd2ad50
---

# From Sequential to Recursive: Enhancing Decision-Focused Learning with Bidirectional Feedback

**Conference**: AAAI 2026
**arXiv**: [2511.08035](https://arxiv.org/abs/2511.08035)  
**Code**: None  
**Area**: Others
**Keywords**: Decision-Focused Learning, Predict-then-Optimize, Recursive Learning, Implicit Differentiation, Bidirectional Feedback

## TL;DR

This paper proposes the first Recursive Decision-Focused Learning (R-DFL) framework, which introduces a bidirectional feedback loop between the prediction module and the optimization module, breaking the unidirectional information flow of conventional sequential DFL. Two gradient propagation methods—explicit unrolling and implicit differentiation—are designed, achieving significant improvements in final decision quality on the newsvendor problem and bipartite matching problem.

## Background & Motivation

Decision-Focused Learning (DFL) represents an important advance in operations research and optimization. The traditional Predict-then-Optimize (PTO) pipeline operates in two stages: first predicting uncertain parameters via machine learning, then solving an optimization problem based on the predictions. The core limitation of PTO lies in the fact that it minimizes intermediate prediction error rather than final decision quality, leading to scenarios where predictions are accurate but decisions are suboptimal.

DFL embeds an optimization layer within a deep learning architecture to directly minimize decision regret end-to-end, achieving better results than PTO. However, existing DFL frameworks (including OptNet, CvxpyLayers, PyEPO, etc.) remain fundamentally **sequential** (Sequential DFL, S-DFL): prediction outputs flow unidirectionally to the optimization module, and optimization results are never fed back to the prediction module.

In many real-world scenarios, there exists **bidirectional coupling** between decision-making and prediction. In ride-hailing matching, for example, after a platform proposes a driver–passenger assignment, user acceptance or rejection feedback should in turn refine subsequent matching decisions. Such closed-loop interactions are prevalent in dynamic pricing, task allocation, and other multi-agent systems. S-DFL cannot capture this recursive interaction, leading to training instability and suboptimal decisions.

The core idea of this paper is to **extend DFL from a sequential to a recursive structure**, allowing optimization outputs to serve as feedback signals back to the prediction module, forming a closed loop of prediction → optimization → feedback → prediction. This is intuitive and natural, yet technically challenging: the recursive structure produces a directed cyclic graph (cyclic graph), violating the DAG assumption fundamental to deep learning, which prevents standard backpropagation from being directly applied.

## Method

### Overall Architecture

R-DFL consists of two core modules: a parameterized prediction model $\mathcal{F}_\theta$ and a convex optimization model $\mathcal{G}$. Unlike S-DFL, the prediction model in R-DFL takes two inputs—a feature vector $\boldsymbol{v}$ and an optimization result $\boldsymbol{x}$—and outputs predicted parameters $\hat{\boldsymbol{c}} = \mathcal{F}_\theta(\boldsymbol{x}, \boldsymbol{v})$. The optimization model generates an optimal decision $\boldsymbol{x}^* = \arg\min_{\boldsymbol{x} \in \mathcal{A}} g(\boldsymbol{x}; \hat{\boldsymbol{c}})$ based on the predicted parameters. The decision is then fed back into the prediction model, forming a closed loop.

To enable gradient propagation through this cyclic structure, two differentiation schemes are proposed: explicit unrolling and implicit differentiation.

### Key Designs

1. **Explicit Unrolling**:

    - Function: The cyclic prediction–optimization interaction is unrolled into $K$ sequential layers, each performing $\hat{\boldsymbol{c}}_i = \mathcal{F}_\theta(\boldsymbol{x}_{i-1}, \boldsymbol{v})$ and $\boldsymbol{x}_i = \mathcal{G}(\hat{\boldsymbol{c}}_i)$.
    - Mechanism: A composite layer $\Phi_\theta = \mathcal{G} \circ \mathcal{F}_\theta$ is defined, and $K$ iterations are unrolled into an explicit computation graph, through which gradients are backpropagated via standard automatic differentiation. The total gradient is $\frac{\partial \mathcal{L}}{\partial \theta} = \frac{\partial \mathcal{L}}{\partial \boldsymbol{x}_K} \sum_{i=1}^{K} \left(\prod_{j=i+1}^{K} J_{\Phi_\theta}|_{\boldsymbol{x}_{j-1}}\right) \frac{\partial \Phi_\theta(\boldsymbol{x}_{i-1})}{\partial \theta}$
    - Design Motivation: Straightforward to implement, directly leveraging PyTorch's automatic differentiation. However, computational cost grows linearly with unrolling depth $K$, and the method is susceptible to vanishing/exploding gradients.

2. **Implicit Differentiation**:

    - Function: The recursive system is modeled as a fixed-point problem $\boldsymbol{x}^* = \Phi_\theta(\boldsymbol{x}^*)$, with gradients computed directly at the equilibrium point.
    - Mechanism: The forward pass solves the fixed-point equation $\mathcal{H}_\theta = \boldsymbol{x}^* - \Phi_\theta(\boldsymbol{x}^*) \to 0$ via RootFind; the backward pass applies the implicit function theorem to directly yield the gradient $\frac{\partial \mathcal{L}}{\partial \theta} = \frac{\partial \mathcal{L}}{\partial \boldsymbol{x}^*}(I - J_{\Phi_\theta}|_{\boldsymbol{x}^*})^{-1} \frac{\partial \Phi_\theta(\boldsymbol{x}^*)}{\partial \theta}$
    - Design Motivation: The inverse Jacobian $(I - J_{\Phi_\theta}|_{\boldsymbol{x}^*})^{-1}$ implicitly encodes the information of the full unrolled structure without storing intermediate states, yielding significantly better memory and computational efficiency than explicit unrolling. However, gradient expressions must be derived manually.

3. **Gradient Equivalence Proof**:

    - Function: Proves that as the number of unrolling steps $K \to \infty$, explicit unrolling and implicit differentiation yield identical gradients.
    - Mechanism: The Neumann series expansion $(I - J)^{-1} = \sum_{k=0}^{\infty} J^k$ (when spectral radius $\rho(J) < 1$) is used to establish the mathematical equivalence between the two methods.
    - Design Motivation: Provides theoretical guarantees for practitioners—regardless of which differentiation method is chosen, the final accuracy is equivalent; the choice affects only computational efficiency.

### Loss & Training

The loss function adopts decision regret: $\mathcal{R}(\hat{\boldsymbol{c}}, \boldsymbol{c}) = g(\boldsymbol{x}^*(\hat{\boldsymbol{c}}), \boldsymbol{c}) - g(\boldsymbol{x}^*(\boldsymbol{c}), \boldsymbol{c})$, which measures the gap in objective value between the decision made based on predicted parameters and the optimal decision made based on true parameters. During training, all unrolled layers share the same parameter set $\theta$, updated via gradient descent.

## Key Experimental Results

### Main Results

Four methods are compared on two benchmark problems (newsvendor problem on synthetic data and bipartite matching on NYC real-world data) across three scales (Small/Mid/Large).

| Dataset | Metric | R-DFL-I | R-DFL-U | S-DFL | PTO |
|--------|------|---------|---------|-------|-----|
| Newsvendor-Small | RMSE↓ | **8.831** | 8.983 | 12.245 | 12.771 |
| Newsvendor-Mid | RMSE↓ | **9.106** | 9.173 | 12.536 | 12.747 |
| Newsvendor-Large | RMSE↓ | **9.327** | 9.343 | 12.649 | 12.684 |
| Matching-Small | RMSE↓ | 0.398 | **0.396** | 0.408 | 0.412 |
| Matching-Mid | RMSE↓ | **0.220** | 0.222 | 0.231 | 0.232 |
| Matching-Large | RMSE↓ | 0.171 | **0.170** | 0.187 | 0.190 |

R-DFL reduces RMSE by approximately 26–28% over S-DFL on the newsvendor problem and by approximately 5–9% on the matching problem.

### Ablation Study

Comparison of different prediction models (LSTM/RNN/Transformer) on large-scale problems:

| Configuration | Newsvendor RMSE | Matching RMSE | Note |
|------|----------------|---------------|------|
| R-DFL-I + LSTM | 10.104 | 0.174 | Implicit method; all prediction models outperform S-DFL |
| R-DFL-U + LSTM | 10.112 | 0.176 | Explicit method; results close to implicit |
| S-DFL + LSTM | 12.693 | 0.230 | Sequential baseline |
| R-DFL-I + Transformer | 11.332 | **0.166** | Transformer achieves best results on matching |
| PTO + RNN | 12.842 | 0.185 | Traditional two-stage baseline |

Sensitivity analysis on unrolling depth shows diminishing marginal accuracy gains as depth increases, while the explicit method's time cost grows linearly and the implicit method's time cost remains largely stable.

### Key Findings
- The two differentiation methods of R-DFL are highly consistent in accuracy (QQ plots show strong linear alignment), validating the gradient equivalence theorem.
- The implicit method trains approximately 1.5× faster than the explicit method on large-scale problems (1867s vs. 2704s per epoch).
- R-DFL's advantages are independent of the prediction model architecture, holding consistently across LSTM, RNN, and Transformer.

## Highlights & Insights
- This is the first work to extend DFL from the sequential paradigm to the recursive paradigm. The concept is simple yet significant, as closed-loop decision-making is a more universal paradigm in real-world settings.
- The gradient equivalence proof between explicit unrolling and implicit differentiation is both elegant and practical, providing theoretical grounding for choosing a differentiation method.
- The paper transfers the ideas of Deep Equilibrium Models (DEQ) into the DFL domain within operations research, constituting a cross-domain methodological transfer.
- The experimental design accounts for varying problem scales and prediction model architectures, validating the generality of the framework.

## Limitations & Future Work
- The current framework only supports convex optimization problems (differentiable convex objectives with linear constraints) and cannot be directly applied to integer programming, combinatorial optimization, or other discrete problems.
- The fixed-point convergence assumptions (PL condition, spectral radius < 1) may not always be satisfied in practice.
- Experimental scale remains relatively small (up to 900 decision variables), and performance on large-scale industrial problems has yet to be validated.
- Stochastic recursive environments, where the feedback signal itself is noisy, have not been considered.

## Related Work & Insights
- The approach is conceptually similar to Deep Equilibrium Models (Bai et al., 2019), both replacing layer-by-layer unrolling with fixed-point methods, but R-DFL applies this to a hybrid learning-plus-optimization system rather than a purely neural architecture.
- Representative DFL works include OptNet, SPO+, and PyEPO; this paper presents the first framework to systematically model bidirectional interactions in DFL.
- Methodologically, this work can inspire model-based planning in RL, Stackelberg game solving in game theory, and related settings.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] RcAE: Recursive Reconstruction Framework for Unsupervised Industrial Anomaly Detection](rcae_recursive_reconstruction_framework_for_unsupervised_industrial_anomaly_dete.md)
- [\[ICML 2026\] Decision Tree Learning on Product Spaces](../../ICML2026/others/decision_tree_learning_on_product_spaces.md)
- [\[ICLR 2026\] Active Learning for Decision Trees with Provable Guarantees](../../ICLR2026/others/active_learning_for_decision_trees_with_provable_guarantees.md)
- [\[AAAI 2026\] Enhancing Noise Resilience in Face Clustering via Sparse Differential Transformer](enhancing_noise_resilience_in_face_clustering_via_sparse_differential_transforme.md)
- [\[AAAI 2026\] A New Strategy for Verifying Reach-Avoid Specifications in Neural Feedback Systems](a_new_strategy_for_verifying_reach-avoid_specifications_in_neural_feedback_syste.md)

</div>

<!-- RELATED:END -->

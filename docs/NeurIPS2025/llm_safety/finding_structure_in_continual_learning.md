---
title: >-
  [Paper Note] Finding Structure in Continual Learning
description: >-
  [NeurIPS 2025][LLM Safety][Continual Learning] This paper proposes a continual learning optimization framework based on Douglas-Rachford Splitting (DRS)…
tags:
  - "NeurIPS 2025"
  - "LLM Safety"
  - "Continual Learning"
  - "Catastrophic Forgetting"
  - "Douglas-Rachford Splitting"
  - "Rényi Divergence"
  - "Bayesian Inference"
  - "Proximal Operator"
date: 2026-05-08
content_hash: 57baf19a1e192488
---

# Finding Structure in Continual Learning

**Conference**: NeurIPS 2025
**arXiv**: [2602.04555](https://arxiv.org/abs/2602.04555)  
**Code**: To be confirmed  
**Area**: LLM Safety
**Keywords**: Continual Learning, Catastrophic Forgetting, Douglas-Rachford Splitting, Rényi Divergence, Bayesian Inference, Proximal Operator

## TL;DR
This paper proposes a continual learning optimization framework based on Douglas-Rachford Splitting (DRS), which decouples stability and plasticity into two independent proximal subproblems, and replaces KL divergence with Rényi divergence for more robust prior alignment, thereby effectively alleviating catastrophic forgetting without replay buffers or additional modules.

## Background & Motivation
Continual learning requires a model to acquire new knowledge (plasticity) while retaining previously learned knowledge (stability) across sequentially presented tasks — a tension known as the "stability-plasticity dilemma." Existing approaches fall into three main categories:

1. **Replay methods**: Store past data for rehearsal, but memory overhead grows linearly with the number of tasks.
2. **Architecture expansion methods**: Add new modules per task, leading to unsustainable model growth.
3. **Regularization methods**: Penalize changes to important parameters via additional loss terms (e.g., EWC uses the Fisher information matrix), but fundamentally couple both objectives into a single loss $\mathcal{L}_{CL} = L_{\text{new}} + R_{\text{reg}}$.

The authors argue that the root issue lies not in the objective function itself, but in the **optimization strategy** — applying standard SGD to a coupled objective causes gradient conflicts, turning stability and plasticity into a zero-sum game. The solution is not to balance the conflict, but to **change the mode of interaction**.

## Core Problem
How can stability and plasticity objectives in continual learning be decoupled at the optimization level to avoid gradient interference and achieve cooperation rather than competition?

## Method

### Overall Architecture
The model adopts a variational autoencoder (VAE) architecture: an encoder $\phi$ maps inputs to a latent posterior $q_\phi(z|x)$, and a decoder $\theta$ predicts outputs from latent variables. The key design is **posterior-to-prior propagation**: after learning task $t-1$, its posterior becomes the prior for task $t$.

The training objective is:

$$\mathcal{L}(\phi, \theta) = \underbrace{\mathbb{E}_{z \sim q_\phi}\left[\sum_{n=1}^{N} \log p_\theta(y_n | x_n, z)\right]}_{f:\ \text{plasticity}} - \underbrace{\lambda \sum_{i=1}^{d} w_i D_\alpha(q_\phi^i \| p^i)}_{g:\ \text{stability}}$$

where $w_i = (\sigma_p^i)^2 / \sum_j (\sigma_p^j)^2$ are adaptive weights. Dimensions with larger prior variance receive looser constraints (allowing more plasticity), while those with smaller prior variance are more tightly constrained (protecting learned features).

### DRS Optimization Procedure (Core Contribution)
The objectives $f$ (plasticity) and $g$ (stability) are handled by their respective proximal operators. The algorithm maintains an auxiliary variable $u$, with each iteration consisting of three steps:

**Step 1 — Plasticity Proximal Step (Task-Fitting)**:
$$x_i = \text{prox}_f(u_{i-1}) = \arg\min_{\phi,\theta}\left[f(\phi,\theta) + \frac{1}{2\gamma}\|(\phi,\theta) - u_{i-1}\|^2\right]$$
Approximately solved via several Adam gradient descent steps, updating both encoder and decoder.

**Step 2 — Stability Reflection Step (Prior-Alignment)**:
$$y_i = \text{prox}_g(2x_i - u_{i-1})$$
Only the encoder $\phi$ is updated to align the posterior to the prior. Decoder parameters are carried over directly from Step 1, preserving their specialization on the new task.

**Step 3 — Relaxation Update**:
$$u_i = u_{i-1} + \lambda_r(y_i - x_i)$$
Interpolates between plasticity and stability, driving the auxiliary variable toward a consensus point.

### Rényi Divergence as a Replacement for KL Divergence
Proposition 3.1 demonstrates a fundamental limitation of KL divergence within the DRS framework: when the plasticity step proposes parameters far from the prior support, the "zero-forcing" property of KL causes the stability step to entirely discard the plasticity proposal, halting learning. The "zero-avoiding" property of Rényi divergence ensures that the penalty remains finite, permitting meaningful interpolation between the prior and new proposals.

### Convergence Guarantees
Proposition 3.2 proves that: (i) fixed points of DRS correspond to stationary points of the continual learning objective, satisfying $0 \in \nabla f(\omega^*) + \partial g(\omega^*)$; and (ii) the discrepancy between the plasticity and stability steps vanishes, $\lim_{k\to\infty}\|x_k - y_k\| = 0$, indicating that both objectives ultimately reach consensus.

## Key Experimental Results
Comparisons against 14 methods across 6 benchmarks (ResNet-18 backbone):

| Metric | Setting | Ours | Best Competitor |
|--------|---------|------|----------------|
| Average Accuracy | Disjoint tasks (4) | **65.7%** | SB-MCL 64.9% |
| Average Accuracy | Joint tasks (2) | **88.2%** | SPG/SB-MCL 87.5% |
| Backward Transfer BWT | Disjoint tasks | **-1.9** | TAG -1.2 (less forgetting but much lower accuracy) |
| Backward Transfer BWT | Joint tasks | **+3.2** | UCL/UPGD +2.0 |
| Forward Transfer FWT | Disjoint tasks | **+7.9** | SB-MCL +7.1 |
| Forward Transfer FWT | Joint tasks | **+10.4** | POCL +9.1 |

- On 100 sequential tasks from CASIA-100, the forgetting rate remains below 4%, while KL-based methods exceed 13%.
- Ablation studies identify $\alpha=2.0$ as optimal, yielding ~77% accuracy; $\alpha=0$ (no stability term) drops to ~72%.
- Replacing stochastic sampling with deterministic latent variables reduces training time by 9% but lowers accuracy from 79.1% to 76.3%.
- Computational overhead is comparable to or lower than baseline methods.

## Highlights & Insights
1. **Novel perspective**: Reframes continual learning as an optimization problem rather than an objective design problem, transforming the stability-plasticity tension from a "tug-of-war" to a "negotiation" via operator splitting.
2. **Replay-free**: Requires no historical data storage; knowledge retention is achieved purely through posterior propagation, ensuring high memory efficiency.
3. **Theoretical rigor**: Provides convergence guarantees and a formal argument for why Rényi divergence outperforms KL, making this more than a purely empirical contribution.
4. **Simplicity and efficiency**: Introduces no new architectural modules or external memory, with computational overhead comparable to standard methods.
5. **Significant forward transfer**: Not only reduces forgetting but actively leverages prior knowledge to accelerate new task learning (FWT +10.4), achieving genuine synergy.

## Limitations & Future Work
1. **Gaussian assumption**: All distributions are constrained to the Gaussian family for closed-form Rényi divergence, which may be insufficient for complex multimodal posteriors.
2. **Proximal operator approximation**: $\text{prox}_f$ is approximated via gradient descent; the impact of approximation quality on final performance is not thoroughly discussed.
3. **Hyperparameter sensitivity**: Although ablations are conducted for $\alpha$, the interaction between $\gamma$ and $\lambda$ is not analyzed in detail.
4. **Limited experimental domains**: Validation is primarily on image classification; continual learning in NLP, reinforcement learning, and other domains is not explored.
5. **Large pretrained models**: Effectiveness on pretrained models (e.g., CLIP, ViT-Large) is not validated, despite the prevalence of fine-tuning large models in practice.
6. **Task boundary assumption**: Explicit task boundaries are still required to trigger prior updates, limiting direct applicability to unsupervised or online continual learning settings.

## Related Work & Insights

| Method | Category | Replay Required | Optimization | Divergence | Characteristics |
|--------|----------|:--------------:|-------------|-----------|----------------|
| EWC | Regularization | No | SGD | — | Fisher-weighted regularization |
| VCL/UCL | Bayesian | Optional | SGD | KL | Posterior propagation + KL constraint |
| SB-MCL | Bayesian | Yes | SGD | KL | Ensemble method + joint training |
| UPGD | Gradient correction | No | Modified gradient | — | Direct gradient direction modification |
| POCL | Proximal | Yes | Proximal point | KL | Single proximal operator on combined loss |
| **Ours** | **Operator splitting** | **No** | **DRS** | **Rényi** | **Dual proximal decoupling + Rényi** |

Compared to POCL, which also employs a proximal method, the key distinction is that POCL applies a single proximal operator to a combined loss (task + replay) and still requires replay data, whereas the proposed method **splits** the objective into two independent proximal subproblems, eliminating the need for replay entirely.

The following broader insights emerge from this work:
1. **Optimization perspective**: Objective conflicts in other ML settings (e.g., multi-task learning, fairness constraints) may similarly benefit from operator splitting rather than simple weighted aggregation.
2. **Broader applicability of Rényi divergence**: Replacing KL with Rényi in variational inference and generative models may yield analogous robustness advantages.
3. **Connection to federated learning**: The local update–global aggregation tension in federated learning shares structural similarities with the stability-plasticity dilemma; the DRS framework warrants investigation in that context.
4. **DRS potential in deep learning optimization**: Traditionally applied in convex optimization and signal processing, this paper demonstrates the feasibility of DRS in non-convex deep learning, opening a new research direction.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — Addresses continual learning at the optimization algorithm level with a distinctly original perspective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Six datasets, 14 baselines, and comprehensive ablations; experimental scenarios are concentrated on image classification.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear motivation, rigorous theoretical derivations, and intuitive illustrations.
- **Value**: ⭐⭐⭐⭐ — Establishes a new paradigm; scalability to large pretrained model settings requires further validation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Attention Retention for Continual Learning with Vision Transformers](../../AAAI2026/llm_safety/attention_retention_for_continual_learning_with_vision_transformers.md)
- [\[ICML 2026\] Understanding Generalization and Forgetting in In-Context Continual Learning](../../ICML2026/llm_safety/understanding_generalization_and_forgetting_in_in-context_continual_learning.md)
- [\[ICCV 2025\] Adversarial Robust Memory-Based Continual Learner](../../ICCV2025/llm_safety/adversarial_robust_memory-based_continual_learner.md)
- [\[NeurIPS 2025\] Reinforcement Learning with Backtracking Feedback](reinforcement_learning_with_backtracking_feedback.md)
- [\[NeurIPS 2025\] Contextual Integrity in LLMs via Reasoning and Reinforcement Learning](contextual_integrity_in_llms_via_reasoning_and_reinforcement_learning.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Efficient Estimation of Kernel Surrogate Models for Task Attribution
description: >-
  [ICLR 2026][Reinforcement Learning][task attribution] This paper proposes a kernel surrogate model (KernelSM) for task attribution. By employing RBF kernel ridge regression to capture nonlinear interaction effects among tasks, combined with a gradient-projection-based efficient estimation algorithm that eliminates repeated retraining, KernelSM achieves a 25% improvement in correlation over linear surrogate and influence function baselines across mathematical reasoning, in-context learning, and multi-objective RL settings.
tags:
  - ICLR 2026
  - Reinforcement Learning
  - task attribution
  - kernel surrogate model
  - influence function
  - data attribution
  - kernel ridge regression
date: 2026-05-08
content_hash: d70d99161ca2b732
---

# Efficient Estimation of Kernel Surrogate Models for Task Attribution

**Conference**: ICLR 2026
**arXiv**: [2602.03783](https://arxiv.org/abs/2602.03783)
**Code**: [https://github.com/VirtuosoResearch/Kernel-surrogate-models](https://github.com/VirtuosoResearch/Kernel-surrogate-models)
**Area**: Reinforcement Learning
**Keywords**: task attribution, kernel surrogate model, influence function, data attribution, kernel ridge regression

## TL;DR
This paper proposes a kernel surrogate model (KernelSM) for task attribution. By employing RBF kernel ridge regression to capture nonlinear interaction effects among tasks, combined with a gradient-projection-based efficient estimation algorithm that eliminates repeated retraining, KernelSM achieves a 25% improvement in correlation over linear surrogate and influence function baselines across mathematical reasoning, in-context learning, and multi-objective RL settings.

## Background & Motivation

**State of the Field**: Modern AI systems (e.g., LLMs) are trained simultaneously on diverse tasks. Quantifying the contribution of each training task to a target task (task attribution) is a central problem in interpretability. Existing approaches include leave-one-out (LOO) retraining (exact but computationally infeasible), influence functions (requiring Hessian computation), and linear surrogate models (fitting a linear function via random subset sampling).

**Limitations of Prior Work**: Linear surrogate models can only capture first-order additive effects and fail to express nonlinear interactions among tasks—such as synergistic effects (joint training on two tasks outperforms the sum of individual contributions), adversarial effects, and XOR-type effects. These interactions are particularly pronounced for training samples near decision boundaries.

**Root Cause**: Stronger surrogate models (e.g., kernel methods) require evaluating model performance $F(\mathbf{s})$ on more subsets, each demanding a full retraining—making the computational cost comparable to LOO.

**Paper Goals**: (1) Provide a unified analysis of the relationship between linear surrogates and influence functions; (2) Design surrogate models capable of capturing nonlinear task interactions; (3) Enable efficient estimation without repeated retraining.

**Starting Point**: The paper first uses a second-order Taylor expansion to prove that linear surrogates $\approx$ influence functions (when second-order interactions are small), thereby exposing the limitations of linear surrogates. It then upgrades the surrogate model with an RBF kernel and employs a first-order gradient approximation to avoid repeated retraining.

**Core Idea**: Efficiently construct kernel surrogate models via gradient-projection-based first-order approximation, capturing nonlinear task interactions with less than 2% relative error.

## Method

### Overall Architecture
Given $K$ training tasks → sample $m$ binary subset vectors $\mathbf{s} \in \{0,1\}^K$ → efficiently estimate model performance $F(\mathbf{s}^{(i)})$ for each subset (via gradient approximation, without retraining) → fit $g_\theta: \{0,1\}^K \to \mathbb{R}$ using kernel ridge regression → output task attribution scores (nonlinear task interactions are implicitly encoded by the kernel function).

### Key Designs

1. **Unified Analysis of Linear Surrogates and Influence Functions**:

   - Function: Prove that linear surrogate model coefficients are equivalent to influence functions under a first-order approximation.
   - Mechanism: Apply a second-order Taylor expansion to $F(\mathbf{s})$, substitute into the linear regression objective, and analyze the regression coefficients via the delta method. Proposition 3.1 establishes $\|\hat{\beta} - \nabla_\mathbf{s} F(\mathbf{s}^*) - \text{second-order correction}\| \lesssim c_3 K^{3/2} p^{-1}$.
   - Design Motivation: Reveals the fundamental limitation of linear surrogates—when the norm of the Hessian $\mathbf{H}_\mathbf{s}$ is non-negligible, neither linear surrogates nor influence functions can accurately attribute task contributions.

2. **RBF Kernel Surrogate Model (KernelSM)**:

   - Function: Replace linear regression with kernel ridge regression to learn nonlinear task interactions.
   - Mechanism: $g_\theta(\mathbf{s}) = \sum_i \theta_i k(\mathbf{s}^{(i)}, \mathbf{s})$, where $k(\mathbf{s}^{(a)}, \mathbf{s}^{(b)}) = \exp(-\gamma \|\mathbf{s}^{(a)} - \mathbf{s}^{(b)}\|^2)$. The closed-form solution for the coefficients is $\theta = (\mathcal{K} + \lambda I)^{-1} \mathbf{F}$.
   - Design Motivation: The RBF kernel is a universal approximator on the binary space $\{0,1\}^K$, and its geometric intuition naturally matches the subset space—similar subsets (close in Hamming distance) should yield similar performance.

3. **Efficient Estimation via Gradient Projection (Core Contribution)**:

   - Function: Avoid retraining the model for each subset $\mathbf{s}^{(i)}$ by using a first-order Taylor approximation to estimate $F(\mathbf{s}^{(i)})$.
   - Mechanism: Perform a first-order expansion at the pretrained weights $W_0$: $f_W(x) \approx f_{W_0}(x) + \langle \nabla f_{W_0}(x), W - W_0 \rangle$. For each subset $\mathbf{s}^{(i)}$, solve for the optimal perturbation $Z^*_{\mathbf{s}^{(i)}}$ via multinomial logistic regression using projected gradients as features, then estimate $\hat{f}(x) = f_{W_0}(x) + \langle \nabla f_{W_0}(x), Z^*_{\mathbf{s}^{(i)}} \rangle$.
   - Design Motivation: Gradients need only be computed once at $W_0$; all subsequent subset estimations reduce to linear algebra on CPU. Empirical validation shows that the first-order approximation error is less than 2%.
   - Key Technique: Gaussian random convolution is used to project high-dimensional gradient vectors into a low-dimensional space, reducing regression solving to a matter of seconds.

### Loss & Training
Kernel ridge regression objective: $\min_{g_\theta} \sum_{i=1}^m (F(\mathbf{s}^{(i)}) - g_\theta(\mathbf{s}^{(i)}))^2 + \lambda \|g_\theta\|^2_\mathcal{K}$

Hyperparameters $\lambda$ and $\gamma$ are selected via cross-validation.

## Key Experimental Results

### Main Results

| Method | CIFAR-10 Corr.↑ | Modular Arith. Corr.↑ | ICL Corr.↑ | Multi-obj. RL Corr.↑ |
|---|---|---|---|---|
| Influence Function | ~0.74 | ~0.55 | ~0.72 | ~0.65 |
| Linear Surrogate | ~0.76 | ~0.58 | ~0.75 | ~0.68 |
| **KernelSM** | ~0.82 | ~0.80 | ~0.88 | ~0.73 |

KernelSM achieves a 25% improvement in correlation over linear baselines and influence functions (compared against LOO ground truth). The largest gain is observed on modular arithmetic tasks (42%), attributable to strong nonlinear interactions in operations such as XOR and division.

### Ablation Study

| Kernel | CIFAR-10 Residual↓ | Modular Arith. Residual↓ |
|---|---|---|
| Linear | 4.4±0.9 | 4.6±1.3 |
| **RBF** | **1.0±0.0** | **1.5±0.4** |

| Task | First-order Approx. Relative Error |
|---|---|
| CIFAR-10 | 1.02±0.69% |
| Modular Arithmetic | 2.40±2.17% |
| ICL | 0.51±0.04% |
| Multi-objective RL | 0.43±0.73% |

### Key Findings
- **Nonlinear interactions are pervasive**: The residual error of the RBF kernel is 1/4 to 1/3 that of the linear model, indicating that nonlinear task interactions are non-negligible.
- **First-order gradient approximation is sufficiently accurate**: Relative error remains below 2% across diverse tasks and model scales (including a 34B-parameter LLM).
- **Downstream task selection benefits substantially**: Using KernelSM for ICL example selection and task selection in multi-objective optimization reduces loss by 40% compared to linear methods.
- Linear surrogates and influence functions are empirically confirmed to be equivalent under first-order approximation (Pearson correlation 0.96–0.98).

## Highlights & Insights
- **Unified theory**: This is the first work to rigorously establish, via second-order Taylor expansion, the conditions under which linear surrogate models and influence functions are equivalent, while revealing their shared limitation—neither can capture task interactions.
- **Efficient estimation via gradient projection**: The approach elegantly circumvents the computational bottleneck of kernel methods through first-order Taylor approximation combined with random projection for dimensionality reduction, making the practical cost of KernelSM comparable to that of linear models.
- **Geometric intuition of the RBF kernel**: On the binary subset space $\{0,1\}^K$, the RBF kernel is equivalent to a heat kernel based on Hamming distance—implying that similar training subsets (differing in only a few tasks) should yield similar performance, which constitutes a sound inductive bias.

## Limitations & Future Work
- The first-order approximation is valid near the pretrained weights $W_0$; when fine-tuning deviates substantially (e.g., full fine-tuning of large models), the approximation error may increase significantly.
- The expressive capacity of the kernel surrogate model is bounded by the number of sampled subsets $m$—for large $K$, a sufficiently large number of subset samples is required.
- Stronger kernel functions such as deep kernels or neural tangent kernels remain unexplored.
- Validation is limited to task-level attribution; extension to instance-level data attribution (though the theoretical framework is general) has not been pursued.
- Evaluation relies primarily on correlation with LOO ground truth; the practical utility of kernel attribution for model debugging or data cleaning has not been validated.

## Related Work & Insights
- **vs. DataModels (Ilyas et al. 2022)**: DataModels employ linear regression for data attribution; KernelSM generalizes this to kernel methods.
- **vs. Influence Functions (Koh & Liang 2017)**: This paper proves that influence functions $\approx$ the first-order approximation of linear surrogates; only KernelSM can capture second-order effects.
- **vs. TRAK (Park et al. 2023)**: TRAK also uses gradient features for data attribution but remains a linear model; KernelSM employs an RBF kernel to capture nonlinear interactions.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of applying kernel methods to task attribution is natural and theoretically grounded; the efficient estimation algorithm is the key contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers classification, mathematical reasoning, ICL, and RL; ablations are comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Theory and experiments are well-organized, though some proof details are overly compressed.
- Value: ⭐⭐⭐⭐ Provides a more powerful tool for task attribution with significant gains in downstream applications (task selection).

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Pruning as a Cooperative Game: Surrogate-Assisted Layer Contribution Estimation for Large Language Models](pruning_as_a_cooperative_game_surrogate-assisted_layer_contribution_estimation_f.md)
- [\[ICLR 2026\] One Model for All Tasks: Leveraging Efficient World Models in Multi-Task Planning](one_model_for_all_tasks_leveraging_efficient_world_models_in_multi-task_planning.md)
- [\[ICLR 2026\] Optimistic Task Inference for Behavior Foundation Models](optimistic_task_inference_behavior_models.md)
- [\[ICLR 2026\] WIMLE: Uncertainty-Aware World Models with IMLE for Sample-Efficient Continuous Control](wimle_uncertainty-aware_world_models_with_imle_for_sample-efficient_continuous_c.md)
- [\[AAAI 2026\] QiMeng-Kernel: Macro-Thinking Micro-Coding Paradigm for LLM-Based High-Performance GPU Kernel Generation](../../AAAI2026/reinforcement_learning/qimeng-kernel_macro-thinking_micro-coding_paradigm_for_llm-based_high-performanc.md)

<!-- RELATED:END -->

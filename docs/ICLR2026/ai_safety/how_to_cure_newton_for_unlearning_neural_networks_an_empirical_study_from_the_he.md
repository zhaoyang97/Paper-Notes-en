---
title: >-
  [Paper Note] How to Cure Newton for Unlearning Neural Networks? An Empirical Study from the Hessian Perspective
description: >-
  [ICLR 2026][AI Safety][Paper Note] This paper discovers that Newton unlearning fails on real-world neural networks and LLMs due to Hessian degeneracy (a large number of zero/negative eigenvalues). It proposes CuReNU, based on cubic regularization, and its stochastic Hessian-free variant CuReNUS. These methods automatically determine the damping factor $
tags:
  - ICLR 2026
  - AI Safety
date: 2026-05-08
content_hash: 3b38610f19f80ac3
---
# How to Cure Newton for Unlearning Neural Networks? An Empirical Study from the Hessian Perspective

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=dHz2LBCyTh](https://openreview.net/forum?id=dHz2LBCyTh)  
**Area**: AI Safety / Machine Unlearning  
**Keywords**: Machine Unlearning, Second-order Unlearning, Newton's Method, Hessian Degeneracy, Cubic Regularization

## TL;DR
This paper discovers that Newton unlearning fails on real-world neural networks and LLMs due to Hessian degeneracy (a large number of zero/negative eigenvalues). It proposes CuReNU, based on cubic regularization, and its stochastic Hessian-free variant CuReNUS. These methods automatically determine the damping factor $\gamma$, guarantee convergence to second-order stationary points, and achieve unlearning performance comparable to SOTA empirical methods across batch, sequential, and LLM-scale unlearning tasks.

## Background & Motivation

**Background**: Machine unlearning aims to erase the influence of specific training samples $D_e$ from a model without full retraining, satisfying the "right to be forgotten" (GDPR/CCPA) or removing noisy/malicious data. While retraining is precise, it is prohibitively expensive (e.g., GPT-4 costs over \$100M). Thus, the field shifts toward **approximate unlearning**—making the unlearned model approximate the behavior of a retrained model. **Second-order unlearning** leverages first-order (gradient) and second-order (Hessian) information, naturally connecting to influence functions and providing theoretical guarantees of "converging to the same loss as retraining," making it more rigorous than empirical heuristics (e.g., maximizing forget set loss, random labels).

**Limitations of Prior Work**: The classic Newton unlearning update $w_{t+1}=w_t-(H_{w_t}^{D_r})^{-1}g_{w_t}^{D_r}$ assumes an invertible Hessian. However, the authors' measurements on CNN×FMNIST and Llama-2×TOFU reveal that the Hessian spectrum follows a "spiked model": most eigenvalues cluster near zero (the bulk), with only a few large eigenvalues (spikes). The Hessian rank collapses rapidly as training converges, and negative eigenvalues (saddle points) appear. This makes the Hessian non-invertible, rendering vanilla Newton unlearning unusable and causing more severe degeneracy in LLMs.

**Key Challenge**: Common remedies address symptoms rather than the root cause. Using a pseudo-inverse (PINV-Newton) leads to an update norm $\|\Delta\|^2=\sum_{\lambda_i\neq0}\frac{1}{\lambda_i^2}(u_i^\top g)^2$ that explodes due to near-zero eigenvalues. Adding damping (Damped Newton) $\|\Delta\|^2=\sum_i\frac{1}{(\gamma+\lambda_i)^2}(u_i^\top g)^2$ also leads to massive updates if $\gamma$ is too small, causing the model to overshoot local minima and collapse in performance (accuracy drops to ~9% in experiments). However, the update norm decreases monotonically with $\gamma$, suggesting an optimal $\gamma$ exists.

**Goal**: Reformulate the problem to **automatically** find a $\gamma$ that is large enough to avoid extreme update norms but small enough to prevent trivial updates that stall convergence, making Newton unlearning viable for NN/LLM without manual hyperparameter tuning.

**Key Insight**: The authors adopt the **Cubic Regularized Newton Method** (Nesterov & Polyak), where the second-order approximation includes a cubic term $\frac{L}{6}\|\Delta\|^3$. This provides a global upper bound, ensuring global convergence guarantees even for non-convex losses. Its dual variables implicitly define the optimal damping factor $\gamma$.

**Core Idea**: Replace vanilla/damped Newton with cubic regularization, allowing the optimal $\gamma$ to be solved via an optimization problem. Design a stochastic Hessian-free version to scale the method to LLMs.

## Method

### Overall Architecture
The method follows one central line: **how to perform robust second-order unlearning given a trained model $w^*$ with a degenerate Hessian**. The authors first attribute the failure of Newton unlearning to Hessian degeneracy, then transform the search for the "optimal damping $\gamma$" into a solvable dual optimization problem (CuReNU). Finally, explicit Hessian computation is replaced by stochastic iterations using Hessian-vector products (HVP) (CuReNUS) to achieve LLM scalability. The workflow starts from the original model, iteratively "pulls" parameters away from the forget set $D_e$, converges toward the second-order stationary point of the retraining loss $L(w;D_r)$, and outputs the unlearned model.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Trained Model w*<br/>(Hessian Degenerate)"] --> B["Hessian Degeneracy Diagnosis<br/>Vanilla/PINV/Damped Newton fail"]
    B --> C["CuReNU: Cubic Regularized Newton Unlearning<br/>Dual solving for optimal γ"]
    C -->|Requires explicit Hessian, O(d²) non-scalable| D["CuReNUS: Stochastic Hessian-free Variant<br/>HVP + GD Inner Loop + Perturbed Gradient"]
    D --> E["Unlearned Model<br/>Approximates Retraining, ε-SOSP"]
```

### Key Designs

**1. Hessian Degeneracy Diagnosis: Attributing Unlearning Failure to Spectral Collapse**

This step explains why existing Newton unlearning fails on real networks. By measuring the spectrum (Observation 4.1), the authors characterize the post-training Hessian structure: there exists $k\ll d$ such that $\lambda_i>0$ only for the first $k$ eigenvalues, while others $\lambda_j\le0$. This destroys the invertibility required by Newton's method. Quantitatively, the pseudo-inverse $\|\Delta_{t+1}\|^2=\sum_{i:\lambda_i\neq0}\frac{1}{\lambda_i^2}(u_i^\top g_{w_t}^{D_r})^2$ and damped updates fail because near-zero eigenvalues or small $\gamma$ cannot suppress the norm. Massive updates cause performance to collapse, pinpointing "choosing the right $\gamma$" as the primary objective.

**2. CuReNU: Automatically Solving Optimal Damping $\gamma$ via Duality**

To address the $\gamma$ tuning problem, CuReNU minimizes a **cubic regularized approximation** of the retraining loss:

$$\tilde{L}(w_{t+1};D_r)=L(w_t;D_r)+\langle g_{w_t}^{D_r},\Delta_{t+1}\rangle+\tfrac{1}{2}\langle H_{w_t}^{D_r}\Delta_{t+1},\Delta_{t+1}\rangle+\tfrac{L}{6}\|\Delta_{t+1}\|^3$$

The cubic term makes $\tilde{L}$ a **global upper bound** of the true loss, providing global convergence guarantees that vanilla Newton (only local quadratic approximation) lacks. Since the problem is ill-defined for the primal update (due to unknown $\|\Delta\|$), the authors solve the **strong dual**: introducing the dual variable $\alpha_{t+1}\triangleq\|w_{t+1}-w_t\|$ yields a convex constrained optimization problem:

$$\sup_{\alpha_{t+1}\in Q}\;-\tfrac{1}{2}\Big\langle\big(H_{w_t}^{D_r}+\tfrac{L}{2}\alpha_{t+1}I\big)^{-1}g_{w_t}^{D_r},\,g_{w_t}^{D_r}\Big\rangle-\tfrac{L}{12}\alpha_{t+1}^3$$

Since it is convex in $\alpha_{t+1}$, it can be solved efficiently using trust-region solvers. Crucially, $\alpha_{t+1}$ **implicitly defines the optimal damping factor** $\gamma=\frac{L}{2}\alpha_{t+1}$. Theoretically, CuReNU converges globally to an $\varepsilon$-second-order stationary point ($\varepsilon$-SOSP) in $O(\varepsilon^{-1.5})$ iterations, a stronger guarantee than the $\varepsilon$-FOSP provided by first-order methods.

**3. CuReNUS: Stochastic Hessian-free Variant for LLM Scalability**

While CuReNU has convergence guarantees, it requires explicit Hessian storage ($O(d^2)$ space) and inversion ($O(nd^2+d^3)$ time), which is infeasible for LLMs. CuReNUS uses a **stochastic approximation** of the cubic regularization: it estimates stochastic gradients $g^{B_1}$ and Hessians $H^{B_2}$ on small batches $B_1, B_2 \subset D_r$. The gradient of $\tilde{L}_\text{sto}$ is computed efficiently via **Hessian-vector products (HVP)** (Pearlmutter trick), **completely avoiding explicit Hessian construction**. The inner loop iteration is:

$$\Delta_{s+1}=\Delta_s-\eta\big(\tilde{g}_{w_t}^{B_1}+H_{w_t}^{B_2}\Delta_s\big)$$

To handle the "hard case" in cubic optimization (where $\lambda_d<0$ and $\langle u_d,g^{B_1}\rangle=0$, causing gradients to stay orthogonal to the optimal direction), the authors add a small perturbation to the gradient $\tilde{g}^{B_1}=g^{B_1}+\sigma\zeta$. It converges to $\varepsilon$-SOSP within $\tilde{O}(\varepsilon^{-3.5})$ gradient/HVP evaluations with only $O(2d)$ memory.

### Loss & Training
Both algorithms target minimizing the **retraining loss** $L(w;D_r)$ (minimizing the retain set loss is shown to be equivalent to unlearning $D_e$). The Lipschitz constant $L$ is set empirically: 5 for FMNIST, 50 for CIFAR-10, 80 for AG-News, and 400 for TOFU. CuReNUS uses $\sigma=0.1$ and $\eta$ matching the training learning rate. Each outer loop samples new batches and runs $T_\text{inner} \approx 5\text{--}10$ steps. Convergence comparison (under non-convex loss):

| Algorithm | Convergence Rate | Guarantee |
|-----------|------------------|-----------|
| GD        | $O(\varepsilon^{-2})$ | $\varepsilon$-FOSP |
| SGD       | $O(\varepsilon^{-4})$ | $\varepsilon$-FOSP |
| Newton    | Local Quadratic  | $\varepsilon$-FOSP |
| CuReNU    | Global $O(\varepsilon^{-1.5})$ | $\varepsilon$-SOSP |
| CuReNUS   | Global $\tilde{O}(\varepsilon^{-3.5})$ | $\varepsilon$-SOSP |

## Key Experimental Results

### Main Results
Covering CNN×FMNIST, ResNet-18×CIFAR-10, and Llama-2-7B(+LoRA)×AG-News/TOFU. Evaluation benchmarks proximity to retraining using ToW (Tug-of-War, higher is better), JS divergence, Truth Ratio, and MIA.

CNN×FMNIST Batch Unlearning (Sample-level / Class-level):

| Method | Sample ToW (↑) | Class ToW (↑) | Class $D_e$ Acc (→Retrain 0) |
|--------|----------------|---------------|------------------------------|
| Retraining | 1.00 | 1.00 | 0.00 |
| PINV-Newton | 0.01 | 0.05 | 1.44 |
| Damped Newton | 0.01 | 0.05 | 0.52 |
| SCRUB (SOTA) | 0.94 | 0.97 | 0.00 |
| DELETE (SOTA) | 0.85 | 0.99 | 0.00 |
| **CuReNU** | 0.98 | 0.93 | 1.37 |
| **CuReNUS** | **0.98** | **0.99** | 0.14 |

Standard remedies (PINV/Damped) collapse, confirming the norm analysis: for class-level unlearning, PINV-Newton's update norm is $3708.78$ and Damped Newton is $838.68$, while CuReNU/CuReNUS are only $0.36/0.38$. CuReNUS achieves the best/tied ToW, equaling or surpassing SOTA empirical methods.

### Sequential Unlearning and Efficiency
5 rounds of sequential unlearning on Llama-2×TOFU (sample) and ResNet-18×CIFAR-10 (class), final round:

| Method | TOFU ToW (↑) | TOFU Truth Ratio (↑) | CIFAR ToW (↑) |
|--------|--------------|----------------------|---------------|
| Retraining | 1.00 | 0.658 | 1.000 |
| GD | 0.60 | 0.538 | 0.057 |
| SCRUB | 0.72 | 0.512 | 0.944 |
| NWA (NWA-like) | 0.08 | 0.831 | 0.732 |
| **CuReNUS** | **0.80** | 0.591 | 0.909 |

Efficiency (Table 4): On Llama-2×TOFU, retraining takes 900s, SCRUB 178s, and CuReNUS 340s but with much lower memory. CuReNU takes 6355s and $O(d^2)$ memory even on CNNs, failing on LLMs—validating CuReNUS as the only scalable option.

### Key Findings
- **Degeneracy diagnosis is the pivot**: Reducing update norms from ~3700 to ~0.36 quantitatively proves that "choosing the right $\gamma$" is the true cure.
- First-order methods (GD) exhibit **under-unlearning** (where $D_e$ remains similar to the original model), proving the necessity of second-order info for high-fidelity retraining approximation.
- CuReNU and CuReNUS exhibit similar performance, suggesting stochastic Hessian-free approximation loses negligible quality while gaining LLM scalability.

## Highlights & Insights
- **Tracing Unlearning Failure to Hessian Degeneracy**: Using spectral analysis and update norm formulas to provide a quantitative diagnosis of why PINV/Damping explode is far more profound than vaguely stating "Newton's method is unstable."
- **Dual Variable as Optimal Damping**: The relationship $\gamma=\frac{L}{2}\alpha_{t+1}$ transforms "manual damping tuning" into "automatic solving," elegantly migrating cubic regularization from optimization to unlearning.
- **HVP + Perturbed Gradient for Scalability**: Using the Pearlmutter trick to bypass explicit Hessians with $O(2d)$ memory on LLMs, combined with random perturbations to escape orthogonal subspaces of negative eigenvectors, forms a toolkit transferable to other massive-scale second-order scenarios.

## Limitations & Future Work
- The cubic coefficient $L$ (Lipschitz Hessian constant) cannot be precisely calculated and relies on empirical values per dataset; though robust, cross-task tuning still requires experience.
- CuReNU is limited to small models due to $O(d^2)$ memory; only CuReNUS scales to LLMs. The stability of stochastic approximation under extreme degeneracy or longer sequences remains to be explored.
- MIA is largely ineffective on well-regularized models; privacy validation relies on an overfitted model scenario, leaving a gap for direct privacy guarantees in realistic unlearning.
- Convergence guarantees rely on smoothness assumptions (Assumptions 3.1–3.3), which may have limited alignment with real LLM loss landscapes.

## Related Work & Insights
- **vs. Empirical Approximate Unlearning (GA / SCRUB / NPO / DELETE)**: These rely on heuristics and lack strict convergence guarantees, often degrading in difficult settings like sequential unlearning or LLMs. Ours follows a second-order path with $\varepsilon$-SOSP global convergence and achieves comparable or better ToW.
- **vs. Classic Newton/Second-order Unlearning (Guo et al. 2020; Golatkar et al. 2020)**: Previous works assumed linear models and invertible positive semi-definite Hessians. Ours shows these assumptions fail in NNs/LLMs and solves Hessian degeneracy via cubic regularization.
- **vs. Existing Hessian-free Unlearning (Qiao et al. 2025)**: Their space complexity $O(dn)$ scales linearly with the number of samples, whereas CuReNUS maintains $O(2d)$ constant memory, offering significantly better scalability.

## Rating
- Novelty: ⭐⭐⭐⭐ Systems-level attribution of failure to Hessian degeneracy solved via cubic regularization is a fresh perspective, though the algorithm is adapted from existing optimization theory.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers CNN/ResNet/LLM, batch/sequential unlearning, and multiple metrics including efficiency.
- Writing Quality: ⭐⭐⭐⭐ Logical flow from diagnosis to method to verification, with strong quantitative analysis.
- Value: ⭐⭐⭐⭐ Provides a scalable and theoretically grounded path for second-order unlearning on LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Designing Affine-Invariant Neural Networks for Photometric Corruption Robustness and Generalization](designing_affine-invariant_neural_networks_for_photometric_corruption_robustness.md)
- [\[ICLR 2026\] ATEX-CF: Attack-Informed Counterfactual Explanations for Graph Neural Networks](atex-cf_attack-informed_counterfactual_explanations_for_graph_neural_networks.md)
- [\[ICLR 2026\] Fisher-Rao Sensitivity for Out-of-Distribution Detection in Deep Neural Networks](fisher-rao_sensitivity_for_out-of-distribution_detection_in_deep_neural_networks.md)
- [\[ICML 2026\] Singular Bayesian Neural Networks](../../ICML2026/ai_safety/singular_bayesian_neural_networks.md)
- [\[ICLR 2026\] Robust Spiking Neural Networks Against Adversarial Attacks](robust_spiking_neural_networks_against_adversarial_attacks.md)

</div>

<!-- RELATED:END -->

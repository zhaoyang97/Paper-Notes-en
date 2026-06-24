---
title: >-
  [Paper Note] Catastrophic Forgetting is Low-Rank: A Function-Space Theory for Continual Adaptation
description: >-
  [ICML 2026][Learning Theory][Neural Tangent Kernel] Instead of treating catastrophic forgetting as "parameter drift," this work provides a closed-form characterization in function space under the NTK framework: new task training drags old task predictions away via the cross-task kernel $K_{AB}$, and this "forgetting vector" is precisely predictable before training. This vector concentrates on an extremely small number of eigenmodes of the old task kernel $K_{AA}$ (1–6 modes c…
tags:
  - "ICML 2026"
  - "Learning Theory"
  - "Continual Learning"
  - "Catastrophic Forgetting"
  - "NTK"
  - "Neural Tangent Kernel"
  - "Function Space"
  - "Spectral Regularization"
date: 2026-05-08
content_hash: c446ad6b569e5c3a
---

# Catastrophic Forgetting is Low-Rank: A Function-Space Theory for Continual Adaptation

**Conference**: ICML 2026  
**arXiv**: [2606.18024](https://arxiv.org/abs/2606.18024)  
**Code**: To be confirmed  
**Area**: Learning Theory / Continual Learning / Catastrophic Forgetting / NTK  
**Keywords**: Catastrophic Forgetting, Neural Tangent Kernel, Function Space, Continual Learning, Spectral Regularization

## TL;DR
Instead of treating catastrophic forgetting as "parameter drift," this work provides a closed-form characterization in function space under the NTK framework: new task training drags old task predictions away via the cross-task kernel $K_{AB}$, and this "forgetting vector" is precisely predictable before training. This vector concentrates on an extremely small number of eigenmodes of the old task kernel $K_{AA}$ (1–6 modes carry 50–90% of the forgetting energy), explaining why parameter-space regularizers fail on shared-head benchmarks and leading to a spectral regularization method that protects only the vulnerable subspace.

## Background & Motivation
**Background**: Continual adaptation—including fine-tuning, alignment maintenance, and adaptation under distribution shift—is long plagued by catastrophic forgetting. Mitigation strategies over the past decade have been diverse: parameter regularization (EWC, SI), replay (replay, DER++), knowledge distillation (LwF), gradient projection (GPM, OGD), and recent parameter-efficient continual learning for frozen backbones (prompt pool, low-rank adapters).

**Limitations of Prior Work**: Most methods "engineer around" forgetting without providing a **mechanistic explanation**—which output directions are being dragged away? Why are some directions vulnerable while others remain unaffected? The closest theoretical works (Doan et al., 2021; Bennani et al., 2020) introduce cross-task NTK overlap matrices but only focus on the **scalar magnitude/risk bounds** of forgetting, emphasizing how task alignment controls the total amount of forgetting without characterizing the **eigenspace structure** of the actual forgetting vector.

**Key Challenge**: Existing perspectives (parameter drift, replay, distillation) cannot answer the question: "which output directions are vulnerable?" This is crucial—if forgetting occurs only in a few output directions, spreading protection effort across all parameter directions (EWC) or all output directions (LwF) is inefficient or even misaligned.

**Goal**: Characterize forgetting in function space—(1) Can the direction and magnitude of the forgetting vector be predicted **before** running new task training? (2) In which subspace does this vector reside? (3) What determines the dimensionality of the vulnerable subspace?

**Key Insight**: Treat forgetting as "NTK interference." Under NTK linearization, the parameter update $\delta$ from new task training is closed-form, so the change in old task predictions $\Delta f_A = J_A \delta$ is also closed-form. A key observation is that constructing the cross-task Jacobian $J_B$ uses **new task inputs but old task weights**, allowing the entire predictor to be calculated before new task training starts.

**Core Idea**: Forgetting = the cross-task kernel $K_{AB}$ acting on the new task residual, with its direction locked by the eigenstructure of the old task kernel $K_{AA}$; thus, "forgetting is low-rank."

## Method

### Overall Architecture
Consider two-task continual regression: first train on task A to reach $\theta_A$, then start from $\theta_A$ and train on task B to reach $\theta_B$. Forgetting is defined as the induced drift in task A predictions, written as a flattened vector:

$$\Delta f_A := \mathrm{vec}\left[f_A(\theta_B) - f_A(\theta_A)\right] \in \mathbb{R}^{n_A d}$$

The paper revolves around a core proposition (Proposition 1, closed-form forgetting predictor), from which three structural consequences are derived—low-rank forgetting, precise linearization under frozen linear heads, and the Kronecker decomposition locking the vulnerable rank. Finally, a spectral regularization probe is derived to validate the diagnosis. All Jacobians and kernels are evaluated at $\theta_A$: $J_A := \nabla_\theta f_A|_{\theta_A}$, $J_B := \nabla_\theta f_B|_{\theta_A}$, with kernels $K_{AA}=J_A J_A^\top$, $K_{BB}=J_B J_B^\top$, and $K_{AB}=J_A J_B^\top$.

### Key Designs

**1. Closed-form forgetting predictor: Calculating forgetting magnitude and direction before training**

The objective for task B is MSE with an $L_2$ drift penalty: $\mathcal{L}_B(\theta) = \frac{1}{2}\|f_B(\theta)-y_B\|^2 + \frac{\lambda}{2}\|\theta-\theta_A\|^2$. Under NTK linearization ($f_B(\theta_A+\delta)\approx f_B(\theta_A)+J_B\delta$), $\mathcal{L}_B$ becomes a convex quadratic whose optimal solution satisfies the ridge regression normal equation. Using the push-through identity to convert this to the dual form $\delta^\star = -J_B^\top(K_{BB}+\lambda I)^{-1} r_B$ and substituting it back into the task A linearization yields:

$$\Delta f_A = -K_{AB}(K_{BB}+\lambda I)^{-1} r_B$$

where $r_B := f_B(\theta_A) - y_B$ is the residual of task B at $\theta_A$. The power of this predictor lies in the fact that $K_{AB}$, $K_{BB}$, and $r_B$ depend only on $\theta_A$ and the data from both tasks, meaning forgetting can be calculated **without actually training on task B**. Experimentally, the cosine similarity between this and the real forgetting vector is $>0.99$ (Split-MNIST/CIFAR-10). In PEFT-CL systems with frozen backbones and linear heads, it is structurally exact, with $1-\cos\text{sim}$ as low as $10^{-6}$ on ViT-B/16 and DINOv2.

**2. Low-rank structure: Forgetting concentrates on a few eigenmodes of $K_{AA}$**

Expanding the predictor in the eigenbasis of $K_{AA}=U\Lambda U^\top$ as $\Delta f_A=\sum_i c_i u_i$, and using the SVD $J_A=U\Sigma V_A^\top$, the coefficient for mode $i$ is:

$$c_i = -\sigma_i\, v_{A,i}^\top J_B^\top (K_{BB}+\lambda I)^{-1} r_B$$

Thus, $|c_i|$ inherits the decay of $\sigma_i$, modulated by the alignment of $v_{A,i}$ with the residual-driven task B direction. Since $\Lambda$ decays rapidly due to spectral bias and as long as cross-task alignment factors do not concentrate abnormally on small $\sigma_i$ directions, $\Delta f_A$ concentrates on the top eigenmodes of $K_{AA}$—defined as the **vulnerable subspace** $\mathrm{span}(u_1,\dots,u_k)$. The authors clarify that it is not "learning" that is low-rank, but specifically "forgetting." The novelty of Eq. (5) is that $c_i$ is determined by a **specific cross-task product**, combining "spectral bias decay" and "cross-task alignment." In practice, 1–6 modes carry 50–90% of the forgetting energy.

**3. Kronecker Decomposition: Locking the vulnerable rank to $k^\star \approx C \cdot k_G$**

When $f$ is linear in $\theta$ (frozen backbone + trainable linear head), the Taylor expansion is an equality, and Proposition 1 holds exactly. For frozen features $\phi(x)\in\mathbb{R}^F$ and a linear head $W\in\mathbb{R}^{C\times F}$, the MSE Jacobian is block-diagonal across output classes, leading to:

$$K_{AA} = I_C \otimes G, \qquad G_{ij}=\phi(x_i)^\top\phi(x_j)$$

where $G$ is the feature Gram matrix for a single output. Consequently, each eigenvalue of $G$ has a multiplicity of $C$ in $K_{AA}$, and the dimension of the vulnerable subspace scales as:

$$k^\star \approx C \cdot k_G$$

where $k_G$ is the effective rank of $G$ (the number of dominant eigenvalues). Empirically $k_G \in [1, 5]$. This provides a clean design rule: for $C=10$ outputs, $k^\star \in [10, 50]$, and for $C=100$, $k^\star \in [100, 500]$. This scaling law explains why for large $C$, "broad" penalties like LwF essentially cover the $C \cdot k_G$ dimensional vulnerable subspace, causing targeted and broad function-space methods to converge.

**4. Spectral Regularization Probe: Suppressing drift only in the vulnerable subspace**

Since forgetting concentrates in the top-$k$ eigenspace of $K_{AA}$, penalties are applied only there. After training task $\tau$, the top-$k$ eigenvectors $\{u_j^{(\tau)}\}$ of $K_{\tau\tau}$ are calculated on $n_{\text{probe}}$ probe samples, and reference outputs $f_\tau^{\text{ref}}$ are stored. The training loss for subsequent tasks is:

$$\mathcal{L}(\theta)=\mathcal{L}_{\text{new}}(\theta)+\sum_{\tau<t}\frac{\mu}{k}\sum_{j=1}^k\left(u_j^{(\tau)\top}\left[f_\tau(\theta)-f_\tau^{\text{ref}}\right]\right)^2$$.

Drift in the $(nd-k)$ dimensional complement space is unconstrained, preserving plasticity—a key difference from EWC (which constrains all $p$ parameter directions) and LwF (which constrains all $nd$ output directions). The authors emphasize that spectral regularization is a **diagnostic tool/probe** to validate the theory rather than the central claim of the paper.

## Key Experimental Results

Three benchmarks: Split-MNIST (5 tasks, MLP), Split-CIFAR-10 (5 tasks, ~200k param CNN), each with **shared-head** (single output layer for all classes) and multi-head variants; Split-CIFAR-100 using frozen ResNet-18 (10 tasks, linear head PEFT-CL). Accuracy (Acc↑) and Forgetting (Fgt↓) are reported.

### Main Results: Parameter-space methods fail on shared heads; function-space methods succeed

| Method | MNIST Acc↑ | MNIST Fgt↓ | CIFAR-10 Acc↑ | CIFAR-10 Fgt↓ |
|------|-----------|-----------|--------------|--------------|
| No reg | 19.7 | 99.6 | 17.5 | 85.8 |
| EWC (param space) | 19.8 | 99.5 | 18.7 | 88.7 |
| SI (param space) | 22.2 | 96.4 | 17.8 | 85.5 |
| Replay | 56.7 | 53.3 | 21.5 | 82.7 |
| DER++ | 65.9 | 42.0 | 28.4 | 78.6 |
| LwF (func space, broad) | 80.3 | 23.9 | 29.0 | 77.4 |
| **Spectral (func space, targeted)** | **77.9** | **11.3** | **32.0** | **51.6** |

Spectral significantly outperforms LwF on CIFAR-10 ($p=0.002$). Parameter-space methods like EWC and SI perform no better than "No reg" (Acc ~20%) because the diagonal Fisher matrix is anisotropic in parameter coordinates and poorly aligned with the rank-$k$ projector $U_k U_k^\top$ in output space.

### Drift Decomposition: Direct verification of the low-rank claim

| Method | Vuln. Subspace Drift↓ | Complement Drift | Ratio |
|------|-------------|----------|------|
| No reg | 305.1 | 142.8 | 0.5:1 |
| EWC (best) | 262.5 | 92.9 | 0.4:1 |
| Replay (100) | 278.9 | 144.5 | 0.5:1 |
| LwF (50p) | 161.4 | 124.4 | 0.8:1 |
| **Spectral (μ=10)** | **2.0** | 146.7 | **75:1** |

On Split-MNIST with a shared head and $k=10$, Spectral regularizer reduces drift in the vulnerable subspace by $150\times$ (305.1 $\rightarrow$ 2.0) while leaving the complement subspace drift almost unchanged. This confirms the structural difference predicted by the theory.

### Key Findings
- **Forgetting lives in output space, not parameter space**: The failure of parameter-space methods on shared heads is consistent with the claim that vulnerable directions are in output space.
- **Function-space methods converge under scaling laws**: On Split-CIFAR-100 ($C=100$), Spectral ($k=100$) and LwF are indistinguishable (39.9% vs 39.8%), matching the $k^\star \approx C \cdot k_G$ prediction.
- **Extremely accurate predictor**: The closed-form forgetting vector has a cosine similarity $>0.99$ with the real vector.

## Highlights & Insights
- **"Predicting forgetting before training"**: The most significant finding—without running task B training, forgetting can be calculated in closed form using the kernel and residuals at $\theta_A$.
- **Diagnosis of EWC's failure**: EWC fails not because of insufficient strength, but because it spreads protection across parameter coordinates that do not align with the vulnerable output subspace.
- **The $k^\star \approx C \cdot k_G$ scaling law**: This provides a computable metric for "how many dimensions to protect" and explains why "broad" methods like LwF work well when $C$ is large.
- **Clean Theory-Probe-Verfication loop**: Spectral regularization serves as a verification tool, proving the "targeted protection" mechanism via drift decomposition ratios (75:1 vs 0.5:1).

## Limitations & Future Work
- **NTK/Linearization Premise**: The core proposition is exact under NTK linearization. Its reliability for non-linear adapters (LoRA) or full fine-tuning far from the linear regime is not fully explored.
- **Regression Setup**: The theory is derived for regression. Extension to cross-entropy and long task sequences remains an open problem.
- **Dependency on Probe Sets**: The estimation of $k_G$ and vulnerable eigenvectors depends on the quality and size of the probe set.
- **Frozen Representation Assumption**: The analysis assumes frozen features. In real continual learning, representations evolve, which would change the eigenstructure of $K_{AA}$.

## Related Work & Insights
- **vs. NTK Overlap Theory**: Earlier works used similar expressions but only derived scalar magnitude bounds. This work treats it as a structured operator in output space.
- **vs. Parameter-Space Methods**: These methods constrain parameter drift; this work proves that the vulnerability resides in output space.
- **vs. Function Regularization (LwF)**: LwF regularizes all output directions; this work concentrates the penalty in the vulnerable subspace identified by the NTK.
- **vs. Concurrent Mechanistic Analysis**: Unlike works that decompose forgetting by model components, this work provides a function-space decomposition by NTK eigenmodes.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Reframing forgetting as low-rank interference in function space with a pre-computable predictor is a fresh perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Strong verification of the low-rank claim through drift decomposition, though focused on small to mid-scale benchmarks.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous logic with clear clarification of subtle theoretical points.
- Value: ⭐⭐⭐⭐ Provides a mechanistic diagnostic metric for designing selective protection in continual learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Active Learning with Low-Rank Structure for Data Selection](active_learning_with_low-rank_structure_for_data_selection.md)
- [\[ICLR 2026\] Understanding the Dynamics of Forgetting and Generalization in Continual Learning via the Neural Tangent Kernel](../../ICLR2026/learning_theory/understanding_the_dynamics_of_forgetting_and_generalization_in_continual_learnin.md)
- [\[ICML 2026\] On the Robustness of Langevin Dynamics to Score Function Error](on_the_robustness_of_langevin_dynamics_to_score_function_error.md)
- [\[ICML 2026\] Performative Learning Theory](performative_learning_theory.md)
- [\[ICML 2026\] On the Learnability of Test-Time Adaptation: A Recovery Complexity Perspective](on_the_learnability_of_test-time_adaptation_a_recovery_complexity_perspective.md)

</div>

<!-- RELATED:END -->

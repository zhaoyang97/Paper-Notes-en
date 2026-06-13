---
title: >-
  [Paper Note] TPV: Parameter Perturbations Through the Lens of Test Prediction Variance
description: >-
  [ICML2026][Optimization][Prediction Variance] The authors formalize the "local prediction sensitivity of a trained model to parameter perturbations" as Test Prediction Variance (TPV). They prove that under a first-order…
tags:
  - "ICML2026"
  - "Optimization"
  - "Prediction Variance"
  - "Parameter Perturbation"
  - "Benign Overfitting"
  - "Wide Minima"
  - "Training Set Model Selection"
date: 2026-05-08
content_hash: 53028ba42a3cc6b7
---

# TPV: Parameter Perturbations Through the Lens of Test Prediction Variance

**Conference**: ICML2026  
**arXiv**: [2512.11089](https://arxiv.org/abs/2512.11089)  
**Code**: Marked as "Code Available Here" at the end of the paper (specific repository requires checking the arXiv page)  
**Area**: optimization  
**Keywords**: Prediction Variance, Parameter Perturbation, Benign Overfitting, Wide Minima, Training Set Model Selection  

## TL;DR
The authors formalize the "local prediction sensitivity of a trained model to parameter perturbations" as Test Prediction Variance (TPV). They prove that under a first-order approximation, it simplifies to the trace form $\mathrm{Tr}(H_{\mathrm{eff}}C)$, thereby unifying SGD noise, label noise, quantization, and pruning into a single curvature-covariance framework. Furthermore, they provide a stability theorem that allows TPV to be estimated using only the training set, leading to the label-free pruning criterion JBR and a model selection signal that requires no test labels.

## Background & Motivation
**Background**: Understanding the robustness of trained networks to "post-training noise" (SGD convergence noise, finite precision, label noise during fine-tuning, pruning masks) currently involves various theoretical branches: wide/flat minima theory (Keskar et al.), implicit optimization bias (Soudry et al.), benign overfitting (Bartlett et al.), and NTK theory (Jacot et al.).

**Limitations of Prior Work**: These theories use distinct tools to answer essentially the same type of question: "Which $w^\star$ does the optimizer eventually land on?" However, in practical deployment, the concern is how much the output of a given trained $w^\star$ changes when facing real perturbations. This is a local, fixed-model problem that the aforementioned theories do not unify into a single direct expression.

**Key Challenge**: Existing perspectives model variations as "re-training to obtain a different $w$," which treats weights as global variables. In contrast, real noise (small-step SGD jitters, quantization, mask-based pruning) acts on the neighborhood of a fixed $w^\star$, essentially representing local parameter perturbations. These two perspectives differ significantly in statistical meaning, and mixing them can lead to errors.

**Goal**: To define a quantity that directly characterizes the "sensitivity of output to parameter perturbations under a fixed $w^\star$"; to prove it can be estimated using the training set; to unify various noise mechanisms using it; and to apply it to practical tasks (pruning, model selection).

**Key Insight**: By applying a first-order Taylor expansion near $w^\star$, $f_{w^\star+\delta w}(x)\approx f_{w^\star}(x)+J(x)\delta w$, the prediction variance naturally decomposes into two parts: the "geometry of the model Jacobian" and the "perturbation covariance." This decomposition matches the physical structure of most post-training noise sources—the noise mechanism determines $C$, while $H_{\mathrm{eff}}$ is a purely geometric object independent of the noise.

**Core Idea**: Use the trace form of first-order TPV, $\mathrm{Tr}(H_{\mathrm{eff}}C)$, as the unique unifying metric to compress the robustness problems of various post-training perturbations into "how the perturbation covariance $C$ couples with the Jacobian geometry $H_{\mathrm{eff}}$."

## Method

### Overall Architecture
The TPV framework is an analytical tool rather than a new training algorithm. The main logic consists of three parts:

1.  **Definition Layer**: Defines TPV as $\mathrm{TPV}:=\mathbb{E}_{x,\delta w}\bigl[\|f_{w^\star+\delta w}(x)-f_{w^\star}(x)\|^2\bigr]$. Under first-order approximation, it becomes $\mathrm{Tr}(H_{\mathrm{eff}}C)$, where $H_{\mathrm{eff}}:=\mathbb{E}_x[J(x)^\top J(x)]$ is the second moment of the model Jacobian and $C:=\mathbb{E}_R[\delta w\delta w^\top]$ is the perturbation covariance.
2.  **Stability Layer**: Proves that under over-parameterization and isotropic perturbations, the difference between $\mathrm{TPV}(w^\star;X_{\mathrm{tr}})$ and $\mathrm{TPV}(w^\star;X_{\mathrm{te}})$ is bounded by a small term unrelated to generalization, allowing the training set to estimate the test set TPV.
3.  **Mechanism Layer**: Explicitly calculates the form of $C$ for each noise source (label noise, SGD steady-state noise, quantization), expresses TPV as an interpretable function, and recovers existing theoretical conclusions (benign overfitting, wide minima).

Downstream applications include the JBR pruning criterion and training-set-based model selection.

### Key Designs

1.  **Trace Decomposition $\mathrm{Tr}(H_{\mathrm{eff}}C)$ as a Unified Lens**:
    - **Function**: Maps heterogeneous perturbation sources (SGD noise, label noise, quantization, pruning) onto the same scalar.
    - **Mechanism**: First write $\bigl(J(x)\delta w\bigr)^2=\mathrm{Tr}\bigl(J(x)^\top J(x)\delta w\delta w^\top\bigr)$ for each $x$, then take dual expectations over $x$ and $\delta w$. Since they are independent, the result is $\mathrm{Tr}(H_{\mathrm{eff}}C)$. $H_{\mathrm{eff}}$ is a label-free geometric quantity (depending only on test inputs and the trained Jacobian), while $C$ is entirely determined by the noise mechanism. Substituting specific $C$: for SGD steady-state noise $C\propto (\eta/b)\nabla^2 L(w^\star)$, for quantization $C=\tfrac{\delta^2}{12}I$, and for label noise $C$ is related to the pseudo-inverse of $J^\top J$.
    - **Design Motivation**: Previous analyses often required different mathematical tools for each noise type. Here, once $C$ is calculated, the rest of the analysis follows the same template, naturally explaining why the "wide minima hypothesis" holds for both SGD and quantization (where $C \propto$ Hessian) but not for label noise (where $C$ follows the Jacobian direction, independent of the Hessian spectrum).

2.  **TPV Trace Stability Theorem (Estimating Test TPV from Training Set)**:
    - **Function**: Explains why sharpness/TPV calculated on the training set can predict robustness at test time.
    - **Mechanism**: Theorem 3.1 provides the bound $\bigl|\mathrm{TPV}(w^\star;X_{\mathrm{tr}})-\mathrm{TPV}(w^\star;X_{\mathrm{te}})\bigr|\le c_1\mathrm{Tr}(C)$, where $c_1=\tfrac{n_{\mathrm{tr}}+n_{\mathrm{te}}}{p}\varepsilon_{\mathrm{NTK}}+o(1)$. The proof relies on two points: (a) NTK remains largely stable throughout training (Jacot/Allen-Zhu); (b) $H_{\mathrm{eff}}(w_0;X)$ concentrates to the population quantity for both training/test sets at initialization (Law of Large Numbers). Combining these implies that $H_{\mathrm{eff}}$ has minimal differences between training and test sets throughout training, thus $\mathrm{Tr}(H_{\mathrm{eff}}C)$ also has minimal differences.
    - **Design Motivation**: Traditional sharpness literature often implicitly uses "training set sharpness to approximate test sharpness" for label-free model selection without rigorous proof. This paper provides the first explicit bound for "training TPV → test TPV," which is independent of the generalization gap $L_{\mathrm{test}}-L_{\mathrm{train}}$. This is crucial: conventional intuition suggests only well-generalized models can use training statistics to represent test statistics, but TPV stability shows otherwise.

3.  **Non-linear Benign Overfitting under Label Noise (Theorem 4.2)**:
    - **Function**: Seamlessly extends the linear regression conclusion $\sigma_\varepsilon^2 \mathrm{Tr}((XX^\top)^{-1})$ to non-linear deep networks.
    - **Mechanism**: Re-derives the process where "training labels are corrupted by $\varepsilon_i$ and the model selects the minimum-norm solution" under first-order linearization, yielding $\mathrm{TPV}_{\mathrm{label}}\approx\sigma_\varepsilon^2\sum_{i=1}^r B_{ii}/s_i^2$, where $s_i$ are the non-zero singular values of the training Jacobian and $B_{ii}=(V^\top H_{\mathrm{eff}}V)_{ii}$ represents the projection energy of the test distribution Jacobian onto the right singular vectors of the training Jacobian. The linear case is a subset (replacing Jacobian with data matrix $X$).
    - **Design Motivation**: Classic benign overfitting only provides linear model analysis. This formula directly shows that label noise sensitivity in non-linear deep networks is dominated by "whether test Jacobian energy exists in the directions of small singular values." NTK theory guarantees a lower bound on the minimum singular value of over-parameterized networks, suppressing $\sum B_{ii}/s_i^2$—this is the geometric explanation of why over-parameterization reduces label noise sensitivity.

### Loss & Training
TPV itself is not a training objective. The only training-related conclusion provided is Theorem 4.3: under squared loss, the TPV of SGD steady-state noise is approximately $\tfrac{\eta\sigma_\varepsilon^2}{2b}\mathrm{Tr}(\nabla_w^2 L(w^\star))$, i.e., "Learning rate/Batch size ratio × Squared residual × Hessian trace," recovering the wide minima intuition. Quantization noise yields $\tfrac{\delta^2}{12}\mathrm{Tr}(\nabla_w^2 L(w^\star))$.

## Key Experimental Results

### Main Results
The experiments are split into "TPV stability validation" and "TPV–test loss correlation validation."

| Experimental Scenario | Configuration Scale | Key Observation | Meaning |
|---------|---------|---------|------|
| Synthetic TPV Stability | 324 configs × 2 noise sources (Label / SGD), $n_{\mathrm{train}}=1000$ | TPV spans 5 orders of magnitude; all points stay close to the $y=x$ diagonal, even for width = 1 | Stability exceeds theorem requirements |
| Synthetic Small Sample Breakpoint | Same as above but width = 256, $n_{\mathrm{train}}\in\{10,1000\}$ | Severe deviation from the diagonal at $n=10$; close fit at $n=1000$ | Stability fails only when samples are too few |
| CIFAR-10 MobileNetV2 | Multiple width multipliers | Stability held across architectures | Holds on real data |
| CIFAR-100 Logit Noise + Fine-tuning | MobileNetV2 with increasing width | Width↑ → Training/Test TPV both↓ → Test CE also↓ | TPV correlates positively with generalization in low training loss regions |

### Ablation Study

| Ablation Dimension | Setting | Conclusion |
|---------|------|------|
| Noise Variance $\sigma_\varepsilon$ | Synthetic $\sigma=0.01$ vs $0.1$ | At $\sigma=0.1$, training TPV saturates to $\sigma^2$ while test TPV still drops with width—stability breaks under large perturbations, consistent with looser theorem bounds |
| Regularization Scan | Weight decay / dropout / label smoothing on CIFAR-10 | TPV and test loss show a U-shape: they decrease together in low loss regions but diverge in high loss regions (underfit region has small TPV but large test loss) |
| Training Trajectory | ResNet-18 / CIFAR-100 / 30% Label Noise | The peak TPV moment exactly separates the underfitting phase from the epoch-wise double descent phase |
| Pruning Criteria Comparison | JBR vs 7 baselines (Jacobian / L1 / BN Scale / FPGM / WHC / Taylor / Random), CIFAR-10/100 + ImageNet | JBR matches or exceeds Prev. SOTA in all 4 settings |
| Training Recipe Robustness | MLP with 7 weight decay settings, Sharpness vs Label-noise TPV | Sharpness does not correlate positively with label noise sensitivity; TPV correctly identifies the most robust recipe |

### Key Findings
- TPV stability is **decoupled** from whether the model generalizes well: models with large generalization gaps still fit the diagonal, overturning the intuition that "using training statistics to predict test statistics requires good generalization."
- The U-shaped relationship between TPV and test loss holds across architectures and regularization methods. Through the "argmax TPV epoch" observable, one can partition the underfit and double descent phases on a single training curve.
- **Unification** of the wide minima hypothesis and benign overfitting: two conclusions long thought to stem from different mechanisms are shown to correspond to $C\propto$ Hessian and $C\propto J^\dagger J^{\dagger\top}$ perturbation covariance forms within the TPV framework.

## Highlights & Insights
- **Unification**: A single $\mathrm{Tr}(H_{\mathrm{eff}}C)$ formula recovers wide minima, benign overfitting, quantization sensitivity, and pruning importance just by "changing $C$." This idea—constant form, varying noise covariance—can be transferred to any "small parameter perturbation" problem.
- **Training Set is Sufficient**: TPV stability elevates "label-free model selection" from an empirical rule to a theoretical conclusion and does not rely on the generalization gap—meaning models can be selected using pure training data in label-expensive scenarios (medicine, privacy).
- **U-shape Mechanism Explanation**: Attributes why model selection curves are sometimes monotonic and sometimes U-shaped to the relative dominance of bias and variance over $L_{\mathrm{train}}$, providing a clear phase division for future hyperparameter auto-tuning.

## Limitations & Future Work
- Theorem 3.1 rigorously holds only for isotropic $C$ and the over-parameterization limit. For anisotropic perturbations (e.g., when label noise $C$ strongly couples with Jacobian geometry), empirical stability is required.
- First-order Taylor approximation fails under large perturbations or when $\sigma_\varepsilon$ is large and the solution leaves the linearized neighborhood; Appendix D.2/D.3 admits that non-linear label noise TPV is difficult to estimate precisely in practice.
- TPV is a quantity within the Mean Squared Error (MSE) framework. Classification tasks practically use logit perturbations or fine-tuning; the paper provides an empirical bridge using CE and logit noise but lacks a formal definition. Extending to structured outputs (detection, generation) remains an open problem.
- Improvement directions: (i) Extending stability proofs from the NTK regime to the feature learning regime; (ii) Incorporating input distribution shift into $C$; (iii) Providing numerically stable practical estimation algorithms for label noise TPV.

## Related Work & Insights
- **vs Wide Minima/Flat Minima (Keskar et al., 2017; Foret et al., 2020 SAM)**: Traditional theory states "small $\mathrm{Tr}(\nabla^2 L)$ leads to good generalization." TPV reformulates this as "$H_{\mathrm{eff}}$ and $C$ produced by SGD/quantization jointly determine robustness," and explains why the reparameterization counterexample of Dinh et al. (2017) did not truly overturn the wide minima intuition—once the noise physics is fixed, $C$ changes alongside the reparameterization.
- **vs Benign Overfitting (Bartlett et al., 2020)**: They prove $\sigma^2\mathrm{Tr}((XX^\top)^{-1})$ under linear regression; Theorem 4.2 replaces $X$ with the Jacobian and the data condition number with the Jacobian spectral condition number, extending it to arbitrary non-linear deep networks.
- **vs Bordelon & Pehlevan (2023) Finite Width Prediction Fluctuations**: They use DMFT to describe the global variance of training dynamics; TPV is a post-training local quantity, making them complementary.
- **vs Jacobian Criterion (Chen et al., 2025) Pruning Criterion**: JBR is re-derived under TPV geometry. The resulting score is homologous to JC but incorporates $\delta_u$, the "output direction against the predicted class negative log-probability," thus preserving predictions for samples already correctly classified by the training set—Appendix H.1 provides specific difference analysis.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ A single $\mathrm{Tr}(H_{\mathrm{eff}}C)$ recovers four long-independent robustness theories, and TPV stability is the first of its kind in literature.
- Experimental Thoroughness: ⭐⭐⭐⭐ Synthetic experiments with 324 configs + CIFAR/ImageNet/NLU coverage, though specific numbers for pruning and model selection are mainly in the appendix.
- Writing Quality: ⭐⭐⭐⭐ Clear main logic, theorems align well with mechanisms; high formula density, and some derivations (e.g., minimum-norm solution assumption in Theorem 4.2) require the appendix for full clarity.
- Value: ⭐⭐⭐⭐⭐ "Using Training TPV for Test Model Selection" could significantly reduce labeling costs if implemented in production pipelines, and its theoretical unification provides a clear scaffold for future post-training robustness research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Balanced LoRA: Removing Parameter Invariance to Accelerate Convergence](balanced_lora_removing_parameter_invariance_to_accelerate_convergence.md)
- [\[ICLR 2026\] Personalized Collaborative Learning with Affinity-Based Variance Reduction](../../ICLR2026/optimization/personalized_collaborative_learning_with_affinity-based_variance_reduction.md)
- [\[ICML 2026\] Test time training enhances in-context learning of nonlinear functions](test_time_training_enhances_in-context_learning_of_nonlinear_functions.md)
- [\[ICLR 2026\] Test-Time Meta-Adaptation with Self-Synthesis](../../ICLR2026/optimization/test-time_meta-adaptation_with_self-synthesis.md)
- [\[ICML 2026\] Colorful Pinball: Density-Weighted Quantile Regression for Conditional Guarantee of Conformal Prediction](colorful_pinball_density-weighted_quantile_regression_for_conditional_guarantee_.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Possibilistic Predictive Uncertainty for Deep Learning
description: >-
  [ICML 2026][epistemic uncertainty] This paper replaces the Bayesian probability framework with possibility theory and proposes DAPPr—a method that projects the possibilistic posterior in parameter space onto the predicti…
tags:
  - "ICML 2026"
  - "epistemic uncertainty"
  - "possibility theory"
  - "Dirichlet"
  - "second-order predictor"
  - "EDL"
date: 2026-05-08
content_hash: 4b7e00b398759e64
---

# Possibilistic Predictive Uncertainty for Deep Learning

**Conference**: ICML 2026  
**arXiv**: [2605.00600](https://arxiv.org/abs/2605.00600)  
**Code**: https://github.com/MaxwellYaoNi/DAPPr  
**Keywords**: epistemic uncertainty, possibility theory, Dirichlet, second-order predictor, EDL

## TL;DR
This paper replaces the Bayesian probability framework with possibility theory and proposes DAPPr—a method that projects the possibilistic posterior in parameter space onto the prediction space via supremum, fits it with a learnable Dirichlet possibility function, and ultimately yields a cognitive uncertainty modeling approach that requires only 10 lines of code, can directly replace cross-entropy, and outperforms the EDL family in OOD detection.

## Background & Motivation
**Background**: It is well known that deep networks are overconfident on out-of-distribution (OOD) samples. Mainstream cognitive uncertainty modeling follows two paths: Bayesian deep learning (BNN / MC Dropout / Deep Ensemble) and second-order predictors (EDL / PostNet / Prior Networks).

**Limitations of Prior Work**: The Bayesian approach is theoretically rigorous but requires posterior marginalization in high-dimensional parameter spaces, which is computationally expensive and hard to scale. Second-order predictors are efficient but their objectives are mostly heuristic, lacking strict derivation from probability axioms. EDL has even been shown to exhibit the pathological behavior of "more data leads to higher uncertainty."

**Key Challenge**: There is a trade-off between theoretical rigor and computational feasibility—Bayesian methods are rigorous but expensive, second-order methods are cheap but ad hoc. The authors argue that the root cause is treating epistemic uncertainty as probability, while the "sum to 1" constraint of probability distributions is more suitable for aleatoric randomness, not for representing "ignorance."

**Goal**: (1) Find a rigorous uncertainty representation framework that does not require integration over parameter space; (2) Derive a closed-form training objective; (3) Directly compare with the EDL family on standard benchmarks.

**Key Insight**: The authors draw from possibility theory, proposed by Zadeh in 1978 but largely ignored in deep learning. It replaces integration with supremum and sum-to-1 normalization with max-normalization, making it naturally suited for expressing "which hypotheses cannot be excluded"—a form of epistemic information.

**Core Idea**: Project the possibilistic posterior of model parameters onto the simplex via supremum, then use a Dirichlet possibility function for parametric approximation on the simplex. The entire pipeline can be closed in closed-form using cross-entropy.

## Method
DAPPr's elegance lies in compressing what would be a constrained optimization in high-dimensional parameter space ("projected posterior") into 10 lines of PyTorch code, via the trio of over-parameterized assumption, Dirichlet parameterization, and the Danskin theorem.

### Overall Architecture
Input is a standard classification sample $(\bm{x}, \bm{y})$; the model $\Phi'_{\bm{\psi}}$ outputs Dirichlet parameters $\bm{\alpha} = \mathrm{softplus}(\mathrm{logits}) + 1$; the Dirichlet possibility function $g_{\bm{\psi}}(\bm{p}|\bm{x})$ is defined. At inference, $1 - \max_k \alpha_k / \alpha_0$ computes aleatoric uncertainty, and $K / \alpha_0$ computes epistemic uncertainty ($\alpha_0 = \sum_k \alpha_k$ is the total evidence). Training constructs a two-step "projection + approximation" pipeline, ultimately deriving a closed-form surrogate loss under cross-entropy.

### Key Designs

1. **Possibilistic Posterior + Supremum Projection**:

    - **Function**: Under a uniform prior, define the parameter space possibilistic posterior $\pi(\bm{\theta}|\mathcal{D}) = \exp(-L(\bm{\theta};\mathcal{D})) / \sup_{\bm{\theta}'}\exp(-L(\bm{\theta}';\mathcal{D}))$, where lower loss implies higher plausibility. Then, project to the simplex using the possibilistic change-of-variable: $g^*_{\bm{x}}(\bm{p}|\mathcal{D}) = \sup\{\pi(\bm{\theta}|\mathcal{D}) : \Phi_{\bm{\theta}}(\bm{x}) = \bm{p}\}$.
    - **Mechanism**: Replace the marginalization (integration) in Bayesian inference with supremum-based constrained optimization—this is the essential difference between possibility and probability theory. Using the over-parameterized assumption (a sufficiently large network can fit any single point without affecting others), it is shown that $\inf_{\Phi_{\bm{\theta}}(\bm{x})=\bm{p}} L(\bm{\theta}; \mathcal{D} \setminus \{(\bm{x},\bm{y})\}) \approx c_{\bm{x}}$ is nearly independent of $\bm{p}$, so the projected posterior simplifies to $g^*_{\bm{x}}(\bm{p}|\mathcal{D}) \propto \exp(-\ell(\bm{p}, \bm{y}))$.
    - **Design Motivation**: Integration over parameter space is the root of Bayesian computational cost; replacing it with supremum plus the over-parameterization assumption reduces it to a sample-wise leave-one-out infimum, which is then approximated as a constant—a clever two-stage simplification.

2. **Maxitive Pseudo-divergence Training Objective**:

    - **Function**: Use $D_{\mathrm{max}}(f\|g) = \max_{\theta} \log(f(\theta)/g(\theta))$ to measure the deviation between two possibility functions, and define the training objective $\mathcal{L}(\bm{\psi}; \mathcal{D}) = \mathbb{E}_{\bm{x}}[\max_{\bm{p}}(\log g_{\bm{\psi}}(\bm{p}|\bm{x}) - \log g^*_{\bm{x}}(\bm{p}|\mathcal{D}))]$, essentially penalizing the maximum pointwise overestimation of the projected posterior by the learned function.
    - **Mechanism**: This is a min-max problem, where the inner maximizer $\bm{p}^*$ depends on $\bm{\psi}$. The Danskin theorem equates the outer gradient to the gradient at the inner maximizer with respect to $\bm{\psi}$. Under Dirichlet parameterization, the inner max of the cross-entropy loss has a closed-form solution $\tilde{\bm{p}}^* = (\bm{\alpha} - \bm{y}) / (\alpha_0 - 1)$, requiring $\alpha_k > 1$ (enforced by softplus + 1).
    - **Design Motivation**: The combination of "maxitive divergence instead of KL," "Danskin for min-max," and "Dirichlet parameterization for closed-form" transforms an abstract possibility theory framework into a differentiable, trainable, and simple loss—the key engineering contribution of the paper.

3. **Spurious Evidence Regularization**:

    - **Function**: Add a regularization term $\mathcal{R}(\bm{x}) = \|(\bm{1} - \bm{y}) \odot \bm{\alpha}\|_2^2$ outside the cross-entropy surrogate to penalize evidence assigned to incorrect classes.
    - **Mechanism**: The surrogate objective encourages arbitrarily precise fitting for each sample, so $\alpha_0$ may grow unbounded, corresponding to unrealistically high evidence. This regularizer penalizes only the $\alpha$ on wrong classes, keeping total evidence controlled without hindering the growth of correct-class evidence.
    - **Design Motivation**: A common issue in EDL methods is the difficulty of controlling evidence; here, a simple mask + L2 directly limits overconfidence on incorrect classes, avoiding the complex Fisher regularization in EDL.

### Loss & Training
The final training objective (implementable in 10 lines of PyTorch):

$\ell_{\bm{\psi}}(\bm{x}) = \alpha_0 \log \alpha_0 + \sum_k \alpha_k \log(\tilde{p}^*_k / \alpha_k) + \lambda \|(\bm{1} - \bm{y}) \odot \bm{\alpha}\|_2^2$

where $\tilde{\bm{p}}^* = (\bm{\alpha} - \bm{y} + \epsilon) / (\alpha_0 - 1)$ is detached to prevent gradient flow. $\lambda$ controls the regularization strength and is the only explicit hyperparameter.

## Key Experimental Results

### Main Results
On MNIST / CIFAR-10 / CIFAR-100, compared with SOTA EDL family ($\mathcal{I}$-EDL / R-EDL / $\mathcal{F}$-EDL) and Bayesian baselines (MC Dropout / DUQ / PostNet):

| Dataset | Metric | DAPPr | $\mathcal{F}$-EDL | R-EDL | $\mathcal{I}$-EDL | EDL |
|------|------|------|------|------|------|------|
| MNIST Test Acc | ↑ | 99.26 | 99.30 | 99.33 | 99.21 | 98.22 |
| MNIST Conf AUPR | ↑ | 99.99 | 99.93 | 99.99 | 99.98 | 99.99 |
| MNIST→KMNIST OOD | ↑ | **98.81** | 98.74 | 98.69 | 98.33 | 96.31 |
| MNIST→FMNIST OOD | ↑ | **99.55** | 99.31 | 99.29 | 98.86 | 98.08 |

DAPPr consistently outperforms the strongest EDL variants in OOD detection, with accuracy and confidence calibration on par.

### Ablation Study
The paper empirically validates the over-parameterization assumption, scans the spurious evidence regularization strength $\lambda$, and compares on more complex benchmarks such as long-tail distribution, distribution shift detection, and fine-grained classification:

| Configuration | Key Effect | Description |
|------|------|------|
| No regularization $\lambda = 0$ | $\alpha_0$ grows unbounded | Arbitrarily precise fitting per sample, undermining uncertainty expression |
| Large $\lambda$ | Evidence suppressed | Overall uncertainty too high, slight drop in accuracy |
| Moderate $\lambda$ | Best trade-off | Highest OOD AUPR |
| Eq. (11) Approximation Validation | Leave-one-out loss nearly independent of $\bm{p}$ | Empirical support for over-param assumption |

### Key Findings
- On OOD detection, where epistemic uncertainty is truly critical, DAPPr consistently surpasses all EDL variants, indicating that the objective derived from possibility theory is more sensitive in OOD scenarios than heuristic EDL.
- The spurious evidence regularizer is not just an engineering trick, but theoretically caps the unbounded behavior of overfitting single samples, thus significantly impacting final calibration.
- The closed-form $\tilde{\bm{p}}^*$ makes training cost identical to standard cross-entropy, with no ensemble or sampling overhead, and can directly replace existing pipelines.

## Highlights & Insights
- Introducing possibility theory into deep uncertainty is the paper's main conceptual contribution—over the past decades, the field has almost exclusively considered uncertainty within the probability theory framework, while the max operator in possibility theory naturally fits the epistemic semantics of "cannot be excluded."
- The Danskin theorem is elegantly applied here: collapsing a min-max problem to a single-layer gradient at the inner maximizer, avoiding the instability of GAN-style adversarial training.
- The fact that 10 lines of PyTorch code can drop-in replace cross-entropy is highly engineering-friendly, with virtually zero migration cost, and can greatly promote adoption in the community.
- The over-parameterized assumption as a simplification trick is very powerful—it approximates a leave-one-out optimization as a constant, and this idea can be transferred to other methods involving parameter space integration (e.g., influence function, data attribution).

## Limitations & Future Work
- The over-parameterization assumption may fail in underparameterized or sample-sensitive scenarios (e.g., few-shot learning or conflicting multi-task settings); while the paper provides empirical validation, theoretical boundaries are lacking.
- The spurious evidence regularization strength $\lambda$ is the only explicit hyperparameter and still requires tuning on new datasets; future work could consider adaptive versions.
- Currently, Dirichlet approximation is only performed on the simplex for classification tasks; extending to regression or structured prediction requires new families of possibility functions.
- Comparison with calibration methods such as conformal prediction is missing; it is unclear whether DAPPr's uncertainty can be directly translated into guaranteed coverage intervals.

## Related Work & Insights
- **vs EDL family**: EDL is based on subjective logic / Dempster-Shafer theory and uses heuristic objectives; DAPPr strictly derives its objective from possibility theory and consistently outperforms the strongest EDL variants on OOD.
- **vs Bayesian deep learning (BNN/MC Dropout/Deep Ensemble)**: Bayesian approaches require ensembles or sampling; DAPPr uses single-model inference, with cost identical to standard classification, yet still expresses epistemic uncertainty.
- **vs PostNet / Natural Posterior Networks**: Those methods use normalizing flows to fit the posterior, which is complex and requires extra components; DAPPr uses Dirichlet parameterization and a closed-form maximizer, making it much simpler.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to systematically introduce possibility theory into deep cognitive uncertainty, with novel theoretical foundations.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive coverage of MNIST / CIFAR / long-tail / distribution shift / fine-grained benchmarks.
- Writing Quality: ⭐⭐⭐⭐ Rigorous and clear derivations, building step-by-step from possibility theory basics to closed-form solutions.
- Value: ⭐⭐⭐⭐⭐ SOTA OOD performance with a 10-line code replacement for cross-entropy, offering immense engineering value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Potential Field Based Deep Metric Learning](../../CVPR2025/llm_evaluation/potential_field_based_deep_metric_learning.md)
- [\[ICLR 2026\] Measuring Uncertainty Calibration](../../ICLR2026/llm_evaluation/measuring_uncertainty_calibration.md)
- [\[ICLR 2026\] Deep FlexQP: Accelerated Nonlinear Programming via Deep Unfolding](../../ICLR2026/llm_evaluation/deep_flexqp_accelerated_nonlinear_programming_via_deep_unfolding.md)
- [\[CVPR 2025\] Improving Accuracy and Calibration via Differentiated Deep Mutual Learning](../../CVPR2025/llm_evaluation/improving_accuracy_and_calibration_via_differentiated_deep_mutual_learning.md)
- [\[NeurIPS 2025\] Conformal Online Learning of Deep Koopman Linear Embeddings](../../NeurIPS2025/llm_evaluation/conformal_online_learning_of_deep_koopman_linear_embeddings.md)

</div>

<!-- RELATED:END -->

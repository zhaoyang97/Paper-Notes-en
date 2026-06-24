---
title: >-
  [Paper Note] Variational Deep Learning via Implicit Regularization
description: >-
  [ICLR 2026][Learning Theory][Implicit Regularization] This paper proposes Implicit Bias VI (IBVI): when training a variational distribution over weights, it **directly discards the KL regularization term** in the ELBO, relying solely on SGD's implicit bias to "select" the distribution. It rigorously proves that in overparameterized linear models, this implicit bias is equivalent to generalized variational inference regularized by the **2-Wasserstein distance** (rather than KL…
tags:
  - "ICLR 2026"
  - "Learning Theory"
  - "Bayesian Deep Learning"
  - "Implicit Regularization"
  - "SGD Implicit Bias"
  - "Variational Inference"
  - "2-Wasserstein"
  - "Uncertainty Quantification"
date: 2026-05-08
content_hash: 01d08b289e6befca
---

# Variational Deep Learning via Implicit Regularization

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=WsN88Ns0i6](https://openreview.net/forum?id=WsN88Ns0i6)  
**Code**: https://github.com/inferno-ml/inferno  
**Area**: Learning Theory / Bayesian Deep Learning  
**Keywords**: Implicit Regularization, SGD Implicit Bias, Variational Inference, 2-Wasserstein, Uncertainty Quantification

## TL;DR
This paper proposes Implicit Bias VI (IBVI): when training a variational distribution over weights, it **directly discards the KL regularization term** in the ELBO, relying solely on SGD's implicit bias to "select" the distribution. It rigorously proves that in overparameterized linear models, this implicit bias is equivalent to generalized variational inference regularized by the **2-Wasserstein distance** (rather than KL). This approach preserves the generalization capabilities of standard neural networks while providing well-calibrated uncertainty for free, with almost no additional computational overhead.

## Background & Motivation
**Background**: Modern deep networks generalize well despite being overparameterized and lacking explicit regularization. The mainstream explanation is "implicit regularization"—the inductive bias imposed by the architecture, hyperparameters, and optimizer (especially SGD) favors a specific class of solutions among infinite global optima with zero training error. This bias takes effect without extra computation.

**Limitations of Prior Work**: Standard networks are highly fragile on out-of-distribution (OOD) data, exhibiting overconfident predictions and sharp drops in generalization. Bayesian Deep Learning (BDL) mitigates this via "model averaging," but suffers from three chronic issues: difficulty in prior elicitation, poor scalability to large models, and the fact that explicit prior regularization combined with approximate inference often yields pathological inductive biases and uncertainty. In other words, **to obtain uncertainty, Bayesian methods use an explicit prior regularization that "masks" the implicit regularization of standard training.**

**Key Challenge**: In the standard VI objective $\ell_r(\theta)=\mathbb{E}_{q_\theta}(-\log p(y\mid w)) + \mathrm{KL}(q_\theta\,\|\,p)$, the KL term is a double-edged sword—it provides regularization towards the prior but also overrides the SGD implicit bias proven crucial for generalization. Is it possible to **eliminate this explicit KL and let the optimizer's implicit bias act as the regularizer**?

**Key Insight**: The authors observe that under overparameterization, "minimum loss" corresponds to a family of solutions rather than a single point. If one directly minimizes the expected loss $\bar\ell(\theta)=\mathbb{E}_{q_\theta}(\ell(y,f_w(X)))$ without any divergence term, the optimal solution might seemingly collapse into a delta point mass (no uncertainty). However, in overparameterized settings, **a delta point mass is only one of many optima**, and which one the model converges to is determined by SGD's initialization and parameterization.

**Core Idea**: Use SGD's implicit bias instead of explicit KL regularization to train variational networks—initialize the variational distribution at the prior and minimize only the expected loss. Theoretically, this is equivalent to selecting the distribution **closest to the prior in terms of 2-Wasserstein distance** among all distributions with zero training error, i.e., generalized variational inference.

## Method

### Overall Architecture
The overall approach of IBVI is to mirror the "standard network training" pipeline for variational networks, replacing point-estimate weights with a Gaussian variational distribution $q_\theta(w)=\mathcal{N}(w;\mu, SS^\top)$. Specifically, given an architecture $f_w$ and a variational family, the training objective reduces to optimizing only the expected loss (removing the divergence term $\lambda D(q_\theta,p)$ from the standard VI formula (3)):

$$\theta^\star \in \arg\min_\theta\ \mathbb{E}_{q_\theta(w)}\big(\ell(y, f_w(X))\big).$$

The pipeline involves three components: (1) Initializing variational parameters $\theta_0=(\mu_0, S_0)$ using the prior; (2) Minimizing the expected loss using SGD with momentum, sampling only **one** parameter $w_m\sim q_\theta$ per forward pass; (3) Obtaining $q_{\theta^\star}$ upon convergence and performing model averaging over the weight distribution for uncertainty-aware inference. Crucially, because there is no KL, all solutions interpolate training data to zero error (uncertainty collapses at training points, identical to standard networks); however, in directions away from the training data manifold, the distribution naturally reverts to the prior, providing OOD uncertainty. This mechanism is rigorously characterized by two theorems in Section 4—the core contribution, making this a **theoretical characterization + training recipe** rather than a multi-module serial pipeline.

### Key Designs

**1. Training with Expected Loss excluding KL: Preventing delta collapse from being the sole optimum**

Addressing the limitation where standard VI's KL term masks SGD bias, the authors remove the divergence regularization entirely, leaving only the expected loss $\bar\ell(\theta)=\mathbb{E}_{q_\theta(w)}(\ell(y,f_w(X)))$. While this might seem problematic as the objective is minimized at $q_\theta=\delta_{w^\star}$ (providing zero variance), the authors clarify that under overparameterization ($P>N$), zero training error corresponds to an entire family of variational distributions $q_{\theta^\star}$. **Which specific distribution is reached is determined by the optimizer's implicit bias.** The problem shifts from "will it collapse" to "which optimum does SGD's implicit bias lead us to"—leading to the theorems in the next point. This design allows the training process to remain nearly identical to standard networks, thereby **inheriting** rather than **overriding** standard implicit regularization.

**2. Characterizing SGD Implicit Bias as 2-Wasserstein Generalized Variational Inference**

This is the theoretical pillar. The authors prove that for overparameterized linear models $f_w(x)=x^\top w$ with Gaussian priors/variational families: if SGD starts from the prior initialization $q_{\theta_0}=p$, its implicit bias is equivalent to selecting the distribution closest to the prior in 2-Wasserstein distance among all minimizers of the expected loss:

$$q_{\theta^{\mathrm{GD}}_\star} = \arg\min_{q_\theta:\ \theta\in\arg\min\bar\ell(\theta)} W_2^2(q_\theta, p).$$

This matches the generalized VI objective (3) where the "regularizer is $W_2^2$ instead of KL." In **regression** (Theorem 1), this holds for SGD and SGD with momentum, and $q_{\theta^{\mathrm{GD}}_\star}$ equals the weight distribution of an "ensemble of linear models initialized independently from the prior"—explaining IBVI's similarity to Deep Ensembles. In **binary classification** (Theorem 2, linearly separable, exponential loss), the conclusion is similar: after rescaling $\theta^{\mathrm{rGD}}_t=(\frac{1}{\log t}\mu^{\mathrm{GD}}_t+P_{\mathrm{null}(X)}\mu_0,\ S^{\mathrm{GD}}_t)$, the mean parameter converges to the **L2 max-margin vector** $\hat w$ (Hard-margin SVM solution). Uncertainty collapses to zero on the data manifold but is pulled back to the prior in the null space (away from the manifold) by the $W_2$ regularizer. Using $W_2$ instead of KL is critical: for a Gaussian with variance approaching zero, KL divergence explodes to infinity, so KL **never** allows uncertainty collapse at training points; $W_2$ permits collapse, allowing IBVI to interpolate training data like a standard network while retaining prior uncertainty elsewhere.

**3. Variational μP: Transferring Hyperparameters from Small to Large Models**

Implicit bias depends on initialization and parameterization, the latter being a "major challenge in Bayesian computation." Under Standard Parameterization (SP), the optimal learning rate drifts as width increases. The authors extend **Maximal Update Parameterization (μP)** to variational networks: the $i$-th hidden unit of layer $l$, $h^{(l)}_i(x)=(\mu_i+S_i z)h^{(l-1)}(x)$, is a function of mean/covariance parameters, forward noise $z$, and previous activations. Since $S_i z$ is a sum of $R$ terms for $S\in\mathbb{R}^{P\times R}$, it is scaled by $R^{-1/2}$ via the Central Limit Theorem, alongside μP scaling for mean and covariance parameters. This brings μP's "width-independent feature learning + hyperparameter transfer" to probabilistic models—learning rates tuned on small models transfer directly to large ones. In CIFAR-10 experiments, the optimal learning rate for SP drops with width, while it remains constant for μP.

**4. Single-Sample + Low-Rank Covariance: Reducing Overhead to Near-Standard Levels**

To make uncertainty "almost free," the authors implement two strategies. First, they use **single** parameter sampling ($M=1$) per forward pass: while this increases noise in the expected loss estimate (similar to a smaller batch), convergence is maintained with small learning rates or momentum—keeping per-step overhead near-identical to standard models. Second, they use a **low-rank decomposition** for covariance $\Sigma=SS^\top$ ($S\in\mathbb{R}^{P\times R}$, $R\le P$) and apply probabilistic layers only to the input and output layers. This decomposition is theoretically required by Theorem 2 to characterize the SGD bias as generalized VI; practically, it reduces memory overhead to ~10% above standard networks with similar training times.

### Loss & Training
The training objective is the minibatched expected loss:

$$\bar\ell(\theta)\approx \frac{1}{N_b M}\sum_{n=1}^{N_b}\sum_{m=1}^{M}\ell\big(y_n, f_{w_m}(x_n)\big),\quad w_m\sim q_\theta(w).$$

Experiments consistently use SGD with momentum $\gamma=0.9$, batch size $N_b=128$, for 200 epochs, single precision, $M=1$, low-rank covariance on the last two layers, and μP where applicable.

## Key Experimental Results

### Main Results (In-Distribution Generalization + Uncertainty)
Evaluated on MNIST / CIFAR10 / CIFAR100 / TinyImageNet against standard networks and uncertainty baselines (Temperature Scaling, Laplace, Weight-space VI, SWAG, Deep Ensembles).

| Dimension | IBVI Performance | Comparison & Cost |
|------|-----------|------------|
| Test Error | Comparable to SWAG | Only Ensembles improve accuracy, but with much higher memory cost |
| NLL (Likelihood) | Significantly improved with TS, DE | LA, WSVI occasionally perform worse |
| Calibration (ECE) | Optimal alongside TS, DE | —— |
| Compute Cost | Memory +≈10%, similar training time | Significantly lower than Ensembles / WSVI |

IBVI performs similarly to Deep Ensembles in-distribution, consistent with theory (equivalence in linear models, see Proposition S1).

### Robustness (Input Corruption OOD)
Evaluated on MNISTC / CIFAR10C / CIFAR100C / TinyImageNetC (15 corruptions, averaged at max severity).

| Configuration | Key Findings |
|------|----------|
| Accuracy | Except for DE, IBVI outperforms all other methods on corrupted data |
| Uncertainty (NLL/ECE) | TS, DE, and IBVI consistently perform well; LA-ML is competitive on NLL |
| IBVI vs Ensembles | IBVI's OOD uncertainty quantification **outperforms** Ensembles across all datasets |

### Key Findings
- **Removing KL does not cause uncertainty collapse**: Provided correct initialization (at prior) and parameterization, SGD's implicit bias automatically retains prior uncertainty outside the training data—a counter-intuitive but theoretically guaranteed conclusion.
- **Fundamental difference between $W_2$ and KL**: KL diverges for zero-variance Gaussians, forbidding collapse at training points; $W_2$ allows it, enabling IBVI to interpolate training data while reverting to the prior elsewhere.
- **Hyperparameter transfer holds only for μP**: For hidden dimensions >256, transferring tuned parameters from small to large models works under μP but fails under SP.
- **Best cost-performance ratio**: IBVI ranks in the top tier for calibration/NLL/OOD robustness while adding almost zero memory or time overhead.

## Highlights & Insights
- **"Subtractive" Innovation**: While others add complex regularizers to VI, Ours removes KL entirely, relying on the optimizer's intrinsic bias. This saves computation and tuning effort, resulting in a very clean methodology.
- **Generalizing Implicit Bias from Points to Distributions**: Existing theories state SGD picks the closest minimizer to initialization in Euclidean distance; Ours upgrades this to the "closest distribution in 2-Wasserstein distance to the prior," a rigorous extension from non-probabilistic to probabilistic models.
- **Transferable Tricks**: The recipe of $M=1$ + low-rank covariance + probabilistic input/output layers provides a "near-zero cost" template for adding lightweight UQ to other networks.
- **Eliminating Prior Elicitation**: Using the prior only for initialization allows its hyperparameter memory to be released post-initialization, bypassing one of the most difficult aspects of BDL.

## Limitations & Future Work
- **Theory restricted to linear models**: Theorems 1/2 strictly hold for overparameterized linear models; implicit regularization in deep networks is more complex and observed only experimentally here.
- **Constrained covariance structure**: The characterization depends on decomposed covariance $\Sigma=SS^\top$; the implicit bias of SGD under arbitrary covariance parameterization remains an open question.
- **Rescaling and assumptions for classification**: Theorem 2 relies on exponential loss, linear separability, and data-spanning SVs, requiring $1/\log t$ mean rescaling for convergence. Extensions to cross-entropy or SGD with momentum remain conjectures.
- **Small learning rate for single-sample training**: $M=1$ introduces instability, requiring smaller learning rates, momentum, or extra training steps to compensate.

## Related Work & Insights
- **vs Standard Variational Inference (mean-field VI / ELBO)**: Standard VI uses KL divergence to regularize towards the prior; Ours uses $W_2$ applied implicitly by SGD. The difference is that KL prevents uncertainty collapse at training points while $W_2$ allows it, making IBVI closer to standard network training and generalization.
- **vs Generalized VI + Wasserstein Regularization**: Closely related theoretically—Ours proves that SGD's implicit bias **precisely implements** $W_2$ generalized VI, removing the need for explicit regularization terms.
- **vs Deep Ensembles**: DE requires multiple independent training runs, scaling cost linearly with ensemble size. IBVI is equivalent to DE for linear models but trains only one variational model with much lower memory cost and better OOD uncertainty.
- **vs Non-probabilistic Implicit Bias Theory**: Previous work showed SGD selects the closest (Euclidean) or max-margin solution. Ours elevates these to variational models and extends μP to probabilistic networks.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ "Removing KL and using implicit bias as regularization" with a rigorous 2-Wasserstein characterization is highly novel and theoretically deep.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 4 datasets + corrupted OOD + multiple baselines, though a gap remains between linear theory and deep net experiments.
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation, theory, and experiments are tightly linked; counter-intuitive conclusions are clearly explained.
- Value: ⭐⭐⭐⭐⭐ Obtaining well-calibrated uncertainty with nearly zero overhead is highly attractive for the practical deployment of Bayesian Deep Learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Variational Inference for Cyclic Learning](variational_inference_for_cyclic_learning.md)
- [\[ICLR 2026\] Diffusion Bridge Variational Inference for Deep Gaussian Processes](diffusion_bridge_variational_inference_for_deep_gaussian_processes.md)
- [\[ICLR 2026\] Implicit bias produces neural scaling laws in learning curves, from perceptrons to deep networks](implicit_bias_produces_neural_scaling_laws_in_learning_curves_from_perceptrons_t.md)
- [\[ICLR 2026\] Memory-Statistics Tradeoff in Continual Learning with Structural Regularization](memory-statistics_tradeoff_in_continual_learning_with_structural_regularization.md)
- [\[ICLR 2026\] Implicit Regularisation in Diffusion Models: An Algorithm-Dependent Generalisation Analysis](implicit_regularisation_in_diffusion_models_an_algorithm-dependent_generalisatio.md)

</div>

<!-- RELATED:END -->

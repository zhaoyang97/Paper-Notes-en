---
title: >-
  [Paper Note] How Does Bayesian Sampling Help Membership Inference Attacks?
description: >-
  [ICML 2026][Membership Inference Attack] This paper proposes BMIA, which expands a single reference model into a "virtual model family" using a Laplace posterior. By estimating the conditional score distribution of each…
tags:
  - "ICML 2026"
  - "Membership Inference Attack"
  - "Bayesian Sampling"
  - "Laplace Approximation"
  - "Conditional Distribution"
  - "Variance Decomposition"
date: 2026-05-08
content_hash: f594870758021751
---

# How Does Bayesian Sampling Help Membership Inference Attacks?

**Conference**: ICML 2026  
**arXiv**: [2503.07482](https://arxiv.org/abs/2503.07482)  
**Code**: https://github.com/zhenlong-liu/BMIA (Available)  
**Area**: AI Security / Privacy Attacks  
**Keywords**: Membership Inference Attack, Bayesian Sampling, Laplace Approximation, Conditional Distribution, Variance Decomposition

## TL;DR
This paper proposes BMIA, which expands a single reference model into a "virtual model family" using a Laplace posterior. By estimating the conditional score distribution of each sample via Bayesian sampling, it achieves a TPR in low FPR regions on datasets like CIFAR-100 that is 54% higher than LiRA (which requires training 8 reference models), while requiring only a single reference model training budget.

## Background & Motivation
**Background**: Membership Inference Attack (MIA) is a standard probe for measuring how much a model memorizes training samples. The current strongest class of attacks is "conditional attacks"—estimating a personalized threshold $\tau_\alpha(x,y)$ for each sample $z=(x,y)$ and determining if the model's score on that sample is abnormally high. LiRA by Carlini et al. and Attack-R by Ye et al. belong to this category.

**Limitations of Prior Work**: To estimate the conditional distribution, the mainstream approach is to **train dozens or even hundreds of shadow models**, each trained on a different subset. Scores for the same sample from all shadow models are then collected to fit a Gaussian or empirical distribution. On ImageNet, each shadow model takes 580 GPU·min; running 8 models takes 78 hours, which is nearly infeasible for real-world auditing scenarios.

**Key Challenge**: The power of conditional attacks stems from "per-instance uncertainty modeling," but existing methods can only obtain this uncertainty through **external retraining**, tightly coupling computational cost with attack strength.

**Goal**: Support conditional distribution estimation using a single reference model, ensuring that TPR in low FPR regions does not drop or even increases.

**Key Insight**: The authors observe that the variance of scores across multiple shadow models can be decomposed using the **law of total variance** into "intra-model variance" $\sigma^2_{\text{intra}}$ (caused by different parameters under the same dataset) and "inter-model variance" $\sigma^2_{\text{inter}}$ (caused by different datasets). LiRA effectively eliminates $\sigma^2_{\text{inter}}$ through external retraining but cannot handle $\sigma^2_{\text{intra}}$. If the reference model weights are treated as random variables from a BNN posterior, multiple weight samples from the posterior can directly capture $\sigma^2_{\text{intra}}$ without retraining.

**Core Idea**: Upgrade a MAP reference model to a family of Bayesian reference models using a Laplace posterior, and replace shadow training with posterior sampling to obtain the conditional score distribution.

## Method

### Overall Architecture
The BMIA attack pipeline: (1) Train a standard reference model on a reference dataset $\mathcal{D}$ disjoint from the target model's training set to obtain MAP weights $\hat w_1$; (2) Fit a Gaussian posterior $\mathcal{N}(w;\hat w_1,\Sigma)$ around $\hat w_1$ using Laplace Approximation; (3) For each sample $z^*=(x^*,y^*)$, sample $M$ sets of weights $\tilde w_i$ from this posterior and calculate a hinge score $s_i$ for each; (4) Treat the target model score $s_0$ as the "variable under test" and perform a one-sided one-sample $t$-test against $\{s_i\}$ to output a $p$-value for membership determination. This process **trains the reference model only once**, and all "expansion" costs are amortized over matrix multiplications and sampling.

### Key Designs

1.  **Laplace Posterior to Convert Single Model into Bayesian Model Family**:
    *   **Function**: Supports the entire conditional score distribution using a single MAP reference model.
    *   **Mechanism**: Performs a second-order Taylor expansion at $\hat w_1$ to approximate the posterior as $p(w\mid\mathcal{D})\approx\mathcal{N}(w;\hat w_1,\Sigma)$, where $\Sigma=(-\nabla_w^2\mathcal{L}(\mathcal{D};w)|_{w=\hat w})^{-1}$. In implementation, LA is applied only to the **last layer**, and the Hessian is approximated using KFAC or Diagonal methods. Prior precision is determined by maximizing marginal likelihood. Sampling $M$ sets of $\tilde w_i$ and passing them into the hinge score $s_{\text{hinge}}(x,y)=f(x)_y-\max_{y'\neq y}f(x)_{y'}$ yields a set of conditional scores under different samples of the same model.
    *   **Design Motivation**: LiRA uses Gaussian fitting of scores from $K$ shadow models to estimate $\tau_\alpha(x,y)$, equivalent to $M=1$ and large $K$. BMIA reverses this—single $K$ and large $M$—transforming external retraining into internal posterior sampling. This reduces "training cost" to "forward inference cost" while Bayesian sampling maintains the Gaussian approximation premise of scores (hinge scores are empirically nearly normal).

2.  **Conditional MIA Decision Rule Based on Student-$t$ Test**:
    *   **Function**: Formalizes the "score magnitude" as a hypothesis test to avoid subjective threshold selection.
    *   **Mechanism**: Defines calibrated scores $d_i=s_0-s_i$. Under the null hypothesis $H_0$ ($z^*$ is a non-member), $\mathbb{E}[d_i]=0$. The variance of $\bar d$ can be derived as $\operatorname{Var}(\bar d)=(1+\frac{1}{M})\sigma^2$. Using sample variance $\hat\sigma^2$ to estimate $\sigma^2$, the statistic $t=\bar d/(\hat\sigma\sqrt{1+1/M})$ follows a $t$-distribution with $M-1$ degrees of freedom. Membership is decided if $p=1-F_t(t;M-1)<\alpha$.
    *   **Design Motivation**: Traditional methods using empirical quantiles or Gaussian tails are unstable for extreme tails (0.1% FPR) with small samples. The $t$-test naturally handles unknown sample variance and small samples, fitting the "sampling dozens of weights" scenario. It also equates attack power to $1-\beta$, directly linking power with variance.

3.  **Total Variance Decomposition and MR-BMIA Multi-Reference Extension**:
    *   **Function**: Explains "why Bayesian sampling works" and extends the method to scenarios with multiple reference models.
    *   **Mechanism**: Uses the law of total variance to split total score variance into $\operatorname{Var}(s)=\sigma^2_{\text{intra}}+\sigma^2_{\text{inter}}$. In a setup with $K$ reference datasets and $M$ samples each, the variance of the difference between the target score and the mean $s_0-\bar s$ is $\operatorname{Var}(s_0-\bar s)=(1+\frac{1}{K})\sigma^2_{\text{inter}}+(1+\frac{1}{KM})\sigma^2_{\text{intra}}$. LiRA corresponds to $M=1$ and relies on increasing $K$; BMIA at $K=1$ reduces $\sigma^2_{\text{intra}}$ to a term of $\frac{1}{M}$. Theorem 3.2 further proves that $\beta(M')>\beta(M)$, meaning larger $M$ yields a tighter rejection region and higher TPR. The multi-reference variant MR-BMIA uses a mixture-Laplace to suppress both variance terms simultaneously, utilizing a two-level estimator in Algorithm 2 with Welch–Satterthwaite style degrees of freedom $v$ correction.
    *   **Design Motivation**: Theory precedes the method—the decomposition explicitly tells the attacker that "adding shadow models suppresses inter-variance, while adding posterior sampling suppresses intra-variance," providing actionable guidance on resource allocation.

### Loss & Training
No special training loss is used. The attacker runs standard SGD to train the reference model (ResNet-50 for CIFAR-10, DenseNet-121 for CIFAR-100, ResNet-50 for ImageNet, 4-layer MLP for tabular, and BERT/DistilBERT fine-tuning for text), followed by posterior fitting. Data is split 20%/20%/40%/20% for target training / target test / reference pool / QMIA validation.

## Key Experimental Results

### Main Results
Evaluations are conducted on CIFAR-10/100, ImageNet, Texas-100, Purchase-100, and 5 text datasets. Main metrics are TPR at low FPR and training time.

| Dataset | Metric | BMIA (n=1) | LiRA (n=8) | Gain / Saving |
| :--- | :--- | :--- | :--- | :--- |
| CIFAR-100 | TPR@FPR=1% | 35.75% | 23.20% | +54% TPR |
| CIFAR-100 | Training Time | 26.4 min | 211.5 min | 8× Speedup |
| CIFAR-10 | TPR@FPR=0.1% | 2.84% | 1.73% | +64% TPR |
| ImageNet | TPR@FPR=1% | 13.59% | 11.90% | Slightly better & 8× faster |
| Texas-100 | TPR@FPR=1% | 11.81% | 8.63% | +37% TPR |

| Setting | Dataset | Method | TPR@FPR=1% |
| :--- | :--- | :--- | :--- |
| Single Ref | CIFAR-100 | RMIA | 10.08% |
| Single Ref | CIFAR-100 | QMIA | 15.26% |
| Single Ref | CIFAR-100 | **BMIA** | **35.75%** |
| 64 Ref | CIFAR-100 | LiRA | 43.33% |
| 64 Ref | CIFAR-100 | RMIA | 36.06% |
| 64 Ref | CIFAR-100 | **MR-BMIA** | **45.57%** |

### Ablation Study
| Configuration | CIFAR-10 TPR@1% | Remarks |
| :--- | :--- | :--- |
| BMIA, $M=1$ | Similar to LiRA(n=1) | Degenerates to single score comparison |
| BMIA, $M$ increases | Monotonic increase | Validates Theorem 3.2 |
| Hessian = Diagonal | Close to KFAC | Lightweight approximation maintains performance |
| Arch mismatch (target=ResNet-50, ref=ResNet-18) | BMIA 8.72% vs LiRA 8.16% | Still leads across architectures |

### Key Findings
*   **Variance decomposition is empirically validated**: Increasing $M$ improves TPR while inference time remains nearly constant (sampling is parallelized), proving that gains come from suppressing $\sigma^2_{\text{intra}}$ rather than extra computation.
*   **Cross-modal robustness**: BMIA achieves SOTA or competitive results across Image, Text, and Tabular modalities with various architectures (ResNet/DenseNet/BERT/MLP).
*   **Robustness to architecture mismatch**: When the reference model is ResNet-18 and the target is ResNet-50, BMIA leads LiRA across all FPR ranges, indicating that Laplace posterior uncertainty is more "universal" than shadow model ensembles.
*   **MR-BMIA is not redundant**: When compute allows for multiple references, MR-BMIA suppresses both variance terms, pushing TPR@1% on CIFAR-100 to 45.57%, which is 2.2 points higher than 64-shadow LiRA.

## Highlights & Insights
*   **BNN posterior as a "free shadow model generator"**: Single MAP model + Laplace Posterior $\approx$ a family of shadow models. It cleverly obtains uncertainty during the inference phase, avoiding the quadratic overhead of the training phase.
*   **Theory precedes practice**: The paper first uses variance decomposition to clarify what "shadow $K$ vs sampling $M$" each controls, then designs BMIA and MR-BMIA to correspond precisely, creating a clean loop between theory and method.
*   **Transferable trick**: Formalizing the attack as a hypothesis test using the $t$-test + calibrated scores $d_i=s_0-s_i$ is more stable than empirical quantiles and can be directly transferred to other score-based tasks (e.g., OOD detection, distribution shift).
*   **Audit-friendly**: The budget of a single reference model plus dozens of posterior samples makes MIA feasible for privacy auditing of actual production-size models for the first time.

## Limitations & Future Work
*   Current implementation only uses **last-layer** LA + KFAC/Diagonal Hessian; costs and benefits of all-layer LA are not fully analyzed. Laplace assumptions might collapse under non-convex or heavy-tailed losses.
*   Gaussian score approximation is a prerequisite for the $t$-test; the authors admit that additional calibration is needed for non-Gaussian scores (e.g., long-tail text tasks). This might be fragile at the LLM scale.
*   The actual gains of BMIA under defense strategies (differential privacy, temperature scaling) are not extensively evaluated; the perspective is stronger on the attacker side than the defender side.
*   Lack of direct comparison with gradient-based or loss-trajectory MIA; whether these can be combined remains for future work.

## Related Work & Insights
*   **vs LiRA (Carlini 2022)**: LiRA trains multiple shadow models to fit a Gaussian score distribution; BMIA trains a single model and uses Laplace posterior expansion. This paper explicitly notes that LiRA is equivalent to $M=1$ and thus inevitably loses to BMIA under low $K$ budgets.
*   **vs RMIA (Zarifzadeh 2024)**: RMIA uses the likelihood ratio of sample pairs; this paper uses the weight distribution within samples. BMIA achieves higher TPR under a single-reference budget.
*   **vs QMIA (Bertran 2024)**: QMIA trains quantile regression to predict thresholds, requiring hyperparameter search for the quantile model. BMIA converts quantile estimation into posterior sampling, saving the second-order training loop.
*   **vs Attack-R (Ye 2022)**: Attack-R uses empirical quantiles for thresholds, requiring more shadows for stability. BMIA uses a parametric $t$-distribution, allowing estimation from small samples.

## Rating
*   Novelty: ⭐⭐⭐⭐⭐ Using Laplace posterior to replace shadow training is a first in MIA; the variance decomposition perspective is also new.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ Coverage of three modalities, multiple architectures, single/multiple references, architecture mismatch, and Hessian factorization.
*   Writing Quality: ⭐⭐⭐⭐ Clear theory and dense tables; minor issues with uncompiled label references in some figures.
*   Value: ⭐⭐⭐⭐⭐ Reduces high-fidelity MIA costs from "hundred-GPU scale" to "single-GPU scale," making real-world privacy auditing feasible.

## Rating
*   Novelty: To be rated
*   Experimental Thoroughness: To be rated
*   Writing Quality: To be rated
*   Value: To be rated

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Membership Inference Attacks with False Discovery Rate Control](../../ICCV2025/others/membership_inference_attacks_with_false_discovery_rate_control.md)
- [\[ICML 2026\] Singular Bayesian Neural Networks](singular_bayesian_neural_networks.md)
- [\[ICML 2026\] SORA: Free Second-Order Attacks in Fast Adversarial Training](sora_free_second-order_attacks_in_fast_adversarial_training.md)
- [\[ICML 2026\] Partitioning for Intrinsic Model Inversion Resistance in Collaborative Inference](partitioning_for_intrinsic_model_inversion_resistance_in_collaborative_inference.md)
- [\[ICML 2026\] Amortized Simulation-Based Inference in Generalized Bayes via Neural Posterior Estimation](amortized_simulation-based_inference_in_generalized_bayes_via_neural_posterior_e.md)

</div>

<!-- RELATED:END -->

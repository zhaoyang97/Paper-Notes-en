---
title: >-
  [Paper Note] Inconsistency-Aware Minimization: Improving Generalization with Unlabeled Data
description: >-
  [ICML 2026][Self-Supervised Learning][Generalization Bound] This paper proposes "Local Inconsistency" $S_\rho(\theta)$—the worst-case KL divergence within a parameter ball—which can be computed using only unlabeled data.…
tags:
  - "ICML 2026"
  - "Self-Supervised Learning"
  - "Generalization Bound"
  - "Fisher Information Matrix"
  - "Sharpness-Aware Minimization"
  - "KL Divergence Regularization"
  - "Unlabeled Data"
date: 2026-05-08
content_hash: ed0c77a99f550547
---

# Inconsistency-Aware Minimization: Improving Generalization with Unlabeled Data

**Conference**: ICML 2026  
**arXiv**: [2605.31324](https://arxiv.org/abs/2605.31324)  
**Code**: https://github.com/heesung-k/IAM  
**Area**: Optimization & Regularization / Semi-supervised / Self-supervised  
**Keywords**: Generalization Bound, Fisher Information Matrix, Sharpness-Aware Minimization, KL Divergence Regularization, Unlabeled Data

## TL;DR
This paper proposes "Local Inconsistency" $S_\rho(\theta)$—the worst-case KL divergence within a parameter ball—which can be computed using only unlabeled data. By employing it as a training regularizer, the resulting IAM optimizer matches or exceeds SAM/ASAM in supervised tasks and provides additional gains in semi-supervised (FixMatch) and self-supervised (SimCLR) scenarios by leveraging unlabeled batches.

## Background & Motivation

**Background**: Research on deep network generalization currently follows two main tracks: first, sharpness-aware optimizers like SAM/ASAM, which use the maximum eigenvalue of the loss Hessian $\lambda_{\max}(H)$ as a proxy for "flatness" to approximate the geometry near minima; second, measures based on "output disagreement" such as disagreement (Jiang et al.) or inconsistency (Johnson-Zhang), which use KL divergence between multiple models/data partitions as a generalization proxy.

**Limitations of Prior Work**: Both tracks have significant drawbacks. Sharpness-based measures exhibit "local positive correlation, global negative correlation" anomalies under different weight decay and data augmentation combinations, which Andriushchenko et al. pointed out as being entangled with training hyperparameters rather than true generalization. Although disagreement/inconsistency can be calculated using only unlabeled data, they are defined as expectations over multiple trained models, making them neither differentiable nor regularizable in a single-model context, and thus impractical for engineering.

**Key Challenge**: The authors identify a core conflict: can a geometric measure be found that is "dependent only on a single model, differentiable, and requires only unlabeled data," allowing it to both predict generalization gaps and be directly integrated into the training loss as a regularizer? Sharpness measures satisfy the first two but require labeled data; inconsistency-based measures satisfy only the "unlabeled" requirement.

**Goal**: Construct a new measure $S_\rho(\theta)$ that simultaneously possesses (i) single-model computability, (ii) differentiability, and (iii) use of unlabeled data only, and design a unified regularizer based on it for supervised, semi-supervised, and self-supervised learning.

**Key Insight**: From an information geometry perspective, the second-order expansion of KL divergence in parameter space is exactly the quadratic form of the Fisher Information Matrix (FIM) $\tfrac12\delta^\top F(\theta)\delta$, and the Gauss–Newton approximation aligns $F$ with the loss Hessian $H$ under cross-entropy. By using the "worst-case KL of the output distribution under parameter perturbation" as a measure, it inherits the Hessian implications of sharpness measures while being computable without labels because KL is defined in the output space.

**Core Idea**: Define $S_\rho(\theta)=\max_{\|\delta\|\le\rho}\mathbb{E}_x[\mathrm{KL}(f(x;\theta)\|f(x;\theta+\delta))]$, prove it approximates $\tfrac12\rho^2\lambda_{\max}(F(\theta))$, use a single step of Power Iteration to calculate its gradient, and insert it into the training objective as a "KL-based proxy" for SAM.

## Method

### Overall Architecture
The method consists of two layers: the measurement layer defines and estimates $S_\rho(\theta)$, and the optimization layer integrates it into the training objective. Measurement estimation follows Algorithm 1: an initial perturbation $\delta_0$ is sampled from an isotropic Gaussian, followed by $K$ steps of normalized gradient ascent. Since the second-order approximation of KL with respect to $\delta$ is $F\delta$, one step of normalized gradient ascent is equivalent to one Power Iteration step, approximating the principal eigenvector of $F$ at the cost of $K$ backpropagations. The optimization layer provides two variants: IAM-D adds $\beta S_\rho(\theta)$ directly to the training loss as soft regularization; IAM-S mimics SAM by calculating the training loss gradient at the estimated perturbation point $\theta+\delta^*$, resulting in a KL-driven adversarial update. The entire pipeline's overhead per step is nearly identical to SAM (both requiring one extra gradient computation), but the KL branch only considers the output distribution without $y$, allowing it to naturally incorporate all unlabeled samples in pipelines like FixMatch or SimCLR.

### Key Designs

1.  **Connection between Local Inconsistency $S_\rho(\theta)$ and FIM**:
    - **Function**: Uses a single-model, single-batch quantity to predict the generalization gap and provide a differentiable regularization signal.
    - **Mechanism**: Define $S_\rho(\theta)=\max_{\|\delta\|\le\rho}\mathbb{E}_x[\mathrm{KL}(f(x;\theta)\|f(x;\theta+\delta))]$. A second-order Taylor expansion for $\delta$ transforms this into $\max\tfrac12\delta^\top F(\theta)\delta=\tfrac12\rho^2\lambda_{\max}(F(\theta))$. Since calculating $F$ only requires $\nabla_\theta z$ and the softmax output $f$, the process involves no ground truth labels $y$. Under cross-entropy settings, $H\approx G=F$, so $S_\rho$ is geometrically equivalent to an "unlabeled version of maximum eigenvalue sharpness" near the solution.
    - **Design Motivation**: To overcome the dilemma of "sharpness requiring labels + inconsistency requiring multiple models," geometry must be performed in the output space rather than the loss space. The second-order expansion of KL translates "output sensitivity" back to the FIM principal axes, providing theoretical interpretability. Theorem 4.1 embeds $\lambda_{\max}(F_S)$ into Luo et al.'s generalization bound, arguing that replacing $\lambda_{\max}(H)$ with $S_\rho$ does not lose precision in near-interpolation regimes.

2.  **Power Iteration Estimation + IAM-S/D Injection**:
    - **Function**: Reduces the insoluble $\max$ problem to an executable $K=1$ step algorithm and provides two interfaces to inject $S_\rho$ into training.
    - **Mechanism**: Use $\delta_{k+1}=\rho\,g_k/\|g_k\|$, where $g_k=\nabla_\delta \mathbb{E}_x \mathrm{KL}(f(x;\theta)\|f(x;\theta+\delta))$, equivalent to normalized Power Iteration on $F$; $K=1$ is sufficient to approximate the principal feature direction. For injection, IAM-D directly minimizes $L(\theta)+\beta S_\rho(\theta)$, while IAM-S minimizes $L(\theta+\delta^*)$, isomorphic to SAM but with the perturbation direction derived from KL rather than the training gradient. The authors argue that since $\pm\delta$ appear with equal probability, the first-order term $\delta^\top\nabla_\theta L$ cancels out in expectation, meaning IAM-S implicitly suppresses the principal eigenvalue of $G(\theta)=F(\theta)$.
    - **Design Motivation**: The computational cost of a single-step normalized gradient ascent is equivalent to SAM's adversarial perturbation, making IAM comparable to SAM in "per-step cost." The D/S interfaces allow it to function as both a plug-in regularizer (D is easy to combine with FixMatch/SimCLR) and a SAM-style worst-case minimization (S is more stable for supervised tasks).

3.  **Natural Adaptation to Unlabeled Data**:
    - **Function**: Allows the regularizer to utilize all unlabeled samples in semi-supervised and self-supervised training, mitigating the bias of "sharpness estimated only on small label subsets."
    - **Mechanism**: Estimating $S_\rho$ requires a forward pass to get $f(x;\theta)$ and a backward pass for $\nabla_\delta \mathrm{KL}$, neither of which involves $y$. In FixMatch, $\beta S_\rho(\theta)$ is added to the original objective, with the KL expectation taken over the entire batch (labeled+unlabeled). In SimCLR, the KL expectation is taken over the projection head outputs, remaining label-independent.
    - **Design Motivation**: The paper notes that "measuring flatness on sparse label sets" does not reflect true flatness over the entire data manifold—applying SAM directly to the FixMatch labeled loss yields no improvement (see Appx. E.4). IAM extends second-order geometric signals to unlabeled distributions via KL label-independence, which is pivotal for its superiority over SAM in semi/self-supervised learning.

### Loss & Training
The supervised training objective is $L_{\text{IAM-D}}=L(\theta)+\beta S_\rho(\theta)$ or $L_{\text{IAM-S}}=L(\theta+\delta^*)$, with $K=1$ in Algorithm 1 to estimate perturbations. For CIFAR-10, $\beta=1.0, \rho=0.1$ is used; for CIFAR-100, $\beta=10.0, \rho=0.1$ (IAM-D) or $\rho=0.5$ (IAM-S). In ImageNet, $\rho=0.2$ (S) / $0.1$ (D). In semi-supervised settings, the KL expectation is taken over the combined labeled and unlabeled batch. In self-supervised settings, KL is calculated on the projection head's output distribution.

## Key Experimental Results

### Main Results

| Dataset | Model | Metric | SGD | SAM | ASAM | IAM-D | IAM-S |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| CIFAR-10 | WRN-16-8 | Test Error | 3.68 | 3.31 | 3.15 | 3.28 | 3.28 |
| CIFAR-100 | WRN-16-8 | Test Error | 19.17 | 17.63 | 17.15 | 17.16 | **16.82** |
| F-MNIST | WRN-28-10 | Test Error | 4.45 | 4.13 | 4.11 | 4.13 | **4.10** |
| SVHN | WRN-28-10 | Test Error | 3.82 | 3.47 | 3.24 | **3.13** | **3.13** |
| ImageNet | ResNet-50 | Top-1 Err | 22.66 | 21.80 | – | **21.36** | 21.72 |
| ImageNet | ResNet-50 | Top-5 Err | 6.51 | 5.99 | – | **5.70** | 5.90 |

In supervised scenarios, IAM performs on par with ASAM/SAM on small datasets, while IAM-S outperforms SAM by 0.81% on the more challenging CIFAR-100. On ImageNet, IAM-D directly outperforms the stronger SAM baseline.

### Ablation Study

| Configuration | CIFAR-10 (250 labels) | CIFAR-10 (4000 labels) | CIFAR-100 (2500 labels) | CIFAR-100 (10000 labels) | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| SGD | 63.82 | 22.45 | 68.91 | 45.94 | No geometric reg. |
| SAM (labeled only) | 63.91 | 19.95 | 69.53 | 43.30 | Sharpness on label subset only |
| IAM-D (labeled+unlabeled) | 61.77 | 15.07 | 66.98 | 40.02 | KL on full batch |
| FixMatch | 6.26 | 4.10 | 32.84 | 22.93 | Strong SSL baseline |
| FixMatch + IAM-D | **5.30** | **3.88** | **28.95** | **21.99** | Plug-in improvement |

Under the extreme scarcity of 250 labels, SAM is slightly worse than SGD (63.91 vs 63.82), confirming the author's assertion that "flatness on small label sets is unreliable." Meanwhile, IAM-D steadily reduces the error to 61.77 by extending the signal to unlabeled batches, and further to 5.30 when layered with FixMatch, representing the largest relative reduction in that setting.

### Key Findings
- On small models like 6CNN, the Kendall $\tau$ between $S_\rho, \mathrm{Tr}(H), \lambda_{\max}(H)$ and the generalization gap is similar (0.51–0.54). However, as data augmentation and weight decay increase on WRN28-2, global correlation for $\mathrm{Tr}(H)$ and $\lambda_{\max}(H)$ flips to negative values ($-0.04, -0.12$), while $S_\rho$ remains positively correlated ($0.37$). This shows the KL measure is more robust to hyperparameter scale effects.
- IAM-D significantly suppresses the rise of $S_\rho$ during training and avoids the overfitting behavior (test accuracy drop + inconsistency rebound) seen in SGD after learning rate decay, suggesting it confines the model to parameter regions with more stable outputs.
- Directly applying SAM to the FixMatch labeled loss yields no improvement (Appx. E.4), but replacing it with IAM-D on the entire batch provides significant gains. This confirms that being "label-independent + using unlabeled data" is the key driver of this gain, rather than simply "adding a KL term."

## Highlights & Insights
- Using the second-order expansion of KL divergence as "output-space sharpness" is the cleanest step of this paper. It solves three problems at once: single-model availability (unlike inconsistency), differentiability (unlike disagreement), and label-independence (unlike sharpness). This "re-choosing coordinates" approach is worthy of reuse for other regularizers.
- The paper uses a Power Iteration perspective to explain why $K=1$ is sufficient: a single step of normalized gradient ascent is equivalent to one Power Iteration, which approximates the FIM principal eigenvector. Symmetric sampling of $\pm\delta$ ensures the first-order term vanishes in expectation, meaning IAM-S implicitly performs principal eigenvalue minimization. This explains SAM's success as "suppression on FIM principal axes," providing a more geometric explanation.
- The improvement of IAM-D + FixMatch in semi-supervised learning suggests that many SSL methods only suppress consistency loss without suppressing the "worst-case shift in output distribution under parameter perturbation." The latter can serve as a new SSL regularization suite applicable to any network with an output probability distribution (classification, projection heads in contrastive learning, score heads in diffusion models, etc.).

## Limitations & Future Work
- Estimating $S_\rho$ still requires an additional full-model backpropagation, matching SAM's cost but doubling that of SGD. The authors acknowledge the need for cheaper versions (e.g., low-rank FIM approximations or Hutchinson estimation).
- The theoretical part (Theorem 4.1) relies on the near-interpolation hypothesis $\varepsilon_R\approx 0$; the gap between $\lambda_{\max}(F)$ and $\lambda_{\max}(H)$ during intermediate stages far from interpolation is not covered.
- Experiments were restricted to CV and ResNet/WRN/ViT architectures; verification on LLMs, diffusion models, and regression tasks is pending. For non-categorical outputs (e.g., continuous Gaussian), the KL expansion form changes and requires re-derivation of the FIM.
- The self-supervised section only tested SimCLR + ResNet-18 + linear probe, not stronger SSL like MAE/DINO/MoCo. Furthermore, system reports on the sensitivity of $\rho$ and $\beta$ in self-supervised settings are missing, requiring manual tuning for deployment.

## Related Work & Insights
- **vs SAM (Foret et al., 2021)**: SAM calculates gradients at the worst perturbation point of the training loss, requiring $y$. This paper calculates the gradient of $L$ at the worst KL perturbation point (IAM-S) or uses it as soft regularization (IAM-D), removing the need for $y$. Both share the same per-step cost, but IAM is stronger on CIFAR-100/ImageNet/Semi-supervised tasks.
- **vs ASAM (Kwon et al., 2021)**: ASAM uses adaptive sharpness to address SAM's scale invariance but still relies on training loss. Starting from output KL naturally provides scale invariance (softmax outputs are invariant to linear reparameterization), eliminating the need for extra reweighting.
- **vs Johnson & Zhang (2023) Inconsistency**: Their inconsistency requires training multiple models to take the KL expectation. The paper proves that under the isotropic posterior assumption, $S_\rho$ is proportional to their conditional inconsistency (coefficients from $m/(2C)$ to $m/2$). Thus, IAM essentially compresses multi-model inconsistency into a single-model differentiable version, removing ensembling costs.
- **vs Explicit Jacobian Regularization (Lee et al., 2023)**: They prove that "random noise projected through the Jacobian column space becomes a meaningful perturbation." $F(\theta)\varepsilon$ in this paper is an instantiation of that mechanism on the FIM principal feature space, providing an output-space explanation for EJR.

## Rating
- Novelty: ⭐⭐⭐⭐ Using KL's second-order expansion as output-space sharpness and linking it to unlabeled SSL is a clear new perspective, though individual components (FIM/SAM/inconsistency) are not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers CIFAR/F-MNIST/SVHN/ImageNet + Semi-supervised + Self-supervised, but lacks LLMs/Diffusion models, and self-supervised testing is limited to one baseline.
- Writing Quality: ⭐⭐⭐⭐ Both theory and algorithm descriptions are clear; formulas and pseudocode are complete, though some figure descriptions are fragmented.
- Value: ⭐⭐⭐⭐ Immediate value for SSL engineers as a plug-in regularizer for FixMatch/SimCLR; provides new output-space coordinates for sharpness generalization theory researchers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Statistical Consistency and Generalization of Contrastive Representation Learning](statistical_consistency_and_generalization_of_contrastive_representation_learnin.md)
- [\[ICML 2026\] A Refined Generalization Analysis for Extreme Multi-class Supervised Contrastive Representation Learning](a_refined_generalization_analysis_for_extreme_multi-class_supervised_contrastive.md)
- [\[ICML 2026\] Data Augmentation of Contrastive Learning is Estimating Positive-incentive Noise](data_augmentation_of_contrastive_learning_is_estimating_positive-incentive_noise.md)
- [\[AAAI 2026\] Improving Sustainability of Adversarial Examples in Class-Incremental Learning](../../AAAI2026/self_supervised/improving_sustainability_of_adversarial_examples_in_class-incremental_learning.md)
- [\[ACL 2026\] LLMSurgeon: Diagnosing Data Mixture of Large Language Models](../../ACL2026/self_supervised/llmsurgeon_diagnosing_data_mixture_of_large_language_models.md)

</div>

<!-- RELATED:END -->

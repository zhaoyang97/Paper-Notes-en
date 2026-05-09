---
title: >-
  [Paper Note] Measuring Uncertainty Calibration
description: >-
  [ICLR 2026][LLM Evaluation][Calibration error] For the problem of estimating the $L_1$ calibration error of binary classifiers from finite samples, this paper proposes the first non-asymptotic, distribution-free certifiable upper bound methods under two structural assumptions—bounded variation and bounded derivatives—where the latter can be guaranteed by applying a small perturbation to classifier outputs. Experiments demonstrate that the calibration error upper bound can be controlled to approximately 0.02 with $10^7$ samples.
tags:
  - ICLR 2026
  - LLM Evaluation
  - Calibration error
  - finite-sample bounds
  - distribution-free
  - bounded variation
  - kernel estimation
date: 2026-05-08
content_hash: 0958a155688ee157
---

# Measuring Uncertainty Calibration

**Conference**: ICLR 2026
**arXiv**: [2512.13872](https://arxiv.org/abs/2512.13872)
**Code**: [GitHub](https://github.com/spotify-research/calibration)
**Area**: Machine Learning Theory / Calibration
**Keywords**: Calibration error, finite-sample bounds, distribution-free, bounded variation, kernel estimation

## TL;DR

For the problem of estimating the $L_1$ calibration error of binary classifiers from finite samples, this paper proposes the first non-asymptotic, distribution-free certifiable upper bound methods under two structural assumptions—bounded variation and bounded derivatives—where the latter can be guaranteed by applying a small perturbation to classifier outputs. Experiments demonstrate that the calibration error upper bound can be controlled to approximately 0.02 with $10^7$ samples.

## Background & Motivation

**Background**: Whether the output probabilities of machine learning models match true event probabilities—i.e., calibration—is critical for decision-making tasks. The most widely used calibration metric is ECE (Expected Calibration Error), which estimates calibration error by binning model outputs and computing the average error within each bin. However, this approach is highly sensitive to the choice of binning scheme, and different numbers of bins or partitioning strategies yield substantially different calibration error estimates.

**Limitations of Prior Work**: Existing calibration measurement methods face a fundamental dilemma. If binning is treated as an external post-processing step, the estimates are unreliable and entirely dependent on bin configuration—a finding empirically confirmed by Arrieta-Ibarra et al. (2022). If binning is treated as an integral component of the classifier, classification performance suffers because binning is not considered during training and gradients cannot backpropagate through discrete bin boundaries. Another class of methods (e.g., KS test, Kuiper test) frames calibration as a frequentist hypothesis test; while statistically powerful, these methods can only determine whether a classifier is perfectly calibrated and cannot quantify the degree of miscalibration across models, with theoretical guarantees that rely on asymptotic analysis. More fundamentally, Lee et al. (2023) proved that even assuming the calibration function is continuous is insufficient for consistent estimation of calibration error from finite samples.

**Key Challenge**: Estimating calibration error requires structural assumptions on the calibration function $\eta(s) = \mathbb{E}[Y|S=s]$. Assumptions that are too strong restrict the applicability of the method, while assumptions that are too weak yield poor sample efficiency and loose bounds. The challenge is to strike a favorable balance between assumption strength and estimation precision.

**Goal**: To provide computable, theoretically guaranteed upper bounds on calibration error satisfying three core requirements: (1) non-asymptotic—valid for any finite sample size; (2) distribution-free—placing no restrictions on the form of the score distribution (discrete, continuous, or mixed); and (3) practically feasible—producing meaningful bounds under reasonable computational and data requirements.

**Key Insight**: The authors observe that while calibration error cannot be estimated for arbitrary $\eta$, guarantees can be established under two realistic and verifiable structural assumptions. The first is bounded variation (a weak but general assumption), and the second is bounded derivatives (stronger but constructively guaranteed via perturbation). The two approaches are complementary and suited to different application scenarios.

**Core Idea**: By constructing a surrogate $\hat{\eta}$ for the calibration function (via TV denoising or Nadaraya-Watson kernel smoothing), the calibration error is decomposed into a "surrogate calibration error" plus a "surrogate construction error," both of which can be computed from data and bounded probabilistically using Bernstein's inequality.

## Method

### Overall Architecture

The core framework consists of three steps: (1) split the dataset into a training set $T$ and a validation set $V$ (with K-fold cross-fitting used in practice); (2) construct a surrogate $\hat{\eta}$ for the calibration function $\eta$ on the training set; (3) estimate the surrogate calibration error on the validation set using Bernstein's inequality, and add a theoretical upper bound on the surrogate construction error to obtain a certifiable probabilistic upper bound on CE. The key mathematical decomposition is:

$$\text{CE} = \mathbb{E}_s[|s - \eta(s)|] \leq \underbrace{\mathbb{E}_s[|\hat{\eta}(s) - s|]}_{\text{surrogate calibration error}} + \underbrace{\mathbb{E}_s[|\hat{\eta}(s) - \eta(s)|]}_{\text{surrogate construction error}}$$

### Key Designs

1. **TV Denoising under Bounded Variation Assumption**:

    - **Function**: Constructs a piecewise-constant surrogate for $\eta$ under the assumption that the calibration function has bounded variation $\text{TV}(\eta, [0,1]) \leq V$.
    - **Mechanism**: After sorting the training set, the method solves the TV denoising optimization problem $\hat{\eta}_T = \arg\min_{v \in [0,1]^{|T|}} \frac{1}{2|T|}\|y_T - v\|_2^2 + \lambda\|Dv\|_1$, where $D$ is the first-order difference matrix and $\lambda$ is set according to the confidence parameter $\delta_1$ as $\sqrt{\frac{1}{8|T|}\ln\frac{4(|T|-1)}{\delta_1}}$. The resulting $\hat{\eta}$ is a piecewise-constant function, interpretable as a data-driven adaptive binning scheme where bin boundaries and counts are determined entirely by the data. The reconstruction error TVB($\delta_1$) is derived from the theoretical results of Hütter & Rigollet (2016), supplemented by a population transfer bound (PTB) that extends training-set guarantees to the population.
    - **Design Motivation**: Bounded variation is among the weakest structural assumptions that permit finite-sample estimation. In practice, if a classifier is well-trained so that high scores correspond to high positive-class probabilities, then $\eta$ is approximately monotone—and all monotone functions on $[0,1]$ have total variation naturally bounded by 1, making $V=1$ a reasonable default.

2. **Perturbation + NW Kernel Smoothing under Bounded Derivative Assumption**:

    - **Function**: For any classifier, applies a small random perturbation to outputs so that the resulting calibration function automatically possesses bounded derivatives, enabling tighter kernel-smoothing estimation.
    - **Mechanism**: The key innovation is that no favorable properties of the original classifier are assumed. For the original classifier output $s_{\text{orig}}$, a perturbed score $s \in [0,1]$ is sampled according to the hyperbolic secant kernel $k(s|s_{\text{orig}}) = \frac{1}{Z}\text{sech}(\frac{s_{\text{orig}} - s}{h})$. A core lemma establishes that regardless of the properties of $\eta_{\text{orig}}$, the perturbed calibration function $\eta$ automatically possesses bounded first-order derivatives ($\leq \frac{1}{2h}$) and second-order derivatives ($\leq \frac{3}{2h^2}$). Given these derivative bounds, a Nadaraya-Watson kernel smoother constructs the surrogate $\hat{\eta}(s') = \sum_{i \in T} w_i(s') y_i$, whose reconstruction error $g_T(s')$ can be computed exactly from data and decays with sample size.
    - **Design Motivation**: (1) Guaranteeing mathematical tractability from first principles avoids unverifiable assumptions. (2) Bounded derivatives is a stronger assumption than bounded variation, improving the theoretical rate from $n^{-1/4}$ to $n^{-1/3}$ with substantially better sample efficiency. (3) The sech kernel is preferred over a truncated Gaussian because its derivative bounds on $[0,1]$ admit more concise expressions.

3. **K-Fold Cross-Fitting**:

    - **Function**: Partitions data into $K$ folds, each serving as the validation set in turn, with results aggregated to reduce variance.
    - **Mechanism**: Each validation point is scored by a surrogate fitted on a training set that has not seen it, preserving the training/validation independence required by theory while avoiding data waste from fixed splits.
    - **Design Motivation**: Simultaneously satisfies theoretical validity and practical efficiency.

### Loss & Training

When using the perturbation approach, the effect of perturbation must be accounted for during training. Concretely, the training loss is modified so that the model optimizes classification performance with awareness that perturbation will be applied at inference. Experiments show that this modification incurs negligible additional training cost.

## Key Experimental Results

### Main Results: Convergence Rates of Each Method on Synthetic Data

Four synthetic calibration functions with known ground truth are used to evaluate the relationship between bound quality and sample size:

| Method | Empirical Convergence Rate | Theoretical Rate | Required Assumptions | Synthetic Data Performance |
|--------|---------------------------|-----------------|---------------------|---------------------------|
| NW (kernel smoothing) | $[-0.406, -0.213]$ | $-1/3$ | Bounded derivatives | Tightest bound on all functions |
| TV (denoising) | $[-0.423, -0.164]$ | $-1/4$ | Bounded variation | Consistent convergence but looser |
| Lip+Bkt | $[-0.574, -0.346]$ | $-1/3$ | Lipschitz | Same rate as NW but larger constants |
| ECE (heuristic) | Inconsistent | No guarantee | None | Complete failure on 4th function |

The NW method yields the tightest upper bounds across all four synthetic functions. ECE performs acceptably on the first three functions but fails entirely on the fourth—the error does not decrease as sample size grows—demonstrating the fundamental risk of guarantee-free heuristic methods.

### Ablation Study: Effect of Perturbation Bandwidth on Classification Performance

| Dataset | Model | AUROC Change at $h = 2^{-6}$ | AUROC Change at $h = 2^{-4}$ | Calibration Error Upper Bound at $h = 2^{-6}$ |
|---------|-------|------------------------------|------------------------------|-----------------------------------------------|
| IMDB | BERT | $< 0.001$ decrease | Notable decrease | ~0.02 ($10^7$ samples) |
| Spam Detection | BERT | $< 0.001$ decrease | Notable decrease | ~0.02 ($10^7$ samples) |
| CIFAR | ViT | $< 0.001$ decrease | Notable decrease | ~0.02 ($10^7$ samples) |

### Key Findings

- **ECE is unreliable**: On the fourth synthetic calibration function, ECE fails to converge to the true value even as the sample size grows without bound, demonstrating that heuristic methods can systematically produce incorrect estimates in certain settings.
- **Constant-factor advantage of NW**: NW and Lipschitz-binning share the same theoretical convergence rate ($n^{-1/3}$), but NW's constant term is significantly smaller, yielding upper bounds that can be several times tighter in practice.
- **Perturbation is nearly cost-free**: Perturbation with $h = 2^{-6}$ affects AUROC by less than 0.001 on all three real datasets, yet is sufficient to guarantee bounded derivatives and enable the NW method.
- **Computationally efficient**: All methods require at most log-linear time; the sliding-window implementation of NW runs in linear time, completing 64 repeated experiments (with sample sizes up to $10^7$) in approximately 4 minutes.
- **All results are statistically significant**: Confidence intervals over 64 repetitions are too narrow to be visible in the figures.

## Highlights & Insights

- **Perturbation as a guarantee of smoothness** is the most elegant idea in the paper. Without making any assumption about the original classifier, a simple random perturbation constructively ensures that the calibration function has bounded derivatives. This paradigm of "obtaining mathematical tractability by construction" is transferable to other statistical estimation problems that require smoothness assumptions.
- **Reinterpreting TV denoising as adaptive binning** provides a rigorous theoretical foundation for the classical binning approach. The problem with traditional binning is that bin selection is arbitrary; TV denoising automatically determines the number and boundaries of bins through optimization.
- **The choice of the sech kernel reflects the unity of mathematical elegance and practicality**: Compared to a truncated Gaussian, the sech kernel has superior properties on $[0,1]$ and yields more concise expressions for derivative bounds—a seemingly minor detail that materially affects the elegance of the theoretical results.

## Limitations & Future Work

- **Binary classification only**: The current theory is entirely restricted to binary classifiers; extension to multi-class calibration (e.g., top-1 calibration, classwise calibration) is an important open problem.
- **High sample requirements**: Approximately $10^7$ samples are needed to bring the calibration error upper bound to ~0.02, which is impractical for small evaluation sets or rare-event prediction.
- **Perturbation requires retraining**: Although the overhead is minimal, the perturbation method requires incorporating a perturbation-aware term into the training loss and retraining the model; for already-deployed models, one must fall back to the weaker TV-based method.
- **Focus on upper bounds**: The paper primarily addresses upper bounds; while lower bounds are technically obtainable, the practical utility of two-sided bounds is not deeply explored.

## Related Work & Insights

- **vs. ECE binning methods**: Traditional ECE is a heuristic with no theoretical guarantees and is sensitive to bin configuration choices. The proposed methods provide certifiable probabilistic upper bounds, with the NW method yielding tighter results. ECE, however, remains advantageous in its computational simplicity and ability to produce estimates from small samples (albeit unreliably).
- **vs. KS/Kuiper test methods** (Arrieta-Ibarra et al., 2022): KS-based methods exploit the random-walk properties of cumulative scores and offer high statistical power, but only support a binary "perfectly calibrated or not" judgment, relying on asymptotic theory. The proposed methods quantify the degree of miscalibration, hold non-asymptotically, and are more interpretable.
- **vs. Lipschitz assumption methods** (Vaicenavicius et al., 2019; Futami & Fujisawa, 2024): Prior work directly assumes Lipschitz continuity without justification. The present paper guarantees bounded derivatives (and hence Lipschitz continuity) from first principles via perturbation, and the NW method achieves smaller constant terms and tighter upper bounds.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The idea of guaranteeing smoothness via perturbation is highly novel and elegant, though the overall "surrogate + concentration inequality" framework is a standard technique.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive coverage with synthetic and real data, statistically significant over 64 repetitions; however, the absence of ground truth on real datasets limits full verification.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Mathematically rigorous, logically clear, with an especially well-written practical guidance section—an exemplary work balancing theory and practice.
- **Value**: ⭐⭐⭐⭐ Addresses a foundational theoretical problem in calibration measurement, with direct practical value for high-stakes applications requiring reliable calibration evaluation.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] TabStruct: Measuring Structural Fidelity of Tabular Data](tabstruct_measuring_structural_fidelity_of_tabular_data.md)
- [\[NeurIPS 2025\] Asymmetric Duos: Sidekicks Improve Uncertainty](../../NeurIPS2025/llm_evaluation/asymmetric_duos_sidekicks_improve_uncertainty.md)
- [\[ACL 2026\] MADE: A Living Benchmark for Multi-Label Text Classification with Uncertainty Quantification](../../ACL2026/llm_evaluation/made_a_living_benchmark_for_multi-label_text_classification_with_uncertainty_qua.md)
- [\[NeurIPS 2025\] On the Entropy Calibration of Language Models](../../NeurIPS2025/llm_evaluation/on_the_entropy_calibration_of_language_models.md)
- [\[AAAI 2026\] Sampling Control for Imbalanced Calibration in Semi-Supervised Learning](../../AAAI2026/llm_evaluation/sampling_control_for_imbalanced_calibration_in_semi-supervised_learning.md)

<!-- RELATED:END -->

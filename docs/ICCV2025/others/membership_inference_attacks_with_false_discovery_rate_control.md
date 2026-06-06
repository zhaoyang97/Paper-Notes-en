---
title: >-
  [Paper Note] Membership Inference Attacks with False Discovery Rate Control
description: >-
  [ICCV 2025][membership inference attack] This paper proposes MIAFdR, the first membership inference attack (MIA) method with theoretical false discovery rate (FDR) guarantees. By designing a novel non-member conformity s…
tags:
  - "ICCV 2025"
  - "membership inference attack"
  - "false discovery rate control"
  - "conformal inference"
  - "multiple hypothesis testing"
  - "privacy security"
date: 2026-05-08
content_hash: c2a227d26c466d42
---

# Membership Inference Attacks with False Discovery Rate Control

**Conference**: ICCV 2025
**arXiv**: [2508.07066](https://arxiv.org/abs/2508.07066)  
**Code**: None  
**Area**: Other / AI Security & Privacy
**Keywords**: membership inference attack, false discovery rate control, conformal inference, multiple hypothesis testing, privacy security

## TL;DR
This paper proposes MIAFdR, the first membership inference attack (MIA) method with theoretical false discovery rate (FDR) guarantees. By designing a novel non-member conformity score function and an adjusted membership decision strategy, MIAFdR controls FDR and can be integrated as a plug-and-play wrapper into existing MIA methods, maintaining attack performance while providing FDR guarantees.

## Background & Motivation

**Background**: Membership inference attacks (MIA) aim to determine whether a given data sample was used to train a target model, and represent a core research direction in deep learning privacy and security. Existing approaches include classifier-based methods (shadow training), metric-based methods (softmax, entropy, loss), likelihood ratio-based methods (LiRA), and quantile regression-based methods.

**Limitations of Prior Work**:
   - **Lack of FDR guarantees**: Existing MIA methods cannot provide theoretical guarantees on the false discovery rate. FDR is defined as the proportion of samples identified as training members that are in fact non-members.
   - **Practical harm**: When the actual proportion of members in the test set is high, uncontrolled FDR leads to a large number of false positives, severely undermining the credibility of the attack.
   - **Technical challenges**: (1) The score distribution of non-training data is unknown and difficult to model; (2) estimated non-member probabilities are mutually dependent, violating the independence assumption required by classical multiple hypothesis testing methods.

**Key Challenge**: How can theoretical FDR guarantees be provided without knowledge of the training data distribution and without being able to ensure independence across samples?

**Goal**: To design a MIA method that can (1) provide theoretical FDR control, (2) simultaneously offer marginal probability guarantees (i.e., the probability that a true non-member is misclassified as a member does not exceed $\alpha$), and (3) serve as a wrapper that can be embedded into any existing MIA method.

**Key Insight**: Drawing inspiration from conformal inference, while resolving the issue that conformal p-values cannot be directly applied to FDR control due to their dependence through a shared calibration set.

## Method

### Overall Architecture
MIAFdR consists of three core modules: (1) non-member conformity score computation — designing a conformity score function to quantify how well a test sample conforms to the non-member distribution; (2) non-member relative probability estimation — estimating the relative probability that each test sample is a non-member based on conformity scores; (3) adjusted membership decision — correcting mutually dependent p-values and comparing them against a pre-specified significance level for final decisions.

### Key Designs

1. **Non-member conformity score function**:

    - Train $K$ surrogate models $\{f(\tilde{\theta}^k)\}_{k=1}^K$ on subsets of auxiliary data $D_{au}$.
    - Construct a membership dataset $D_{me}$: predictions on the surrogate models' training sets (labeled 0 = member) ∪ predictions on samples outside the training sets (labeled +1 = non-member).
    - Train a binary classifier $f_{bc}(\theta_{bc})$ to distinguish members from non-members.
    - Conformity score:
    $S(y^t; \theta_{bc}) = \lambda \log\frac{f_{bc}(y^t; \theta_{bc})}{1-f_{bc}(y^t; \theta_{bc})} + (1-\lambda) f_{bc}(y^t; \theta_{bc})$
    A higher score indicates a greater likelihood of being a non-member.

2. **Non-member relative probability estimation**:
    $p(x^t) = \frac{|\{\mathbb{S}^k \in \mathcal{C}_{au}^{2,ca} \cup \{S(y^t;\theta_{bc})\}: \mathbb{S}^k \leq S(y^t;\theta_{bc})\}|}{1 + |\mathcal{C}_{au}^{2,ca}|}$

   **Theorem 1 (Marginal probability guarantee)**: Under the exchangeability assumption, for significance level $\alpha$:
    $\mathcal{P}(p(x^t) \leq \alpha \mid x^t \notin D_{tr}) \leq \alpha$
   That is, the probability of a true non-member being misclassified as a member does not exceed $\alpha$.

3. **Adjusted membership decision**:

    - Problem: p-values are mutually dependent due to the shared calibration set and cannot be directly used with classical methods such as BH.
    - Solution: Sort all p-values in ascending order $\{p^{(t)}\}_{t=1}^T$ and compute the adjusted non-member probability:
    $p_{\text{adj}}^{(t)} = \min\left\{1, \min_{m \in \{t,...,n\}} \frac{n}{m} \cdot p^{(m)}\right\}$
    - Decision: If $p_{\text{adj}}^{(t)} \leq \alpha$, reject the null hypothesis (classify as member).

   **Theorem 2 (FDR control)**:
    $\mathbb{E}\left[\frac{|\mathcal{R}(D_{ts}) \cap \mathcal{H}_0^*(D_{ts})|}{\max\{1, |\mathcal{R}(D_{ts})|\}}\right] \leq \alpha \cdot \frac{\mathcal{H}_0^*(D_{ts})}{T} \leq \alpha$

## Experiments

### Main Results 1: Attack Performance (Classifier-based MIAFdR)

| Dataset | Method | Accuracy (%) | AUROC (%) |
|:---|:---|:---|:---|
| CIFAR-100 | Classifier baseline | 76.81±1.01 | 84.35±0.98 |
| CIFAR-100 | **Classifier+MIAFdR** | **78.19±0.79** | **84.46±0.93** |
| Tiny-ImageNet | Classifier baseline | 69.67±0.85 | 76.99±1.63 |
| Tiny-ImageNet | **Classifier+MIAFdR** | **71.18±1.53** | **77.06±1.52** |

**Key Findings**: MIAFdR not only provides FDR control but also improves attack accuracy (+1.4% on CIFAR-100).

### Main Results 2: FDR Control Effectiveness

| Setting | Method | $\alpha$=0.05 | $\alpha$=0.10 | $\alpha$=0.15 | $\alpha$=0.20 |
|:---|:---|:---|:---|:---|:---|
| Classifier, $\pi_0$=0.5 | MIAFdR | FDR≤0.05 ✓ | FDR≤0.10 ✓ | FDR≤0.15 ✓ | FDR≤0.20 ✓ |
| Metric (Softmax), $\pi_0$=0.5 | MIAFdR | FDR≤0.05 ✓ | FDR≤0.10 ✓ | FDR≤0.15 ✓ | FDR≤0.20 ✓ |
| LiRA, $\pi_0$=0.5 | MIAFdR | - | - | FDR=0.145 ✓ | - |

FDR is effectively controlled across different significance levels and different MIA methods.

### Ablation Study

| Analysis Dimension | Key Findings |
|:---|:---|
| Calibration set size | Larger calibration set → higher attack accuracy and more reliable non-member probability estimation |
| Member/non-member ratio | AUROC remains stable across different ratios, demonstrating robustness |
| Under KD defense | FDR control remains effective; attack accuracy and AUROC are maintained under the defense mechanism |
| Black-box transferability | Attack performance remains robust when surrogate models with different architectures are used |
| Computational overhead | Only ~0.01 seconds of additional inference time for 7,000 samples |
| Machine unlearning | Effectively controls the proportion of falsely reported unlearned samples; accuracy significantly outperforms baselines |
| Continual learning | Effectively controls the proportion of samples incorrectly reported as memorized |

### Key Findings Summary
1. MIAFdR as a wrapper does not degrade and even improves the original MIA's attack performance.
2. FDR is effectively controlled across various settings (gray-box/black-box, classifier-based/metric-based/likelihood ratio-based MIA).
3. The additional computational overhead is negligible (~0.01% increase relative to the original MIA).
4. FDR control and attack effectiveness are maintained under defense mechanisms (knowledge distillation).
5. The framework naturally extends to machine unlearning verification and data memorization evaluation in continual learning.

## Highlights & Insights

1. **First MIA with FDR guarantees**: Fills a gap in theoretical guarantees within the MIA literature; FDR more accurately reflects practical error costs than simple TPR/FPR metrics.
2. **Plug-and-play design**: As a wrapper, MIAFdR can be seamlessly integrated into any existing MIA method without modifying its training procedure.
3. **Theoretical rigor**: The proofs of Theorem 1 (marginal probability guarantee) and Theorem 2 (FDR control) are based on exchangeability rather than the stronger i.i.d. assumption.
4. **Multi-domain applicability**: Applicable not only to privacy attacks but also to machine unlearning verification and memorization evaluation in continual learning.

## Limitations & Future Work

1. An auxiliary dataset $D_{au}$ is required, which, although a common assumption, may not be feasible in certain privacy-sensitive scenarios.
2. The exchangeability assumption, while weaker than i.i.d., may not hold under certain distribution shift scenarios.
3. Evaluation is primarily conducted on classification tasks; membership inference for generative models or segmentation models has not been explored.
4. The tightness of FDR control depends on the size of the calibration set; control may be overly conservative with small calibration sets.

## Related Work & Insights

- **MIA methods**: Shadow Training (Shokri et al., 2017), LiRA (Carlini et al., 2022), Difficulty Calibration (Watson et al., 2021), Quantile Regression Attack (Bertran et al., 2024)
- **Conformal inference**: Conformal prediction, FDR control extensions of conformal inference
- **MIA applications**: Semantic segmentation MIA, healthcare MIA, recommender system MIA
- **Defense methods**: Knowledge distillation defense, differential privacy defense

## Rating
- Novelty: ★★★★☆ (Adapts the conformal inference framework to FDR control in MIA, resolving the technical challenge of p-value dependence)
- Experimental Thoroughness: ★★★★★ (Multiple datasets, diverse attack settings, thorough ablation, and extensions to unlearning/continual learning)
- Value: ★★★★☆ (The wrapper design is highly practical, though the auxiliary data requirement limits applicability in some scenarios)
- Writing Quality: ★★★★☆ (Theoretical derivations are clear, but dense notation raises the reading barrier)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] How Does Bayesian Sampling Help Membership Inference Attacks?](../../ICML2026/others/how_does_bayesian_sampling_help_membership_inference_attacks.md)
- [\[ICCV 2025\] A Hidden Stumbling Block in Generalized Category Discovery: Distracted Attention](a_hidden_stumbling_block_in_generalized_category_discovery_d.md)
- [\[ICCV 2025\] Intra-view and Inter-view Correlation Guided Multi-view Novel Class Discovery](intra-view_and_inter-view_correlation_guided_multi-view_novel_class_discovery.md)
- [\[ICCV 2025\] Generate, Refine, and Encode: Leveraging Synthesized Novel Samples for On-the-Fly Fine-Grained Category Discovery](generate_refine_and_encode_leveraging_synthesized_novel_samples_for_on-the-fly_f.md)
- [\[NeurIPS 2025\] Statistical Inference Under Performativity](../../NeurIPS2025/others/statistical_inference_under_performativity.md)

</div>

<!-- RELATED:END -->

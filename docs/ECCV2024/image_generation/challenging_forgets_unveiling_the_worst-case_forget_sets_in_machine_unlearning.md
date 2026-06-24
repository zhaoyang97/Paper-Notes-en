---
title: >-
  [Paper Note] Challenging Forgets: Unveiling the Worst-Case Forget Sets in Machine Unlearning
description: >-
  [ECCV 2024][Image Generation][machine unlearning] This work proposes a method to identify "worst-case forget sets" from an adversarial perspective. It uses a bi-level optimization framework to find the hardest-to-forget data subsets, and leverages SignSGD to simplify the second-order BLO into a first-order problem, thereby more reliably evaluating the true efficacy of machine unlearning methods.
tags:
  - "ECCV 2024"
  - "Image Generation"
  - "machine unlearning"
  - "Bi-Level Optimization"
  - "Worst-Case Evaluation"
  - "Forget Set Selection"
  - "SignSGD"
date: 2026-05-08
content_hash: 39fb7370266206d9
---

# Challenging Forgets: Unveiling the Worst-Case Forget Sets in Machine Unlearning

**Conference**: ECCV 2024  
**arXiv**: [2403.07362](https://arxiv.org/abs/2403.07362)  
**Code**: [Yes](https://github.com/OPTML-Group/Unlearn-WorstCase)  
**Area**: Trustworthy Machine Learning / Machine Unlearning  
**Keywords**: machine unlearning, Bi-Level Optimization, Worst-Case Evaluation, Forget Set Selection, SignSGD

## TL;DR

This work proposes a method to identify "worst-case forget sets" from an adversarial perspective. It uses a bi-level optimization framework to find the hardest-to-forget data subsets, and leverages SignSGD to simplify the second-order BLO into a first-order problem, thereby more reliably evaluating the true efficacy of machine unlearning methods.

## Background & Motivation

Machine Unlearning (MU) aims to eliminate the influence of specific data from a trained model while maintaining its utility on the remaining data. This field, originating from data protection regulations such as the "right to be forgotten," has expanded to multiple trustworthy ML directions including model safety, fairness, and copyright protection.

However, existing MU evaluations suffer from fundamental reliability flaws:

**High Variance Problem**: Existing evaluations are almost entirely based on randomly selected forget sets. The performance of the same unlearning method varies significantly across different random forget sets (UA variance can reach up to $\pm 0.69$), making fair comparisons across methods difficult.

**Inability to Reveal the Worst-Case**: Random unlearning scenarios fail to expose the true vulnerability of MU methods under extreme conditions, leaving the critical question of "how robust is this unlearning method" unanswered.

**Lack of Systematic Stress Testing**: Compared to standard adversarial evaluation practices in the security field, MU evaluation lacks a comprehensive adversarial benchmark.

The authors experimentally reveal that even for "perfect" exact unlearning methods like Retrain, the variations in UA and MIA under different selections of forget sets are substantial. This motivates them to propose the core research question: **How to systematically find the most challenging data subsets to unlearn?**

## Method

### Overall Architecture

The core of the method is a **Bi-Level Optimization (BLO)** framework, which identifies the worst-case forget set by alternating iterations between upper-level and lower-level optimizations:

- **Upper-level optimization** (data selection level): Optimizes the data selection variable $\mathbf{w}$ so that the unlearned model still maintains low loss on the selected data (i.e., unlearning failure).
- **Lower-level optimization** (unlearning execution level): Performs standard unlearning training given a forget set to obtain the unlearned model $\theta_u(\mathbf{w})$.

A binary selection variable $\mathbf{w} \in \{0,1\}^N$ is introduced, where $w_i=1$ denotes that the $i$-th training sample is selected into the forget set $\mathcal{D}_f$. The complete BLO problem is formulated as:

$$\min_{\mathbf{w} \in \mathcal{S}} \sum_{\mathbf{z}_i \in \mathcal{D}} [w_i \ell(\theta_u(\mathbf{w}); \mathbf{z}_i)] + \gamma \|\mathbf{w}\|_2^2 \quad \text{s.t.} \quad \theta_u(\mathbf{w}) = \arg\min_\theta \ell_{\text{MU}}(\theta; \mathbf{w})$$

The physical meaning of the upper-level objective is: finding a group of data such that the unlearned model still "remembers" them well—namely, the data points where unlearning is hardest to succeed.

### Key Designs

**1. Simplifying Implicit Gradients via SignSGD (Core Innovation)**

The key bottleneck of BLO lies in the implicit gradient term $\frac{d\theta_u(\mathbf{w})}{d\mathbf{w}}$ within the upper-level gradient. Traditional methods require computing second-order derivatives: the Influence Function (IF) method requires the inverse Hessian-gradient product, while the Gradient Unrolling (GU) method must backpropagate along the optimization path, both of which suffer from severe scalability issues on deep networks.

The authors' core insight is to replace the standard SGD with SignSGD as the optimizer in the lower-level optimization:

$$\theta_j = \theta_{j-1} - \beta \cdot \text{sign}(\nabla_\theta \ell_{\text{MU}}(\theta_{j-1}; \mathbf{w}))$$

Since the derivative of the sign function is zero almost everywhere ($\frac{d\,\text{sign}(\mathbf{x})}{d\mathbf{x}} = \mathbf{0}$), every step of the implicit gradient in the gradient unrolling chain is truncated, and eventually, the entire implicit gradient term vanishes: $\text{IG} = \mathbf{0}$. This simplifies the BLO problem, which originally required expensive second-order information, into an alternating optimization requiring only the first-order partial derivative $\nabla_{\mathbf{w}} f$.

**2. Alternating Optimization Solving**

- Upper level: Projected Gradient Descent (PGD) updates $\mathbf{w}$ with 20 iterations and learning rate $\alpha = 10^{-3}$.
- Lower level: SignSGD performs unlearning training with $K = 10$ epochs.
- Constraint relaxation: Relax the binary constraint $\{0,1\}^N$ to a continuous constraint $[0,1]^N$ with $\mathbf{1}^\top \mathbf{w} = m$, where the projection step has a closed-form solution.
- The relaxed continuous $\mathbf{w}$ values can serve as "unlearning difficulty scores": the maximum values correspond to the hardest to forget, and the minimum values correspond to the easiest to forget.

**3. Extension to Class-wise and Prompt-wise Unlearning**

The framework possesses excellent generality: by reinterpreting $\mathbf{w}$ as class selection variables (class-wise forgetting for class unlearning in image classification) or text prompt selection variables (prompt-wise forgetting for concept unlearning in text-to-image models), it can be seamlessly extended to other MU scenarios.

### Loss & Training

- The lower-level unlearning objective is based on the FT strategy: $\ell_f = -\ell_r$ (gradient ascent on the forget set, standard training on the retain set), with $\lambda = 1$.
- The upper-level $\ell_2$ regularization parameter is set to $\gamma = 10^{-4}$, which simultaneously serves the dual purpose of encouraging the sparsity of $\mathbf{w}$ and enhancing the stability of the BLO optimization.

## Key Experimental Results

### Exact Unlearning Validation

On CIFAR-10 / ResNet-18, Retrain (exact unlearning) is used to validate the effectiveness of the worst-case forget set:

| Metric | Unlearning Rate | Random Forget Set | Worst-Case Forget Set | Difference |
|------|--------|-----------|-------------|------|
| UA   | 10%    | $5.28 \pm 0.33$ | $0.00 \pm 0.00$ | ▼5.28 |
| MIA  | 10%    | $12.86 \pm 0.61$ | $0.00 \pm 0.00$ | ▼12.86 |
| RA   | 10%    | 100.00 | 100.00 | — |
| TA   | 10%    | $94.38 \pm 0.15$ | $94.66 \pm 0.09$ | ▲0.28 |

Under four unlearning ratios of 1%/5%/10%/20%, the worst-case forget set consistently depresses UA and MIA to $\approx 0\%$ with zero variance, while the model utility (RA/TA) remains completely undamaged.

### Approximate Unlearning Methods Re-evaluation

Evaluates 9 approximate unlearning methods on CIFAR-10 with a 10% unlearning rate, comparing the average performance gap to Retrain under random vs. worst-case forget sets:

| Method Category | Representative Method | Random Forget Set Avg. Gap | Worst-Case Forget Set Avg. Gap | Trend |
|---------|---------|-------------------|-------------------|------|
| No Re-labeling | FT | 2.00 | 1.37 | Gap narrows |
| No Re-labeling | CF-k | 4.36 | 0.08 | Gap significantly narrows |
| No Re-labeling | SCRUB | 9.91 | 0.82 | Gap significantly narrows |
| Re-labeling | RL | 4.38 | **24.88** | Gap dramatically widens |
| Re-labeling | BS | 6.84 | **37.22** | Gap dramatically widens |
| Re-labeling | BE | 4.28 | **34.70** | Gap dramatically widens |
| Re-labeling | SalUn | 1.89 | **24.51** | Gap dramatically widens |

**Core Finding**: Re-labeling methods (RL, BE, BS, SalUn) expose severe flaws under the worst-case forget set, with MIA soaring to 80-96%, indicating that they provide a "false sense of unlearning safety." Conversely, non-re-labeling methods (FT, CF-k, SCRUB) perform stably, following the trend of Retrain.

### Coreset Analysis

When the complement of the worst-case forget set (i.e., the retain set) is trained on CIFAR-10 and CIFAR-100, the test accuracy is comparable to SOTA coreset selection methods like EL2N and GraNd, and even surpasses the accuracy of the fully trained model under 90%/95% selection ratios on CIFAR-100. This indicates that the essence of the worst-case forget set is a "non-core subset"—the data hardest to forget happens to be the most unimportant redundant data during training.

### Further Validation

- **Multiple Datasets**: Consistent findings across CIFAR-100, CelebA, and Tiny ImageNet.
- **Multiple Models**: Similar trends observed on VGG and ResNet-50.
- **Class-level Unlearning**: Successfully identifies the hardest-to-forget classes on ImageNet / ResNet-18.
- **Prompt-level Unlearning**: Successfully applied to concept unlearning of generative models on UnlearnCanvas / Latent Diffusion.
- **Biased Dataset Case**: In the hair color prediction task on CelebA, the worst-case forget set heavily consists of (Blond + Female) samples—these are the "easy-to-learn" samples with the strongest spurious correlation in the data, confirming the non-core subset hypothesis.

## Highlights & Insights

- **Clever Utilization of SignSGD**: Exploiting the zero derivative of the sign function to degenerate the second-order BLO into a first-order optimization. This simplification is both elegant and practical, representing the most brilliant technical contribution of the paper. It accomplishes the originally costly bi-level optimization using only first-order gradients.
- **Evaluation Paradigm Shift**: Shifting from "random unlearning $\rightarrow$ compute metrics" to "adversarial forget set selection $\rightarrow$ expose the worst-case," providing a more reliable stress-testing benchmark for the MU field.
- **Deep Connection between MU and Coreset**: The hardest-to-forget data $\approx$ the least important training data. This finding connects data selection and machine unlearning, two seemingly independent fields.
- **Reversal of Method Robustness Rankings**: Re-labeling methods appear superior under random evaluations but completely collapse in worst-case scenarios, which provides critical guidance for the selection of MU methods.

## Limitations & Future Work

1. **Lower-Level Unlearning Strategy Fixed to FT**: The lower level of the BLO uniformly employs the FT + gradient ascent strategy. However, different MU methods have their own unique unlearning objective functions; ideally, the worst-case forget set should be solved individually for each method.
2. **Computational Overhead**: Each iteration of the upper level requires executing the full lower-level unlearning training ($K$ epochs), which still incurs high total overhead on large-scale datasets.
3. **Approximation Error of Binary Relaxation**: Continuous relaxation combined with top-$m$ selection can lead to sub-optimal forget sets, especially when the distribution of $\mathbf{w}$ is not sharp enough.
4. **Limited Domain Coverage**: Only covers visual tasks (classification and text-to-image), while knowledge unlearning scenarios in Large Language Models (LLMs) remain unexplored.

## Related Work & Insights

- Unlike existing adversarial metrics such as MIA, this work evaluates MU from the perspective of **data selection**, which is orthogonal and complementary to other performance metrics.
- Unlike saliency-based unlearning methods such as SalUn, this work focuses on improvements on the **evaluation side** rather than the methodology side.
- The BLO framework borrows data selection ideas from coreset selection, but defines the objective function with respect to unlearning difficulty rather than training efficiency.

## Inspirations & Extensions

- This framework can be directly applied to the safety evaluation of LLM knowledge unlearning, such as evaluating whether the safety guardrails after RLHF are truly irreversible.
- The intersection between coreset and MU is worth deeply exploring: the most important training data $\approx$ the hardest-to-forget data, which can be useful for understanding the mechanism of data influence on models.
- The concept of adversarial evaluation can be generalized to other MU variants such as federated unlearning and graph neural network unlearning.

## Rating

| Dimension | Rating | Reason |
|------|------|------|
| Novelty | ⭐⭐⭐⭐ | First to systematically propose the problem of identifying worst-case forget sets |
| Technical Depth | ⭐⭐⭐⭐⭐ | The derivation of using SignSGD to simplify BLO is elegant and exquisite |
| Experimental Thoroughness | ⭐⭐⭐⭐ | Comprehensive validation across multiple datasets, models, and scenarios |
| Writing Quality | ⭐⭐⭐⭐ | Clear motivation of the problem, rigorous mathematical derivation |
| Practical Value | ⭐⭐⭐⭐ | Provides a standardized adversarial evaluation tool for the MU field |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Unveiling Advanced Frequency Disentanglement Paradigm for Low-Light Image Enhancement](unveiling_advanced_frequency_disentanglement_paradigm_for_low-light_image_enhanc.md)
- [\[ICML 2026\] Forget-It-All: Multi-Concept Machine Unlearning via Concept-Aware Neuron Masking](../../ICML2026/image_generation/forget-it-all_multi-concept_machine_unlearning_via_concept-aware_neuron_masking.md)
- [\[ICCV 2025\] Invisible Watermarks, Visible Gains: Steering Machine Unlearning with Bi-Level Watermarking Design](../../ICCV2025/image_generation/invisible_watermarks_visible_gains_steering_machine_unlearning_with_bi-level_wat.md)
- [\[ICLR 2026\] Forget Many, Forget Right: Scalable and Precise Concept Unlearning in Diffusion Models](../../ICLR2026/image_generation/forget_many_forget_right_scalable_and_precise_concept_unlearning_in_diffusion_mo.md)
- [\[ECCV 2024\] NeuSDFusion: A Spatial-Aware Generative Model for 3D Shape Completion, Reconstruction, and Generation](neusdfusion_a_spatial-aware_generative_model_for_3d_shape_completion_reconstruct.md)

</div>

<!-- RELATED:END -->

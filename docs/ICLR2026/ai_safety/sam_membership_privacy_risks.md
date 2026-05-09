---
title: >-
  [Paper Note] Membership Privacy Risks of Sharpness Aware Minimization
description: >-
  [ICLR 2026][AI Safety][Sharpness-Aware Minimization] This paper presents the first systematic study demonstrating that models trained with SAM (Sharpness-Aware Minimization), despite achieving better generalization, are more vulnerable to membership inference attacks (MIA) than SGD-trained models. Two complementary explanations are provided through theoretical analysis and experiments: memorization behavior and variance contraction.
tags:
  - ICLR 2026
  - AI Safety
  - Sharpness-Aware Minimization
  - Membership Inference Attack
  - Privacy Leakage
  - Memorization
  - Variance Reduction
date: 2026-05-08
content_hash: 857c9052a3faf15a
---

# Membership Privacy Risks of Sharpness Aware Minimization

**Conference**: ICLR 2026
**arXiv**: [2310.00488](https://arxiv.org/abs/2310.00488)
**Code**: None
**Area**: AI Security / Privacy
**Keywords**: Sharpness-Aware Minimization, Membership Inference Attack, Privacy Leakage, Memorization, Variance Reduction

## TL;DR

This paper presents the first systematic study demonstrating that models trained with SAM (Sharpness-Aware Minimization), despite achieving better generalization, are more vulnerable to membership inference attacks (MIA) than SGD-trained models. Two complementary explanations are provided through theoretical analysis and experiments: memorization behavior and variance contraction.

## Background & Motivation

**State of the Field**: SAM improves the generalization of deep learning models by seeking flatter loss minima and has become a widely adopted optimization technique. Intuitively, a model with better generalization should rely less on memorizing training data and therefore pose lower privacy risks.

**Limitations of Prior Work**: Yeom et al. formally established that the upper bound on MIA advantage is given by the generalization gap, implying that better generalization should reduce MIA risk. However, the relationship between generalization and privacy in practice is far more complex than this bound suggests, and precedents of utility–privacy tradeoffs exist.

**Root Cause**: SAM improves generalization by better capturing atypical subclass patterns, but this form of "structured memorization" simultaneously leaves stronger traces of training samples in model outputs, thereby increasing privacy leakage.

**Paper Goals**: (1) Systematically verify whether SAM indeed increases MIA risk; (2) explain the root cause through the lens of memorization and influence scores; (3) theoretically prove how SAM's variance contraction effect amplifies MIA advantage.

**Starting Point**: The authors observe that SAM models exhibit smaller variance in output confidence—SGD produces more extreme high-confidence predictions (including on non-members), and these non-members exceeding the decision threshold cause the attacker to make more errors. SAM compresses the variance, making the confidence distributions of members and non-members more separable.

**Core Idea**: Flat minima ≠ privacy safety. SAM's sharpness penalty suppresses over-amplification of dominant features, forcing the model to spread reliance across diverse subclass features. While this improves generalization, it reduces output variance, thereby amplifying the membership inference signal.

## Method

### Overall Architecture

This paper is an analytical study rather than a method proposal. The overall framework consists of: (1) comparing MIA vulnerability of SAM and SGD across datasets and attack methods; (2) analyzing root causes via memorization scores and influence scores; (3) providing theoretical proofs under interpolating solutions for linear models.

### Key Designs

1. **Memorization Analysis**:

    - Function: Reveal the fundamental differences in memorization behavior between SAM and SGD.
    - Mechanism: Use Leave-One-Out (LOO)-based memorization scores $mem(\mathcal{A},\mathcal{D},i)$ to compare the two optimizers. SAM's memorization score distribution is more concentrated in the moderate range (rather than the high end), indicating that SAM focuses on atypical but generalizable subpatterns rather than pure noise.
    - Design Motivation: High memorization does not necessarily imply overfitting to noise—SAM's "structured memorization" selectively focuses on underrepresented subgroups.

2. **Influence Analysis and Generalization Decomposition**:

    - Function: Demonstrate that SAM's generalization gain originates from correctly predicting atypical test samples.
    - Mechanism: Introduce an influence-score-entropy-based metric $\mathcal{I}_{ent}$, partitioning test data into 5 buckets—low-entropy buckets represent atypical test points that heavily rely on a small number of high-memorization training samples, while high-entropy buckets represent typical test points. SAM's gain in low-entropy buckets far exceeds that in high-entropy buckets.
    - Design Motivation: Reveal that SAM does not simply improve all samples uniformly, but specifically benefits atypical samples that require memorization for correct classification.

3. **Variance Contraction Theory**:

    - Function: Prove within a linear model framework that SAM's geometry necessarily reduces non-member output variance.
    - Mechanism: Construct minimum $G$-norm interpolating solutions, where SGD corresponds to $G_0 = I_d$ and SAM corresponds to $G_\eta = I + \eta\Sigma$. It is proved that $\sigma_{G_\eta}^2 < \sigma_{G_0}^2$, i.e., SAM's output variance is strictly smaller. Since the confidence of training samples remains unchanged under interpolation, the decrease in non-member variance directly increases MIA advantage.
    - Design Motivation: Provide a rigorous theoretical foundation for the empirical findings, proving that variance contraction is an intrinsic property of SAM's geometry.

### Loss & Training

This is an analytical work and introduces no new training strategy. The standard SAM objective is used in the analysis: $\min_w \max_{\epsilon \in B(\rho)} L_S(w+\epsilon)$.

## Key Experimental Results

### Main Results

| Dataset | Attack Method | SGD Attack Acc. | SAM Attack Acc. | SGD Test Acc. | SAM Test Acc. |
|--------|----------|---------------|---------------|---------------|---------------|
| CIFAR-100 | Confidence | 77.19% | 79.10% | 80.30% | 81.60% |
| CIFAR-10 | M-entropy | 59.51% | 61.70% | 96.00% | 96.72% |
| EyePacs | Confidence | 73.40% | 77.07% | 73.67% | 75.41% |
| CIFAR-100 | RMIA (AUC) | 90.4% | 91.6% | 67.7% | 69.1% |
| CIFAR-10 | LiRA (TPR@0.1) | 8.8% | 12.5% | 92.3% | 93.1% |

### Ablation Study

| Analysis Dimension | Key Findings |
|---------|---------|
| Memorization Density Distribution | SAM exhibits lower density at the low end and a more uniform distribution in the middle range |
| Generalization Decomposition (Bucket 1 vs. Bucket 5) | SAM achieves the largest gain in the atypical bucket (Bucket 1); nearly no difference in the typical bucket (Bucket 5) |
| Other Sharpness-Aware Optimizers | GSAM, LookSAM, and others exhibit similar patterns of increased privacy risk |
| Different Model Architectures | The phenomenon is consistently reproduced on both ResNet and VGG |

### Key Findings

- SAM is more susceptible to MIA than SGD across all 5 datasets and all attack methods, despite consistently achieving higher test accuracy.
- SAM's memorization gain concentrates in the "moderate memorization" interval (0.6–0.85) rather than the high end (noise memorization), confirming the structured memorization hypothesis.
- On CIFAR-10, SAM's LiRA TPR@0.1%FPR increases from 8.8% to 12.5%, a relative gain of 42%—particularly concerning under strict low false-positive-rate regimes.

## Highlights & Insights

- **Systematic Validation of a Counter-Intuitive Finding**: The paper overturns the naive assumption that "flat minima = good privacy," supported by comprehensive experiments across multiple datasets, attack methods, and architectures. This finding carries important practical implications for the deployment of SAM.
- **Highly Elegant Theoretical Explanation via Variance Contraction**: The problem is reduced to geometric differences between minimum $G$-norm interpolating solutions, with a clear proof chain—sharpness penalty → suppression of high-curvature directions → reduced non-member variance → increased MIA advantage.
- **The Concept of Structured Memorization is Transferable**: Memorization induced by different optimizers and regularization methods is not homogeneous. The framework for distinguishing "beneficial memorization" from "noise memorization" offers reference value for other privacy research.

## Limitations & Future Work

- The theoretical analysis is limited to the perfect interpolation setting for linear models; extension to nonlinear deep networks remains unverified.
- No concrete mitigation strategies for SAM's privacy risks are proposed (e.g., combining differential privacy with SAM).
- Computing memorization and influence scores requires extensive LOO retraining, making the experimental cost prohibitively high and limiting validation on larger-scale models.
- Whether SAM variants (e.g., adaptive SAM, mSAM) could alleviate this issue is not explored.

## Related Work & Insights

- **vs. Yeom et al. (2018)**: They proved that MIA advantage ≤ generalization gap; this paper empirically challenges that intuitive upper bound.
- **vs. Feldman (2020)**: This paper builds on Feldman's memorization framework but is the first to compare memorization patterns across different optimizers.
- **vs. Tan et al. (2022)**: They analyzed the effects of model size and ridge regression on privacy; this paper introduces a curvature-aligned geometric model to characterize SAM.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First to reveal the privacy risk paradox of SAM; the finding is important and counter-intuitive.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Five datasets, multiple attack methods, multi-architecture ablations, and theoretical validation.
- Writing Quality: ⭐⭐⭐⭐ The progressive logic from experiments to analysis to theory is clear; notation is occasionally heavy.
- Value: ⭐⭐⭐⭐ Directly relevant as a warning for real-world systems deploying SAM; the absence of mitigation strategies is a minor shortcoming.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Transferable Backdoor Attacks for Code Models via Sharpness-Aware Adversarial Perturbation](../../AAAI2026/ai_safety/transferable_backdoor_attacks_for_code_models_via_sharpness-aware_adversarial_pe.md)
- [\[ICLR 2026\] Learnability and Privacy Vulnerability are Entangled in a Few Critical Weights](learnability_and_privacy_vulnerability_are_entangled_in_a_few_critical_weights.md)
- [\[AAAI 2026\] Privacy Auditing of Multi-Domain Graph Pre-Trained Model under Membership Inference Attack](../../AAAI2026/ai_safety/privacy_auditing_of_multi-domain_graph_pre-trained_model_under_membership_infere.md)
- [\[NeurIPS 2025\] Unifying Re-Identification, Attribute Inference, and Data Reconstruction Risks in Differential Privacy](../../NeurIPS2025/ai_safety/unifying_re-identification_attribute_inference_and_data_reconstruction_risks_in_.md)
- [\[ICLR 2026\] Unified Privacy Guarantees for Decentralized Learning via Matrix Factorization](unified_privacy_guarantees_for_decentralized_learning_via_matrix_factorization.md)

<!-- RELATED:END -->

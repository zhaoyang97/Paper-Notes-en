---
title: >-
  [Paper Note] Conformal Prediction Adaptive to Unknown Subpopulation Shifts
description: >-
  [ICLR 2026][LLM Evaluation][conformal prediction] To address the failure of standard conformal prediction under subpopulation shift, this paper proposes three adaptive algorithms: weighting calibration data via a learned domain classifier (Algorithms 1/2) or via embedding similarity (Algorithm 3). Coverage guarantees are maintained even with imperfect or absent domain labels, with applications to visual classification and LLM hallucination detection.
tags:
  - ICLR 2026
  - LLM Evaluation
  - conformal prediction
  - distribution shift
  - subpopulation shift
  - uncertainty quantification
  - LLM hallucination
date: 2026-05-08
content_hash: fd4449e166a9f16d
---

# Conformal Prediction Adaptive to Unknown Subpopulation Shifts

**Conference**: ICLR 2026
**arXiv**: [2506.05583](https://arxiv.org/abs/2506.05583)
**Code**: To be confirmed
**Area**: LLM Evaluation
**Keywords**: conformal prediction, distribution shift, subpopulation shift, uncertainty quantification, LLM hallucination

## TL;DR
To address the failure of standard conformal prediction under subpopulation shift, this paper proposes three adaptive algorithms: weighting calibration data via a learned domain classifier (Algorithms 1/2) or via embedding similarity (Algorithm 3). Coverage guarantees are maintained even with imperfect or absent domain labels, with applications to visual classification and LLM hallucination detection.

## Background & Motivation

**Background**: Conformal prediction (CP) provides uncertainty quantification for black-box ML models, guaranteeing marginal coverage under exchangeable data: $\Pr(Y_{\text{test}} \in C_\alpha(X_{\text{test}})) \geq 1-\alpha$.

**Limitations of Prior Work**: In practice, the subpopulation mixture of the test environment often differs from that of calibration data (subpopulation shift), causing standard CP to severely under- or over-cover in certain test environments. Existing solutions either require known distribution shifts (Tibshirani et al.), rely on worst-case thresholds (max CP, which severely over-covers), or demand perfect group labels (group-conditional CP).

**Key Challenge**: Group-conditional CP theoretically resolves subpopulation shift but requires precise group membership information. Theorem 2.1 proves that an imperfect domain classifier leads to severe degradation of coverage guarantees — if the classifier accuracy is $\gamma$, coverage can drop as low as $\max(0, \gamma - \alpha)$.

**Goal**: Design adaptive CP algorithms with theoretical guarantees when domain labels are unknown or imperfect.

**Key Insight**: Rather than requiring a perfect domain classifier, the paper exploits weaker assumptions such as multicalibration and multiaccuracy to ensure coverage; one algorithm requires no domain classifier at all, relying solely on embedding similarity weighting.

**Core Idea**: Use an imperfect domain classifier or embedding similarity to adaptively reweight calibration data, maintaining conformal prediction coverage guarantees under unknown subpopulation shifts.

## Method

### Overall Architecture

The test environment is $\mathbb{P}_{\text{test}} = \sum_{k=1}^K \lambda_k \mathbb{P}_k$, where $\lambda_k$ is unknown and differs from the calibration distribution. The core mechanism is to estimate the domain membership probability $\hat{\lambda}$ for each test point, then compute an adaptive threshold by weighting calibration scores from different domains using $\hat{\lambda}$.

The three algorithms progressively relax assumptions:
1. **Algorithm 1**: Domain classifier available + per-instance weighting (requires multicalibrated classifier)
2. **Algorithm 2**: Domain classifier available + batch-averaged weighting (requires multiaccurate classifier, a weaker condition)
3. **Algorithm 3**: No domain classifier; weighting by embedding similarity (no theoretical guarantee, but empirically effective)

### Key Designs

1. **Algorithm 1: Weighted CP with Domain Classifier (Per-Instance)**:

    - **Function**: For each test point $X_{\text{test}}$, a domain classifier $c(X_{\text{test}})$ predicts the probability $\hat{\lambda}$ of belonging to each domain.
    - **Mechanism**: $\hat{q}_\alpha \leftarrow \min_{\hat{q}} \sum_{k=1}^K \frac{\hat{\lambda}_k m_k(\hat{q}_\alpha)}{n_k + 1} \geq (1-\alpha)$, where $m_k(\hat{q})$ is the number of calibration samples in domain $k$ with scores not exceeding $\hat{q}$.
    - **Theoretical Guarantee (Theorem 3.1)**: If $c$ is the Bayes-optimal classifier, $\Pr(Y_{\text{test}} \in C_\alpha(X_{\text{test}})) \geq 1-\alpha$ is guaranteed.
    - Theorem 3.3 extends this guarantee to multicalibrated classifiers.

2. **Algorithm 2: Weighted CP with Batch Averaging**:

    - **Function**: Replaces per-instance domain probability estimates with the mean over the test set.
    - **Mechanism**: $\hat{\lambda} = \text{mean}_{i=1}^{n_{\text{test}}} c(X_{\text{test}}^i)$, followed by the same weighted threshold computation.
    - **Theoretical Guarantee (Theorem 3.5)**: Coverage is guaranteed under the weaker condition of multiaccuracy (as opposed to multicalibration).
    - **Design Motivation**: Multiaccuracy imposes lower computational and sample complexity than multicalibration and is easier to satisfy in practice.

3. **Algorithm 3: Similarity-Weighted CP (No Domain Classifier)**:

    - **Function**: Requires no domain labels or domain classifier; reweights calibration data by embedding-space similarity to the test point.
    - **Mechanism**:
        - Retain the top $\beta$ fraction of calibration data ranked by embedding similarity to the test point.
        - Apply softmax weighting: $\gamma_i = h(z(X_{\text{test}}), z(X_i'))$, $m = \text{Softmax}(\{\gamma_i/\sigma\})$.
        - Compute the weighted quantile as the threshold.
    - **Design Motivation**: Semantically similar data points are more likely to originate from the same domain; similarity thus serves as a proxy for domain membership.

4. **Theorem 2.1 (Key Theoretical Contribution)**:

    - **Function**: Proves coverage degradation of group-conditional CP when the domain classifier is imperfect.
    - **Core Result**: There exist distributions under which coverage degrades to $\max(0, \gamma - \alpha)$, where $\gamma$ is the conditional accuracy of the classifier.
    - This fundamentally demonstrates why a naive plug-in of an imperfect classifier into group-conditional CP is insufficient.

### Loss & Training

- Domain classifier training: the pretrained backbone is frozen; only the final 3 FC layers (2048→1024→512→K) are trained using Adam with cross-entropy loss.
- Post-training calibration via multi-domain temperature scaling.
- LLM hallucination detection: GPT-4o serves as the correctness evaluator; LLaMA-3-8B is used as the generation model.

## Key Experimental Results

### Main Results

Coverage distribution across 100 test environments on ImageNet (26 domains, ViT, LAC score, $\alpha=0.05$):

| Method | Mean Coverage | Std. Dev. | Remarks |
|--------|--------------|-----------|---------|
| Target Coverage | 0.950 | — | Ideal |
| Standard CP | ~0.95 | **High** | Severe under-coverage in some environments |
| Max CP | ~0.99 | Low | Severe over-coverage |
| Conditional Calibration | ~0.94 | Medium | Under-coverage in certain environments |
| Algorithm 1 (A1) | ~0.95 | **Low** | Closely tracks target |
| Algorithm 2 (A2) | ~0.95 | **Low** | Closely tracks target |
| Oracle | ~0.95 | **Lowest** | Ideal upper bound |
| Algorithm 3 (A3) | ~0.95 | Low | Effective without domain information |

### Ablation Study

| Dimension | Finding |
|-----------|---------|
| Score function (HPS/APS/LAC) | Consistently effective across all |
| Model architecture (ResNet50/ViT/CLIP-ViT) | Consistently effective across all |
| Shift magnitude ($\alpha'=0.1/0.5/1.0$) | Larger shift → larger gap between A1–A3 and Standard CP |
| LLM hallucination detection (LLaMA-3-8B) | A3 significantly reduces std. dev. of recall |

### Key Findings
- **Standard CP is unreliable under shift**: A substantial fraction of the 100 test environments exhibit severe under-coverage, with high variance.
- **Max CP is overly conservative**: While coverage is guaranteed, the severe over-coverage (~0.99 vs. target 0.95) yields excessively large prediction sets, limiting practical utility.
- **A1/A2 closely track the target**: Both mean coverage and standard deviation approach the oracle, indicating that the multicalibrated/multiaccurate classifier assumptions hold in practice.
- **A3 is effective without domain information**: Reweighting by embedding similarity alone maintains coverage across most environments, making it the most practically accessible method.
- **LLM hallucination detection**: Standard CP exhibits high variance in recall across test environments; A3 significantly reduces this variance, yielding more reliable behavior.

## Highlights & Insights
- **Theorem 2.1 reveals a fundamental flaw in group-conditional CP**: Imperfect group information can cause complete failure of coverage guarantees (coverage can drop to $\gamma - \alpha$), not merely a minor degradation. This provides strong motivation for the proposed approach.
- **Assumption relaxation chain from Bayes-optimal → multicalibrated → multiaccurate**: The paper elegantly weakens requirements on the domain classifier in stages while preserving coverage guarantees, allowing practitioners to select the appropriate algorithm based on classifier quality.
- **Unsupervised adaptation in Algorithm 3**: With no domain information whatsoever, embedding similarity alone achieves comparable performance — making the method directly applicable to any pretrained model.

## Limitations & Future Work
- **Theory does not exploit sample independence**: Current theoretical analysis does not leverage independence across domain samples, leading to mild over-coverage under small shifts.
- **Algorithm 3 lacks formal guarantees**: Despite empirical effectiveness, it does not admit coverage guarantees analogous to those of A1/A2.
- **No guidance on score function selection**: No theoretical or empirical guidance is provided regarding which score function is optimal in which setting.
- **Limited LLM experiments**: Hallucination detection is evaluated only on short-answer QA; more complex generation tasks are not addressed.
- **Single-risk control only**: Extending the framework to simultaneously control multiple risks (hallucination, toxicity, sycophancy, etc.) remains an important direction for future work.

## Related Work & Insights
- **vs. Tibshirani et al. (2020)**: That work requires known covariate likelihood ratios, which is infeasible in high-dimensional ML settings. The proposed methods only require a learned domain classifier or similarity approximation.
- **vs. Gibbs et al. (2024, Conditional Calibration)**: Also employs a two-stage domain classifier approach but assumes perfect group information. Theorem 2.1 in this paper exposes the fragility of that assumption and offers a more robust alternative.
- **vs. Max/Robust CP (Cauchois et al.)**: Coverage is guaranteed but overly conservative. The proposed methods adaptively match the actual test distribution rather than optimizing for the worst case.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of multicalibration/multiaccuracy with CP is novel; Theorem 2.1 is of practical significance.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated across vision and LLM domains, multiple models, score functions, and shift magnitudes; A3 lacks theoretical analysis.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are clear; the assumption relaxation chain is logically coherent; algorithm pseudocode is readable.
- Value: ⭐⭐⭐⭐ Directly contributes to the reliability of CP in practical deployment; the LLM hallucination detection application shows strong potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Semi-Supervised Conformal Prediction With Unlabeled Nonconformity Score](../../CVPR2026/llm_evaluation/semi-supervised_conformal_prediction_with_unlabeled_nonconformity_score.md)
- [\[ICLR 2026\] Multi-LLM Adaptive Conformal Inference for Reliable LLM Responses](multi-llm_adaptive_conformal_inference_for_reliable_llm_responses.md)
- [\[ICLR 2026\] Truthfulness Despite Weak Supervision: Evaluating and Training LLMs Using Peer Prediction](truthfulness_despite_weak_supervision_evaluating_and_training_llms_using_peer_pr.md)
- [\[NeurIPS 2025\] Document Summarization with Conformal Importance Guarantees](../../NeurIPS2025/llm_evaluation/document_summarization_with_conformal_importance_guarantees.md)
- [\[NeurIPS 2025\] Leveraging Robust Optimization for LLM Alignment under Distribution Shifts](../../NeurIPS2025/llm_evaluation/leveraging_robust_optimization_for_llm_alignment_under_distribution_shifts.md)

</div>

<!-- RELATED:END -->

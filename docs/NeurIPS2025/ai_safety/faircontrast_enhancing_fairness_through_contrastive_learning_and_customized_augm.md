---
title: >-
  [Paper Note] FairContrast: Enhancing Fairness through Contrastive Learning and Customized Augmentation
description: >-
  [NeurIPS 2025][AI Safety][Fairness] FairContrast proposes a fair contrastive learning framework for tabular data. By strategically selecting positive pairs—pairing advantaged-group samples with favorable outcomes against…
tags:
  - "NeurIPS 2025"
  - "AI Safety"
  - "Fairness"
  - "Contrastive Learning"
  - "Tabular Data"
  - "Representation Learning"
  - "Demographic Parity"
date: 2026-05-08
content_hash: 74e95e348ec53671
---

# FairContrast: Enhancing Fairness through Contrastive Learning and Customized Augmentation

**Conference**: NeurIPS 2025
**arXiv**: [2510.02017](https://arxiv.org/abs/2510.02017)
**Code**: None
**Area**: AI Safety
**Keywords**: Fairness, Contrastive Learning, Tabular Data, Representation Learning, Demographic Parity

## TL;DR

FairContrast proposes a fair contrastive learning framework for tabular data. By strategically selecting positive pairs—pairing advantaged-group samples with favorable outcomes against their disadvantaged-group counterparts—and training end-to-end with supervised or self-supervised contrastive loss combined with cross-entropy loss, the framework achieves significant bias reduction with minimal accuracy loss, without introducing any additional fairness constraint losses.

## Background & Motivation

**Background**: Deep learning models tend to inherit social biases present in training data, resulting in unfair treatment of groups defined by sensitive attributes. Classic examples include recidivism prediction models exhibiting racial bias and hiring models discriminating against female candidates. Representation learning, particularly contrastive learning, has demonstrated strong robustness and generalization in CV and NLP domains.

**Limitations of Prior Work**:
   - State-of-the-art contrastive learning methods for tabular data (VIME, SCARF) do not account for fairness, and the learned representations still encode sensitive attribute bias.
   - Fair contrastive learning methods from the vision domain (e.g., CCL) apply Gaussian noise as an augmentation strategy for tabular data, which is semantically inappropriate for discrete and categorical features.
   - Existing fair representation learning methods (FCRL, CVIB, adversarial forgetting) typically require additional fairness constraint modules or adversarial training, increasing model complexity.

**Core Problem**: Can fairness be naturally embedded into the standard contrastive learning process solely through carefully designed pairing strategies, without any additional fairness constraint losses?

## Method

### Overall Architecture

The core mechanism of FairContrast:
1. Input samples are mapped to a representation space via an encoder: $z = \text{Enc}(x)$.
2. Positive pairs are selected through a customized positive-pair sampling strategy.
3. End-to-end training is performed with a contrastive loss (supervised or self-supervised) plus binary cross-entropy loss.
4. The learned fair representations can be applied to arbitrary downstream tasks.

### Key Designs

**1. Positive Pair Sampling Strategy (Core Innovation)**

For each anchor sample, the positive pair selection rule is as follows:

- **Non-advantaged group or unfavorable outcome**: Paired with a sample from the same class and same sensitive attribute group. For example, a low-income female is paired with another low-income female. This preserves intra-subgroup feature similarity.
- **Advantaged group with favorable outcome**: Paired with a sample from the disadvantaged group that also has a favorable outcome. For example, a high-income male is paired with a high-income female.

Core intuition: Pulling advantaged-group samples with favorable outcomes toward disadvantaged-group samples with favorable outcomes encourages the model not to rely on sensitive attributes for making favorable predictions, aligning with the equal opportunity fairness criterion.

**Negative Pair Strategy**:
- Supervised: Samples from different classes serve as negative pairs.
- Self-supervised: All samples in the batch other than the positive pair serve as negative pairs.

**2. Self-Supervised Contrastive Loss (InfoNCE)**

$$L_{\text{self}} = -\frac{1}{N} \sum_i \log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k \neq i} \exp(\text{sim}(z_i, z_k)/\tau)}$$

where $\text{sim}$ denotes cosine similarity and $\tau$ is the temperature parameter.

**3. Supervised Contrastive Loss**

$L_{\text{sup}}$ extends InfoNCE to multiple positive pairs: for anchor $i$, all samples in the same class $P(i)$ serve as positive pairs, while samples from different classes serve as negative pairs. This encourages all same-class samples to form more compact clusters in the embedding space.

### Loss & Training

**Total Loss**: $L_{\text{total}} = \alpha \cdot L_{\text{BCE}} + L_{\text{SCL}}$

$\alpha$ controls the balance between classification loss and contrastive loss. Ablation studies show that the fairness–accuracy trade-off stabilizes when $\alpha > 1$.

**Theoretical Analysis (Mutual Information Decomposition)**:

Under the Markov assumption on positive pairs, the mutual information between positive pairs can be decomposed as:

$$I(Z; Z^+) = I(Z; Y) + (1 - \pi) \cdot I(Z; S \mid Y)$$

where $\pi = \Pr[S^+ \neq S \mid Y=1]$ is the cross-group pairing probability.

**Key Theorem (InfoNCE as Equivalent Information Bottleneck)**:

$$\arg\min_\theta L_{\text{NCE}}(\theta) = \arg\max_\theta \{ I(Z; Y) - \lambda \cdot I(Z; S \mid Y) \}$$

where $\lambda = 1 - \pi$. This implies that through the pairing strategy alone, standard contrastive learning naturally becomes an information bottleneck: preserving label-relevant information $I(Z; Y)$ while suppressing conditionally sensitive information $I(Z; S \mid Y)$. The trade-off coefficient is entirely data-driven, requiring no additional hyperparameter tuning.

**Training Configuration**: Adam optimizer, lr=0.001, $\tau=1$, 100 epochs, NVIDIA RTX 3090 GPU.

## Key Experimental Results

### Main Results

Accuracy and Demographic Parity (DP, lower is fairer) on three fairness benchmark datasets:

| Dataset | Model | Accuracy | DP |
|--------|------|----------|-----|
| Adult | Unfair MLP | 84.5 | 0.1855 |
| | FCRL | 83.29 | 0.1600 |
| | CVIB | 81.28 | 0.1350 |
| | SCARF | 82.13 | 0.1848 |
| | VIME-self | 84.47 | 0.1779 |
| | FairContrast-sup | 84.4 | 0.0255 |
| | FairContrast-unsup | 84.4 | 0.1201 |
| German | Unfair MLP | 78.5 | 0.3125 |
| | CVIB | 69.5 | 0.0244 |
| | VIME-self | 78.0 | 0.0482 |
| | FairContrast-sup | 78.0 | 0.0099 |
| Health | Unfair MLP | 84.64 | 0.6468 |
| | FCRL | 78.27 | 0.4407 |
| | VIME-self | 84.22 | 0.6192 |
| | FairContrast-sup | 84.3 | 0.4135 |

Key Findings:
- On the Adult dataset, the supervised variant achieves a DP of only 0.0255 (vs. 0.1855 for Unfair MLP), a reduction of approximately 86%, with virtually no accuracy loss.
- On the German dataset, DP drops to 0.0099 (near-perfect fairness) while maintaining 78% accuracy.
- General contrastive learning methods such as SCARF and VIME that ignore fairness exhibit bias levels comparable to Unfair MLP.

### Ablation Study

**Loss Weight $\alpha$ Ablation** (Adult dataset):
- Evaluated using AOC (area under the fairness–accuracy Pareto curve).
- Both supervised and self-supervised settings stabilize when $\alpha > 1$.
- Smaller $\alpha$ allows the contrastive loss to dominate, potentially over-suppressing useful information; larger $\alpha$ allows the classification loss to dominate.

**Counterfactual Data Augmentation Comparison**:
- Counterfactual augmentation (flipping sensitive attributes) can also be integrated into the framework.
- However, it underperforms FairContrast's strategic pairing (DP = 0.1639 vs. 0.0255 on Adult).

### Key Findings

- Significant debiasing can be achieved through carefully designed pairing strategies alone, without adversarial training or additional constraints.
- The supervised variant consistently outperforms the self-supervised variant, indicating that label information is critical for fair representation learning.
- VIME and SCARF perform similarly to the undebiased MLP in terms of fairness, demonstrating that general-purpose contrastive learning does not automatically improve fairness.
- Theoretically, the fairness–accuracy trade-off coefficient $\lambda = 1 - \pi$ is shown to be entirely determined by the data distribution.

## Highlights & Insights

1. **Minimalist Elegance**: Without introducing adversarial components, fairness constraint losses, or architectural modifications, the framework transforms standard contrastive learning into a fair information bottleneck through pairing strategy alone—achieving maximal effect with minimal modification.
2. **Theory–Practice Alignment**: The mutual information decomposition precisely reveals how the pairing strategy automatically balances label information preservation and sensitive information suppression.
3. **Strong Generalizability**: The learned fair representations can be directly applied to arbitrary downstream tasks, offering cross-task transfer potential.
4. **An 86% reduction in DP on the Adult dataset with virtually no accuracy sacrifice** demonstrates that fairness and accuracy need not be in conflict.

## Limitations & Future Work

1. Only group fairness (demographic parity) is considered; individual fairness is not addressed.
2. The current focus is on a single fairness metric (DP), whereas multiple fairness definitions may conflict in practice.
3. Validation is limited to tabular data; extensions to image, text, and multimodal settings remain unexplored.
4. The positive pair strategy requires known sensitive attribute labels, making it inapplicable when sensitive attributes are unobservable.
5. Experimental datasets are relatively small (Adult: 48K, German: 1K, Health: 50K); validation at large scale is lacking.

## Related Work & Insights

- **SupCon** (Khosla et al., NeurIPS 2020): Foundation for supervised contrastive loss.
- **SCARF** (Bahri et al., 2021): State-of-the-art contrastive learning for tabular data, without fairness consideration.
- **FCRL** (Gupta et al., AAAI 2021): Fair representation learning based on mutual information estimation.
- **CCL** (Park et al.): Conditional contrastive learning for fairness, but with augmentation strategies unsuited to tabular data.
- Insight: Many fairness problems can be addressed through data-level pairing and sampling strategies rather than complex model-level constraints.

## Rating

4/5 — Excellent. The method is elegant and concise, the theoretical analysis is complete, and the experiments provide thorough comparisons. The core insight—that the pairing strategy is equivalent to an information bottleneck—is particularly impressive. The primary limitation is that validation is restricted to tabular data and group fairness settings.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Mitigating Disparate Impact of Differentially Private Learning through Bounded Adaptive Clipping](mitigating_disparate_impact_of_differentially_private_learning_through_bounded_a.md)
- [\[ICCV 2025\] Backdooring Self-Supervised Contrastive Learning by Noisy Alignment](../../ICCV2025/ai_safety/backdooring_self-supervised_contrastive_learning_by_noisy_alignment.md)
- [\[NeurIPS 2025\] Enhancing Graph Classification Robustness with Singular Pooling](enhancing_graph_classification_robustness_with_singular_pooling.md)
- [\[NeurIPS 2025\] FedFACT: A Provable Framework for Controllable Group-Fairness Calibration in Federated Learning](fedfact_a_provable_framework_for_controllable_group-fairness_calibration_in_fede.md)
- [\[NeurIPS 2025\] Incentivizing Time-Aware Fairness in Data Sharing](incentivizing_time-aware_fairness_in_data_sharing.md)

</div>

<!-- RELATED:END -->

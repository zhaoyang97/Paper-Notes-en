---
title: >-
  [Paper Note] Versatile Incremental Learning: Towards Class and Domain-Agnostic Incremental Learning
description: >-
  [ECCV 2024][LLM Evaluation][Incremental Learning] This work is the first to define the Versatile Incremental Learning (VIL) scenario—where the class or domain incremental type of subsequent tasks is unknown. It proposes the ICON framework, which controls learning directions via CAST loss to avoid conflicts with historical tasks, and dynamically expands output nodes with the IC incremental classifier to address the cross-domain intra-class overwriting problem…
tags:
  - "ECCV 2024"
  - "LLM Evaluation"
  - "Incremental Learning"
  - "Catastrophic Forgetting"
  - "Class-Domain Joint Incremental"
  - "Adaptation Shift Control"
  - "Incremental Classifier"
date: 2026-05-08
content_hash: 24fd54efc9dfb192
---

# Versatile Incremental Learning: Towards Class and Domain-Agnostic Incremental Learning

**Conference**: ECCV 2024  
**arXiv**: [2409.10956](https://arxiv.org/abs/2409.10956)  
**Code**: [KHU-AGI/VIL](https://github.com/KHU-AGI/VIL)  
**Area**: LLM Evaluation  
**Keywords**: Incremental Learning, Catastrophic Forgetting, Class-Domain Joint Incremental, Adaptation Shift Control, Incremental Classifier

## TL;DR

This work is the first to define the Versatile Incremental Learning (VIL) scenario—where the class or domain incremental type of subsequent tasks is unknown. It proposes the ICON framework, which controls learning directions via CAST loss to avoid conflicts with historical tasks, and dynamically expands output nodes with the IC incremental classifier to address the cross-domain intra-class overwriting problem, comprehensively surpassing existing CIL/DIL methods on three benchmarks.

## Background & Motivation

Incremental Learning (IL) aims to continuously accumulate knowledge from sequentially arriving tasks while overcoming catastrophic forgetting. Existing IL scenarios are divided into two categories:

- **Class IL (CIL)**: Different classes across tasks under the same domain (e.g., continuously learning new classes of objects)
- **Domain IL (DIL)**: Different domains across tasks for the same classes (e.g., the same classes under different weather/environments)

**Core Problem**: Existing methods strongly assume that subsequent tasks only introduce new classes or only introduce new domains, whereas in reality, both variations may appear randomly. For instance, in autonomous driving, the model must learn new object classes while adapting to new environmental conditions, without knowing beforehand what the next task will introduce.

**New Challenges Introduced by VIL**:

**Intra-class domain confusion**: Distribution discrepancies of the same class across different domains lead to overwriting of classifier weights.

**Inter-domain class confusion**: DIL methods assume constant classes and fail to adapt when encountering new classes.

**Classifier drift**: Classifier weights are heavily overwritten when learning new domains of existing classes.

## Method

### Overall Architecture

ICON (Incremental Classifier with Adaptation Shift cONtrol) is based on a frozen ViT backbone + trainable adapter architecture, consisting of two core components:

1. **CAST (Cluster-based Adaptation Shift conTrol)**: Cluster-based regularization of adapter weight shifts to control the learning direction.
2. **IC (Incremental Classifier)**: Dynamically expands classifier output nodes based on class difficulty.

### Key Designs

1. **CAST Loss — Cluster-based Adaptation Shift Control**:

Key Observation: When the IL type changes between consecutive tasks (e.g., switching from CIL to DIL), the shift directions of the adapter weights differ significantly. Without constraints, the learning directions of the model across different task types will conflict with each other.

Mechanism:
- Before and after learning each task, the adapter weight difference $V_{t-1} = A_{t-1}^{after} - A_{t-1}^{prev}$ is recorded and stored in a shift pool.
- K-Means clustering is performed on all historical shifts in the shift pool.
- During current task training, the shift at the current iteration is calculated as $V_t^i = A_t^i - A_t^{prev}$.
- The cluster $S_t^i$ to which $V_t^i$ belongs is identified, and shifts in other clusters $S_t^{i'}$ are treated as "different types" of historical learning directions.
- The current shift is regularized to be orthogonal to the shifts in different clusters:

$$\mathcal{L}_{CAST} = \sum_j w_j \cdot \frac{V_t^i \cdot V_j}{\|V_t^i\| \|V_j\|}$$

where $w_j = \frac{\|V_t^i - V_j\|_2}{\sum_{V_k \in S_t^{i'}} \|V_t^i - V_k\|_2}$, $V_j \in S_t^{i'}$

The weight $w_j$ assigns larger regularization weights to more distant historical shifts, thereby differentially controlling the learning directions. The mathematical essence of the shift difference is the accumulated gradient (derived from the gradient descent formula), so the shift direction is equivalent to the learning direction.

2. **IC — Incremental Classifier**:

To address the problem where classifier weights are overwritten when "the same class appears in different domains" in VIL, the IC dynamically expands the classifier's output nodes as needed:

- **Dynamic Threshold Decision**: For each class $i$, compute the gap between the average accuracy on learned domains and the accuracy on the new domain:

$$\delta_i = \tanh(p_i), \quad p_i = \gamma \cdot \frac{\frac{1}{|D^{prev}|}\sum_{d \in D^{prev}} Acc(C_i^d) - Acc(C_i^{d_{new}})}{\frac{1}{|D^{prev}|}\sum_{d \in D^{prev}} Acc(C_i^d)}$$

If the new domain's accuracy is significantly lower than the historical average (i.e., this class is "difficult" in the new domain), a new output node is added for it.

- **Node Selection Strategy**: During inference, the maximum logit is taken among multiple nodes of the same class (based on energy-based model theory—nodes with lower energy, i.e., higher logits, are more in-distribution).
- **Knowledge Distillation**: For the unselected old nodes, a KL-divergence loss is used to distill knowledge from the classifier of the previous task.

### Loss & Training

$$\mathcal{L}_{Total} = \beta \mathcal{L}_{CAST} + \mathcal{L}_{IC}$$

where $\mathcal{L}_{IC} = \mathcal{L}_{CE}(O^t, y) + \alpha \mathcal{L}_{KL}(O^t, O^{t-1})$. ViT parameters are frozen, and only adapter and classifier parameters are updated.

## Key Experimental Results

### Main Results

Average Accuracy (%) in the VIL scenario:

| Method | iDigits | CORe50 | DomainNet | Average |
|------|---------|--------|-----------|------|
| Fine-tuning | 19.89 | 14.04 | 20.35 | 18.09 |
| L2P | 59.07 | 64.85 | 48.98 | 57.63 |
| CODA-Prompt | 63.30 | 69.28 | 49.45 | 60.68 |
| LAE | 59.34 | 77.11 | 49.01 | 61.82 |
| **ICON (Ours)** | **75.11** | **83.18** | **53.37** | **70.55** |

Average Accuracy across all scenarios (CIL+DIL+VIL):

| Method | iDigits | CORe50 | DomainNet |
|------|---------|--------|-----------|
| CODA-Prompt | 70.95 | 74.52 | 58.73 |
| LAE | 68.12 | 75.89 | 55.26 |
| **ICON** | **77.15** | **84.34** | **59.74** |

### Ablation Study

Ablation of CAST and IC in the VIL scenario (Avg. Acc %):

| CAST | IC | iDigits | CORe50 | DomainNet | Average |
|------|-----|---------|--------|-----------|------|
| ✗ | ✗ | 59.34 | 77.11 | 49.01 | 61.82 |
| ✓ | ✗ | 68.34 | 79.20 | 50.56 | 66.03 |
| ✗ | ✓ | 66.97 | 81.13 | 51.60 | 66.57 |
| ✓ | ✓ | **75.11** | **83.18** | **53.37** | **69.98** |

Further decomposition of IC (iDigits VIL):

| Node Expansion | Knowledge Distillation | Avg. Acc | Forgetting |
|---------|---------|----------|------------|
| ✗ | ✗ | 59.34 | 29.32 |
| ✓ | ✗ | 63.10 | 25.50 |
| ✓ | ✓ | **66.97** | **14.32** |

### Key Findings

- Both CAST and IC make significant individual contributions (around +4% on average), and their combination exhibits synergistic effects.
- The number of clusters $K=2$ is optimal for short sequences (iDigits, 20 tasks), while $K=3$ is optimal for long sequences (CORe50, 40 tasks).
- Performing node expansion alone (without distillation) already yields a considerable improvement, indicating that output node separation itself is critical to resolving weight overwriting.
- ICON also achieves SOTA in Cross-Domain IL (72.88% average vs. CODA-P 69.08%), proving the method is effective in VIL sub-scenarios as well.

## Highlights & Insights

1. **Valuable Scenario Definition**: VIL unifies CIL and DIL, which is closer to the real-world. The significant degradation of existing methods on VIL demonstrates that it is a problem highly worth studying.
2. **Ingenious Design of CAST**: It indirectly measures learning directions through weight shifts (= cumulative gradients), then uses clustering to distinguish between "similar" and "different" historical tasks, concentrating regularization on "different type" tasks—avoiding one-size-fits-all regularization.
3. **Low-cost Expansion of IC**: Unlike DER/DyTox which expand the entire network, IC only expands a few nodes in the final classifier layer with extremely low cost.

## Limitations & Future Work

1. In the VIL setting, the number of classes and domains per task is fixed. The paper also acknowledges that "varying numbers of classes/domains is a more realistic scenario".
2. CAST relies on K-Means clustering, where the number of clusters $K$ requires hyperparameter tuning, and the optimal value varies across datasets.
3. The dynamic threshold calculation of IC requires evaluating the classifier accuracy across domains, introducing additional computational overhead.
4. The method is only validated on the ViT+adapter architecture, and its applicability to other architectures (such as CNN-based) remains unknown.
5. The domain differences in the three datasets are relatively distinct (e.g., different styles of handwritings). More subtle domain variations might pose a greater challenge.

## Related Work & Insights

- **CODA-Prompt / DualPrompt / L2P**: Prompt-based IL methods that perform well on CIL but degrade significantly on VIL.
- **S-Prompts**: A domain-specific DIL method that degrades severely in VIL due to the lack of incremental class handling.
- **DER / DyTox**: Model expansion methods that do not address the issue of cross-domain overwriting for the same class.
- **EWC / LwF**: Classic regularization methods that show limited effectiveness in VIL.

## Rating

- Novelty: ⭐⭐⭐⭐ — The definition of the VIL scenario is practically meaningful, and both CAST and IC are reasonable new designs.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Evaluated on three datasets, three scenarios, multiple baselines, with detailed ablations and analyses.
- Writing Quality: ⭐⭐⭐⭐ — The motivation is clearly stated, and the formulation of the method is complete; despite many data tables, they are well-organized.
- Value: ⭐⭐⭐⭐ — This work pioneers the new VIL scenario, and the method achieves comprehensive SOTA performance across scenarios, showing both theoretical value and practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Image-Feature Weak-to-Strong Consistency: An Enhanced Paradigm for Semi-Supervised Learning](image-feature_weak-to-strong_consistency_an_enhanced_paradigm_for_semi-supervise.md)
- [\[ECCV 2024\] Instance-dependent Noisy-label Learning with Graphical Model Based Noise-rate Estimation](instance-dependent_noisy-label_learning_with_graphical_model_based_noise-rate_es.md)
- [\[ICLR 2026\] In-Context Learning for Pure Exploration](../../ICLR2026/llm_evaluation/in-context_learning_for_pure_exploration.md)
- [\[ICML 2025\] Sample Efficient Demonstration Selection for In-Context Learning](../../ICML2025/llm_evaluation/sample_efficient_demonstration_selection_for_in-context_learning.md)
- [\[ICLR 2026\] Detecting Data Contamination in LLMs via In-Context Learning](../../ICLR2026/llm_evaluation/detecting_data_contamination_in_llms_via_in-context_learning.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Harnessing Feature Resonance under Arbitrary Target Alignment for Out-of-Distribution Node Detection
description: >-
  [NeurIPS 2025][OOD detection] This paper discovers the *Feature Resonance* phenomenon—when optimizing the representations of known in-distribution (ID) nodes…
tags:
  - "NeurIPS 2025"
  - "OOD detection"
  - "graph neural networks"
  - "feature resonance"
  - "unsupervised"
  - "node-level"
  - "label-agnostic"
date: 2026-05-08
content_hash: 60fd73a1c9ddb049
---

# Harnessing Feature Resonance under Arbitrary Target Alignment for Out-of-Distribution Node Detection

**Conference**: NeurIPS 2025
**arXiv**: [2502.16076](https://arxiv.org/abs/2502.16076)  
**Code**: [ShenzhiYang2000/RSL](https://github.com/ShenzhiYang2000/RSL)  
**Area**: Other
**Keywords**: OOD detection, graph neural networks, feature resonance, unsupervised, node-level, label-agnostic

## TL;DR

This paper discovers the *Feature Resonance* phenomenon—when optimizing the representations of known in-distribution (ID) nodes, unknown ID nodes undergo significantly larger representational changes than OOD nodes, and this phenomenon is label-agnostic. Based on this observation, the authors propose RSL, a graph OOD node detection framework that requires no multi-class labels, achieving state-of-the-art performance across 13 datasets.

---

## Background & Motivation

**Background**: OOD node detection on graphs is a critical task for ensuring the deployment reliability of GNNs. Existing methods (MSP, Energy, KNN, NNGuide, etc.) fall into two broad categories: classifier-output-based (entropy/energy scores) and supervised-representation-based (KNN distance), both of which rely heavily on a pre-trained multi-class classifier.

**Limitations of Prior Work**:
- **Strong label assumptions**: Multi-class labels must be available and the upstream task must be classification.
- **Limited applicability**: Many real-world OOD detection scenarios do not satisfy the classification prerequisite—generative models, regression tasks, reinforcement learning, and one-class detection all lack multi-class labels.
- **Severe under-exploration of unsupervised node-level OOD detection on graphs**: Only a handful of works such as EnergyDef exist, and there remains substantial room for performance improvement.

**Key Challenge**: Existing methods are coupled to classifier outputs and label spaces, making them inapplicable to label-free settings. Meanwhile, local connectivity structures among graph nodes encode rich distributional information. Exploiting this information without relying on labels is the core challenge.

**Key Insight**: The paper shifts focus from label space to feature space, examining behavioral differences between ID and OOD nodes during representation optimization, and proposes an OOD detection method that is entirely independent of labels and upstream tasks.

---

## Core Idea

### Feature Resonance Phenomenon

The authors identify a key phenomenon:

> When the representations of known ID nodes are optimized—even when aligned to **arbitrary random vectors**—the representations of unknown ID nodes change **significantly more** than those of OOD nodes.

**Physical Analogy**: This resembles resonance in forced oscillation—when the external forcing frequency matches the natural frequency of the oscillator, the amplitude is maximized. ID nodes "resonate" due to their shared underlying data manifold, while OOD nodes respond weakly because they belong to a different manifold structure.

**Core Insight**: Feature Resonance is label-agnostic, stemming from the intrinsic manifold relationships among ID node representations. This makes it naturally suited for class-label-free, task-agnostic OOD detection.

---

## Method

### 1. Macroscopic Feature Resonance

The feature trajectory metric is defined as:

$$\hat{F}(\tilde{\mathbf{x}}_i) = \sum_t h_{\theta_{t+1}}(\tilde{\mathbf{x}}_i) - h_{\theta_t}(\tilde{\mathbf{x}}_i)$$

where $h_{\theta_t}$ is the model at epoch $t$. Under ideal conditions, $\|\hat{F}\|$ is larger for ID nodes than for OOD nodes. However, on complex real-world data, the full-trajectory metric is corrupted by noise in early training and overfitting in later stages, limiting its reliability.

### 2. Microscopic Feature Resonance

To address the instability of macroscopic trajectories, the paper introduces **single-step representational change** as a microscopic proxy:

$$\tau_i = \|\Delta h_{\theta_t}(\tilde{\mathbf{x}}_i)\|_2 = \|h_{\theta_{t+1}}(\tilde{\mathbf{x}}_i) - h_{\theta_t}(\tilde{\mathbf{x}}_i)\|_2$$

A key finding is that Feature Resonance is **not sustained throughout training**: it is most pronounced in the middle training phase—early on, the model searches for an optimization path causing confusion; in the middle phase, once a path aligned with ID patterns is found, resonance peaks; in the late phase, overfitting causes resonance to dissipate. A simple ID/OOD validation set suffices to precisely locate the resonance period.

### 3. Arbitrary Target Alignment Training

The representations of known ID nodes are aligned to a randomly generated target vector $e$:

$$\ell(h_{\theta_t}(\mathbf{X}_{\text{known}}), e) = \mathbb{E}(\|\mathbf{1}^\top e - \mathbf{X}_{\text{known}} \mathbf{W}^\top\|_2^2)$$

Key empirical validation shows that regardless of whether the alignment target is true multi-class labels, random multi-class labels, or a single random vector, the $\tau$ of unknown ID nodes consistently exceeds that of OOD nodes, confirming the label-agnostic nature of Feature Resonance.

### 4. Synthetic OOD Node Augmentation (RSL Framework)

Building on the microscopic resonance scores, the framework incorporates a synthetic OOD node strategy to further improve detection performance:

- **Candidate OOD selection**: The $n$ wild nodes with the smallest $\tau$ values are selected as the candidate OOD set $\mathcal{V}_{\text{cand}}$.
- **SGLD synthesis**: Stochastic gradient Langevin dynamics is applied to generate synthetic OOD nodes from the candidate OOD nodes, making them closer to the true OOD distribution:

$$\hat{\mathbf{x}}_j^{(t+1)} = \lambda\big(\hat{\mathbf{x}}_j^{(t)} - \frac{\alpha}{2}\nabla E_\theta(\hat{v}_j^{(t)}) + \epsilon\big) + (1-\lambda)\mathbb{E}_{\mathbf{x} \sim \mathbf{X}_{\text{cand}}}(\mathbf{x} - \hat{\mathbf{x}}_j^{(t)})$$

- **OOD classifier training**: A binary classifier is trained using known ID nodes (label 1) together with candidate and synthetic OOD nodes (label 0), optimized with binary cross-entropy loss.

### 5. Theoretical Guarantees

The paper provides an upper bound on the OOD separability error of the resonance score $\tau$ (Theorem 1): when the number of known ID samples $n$ and wild samples $m$ is sufficiently large and the optimal ID risk $R_{in}^*$ is sufficiently small, the OOD misclassification rate $\text{ERR}_{\text{out}}$ is bounded above and can approach zero (Theorem 2).

---

## Key Experimental Results

### Main Results — Unsupervised OOD Node Detection (9 Datasets)

| Method | YelpChi AUROC↑ | Amazon AUROC↑ | Reddit AUROC↑ | Requires Labels |
|--------|---------------|--------------|--------------|----------------|
| EnergyDef | 62.04 | 86.57 | 63.32 | No |
| GRASP (pseudo-labels) | 58.05 | 70.31 | 51.82 | K-means pseudo-labels |
| **RSL** | **66.11** | **90.03** | **64.83** | **No** |

On YelpChi/Amazon/Reddit, RSL improves over the previous SOTA by an average of 3.01%/7.09%/8.95% in AUROC/AUPR/FPR95, respectively.

### Ablation Study

| Variant | Description | Amazon FPR95↓ |
|---------|-------------|--------------|
| RSL w/o classifier | Resonance score $\tau$ only | 19.56 |
| RSL w/o $\mathcal{V}_{\text{syn}}$ | No synthetic OOD nodes | 25.18 |
| **RSL (full)** | Resonance + synthesis + classifier | **19.60** |

Using only the resonance score $\tau$ already reduces average FPR95 by 9.70% compared to the previous SOTA, validating the standalone effectiveness of Feature Resonance.

### Effect of Different Alignment Targets (WikiCS)

| Alignment Target | AUROC↑ | AUPR↑ |
|-----------------|--------|-------|
| True multi-class labels | 71.03 | 72.47 |
| Random multi-class vectors | 73.64 | 74.13 |
| Single random vector | **79.15** | **78.65** |

A single random vector yields the best results—simple alignment targets are sufficient to elicit resonance while avoiding label noise interference.

### Advantage on Heterophilic Graphs

On heterophilic graphs (Squirrel, WikiCS, Chameleon), RSL reduces FPR95 by an average of 14.93% over the previous SOTA, as RSL does not rely on graph homophily assumptions.

---

## Highlights & Insights

- **Feature Resonance is the central contribution**: It reveals a fundamental difference in the representational dynamics of ID and OOD nodes during training, offering an entirely new perspective on OOD detection. This finding may extend beyond graph-structured data and is worth exploring in other modalities.
- **Elegance of label-agnosticity**: Aligning to an arbitrary random vector is sufficient to induce the resonance effect, completely eliminating dependence on upstream classification tasks and multi-class labels.
- **Automatic resonance period localization**: A simple ID/OOD validation set is used to identify the epoch of maximal microscopic resonance during training, which is practically convenient.
- **Synergy between synthetic OOD nodes and resonance**: Using resonance scores to select candidate OOD nodes as guidance for SGLD generation produces synthetic nodes closer to the true OOD distribution, outperforming the unguided generation in EnergyDef.

## Limitations & Future Work

- **Validation set assumption**: Although multi-class labels are not required, a binary-annotated (ID/OOD) validation set is still needed to locate the resonance period, which may be difficult to obtain in some settings.
- **Computational overhead**: The method requires a full training run and per-epoch computation of representational changes for all nodes, which may present efficiency bottlenecks on large-scale graphs.
- **Stability of the resonance period**: The location of the resonance period may vary across datasets, model architectures, and hyperparameter settings; the robustness of automated localization warrants further investigation.
- **Future directions**:
    - Explore Feature Resonance in non-graph modalities such as images and text.
    - Develop validation-set-free methods for automatic resonance period localization.
    - Combine resonance scores with other self-supervised approaches.

## Related Work & Insights

- **vs. EnergyDef (Gong & Sun, 2024)**: Both are unsupervised graph OOD detection methods, but EnergyDef generates synthetic OOD nodes without guidance; RSL uses resonance scores to select candidate OOD nodes for guided generation, achieving comprehensive performance gains.
- **vs. GRASP (Ma et al., 2024)**: GRASP is the current supervised SOTA, but its performance degrades substantially when K-means pseudo-labels are used; RSL surpasses the pseudo-label version of GRASP without using any labels.
- **vs. SSD (Sehwag et al., 2021)**: SSD uses self-supervised learning to avoid label dependency but is inefficient on graphs (exceeds time limits on some datasets); RSL is both more efficient and more effective.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Feature Resonance is a genuinely new discovery that offers a fresh perspective on OOD detection.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 13 datasets, 21 baselines, extensive ablations and analyses.
- Writing Quality: ⭐⭐⭐⭐ Strong physical intuition via analogy, complete theoretical analysis, clearly organized experiments.
- Value: ⭐⭐⭐⭐ A significant advance in unsupervised graph OOD detection with broadly applicable label-agnostic properties.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Double Descent Meets Out-of-Distribution Detection: Theoretical Insights and Empirical Analysis](double_descent_meets_out-of-distribution_detection_theoretical_insights_and_empi.md)
- [\[NeurIPS 2025\] Redundancy-Aware Test-Time Graph Out-of-Distribution Detection](redundancy-aware_test-time_graph_out-of-distribution_detection.md)
- [\[NeurIPS 2025\] Out-of-distribution Generalisation is Hard: Evidence from ARC-like Tasks](out-of-distribution_generalisation_is_hard_evidence_from_arc-like_tasks.md)
- [\[ICML 2026\] Target-Agnostic Calibration under Distribution Shift with Frequency-Aware Gradient Rectification](../../ICML2026/others/target-agnostic_calibration_under_distribution_shift_with_frequency-aware_gradie.md)
- [\[NeurIPS 2025\] EPHAD: An Evidence-Based Post-Hoc Adjustment Framework for Anomaly Detection Under Data Contamination](an_evidence-based_post-hoc_adjustment_framework_for_anomaly_detection_under_data.md)

</div>

<!-- RELATED:END -->

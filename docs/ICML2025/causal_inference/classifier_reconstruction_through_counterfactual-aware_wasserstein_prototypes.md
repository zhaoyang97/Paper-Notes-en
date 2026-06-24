---
title: >-
  [Paper Note] Classifier Reconstruction Through Counterfactual-Aware Wasserstein Prototypes
description: >-
  [ICML 2025][Causal Inference][Counterfactual Explanation] This paper proposes using Wasserstein barycenters to fuse original and counterfactual samples into class prototypes, enabling high-fidelity reconstruction of target binary classifiers under limited query budgets and effectively mitigating the decision boundary shift problem caused by the naive use of counterfactual samples.
tags:
  - "ICML 2025"
  - "Causal Inference"
  - "Counterfactual Explanation"
  - "Wasserstein Barycenter"
  - "Model Reconstruction"
  - "Prototypical Classification"
  - "Decision Boundary"
date: 2026-05-08
content_hash: 5f52dd3e93e6b440
---

# Classifier Reconstruction Through Counterfactual-Aware Wasserstein Prototypes

**Conference**: ICML 2025  
**arXiv**: [2512.10878](https://arxiv.org/abs/2512.10878)  
**Code**: None  
**Area**: Causal Inference  
**Keywords**: Counterfactual Explanation, Wasserstein Barycenter, Model Reconstruction, Prototypical Classification, Decision Boundary

## TL;DR

This paper proposes using Wasserstein barycenters to fuse original and counterfactual samples into class prototypes, enabling high-fidelity reconstruction of target binary classifiers under limited query budgets and effectively mitigating the decision boundary shift problem caused by the naive use of counterfactual samples.

## Background & Motivation

Counterfactual Explanations (CE) provide users with actionable insights by identifying the minimal input modifications that flip model predictions (e.g., "if income increases by 10k, the loan application will be approved"). However, counterfactuals can also expose internal model structures: attackers can leverage counterfactual queries to train surrogate models, thereby achieving model extraction and threatening the intellectual property of MLaaS platforms. On the other hand, model reconstruction also has positive use cases—applicants can estimate their approval probability without formally submitting sensitive data.

**Core Problems of Existing Methods**:

**Decision Boundary Shift**: Prior work (Aïvodji et al., 2020) directly treats counterfactuals as labeled training samples. Since counterfactuals naturally lie close to the decision boundary, under class imbalance or one-sided counterfactual scenarios (e.g., only "rejection $\to$ approval" direction), the decision boundary of the surrogate model significantly shifts away from that of the target model.

**Overfitting**: Although the Counterfactual Clamping loss (Dissanayake & Dutta, 2024) handles counterfactuals by modifying the cross-entropy loss, it is prone to overfitting under limited query samples, especially when employing complex neural networks as surrogate models.

**Two-sided Query Constraints**: While two-sided counterfactual queries can mitigate the shift, in reality, only one-sided counterfactuals are often available (e.g., only counterfactuals for rejected applications).

**Motivation**: Although counterfactual samples are highly informative (being close to the decision boundary), they cannot replace the representative samples of the two classes. How can one leverage the counterfactual information while avoiding boundary shift under limited queries? The authors' **Key Insight** is to treat counterfactuals as "soft samples" of both classes (with label 0.5) and calculate class prototypes along with the original data based on optimal transport theory.

## Method

### Overall Architecture

The overall workflow of the proposed method is as follows:

1. **Data Organization**: Partition the data into three categories—class 0 sample set $\mathcal{D}_0$, class 1 sample set $\mathcal{D}_1$, and counterfactual sample set $\mathcal{D}_{cf}$ (assigned a soft label of 0.5).
2. **Wasserstein Barycenter Computation**: For each class $c \in \{0, 1\}$, compute a barycenter distribution $\mathbb{Q}_c$ that fuses the original samples and the counterfactual samples.
3. **Classification Inference**: For a new sample, compute its Wasserstein distances to both barycenters, and predict the class with the smaller distance.

The target model $m: \mathbb{R}^d \to [0,1]$ is a binary classifier that outputs probability scores, determining classes with a threshold of 0.5. The counterfactual generator $g_m$ is triggered only when the model predicts class 0 (i.e., a one-sided counterfactual setting). The goal is to learn a surrogate model $\hat{m}$ to replicate the behavior of the target model with a minimal number of queries.

### Key Designs

#### 1. Soft Labeling of Counterfactuals

Traditional approaches directly treat counterfactuals as training samples for the target class, which introduces boundary shift. This paper assigns a soft label of 0.5 to counterfactuals:

$$\mathcal{Y} = \{0, 0.5, 1\}$$

This design offers two advantages:

- **Mitigating Class Imbalance**: Counterfactuals are no longer unilaterally assigned to a single class, avoiding the boundary shift caused by an excessive number of samples on one side.
- **Exploiting Boundary Information**: The 0.5 label naturally expresses the semantic meaning that counterfactuals are located between the two classes.

#### 2. Class Prototypes Based on Wasserstein Barycenters

For each class $c$, a barycenter distribution $\mathbb{Q}_c$ is computed to be close to both the original distribution $\mathbb{P}_c$ of that class and the counterfactual distribution $\mathbb{P}_{cf}$ of that class:

$$\mathbb{Q}_c = \arg\min_{\mathbb{Q} \in \mathbb{P}(\mathcal{X})} \left( W_2^2(\mathbb{Q}, \mathbb{P}_c) + \lambda_c W_2^2(\mathbb{Q}, \mathbb{P}_{cf}) \right)$$

where $\lambda_c = 0.5$ balances the influence of the original distribution and the counterfactual distribution. The 2-Wasserstein distance is used as the metric between distributions here; it accounts for the underlying spatial geometry and is better suited to capturing distribution shapes compared to metrics like KL divergence.

Unlike traditional Prototypical Networks (Snell et al., 2017) which use embedding space means as prototypes, the Wasserstein barycenter preserves intra-class distribution variability, providing more robust class representations in few-shot scenarios.

#### 3. Symmetry Regularization

To ensure that counterfactuals maintain a symmetric position between the prototypes of the two classes (i.e., equidistant from both barycenters), a regularization term is introduced:

$$\mathcal{R}(\mathbb{Q}_0, \mathbb{Q}_1) = \left( W_2(\mathbb{Q}_0, \mathbb{P}_{cf}) - W_2(\mathbb{Q}_1, \mathbb{P}_{cf}) \right)^2$$

This regularization penalizes the asymmetry of distances from the counterfactuals to the two barycenters, encouraging the decision boundary to pass exactly through the central region of the counterfactual distribution, thereby aligning with the decision boundary of the target model.

#### 4. Classification Rule with Margin

For a test sample $x$, classification is performed based on the Wasserstein distances from its Dirac distribution $\delta_x$ to the two barycenters:

$$\hat{y}(x) = \begin{cases} 0 & \text{if } W_2(\delta_x, \mathbb{Q}_0) < W_2(\delta_x, \mathbb{Q}_1) - \tau \\ 1 & \text{if } W_2(\delta_x, \mathbb{Q}_1) < W_2(\delta_x, \mathbb{Q}_0) - \tau \end{cases}$$

The threshold parameter $\tau \geq 0$ introduces a margin, preventing overly confident predictions near the decision boundary.

### Loss & Training

The overall optimization objective merges the class barycenter loss with the symmetry regularization:

$$\min_{\mathbb{Q}_0, \mathbb{Q}_1} \sum_{c \in \{0,1\}} \left( W_2^2(\mathbb{Q}_c, \mathbb{P}_c) + \lambda_c W_2^2(\mathbb{Q}_c, \mathbb{P}_{cf}) \right) + \gamma \mathcal{R}(\mathbb{Q}_0, \mathbb{Q}_1)$$

- $\lambda_c = 0.5$: Weight of the influence of counterfactuals on the two class prototypes
- $\gamma > 0$: Intensity of the symmetry regularization
- The free-support barycenter algorithm in the Python Optimal Transport (POT) library (Flamary et al., 2021) is used for iterative solving.

Key Training Strategy Points:

- **No Neural Network Training Required**: Unlike Baseline 2, which requires training a surrogate classifier, the proposed method directly computes Wasserstein barycenters through an optimization problem, avoiding the risk of overfitting inherent in neural network training.
- **Low Query Requirement**: The method achieves high fidelity with only a few hundred queries, matching scenarios with constrained query budgets.

## Key Experimental Results

### Main Results

On four tabular datasets (Adult Income, COMPAS, DCCC, and HELOC), the proposed method is compared with two SOTA methods (using logistic regression as the target classifier and MCCF as the counterfactual generation method):

| Dataset | Query Count | Baseline 1 | Baseline 2 | **Ours** | Gain (vs B2) |
|--------|--------|-----------|-----------|---------|-------------|
| Adult Income | 500 | 91±3.2 | 94±3.2 | **96±2.5** | +2.0 |
| COMPAS | 500 | 92±3.2 | 94±2.0 | **96±2.3** | +2.0 |
| DCCC | 500 | 89±8.9 | 91±0.9 | **97±1.5** | +6.0 |
| HELOC | 500 | 91±4.7 | 93±2.2 | **95±2.0** | +2.0 |
| Adult Income | 300 | 87±3.8 | 90±3.8 | **93±3.2** | +3.0 |
| COMPAS | 300 | 88±3.8 | 90±2.6 | **94±3.0** | +4.0 |
| DCCC | 300 | 85±9.5 | 87±1.5 | **93±2.1** | +6.0 |
| HELOC | 300 | 87±5.3 | 89±2.8 | **93±2.6** | +4.0 |

### Ablation Study

The impact of different counterfactual generation methods on fidelity on the Adult dataset:

| Counterfactual Method | Characteristics / Property | Fidelity Performance | Description / Explanation |
|-----------|------|----------|------|
| MCCF (Wachter) | L2 distance minimization | High | Default method, robust performance |
| L1-Sparse | Sparse changes | High | Minimal feature changes |
| DiCE | Actionability constraints | High | Supports immutable features |
| Nearest Neighbor | Feasibility (nearest neighbor) | High | Close to the data manifold |
| C-CHVAE | VAE-based generation | Lower | Generative models perform poorly on tabular data |
| ROAR | Robustness | High | Robust against model drift |

### Key Findings

1. **The fewer the queries, the larger the advantage**: When decreasing queries from 500 to 300, the fidelity decline of the proposed method is much smaller than that of the two baselines, demonstrating its superiority under low query budgets.
2. **Greater stability under complex surrogate models**: When Baseline 2 uses a more complex neural network as the surrogate model, its fidelity actually decreases (due to overfitting under constrained queries), whereas the proposed method, relying on Wasserstein barycenters, does not suffer from this issue.
3. **Significant impact of counterfactual quality**: Counterfactuals that lie closer to the data manifold (such as nearest-neighbor approaches) yield better results, while VAE-based C-CHVAE performs poorly due to limited generation quality.
4. **Lower variance**: In most settings, the standard deviation of the proposed method is smaller than that of the two baselines (e.g., dropping from 8.9 to 1.5 on the DCCC dataset), indicating higher stability in the results.

## Highlights & Insights

1. **Novel Perspective**: It formulates the "side effect" of counterfactual explanations (exposing model information) into a formal optimization problem, and elegantly solves it by leveraging optimal transport theory.
2. **No Neural Network Training**: Departing from the dominant surrogate model training paradigm, the proposed method directly computes distribution barycenters as prototypes, which naturally circumvents overfitting in low-data regimes.
3. **Unification of Theory and Practice**: The organic combination of the soft label (0.5), Wasserstein barycenters, and symmetry regularization is supported by clear geometric intuition and theoretical backing for each component.
4. **Comprehensive Comparison of Counterfactual Generation**: It systematically evaluates the impact of six different counterfactual generation methods on model reconstruction, which is a rare analysis in the literature.

## Limitations & Future Work

1. **Limited to Binary Classification**: The current method is tailored to binary classifiers. Extending it to multi-class scenarios requires handling more complex multi-class prototypes and boundaries.
2. **Evaluated only on Tabular Data**: The experiments are conducted exclusively on tabular datasets; the effectiveness on high-dimensional data, such as images or text, remains unexplored.
3. **Target Model Constraints**: The target model in the experiments is restricted to logistic regression. The reconstruction effectiveness on more complex target classifiers (e.g., deep neural networks) remains to be verified.
4. **Computational Cost of Wasserstein Distance**: Exact computation of Wasserstein distances in high-dimensional spaces is computationally heavy; approximate schemes like Sliced Wasserstein or Sinkhorn distance could be considered.
5. **Fixed One-sided Counterfactual Assumption**: The study only considers counterfactuals in the direction of class 0 $\rightarrow$ class 1. More flexible settings incorporating mixed one-sided/two-sided counterfactuals could be explored.

## Related Work & Insights

- **Aïvodji et al. (2020)**: First to propose model reconstruction leveraging counterfactuals (Baseline 1), applying a naive approach that directly treats counterfactuals as training samples.
- **Dissanayake & Dutta (2024)**: Proposed the Counterfactual Clamping loss to improve surrogate training (Baseline 2), which is however limited by overfitting under small query counts.
- **Prototypical Networks (Snell et al., 2017)**: The concept in prototypical networks of using class means as prototypes for few-shot learning inspired this paper; however, this work replaces simple means with Wasserstein barycenters.
- **Optimal Transport (Flamary et al., 2021, POT)**: Provides an efficient implementation of Wasserstein barycenters.
- **Insight**: The approach in this work can be extended to model privacy analysis in federated learning—specifically, evaluating under what information-sharing conditions an adversary can reconstruct local models.

## Rating

| Metric | Score (1-5) | Description |
|------|-----------|------|
| Novelty | 4 | Introducing optimal transport to counterfactual model reconstruction is a novel combination |
| Technical Depth | 4 | Rigorous mathematical derivation and reasonable framework design |
| Experimental Thoroughness | 3 | Limited datasets and restricted to tabular data and simple models |
| Writing Quality | 4 | Clear problem definition and well-motivated approach |
| Practicality | 3 | Restricted to binary tabular data, offering limited application scenarios |
| **Overall** | **3.5** | Elegant methodology but needs enhancement in experimental scale and scenario generalization |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Adjusting Prediction Model Through Wasserstein Geodesic for Causal Inference](../../ICLR2026/causal_inference/adjusting_prediction_model_through_wasserstein_geodesic_for_causal_inference.md)
- [\[ICCV 2025\] A Visual Leap in CLIP Compositionality Reasoning through Generation of Counterfactual Sets](../../ICCV2025/causal_inference/a_visual_leap_in_clip_compositionality_reasoning_through_gen.md)
- [\[ICML 2025\] Learning Time-Aware Causal Representation for Model Generalization in Evolving Domains](learning_time-aware_causal_representation_for_model_generalization_in_evolving_d.md)
- [\[ICML 2025\] Exogenous Isomorphism for Counterfactual Identifiability](exogenous_isomorphism_for_counterfactual_identifiability.md)
- [\[ICML 2025\] Transformer-Based Spatial-Temporal Counterfactual Outcomes Estimation](transformer-based_spatial-temporal_counterfactual_outcomes_estimation.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Fine-Grained Scene Graph Generation via Sample-Level Bias Prediction
description: >-
  [ECCV2024][Optimization][Scene Graph Generation] This paper proposes a sample-level bias prediction method named SBP. By leveraging a Bias-Oriented GAN, it utilizes the contextual information of the union region of object pairs to predict sample-specific bias correction vectors, reforming coarse-grained relationships into fine-grained ones. SBP outperforms dataset-level bias correction methods by an average of 5.6%/3.9%/3.2% in Average@K on VG/GQA/VG-1800 datasets…
tags:
  - "ECCV2024"
  - "Optimization"
  - "Scene Graph Generation"
  - "Long-Tailed Distribution"
  - "Bias Correction"
  - "GAN"
  - "Fine-Grained Relationships"
date: 2026-05-08
content_hash: 28490bdc8e007e17
---

# Fine-Grained Scene Graph Generation via Sample-Level Bias Prediction

**Conference**: ECCV2024  
**arXiv**: [2407.19259](https://arxiv.org/abs/2407.19259)  
**Code**: [Zhuzi24/SBG](https://github.com/Zhuzi24/SBG)  
**Area**: Graph Learning  
**Keywords**: Scene Graph Generation, Long-Tailed Distribution, Bias Correction, GAN, Fine-Grained Relationships

## TL;DR

This paper proposes a sample-level bias prediction method named SBP. By leveraging a Bias-Oriented GAN, it utilizes the contextual information of the union region of object pairs to predict sample-specific bias correction vectors, reforming coarse-grained relationships into fine-grained ones. SBP outperforms dataset-level bias correction methods by an average of 5.6%/3.9%/3.2% in Average@K on VG/GQA/VG-1800 datasets, respectively.

## Background & Motivation

Scene Graph Generation (SGG) extracts objects and their relationships from images to construct structured semantic graphs $\mathcal{G}=\{(o_i, r_{i\to j}, o_j)\}$, which widely support downstream tasks such as visual question answering, image retrieval, and image captioning. However, SGG suffers from a severe **long-tailed distribution problem**: using the Visual Genome (VG) dataset as an example, coarse-grained head classes (e.g., "on", "has", "wearing") have abundant samples, whereas fine-grained tail classes (e.g., "walking on", "part of", "flying in") are extremely scarce. This causes model predictions to heavily bias towards coarse-grained head relationships, yielding uninformative scene graphs.

Existing bias correction methods can be categorized as **dataset-level bias correction**: DLFE estimates the frequency of each label class $\mathbf{c}$ and divides the biased probability by $\mathbf{c}$ to restore unbiased probabilities, while RTPB learns a resistance bias $\mathbf{b}$ and subtracts it from classification logits to enhance tail class detection. The common limitation of both methods is that they apply a single global bias correction vector uniformly to all object pairs, neglecting the contextual differences across different object pairs. For example, the union regions of ⟨man, beach⟩ and ⟨cat, table⟩ contain drastically different visual semantics, which warrants differentiated bias correction strategies.

Key Insight: The union region of each object pair contains rich and exclusive contextual information. This can be exploited to predict **sample-specific bias correction offsets**, elevating the bias correction scale from the dataset level to the sample level.

## Core Problem

How to predict an exclusive bias correction vector $\mathbf{b}_s$ for each object pair, enabling the model to specifically refine coarse-grained relationships (e.g., "on") into fine-grained ones (e.g., "walking on") while avoiding the over-suppression of head classes to achieve an optimal balance between R@K and mR@K?

## Method

The overall framework, SBG (Sample-Level Bias Prediction for Fine-Grained SGG), follows the standard two-stage SGG pipeline, with its core innovation being the proposed SBP module for sample-level bias correction. It consists of three stages:

### Stage 1: Classical SGG Model Training

Faster R-CNN (ResNeXt-101-FPN backbone) is employed to detect objects, and then classical models like Motif, VCtree, or Transformer are used to predict relationships. This stage yields a biased relationship prediction vector $\mathbf{z}=[z_1, z_2, \ldots, z_M]$, where $M$ is the number of relation categories. Due to the long-tailed effect, the scores of head categories in $\mathbf{z}$ are disproportionately high.

### Stage 2: Constructing the Bias Correction Set $\mathcal{S}$

Theoretical Motivation: Ideally, there exists a difference vector $\mathbf{b}_u = \mathbf{z}_u - \mathbf{z}$ between the unbiased prediction $\mathbf{z}_u$ and the biased prediction $\mathbf{z}$, but $\mathbf{z}_u$ is inaccessible. Alternatively, as long as we can find a vector $\hat{\mathbf{z}}$ that satisfies $\text{argmax}(\text{softmax}(\hat{\mathbf{z}})) = \text{argmax}(\text{softmax}(\mathbf{z}_u))$ (i.e., making the prediction correct, without requiring complete unbiasedness), we can correct the bias via $\hat{\mathbf{z}} = \mathbf{z} + \mathbf{b}_s$.

Specific construction workflow:

1. Utilize the union region feature $f_{uni}$ of the object pair (by extracting visual features using a feature extractor sharing weights with Faster R-CNN, and then fusing spatial features).
2. Map high-dimensional features to one-dimensional vectors via a single-layer Transformer encoder $\phi$, and fuse them with global bias: $\mathbf{b}^{tru} = \phi(f_{uni}) + \mathbf{b}^{glo}$.
3. The global bias is $\mathbf{b}^{glo} = -\log(w^a / \sum_{j \in M} w_j^a + \epsilon)$, calculated from the relationship category weights based on dataset statistics.
4. Verify if the corrected prediction $r_{pre} = \text{argmax}(\text{softmax}(\mathbf{z} + \mathbf{b}^{tru}))$ equals the ground truth $r_{tru}$.
5. If satisfied, directly add it to $\mathcal{S}$; otherwise, calculate the difference $d = \hat{\mathbf{z}}[r_{tru}] - \hat{\mathbf{z}}[r_{pre}]$, update $\mathbf{b}^{tru}[r_{pre}] = \mathbf{b}^{tru}[r_{pre}] + d + \varepsilon$, and then add it to $\mathcal{S}$.

### Stage 3: Bias-Oriented GAN (BGAN) Training

Freeze the parameters of the classical SGG model $\Lambda^O$ from Stage 1, and train the BGAN to learn to predict the bias correction offsets.

**Generator G** (5-layer 1D CNN):

- Input: union feature $f_{uni}$, global bias $\mathbf{b}^{glo}$, and raw prediction $\mathbf{z}$
- Output: predicted bias $\mathbf{b}^{pre} = \Upsilon(\Lambda^{B_G}; f_{uni}, \mathbf{b}^{glo}, \mathbf{z})$
- Loss: $\mathcal{L}_G = -\text{mean}(\mathcal{T}_G) + \alpha \cdot \mathcal{L}_{CE}(\hat{\mathbf{z}})$, where the GAN loss guides fitting to the true bias distribution and the cross-entropy loss constrains the correction result's accuracy.

**Discriminator D** (3-layer 1D CNN):

- Input: predicted bias $\mathbf{b}^{pre}$ or true bias $\mathbf{b}^{tru}$
- Output: discriminator scores $\mathcal{T}_G$ and $\mathcal{T}_S$
- Loss: $\mathcal{L}_D = -\text{mean}(\mathcal{T}_S) + \text{mean}(\mathcal{T}_G)$

Training Strategy: The discriminator is updated 5 times and the generator is updated 1 time per iteration. The optimizer used is RMSProp, with learning rates of 0.0001 for G and 0.0005 for D. During inference, only the classical SGG model + Generator G are used, and the correction formula is:

$$\hat{\mathbf{z}} = \mathbf{z} + \mathbf{b}_s$$

## Key Experimental Results

### Main Results on the VG Dataset (Average@K, three backbones)

| Backbone | PredCls A@50/100 | SGCls A@50/100 | SGDet A@50/100 |
|---|---|---|---|
| Motif+SBG | **43.8 / 45.9** | **26.2 / 27.1** | **20.4 / 23.7** |
| VCtree+SBG | **44.0 / 45.9** | **31.3 / 32.5** | 19.4 / 22.4 |
| Transformer+SBG | **44.6 / 46.7** | **27.1 / 28.0** | 20.1 / 23.3 |

Optimal or near-optimal A@K is achieved across all three backbones, demonstrating high performance consistency—whereas other methods often show competitiveness on only a single backbone.

### Detailed Comparison of Sample-Level vs. Dataset-Level Bias Correction (Motif Model, VG, R@100/mR@100/A@100)

| Method | R@100 Change | mR@100 Change | A@100 Change |
|---|---|---|---|
| DLFE | −12.9 | +9.4 | −1.8 |
| RTPB | −24.6 | +18.1 | −3.3 |
| **SBG** | **−9.9** | **+15.1** | **+2.6** |

Key Findings: Although DLFE and RTPB improve mR@K, they suffer from severe drops in R@K, leading to a decrease in A@K. SBG is the only method that achieves a positive gain in A@K, realizing the best balance between head and tail classes. The average gains across the three tasks (PredCls/SGCls/SGDet) are **5.6%/3.9%/3.2%** respectively.

### Efficiency Comparison (Motif PredCls)

| Metric | vs. DLFE | vs. RTPB |
|---|---|---|
| A@100 Gain | +10.6% | +14.5% |
| Training Time | −45.0% (12.6h vs 22.9h) | +4.1% |
| Inference Speed Increase | +1.2% | +2.2% |
| Parameter Count Increase | +0.5% | +0.3% |

### Cross-Dataset Generalization (Motif PredCls)

| Dataset | A@50 Gain vs. CFA | A@100 Gain vs. CFA | Characteristics |
|---|---|---|---|
| GQA | +0.2% | +0.1% | Weak long-tailed effect |
| VG | +2.7% | +1.8% | Moderate long-tailed effect |
| VG-1800 | F-Acc Top-10 +10.48% | - | 1800 relation categories, severe long-tailed effect |

The larger the dataset and the more severe the long-tailed effect, the more prominent SBG's advantages become.

### Ablation Study (Transformer PredCls)

Union region vs. Entire Image: Using union region features yields an A@50/100 of 44.6/46.7, outperforming the entire image features (40.3/42.0). This indicates that the complete image introduces noise interference.

Role of Global Bias $\mathbf{b}^{glo}$: Relying solely on $f_{uni}$ results in an A@50/100 of 42.6/44.8, which increases to 44.6/46.7 when incorporating $\mathbf{b}^{glo}$. However, using $\mathbf{b}^{glo}$ alone (without $f_{uni}$) yields poor results (42.1/43.5), confirming that the sample-specific union feature is the core component.

### Scalability Verification

**One-Stage SGG Methods** (VG SGDet): A@50/100 improves from 25.2/29.5 to 26.7/30.9 on ISG; from 18.3/21.8 to 20.2/23.6 on SGTR; and from 21.1/24.4 to 23.7/27.3 on SS R-CNN.

**Object Detection** (COCO): Faster R-CNN + SBP improves mAP from 36.4 to 37.6 (+1.2%), and tail category mAP from 41.6 to 44.4 (+2.8%).

## Highlights & Insights

- **Fundamental upgrade in bias correction granularity**: This work is the first to introduce sample-level bias correction to SGG. It transitions from "applying a single bias to all samples" to "predicting an exclusive bias for each sample", demonstrating a clear conceptual evolution.
- **Using GAN for bias prediction rather than data generation**: Drawing on the robust fitting capacity of adversarial training to approximate the true bias correction vectors, it acts as a much stronger alternative than fully connected networks.
- **Optimal balance between R@K and mR@K**: SBP is the only bias correction method that achieves a positive increase in Average@K, successfully avoiding the common pitfall of sacrificing head category performance to boost tail category performance.
- **Plug-and-play general framework**: Dynamically adapts to three mainstream two-stage backbones (Motif/VCtree/Transformer), three one-stage methods, and even object detection tasks.
- **Extremely low inference overhead**: The parameter count increases by only 0.3-0.5%, and the inference speed decreases slightly by 1.2-2.2%, proving highly feasible for actual deployment.

## Limitations & Future Work

- GAN training is inherently unstable and requires meticulous parameter tuning (a 5:1 updates ratio between D and G, high sensitivity to learning rate), which elevates tuning costs.
- The construction workflow of the bias correction set $\mathcal{S}$ is relatively complex, involving conditional judgments and iterative modifications, lacking simplicity.
- It remains a two-stage training scheme (first training the SGG model and freezing its parameters, then training BGAN), missing end-to-end joint optimization.
- Performance improvement on datasets with weak long-tailed distributions (e.g., GQA) is limited (A@K of only +0.1-0.2%).
- The global bias $\mathbf{b}^{glo}$ still relies on statistical priors of the dataset, which does not entirely break free from dataset-level information.
- The potential of utilizing Transformer-based generators or diffusion models to replace GAN has not yet been explored.

## Related Work & Insights

| Method | Type | Core Idea | Limitations |
|---|---|---|---|
| DLFE | Dataset-level bias correction | Estimate class frequency, divide biased probability to recover unbiased state | All samples share the same $\mathbf{c}$, severe drop in R@K |
| RTPB | Dataset-level bias correction | Learn resistance bias, subtract from classification logits | All samples share the same $\mathbf{b}$, A@K decreases instead |
| CFA | Data augmentation | Triplet feature combination to enhance tail diversity | No direct bias correction involved, offers only indirect relief |
| HML | Hierarchical learning | Level-by-level progressive relationship learning | High computational overhead, weak generalization |
| GCL | Grouping learning | Cooperative grouping to increase tail attention | Massive sacrifice in R@K |
| IETrans | Data augmentation | Relation transition to enrich tail training samples | Still operates at the dataset level |
| **SBG (Ours)** | **Sample-level bias correction** | **GAN predicts sample-specific bias vectors** | **Requires extra BGAN training** |

## Insights & Connections

- **Paradigm Shift in Bias Correction Granularity**: The evolutionary progression from category-level to dataset-level and finally to sample-level bias correction can be migrated to any task facing long-tailed issues (e.g., detection, segmentation, NLP classification).
- **GAN as a Correction Predictor**: While traditional GANs are used for image or text generation, using them here to generate correction signals expands the practical application scenarios of GANs.
- **Semantic Value of the Union Region**: The union region of object pairs not only encapsulates relation semantics but also supports bias prediction, inspiring future work to further explore the hidden potential of this area.
- **The Relaxation Principle of "Accurate but not necessarily perfect"**: Abandoning the pursuit of the ideal unbiased prediction $\mathbf{z}_u$ and instead aiming for a "correct prediction result" $\hat{\mathbf{z}}$ reduces the complexity of mathematical modeling without sacrificing practical utility.
- **High-Efficiency Pattern of Minimal Overhead for Substantial Gain**: Adding <3% computational overhead during inference while yielding a 10%+ boost in A@K constitutes a highly recommended post-processing strategy.

## Rating

- Novelty: ⭐⭐⭐⭐ (The sample-level bias correction approach is elegant and novel, and using GAN for bias prediction is highly creative)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Tested on 3 datasets × 3 backbones × 3 tasks + one-stage methods + object detection generalization + comprehensive ablation studies)
- Writing Quality: ⭐⭐⭐⭐ (The motivation progresses step-by-step, the mathematical derivation is complete, and the comparison with related works is clear and thorough)
- Value: ⭐⭐⭐⭐ (A plug-and-play general framework for long-tailed bias correction; low overhead and high yield with validated generalizability)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] One Sample is Enough to Make Conformal Prediction Robust](../../NeurIPS2025/optimization/one_sample_is_enough_to_make_conformal_prediction_robust.md)
- [\[NeurIPS 2025\] Learning Orthogonal Multi-Index Models: A Fine-Grained Information Exponent Analysis](../../NeurIPS2025/optimization/learning_orthogonal_multi-index_models_a_fine-grained_information_exponent_analy.md)
- [\[ICLR 2026\] Implicit Bias of Per-sample Adam on Separable Data: Departure from the Full-batch Regime](../../ICLR2026/optimization/implicit_bias_of_per-sample_adam_on_separable_data_departure_from_the_full-batch.md)
- [\[ICLR 2026\] Bilevel Optimization with Lower-Level Uniform Convexity: Theory and Algorithm](../../ICLR2026/optimization/bilevel_optimization_with_lower-level_uniform_convexity_theory_and_algorithm.md)
- [\[ICML 2025\] Nonparametric Teaching for Graph Property Learners](../../ICML2025/optimization/nonparametric_teaching_for_graph_property_learners.md)

</div>

<!-- RELATED:END -->

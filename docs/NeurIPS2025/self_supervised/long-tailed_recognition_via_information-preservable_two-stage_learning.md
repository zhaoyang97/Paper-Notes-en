---
title: >-
  [Paper Note] Long-Tailed Recognition via Information-Preservable Two-Stage Learning
description: >-
  [NeurIPS 2025][Self-Supervised Learning][Long-tailed recognition] This paper proposes an information-preservable two-stage learning framework: Stage 1 employs Balanced Negative Sampling (BNS) to learn an effective and se…
tags:
  - "NeurIPS 2025"
  - "Self-Supervised Learning"
  - "Long-tailed recognition"
  - "contrastive learning"
  - "information theory"
  - "determinantal point process"
  - "two-stage learning"
  - "representation learning"
date: 2026-05-08
content_hash: 2a6ec90450e4bc7a
---

# Long-Tailed Recognition via Information-Preservable Two-Stage Learning

**Conference**: NeurIPS 2025
**arXiv**: [2510.08836](https://arxiv.org/abs/2510.08836)  
**Code**: [github.com/fudong03/BNS_IPDPP](https://github.com/fudong03/BNS_IPDPP)  
**Authors**: Fudong Lin, Xu Yuan (University of Delaware)
**Area**: Self-Supervised Learning
**Keywords**: Long-tailed recognition, contrastive learning, information theory, determinantal point process, two-stage learning, representation learning

## TL;DR

This paper proposes an information-preservable two-stage learning framework: Stage 1 employs Balanced Negative Sampling (BNS) to learn an effective and separable feature space via mutual information maximization; Stage 2 uses Information-Preservable DPP (IP-DPP) to sample the most informative examples in a mathematically principled manner to correct majority-biased decision boundaries. The method achieves state-of-the-art performance on multiple long-tailed benchmarks.

## Background & Motivation

Long-tailed distributions are ubiquitous in real-world data. Head classes dominate the training process, causing decision boundaries to be biased toward majority classes and severely degrading tail-class performance. Existing methods fall into two categories:

**One-stage methods** (re-weighting, oversampling/undersampling): Limited by representation learning capacity, they struggle to strike a balance between head and tail classes.

**Two-stage methods** (decoupled representation learning and classification): Existing works suffer from two critical bottlenecks:
- **Stage 1**: Existing contrastive learning methods (KCL, TSC, SBCL) fail to learn a clearly separable feature space between head and tail classes.
- **Stage 2**: Oversampling is prone to mode collapse, while undersampling causes severe information loss.

This paper approaches both stages from an information-theoretic perspective and proposes novel solutions for each.

## Method

### Overall Architecture

A two-stage pipeline: Stage 1 trains the feature extractor with BNS → Stage 2 fine-tunes the classifier on a balanced subset sampled by IP-DPP.

### Stage 1: Balanced Negative Sampling (BNS)

**Core Idea**: Maximize the mutual information $MI(\boldsymbol{Q}, \boldsymbol{V})$ between two augmented views of the same data, and prove this is equivalent to minimizing intra-class distance.

- Given an image $\boldsymbol{x}$, two views $\boldsymbol{x}_i \in \mathbb{X}_Q$ and $\boldsymbol{x}_j \in \mathbb{X}_V$ are generated via data augmentation.
- Drawing on Noise Contrastive Estimation (NCE), positive and negative pairs are constructed for contrastive learning.
- The basic negative sampling loss $\mathcal{L}_{NS}$ alleviates "label bias" but fails to learn a well-separated space.

**Novelty of BNS**: For a given anchor image $\boldsymbol{x}_i$, $m$ additional same-class images $\{\boldsymbol{x}_k\}_{k=1}^m$ are sampled, forming $(m+1)$ positive pairs and $n(m+1)$ negative pairs:

$$\mathcal{L}_{BNS} = -\frac{1}{m+1}\left[\sum_{q_* \in \{q_i\} \cup \boldsymbol{Q}_{i,m}^+} \log\sigma\left(\frac{\boldsymbol{q}_*^\top \boldsymbol{v}_{j,i}^+}{\tau}\right) + \sum_{q_* \in \{q_i\} \cup \boldsymbol{Q}_{i,m}^+} \sum_{j=1}^n \log\sigma\left(-\frac{\boldsymbol{q}_*^\top \boldsymbol{v}_j^-}{\tau}\right)\right]$$

**Two-level semantic decomposition**:

- **Instance-level semantics**: Anchor $\boldsymbol{q}_i$ and its positive pair $\boldsymbol{v}_{j,i}^+$ come from the same instance, ensuring high-quality features.
- **Class-level semantics**: Additional same-class samples $\boldsymbol{q}_k$ and the positive pair $\boldsymbol{v}_{j,i}^+$ share the same class label, promoting inter-class separation.

**Theorem 4.1 (Intra-class Distance–Mutual Information Theorem)**: $\max MI(\boldsymbol{Q}^c, \boldsymbol{V}^c) \propto \min D(\boldsymbol{Q}^c, \boldsymbol{V}^c)$, i.e., maximizing mutual information is equivalent to minimizing intra-class distance.

### Stage 2: Information-Preservable DPP (IP-DPP)

**Objective**: Sample a balanced subset from majority classes to correct decision boundaries while preserving information content.

- Based on determinantal point processes (DPP), a symmetric random matrix $\boldsymbol{S}$ is constructed, where $\boldsymbol{S}_{i,j} = \frac{p(i)p(j)}{N}$ ($i \neq j$), and $p(i) = p_\phi(y_i|\boldsymbol{x}_i)$ is the probability of correct classification.
- It is proven that $\boldsymbol{S}$ is positive semi-definite with eigenvalues in $[0,1]$ (Lemmas 4.4 and 4.5), satisfying the DPP kernel matrix conditions.
- **Information preservation** (Remark 4.7): $\mathcal{P}_S(\mathbb{Y} \cup \{\boldsymbol{x}\}) \propto I(\boldsymbol{x})$, meaning samples with lower classification confidence (higher information content) are more likely to be selected.
- Standard DPP expects to sample $N(1-\ln 2)$ instances (roughly one-third), which is insufficient to balance class priors; thus, a fixed-cardinality $k$-DPP is introduced to sample a fixed $k$ instances from each majority class.
- An efficient sampling algorithm based on spectral decomposition is designed (Algorithm 1).

## Key Experimental Results

### Table 1: CIFAR-10/100-LT Results (IF=100)

| Method | CIFAR-10-LT Overall | CIFAR-100-LT Overall |
|--------|:---:|:---:|
| Focal Loss | 69.2 | 43.5 |
| LDAM Loss | 71.5 | 44.1 |
| RIDE | 73.4 | 47.2 |
| SBCL | 72.6 | 48.5 |
| OTmix | 73.8 | 48.1 |
| DisA | 73.6 | 49.2 |
| **Ours** | **76.4** | **52.4** |

- Overall accuracy of 76.4% on CIFAR-10-LT, surpassing the second-best by at least 2.6%.
- Overall accuracy of 52.4% on CIFAR-100-LT, surpassing DisA by 3.2%.
- Although many-shot accuracy is slightly below OTmix, medium-shot and few-shot accuracy are substantially higher (few-shot gains of 19.9% / 12.8%).

### Table 2: ImageNet-LT and iNaturalist 2018

| Method | ImageNet-LT Overall | iNaturalist 2018 Overall |
|--------|:---:|:---:|
| DisA | 49.4 | 69.8 |
| SBCL | 47.1 | 70.4 |
| **Ours** | **51.7** | **74.0** |

- Surpasses DisA by 2.3% on ImageNet-LT and SBCL by 3.6% on iNaturalist 2018.
- State-of-the-art is maintained on large-scale datasets, demonstrating strong generalization.

### Robustness Across Imbalance Factors (IF)

- As IF increases from 10 to 200, the proposed method drops only 10.2% on CIFAR-10-LT (83.7%→73.5%), compared to a 15.2% drop for DisA.
- Achieves the highest overall accuracy across all IF values (10/20/50/100/200).

### Representation Learning Evaluation

- Linear probing accuracy: 68.2% on CIFAR-10-LT, surpassing KCL/TSC/SBCL by 7.2%/4.4%/3.5%, respectively.
- The gap between many-shot and few-shot accuracy under BNS is only 1.8% (69.4% vs. 67.6%), compared to 43.5% for SBCL.
- t-SNE visualizations clearly show that BNS produces significantly better inter-class separation than SBCL.

## Highlights & Insights

1. **Rigorous theoretical grounding**: Starting from information theory, the paper formally proves that mutual information maximization is equivalent to intra-class distance minimization (Theorem 4.1), and provides theoretical guarantees for the information-preserving property of DPP sampling.
2. **Addresses both stage bottlenecks**: BNS resolves the feature inseparability problem through dual-level semantics (instance-level + class-level); IP-DPP addresses information loss in undersampling by prioritizing high-information samples.
3. **Substantial tail-class improvement**: Few-shot accuracy is dramatically improved without heavily sacrificing head-class performance (19.9% higher than OTmix on CIFAR-10-LT).
4. **Strong robustness**: Minimal performance degradation under extreme imbalance (IF=200), with state-of-the-art results on both small- and large-scale datasets.

## Limitations & Future Work

1. **Many-shot accuracy trade-off**: The method underperforms certain baselines on head classes, reflecting an inherent head-tail trade-off.
2. **Computational overhead**: IP-DPP requires spectral decomposition of the symmetric random matrix, and computational cost on large-scale datasets warrants attention.
3. **Hyperparameter sensitivity**: $m$ (number of same-class samples) in BNS and $k$ (fixed sampling size) in IP-DPP require tuning; in particular, $m$ is constrained by the number of available tail-class samples.
4. **Evaluation limited to classification**: The method has not been validated on downstream tasks such as detection or segmentation, and broader generalizability remains to be tested.
5. **Integration with foundation models**: The potential of combining the proposed framework with large pretrained models (e.g., ViT-L/CLIP) for long-tailed fine-tuning has not been explored.

## Related Work & Insights

| Category | Representative Methods | Advantages of Ours |
|----------|----------------------|-------------------|
| Re-weighting | Focal Loss, LDAM | Two-stage decoupling avoids over-reliance on loss re-weighting |
| Oversampling | SMOTE, OTmix | IP-DPP undersamples while preserving information, avoiding mode collapse |
| Undersampling | Random, Informed | IP-DPP prioritizes high-information samples based on information theory, reducing information loss |
| Contrastive learning | KCL, TSC, SBCL | BNS captures both instance-level and class-level semantics, yielding better feature space separation |
| Prev. SOTA | DisA (ICML'24) | Consistently superior overall accuracy, with especially notable tail-class gains |

## Rating

- Novelty: ⭐⭐⭐⭐ — The mutual information perspective in BNS and the information-preservable sampling in IP-DPP are both original, with thorough theoretical proofs.
- Experimental Thoroughness: ⭐⭐⭐⭐ — 4 datasets × 9 baselines, with ablations, varying IF settings, linear probing, and t-SNE visualizations.
- Writing Quality: ⭐⭐⭐⭐ — Mathematical derivations are clear, the structure is complete, and theory aligns well with experiments.
- Value: ⭐⭐⭐⭐ — Consistent state-of-the-art in long-tailed recognition, with tight integration of theory and practice.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] BCE3S: Binary Cross-Entropy Based Tripartite Synergistic Learning for Long-tailed Recognition](../../AAAI2026/self_supervised/bce3s_binary_cross-entropy_based_tripartite_synergistic_learning_for_long-tailed.md)
- [\[CVPR 2026\] Reframing Long-Tailed Learning via Loss Landscape Geometry](../../CVPR2026/self_supervised/reframing_long-tailed_learning_via_loss_landscape_geometry.md)
- [\[NeurIPS 2025\] Self-Supervised Contrastive Learning is Approximately Supervised Contrastive Learning](self-supervised_contrastive_learning_is_approximately_supervised_contrastive_lea.md)
- [\[NeurIPS 2025\] Know Thyself by Knowing Others: Learning Neuron Identity from Population Context](know_thyself_by_knowing_others_learning_neuron_identity_from_population_context.md)
- [\[ICLR 2026\] Maximizing Incremental Information Entropy for Contrastive Learning](../../ICLR2026/self_supervised/maximizing_incremental_information_entropy_for_contrastive_learning.md)

</div>

<!-- RELATED:END -->

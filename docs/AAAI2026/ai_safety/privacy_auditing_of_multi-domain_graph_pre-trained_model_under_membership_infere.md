---
title: >-
  [Paper Note] Privacy Auditing of Multi-Domain Graph Pre-Trained Model under Membership Inference Attack
description: >-
  [AI Safety] This paper proposes MGP-MIA, the first framework targeting membership inference attacks (MIA) against multi-domain graph pre-trained models. It amplifies membership signals via machine unlearning, constructs shadow models through incremental learning, and employs a similarity-based inference mechanism to effectively expose privacy leakage risks in multi-domain graph pre-training.
tags:
  - AI Safety
date: 2026-05-08
content_hash: 1e2da63dc120e3e0
---

# Privacy Auditing of Multi-Domain Graph Pre-Trained Model under Membership Inference Attack

- **Conference**: AAAI 2026
- **arXiv**: [2511.17989](https://arxiv.org/abs/2511.17989)
- **Code**: [RingBDStack/MGP-MIA](https://github.com/RingBDStack/MGP-MIA)
- **Area**: AI Security
- **Keywords**: Membership Inference Attack, Multi-Domain Graph Pre-Training, Privacy Auditing, Machine Unlearning, Graph Neural Networks

## TL;DR

This paper proposes MGP-MIA, the first framework targeting membership inference attacks (MIA) against multi-domain graph pre-trained models. It amplifies membership signals via machine unlearning, constructs shadow models through incremental learning, and employs a similarity-based inference mechanism to effectively expose privacy leakage risks in multi-domain graph pre-training.

## Background & Motivation

Multi-domain graph pre-training is a key technique for building graph foundation models. By conducting self-supervised pre-training (e.g., link prediction, contrastive learning) across graph data from multiple domains, GNNs acquire transferable structural and semantic representations. When developers publicly release pre-trained models to support downstream tasks, adversaries can exploit the model to infer whether specific samples were included in the training data, causing serious privacy leakage.

However, performing MIA against multi-domain graph pre-trained models faces three major challenges:

1. **Enhanced generalization**: Multi-domain pre-training reduces overfitting, which is precisely the core signal that traditional MIA relies on.
2. **Non-representative shadow datasets**: Training data spans multiple domains, making it difficult for adversaries to obtain shadow graphs aligned with all training domains.
3. **Weakened membership signals**: Pre-trained encoders output embedding vectors rather than logits, which carry weaker overfitting signals.

The authors validate two key observations through PCA visualization and perturbation stability experiments: (1) the separability between member and non-member embeddings is weak; (2) member embeddings are not more stable under perturbation than non-member embeddings. This demonstrates that existing graph MIA methods cannot be directly applied to multi-domain graph pre-training scenarios.

## Method

The MGP-MIA framework consists of three core modules: a membership signal amplification mechanism, an incremental shadow model construction mechanism, and a similarity-based inference mechanism.

### 1. Membership Signal Amplification

This module leverages machine unlearning to intensify the model's overfitting to the remaining data, thereby amplifying membership signals.

Specifically, a subgraph $\mathcal{G}_{\text{Unlearn}}$ is randomly sampled from the shadow graph $\mathcal{G}_{\text{Shadow}}$ as the unlearning target. The target model $\mathcal{F}_{\text{Target}}$ is first fine-tuned on $\mathcal{G}_{\text{Unlearn}}$ for several epochs to obtain an augmented model $\mathcal{F}_{\text{Augment}}$. The similarity differences between each node and its positive/negative samples under both models are then compared, yielding teacher similarity scores:

$$\mathbf{s}_{\text{Teacher}}^{i} = \mathbf{s}_{\text{Target}}^{i} - \lambda \cdot (\mathbf{s}_{\text{Target}}^{i} - \mathbf{s}_{\text{Augment}}^{i})$$

where $\lambda$ controls the unlearning intensity. The similarity vector is defined as the concatenation of cosine similarities between node $v_i$ and its $P$ positive samples and $N$ negative samples:

$$\mathbf{s}^{i} = [\text{sim}(\mathbf{h}_i, \mathbf{h}_{i_1^+}), \ldots, \text{sim}(\mathbf{h}_i, \mathbf{h}_{i_P^+}), \text{sim}(\mathbf{h}_i, \mathbf{h}_{i_1^-}), \ldots, \text{sim}(\mathbf{h}_i, \mathbf{h}_{i_N^-})]$$

Unlearning is completed by minimizing the deviation between the unlearned model's output and the teacher scores:

$$\min_{\mathcal{F}_{\text{Unlearn}}} \sum_{n_i \in \mathcal{V}_{\text{Unlearn}}} \|\mathbf{s}_{\text{Unlearn}}^{i} - \mathbf{s}_{\text{Teacher}}^{i}\|^2$$

**Core Idea**: Imprecise machine unlearning releases model capacity, causing stronger memorization (overfitting) on the remaining data, which amplifies behavioral differences between members and non-members.

### 2. Incremental Shadow Model Construction

Adversaries typically possess only a shadow graph from the same domain as the target node, which cannot cover all training domains of the target model. This module constructs a reliable shadow model from limited data via incremental learning.

The shadow graph is split into a training set $\mathcal{G}_{\text{Shadow}}^{\text{Train}}$ and a test set $\mathcal{G}_{\text{Shadow}}^{\text{Test}}$. The Fisher information matrix is estimated using the shadow data to quantify the importance of each parameter in the unlearned model:

$$\mathbf{I}_{\text{Unlearn}}(\theta) = \mathbb{E}_{v \sim \mathcal{G}_{\text{Shadow}}^{\text{Train}}} \left[\frac{\partial^2 \mathcal{L}_{\text{task}}(\mathcal{F}_{\text{Unlearn}}; v)}{\partial \theta^2}\bigg|\theta\right]$$

The unlearned model is then fine-tuned with parameter regularization to obtain the shadow model:

$$\min_{\mathbf{\Theta}_{\text{Shadow}}} \sum_{v \in \mathcal{G}_{\text{Shadow}}^{\text{Train}}} \mathcal{L}_{\text{task}}(\mathcal{F}_{\text{Shadow}}; v) + \alpha \sum_i \mathbf{I}_{\text{Unlearn}}^{(i)} (\mathbf{\Theta}_{\text{Shadow}}^{(i)} - \mathbf{\Theta}_{\text{Unlearn}}^{(i)})^2$$

where $\alpha$ controls the regularization strength. The Fisher information matrix constrains important parameters from deviating too far, enabling the shadow model to better replicate the membership inference characteristics of the target model.

### 3. Similarity-Based Inference

To extract membership signals from embeddings, this module exploits the inherent principle of self-supervised pre-training—pulling positive samples closer and pushing negative samples apart—to construct attack features. For each target node $v$, $m$ positive and $m$ negative samples are randomly selected, and the similarity vector $\mathbf{s}_v$ between the node and the shadow model's output embeddings is computed as the attack feature. A two-layer MLP attack model is then trained on these features with member/non-member labels.

## Key Experimental Results

### Experimental Setup

- **Datasets**: Cora, CiteSeer, PubMed (citation networks), Photo, Computers (Amazon co-purchase graphs)
- **Target Models**: MDGPT, BRIDGE (link prediction); GCOPE, SAMGPT (contrastive learning)
- **Baselines**: Embed-MIA, Grad-MIA, NLO-MIA, GLO-MIA, GE-MIA, GPIA
- **Metrics**: Accuracy (ACC), F1-score
- **Hardware**: Single NVIDIA V100 GPU, repeated 5 times

### Table 1: Attacking Link-Prediction-Based Multi-Domain Graph Pre-Trained Models (MDGPT)

| Method | Cora ACC | Cora F1 | CiteSeer ACC | PubMed ACC | Computers ACC |
|--------|----------|---------|-------------|------------|---------------|
| Embed-MIA | 68.89 | 60.31 | 66.53 | 60.60 | 61.54 |
| Grad-MIA | 51.51 | 22.03 | 50.76 | 49.21 | 55.15 |
| GPIA | 72.20 | 76.41 | 68.58 | 65.75 | 68.35 |
| **MGP-MIA** | **81.79** | **83.99** | **77.36** | **74.77** | **80.66** |

### Table 2: Attacking Contrastive-Learning-Based Multi-Domain Graph Pre-Trained Models (SAMGPT)

| Method | Cora ACC | Cora F1 | CiteSeer ACC | PubMed ACC | Computers ACC |
|--------|----------|---------|-------------|------------|---------------|
| Grad-MIA | 61.82 | 60.71 | 52.19 | 50.03 | 54.70 |
| GE-MIA | 73.32 | 74.99 | 73.97 | 55.21 | 55.77 |
| GPIA | 58.55 | 59.11 | 55.31 | 54.59 | 73.33 |
| **MGP-MIA** | **99.91** | **99.88** | **98.83** | **91.30** | **91.72** |

MGP-MIA performs particularly well on SAMGPT, achieving an ACC of 99.91% on Cora—approximately 26.6 percentage points higher than the strongest baseline, GE-MIA.

## Key Findings

1. **Multi-domain pre-training is not privacy-safe**: Despite enhanced generalization from multi-domain pre-training, MGP-MIA can identify member nodes with high accuracy, revealing serious privacy risks in such models.
2. **Contrastive learning models are more vulnerable**: Attack performance on SAMGPT (contrastive learning) significantly surpasses that on MDGPT (link prediction), as contrastive learning explicitly encodes positive/negative sample relationships, providing stronger signals for similarity-based inference.
3. **Ablation study**: Both the machine unlearning module (UL) and the incremental learning module (IL) contribute to performance, with IL providing the primary intrinsic gain and UL further amplifying membership signals.
4. **Hyperparameter robustness**: The framework is insensitive to the regularization strength $\alpha$, maintaining stable performance across a wide range.

## Highlights & Insights

- **Pioneer work**: The first MIA study targeting multi-domain graph pre-trained models, filling a gap in this research direction.
- **Adversarial repurposing of machine unlearning**: A creative inversion of a privacy-preserving tool (machine unlearning) to enhance attack effectiveness.
- **Incremental learning for shadow model construction**: The Fisher information matrix is cleverly employed for parameter regularization, enabling high-quality shadow model construction from limited data.
- **Pre-training-paradigm-aware attack features**: Similarity features are designed by exploiting the inherent pull/push mechanism of self-supervised learning, proving more effective than directly using embeddings or gradients.
- **Comprehensive experiments**: Covers four target models (two pre-training paradigms), five datasets, and six baselines.

## Limitations & Future Work

1. **Strong white-box assumption**: The adversary requires full access to the target model's architecture and parameters, which may not hold in some real-world scenarios.
2. **Node-level MIA only**: Edge-level and graph-level membership inference are not addressed, limiting the attack granularity.
3. **Reliance on same-domain shadow data**: Adversaries still need to obtain shadow graphs from the same domain as the target node, incurring non-negligible data acquisition costs.
4. **Absence of defense discussion**: The paper focuses primarily on attack effectiveness without in-depth discussion of countermeasures against such attacks.
5. **Scalability unverified**: The datasets used in experiments are relatively small (Cora has only 2,708 nodes); applicability to large-scale graphs remains to be validated.

## Related Work & Insights

- **Multi-domain graph pre-training**: GCOPE (connecting domains via virtual nodes), SAMGPT (unified message aggregation with structural tokens), MDGPT (domain token-based semantic alignment), BRIDGE (domain aligners for extracting shared representations).
- **Graph membership inference attacks**: He et al. and Olatunji et al. first extended MIA to GNNs; ProIA introduced prompt-augmented attack model priors; GCL-Leak targeted federated contrastive learning; Conti et al. and Dai & Lu proposed label-only black-box MIA.
- **Machine unlearning**: Chen et al. 2022 proposed machine unlearning as a privacy protection strategy; Hayes et al. 2025 found that imprecise unlearning leads to stronger overfitting on remaining samples.

## Rating

⭐⭐⭐⭐ — The first MIA study targeting multi-domain graph pre-trained models, featuring a cleverly designed methodology (particularly the adversarial repurposing of machine unlearning), comprehensive experiments, and significant results. The white-box assumption and small-scale datasets are the primary limitations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Rethinking Target Label Conditioning in Adversarial Attacks: A 2D Tensor-Guided Generative Approach](rethinking_target_label_conditioning_in_adversarial_attacks_a_2d_tensor-guided_g.md)
- [\[AAAI 2026\] Robust Watermarking on Gradient Boosting Decision Trees](robust_watermarking_on_gradient_boosting_decision_trees.md)
- [\[AAAI 2026\] ProbLog4Fairness: A Neurosymbolic Approach to Modeling and Mitigating Bias](problog4fairness_a_neurosymbolic_approach_to_modeling_and_mitigating_bias.md)
- [\[AAAI 2026\] Learning to Collaborate: An Orchestrated-Decentralized Framework for Peer-to-Peer Collaborative Learning](learning_to_collaborate_an_orchestrated-decentralized_framework_for_peer-to-peer.md)
- [\[ICLR 2026\] Skirting Additive Error Barriers for Private Turnstile Streams](../../ICLR2026/ai_safety/skirting_additive_error_barriers_for_private_turnstile_streaming.md)

</div>

<!-- RELATED:END -->

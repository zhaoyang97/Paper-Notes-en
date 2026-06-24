---
title: >-
  [Paper Note] Privacy Auditing of Multi-Domain Graph Pre-Trained Model under Membership Inference Attack
description: >-
  [AI Safety] This paper proposes the MGP-MIA framework, conducting the first membership inference attack (MIA) against multi-domain graph pre-trained models. By amplifying membership signals via machine unlearning, constructing shadow models through incremental learning, and employing a similarity-based inference mechanism, MGP-MIA effectively reveals the privacy leakage risks of multi-domain graph pre-training.
tags:
  - "AI Safety"
date: 2026-05-08
content_hash: f42e4d5fba77e1c5
---

# Privacy Auditing of Multi-Domain Graph Pre-Trained Model under Membership Inference Attack

- **Conference**: AAAI 2026
- **arXiv**: [2511.17989](https://arxiv.org/abs/2511.17989)
- **Code**: [RingBDStack/MGP-MIA](https://github.com/RingBDStack/MGP-MIA)
- **Area**: AI Security
- **Keywords**: Membership Inference Attack, Multi-Domain Graph Pre-training, Privacy Auditing, Machine Unlearning, Graph Neural Networks

## TL;DR

This paper proposes the MGP-MIA framework, conducting the first membership inference attack (MIA) against multi-domain graph pre-trained models. By amplifying membership signals via machine unlearning, constructing shadow models through incremental learning, and employing a similarity-based inference mechanism, MGP-MIA effectively reveals the privacy leakage risks of multi-domain graph pre-training.

## Background & Motivation

Multi-domain graph pre-training is a critical technique for building graph foundation models. By performing self-supervised pre-training (e.g., link prediction, contrastive learning) on graph data across multiple domains, GNNs acquire cross-domain transferable structural and semantic representations. When developers publicly release pre-trained models to support downstream tasks, adversaries can exploit the models to infer whether specific samples were included in the training data, leading to severe privacy leakage risks.

However, executing MIA against multi-domain graph pre-trained models faces three major challenges:

1. **Enhanced generalization ability**: Multi-domain pre-training reduces overfitting, which is the core signal trusted by traditional MIAs.
2. **Unrepresentative shadow datasets**: The training data spans multiple domains, making it difficult for adversaries to obtain shadow graphs aligned with all training domains.
3. **Weakened membership signals**: Pre-trained encoders output embedding vectors instead of logits, containing much weaker overfitting signals.

The authors validate two key observations through PCA visualization and perturbation stability experiments: (1) the separability between member and non-member embeddings is weak; (2) member embeddings are not more stable under perturbations than non-member ones. This indicates that existing graph MIA methods cannot be directly applied to multi-domain graph pre-training scenarios.

## Method

The MGP-MIA framework consists of three core modules: the membership signal amplification mechanism, the incremental shadow model construction mechanism, and the similarity-based inference mechanism.

### 1. Membership Signal Amplification Mechanism

This module utilizes machine unlearning to enhance the model's overfitting on the remaining data, thereby amplifying the membership signals.

Specific process: Randomly sample a subgraph $\mathcal{G}_{\text{Unlearn}}$ from the shadow graph $\mathcal{G}_{\text{Shadow}}$ as the unlearning target. First, fine-tune the target model $\mathcal{F}_{\text{Target}}$ on $\mathcal{G}_{\text{Unlearn}}$ for a few epochs to obtain an augmented model $\mathcal{F}_{\text{Augment}}$. Then, compare the similarity differences of each node with its positive/negative samples under both models to compute the teacher similarity score:

$$\mathbf{s}_{\text{Teacher}}^{i} = \mathbf{s}_{\text{Target}}^{i} - \lambda \cdot (\mathbf{s}_{\text{Target}}^{i} - \mathbf{s}_{\text{Augment}}^{i})$$

where $\lambda$ controls the unlearning intensity. The similarity vector is defined as the concatenation of the cosine similarities of node $v_i$ with its $P$ positive samples and $N$ negative samples:

$$\mathbf{s}^{i} = [\text{sim}(\mathbf{h}_i, \mathbf{h}_{i_1^+}), \ldots, \text{sim}(\mathbf{h}_i, \mathbf{h}_{i_P^+}), \text{sim}(\mathbf{h}_i, \mathbf{h}_{i_1^-}), \ldots, \text{sim}(\mathbf{h}_i, \mathbf{h}_{i_N^-})]$$

Finally, unlearning is accomplished by minimizing the deviation of the unlearned model's output from the teacher score:

$$\min_{\mathcal{F}_{\text{Unlearn}}} \sum_{n_i \in \mathcal{V}_{\text{Unlearn}}} \|\mathbf{s}_{\text{Unlearn}}^{i} - \mathbf{s}_{\text{Teacher}}^{i}\|^2$$

**Core Idea**: Imprecise machine unlearning releases model capacity, causing the model to develop stronger memory (overfitting) regarding the remaining data, thereby amplifying the behavioral differences between members and non-members.

### 2. Incremental Shadow Model Construction Mechanism

An adversary typically only possesses a shadow graph from the same domain as the target node, which fails to cover all training domains of the target model. This module constructs a reliable shadow model on limited data via incremental learning.

Divide the shadow graph into a training set $\mathcal{G}_{\text{Shadow}}^{\text{Train}}$ and a test-set $\mathcal{G}_{\text{Shadow}}^{\text{Test}}$. Estimate the Fisher information matrix using the shadow data to quantify the importance of each parameter in the unlearned model:

$$\mathbf{I}_{\text{Unlearn}}(\theta) = \mathbb{E}_{v \sim \mathcal{G}_{\text{Shadow}}^{\text{Train}}} \left[\frac{\partial^2 \mathcal{L}_{\text{task}}(\mathcal{F}_{\text{Unlearn}}; v)}{\partial \theta^2}\bigg|\theta\right]$$

Then, fine-tune the unlearned model with parameter regularization to obtain the shadow model:

$$\min_{\mathbf{\Theta}_{\text{Shadow}}} \sum_{v \in \mathcal{G}_{\text{Shadow}}^{\text{Train}}} \mathcal{L}_{\text{task}}(\mathcal{F}_{\text{Shadow}}; v) + \alpha \sum_i \mathbf{I}_{\text{Unlearn}}^{(i)} (\mathbf{\Theta}_{\text{Shadow}}^{(i)} - \mathbf{\Theta}_{\text{Unlearn}}^{(i)})^2$$

where $\alpha$ controls the regularization strength. The Fisher information matrix constrains important parameters from shifting too far, enabling the shadow model to better replicate the membership inference behavior of the target model.

### 3. Similarity-Based Inference Mechanism

To extract membership signals from embeddings, this mechanism exploits the self-supervised pre-training principle of "pulling positive samples closer and pushing negative samples further" to construct attack features. For each target node $v$, randomly select $m$ positive samples and $m$ negative samples, and calculate the similarity vector $\mathbf{s}_v$ with the shadow model's output embeddings as the attack feature. After labeling with member/non-member ground truth, train a two-layer MLP attack model.

## Key Experimental Results

### Experimental Setup

- **Datasets**: Cora, CiteSeer, PubMed (citation networks), Photo, Computers (Amazon co-purchase graphs)
- **Target Models**: MDGPT, BRIDGE (link prediction), GCOPE, SAMGPT (contrastive learning)
- **Baselines**: Embed-MIA, Grad-MIA, NLO-MIA, GLO-MIA, GE-MIA, GPIA
- **Metrics**: Accuracy (ACC), F1-score
- **Hardware**: Single NVIDIA V100 GPU, repeated 5 times

### Table 1: Attacking Link Prediction Multi-Domain Graph Pre-Trained Model (MDGPT)

| Method | Cora ACC | Cora F1 | CiteSeer ACC | PubMed ACC | Computers ACC |
|------|----------|---------|-------------|------------|---------------|
| Embed-MIA | 68.89 | 60.31 | 66.53 | 60.60 | 61.54 |
| Grad-MIA | 51.51 | 22.03 | 50.76 | 49.21 | 55.15 |
| GPIA | 72.20 | 76.41 | 68.58 | 65.75 | 68.35 |
| **MGP-MIA** | **81.79** | **83.99** | **77.36** | **74.77** | **80.66** |

### Table 2: Attacking Contrastive Learning Multi-Domain Graph Pre-Trained Model (SAMGPT)

| Method | Cora ACC | Cora F1 | CiteSeer ACC | PubMed ACC | Computers ACC |
|------|----------|---------|-------------|------------|---------------|
| Grad-MIA | 61.82 | 60.71 | 52.19 | 50.03 | 54.70 |
| GE-MIA | 73.32 | 74.99 | 73.97 | 55.21 | 55.77 |
| GPIA | 58.55 | 59.11 | 55.31 | 54.59 | 73.33 |
| **MGP-MIA** | **99.91** | **99.88** | **98.83** | **91.30** | **91.72** |

The performance of MGP-MIA is particularly striking on SAMGPT, achieving 99.91% ACC on Cora, which is an improvement of approximately 26.6 percentage points over the strongest baseline, GE-MIA.

## Key Findings

1. **Multi-domain pre-training is not secure**: Despite the enhanced generalization ability brought by multi-domain pre-training, MGP-MIA can still accurately identify member nodes, revealing severe privacy risks in these types of models.
2. **Contrastive learning models are more vulnerable**: The attack performance on SAMGPT (contrastive learning) is far superior to that on MDGPT (link prediction) because contrastive learning explicitly encodes positive/negative sample relationships, providing a stronger signal for similarity-based inference.
3. **Ablation Study**: Both the Machine Unlearning (UL) module and the Incremental Learning (IL) module contribute to the performance. IL provides the primary intrinsic gain, whereas UL further amplifies the membership signals.
4. **Hyperparameter robustness**: The framework is robust to the regularization strength $\alpha$, maintaining stable performance over a wide range.

## Highlights & Insights

- **Pioneering Nature**: The first work to study membership inference attacks on multi-domain graph pre-trained models, filling a critical gap in this field.
- **Reverse Exploitation of Machine Unlearning**: Imaginatively repurposing a privacy-preserving tool (machine unlearning) to enhance attack effectiveness, which is a highly novel perspective.
- **Incremental Learning for Shadow Model Construction**: Ingeniously utilizing the Fisher information matrix for parameter regularization to construct high-quality shadow models under data scarcity.
- **Pre-training Paradigm-aligned Attack Features**: Designing similarity features based on the inherent self-supervised mechanisms (pulling positive and pushing negative samples), which is more effective than directly using embeddings or gradients.
- **Experimental Thoroughness**: Covers four target models across two pre-training paradigms, five datasets, and six baselines.

## Limitations & Future Work

1. **Strong white-box assumption**: The adversary needs complete access to the target model's architecture and parameters, which might not hold in some practical scenarios.
2. **Limited to node-level MIA**: The work does not explore edge-level or graph-level membership inference, resulting in a limited attack granularity.
3. **Reliance on same-domain shadow data**: The adversary still needs to obtain a shadow graph of the same domain as the target node, incurring non-negligible data acquisition costs.
4. **Lack of defense mechanisms**: The paper mainly focuses on attack effectiveness without deeply discussing how to defend against such attacks.
5. **Unverified scalability**: The datasets utilized in experiments are relatively small (e.g., Cora has only 2,708 nodes), leaving the scalability to large-scale graphs to be validated.

## Related Work & Insights

- **Multi-Domain Graph Pre-training**: GCOPE (virtual nodes to connect domains), SAMGPT (structural tokens for unified message aggregation), MDGPT (domain tokens aligning semantics), BRIDGE (domain aligners to extract shared representations).
- **Graph Membership Inference Attacks**: He et al. and Olatunji et al. first extended MIA to GNNs; ProIA introduced prompt enhancement for prior knowledge in attack models; GCL-Leak targeted federated contrastive learning scenarios; Conti et al. and Dai & Lu proposed label-only black-box MIAs.
- **Machine Unlearning**: Chen et al. 2022 proposed machine unlearning privacy protection strategies; Hayes et al. 2025 discovered that imprecise unlearning leads to stronger overfitting on the remaining samples.

## Rating

⭐⭐⭐⭐ — The first study on MIA targeting multi-domain graph pre-trained models. The method is cleverly designed (especially the reverse exploitation of machine unlearning), and the experiments are comprehensive and effective. The strong white-box assumption and small-scale datasets are the main limitations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Rethinking Target Label Conditioning in Adversarial Attacks: A 2D Tensor-Guided Generative Approach](rethinking_target_label_conditioning_in_adversarial_attacks_a_2d_tensor-guided_g.md)
- [\[AAAI 2026\] ProbLog4Fairness: A Neurosymbolic Approach to Modeling and Mitigating Bias](problog4fairness_a_neurosymbolic_approach_to_modeling_and_mitigating_bias.md)
- [\[AAAI 2026\] Robust Watermarking on Gradient Boosting Decision Trees](robust_watermarking_on_gradient_boosting_decision_trees.md)
- [\[AAAI 2026\] Learning to Collaborate: An Orchestrated-Decentralized Framework for Peer-to-Peer Collaborative Learning](learning_to_collaborate_an_orchestrated-decentralized_framework_for_peer-to-peer.md)
- [\[ICML 2025\] Retraining with Predicted Hard Labels Provably Increases Model Accuracy](../../ICML2025/ai_safety/retraining_with_predicted_hard_labels_provably_increases_model_accuracy.md)

</div>

<!-- RELATED:END -->

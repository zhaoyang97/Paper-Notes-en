---
title: >-
  [Paper Note] Dual Consolidation for Pre-Trained Model-Based Domain-Incremental Learning
description: >-
  [CVPR 2025][LLM Safety][Domain-Incremental Learning] This paper proposes the Duct method, which addresses exemplar-free domain-incremental learning on pre-trained models. Duct employs representation consolidation (accumulating task vectors to build a unified embedding space) and classifier consolidation (utilizing category semantic information via optimal transport to estimate the weights of classifiers for old domains). It outperforms state-of-the-art methods by 1% to 7% acr…
tags:
  - "CVPR 2025"
  - "LLM Safety"
  - "Domain-Incremental Learning"
  - "Model Merging"
  - "Classifier Calibration"
  - "Task Vectors"
  - "Pre-Trained Models"
date: 2026-05-08
content_hash: 932012e09d9cf3aa
---

# Dual Consolidation for Pre-Trained Model-Based Domain-Incremental Learning

**Conference**: CVPR 2025  
**arXiv**: [2410.00911](https://arxiv.org/abs/2410.00911)  
**Code**: [https://github.com/Estrella-fugaz/CVPR25-Duct](https://github.com/Estrella-fugaz/CVPR25-Duct)  
**Area**: LLM Evaluation  
**Keywords**: Domain-Incremental Learning, Model Merging, Classifier Calibration, Task Vectors, Pre-Trained Models

## TL;DR
This paper proposes the Duct method, which addresses exemplar-free domain-incremental learning on pre-trained models. Duct employs representation consolidation (accumulating task vectors to build a unified embedding space) and classifier consolidation (utilizing category semantic information via optimal transport to estimate the weights of classifiers for old domains). It outperforms state-of-the-art methods by 1% to 7% across four benchmark datasets.

## Background & Motivation

**Background**: Domain-Incremental Learning (DIL) requires a model to adapt to continuously changing data domains (e.g., various weather conditions or artistic styles) without altering the label space, while preventing catastrophic forgetting of learned domains. Existing pre-trained model (PTM)-based methods encode domain-specific knowledge by freezing the backbone and training domain-specific prompt pools (e.g., L2P, CODA-Prompt).

**Limitations of Prior Work**: Forgetting occurs at two distinct levels: (1) representation level, where sequentially updating prompts or the backbone biases features towards the most recently learned domain; and (2) classifier level, where classifiers mismatch the continuously shifting feature space. Although prompt-based methods freeze the backbone, the learned prompt pools can still be overwritten, and classifier bias remains uncorrected.

**Key Challenge**: DIL demands a unified embedding space suitable for all observed domains. However, streamingly arrived data prevents simultaneous access to training data from all domains to construct such a space.

**Goal**: To simultaneously address representation and classifier forgetting in PTM-based DIL without storing any historical exemplars.

**Key Insight**: Leveraging the concept of model merging, this work builds a unified representation space covering all domains by accumulating "task vectors" generated from independently fine-tuning each domain onto pre-trained weights. Then, it utilizes category semantic relationships to estimate old classifier weights in the new embedding space through optimal transport.

**Core Idea**: By consolidating the representation space via task vector accumulation and aligning old classifiers using semantics-driven optimal transport, this method counteracts forgetting in domain-incremental learning from both representation and classifier perspectives.

## Method

### Overall Architecture
Upon the arrival of each new domain: (1) A domain expert model is fine-tuned on the new domain data, initialized from the pre-trained model. (2) A task vector $\delta_{\phi_i} = \phi_i - \phi_0$ is extracted and accumulated with task similarity weights to obtain a unified backbone $\phi_i^m$. (3) The classifier for the new domain is retrained on top of this unified backbone. (4) Optimal transport is utilized to transfer category semantic relationships from the new classifier to estimate the old classifier. During inference, only a single merged backbone and a merged classifier are utilized.

### Key Designs

1. **Representation Consolidation**:

    - **Function**: Building a unified embedding space suitable for all observed domains.
    - **Mechanism**: Each domain is independently fine-tuned to yield a task vector $\delta_{\phi_k}$, and the unified backbone is computed as $\phi_i^m = \phi_0 + \alpha_\phi \sum_{k=1}^{i} \text{Sim}_{0,k} \cdot \delta_{\phi_k}$. The task similarity $\text{Sim}_{0,k}$ is measured by calculating the cosine similarity of class prototypes between the pre-trained model and the domain-expert model on the current data. Since task vectors are based on the same pre-trained weights and the domain discrepancy is large, their low similarity makes accumulation effective.
    - **Design Motivation**: Preventing forgetting caused by sequential updates—the task vector for each domain is computed independently of other domains, rendering the merging process non-destructive. This can be executed incrementally: $\phi_i^m = \phi_{i-1}^m + \alpha_\phi \text{Sim}_{0,i} \delta_{\phi_i}$, requiring the storage of only two backbones.

2. **Classifier Consolidation - New Classifier Retraining**:

    - **Function**: Aligning the classifier of the current domain with the consolidated embedding space.
    - **Mechanism**: The merged backbone $\phi_i^m$ is frozen, and the classifier $W_n$ is retrained on the current domain's training data. This represents a standard linear probing step.
    - **Design Motivation**: The consolidated embedding space differs from the space during fine-tuning, necessitating classifier realignment.

3. **Classifier Consolidation - Old Classifier Transport**:

    - **Function**: Estimating the weights of old-domain classifiers in the new embedding space without historical exemplars.
    - **Mechanism**: Leveraging the inter-class semantic relations encoded in the new-domain classifier to estimate the old-domain classifier. Specifically, a semantic transport matrix $S$ is constructed based on category embeddings, and the optimal transport (Sinkhorn algorithm) is employed to solve the optimal matching relationship between categories. This relationship is then applied to the new-domain classifier to obtain an estimate of the old-domain classifier: $\hat{W}_o = \mathcal{T}(W_n, S)$. The final old classifier is a weighted average of the historical classifier and this estimated value.
    - **Design Motivation**: The class weight for "lion" in different domains (e.g., clipart vs. real photos) shares semantic correlation; leveraging inter-class relationships learned from the new domain helps infer what the classifier for the old domain should look like.

### Loss & Training
During the fine-tuning stage, standard cross-entropy loss is used, and cross-entropy loss is also utilized for classifier retraining. A cosine classifier is employed. After each domain arrives, it is independently fine-tuned for 15 epochs using SGD with lr=0.001. During inference, only a single consolidated backbone is required, introducing no extra inference overhead.

## Key Experimental Results

### Main Results

| Dataset | Duct $\bar{\mathcal{A}}$ | Duct $\mathcal{A}_B$ | CODA-Prompt | S-iPrompt | Gain |
|--------|---------|---------|-------------|-----------|------|
| Office-Home | 86.27% | 86.91% | 85.07% | 80.51% | +1.8% |
| DomainNet | 67.16% | 67.01% | 59.99% | 60.46% | +7.0% |
| CORe50 | 91.95% | 94.47% | 91.57% | 83.38% | +2.9% |
| CDDB | 84.14% | 85.10% | 74.18% | 72.76% | +10.9% |

### Ablation Study

| Configuration | CDDB $\mathcal{A}_B$ | Description |
|------|---------------------|------|
| Baseline (frozen backbone, prototype classifier) | ~63% | No merging of any kind |
| + Representation Consolidation | ~79% | Merging backbones brings substantial improvements |
| + New Classifier Retraining | ~83% | Classifier alignment is effective |
| + Old Classifier Transport (Full Duct) | 85.10% | Recovers old-domain data knowledge by +2% |

### Key Findings
- Representation consolidation is the most critical component, contributing the largest performance gain (~16%), indicating that a unified embedding space is vital for DIL.
- Duct achieves the lowest forgetting measure (0.12), significantly lower than Fine-tuning and MEMO, validating that the dual-level consolidation effectively counteracts forgetting.
- The standard deviation across five task orders is small (e.g., ±0.15 on CORe50), indicating that the method is robust to domain arrival order.
- Standard inference requires only a single backbone without extra computational cost, unlike prompt-based methods that demand prompt selection.
- Even without storing any old exemplars, the proposed method outperforms experience replay methods that store exemplars (e.g., 85.1% vs. 63.2% on CDDB).

## Highlights & Insights
- **Innovative Application of Model Merging**: This work transfers model merging from multi-task learning to incremental learning, leveraging the low similarity of task vectors to ensure merging effectiveness. The approach is simple yet theoretically well-motivated.
- **Ingenious Design of Old Classifier Transport**: It leverages inter-class semantic relationships to estimate old classifiers via OT without requiring historical samples, bypassing the need for a replay buffer.
- **Efficient Incremental Storage**: It only requires storing the current consolidated backbone and one active backbone being trained. During inference, only one model is used, maintaining O(1) storage complexity.

## Limitations & Future Work
- The scaling factor $\alpha_\phi$ for task vector accumulation is globally fixed; different domains might require varying merging intensities.
- Old classifier transport relies on the quality of semantic information; it might be inaccurate if domain-specific semantic relationships differ extremely.
- The evaluation is only verified on ViT-B/16, lacking experiments on larger or smaller models.
- The task-vector approach assumes that all domains start from the same pre-trained weights, making it inapplicable to scenarios without a shared initialization.

## Related Work & Insights
- **vs L2P/CODA-Prompt**: These methods encode domain knowledge with a prompt pool, but the prompts themselves can be overwritten. Duct directly merges domain information in the weight space, which is more stable.
- **vs S-iPrompt**: S-iPrompt uses KNN to retrieve domain-specific prompts, relying on accurate domain identification; Duct directly performs classification in a unified space without requiring a domain identification step.
- **vs Model Merging (Task Arithmetic)**: Duct extends model merging, originally used for static multi-task learning, to incremental scenarios by adding task-similarity weighting and classifier transport, marking the first systematic application of model merging in CL.

## Rating
- Novelty: ⭐⭐⭐⭐ The dual-level consolidation framework is elegantly designed, presenting a novel combination of model merging and OT-based classifier transport.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated across four datasets, five task sequences, multiple baselines, detailed ablations, and forgetting metrics.
- Writing Quality: ⭐⭐⭐⭐ The problem motivation is thoroughly analyzed and the methodological derivation is clear.
- Value: ⭐⭐⭐⭐ Highly advances PTM-based incremental learning, outperforming replay-based methods without requiring exemplar storage.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Low-Rank Adaptation in Multilinear Operator Networks for Security-Preserving Incremental Learning](low-rank_adaptation_in_multilinear_operator_networks_for_security-preserving_inc.md)
- [\[AAAI 2026\] LAMP: Learning Universal Adversarial Perturbations for Multi-Image Tasks via Pre-trained Models](../../AAAI2026/llm_safety/lamp_learning_universal_adversarial_perturbations_for_multi-image_tasks_via_pre-.md)
- [\[ACL 2025\] Exploring Forgetting in Large Language Model Pre-Training](../../ACL2025/llm_safety/exploring_forgetting_in_large_language_model_pre-training.md)
- [\[ICLR 2026\] Lifelong Learning with Behavior Consolidation for Vehicle Routing](../../ICLR2026/llm_safety/lifelong_learning_with_behavior_consolidation_for_vehicle_routing.md)
- [\[CVPR 2026\] Elastic Weight Consolidation Done Right for Continual Learning](../../CVPR2026/llm_safety/elastic_weight_consolidation_done_right_for_continual_learning.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] PromptCCD: Learning Gaussian Mixture Prompt Pool for Continual Category Discovery
description: >-
  [ECCV 2024][Self-Supervised Learning][Continual Category Discovery] This paper proposes the PromptCCD framework, which utilizes a Gaussian Mixture Model (GMM) as a prompt pool to achieve continual discovery of novel categories in unlabeled data streams while mitigating catastrophic forgetting.
tags:
  - "ECCV 2024"
  - "Self-Supervised Learning"
  - "Continual Category Discovery"
  - "Gaussian Mixture Models"
  - "Prompt Learning"
  - "Catastrophic Forgetting"
  - "Unlabeled Categories"
date: 2026-05-08
content_hash: 51420eb37a06595c
---

# PromptCCD: Learning Gaussian Mixture Prompt Pool for Continual Category Discovery

**Conference**: ECCV 2024  
**arXiv**: [2407.19001](https://arxiv.org/abs/2407.19001)  
**Code**: Yes (Project Page)  
**Area**: Self-supervised Learning  
**Keywords**: Continual Category Discovery, Gaussian Mixture Models, Prompt Learning, Catastrophic Forgetting, Unlabeled Categories

## TL;DR

This paper proposes the PromptCCD framework, which utilizes a Gaussian Mixture Model (GMM) as a prompt pool to achieve continual discovery of novel categories in unlabeled data streams while mitigating catastrophic forgetting.

## Background & Motivation

**Continual Category Discovery (CCD)** is an emerging and highly challenging problem: automatically discovering new categories in a continuously arriving stream of unlabeled data while not forgetting previously discovered categories. This problem combines the challenges of two notoriously difficult tasks: (1) **Generalized Category Discovery (GCD)**—discovering novel categories from partially labeled data; and (2) **Continual Learning**—learning new knowledge without forgetting old knowledge.

Key Challenge: In the CCD setting, data continues to arrive as a stream, where each phase's data may contain samples of both known and unknown new categories and is completely unlabeled. The model must simultaneously accomplish two difficult tasks: discovering the structure of new categories and maintaining memory of old categories. Even in fully-supervised continual learning, catastrophic forgetting remains an unsolved issue, and it becomes significantly more severe in completely unsupervised CCD.

Additionally, an extra challenge arises: the model needs to automatically estimate the number of new categories in each phase without knowing the ground truth. This requires the method to not only cluster but also automatically determine the cluster count.

This paper proposes PromptCCD, utilizing a Gaussian Mixture Prompt pool (GMP) as the core mechanism to concurrently address three challenges: category discovery, forgetting prevention, and category number estimation.

## Method

### Overall Architecture

PromptCCD is based on a pre-trained ViT feature extractor and a learnable prompt pool. At each phase: (1) the current prompt pool is used to enhance the features of new data; (2) GMM is utilized to cluster and discover new categories; (3) the prompt pool is updated to accommodate newly discovered categories; (4) catastrophic forgetting is prevented through the structured memory mechanism of GMP.

### Key Designs

1. **Gaussian Mixture Prompting (GMP) Module**:
    - Function: Acts as a dynamic prompt pool that updates over time to support category discovery and forgetting prevention.
    - Mechanism: Models the prompt pool as a Gaussian Mixture Model, where each Gaussian component corresponds to a discovered category. Each component stores a mean (class center), covariance (class distribution), and an associated prompt vector. When new samples arrive, the most matching prompt is selected for feature enhancement based on its posterior probability with respect to each component.
    - Design Motivation: Traditional prompt pools (such as L2P and DualPrompt) use discrete key-value matching, which cannot naturally scale to new categories. GMM provides a probabilistic framework that can naturally append new components (new categories) while maintaining memory of old components.

2. **Online Category Number Estimation**:
    - Function: Automatically estimates the number of new categories in each phase without knowing the ground-truth category count.
    - Mechanism: Utilizes GMM's BIC (Bayesian Information Criterion) or silhouette coefficient to automatically determine the optimal number of Gaussian components. In each new phase, a GMM is first fitted to the current data with different numbers of components, and the component number that minimizes BIC is chosen as the category count estimate.
    - Design Motivation: In practical applications, the number of categories is typically unknown; this design brings the method closer to real-world scenarios.

3. **Progressive Prompt Pool Updating Strategy**:
    - Function: Expands the prompt pool upon discovering new categories while protecting the knowledge of old categories.
    - Mechanism: When new categories are discovered, new Gaussian components and corresponding prompt vectors are added to the GMM. The parameters of old components drift slowly via EMA updates to adapt to data distribution shifts, but their core structure remains intact. The prompt selection mechanism ensures that samples belonging to old categories always match the correct old prompts.
    - Design Motivation: The progressive expansion of the prompt pool prevents overwriting old knowledge, while EMA updates allow for necessary parameter fine-tuning.

### Loss & Training

- Contrastive Loss: Utilizes self-supervised contrastive learning to learn compact category representations in the feature space.
- GMM Likelihood Loss: Maximizes the log-likelihood of the data under the current GMM model.
- Regularization Loss: Constrains the magnitude of prompt updates to prevent forgetting.
- Training Strategy: Each phase consists of two steps—first performing feature extraction and clustering, and then updating the prompt pool and GMM parameters.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours | Prev. SOTA | Gain |
|--------|------|------|----------|------|
| CIFAR-100 (CCD) | All Acc | Significant lead | GCD + Continual Learning baselines | +5-10% |
| ImageNet-100 (CCD) | All Acc | SOTA | Multiple baselines | +3-8% |
| CUB-200 (CCD) | New Acc | Substantial lead | GM methods | +8-15% |
| Category Number Estimation | Estimation Error ↓ | Lower | K-means + Gap Statistic | More accurate |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| w/o GMP (Fixed Prompts) | Severe forgetting | Significant performance drop on old categories |
| $GMM \to K-means$ | Performance drop | K-means cannot provide category number estimation |
| w/o Category Number Estimation (Fixed K) | Correlated with ground-truth K | Performance is sensitive to the choice of K |
| Full PromptCCD | Optimal | Mutual cooperation among the three components |

### Key Findings

- The GMM prompt pool significantly outperforms traditional key-value prompt pools on CCD tasks.
- Automatic category number estimation frees the method from relying on prior knowledge.
- The improvement in "novel category discovery" metrics is particularly significant, showing that GMP effectively facilitates the discovery of new categories.
- The method also performs well in mitigating the forgetting of old categories, validating the memory-retention capability of the prompt pool.

## Highlights & Insights

- Elegantly combines GMM with prompt learning, where each Gaussian component naturally corresponds to a category.
- Simultaneously addresses three challenges: category discovery, forgetting prevention, and category quantity estimation.
- The probabilistic framework of GMM is more flexible than discrete prompt matching.
- Normalizes and extends evaluation metrics from GCD to CCD, providing a solid benchmark for subsequent work.

## Limitations & Future Work

- GMM assumes that the category distribution is approximately Gaussian, which may be insufficient for complex multimodal distributions.
- The prompt pool grows linearly with the number of categories, which may present memory constraints in long-term operations.
- The BIC-based method for choosing the category count may lack stability in few-shot scenarios.
- Hierarchical GMMs could be explored to model hierarchical relationships between classes.
- Cross-modal extension (such as text-assisted visual category discovery) is a valuable direction for future research.

## Related Work & Insights

- **GCD**: Generalized Category Discovery introduced by Vaze et al., which serves as a single-stage version of CCD.
- **L2P / DualPrompt**: Applications of prompt learning in continual learning, which do not support unsupervised category discovery.
- **DCCL**: Continual contrastive learning methods, which do not employ a prompt pool.
- Insight: The application of probabilistic models (e.g., GMM) in continual learning is highly worthy of further investigation.

## Rating

- Novelty: ⭐⭐⭐⭐ The design of the GMM prompt pool is clever, and the CCD problem formulation is highly valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated across multiple datasets, including experiments on category number estimation.
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition with a systematic methodology description.
- Value: ⭐⭐⭐⭐ CCD is a highly practical and novel research direction; PromptCCD provides a strong baseline.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Is Parameter Isolation Better for Prompt-Based Continual Learning?](../../CVPR2026/self_supervised/is_parameter_isolation_better_for_prompt-based_continual_learning.md)
- [\[CVPR 2026\] Decouple Your Discovery and Memory in Continual Generalized Category Discovery](../../CVPR2026/self_supervised/decouple_your_discovery_and_memory_in_continual_generalized_category_discovery.md)
- [\[ICLR 2026\] Adaptive Gaussian Expansion for On-the-fly Category Discovery](../../ICLR2026/self_supervised/adaptive_gaussian_expansion_for_on-the-fly_category_discovery.md)
- [\[ICLR 2026\] PRISM: Progressive Robust Learning for Open-World Continual Category Discovery](../../ICLR2026/self_supervised/prism_progressive_robust_learning_for_open-world_continual_category_discovery.md)
- [\[CVPR 2026\] Spectral Mixture-of-Experts for Continual Learning](../../CVPR2026/self_supervised/spectral_mixture-of-experts_for_continual_learning.md)

</div>

<!-- RELATED:END -->

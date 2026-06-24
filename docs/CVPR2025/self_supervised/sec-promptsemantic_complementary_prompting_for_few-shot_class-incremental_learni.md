---
title: >-
  [Paper Note] SEC-Prompt: SEmantic Complementary Prompting for Few-Shot Class-Incremental Learning
description: >-
  [CVPR 2025][Self-Supervised Learning][Few-Shot Incremental Learning] The SEC-Prompt (SEmantic Complementary Prompting) framework is proposed to learn two sets of semantically complementary prompts—discriminative prompts (D-Prompt) and non-discriminative prompts (ND-Prompt). Working cooperatively through an adaptive query mechanism to reinforce inter-class discrimination and facilitate generalization to new classes respectively, they achieve SOTA performance on three benchmark…
tags:
  - "CVPR 2025"
  - "Self-Supervised Learning"
  - "Few-Shot Incremental Learning"
  - "Semantic Complementary Prompt"
  - "Discriminative Features"
  - "Data Augmentation"
  - "Prompt Clustering Loss"
date: 2026-05-08
content_hash: 61f32db257fa0a51
---

# SEC-Prompt: SEmantic Complementary Prompting for Few-Shot Class-Incremental Learning

**Conference**: CVPR 2025  
**Code**: None  
**Area**: Few-Shot Class-Incremental Learning  
**Keywords**: Few-Shot Incremental Learning, Semantic Complementary Prompt, Discriminative Features, Data Augmentation, Prompt Clustering Loss

## TL;DR

The SEC-Prompt (SEmantic Complementary Prompting) framework is proposed to learn two sets of semantically complementary prompts—discriminative prompts (D-Prompt) and non-discriminative prompts (ND-Prompt). Working cooperatively through an adaptive query mechanism to reinforce inter-class discrimination and facilitate generalization to new classes respectively, they achieve SOTA performance on three benchmark datasets.

## Background & Motivation

**Background**: Few-Shot Class-Incremental Learning (FSCIL) is a critical challenge in machine learning, requiring models to learn new classes from a small number of samples while maintaining performance on previously learned classes. Recently, prompt-based methods have demonstrated effectiveness in Class-Incremental Learning (CIL) by training learnable prompts to mitigate catastrophic forgetting.

**Limitations of Prior Work**:
- Existing prompt-based CIL methods require **sufficient data** to train prompts, whereas in FSCIL, each new class has only **extremely few samples** (e.g., 1-5), leading to a severe lack of training signals.
- Existing methods **do not consider the semantic features embedded in prompts**, leading to mixed knowledge learned by prompts, which traps the model in the plasticity-stability dilemma.
- There is a lack of explicit mechanisms to distinguish which information in the prompts contributes to class discrimination (discriminative) and which contributes to generalization to new classes (non-discriminative).

**Key Challenge**: FSCIL requires models to simultaneously possess plasticity (learning new classes) and stability (retaining old classes), but prompt training under few-shot conditions struggles to achieve both simultaneously.

**Goal**: To design a prompt learning method that can be efficiently learned under extremely few-shot conditions while balancing both plasticity and stability.

**Key Insight**: Disentangle prompts into two semantically complementary sets—discriminative prompts focusing on inter-class discrimination and non-discriminative prompts focusing on cross-class generalization, which work cooperatively.

**Core Idea**: Decompose the feature space into two complementary subspaces—discriminative and non-discriminative—via adaptive queries, learn them using specialized prompts respectively, and utilize non-discriminative prompts for data augmentation to compensate for the lack of few-shot samples.

## Method

### Overall Architecture

SEC-Prompt learns two sets of prompts on a pre-trained vision model (such as ViT). Through an adaptive query mechanism, input features are decomposed into discriminative and non-discriminative components. D-Prompt reinforces discriminative features to distinguish classes, while ND-Prompt balances non-discriminative information to facilitate generalization to new classes.

### Key Designs

1. **Adaptive Query Decomposition Mechanism**:
    - **Function**: Adaptively decomposes input features into two complementary subspaces: discriminative and non-discriminative.
    - **Mechanism**: Learn an adaptive query module to dynamically determine which dimensions/directions belong to discriminative information (class-relevant) and which belong to non-discriminative information (shared across classes) based on the input features. The union of these two parts covers the entire feature space.
    - **Design Motivation**: Directly training a single prompt cannot balance discriminability and generalizability, whereas explicit decomposition allows separate optimization.

2. **Discriminative Prompt (D-Prompt)**:
    - **Function**: Enhance the separability of class-specific features to make feature distributions of different classes more distinguishable.
    - **Mechanism**: D-Prompt receives signals from the discriminative feature subspace and is trained to reinforce key class-discriminative features. Combined with Prompt Clustering Loss, this prevents noise pollution and ensures robust discriminative feature learning.
    - **Design Motivation**: Under few-shot settings, discriminative features are highly susceptible to noise interference, necessitating specialized prompts and losses for protection.

3. **Non-Discriminative Prompt (ND-Prompt) + Data Augmentation**:
    - **Function**: Balance non-discriminative information to facilitate generalization to new classes, and use it for data augmentation to compensate for the scarcity of few-shot samples.
    - **Mechanism**: ND-Prompt learns general feature patterns shared across classes. Since non-discriminative features possess class-sharing attributes, learned ND-Prompts can be used to augment few-shot data, increasing the diversity of training samples.
    - **Design Motivation**: Non-discriminative features are crucial for generalization to new classes; leveraging them for augmentation is an ingenious way to obtain more training signals under few-shot conditions.

### Loss & Training

- **Classification Loss**: Standard cross-entropy loss, used for classification objectives.
- **Prompt Clustering Loss**: Prevents noise pollution in D-Prompt, ensuring that discriminative prompts of the same class aggregate together while prompts of different classes stay far apart.
- **Data Augmentation Strategy**: Leverages ND-Prompt to perform feature-level augmentation on few-shot data, increasing sample diversity.
- **Incremental Training**: The base stage uses sufficient data to learn the initial prompts, and the incremental stage fine-tunes them using few-shot samples.

## Key Experimental Results

### Main Results

Achieves SOTA on three standard FSCIL benchmark datasets:

| Dataset | SEC-Prompt Performance |
|--------|----------------|
| CIFAR-100 | SOTA |
| ImageNet-R | SOTA |
| CUB-200 | SOTA |

The paper spans pp. 25643-25656, totaling 14 pages (including supplementary materials), containing comprehensive experimental comparisons.

### Ablation Study

- **D-Prompt alone vs ND-Prompt alone vs SEC-Prompt**: Both contribute to different aspects, and their joint usage achieves the best performance.
- **With/Without Prompt Clustering Loss**: This loss is crucial for the quality of discriminative prompts.
- **With/Without ND-Prompt Data Augmentation**: The augmentation strategy contributes significantly during the few-shot incremental stage.
- **Query Methods**: Adaptive query outperforms fixed splitting or random splitting.

### Key Findings

- The design of semantically complementary prompts effectively mitigates the plasticity-stability dilemma.
- Utilizing non-discriminative features for data augmentation is an effective strategy for obtaining additional training signals in few-shot scenarios.
- Prompt Clustering Loss effectively prevents noise overfitting under few-shot conditions.
- The method maintains stability across different incremental learning settings (different numbers of tasks, classes per task, etc.).

## Highlights & Insights

1. **Novel Semantic Decomposition Perspective**: Dividing prompts into discriminative and non-discriminative parts based on semantic functions is more targeted than single-prompt methods.
2. **Ingenious Few-Shot Augmentation Strategy**: Utilizing non-discriminative prompts for data augmentation is both reasonable (cross-class sharing) and effective.
3. **Practical Prompt Clustering Loss**: A concise solution to prevent noise pollution under few-shot conditions.
4. **Concise Framework**: The overall architecture of the method is clear, with well-defined functions for each module, making it easy to understand and implement.

## Limitations & Future Work

1. **Dependency on Pre-trained Models**: The performance of the method heavily relies on the quality of the pre-trained vision model.
2. **Adaptive Query Overhead**: Introducing the adaptive query mechanism introduces some parameter and computational overhead.
3. **Discriminative/Non-Discriminative Boundary**: Whether the division of the two subspaces is optimal deserves further investigation.
4. **More Extreme Few-Shot Settings**: Performance under 1-shot settings and comparison with meta-learning methods warrant attention.
5. **Cross-Domain Scaling**: In scenarios with large domain shifts, the assumption of cross-class sharing of non-discriminative features may not hold.

## Related Work & Insights

- **FSCIL Methods**: Such as CEC, FACT, etc., attempting to address few-shot incremental learning from various angles.
- **Prompt Learning**: Such as L2P, DualPrompt, etc., learning prompts on pre-trained models for incremental learning.
- **Meta-Learning**: Another technical route to address few-shot problems.
- **Inspirations for Future Work**: The idea of semantic decomposition prompting can be generalized to other prompt learning scenarios that need to balance multiple objectives.

## Rating

- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Semantic-Guided Global-Local Collaborative Prompt Learning for Few-Shot Class Incremental Learning](../../CVPR2026/self_supervised/semantic-guided_global-local_collaborative_prompt_learning_for_few-shot_class_in.md)
- [\[CVPR 2025\] Few-Shot Implicit Function Generation via Equivariance](few-shot_implicit_function_generation_via_equivariance.md)
- [\[CVPR 2025\] Task-Agnostic Guided Feature Expansion for Class-Incremental Learning](task-agnostic_guided_feature_expansion_for_class-incremental_learning.md)
- [\[CVPR 2025\] Order-Robust Class Incremental Learning: Graph-Driven Dynamic Similarity Grouping](order-robust_class_incremental_learning_graph-driven_dynamic_similarity_grouping.md)
- [\[CVPR 2025\] MetaWriter: Personalized Handwritten Text Recognition Using Meta-Learned Prompt Tuning](metawriter_personalized_handwritten_text_recognition_using_meta-learned_prompt_t.md)

</div>

<!-- RELATED:END -->

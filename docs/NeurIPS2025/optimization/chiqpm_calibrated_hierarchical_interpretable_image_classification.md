---
title: >-
  [Paper Note] CHiQPM: Calibrated Hierarchical Interpretable Image Classification
description: >-
  [NeurIPS 2025][Optimization][Interpretable Machine Learning] CHiQPM proposes a calibrated hierarchical interpretable image classification method that selects and assigns features to classes via quadratic programming, constructs hierarchical explanation paths, and incorporates interpretable Conformal Prediction set prediction, retaining 99% of black-box model accuracy while providing both global and local interpretability.
tags:
  - "NeurIPS 2025"
  - "Optimization"
  - "Interpretable Machine Learning"
  - "Hierarchical Explanation"
  - "Conformal Prediction"
  - "Image Classification"
  - "Human-AI Complementarity"
date: 2026-05-08
content_hash: 7ff78a0a96989edd
---

# CHiQPM: Calibrated Hierarchical Interpretable Image Classification

**Conference**: NeurIPS 2025
**arXiv**: [2511.20779](https://arxiv.org/abs/2511.20779)  
**Code**: None  
**Area**: Explainable AI / Image Classification
**Keywords**: Interpretable Machine Learning, Hierarchical Explanation, Conformal Prediction, Image Classification, Human-AI Complementarity

## TL;DR

CHiQPM proposes a calibrated hierarchical interpretable image classification method that selects and assigns features to classes via quadratic programming, constructs hierarchical explanation paths, and incorporates interpretable Conformal Prediction set prediction, retaining 99% of black-box model accuracy while providing both global and local interpretability.

## Background & Motivation

**Background**: Deep learning is increasingly adopted in safety-critical domains such as medical diagnosis and autonomous driving. Interpretable-by-design models represent an important direction toward trustworthy AI. The QPM family of methods provides global interpretability through compact binary class representations (each class described by a small number of features), enabling contrastive explanations of inter-class differences.

**Limitations of Prior Work**:
- Although QPM generates contrastive class representations, such contrasts are sparse (e.g., on CUB-2011, only an average of 0.13% of class pairs are contrastive).
- Existing interpretable models lack principled uncertainty quantification for local explanations—saliency maps convey no confidence information.
- Prototype networks (e.g., ProtoTree) appear interpretable, yet their similarity spaces are learned freely, making their behavior unpredictable to humans.
- Conformal Prediction (CP) provides set-prediction guarantees, but the resulting prediction sets typically contain semantically unrelated classes.

**Key Challenge**: The trade-off between interpretability and accuracy—more compact representations are easier to interpret but limit accuracy; CP-generated sets lack semantic coherence.

**Goal**:
- How to increase the proportion of contrastive class pairs while maintaining high accuracy?
- How to provide hierarchical local explanations that mirror human reasoning?
- How to produce semantically coherent class sets via CP set prediction?

**Key Insight**: Combining feature detection with a hierarchical class structure. The binary feature assignments from QPM naturally induce a class hierarchy tree; traversing this tree yields semantically coherent set predictions.

**Core Idea**: Introduce hierarchical constraints and a Feature Grounding Loss into the QPM framework so that feature assignments naturally form a traversable class hierarchy, and integrate CP into this structure to realize interpretable, calibrated set prediction.

## Method

### Overall Architecture

The CHiQPM pipeline consists of five stages:
1. **Train a dense model**: Train a black-box backbone with a Feature Diversity Loss $\mathcal{L}_{div}$ to ensure that initial feature map activations are distributed across different spatial locations.
2. **Compute QP constants**: Compute the class–feature similarity matrix $\mathbf{A}$, the feature–feature similarity matrix $\mathbf{R}$, the linear bias term $\mathbf{b}$, and the set of similar class pairs $\mathbb{K}$.
3. **Solve the QP with hierarchical constraints**: Extend QPM's objective with hierarchical constraints, jointly optimizing feature selection $\mathbf{s} \in \{0,1\}^{n_f}$ and class–feature assignment $\mathbf{W} \in \{0,1\}^{n_c \times n_f}$.
4. **Fine-tune features**: Fine-tune the compressed model using the Feature Grounding Loss $\mathcal{L}_{feat}$ with ReLU activations.
5. **Calibrate**: Calibrate the hierarchical set predictor using CP.

### Key Designs

**Quadratic Programming with Hierarchical Constraints**: The original QPM objective maximizes the correlation between selected features and classes while enforcing feature distinctiveness and locality. CHiQPM augments this with hierarchical constraints that require the feature assignment $\mathbf{W}^*$ to induce a meaningful hierarchy over classes. Specifically, QP constraints ensure that each class is assigned exactly $k$ features (compactness) and that their combination uniquely identifies the class. The hierarchical constraints further encourage similar classes to share more features, forming semantically meaningful groupings in the class tree.

**Feature Grounding Loss $\mathcal{L}_{feat}$**: This is one of the paper's key contributions. Conventional features may be polysemantic—a single feature simultaneously detects multiple human concepts. $\mathcal{L}_{feat}$, combined with ReLU activations, encourages learned features to be more grounded and sparse, meaning each feature is more likely to correspond to a single human-interpretable concept (e.g., a "red eye" feature distinguishing two species of black birds in Figure 1). The ReLU activation ensures non-negative feature responses, giving "feature not detected" a clear semantic meaning.

**Hierarchical Local Explanations**: Given an input image, CHiQPM constructs an instance-specific explanation hierarchy. Each level $n$ corresponds to classification using the top-$n$ most salient features:
- Level 1: Only the strongest feature is used, potentially matching multiple classes that share it.
- Level 2: The second feature is added, further narrowing the candidate set.
- Level $k$: The class is uniquely identified.

This hierarchy answers multiple questions: which meaningful features are detected in the image? How does each successive feature narrow the candidate class set? Which set should be predicted to guarantee a target coverage rate?

**Built-in Interpretable Conformal Prediction**: CHiQPM uniquely integrates CP into its hierarchical explanation. The prediction set at level $n$ is defined as all classes that share the top-$n$ salient features with the most probable class in the hierarchy tree. Through CP calibration, the system dynamically selects an appropriate level for each instance: easy instances may require only level $k$ (a single class), while difficult instances may fall back to level 2 or 1, predicting a semantically coherent group (e.g., all black birds).

### Loss & Training

The overall training procedure:
1. Stage 1: Train the dense model with $\mathcal{L}_{CE} + \lambda_{div} \mathcal{L}_{div}$.
2. Stage 2: Solve the QP to obtain optimal feature selection and assignment.
3. Stage 3: Fine-tune the compressed model with $\mathcal{L}_{CE} + \lambda_{feat} \mathcal{L}_{feat}$ and ReLU activations to improve feature grounding.

## Key Experimental Results

### Main Results

CHiQPM is evaluated across multiple datasets and architectures. Core results:

| Dataset | Architecture | Black-box Acc. | CHiQPM Acc. | Retention Ratio | Contrastive Pair Ratio |
|---------|-------------|---------------|-------------|-----------------|----------------------|
| CUB-2011 | ResNet-50 | ~82% | ~81% | 99%+ | Significant gain vs. QPM |
| ImageNet-1K | ResNet-50 | ~76% | ~75% | ~99% | Gap halved |
| CUB-2011 | Multiple archs | Varies | Close to black-box | 99%+ | Most classes gain contrastive explanations |

Key finding: CHiQPM achieves state-of-the-art accuracy as a point predictor, retaining over 99% of non-interpretable model accuracy. On ImageNet-1K, the accuracy gap relative to the black-box baseline is reduced by more than half.

### Ablation Study

| Component | Contrastive Pair Ratio | Accuracy | Coverage Efficiency |
|-----------|----------------------|----------|---------------------|
| Base QPM | Baseline | Baseline | — |
| + Hierarchical constraints | ↑ Significant | Maintained | Usable |
| + Feature Grounding Loss | ↑ | Maintained or ↑ | Improved |
| + ReLU activation | ↑ | Maintained | Improved |
| Full CHiQPM | Best | Best | Best |

CP set prediction efficiency: On CUB-2011 (5/50 features per class), CHiQPM's built-in CP method achieves coverage–set-size curves competitive with standard CP baselines (THR, APS), while producing semantically coherent prediction sets.

### Key Findings

1. **Interpretability without sacrificing accuracy**: CHiQPM retains over 99% of black-box model accuracy while providing comprehensive global and local interpretability.
2. **Substantial gain in contrastive coverage**: Compared to QPM, CHiQPM significantly increases the proportion of class pairs with contrastive explanations, enabling more classes to be distinguished by feature differences.
3. **Cognitive plausibility of hierarchical explanations**: The hierarchical explanation structure more closely mirrors human reasoning—first identifying a broad category (black bird), then narrowing to a specific species.
4. **Semantically coherent set predictions**: Classes within a CP prediction set are adjacent in the hierarchy and are semantically similar.

## Highlights & Insights

- **Natural integration of CP with interpretable models**: Rather than post-hoc application of CP, the model architecture itself supports hierarchical set prediction—an elegant design choice.
- **Feature Grounding Loss addresses polysemanticity**: The approach directly tackles the fundamental problem of feature polysemanticity in interpretable ML.
- **Scalable to ImageNet**: The method operates on ImageNet with 1,000 classes, demonstrating scalability.
- **Human-AI complementarity perspective**: Unlike AI systems designed to replace human judgment, CHiQPM is designed to assist human expert decision-making.

## Limitations & Future Work

- The computational cost of QP solving grows with the number of classes, which may limit applicability to extremely large-scale classification tasks.
- The semantic grounding quality of features still depends on the representational quality of the backbone model.
- The hierarchy depth is directly tied to the number of features $k$; with very few features, the resulting hierarchy may be too shallow.
- The paper validates the method primarily on visual classification tasks; applicability to other modalities (text, multimodal) remains to be explored.

## Related Work & Insights

- **QPM / Q-SENN family**: The direct predecessors of CHiQPM; this work extends them with hierarchical structure and calibration capabilities.
- **Conformal Prediction**: A distribution-free set-prediction framework; CHiQPM is the first to intrinsically integrate CP with an interpretable model.
- **Prototype networks (ProtoPNet, ProtoTree)**: Another class of interpretable models, but the unpredictability of their learned similarity spaces is an inherent limitation.
- **Concept Bottleneck Models (CBM)**: Use human concepts as a bottleneck, but require concept-level annotations.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The integration of hierarchical CP with interpretable models is a genuine contribution.
- **Technical Depth**: ⭐⭐⭐⭐ — QP optimization and CP theory are tightly coupled.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multi-dataset, multi-architecture validation with thorough ablation.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure and intuitive illustrations.
- **Value**: ⭐⭐⭐⭐ — Interpretability without accuracy sacrifice has practical value in safety-critical domains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Learning Interpretable Queries for Explainable Image Classification with Information Pursuit](../../ICCV2025/optimization/learning_interpretable_queries_for_explainable_image_classification_with_informa.md)
- [\[NeurIPS 2025\] Optimal Rates for Generalization of Gradient Descent for Deep ReLU Classification](optimal_rates_for_generalization_of_gradient_descent_for_deep_relu_classificatio.md)
- [\[NeurIPS 2025\] Deep Taxonomic Networks for Unsupervised Hierarchical Prototype Discovery](deep_taxonomic_networks_for_unsupervised_hierarchical_prototype_discovery.md)
- [\[NeurIPS 2025\] Natural Gradient Descent for Improving Variational Inference Based Classification of Radio Galaxies](natural_gradient_descent_for_improving_variational_inference_based_classificatio.md)
- [\[NeurIPS 2025\] Adaptive Algorithms with Sharp Convergence Rates for Stochastic Hierarchical Optimization](adaptive_algorithms_with_sharp_convergence_rates_for_stochas.md)

</div>

<!-- RELATED:END -->

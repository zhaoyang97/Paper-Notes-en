---
title: >-
  [Paper Note] C2AL: Cohort-Contrastive Auxiliary Learning for Large-scale Recommendation Systems
description: >-
  [ICLR2026][Recommender Systems][recommendation system] This paper proposes C2AL (Cohort-Contrastive Auxiliary Learning), which data-drivenly identifies user cohort pairs with maximal distributional divergence and constru…
tags:
  - "ICLR2026"
  - "Recommender Systems"
  - "recommendation system"
  - "auxiliary learning"
  - "cohort contrastive"
  - "attention mechanism"
  - "representation bias"
date: 2026-05-08
content_hash: dc4c3e380bde0453
---

# C2AL: Cohort-Contrastive Auxiliary Learning for Large-scale Recommendation Systems

**Conference**: ICLR2026
**arXiv**: [2510.02215](https://arxiv.org/abs/2510.02215)  
**Code**: To be confirmed  
**Area**: Recommendation Systems
**Keywords**: recommendation system, auxiliary learning, cohort contrastive, attention mechanism, representation bias

## TL;DR
This paper proposes C2AL (Cohort-Contrastive Auxiliary Learning), which data-drivenly identifies user cohort pairs with maximal distributional divergence and constructs contrastive auxiliary binary classification tasks to regularize the shared encoder. This transforms FM attention weights from sparse to dense, mitigating representation bias for minority cohorts in large-scale recommendation systems. The approach is validated on 6 Meta production models with billions of data points.

## Background & Motivation

**Background**: Large-scale recommendation models (e.g., DHEN) are trained under a single global objective, implicitly assuming homogeneous user cohort distributions. Multi-task learning (MTL) with auxiliary tasks is commonly used in industry to improve representations, but auxiliary task design largely relies on empirical heuristics.

**Limitations of Prior Work**: Real-world data consists of heterogeneous cohorts. As models and data scale up, optimization favors high-density regions (majority cohorts), resulting in: (a) sparse and concentrated FM attention weights—a large number of feature interaction paths are wasted; (b) feature patterns of minority cohorts are ignored, causing representation bias.

**Key Challenge**: Global optimization provides only "averaged" gradient signals, causing FM attention to converge to a sparse state that captures only globally frequent feature interactions, lacking cohort-specific gradient driving forces. Existing multi-task gradient methods such as PCGrad and CAGrad focus on managing task conflicts but do not establish a causal chain from auxiliary loss → attention mechanism → representation improvement.

**Goal**: (a) Principally discover cohort pairs with maximal distributional divergence; (b) construct auxiliary tasks to inject cohort-specific gradients; (c) provide interpretable mechanistic analysis of how auxiliary losses precisely alter FM attention.

**Key Insight**: By analyzing the FM attention update equations from a gradient propagation perspective, the paper finds that gradients from auxiliary losses are directly superimposed onto the update of attention matrix $\mathbf{Y}$, providing a precise mechanistic explanation.

**Core Idea**: Distributional divergence is used to identify maximally contrasting head/tail cohort pairs, which are used to construct auxiliary binary classification tasks. During training, cohort-specific gradients densify FM attention weights; auxiliary heads are discarded at inference with zero additional overhead.

## Method

### Overall Architecture
Input: user-ad feature vector $\mathbf{x}$ → shared encoder $f(\mathbf{x};\theta_S)$ generates embedding $\mathbf{h}$ → primary task head $g_{\text{primary}}$ predicts CTR. C2AL adds two auxiliary heads $g_{\text{head}}, g_{\text{tail}}$ that share the encoder but each predicts the label of the corresponding cohort. Auxiliary heads are discarded after training, leaving the inference architecture unchanged.

### Key Designs

1. **Contrastive Cohort Discovery**:

    - Function: Automatically identifies cohort pairs with maximal distributional divergence from data.
    - Mechanism: Data is partitioned into $\{\mathcal{C}_1, \ldots, \mathcal{C}_N\}$ along interpretable semantic axes (e.g., user value, age). A baseline model's prediction distributions are used to compute pairwise divergences (KL, JS, Wasserstein, cosine similarity), and the pair with the largest divergence is selected as $\mathcal{C}_{\text{head}}$ and $\mathcal{C}_{\text{tail}}$.
    - Design Motivation: Rather than arbitrarily selecting cohorts, this principled approach identifies the pair with the greatest difference in model prediction behavior, ensuring that auxiliary gradient signals are "partially conflicting" with primary task gradients for maximum information gain.

2. **Contrastive Auxiliary Learning**:

    - Function: Constructs two cohort-specific auxiliary binary classification tasks.
    - Mechanism: $y_{\text{head}} = y \cdot \mathbb{I}(\mathbf{x} \in \mathcal{C}_{\text{head}})$, $y_{\text{tail}} = y \cdot \mathbb{I}(\mathbf{x} \in \mathcal{C}_{\text{tail}})$, with total loss: $\mathcal{L}_{\text{C2AL}} = \mathcal{L}_{\text{primary}} + \lambda_{\text{head}} \mathcal{L}_{\text{head}} + \lambda_{\text{tail}} \mathcal{L}_{\text{tail}}$
    - Design Motivation: Auxiliary labels are positive only within the corresponding cohort and zero otherwise, forcing the shared encoder to learn cohort-discriminative representations. The partially conflicting distributions of the two cohorts allow auxiliary gradients to break the majority-cohort dominance of the primary task.

3. **Mechanistic Interpretability Analysis**:

    - Function: Mathematically demonstrates how auxiliary losses precisely alter FM attention.
    - Mechanism: DHEN's FM attention computes $\mathbf{G} = \mathbf{X}\mathbf{X}^\top \mathbf{Y}$; taking the gradient with respect to attention matrix $\mathbf{Y}$ yields: $\nabla_{\mathbf{Y}} \mathcal{L}_{\text{C2AL}} = (\mathbf{X}\mathbf{X}^\top)(\nabla_{\mathbf{G}} \mathcal{L}_{\text{primary}} + \lambda_{\text{aux}} \nabla_{\mathbf{G}} \mathcal{L}_{\text{aux}})$
    - Key Insight: The auxiliary gradient $\nabla_{\mathbf{G}} \mathcal{L}_{\text{aux}}$ is directly injected into the update of $\mathbf{Y}$—this is not indirect regularization but a direct modification of attention weights. Since auxiliary gradients encode feature interaction patterns of minority cohorts, $\mathbf{Y}$ is forced to transition from sparse (capturing only majority-cohort high-frequency interactions) to dense and diverse (also capturing minority-cohort-specific interactions).
    - Empirical Validation: Visualizations confirm that C2AL primarily affects attention layer weights, while preceding layer weights change minimally—corroborating the theoretical predictions.

### Loss & Training
- Training: Three-head joint optimization (primary + head + tail); auxiliary weights $\lambda_{\text{head}}, \lambda_{\text{tail}}$ are hyperparameters.
- Inference: Auxiliary heads are discarded, reverting to a single-task architecture—zero additional inference overhead.
- This is the core engineering advantage of C2AL: minimal increase in training cost (auxiliary heads are lightweight) with completely unchanged inference cost.

## Key Experimental Results

### Main Results

| Model/Platform | Normalized Entropy Reduction | Minority Cohort Gain | Notes |
|---|---|---|---|
| Model A (Instagram CTR) | ↓ 0.16% | > 0.30% | DHEN baseline |
| Model B | Significant improvement | > 0.30% | Different business scenario |
| Models C–F | Consistent positive | Consistent positive | Effective across all 6 production models |

### Ablation Study (Attention Weight Distribution Analysis)

| Configuration | Attention Weight Distribution | Preceding Layer Change | Notes |
|---|---|---|---|
| Baseline | Sparse, light-tailed, concentrated on few paths | — | Majority-cohort dominated |
| + C2AL | Dense, diverse, more paths activated | Nearly unchanged | Auxiliary gradients precisely modify attention |
| Preceding layer comparison | — | Minimal change | C2AL effect is attention-layer-specific |

### Key Findings
- **C2AL selectively modifies the attention layer**: Preceding layer weight distributions remain nearly unchanged, while attention MLP weights shift significantly—validating the theoretical prediction that "auxiliary losses directly inject into attention updates."
- **Weight densification = better minority cohort representation**: A denser $\mathbf{Y}$ means more sparse embeddings participate in meaningful second-order interactions, and minority-cohort-specific feature combinations are no longer ignored.
- **Cross-model consistency**: All 6 production models across different scenarios exhibit the same pattern, indicating that the mechanism is general and not scenario-specific.
- **0.16% normalized entropy reduction is significant at scale**: At billions-of-data-points scale, this corresponds to substantial advertising revenue gains and user experience improvements.

## Highlights & Insights
- **Interpretable auxiliary learning mechanism**: Unlike prior work that explains auxiliary tasks as "making representations better without knowing why," C2AL provides a complete causal chain from auxiliary loss → gradient → attention matrix → representation. This is the paper's most central contribution—elevating auxiliary learning from "empirically effective" to "mechanistically interpretable."
- **Zero inference overhead regularization**: Auxiliary heads are used only during training and completely discarded at serving time—a critical property for industrial systems where inference latency directly impacts revenue.
- **Data-driven cohort discovery**: No manual specification of "important" cohorts is required; distributional divergence automatically identifies them, reducing dependence on domain knowledge and making the approach more general.
- **First systematic analysis of FM attention sparsity**: The paper reveals the mechanism by which global optimization causes attention degradation; this finding has independent value for understanding large-scale recommendation models.

## Limitations & Future Work
- **Cohort discovery still requires choosing semantic axes**: Although divergence computation is automated, the choice of "which dimension to partition along" still requires domain knowledge. Fully automatic cohort discovery (e.g., clustering + divergence) is a natural improvement direction.
- **Validation limited to FM-based attention**: The analysis is specific to the DHEN architecture. Whether Transformer-based recommendation models (e.g., SASRec) exhibit similar sparse degradation in self-attention warrants exploration.
- **Absence of A/B test results**: Offline evaluation across 6 models is thorough, but no online A/B test results are reported. Industrial papers typically provide such data to demonstrate real-world effectiveness.
- **Auxiliary weight $\lambda$ selection**: The paper does not discuss hyperparameter sensitivity in detail. Whether retuning is required across different models and scenarios remains unknown.

## Related Work & Insights
- **vs. PCGrad/CAGrad and other multi-task gradient methods**: These methods manage gradient conflicts among pre-defined tasks but do not construct new tasks. C2AL actively constructs "partially conflicting" auxiliary tasks—shifting from passive coordination to proactive design.
- **vs. MMoE/PLE and other multi-task architectures**: These learn task-specific parameter sharing strategies through architectural design. C2AL takes a different route—no architectural changes, only auxiliary losses added—making it more suitable for scenarios where modifying the production model architecture is undesirable.
- **vs. fairness/bias mitigation methods**: Traditional fairness methods address cohort imbalance directly through reweighting or constrained optimization. C2AL takes a different perspective—it does not directly optimize fairness metrics but indirectly benefits minority cohorts by improving attention representations.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of cohort contrast, auxiliary learning, and gradient mechanistic analysis is novel; elevating auxiliary learning from "empirically effective" to "mechanistically interpretable" represents a qualitative advancement.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Industrial-scale validation on 6 Meta production models with billions of data points; weight visualization analyses robustly support theoretical predictions.
- Writing Quality: ⭐⭐⭐⭐ The narrative arc from problem identification → mechanistic analysis → method design → empirical validation is clear; mathematical derivations are concise and transparent.
- Value: ⭐⭐⭐⭐ Directly applicable engineering value for large-scale recommendation systems; mechanistic analysis has academic value for understanding FM attention.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Semantic Retrieval Augmented Contrastive Learning for Sequential Recommendation](../../NeurIPS2025/recommender/semantic_retrieval_augmented_contrastive_learning_for_sequential_recommendation.md)
- [\[ICML 2026\] Rethinking Contrastive Learning for Graph Collaborative Filtering: Limitations and a Simple Remedy](../../ICML2026/recommender/rethinking_contrastive_learning_for_graph_collaborative_filtering_limitations_an.md)
- [\[AAAI 2026\] TraveLLaMA: A Multimodal Travel Assistant with Large-Scale Dataset and Structured Reasoning](../../AAAI2026/recommender/travellama_a_multimodal_travel_assistant_with_large-scale_dataset_and_structured.md)
- [\[ICML 2026\] GCIB: Graph Contrastive Information Bottleneck for Multi-Behavior Recommendation](../../ICML2026/recommender/gcib_graph_contrastive_information_bottleneck_for_multi-behavior_recommendation.md)
- [\[ICLR 2026\] From Evaluation to Defense: Advancing Safety in Video Large Language Models](from_evaluation_to_defense_advancing_safety_in_video_large_language_models.md)

</div>

<!-- RELATED:END -->

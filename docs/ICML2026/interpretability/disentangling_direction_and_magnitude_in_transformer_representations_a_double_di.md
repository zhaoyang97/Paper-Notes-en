---
title: >-
  [Paper Note] Disentangling Direction and Magnitude in Transformer Representations: A Double Dissociation Through L2-Matched Perturbation Analysis
description: >-
  [ICML 2026][Interpretability][Linear Representation Hypothesis] This paper employs an L2-matched perturbation protocol to demonstrate that in the Pythia model series…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "Linear Representation Hypothesis"
  - "Direction vs. Magnitude"
  - "L2-Matched Perturbation"
  - "LayerNorm"
  - "Attention Path"
date: 2026-05-08
content_hash: 55802550b5545892
---

# Disentangling Direction and Magnitude in Transformer Representations: A Double Dissociation Through L2-Matched Perturbation Analysis

**Conference**: ICML 2026  
**arXiv**: [2602.11169](https://arxiv.org/abs/2602.11169)  
**Code**: Not released  
**Area**: Interpretability / Representation Geometry / Causal Intervention  
**Keywords**: Linear Representation Hypothesis, Direction vs. Magnitude, L2-Matched Perturbation, LayerNorm, Attention Path

## TL;DR
This paper employs an L2-matched perturbation protocol to demonstrate that in the Pythia model series, angular (direction) perturbations are up to 42.9 times more destructive to language modeling loss than magnitude perturbations of equal Euclidean displacement. Conversely, magnitude perturbations damage syntax (subject-verb agreement) significantly more than angular ones—forming a "double dissociation" in the cognitive neuroscience sense, where direction maps to the attention pathway and magnitude to the LayerNorm pathway.

## Background & Motivation
**Background**: The Linear Representation Hypothesis (LRH) is a cornerstone of current interpretability research, encoding concepts as directions in activation space and extracting semantic features with linear probes. Activation patching, TunedLens, and representation engineering are all built upon the assumption that "direction matters."

**Limitations of Prior Work**: LRH is largely silent on magnitude (norm). However, norms in Transformers are not constant: Kobayashi et al. found they vary significantly across tokens and layers; LayerNorm explicitly manipulates norms; and representation engineering modifies behavior by scaling vectors. There has been no systematic comparison of the causal importance of direction versus magnitude.

**Key Challenge**: A naive comparison—perturbing direction by a small angle versus scaling magnitude by a small factor—involves completely different actual displacements in the representation space. If angular perturbation causes more damage, is it because direction is inherently more important, or simply because that specific perturbation was "more violent"? Without controlling for the magnitude of displacement, all comparisons are invalid.

**Goal**: (1) Construct an L2-matched perturbation protocol to eliminate displacement-size confounding; (2) Systematically measure the causal importance of direction and magnitude for various downstream tasks in Pythia; (3) Locate the mechanistic paths of influence through pathway restoration experiments.

**Key Insight**: Borrowing the "double dissociation" tool from cognitive neuroscience—if operation A primarily impairs task X but not Y, while operation B primarily impairs Y but not X, it suggests that X and Y are supported by separable functional subsystems.

**Core Idea**: Parameterize "perturbation intensity" with $\delta$, forcing the Euclidean displacement of both angular and magnitude perturbations at the intervention layer to be exactly $\delta$, then compare their impact on loss and syntactic accuracy.

## Method

### Overall Architecture
Applying one of two perturbations to the hidden states $\mathbf{h}$ at each token position in the middle layers (layers 8-15) of Pythia-410M:

1.  **Magnitude Perturbation**: $\mathbf{h}'_{\text{mag}} = \alpha \mathbf{h}$, direction remains constant, length changes.
2.  **Angular Perturbation**: $\mathbf{h}'_{\text{ang}} = \|\mathbf{h}\| \cdot \hat{\mathbf{h}}'$, length remains constant, direction rotates by $\theta$.

Analytical formulas ensure both satisfy $\|\mathbf{h} - \mathbf{h}'_{\text{mag}}\| = \|\mathbf{h} - \mathbf{h}'_{\text{ang}}\| = \delta$. The downstream effects are measured via (a) WikiText cross-entropy loss; (b) BLiMP subject-verb agreement accuracy; and (c) recovery rates after attention/LayerNorm pathway restoration.

### Key Designs

1.  **L2-Matched Perturbation Formula**:
    - **Function**: Eliminates the confounding effect of perturbation size when comparing direction and magnitude.
    - **Mechanism**: For magnitude, $\alpha = 1 \pm \delta / \|\mathbf{h}\|$ is derived from $|1-\alpha| \cdot \|\mathbf{h}\| = \delta$, with the sign chosen randomly (scaling up or down), requiring $\delta < \|\mathbf{h}\|$. For angle, a unit vector $\mathbf{v} \perp \mathbf{h}$ is sampled, such that $\mathbf{h}'_{\text{ang}} = \|\mathbf{h}\|(\cos\theta \cdot \hat{\mathbf{h}} + \sin\theta \cdot \hat{\mathbf{v}})$. From $\|\mathbf{h} - \mathbf{h}'_{\text{ang}}\| = \delta$, it follows that $\theta = \arccos(1 - \delta^2 / 2\|\mathbf{h}\|^2)$. Post-perturbation displacement error is empirically verified to be < 0.01.
    - **Design Motivation**: Projecting "direction vs. magnitude" from two incomparable axes onto a unified $\delta$ dimension ensures that differences in causal effects are entirely attributable to the "type" of perturbation.

2.  **Cross-over Dissociation Measurement**:
    - **Function**: Measures the impact of perturbations across "macro loss" and "fine-grained syntax."
    - **Mechanism**: For macro-impact, next-token cross-entropy on 281 WikiText-103 sentences (lengths 10-64) is used. For fine-grained impact, 200 BLiMP minimal pairs for subject-verb agreement (e.g., "The dogs run" vs. "The dogs runs") are tested. Experiments use 6 intensities of $\delta \in \{1, 2, 5, 10, 15, 20\}$ with 5 random seeds and pair t-tests with Bonferroni correction.
    - **Design Motivation**: These tasks are complementary—next-token prediction is high-entropy and sensitive to direction, while subject-verb agreement is a low-dimensional discrete decision more sensitive to numerical magnitudes like the norm.

3.  **Attention / LayerNorm Pathway Restoration**:
    - **Function**: Pinpoints which computational path carries the perturbation effects.
    - **Mechanism**: For a perturbed state $\mathbf{h}'$, specific intermediate products (attention patterns or LayerNorm outputs) are replaced with their "clean" versions to see how much downstream loss is recovered. High recovery implies the path carried the primary perturbation effect. Attention restoration uses unperturbed attention weights; LayerNorm restoration replaces perturbed LN outputs with unperturbed ones.
    - **Design Motivation**: Correlational observations only suggest "direction is important," but causal interventions via pathway restoration are necessary to establish mechanistic claims like "direction influences loss through attention."

### Loss & Training
No training is involved; this is a pure inference-time intervention study. Pythia-410M / 1.4B are run in float32. Orthogonal directions are sampled independently across 5 seeds for each $\delta$.

## Key Experimental Results

### Main Results
Loss Damage (Table 1, baseline loss = 4.107):

| $\delta$ | Magnitude $\Delta$loss | Angular $\Delta$loss | Ang/Mag Ratio | p |
|----------|-------------------|-------------------|----------|-----|
| 1.0 | 0.009 | 0.368 | **42.9×** | <0.001 |
| 2.0 | 0.042 | 0.983 | 23.2× | <0.001 |
| 5.0 | 0.700 | 3.757 | 5.4× | <0.001 |
| 10.0 | 3.262 | 7.061 | 2.2× | <0.001 |
| 20.0 | 5.433 | 7.750 | 1.4× | <0.001 |

Syntax Accuracy (Table 2, baseline 89.5%):

| $\delta$ | Post-Mag Acc | Post-Ang Acc | Mag Drop | Ang Drop |
|----------|--------------|--------------|----------|----------|
| 5.0 | 69.1% | 87.9% | **20.4%** | 1.6% |
| 10.0 | 56.0% | 77.1% | 33.5% | 12.4% |
| 15.0 | 53.5% | 67.4% | 36.0% | 22.1% |

At $\delta = 5$, direction dominates loss (5.4x difference), while magnitude dominates syntax damage (12.8x difference)—these opposing advantages constitute a double dissociation.

### Ablation Study
Pathway Restoration (Percentage of total damage recovered):

| Restoration Path | Angular Recovery | Magnitude Recovery | Bias |
|----------|--------------|--------------|------|
| Attention | **28.4%** | 15.2% | Angle → Attention |
| LayerNorm | 13.7% | **29.9%** | Magnitude → LayerNorm |

The pattern is replicated on Pythia-1.4B (the Angular/Magnitude ratio grows from 23.2x to 56.8x). The dissociation vanishes on RMSNorm architectures (lacking affine LN), indicating the phenomenon is tightly coupled with LayerNorm's normalization mechanism.

Inter-layer Propagation (Table 4, $\delta = 5$):

| Layer | Angular L2 Displacement | Magnitude L2 Displacement | Ratio |
|-------|-------------|-------------|------|
| 8 (Start) | 5.00 | 5.00 | 1.00× |
| 15 (End) | 35.9 | 12.7 | 2.82× |
| 23 (Final) | 123.8 | 38.9 | 3.18× |

Angular perturbations are amplified by 24.8x compared to only 7.8x for magnitude—LayerNorm naturally suppresses magnitude shifts while leaving angular shifts unchecked.

### Key Findings
- **Direction acts via attention channels**: Since attention is essentially $\text{softmax}(QK^T / \sqrt{d})$ and relies on cosine similarity, angular perturbations directly alter routing. LayerNorm re-normalizes the norm, thus absorbing magnitude changes.
- **Syntax is a norm-sensitive task**: Decisions like subject-verb agreement require precise numerical comparison and rely more on norm-regulated "processing intensity" than on which token the attention is routed to.
- **Extreme asymmetry at small $\delta$**: The angular advantage is 42.9x in the low $\delta$ region but drops to 1.4x at high $\delta$ as model predictions hit a "floor."
- **Architectural dependence**: The dissociation does not appear in RMSNorm architectures, suggesting it is a specific geometric division of labor in LayerNorm, not a universal Transformer property.

## Highlights & Insights
- **L2 Matching = Clean Experimental Design**: Maintaining mathematical simplicity while achieving conceptual clarity represents a paradigm-level contribution to geometric causality studies.
- **Interdisciplinary Borrowing**: Introducing the "double dissociation" framework from cognitive neuroscience into interpretability provides much stronger argumentative weight than single-direction ablations.
- **Mechanism Localization + Boundary Conditions**: The chain of "phenomenon → mechanism → boundary condition" (using RMSNorm as a counter-example) is logically rigorous.
- **Warning for Representation Engineering**: Directional editing (steering vectors) and magnitude scaling (activation scaling) are not interchangeable; they correspond to different sub-capabilities.

## Limitations & Future Work
- **Pathways explain ~30%**: Combined attention/LN restoration accounts for less than half the damage; the remaining 70% of the path remains a black box.
- **Statistical Power**: Only 5 seeds were used, which limits statistical power despite the large effect size.
- **Single Syntax Task**: Only subject-verb agreement was tested; it remains unknown if other syntactic phenomena (NPI licensing, island constraints) are equally norm-sensitive.
- **Fixed Intervention Layers**: Only layers 8-15 were systematically scanned; dissociation might vary in earlier or later processing stages.
- **Random Orthogonal Perturbation**: "Random" does not guarantee "semantic neutrality" due to representation space anisotropy; some perturbations might inadvertently hit critical subspaces.
- **Generalization**: Future work should test combinations of RMSNorm, sandwich norm, and different position encodings.

## Related Work & Insights
- **vs. Park et al. 2023 (LRH Formalization)**: That work defined the directional encoding hypothesis; this paper is a refinement that adds the neglected dimension of magnitude.
- **vs. Kobayashi et al. 2020 (Norm in Attention)**: They observed norm modulation in attention weights; this paper uses causal intervention to prove the specific importance of norm for syntax.
- **vs. Meng et al. 2022 (ROME/Activation Patching)**: While belonging to the same causal intervention family, this work decomposes activations into direction/magnitude before patching, providing finer granularity.
- **Insight**: Model editing should distinguish between steering (altering attention routing behavior) and scaling (altering processing intensity). Safety research could explore whether "jailbreak prompts" primarily perturb direction or magnitude.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The L2-matched protocol and double dissociation framework are rare methodological contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multiple models, dual tasks, pathway restoration, and architectural counter-examples; points deducted for low seed count and model size.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The argumentation chain is excellent, derivations are clear, and boundary conditions are well-discussed.
- **Value**: ⭐⭐⭐⭐ Important for refining LRH and guiding representation engineering, though it requires specific causal intervention setups.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Similarity-Distance-Magnitude Activations](../../ACL2026/interpretability/similarity-distance-magnitude_activations.md)
- [\[AAAI 2026\] Probing Preference Representations: A Multi-Dimensional Evaluation and Analysis Method for Reward Models](../../AAAI2026/interpretability/probing_preference_representations_a_multi-dimensional_evaluation_and_analysis_m.md)
- [\[ICML 2026\] Prototype Transformer: Towards Language Model Architectures Interpretable by Design](prototype_transformer_towards_language_model_architectures_interpretable_by_desi.md)
- [\[ACL 2026\] Crosscoding Through Time: Tracking Emergence & Consolidation Of Linguistic Representations Throughout LLM Pretraining](../../ACL2026/interpretability/crosscoding_through_time_tracking_emergence_consolidation_of_linguistic_represen.md)
- [\[ICML 2026\] Learning Coherent Representations: A Topological Approach to Interpretability](learning_coherent_representations_a_topological_approach_to_interpretability.md)

</div>

<!-- RELATED:END -->

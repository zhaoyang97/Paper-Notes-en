---
title: >-
  [Paper Note] Disentangling Direction and Magnitude in Transformer Representations: A Double Dissociation Through L2-Matched Perturbation Analysis
description: >-
  [ICML 2026][Interpretability][Linear Representation Hypothesis] This paper employs an L2-matched perturbation protocol to demonstrate that in the Pythia series, angular (direction) perturbations are 42.9 times more destructive to language modeling loss than magnitude perturbations of equal displacement. Conversely, magnitude perturbations damage syntax (subject-verb agreement) significantly more than angular ones. This constitutes a "double dissociation" in the cognitive neur…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "Linear Representation Hypothesis"
  - "Direction vs Magnitude"
  - "L2-Matched Perturbation"
  - "LayerNorm"
  - "Attention Pathways"
date: 2026-05-08
content_hash: 9fb43b3e12e2f3a3
---

# Disentangling Direction and Magnitude in Transformer Representations: A Double Dissociation Through L2-Matched Perturbation Analysis

**Conference**: ICML 2026  
**arXiv**: [2602.11169](https://arxiv.org/abs/2602.11169)  
**Code**: Not released  
**Area**: Interpretability / Representation Geometry / Causal Intervention  
**Keywords**: Linear Representation Hypothesis, Direction vs Magnitude, L2-Matched Perturbation, LayerNorm, Attention Pathways

## TL;DR
This paper employs an L2-matched perturbation protocol to demonstrate that in the Pythia series, angular (direction) perturbations are 42.9 times more destructive to language modeling loss than magnitude perturbations of equal displacement. Conversely, magnitude perturbations damage syntax (subject-verb agreement) significantly more than angular ones. This constitutes a "double dissociation" in the cognitive neuroscience sense, corresponding to attention pathways for direction and LayerNorm pathways for magnitude.

## Background & Motivation
**Background**: The Linear Representation Hypothesis (LRH) is a cornerstone of current interpretability research, positing that concepts are encoded as directions in activation space and extracted via linear probes. Activation patching, TunedLens, and representation engineering are all built on the assumption that "direction matters."

**Limitations of Prior Work**: LRH remains silent regarding magnitude (norm), yet norms in Transformers are not constant. Kobayashi et al. found that norms vary significantly across tokens and layers; LayerNorm explicitly manipulates norms, and representation engineering modifies behavior by scaling vectors. No systematic comparison has been conducted regarding the causal importance of direction versus magnitude.

**Key Challenge**: Naive comparisons—such as perturbing direction by a small angle versus scaling magnitude by a small factor—are flawed because the actual Euclidean displacements in representation space differ. If angular perturbations cause greater damage, it is unclear whether direction is more important or if the perturbation is simply more "violent." Without controlling for displacement, all comparisons are invalid.

**Goal**: (1) Construct an L2-matched perturbation protocol to eliminate size confounding. (2) Systematically measure the causal importance of direction and magnitude on various downstream tasks in Pythia. (3) Locate the mechanistic paths of influence through pathway restoration experiments.

**Key Insight**: Borrowing the "double dissociation" tool from cognitive neuroscience—if intervention A primarily impairs task X but not Y, and intervention B primarily impairs Y but not X, it indicates that X and Y are supported by separable subsystems.

**Core Idea**: Parameterize "perturbation intensity" $\delta$ to force the Euclidean displacement of both angular and magnitude perturbations at the intervention layer to be exactly $\delta$, then compare their effects on loss and syntactic accuracy.

## Method

### Overall Architecture
This paper addresses whether direction or magnitude is more important for Transformers. The difficulty lies in the inherent incomparability of these quantities. The approach parameterizes both perturbations to the same displacement intensity $\delta$: for hidden states $\mathbf{h}$ at mid-layers (layers 8-15) of Pythia-410M, the model either scales the length (magnitude perturbation $\mathbf{h}'_{\text{mag}} = \alpha \mathbf{h}$) or rotates the direction (angular perturbation $\mathbf{h}'_{\text{ang}} = \|\mathbf{h}\| \cdot \hat{\mathbf{h}}'$), forcing the Euclidean distance to $\mathbf{h}$ to equal $\delta$. After matching the magnitude, downstream metrics like language modeling loss and syntactic accuracy are measured.

### Key Designs

**1. L2-Matched Perturbation Protocol: Projecting Incomparable Axes to $\delta$**

To prevent confounding "intensity" with "type," this paper uses analytical formulas to force $\|\mathbf{h} - \mathbf{h}'_{\text{mag}}\| = \|\mathbf{h} - \mathbf{h}'_{\text{ang}}\| = \delta$. For magnitude, $\alpha = 1 \pm \delta / \|\mathbf{h}\|$ is solved from $|1-\alpha| \cdot \|\mathbf{h}\| = \delta$, with the sign chosen randomly. For direction, an orthogonal unit vector $\mathbf{v} \perp \mathbf{h}$ is sampled to form $\mathbf{h}'_{\text{ang}} = \|\mathbf{h}\|(\cos\theta \cdot \hat{\mathbf{h}} + \sin\theta \cdot \hat{\mathbf{v}})$, where the rotation angle $\theta = \arccos(1 - \delta^2 / 2\|\mathbf{h}\|^2)$ is derived from the distance constraint. Displacement errors are empirically verified to be $< 0.01$.

**2. Cross-Over Dissociation: Capturing Separation via Complementary Tasks**

To prove separable subsystems, a pair of tasks with complementary "geometric sensitivity" is used. For the macro-scale, next-token cross-entropy on 281 WikiText-103 sentences is used, representing high-entropy global prediction sensitive to direction. For the fine-grained scale, 200 subject-verb agreement minimal pairs from BLiMP (e.g., "The dogs run" vs "The dogs runs") are used to see if the model still assigns higher probability to the grammatical sentence—a discrete decision reliant on norm-regulated numerical magnitudes.

**3. Pathway Restoration: Identifying Paths via Causal Intervention**

To establish mechanistic statements like "direction influences loss via attention," causal interventions are performed. For a perturbed state $\mathbf{h}'$, specific intermediate products of a pathway are replaced with clean versions. Attention restoration involves replaying with unperturbed attention weights, while LayerNorm restoration replaces perturbed LN outputs with clean ones. High recovery after restoring a pathway indicates it carries the primary effect of that perturbation.

### Loss & Training
No training is involved; experiments consist of pure inference-time interventions. Pythia-410M/1.4B models are run in float32 precision. For each $\delta$, orthogonal directions are sampled independently across 5 seeds to ensure statistical confidence.

## Key Experimental Results

### Main Results
Loss Impairment (Table 1, baseline loss = 4.107):

| $\delta$ | Magnitude $\Delta$loss | Angular $\Delta$loss | Ang/Mag Ratio | p |
|----------|-------------------|-------------------|----------|-----|
| 1.0 | 0.009 | 0.368 | **42.9×** | <0.001 |
| 2.0 | 0.042 | 0.983 | 23.2× | <0.001 |
| 5.0 | 0.700 | 3.757 | 5.4× | <0.001 |
| 10.0 | 3.262 | 7.061 | 2.2× | <0.001 |
| 20.0 | 5.433 | 7.750 | 1.4× | <0.001 |

Syntactic Accuracy (Table 2, baseline 89.5%):

| $\delta$ | Mag Accuracy | Ang Accuracy | Mag Drop | Ang Drop |
|----------|--------------|--------------|----------|----------|
| 5.0 | 69.1% | 87.9% | **20.4%** | 1.6% |
| 10.0 | 56.0% | 77.1% | 33.5% | 12.4% |
| 15.0 | 53.5% | 67.4% | 36.0% | 22.1% |

At $\delta = 5$, direction is 5.4 times more damaging to loss, while magnitude is 12.8 times more damaging to syntax. This cross-over constitutes a double dissociation.

### Ablation Study
Pathway Restoration (Proportion of total damage recovered):

| Restored Pathway | Ang Recovery | Mag Recovery | Bias |
|----------|--------------|--------------|------|
| Attention | **28.4%** | 15.2% | Ang → Attention |
| LayerNorm | 13.7% | **29.9%** | Mag → LayerNorm |

The pattern replicates in Pythia-1.4B. In RMSNorm architectures (lacking affine LN), the dissociation vanishes, suggesting the phenomenon is tied to LayerNorm's specific norm-handling mechanism.

Inter-layer Propagation (Table 4, $\delta = 5$):

| Layer | Ang Displacement L2 | Mag Displacement L2 | Ratio |
|-------|-------------|-------------|------|
| 8 (Start) | 5.00 | 5.00 | 1.00× |
| 15 (End) | 35.9 | 12.7 | 2.82× |
| 23 (Final) | 123.8 | 38.9 | 3.18× |

Angular perturbations are amplified 24.8×, while magnitude ones only 7.8×. LayerNorm naturally suppresses magnitude deviations while allowing directional ones to propagate.

### Key Findings
- **Direction acts via attention channels**: Since attention relies on $\text{softmax}(QK^T / \sqrt{d})$ (cosine similarity), directional shifts directly alter routing. LayerNorm re-normalizes the norm, absorbing much of the magnitude variation.
- **Syntax is a norm-sensitive task**: Discrete decisions like subject-verb agreement, which require numerical comparison, depend more on norm-regulated "processing intensity" than on attention routing.
- **Asymmetry at small $\delta$**: The directional advantage is 42.9× at low $\delta$ but drops to 1.4× at high $\delta$ as prediction reached a "floor" of random noise.
- **Architecture dependence**: The dissociation disappears in RMSNorm architectures, indicating it is a specific geometric division of labor for LayerNorm.

## Highlights & Insights
- **L2 Matching as Clean Design**: Using mathematically simple yet conceptually clear L2 matching is a methodological contribution that will likely be adopted by subsequent geometric research.
- **Cognitive Neuroscience Framework**: Introducing "double dissociation" makes the causal claims significantly stronger than one-way ablations.
- **Mechanism Localization + Counter-examples**: The argument chain "Phenomenon → Mechanism → Boundary conditions" (using RMSNorm) is highly rigorous.
- **Warning for Representation Engineering**: Steering vectors (direction) and activation scaling (magnitude) are not interchangeable; they correspond to different sub-capabilities.

## Limitations & Future Work
- **Pathway Explanation ~30%**: Attention and LN restoration combined account for less than half the damage; the remaining 70% of the path remains a black box.
- **Small Sample Size**: Only 5 seeds were used, limiting statistical power despite large observed effects.
- **Limited Syntax Tasks**: Only subject-verb agreement was tested; whether other syntax (NPI licensing, island constraints) is equally norm-sensitive is unknown.
- **Fixed Intervention Layers**: The study did not systematically scan early or late layers.
- **Random Orthogonal Directions**: Since representation space is anisotropic, random orthogonal vectors might not be "semantically neutral."

## Related Work & Insights
- **vs. Park et al. 2023 (LRH Formalization)**: That work established the direction-encoding hypothesis; this paper extends it by adding the neglected dimension of magnitude.
- **vs. Kobayashi et al. 2020 (Norm in Attention)**: They found norms modulate attention; this paper provides causal evidence that norms are uniquely critical for syntactic functions.
- **vs. Meng et al. 2022 (ROME)**: Both use causal intervention, but this work decomposes activations into direction and magnitude for finer granularity.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ L2-matched protocol and double dissociation framework are significant methodological contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Includes two models, dual tasks, pathway restoration, and architectural counter-examples; lost one star for low seed count.
- Writing Quality: ⭐⭐⭐⭐⭐ Extremely high readability with a logical "Phenomenon → Mechanism → Boundary" structure.
- Value: ⭐⭐⭐⭐ Critical for refining LRH and guiding representation engineering practices.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Why Low-Precision Transformer Training Fails: An Analysis on Flash Attention](../../ICLR2026/interpretability/why_low-precision_transformer_training_fails_an_analysis_on_flash_attention.md)
- [\[ACL 2026\] Similarity-Distance-Magnitude Activations](../../ACL2026/interpretability/similarity-distance-magnitude_activations.md)
- [\[AAAI 2026\] Probing Preference Representations: A Multi-Dimensional Evaluation and Analysis Method for Reward Models](../../AAAI2026/interpretability/probing_preference_representations_a_multi-dimensional_evaluation_and_analysis_m.md)
- [\[ICML 2026\] Prototype Transformer: Towards Language Model Architectures Interpretable by Design](prototype_transformer_towards_language_model_architectures_interpretable_by_desi.md)
- [\[ICML 2026\] Learning Coherent Representations: A Topological Approach to Interpretability](learning_coherent_representations_a_topological_approach_to_interpretability.md)

</div>

<!-- RELATED:END -->

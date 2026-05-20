---
title: >-
  [Paper Note] Disentangling Direction and Magnitude in Transformer Representations: A Double Dissociation Through L2-Matched Perturbation Analysis
description: >-
  [ICML 2026][Interpretability][Linear Representation Hypothesis] This paper uses an L2-matched perturbation protocol to demonstrate that, in the Pythia series…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "Linear Representation Hypothesis"
  - "Direction vs Magnitude"
  - "L2-matched Perturbation"
  - "LayerNorm"
  - "Attention Pathway"
date: 2026-05-08
content_hash: bad20dcf4fd8233b
---

# Disentangling Direction and Magnitude in Transformer Representations: A Double Dissociation Through L2-Matched Perturbation Analysis

**Conference**: ICML 2026  
**arXiv**: [2602.11169](https://arxiv.org/abs/2602.11169)  
**Code**: Not released  
**Area**: Interpretability / Representation Geometry / Causal Intervention  
**Keywords**: Linear Representation Hypothesis, Direction vs Magnitude, L2-matched Perturbation, LayerNorm, Attention Pathway

## TL;DR
This paper uses an L2-matched perturbation protocol to demonstrate that, in the Pythia series, direction (angle) perturbations are 42.9 times more destructive to language modeling loss than magnitude perturbations of the same displacement, while magnitude perturbations are far more damaging to syntax (subject-verb agreement) than angle—constituting a "double dissociation" in the cognitive neuroscience sense, with direction effects propagating via the attention pathway and magnitude via the LayerNorm pathway.

## Background & Motivation
**Background**: The Linear Representation Hypothesis (LRH) is foundational in current interpretability research—encoding concepts as directions in activation space and extracting semantic features with linear probes. Methods like activation patching, TunedLens, and representation engineering all rest on the assumption that "direction matters."

**Limitations of Prior Work**: LRH is silent by default on magnitude (norm)—yet norm is not constant in transformers: Kobayashi et al. found it varies significantly across tokens and layers; LayerNorm explicitly manipulates norm; representation engineering modifies behavior by scaling vectors. No one has systematically compared the causal importance of direction and magnitude.

**Key Challenge**: Naive comparisons—perturbing direction by a small angle or scaling magnitude by a small factor—result in different actual displacements in representation space. If angle perturbations are more damaging, is it because direction is more important, or because the perturbation is "more violent"? Without controlling for displacement, all comparisons are invalid.

**Goal**: (1) Construct an L2-matched perturbation protocol to eliminate confounding by displacement size; (2) Systematically measure the causal importance of direction and magnitude for different downstream tasks on Pythia; (3) Use pathway repair experiments to localize the mechanism of effect propagation.

**Key Insight**: Borrowing the "double dissociation" tool from cognitive neuroscience—if operation A mainly impairs task X but not Y, and operation B mainly impairs Y but not X, then X and Y are supported by separable subsystems.

**Core Idea**: Use $\delta$ to parameterize "perturbation strength," forcing both angle and magnitude perturbations to have exactly $\delta$ Euclidean displacement at the intervention layer, then compare their effects on loss/syntactic accuracy.

## Method

### Overall Architecture
For mid-layers (layers 8-15) of Pythia-410M, apply one of two perturbations to each token's hidden state $\mathbf{h}$:

1. **Magnitude perturbation**: $\mathbf{h}'_{\text{mag}} = \alpha \mathbf{h}$, direction unchanged, length changes.
2. **Angle perturbation**: $\mathbf{h}'_{\text{ang}} = \|\mathbf{h}\| \cdot \hat{\mathbf{h}}'$, length unchanged, direction rotated by $\theta$.

Analytical formulas ensure both perturbations satisfy $\|\mathbf{h} - \mathbf{h}'_{\text{mag}}\| = \|\mathbf{h} - \mathbf{h}'_{\text{ang}}\| = \delta$, then measure downstream (a) WikiText cross-entropy loss; (b) BLiMP subject-verb agreement accuracy; (c) recovery rate after attention/LayerNorm pathway repair.

### Key Designs

1. **L2-matched Perturbation Formula**:

    - **Function**: Eliminates confounding from perturbation size when comparing direction/magnitude.
    - **Mechanism**: For magnitude, solve $|1-\alpha| \cdot \|\mathbf{h}\| = \delta$ to get $\alpha = 1 \pm \delta / \|\mathbf{h}\|$, with sign randomly chosen (half scaling up, half down), requiring $\delta < \|\mathbf{h}\|$. For angle, sample an orthogonal unit vector $\mathbf{v} \perp \mathbf{h}$, write $\mathbf{h}'_{\text{ang}} = \|\mathbf{h}\|(\cos\theta \cdot \hat{\mathbf{h}} + \sin\theta \cdot \hat{\mathbf{v}})$, and from $\|\mathbf{h} - \mathbf{h}'_{\text{ang}}\| = \delta$ derive $\theta = \arccos(1 - \delta^2 / 2\|\mathbf{h}\|^2)$. Empirically, post-perturbation displacement error $<$ 0.01.
    - **Design Motivation**: Projects "direction vs magnitude" onto a unified $\delta$ axis, making all causal effect differences attributable solely to perturbation "type"—the methodological foundation of the paper.

2. **Cross-over Dissociation Measurement**:

    - **Function**: Measures the impact of both perturbations on "macroscopic loss" and "fine-grained syntax."
    - **Mechanism**: For macroscopic, use next-token cross-entropy on 281 WikiText-103 sentences (10-64 tokens each); for fine-grained, use BLiMP's 200 minimal pairs for irregular/regular plural subject-verb agreement (e.g., "The dogs run" vs "The dogs runs"), checking if the model still assigns higher probability to grammatical sentences. Six $\delta$ levels: $\{1, 2, 5, 10, 15, 20\}$. Five random seeds, pair t-test + Bonferroni correction.
    - **Design Motivation**: These two tasks are complementary in "information density" and "sensitivity to geometric properties"—next-token prediction is high-entropy and globally direction-sensitive; subject-verb agreement is a low-dimensional discrete decision, more sensitive to norm as a control of "processing strength."

3. **Attention / LayerNorm Pathway Repair**:

    - **Function**: Localizes which computational pathway mediates the perturbation's effect.
    - **Mechanism**: For perturbed state $\mathbf{h}'$, replace the intermediate product (attention pattern or LayerNorm output) on a single pathway with the clean version, and observe how much downstream loss is recovered. If repairing a pathway yields high recovery, that pathway carries the main effect. Specifically, attention repair = replay attention weights from the unperturbed forward pass; LayerNorm repair = substitute the unperturbed LN output for the perturbed version.
    - **Design Motivation**: Correlational observations only show "direction matters," but to establish "direction affects loss via attention," causal intervention is needed—repairing a pathway and observing effect disappearance infers the causal route.

### Loss & Training
No training is performed; all interventions are at inference time. Pythia-410M/1.4B are run in float32 precision; for each $\delta$, five seeds independently sample orthogonal directions to estimate confidence.

## Key Experimental Results

### Main Results
Loss damage (Table 1, baseline loss = 4.107):

| $\delta$ | Magnitude $\Delta$loss | Angle $\Delta$loss | Angle/Mag Ratio | p |
|----------|-----------------------|--------------------|-----------------|-----|
| 1.0 | 0.009 | 0.368 | **42.9×** | <0.001 |
| 2.0 | 0.042 | 0.983 | 23.2× | <0.001 |
| 5.0 | 0.700 | 3.757 | 5.4× | <0.001 |
| 10.0 | 3.262 | 7.061 | 2.2× | <0.001 |
| 20.0 | 5.433 | 7.750 | 1.4× | <0.001 |

Syntactic accuracy (Table 2, baseline 89.5%):

| $\delta$ | Accuracy after Mag | Accuracy after Angle | Mag Drop | Angle Drop |
|----------|-------------------|---------------------|----------|------------|
| 5.0 | 69.1% | 87.9% | **20.4%** | 1.6% |
| 10.0 | 56.0% | 77.1% | 33.5% | 12.4% |
| 15.0 | 53.5% | 67.4% | 36.0% | 22.1% |

At $\delta = 5$, loss difference is 5.4× in favor of direction, syntactic difference is 12.8× in favor of magnitude—these opposing advantages constitute a double dissociation.

### Ablation Study
Pathway repair (proportion of total damage recovered):

| Repair Pathway | Angle Perturbation Recovery | Magnitude Perturbation Recovery | Bias |
|---------------|----------------------------|-------------------------------|------|
| Attention | **28.4%** | 15.2% | Angle→attention |
| LayerNorm | 13.7% | **29.9%** | Magnitude→LayerNorm |

This pattern replicates on Pythia-1.4B (angle/magnitude ratio rises from 23.2× at 410M to 56.8×). On RMSNorm architectures (no affine LN), the dissociation disappears, indicating the phenomenon is tightly coupled to LayerNorm's norm operation.

Inter-layer propagation (Table 4, $\delta = 5$):

| Layer | Angle L2 Displacement | Magnitude L2 Displacement | Ratio |
|-------|----------------------|--------------------------|-------|
| 8 (intervention start) | 5.00 | 5.00 | 1.00× |
| 15 (intervention end) | 35.9 | 12.7 | 2.82× |
| 23 (final) | 123.8 | 38.9 | 3.18× |

Angle perturbation amplifies 24.8×, magnitude only 7.8×—LayerNorm naturally suppresses magnitude, but lets direction propagate freely.

### Key Findings
- **Direction acts via attention**: Since attention is essentially $\text{softmax}(QK^T / \sqrt{d})$, relying on cosine similarity, direction perturbations directly alter routing; LayerNorm re-normalizes norm, absorbing magnitude changes.
- **Syntax is norm-sensitive**: Subject-verb agreement and similar tasks requiring fine-grained numerical comparison depend more on norm to regulate "processing strength" than on attention routing.
- **Small $\delta$ is highly asymmetric, large $\delta$ saturates**: At low $\delta$, angle advantage is 6.80×, dropping to 1.69× at high $\delta$, as model predictions have a "floor"—further perturbation only degrades to random.
- **Architecture dependence**: The dissociation disappears on RMSNorm, indicating this geometric division of labor is unique to LayerNorm, not a universal transformer property.

## Highlights & Insights
- **L2-matching = clean experimental design**: Achieves conceptual clarity and mathematical simplicity, a paradigm-level contribution for geometric causal studies, likely to be widely adopted.
- **Borrowing cognitive neuroscience terminology**: Introducing "double dissociation" as a mature causal inference framework strengthens interpretability arguments, far beyond unidirectional ablation.
- **Mechanism localization + architectural counterexample**: Establishes the phenomenon (double dissociation), localizes the mechanism (attention/LN pathways), then uses RMSNorm to test dependency—a rigorous "phenomenon → mechanism → boundary" argument chain.
- **Implicit warning for representation engineering**: Direction editing (steering vectors) and magnitude scaling (activation scaling) are not interchangeable, corresponding to different sub-capabilities.

## Limitations & Future Work
- **Pathway explains only ~30%**: Attention/LN repair together recover less than half the damage; the remaining 70% of effect pathways remain a black box—the paper acknowledges the "mechanistic picture is incomplete."
- **Only 5 seeds**: Authors admit limited statistical power, though effects are large.
- **Only tested subject-verb agreement**: BLiMP contains many other syntactic phenomena (NPI licensing, island constraints, etc.); whether these are also magnitude-sensitive is unknown.
- **Intervention layers fixed at 8-15**: Early/final layers not systematically scanned; dissociation may depend on processing stage.
- **Orthogonal perturbation directions are random**: But representation space is anisotropic (Ethayarajh 2019), so "random orthogonal" is not necessarily "semantically neutral"—some perturbations may hit key subspaces.
- **Future extensions**: Suggests testing RMSNorm + sandwich norm + different positional encoding combinations to map the dissociation phenomenon to specific norm forms.

## Related Work & Insights
- **vs Park et al. 2023 (LRH formalization)**: That work defined and empirically validated the direction-encoding assumption of LRH; this paper refines LRH by adding the previously ignored magnitude dimension.
- **vs Kobayashi et al. 2020 (norm in attention)**: They found norm modulates attention weights; this paper uses causal intervention to show norm is especially important for syntactic function.
- **vs Meng et al. 2022 (activation patching ROME)**: Also in the causal intervention family, but ROME patches the entire activation, while this paper decomposes into direction/magnitude before patching, achieving finer granularity.
- **Insights**: In model editing practice, direction steering (affecting attention routing) and magnitude scaling (affecting processing strength) should be distinguished, not conflated; safety research can test whether "jailbreak prompts" mainly perturb direction or magnitude—a new dimension for analysis.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ L2-matched perturbation protocol + interdisciplinary double dissociation—rarely seen methodological contribution in interpretability.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two Pythia models + dual tasks + dual pathway repair + RMSNorm counterexample; point deducted for only 5 seeds and relatively small model scale.
- Writing Quality: ⭐⭐⭐⭐⭐ Argument chain "phenomenon → mechanism → boundary" is well-structured, formula derivations are clear, counterexamples and confidence intervals are thoroughly discussed, readability is excellent.
- Value: ⭐⭐⭐⭐ Fine-grained extension of LRH and practical guidance for representation engineering are both important, but practical threshold is high (requires causal intervention infrastructure).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Noise Stability of Transformer Models](../../ICLR2026/interpretability/noise_stability_of_transformer_models.md)
- [\[ICLR 2026\] Closing the Curvature Gap: Full Transformer Hessians and Their Implications for Scaling Laws](../../ICLR2026/interpretability/closing_the_curvature_gap_full_transformer_hessians_and_their_implications_for_s.md)
- [\[NeurIPS 2025\] Why Is Attention Sparse in Particle Transformer?](../../NeurIPS2025/interpretability/why_is_attention_sparse_in_particle_transformer.md)
- [\[AAAI 2026\] FlashKAT: Understanding and Addressing Performance Bottlenecks in the Kolmogorov-Arnold Transformer](../../AAAI2026/interpretability/flashkat_understanding_and_addressing_performance_bottlenecks_in_the_kolmogorov-.md)
- [\[NeurIPS 2025\] Beyond Components: Singular Vector-Based Interpretability of Transformer Circuits](../../NeurIPS2025/interpretability/beyond_components_singular_vector-based_interpretability_of_transformer_circuits.md)

</div>

<!-- RELATED:END -->

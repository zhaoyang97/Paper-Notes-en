---
title: >-
  [Paper Note] Bigram Subnetworks: Mapping to Next Tokens in Transformer Language Models
description: >-
  [NeurIPS 2025][Interpretability][Bigram Subnetworks] Using continuous sparsification, the authors identify bigram subnetworks containing only ~10M parameters within Transformer language models. These subnetworks are conc…
tags:
  - "NeurIPS 2025"
  - "Interpretability"
  - "Bigram Subnetworks"
  - "Mechanistic Interpretability"
  - "Continuous Sparsification"
  - "Residual Stream"
  - "Minimal Circuit"
date: 2026-05-08
content_hash: 719cfa13e33fb486
---

# Bigram Subnetworks: Mapping to Next Tokens in Transformer Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2504.15471](https://arxiv.org/abs/2504.15471)  
**Code**: [https://github.com/tylerachang/bigram-subnetworks](https://github.com/tylerachang/bigram-subnetworks)  
**Area**: Interpretability
**Keywords**: Bigram Subnetworks, Mechanistic Interpretability, Continuous Sparsification, Residual Stream, Minimal Circuit

## TL;DR
Using continuous sparsification, the authors identify bigram subnetworks containing only ~10M parameters within Transformer language models. These subnetworks are concentrated in the first MLP layer, suffice to reproduce bigram predictions ($r>0.95$), and cause dramatic performance degradation when ablated — demonstrating that they constitute minimal next-token prediction circuits that are both necessary and sufficient in language models.

## Background & Motivation
**Background**: Mechanistic interpretability research has uncovered specific circuits such as induction heads and name mover heads, but these circuits typically cover only particular behaviors. A "minimal base circuit" defined over the entire input space — serving as a foundation for studying more complex circuits — has been lacking.

**Limitations of Prior Work**: Circuit studies generally verify only **necessity** (behavior disappears after ablation) but not **sufficiency** (whether the circuit alone can sustain the behavior). Verifying sufficiency requires composing the target circuit on top of some already-understood minimal circuit, yet what that minimal circuit should be has remained unclear.

**Key Challenge**: It is known that Transformers overfit bigram distributions in the early stages of pretraining, but it is unclear whether bigram information remains encoded in model parameters even after the model later diverges from bigram predictions — and if so, in what form.

**Key Insight**: Bigram prediction $P(w_i|w_{i-1})$ is the simplest non-trivial next-token prediction, defined over the entire input space. A subnetwork that implements bigram prediction would serve as an ideal foundation for studying more complex circuits.

**Core Idea**: Continuous sparsification is applied to frozen LLMs to search for parameter masks, yielding subnetworks that occupy only 0.17% of parameters yet achieve a bigram correlation of $r=0.96$, concentrated primarily in the first MLP layer.

## Method

### Overall Architecture
Freeze LLM parameters → Learn parameter masks $M$ via continuous sparsification → Minimize cross-entropy between masked model output and the bigram distribution plus an L1 sparsity penalty → Obtain subnetworks defined by binarized masks. Experiments are conducted on Pythia (70M–1B) and GPT-2 (small–large).

### Key Designs

1. **Subnetwork Discovery via Continuous Sparsification**:

    - Each model parameter is associated with a learnable mask value $m \in (-\infty, +\infty)$, mapped to $(0,1)$ via sigmoid.
    - During training, the sigmoid temperature is gradually reduced to push masks toward binary values.
    - Loss: $\text{CE}(P(x), \text{MaskedModel}_M(x)) + \lambda \|M\|_1/|M|$
    - $\lambda$ controls sparsity, varied from 0 to 1000 to observe behavior across different sparsity levels.

2. **Key Finding: Universal ~10M Parameter Count**:

    - Regardless of model size (70M to 1B), bigram subnetworks reach a performance plateau at ~10M active parameters.
    - In Pythia 1B, only 0.17% of non-embedding parameters suffice to achieve a bigram correlation of $r=0.959$.
    - This indicates that the "circuit capacity" required for bigram prediction is independent of model scale.

3. **Structural Analysis: Dominance of the First MLP Layer**:

    - Across all models and pretraining checkpoints, the majority of bigram subnetwork parameters are concentrated in the first Transformer MLP layer.
    - This holds even for randomly initialized models, suggesting an inherent bias of the architecture and loss function.
    - Mechanistic interpretation: the first MLP layer is responsible for rotating activations from the "current token representation space" into the "next-token prediction space."

### Residual Stream Rotation Analysis
- In the full model, token prediction accuracy rises sharply after the first layer — a jump from the current-token space to the next-token space.
- The bigram subnetwork precisely reproduces this jump, with minimal change in subsequent layers.
- This indicates that the bigram subnetwork captures the most fundamental mechanism by which Transformers perform next-token prediction.

## Key Experimental Results

### Main Results (Bigram Correlation $r$)

| Model | Subnetwork Parameter Fraction | Bigram $r$ | Full Model $r$ |
|-------|------------------------------|-----------|---------------|
| Pythia 70M | ~15% | 0.961 | 0.737 |
| Pythia 410M | ~2.5% | 0.983 | 0.650 |
| Pythia 1B | 0.17% | 0.959 | 0.632 |
| GPT-2 medium | ~1% | 0.985 | 0.582 |
| GPT-2 large | ~1% | 0.986 | 0.583 |

### Ablation Study

| Operation | Change in Model Perplexity | Notes |
|-----------|--------------------------|-------|
| Ablate bigram subnetwork | Severe degradation | 0.17% of parameters are critical to performance |
| Ablate equivalent random parameters | Minor degradation | Bigram parameters far more important than random ones |
| Bigram subnetwork ∩ optimal pruning subnetwork | High overlap | Bigram parameters are also the key parameters retained by pruning |

### Key Findings
- Larger models have smaller bigram subnetwork fractions, but the absolute parameter count remains constant (~10M).
- During pretraining, bigram subnetworks first shrink then expand, reaching their most efficient representation at ~4K steps.
- The first MLP layer plays a critical role in rotating representations from the current-token to the next-token space.
- Bigram subnetworks overlap substantially with optimal pruning subnetworks, indicating that bigram prediction constitutes a "core function" of language models.

## Highlights & Insights
- The notion of a **"minimal circuit"** is highly constructive: it provides a well-defined baseline for future circuit discovery research — identify the bigram circuit first, then compose more complex circuits on top of it.
- The **universal constant of ~10M parameters** is intriguing: it suggests that the information-theoretic complexity of bigram prediction corresponds to approximately 10M parameters, independent of model scale.
- The **special role of the first MLP layer** is precisely quantified: it is not merely arbitrary initial processing, but rather performs the critical spatial rotation from "what token am I" to "what token comes next."

## Limitations & Future Work
- Evaluation is limited to the 1B scale — do bigram subnetworks in 7B/70B models still contain ~10M parameters?
- Continuous sparsification may not find globally optimal masks — different initializations may yield different subnetworks.
- Only bigrams (1-token context) are studied — can the approach generalize to trigram or 4-gram subnetworks, and how do they compose?
- No connection is drawn to known circuits such as induction heads — what is the relationship between bigram subnetworks and induction heads?

## Related Work & Insights
- **vs. Voita et al. (2024)**: They identify bigram-facilitating neurons distributed across all layers. The present work finds that bigram subnetworks are concentrated in the first layer, suggesting that bigram-related activity at different layers may serve distinct functions.
- **vs. Traditional circuit discovery (e.g., IOI)**: IOI circuits target specific tasks, whereas bigram subnetworks cover the entire input space and constitute a more fundamental circuit.

## Rating
- Novelty: ⭐⭐⭐⭐ The concept of bigram subnetworks is novel, and the ~10M universal constant is an interesting finding.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Eight models, multiple checkpoints, ablations, pruning overlap analysis, and residual stream analysis — extremely thorough.
- Writing Quality: ⭐⭐⭐⭐⭐ Arguments build progressively, figures are rich, and the narrative is clear.
- Value: ⭐⭐⭐⭐ Provides a "minimal circuit" foundation for mechanistic interpretability and inspires future work on circuit composition.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Compositional Steering of Large Language Models with Steering Tokens](../../ACL2026/interpretability/compositional_steering_of_large_language_models_with_steering_tokens.md)
- [\[NeurIPS 2025\] Beyond Components: Singular Vector-Based Interpretability of Transformer Circuits](beyond_components_singular_vector-based_interpretability_of_transformer_circuits.md)
- [\[NeurIPS 2025\] The Trilemma of Truth in Large Language Models](the_trilemma_of_truth_in_large_language_models.md)
- [\[NeurIPS 2025\] Table as a Modality for Large Language Models](table_as_a_modality_for_large_language_models.md)
- [\[NeurIPS 2025\] Discovering Transformer Circuits via a Hybrid Attribution and Pruning Framework](discovering_transformer_circuits_via_a_hybrid_attribution_and_pruning_framework.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Far from the Shallow: Brain-Predictive Reasoning Embedding through Residual Disentanglement
description: >-
  [NeurIPS 2025][Interpretability][Brain encoding] This paper proposes a residual disentanglement method that decomposes LLM hidden states into four approximately orthogonal embeddings—lexical, syntactic, semantic…
tags:
  - "NeurIPS 2025"
  - "Interpretability"
  - "Brain encoding"
  - "LLM alignment"
  - "residual disentanglement"
  - "ECoG"
  - "reasoning representations"
  - "language hierarchy"
date: 2026-05-08
content_hash: 18f7c862600a37e1
---

# Far from the Shallow: Brain-Predictive Reasoning Embedding through Residual Disentanglement

**Conference**: NeurIPS 2025
**arXiv**: [2510.22860](https://arxiv.org/abs/2510.22860)  
**Code**: Available (footnote link in paper)  
**Area**: Neuroscience / Language Encoding Models
**Keywords**: Brain encoding, LLM alignment, residual disentanglement, ECoG, reasoning representations, language hierarchy

## TL;DR

This paper proposes a residual disentanglement method that decomposes LLM hidden states into four approximately orthogonal embeddings—lexical, syntactic, semantic, and reasoning—for predicting intracranial ECoG brain signals. The study finds that reasoning signals exhibit independent neural signatures both temporally (~350–400 ms) and spatially (extending beyond classical language areas into visual cortex), revealing a computational alignment between LLM reasoning and human brain processing.

## Background & Motivation

**Limitations of brain–LLM alignment research**: Existing studies demonstrate strong alignment between LLM internal representations and human language processing, yet the vast majority focus on semantic and low-level phonological relationships, leaving the neural alignment of higher-order reasoning largely unexplored.

**Feature entanglement problem**: LLM hidden states are highly entangled, mixing lexical information, syntactic structure, semantic meaning, and reasoning processes. Conventional brain encoding analyses are inherently biased toward shallow linguistic features (lexical, syntactic), obscuring the contribution of deeper cognitive processes.

**Representational bias**: Research shows that when models are trained on tasks of varying complexity, internal representations are biased toward simpler, linearly extractable features—even when accuracy on complex tasks is equally high, shallow features continue to dominate the representation.

**Emergence of LLM reasoning**: Robust reasoning capabilities have only recently emerged in modern LLMs, making it newly feasible to study the neural basis of reasoning.

## Method

### Step 1: Minimal Pair Probing to Localize Feature Layers

Logistic regression classifiers are trained on each layer of Qwen2.5-14B (48 layers, 5120 dimensions) using diagnostic datasets. Minimal pair tasks identify the saturation layer for each linguistic feature:

$$L_x := \min\{l \mid \forall l' > l,\; \text{Acc}^{\mathcal{D}_x}(H_{l'}) - \text{Acc}^{\mathcal{D}_x}(H_l) < \varepsilon\}, \quad x \in \{s, m, r\}$$

- **Syntactic** $L_s = 6$ (BLiMP dataset)
- **Semantic** $L_m = 20$ (COMPS-BASE dataset)
- **Reasoning** $L_r = 30$ (COMPS-WUGS-DIST dataset)

### Step 2: Residual Embedding Construction

Lower-level representations are progressively regressed out along the LLM hierarchy, yielding four disentangled embeddings:

**Lexical embedding**: $E_l = H_0$ (uncontextualized representation at layer 0)

**Syntactic residual**: $E_s = H_s - g_l(H_l)$, where $g_l = \arg\min_W \|H_s - WH_l\|_F^2 + \alpha\|W\|_F^2$

**Semantic residual**: $E_m = H_m - g_s(H_s)$ (semantic layer minus syntactic projection)

**Reasoning residual**: $E_r = H_r - g_r(H_m)$ (reasoning layer minus semantic projection)

Regression models are trained on 16 podcast episodes (~160K tokens) using Ridge regression with 4-fold cross-validation.

### Step 3: Orthogonality Verification

**Matrix-level orthogonality**: Because linguistic features emerge in a sequential order (syntactic → semantic → reasoning), higher-layer representations $H_m$ already subsume information from $H_l$ and $H_s$. The residual $E_r'$ is the residual from a linear projection onto $[H_l, H_s, H_m]$, yielding:

$$\langle E_j, E_k \rangle \approx 0 \quad \forall j \neq k$$

**Token-level cosine similarity**: Before disentanglement, the cosine similarity between the semantic and reasoning layers is 0.751; after disentanglement, all pairwise cosine similarities are ≤ 0.045.

### Step 4: Brain Encoding Model

The Podcast ECoG dataset is used (9 neurosurgical patients, 1330 electrodes, high-γ band 70–200 Hz), aligned to word onset with a ±2 s window downsampled to 32 Hz (128 time bins):

$$W^* = \arg\min_W \|Y - XW\|_F^2 + \alpha\|W\|_F^2$$

Prediction quality is evaluated via Pearson correlation; a null distribution is constructed from 500 shuffles, with $z > 3.95$ (Bonferroni-corrected $\alpha = 0.05$, $N = 1268$) as the significance threshold.

## Key Experimental Results

### Neural Prediction Results

| Feature | Responsive Electrodes | Peak Correlation | Peak Latency |
|---|---|---|---|
| Lexical | Fewest (high precision but limited) | Highest *** | ~Immediate |
| Syntactic | 166 (dominant) | Second highest *** | Pre-onset |
| Semantic | 161 (dominant) | Moderate | Around onset |
| Reasoning | 128 (dominant) | Lowest but significant | ~362 ms |
| Full embedding | — | Dominated by lexical/syntactic | — |

> Welch's t-test: lexical/syntactic vs. semantic/reasoning $p < 0.001$

### Spatial Distribution

- **Lexical/syntactic**: Concentrated in classical language areas (IFG, STG)
- **Reasoning-specific**: Extended into the superior frontal gyrus (SFG) and visual cortex (occipital lobe); reasoning feature correlations in visual cortex are significantly higher than those of other features ($p < 0.001$)

### Cross-Model Validation

Validated across multiple generations of the Qwen model family (1.8B–14B); the feature emergence order is consistently preserved: syntactic earliest → semantic intermediate → reasoning deepest. The sole exception is Qwen-1.8B, where syntactic and semantic features saturate at the same layer.

## Highlights & Insights

1. ⭐⭐⭐ **First isolation of neural signatures for reasoning**: Residual disentanglement reveals reasoning-specific brain activity, with a peak at ~362 ms—over 100 ms later than other features—consistent with cognitive hierarchy theory.
2. ⭐⭐⭐ **Recruitment of visual cortex by reasoning**: Reasoning signals uniquely activate visual regions, suggesting that reasoning may involve visual imagery and cross-modal integration.
3. ⭐⭐ **Revealing the bias of full embeddings**: The brain-predictive success of standard LLM embeddings is largely attributable to shallow features, obscuring the contribution of deeper cognitive processes—an important methodological caveat for all brain–LLM alignment research.
4. ⭐⭐ **Methodological generalizability**: The disentanglement framework is extensible to the analysis of any hierarchically structured representations.

## Limitations & Future Work

1. **Data scale**: Only 9 patients' ECoG data, with electrode coverage constrained by clinical necessity rather than experimental design.
2. **Linearity assumption**: Both residual disentanglement and brain encoding rely on linear models (Ridge regression), potentially missing nonlinear mappings.
3. **Ambiguous definition of reasoning**: The paper's operationalization of "reasoning" is primarily defined by COMPS and ProntoQA, offering limited coverage and excluding broader reasoning types such as mathematical or causal reasoning.
4. **Single primary LLM**: The main experiments center on Qwen2.5-14B; while cross-model validation is provided, non-Qwen architectures are not examined.

## Rating

| Dimension | Score |
|---|---|
| Novelty | ⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐⭐ |
| Overall Recommendation | ⭐⭐⭐⭐ |

## Related Work & Insights

## Rating

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Conditional Distribution Compression via the Kernel Conditional Mean Embedding](conditional_distribution_compression_via_the_kernel_conditional_mean_embedding.md)
- [\[NeurIPS 2025\] Deep Value Benchmark: Measuring Whether Models Generalize Deep Values or Shallow Preferences](deep_value_benchmark_measuring_whether_models_generalize_deep_values_or_shallow_.md)
- [\[NeurIPS 2025\] Saying the Unsaid: Revealing the Hidden Language of Multimodal Systems Through Telephone Games](saying_the_unsaid_revealing_the_hidden_language_of_multimodal_systems_through_te.md)
- [\[NeurIPS 2025\] Tropical Attention: Neural Algorithmic Reasoning for Combinatorial Algorithms](tropical_attention_neural_algorithmic_reasoning_for_combinatorial_algorithms.md)
- [\[NeurIPS 2025\] Auditing Meta-Cognitive Hallucinations in Reasoning Large Language Models](auditing_meta-cognitive_hallucinations_in_reasoning_large_language_models.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] An Empirical Study of Mechanistic Interpretability Approaches for Factual Recall
description: >-
  [ACL 2025][Interpretability][Mechanistic Interpretability] This paper systematically compares multiple mechanistic interpretability methods (such as causal tracing, activation patching, and probing analysis) in localizing and explaining the mechanisms of factual recall in LLMs, revealing the consistencies, discrepancies, and respective application scenarios of different approaches.
tags:
  - "ACL 2025"
  - "Interpretability"
  - "Mechanistic Interpretability"
  - "Factual Recall"
  - "Causal Tracing"
  - "Activation Patching"
  - "Knowledge Localization"
date: 2026-05-08
content_hash: 26d7d4fafed1462e
---

# An Empirical Study of Mechanistic Interpretability Approaches for Factual Recall

**Conference**: ACL 2025  
**Area**: LLM/NLP (Interpretability)  
**Keywords**: Mechanistic Interpretability, Factual Recall, Causal Tracing, Activation Patching, Knowledge Localization

## TL;DR
This paper systematically compares multiple mechanistic interpretability methods (such as causal tracing, activation patching, and probing analysis) in localizing and explaining the mechanisms of factual recall in LLMs, revealing the consistencies, discrepancies, and respective application scenarios of different approaches.

## Background & Motivation

**Background**: Mechanistic interpretability aims to understand the internal information processing mechanisms of neural networks, particularly how LLMs store and recall factual knowledge. Mainstream methods include causal tracing, activation patching, linear probing, and attention analysis. These methods have been used in various papers to explain factual recall in LLMs, but they employ different experimental setups and evaluation metrics.

**Limitations of Prior Work**: Conclusions drawn from different interpretability methods are sometimes mutually contradictory—for instance, some methods suggest that MLP layers are key to factual storage, while others point to attention heads. Because each method is evaluated using different models, datasets, and evaluation metrics, it is difficult to determine whether the discrepancies arise from the methods themselves or from the experimental setups.

**Key Challenge**: Interpretability research itself lacks "interpretability"—there is no unified framework to judge which methods are more reliable and under what conditions they apply.

**Goal**: To systematically compare multiple mechanistic interpretability methods under a unified experimental environment (same model, dataset, and evaluation metrics) to clarify their consistencies and discrepancies.

**Key Insight**: Choosing factual recall as the unified test task—given prompts like "Paris is the capital of ___", analyzing how the model internally retrieves and outputs "France".

**Core Idea**: Establishing a reliability benchmark for mechanistic interpretability methods through standardized apple-to-apple comparisons.

## Method

### Overall Architecture
Five mainstream mechanistic interpretability methods are selected and systematically compared on three Transformer models of different scales (GPT-2 Small/Medium/Large or equivalent open-source models) targeting the factual recall task. A factual triplet dataset covering various relation types (e.g., capital-country, person-profession, object-material) is utilized. The findings of each method in localizing factual storage locations and identifying key components (layers, attention heads, MLPs) are evaluated under a unified standard.

### Key Designs

1. **Unified Evaluation Framework**:

    - Function: Ensure all methods are comparable under identical conditions.
    - Mechanism: Fix the model, dataset, and evaluation metrics, and implement a standardized experimental pipeline for each method. The dataset contains 2,000 factual triplets, stratified by relation type, subject frequency (high/low frequency), and answer uniqueness. The evaluation metrics are unified into: (1) localization precision—whether it can pinpoint specific layers and components; (2) localization consistency—whether localization results are consistent across different facts; (3) intervention effectiveness—whether factual recall is affected after intervening on the localized components.
    - Design Motivation: Eliminate spurious discrepancies caused by differences in experimental setups.

2. **Comparative Analysis of Methods**:

    - Function: Reveal the strengths, weaknesses, and applicable conditions of different methods.
    - Mechanism: Systematically compare five methods: (1) **Causal Tracing** (corrupted-restore)—restores activations layer-by-layer after introducing noise to the input to localize key layers; (2) **Activation Patching**—tests the causal importance of components by replacing current activations with those from a clean run; (3) **Linear Probing**—trains linear classifiers at each layer to detect the presence of factual information; (4) **Attention Attribution**—analyzes attention weight distributions to reveal information flow; (5) **Logit Lens / Tuned Lens**—directly projects intermediate representations to the vocabulary space to observe the emergence timing of the answer token. For each method, the "key components" localized are recorded, and the agreement rate between methods is computed.
    - Design Motivation: Different methods are based on different mechanistic assumptions; understanding when they agree or disagree helps assess the reliability of their conclusions.

3. **Cross-Model Generalization Analysis**:

    - Function: Validate the robustness of interpretability conclusions across different model scales.
    - Mechanism: Repeat all experiments on three models of different scales to analyze whether the locations of key components shift with model scale. The key questions investigated include: Does factual storage shift from shallow layers in small models to deeper layers in large models? Does the relative importance of MLPs vs. attention heads change with scale?
    - Design Motivation: If interpretability conclusions highly depend on model scale, their applicability to larger models needs to be re-evaluated.

### Loss & Training
This work is analytical and does not involve model training. Linear probes are trained using logistic regression, while causal tracing and activation patching are intervention experiments conducted during inference.

## Key Experimental Results

### Main Results

| Method Pairs | Key Layer Localization Agreement Rate | MLP/Attn Attribution Agreement Rate | Correlation of Intervention Effects |
|--------|-----------------|-------------------|-------------|
| Causal Tracing vs. Activation Patching | 82.5% | 76.3% | 0.84 |
| Causal Tracing vs. Linear Probing | 68.2% | 61.5% | 0.67 |
| Activation Patching vs. Logit Lens | 74.8% | 69.2% | 0.73 |
| Linear Probing vs. Attention Attribution | 55.3% | 48.7% | 0.52 |
| Consensus Area of All Methods | 47.6% | - | - |

### Cross-Model Scale Analysis

| Component Type | GPT-2 Small | GPT-2 Medium | GPT-2 Large | Trend |
|---------|------------|-------------|-------------|------|
| Key MLP Layer Location (Relative) | Layers 50-70% | Layers 55-75% | Layers 60-80% | Shifts backward with scale |
| Causal Contribution of MLP (%) | 62.3 | 58.7 | 55.2 | MLP contribution decreases |
| Causal Contribution of Attention Heads (%) | 37.7 | 41.3 | 44.8 | Attention contribution increases |
| Factual Recall Success Rate (%) | 45.2 | 63.8 | 78.1 | Larger models recall more accurately |

### Key Findings
- **Causal tracing and activation patching are most consistent**: These two intervention-based methods share an 82.5% agreement rate in locating key layers, suggesting that causal methods are more reliable than correlation-based methods (e.g., linear probing).
- **Linear probing diverges most from other methods**: High probing performance does not entail the causal importance of the layer for factual recall; probes might detect redundant storage rather than necessary storage.
- **The importance of MLPs decreases as the model size increases**: MLPs shoulder more factual storage in small models, whereas the attention mechanism shares more of this functionality in larger models.
- **Consensus is reached across all methods for about half of the facts**: This implies that the interpretability conclusions for the other half are highly dependent on the chosen method and should be interpreted with caution.

## Highlights & Insights
- For the first time, multiple mechanistic interpretability methods are compared under completely unified experimental conditions, providing a much-needed standardized benchmark for the field. This "meta-study" paradigm holds strong methodological value.
- The finding that "probing detection $\neq$ causal importance" provides vital guidance for the correct application of probing analysis.

## Limitations & Future Work
- Experiments are only conducted on the GPT-2 family; applicability to larger-scale models (e.g., 70B+) remains unknown.
- Factual recall is a relatively simple form of knowledge utilization; the interpretability of more complex reasoning (e.g., multi-step reasoning) could be entirely different.
- The mechanism of factual recall under in-context learning scenarios is not considered.
- Emerging methods like Sparse Autoencoders (SAEs) are not included in the comparison and should be integrated into the unified evaluation framework in the future.
- The comparative framework should be extended to more task types and larger models in the future.

## Related Work & Insights
- **vs. Meng et al. (ROME)**: ROME localizes and edits knowledge based on causal tracing; this paper validates the relative reliability of causal tracing.
- **vs. Geva et al.**: The conclusion that MLPs serve as "memory" layers is verified in small models, but the role of MLPs diminishes in larger models, with the attention mechanism undertaking more knowledge retrieval functions.
- **vs. Makelov et al.**: They point out that subspace activation patching may lead to "interpretability illusions"; this paper similarly finds that patching at different granularities can yield different conclusions.

## Rating
- Novelty: ⭐⭐⭐⭐ The standardized comparison framework itself is a significant methodological contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ The systematic comparison across multiple methods, models, and metrics is highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐ The complex comparative analyses are clearly presented.
- Value: ⭐⭐⭐⭐⭐ Provides a reliability benchmark for mechanistic interpretability research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Mechanistic Interpretability of Emotion Inference in Large Language Models](mechanistic_interpretability_of_emotion_inference_in_large_language_models.md)
- [\[ACL 2025\] Reasoning Circuits in Language Models: A Mechanistic Interpretation of Syllogistic Inference](reasoning_circuits_in_language_models_a_mechanistic_interpretation_of_syllogisti.md)
- [\[ICML 2025\] MIB: A Mechanistic Interpretability Benchmark](../../ICML2025/interpretability/mib_a_mechanistic_interpretability_benchmark.md)
- [\[ACL 2026\] Through a Compressed Lens: Investigating The Impact of Quantization on Factual Knowledge Recall](../../ACL2026/interpretability/through_a_compressed_lens_investigating_the_impact_of_quantization_on_factual_kn.md)
- [\[NeurIPS 2025\] nnterp: A Standardized Interface for Mechanistic Interpretability of Transformers](../../NeurIPS2025/interpretability/nnterp_a_standardized_interface_for_mechanistic_interpretability_of_transformers.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability
description: >-
  [ICML2025][LLM Safety][Sparse Autoencoder] This paper proposes SAEBench—a comprehensive benchmark containing 8 evaluation metrics that systematically evaluates the performance of Sparse Autoencoders (SAEs) in language model interpretability, revealing a severe disconnect between proxy metrics (sparsity-fidelity) and downstream task performance.
tags:
  - "ICML2025"
  - "LLM Safety"
  - "Sparse Autoencoder"
  - "benchmark"
  - "interpretability"
  - "Feature Disentanglement"
  - "mechanistic interpretability"
date: 2026-05-08
content_hash: d6d794a2b9d39950
---

# SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability

**Conference**: ICML2025  
**arXiv**: [2503.09532](https://arxiv.org/abs/2503.09532)  
**Code**: [github.com/adamkarvonen/SAEBench](https://github.com/adamkarvonen/SAEBench)  
**Area**: SAE benchmark / Interpretability evaluation  
**Keywords**: Sparse Autoencoder, benchmark, interpretability, Feature Disentanglement, mechanistic interpretability

## TL;DR

This paper proposes SAEBench—a comprehensive benchmark containing 8 evaluation metrics that systematically evaluates the performance of Sparse Autoencoders (SAEs) in language model interpretability, revealing a severe disconnect between proxy metrics (sparsity-fidelity) and downstream task performance.

## Background & Motivation

Sparse Autoencoders (SAEs) are currently one of the most popular tools in mechanistic interpretability, decomposing language model activations into sparse, interpretable feature directions via dictionary learning. In recent years, extensive work has been dedicated to improving SAE architectures (Gated, Switch), activation functions (TopK, JumpReLU), and loss functions (P-anneal, Matryoshka), but almost all improvements rely on the **sparsity-fidelity tradeoff** as the primary evaluation standard.

**Core Problem**: Does the unsupervised proxy metric of sparsity-fidelity truly reflect the actual interpretability quality of SAEs? The authors discover that:

- Improvements in proxy metrics do not reliably translate to improvements in downstream tasks.
- A single metric obscures important trade-offs between different architectures.
- The lack of a unified multi-dimensional evaluation framework hinders the progress of the field.

## Method

### SAE Foundation Architecture

A standard SAE consists of an encoder and a decoder. The forward pass and optimization objective are defined as:

$$h = \text{ReLU}(W_E x + b_E)$$

$$\hat{x} = W_D h + b_D$$

$$\mathcal{L} = \underbrace{\|x - \hat{x}\|_2^2}_{\text{reconstruction}} + \lambda \underbrace{\|h\|_1}_{\text{sparsity}}$$

where $x$ is the input activation, $h$ is the sparse hidden representation, $\hat{x}$ is the reconstructed activation, and $\lambda$ is the sparsity coefficient.

### SAEBench Evaluation Framework

SAEBench is organized around four fundamental capability dimensions and contains 8 metrics:

**1. Concept Detection**

- **k-Sparse Probing**: For each concept, the $k \in \{1,2,5\}$ most relevant latents are selected to train linear probes, evaluating whether the SAE isolates the predefined concepts.
- **Feature Absorption**: Detects the undesirable merging of features in hierarchical concepts (e.g., "pig → mammal") caused by sparsity incentives.

**2. Interpretability**

- **Auto-Interpretability**: Employs an LLM as a judge to generate feature descriptions for each latent, then predicts which sequences will activate the latent, using prediction accuracy as the score.

**3. Reconstruction**

- **Loss Recovered**: Measures the fidelity of the SAE reconstruction to the model's original behavior, defined as:

$$\text{LR} = \frac{H^* - H_0}{H_{\text{orig}} - H_0}$$

where $H_{\text{orig}}$ is the original cross-entropy loss, $H^*$ is the loss after SAE reconstruction, and $H_0$ is the loss after zero ablation.

**4. Feature Disentanglement**

- **RAVEL**: Tests whether intervening in SAE latents can selectively modify the model's prediction of a specific attribute without affecting other attributes (e.g., making the model believe Paris is in Japan while preserving the knowledge that French is spoken there).
- **Unlearning**: Selectively deletes specific knowledge domains in the model using conditional negative steering.
- **Spurious Correlation Removal (SCR)**: Zero-ablates a small number of SAE latents to remove spurious correlations in biased linear probes.
- **Targeted Probe Perturbation (TPP)**: Evaluates whether ablating the latents of one concept category only affects the accuracy of probes for that category without disrupting others.

### Evaluated SAE Architectures

| Architecture | Characteristics |
|------|------|
| ReLU | Classic baseline architecture |
| TopK | Activates a fixed number of top K latents |
| BatchTopK | Batch-level TopK |
| Gated | Gating mechanism |
| JumpReLU | Jump ReLU |
| P-Annealing | Sparsity penalty annealing |
| **Matryoshka BatchTopK** | **Hierarchical multi-scale design** |

A total of **200+ SAEs** were trained, covering dictionary widths of 4k/16k/65k latents and sparsity $L_0 \in [20, 1000]$, evaluated on Gemma-2-2B (Layer 12) and Pythia-160M (Layer 8).

## Key Experimental Results

### Architecture Comparison (65k width, Gemma-2-2B, $L_0 \in [40,200]$)

| Metric | Best Architecture | Key Findings |
|------|---------|-------------|
| Loss Recovered | BatchTopK/TopK | ReLU is the worst |
| Sparse Probing | Matryoshka | Best concept detection |
| Feature Absorption | **Matryoshka** | Least absorption phenomenon |
| SCR | **Matryoshka** | Best spurious correlation removal |
| TPP | **Matryoshka** | Best feature isolation |
| RAVEL | **Matryoshka** | Strongest disentanglement capability |
| Autointerp | Similar across architectures | Limited differentiability |
| Unlearning | ReLU comparable | Little difference |

**Core Conclusion: Matryoshka SAE achieves the best performance in 5 out of 8 metrics, despite its sparsity-fidelity Pareto frontier being inferior to TopK/BatchTopK.**

### Dictionary Width Scaling (4k → 16k → 65k)

| Scaling Trend | Loss Recovered | Autointerp | Absorption | SCR |
|----------|---------------|------------|------------|-----|
| Most Architectures | ↑ Improves | ↑ Improves | ↓ Worsened | ↓ Worsened |
| **Matryoshka** | ↑ Improves | ↑ Improves | ≈ Stable | **↑ Improves** |

Matryoshka is the **only** architecture that scales positively with size on feature disentanglement metrics, which is attributed to its hierarchical design preventing excessive feature splitting.

### Optimal Sparsity

- Low $L_0$ (high sparsity): Benefits human interpretability.
- High $L_0$: Better reconstruction fidelity, RAVEL, and TPP scores.
- **Medium $L_0 \in [50, 150]$**: Best compromise across metrics.

## Highlights & Insights

1. **Proxy Metrics $\neq$ Practical Performance**: The ranking on the sparsity-fidelity frontier is highly inconsistent with downstream task performance, serving as an important warning to the current SAE development paradigm.
2. **Matryoshka's Comeback**: Despite mediocre performance on traditional proxy metrics, it comprehensively leads in concept detection and feature disentanglement tasks, and is the only architecture that scales positively.
3. **Inverse Scaling**: Except for Matryoshka, all architectures exhibit a decrease in feature disentanglement performance when dictionary width is increased, which might be related to feature splitting.
4. **Practical Design Principles**: $L_0 \in [50, 150]$ is the optimal compromise range across tasks.
5. **Infrastructure Contribution**: Open-sourcing 200+ SAE models + a unified evaluation framework + interactive visualization on neuronpedia.org.

## Limitations & Future Work

1. **Supervised metrics are limited by annotated data**: They can only evaluate a small number of concepts with ground truth, resulting in narrow coverage.
2. **Quantitative metrics struggle to capture qualitative interpretability**: Automated metrics cannot fully substitute the observational value of in-depth human analysis.
3. **Limited model coverage**: Verified only on Gemma-2-2B and Pythia-160M, leaving transferability across different model sizes/architectures unknown.
4. **Non-aggregatable metrics**: Different metrics have varying scales and noise levels, making it impossible to synthesize them into a single score.
5. **Unlearning evaluation limited by model capabilities**: Gemma-2-2B only achieved sufficient baseline performance on a single unlearning test set.
6. **No multimodal coverage**: Currently applicable only to text; SAE evaluations for modalities such as vision or biology remain to be explored.

## Related Work & Insights

- **Monosemanticity Series** (Bricken et al., 2023): Cornerstone work of SAE interpretability.
- **Gated SAE / TopK SAE**: Representatives of architectural improvements.
- **Matryoshka SAE** (Bussmann et al., 2024b): A key innovation in hierarchical design.
- **Feature Absorption** (Chanin et al., 2024): Uncovering feature absorption issues caused by sparsity.
- **RAVEL** (Huang et al., 2024): Attribute-value disentanglement evaluation method.

The significance of this work to the SAE field is analogous to GLUE/SuperGLUE for NLP—providing a standardized evaluation platform to drive meaningful progress.

## Rating

- Novelty: ⭐⭐⭐⭐ (First multi-dimensional comprehensive SAE benchmark, including two new metrics, SCR/TPP)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (200+ SAEs, 7 architectures, 3 widths, various sparsities, 2 models)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, deep analysis, rich figures and tables)
- Value: ⭐⭐⭐⭐⭐ (Fills a gap in the field, reveals the misleading nature of proxy metrics, with highly practically significant findings on Matryoshka)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SAUCE: Selective Concept Unlearning in Vision-Language Models with Sparse Autoencoders](../../ICCV2025/llm_safety/sauce_selective_concept_unlearning_in_vision-language_models_with_sparse_autoenc.md)
- [\[NeurIPS 2025\] SAEMark: Steering Personalized Multilingual LLM Watermarks with Sparse Autoencoders](../../NeurIPS2025/llm_safety/saemark_steering_personalized_multilingual_llm_watermarks_with_sparse_autoencode.md)
- [\[ACL 2026\] CRISP: Persistent Concept Unlearning via Sparse Autoencoders](../../ACL2026/llm_safety/crisp_persistent_concept_unlearning_via_sparse_autoencoders.md)
- [\[NeurIPS 2025\] CPRet: A Dataset, Benchmark, and Model for Retrieval in Competitive Programming](../../NeurIPS2025/llm_safety/cpret_a_dataset_benchmark_and_model_for_retrieval_in_competitive_programming.md)
- [\[ACL 2025\] ELBA-Bench: An Efficient Learning Backdoor Attacks Benchmark for Large Language Models](../../ACL2025/llm_safety/elba-bench_an_efficient_learning_backdoor_attacks_benchmark_for_large_language_m.md)

</div>

<!-- RELATED:END -->

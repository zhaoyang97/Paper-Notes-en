---
title: >-
  [Paper Note] Transformer Key-Value Memories Are Nearly as Interpretable as Sparse Autoencoders
description: >-
  [NeurIPS 2025][Interpretability][Sparse Autoencoders] This paper systematically compares the interpretability of features derived from Transformer feed-forward (FF) layer key-value memories with those learned by sparse a…
tags:
  - "NeurIPS 2025"
  - "Interpretability"
  - "Sparse Autoencoders"
  - "Feed-Forward Layers"
  - "Key-Value Memories"
  - "Feature Discovery"
date: 2026-05-08
content_hash: 0299c78830e19d91
---

# Transformer Key-Value Memories Are Nearly as Interpretable as Sparse Autoencoders

**Conference**: NeurIPS 2025
**arXiv**: [2510.22332](https://arxiv.org/abs/2510.22332)  
**Code**: [https://muyo8692.com/projects/ff-kv-sae](https://muyo8692.com/projects/ff-kv-sae)  
**Area**: Model Compression / Interpretability
**Keywords**: Sparse Autoencoders, Feed-Forward Layers, Key-Value Memories, Interpretability, Feature Discovery

## TL;DR

This paper systematically compares the interpretability of features derived from Transformer feed-forward (FF) layer key-value memories with those learned by sparse autoencoders (SAEs), finding the two approaches perform comparably on existing evaluation metrics—with FF-KV outperforming SAEs on certain dimensions—thereby questioning the necessity of SAEs as a feature discovery tool.

## Background & Motivation

Interpretability research on large language models (LLMs) has recently undergone a paradigm shift from top-down analysis toward bottom-up feature discovery. Two concurrent trends have emerged in this era: (1) training external proxy modules (e.g., SAEs) to decompose neuron activations, and (2) developing comprehensive interpretability benchmarks (e.g., SAEBench) to evaluate feature quality.

A critical yet overlooked question is: **do features learned by proxy modules actually surpass those already present in the model's original parameters?** The FF layer itself can be viewed as key-value memories—each row of the Key matrix $\mathbf{W}_K$ is a "key," and the corresponding row of the Value matrix $\mathbf{W}_V$ is the associated "value" (feature vector). The FF layer naturally decomposes activations into a set of feature vectors, a structure that is architecturally identical to that of SAEs (both are MLP-based).

Proxy approaches and FF-KV analysis each offer complementary advantages. SAEs have theoretical motivation for handling superposition, but they also introduce additional biases—certain features may be repeatedly discovered, proxy modules may "hallucinate" non-existent features, and extra computational cost is incurred. Moreover, FF activations are already naturally sparse. **If FF-KV and SAE analyses yield comparable results, Occam's Razor favors directly analyzing FF-KV.**

## Method

### Overall Architecture

The key-value structure of the FF layer is used directly as a feature discovery method. Modern interpretability benchmarks (SAEBench) and human evaluation are then employed to systematically compare the interpretability of FF-KV features against those of SAEs and Transcoders. A fidelity analysis further examines the degree of overlap between features discovered by proxy modules and those of the original FF module—assessing whether proxy modules genuinely "translate" the original module's computations or instead "hallucinate" new features.

### Key Designs

1. **FF-KV Method Family**:

    - **Vanilla FF-KV**: Directly uses the FF layer activations $\phi(\mathbf{x}_{FF_{in}}\mathbf{W}_K + \mathbf{b}_K)$ as feature activations, with rows of $\mathbf{W}_V$ as feature vectors.
    - **TopK FF-KV**: Applies Top-$k$ sparsification to FF activations, retaining only the $k$ largest values to align with SAE sparsity.
    - **Normalized FF-KV**: L2-normalizes each row of $\mathbf{W}_V$, transferring the absorbed norm as a weight on the activation to avoid bias introduced by varying feature vector norms.
    - **SwiGLU Compatibility**: All variants above naturally extend to modern LMs that adopt SwiGLU gated activations.

2. **SAEBench Evaluation Framework**: Eight complementary metrics are used for comprehensive assessment:

    - **Feature Alive Rate**: Proportion of active features.
    - **Explained Variance**: Reconstruction quality (FF-KV achieves a perfect score by construction).
    - **Absorption Score**: Degree to which concepts are over-fragmented (lower is better).
    - **Sparse Probing**: Discriminability and generalization of features.
    - **Auto-Interpretation**: Whether an LLM can summarize a feature's activation pattern in natural language.
    - **SCR/TPP**: Ability to disentangle spuriously correlated features.
    - **RAVEL**: Separability and controllability of different attributes of the same entity.

3. **Fidelity Analysis**: Transcoders (TC) are used as the analysis target (TC being the closest proxy counterpart to FF-KV) to examine the overlap between TC features and original FF features—i.e., whether the proxy module genuinely reflects the original module's behavior or fabricates novel features.

### Loss & Training

FF-KV methods require **no training whatsoever** (they directly use the model's existing parameters), which is a key advantage over SAEs. SAEs are trained with a reconstruction loss plus sparsity regularization, requiring additional computational resources. Evaluation uses pretrained SAEs (Gemma Scope, Llama Scope, etc.).

## Key Experimental Results

### Main Results (SAEBench Evaluation, Gemma-2-2B Layer 13)

| Method | Absorption↓ | Sparse Prob.↑ | AutoInterp↑ | RAVEL-ISO↑ | SCR(k=20)↑ |
|---|---|---|---|---|---|
| SAE | 0.087 | **0.846** | **0.782** | **0.985** | **0.170** |
| Transcoder | 0.025 | 0.854 | 0.790 | 0.940 | 0.104 |
| FF-KV | **0.000** | 0.827 | 0.710 | 0.952 | 0.041 |
| TopK FF-KV | 0.000 | 0.768 | 0.772 | 0.943 | 0.045 |
| Random Transformer | 0.007 | 0.798 | 0.679 | - | 0.004 |

### Ablation Study (Human Evaluation, 50 Features per Method)

| Method | Surface Features | Conceptual Features | Uninterpretable | Source Attribution Accuracy |
|---|---|---|---|---|
| FF-KV | 6 | 8 | **36** | **0.86** |
| TopK FF-KV | 9 | 9 | 32 | 0.28 |
| SAE | 6 | **9** | 35 | 0.13 |
| Transcoder | **16** | **11** | 23 | 0.18 |

### Key Findings

- **Overall Comparability**: SAEs and FF-KV yield scores in similar ranges across the 8 SAEBench metrics; absolute differences are typically far smaller than variance across seeds or layers.
- **FF-KV Advantage — Absorption**: FF-KV's Absorption score is nearly zero (far superior to SAE's 0.087), indicating that FF-KV features do not over-fragment simple concepts—feature redundancy is lower.
- **Marginal SAE Advantage — AutoInterp/SCR**: SAEs perform slightly better on auto-interpretation and spurious correlation disentanglement, though the gap is small.
- **Comparable Conceptual Feature Counts**: Human evaluation reveals nearly identical numbers of concept-level features between FF-KV and SAE (8 vs. 9).
- **Fidelity Concerns**: Most Transcoder features have no identifiable counterpart in the original FF module, suggesting proxy modules may hallucinate new features rather than translate the original module's behavior.
- **Random Transformer Baseline**: A randomly initialized Transformer also achieves non-trivial interpretability scores, further casting doubt on the feature quality attributable to proxy methods.

## Highlights & Insights

- This work raises an important challenge to the SAE-dominated interpretability paradigm—a zero-training-cost baseline achieves comparable performance.
- The Absorption score comparison is particularly compelling: SAE's tendency to over-fragment features is an inherent limitation, whereas FF-KV naturally avoids this problem.
- The fidelity analysis exposes the risk of proxy modules hallucinating features, echoing existing critiques (e.g., SAEs can explain randomly initialized Transformers).
- The core message is concise and forceful: FF-KV should serve as a strong baseline for interpretability research.

## Limitations & Future Work

- SAEs theoretically handle superposition, but current evaluation metrics may not adequately capture this advantage.
- SCR/TPP metrics are unstable and results should be treated as supplementary evidence.
- Human evaluation involved only a single annotator, potentially introducing personal bias.
- Evaluation is limited to Gemma-2 and Llama-3.1; behavior on other architectures (e.g., MoE models) remains to be verified.
- The paper does not examine FF-KV's performance on model behavior steering, which is an important application domain for SAEs.

## Related Work & Insights

- **vs. Geva et al. (2021)**: The earliest work to propose the FF-as-KV-memories perspective; this paper validates that view systematically using modern benchmarks.
- **vs. SAEs (Cunningham et al., Bricken et al.)**: SAEs demonstrated promising interpretable features early on, but this paper shows FF-KV achieves a comparable level.
- **vs. Transcoder (Dunefsky et al.)**: Transcoders are the closest proxy counterpart to FF-KV; this paper finds their features exhibit low overlap with the original FF module.
- **vs. Critiques of SAEs (Makelov et al., Huang et al.)**: This paper provides additional evidence questioning the general superiority of SAEs—FF-KV as a "zero-cost" baseline is already sufficiently strong.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Reintroduces the overlooked FF-KV perspective into modern interpretability research with a novel angle, though the method itself is not complex.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Combines automated evaluation (8 metrics), human evaluation, fidelity analysis, multiple models, and multiple variants—very comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clear and experiments are well-organized, though LaTeX rendering issues affect the readability of some equations.
- **Value**: ⭐⭐⭐⭐⭐ Highly instructive for the interpretability community—before pursuing more complex proxy methods, one should first establish how well simple baselines perform.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Why Is Attention Sparse in Particle Transformer?](why_is_attention_sparse_in_particle_transformer.md)
- [\[NeurIPS 2025\] Steering Information Utility in Key-Value Memory for Language Model Post-Training](steering_information_utility_in_key-value_memory_for_language_model_post-trainin.md)
- [\[NeurIPS 2025\] A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders](a_is_for_absorption_studying_feature_splitting_and_absorption_in_sparse_autoenco.md)
- [\[ICML 2026\] Sparse Autoencoders are Topic Models](../../ICML2026/interpretability/sparse_autoencoders_are_topic_models.md)
- [\[ACL 2026\] AdaptiveK: Complexity-Driven Sparse Autoencoders for Interpretable Language Model Representations](../../ACL2026/interpretability/adaptivek_complexity-driven_sparse_autoencoders_for_interpretable_language_model.md)

</div>

<!-- RELATED:END -->

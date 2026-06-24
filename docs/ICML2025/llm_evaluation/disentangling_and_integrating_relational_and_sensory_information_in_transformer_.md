---
title: >-
  [Paper Note] Disentangling and Integrating Relational and Sensory Information in Transformer Architectures
description: >-
  [ICML 2025][LLM Evaluation][Dual Attention] This paper proposes the Dual Attention Transformer (DAT). By introducing "relational attention" heads into the standard attention mechanism, it decouples and parallelly processes sensory and relational information before integrating them. DAT exhibits significant improvements in data and parameter efficiency across relational reasoning benchmarks, mathematical problem solving, image recognition, and language modeling.
tags:
  - "ICML 2025"
  - "LLM Evaluation"
  - "Dual Attention"
  - "Relational Reasoning"
  - "Transformer Architecture"
  - "Inductive Bias"
  - "Relational Attention"
date: 2026-05-08
content_hash: 73a57dd902dd72c0
---

# Disentangling and Integrating Relational and Sensory Information in Transformer Architectures

**Conference**: ICML 2025  
**arXiv**: [2405.16727](https://arxiv.org/abs/2405.16727)  
**Code**: [https://github.com/Awni00/dual-attention](https://github.com/Awni00/dual-attention)  
**Area**: LLM Evaluation  
**Keywords**: Dual Attention, Relational Reasoning, Transformer Architecture, Inductive Bias, Relational Attention

## TL;DR
This paper proposes the Dual Attention Transformer (DAT). By introducing "relational attention" heads into the standard attention mechanism, it decouples and parallelly processes sensory and relational information before integrating them. DAT exhibits significant improvements in data and parameter efficiency across relational reasoning benchmarks, mathematical problem solving, image recognition, and language modeling.

## Background & Motivation
**Background**: Transformers have become the most general neural network architecture, but their attention mechanism fundamentally only routes "sensory information" (features of individual objects/tokens) and lacks explicit processing of "relational information" (relations/comparisons between objects).  

**Limitations of Prior Work**: Numerous experiments show that Transformers perform poorly on tasks requiring relational reasoning. Although LLMs acquire a certain degree of relational reasoning capability through large-scale training, their data efficiency is extremely low.  

**Key Challenge**: The tension between general architectures and inductive biases—purely general architectures suffer from low data efficiency in relational reasoning, while architectures with strong inductive biases suffer from limited generality.  

**Goal**: To introduce inductive biases for relational reasoning without compromising the generality of the Transformer.  

**Key Insight**: Distinguishing between sensory information (object properties) and relational information (comparisons between previous objects), and designing dedicated attention mechanisms for each type.  

**Core Idea**: Utilizing a subset of heads in multi-head attention for standard sensory attention and another subset for relational attention, allowing the model to simultaneously route both types of information.

## Method

### Overall Architecture
DAT maintains the Transformer structure (alternating stacks of attention and MLP) but replaces the multi-head attention with "Dual Attention": $n_h^{sa}$ sensory attention heads + $n_h^{ra}$ relational attention heads. The outputs of both types of heads are concatenated and projected before being fed into the MLP. It is compatible with all Transformer variants and supports causal masking, RoPE, and other techniques.

### Key Designs

1. **Sensory Attention (Standard Self-Attention)**: 

    - **Function**: Routing features of individual objects/tokens.
    - **Mechanism**: Standard QKV attention, where the value is a linear transformation of the source object features.
    - **Output**: $\text{Attention}(x, y) = \sum_i \alpha_i \phi_v(y_i)$

2. **Relational Attention**: 

    - **Function**: Routing relational information between objects.
    - **Mechanism**: Instead of retrieving source object features, it computes a relation vector $r(x, y_i)$ between the target and source, performing inner product comparisons across $d_r$ feature subspaces.
    - **Design Motivation**: Each dimension corresponds to a "comparison perspective", forming a fine-grained relation description; symbol identifiers $s_i$ are introduced to tag sources.
    - **Key Formula**: $\text{RelAttn}(x, y) = \sum_i \alpha_i (r(x, y_i) W_r + s_i W_s)$
    - **Difference from Standard Attention**: Standard attention retrieves "what the source object is"; relational attention retrieves "what relation exists between source and target".

3. **Symbol Assignment Mechanisms**: 

    - **Function**: Assigning abstract identifiers to each source object so the receiver knows which object a relation corresponds to.
    - **Three Variants**:
        - **Positional Symbols**: Learnable positional embeddings.
        - **Relative Positional Symbols**: Relative positional embeddings $s_{j-i}$, which perform better in language modeling.
        - **Symbolic Attention**: Matching and assigning symbol vectors from a finite symbol library through a feature template library.
    - **Design Motivation**: Maintaining relation-centric representation through a finite symbol library.

4. **Theoretical Guarantee of Expressiveness (Theorem 1)**: 

    - Proving that relational attention can approximate any "compute-relation-after-selection" function $\text{Rel}(x, \text{Select}(x, y))$ with arbitrary precision.

### Loss & Training
- Cross-entropy is used for classification, and autoregressive next-token prediction loss is used for language modeling.
- The training configurations are identical to standard Transformer baselines, with only the attention head types being modified.
- Language modeling is trained on FineWeb-Edu (10 billion tokens) with a maximum scale of 1.3B parameters.

## Key Experimental Results

### Main Results

| Task/Dataset | Metric | DAT | Transformer | Gain |
|------------|------|-----|------------|------|
| CIFAR-10 (ViDAT vs ViT) | Classification Accuracy | 89.7±0.1% (6.0M) | 86.4±0.1% (7.1M) | +3.3%, fewer parameters |
| CIFAR-100 (ViDAT vs ViT) | Classification Accuracy | 70.5±0.1% (6.1M) | 68.8±0.2% (7.2M) | +1.7%, fewer parameters |
| Language Modeling (1.3B) | scaling curve | Superior | Baseline | Better data and parameter efficiency |
| Relational Games | Sample Efficiency | Significantly better | Baseline | Huge difference on match_pattern |
| Mathematical Reasoning (Multi-task) | Character-level Accuracy | Superior across all tasks | Baseline | Overall advantage |

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| Relational-only attention heads | Slightly better on purely relational tasks | Minimal difference; the model automatically selects the appropriate computation mode |
| Symmetric vs. Asymmetric Relations | Symmetric is better for vision tasks | Attribute similarity relations are naturally symmetric |
| Positional vs. Relative Positional Symbols | Relative position is better in LM | Relative position is more important in language processing |
| Visualizing Relation Activations | Encodes semantic relations | High activation values between model/state/machine |

### Key Findings
- The sample efficiency gain is most significant on the most challenging "match_pattern" task (which requires second-order relational reasoning).
- Relational attention learns human-interpretable semantic relations (rather than just syntactic relations) in language modeling.
- The computational complexity is $O(n^2)$, identical to the standard Transformer.
- Relational attention not only improves performance but also provides new perspectives for interpretability.

## Highlights & Insights
- Elegantly decouples and integrates both "sensory" and "relational" information flows within the same attention framework.
- The language model scales up to 1.3B and its weights are open-sourced.
- The concept of "symbols" bridges connectionism and symbolism.
- Comprehensive validation is performed across multi-modal and multi-task paradigms.

## Limitations & Future Work
- Lack of hardware-level optimizations such as Flash-Attention, resulting in slower actual training speeds.
- The ratio between the number of sensory heads and relational heads requires tuning and lacks an adaptive mechanism.
- Lacks validation at the scale of 10B+ parameters.
- Mechanistic interpretability is left for future work.

## Related Work & Insights
- The Abstractor architecture (Altabaa et al., 2024) is the direct inspiration for relational attention.
- Insight: Treating relational information as a first-class citizen in Transformers may be the key to more efficient reasoning.

## Rating
- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] HybridNorm: Towards Stable and Efficient Transformer Training via Hybrid Normalization](../../NeurIPS2025/llm_evaluation/hybridnorm_towards_stable_and_efficient_transformer_training_via_hybrid_normaliz.md)
- [\[ICLR 2026\] Fewer Battles, More Gain: An Information-Efficient Framework for Arena-based LLM Evaluation](../../ICLR2026/llm_evaluation/fewer_battles_more_gain_an_information-efficient_framework_for_arena-based_llm_e.md)
- [\[ICML 2026\] REAL: Integrating Regression-Aware Rewards into RL, Teaching LLM-as-a-Judge that "Even a One-Point Difference Matters"](../../ICML2026/llm_evaluation/real_regression-aware_reinforcement_learning_for_llm-as-a-judge.md)
- [\[ICLR 2026\] Do LLM Agents Know How to Ground, Recover, and Assess? Evaluating Epistemic Competence in Information-Seeking Agents](../../ICLR2026/llm_evaluation/do_llm_agents_know_how_to_ground_recover_and_assess_evaluating_epistemic_compete.md)
- [\[ICML 2025\] Sample Efficient Demonstration Selection for In-Context Learning](sample_efficient_demonstration_selection_for_in-context_learning.md)

</div>

<!-- RELATED:END -->

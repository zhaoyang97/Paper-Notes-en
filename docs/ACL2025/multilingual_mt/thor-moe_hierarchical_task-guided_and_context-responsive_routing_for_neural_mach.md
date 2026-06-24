---
title: >-
  [Paper Note] THOR-MoE: Hierarchical Task-Guided and Context-Responsive Routing for Neural Machine Translation
description: >-
  [ACL 2025][Multilingual & Machine Translation][Mixture-of-Experts] This paper proposes the THOR-MoE framework, which utilizes hierarchical task-guided routing (automatically predicting domain/language and generating soft-mixed task representations to select a task-level expert subset) and context-responsive routing (injecting global context into token representations to assist expert selection). It achieves significant performance gains in multi-domain and multilingual transl…
tags:
  - "ACL 2025"
  - "Multilingual & Machine Translation"
  - "Mixture-of-Experts"
  - "Neural Machine Translation"
  - "Hierarchical Routing"
  - "Context-Aware Routing"
  - "Multi-Domain Translation"
  - "Multilingual Translation"
date: 2026-05-08
content_hash: f74f7f977c0bd8bb
---

# THOR-MoE: Hierarchical Task-Guided and Context-Responsive Routing for Neural Machine Translation

**Conference**: ACL 2025  
**arXiv**: [2505.14173](https://arxiv.org/abs/2505.14173)  
**Authors**: Yunlong Liang, Fandong Meng, Jie Zhou (Tencent WeChat AI)  
**Code**: Not publicly available  
**Area**: Multilingual Translation  
**Keywords**: Mixture-of-Experts, Neural Machine Translation, Hierarchical Routing, Context-Aware Routing, Multi-Domain Translation, Multilingual Translation

## TL;DR

This paper proposes the THOR-MoE framework, which utilizes hierarchical task-guided routing (automatically predicting domain/language and generating soft-mixed task representations to select a task-level expert subset) and context-responsive routing (injecting global context into token representations to assist expert selection). It achieves significant performance gains in multi-domain and multilingual translation with fewer activated parameters.

## Background & Motivation

### Problem Background
Sparse MoE architectures expand model capacity without increasing inference overhead through conditional computation, achieving significant progress in NMT. Existing MoE routing schemes mainly fall into two categories: (1) introducing task-specific knowledge (domain/language labels) to design specialized routing modules; (2) improving efficiency by reducing the number of activated experts.

### Limitations of Prior Work

**Reliance on explicit task labels**: Existing methods (e.g., Lingual-MoE) directly use hard language/domain labels to guide routing, but these labels are often unavailable during actual test stages. For code-mixed sentences or cross-domain texts, a single label cannot accurately describe the input characteristics.

**Routing depends solely on local token representations**: Traditional routing mechanisms allocate experts based only on the local representation of the current token, ignoring global context information. Context reflects the difficulty and contextual role of each token, which is crucial for optimal routing decisions.

**Lack of hierarchical design**: Selecting experts for each token directly from the entire pool of experts fails to utilize the natural grouping characteristics at the task level to narrow down the candidate search space.

### Core Motivation
To design a plug-and-play MoE routing framework that: (a) automatically acquires rather than relies on predefined task knowledge, using mixed representations to handle ambiguous boundaries; (b) hierarchically selects task-level expert subsets before performing token-level routing; and (c) integrates global context into the token routing process.

## Method

### Overall Architecture
THOR-MoE adds two modules on top of standard MoE routing, forming a three-level routing process:
1. **Task Prediction and Mixed Representation Generation**: Automatically predicts the domain/language of the input and generates a soft-mixed task representation.
2. **Hierarchical Task-Guided Routing**: Uses the mixed task representation to select a task-level candidate expert subset $\mathcal{S}^t$ from the entire expert pool.
3. **Context-Responsive Token Routing**: Injecting global context into token representations, performing final token-level expert selection from the candidate subset $\mathcal{S}^t$.

This framework is compatible with both Top-k and Top-p routing strategies.

### Key Design 1: Hierarchical Task-Guided Routing

**Task Predictor**: A special `[CLS]` token is added before the input. After Transformer encoding, it passes through MaxPooling and a fully-connected layer to obtain the task distribution prediction:

$$\mathcal{P}^t = \text{Softmax}(\mathbf{W}^p \cdot \text{MaxPooling}(\mathbf{H}^{cls}))$$

**Mixed Task Representation**: Instead of directly taking the hard label via argmax, the predicted probability distribution is used to compute a weighted sum of the task embedding matrix:

$$\mathbf{E}_p = \sum(\mathcal{P}^t \cdot \mathbf{EMB1})$$

This soft-mixed representation naturally possesses fault tolerance for code-mixed inputs and cross-domain texts—experiments demonstrate that it even outperforms using golden labels.

**Hierarchical Routing**: The mixed task representation is fed into a specialized task router $g^t$ to select a task-level candidate subset $\mathcal{S}^t$ from the complete expert pool via TopK. Subsequent token-level routing is performed solely within $\mathcal{S}^t$, significantly narrowing down the search space.

### Key Design 2: Context-Responsive Routing

Before token-level routing, a gating mechanism is used to integrate the global context into each token's representation:

$$\mathbf{x}_i = g \odot \mathbf{x}_i + (1-g) \odot \mathbf{H}_{ctx}$$

Where $g = \sigma([\mathbf{x}_i; \mathbf{H}_{ctx}]\mathbf{W}^g + \mathbf{b}^g)$ is a learnable gating function, and $\mathbf{H}_{ctx}$ is the average hidden state of all tokens in the sequence. On the decoder side, the context is dynamically updated with decoding steps (using the representation of the generated prefix).

This allows the router to evaluate the difficulty and role of each token from a global perspective, assigning more appropriate experts to them.

### Loss & Training

The total loss consists of four or five components:
- $\mathcal{L}_{NMT}$: Standard translation loss
- $\mathcal{L}_{tp}$: Task prediction cross-entropy loss
- $\mathcal{L}_{bd}$: Task-level load balancing loss, ensuring even utilization of experts across different tasks
- $\mathcal{L}_{bt}$: Token-level load balancing loss, balanced within the candidate subset
- $\mathcal{L}_{topp}$: (Top-p only) Dynamic routing entropy constraint, preventing the activation of too many experts

## Key Experimental Results

### Experiment 1: Multi-Domain Translation (De→En, Decoder-only Architecture)

Fine-tuned on 5 domains based on a pruned version of Qwen1.5-MoE (Trim-MoE, 3.5B total parameters / 2.3B active parameters).

| Model | IT | Koran | Medical | Law | Subtitles | Avg. |
|------|-----|-------|---------|-----|-----------|------|
| Dense SFT-3B | 40.65 | 20.40 | 51.40 | 54.80 | 28.33 | 39.12 |
| Trim-MoE (Top-2) | 45.10 | 22.68 | 51.84 | 57.12 | 29.02 | 41.15 |
| Trim-MoE (Top-p) | 39.39 | 19.21 | 55.67 | 60.18 | 29.21 | 40.73 |
| **THOR-MoE (Top-2)** | **46.00** | **23.35** | **55.79** | **61.06** | 28.23 | **42.89** |
| **THOR-MoE (Top-p)** | 44.63 | 22.53 | 53.58 | 58.65 | 27.99 | 41.48 |

THOR-MoE (Top-2) achieves an average improvement of **+1.74 BLEU** over Trim-MoE (Top-2), with multiple significance tests yielding p<0.01.

### Experiment 2: Multilingual Translation (OPUS-16, Encoder-Decoder Architecture)

16 languages (8 high / 4 medium / 4-low resource), based on Transformer-Base + 32 experts.

| Model | En→XX Avg. | XX→En Avg. | Total Avg. |
|------|-----------|-----------|--------|
| Dense Transformer-base | 26.16 | 30.27 | 28.21 |
| ST-MoE (Top-1) | 29.09 | 33.71 | 31.40 |
| Lingual-MoE | 30.95 | 33.81 | 32.38 |
| **THOR-MoE (Top-2)** | **31.98** | **34.64** | **33.31** |
| THOR-MoE (Top-p) | 31.55 | 34.26 | 32.91 |

THOR-MoE (Top-2) improves by **+0.93 BLEU** (p<0.05) on the Total Avg. compared to the strongest baseline, Lingual-MoE. The largest gain is achieved on low-resource languages (+1.51 vs. Lingual-MoE).

### Experiment 3: Efficiency Analysis

| Routing Strategy | IT | Koran | Medical | Law | Subtitles | Average Activated Experts |
|---------|-----|-------|---------|-----|-----------|------------|
| Top-p (Original) | 1.87 | 1.95 | 1.82 | 1.92 | 1.77 | 1.87 |
| Top-p + Context | 1.37 | 1.61 | 1.42 | 1.56 | 1.31 | **1.45** |

Context-responsive routing reduces the average number of activated experts from 1.87 to 1.45 (a 22% reduction) while achieving better performance—context helps the model route more confidently to a smaller number of experts.

## Key Findings

1. **Mixed Representation > Golden Label**: Using mixed task representations weighted by predicted probabilities outperforms directly using golden labels (41.48 vs. 41.32 BLEU), demonstrating the fault-tolerance advantage of soft distributions.
2. **Hierarchical Design > Direct Fusion**: Employing task knowledge hierarchically—first filtering candidates and then performing token routing—substantially outperforms directly concatenating task information into token representations (41.48 vs. 40.95).
3. **Context Enhances Routing Efficiency**: By introducing context, the model uses fewer experts (1.45 vs. 1.87) to achieve better performance and converges faster to a lower activation count during training.
4. **Strong Framework Generality**: Consistently effective across Top-1, Top-2, and Top-p routing strategies, and holds true for both Decoder-only and Encoder-Decoder architectures.

## Highlights & Insights

- **Plug-and-Play Design**: THOR-MoE, as a modular component, can be seamlessly integrated into any MoE architecture using Top-k or Top-p routing, without modifying the base model structure.
- **Automatic Task Knowledge Acquisition**: The task predictor automatically acquires domain/language knowledge, removing reliance on explicit labels and enhancing feasibility for practical deployment.
- **Elegant Handling via Mixed Representations**: Soft probability-weighted task representations naturally handle code-mixed and cross-domain ambiguous inputs, demonstrating greater robustness than hard labels.
- **Dual Benefits of Context**: Global context not only improves routing accuracy (better expert matching) but also enhances efficiency (fewer active experts), embodying the routing philosophy of "knowing the difficulty and specializing."
- **Prior-Tightening Effect of Hierarchical Routing**: Narrowing down candidate options at the task level before selecting at the token level closely resembles the concept of prior-constrained search space within a Bayesian framework.

## Limitations & Future Work

- **Reliance on Task Number Prior**: The design requires predefining the number of domain/language groups, which limits scalability and makes it inflexible when facing open domains or a large number of fine-grained tasks.
- **Validation Limited to Translation Tasks**: All experiments are limited to NMT. Performance on other generative tasks (such as summarization and dialogue) or discriminative tasks is not evaluated.
- **Simplistic Context Representation**: The global context is aggregated using simple average pooling; more sophisticated context aggregation methods, such as attention-weighting, have not been explored.
- **Insufficient Discussion of Computational Overhead**: The additional parameters and computational cost introduced by the task predictor and gating mechanism are not profiled in detail.
- **Small-Scale Multi-Domain Experiments**: Multi-domain translation is evaluated on only one language pair (De→En) across 5 domains, indicating a limited scale.

## Related Work & Insights

- **Lingual-MoE (Zhao et al., 2024)**: The most direct baseline, which also employs hierarchical language guidance and dynamic routing but relies on hard language ID embeddings and lacks contextual information. THOR-MoE completely outperforms it through mixed representations and context injection.
- **Top-p Routing (Huang et al., 2024)**: A routing strategy with a dynamic number of activated experts, on top of which THOR-MoE achieves superior performance with fewer active parameters.
- **Hybrid-MoE (Kudugunta et al., 2021)**: A hybrid scheme with token routing on the encoder side and language routing on the decoder side, which is outperformed by a unified hierarchical design.
- **Insights**: The core idea of this work—hierarchical routing featuring a "coarse-to-fine" selection strategy integrated with global information—can be extended to MoE routing designs during LLM pre-training, especially in multi-task instruction tuning scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐ — The combined design of hierarchical routing and context responsiveness is novel, and the finding that mixed task representations outperform golden labels is interesting.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Two architectures, two benchmarks, and rich ablations, but lacks validation beyond translation.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, well-articulated motivation, and good alignment between text and figures.
- Value: ⭐⭐⭐⭐ — A plug-and-play MoE routing enhancement scheme with high practical value for NMT, though limited to translation scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Exploring In-context Example Generation for Machine Translation](exploring_in-context_example_generation_for_machine_translation.md)
- [\[ACL 2025\] Multi-perspective Alignment for Increasing Naturalness in Neural Machine Translation](multi-perspective_alignment_for_increasing_naturalness_in_neural_machine_transla.md)
- [\[ACL 2025\] Memorization Inheritance in Sequence-Level Knowledge Distillation for Neural Machine Translation](memorization_inheritance_seqkd.md)
- [\[ACL 2025\] GrammaMT: Improving Machine Translation with Grammar-Informed In-Context Learning](grammamt_improving_machine_translation_with_grammar-informed_in-context_learning.md)
- [\[ACL 2025\] Registering Source Tokens to Target Language Spaces in Multilingual Neural Machine Translation](registering_source_tokens_to_target_language_spaces_in_multilingual_neural_machi.md)

</div>

<!-- RELATED:END -->

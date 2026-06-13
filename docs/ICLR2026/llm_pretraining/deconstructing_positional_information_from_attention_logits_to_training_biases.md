---
title: >-
  [Paper Note] Deconstructing Positional Information: From Attention Logits to Training Biases
description: >-
  [ICLR2026][LLM Pretraining][Positional Encoding] This paper proposes a unified analytical framework based on Toeplitz matrices, categorizing positional encodings into additive (Absolute/T5/ALiBi) and multiplicative (RoPE…
tags:
  - "ICLR2026"
  - "LLM Pretraining"
  - "Positional Encoding"
  - "RoPE"
  - "Toeplitz Matrix"
  - "Attention Mechanism"
  - "Single-Head Deposit Pattern"
date: 2026-05-08
content_hash: 8ac74a8e191a730f
---

# Deconstructing Positional Information: From Attention Logits to Training Biases

**Conference**: ICLR2026  
**arXiv**: [2505.13027](https://arxiv.org/abs/2505.13027)  
**Code**: To be confirmed  
**Area**: LLM Pretraining  
**Keywords**: Positional Encoding, RoPE, Toeplitz Matrix, Attention Mechanism, Single-Head Deposit Pattern  

## TL;DR
This paper proposes a unified analytical framework based on Toeplitz matrices, categorizing positional encodings into additive (Absolute/T5/ALiBi) and multiplicative (RoPE) types. Through synthetic tasks, it reveals that RoPE exhibits significant advantages on position-sensitive tasks but suffers from a "single-head deposit pattern" in shallow layers, where nearly all positional reasoning concentrates in a single attention head. The paper further provides a theoretical proof that this pattern is an intrinsic property of RoPE's multiplicative structure.

## Background & Motivation

**Background**: Positional encoding (PE) is a core component of Transformers, evolving from additive schemes (Sinusoidal, T5 Bias, ALiBi) to multiplicative ones (RoPE). However, mechanistic understanding remains limited to two properties: distance decay and translation invariance.

**Limitations of Prior Work**: Despite RoPE's theoretically desirable properties (e.g., decay characteristics supporting length generalization), it underperforms simpler relative PE or even NoPE models on certain tasks—a "performance paradox" that lacks explanation.

**Core Idea**: The attention logit computation is deconstructed into content–position interaction terms and described uniformly via Toeplitz matrices. Additive PE introduces positional information through independent bias terms, whereas multiplicative PE (RoPE) couples positional signals with content via Hadamard products—this strong coupling leads to excessive concentration of positional reasoning.

## Method

### Overall Architecture
Token representations are decomposed into a content component $c_i$ and a positional component $p_i$ (i.e., $x_i = c_i + p_i$), and the attention logit matrix is analyzed. For additive PE, the logit matrix is a sum of interaction terms ($\mathbf{L}_{\text{Add}} = G_{q^c,k^c} + G_{q^c,k^p} + G_{q^p,k^c} + G_{q^p,k^p} + \mathbf{B}$); for multiplicative PE (RoPE), it takes a Hadamard product form ($\mathbf{L}_{\text{RoPE}} = \text{Re}\{(\cdots) \circ G_{\mathbf{e}}\}$), where $G_{\mathbf{e}}$ is a Toeplitz kernel.

### Key Design 1: Synthetic Task Design
- **Task 1 (Position-Sensitive)**: Given two trigger words in a sequence, predict their relative distance (classification); requires the model to know both "what" and "where."
- **Task 2 (Position-Agnostic)**: Count the number of occurrences of a specific trigger word in a sequence; positional information acts as a nuisance variable.
- These two contrastive tasks precisely isolate the content–position coupling capability.

### Key Design 2: Discovery and Validation of the Single-Head Deposit Pattern
Through per-head ablation (zeroing out individual heads), it is found that removing a single attention head in the first layer of a RoPE model causes accuracy on Task 1 to drop by approximately 60%, while removing other heads has negligible effect. This pattern only emerges in the combination of "RoPE + position-sensitive task"—NoPE does not exhibit it, nor does RoPE on Task 2.

### Key Design 3: Theoretical Derivation
- **Proposition 6.1**: The multiplicative structure of RoPE provides a deterministic lower bound (non-zero seed) for gradient signals, ensuring that some head receives a positive positional learning signal.
- **Proposition 6.2**: The additive bias of ALiBi causes gradient signals to cancel out during batch aggregation, preventing the formation of a stable seed.
- **Theorem 6.1**: During backpropagation, the seed advantage is amplified exponentially ($\text{Margin}_l \geq \text{Margin}_L \prod_{k=l}^{L-1} \gamma_k$, $\gamma_k > 1$), ultimately leading to monopolization of positional reasoning by a single head.

## Key Experimental Results

### Synthetic Task Performance

| PE Method | Task 1 (Position-Sensitive) Acc | Task 2 (Position-Agnostic) Acc |
|-----------|--------------------------------|-------------------------------|
| RoPE | **92.64%** | 69.43% |
| MLA | 88.34% | **97.41%** |
| Absolute | Second best | Moderate |
| ALiBi | Fails | Worst (strong bias harmful) |
| NoPE | Fails | **77.69%** |

### Ablation Study: Minimum Number of RoPE Heads

| No. of RoPE Heads | Task 1 Acc |
|-------------------|-----------|
| All heads | 92.64% |
| 2 heads | ≈90%+ |
| **1 head** | **≈90%** |

### Key Findings
- RoPE requires only 1–2 heads to perform all positional reasoning; the remaining heads are redundant for position-sensitive tasks.
- The hybrid architecture MLA (the attention design in DeepSeek-V3) successfully eliminates the deposit pattern while achieving near-optimal performance on both tasks (88.34% / 97.41%).
- RoPE suppresses the formation of implicit positional representations: in a hybrid Absolute+RoPE model, additive positional directions are completely displaced after Layer 2.

## Highlights & Insights
- **Elegant Theoretical Framework**: All PE methods are unified under an additive vs. multiplicative dichotomy via Toeplitz matrices, yielding strong explanatory power.
- **Complete Chain from Phenomenon to Mechanism**: Synthetic task discovery → ablation validation → mathematical proof, forming a closed three-step loop.
- **Theoretical Validation of MLA**: This work provides the first positional-encoding-based explanation for why the MLA design in DeepSeek-V3 is effective.

## Limitations & Future Work
- The causal relationship between the deposit pattern and length extrapolation capability remains a hypothesis and is not directly validated.
- The synthetic tasks are overly simplified; applicability to complex NLP tasks (e.g., sequence reversal, Dyck languages) is unknown.
- Analysis is limited to 6-layer small models; whether the deposit pattern persists in large-scale models remains unclear.

## Related Work & Insights
- Explains the counterintuitive finding of Kazemnejad et al. (2023) that NoPE outperforms RoPE on certain tasks—because multiplicative bias is harmful on position-agnostic tasks.
- Provides design principles for future PE research: pure multiplicative coupling should be avoided in favor of MLA-style hybrid strategies (parallel NoPE + RoPE pathways).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Toeplitz unified framework + discovery and theoretical proof of the deposit pattern
- Experimental Thoroughness: ⭐⭐⭐⭐ Synthetic experiments are elegantly designed with thorough ablations, but natural language experiments are absent
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretical exposition is clear, with a complete logical chain from framework to findings to proofs
- Value: ⭐⭐⭐⭐ Substantially advances mechanistic understanding of positional encoding and offers guidance for novel designs such as MLA

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Explaining Grokking and Information Bottleneck through Neural Collapse Emergence](explaining_grokking_and_information_bottleneck_through_neural_collapse_emergence.md)
- [\[ICML 2026\] Focus and Dilution: The Multi-stage Learning Process of Attention](../../ICML2026/llm_pretraining/focus_and_dilution_the_multi-stage_learning_process_of_attention.md)
- [\[AAAI 2026\] No-Regret Strategy Solving in Imperfect-Information Games via Pre-Trained Embedding](../../AAAI2026/llm_pretraining/no-regret_strategy_solving_in_imperfect-information_games_via_pre-trained_embedd.md)
- [\[ICML 2026\] InfoLaw: Information Scaling Laws for Large Language Models with Quality-Weighted Mixture Data and Repetition](../../ICML2026/llm_pretraining/infolaw_information_scaling_laws_for_large_language_models_with_quality-weighted.md)
- [\[NeurIPS 2025\] The Atlas of In-Context Learning: How Attention Heads Shape In-Context Retrieval Augmentation](../../NeurIPS2025/llm_pretraining/the_atlas_of_in-context_learning_how_attention_heads_shape_in-context_retrieval_.md)

</div>

<!-- RELATED:END -->

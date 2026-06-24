---
title: >-
  [Paper Note] Counting in Small Transformers: The Delicate Interplay between Attention and Feed-Forward Layers
description: >-
  [ICML2025][LLM Pretraining][Transformer mechanistic analysis] Through a histogram counting task, this paper reveals the delicate division of labor between attention layers and feed-forward networks (FFNs) in small Transformers: attention excels at relation-based counting, whereas FFNs are responsible for inventory-based counting (dictionary memorization). The emergence of these two strategies is determined by the relative relationship among embedding dimension $d$…
tags:
  - "ICML2025"
  - "LLM Pretraining"
  - "Transformer mechanistic analysis"
  - "counting task"
  - "attention-FFN interaction"
  - "embedding orthogonality"
  - "mechanistic interpretability"
date: 2026-05-08
content_hash: f278400d7a42e798
---

# Counting in Small Transformers: The Delicate Interplay between Attention and Feed-Forward Layers

**Conference**: ICML2025  
**arXiv**: [2407.11542](https://arxiv.org/abs/2407.11542)  
**Code**: [GitHub](https://github.com/SPOC-group/counting-attention)  
**Area**: Transformer Theory  
**Keywords**: Transformer mechanistic analysis, counting task, attention-FFN interaction, embedding orthogonality, mechanistic interpretability

## TL;DR

Through a histogram counting task, this paper reveals the delicate division of labor between attention layers and feed-forward networks (FFNs) in small Transformers: attention excels at relation-based counting, whereas FFNs are responsible for inventory-based counting (dictionary memorization). The emergence of these two strategies is determined by the relative relationship among embedding dimension $d$, hidden layer size $p$, and vocabulary size $T$.

## Background & Motivation

While Transformer architectures have achieved widespread success, there is still a lack of clear understanding regarding the specific functions and collaboration of their internal components (attention vs. FFN). This paper chooses a minimalist yet revealing task—the **histogram task**—as an analytical tool: given an input sequence $\mathbf{x} = [A, B, D, D, B, B]$, output the occurrence frequency of the corresponding token at each position in the sequence, $\mathbf{y} = [1, 3, 2, 2, 3, 3]$.

This task appears simple, but even modern LLMs with 8B parameters cannot reliably solve it in-weights (relying instead on chain-of-thought). Based on this, the authors investigate how different token-mixing mechanisms (linear mixing vs. dot-product attention) and FFN capacities influence the types of algorithms the model can learn.

## Method

### Model Architecture

A single-layer Transformer block, containing two components:

1. **Token mixing layer**: produces the mixed token $\bar{x}'_\ell = \bar{x}_\ell + [\mathbf{A}(\bar{\mathbf{x}})\bar{\mathbf{x}}]_\ell$
2. **Feed-forward network (FFN)**: $f(\bar{x}'_\ell) = \text{ReLU}(\bar{x}'_\ell W_1 + b_1) W_2 + b_2$, with hidden dimension $p$

Final prediction: $F(\bar{\mathbf{x}})_\ell = \arg\max_{c \in \{1,\dots,C\}} f(\bar{x}'_\ell)_c$

### Four Token-Mixing Mechanisms

| Mechanism | Formula | Parameter Count |
|------|------|--------|
| **lin** (Linear mixing) | $\mathbf{A} = A$ (learnable matrix) | $L^2$ |
| **lin+sftm** | $\mathbf{A} = \text{softmax}(A)$ | $L^2$ |
| **dot** (Dot-product mixing) | $\mathbf{A} = \frac{1}{\sqrt{d}} \bar{\mathbf{x}} W_Q W_K^T \bar{\mathbf{x}}^T$ | $2d^2$ |
| **dot+sftm** | $\mathbf{A} = \text{softmax}(\mathbf{A}_{\text{dot}})$ | $2d^2$ |

Additionally, there is the **bos** variant: adding a BOS token at the beginning of the input: $\tilde{\mathbf{x}} = (\$, x_1, \dots, x_L)$.

### Two Counting Strategies

**Strategy 1: Relation-based Counting (RC)**

- Leverages the **pairwise comparison** capability of dot-product attention: attention scores are high for identical tokens and low for different tokens.
- Requires only $p=1$ hidden neuron to extract counting information through a single direction (e.g., the BOS direction).
- In the BOS model: $\langle \bar{x}'_\ell, e_{\text{BOS}} \rangle = T + \text{hist}_{\mathbf{x}}(\ell) + 1$, where the count is linearly encoded in the projection onto the BOS direction.
- In the dot model: uses a "tagged embedding"—adding a common direction $e_{\text{cnt}}$ to orthogonal embeddings, such that $\langle e_{\text{cnt}}, \bar{x}'_\ell \rangle \propto 1 + \text{hist}(\ell) \cdot a_= + (L - \text{hist}(\ell)) \cdot a_{\neq}$.

**Strategy 2: Inventory-based Counting (IC)**

- The FFN acts as a **lookup table**, employing $p \geq T$ hidden neurons to memorize the entire alphabet.
- Let $(W_1)_t = e_t$. Then $\text{hist}_{\mathbf{x}}(\ell) = \frac{1}{a} \sum_{t \in \mathcal{T}} \text{ReLU}(\langle \bar{x}'_\ell, e_t \rangle - 1)$.
- Due to the bias term $-1$, only the neuron corresponding to the token $x_\ell$ in the residual connection is activated.
- Suitable for architectures such as lin, lin+sftm, and dot+sftm, which cannot perform RC.

### Key Theoretical Results

**Why can't dot+sftm perform RC?** Softmax normalization forces $\sum_m a_{\ell m} = 1$. Any common direction present in all tokens will have a constant scaling factor of 1 after mixing, thereby losing the counting information. Consequently, it must resort to the IC strategy ($p \geq T$).

**Robustness with non-orthogonal embeddings ($d < T$):** Utilizing the Welch bound to constrain mutual coherence, the paper derives the minimum $d$ required for perfect counting across different architectures:

- lin/lin+sftm ($p=T$): $d \geq \lceil \frac{T(2L-3)^2}{T-1+(2L-3)^2} \rceil$
- dot/bos ($p=1$): the above equation $+1$
- dot/bos ($p=T$): $d \geq \lceil \frac{T(L-1)}{T-1+(L-1)} \rceil$

**Denoising effect of softmax:** High-temperature softmax can non-linearly suppress attention scores of non-matching tokens, reducing the required embedding dimension to $d \geq \lceil \log_2(T+1) \rceil + 2$ (requiring only $d=7$ for $T=32$).

## Key Experimental Results

Experimental setup: $T=32$ unique tokens, sequence length $L=10$, Adam optimizer (lr=1e-3), 500 epochs, 10,000 new samples per epoch.

| Architecture | Minimum Configuration for Perfect Accuracy | Requirement on $p$ | RC/IC |
|------|-------------------|--------------|-------|
| **bos+sftm** | $d \geq T, p=1$ | $p=1$ is sufficient | RC |
| **bos** | $d \geq T, p=1$ | $p=1$ is sufficient | RC |
| **dot** | $d \geq T, p=1$ | $p=1$ is sufficient | RC |
| **dot+sftm** | $d \geq T, p \geq T$ | Requires $p \geq T$ | IC |
| **lin+sftm** | $d \geq T, p \geq T$ | Requires $p \geq T$ | IC |
| **lin** | $d \geq T, p \geq T$ | Requires $p \geq T$ | IC |

Key Findings:
- bos+sftm can still achieve near 100% accuracy at $d < T$ (requiring only $d=7$) due to the denoising capability of softmax.
- For the dot model, accuracy slightly drops from 100% to 99% when $p$ increases from 1 to $T$, suggesting a potential superposition between RC and IC.
- The performance of a 2-layer Transformer is highly consistent with the 1-layer baseline, indicating that depth does not change the fundamental mechanism.

## Highlights & Insights

1. **Minimalist task reveals deep mechanisms**: A seemingly trivial counting task precisely exposes the functional division of labor between attention and FFN—attention performs comparisons, and FFN acts as memory.
2. **Essential explanation of the BOS token**: BOS is not merely a sequence marker; it acts as a "counter" in the RC strategy—its attention weights encode the frequency of occurrence of each token.
3. **The double-edged sword of softmax**: Softmax **destroys** the RC ability in dot+sftm (normalization erases the counting signal), but **enhances** robustness in bos+sftm (suppresses noise from non-matching tokens).
4. **Closed loop of theoretical construction and experimental validation**: Every theoretical proposition has a corresponding weight construction and trained model validation. Mechanistic analyses (attention matrices, FFN response curves) are highly consistent with theoretical predictions.
5. **Implications for LLMs**: In real-world LLMs, vocabulary size $T$ is much larger than model dimension $d$, making the non-orthogonal embedding analysis in this paper directly relevant.

## Limitations & Future Work

1. **Limited to 1-layer non-autoregressive models**: The effects of causal masking and positional encodings are not considered, so generalization to multi-layer or autoregressive architectures requires further verification.
2. **Single task**: It remains uncertain whether the conclusions from the histogram task can transfer to other basic tasks such as sorting or table lookup.
3. **Fixed training budget**: Performance degradation at $L=30$ might be due to insufficient training rather than architectural limitations.
4. **Hard-to-achieve Welch bound**: Theoretical derivations rely on the attainability of the Welch bound, but constructing embedding matrices that meet this bound is difficult in practice.
5. **Superposition phenomenon not deeply explored**: The exact mechanism when RC and IC coexist (e.g., the SVD analysis was only a preliminary exploration) warrants more detailed study.

## Related Work & Insights

- **RASP Language** (Weiss et al., 2021): Predicting histograms requires a BOS token, whereas this paper proves that dot models can achieve RC without a BOS token.
- **FFN as memory modules** (Geva et al., 2021; Meng et al., 2022): This paper corroborates the lookup-table function of FFN from the perspective of a counting task.
- **Algorithm-architecture alignment** (Dziri et al., 2023): LLM hallucinations may stem from a mismatch between the computational graph and the task. This paper demonstrates how subtle architectural changes can lead to the emergence of drastically different algorithms.

## Rating

- Novelty: ⭐⭐⭐⭐ — Systematically reveals the division of labor between attention and FFN starting from a minimalist task. The RC vs. IC dichotomy is highly inspiring.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Solid theoretical constructions, large-scale hyperparameter sweeps, and mechanistic validations form a complete closed loop.
- Writing Quality: ⭐⭐⭐⭐ — Clearly structured, though formulas and notation are dense, requiring some prior background.
- Value: ⭐⭐⭐⭐ — Offers significant reference value for understanding the internal mechanisms of Transformers and provides guidance for architectural design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Scaling Embedding Layers in Language Models](../../NeurIPS2025/llm_pretraining/scaling_embedding_layers_in_language_models.md)
- [\[ICML 2025\] Benign Overfitting in Token Selection of Attention Mechanism](benign_overfitting_in_token_selection_of_attention_mechanism.md)
- [\[ICLR 2026\] Energy-Based Transformers are Scalable Learners and Thinkers](../../ICLR2026/llm_pretraining/energy-based_transformers_are_scalable_learners_and_thinkers.md)
- [\[ICCV 2025\] Image Intrinsic Scale Assessment: Bridging the Gap Between Quality and Resolution](../../ICCV2025/llm_pretraining/image_intrinsic_scale_assessment_bridging_the_gap_between_quality_and_resolution.md)
- [\[ICLR 2026\] Conditioned Initialization for Attention](../../ICLR2026/llm_pretraining/conditioned_initialization_for_attention.md)

</div>

<!-- RELATED:END -->

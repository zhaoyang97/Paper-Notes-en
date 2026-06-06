---
title: >-
  [Paper Note] The Expressive Power of Low Precision Softmax Transformers with (Summarized) Chain-of-Thought
description: >-
  [ICML 2026][LLM Reasoning][Low-precision softmax] This paper demonstrates for the first time that standard Transformer decoders using softmax attention with bfloat16-level precision (with both activations and attention w…
tags:
  - "ICML 2026"
  - "LLM Reasoning"
  - "Low-precision softmax"
  - "Chain-of-Thought"
  - "Turing Machine simulation"
  - "Ternary activation"
  - "Summarized CoT"
date: 2026-05-08
content_hash: f08bb73c0a9397a3
---

# The Expressive Power of Low Precision Softmax Transformers with (Summarized) Chain-of-Thought

**Conference**: ICML 2026  
**arXiv**: [2605.18079](https://arxiv.org/abs/2605.18079)  
**Code**: https://github.com/moritzbroe/transformer-expressivity (Available)  
**Area**: LLM Reasoning / Transformer Expressivity Theory  
**Keywords**: Low-precision softmax, Chain-of-Thought, Turing Machine simulation, Ternary activation, Summarized CoT

## TL;DR
This paper demonstrates for the first time that standard Transformer decoders using softmax attention with bfloat16-level precision (with both activations and attention weights rounded) can simulate arbitrary Turing Machines using CoT, provided that depth and width grow logarithmically with the context. It further proves that Summarized CoT (SCoT) reduces the scale from a time bound $\hat{t}$ to a spatial bound $\hat{s}$ and empirically finds that "increasing depth instead of precision" is the true remedy for CoT long-context failures in Sudoku tasks.

## Background & Motivation

**Background**: Transformer expressivity theory has proven that these models can simulate Turing Machines, but at the cost of being disconnected from reality—mainstream constructions (Merrill & Sabharwal 2024; Yang et al. 2025) rely on hardmax attention and at least $\mathcal{O}(\log \hat{t})$ activation precision, with some introducing non-standard positional encodings.

**Limitations of Prior Work**: Modern LLMs universally use softmax attention with bfloat16/int8 precision and improve capabilities by increasing depth and width. Theoretical "logarithmic precision + constant depth/width" and practical "constant precision + logarithmic depth/width" are entirely different directions, leaving theoretical results with almost no predictive power for practice.

**Key Challenge**: Forcing existing hardmax constructions into softmax requires multiplying query/key projections by a constant $c$ to sharpen the softmax and approximate hardmax. However, in existing constructions, the attention score gap $\beta$ decays polynomially with sequence length, causing $c$ to grow polynomially—exploding parameter magnitudes and destroying "uniformity," not to mention the $\mathcal{O}(\log \hat{t})$ precision required to store such numbers.

**Goal**: Construct a Transformer model that truly fits practical usage—softmax attention, bfloat16-level precision, moderate parameter magnitudes—to prove it remains Turing complete, while answering how much depth, width, and precision CoT and SCoT each require.

**Key Insight**: Instead of patching existing hardmax constructions, the authors **construct a class of "softmax-friendly" hardmax Transformers from scratch**: all activations are restricted to $\{-1, 0, 1\}$ (ternary), and attention score gaps do not decay with context (always $\geq 1/\sqrt{d_k}$), making a moderate $c = \mathcal{O}((\log \hat{l})^{3/4})$ sufficient to suppress softmax errors.

**Core Idea**: Use "ternary activations + binary position registers + binary search for symbol reading" instead of "log-precision floating-point accumulation" to reduce the hardmax $\to$ softmax conversion cost to a logarithmic level. This same construction path is applied to the SCoT paradigm.

## Method

The paper provides a two-stage constructive proof followed by a set of small-scale empirical studies. The theoretical part first presents the hardmax construction (Section 3), then uses a unified "rescaling + MLP denoising" mechanism to convert it into a softmax construction with precision quantization (Section 4); the empirical part (Section 5) trains small Transformers to solve Sudoku to compare the learnability of different modeling choices.

### Overall Architecture

The theoretical pipeline is as follows:

1.  **Input**: A multi-tape Turing Machine $M$ and a time bound $\hat{t}$ (or spatial bound $\hat{s}$);
2.  **First Stage (Section 3)**: Construct a hardmax Transformer $T$ with ternary activations and size $\mathcal{O}(\log \hat{t})$, which can output $f_M(w)$ within $\mathcal{O}(t_M(w)+|w|)$ steps using CoT/SCoT;
3.  **Second Stage (Section 4)**: Multiply query/key projections of $T$ by $c$, replace hardmax with softmax, and insert "denoising MLPs" to eliminate biases introduced by rounding attention weights, resulting in a softmax version $\tilde T_c$ that matches the output of $T$ for all valid inputs;
4.  **Precision Budget**: $c = \mathcal{O}((\log \hat{l})^{3/4})$, activation exponent bits $\mathcal{O}(\log\log\log \hat{l})$, and attention weight exponent bits $\mathcal{O}(\log\log \hat{l})$. This implies bfloat16 is sufficient until $\hat{l} \approx 10^{38}$.

The key difference between CoT and SCoT lies in the bound of the second-stage input: CoT relies entirely on the time bound $\hat{t}$ as the entire decoded sequence remains in the same context; SCoT allows writing `<summ>...</summ>` blocks as "checkpoints," where the length of each non-summarized region is $\mathcal{O}(s_M(w))$, thus model size only grows logarithmically with the spatial bound, providing significant gains for tasks like Sudoku that are "space-efficient but time-intensive."

### Key Designs

1.  **Ternary Activations + Binary Register Representation**:
    - **Function**: Encodes Turing Machine configurations (state, head position, tape symbols) into $\{-1,0,1\}^d$-dimensional residual streams.
    - **Mechanism**: Each hidden dimension carries specific semantics—several bits for positional registers (binary representations of absolute token position + tape head positions), several bits for state/symbol one-hot encodings, and bits as flags to distinguish token types. MLPs perform bitwise operations like copy, clear, and increment/decrement within the ternary domain; attention retrieves registers from history. Positional encodings are also converted to binary (similar to original sinusoidals, with geometrically decreasing frequencies) to avoid precision dependency.
    - **Design Motivation**: Ternary is the key. Once activations are limited to $\{-1, 0, 1\}$, the initial attention score gap $\beta \geq 1/\sqrt{d_k}$ is not diluted by sequence length, which is why a moderate $c = \mathcal{O}((\log \hat{l})^{3/4})$—rather than a polynomial one—suffices for softmax conversion. Simultaneously, the denoising MLP (see Key Design 3) can perform "hard clamping" on ternary points, which is unfeasible for arbitrary activation sets.

2.  **Binary Search for Symbol Reading + Sequential Head Position Accumulation**:
    - **Function**: In hardmax mode, every generated run token must determine the "most recent symbol written at the current head position."
    - **Mechanism**: Head positions are recovered by explicitly writing coordinates in `<p>...</p>` blocks inserted every $r$ run tokens, with internal token layers sequentially adding head movements $\{L, S, R\}$. Symbol reading is performed via binary search over $\mathcal{O}(\log \hat{t})$ layers, pruning half of the candidate historical tokens at each layer to eventually isolate the "most recently written" token and retrieve its symbol via attention.
    - **Design Motivation**: Compared to the approach of Merrill & Sabharwal (2024), which uses uniform attention over history and divides by $n$ (requiring $\mathcal{O}(\log n)$ precision to store $1/n$), binary search replaces "global summation" with "logarithmic localization." Each localization only requires hardmax to select one token, ensuring the attention score gap does not decay.

3.  **Rescaling + Denoising MLP (Bridge from hardmax to softmax)**:
    - **Function**: Losslessly converts hardmax constructions to softmax versions with double quantization of activations and attention weights.
    - **Mechanism**: All query/key projections are multiplied by $c$, controlling the $\ell_1$ error between softmax and hardmax outputs by $2n e^{-\beta c^2}$. When only activations are rounded (Theorem 4.1), ternary values can be precisely represented in floating-point formats. However, actual attention weights $\alpha_{ij}$ often take values like $1/k$, which cannot be precisely represented. Thus, a **coordinate-wise denoising MLP** is inserted after each attention layer. It implements a step function $f$ that snaps any value within $(-\frac{5}{4},-\frac{3}{4}), (-\frac{1}{4},\frac{1}{4}), (\frac{3}{4},\frac{5}{4})$ back to $-1, 0, 1$ via $x \mapsto x+f(x)$.
    - **Design Motivation**: The denoising MLP is critical for compressing attention weight precision requirements from "linear growth" down to $\mathcal{O}(\log\log \hat{l})$. Its existence depends on the ternary constraint—if the activation range grew linearly, the MLP size would grow linearly, collapsing the approach.

### Loss & Training
No training for the theoretical part. In the empirical section (Section 5), a small Transformer is trained to imitate a deterministic MRV depth-first search Sudoku solver: using the `sudoku-extreme` dataset (~4 million puzzles), SCoT segments of 512 non-summarized tokens, and summary blocks encoding the complete solver configuration. Training uses 20B tokens, bfloat16 mixed precision, temperature 0 greedy decoding, and success is defined as outputting the correct solution.

## Key Experimental Results

### Main Results: Learnability under SCoT vs. CoT for Small Models

**SCoT Model Size Scan** ($d=512$, $H=8$, accuracy on random 10k test subset):

| Depth $L$ | 4 | 5 | 6 | 7 | 8 |
|-----------|---|---|---|---|---|
| Acc (%) | 16.3 | 97.2 | 99.5 | 99.7 | 99.6 |

Performance saturates at $L=6$; same-sized CoT models trained with $N=2^{14}$ context collapse in accuracy far before reaching the length upper bound.

### Ablation Study: Depth vs. Precision (Cracking CoT Long-Context Failure)

| Setting | Max Training Context $N$ | Phenomenon | Conclusion |
|---------|-------------------------|------------|------------|
| $L=6$, bf16 (baseline) | $2^{14}$ | CoT fails far before $N$ | Model size insufficient |
| $L=8$, bf16 | $2^{14}$ | CoT stable throughout | **Increasing depth works** |
| $L=6$, **fp32** | $2^{14}$ | Almost identical to bf16, still fails | **Increasing precision fails** |

### Gain from SCoT (Long-tail Generalization)
A larger SCoT model ($d=768, L=12, H=12$) trained only on $N \leq 2^{12}$ achieved 99.96% on the 10k test set and solved 92 of the 100 "hardest Sudoku puzzles"—puzzles requiring between 1 and 9 million CoT tokens, far exceeding training length.

### Key Findings
- **Theoretical learnability predictions outperform log-precision results**: Log-precision theory suggests "increasing precision," while this paper's theory suggests "increasing model size"; empirical tests clearly support the latter.
- **SCoT's spatial bound advantage exists in practice**: Since the Sudoku algorithm's space requirement is constant regardless of difficulty, the SCoT model can extrapolate to total token counts 9 orders of magnitude beyond its training length once it learns the algorithm.
- **No difference between bfloat16 and fp32 for long-context CoT**, aligning with the community experience that mixed-precision rarely hurts performance, while providing the first theoretical explanation.

## Highlights & Insights
- **The "softmax-friendly hardmax construction" philosophy is ingenious**: Instead of patching existing constructions to handle softmax, reverse-engineering a hardmax construction with "constant attention score gaps" reduces the cost of softmax conversion to logarithmic.
- **Denoising MLPs are a standard abstraction for truncating "continuous noise" into "discrete symbols"**: By limiting state space to a small discrete set (ternary), a fixed-size MLP can always clamp perturbations. This technique is applied here to accommodate attention weight rounding, completing the expressivity proof.
- **Theory provides falsifiable predictions on "depth vs. precision" for the first time**: Expressivity theory is often criticized for being "unfalsifiable." This paper conducts a clean controlled experiment, translating abstract $\mathcal{O}(\log \hat{t})$ into concrete "depth $L=6$ vs $L=8$" and "bf16 vs fp32."

## Limitations & Future Work
- **Expressivity $\neq$ Learnability**: Even if logarithmic scale + bfloat16 is "sufficiently expressive," whether SGD can find such solutions remains an open question (e.g., Liu et al. 2023 shows automaton tasks are hard to train).
- **Construction is not yet fully uniform**: Size still grows with $\log(\hat{t}/\hat{s})$. While not as "decoupled" as the constant-size construction of Merrill & Sabharwal 2024, it is more practical.
- **Empirical study is limited to Sudoku**: While algorithmic, Sudoku differs significantly from natural language reasoning; evidence for SCoT generalization on LLM tasks like math or code remains to be seen.

## Related Work & Insights
- **vs. Merrill & Sabharwal 2024**: They use constant size + $\mathcal{O}(\log \hat{t})$ precision + hardmax; Ours uses logarithmic size + $\mathcal{O}(\log\log \hat{t})$ precision + softmax. Ours is closer to practice and suggests different remedies for long-context failure.
- **vs. Yang et al. 2025 (SCoT)**: They also use the SCoT framework for expressivity but rely on hardmax + logarithmic precision. Ours uses softmax + double-logarithmic precision and proves that size must grow logarithmically to maintain its advantages.
- **Inspiration**: The "designing prototypes for conversion" approach can be used for model distillation (high-precision training $\to$ low-precision deployment); the "increasing depth instead of precision" finding has direct implications for long-context LLM design.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First Turing-complete construction matching practical precision levels, integrating SCoT.
- Experimental Thoroughness: ⭐⭐⭐⭐ Sudoku experiments are clean and powerful, but single-task focused.
- Writing Quality: ⭐⭐⭐⭐⭐ Proves clearly with asymptotic comparisons and well-defined technical outlines.
- Value: ⭐⭐⭐⭐⭐ Bridges the gap between "expressivity theory" and "actual model design choices."

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Exact Expressive Power of Transformers with Padding](../../NeurIPS2025/llm_reasoning/exact_expressive_power_of_transformers_with_padding.md)
- [\[ICML 2026\] Clustering as Reasoning: A $k$-Means Interpretation of Chain-of-Thought Graph Learning](clustering_as_reasoning_a_k-means_interpretation_of_chain-of-thought_graph_learn.md)
- [\[ICML 2026\] Hidden Error Awareness in Chain-of-Thought Reasoning: The Signal Is Diagnostic, Not Causal](hidden_error_awareness_in_chain-of-thought_reasoning_the_signal_is_diagnostic_no.md)
- [\[ICML 2026\] How Far Ahead Do LLMs Plan? Uncovering the Latent Horizon in Chain-of-Thought Reasoning](how_far_ahead_do_llms_plan_uncovering_the_latent_horizon_in_chain-of-thought_rea.md)
- [\[ICML 2026\] A Formal Comparison Between Chain of Thought and Latent Thought](a_formal_comparison_between_chain_of_thought_and_latent_thought.md)

</div>

<!-- RELATED:END -->

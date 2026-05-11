---
title: >-
  [Paper Note] Correlation Dimension of Auto-Regressive Large Language Models
description: >-
  [NeurIPS 2025][Model Compression][Correlation Dimension] This paper introduces the correlation dimension from fractal geometry into LLM analysis. By measuring the recursive structure among next-token log-probability vect…
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "Correlation Dimension"
  - "Fractal Geometry"
  - "Text Degeneration Detection"
  - "LLM Pretraining Dynamics"
  - "Hallucination Indicator"
date: 2026-05-08
content_hash: 20afd3db67be1620
---

# Correlation Dimension of Auto-Regressive Large Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2510.21258](https://arxiv.org/abs/2510.21258)
**Code**: None
**Area**: Model Compression
**Keywords**: Correlation Dimension, Fractal Geometry, Text Degeneration Detection, LLM Pretraining Dynamics, Hallucination Indicator

## TL;DR

This paper introduces the correlation dimension from fractal geometry into LLM analysis. By measuring the recursive structure among next-token log-probability vectors, it quantifies the hierarchical complexity of text, revealing a three-stage evolution of LLM pretraining, an indicator of hallucination tendency, and a unified detection capability for multiple text degeneration patterns — none of which can be captured by perplexity.

## Background & Motivation

**Background**: LLM evaluation primarily relies on two categories of metrics: local text attribute-based measures (e.g., n-gram frequency, Rep-N deduplication), which are interpretable but fail to capture deep semantic structure; and global measures (e.g., average perplexity, BERTScore), which provide holistic assessments but lack interpretability and sensitivity to local texture. A gap exists between "local interpretability" and "global comprehensiveness."

**Limitations of Prior Work**: Perplexity is the most widely used LLM evaluation metric, yet it has fundamental blind spots. A model may achieve low perplexity while still producing degenerate text exhibiting repetition, incoherence, or blandness. More critically, perplexity is insensitive to rare tokens (whose gradients are proportional to frequency) and cannot distinguish between "genuine contextual understanding" and "statistically driven generation."

**Key Challenge**: LLMs operate through token-by-token prediction, yet their emergent higher-order capabilities — such as reasoning and planning — imply complex nonlinear hierarchical mechanisms. Existing evaluation metrics either focus on the micro level (token-level) or the macro level (global), lacking a bridging measure that connects micro-level recursive structure to macro-level textual complexity.

**Goal**: To propose a computationally efficient, theoretically grounded LLM evaluation metric that simultaneously reflects local recursion and global complexity.

**Key Insight**: Natural language exhibits statistical self-similarity across multiple linguistic levels (lexical, syntactic, semantic) — analogous to fractal structures. The correlation dimension is a classical tool for measuring such self-similarity and can be computed by analyzing the distance distribution among next-token probability vectors.

**Core Idea**: Use the correlation dimension of the next-token log-probability vector sequence to quantify the hierarchical complexity perceived by LLMs, thereby addressing the blind spots of perplexity.

## Method

### Overall Architecture

Given a text passage and an autoregressive LLM, the model obtains the next-token log-probability vector $x_t \in \mathbb{R}^{|\Omega|}$ over the full vocabulary at each position $t$. Euclidean distances are computed between all pairs of vectors; the proportion of pairs with distance below threshold $\varepsilon$ is recorded as the correlation integral $S(\varepsilon)$. The correlation dimension $d$ is then extracted via the power-law relationship $S(\varepsilon) \propto \varepsilon^d$. The entire process requires only a single forward pass, with no additional training or model modification.

### Key Designs

1. **Correlation Dimension Definition Based on Log-Probability Vectors**

   - **Function**: Applies the correlation dimension from fractal geometry to the output space of LLMs.
   - **Mechanism**: At position $t$, the LLM outputs a $|\Omega|$-dimensional log-probability vector $x_t(\omega) = \log P_\theta(\omega_t = \omega | \omega_{<t})$. The correlation integral is defined as $S(\varepsilon) = \lim_{T\to\infty}\frac{2}{T(T-1)}\sum_{i<j}\mathbf{1}\{\|x_i - x_j\| < \varepsilon\}$, and the correlation dimension $d$ is the slope of $\log S(\varepsilon)$ vs. $\log \varepsilon$. Intuitively, when the probability vectors at two positions are close, the model "perceives" similar linguistic patterns at both locations — a form of recursion spanning multiple scales from the word level to the sentence level.
   - **Design Motivation**: Perplexity measures only prediction accuracy, whereas the correlation dimension measures the recursive structure of predictions — the latter reflects the hierarchical organization of text.

2. **Textual Skips as a Linguistic Interpretation of Recursion**

   - **Function**: Provides a linguistic interpretation for the mathematical concept.
   - **Mechanism**: If two positions $s, t$ have similar log-probability vectors, i.e., $\|x_s - x_t\| < \varepsilon$, then the text segment $[s, t)$ can theoretically be "skipped" without significantly affecting subsequent generation. Small $\varepsilon$ corresponds to local skips (e.g., word-level), while large $\varepsilon$ corresponds to long-range skips (e.g., sentence-level). This naturally corresponds to subtree ellipsis in Chomskyan generative grammar.
   - **Design Motivation**: Connects the purely mathematical distance threshold to interpretable linguistic structures, thereby enhancing explainability.

3. **Sufficiency of Single-Step Probability Vectors**

   - **Function**: Demonstrates that next-token probabilities alone (without multi-step time-delay embeddings) are sufficient.
   - **Mechanism**: In principle, the full state of an LLM encodes distributions over all future tokens, making single-step probabilities only partial information. The authors construct time-delay embeddings $y_t = [x_t; x_{t+1}; ...; x_{t+k}]$ via a stochastic extension of Takens' embedding theorem, but experiments show that the correlation dimension is virtually identical for $k=1$ and $k>1$ — indicating that single-step probability vectors already implicitly encode long-range structural information. This is consistent with findings in knowledge distillation, where single-step probability distributions effectively summarize model knowledge.
   - **Design Motivation**: Ensures computational efficiency — only a single forward pass is required.

### Computational Optimizations

GPU kernel fusion and vocabulary reduction together achieve over 10× speedup with zero additional memory overhead. Under 4-bit quantization (GPTQ/AWQ), the correlation dimension changes by less than 3%, ensuring practical usability in production environments.

## Key Experimental Results

### Main Results

| Experiment | Key Data | Notes |
|---|---|---|
| Natural language correlation dimension | ~6.5 (consistent across models) | GPT2/Pythia/Falcon3/OpenLLaMA/Yi1.5/Mamba converge to a consistent value on the SEP dataset |
| Programming language correlation dimension | ~5.0 | Consistent across Python/Java/C; lower than natural language |
| Randomly shuffled text | >10 | High dimension reflects lack of structure |
| Repetitive text | <2.0 | Low dimension reflects simple patterns |
| Pólya urn process | <2.0 | Self-reinforcing process yields extremely low dimension |

### Ablation Study (Degeneration Detection)

| Text Type | Correlation Dimension (Falcon3-10B) | Perplexity | Notes |
|---|---|---|---|
| Normal text | 5.04 | 10.79 | Baseline |
| Repetitive degeneration | 3.80 (p=9.5E-7) | 1.25 | Both metrics detect it; directions agree |
| Incoherent degeneration | 3.96 (p=2.9E-6) | 13.24 | Perplexity rises but correlation dimension drops — directions diverge! |
| Bland degeneration | 4.51 (p=1.1E-3) | 4.24 | Perplexity decreases and so does correlation dimension |

### Key Findings

- **Three-Stage Evolution of LLM Pretraining**: (1) Correlation dimension decreases rapidly (learning short-range structures such as bigrams); (2) dimension rises (model begins capturing long-range dependencies); (3) dimension slowly decreases (generalization compression). This dynamics is entirely invisible in the monotonic decrease of perplexity. Small models (Pythia-14M/160M) exhibit a dimension increase in stage three, coinciding with the collapse of in-context learning capability.
- **Distinguishing Hallucination from Recall**: On knowledge-intensive texts, models that successfully recall facts exhibit significantly higher correlation dimensions than those that hallucinate — Falcon3-7B (6.68) vs. Qwen2.5-32B (4.42) — suggesting that recall requires long-range dependencies (high dimension) whereas hallucination relies on format imitation (low dimension).
- **Stress Test Validation**: When using random text as a prompt for model continuation, the Spearman correlation between correlation dimension and HelloEval scores reaches 0.952, indicating that correlation dimension serves as an intrinsic indicator of robustness in long-form generation.

## Highlights & Insights

- **Unified degeneration detection** is the paper's most significant contribution: among existing metrics, Rep-N detects only repetition; BERTScore/MAUVE detect only incoherence; no single metric can simultaneously detect all three degeneration types. The correlation dimension is the first — and possibly the only — metric capable of unifying the detection of repetition, incoherence, and blandness.
- **The three-stage pretraining discovery** is highly insightful: the monotonic decrease in perplexity conceals the nonlinear internal evolution from "learning short-range patterns → exploring long-range dependencies → compression and generalization." The anomalous dimension increase in small models during stage three coincides with capability collapse, which could inform training decisions such as early stopping.
- **The Japanese dual-script experiment** elegantly validates that correlation dimension measures semantic complexity rather than surface lexical repetition: kanji+kana vs. pure kana yields a vocabulary size difference of 10×, yet a correlation dimension difference of only 5.7%, compared to Rep-N's difference of 29.8%.

## Limitations & Future Work

- The empirically observed convergence value (~6.5) of the correlation dimension lacks a theoretical explanation — why does natural language's correlation dimension converge to this specific value, and what intrinsic property of language does it reflect?
- The method requires access to full logits (the complete vocabulary probability distribution), making it inapplicable to closed-source models (e.g., GPT-4).
- The degeneration detection experiments rely on GPT-4o-generated controlled data (20 questions × 10 normal/degenerate responses), which is relatively small in scale and depends on human-defined notions of what constitutes "normal" or "degenerate" text.
- The correlation dimension is a global statistic and cannot localize where degeneration occurs within a text. Application to online generation control would require investigation of a sliding-window variant.

## Related Work & Insights

- **vs. Alabdulmohsin et al. (NeurIPS 2024)**: That work also applies fractal dimensions and the Hurst exponent to analyze LLMs, but measures long-range dependencies in the cumulative log-perplexity sequence rather than the recursive structure of the generation process itself — capturing information at a different level.
- **vs. Standard Perplexity**: Perplexity measures local prediction accuracy; correlation dimension measures global recursive complexity. The two are complementary rather than substitutes. Perplexity can be low while degeneration is severe (e.g., bland text), a situation that correlation dimension successfully captures.
- **vs. MAUVE (Pillutla et al.)**: MAUVE measures the divergence between generated and reference distributions via KL divergence, whereas correlation dimension requires no reference distribution — it is an intrinsic property of the text itself.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Introducing the correlation dimension from fractal geometry into LLM evaluation represents an entirely new perspective with solid theoretical foundations.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers multiple dimensions including pretraining dynamics, context dependence, hallucination detection, degeneration detection, and stress testing.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Progresses logically from physical intuition to mathematical definition to linguistic interpretation, with excellent readability.
- **Value**: ⭐⭐⭐⭐⭐ Provides a complementary perspective that perplexity cannot replace, with direct practical value for LLM training monitoring and generation quality control.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Restoring Pruned Large Language Models via Lost Component Compensation](restoring_pruned_large_language_models_via_lost_component_compensation.md)
- [\[NeurIPS 2025\] The Structure of Relation Decoding Linear Operators in Large Language Models](the_structure_of_relation_decoding_linear_operators_in_large_language_models.md)
- [\[NeurIPS 2025\] PermLLM: Learnable Channel Permutation for N:M Sparse Large Language Models](permllm_learnable_channel_permutation_for_nm_sparse_large_language_models.md)
- [\[NeurIPS 2025\] A Simple Linear Patch Revives Layer-Pruned Large Language Models](a_simple_linear_patch_revives_layerpruned_large_language_mod.md)
- [\[NeurIPS 2025\] LayerIF: Estimating Layer Quality for Large Language Models using Influence Functions](layerif_estimating_layer_quality_for_large_language_models_using_influence_funct.md)

</div>

<!-- RELATED:END -->

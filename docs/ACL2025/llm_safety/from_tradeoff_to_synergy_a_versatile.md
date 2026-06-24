---
title: >-
  [Paper Note] From Trade-off to Synergy: A Versatile Symbiotic Watermarking Framework for Large Language Models
description: >-
  [ACL 2025][LLM Safety][LLM watermarking] This paper proposes SymMark, a symbiotic watermarking framework that integrates logits-based and sampling-based watermarking methods (via three strategies: serial, parallel, and hybrid). By adaptively selecting watermarking strategies using token entropy and semantic entropy, it achieves SOTA performance in terms of detectability, robustness, text quality, and security.
tags:
  - "ACL 2025"
  - "LLM Safety"
  - "LLM watermarking"
  - "symbiotic watermark"
  - "token entropy"
  - "semantic entropy"
  - "text traceability"
date: 2026-05-08
content_hash: 315c720eb4d32af6
---

# From Trade-off to Synergy: A Versatile Symbiotic Watermarking Framework for Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2505.09924](https://arxiv.org/abs/2505.09924)  
**Code**: [https://github.com/redwyd/SymMark](https://github.com/redwyd/SymMark)  
**Area**: AI Safety  
**Keywords**: LLM watermarking, symbiotic watermark, token entropy, semantic entropy, text traceability

## TL;DR

This paper proposes SymMark, a symbiotic watermarking framework that integrates logits-based and sampling-based watermarking methods (via three strategies: serial, parallel, and hybrid). By adaptively selecting watermarking strategies using token entropy and semantic entropy, it achieves SOTA performance in terms of detectability, robustness, text quality, and security.

## Background & Motivation
**Background**: LLM watermarking can be categorized into two major types: logits-based (e.g., KGW, which modifies logit distributions) and sampling-based (e.g., AAR, which alters the sampling process).

**Limitations of Prior Work**: Both methodologies have distinct advantages and drawbacks. Logits-based methods are robust but degrade text quality, whereas sampling-based methods preserve text quality but exhibit weaker detectability. Additionally, both suffer from security vulnerabilities such as watermark extraction attacks.

**Key Challenge**: A fundamental trade-off exists among robustness, text quality, and security, making simultaneous optimization challenging.

**Goal**: To integrate the two types of watermarking methods, shifting from trade-off to synergy.

**Key Insight**: Drawing inspiration from symbiotic relationships in nature, this work designs three integration strategies and employs entropy metrics for adaptive selection.

**Core Idea**: Token entropy is used to determine whether to embed a logits watermark, and semantic entropy determines whether to embed a sampling watermark, achieving adaptive hybrid watermark embedding.

## Method

### Overall Architecture
SymMark provides three symbiotic strategies: Serial (embedding both watermarks simultaneously for each token), Parallel (alternating embedding at odd and even positions), and Hybrid (adaptive selection based on entropy), alongside a unified detection algorithm. The experiments default to using Unigram as the logits watermark and AAR as the sampling watermark, validated across three model families: OPT, LLaMA, and GPT-J.

### Key Designs
1. **Serial strategy**: $y_t = \mathcal{S}_w(\text{softmax}(\mathcal{A}_w(l_t)))$, which modifies the logits first and then applies watermarked sampling. This maximizes the watermark signal but may affect text quality.
2. **Parallel strategy**: Uses the logits watermark with original sampling at odd positions, and the original logits with the sampling watermark at even positions. This independent embedding reduces mutual interference.
3. **Hybrid strategy (Core)**: Evaluates two entropy criteria—embedding the digits watermark when the token entropy $H_{TE}$ is higher than a threshold $\alpha$ (modifying logits has less impact when model uncertainty is high), and embedding the sampling watermark when the semantic entropy $H_{SE}$ is below a threshold $\beta$ (replacing tokens has less impact when candidate semantics are similar).

### Loss & Training
- A training-free method that embeds watermarks directly during inference.
- The Hybrid strategy uses K-means clustering ($k=64$, $n=10$ clusters) to calculate semantic entropy.
- Default hyperparameters: token entropy threshold $\alpha=1.0$, semantic entropy threshold $\beta=0.5$.
- Detection uses logical OR: $I = I_l \mid I_s$ (a text is determined to be watermarked if either watermark is detected).

## Key Experimental Results

### Main Results (Detectability on C4 dataset with OPT-6.7B)

| Method | TPR | TNR | F1 | AUC |
|------|-----|-----|-----|-----|
| KGW (logits) | 0.990 | 1.000 | 0.994 | 0.999 |
| Unigram (logits) | 0.995 | 1.000 | 0.997 | 0.998 |
| AAR (sampling) | 0.995 | 1.000 | 0.997 | 0.999 |
| EXP (sampling) | 0.975 | 0.925 | 0.951 | 0.960 |
| **SymMark-Serial** | **1.000** | **1.000** | **1.000** | **1.000** |
| **SymMark-Hybrid** | **1.000** | **1.000** | **1.000** | **1.000** |

### Ablation Study (Comparison of Strategy Characteristics)

| Strategy | Detectability | Robustness | Text Quality | Security |
|------|---------|--------|---------|-------|
| Serial | Optimal | Optimal | Poor | Fair |
| Parallel | Moderate | Moderate | Optimal | Fair |
| Hybrid | Excellent | Excellent | Excellent | Optimal |

### Key Findings
- SymMark achieves a perfect detectability of $F1=1.000$ on both the C4 and OpenGen datasets.
- It consistently maintains robust advantages across three model families (OPT, LLaMA, GPT-J).
- It shows distinct advantages compared to sampling methods like EXP ($F1=0.951$) and ITS ($F1=0.957$).
- Serial is optimal in detectability and robustness (due to the superposition of dual watermark signals).
- Parallel performs best in text quality (alternating embedding reduces mutual interference).
- Hybrid achieves the best comprehensive performance, adaptively balancing all metrics via entropy.
- Semantic entropy effectively identifies when embedding the sampling watermark will not distort semantics.
- The unified detection algorithm can detect watermarks from all three strategies simultaneously.

## Highlights & Insights
- This work is the first to systematically explore the integration of logits-based and sampling-based watermarking, pioneering the symbiotic watermarking paradigm.
- The paradigm shift from trade-off to synergy offers general inspiration, showing that the strengths of different methods can complement rather than exclude each other.
- The dual-entropy criterion is cleverly designed: token entropy controls the logits watermark (altering logits when uncertainty is high has less impact), while semantic entropy governs the sampling watermark (substituting words when semantics are similar has less impact).
- The Hybrid strategy achieves the Pareto-optimal balance across four dimensions: detectability, robustness, quality, and security.
- The unified detection algorithm is simple and efficient: it determines presence if either watermark is detected, ensuring low false positive rates through a logical OR operation.
- It achieves comprehensive dominance over 11 baseline methods, validated by large-scale experiments.

## Limitations & Future Work
- The computation of semantic entropy relies on K-means clustering (top-64 token embeddings), introducing additional computational overhead.
- It requires a model with the same tokenizer as the original model for semantic clustering, which limits its general applicability.
- Watermark detection on long texts is not fully validated, as experimental text length is fixed at 200±30 tokens.
- Robustness under stronger adversarial attacks (e.g., model distillation, paraphrase attacks) needs further investigation.
- The choice of hyperparameters $\alpha$ and $\beta$ impacts performance, potentially requiring tuning for different scenarios.
- Multi-bit watermarking scenarios have not been explored (currently limited to 1-bit detection: presence or absence of watermark).

## Related Work & Insights
- Bridges the gap between two foundational methods: KGW (Kirchenbauer et al., 2023) and AAR (Aaronson, 2023).
- SynthID's (Dathathri et al., 2024) tournament sampling represents another high-quality sampling direction.
- Designing watermarking strategies from an information-theoretic perspective (Shannon entropy and semantic entropy) is highly inspiring.
- The concept of leveraging entropy to optimize a single method from SWEET (Lee et al., 2024) and EWD (Lu et al., 2024) is extended in this work to a fused framework.
- Provides direct practical value for AIGC regulation and intellectual property protection.

## Rating
- Novelty: ⭐⭐⭐⭐ The integration of the two types of watermarking is highly novel, though the individual components are existing methods.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across multiple datasets, models, and baselines (11 baselines), covering four key dimensions.
- Writing Quality: ⭐⭐⭐⭐ Well-structured and highly readable, presenting the three strategies in a progressive manner.
- Value: ⭐⭐⭐⭐ Provides practical advancements to the LLM watermarking field.
- Overall Evaluation: Strong engineering, high practical value, and open-source code for ease of reproduction.
- Application Scenarios: AIGC regulation, copyright protection, academic integrity detection.
- Reproducibility: Code is open-sourced (SymMark) and can be directly integrated into existing LLM services.
- Extensibility: Future work can explore the integration of more watermarking types (e.g., sentence-level + token-level).
- Open Questions: How to defend against more complex paraphrase attacks while maintaining watermark strength?
- Impact: Establishes a new "synergy over trade-off" paradigm for designing future watermarking methodologies.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] MorphMark: Flexible Adaptive Watermarking for Large Language Models](morphmark_adaptive_watermarking.md)
- [\[NeurIPS 2025\] Learning to Watermark: A Selective Watermarking Framework for Large Language Models via Multi-Objective Optimization](../../NeurIPS2025/llm_safety/learning_to_watermark_a_selective_watermarking_framework_for_large_language_mode.md)
- [\[ICLR 2026\] Improving the Trade-off Between Watermark Strength and Speculative Sampling Efficiency for Language Models](../../ICLR2026/llm_safety/improving_the_trade-off_between_watermark_strength_and_speculative_sampling_effi.md)
- [\[ACL 2025\] Robust Data Watermarking in Language Models by Injecting Fictitious Knowledge](robust_data_watermarking_in_language_models_by_injecting_fictitious_knowledge.md)
- [\[ACL 2025\] Ensemble Watermarks for Large Language Models](ensemble_watermarks_llm.md)

</div>

<!-- RELATED:END -->

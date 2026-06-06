---
title: >-
  [Paper Note] Working Memory Constraints Scaffold Learning in Transformers under Data Scarcity
description: >-
  [ACL 2026][LLM Pretraining][Working Memory] This paper integrates human working memory constraints (fixed window, exponential decay, logistic decay, and primacy-recency effects) into the GPT-2 attention mechanism…
tags:
  - "ACL 2026"
  - "LLM Pretraining"
  - "Working Memory"
  - "Attention Constraints"
  - "Inductive Bias"
  - "Data Scarcity"
  - "Cognitive Alignment"
date: 2026-05-08
content_hash: ab0e110d654a40d1
---

# Working Memory Constraints Scaffold Learning in Transformers under Data Scarcity

**Conference**: ACL 2026
**arXiv**: [2604.20789](https://arxiv.org/abs/2604.20789)  
**Code**: None  
**Area**: Cognitive Modeling / Language Model Pretraining
**Keywords**: Working Memory, Attention Constraints, Inductive Bias, Data Scarcity, Cognitive Alignment

## TL;DR

This paper integrates human working memory constraints (fixed window, exponential decay, logistic decay, and primacy-recency effects) into the GPT-2 attention mechanism, training from scratch on developmentally plausible small-scale corpora (10M/100M words). The results demonstrate that these constraints significantly improve grammatical accuracy and human reading time predictability under data scarcity, while also promoting functional specialization of attention heads.

## Background & Motivation

**Background**: The self-attention mechanism in standard Transformers permits nearly uniform access to all tokens within the context window, lacking inherent structural biases that reflect human cognitive limitations. Although models such as GPT-2 correlate well with human reading times and neural activity, this correlation may arise *despite* rather than *because of* the architecture.

**Limitations of Prior Work**: Existing efforts to incorporate cognitive constraints into Transformers are either post-hoc modifications (e.g., applying decay biases to pretrained models), inference-time context truncation, or soft constraints (e.g., ALiBi). None of these approaches allow the constraints to shape representational learning from the very beginning of training.

**Key Challenge**: The NLP field trends toward longer contexts and weaker inductive biases, yet human language processing is strongly constrained by limited working memory capacity — raising the question of whether such constraints might actually facilitate learning.

**Goal**: (1) Systematically compare four cognitively inspired attention constraints in from-scratch training; (2) Evaluate their effects on grammatical competence and cognitive alignment under data-scarce conditions.

**Key Insight**: The BabyLM challenge's developmentally plausible datasets (10M/100M words) are used to train models at a data scale comparable to human language acquisition, providing a more meaningful testbed for cognitive constraint hypotheses.

**Core Idea**: Hard cognitive constraints — particularly fixed-window attention — serve as inductive biases that actively *scaffold* learning under data scarcity rather than impeding it.

## Method

### Overall Architecture

Four attention variants are implemented on the GPT-2-small architecture and trained from scratch on the BabyLM 10M and 100M word corpora. Grammatical competence is evaluated using BLiMP, and alignment with human processing data is assessed via psychometric benchmarks.

### Key Designs

1. **Fixed Window Attention**:

    - Function: Simulates the limited capacity of working memory.
    - Mechanism: Each token attends only to the preceding $W$ tokens; positions outside the window are masked to $-\infty$ (i.e., zero after softmax). Window sizes $W \in \{4, 5, 7, 9\}$ correspond to Cowan's 4-chunk theory and Miller's $7 \pm 2$ theory, respectively.
    - Design Motivation: Forces the model to operate within strictly local context, isolating the learning effects of local dependencies.

2. **Exponential / Logistic Decay Attention**:

    - Function: Simulates the recency effect — recent information is highly accessible while distant information is gradually forgotten.
    - Mechanism: Exponential decay blends raw attention weights with a distance decay factor: $a'_{ij} = (1-\alpha)a_{ij} + \alpha e^{-|i-j| \cdot \lambda}$. Logistic decay applies a sigmoid curve to produce a pattern of sustained access followed by an abrupt drop: $w_{ij} = 1/(1 + e^{k(d_{ij} - m)})$.
    - Design Motivation: Exponential decay provides a continuous decline, while logistic decay offers a "sustain-then-drop" pattern more analogous to discrete capacity limits.

3. **Primacy-Recency Attention**:

    - Function: Simulates the memory advantage for items at the beginning and end of a sequence.
    - Mechanism: Two learnable parameters, $w_{\text{primacy}}$ and $w_{\text{recency}}$, control exponential decay biases toward the beginning and end of the sequence, respectively, which are superimposed on the standard attention weights.
    - Design Motivation: Human recall rates for list items are higher at the beginning and end than in the middle.

### Loss & Training

Standard language modeling loss (next-token prediction). AdamW optimizer with lr=5e-5, trained for 5 epochs. All variants share identical hyperparameters.

## Key Experimental Results

### Main Results

**BLiMP Average Accuracy**

| Model | 10M Data | 100M Data |
|---|---|---|
| Baseline GPT-2 | ~61% | ~71% |
| Fixed Window 5 | **~68%** | ~72% |
| Exponential Decay | ~65% | ~71% |
| Logistic Decay | ~66% | ~72% |
| Primacy-Recency | ~63% | ~71% |

**Psychometric Alignment (ΔLog-Likelihood)**

| Model | 10M | 100M |
|---|---|---|
| Baseline | ~3.2 | slightly higher |
| Fixed Window 7/9 | **~6.0** | decreased |

### Key Findings

- **Constraints are most effective under data scarcity**: At 10M words, Fixed Window 5 outperforms the baseline by approximately 7 percentage points; at 100M words, the gap narrows to 1–2 points.
- Constrained models excel on Argument Structure; Fixed Window 5 approaches the performance of pretrained GPT-2-large on this category.
- **Counterintuitive finding**: Models with highly local attention (window of only 5 tokens) also perform well on non-local linguistic phenomena such as Binding and Argument Structure.
- Psychometric alignment *decreases* at 100M words — all models (including the baseline) show reduced alignment with human processing as data increases, supporting the hypothesis that the language modeling objective does not asymptotically converge toward human comprehension.
- Attention visualizations reveal that constrained models develop functionally specialized attention heads (e.g., subject-verb-object heads, verb-specific heads, noun-specific heads), whereas the baseline exhibits diffuse, unspecialized attention patterns.

## Highlights & Insights

- The central argument that "cognitive constraints scaffold rather than hinder learning" is compelling — it directly challenges the prevailing NLP trend of pursuing ever-longer contexts.
- The strong performance of fixed-window models on non-local tasks is the most surprising finding, suggesting that local constraints compel the model to develop more explicit syntactic encodings.
- The finding that more data reduces psychometric alignment provides new evidence for ongoing debates on whether large language models genuinely understand language.

## Limitations & Future Work

- Experiments are conducted solely on GPT-2-small; generalizability to large-scale models remains unknown.
- Validation is limited to English; languages with free word order or head-final typology may yield different results.
- The simplified implementation of working memory (fixed window) is far less sophisticated than real cognitive systems.
- Phenomena such as Island Effects remain unimproved, indicating that architectural constraints alone cannot account for all linguistic phenomena.

## Related Work & Insights

- **vs. ALiBi**: ALiBi uses soft biases to discourage but not prohibit long-range attention; this paper employs hard constraints that entirely block such attention.
- **vs. De Varda & Marelli (2024)**: Their approach applies post-hoc decay to pretrained models, whereas this paper trains from scratch so that constraints shape the learning process from the outset.

## Rating

- Novelty: ⭐⭐⭐⭐ — Valuable interdisciplinary bridging from cognitive science to computational linguistics, though individual components are not entirely novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Multi-faceted validation via BLiMP, psychometric benchmarks, attention visualization, and syntactic probing.
- Writing Quality: ⭐⭐⭐⭐⭐ — Well-argued, with thorough experimental analysis and an honest discussion of limitations.
- Value: ⭐⭐⭐⭐ — Important implications for cognitively inspired model design and data-efficient learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Data Mixing Agent: Learning to Re-weight Domains for Continual Pre-training](data_mixing_agent_learning_to_re-weight_domains_for_continual_pre-training.md)
- [\[ACL 2026\] SAGE: Sign-Adaptive Gradient for Memory-Efficient LLM Optimization](sage_sign-adaptive_gradient_for_memory-efficient_llm_optimization.md)
- [\[NeurIPS 2025\] Neural Collapse under Gradient Flow on Shallow ReLU Networks for Orthogonally Separable Data](../../NeurIPS2025/llm_pretraining/neural_collapse_under_gradient_flow_on_shallow_relu_networks_for_orthogonally_se.md)
- [\[NeurIPS 2025\] Memory Mosaics at Scale](../../NeurIPS2025/llm_pretraining/memory_mosaics_at_scale.md)
- [\[NeurIPS 2025\] Does Object Binding Naturally Emerge in Large Pretrained Vision Transformers?](../../NeurIPS2025/llm_pretraining/does_object_binding_naturally_emerge_in_large_pretrained_vision_transformers.md)

</div>

<!-- RELATED:END -->

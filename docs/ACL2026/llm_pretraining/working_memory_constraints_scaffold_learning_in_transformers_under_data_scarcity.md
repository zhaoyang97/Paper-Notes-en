---
title: >-
  [Paper Note] Working Memory Constraints Scaffold Learning in Transformers under Data Scarcity
description: >-
  [ACL 2026][LLM Pretraining][Working Memory] This paper integrates human working memory constraints (fixed window, exponential decay, logistic decay…
tags:
  - "ACL 2026"
  - "LLM Pretraining"
  - "Working Memory"
  - "Attention Constraints"
  - "Inductive Bias"
  - "Data Scarcity"
  - "Cognitive Alignment"
date: 2026-05-08
content_hash: ae6f8c1684d80e84
---

# Working Memory Constraints Scaffold Learning in Transformers under Data Scarcity

**Conference**: ACL 2026  
**arXiv**: [2604.20789](https://arxiv.org/abs/2604.20789)  
**Code**: None  
**Area**: Cognitive Modeling / LLM Pre-training  
**Keywords**: Working Memory, Attention Constraints, Inductive Bias, Data Scarcity, Cognitive Alignment

## TL;DR

This paper integrates human working memory constraints (fixed window, exponential decay, logistic decay, and primacy-recency effects) into the GPT-2 attention mechanism. By training from scratch on developmentally plausible small-scale corpora (10M/100M words), the authors find that these constraints significantly improve grammatical accuracy and the predictability of human reading times under data scarcity, while also promoting functional specialization of attention heads.

## Background & Motivation

**Background**: The self-attention mechanism in standard Transformers allows for nearly uniform access to all tokens within the context window, lacking intrinsic structural biases that reflect human cognitive limitations. While models like GPT-2 correlate well with human reading times and neural activity, this may occur "in spite of" rather than "because of" their architecture.

**Limitations of Prior Work**: Existing efforts to introduce cognitive constraints into Transformers are often post-hoc modifications (e.g., imposing decay biases on pre-trained models), involve truncating context during inference, or rely on soft constraints (e.g., ALiBi). None of these methods allow the constraints to shape representation learning from the onset of training.

**Key Challenge**: The field of NLP tends toward longer contexts and weaker inductive biases, yet human language processing is strictly constrained by limited working memory capacity. Does this constraint actually facilitate learning?

**Goal**: (1) Systematically compare four cognitively-inspired attention constraints during training from scratch; (2) Evaluate their impact on grammatical competence and cognitive alignment in data-scarce scenarios.

**Key Insight**: By utilizing developmentally plausible datasets from the BabyLM Challenge (10M/100M words) and training at a scale comparable to human language acquisition, the cognitive constraint hypothesis becomes more comparable.

**Core Idea**: Rigid cognitive constraints (especially fixed-window attention) act as inductive biases that actively "scaffold" learning during data scarcity, rather than hindering it.

## Method

### Overall Architecture

Four attention variants are implemented on the GPT-2-small architecture and trained from scratch on BabyLM's 10M and 100M word corpora. Grammatical ability is evaluated using BLiMP, and alignment with human processing data is assessed using psychometric benchmarks.

### Key Designs

1.  **Fixed Window Attention**:
    - **Function**: Simulates the limited capacity of working memory.
    - **Mechanism**: Attention for each token is only calculated over the preceding $W$ tokens; positions outside the window are set to $-\infty$ (vanishing to zero after softmax). Window sizes $W \in \{4, 5, 7, 9\}$ correspond to Cowan’s 4-chunk theory and Miller’s $7 \pm 2$ theory.
    - **Design Motivation**: Forces the model to operate within a strictly local context to isolate the learning effects of local dependencies.

2.  **Exponential/Logistic Decay Attention**:
    - **Function**: Simulates the recency effect—recent information is highly accessible, while distal information gradually fades.
    - **Mechanism**: Exponential decay mixes raw attention weights with a distance decay factor: $a'_{ij} = (1-\alpha)a_{ij} + \alpha e^{-|i-j| \cdot \lambda}$. Logistic decay uses a sigmoid curve to maintain information before a sharp drop: $w_{ij} = 1/(1 + e^{k(d_{ij} - m)})$.
    - **Design Motivation**: Exponential decay provides a continuous decline, while logistic decay offers a "hold-then-drop" pattern closer to discrete capacity models.

3.  **Primacy-Recency Attention**:
    - **Function**: Simulates memory advantages for the beginning and end of sequences.
    - **Mechanism**: Two trainable parameters $w_{\text{primacy}}$ and $w_{\text{recency}}$ are learned to control exponential decay biases at the start and end of sequences, which are then superimposed onto standard attention weights.
    - **Design Motivation**: Humans exhibit higher recall rates for the first and last items in a list compared to middle items.

### Loss & Training

Standard language modeling loss (next-token prediction). AdamW optimizer, lr=5e-5, trained for 5 epochs. All variants use identical hyperparameters.

## Key Experimental Results

### Main Results

**BLiMP Average Accuracy**

| Model | 10M Data | 100M Data |
|------|---------|----------|
| Baseline GPT-2 | ~61% | ~71% |
| Fixed Window 5 | **~68%** | ~72% |
| Exponential Decay | ~65% | ~71% |
| Logistic Decay | ~66% | ~72% |
| Primacy-Recency | ~63% | ~71% |

**Psychometric Alignment (ΔLog-Likelihood)**

| Model | 10M | 100M |
|------|-----|------|
| Baseline | ~3.2 | Slightly higher |
| Fixed Window 7/9 | **~6.0** | Lower |

### Key Findings

-   **Constraints are most effective under data scarcity**: With 10M words, Fixed Window 5 outperforms the baseline by approximately 7 percentage points; at 100M words, the gap narrows to 1-2 percentage points.
-   Constrained models perform exceptionally well on Argument Structure, with Fixed Window 5 nearly reaching the level of pre-trained GPT-2-large.
-   **Counter-intuitive Finding**: Extremely local models (window of only 5 tokens) perform well even on non-local linguistic phenomena (e.g., Binding, Argument Structure).
-   Psychometric alignment actually decreases at 100M—all models (including the baseline) show lower alignment with human processing as data increases, supporting the hypothesis that language modeling objectives do not asymptotically converge with human understanding.
-   Attention visualization shows that constrained models develop functional specialization in attention heads (e.g., subject-verb-object heads, verb-specific heads, noun-specific heads), whereas the baseline attention distribution remains diffuse and unspecialized.

## Highlights & Insights

-   The core argument that "cognitive constraints are a scaffold, not a hindrance" is powerful, challenging the mainstream NLP trend of pursuing ever-longer contexts.
-   The finding that Fixed Window models perform well even on non-local tasks is surprising, suggesting that local constraints force the model to develop more explicit syntactic encodings.
-   The observation that more data reduces psychometric alignment provides new evidence for discussions regarding whether large models truly "understand" language.

## Limitations & Future Work

-   Experiments were limited to GPT-2-small; it is unknown if the conclusions scale to larger models.
-   Validations were only conducted in English; free word order or head-final languages might yield different results.
-   The implementation of working memory (fixed window) is significantly simpler than the actual human cognitive system.
-   Phenomena like Island Effects remain unimproved, indicating that architectural constraints cannot solve all linguistic phenomena.

## Related Work & Insights

-   **vs ALiBi**: ALiBi uses soft biases to discourage but not prohibit long-range attention, whereas this work uses hard constraints to block it entirely.
-   **vs De Varda & Marelli (2024)**: They applied post-hoc decay to pre-trained models; this work trains from scratch to allow constraints to shape the learning process.

## Rating

-   Novelty: ⭐⭐⭐⭐ The interdisciplinary bridge from cognitive science to computational linguistics is valuable, though the individual components are not entirely new.
-   Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive validation via BLiMP, psychometrics, attention visualization, and syntactic probing.
-   Writing Quality: ⭐⭐⭐⭐⭐ Strong argumentation, in-depth experimental analysis, and honest discussion of limitations.
-   Value: ⭐⭐⭐⭐ Provides important insights for cognitively-inspired model design and data-efficient learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] FOREVER: Forgetting Curve-Inspired Memory Replay for Language Model Continual Learning](forever_forgetting_curve-inspired_memory_replay_for_language_model_continual_lea.md)
- [\[ACL 2026\] SAGE: Sign-Adaptive Gradient for Memory-Efficient LLM Optimization](sage_sign-adaptive_gradient_for_memory-efficient_llm_optimization.md)
- [\[ACL 2026\] Data Mixing Agent: Learning to Re-weight Domains for Continual Pre-training](data_mixing_agent_learning_to_re-weight_domains_for_continual_pre-training.md)
- [\[NeurIPS 2025\] Neural Collapse under Gradient Flow on Shallow ReLU Networks for Orthogonally Separable Data](../../NeurIPS2025/llm_pretraining/neural_collapse_under_gradient_flow_on_shallow_relu_networks_for_orthogonally_se.md)
- [\[NeurIPS 2025\] Memory Mosaics at Scale](../../NeurIPS2025/llm_pretraining/memory_mosaics_at_scale.md)

</div>

<!-- RELATED:END -->

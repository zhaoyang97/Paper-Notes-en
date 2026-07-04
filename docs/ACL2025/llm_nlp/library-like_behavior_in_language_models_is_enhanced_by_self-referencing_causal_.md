---
title: >-
  [Paper Note] ReCall: Library-Like Behavior In Language Models is Enhanced by Self-Referencing Causal Cycles
description: >-
  [ACL2025][LLM (Other)][reversal curse] This paper introduces the concept of "Self-Referencing Causal Cycles" (ReCall), revealing how naturally occurring repetitive token sequences in LLM pre-training data form circular references. This enables autoregressive models to bypass unidirectional causal constraints and overcome the reversal curse. Based on this, a two-step ReCall-aware prompting strategy is designed.
tags:
  - "ACL2025"
  - "LLM (Other)"
  - "reversal curse"
  - "causal cycle"
  - "cycle tokens"
  - "autoregressive model"
  - "information retrieval"
  - "prompting"
date: 2026-05-08
content_hash: 4e6c57940ee33139
---

# ReCall: Library-Like Behavior In Language Models is Enhanced by Self-Referencing Causal Cycles

**Conference**: ACL2025  
**arXiv**: [2501.13491](https://arxiv.org/abs/2501.13491)  
**Code**: [samunaai/remember](https://github.com/samunaai/remember)  
**Area**: LLM/NLP  
**Keywords**: reversal curse, causal cycle, cycle tokens, autoregressive model, information retrieval, prompting  

## TL;DR

This paper introduces the concept of "Self-Referencing Causal Cycles" (ReCall), revealing how naturally occurring repetitive token sequences in LLM pre-training data form circular references. This enables autoregressive models to bypass unidirectional causal constraints and overcome the reversal curse. Based on this, a two-step ReCall-aware prompting strategy is designed.

## Background & Motivation

The **Reversal Curse** is a well-known limitation of autoregressive language models: after being trained on "A is B", the model fails to correctly infer "B is A". For example, an LLM can correctly state that the line following "Gave proof through the night..." is "O say does that star-spangled banner yet wave", but fails to provide the correct answer when asked for the preceding line.

This phenomenon stems from the unidirectional causality of autoregressive models—models generate subsequent tokens based on preceding tokens, requiring knowledge to be learned and recalled in a consistent token order. Formally:

$$S_r = \arg\max_{s \in \mathcal{S}} P_\mathcal{M}(s|S_l) \quad \text{(easy)}$$
$$S_l \neq \arg\max_{s \in \mathcal{S}} P_\mathcal{M}(s|S_r) \quad \text{(hard)}$$

Existing solutions primarily rely on data augmentation (such as token permutation, reversal training, and other manual interventions). The authors propose a fresh perspective: **the reversal curse is not always an insurmountable barrier—naturally occurring patterns in pre-training data are sufficient to mitigate this issue**.

Core Analogy: The LLM is analogy-mapped to a library, where prompts serve as cross-referencing indexes. Frequently recurring token sequences in pre-training data (such as poem titles, song names, etc.) act like "hyperlinks" that connect different sections of the text.

## Method

### 1. Self-Referencing Causal Cycles

**Core Concept — Cycle Token**:

Consider an original sequence $\mathcal{S}_{seq} = [e_1, e_2, ..., e_n]$, split at position i into a left segment $S_l = [e_1, ..., e_i]$ and a right segment $S_r = [e_{i+1}, ..., e_n]$.

Directly recovering $S_l$ from $S_r$ requires traversing all possible sequences, which is computationally intractable. The key insight is: if $e_1$ (or a specific token sequence) **frequently recurs** in the text, it acts as a "cycle token"—connecting the end of the sequence back to its beginning, thereby forming a causal cycle.

Constructing a modified sequence: $S_r' = [e_{i+1}, ..., e_n, e_1]$ (appending $e_1$ to the end of $S_r$). Starting from $S_r'$, the model can continue to predict $S_l' = [e_2, e_3, ..., e_i]$, and when i is sufficiently large, $S_l' \approx S_l$.

**Intuition**: Cycle tokens function like "anchors" on a webpage or "cross-references" in a book, allowing the model to "jump" back to earlier parts of the text within a unidirectional generation flow.

### 2. Formal Framework

Given the right segment $S_r$, candidate set $S_{l_c}$ is generated via cycle tokens, from which the optimal sequence is selected:

$$S_l = \arg\max_{s \in S_{l_c}} P_\mathcal{M}(S_r|s) P_\mathcal{M}(s)$$

This transforms the search problem, which originally required traversing all possible sequences, into a selection problem over a finite candidate set.

### 3. Few-Token Experimental Design

**Deterministic Experiments** (6 settings):

| Experiment | Training Sequence | Test Path | Verification Goal |
|------|---------|---------|---------|
| Baseline | (e1, e2, e3, e1) | e3→e1→e2 | Basic cycling capability |
| Length of Path | (e1, e2, e3, E4, e1) | e3→E4→e1→e2 | Effect of path length |
| Length of 'Out-of' Path | (e1, e2, E3, e4, e1) | e4→e1→e2 | Out-of-path noise |
| Cycle Composability | Two sequences share e3 | e3→e1→e2 | Compositionality across samples |
| Hyperlink Composability | Two sequences share e3 | e2→e3→e1→e4 | Hyperlink jumping |

All experiments utilize a small 2-layer, 8-head Transformer, demonstrating the fundamental feasibility of the cycle token mechanism.

**Randomized Experiments**: Extended to scenarios where each cycle token maps to multiple ($n$) candidate subsequent tokens, validating the choice behavior of cycle tokens under ambiguity.

### 4. ReCall-aware Prompting Strategy

A two-step prompting strategy is designed for practical LLMs:

**Step 1 - Recall Context**:
> "Tell me the lines surrounding this line 'X'."

This prompts the model to leverage self-referencing causal cycles to output all relevant surrounding text for the target line, forming a candidate set.

**Step 2 - Extract Answer**:
Utilizing the model's in-context learning capabilities, the correct preceding line is extracted from the output of Step 1.

## Key Experimental Results

### Deterministic Few-Token Experiments

**Core Results**: Across all six experimental settings, the model achieved **100% validation accuracy** after training (except for Cycle Composability). This directly demonstrates that cycle tokens enable the model to reconstruct the left segment from the right segment.

**Sequence Length Ablation**: Increasing the path length N from 4 to 64 slowed down generalization under the default embedding dimension of 36. However, when the dimension was increased to 256, fast generalization was achieved across all lengths, indicating that the ReCall mechanism remains consistent across varying sequence lengths.

**Exception in Cycle Composability**: When the left context alters the semantics of the cycle token (e.g., in [e3, e1, e4] where e3 preceding e1 changes the meaning of e1), the model tends to predict e4 instead of e2 according to the training pattern. This is in fact the **correct behavior**—self-attention naturally adjusts token semantics based on context.

### Randomized Experiments

When the candidate set size is $n$, the accuracy of predicting the next token precisely follows the **$1/n$** rule. For example:
- $n=2 \rightarrow 50\%$ accuracy
- $n=3 \rightarrow 33\%$ accuracy  
- $n=4 \rightarrow 25\%$ accuracy

This is substantially better than random guessing ($1/V$, where $V$ is the vocabulary size, and $V \gg n$), indicating that cycle tokens effectively narrow down the search space to a semantically relevant candidate set.

### Analysis of Cycle Tokens in Pre-training Corpora

An analysis of 50 classical cultural texts (poems, speeches, nursery rhymes) was conducted:
- **High Frequency**: For instance, "Star-Spangled Banner" appears 73 times in its corresponding Wikipedia article.
- **Uniform Distribution**: Cycle token sequences are widely distributed throughout the text, forming a rich network of causal paths.
- These repeating titles or key phrases naturally form "hyperlinks" without manual intervention.

### ReCall-aware Prompting Effectiveness

Evaluations on GPT-4o (2024-12-23) and LLaMA-3.3-70B show:
- **100% Success Rate**: For any complete sentence in all 50 classical texts, the two-step ReCall-aware prompting correctly retrieves the preceding line.
- In contrast, conventional prompting (directly asking "what is the preceding line") fails in most cases.
- Even advanced prompting strategies such as chain-of-thought or process-of-elimination fail.

## Highlights & Insights

1. **Elegance of the Library Analogy**: Analogizing the LLM to a library, prompts to directories, and cycle tokens to cross-references is both intuitive and profound, fundamentally reshaping our understanding of the reversal curse.
2. **Perspective Shift from "Bug" to "Feature"**: While the reversal curse has been traditionally viewed as a defect requiring correction, this work demonstrates that naturally occurring structures within the pre-training data are sufficient to mitigate it.
3. **Bridge Between Theory and Practice**: From axiom-level few-token experiments to practical prompting strategies on LLMs, a complete pipeline from theory to execution is established.
4. **The $1/n$ Random Selection Pattern**: Under ambiguity, the cycle token selection precisely follows a uniform distribution, indicating that a structured candidate set is physically formed within the model.
5. **Lightweight Solution**: ReCall-aware prompting requires no modifications to the model or its training data, overcoming the reversal curse purely through a two-step prompt.

## Limitations & Future Work

1. **Over-simplicity of Controlled Experiments**: Few-token experiments utilize extremely small models and simplified datasets, presenting a gap compared to the complexity of real LLMs.
2. **Attribution Difficulty**: In large models, it is challenging to precisely attribute information retrieval to specific cycle tokens, particularly when pre-training data is proprietary.
3. **Privacy and Security Constraints**: Extracting pre-training data from LLMs is fundamentally restricted, which complicates the empirical validation of the cycle token mechanism.
4. **Real-world Deployment Reliability**: Although a 100% success rate is reported on 50 classical texts, generalization across arbitrary texts remains unverified.
5. **Dependency on Natural Repetition**: If the pre-training data lacks sufficient repetitive patterns, the cycle token mechanism may fail to apply.

## Related Work & Insights

- **Reversal Curse**: First systematically identified by Berglund et al. (2023); theoretical analysis by Allen-Zhu & Li (2023).
- **Data Augmentation Schemes**: Token permutation by Guo et al. (2024), reversed training by Golovneva et al. (2024), and token duplication by Springer et al. (2024).
- **Bidirectional Models**: BERT (Devlin et al., 2019) circumvents the reversal curse via the masked language modeling objective, but does not extend to autoregressive generation.
- **LLMs as Knowledge Bases**: Petroni et al. (2019), Heinzerling & Inui (2020), and the library analogies of Lederman & Mahowald (2024).

## Rating

⭐⭐⭐⭐ (4/5)

The concept is highly novel, offering a fresh perspective on the reversal curse. The proposed cycle token mechanism is both intuitively elegant and theoretically rigorous. The few-token experimental design is meticulous and persuasive. However, scaling insights from tiny models to large scale LLMs requires further validation, and the applicability of the prompting strategy needs expansion to more diverse text domains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] ExpliCa: Evaluating Explicit Causal Reasoning in Large Language Models](explica_evaluating_explicit_causal_reasoning_in_large_language_models.md)
- [\[ACL 2025\] Assessing and Enhancing the Causal Reasoning Abilities of Language Models via Faithful Textual Interpretation](assessing_and_enhancing_the_causal_reasoning_abilities_of_language_models_via_fai.md)
- [\[ACL 2025\] SQLong: Enhanced NL2SQL for Longer Contexts with LLMs](sqlong_enhanced_nl2sql_for_longer_contexts_with_llms.md)
- [\[ACL 2025\] HumT DumT: Measuring and Controlling Human-like Language in LLMs](humt_dumt_measuring_and_controlling_human-like_language_in_llms.md)
- [\[ACL 2025\] Does Time Have Its Place? Temporal Heads Where Language Models Recall Time-specific Information](does_time_have_its_place_temporal_heads_where_language_models_recall_time-specif.md)

</div>

<!-- RELATED:END -->

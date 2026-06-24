---
title: >-
  [Paper Note] Context-Robust Knowledge Editing for Language Models
description: >-
  [ACL 2025][Knowledge Editing][Context Robustness] This work identifies that existing knowledge editing methods significantly fail when prefix contexts are present (with editing success rates dropping from 90.9% to 69.1%). It introduces the CHED benchmark to evaluate context robustness and designs CoRE, a method that enhances the context robustness of editing through diversified prefix contexts and cross-prefix hidden state variance regularization…
tags:
  - "ACL 2025"
  - "Knowledge Editing"
  - "Context Robustness"
  - "MEMIT"
  - "Variance Regularization"
  - "Attention Analysis"
date: 2026-05-08
content_hash: 5cc32954a382631d
---

# Context-Robust Knowledge Editing for Language Models

**Conference**: ACL 2025  
**arXiv**: [2505.23026](https://arxiv.org/abs/2505.23026)  
**Code**: [https://github.com/holi-lab/CoRE](https://github.com/holi-lab/CoRE)  
**Area**: Knowledge Editing  
**Keywords**: Knowledge Editing, Context Robustness, MEMIT, Variance Regularization, Attention Analysis

## TL;DR
This work identifies that existing knowledge editing methods significantly fail when prefix contexts are present (with editing success rates dropping from 90.9% to 69.1%). It introduces the CHED benchmark to evaluate context robustness and designs CoRE, a method that enhances the context robustness of editing through diversified prefix contexts and cross-prefix hidden state variance regularization, significantly narrowing the performance gap between settings with and without context while maintaining general model capabilities.

## Background & Motivation

**Background**: Knowledge Editing is an efficient approach to modify the internal knowledge of LLMs without full retraining. Recurrent methods like MEMIT directly update the weights of MLP layers to map key vectors to new value vectors, thereby updating knowledge.

**Limitations of Prior Work**: (a) Existing evaluations (e.g., CounterFact, zsRE) only test isolated editing prompts (e.g., "Tim Cook, who works for") without any prefix context; (b) In practice, dialogue histories or preceding context often trigger the model to recall original knowledge, rendering the editing ineffective; (c) As shown in Figure 1, when "iPhone" appears in the prefix, attention shifts to this token, causing the model to revert to the original knowledge "Apple".

**Key Challenge**: While knowledge editing modifies key-value associations in MLPs, the prefix context injects information from semantically related tokens into the hidden state of the subject token via the attention mechanism, which interferes with the edited key-value mapping.

**Goal**: (a) Build the CHED benchmark to evaluate context robustness; (b) Propose CoRE, an editing method to enhance context robustness.

**Key Insight**: Hop words highly related to the edited entities are mined from the Wikidata knowledge graph to construct natural, distracting prefix contexts.

**Core Idea**: Distracting prefixes are constructed using associated words of edited entities to evaluate robustness, and variance regularization is employed to ensure that editing remains stable across different prefixes.

## Method

### Overall Architecture
1. **CHED Benchmark Construction**: Knowledge triplet $\to$ Wikidata extraction of hop words $\to$ Selection of highly distracting words $\to$ Generation of prefix context sentences.
2. **CoRE Method**: Diversified prefix contexts + cross-prefix variance regularization $\to$ Enhancing the robustness of MEMIT editing.

### Key Designs

1. **CHED Benchmark Construction**:

    - Based on 21,919 knowledge triplets $(s, r, o) \to (s, r, o^*)$ from CounterFact.
    - Extract all entities one-hop away from $s$, $o$, and $o^*$ from Wikidata as hop words.
    - Word selection strategy (Freq-Sim): First select the 10 lowest-frequency words in the corpus, then select the 5 words with the highest cosine similarity to the subject entity.
    - Generate natural prefix context sentences using GPT-4o-mini (containing the hop word, $\le 20$ words, and naturally transitioning into the editing prompt).
    - 6 types of prefixes: $s$, $o$, $o^*$, $s_{hop}$, $o_{hop}$, $o^*_{hop}$.
    - Final dataset: 21,782 triplets $\times$ 314,385 hop-word prefixes + 326,730 direct-word prefixes.

2. **CoRE Method**:

    - **Diversified Prefix Contexts (Figure 4-A)**: Unlike the original MEMIT which uses generalized prefixes such as "The" and "Therefore", CoRE uses combinations of $s$, $o$, and $o^*$ (e.g., "$s$ + $o$") as prefixes.
    - **Design Motivation**: These words are naturally highly related to the original/edited knowledge, inducing larger value vector variance than generalized prefixes.
    - **Cross-Prefix Variance Regularization (Figure 4-B)**: When optimizing the edited value vector $\mathbf{v}^*$, a regularization term is added to minimize the variance of hidden states under different prefixes.
    - Objective function: $$\mathbf{v}^* = \arg\min_\mathbf{v} \frac{1}{N}\sum_j [-\log \mathbb{P}[o^* | z_j]] + D_{KL}(\mathbf{v}) + \lambda_{var} \cdot \text{Var}(\mathbf{v})$$
    - **Design Motivation**: Large variance indicates that different contexts lead to different editing behaviors; regularization ensures that only necessary parameter modifications are applied.

## Key Experimental Results

### Main Results: Impact of Prefix Contexts on Editing Success Rate

| Method | No Prefix | $s$ | $o$ | $o^*$ | $s_{hop}$ | $o_{hop}$ | $o^*_{hop}$ |
|------|--------|-----|-----|-------|----------|----------|------------|
| MEMIT | 90.9% | 84.8% | 82.1% | 87.8% | 83.1% | 69.1% | 78.1% |
| FT-W | 82.6% | 74.2% | 62.5% | 70.5% | 72.8% | 58.3% | 66.7% |
| PMET | 88.7% | 83.2% | 79.8% | 85.9% | 81.5% | 67.2% | 76.3% |

- $o_{hop}$ (associated words of the original object) causes the largest interference: MEMIT drops from 90.9% to 69.1% (-21.8%).
- Hop words selected by the Freq-Sim word selection strategy show the strongest interference.

### CoRE Performance Improvements

| Method | No Prefix | $o_{hop}$ (Hardest) | Average |
|------|--------|-----------------|------|
| MEMIT | 90.9% | 69.1% | 82.5% |
| CoRE (Ours) | 90.2% | 79.8% | 87.3% |
| CoRE Gain | -0.7% | +10.7% | +4.8% |

- CoRE improves by 10.7% in the hardest scenario ($o_{hop}$), with almost no performance loss in the no-prefix scenario.
- Fluency and general capabilities (such as MMLU) show no significant degradation.

### Ablation Study
- **Prefix Type**: Comparing user utterances vs. assistant utterances as prefixes; user utterances cause greater interference (the model trusts user inputs more).
- **Variance Regularization Weight**: An excessively large $\lambda_{var}$ leads to under-editing, while a too-small weight fails to suppress variance; the optimal value is within 0.1-1.0.
- **Attention Analysis**: Tokens in the prefix related to the original knowledge receive abnormally high attention scores; CoRE mitigates this attention shift.

### Key Findings
- Prefix context is a major blind spot in knowledge editing—all existing methods deteriorate significantly when prefixes are present.
- Prefixes related to the original object $o$ cause larger interference than those related to the subject $s$.
- Variance regularization is a simple yet effective means to enhance context robustness.

## Highlights & Insights
- **Problem Importance**: This work is the first to systematically reveal the catastrophic impact of prefix contexts on knowledge editing, filling an evaluation gap.
- **Exquisite Benchmark Design**: The hop words selected by the Freq-Sim strategy are both natural and highly distracting, making CHED a versatile evaluation tool.
- **Simple and Effective Method**: CoRE only introduces prefix diversification and variance regularization on top of MEMIT without requiring extra models or complex pipelines.

## Limitations & Future Work
1. CoRE is based on the locate-then-edit paradigm (MEMIT) and has not been validated on weight-preserved methods (e.g., SERAC).
2. The prefix contexts in CHED are generated by GPT-4o-mini, with a moderate coherence score (3.4/5), containing some unnatural cases.
3. The evaluation is built on top of CounterFact, limiting it to factual triplet editing, which does not cover more complex knowledge structures.
4. Variance regularization assumes that hidden states under different prefixes should be similar, yet some contexts indeed ought to affect the output—requiring more fine-grained control.

## Related Work & Insights
- Difference from MQuAKE (Zhong et al., 2023): MQuAKE evaluates chain editing for multi-hop questions, whereas CHED evaluates the context robustness of single-hop editing—making them complementary.
- Difference from CounterFact+: The latter only appends samples of the same relation and object as prefixes, whereas CHED mines more targeted distractors using the knowledge graph.
- Insight: The evaluation of knowledge editing should expand from "single-turn QA" to "multi-turn dialogue"—CHED is the first step, but longer dialogue chains are still needed.

## Rating
- Novelty: ⭐⭐⭐⭐ (Novel problem, valuable benchmark)
- Theoretical Depth: ⭐⭐⭐ (Attention analysis is insightful but lacks deep theory)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Comparison of multiple methods, complete ablation)
- Value: ⭐⭐⭐⭐⭐ (Both CHED benchmark and CoRE method are directly applicable)
- Overall Recommendation: ⭐⭐⭐⭐ (Important advancement in the field of knowledge editing evaluation)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] BMIKE-53: Investigating Cross-Lingual Knowledge Editing with In-Context Learning](bmike-53_investigating_cross-lingual_knowledge_editing_with_in-context_learning.md)
- [\[ACL 2025\] Structure-aware Domain Knowledge Injection for Large Language Models](structure-aware_domain_knowledge_injection_for_large_language_models.md)
- [\[ACL 2025\] Neuron-Level Sequential Editing for Large Language Models](neuron-level_sequential_editing_for_large_language_models.md)
- [\[NeurIPS 2025\] UniEdit: A Unified Knowledge Editing Benchmark for Large Language Models](../../NeurIPS2025/knowledge_editing/uniedit_a_unified_knowledge_editing_benchmark_for_large_language_models.md)
- [\[NeurIPS 2025\] KScope: A Framework for Characterizing the Knowledge Status of Language Models](../../NeurIPS2025/knowledge_editing/kscope_a_framework_for_characterizing_the_knowledge_status_of_language_models.md)

</div>

<!-- RELATED:END -->

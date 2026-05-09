---
title: >-
  [Paper Note] Rethinking Benign Relearning: Syntax as the Hidden Driver of Unlearning Failures
description: >-
  [ICLR 2026][LLM Evaluation][machine_unlearning] This paper identifies that the true driver of "benign relearning" in LLM machine unlearning is not topical relevance but **syntactic similarity**, and proposes a **syntactic diversification** strategy to improve unlearning robustness.
tags:
  - ICLR 2026
  - LLM Evaluation
  - machine_unlearning
  - LLM_safety
  - syntactic_similarity
  - benign_relearning
date: 2026-05-08
content_hash: 8b24adde2eb608be
---

# Rethinking Benign Relearning: Syntax as the Hidden Driver of Unlearning Failures

**Conference**: ICLR 2026
**arXiv**: [2602.03379](https://arxiv.org/abs/2602.03379)
**Code**: Not released
**Area**: LLM Evaluation
**Keywords**: machine_unlearning, LLM_safety, syntactic_similarity, benign_relearning

## TL;DR

This paper identifies that the true driver of "benign relearning" in LLM machine unlearning is not topical relevance but **syntactic similarity**, and proposes a **syntactic diversification** strategy to improve unlearning robustness.

## Background & Motivation

Machine unlearning aims to remove specific content from a trained model while preserving overall performance. However, the phenomenon of "benign relearning" shows that forgotten information can resurface even when fine-tuning on seemingly unrelated benign data.

**Limitations of prior understanding**:
- The BLUR benchmark attributes benign relearning to **topical relevance** — i.e., entity/topic overlap between relearning data and forgotten data
- For example, after unlearning passages about Harry Potter, fine-tuning on GPT-generated descriptions of the same characters can recover the forgotten content
- This intuitive explanation has been widely accepted, but the authors find it incomplete

**Key observations**:
- BLUR experiments contain two confounding factors: (1) inconsistent dataset sizes across relevance levels lead to different numbers of gradient update steps; (2) recovery is non-monotonic, and evaluating only at epoch boundaries may miss peak recovery
- Under fair evaluation (standardized step budgets + reporting maximum recovery), the advantage of topical relevance largely disappears

## Method

### Overall Architecture

The study proceeds in three stages:
1. **Re-evaluating topical relevance**: After normalizing evaluation steps on WMDP, WHP, and RWKU benchmarks, topical relevance is found not to be the primary driver
2. **Validating syntactic similarity**: Controlled experiments on TOFU compare topically relevant sets vs. syntactically similar sets
3. **Proposing syntactic diversification**: GPT-4o is used to rewrite the forget set with diverse syntactic patterns, breaking syntactic rigidity

### Key Designs: Syntactic Similarity Metric

Normalized Levenshtein distance is used to quantify syntactic similarity:

$$\text{Sim}(s_1, s_2) = 1 - \frac{d_{\text{Lev}}(s_1, s_2)}{\max(|s_1|, |s_2|)}$$

where $d_{\text{Lev}}$ denotes the minimum number of single-character edits. This metric lies in $[0, 1]$ and captures surface structural alignment without semantic content.

### Controlled Experiment Design (TOFU Dataset)

Under the forget05 scenario of TOFU (unlearning knowledge of 10 fictional authors), three sets are defined:
- **Target set $D_{\text{target}}$**: QA pairs querying authors' full names
- **Topically relevant relearning set $D_{\text{relearn}}^{\text{topic}}$**: Non-name questions about the same authors (e.g., birthplace, occupation)
- **Syntactically similar relearning set $D_{\text{relearn}}^{\text{syntactic}}$**: Name questions about different authors sharing the same sentence structure as the target set

Key statistics: $D_{\text{relearn}}^{\text{syntactic}}$ has a syntactic similarity of 0.4513 with $D_{\text{target}}$, whereas $D_{\text{relearn}}^{\text{topic}}$ scores only 0.2349.

### Why Syntactic Similarity Drives Relearning

**Representation and gradient alignment analysis**:
- The syntactically similar set exhibits substantially higher cosine similarity to the target set in both hidden representations and gradient directions within the unlearned model, compared to the topically relevant set
- This indicates that syntactic overlap pulls the model's internal representations and optimization trajectory back toward the forgotten content

**Template vs. keyword unlearning analysis**:
- Target response tokens are categorized into **template tokens** (generic phrases) and **keyword tokens** (specific information to be forgotten, e.g., author names)
- A loss ratio is defined as: $\text{Loss Ratio} = \frac{\mathcal{L}_{\text{template}}}{\mathcal{L}_{\text{keyword}}}$
- During unlearning, the loss ratio rises consistently, indicating that unlearning primarily suppresses template tokens rather than the target keyword tokens
- When fine-tuning on syntactically similar data, the suppressed template structures are rapidly recovered, which in turn triggers the resurfacing of keyword content

### Syntactic Diversification Strategy

The procedure is as follows:
1. GPT-4o generates multiple syntactic variants of queries in $D_{\text{forget}}$
2. Rewrites with low similarity scores are retained to ensure diversity
3. The diversified set $D_{\text{forget}}'$ replaces the original forget set for unlearning

Effect: The average syntactic similarity between $D_{\text{relearn}}^{\text{syntactic}}$ and $D_{\text{forget}}'$ drops from 0.4513 to 0.2241.

### Loss & Training

Three standard unlearning methods are evaluated:
- **Gradient Ascent (GA)**: Maximizes loss on the forget set
- **Negative Preference Optimization (NPO)**: Suppresses forgotten content via preference optimization
- **SCRUB**: Joint optimization combining the forget set and a retain set

Syntactic diversification serves as a preprocessing step compatible with any of the above methods.

## Key Experimental Results

### Main Results: Syntactic Similarity vs. Topical Relevance for Relearning on TOFU

| Unlearning Method | Relearning Set | Recovery after 50 Unlearning Steps |
|---|---|---|
| GA | $D_{\text{relearn}}^{\text{topic}}$ | No recovery |
| GA | $D_{\text{relearn}}^{\text{syntactic}}$ | Keyword recovery with few updates |
| NPO | $D_{\text{relearn}}^{\text{topic}}$ | Marginal recovery |
| NPO | $D_{\text{relearn}}^{\text{syntactic}}$ | Significant recovery |
| SCRUB | $D_{\text{relearn}}^{\text{topic}}$ | Limited recovery |
| SCRUB | $D_{\text{relearn}}^{\text{syntactic}}$ | Full recovery of forgotten content |

Across all unlearning methods, the syntactically similar set consistently and substantially outperforms the topically relevant set in recovery. While SCRUB achieves the fastest forgetting, it is the most vulnerable to relearning.

### Ablation Study: Effect of Syntactic Diversification

**Model utility preservation (GA method)**:

| Metric | $D_{\text{forget}}$ | $D_{\text{forget}}'$ (Ours) |
|---|---|---|
| Real Authors ROUGE↑ | 0.2608 | **0.4257** |
| Real Authors Prob↑ | 0.3665 | **0.4223** |
| Real Authors TR↑ | 0.5769 | **0.6075** |
| World Facts Avg↑ | 0.6056 | **0.6104** |
| Retain Set ROUGE↑ | 0.1036 | **0.4052** |
| Retain Set Avg↑ | 0.1607 | **0.3128** |

Syntactic diversification not only improves unlearning robustness but also substantially enhances model utility, with Retain Set ROUGE improving from 0.10 to 0.41.

### Re-analysis of BLUR Benchmarks

| Benchmark | $D_{\text{hi}}$ Syntactic Sim. | $D_{\text{mid}}$ | $D_{\text{low}}$ |
|---|---|---|---|
| WMDP | 0.2244 | 0.2059 | 0.1771 |
| WHP | 0.1894 | 0.1767 | 0.1818 |
| RWKU | 0.2250 | 0.2215 | 0.1883 |

In WHP, $D_{\text{low}}$ (Lorem Ipsum filler text) exhibits syntactic similarity comparable to $D_{\text{hi}}$ and $D_{\text{mid}}$, explaining why its relearning effect is similarly strong.

### Key Findings

1. **Syntactic similarity > topical relevance**: Across all benchmarks and unlearning methods, syntactic similarity consistently serves as the primary driver of benign relearning
2. **Skewed unlearning**: Current unlearning methods disproportionately suppress template tokens rather than keyword tokens, creating structural vulnerability
3. **Triple benefit of syntactic diversification**: (a) suppresses relearning, (b) accelerates forgetting, (c) alleviates the trade-off between forgetting efficacy and model utility
4. **Safety training ≠ unlearning**: Methods such as DPO suppress outputs without removing knowledge, making them more vulnerable under syntactic relearning
5. **Risk of LoRA**: Although LoRA fine-tuning involves fewer parameters, it achieves faster and more effective recovery in relearning scenarios

## Highlights & Insights

- **Perspective shift**: Reframing unlearning failures from a semantic level (topical relevance) to a surface-form level (syntactic similarity) is a counterintuitive yet experimentally well-supported finding
- **Elegant experimental design**: Constructing datasets that share syntactic patterns but have no topical overlap cleanly disentangles the two factors
- **High practicality**: Syntactic diversification is straightforward to implement, requiring only a single LLM rewriting step with significant empirical gains
- **Critical security implications**: The work exposes an attack vector that is difficult to defend against in deployment — fine-tuning on syntactically similar but topically unrelated data is sufficient to recover forgotten knowledge

## Limitations & Future Work

1. Experiments are conducted primarily on TOFU, a synthetic dataset; unstructured text in real-world settings exhibits higher natural syntactic diversity
2. Syntactic diversification relies on GPT-4o rewriting quality, introducing additional computational cost
3. Only Llama-2-7b-chat and Phi model families are evaluated; behavior at larger scales remains to be verified
4. The syntactic similarity metric (Levenshtein distance) is relatively simple and does not capture more complex syntactic structural alignments

## Related Work & Insights

- **BLUR (Hu et al., 2025b)**: Proposed a three-level topical relevance taxonomy; the present work refutes its core conclusion through improved experimental design
- **TOFU (Maini et al., 2024)**: A standard LLM unlearning benchmark used here as the foundation for fine-grained controlled experiments
- **GA/NPO/SCRUB**: Three mainstream unlearning methods, all shown to share the same syntactic vulnerability
- Implications for unlearning robustness evaluation: Beyond assessing content-level recovery, future work should address structural attack surfaces

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Identifying syntactic similarity as the driver of relearning is a genuinely novel insight
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Controlled experiment design is elegant, ablations are comprehensive, and analysis is thorough
- **Value**: ⭐⭐⭐⭐ — Syntactic diversification is simple and effective
- **Writing Quality**: ⭐⭐⭐⭐ — Logically clear with intuitive figures
- **Overall**: ⭐⭐⭐⭐ (4/5)

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] LLM Unlearning with LLM Beliefs](llm_unlearning_with_llm_beliefs.md)
- [\[ICLR 2026\] Revisiting the Past: Data Unlearning with Model State History](revisiting_the_past_data_unlearning_with_model_state_history.md)
- [\[ICLR 2026\] Truthfulness Despite Weak Supervision: Evaluating and Training LLMs Using Peer Prediction](truthfulness_despite_weak_supervision_evaluating_and_training_llms_using_peer_pr.md)
- [\[ICLR 2026\] Unpacking Human Preference for LLMs: Demographically Aware Evaluation with the HUMAINE Framework](unpacking_human_preference_for_llms_demographically_aware_evaluation_with_the_hu.md)
- [\[ICLR 2026\] TabStruct: Measuring Structural Fidelity of Tabular Data](tabstruct_measuring_structural_fidelity_of_tabular_data.md)

<!-- RELATED:END -->

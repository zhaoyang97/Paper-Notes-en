---
title: >-
  [Paper Note] Combining the Best of Both Worlds: A Method for Hybrid NMT and LLM Translation
description: >-
  [ACL2025][LLM (Other)][Machine Translation] A hybrid NMT and LLM translation scheduling strategy (PPLT and JDM) based on source sentence features is proposed. It maintains optimal translation quality while reducing the LLM invocation rate to approximately 25-30%, significantly decreasing computational overhead.
tags:
  - "ACL2025"
  - "LLM (Other)"
  - "Machine Translation"
  - "NMT-LLM Integration"
  - "Scheduling Policy"
  - "Binary Decision Maker"
  - "Translation Quality Estimation"
date: 2026-05-08
content_hash: 47a2f242eac94eef
---

# Combining the Best of Both Worlds: A Method for Hybrid NMT and LLM Translation

**Conference**: ACL2025  
**arXiv**: [2505.13554](https://arxiv.org/abs/2505.13554)  
**Code**: To be confirmed  
**Area**: LLM/NLP  
**Keywords**: Machine Translation, NMT-LLM Integration, Scheduling Policy, Binary Decision Maker, Translation Quality Estimation

## TL;DR
A hybrid NMT and LLM translation scheduling strategy (PPLT and JDM) based on source sentence features is proposed. It maintains optimal translation quality while reducing the LLM invocation rate to approximately 25-30%, significantly decreasing computational overhead.

## Background & Motivation

1. **High cost and high latency of LLM translation**: Although LLMs perform exceptionally well on translation tasks, their inference cost is much higher than that of traditional NMT systems, creating barriers to practical deployment.
2. **NMT is sufficient for most sentences**: The authors annotated and found that about 95% of sentences belong to the "simple" category, where the DA score gap between NMT and LLM is only 1.41 points, making the LLM unnecessary.
3. **LLM shows clear advantages on complex sentences**: For only about 5% of "difficult" sentences, the LLM outperforms NMT by 3.80 DA points, showing that their complementarity is concentrated in a small fraction of hard sentences.
4. **Existing QE methods have limitations**: The QET method proposed by Hendy et al. relies on quality estimation models to score NMT translations but cannot determine whether the LLM translation is actually better, potentially introducing worse LLM results.
5. **NMT and LLMs have distinct strengths in different domains**: LLMs perform better in literature and internet slang, while NMT excels in technical fields. A single model cannot cover all scenarios.
6. **Urgent demand for low-cost, high-speed hybrid solutions**: The industry requires obtaining optimal translation quality without significantly increasing latency. Invoking the LLM only "when necessary" is the most pragmatic approach.

## Core Problem
How to quickly determine whether to use NMT or LLM for translation based solely on source sentence features, achieving optimal translation quality while minimizing LLM usage?

## Method

### Overall Architecture
Unlike QET (which does NMT translation first $\to$ QE scoring $\to$ deciding whether to run LLM), this work directly makes a binary decision from the source sentence without running NMT translation and QE models, making the inference pipeline simpler and much faster.

### Method 1: PPL Thresholding (PPLT)
- Train a small language model (LM) using the monolingual data in the NMT training data.
- Calculate perplexity (PPL) for the input source sentence; higher PPL indicates a more complex sentence.
- Set a threshold: if PPL > threshold, invoke LLM translation; otherwise, use NMT.
- Threshold determination: Sort 1 million monolingual data points and use the PPL value corresponding to the top 25% as the threshold.
- **Core Idea**: Sentence complexity (PPL) serves as an effective proxy indicator for predicting LLM superiority.

### Method 2: Joint Decision Method (JDM)
- **Training Data Construction**: Obtain NMT and LLM translation results for 1 million bilingual data points, and compute translation quality scores $Q_{NMT}$ and $Q_{LLM}$ using COMET.
- **Positive Sample Filtering Conditions**: $Q_{NMT} < T_1$ and $Q_{LLM} - Q_{NMT} > T_2$. This ensures that NMT translation is poor and the LLM is significantly better.
- The threshold $T_1$ is set to the 100,000th position after sorting $Q_{NMT}$, and $T_2$ is set to the 10,000th position after sorting the differences.
- This ultimately yields 10,000 positive samples + 30,000 negative samples.
- **Decision Maker**: A binary classification model fine-tuned on xlm-roberta-base. The input is solely the source sentence, and the output is "use NMT" or "use LLM".
- **Inference Process**: The source sentence directly passes through the decision maker without needing pre-translation or external evaluation.

### Key Designs
- The dual conditions of JDM ensure that LLM is only called when "NMT is poor + LLM is better," addressing the issue in QET where LLM output is not guaranteed to be superior.
- The decision maker relies exclusively on the source sentence, avoiding NMT translation overhead and resulting in extremely low inference latency.

## Key Experimental Results

### Experimental Settings
- **NMT Model**: Deep Transformer-Big architecture, trained on 100 million bilingual data points per language pair.
- **LLM**: Llama-3.1-8B-Instruct, used without fine-tuning.
- **Test Sets**: WMT22 News, Flores, self-constructed Literary (500 sentences), self-constructed Tech (500 sentences), covering ZH-EN, EN-ZH, DE-EN, and JA-EN.
- **Evaluation Metrics**: DA score (COMET), BLEURT, and LLM invocation rate ($LLM_p$).

### ZH $\to$ EN Results (Table 2)

| Method | News DA | Literary DA | Tech DA | Avg DA | Avg BLEURT | Avg LLM% |
|------|---------|-------------|---------|--------|------------|----------|
| NMT | 78.99 | 59.71 | 83.38 | 77.29 | 65.12 | 0% |
| LLM | 80.13 | 66.69 | 77.68 | 77.80 | 64.49 | 100% |
| QET | 79.13 | 63.88 | 80.21 | 77.58 | 64.78 | 30.55% |
| PPLT | 79.24 | 63.49 | 82.02 | 77.95 | 65.58 | 32.31% |
| **JDM** | **79.69** | **65.70** | **82.71** | **78.81** | **66.65** | **29.52%** |
| Oracle | 82.25 | 68.41 | 84.80 | 80.91 | 68.78 | 51.56% |

### EN $\to$ ZH Results (Table 3)

| Method | News DA | Literary DA | Tech DA | Avg DA | Avg BLEURT | Avg LLM% |
|------|---------|-------------|---------|--------|------------|----------|
| NMT | 86.17 | 71.68 | 86.30 | 83.01 | 67.65 | 0% |
| LLM | 85.17 | 76.30 | 78.40 | 81.63 | 63.24 | 100% |
| QET | 86.08 | 72.73 | 80.57 | 81.80 | 65.36 | 22.16% |
| **JDM** | **86.18** | **75.15** | **85.39** | **83.62** | **67.86** | **23.37%** |

### Key Findings
- JDM achieves the best average DA and BLEURT across all settings, while maintaining an LLM invocation rate of approximately 25-30%.
- JDM adaptively adjusts the LLM usage ratio: up to 80% on the literary test set (where LLM excels) and down to 7% on the technical test set (where NMT excels).
- Simply using PPL (PPLT) also outperforms QET, demonstrating that source sentence complexity serves as an effective scheduling signal.
- QET performs poorly in the technical domain because it cannot verify whether the LLM output is actually better.

## Highlights & Insights

1. **Simple and Effective Idea**: Scheduling decisions are based solely on source sentence features, eliminating the need for the two-step NMT+QE evaluation, leading to faster inference.
2. **Highly Adaptive JDM Decision Maker**: Automatically adjusts the proportion of LLM usage across different domains—relying heavily on LLM in the literary domain and almost completely bypassing it in the technical domain.
3. **Elegant Dual-Condition Design for Filtering Positive Samples**: Simultaneously requiring poor NMT quality and superior LLM quality prevents the introduction of suboptimal LLM translations.
4. **Highly Applicable for Practical Industrial Deployment**: Originating from the Huawei Translation Service Center, the solution targets real production environments with high cost-sensitivity.

## Limitations & Future Work

1. **Dependency on Complementarity**: When NMT and LLM capabilities reach complete parity, hybrid scheduling cannot bring improvements. The premise of the method is that both models have differentiated strengths in different scenarios.
2. **Decision Maker Requires Substantial Training Data**: JDM requires running both NMT and LLM inference on 1 million bilingual sentences to obtain training signals, resulting in non-trivial initial costs.
3. **Evaluations limited to 8B Scale LLM**: The potential of larger scale LLMs (such as 70B+) or GPT-4 level models has not been fully explored. As LLM capabilities improve, the margin for complementarity may shrink.
4. **Threshold Settings are Language-Pair Dependent**: The threshold needs to be adjusted separately for each language pair, lacking a comprehensive discussion on generalizability.
5. **Other Advantages of LLMs left unconsidered**: Quality dimensions that are difficult to quantify with standard metrics, such as style consistency and terminology translation, were not evaluated.

## Related Work & Insights

### vs QET (Hendy et al., 2023)
QET translates with NMT first, then uses a QE model to score and decide whether to call the LLM, presenting two problems: (1) it incurs additional inference costs for the QE model; (2) it cannot determine whether the LLM result is indeed better. In contrast, the JDM proposed in this paper makes decisions based only on the source sentence and guarantees during training that the LLM is superior to NMT for positive samples, avoiding "blindly switching to LLM."

### vs Cooperative Decoding (Zeng et al., 2024)
Cooperative decoding requires running both NMT and LLM concurrently for joint decoding during every translation step, with a computational cost equivalent to fully using the LLM. The proposed method only calls the LLM for ~25% of sentences, drastically reducing computational overhead.

### vs MBR Ensembling (Farinhas et al., 2023)
MBR ensembling methods require sampling multiple candidate translations from multiple models and selecting the optimal one, leading to extremely high inference overhead. This study uses a "routing" rather than an "ensembling" approach, querying only one model per sentence.

### Inspiration & Association
- **Generality of the Routing/Scheduling Approach**: The strategy of "using the small model if it can handle it, avoiding the large model" can be generalized to collaboration between large and small models in various NLP tasks.
- The finding that source sentence complexity serves as a scheduling signal can inspire model selection strategies in other tasks (e.g., summarization, QA).
- The training paradigm of the decision maker (constructing positive/negative samples based on performance discrepancies between two models) is transferable to other model-routing scenarios.

## Rating
- Novelty: ⭐⭐⭐ — Applying "model routing" to MT. PPLT and JDM designs are sensible but not highly groundbreaking.
- Experimental Thoroughness: ⭐⭐⭐⭐ — 4 language pairs × 4 domains, evaluation with multiple metrics, along with an Oracle upper-bound analysis.
- Writing Quality: ⭐⭐⭐⭐ — Clear motivation, intuitive comparison diagrams of the methods, and a compact structure.
- Value: ⭐⭐⭐⭐ — Strong industrial practicality. Coming from the Huawei Translation Center, cost-quality trade-offs are well analyzed.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] LLM Social Simulations Are a Promising Research Method](../../ICML2025/llm_nlp/llm_social_simulations_are_a_promising_research_method.md)
- [\[ACL 2025\] HyGenar: An LLM-Driven Hybrid Genetic Algorithm for Few-Shot Grammar Generation](hygenar_an_llm-driven_hybrid_genetic_algorithm_for_few-shot_grammar_generation.md)
- [\[ACL 2025\] Towards Style Alignment in Cross-Cultural Translation](towards_style_alignment_in_cross-cultural_translation.md)
- [\[ACL 2025\] BFS-Prover: Scalable Best-First Tree Search for LLM-Based Automatic Theorem Proving](bfs-prover_scalable_best-first_tree_search_for_llm-based_automatic_theorem_provi.md)
- [\[ACL 2025\] Can we Retrieve Everything All at Once? ARM: An Alignment-Oriented LLM-based Retrieval Method](can_we_retrieve_everything_all_at_once_arm_an_alignment-oriented_llm-based_retri.md)

</div>

<!-- RELATED:END -->

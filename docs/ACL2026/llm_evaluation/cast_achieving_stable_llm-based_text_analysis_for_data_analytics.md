---
title: >-
  [Paper Note] CAST: Achieving Stable LLM-based Text Analysis for Data Analytics
description: >-
  [ACL 2026][LLM Evaluation][Output Stability] This paper proposes the CAST framework, which constrains the latent reasoning trajectories of LLMs through two complementary mechanisms—Algorithmic Prompting and Thinking-before-Speaking—to significantly improve run-to-run stability in text summarization and annotation tasks without sacrificing output quality.
tags:
  - ACL 2026
  - LLM Evaluation
  - Output Stability
  - Text Analysis
  - Tabular Data
  - Algorithmic Prompting
  - Intermediate State Commitment
date: 2026-05-08
content_hash: d234c8c881e2e7c4
---

# CAST: Achieving Stable LLM-based Text Analysis for Data Analytics

**Conference**: ACL 2026
**arXiv**: [2602.15861](https://arxiv.org/abs/2602.15861)
**Code**: [https://github.com/jxtse/CAST-text-analysis](https://github.com/jxtse/CAST-text-analysis)
**Area**: LLM Evaluation
**Keywords**: Output Stability, Text Analysis, Tabular Data, Algorithmic Prompting, Intermediate State Commitment

## TL;DR

This paper proposes the CAST framework, which constrains the latent reasoning trajectories of LLMs through two complementary mechanisms—Algorithmic Prompting and Thinking-before-Speaking—to significantly improve run-to-run stability in text summarization and annotation tasks without sacrificing output quality.

## Background & Motivation

**State of the Field**: Text Analysis for Data Analysis (TADA) is a paradigm that converts free-text columns in tables into structured representations such as summary topics or row-level labels. LLMs are natural candidates for TADA, as a single model can perform diverse text analysis tasks via natural language queries.

**Limitations of Prior Work**: A fundamental tension exists between the probabilistic nature of LLM generation and the deterministic requirements of data analysis. The same input may produce semantically drifted outputs across different runs (e.g., the same review being labeled "Customer Service" in one run and "Support Team" in another), leading to inconsistent downstream filtering, grouping, and aggregation results that undermine reproducibility and user trust.

**Root Cause**: The source of instability lies in unconstrained latent reasoning trajectories within LLMs. From a probabilistic perspective, prompting an LLM induces a distribution over possible reasoning paths; when this distribution has high entropy (i.e., the model is uncertain about the next reasoning step), minor stochastic fluctuations cause output drift. Existing approaches such as Self-Consistency improve correctness through repeated sampling and majority voting, but are not designed to address stability.

**Paper Goals**: To achieve output stability by constraining the reasoning paths during generation, without relying on repeated sampling.

**Starting Point**: The authors observe that requiring the model to produce relevant intermediate reasoning states before generating the final output—even without specifying their exact content—substantially reduces variance in both output length and content.

**Core Idea**: Algorithmic Prompting provides procedural scaffolding to constrain reasoning transitions, while the Thinking-before-Speaking mechanism explicitly anchors critical intermediate states. Together, these mechanisms concentrate reasoning trajectories onto a small number of high-probability paths.

## Method

### Overall Architecture

CAST is realized via a single structured LLM call. The input consists of tabular text data and an analysis query; the output is a stable structured result (summary or labels). The same template adapts to different tasks by switching task-specific schemas and constraints, with the internal pipeline incorporating intermediate commitment writing and self-verification.

### Key Designs

1. **Algorithmic Prompting (AP)**:

   - **Function**: Provides procedural scaffolding for the task, constraining valid reasoning state transitions.
   - **Mechanism**: Classic deterministic workflows and expert heuristics are encoded as structured prompt sequences. For summarization tasks, the LLM is guided to first interpret the query and decompose constraints before executing the task according to an algorithmic workflow. Formally, AP introduces a gating function $g_t(z_t, z_{<t}, x)$ at each step, which concentrates probability mass onto fewer plausible next states via hard masking or soft weighting, reducing local uncertainty $H(Z_t|Z_{<t}, x, \mathcal{C}_{AP})$.
   - **Design Motivation**: Unconstrained reasoning transitions are the root cause of instability. AP "prunes" invalid reasoning paths by supplying a deterministic analytical workflow as a strong prior.

2. **Thinking-before-Speaking (TbS)**:

   - **Function**: Reduces path divergence by forcing the model to explicitly commit to critical intermediate states.
   - **Mechanism**: Rather than allowing the model to traverse reasoning trajectories implicitly and expose only the final output, TbS requires the model to generate intermediate states sequentially (e.g., domain judgment, topic schema, clustering results), with each subsequent generation conditioned on prior commitments. This is grounded in the information-theoretic principle that conditioning reduces entropy: $H(Z_{>t}|X=x, Z_{\leq t}) \leq H(Z_{>t}|X=x)$.
   - **Design Motivation**: Once the schema, topic set, or domain decision is fixed, subsequent generation is forced to remain consistent with it, rendering the reasoning path insensitive to minor stochastic perturbations.

3. **Stability Evaluation Metrics (CAST-S / CAST-T)**:

   - **Function**: Metrics specifically designed to quantify run-to-run stability.
   - **Mechanism**: CAST-S targets summarization and combines a semantic score $S_{sem}$ (content overlap) and a positional score $S_{pos}$ (ranking consistency based on Kendall's Tau): $S_{CAST-S}(\alpha) = \alpha \cdot S_{sem} + (1-\alpha) \cdot S_{pos}$, with $\alpha=0.9$ yielding the highest correlation with human judgment ($r=0.813$). CAST-T targets annotation: an LLM first clusters labels from multiple runs by semantic equivalence, then computes the proportion of dominant clusters.
   - **Design Motivation**: Existing metrics such as ROUGE-L and cosine similarity are insensitive to the semantic drift and ordering changes that matter most in analytical scenarios.

### Loss & Training

CAST is a pure inference-time method and involves no training. Constrained reasoning is achieved within a single API call through carefully designed structured prompts, without requiring multiple sampling rounds or voting.

## Key Experimental Results

### Main Results

| Model | Method | Summarization Stability (CAST-S) ↑ | Standalone Annotation Accuracy ↑ | Joint Annotation Stability (CAST-T) ↑ |
|---|---|---|---|---|
| GPT-5.2 | Baseline | 9.24 | 95.0% | 9.40 |
| GPT-5.2 | Self-Consistency | 7.40 | 96.2% | 9.16 |
| GPT-5.2 | CAST | **9.39** | **98.2%** | **9.60** |
| DeepSeek-V3.2 | Baseline | 8.15 | 92.7% | 8.78 |
| DeepSeek-V3.2 | CAST | **9.47** | 95.6% | **9.14** |
| Gemini-3-Flash | Baseline | 9.80 | 96.0% | 8.18 |
| Gemini-3-Flash | CAST | **9.93** | **96.8%** | 8.26 |

### Ablation Study

| Configuration | Summarization Stability (DeepSeek) | Notes |
|---|---|---|
| Full CAST (AP+TbS) | 9.47 | Full model |
| AP Only | 8.97 | Algorithmic Prompting only |
| TbS Only | 9.46 | Thinking-before-Speaking only |
| Few-shot | 8.96 | Few-shot prompting |
| Self-Consistency | 7.06 | Multi-sample voting performs worst |

### Key Findings

- Self-Consistency yields the worst stability, as its diffuse sampling is ill-suited for reliable post-hoc aggregation, and its computational overhead exceeds CAST's by more than 3×.
- AP and TbS exhibit synergistic effects; full CAST consistently outperforms either component used in isolation.
- CAST improves stability while also slightly enhancing summarization quality (recall increases from 0.854 to 0.879).

## Highlights & Insights

- The paper formalizes the mechanism of LLM output instability—high entropy over reasoning trajectories—through an information-theoretic lens, and provides a theoretical framework showing how constraining reasoning reduces entropy, which is more principled than empirical prompt tuning.
- The observation that requiring the model to produce relevant intermediate reasoning steps—without specifying their content—already reduces output variance is highly practical and broadly applicable.
- The CAST-S/CAST-T metrics fill a gap in stability quantification and are applicable to any scenario requiring consistent LLM outputs.

## Limitations & Future Work

- The algorithmic scaffolding currently requires manual design, which may limit extensibility to entirely new task domains.
- Experiments primarily cover summarization and annotation; effectiveness in more complex composite TADA workflows remains unverified.
- Excessive constraints may suppress nuanced variation that is valuable in certain analytical contexts.

## Related Work & Insights

- **vs. Self-Consistency**: SC improves correctness through repeated sampling and majority voting but does not guarantee stability and incurs high computational cost. CAST achieves stability via a single call by constraining reasoning paths.
- **vs. Algorithm-of-Thoughts**: AoT aims to improve correctness, whereas CAST targets stability—representing two distinct applications of the "constrained reasoning" paradigm.

## Rating

- Novelty: ⭐⭐⭐⭐ First systematic study of LLM output stability in data analytics scenarios.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three models, multiple baselines, ablations, and human evaluation for metric validation.
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretical framework and empirical observations are tightly integrated with clear exposition.
- Value: ⭐⭐⭐⭐ Significant reference value for the reliable deployment of LLMs in production environments.

## Rating

- Novelty: Pending
- Experimental Thoroughness: Pending
- Writing Quality: Pending
- Value: Pending

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] From Domains to Instances: Dual-Granularity Data Synthesis for LLM Unlearning](from_domains_to_instances_dual-granularity_data_synthesis_for_llm_unlearning.md)
- [\[ICLR 2026\] When to Ensemble: Identifying Token-Level Points for Stable and Fast LLM Ensembling](../../ICLR2026/llm_evaluation/when_to_ensemble_identifying_token-level_points_for_stable_and_fast_llm_ensembli.md)
- [\[ACL 2026\] Task-Aware LLM Routing with Multi-Level Task-Profile-Guided Data Synthesis for Cold-Start Scenarios](task-aware_llm_routing_with_multi-level_task-profile-guided_data_synthesis_for_c.md)
- [\[ACL 2026\] Text-to-Distribution Prediction with Quantile Tokens and Neighbor Context](text-to-distribution_prediction_with_quantile_tokens_and_neighbor_context.md)
- [\[ACL 2026\] Subject-level Inference for Realistic Text Anonymization Evaluation](subject-level_inference_for_realistic_text_anonymization_evaluation.md)

<!-- RELATED:END -->

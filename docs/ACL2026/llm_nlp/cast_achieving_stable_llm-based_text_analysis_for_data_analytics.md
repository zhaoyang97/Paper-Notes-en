---
title: >-
  [Paper Note] CAST: Achieving Stable LLM-based Text Analysis for Data Analytics
description: >-
  [ACL 2026][LLM/NLP][Output Stability] Proposes the CAST framework, which constrains the latent reasoning paths of LLMs through two mechanisms: Algorithmic Prompting and Thinking-before-Speaking. It significantly improves…
tags:
  - "ACL 2026"
  - "LLM/NLP"
  - "Output Stability"
  - "Text Analysis"
  - "Tabular Data"
  - "Algorithmic Prompting"
  - "Intermediate State Commitment"
date: 2026-05-08
content_hash: 5575bb51461fb445
---

# CAST: Achieving Stable LLM-based Text Analysis for Data Analytics

**Conference**: ACL 2026  
**arXiv**: [2602.15861](https://arxiv.org/abs/2602.15861)  
**Code**: [https://github.com/jxtse/CAST-text-analysis](https://github.com/jxtse/CAST-text-analysis)  
**Area**: LLM Evaluation  
**Keywords**: Output Stability, Text Analysis, Tabular Data, Algorithmic Prompting, Intermediate State Commitment

## TL;DR

Proposes the CAST framework, which constrains the latent reasoning paths of LLMs through two mechanisms: Algorithmic Prompting and Thinking-before-Speaking. It significantly improves inter-run stability for text summarization and labeling tasks without compromising output quality.

## Background & Motivation

**Background**: Text Analysis for Data Analysis (TADA) is a paradigm that transforms free-text columns in tables into structured representations (e.g., summary topics, row-level labels). LLMs are natural candidates for TADA, as a single model can perform various text analysis tasks via natural language queries.

**Limitations of Prior Work**: There is a fundamental conflict between the probabilistic nature of LLM generation and the deterministic requirements of data analysis. The same input may produce outputs with semantic drift across different runs (e.g., the same comment labeled as "Customer Service" or "Support Team"), leading to inconsistent filtering, grouping, and aggregation results downstream, which undermines reproducibility and trust.

**Key Challenge**: The root cause of instability lies in the unconstrained latent reasoning trajectories within the LLM. From a probabilistic perspective, prompting an LLM induces a distribution over possible reasoning paths; when the entropy of this distribution is high (high uncertainty for the next step), small random fluctuations lead to output drift. Existing methods like Self-Consistency focus on improving correctness through multi-sample voting but are not designed for stability.

**Goal**: To achieve output stability by constraining the reasoning path during the generation process without relying on repeated sampling.

**Key Insight**: The authors observe that requiring the model to generate relevant intermediate reasoning states before the final output—even without specifying the exact content—significantly reduces the variance in output length and content.

**Core Idea**: Use Algorithmic Prompting to provide a procedural scaffold that constrains reasoning transitions, and the Thinking-before-Speaking mechanism to explicitly fix key intermediate states. These work together to concentrate reasoning paths onto a few high-probability trajectories.

## Method

### Overall Architecture

CAST is implemented as a single structured LLM call: inputs are tabular text data and analysis queries, and outputs are stable structured results (summaries or labels). The same template adapts to different tasks by switching task-specific schemas and constraints, with an internal process including intermediate commitment writing and self-verification.

### Key Designs

1.  **Algorithmic Prompting (AP)**:
    - **Function**: Provides a procedural scaffold for tasks to constrain valid reasoning state transitions.
    - **Mechanism**: Encodes classic deterministic workflows and expert heuristics into structured prompt sequences. For summarization, it guides the LLM to interpret queries, decompose constraints, and follow algorithmic steps. Formally, AP introduces a gating function $g_t(z_t, z_{<t}, x)$ at each step, concentrating probability mass on fewer plausible next states via hard masking or soft weighting, thus reducing local uncertainty $H(Z_t|Z_{<t}, x, \mathcal{C}_{AP})$.
    - **Design Motivation**: Unconstrained reasoning transitions cause instability; AP "prunes" invalid paths by providing a deterministic analysis workflow as a strong prior.

2.  **Thinking-before-Speaking (TbS)**:
    - **Function**: Reduces path divergence by forcing the model to explicitly commit to key intermediate states.
    - **Mechanism**: Instead of letting the model traverse reasoning trajectories implicitly and only exposing the final output, the model is required to sequentially generate intermediate states (e.g., domain judgment, topic schema, clustering results), with each subsequent generation conditioned on previous commitments. This is based on the information-theoretic principle that conditioning reduces entropy: $H(Z_{>t}|X=x, Z_{\leq t}) \leq H(Z_{>t}|X=x)$.
    - **Design Motivation**: Once schemas, topic sets, or domain decisions are fixed, subsequent generation is forced to remain consistent, making the reasoning path insensitive to minor random fluctuations.

3.  **Stability Metrics (CAST-S/CAST-T)**:
    - **Function**: Metrics specifically designed to quantify inter-run stability.
    - **Mechanism**: CAST-S is used for summarization, combining a semantic score $S_{sem}$ (content overlap) and a positional score $S_{pos}$ (ranking consistency based on Kendall's Tau), $S_{CAST-S}(\alpha) = \alpha \cdot S_{sem} + (1-\alpha) \cdot S_{pos}$. It correlates best with human judgment ($r=0.813$) when $\alpha=0.9$. CAST-T is for labeling; an LLM clusters labels from multiple runs by semantic equivalence, then calculates the proportion of the dominant cluster.
    - **Design Motivation**: Existing metrics like ROUGE-L or cosine similarity are insensitive to semantic drift and ranking changes critical in analysis scenarios.

### Loss & Training

CAST is a purely inference-time method involving no training. It implements constrained reasoning via carefully designed structured prompts in a single API call, without requiring multiple samples or voting.

## Key Experimental Results

### Main Results

| Model | Method | Summary Stability (CAST-S)↑ | Independent Labeling Accuracy↑ | Joint Labeling Stability (CAST-T)↑ |
|-------|--------|-----------------------------|--------------------------------|------------------------------------|
| GPT-5.2 | Baseline | 9.24 | 95.0% | 9.40 |
| GPT-5.2 | Self-Consistency | 7.40 | 96.2% | 9.16 |
| GPT-5.2 | CAST | **9.39** | **98.2%** | **9.60** |
| DeepSeek-V3.2 | Baseline | 8.15 | 92.7% | 8.78 |
| DeepSeek-V3.2 | CAST | **9.47** | 95.6% | **9.14** |
| Gemini-3-Flash | Baseline | 9.80 | 96.0% | 8.18 |
| Gemini-3-Flash | CAST | **9.93** | **96.8%** | 8.26 |

### Ablation Study

| Config | Summary Stability (DeepSeek) | Description |
|--------|-----------------------------|-------------|
| Full CAST (AP+TbS) | 9.47 | Complete model |
| AP Only | 8.97 | Algorithmic Prompting only |
| TbS Only | 9.46 | Thinking-before-Speaking only |
| Few-shot | 8.96 | Few-shot prompting |
| Self-Consistency | 7.06 | Multi-sample voting performed worse |

### Key Findings
- Self-Consistency is the worst for stability because its diffused sampling is unsuitable for reliable post-aggregation, and its computational overhead is over 3x that of CAST.
- AP and TbS have synergistic effects; full CAST typically outperforms either component alone.
- CAST slightly improves summarization quality (recall from 0.854 to 0.879) while increasing stability.

## Highlights & Insights
- It formalizes the mechanism of LLM output instability—high entropy of reasoning paths—from an information-theoretic perspective and provides a theoretical framework for reducing entropy via constrained reasoning, which is more convincing than empirical tuning.
- The observation that requiring the model to produce relevant intermediate reasoning reduces output variance even without specifying the content is highly practical.
- CAST-S/CAST-T metrics fill the gap in stability quantification and are applicable to any scenario requiring consistent LLM outputs.

## Limitations & Future Work
- Algorithmic scaffolds currently require manual design, which may limit scalability to entirely new task domains.
- Experiments primarily cover summarization and labeling; effectiveness in more complex TADA composite workflows has not been verified.
- Excessive constraints might suppress valuable nuances in certain analysis scenarios.

## Related Work & Insights
- **vs Self-Consistency**: SC improves correctness via multi-sample voting but does not guarantee stability and is computationally expensive. CAST achieves stability through a single call by constraining the reasoning path.
- **vs Algorithm-of-Thoughts**: AoT aims to improve correctness, whereas CAST aims to improve stability, representing different applications of the "constrained reasoning" approach.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic study of LLM output stability in data analysis scenarios.
- Experimental Thoroughness: ⭐⭐⭐⭐ 3 models, multiple baselines and ablations, human evaluation for metrics.
- Writing Quality: ⭐⭐⭐⭐⭐ Close integration of theoretical framework and empirical observations, clear narrative.
- Value: ⭐⭐⭐⭐ Significant reference value for the reliable deployment of LLMs in production environments.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Text-to-Distribution Prediction with Quantile Tokens and Neighbor Context](text-to-distribution_prediction_with_quantile_tokens_and_neighbor_context.md)
- [\[ACL 2026\] Understanding Structured Financial Data with LLMs: A Case Study on Fraud Detection](understanding_structured_financial_data_with_llms_a_case_study_on_fraud_detectio.md)
- [\[AAAI 2026\] Collaborative LLM Numerical Reasoning with Local Data Protection](../../AAAI2026/llm_nlp/collaborative_llm_numerical_reasoning_with_local_data_protection.md)
- [\[ICML 2026\] Rare Event Analysis of Large Language Models](../../ICML2026/llm_nlp/rare_event_analysis_of_large_language_models.md)
- [\[ACL 2026\] Leveraging Pretrained Language Models as Energy Functions for Glauber Dynamics Text Diffusion](leveraging_pretrained_language_models_as_energy_functions_for_glauber_dynamics_t.md)

</div>

<!-- RELATED:END -->

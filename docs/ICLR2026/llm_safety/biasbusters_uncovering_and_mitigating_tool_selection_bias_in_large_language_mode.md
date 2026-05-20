---
title: >-
  [Paper Note] BiasBusters: Uncovering and Mitigating Tool Selection Bias in Large Language Models
description: >-
  [ICLR 2026][LLM Safety][tool selection bias] This paper presents the first systematic study of bias in LLM tool selection. When multiple functionally equivalent APIs are available…
tags:
  - "ICLR 2026"
  - "LLM Safety"
  - "tool selection bias"
  - "LLM agent"
  - "fairness"
  - "API marketplace"
  - "debiasing"
date: 2026-05-08
content_hash: 59b1fbaea5398fc7
---

# BiasBusters: Uncovering and Mitigating Tool Selection Bias in Large Language Models

**Conference**: ICLR 2026
**arXiv**: [2510.00307](https://arxiv.org/abs/2510.00307)  
**Code**: [https://github.com/thierry123454/tool-selection-bias](https://github.com/thierry123454/tool-selection-bias)  
**Area**: AI Safety
**Keywords**: tool selection bias, LLM agent, fairness, API marketplace, debiasing

## TL;DR
This paper presents the first systematic study of bias in LLM tool selection. When multiple functionally equivalent APIs are available, LLMs systematically favor certain tools due to semantic alignment, positional effects, and pretraining exposure. The authors propose a total variation–based bias metric, a benchmark spanning 10 tool categories, and a lightweight debiasing strategy based on filtering followed by uniform sampling.

## Background & Motivation

**Background**: LLM agents increasingly rely on external tools/APIs to perform tasks that cannot be executed directly, such as querying databases, retrieving real-time information, or invoking external services. Tool selection is a critical step in the agent pipeline—candidate tools are first retrieved, then the LLM reasons over them to select the final API to call.

**Limitations of Prior Work**: In API marketplaces, multiple providers offer functionally equivalent tools (e.g., multiple weather APIs, multiple translation APIs). Ideally, LLMs should treat such equivalent tools fairly; in practice, however, models systematically favor certain providers. This not only degrades user experience (repeatedly selecting slower or less reliable services) but also creates market unfairness under pay-per-request pricing models.

**Key Challenge**: Existing LLM bias research focuses primarily on social biases (gender, race, etc.) and cognitive biases (anchoring effects, etc.). Tool selection bias—a critical blind spot—has received almost no attention. A small body of work examines tool selection under adversarial attacks (e.g., malicious metadata injection), but biases arising under non-adversarial conditions—selection unfairness caused by subtle differences in tool names, descriptions, or list positions—have not been systematically analyzed.

**Goal**: Three sub-questions are addressed: (a) How severe is LLM tool selection bias? (b) What are the root causes of the bias? (c) How can it be mitigated?

**Key Insight**: The paper constructs clusters of functionally equivalent tools, quantifies the deviation of selection distributions from uniformity using total variation distance, and isolates the contribution of individual factors through controlled experiments (metadata perturbation, continual pretraining).

**Core Idea**: A benchmark of equivalent tool clusters combined with total variation distance metrics is used to systematically characterize LLM tool selection bias. Semantic alignment is identified as the primary driver, and a lightweight filter-then-uniform-sample mitigation strategy is proposed.

## Method

### Overall Architecture
BiasBusters is an end-to-end "uncover–explain–mitigate" framework. It first defines bias metrics and constructs a test benchmark to measure bias (Uncover), then explains the sources of bias through feature analysis, metadata perturbation, and biased pretraining experiments (Explain), and finally proposes a filtering + uniform sampling mitigation strategy (Mitigate).

### Key Designs

1. **Bias Measurement Framework**

    - **Function**: Quantifies the unfairness of LLM tool selection among functionally equivalent tools.
    - **Mechanism**: Bias is decomposed into two dimensions—API bias $\delta_{\text{API}}$ (preference for a specific API) and positional bias $\delta_{\text{pos}}$ (preference for a specific list position). Both are measured as the total variation distance between the selection distribution and a uniform distribution: $\delta_{\text{API}} = \text{TV}(P^{\text{API}}, U)$. The overall metric is $\delta_{\text{model}} = (\delta_{\text{API}} + \delta_{\text{pos}}) / 2$.
    - **Design Motivation**: Distinguishing API bias from positional bias is essential—positional bias can be eliminated by random shuffling, whereas API bias requires deeper intervention. Cyclic rotation (rotating the API order so each API appears at every position exactly once) ensures unconfounded measurement.

2. **Equivalent Tool Cluster Benchmark**

    - **Function**: Constructs a controlled dataset for bias evaluation.
    - **Mechanism**: Drawing from the RapidAPI database in ToolLLM, APIs are clustered into 10 functionally equivalent groups (e.g., weather forecasting, translation, geocoding), each comprising 5 APIs and 100 user queries, yielding 1,000 test pairs. Queries are generated by an LLM to be balanced and provider-agnostic.
    - **Design Motivation**: Functional equivalence is a prerequisite for fair evaluation—only when tools genuinely perform the same task does preferring one over another constitute "bias."

3. **Multi-Dimensional Bias Explanation**

    - **Feature-Level Analysis**: Seven API features are extracted (semantic similarity, parameter count, description length, readability, promotional language, etc.), and Pearson correlation, linear regression, and random forest models are used to identify which features predict selection rate. **Query–description semantic similarity emerges as the strongest predictor**, yet $R^2 < 0.4$, indicating a large residual of unexplained variance.
    - **Metadata Perturbation Experiments**: Eight controlled perturbations are designed (shuffled names, shuffled descriptions, swapped descriptions, etc.). Description-level semantics are found to be the primary cue used by models to differentiate equivalent APIs. Shuffling descriptions and parameters produces the largest shifts in selection; name perturbations have smaller and more variable effects.
    - **Biased Continual Pretraining**: Continual pretraining of Qwen3-8B on 3.5 million tokens saturated with a single API's metadata raises the target API's selection rate from 0.6% to 12.8% (>20×). The target API still does not dominate overall selections, indicating that pretraining exposure only partially accounts for the observed bias.

4. **Lightweight Mitigation Strategy**

    - **Function**: Reduces bias without sacrificing task coverage.
    - **Mechanism**: A small LLM (Qwen3-14B) serves as a filter, first identifying the subset of candidate APIs capable of addressing the current query, then uniformly sampling at random from that subset. This decouples "recognition capability" from "selection behavior."
    - **Effect**: Micro-Precision ≈ 1.00 (almost no incorrect APIs introduced); Micro-Recall ≈ 0.89 (most correct APIs retained); bias metrics decrease substantially.

### Experimental Setup
- Seven LLMs are evaluated: GPT-3.5-turbo, GPT-4.1 mini, Claude 3.5 Sonnet, DeepSeek-V3.2-Exp, Gemini 2.5 Flash, ToolLLaMA-2-7B, and Qwen3 (1.7B–235B).
- Each query is executed under 5 cyclic rotation orders; temperature = 0.5, top-p = 1.0.
- Approximately 500,000 inference runs in total.

## Key Experimental Results

### Main Results

| Model | $\delta_{\text{API}}$ | $\delta_{\text{pos}}$ | $\delta_{\text{model}}$ |
|------|-------|-------|---------|
| Qwen3 235B | 0.330 | 0.168 | **0.249** |
| GPT-3.5-turbo | 0.320 | 0.336 | 0.328 |
| Gemini 2.5 Flash | 0.365 | 0.306 | 0.335 |
| ToolLLaMA | 0.277 | 0.391 | 0.334 |
| Claude 3.5 Sonnet | 0.370 | 0.325 | 0.347 |
| DeepSeek-V3.2-Exp | 0.249 | 0.504 | 0.377 |
| GPT-4.1 mini | 0.331 | 0.423 | 0.377 |

### Ablation Study / Mitigation Results

| Configuration | Micro-Precision | Micro-Recall | Exact Match | Notes |
|------|----------------|-------------|-------------|------|
| Filter + Uniform Sampling (overall) | 0.9964 | 0.8856 | 0.69 | Almost no incorrect tools introduced |
| K=2 | 1.0000 | 0.7717 | 0.5433 | Lower recall for small candidate sets |
| K=4 | 0.9940 | 0.9633 | 0.9100 | Best overall performance |
| K=5 | 1.0000 | 0.8610 | 0.5350 | Slight omissions for larger sets |

### Key Findings
- **Significant bias is present across all evaluated models**: $\delta_{\text{model}}$ ranges from 0.25 to 0.38, implying that 25%–38% of selection probability mass must be redistributed to achieve fairness.
- **The two bias types are complementary**: High API bias co-occurs with low positional bias and vice versa; when no strong API preference exists, models rely on positional cues, favoring tools listed earlier.
- **Bias is highly aligned across models**: GPT-4.1 mini, Claude, Gemini, DeepSeek, and Qwen3 235B tend to favor the same APIs, suggesting that biases stem from shared implicit decision rules.
- **The effect of metadata perturbation is context-dependent**: The same perturbation may reverse, redistribute, or barely affect preferences depending on the tool cluster.
- Higher temperature slightly reduces bias; larger models exhibit less bias; system prompts shift the preferred API without eliminating bias.

## Highlights & Insights
- **Formalizing tool selection bias as a fairness problem**: This is a practical and overlooked issue—as the agent ecosystem matures, fair competition in API marketplaces becomes increasingly important. Using TV distance to quantify bias is both principled and concise.
- **The separation of API bias and positional bias is a clever design**: The cyclic rotation experimental protocol elegantly controls for positional confounds, enabling independent measurement of the two bias types.
- **The filter + uniform sampling mitigation is transferable**: The idea of decoupling "recognition" from "selection" generalizes to other LLM scenarios requiring fair choice (e.g., recommendation systems, content distribution).
- **The biased continual pretraining experiment** quantitatively demonstrates that pretraining data can "implant" tool preferences, offering insight into how biases propagate through LLM pretraining.

## Limitations & Future Work
- **Limited benchmark scale**: Only 10 clusters, 5 APIs per cluster, and 100 synthetic queries—generalization to production environments with hundreds of API categories remains to be validated.
- **Insufficient explanatory power of features**: Linear regression achieves $R^2 < 0.4$, indicating that a large portion of bias arises from factors not captured by the extracted features (possibly implicit associations from pretraining).
- **Mitigation strategy requires an additional LLM**: The filter itself may introduce its own biases (though experiments suggest the impact is minor), and it adds inference overhead.
- **Coverage limited to English queries and RapidAPI**: Cross-lingual and cross-platform generalizability has not been verified.
- The contribution of RLHF/preference tuning to tool selection bias is not analyzed, despite being a potentially important source.

## Related Work & Insights
- **vs. Mo et al. (2025) / Faghih et al. (2025)**: These works study tool selection vulnerabilities under adversarial attacks (malicious metadata injection). The present paper addresses natural, non-adversarial bias, which is more pervasive.
- **vs. positional bias research (Pezeshkpour, Zheng, et al.)**: Prior work examines positional bias in multiple-choice settings; this paper extends the investigation to tool selection and introduces an integrated framework that jointly accounts for both API bias and positional bias.
- **vs. LLM fairness research**: Existing studies focus on social and cognitive biases; this paper opens a new direction centered on tool selection bias.
- The bias measurement methodology proposed here could be applied to assess LLM agent fairness in other decision-making scenarios (e.g., routing selection, model selection).

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic study of tool selection bias; problem formulation and benchmark design are original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Seven models, ~500K inference runs, and multi-dimensional analysis, though the benchmark scale is relatively small.
- Writing Quality: ⭐⭐⭐⭐ Clear structure; the uncover–explain–mitigate logical chain is coherent and complete.
- Value: ⭐⭐⭐⭐ Practically significant for fairness in the agent ecosystem; the mitigation strategy is simple and deployable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Gender Bias in Emotion Recognition by Large Language Models](../../AAAI2026/llm_safety/gender_bias_in_emotion_recognition_by_large_language_models.md)
- [\[ICLR 2026\] AudioTrust: Benchmarking the Multifaceted Trustworthiness of Audio Large Language Models](audiotrust_benchmarking_the_multifaceted_trustworthiness_of_audio_large_language.md)
- [\[ICLR 2026\] Measuring Physical-World Privacy Awareness of Large Language Models: An Evaluation Benchmark](measuring_physical-world_privacy_awareness_of_large_language_models_an_evaluatio.md)
- [\[ICLR 2026\] SecP-Tuning: Efficient Privacy-Preserving Prompt Tuning for Large Language Models via MPC](secp-tuning_efficient_privacy-preserving_prompt_tuning_for_large_language_mode.md)
- [\[ACL 2026\] Topic-Based Watermarks for Large Language Models](../../ACL2026/llm_safety/topic-based_watermarks_for_large_language_models.md)

</div>

<!-- RELATED:END -->

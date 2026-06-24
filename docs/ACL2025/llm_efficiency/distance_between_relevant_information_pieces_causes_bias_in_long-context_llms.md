---
title: >-
  [Paper Note] Distance between Relevant Information Pieces Causes Bias in Long-Context LLMs
description: >-
  [ACL 2025 (Findings)][LLM Efficiency][Positional Bias] This paper proposes the LongPiBench benchmark to systematically investigate, for the first time, LLMs' sensitivity to the distance (spacing) between multiple relevant information pieces in long contexts. It reveals that while current models have largely overcome the "lost-in-the-middle" problem, they still exhibit significant positional bias when the spacing between relevant information pieces varies.
tags:
  - "ACL 2025 (Findings)"
  - "LLM Efficiency"
  - "Positional Bias"
  - "Long-Context LLMs"
  - "Multiple Information Pieces"
  - "LongPiBench"
  - "Information Spacing"
date: 2026-05-08
content_hash: 1068c69dfbbe909b
---

# Distance between Relevant Information Pieces Causes Bias in Long-Context LLMs

**Conference**: ACL 2025 (Findings)  
**arXiv**: [2410.14641](https://arxiv.org/abs/2410.14641)  
**Code**: None  
**Area**: LLM Efficiency / Long-Context Understanding  
**Keywords**: Positional Bias, Long-Context LLMs, Multiple Information Pieces, LongPiBench, Information Spacing

## TL;DR

This paper proposes the LongPiBench benchmark to systematically investigate, for the first time, LLMs' sensitivity to the distance (spacing) between multiple relevant information pieces in long contexts. It reveals that while current models have largely overcome the "lost-in-the-middle" problem, they still exhibit significant positional bias when the spacing between relevant information pieces varies.

## Background & Motivation

**Background**: In recent years, long-context LLMs have developed rapidly, with supported context windows expanding from 4K to 128K or even longer. Positional bias remains a core challenge hindering the quality of long-context understanding. The most notable prior finding is the "lost-in-the-middle" phenomenon, where LLMs tend to focus on information at the beginning and end of the input while ignoring content in the middle.

**Limitations of Prior Work**: Existing research on positional bias almost exclusively focuses on the positional effects of a single relevant information piece. However, in real-world application scenarios (such as multi-hop reasoning, multi-document QA, and RAG), answering a question often requires integrating multiple information pieces scattered across different locations in a long context. Prior works fail to measure positional bias in such multi-information scenarios.

**Key Challenge**: The evaluation paradigm utilizing a single information piece is insufficient to reflect the complexity of practical usage. When multiple relevant information pieces are present, does their relative distance (spacing) affect the model's comprehension capability? This critical question has not been systematically investigated.

**Goal**: (1) Build a benchmark (LongPiBench) specifically designed to evaluate positional bias under multiple information pieces; (2) Systematically evaluate the performance of commercial and open-source models under different information spacing configurations; (3) Identify new patterns of positional bias.

**Key Insight**: The authors hypothesize that the distance between information pieces (spacing/proximity) is a key factor influencing the information integration capability of LLMs—relevant information might be easier to utilize when clustered together, whereas being scattered throughout a long text may lead to performance degradation.

**Core Idea**: Build a controllable long-context evaluation benchmark with multiple information pieces. By systematically varying the quantity, position, and spacing of these information pieces, the benchmark uncovers a new bias pattern, termed "information spacing bias."

## Method

### Overall Architecture

The construction of LongPiBench follows this pipeline: (1) Design questions that require multiple information pieces to answer; (2) Embed these information pieces into a large amount of distractor text to construct long-context inputs; (3) Systematically control the positions and spacing of these information pieces; (4) Evaluate the answer accuracy of various LLMs under diverse configurations. The overall framework covers multiple task types, ranging from simple information retrieval to complex tasks requiring cross-piece reasoning.

### Key Designs

1. **Task Design with Multiple Information Pieces**:

    - Function: Ensure that the evaluated questions naturally require multiple scattered information pieces to answer, rather than a single "needle-in-a-haystack" style retrieval
    - Mechanism: The authors design various task types, including multi-hop reasoning (requiring chained integration of $A \rightarrow B \rightarrow C$ information), aggregation questions (requiring synthesis after gathering multiple independent facts), and conditional filtering questions (requiring simultaneous satisfaction of multiple constraints distributed in different locations). Each task guarantees that at least 2–5 information pieces are required for a correct answer
    - Design Motivation: Simulate the scattered nature of information in real-world application scenarios, making the benchmark evaluation closer to actual demands

2. **Spacing Control Mechanism**:

    - Function: Precisely control the distance between multiple relevant information pieces
    - Mechanism: Under the condition of a fixed total context length, irrelevant padding text is inserted between the information pieces. The distance between information pieces is controlled by adjusting the amount of distractor text. Various spacing configurations are designed: tightly clustered (all information pieces placed adjacently), evenly scattered (equidistantly distributed across the entire context), and unevenly scattered (some close, some far). Meanwhile, absolute positions (information in the first, middle, or last third) are controlled to rule out interference from pure positional effects
    - Design Motivation: Precisely quantify the impact of the "information spacing" variable on model performance while excluding other confounding factors

3. **Multi-dimensional Evaluation System**:

    - Function: Comprehensively evaluate positional bias from different perspectives
    - Mechanism: Evaluation dimensions include: (a) a complete experimental matrix of the number of information pieces (2/3/5) $\times$ spacing configuration (tight/even/scattered) $\times$ absolute position (front/middle/back); (b) performance variations under different context lengths (8K/16K/32K/64K); (c) consistency of bias patterns across different task types. Standard repetitions and averaging are applied across all configurations to ensure statistical reliability
    - Design Motivation: Avoid drawing one-sided conclusions from a single dimension and provide a comprehensive profile of the biases

### Loss & Training

This paper is a benchmark/evaluation work and does not involve the design of new training strategies or loss functions. During evaluation, few-shot prompting is used for each model to ensure understanding of the task format, with Exact Match (EM) and F1 score as the primary evaluation metrics.

## Key Experimental Results

### Main Results

Evaluation results across 11 models (5 commercial + 6 open-source) under a 32K context length with 3 information fragments:

| Model | Tightly Clustered | Evenly Scattered | Unevenly Scattered | Spacing Bias $\Delta$ |
|------|---------|---------|-----------|----------|
| GPT-4o | 87.3 | 79.2 | 76.8 | -10.5 |
| Claude-3.5 | 89.1 | 81.5 | 78.3 | -10.8 |
| Gemini-1.5-Pro | 85.6 | 78.9 | 75.4 | -10.2 |
| Llama-3-70B | 82.4 | 72.1 | 68.5 | -13.9 |
| Mistral-Large | 80.8 | 70.3 | 66.7 | -14.1 |
| Qwen2-72B | 83.5 | 74.8 | 71.2 | -12.3 |

> Spacing Bias $\Delta$ = Unevenly Scattered - Tightly Clustered; a larger negative value indicates more severe bias.

### Ablation Study

| Experimental Variable | Tight Clustering Accuracy | Max Scattering Accuracy | Performance Gap |
|----------|-------------|-------------|---------|
| 2 information pieces | 90.2 | 83.7 | -6.5 |
| 3 information pieces | 85.6 | 75.4 | -10.2 |
| 5 information pieces | 78.9 | 62.3 | -16.6 |
| Context 8K | 91.3 | 87.1 | -4.2 |
| Context 32K | 85.6 | 75.4 | -10.2 |
| Context 64K | 79.8 | 64.2 | -15.6 |
| Single information piece (Front/Middle/Back) | 92.1/90.8/91.5 | — | $\Delta$ < 2% |

### Key Findings

- **"Lost-in-the-middle" is largely mitigated**: Under the single information piece setting, the performance gap of current mainstream models among front/middle/back positions has narrowed to less than 2%, indicating that this long-standing issue is being addressed.
- **Information spacing bias is the new core challenge**: Even for the strongest commercial models, performance drops by more than 10% when relevant information pieces transition from tightly clustered to scattered distributions.
- **The number of information pieces amplifies the bias effect**: Moving from 2 to 5 information pieces, the performance gap caused by spacing expands from ~6% to ~16%, demonstrating a superlinear growth.
- **Longer context exacerbates the problem**: As the context length increases, the spacing bias becomes more severe, aligning with the intuition of scattered model attention.
- **Open-source models experience more severe bias**: In terms of spacing bias, open-source models generally underperform commercial models by 2–4 percentage points.

## Highlights & Insights

- **The shift from "where is the information" to "how far apart is the information"** is the most core contribution of this work. While most current long-context evaluations are capped at position sensitivity tests of a single information piece, this work extends the perspective to spatial relationships among multiple information pieces, which is much closer to real-world application scenarios.
- **The controlled variable methodology in experimental design** is highly instructive. By fixing the total length and varying only the spacing, the spacing effect is cleanly isolated. This benchmark design methodology can be transferred to the evaluation of other long-context capabilities.
- The discovered rule that "more information pieces lead to more severe spacing bias" has direct guiding significance for RAG system design: retrieved fragments should be organized together as much as possible rather than scattered throughout the prompt.

## Limitations & Future Work

- **Limited coverage of task types**: Currently, the tasks mainly consist of QA and information retrieval, lacking evaluation on generative tasks (such as long-document summarization that requires synthesizing information from the entire text).
- **Insufficient causal analysis**: The paper reveals the existence of spacing bias but does not deeply analyze its root causes at the attention mechanism level. Accompanying attention map visualizations to explain why scattered information is harder to integrate would further strengthen the paper's persuasiveness.
- **Lack of mitigation strategies**: The problem is identified, but no solution is proposed. Future work could explore methods to reduce spacing bias through position encoding, attention mechanisms, or training data construction.
- In practical applications, the positions of information pieces are uncontrollable. Investigating how to mitigate bias during the inference stage through input reranking is a direction worthy of study.

## Related Work & Insights

- **vs "Lost in the Middle" (Liu et al., 2024)**: The former focuses on the absolute positional effect of a single piece of information, while this work focuses on the relative distance effect of multiple pieces, making them complementary. This paper finds that single-information positional bias has been substantially mitigated, whereas multi-information spacing bias remains severe.
- **vs RULER (Hsieh et al., 2024)**: RULER also evaluates long-context capability but focuses on "needle-in-a-haystack" style single-information retrieval. The multi-information setting in this work is more challenging and practically meaningful.
- **vs BABILong**: This benchmark evaluates multi-hop reasoning performance in long contexts, but it does not systematically control the information spacing variable. LongPiBench provides finer-grained control.

## Rating

- Novelty: ⭐⭐⭐⭐ "Information spacing bias" is a novel and important finding that fills a gap in long-context evaluation.
- Experimental Thoroughness: ⭐⭐⭐⭐ The systematic evaluation across 11 models and multiple configurations is highly thorough, though it lacks an in-depth analysis of the underlying causes.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and reasonable experimental design, though some experimental details could be further elaborated.
- Value: ⭐⭐⭐⭐ Provides clear guidance for the evaluation and improvement direction of long-context LLMs, with practical reference value for RAG system design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Boosting Long-Context Information Seeking via Query-Guided Activation Refilling](boosting_long-context_information_seeking_via_query-guided_activation_refilling.md)
- [\[ICML 2025\] Long-Short Alignment for Effective Long-Context Modeling in LLMs](../../ICML2025/llm_efficiency/long-short_alignment_for_effective_long-context_modeling_in_llms.md)
- [\[ACL 2025\] LADM: Long-context Training Data Selection with Attention-based Dependency Measurement for LLMs](ladm_long_context_data.md)
- [\[ACL 2025\] Mitigating Posterior Salience Attenuation in Long-Context LLMs with Positional Contrastive Decoding](mitigating_posterior_salience_attenuation_in_long-context_llms_with_positional_c.md)
- [\[ACL 2025\] What Really Matters in Many-Shot Attacks? An Empirical Study of Long-Context Vulnerabilities in LLMs](many_shot_attacks_long_context.md)

</div>

<!-- RELATED:END -->

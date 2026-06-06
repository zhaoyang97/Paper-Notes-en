---
title: >-
  [Paper Note] What Factors Affect LLMs and RLLMs in Financial Question Answering?
description: >-
  [ACL 2026][Multilingual & Machine Translation][Financial Question Answering] This paper systematically investigates the impact of prompting methods, Agent frameworks…
tags:
  - "ACL 2026"
  - "Multilingual & Machine Translation"
  - "Financial Question Answering"
  - "Reasoning Large Language Models"
  - "Long CoT"
  - "Prompting Methods"
  - "Multilingual Alignment"
date: 2026-05-08
content_hash: 282119d012671cd8
---

# What Factors Affect LLMs and RLLMs in Financial Question Answering?

**Conference**: ACL 2026  
**arXiv**: [2507.08339](https://arxiv.org/abs/2507.08339)  
**Code**: [https://github.com/WPENGxs/LLM_RLLM_financial_analysis](https://github.com/WPENGxs/LLM_RLLM_financial_analysis)  
**Area**: Multilingual/Financial NLP  
**Keywords**: Financial Question Answering, Reasoning Large Language Models, Long CoT, Prompting Methods, Multilingual Alignment

## TL;DR

This paper systematically investigates the impact of prompting methods, Agent frameworks, and multilingual alignment methods on LLMs and RLLMs (Reasoning Large Language Models) for financial question-answering tasks. It finds that existing methods essentially improve LLM performance by simulating Long CoT, but have limited effects on RLLMs that already possess inherent Long CoT capabilities.

## Background & Motivation

**Background**: Large Language Models have made significant progress in the field of financial question answering. Researchers utilize prompting methods (such as CoT), Agent frameworks, and multilingual alignment to enhance the financial reasoning capabilities of LLMs. Meanwhile, Reasoning Large Language Models (RLLMs, e.g., DeepSeek-R1, O4-mini) significantly strengthen complex problem-solving through Long CoT.

**Limitations of Prior Work**: Although various enhancement methods continue to emerge, there is a lack of systematic research to determine which methods truly unlock the potential of LLMs and RLLMs in the financial domain. Particularly since the advent of RLLMs, it remains unclear whether traditional enhancement methods are still effective.

**Key Challenge**: Existing prompting methods and Agent frameworks mainly improve performance by extending the reasoning chain. This creates redundancy with the built-in Long CoT capabilities of RLLMs, leading to extremely limited gains or even negative effects on RLLMs.

**Goal**: Using 5 LLMs and 4 RLLMs, systematically evaluate the effects of prompting methods, Agent frameworks, and multilingual alignment methods on financial question-answering tasks.

**Key Insight**: Starting from the hypothesis that "long reasoning chains are the key bottleneck for performance improvement," the study verifies this by comparing the performance differences of LLMs and RLLMs under the same methods.

**Core Idea**: Effective methods currently used to improve LLM performance in financial QA are essentially simulating Long CoT. Since RLLMs naturally possess this capability, traditional methods yield diminishing marginal returns for them.

## Method

### Overall Architecture

This paper is a systematic empirical study that does not propose a new method but tests combinations of 9 models $\times$ 7 methods on the FAMMA financial QA benchmark. The evaluation covers three dimensions: prompting methods (Direct, Zero-shot CoT, Plan-and-Solve), Agent frameworks (Self-Refine, S3 Agent), and multilingual alignment methods (Direct, Translate-en, Cross-lingual Prompting).

### Key Designs

1.  **Prompting Method Comparison**:
    - **Function**: Evaluate the impact of different prompting strategies on the financial reasoning of LLMs/RLLMs.
    - **Mechanism**: Three representative prompting methods are selected—Direct (direct input), Zero-shot CoT ("let's think step by step"), and Plan-and-Solve (understanding the problem first, then formulating a plan for step-by-step solving). Plan-and-Solve performs best on most LLMs but may decrease RLLM performance.
    - **Design Motivation**: Verify whether the source of gain for prompting methods is the simulation of Long CoT and whether this simulation is redundant for RLLMs already capable of Long CoT.

2.  **Agent Framework Comparison**:
    - **Function**: Evaluate the gains of multi-Agent collaboration for LLMs/RLLMs.
    - **Mechanism**: Test Self-Refine (LLM iterates on its own output feedback for only one round) and S3 Agent (collaborative reasoning from three perspectives: surface expression, semantic information, and emotional expression). Smaller LLMs (e.g., Llama-3.1-8B) benefit more from Agent frameworks, while gains for large LLMs and RLLMs are limited.
    - **Design Motivation**: Explore whether Agent frameworks can compensate for LLM reasoning deficiencies through structured collaboration and whether they remain valuable for RLLMs.

3.  **Multilingual Alignment Method Comparison**:
    - **Function**: Evaluate the enhancement effects of multilingual methods on Chinese and French financial QA.
    - **Mechanism**: Compare Direct (English prompt + native language question), Translate-en (translate to English then answer), and Cross-lingual Prompting (CLP, a two-stage approach with cross-lingual alignment prompting + task solver). CLP achieves the best results for LLMs (average gain of 4-5%) but shows limited or negative effects for RLLMs.
    - **Design Motivation**: Verify whether the gains from multilingual alignment also stem from extending the reasoning chain and whether RLLMs have achieved self-alignment through Long CoT.

### Loss & Training

This is a pure evaluation study and does not involve training. All models are used in inference mode, and open-ended questions are scored by GPT-4o-mini based on standard answers.

## Key Experimental Results

### Main Results

| Model | Method | Overall Acc | Gain compared to Direct |
| :--- | :--- | :--- | :--- |
| DeepSeek-V3 (LLM) | Direct | 58.86 | - |
| DeepSeek-V3 (LLM) | Plan-and-Solve | 58.81 | -0.05 |
| DeepSeek-V3 (LLM) | S3 Agent | 56.81 | -2.05 |
| DeepSeek-R1-Distill-32B (RLLM) | Direct | 53.41 | - |
| DeepSeek-R1-Distill-32B (RLLM) | S3 Agent | 54.29 | +0.88 |
| O4-mini (RLLM) | Direct | 65.29 | - |
| O4-mini (RLLM) | Zero-shot CoT | 66.52 | +1.23 |
| Llama-3.1-8B (LLM) | Direct | 16.50 | - |
| Llama-3.1-8B (LLM) | S3 Agent | 24.62 | +8.12 |

### Ablation Study

| Configuration | Qwen-2.5-32B | R1-Distill-32B | Description |
| :--- | :--- | :--- | :--- |
| Direct | 44.88 | 53.41 | Average gain of 7.4% after R1 distillation |
| Zero-shot CoT | 46.11 | 53.62 | Weak gain for RLLM from prompting |
| Plan-and-Solve | 44.06 | 53.26 | Plan-and-Solve even decreases RLLM performance |
| Self-Refine | 45.19 | 47.96 | Self-Refine significantly decreases RLLM performance |
| S3 Agent | 45.34 | 54.29 | Agent collaboration provides some gain for RLLM |

### Key Findings

- **Small models benefit most from Agent frameworks**: Llama-3.1-8B performance improved from 16.50% to 24.62% (+49%) using S3 Agent, while the larger DeepSeek-V3 showed a decline.
- **Long CoT is the core bottleneck**: Effective methods for LLMs essentially simulate Long CoT; output token count is positively correlated with performance. RLLMs output approximately 2000 tokens on average, compared to only 250-470 tokens for LLMs.
- **Self-alignment capability of RLLMs**: In multilingual scenarios, RLLMs automatically achieve cross-lingual reasoning through Long CoT without requiring additional multilingual alignment methods.
- **The "Overthinking" problem**: RLLMs generate excessive tokens for simple questions without a corresponding performance boost, exhibiting a clear overthinking phenomenon.
- **Scaling Law still holds**: For the Qwen-3 series across 0.6B to 32B, performance improves and output length increases with parameter size. Enabling thinking mode results in an average 16.9% gain.

## Highlights & Insights

- **Systematic comparison of LLM vs RLLM**: For the first time in a financial QA context, the differentiated impacts of prompting, Agent frameworks, and multilingual methods on LLMs and RLLMs are systematically compared, revealing the importance of Long CoT as a unifying explanatory framework.
- **Methodological implications**: For LLMs, more effort should be invested in designing methods that extend the reasoning chain; for RLLMs, the focus should shift toward more complex Agent mechanisms to regulate output rather than simply extending thinking time.
- **Dynamic CoT length control**: Addressing the overthinking issue in RLLMs by dynamically adjusting CoT length based on problem complexity is an important future research direction.

## Limitations & Future Work

- All models were run only once, lacking statistical significance tests across multiple runs.
- Only the text subset of FAMMA was used, without involving multimodal financial QA.
- The exploration of Agent frameworks (e.g., only 1 round of iteration for Self-Refine) is shallow, and more complex multi-turn Agent systems were not tested.
- Enhancement methods specifically designed for RLLMs were not explored.

## Related Work & Insights

- **vs BloombergGPT**: BloombergGPT trained a 50 billion parameter finance-specific LLM, whereas this paper explores the release of financial capabilities in general LLMs from a reasoning strategy perspective.
- **vs FinBen**: FinBen is a comprehensive financial benchmark; this paper uses FAMMA but focuses on method comparison rather than model ranking.

## Rating

- **Novelty**: ⭐⭐⭐ The research perspective is valuable, but it does not propose new methods, qualifying it as an empirical investigation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Large-scale comparison of 9 models and 7 methods provides sufficient data.
- **Writing Quality**: ⭐⭐⭐⭐ The analysis is clear, and the findings are well-summarized.
- **Value**: ⭐⭐⭐⭐ Provides practical guidance for the financial NLP community in choosing LLM/RLLM strategies.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Cross-Cultural Transfer of Emoji Semantics and Sentiment in Financial Social Media](cross-cultural_transfer_of_emoji_semantics_and_sentiment_in_financial_social_med.md)
- [\[ACL 2026\] Vocab Diet: Reshaping the Vocabulary of LLMs via Vector Arithmetic](vocab_diet_reshaping_the_vocabulary_of_llms_via_vector_arithmetic.md)
- [\[ACL 2026\] NiuTrans.LMT: Toward Inclusive and Scalable Multilingual Machine Translation with LLMs](niutranslmt_toward_inclusive_and_scalable_multilingual_machine_translation_with_.md)
- [\[ACL 2026\] From Traditional Taggers to LLMs: A Comparative Study of POS Tagging for Medieval Romance Languages](from_traditional_taggers_to_llms_a_comparative_study_of_pos_tagging_for_medieval.md)
- [\[ACL 2026\] Location Not Found: Exposing Implicit Local and Global Biases in Multilingual LLMs](location_not_found_exposing_implicit_local_and_global_biases_in_multilingual_llm.md)

</div>

<!-- RELATED:END -->

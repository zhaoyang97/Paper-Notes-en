---
title: >-
  [Paper Note] What Factors Affect LLMs and RLLMs in Financial Question Answering?
description: >-
  [ACL 2026 Findings][Multilingual & Machine Translation][Financial QA] This paper systematically investigates the impact of prompting methods, Agent frameworks, and multilingual alignment methods on LLMs and RLLMs (Reasoning LLMs) in financial QA tasks. It finds that existing methods essentially improve LLM performance by simulating Long CoT, but provide limited benefits to RLLMs that already possess inherent Long CoT capabilities.
tags:
  - "ACL 2026 Findings"
  - "Multilingual & Machine Translation"
  - "Financial QA"
  - "Reasoning LLMs"
  - "Long CoT"
  - "Prompting Methods"
  - "Multilingual Alignment"
date: 2026-05-08
content_hash: 45d926630a16a46b
---

# What Factors Affect LLMs and RLLMs in Financial Question Answering?

**Conference**: ACL 2026 Findings  
**arXiv**: [2507.08339](https://arxiv.org/abs/2507.08339)  
**Code**: [https://github.com/WPENGxs/LLM_RLLM_financial_analysis](https://github.com/WPENGxs/LLM_RLLM_financial_analysis)  
**Area**: Multilingual/Financial NLP  
**Keywords**: Financial QA, Reasoning LLMs, Long CoT, Prompting Methods, Multilingual Alignment

## TL;DR

This paper systematically investigates the impact of prompting methods, Agent frameworks, and multilingual alignment methods on LLMs and RLLMs (Reasoning LLMs) in financial QA tasks. It finds that existing methods essentially improve LLM performance by simulating Long CoT, but provide limited benefits to RLLMs that already possess inherent Long CoT capabilities.

## Background & Motivation

**Background**: Large Language Models (LLMs) have achieved significant progress in the financial QA domain. Researchers use prompting methods (e.g., CoT), Agent frameworks, and multilingual alignment to enhance the financial reasoning capabilities of LLMs. Meanwhile, Reasoning LLMs (RLLMs, such as DeepSeek-R1 and O4-mini) have significantly enhanced complex reasoning through Long CoT.

**Limitations of Prior Work**: Despite the emergence of various enhancement methods, there is a lack of systematic research to identify which methods truly release the potential of LLMs and RLLMs in the financial sector. Specifically, it remains unclear whether traditional enhancement methods remain effective after the emergence of RLLMs.

**Key Challenge**: Existing prompting methods and Agent frameworks primarily improve performance by extending the reasoning chain. This creates redundancy with the built-in Long CoT capabilities of RLLMs, leading to marginal gains or even negative effects on RLLMs.

**Goal**: Systematically evaluate the effects of prompting methods, Agent frameworks, and multilingual alignment methods on financial QA tasks using 5 LLMs and 4 RLLMs.

**Key Insight**: Starting from the hypothesis that "the length of the reasoning chain is the key performance bottleneck," the paper validates this by comparing the performance differences between LLMs and RLLMs under the same methods.

**Core Idea**: Effective methods currently used to improve LLM performance in financial QA are essentially simulating Long CoT. Since RLLMs naturally possess this capability, traditional methods show diminishing marginal returns for RLLMs.

## Method

### Overall Architecture

This study is a systematic empirical research project that does not propose a new method but tests combinations of 9 models × 7 methods on the FAMMA financial QA benchmark. The evaluation covers three dimensions: prompting methods (Direct, Zero-shot CoT, Plan-and-Solve), Agent frameworks (Self-Refine, S3 Agent), and multilingual alignment methods (Direct, Translate-en, Cross-lingual Prompting).

### Key Designs

**1. Prompting Method Comparison: Examining whether prompting gains come from "simulating Long CoT for the model"**

If the benefit of a prompting method is essentially to temporarily extend the reasoning chain, it should be redundant for an RLLM with built-in Long CoT. The authors tested three representative prompts: Direct (direct input), Zero-shot CoT (adding "let's think step by step"), and Plan-and-Solve (understanding the problem first, then planning a step-by-step solution).

Results confirmed the hypothesis: Plan-and-Solve, the most "reasoning-intensive" prompt, performed best on most standard LLMs but often degraded performance on RLLMs. This is because RLLMs already generate long reasoning chains; external structured prompts conflict with their internal Long CoT, interfering with their natural rhythm rather than complementing it.

**2. Agent Framework Comparison: Whether the value of multi-agent collaboration diminishes with the model's own reasoning ability**

Beyond prompting, another enhancement involves multi-round self-feedback or multi-perspective collaboration. The authors tested two: Self-Refine (iterative feedback on the model's own output, limited to 1 round here) and S3 Agent (collaborative reasoning from surface expression, semantic information, and emotional expression perspectives).

A clear inverse correlation was observed: weaker models benefit more from Agent frameworks. Llama-3.1-8B jumped from $16.50\%$ to $24.62\%$ using S3 Agent, while gains for large LLMs and RLLMs were limited. This indicates that Agent collaboration primarily "compensates" for small models with insufficient reasoning; for RLLMs with strong native reasoning, the marginal value of such external scaffolding is small.

**3. Multilingual Alignment Method Comparison: Verifying if cross-lingual gains also stem from extending reasoning chains**

In non-English financial QA (e.g., Chinese, French), a common practice is to align the problem to English before reasoning. Three methods were compared: Direct (English prompt + native language question), Translate-en (translate to English then answer), and Cross-lingual Prompting (CLP, a two-stage process of cross-lingual alignment prompt + task solver).

CLP worked best for standard LLMs, providing an average gain of $4-5\%$, but was less effective or even negative for RLLMs. Viewed alongside the first two designs, all three enhancements (Prompt/Agent/Multilingual) point to the same explanation: their gains arise from "finding ways to make the model reason more steps," whereas RLLMs already complete cross-lingual self-alignment internally through Long CoT.

### Loss & Training

This is a pure evaluation study and does not involve training. All models used inference mode, and open-ended questions were scored by GPT-4o-mini based on standard answers.

## Key Experimental Results

### Main Results

| Model | Method | Overall Acc | Gain vs Direct |
|--------|------|------|----------|
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
|------|---------|------|------|
| Direct | 44.88 | 53.41 | Average gain of 7.4% after R1 distillation |
| Zero-shot CoT | 46.11 | 53.62 | Minimal gain for RLLM from prompting |
| Plan-and-Solve | 44.06 | 53.26 | Plan-and-Solve even decreases RLLM performance |
| Self-Refine | 45.19 | 47.96 | Significant decrease for RLLM with Self-Refine |
| S3 Agent | 45.34 | 54.29 | Some gain for RLLM from Agent collaboration |

### Key Findings

- **Small models benefit most from Agent frameworks**: Llama-3.1-8B performance increased from $16.50\%$ to $24.62\%$ ($+49\%$) with S3 Agent, while DeepSeek-V3 experienced a decline.
- **Long CoT is the core bottleneck**: Effective methods for LLMs essentially simulate Long CoT; output token count is positively correlated with performance. RLLMs output ~2000 tokens on average, while LLMs output only 250-470 tokens.
- **Self-alignment capability of RLLMs**: RLLMs automatically achieve cross-lingual reasoning via Long CoT in multilingual scenarios, requiring no additional alignment methods.
- **Overthinking issue**: RLLMs generate excessive tokens on simple questions without improving performance, indicating a clear overthinking phenomenon.
- **Scaling Law still holds**: For the Qwen-3 series (0.6B to 32B), larger parameters correlate with better performance and longer outputs. Enabling thinking mode yielded an average improvement of $16.9\%$.

## Highlights & Insights

- **Systematic LLM vs RLLM Comparison**: This is the first study to systematically compare the differentiated impacts of prompting, Agents, and multilingual methods on LLMs and RLLMs in financial QA, revealing Long CoT as a unifying explanatory framework.
- **Methodological Insights**: For LLMs, more effort should be invested in designing methods that extend reasoning chains; for RLLMs, the focus should shift to more complex Agent mechanisms to regulate output rather than simply extending thinking.
- **Dynamic CoT Length Control**: To address RLLM overthinking, dynamically adjusting CoT length based on question complexity is an important future research direction.

## Limitations & Future Work

- All models were run only once, lacking statistical significance tests across multiple runs.
- Only the text subset of FAMMA was used, omitting multimodal financial QA.
- The exploration of Agent frameworks (e.g., only 1 round for Self-Refine) was shallow, and more complex multi-round systems were not tested.
- Enhancement methods specifically designed for RLLMs were not explored.

## Related Work & Insights

- **vs BloombergGPT**: While BloombergGPT trained a 50B parameter finance-specific LLM, this paper explores releasing the financial capabilities of general LLMs through inference strategies.
- **vs FinBen**: FinBen is a comprehensive financial benchmark; this paper uses FAMMA but focuses on method comparison rather than model ranking.

## Rating

- Novelty: ⭐⭐⭐ The research perspective is valuable, though it does not propose new methods (empirical investigation).
- Experimental Thoroughness: ⭐⭐⭐⭐ Large-scale comparison across 9 models and 7 methods with sufficient data.
- Writing Quality: ⭐⭐⭐⭐ Clear analysis and well-summarized findings.
- Value: ⭐⭐⭐⭐ Provides practical guidance for the financial NLP community in choosing strategies for LLMs/RLLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] AskQE: Question Answering as Automatic Evaluation for Machine Translation](../../ACL2025/multilingual_mt/askqe_question_answering_as_automatic_evaluation_for_machine_translation.md)
- [\[ACL 2026\] Cross-Cultural Transfer of Emoji Semantics and Sentiment in Financial Social Media](cross-cultural_transfer_of_emoji_semantics_and_sentiment_in_financial_social_med.md)
- [\[ACL 2026\] Vocab Diet: Reshaping the Vocabulary of LLMs via Vector Arithmetic](vocab_diet_reshaping_the_vocabulary_of_llms_via_vector_arithmetic.md)
- [\[ACL 2026\] No One Fits All: From Fixed Prompting to Learned Routing in Multilingual LLMs](no_one_fits_all_from_fixed_prompting_to_learned_routing_in_multilingual_llms.md)
- [\[ACL 2026\] NiuTrans.LMT: Toward Inclusive and Scalable Multilingual Machine Translation with LLMs](niutranslmt_toward_inclusive_and_scalable_multilingual_machine_translation_with_.md)

</div>

<!-- RELATED:END -->

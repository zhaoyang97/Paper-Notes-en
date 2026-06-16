---
title: >-
  [Paper Note] What Factors Affect LLMs and RLLMs in Financial Question Answering?
description: >-
  [ACL 2026][Multilingual & Translation][Long CoT] This paper systematically investigates the impact of prompting methods, agent frameworks, and multilingual alignment methods on Large Language Models (LLMs) and Reasoning LLMs (RLLMs) in financial question-answering tasks. It finds that existing enhancement methods essentially improve LLM performance by simulating Long
tags:
  - ACL 2026
  - Multilingual & Translation
  - Long CoT
date: 2026-05-08
content_hash: 05d2b0338b8c5389
---
# What Factors Affect LLMs and RLLMs in Financial Question Answering?

**Conference**: ACL 2026 Findings  
**arXiv**: [2507.08339](https://arxiv.org/abs/2507.08339)  
**Code**: [https://github.com/WPENGxs/LLM_RLLM_financial_analysis](https://github.com/WPENGxs/LLM_RLLM_financial_analysis)  
**Area**: Multilingual/Financial NLP  
**Keywords**: Financial QA, Reasoning LLMs, Long CoT, Prompting Methods, Multilingual Alignment

## TL;DR

This paper systematically investigates the impact of prompting methods, agent frameworks, and multilingual alignment methods on Large Language Models (LLMs) and Reasoning LLMs (RLLMs) in financial question-answering tasks. It finds that existing enhancement methods essentially improve LLM performance by simulating Long Chain-of-Thought (CoT), but provide limited benefits for RLLMs that natively possess Long CoT capabilities.

## Background & Motivation

**Background**: LLMs have achieved significant progress in the financial QA domain. Researchers utilize prompting methods (e.g., CoT), agent frameworks, and multilingual alignment to enhance the financial reasoning capabilities of LLMs. Concurrently, RLLMs (e.g., DeepSeek-R1, o4-mini) have remarkably strengthened complex reasoning through Long CoT.

**Limitations of Prior Work**: Despite the emergence of various enhancement methods, there is a lack of systematic research to determine which methods truly unlock the potential of LLMs and RLLMs in finance. Particularly after the rise of RLLMs, it remains unclear whether traditional enhancement methods remain effective.

**Key Challenge**: Existing prompting methods and agent frameworks primarily improve performance by extending the reasoning chain. This creates redundancy with the built-in Long CoT capability of RLLMs, resulting in marginal gains or even negative effects on RLLMs.

**Goal**: Systematically evaluate the impact of prompting methods, agent frameworks, and multilingual alignment methods on financial QA tasks using 5 LLMs and 4 RLLMs.

**Key Insight**: Starting from the hypothesis that "long reasoning chains are the key performance bottleneck," the study verifies this by comparing the performance differences between LLMs and RLLMs under identical methods.

**Core Idea**: Effective methods for improving LLM financial QA performance essentially simulate Long CoT. Since RLLMs natively possess this capability, traditional methods yield diminishing marginal returns for them.

## Method

### Overall Architecture

This study is a systematic empirical investigation. It does not propose a new method but tests combinations of 9 models × 7 methods on the FAMMA financial QA benchmark. The evaluation covers three dimensions: prompting methods (Direct, Zero-shot CoT, Plan-and-Solve), agent frameworks (Self-Refine, S3 Agent), and multilingual alignment methods (Direct, Translate-en, Cross-lingual Prompting).

### Key Designs

**1. Prompting Method Comparison: Verifying if prompt gains "simulate Long CoT"**

The authors hypothesize that if prompting benefits stem from temporarily extending the reasoning chain, they should be redundant for an RLLM with built-in Long CoT. They selected three representative prompts: Direct, Zero-shot CoT ("let's think step by step"), and Plan-and-Solve (understanding, planning, and solving).  
Results confirm the hypothesis: Plan-and-Solve, the most "reasoning-intensive" prompt, performs best on most standard LLMs but may degrade RLLM performance. This is because structured external prompts interfere with the RLLM's internal Long CoT rhythm.

**2. Agent Framework Comparison: Assessing if multi-agent collaboration value diminishes with model reasoning capability**

Beyond prompting, the authors tested two frameworks: Self-Refine (1-round iterative feedback) and S3 Agent (collaborative reasoning from surface expression, semantic information, and emotional expression perspectives).  
A clear inverse correlation was observed: weaker models benefit more from agent frameworks. Llama-3.1-8B improved from 16.50% to 24.62% using S3 Agent, while gains for large LLMs and RLLMs were limited. This suggests agent collaboration primarily compensates for the reasoning deficiencies of smaller models.

**3. Multilingual Alignment Method Comparison: Verifying if cross-lingual gains also stem from reasoning extension**

For non-English financial QA, common practices involve aligning questions to English. The authors compared Direct (English prompt + native question), Translate-en (translation before answering), and Cross-lingual Prompting (CLP, a two-stage alignment and solver prompt).  
CLP performed best for standard LLMs (4-5% average gain) but had limited or negative effects on RLLMs. This indicates that RLLMs complete cross-lingual alignment internally via Long CoT, making external alignment layers redundant.

### Loss & Training

Ours is a pure evaluation study and does not involve training. All models used inference mode; open-ended questions were scored by GPT-4o-mini based on gold standard answers.

## Key Experimental Results

### Main Results

| Model | Method | Overall Acc | Gain over Direct |
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
| Zero-shot CoT | 46.11 | 53.62 | Weak gain for RLLMs from prompting |
| Plan-and-Solve | 44.06 | 53.26 | Performance decrease for RLLMs |
| Self-Refine | 45.19 | 47.96 | Significant performance decrease for RLLMs |
| S3 Agent | 45.34 | 54.29 | Some gain for RLLMs from agent collaboration |

### Key Findings

- **Small models benefit most from agent frameworks**: Llama-3.1-8B performance improved from 16.50% to 24.62% (+49%) with S3 Agent, whereas the large DeepSeek-V3 showed a decline.
- **Long CoT is the core bottleneck**: Effective LLM methods essentially simulate Long CoT; output token count correlates positively with performance. RLLMs output ~2000 tokens on average, while LLMs output only 250-470 tokens.
- **RLLM self-alignment**: RLLMs automatically achieve cross-lingual reasoning via Long CoT in multilingual scenarios, eliminating the need for external alignment methods.
- **Overthinking**: RLLMs generate excessive tokens for simple questions without performance gains, indicating a clear overthinking phenomenon.
- **Scaling Law persists**: For the Qwen-3 series (0.6B to 32B), larger parameters yield better performance and longer outputs. Enabling reasoning mode provides an average gain of 16.9%.

## Highlights & Insights

- **Systematic Comparison of LLMs vs RLLMs**: This study provides the first systematic comparison of how prompting, agents, and multilingual methods differentially affect LLMs and RLLMs in financial QA, highlighting Long CoT as a unifying explanatory framework.
- **Methodological Implications**: For LLMs, research should focus on methods that extend the reasoning chain. For RLLMs, the focus should shift toward more complex agent mechanisms that regulate output rather than simply extending thinking.
- **Dynamic CoT Length Control**: To address RLLM overthinking, dynamically adjusting CoT length based on question complexity is an important future research direction.

## Limitations & Future Work

- All models were run only once, lacking statistical significance tests across multiple runs.
- Only the text subset of FAMMA was used, excluding multimodal financial QA.
- Agent framework exploration was limited (Self-Refine used only 1 iteration) and did not test complex multi-turn systems.
- Enhancement methods specifically designed for RLLMs were not explored.

## Related Work & Insights

- **vs BloombergGPT**: While BloombergGPT trained a 50B parameter finance-specific LLM, ours explores unlocking financial capabilities in general LLMs through reasoning strategies.
- **vs FinBen**: FinBen is a comprehensive financial benchmark; ours uses FAMMA but focuses on method comparison rather than model rankings.

## Rating

- Novelty: ⭐⭐⭐ Valuable perspective but does not propose new methods (empirical survey).
- Experimental Thoroughness: ⭐⭐⭐⭐ Large-scale comparison across 9 models and 7 methods.
- Writing Quality: ⭐⭐⭐⭐ Clear analysis and well-summarized findings.
- Value: ⭐⭐⭐⭐ Provides practical guidance for the financial NLP community when choosing LLM/RLLM strategies.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] AskQE: Question Answering as Automatic Evaluation for Machine Translation](../../ACL2025/multilingual_mt/askqe_question_answering_as_automatic_evaluation_for_machine_translation.md)
- [\[ACL 2025\] MTVQA: Benchmarking Multilingual Text-Centric Visual Question Answering](../../ACL2025/multilingual_mt/mtvqa_benchmarking_multilingual_text-centric_visual_question_answering.md)
- [\[ACL 2026\] Cross-Cultural Transfer of Emoji Semantics and Sentiment in Financial Social Media](cross-cultural_transfer_of_emoji_semantics_and_sentiment_in_financial_social_med.md)
- [\[ACL 2026\] Vocab Diet: Reshaping the Vocabulary of LLMs via Vector Arithmetic](vocab_diet_reshaping_the_vocabulary_of_llms_via_vector_arithmetic.md)
- [\[ACL 2026\] No One Fits All: From Fixed Prompting to Learned Routing in Multilingual LLMs](no_one_fits_all_from_fixed_prompting_to_learned_routing_in_multilingual_llms.md)

</div>

<!-- RELATED:END -->

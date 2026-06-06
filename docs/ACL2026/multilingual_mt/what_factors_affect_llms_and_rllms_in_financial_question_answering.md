---
title: >-
  [Paper Note] What Factors Affect LLMs and RLLMs in Financial Question Answering?
description: >-
  [ACL 2026][Multilingual & Machine Translation][Financial QA] This paper systematically investigates how prompting methods, agent frameworks, and multilingual alignment approaches affect LLMs and RLLMs (Reasoning Large La…
tags:
  - "ACL 2026"
  - "Multilingual & Machine Translation"
  - "Financial QA"
  - "Reasoning LLMs"
  - "Long CoT"
  - "Prompting Methods"
  - "Multilingual Alignment"
date: 2026-05-08
content_hash: f44c192575a777ab
---

# What Factors Affect LLMs and RLLMs in Financial Question Answering?

**Conference**: ACL 2026
**arXiv**: [2507.08339](https://arxiv.org/abs/2507.08339)  
**Code**: [https://github.com/WPENGxs/LLM_RLLM_financial_analysis](https://github.com/WPENGxs/LLM_RLLM_financial_analysis)  
**Area**: Multilingual / Financial NLP
**Keywords**: Financial QA, Reasoning LLMs, Long CoT, Prompting Methods, Multilingual Alignment

## TL;DR

This paper systematically investigates how prompting methods, agent frameworks, and multilingual alignment approaches affect LLMs and RLLMs (Reasoning Large Language Models) on financial question answering tasks. The key finding is that existing methods essentially improve LLM performance by simulating Long CoT, but offer limited gains for RLLMs that already possess native Long CoT capabilities.

## Background & Motivation

**Background**: Large language models have achieved notable progress in financial QA. Researchers have employed prompting strategies (e.g., CoT), agent frameworks, and multilingual alignment techniques to enhance LLMs' financial reasoning. Concurrently, RLLMs (e.g., DeepSeek-R1, O4-mini) have significantly strengthened complex reasoning through Long CoT.

**Limitations of Prior Work**: Despite a proliferation of enhancement methods, systematic studies examining which approaches truly unlock the potential of LLMs and RLLMs in the financial domain are lacking. In particular, it remains unclear whether traditional enhancement methods remain effective following the emergence of RLLMs.

**Key Challenge**: Existing prompting methods and agent frameworks primarily improve performance by extending the reasoning chain, which is redundant with the native Long CoT capability of RLLMs, resulting in minimal or even negative gains for RLLMs.

**Goal**: To systematically evaluate the effects of prompting methods, agent frameworks, and multilingual alignment approaches on financial QA tasks using 5 LLMs and 4 RLLMs.

**Key Insight**: The study proceeds from the hypothesis that long reasoning chains constitute the critical performance bottleneck, validating this by comparing LLM and RLLM performance under identical methods.

**Core Idea**: Effective methods for improving LLM performance on financial QA essentially simulate Long CoT, which RLLMs already possess natively; consequently, the marginal gains of traditional methods diminish for RLLMs.

## Method

### Overall Architecture

This paper presents a systematic empirical study that introduces no new methods, but instead evaluates combinations of 9 models × 7 methods on the FAMMA financial QA benchmark. The evaluation covers three dimensions: prompting methods (Direct, Zero-shot CoT, Plan-and-Solve), agent frameworks (Self-Refine, S3 Agent), and multilingual alignment methods (Direct, Translate-en, Cross-lingual Prompting).

### Key Designs

1. **Prompting Method Comparison**:

    - Function: Assess the impact of different prompting strategies on LLM/RLLM financial reasoning.
    - Mechanism: Three representative prompting methods are selected — Direct (straightforward input), Zero-shot CoT ("let's think step by step"), and Plan-and-Solve (comprehend the problem first, then devise a step-by-step plan). Plan-and-Solve yields the best performance on most LLMs but may degrade performance on RLLMs.
    - Design Motivation: Verify whether the gains from prompting methods originate from simulating Long CoT, and whether such simulation is redundant for RLLMs that already possess this capability.

2. **Agent Framework Comparison**:

    - Function: Evaluate the gains of multi-agent collaboration for LLMs/RLLMs.
    - Mechanism: Tests Self-Refine (iterative self-feedback optimization, 1 round only) and S3 Agent (collaborative reasoning from three perspectives: surface expression, semantic information, and sentiment). Smaller LLMs (e.g., Llama-3.1-8B) benefit more from agent frameworks, while large LLMs and RLLMs show limited gains.
    - Design Motivation: Explore whether agent frameworks can compensate for LLM reasoning deficiencies through structured collaboration, and whether they retain value for RLLMs.

3. **Multilingual Alignment Method Comparison**:

    - Function: Evaluate the effectiveness of multilingual methods on Chinese and French financial QA.
    - Mechanism: Three approaches are compared — Direct (English prompts with native-language questions), Translate-en (translate to English before answering), and Cross-lingual Prompting (CLP; a two-stage pipeline with cross-lingual alignment prompts and a task solver). CLP yields the best results for LLMs (average improvement of 4–5%), but provides limited or even negative gains for RLLMs.
    - Design Motivation: Verify whether multilingual alignment gains also stem from extending the reasoning chain, and whether RLLMs already achieve self-alignment through Long CoT.

### Loss & Training

This paper is a purely evaluative study and involves no training. All models are used in inference mode; open-ended questions are scored by GPT-4o-mini against reference answers.

## Key Experimental Results

### Main Results

| Model | Method | Overall Acc | Gain vs. Direct |
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

| Configuration | Qwen-2.5-32B | R1-Distill-32B | Notes |
|------|---------|------|------|
| Direct | 44.88 | 53.41 | R1 distillation yields avg. +7.4% |
| Zero-shot CoT | 46.11 | 53.62 | Minimal gain for RLLMs from prompting |
| Plan-and-Solve | 44.06 | 53.26 | Plan-and-Solve even degrades RLLMs |
| Self-Refine | 45.19 | 47.96 | Self-Refine substantially degrades RLLMs |
| S3 Agent | 45.34 | 54.29 | Agent collaboration offers modest gains for RLLMs |

### Key Findings

- **Smaller models benefit most from agent frameworks**: Llama-3.1-8B improves from 16.50% to 24.62% (+49%) with S3 Agent, while the large model DeepSeek-V3 actually declines.
- **Long CoT is the core bottleneck**: Effective methods for LLMs essentially simulate Long CoT; output token count is positively correlated with performance (Table 3). RLLMs generate approximately 2,000 tokens on average, compared to only 250–470 for LLMs.
- **RLLM self-alignment capability**: In multilingual settings, RLLMs automatically perform cross-lingual reasoning through Long CoT, without requiring additional multilingual alignment methods.
- **Overthinking phenomenon**: RLLMs generate excessive tokens on simple questions without corresponding performance gains, exhibiting a clear overthinking pattern.
- **Scaling Law holds**: Across the Qwen-3 series from 0.6B to 32B, larger models consistently achieve better performance and produce longer outputs. Enabling thinking mode yields an average improvement of 16.9%.

## Highlights & Insights

- **Systematic LLM vs. RLLM comparison**: This is the first study in the financial QA setting to systematically compare the differential effects of prompting methods, agent frameworks, and multilingual approaches on LLMs and RLLMs, revealing the importance of Long CoT as a unified explanatory framework.
- **Methodological implications**: For LLMs, greater effort should be devoted to designing methods that extend reasoning chains; for RLLMs, research should shift toward more sophisticated agent mechanisms to regulate outputs, rather than simply prolonging reasoning.
- **Dynamic CoT length control**: Dynamically adjusting CoT length according to question complexity to address RLLM overthinking represents an important future research direction.

## Limitations & Future Work

- All models are evaluated with a single run, lacking statistical significance testing across multiple runs.
- Only the text subset of FAMMA is used; multimodal financial QA is not addressed.
- Exploration of agent frameworks remains shallow (Self-Refine uses only 1 iteration), with no evaluation of more complex multi-turn agent systems.
- Enhancement methods specifically designed for RLLMs are not explored.

## Related Work & Insights

- **vs. BloombergGPT**: BloombergGPT trains a finance-specific LLM with 50 billion parameters, whereas this paper explores unlocking general-purpose LLM financial capabilities from the perspective of reasoning strategies.
- **vs. FinBen**: FinBen is a comprehensive financial benchmark; this paper uses FAMMA but focuses on method comparison rather than model ranking.

## Rating

- Novelty: ⭐⭐⭐ The research perspective is valuable, but no new methods are proposed; this is an empirical survey.
- Experimental Thoroughness: ⭐⭐⭐⭐ Large-scale comparison across 9 models and 7 methods with sufficient data.
- Writing Quality: ⭐⭐⭐⭐ Analysis is clear and findings are well summarized.
- Value: ⭐⭐⭐⭐ Provides practical guidance for the financial NLP community in selecting LLM/RLLM strategies.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Vocab Diet: Reshaping the Vocabulary of LLMs via Vector Arithmetic](vocab_diet_reshaping_the_vocabulary_of_llms_via_vector_arithmetic.md)
- [\[ACL 2026\] Location Not Found: Exposing Implicit Local and Global Biases in Multilingual LLMs](location_not_found_exposing_implicit_local_and_global_biases_in_multilingual_llm.md)
- [\[ACL 2026\] No One Fits All: From Fixed Prompting to Learned Routing in Multilingual LLMs](no_one_fits_all_from_fixed_prompting_to_learned_routing_in_multilingual_llms.md)
- [\[ACL 2026\] Mitigating Catastrophic Forgetting in Target Language Adaptation of LLMs via Source-Shielded Updates](mitigating_catastrophic_forgetting_in_target_language_adaptation_of_llms_via_sou.md)
- [\[ACL 2026\] Language on Demand, Knowledge at Core: Composing LLMs with Encoder-Decoder Translation Models for Extensible Multilinguality](language_on_demand_knowledge_at_core_composing_llms_with_encoder-decoder_transla.md)

</div>

<!-- RELATED:END -->

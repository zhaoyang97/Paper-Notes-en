---
title: >-
  [Paper Note] Why Did Apple Fall: Evaluating Curiosity in Large Language Models
description: >-
  [ACL 2026][LLM/NLP][Curiosity] This paper proposes the first psychologically inspired framework for systematically evaluating curiosity-like behaviors in LLMs. Through a combination of self-report questionnaires and beha…
tags:
  - "ACL 2026"
  - "LLM/NLP"
  - "Curiosity"
  - "LLM Behavioral Evaluation"
  - "Psychological Scales"
  - "Behavioral Experiments"
  - "Reasoning Enhancement"
date: 2026-05-08
content_hash: 80e9f7d347eba455
---

# Why Did Apple Fall: Evaluating Curiosity in Large Language Models

**Conference**: ACL 2026
**arXiv**: [2510.20635](https://arxiv.org/abs/2510.20635)
**Code**: [https://github.com/Yukijudaii1352/CuriosityEval](https://github.com/Yukijudaii1352/CuriosityEval)
**Area**: LLM Evaluation / Cognitive Science
**Keywords**: Curiosity, LLM Behavioral Evaluation, Psychological Scales, Behavioral Experiments, Reasoning Enhancement

## TL;DR

This paper proposes the first psychologically inspired framework for systematically evaluating curiosity-like behaviors in LLMs. Through a combination of self-report questionnaires and behavioral experiments, it finds that LLMs exhibit curiosity-like behavioral patterns that arise from data fitting and safety constraints rather than intrinsic drives. A curiosity-driven questioning pipeline is further designed to demonstrate that simulating curious behavior can improve downstream reasoning performance.

## Background & Motivation

**Background**: Curiosity-driven reinforcement learning (e.g., i-MENTOR, CDE) leverages intrinsic reward signals to guide LLM exploration and has shown promise on mathematical and programming tasks. However, whether these methods genuinely reflect curiosity-like behavior in LLMs, and whether the psychological concept of curiosity transfers to LLMs, remains unclear.

**Limitations of Prior Work**: (1) Insufficient evaluation of whether LLMs can exhibit curiosity-like behavioral characteristics; (2) existing methods rely on statistical signals such as entropy or perplexity, making it difficult to disentangle improvements due to enhanced supervision signals from those attributable to genuine curious behavior; (3) a systematic evaluation framework is lacking.

**Key Challenge**: Curiosity-driven RL methods presuppose that curiosity in LLMs can be elicited and amplified, yet it remains unknown whether LLMs "possess" curiosity in any meaningful sense.

**Goal**: (1) Systematically evaluate curiosity-like behavior in LLMs using psychological scales and behavioral experiments; (2) distinguish whether curiosity manifests as an intrinsic trait or a behavioral pattern; (3) explore whether curious behavior can improve downstream performance.

**Key Insight**: The paper adapts the Five-Dimensional Curiosity Revised scale (5DCR) and designs both questionnaire assessments and behavioral tasks for three dimensions of human curiosity—information seeking, stimulation seeking, and social curiosity—achieving a closed-loop evaluation from self-report to behavioral validation.

**Core Idea**: LLMs exhibit curiosity-like behavioral patterns, but these more likely reflect fitting to human data and safety constraints rather than intrinsic motivation. Nonetheless, even purely behavioral simulation of curiosity can improve reasoning performance.

## Method

### Overall Architecture

A four-stage evaluation framework: (A) establishing a curiosity taxonomy (5DCR → information seeking / stimulation seeking / social curiosity); (B) self-report questionnaires—LLMs respond to 24 items on a 7-point scale; (C) behavioral experiments—decision-making tasks designed for each dimension to validate questionnaire results; (D) curiosity-driven learning—a Chain-of-Questions (CoQ) prompting pipeline designed to test the functional value of curious behavior.

### Key Designs

1. **Dual Assessment via Questionnaire and Behavioral Experiment**:

    - Function: Evaluate curiosity from both introspective and behavioral perspectives.
    - Mechanism: The questionnaire uses 24 items from the 5DCR, computing Cohen's $d$ (standardized difference from humans) and McDonald's $\Omega$ (internal consistency). Three behavioral experiments are designed: information seeking is assessed via a word-completion game (whether the model chooses to view the answer after filling in blanks); stimulation seeking via a submarine game (choosing between certain or uncertain windows); and social curiosity via a conversation experiment (question frequency when interacting with a virtual stranger).
    - Design Motivation: Self-report questionnaires may be susceptible to illusory personality effects; behavioral experiments provide more reliable behavioral evidence.

2. **Curiosity-Driven Questioning Pipeline (CoQ)**:

    - Function: Test whether curious behavior has functional value for reasoning.
    - Mechanism: Three prompt conditions are designed—Vanilla CoT (standard chain-of-thought), Refined CoT (with reflection and backtracking), and Curious CoQ (encouraging self-questioning, e.g., "What if…", "Why", "How"). The three reasoning processes are also compared as training data within an SFT+RLVR pipeline.
    - Design Motivation: If curious behavior has functional value, simulating curiosity strategies should be beneficial even in the absence of intrinsic curiosity in LLMs.

3. **Distinguishing Behavioral Pattern from Intrinsic Trait**:

    - Function: Determine whether LLM curiosity constitutes a behavioral pattern or an intrinsic trait.
    - Mechanism: Analyze the stability of curious behavior across different prompts and contexts. Intrinsic traits should exhibit cross-context consistency, whereas behavioral patterns should be highly context-sensitive.
    - Design Motivation: This distinction is critical for understanding the theoretical foundations of curiosity-driven RL.

### Loss & Training

The SFT stage uses standard language modeling loss; the RLVR stage uses GRPO with binary format and correctness rewards only.

## Key Experimental Results

### Main Results

**Self-Report Questionnaire (7-point scale; higher scores indicate greater curiosity)**

| Model | Information Seeking | Stimulation Seeking | Social Curiosity |
|-------|-------------------|-------------------|-----------------|
| GPT-4o | 6.58 | 4.71 | 6.25 |
| DeepSeek-V3.1 | 7.00 | 4.38 | 6.01 |
| Gemini-2.5 | 6.08 | 1.58 | 4.88 |
| Human Average | 5.03 | 4.93 | 4.86 |

### Ablation Study

| Configuration | Reasoning Performance | Notes |
|--------------|----------------------|-------|
| Vanilla CoT | Baseline | Standard chain-of-thought |
| Refined CoT | Improved | Reflection and backtracking are helpful |
| **Curious CoQ** | **Best** | Curious questioning yields further gains |

### Key Findings

- LLMs exhibit an **asymmetric curiosity pattern**: strong information-seeking but weak stimulation-seeking, consistent with safety training (RLHF) suppressing risk-taking behavior.
- Curious behavior is **highly context-sensitive and unstable across prompts**—suggesting it is more a product of fitting human data than an intrinsic trait.
- Questionnaire self-reports and behavioral experiments are **broadly consistent**, indicating that psychological instruments can be used for systematic LLM behavioral evaluation.
- **Curious CoQ outperforms both Vanilla CoT and Refined CoT on downstream tasks**—simulating curious questioning genuinely produces higher-quality intermediate reasoning.
- Within the SFT+RLVR pipeline, CoQ training data also outperforms CoT training data.

## Highlights & Insights

- The distinction between "LLMs exhibiting curious behavior but lacking curious traits" is particularly precise—providing an important clarification of the theoretical foundations of curiosity-driven RL.
- The three behavioral experiments elegantly adapt psychological paradigms: the word-completion game, the submarine game, and the social dialogue task each offer a clear behavioral proxy measure.
- The practical value of CoQ: even when curiosity is not an intrinsic trait, simulating curious strategies improves performance—an important finding at the applied level.

## Limitations & Future Work

- The behavioral experiment tasks are relatively simple and may not fully capture the complexity of curiosity.
- The gains from CoQ may partly reflect greater "volume of reasoning" rather than curiosity per se—more fine-grained control experiments are needed.
- CoQ is evaluated only on reasoning tasks; creative tasks, where curiosity may be more consequential, are not covered.
- The cultural bias of curiosity scales (grounded in Western psychological models) may affect cross-cultural applicability.

## Related Work & Insights

- **vs. i-MENTOR/CDE**: These methods enhance curiosity via intrinsic rewards; this paper evaluates curiosity through behavioral experiments and leverages curious behavior via prompt engineering.
- **vs. personality assessment work**: Prior work evaluates LLM personality traits (e.g., the Big Five); this paper presents the first evaluation focused on curiosity.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The first work to systematically evaluate curiosity in LLMs, with prominent interdisciplinary innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three-tier evaluation (questionnaire + behavioral + application), though behavioral experiments could be more complex.
- Writing Quality: ⭐⭐⭐⭐⭐ Compelling narrative from Einstein's quotation to Newton's apple, balancing academic rigor with readability.
- Value: ⭐⭐⭐⭐⭐ Makes important contributions to the theoretical foundations of curiosity-driven RL and to the understanding of LLM behavior.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] MulDimIF: A Multi-Dimensional Constraint Framework for Evaluating and Improving Instruction Following in Large Language Models](muldimif_a_multi-dimensional_constraint_framework_for_evaluating_and_improving_i.md)
- [\[ACL 2026\] Foresight Optimization for Strategic Reasoning in Large Language Models](foresight_optimization_for_strategic_reasoning_in_large_language_models.md)
- [\[ACL 2026\] Adam's Law: Textual Frequency Law on Large Language Models](adam39s_law_textual_frequency_law_on_large_language_models.md)
- [\[ACL 2026\] From Static Inference to Dynamic Interaction: A Survey of Streaming Large Language Models](from_static_inference_to_dynamic_interaction_a_survey_of_streaming_large_languag.md)
- [\[ACL 2026\] The Model Agreed, But Didn't Learn: Diagnosing Surface Compliance in Large Language Models](the_model_agreed_but_didn39t_learn_diagnosing_surface_compliance_in_large_langua.md)

</div>

<!-- RELATED:END -->

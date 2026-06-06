---
title: >-
  [Paper Note] The Imperfective Paradox in Large Language Models
description: >-
  [ACL2026][NLP Understanding][Imperfective Paradox] This paper evaluates whether LLMs understand that "doing something" does not necessarily entail "having completed something" using the newly constructed ImperfectiveNLI…
tags:
  - "ACL2026"
  - "NLP Understanding"
  - "Imperfective Paradox"
  - "Event Semantics"
  - "Teleological Bias"
  - "Natural Language Inference"
  - "ImperfectiveNLI"
date: 2026-05-08
content_hash: a8d65bae7fbd6428
---

# The Imperfective Paradox in Large Language Models

**Conference**: ACL2026  
**arXiv**: [2601.09373](https://arxiv.org/abs/2601.09373)  
**Code**: https://github.com/boleima/ImperfectiveParadox  
**Area**: Semantic Reasoning / LLM Evaluation  
**Keywords**: Imperfective Paradox, Event Semantics, Teleological Bias, Natural Language Inference, ImperfectiveNLI  

## TL;DR
This paper evaluates whether LLMs understand that "doing something" does not necessarily entail "having completed something" using the newly constructed ImperfectiveNLI diagnostic set. It reveals that open-source LLMs generally misjudge telic events as completed. Prompt engineering tends to fluctuate between reducing completion hallucinations and losing legitimate atelic entailments; the fundamental issue lies in the reasoning phase being dominated by teleological priors.

## Background & Motivation
**Background**: Large language models have demonstrated strong performance in NLI, QA, and complex reasoning, yet high scores do not equate to a mastery of formal semantics. Event semantics are particularly subtle, as a sentence may describe a process or assert a result. For humans, "The boy was running" entails "The boy ran," but "The carpenter was building a gazebo" does not entail "The carpenter built a gazebo."

**Limitations of Prior Work**: Much NLP research evaluates whether models can classify verb aspect, tense, or event type. However, classifying a verb as telic or atelic does not guarantee that the model can apply this knowledge for reasoning. LLMs might recognize that "building is a bounded activity" yet still use commonsense narrative completion to infer "the house was built" during NLI tasks.

**Key Challenge**: Formal semantics require models to distinguish between a process and its completion. In contrast, narrative biases in pre-training corpora often default to the success of goal-oriented actions. When a goal-directed event is mentioned, the language model tends to predict its typical outcome; strict logical reasoning, however, must remain Unknown in the absence of evidence regarding the outcome.

**Goal**: The authors aim to systematically measure LLM performance on the "imperfective paradox": whether models exhibit completion hallucinations for telic progressives; whether explicit rules, CoT, or counterfactual prompts can correct this bias; whether larger models naturally improve; and whether different telic verb categories trigger bias equally.

**Key Insight**: The paper converts the linguistic "imperfective paradox" into a controlled NLI diagnostic task. Through a 2×2 combination of telic/atelic and interrupted/ambiguous conditions, it simultaneously probes whether models acknowledge interruption facts, preserve legitimate atelic entailments, and avoid incorrect completions in ambiguous accomplishments.

**Core Idea**: By using a minimal pair NLI dataset to decouple "process description" from "result realization," the study demonstrates that current LLMs function more as narrative outcome predictors than as logical reasoners adhering to event semantic boundaries.

## Method
The core contribution is the ImperfectiveNLI dataset and a corresponding set of diagnostic metrics. Following Vendler’s aspectual classification, the authors selected 100 telic accomplishment verbs and 100 atelic activity verbs to construct premise-hypothesis pairs. Models are required to output True, False, or Unknown.

### Overall Architecture
ImperfectiveNLI employs a 2×2 design. The first dimension is whether the verb is telic: accomplishments have inherent endpoints (e.g., build, write), while activities do not (e.g., run, swim). The second dimension is whether the context explicitly interrupts the event: "interrupted" conditions provide cancellation information, while "ambiguous" conditions only state the action is ongoing. The four groups are: A: interrupted accomplishment (False); B: interrupted activity (True); C: ambiguous accomplishment (Unknown); D: ambiguous activity (True).

Evaluated models include Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3, Qwen2.5-7B-Instruct, and others. The authors also separately evaluated Qwen2.5 across scales from 1.5B to 72B. All generation used greedy decoding.

### Key Designs
1.  **Four Diagnostic Groups (Minimal Pairs)**:
    - **Function**: Decouples event telicity and contextual interruption to locate specific model failures.
    - **Mechanism**: Group C serves as the critical probe. The correct label for "The carpenter was building a gazebo" to "The carpenter built a gazebo" is Unknown. Group D prevents models from judging all progressives as Unknown, as "was running" entails "ran."
    - **Design Motivation**: If only ambiguous accomplishments were tested, models could score high by conservatively answering Unknown. The 2×2 combination forces models to process cancellation, process, result, and aspect simultaneously.

2.  **Teleological Bias and Aspectual Awareness Metrics**:
    - **Function**: Separately measures "completion hallucination" and the ability to distinguish telic/atelic types.
    - **Mechanism**: Teleological Bias Rate (TBR) is the proportion of True predictions in Group C: $TBR_C=\sum_{i\in C}\mathbb{I}(\hat{y}_i=True)/|C|$. Aspectual Awareness Gap is defined as $\Delta_{AA}=ACC_D-TBR_C$, rewarding models that are accurate in Group D while maintaining low hallucination in Group C.
    - **Design Motivation**: Accuracy in Group C alone is insufficient, as overly conservative models might label all Group D cases as Unknown. $\Delta_{AA}$ is a stricter metric for aspectual reasoning.

3.  **Prompt Intervention & Representation/Behavior Analysis**:
    - **Function**: Determines if errors are correctable via prompting and whether bias originates in the representation or reasoning layer.
    - **Mechanism**: The authors compared Zero-shot, Definition-Aware Prompt (DAP), CoT, and Counterfactual prompts. Representation analysis compared contextual embedding cosine similarity between progressive and perfective phrases.
    - **Design Motivation**: If explicit rules solve the problem, it is a knowledge gap; if representations cannot distinguish process from result, it is an encoding issue. The findings suggest models encode the difference but fail during decision-making due to completion priors.

### Loss & Training
This study is based on evaluation and prompt intervention rather than model training. Data construction used Gemini for rewriting followed by strict human auditing (mean quality score 4.80, 96.3% agreement). Model evaluation employed deterministic greedy decoding to avoid sampling noise.

## Key Experimental Results

### Main Results
In the zero-shot setting, most models treated nearly all progressives as simple past completions. While Llama-3.1 achieved 0.98 in Group D, it scored only 0.02 in Group C (TBR 0.98, $\Delta_{AA}$ 0.00). This indicates that the model relies on a shallow "was V-ing implies V-ed" heuristic rather than understanding atelic properties.

| Model | Acc A | Acc B | Acc C | Acc D | TBR_C | ΔAA | Interpretation |
|------|-------|-------|-------|-------|-------|-----|------|
| Llama-3.1-8B | 0.47 | 0.85 | 0.02 | 0.98 | 0.98 | 0.00 | Almost always judges telic as completed |
| Mistral-7B | 0.37 | 0.92 | 0.02 | 1.00 | 0.97 | 0.03 | Strong completion bias similar to Llama |
| Qwen2.5-7B | 0.20 | 0.98 | 0.47 | 0.97 | 0.53 | 0.44 | Relatively best; can partially suspend judgment |
| Yi-1.5-9B | 0.35 | 0.94 | 0.02 | 1.00 | 0.98 | 0.02 | Near-maximum teleological bias |
| DeepSeek-7B | 0.04 | 0.88 | 0.00 | 1.00 | 1.00 | 0.00 | Complete hallucination of completion |
| Gemma-2-9B | 0.03 | 0.96 | 0.06 | 1.00 | 0.94 | 0.06 | Unable to handle accomplishments |
| GLM-4-9B | 0.14 | 0.98 | 0.03 | 1.00 | 0.97 | 0.03 | High atelic accuracy masks surface heuristics |

Prompt interventions show a clear trade-off. DAP provides some rules, while CoT reduces TBR but causes atelic entailments in Group D to drop. Counterfactual prompts are most effective for Group C but drive models toward extreme uncertainty for all progressives.

| Prompt | Example Model | Group C Gain | Group D Cost | TBR_C | Conclusion |
|--------|----------|--------------|--------------|-------|------|
| Zero-shot | Llama-3.1 | 0.02 | 0.98 | 0.98 | Naive teleology, defaults to completion |
| DAP | Llama-3.1 | 0.36 | 0.99 | 0.45 | Explicit rules help but are not exhaustive |
| CoT | Llama-3.1 | 0.67 | 0.65 | 0.33 | Reduces hallucination while over-doubting atelic |
| Counterfactual | Llama-3.1 | 0.97 | 0.00 | 0.00 | Corrects telic but causes calibration collapse |

### Ablation Study
Scale analysis reveals a non-linear improvement in the Qwen2.5 family. The 1.5B model has a TBR of 1.00; however, the 32B model exhibits a significant jump with Group C accuracy reaching 0.91 and $\Delta_{AA}$ reaching 0.83.

| Qwen2.5 Scale | Acc A | Acc B | Acc C | Acc D | TBR_C | ΔAA |
|--------------|-------|-------|-------|-------|-------|-----|
| 1.5B | 0.21 | 0.96 | 0.00 | 1.00 | 1.00 | 0.00 |
| 7B | 0.20 | 0.98 | 0.47 | 0.97 | 0.53 | 0.44 |
| 14B | 0.24 | 0.86 | 0.39 | 0.98 | 0.61 | 0.37 |
| 32B | 0.53 | 0.90 | 0.91 | 0.92 | 0.09 | 0.83 |
| 72B | 0.43 | 0.88 | 0.84 | 0.97 | 0.16 | 0.81 |

Semantic category analysis shows that "Motion to Goal" events (e.g., walking to a park) are handled much better than "Creation" (e.g., building a house) or "Change of State" events. Creation verbs strongly activate priors regarding the existence of the result, making them more prone to completion hallucinations.

Representation analysis shows that while "Motion to Goal" has the highest progressive/perfective embedding similarity (~0.88), it performs most accurately. Creation has lower similarity (~0.85) but higher hallucination. This suggests that errors are not due to a total lack of distinction in the representation layer, but rather because the reasoning phase is overridden by narrative completion priors.

### Key Findings
- High scores in atelic groups (Group D) are deceptive; models often map past progressives to simple past completions indiscriminately.
- Teleological priors often override explicit cancellations (Group A).
- Prompt engineering leads to a calibration crisis: stronger reminders about incomplete progressives cause models to incorrectly reject legitimate atelic entailments.
- Model scale helps non-linearly, with significant improvements occurring at specific capacity thresholds (e.g., 32B).
- Bias is primarily a reasoning-time failure; internal representations differentiate aspects, but final decisions favor narrative completion.

## Highlights & Insights
- The study transforms a classic linguistic problem into a clean diagnostic task. The four-group design effectively excludes speculative strategies like "always True" or "always Unknown."
- The combination of TBR and $\Delta_{AA}$ is elegant. It targets completion hallucinations while preventing models from masking ignorance with excessive skepticism.
- The concept of LLMs as "predictive narrative engines" is insightful. LLMs do not just make random errors; they complete goal-oriented events according to common patterns found in training data.
- The separation of representation and behavior demonstrates that models possess the underlying knowledge, but fail to weight it correctly during decoding.

## Limitations & Future Work
- The dataset is template-based, providing high internal validity but limited syntactic and discourse diversity compared to natural text.
- The scope is currently limited to English. Different aspectual marking systems in other languages could result in different model behaviors.
- The theoretical gold labels might differ from human probabilistic intuition; subsequent research could incorporate human judgment distributions.
- Interventions were limited to prompting; the effectiveness of PEFT, RL, or activation steering remains to be tested.

## Related Work & Insights
- **vs Traditional aspect classification**: Unlike previous work that merely classifies aspect, this study tests whether models can utilize that knowledge for entailment.
- **vs NLI heuristic diagnostics**: This work extends the study of surface heuristics (e.g., word overlap) into the domain of event semantics where "was V-ing → V-ed" acts as a shallow rule.
- **vs Prompt-based reasoning**: It demonstrates that CoT can lead models into a different type of error—excessive doubt—in semantic calibration tasks.
- **vs Hallucination research**: It places hallucinations at a granular semantic level, where typical results are mistaken for logical entailments.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Systematically evaluates LLM event semantics via the imperfective paradox with high precision.
- **Experimental Thoroughness**: ⭐⭐⭐⭐☆ Covers interventions, scale, and categories, though currently limited to English.
- **Writing Quality**: ⭐⭐⭐⭐☆ Clear narrative with thorough linguistic background.
- **Value**: ⭐⭐⭐⭐⭐ Highly insightful for NLI and semantic evaluation; a model for fine-grained formal semantic benchmarks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] AdapTime: Enabling Adaptive Temporal Reasoning in Large Language Models](adaptime_enabling_adaptive_temporal_reasoning_in_large_language_models.md)
- [\[ACL 2026\] Table Question Answering in the Era of Large Language Models: A Comprehensive Survey](table_question_answering_in_the_era_of_large_language_models_a_comprehensive_sur.md)
- [\[ACL 2026\] Lost in the Prompt Order: Revealing the Limitations of Causal Attention in Language Models](lost_in_the_prompt_order_revealing_the_limitations_of_causal_attention_in_langua.md)
- [\[AAAI 2026\] Language Models and Logic Programs for Trustworthy Tax Reasoning](../../AAAI2026/nlp_understanding/language_models_and_logic_programs_for_trustworthy_tax_reasoning.md)
- [\[ICML 2026\] Controlling the Risk of Corrupted Contexts for Language Models via Early-Exiting](../../ICML2026/nlp_understanding/controlling_the_risk_of_corrupted_contexts_for_language_models_via_early-exiting.md)

</div>

<!-- RELATED:END -->

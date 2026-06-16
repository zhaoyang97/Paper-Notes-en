---
title: >-
  [Paper Note] The Imperfective Paradox in Large Language Models
description: >-
  [ACL 2026][NLP Understanding][ImperfectiveNLI] This paper evaluates whether LLMs understand that "doing something" does not necessarily imply "having finished something" using the newly constructed ImperfectiveNLI diagnostic set. It finds that open-source LLMs generally misjudge telic events as completed; prompt engineering merely oscillates between reducing comple
tags:
  - ACL 2026
  - NLP Understanding
  - ImperfectiveNLI
date: 2026-05-08
content_hash: a1a9ba47c630f965
---
# The Imperfective Paradox in Large Language Models

**Conference**: ACL2026  
**arXiv**: [2601.09373](https://arxiv.org/abs/2601.09373)  
**Code**: https://github.com/boleima/ImperfectiveParadox  
**Area**: Semantic Reasoning / LLM Evaluation  
**Keywords**: Imperfective paradox, event semantics, teleological bias, Natural Language Inference, ImperfectiveNLI  

## TL;DR
This paper evaluates whether LLMs understand that "doing something" does not necessarily imply "having finished something" using the newly constructed ImperfectiveNLI diagnostic set. It finds that open-source LLMs generally misjudge telic events as completed; prompt engineering merely oscillates between reducing completion hallucinations and preserving legitimate entailments, suggesting the core issue is the dominance of teleological priors during the reasoning phase.

## Background & Motivation
**Background**: Large Language Models have shown strong performance in NLI, QA, and complex reasoning, but high scores do not equate to mastering formal semantics. Event semantics are particularly subtle because a sentence may describe a process or assert a result. For humans, "The boy was running" entails "The boy ran," but "The carpenter was building a gazebo" does not entail "The carpenter built a gazebo."

**Limitations of Prior Work**: Many NLP studies evaluate whether models can classify verb aspect, tense, or event types, but classifying a verb as telic or atelic does not mean the model uses this knowledge for reasoning. An LLM might know "building is a telic activity" yet still use commonsense narrative completion to conclude "the house was built" during NLI.

**Key Challenge**: Formal semantics require models to distinguish between process and completion, whereas narrative bias in pre-training corpora often defaults to goal-oriented actions succeeding. When a goal-directed event is mentioned, language models tend to predict its typical resolution; however, strict logical reasoning must remain Unknown in the absence of evidence for the result.

**Goal**: The authors aim to systematically measure LLM performance on the imperfective paradox: whether they develop completion hallucinations for telic progressives; whether explicit rules, CoT, and counterfactual prompts fix the bias; whether larger models improve naturally; whether different telic verb categories trigger bias equally; and whether bias stems from representation confusion or the reasoning/decoding stage.

**Key Insight**: The paper transforms the linguistic imperfective paradox into a controlled NLI diagnostic task. Through a 2×2 combination of telic/atelic and interrupted/ambiguous conditions, it simultaneously checks for recognition of interruptions, preservation of atelic entailments, and erroneous completion of ambiguous accomplishments.

**Core Idea**: By using minimal pair NLI datasets to decouple "process described" from "result realized," the paper demonstrates that current LLMs function more like narrative outcome predictors than logical reasoners that respect event semantic boundaries.

## Method

### Overall Architecture
The core contribution is the ImperfectiveNLI diagnostic set and its two associated metrics. Starting from Vendler's aspectual categories, the authors selected 100 telic accomplishment verbs (e.g., build, write, fix) and 100 atelic activity verbs (e.g., run, swim, wander). Premise-hypothesis pairs were constructed for each verb, requiring models to output True / False / Unknown (corresponding to Entailment / Contradiction / Neutral). The design follows a 2×2 grid: telic/atelic verb $\times$ explicit interruption/ambiguous context, forming four groups: A (interrupted accomplishment, gold: False), B (interrupted activity, gold: True), C (ambiguous accomplishment, gold: Unknown), and D (ambiguous activity, gold: True). Teleological bias rate and aspectual awareness gap are defined, and prompt intervention with representation analysis is used to locate whether errors occur in encoding or decoding.

### Key Designs

**1. Four groups of minimal pair diagnostic data: Decoupling verb aspect and interruption info**

If only ambiguous accomplishments were tested, a model could achieve a high score by conservatively answering Unknown for everything without truly understanding aspect. The four-group combination forces models to handle cancellation, process, and result simultaneously. Group C is the key probe: "The carpenter was building a gazebo" $\rightarrow$ "The carpenter built a gazebo" (Correct: Unknown). Group D serves as a control: any sub-interval of an atelic activity constitutes the event itself, so "was running" does entail "ran." Models must distinguish between "suspending telic completion" and "accepting atelic entailment."

**2. Teleological Bias Rate and Aspectual Awareness Gap: Separating completion hallucinations from true discriminative ability**

Accuracy on Group C alone is insufficient, as an overly conservative model might also label Group D as Unknown. The authors decouple these via two metrics: Teleological Bias Rate ($TBR_C$) counts the proportion of True predictions in Group C, $TBR_C=\sum_{i\in C}\mathbb{I}(\hat{y}_i=True)/|C|$, specifically targeting completion hallucinations. The Aspectual Awareness Gap is defined as $\Delta_{AA}=ACC_D-TBR_C$, combining "suppressing completion hallucinations" and "preserving legitimate entailments" into a single score. Only a model with high Group D accuracy and low Group C TBR achieves a high $\Delta_{AA}$.

**3. Prompt intervention and representation/behavior separation: Locating where errors occur**

To determine if errors stem from missing knowledge, representation confusion, or reasoning failures, the authors compared four prompt types: zero-shot strict logician, Definition-Aware Prompt (DAP), CoT, and Counterfactual (asking the model to imagine three unfinished scenarios before judging). In the representation space, they calculated the cosine similarity between progressive and perfective phrase contextual embeddings and correlated this with $TBR_C$ across verb classes.

### Loss & Training
No training was performed; the study focused on evaluation and prompt intervention. Data was augmented via Gemini and strictly human-verified: three native speakers scored Grammar, Fluency, and Adequacy (avg. score $4.80$, agreement $96.3\%$). All evaluations used deterministic greedy decoding (max 512 tokens). Evaluated models include Llama-3.1-8B, Mistral-7B-v0.3, Qwen2.5-7B, DeepSeek-7B-Chat, Gemma-2-9B, GLM-4-9B, Yi-1.5-9B, and a Qwen2.5 scale analysis (1.5B/7B/14B/32B/72B).

## Key Experimental Results

### Main Results
In the zero-shot setting, most models treated almost all progressives as completed simple past facts. Llama-3.1 achieved 0.98 on Group D, but its Group C accuracy was only 0.02 with a $TBR_C$ of 0.98 and $\Delta_{AA}$ of 0.00. This suggesting it relies on a shallow "was V-ing implies V-ed" heuristic rather than understanding sub-interval properties.

| Model | Acc A | Acc B | Acc C | Acc D | $TBR_C$ | $\Delta_{AA}$ | Interpretation |
|------|-------|-------|-------|-------|-------|-----|------|
| Llama-3.1-8B | 0.47 | 0.85 | 0.02 | 0.98 | 0.98 | 0.00 | Almost always judges telic as finished |
| Mistral-7B | 0.37 | 0.92 | 0.02 | 1.00 | 0.97 | 0.03 | Strong completion bias |
| Qwen2.5-7B | 0.20 | 0.98 | 0.47 | 0.97 | 0.53 | 0.44 | Relatively best; partially suspends judgment |
| Yi-1.5-9B | 0.35 | 0.94 | 0.02 | 1.00 | 0.98 | 0.02 | Nearly maximum teleological bias |
| DeepSeek-7B | 0.04 | 0.88 | 0.00 | 1.00 | 1.00 | 0.00 | Complete hallucination of telic results |
| Gemma-2-9B | 0.03 | 0.96 | 0.06 | 1.00 | 0.94 | 0.06 | Fails to handle accomplishments |
| GLM-4-9B | 0.14 | 0.98 | 0.03 | 1.00 | 0.97 | 0.03 | High atelic accuracy masks heuristics |

Prompt intervention reveals a clear trade-off. DAP improves Llama’s Group C from 0.02 to 0.36; CoT reduces TBR but harms atelic entailment in Group D; Counterfactual is most effective for Group C but pushes models toward an "everything is uncertain" extreme.

| Prompt | Model Example | Group C Imp. | Group D Cost | $TBR_C$ | Conclusion |
|--------|----------|--------------|--------------|-------|------|
| Zero-shot | Llama-3.1 | 0.02 | 0.98 | 0.98 | Naive teleology, defaults to complete |
| DAP | Llama-3.1 | 0.36 | 0.99 | 0.45 | Explicit rules help but are incomplete |
| CoT | Llama-3.1 | 0.67 | 0.65 | 0.33 | Reduces hallucination, leads to over-skepticism |
| Counterfactual | Llama-3.1 | 0.97 | 0.00 | 0.00 | Corrects telic, causes calibration collapse |

### Ablation Study
Scaling analysis reveals non-linear improvements in the Qwen2.5 family. While 1.5B has a $TBR_C$ of 1.00, a significant jump occurs near 32B, where Group C accuracy reaches 0.91 and $\Delta_{AA}$ hits 0.83.

| Qwen2.5 Scale | Acc A | Acc B | Acc C | Acc D | $TBR_C$ | $\Delta_{AA}$ |
|--------------|-------|-------|-------|-------|-------|-----|
| 1.5B | 0.21 | 0.96 | 0.00 | 1.00 | 1.00 | 0.00 |
| 7B | 0.20 | 0.98 | 0.47 | 0.97 | 0.53 | 0.44 |
| 14B | 0.24 | 0.86 | 0.39 | 0.98 | 0.61 | 0.37 |
| 32B | 0.53 | 0.90 | 0.91 | 0.92 | 0.09 | 0.83 |
| 72B | 0.43 | 0.88 | 0.84 | 0.97 | 0.16 | 0.81 |

Semantic category analysis shows that Creation verbs (e.g., build, write) trigger stronger completion hallucinations compared to Motion to Goal verbs. The authors suggest that Creation verbs strongly activate a "result existence" prior that is harder for models to suspend.

| Semantic Class | Group A Avg | Group C Avg | Phenomenon |
|----------|------------------|------------------|----------|
| Creation | Lowest (~18%) | High Hallucination | Activates result-existence priors |
| Change of State | ~21% | Strong Bias | Goal states are assumed completed |
| Motion to Goal | Highest (~46%) | Lowest $TBR_C$ | Destination/traversal is easier to distinguish |

Representation analysis shows that similarity and $TBR_C$ are negatively correlated (Pearson $r=-0.97, p=0.03$). Models can distinguish process from result in their representation space, but the reasoning phase is overridden by narrative priors.

### Key Findings
- High scores on Group D are deceptive; models often map past progressives to simple past results as a crude heuristic.
- Explicit cancellation is often ignored due to teleological priors: even if a context says a frame was destroyed, models often assume the gazebo was "built."
- Prompt engineering causes a calibration crisis; reminding models that the progressive doesn't imply completion often makes them reject legitimate atelic entailments.
- Scaling helps non-linearly; Qwen2.5 shows a jump at 32B, suggesting aspectual reasoning requires sufficient capacity to suppress shallow heuristics.
- The failure primarily occurs at reasoning-time; representations distinguish process and result, but decoding favors narrative completion.

## Highlights & Insights
- The paper adapts a classic linguistic problem into a clean LLM diagnostic task, using a four-group design that excludes simple guessing strategies.
- The combination of TBR and $\Delta_{AA}$ is effective for isolating completion hallucinations while preventing models from masking ignorance with over-skepticism.
- The concept of LLMs as "predictive narrative engines" is insightful; hallucinations are not just random errors but are completions of goal events following common narrative patterns.
- Representation/behavior separation shows that the "knowledge" exists internally, but isn't weighted correctly in final decision-making.

## Limitations & Future Work
- The dataset is template-based, providing high internal validity but limited syntactic and discourse diversity.
- The study is limited to English; different aspectual marking systems in other languages (e.g., Chinese markers, Slavic morphology) may lead to different behaviors.
- Theoretical gold labels might differ from human intuition, which can be probabilistic based on world knowledge.
- Interventions were limited to prompting; the efficacy of PEFT, RL, or activation steering was not tested.

## Related Work & Insights
- **vs. Traditional Aspect Classification**: Unlike prior work that merely labels aspects, this study tests whether models use aspectual knowledge for entailment.
- **vs. NLI Heuristic Diagnostics**: Similar to work showing models use word-overlap heuristics, this highlights a "progressive-to-perfective" heuristic for event semantics.
- **vs. Hallucination Research**: Frames hallucination as a fine-grained semantic issue where models mistake typical narrative outcomes for logical entailments.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐☆ 
- Writing Quality: ⭐⭐⭐⭐☆ 
- Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->
<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] AdapTime: Enabling Adaptive Temporal Reasoning in Large Language Models](adaptime_enabling_adaptive_temporal_reasoning_in_large_language_models.md)
- [\[ACL 2026\] Table Question Answering in the Era of Large Language Models: A Comprehensive Survey](table_question_answering_in_the_era_of_large_language_models_a_comprehensive_sur.md)
- [\[ACL 2025\] BQA: Body Language Question Answering Dataset for Video Large Language Models](../../ACL2025/nlp_understanding/bqa_body_language_question_answering_dataset_for_video_large_language_models.md)
- [\[ACL 2025\] Generating Diverse Training Samples for Relation Extraction with Large Language Models](../../ACL2025/nlp_understanding/generating_diverse_training_samples_for_relation_extraction_with_large_language_.md)
- [\[ACL 2026\] Lost in the Prompt Order: Revealing the Limitations of Causal Attention in Language Models](lost_in_the_prompt_order_revealing_the_limitations_of_causal_attention_in_langua.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] How LLMs Comprehend Temporal Meaning in Narratives: A Case Study in Cognitive Evaluation of LLMs
description: >-
  [ACL 2025][LLM (Other)][temporal comprehension] By constructing an Expert-in-the-Loop probing pipeline with three sets of cognitive linguistics experiments (truth-value judgment, word completion, and open-ended causal questioning) across 16 narratives, 30 prompt variants, and 7 LLMs, this study systematically evaluates LLMs' understanding of grammatical aspect (perfective vs. imperfective) in narratives. The results show that LLMs achieve only 18% accuracy under non-prototypi…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "temporal comprehension"
  - "grammatical aspect"
  - "cognitive evaluation"
  - "narrative understanding"
  - "causal reasoning"
date: 2026-05-08
content_hash: 874b343534978035
---

# How LLMs Comprehend Temporal Meaning in Narratives: A Case Study in Cognitive Evaluation of LLMs

**Conference**: ACL 2025  
**arXiv**: [2507.14307](https://arxiv.org/abs/2507.14307)  
**Code**: None  
**Area**: LLM/NLP  
**Keywords**: temporal comprehension, grammatical aspect, cognitive evaluation, narrative understanding, causal reasoning

## TL;DR

By constructing an Expert-in-the-Loop probing pipeline with three sets of cognitive linguistics experiments (truth-value judgment, word completion, and open-ended causal questioning) across 16 narratives, 30 prompt variants, and 7 LLMs, this study systematically evaluates LLMs' understanding of grammatical aspect (perfective vs. imperfective) in narratives. The results show that LLMs achieve only 18% accuracy under non-prototypical aspect conditions (compared to 71% for humans) and lack long-distance causal reasoning capabilities.

## Background & Motivation

- **Background**: While LLMs exhibit increasingly powerful linguistic capabilities, whether these behaviors reflect human-like cognitive understanding or advanced statistical pattern matching remains an open question. A growing body of research has emerged in recent years to evaluate LLMs using experimental paradigms from cognitive science (such as the Fan Effect, decision-making biases, Theory of Mind, etc.). However, cognitive evaluation of temporal aspect semantics remains a blank.
- **Limitations of Prior Work**: Existing LLM evaluations rely heavily on NLP benchmarks, lacking fine-grained comparisons with human cognitive processes. The few studies utilizing human experiments often employ a single metric and fail to sufficiently control for pseudo-discoveries caused by prompt perturbation.
- **Key Challenge**: LLMs can fluently explain the academic definition of grammatical aspect (declarative knowledge), yet fail severely when required to implicitly apply aspect semantics for verification. This dissociation—knowing the rules but being unable to use them—highly resembles the performance of intermediate-stage second language (L2) learners.
- **Goal**: Can LLMs, like humans, leverage the semantic differences of grammatical aspect (perfective vs. imperfective) to construct narrative situation models, maintain focal information in working memory, and perform long-distance causal reasoning?
- **Key Insight**: Choose grammatical "aspect" in linguistics as a probe—where the perfective aspect ("washed") indicates that an event has ended, and the imperfective aspect ("was washing") indicates that an event is ongoing. In narratives, the imperfective aspect keeps an event "open," making it more likely to be retained in working memory and link causally with subsequent events. This mechanism is extensively empirically validated in human cognition, providing an ideal comparative baseline for LLM evaluation.
- **Core Idea**: Reuse validated narrative aspect experimental materials from cognitive linguistics to build a robust testing pipeline with 30 prompt variants, step-by-step evaluating the gap between LLMs and humans in aspect comprehension across three levels: semantics, cognitive processing, and pragmatics.

## Method

### Overall Architecture

Expert-in-the-Loop Probing Pipeline: Centered around an iterative evaluation framework involving cognitive linguists, human experimental materials (16 narratives from Schramm 1998) are converted into LLM-usable prompts. After systematic prompt perturbation (30 variants), three progressive experiments (truth-value judgment, word completion, and open-ended causal reasoning) are executed on 7 LLMs. Finally, linear mixed-effects models (lme4) are used for statistical analysis to directly compare results with published human data.

### Key Designs

**1. Narrative Stimuli and Aspect Manipulation**

- **Function**: Provide controlled experimental stimuli to manipulate the direction of causal reasoning throughout the narrative solely by altering the grammatical aspect of a single verb.
- **Mechanism**: Each narrative contains two potential cause events ($C_1$, $C_2$) and a surprising effect (Effect). $C_1$ uses an accomplishment event (e.g., "wash the dishes"), and its verb alternates between the perfective ("washed") and imperfective ("was washing") aspects. When $C_1$ is imperfective, the event remains open, and humans show a stronger tendency to infer $C_1$ as the cause of the Effect. When $C_1$ is perfective, the event is viewed as completed, making $C_2$ the more likely cause. There are a total of 16 narratives $\times$ 2 versions = 32 stimuli.
- **Design Motivation**: Directly reuse linguistics experimental materials validated on human participants to ensure strict comparability with human data. Simultaneously, utilize the non-prototypical pairing of Accomplishment + imperfective to test whether LLMs truly understand aspect semantics rather than merely relying on high-frequency co-occurrence patterns.

**2. Prompt Perturbation and Robustness Control**

- **Function**: Eliminate the systematic influence of prompt phrasing on LLM responses, ensuring the reliability of experimental conclusions.
- **Mechanism**: Systematic perturbation is applied to each prompt across two axes: (a) general instruction paraphrasing, generating 3 instruction versions based on syntactic and semantic variation strategies; (b) data format variations, introducing 10 format variants based on the FormatSpread protocol (varying spaces, casing, ordering, punctuation). Every experimental condition yields $3 \times 10 = 30$ prompt variants, and model results are aggregated across all variants.
- **Design Motivation**: Sclar et al. (2024) and Wahle et al. (2024) have demonstrated that prompt format and phrasing significantly influence LLM responses, meaning conclusions from a single prompt could be artifacts. The 30 variants provide sufficient statistical coverage.

**3. Three-Tier Progressive Probing Framework**

- **Function**: Comprehensively evaluate LLMs' aspect processing capabilities across three levels: semantic understanding, cognitive processing, and pragmatic reasoning.
- **Mechanism**: (a) **Experiment 1 - Truth-Value Judgment**: After presenting the narrative, LLMs are asked to judge the truth/falsity of the event's final status (e.g., "Lena was running downstairs" $\rightarrow$ "Lena downstairs" is True/False) to test whether they understand semantically that the imperfective does not entail completion. (b) **Experiment 2 - Word Completion**: Partial letters of target words are inserted into the narrative (e.g., "D I _ _ _ _"), and LLMs are asked to complete them. Measuring the activation of $C_1$ key terms through completion performance allows assessing memory retention at two probe locations: Near $C_1$ and Near Effect, evaluating long-distance maintenance capabilities. (c) **Experiment 3 - Open-Ended Causal Questioning**: After reading the narrative, LLMs answer "What caused the effect?", measuring the frequency of choosing $C_1$ as the cause to directly benchmark against human causal reasoning data.
- **Design Motivation**: A single metric cannot fully reflect cognitive processes. The three experiments correspond to semantic representation, working memory encoding, and causal models in episodic memory in human cognition, providing converging evidence.

## Key Experimental Results

### Main Results

**Experiment 1a — Truth-Value Judgment Accuracy (LLM vs. Human)**:

| Aspect | Polarity | Correct Answer | LLM Accuracy | Human Accuracy |
|------|------|---------|----------|----------|
| Perfective | Positive | True | 88% | 88% |
| Imperfective | Negative | True | **18%** | **71%** |
| Perfective | Negative | False | 89% | 93% |
| Imperfective | Positive | False | **35%** | **61%** |

Statistical test: Main effect of aspect is significant ($F=66.5, p < .01$); main effect of polarity is significant ($F=10363, p < .01$); interaction effect is significant ($F=661.5, p < .01$).

**Experiment 3 — Open-Ended Causal Reasoning (Rate of selecting C1 as the cause)**:

| Model | Imperfective Condition | Perfective Condition | Gap with Human |
|------|------------|-----------|----------|
| Human | ~68% | ~33% | — |
| Qwen2-72B | ~60% | ~10% | Small |
| Llama3.1-70B | ~55% | ~8% | Medium |
| GPT-4o | ~50% | ~12% | Medium |
| Small Models (<10B average) | ~35% | ~8% | Large |

Main effect of aspect is significant ($F=98.5, p < .01$).

### Ablation Study

| Comparative Condition | Key Metric | Conclusion |
|---------|---------|------|
| Experiment 1b: Narrative Context vs. No Context | Imperfective accuracy remains low without narratives | Difficulties stem from aspect processing itself rather than narrative structure distraction; change directions for individual models are inconsistent |
| Experiment 2: Near C1 vs. Near Effect | Match rate for target words drops by 33% on average | LLMs lack long-distance working memory retention and fail to keep imperfective events in focus like humans |
| Large Models (70B+) vs. Small Models (<10B) | Causal reasoning closer to humans | Scaling effect observed only in Experiment 3; no significant differences between large and small models in Experiments 1 and 2 |
| Model Families Comparison (Gemma/Llama/Qwen/GPT-4o) | Significant differences among families with inconsistent directions | No single model maintains human-level performance across all experiments |

### Key Findings

1. **Normal Performance on Prototypical Conditions, Severe Failure on Non-Prototypical Conditions**: Accuracy on perfective aspect (prototypical pairing) is on par with humans at 88-89%, but imperfective aspect (non-prototypical pairing) accuracy is only 18%, far below human accuracy of 71%.
2. **Over-reliance on Prototypicality**: Perfective + Accomplishment events are the most common pairing in narratives. LLMs tend to default to assuming events are completed, showing distributed representation reliance rather than meaning-driven understanding.
3. **Lack of Long-Distance Causal Reasoning**: The activation drop of word completion from Near $C_1$ to Near Effect is 33%, indicating that LLMs fail to maintain the focus of imperfective events in working memory.
4. **Dissociation of Declarative Knowledge vs. Implicit Comprehension**: LLMs can accurately state the definition of grammatical aspect but fail to apply it during actual judgments—highly reminiscent of L2 learners who can memorize grammatical rules but fail to use them.
5. **More Flexible Narrative Interpretation in Humans**: Under perfective conditions, approximately 1/3 of humans still select $C_1$ as the cause, whereas LLMs rarely do. LLMs lack the flexible modeling capacity that humans employ to seek narrative coherence.

## Highlights & Insights

- **Exquisite Research Design**: Directly reuses validated experimental materials from cognitive linguistics (Schramm 1998), ensuring strict comparability with human data instead of engineering another arbitrary benchmark.
- **Outstanding Methodological Contribution**: The Expert-in-the-Loop pipeline combined with robustness controls across 30 prompt variants serves as a paradigmatic approach for LLM cognitive evaluation, highly transferable to other cognitive domains.
- **Converging Evidence from a Three-Tier Design**: Spans semantics (truth-value judgment), cognitive processing (word completion/working memory), and pragmatics (causal reasoning), systematically locating where LLMs fail in aspect processing.
- **Deep Insight on "Declarative vs. Implicit"**: LLM behavior is highly consistent with the intermediate L2 learner stage in second language acquisition (Salaberry 2024), suggesting that LLM aspect representation is fundamentally distributed rather than conceptual.
- **Rigorous Statistical Analysis**: Utilizes linear mixed-effects models to control for narrative random effects, applying Bonferroni correction for multiple comparisons with $\alpha = .01$.

## Limitations & Future Work

1. Relies on LLM "self-reported" responses, making it difficult to directly inspect internal processing mechanisms; future work could combine attention analysis or probing classifiers to capture implicit signals.
2. The sample size of 16 narratives is small, which may result in insufficient statistical power for certain condition combinations; expanding the stimuli pool will enhance the robustness of conclusions.
3. Only aspect phenomena in English narratives were tested. The aspectual systems of other languages (e.g., Chinese, Spanish) differ significantly, leaving cross-lingual generalizability an open question.
4. Did not evaluate the latest reasoning models (e.g., o1, Claude 3.5) or performance under Chain-of-Thought prompting, which might underestimate LLM potential.
5. The open-ended responses in Experiment 3 were automatically annotated using GPT-4o (Cohen's $\kappa = .93$), which, despite high reliability, introduces a circular dependency of using an LLM to evaluate LLMs.

## Related Work & Insights

- **LLM Cognitive Evaluation Frameworks**: Consistent with the evaluation principles proposed by Ivanova (2025), emphasizing multiple metrics and human control data. While Roberts et al. (2024) use token probability as a proxy for memory retrieval difficulty in Fan Effect studies, this paper employs various behavioral indicators to study more subtle aspectual effects.
- **LLM Pragmatic Understanding**: Both Beuls & Van Eecke (2024) and Sravanthi et al. (2024) point out deficiencies in LLM pragmatic reasoning. This paper provides new evidence in the dimension of narrative aspect.
- **Cognitive Biases and Pattern Dependency**: Aligns with the LLM decision-making biases found by Hagendorff et al. (2023)—LLMs tend to rely on high-frequency patterns rather than flexible reasoning.
- **Insights**: There is a need to introduce richer temporal/causal reasoning signals during pre-training or alignment. Distinguishing between declarative knowledge and implicit understanding remains a key methodological principle in evaluating LLM cognition.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Porting cognitive linguistics aspect experiments precisely to LLM evaluation is a rare and refreshing angle in the NLP community.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three sets of experiments + 30 prompt variants + 7 models + mixed-effects statistical analysis, highly rigorous.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear introduction of the linguistic background with a smooth logical flow, highly accessible to interdisciplinary readers.
- **Value**: ⭐⭐⭐⭐ Provides solid evidence on the boundaries of LLM cognitive abilities with high methodological reproducibility, though direct recipes for technical improvement are limited.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Can LLMs Interpret and Leverage Structured Linguistic Representations? A Case Study with AMRs](can_llms_interpret_and_leverage_structured_linguistic_representations_a_case_stu.md)
- [\[ACL 2025\] Is It JUST Semantics? A Case Study of Discourse Particle Understanding in LLMs](is_it_just_semantics_a_case_study_of_discourse_particle_understanding_in_llms.md)
- [\[ACL 2025\] Assessing the Vulnerability of LLMs to Cognitive Biases in Scientific Research](assessing_the_vulnerability_of_llms_to_cognitive_biases_in_scientific_research.md)
- [\[ACL 2025\] LLMs instead of Human Judges? A Large Scale Empirical Study across 20 NLP Evaluation Tasks](llm_vs_human_judges_study.md)
- [\[ACL 2025\] SkillVerse: Assessing and Enhancing LLMs with Tree Evaluation](skillverse_tree_eval.md)

</div>

<!-- RELATED:END -->

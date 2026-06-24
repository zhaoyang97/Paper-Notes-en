---
title: >-
  [Paper Note] Not Quite Sherlock Holmes: Language Model Predictions Do Not Reliably Differentiate Impossible from Improbable Events
description: >-
  [ACL 2025 (Findings)][LLM (Other)][Event Possibility] Through meticulously designed minimal pair experiments, this paper reveals that language models cannot reliably differentiate "impossible events" from "improbable but possible events." Under adversarial conditions (where possible sentences contain unrelated words and impossible sentences contain related ones), all 35 tested models, including Llama 3, Gemma 2, and Mistral NeMo, perform below chance level.
tags:
  - "ACL 2025 (Findings)"
  - "LLM (Other)"
  - "Event Possibility"
  - "Semantic Relatedness"
  - "Language Model Predictions"
  - "Commonsense Reasoning"
  - "Minimal Pairs"
date: 2026-05-08
content_hash: 22a4231746b83dd4
---

# Not Quite Sherlock Holmes: Language Model Predictions Do Not Reliably Differentiate Impossible from Improbable Events

**Conference**: ACL 2025 (Findings)  
**arXiv**: [2506.06808](https://arxiv.org/abs/2506.06808)  
**Code**: [https://osf.io/r6xns/](https://osf.io/r6xns/)  
**Area**: LLM / NLP Understanding  
**Keywords**: Event Possibility, Semantic Relatedness, Language Model Predictions, Commonsense Reasoning, Minimal Pairs

## TL;DR

Through meticulously designed minimal pair experiments, this paper reveals that language models cannot reliably differentiate "impossible events" from "improbable but possible events." Under adversarial conditions (where possible sentences contain unrelated words and impossible sentences contain related ones), all 35 tested models, including Llama 3, Gemma 2, and Mistral NeMo, perform below chance level.

## Background & Motivation

**Background**: Numerous studies evaluate the world knowledge of language models through commonsense reasoning benchmarks (e.g., HellaSwag, WinoGrande, PIQA), which typically require models to select the most plausible continuation from multiple options. Prior work (e.g., Kauf et al., 2023) suggests that language models can distinguish between possible and impossible events.

**Limitations of Prior Work**: Existing benchmarks conflate "incorrect" options—some represent genuinely impossible events, while others are merely atypical. For example, "incorrect" continuations in HellaSwag are often still possible, just atypical. This implies that when models perform well, it remains unclear whether they truly understand event "possibility" or merely rely on superficial cues like "typicality" and "semantic relatedness."

**Key Challenge**: Possibility, typicality, and semantic relatedness are highly entangled in natural text—typical events are usually also possible, and words describing possible/typical events tend to be semantically related to the context. Without disentangling these three factors, it is impossible to determine what the models have actually learned.

**Goal**: To systematically disentangle the effects of possibility, typicality, and semantic relatedness, addressing a key question: "Can language models still differentiate possible from impossible events when typicality and semantic relatedness are no longer useful cues?"

**Key Insight**: Leveraging the minimal pair paradigm and structured stimulus materials from psycholinguistics (from Vega-Mendoza et al., 2021 and Chow & Phillips, 2013), the authors manipulate key target words across five conditions: Possible-Typical-Related (PTR), Possible-Atypical-Related (PAR), Possible-Atypical-Unrelated (PAU), Impossible-Atypical-Related (IAR), and Impossible-Atypical-Unrelated (IAU).

**Core Idea**: By orthogonally manipulating possibility, typicality, and semantic relatedness to construct minimal pairs, the authors demonstrate that language models rely heavily on typicality and semantic relatedness heuristics during prediction, failing almost completely when these cues conflict with actual possibility.

## Method

### Overall Architecture

Rather than proposing a new model, this paper presents a meticulously designed empirical study. The experiments utilize the minimal pair paradigm: only one key word differs between each sentence pair, which determines whether the event is possible. Two sentences are input to the language model to assess whether it assigns a higher probability to the possible sentence. The analysis progresses through four experiments: Experiment 1 tests typical vs. atypical/impossible; Experiment 2 tests atypical-but-possible vs. impossible (the core experiment); Experiment 3 uses mixed-effects regression to verify statistical reliability; Experiment 4 investigates scaling effects using the Pythia model suite.

### Key Designs

1. **Five-Dimensional Conditionally-Manipulated Minimal Pair Stimuli**:

    - Function: Orthogonally separates the three factors of possibility, typicality, and semantic relatedness.
    - Mechanism: Taking the sentence "the cure for the disease was discovered by the ___" as an example, the target keywords are designed as: *doctor* (PTR: possible + typical + related), *patient* (PAR: possible + atypical + related), *guest* (PAU: possible + atypical + unrelated), *medication* (IAR: impossible + atypical + related), and *stamp* (IAU: impossible + atypical + unrelated). The English stimuli are adapted from Vega-Mendoza et al. (2021) containing 154 pairs; the Chinese stimuli are from Chow & Phillips (2013) containing 57 pairs.
    - Design Motivation: Prior research either failed to distinguish impossibility from atypicality or controlled semantic relatedness to prevent it from acting as a confound. This study needs to cover scenarios where relatedness acts as a "hindrance"—specifically, when an impossible word happens to be highly related to the context.

2. **Incrementally Escalating Adversarial Experimental Design**:

    - Function: Progressively exposes the vulnerability of the models from easy to hard levels.
    - Mechanism: Experiment 1 first tests the simplest scenario (PTR vs. IAR/IAU) to confirm that models can distinguish typical from impossible events. Experiment 2 introduces the core upgrade: atypical-but-possible vs. impossible (PAU vs. IAR), which constitutes the critical Sherlock Holmes task. Experiment 3 validates the independent contributions of semantic relatedness and typicality at the item level. Experiment 4 evaluates scaling trends using the entire Pythia suite (14M-12B) across $10 \text{ scales} \times 20 \text{ checkpoints} = 200 \text{ models}$.
    - Design Motivation: Simply reporting that "models perform poorly" is insufficient; an incrementally progressive experimental design allows readers to observe exactly how performance degradation unfolds.

3. **Cross-Linguistic and Cross-Scale Comprehensive Validation**:

    - Function: Verifies the generalizability of the findings.
    - Mechanism: The experiments are replicated across 35 English models (from families including BLOOM, Gemma, Llama, Mistral, OLMo, Qwen, SmolLM, XGLM, Yi, mGPT) and a Chinese subset. Evaluation is standardized using the LM Evaluation Harness. Additionally, the training checkpoints of the Pythia suite are used to track how capabilities evolve throughout training.
    - Design Motivation: Findings restricted to a single language or model family could be anomalous; cross-linguistic and cross-scale validation is necessary to demonstrate the robustness and universality of the conclusions.

### Loss & Training

This study is purely evaluative and does not involve model training. The evaluation method directly compares the log-likelihood assigned by the model to both sentences: a decision is deemed correct if the possible sentence receives a higher probability. All models are evaluated in their pre-trained (base) versions, without instruction tuning, to assess their raw predictive capabilities.

## Key Experimental Results

### Main Results — Experiment 2: Atypical Possible vs. Impossible

| Comparison Task | Average Accuracy of 35 Models (English) | Chinese Subset Average | Description |
|---------|----------------------|------------|------|
| PAU vs. IAU (Unrelated Possible vs. Unrelated Impossible) | ~73% | ~70% | Models perform well when both are unrelated |
| PAR vs. IAU (Related Possible vs. Unrelated Impossible) | ~76% | — | Slightly beneficial when the possible word is related |
| PAU vs. IAR (Unrelated Possible vs. Related Impossible) | **~28%** | **~35%** | **Far below the 50% chance level!** |
| PAR vs. IAR (Related Possible vs. Related Impossible) | ~68% | — | Models recover partial capability when both are related |

### Ablation Study — Experiment 3: Mixed-Effects Regression

| Predictor | Impact on Model Accuracy | $\chi^2$ Statistic | p-value |
|---------|------------------|----------|------|
| Semantic relatedness of possible word $\uparrow$ | Significantly improves accuracy | 176.17 | <0.0001 |
| Semantic relatedness of impossible word $\uparrow$ | Significantly reduces accuracy | 197.58 | <0.0001 |
| Typicality of possible word $\uparrow$ | Significantly improves accuracy | 128.76 | <0.0001 |
| Typicality of impossible word $\uparrow$ | Significantly reduces accuracy | 127.39 | <0.0001 |
| Frequency of possible word $\uparrow$ | Significantly improves accuracy | 394.32 | <0.0001 |
| Frequency of impossible word $\uparrow$ | Significantly reduces accuracy | 302.54 | <0.0001 |

### Key Findings

- **Core Finding**: In the PAU vs. IAR condition (unrelated-but-possible vs. related-but-impossible), all 35 models perform below the 50% chance level—meaning they judge impossible events as more likely in more than half of the cases. For example, they predict "the car was given a parking ticket by the brake" to be more probable than "...by the explorer".
- **Scaling Does Not Solve the Problem**: Pythia experiments show that on the most challenging PAU vs. IAR task, model accuracy does not scale with parameter sizes or training tokens; larger models even perform slightly worse than smaller ones. This is not simply a matter of "insufficient model size."
- **Semantic Relatedness is the Primary Confound**: Replacing an impossible word with a contextually related word results in a precipitous drop in accuracy. This indicates that models rely heavily on the heuristic of "whether the word is topically related to the context" during prediction, rather than possessing a genuine understanding of the physical or logical constraints of events.

## Highlights & Insights

- **Ingenious Conceptualization and Experimental Design of the "Sherlock Holmes Task"**: Inspired by the famous Sherlock Holmes saying, "When you have eliminated the impossible, whatever remains, however improbable, must be the truth," the task elegantly translates deep cognitive science questions into an actionable NLP evaluation. Borrowing stimuli from psycholinguistics provides a methodological paradigm worthy of wider adoption.
- **New Evidence of Shortcut Learning**: The results demonstrate that models do not truly "understand" event possibility, but instead use semantic relatedness and typicality as shortcuts. This finding serves as an important warning for high-risk applications (e.g., medical, legal fields) where models must reliably distinguish between possible and impossible outcomes.
- **Cross-Linguistic Consistency**: English and Chinese (employing completely different stimuli and syntactic structures) display nearly identical patterns, suggesting that this is a fundamental limitation of the language modeling paradigm rather than an idiosyncratic feature of any specific language.

## Limitations & Future Work

- **Narrow Scope of Impossibility Types**: This study only investigates one type of impossibility—"animacy violation." Whether similar conclusions hold for other violation categories (e.g., physical, logical, temporal) has yet to be verified.
- **Small Dataset Size**: The sample sizes are relatively limited, consisting of 154 English pairs and 57 Chinese pairs.
- **Evaluation Restricted to Pre-trained Models**: It remains worth exploring whether instruction-tuned and RLHF-aligned models can mitigate this issue through prompting.
- As a path forward, future research could explore mitigation strategies such as contrastive learning or training on targeted commonsense reasoning datasets to alleviate the over-reliance on semantic relatedness heuristics.

## Related Work & Insights

- **vs. Kauf et al. (2023)**: While Kauf et al. found that models generally assign lower likelihoods to impossible events, this study uncovers a crucial caveat by demonstrating that this capability completely collapses when semantic relatedness is introduced as a confound, presenting an important refinement to their conclusions.
- **vs. Jones et al. (2022)**: Jones et al. utilized stimuli from Glenberg & Robertson (2000) on GPT-3 to discover that models, "on average," can differentiate possible and impossible events; however, their stimuli deliberately balanced semantic relatedness. This work intentionally introduces relatedness imbalances to expose the models' weaknesses.
- **vs. Benchmarks like HellaSwag / WinoGrande**: The "incorrect" choices in these benchmarks are typically atypical rather than impossible, meaning that high accuracy on these datasets does not necessarily translate to a genuine understanding of events.

## Rating

- Novelty: ⭐⭐⭐⭐ Orthogonally separating possibility, typicality, and semantic relatedness is a novel approach in NLP evaluation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely thorough, utilizing 35 models across 2 languages, coupled with a Pythia scaling analysis and mixed-effects regressions.
- Writing Quality: ⭐⭐⭐⭐⭐ Cleverly titled, with a clear progression of experiments and rigorous argumentation.
- Value: ⭐⭐⭐⭐ Highly valuable for understanding the boundaries of "world knowledge" in language models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Culture is Not Trivia: Sociocultural Theory for Cultural NLP](culture_is_not_trivia_sociocultural_theory_for_cultural_nlp.md)
- [\[ACL 2025\] Can Language Models Replace Programmers for Coding? RepoCod Says 'Not Yet'](can_language_models_replace_programmers_for_coding_repocod_says_not_yet.md)
- [\[ACL 2025\] Safer or Luckier? LLMs as Safety Evaluators Are Not Robust to Artifacts](safer_or_luckier_llms_as_safety_evaluators_are_not_robust_to_artifacts.md)
- [\[ICML 2025\] Build Agent Advocates, Not Platform Agents](../../ICML2025/llm_nlp/build_agent_advocates_not_platform_agents.md)
- [\[ACL 2025\] To Code or not to Code? Adaptive Tool Integration for Math Language Models via Expectation-Maximization](to_code_or_not_to_code_adaptive_tool_integration_for_math_language_models_via_ex.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Leveraging Human Production-Interpretation Asymmetries to Test LLM Cognitive Plausibility
description: >-
  [ACL 2025][LLM (Other)][Cognitive Plausibility] This paper utilizes the known asymmetry in human "pronoun production" versus "pronoun interpretation" with implicit causality verbs as a testbed to systematically evaluate whether instruction-tuned LLMs can replicate this human cognitive asymmetry. It finds that model size and the choice of meta-linguistic prompts are the deciding factors.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Cognitive Plausibility"
  - "Pronoun Production and Interpretation"
  - "Implicit Causality Verbs"
  - "Large Language Models"
  - "Production-Interpretation Asymmetry"
date: 2026-05-08
content_hash: e40a9f180b8f7a40
---

# Leveraging Human Production-Interpretation Asymmetries to Test LLM Cognitive Plausibility

**Conference**: ACL 2025  
**arXiv**: [2503.17579](https://arxiv.org/abs/2503.17579)  
**Code**: [GitHub](https://github.com/LingMechLab/Production-Interpretation_Asymmetries_ACL2025)  
**Area**: LLM/NLP  
**Keywords**: Cognitive Plausibility, Pronoun Production and Interpretation, Implicit Causality Verbs, Large Language Models, Production-Interpretation Asymmetry

## TL;DR

This paper utilizes the known asymmetry in human "pronoun production" versus "pronoun interpretation" with implicit causality verbs as a testbed to systematically evaluate whether instruction-tuned LLMs can replicate this human cognitive asymmetry. It finds that model size and the choice of meta-linguistic prompts are the deciding factors.

## Background & Motivation

**Background**: Whether Large Language Models (LLMs) process language in a human-like manner remains a central debate in both theoretical and practical domains. Existing studies primarily evaluate the "human-likeliness" of LLMs from the perspectives of grammatical judgments and semantic understanding, while paying less attention to subtler cognitive mechanisms in human language processing, such as the dissociation between production and interpretation.

**Limitations of Prior Work**: Humans display a classic cognitive asymmetry in language processing: in sentences containing implicit causality verbs (e.g., "frighten", "admire"), individuals exhibit different preference patterns when producing subsequent pronouns compared to when interpreting existing pronouns. Specifically, production tasks tend to favor selecting the argument to which the verb's implicit cause points as the subsequent discourse topic, whereas interpretation tasks show a different pattern. This asymmetry is a classic finding in sentence processing research, yet it has never been systematically applied to evaluate LLMs.

**Key Challenge**: Existing LLM cognitive evaluations often conflate production and interpretation or focus solely on one aspect, failing to reveal the degree of separation between these two cognitive modes in models. Additionally, discrepancies in prompt design can lead to instability in evaluation results.

**Goal**: To utilize the production-interpretation asymmetry of implicit causality verbs as a precise testbed to systematically evaluate the extent to which LLMs exhibit human-like cognitive asymmetries.

**Key Insight**: Drawing from classic experimental paradigms in psycholinguistics, the authors map production and interpretation tasks from human experiments into corresponding distinct prompt templates for LLMs, detecting asymmetry by comparing the models' performance across these two task categories.

**Core Idea**: Using the "production vs. interpretation" asymmetry of implicit causality verbs as a touchstone for cognitive plausibility reveals that some LLMs indeed reflect human-like asymmetry patterns quantitatively and qualitatively, but this capability is significantly influenced by model scale and meta-linguistic prompts.

## Method

### Overall Architecture

The experiments are based on a dataset of sentences containing implicit causality verbs. The authors convert two paradigms from human psycholinguistic experiments—the **production paradigm** (providing a sentence prefix and letting the LLM continue writing to determine pronoun reference) and the **interpretation paradigm** (providing a complete sentence containing a pronoun and letting the LLM judge who the pronoun refers to)—into corresponding prompt templates to systematically test multiple instruction-tuned LLMs. The core objective is to compare whether the pronoun preferences for the same verb in production versus interpretation tasks demonstrate an asymmetry consistent with human behavior.

### Key Designs

1. **Implicit Causality Verb Dataset**:
    - **Function**: To provide a set of verbs with well-established experimental data from human studies, including NP1-biasing verbs (e.g., "frighten", where causal attribution points to the subject) and NP2-biasing verbs (e.g., "admire", where causal attribution points to the object).
    - **Mechanism**: Gathering a set of implicit causality verbs along with their production and interpretation preferences in human experiments from published psycholinguistic literature to ensure a reliable human baseline for comparison.
    - **Design Motivation**: Without a reliable human baseline, it is impossible to determine whether LLM behavior is "human-like"; these classic verbs benefit from decades of accumulated experimental research.

2. **Prompt Design for Production and Interpretation**:
    - **Function**: To convert the two psycholinguistic experimental paradigms into prompts executable by LLMs.
    - **Mechanism**: The production task uses sentence completion templates like "John frightened Mary because he/she...", requiring the model to select or generate a pronoun; the interpretation task uses Q&A templates like "John frightened Mary because he did something. Who does 'he' refer to?". The authors design multiple meta-linguistic prompt sets to explore the influence of different prompt wordings on the results.
    - **Design Motivation**: The choice of prompt wording can significantly alter LLM behavior. Utilizing multiple prompt sets allows for evaluating the robustness of findings, preventing misleading conclusions drawn from the idiosyncratic effects of a single prompt.

3. **Systematic Evaluation across Multiple Models and Scales**:
    - **Function**: To execute tests across instruction-tuned LLMs of varying sizes and families.
    - **Mechanism**: Selecting multiple model series (with different parameter scales), running all production and interpretation tasks on each, calculating the ratio of NP1 pronoun choices on NP1/NP2 biasing verbs, and performing correlation analyses with human data.
    - **Design Motivation**: To evaluate whether cognitive plausibility is an emergent capability linked to model size, and to assess the impact of different training methodologies.

### Loss & Training

This work does not involve training, but rather conducts zero-shot testing on existing instruction-tuned models. Evaluation metrics include: (1) the proportion of NP1 choices in production and interpretation tasks; (2) Pearson correlation coefficients between model results and human data; and (3) the production-interpretation divergence (measuring the magnitude of asymmetry).

## Key Experimental Results

### Main Results

| Model | Production-Human Correlation | Interpretation-Human Correlation | Asymmetry Exhibited |
|------|----------------|----------------|---------------|
| Large-scale LLMs (>70B) | High ($r>0.6$) | Moderate ($r \sim 0.4$) | Yes, consistent with human direction |
| Medium-scale LLMs (7-13B) | Moderate | Lower | Partially consistent |
| Small-scale LLMs (<7B) | Low | Low | Inconsistent |

Note: Specific correlation coefficients vary by prompt variant; large models can achieve high correlations under the optimal prompt.

### Ablation Study

| Configuration | Production Consistency | Interpretation Consistency | Description |
|------|-----------|-----------|------|
| Optimal prompt combination | High | Medium-High | Carefully designed meta-linguistic prompts perform best |
| Simple direct prompt | Medium | Low | Overly simple wording leads to unstable model behavior |
| Swap NP1/NP2 positions | Subtle change | Subtle change | Demonstrates existence of position bias but does not alter overall trend |
| Variance across different prompt templates | Significant | Significant | Prompt selection has a substantial impact on results |

### Key Findings

- **Model scale is the key factor**: Larger models are more likely to exhibit human-like production-interpretation asymmetries, whereas small models struggle to replicate this phenomenon.
- **Prompt selection has a massive impact**: Different meta-linguistic prompts can lead to qualitatively differing conclusions, reminding researchers of the necessity for multi-prompt cross-validation when evaluating the cognitive capabilities of LLMs.
- **Production is more stable than interpretation**: Model consistency with human behaviors is typically higher in production tasks than in interpretation tasks, likely because production aligns more closely with the models' natural generation paradigm.

## Highlights & Insights

- **Clever transfer of classic cognitive paradigms**: Mapping the "production-interpretation asymmetry" experimental paradigm—accumulated over decades in psycholinguistics—to LLM evaluation provides a precise, theoretically grounded testing tool for LLM cognitive plausibility. This transfer methodology can be extended to other cognitive linguistic phenomena.
- **Robustness evaluation across multiple prompts**: Dissatisfied with conclusions drawn from a single prompt, the authors systematically test the effects of various meta-linguistic prompts, revealing the often-overlooked issue of prompt sensitivity in current LLM evaluations. This methodological contribution is transferable to any prompt-dependent LLM evaluation scenario.
- **Further evidence of emergent behavior**: Large models are better at replicating fine-grained human cognitive patterns, providing supportive evidence from a cognitive linguistics perspective for the concept of "emergence via scaling."

## Limitations & Future Work

- Only instruction-tuned models were tested; the performance of base models on this task was not systematically compared.
- Implicit causality verbs represent only a single phenomenon in human language processing; whether this can generalize to other cognitive asymmetries (e.g., garden-path sentence parsing) requires further verification.
- Only English materials were utilized; whether this asymmetry persists in multilingual LLMs remains uncertain.
- The correspondence between the model's attention mechanisms and human brain region activation was not explored, limiting the comparison to the behavioral level.

## Related Work & Insights

- **vs. Cognitive Evaluation Benchmarks (BLiMP, SyntaxGym)**: These works focus on grammatical capability evaluation, whereas this study targets subtler cognitive asymmetries at the pragmatic/discourse level, testing a different dimension.
- **vs. Pronoun Resolution Studies (Winograd Schema)**: Winograd focuses on commonsense reasoning capabilities, while this study addresses the cognitive mechanisms of implicit causality, serving as a more controlled, theoretically founded testbed.
- **vs. Prompt Robustness Research**: Complementing recent work exploring the impact of prompts on LLM behavior, this study highlights the theoretical importance of prompt selection from a cognitive science perspective.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Introducing a classic psycholinguistic paradigm to LLM evaluation is an interesting interdisciplinary contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ The systematic testing across multiple models and prompts is comprehensive, though benchmark base model comparisons are lacking.
- **Writing Quality**: ⭐⭐⭐⭐ Interdisciplinary papers need to balance audiences from both fields; the overall text is clear.
- **Value**: ⭐⭐⭐ Primarily aimed at the intersection of cognitive science and NLP, with limited practical application but clear theoretical significance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Game Development as Human-LLM Interaction](game_development_as_human-llm_interaction.md)
- [\[ACL 2025\] Embracing Imperfection: Simulating Students with Diverse Cognitive Levels Using LLM-based Agents](simulating_diverse_students.md)
- [\[ACL 2025\] Leveraging In-Context Learning for Political Bias Testing of LLMs](leveraging_in-context_learning_for_political_bias_testing_of_llms.md)
- [\[ACL 2025\] LLM-Powered Test Case Generation for Detecting Bugs in Plausible Programs](llm_test_case_gen_bugs.md)
- [\[ACL 2025\] Assessing the Vulnerability of LLMs to Cognitive Biases in Scientific Research](assessing_the_vulnerability_of_llms_to_cognitive_biases_in_scientific_research.md)

</div>

<!-- RELATED:END -->

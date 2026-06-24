---
title: >-
  [Paper Note] Logical Forms Complement Probability in Understanding Language Model (and Human) Performance
description: >-
  [ACL 2025][LLM (Other)][logical reasoning] This paper systematically investigates LLM capabilities in propositional and modal logic reasoning, finding that in addition to input probability (perplexity), logical form (modality, argument form) is an important complementary factor in predicting LLM performance. It further compares these findings with human behavioral data to reveal similarities and differences in human-machine reasoning.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "logical reasoning"
  - "modal logic"
  - "propositional logic"
  - "syllogism"
  - "LLM evaluation"
date: 2026-05-08
content_hash: faa00e5b95c3727a
---

# Logical Forms Complement Probability in Understanding Language Model (and Human) Performance

**Conference**: ACL 2025  
**arXiv**: [2502.09589](https://arxiv.org/abs/2502.09589)  
**Code**: —  
**Area**: logical reasoning / LLM behavior analysis / cognitive science  
**Keywords**: logical reasoning, modal logic, propositional logic, syllogism, LLM evaluation  

## TL;DR

This paper systematically investigates LLM capabilities in propositional and modal logic reasoning, finding that in addition to input probability (perplexity), logical form (modality, argument form) is an important complementary factor in predicting LLM performance. It further compares these findings with human behavioral data to reveal similarities and differences in human-machine reasoning.

## Background & Motivation

### Background
As LLMs are widely deployed for planning and decision-making, understanding their logical reasoning capabilities becomes critical. Although existing studies show LLMs perform reasonably well on logical reasoning, they lack a fine-grained analysis of different **logical forms**. Specifically, do LLMs perform consistently across various argument forms presented in natural language, and do they exhibit preferences for certain forms?

### Core Problem
1. Is input probability (probability/perplexity) sufficient to predict LLM performance in logical reasoning?
2. Are logical forms (including modalities and argument forms) important additional predictors?
3. What are the similarities and differences between LLM and human performance in logical reasoning?

### Related Work & Insights
- Generally consistent with the probability hypotheses of Gonen et al. (2023) and McCoy et al. (2024), but **complements them with the key dimension of logical form**
- Complementary to the study on categorical syllogisms by Eisape et al. (2024), this work focuses on **hypothetical syllogisms and disjunctive syllogisms**
- First to introduce **modal logic** (necessity and possibility) into LLM benchmarks

## Method

### Dataset Construction

**Logical Form System**: Based on propositional logic and normal modal logic (alethic modal logic), using the following core operators:
- $\neg$ (negation), $\Box$ (necessity, must), $\Diamond$ (possibility, may)
- $\vee$ (disjunction, or), $\wedge$ (conjunction, and), $\to$ (implication, if...then)

**Logical Forms Involved** (4 valid inferences + 4 fallacies $\times$ 3 modalities = 24 forms):

Valid inferences (ground truth = Yes):
- $\vee^L$: Disjunctive syllogism (left negation of antecedent)
- $\to^L$: Hypothetical syllogism - Modus Ponens
- $\vee^R$: Disjunctive syllogism (right negation of consequent)
- $\to^R$: Hypothetical syllogism - Modus Tollens

Fallacies (ground truth = No):
- $\vee^L_\nvdash$: Affirming a disjunct
- $\to^L_\nvdash$: Affirming the Consequent
- $\vee^R_\nvdash$: Affirming a disjunct
- $\to^R_\nvdash$: Denying the Antecedent

**Modalities**: Each group of variables can operate under three modalities: $\Box$ (necessity), $\Diamond$ (possibility), $\varnothing$ (no modality / propositional logic)

### Control of Knowledge Bias

Key Design - Avoiding the influence of common-sense bias of interpreted content on reasoning:
- Generate 204 verb phrases + 200 common names to form subject-verb-object triples
- The content of the two variables in each interpretation is **independent**
- Randomly generate 1,000 interpretations applied to the 24 logical forms, totaling **24,000 problems**

### Translation to Natural Language
Translating logical forms into Yes/No question formats, for example:
> Consider the following statements: Jane is watching a show or John is reading a book. Jane isn't watching a show. Question: Based on these statements, can we infer that John is reading a book?

### Evaluation Metrics
Adopting a **probability-based soft accuracy**, avoiding the underestimation of model capability due to greedy decoding:

$$\hat{p} = \frac{p(\text{No}|s)\mathbb{1}[y=\text{No}] + p(\text{Yes}|s)\mathbb{1}[y=\text{Yes}]}{p(\text{No}|s) + p(\text{Yes}|s)}$$

## Experiments

### Evaluated Models
10 open-source models: mistral-7b/8x7b, llama-2-7b/13b/70b, llama-3-8b/70b, yi-34b, phi-2, phi-3-mini  
Commercial models (for reference): OpenAI o1, Gemini-1.5-Pro  

### Main Results

| Model | Overall Accuracy | No Modality | Necessity (□) | Possibility (◇) | MP | MT | Affirming Consequent |
|------|----------|-------|---------|---------|----|----|---------|
| llama-3-70b | 0.714 | 0.745 | 0.554 | **0.843** | 0.773 | **0.515** | 0.661 |
| mistral-8x7b | **0.724** | 0.698 | 0.601 | 0.874 | **0.873** | 0.023 | 0.648 |
| phi-3-mini | 0.690 | 0.657 | 0.536 | 0.877 | **0.974** | 0.475 | 0.462 |
| OpenAI o1 | 0.926 | 1.000 | 0.773 | 1.000 | 1.000 | 0.895 | 1.000 |
| Humans | 0.595 | 0.589 | 0.566 | 0.640 | **0.901** | 0.628 | **0.225** |

### Key Findings

**Finding 1: Modality has a significant impact**
- All models perform best under the **possibility (◇) modality** and worst under the **necessity (□) modality**
- Statistical tests are highly significant ($p < 0.001$)

**Finding 2: Significant differences across argument forms**
- Most models perform worst on **Modus Tollens** (among valid inferences)
- Most models perform worst on the **Affirming the Consequent** fallacy (among fallacies)
- Modus Ponens is the easiest form
- These patterns are all significant in linear mixed-effects models ($p < 0.001$)

**Finding 3: Perplexity is not a reliable predictor**
- The correlation between perplexity and accuracy is present but very weak ($\rho = -0.09$)
- Linear mixed-effects model: marginal $R^2 = 0.342$, conditional $R^2 = 0.543$
- Control experiments (replacing interpretations with nonsense words) show huge differences in perplexity (~30 vs. ~10) but performance differences depend on the logical form

**Finding 4: Modal biases of LLMs**
- Under the possibility modality, LLMs tend to give affirmative answers (Yes)
- Under the necessity modality, LLMs tend to give negative answers (No)

### Human-Machine Comparison
- **Communalities**: Both humans and LLMs perform best on Modus Ponens
- **Differences**: Humans perform extremely poorly on the "Affirming the Consequent" fallacy (0.225), whereas some LLMs perform acceptably
- LLM preferences for specific logical forms are sometimes unsupported by human intuition

### Statistical Analysis Method
Using a **linear mixed-effects model**:
$$\text{Acc}_\text{soft} \sim \text{Modality} + \text{ArgForm} + \text{Perplexity} + (1 + \text{Perplexity} | \text{LLM})$$

Fixed effects: modality, argument form, perplexity; Random effects: model-specific deviations

## Highlights & Insights

1. **Discovery of Logical Form as a Predictor**: The first study to systematically prove that logical form (rather than just probability) is a key factor in predicting LLM reasoning performance.
2. **Introduction of Modal Logic**: The first work to systematically incorporate necessity/possibility modalities in an LLM benchmark.
3. **Rigorous Experimental Controls**: The data construction method to eliminate knowledge bias provides a valuable reference.
4. **Soft Accuracy Metric**: The evaluation method avoids underestimating model capability due to greedy decoding.
5. **Human-Machine Comparison**: Provides new behavioral data, revealing fundamental similarities and differences between LLM reasoning and human reasoning.
6. **Nonsense Word Control Experiment**: Cleverly demonstrates the unreliability of perplexity.

## Limitations & Future Work

1. Only atomic-level propositional and modal logic reasoning are considered; multi-step complex reasoning is not evaluated.
2. Evaluated in a zero-shot setting; adding few-shot exemplars might alter absolute performance and patterns.
3. The sample size for human experiments is limited, which might introduce demographic bias.
4. Commercial models (o1, Gemini) could only be evaluated using greedy decoding due to API limitations, making them not directly comparable.
5. Other logical systems beyond modal logic (such as temporal logic or epistemic logic) have not been explored.

## Related Work & Insights

- **Logical Reasoning Benchmarks**: Synthetic datasets like ProofWriter and PrOntoQA exist but do not cover modal logic.
- **Logical Reasoning by LLMs**: Training/fine-tuning approaches such as Clark et al. (2021) and Hahn et al. (2021) have been developed, whereas this study focuses on prompting evaluation.
- **Human Logical Reasoning**: Categorical syllogisms studied by Eisape et al. (2024), and content effects investigated by Lampinen et al. (2024).
- **Probability Hypothesis**: Gonen et al. (2023) and McCoy et al. (2024) propose that LLMs perform better on high-probability inputs.

## Rating

⭐⭐⭐⭐ — The research design is rigorous (controlled experiments + mixed-effects models), and the findings are insightful (logical form > probability). The debiasing method in dataset construction and the human-machine comparison add depth to the study. The main limitation lies in its focus on atomic reasoning and a specific subset of modal logic.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Exploring Graph Representations of Logical Forms for Language Modeling](exploring_graph_representations_of_logical_forms_for_language_modeling.md)
- [\[ACL 2025\] Collaborative Performance Prediction for Large Language Models](collaborative_performance_prediction_for_large_language_models.md)
- [\[ACL 2025\] ECLM: Entity Level Language Model for Spoken Language Understanding with Chain of Intent](eclm_entity_level_language_model_spoken_language_understanding.md)
- [\[ACL 2025\] Sample-Efficient Human Evaluation of Large Language Models via Maximum Discrepancy Competition](sample-efficient_human_evaluation_of_large_language_models_via_maximum_discrepan.md)
- [\[ACL 2025\] Meaning Beyond Truth Conditions: Evaluating Discourse Level Understanding via Anaphora Accessibility](meaning_beyond_truth_conditions_evaluating_discourse_level_understanding_via_ana.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] A Theory of Response Sampling in LLMs: Part Descriptive and Part Prescriptive
description: >-
  [ACL 2025][LLM (Other)][LLM Sampling] Proposes and validates a theory of response sampling in LLMs, demonstrating that the sampling process is simultaneously driven by dual components: descriptive forces (statistical norms) and prescriptive forces (implicit ideals). This causes samples to systematically skew away from statistical averages toward idealized values. This bias is statistically significant across 15 models and 500 concepts, and scales stronger with larger model si…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "LLM Sampling"
  - "Prescriptive Bias"
  - "Descriptive Norms"
  - "Prototype Theory"
  - "System-1"
date: 2026-05-08
content_hash: 81b3370f4dde683d
---

# A Theory of Response Sampling in LLMs: Part Descriptive and Part Prescriptive

**Conference**: ACL 2025  
**arXiv**: [2402.11005](https://arxiv.org/abs/2402.11005)  
**Code**: None  
**Area**: LLM Theory / Cognitive Science  
**Keywords**: LLM Sampling, Prescriptive Bias, Descriptive Norms, Prototype Theory, System-1

## TL;DR

Proposes and validates a theory of response sampling in LLMs, demonstrating that the sampling process is simultaneously driven by dual components: descriptive forces (statistical norms) and prescriptive forces (implicit ideals). This causes samples to systematically skew away from statistical averages toward idealized values. This bias is statistically significant across 15 models and 500 concepts, and scales stronger with larger model sizes.

## Background & Motivation

**Background**: LLMs are increasingly deployed to make autonomous decisions, sampling options from massive action spaces. Prior research viewed LLMs merely as "stochastic parrots" that probabilistically assemble patterns, but growing evidence suggests that LLMs can construct structured internal representations.

**Limitations of Prior Work**: The heuristics guiding LLM response sampling remain un-systematically studied. Although LLMs are known to perform poorly on probabilistic sampling tasks (generating samples inconsistent with expected probability distributions), an explanatory framework is lacking.

**Key Challenge**: In scenarios requiring decisions based on statistical norms (e.g., estimating medical recovery time), does LLM sampling faithfully reflect statistical distributions? If systematically biased, what are the sources and directions of such biases?

**Goal**: To systematically understand the sampling heuristics of LLMs through the lens of cognitive science, revealing their convergence and divergence with human decision-making.

**Key Insight**: Drawing on the "Theory of Normality" from human cognitive science (Bear & Knobe, 2017), which posits that human conceptual normality comprises both descriptive (statistically common) and prescriptive (valued ideals) components.

## Method

### Overall Architecture

Two core sets of experiments: (1) Constructive experiments — introducing novel concepts to systematically control descriptive and prescriptive components; (2) Natural concept experiments — validating the theory on 500 real-world concepts.

### Key Designs

1. **Novel Concept "Glubbing"**:
    - **Function**: Introduces a novel concept "glubbing" never seen by LLMs, accompanied by 100 samples (drawn from a Gaussian distribution $C_\mu=45$) and corresponding grades (A+ to D-).
    - **Mechanism**: Systematically varies the prescriptive direction $C_v$ (positive = high values are good, negative = low values are good, neutral = balanced), and observes whether the LLM sampling $S(C)$ deviates from the reported average $A(C)$. Key control: $A(C) \approx C_\mu$ proves that the LLM understands the distribution, but $S(C)$ systematically deviates from $A(C)$ in the direction aligned with $C_v$.
    - **Design Motivation**: Utilizing novel concepts eliminates confounding pre-training knowledge. By only changing $C_v$ while keeping everything else constant, the effect of the prescriptive component is isolated.

2. **500 Natural Concepts Experiment**:
    - **Function**: Obtains the LLM-reported average $A(C)$, ideal $I(C)$, and sampled value $S(C)$ across 500 real-world concepts in 10 different domains.
    - **Mechanism**: Applies a binomial test to verify whether $S(C)$ systematically falls on the ideal side of $A(C)$. The deviation is defined as $\alpha = (A(C) - S(C)) \times \text{sign}(A(C) - I(C))$, where positive values indicate a sample shift towards the ideal.
    - **Design Motivation**: Generalizes findings from constructive validation to real-world knowledge internalized within LLMs.

3. **Conceptual Prototype Analysis**:
    - **Function**: Evaluates whether the LLM's prototypicality ratings for 8 concepts contain prescriptive components.
    - **Mechanism**: For 6 exemplars of each concept, averages, ideals, and prototypicality scores are gathered. If prototypicality ratings systematically lean towards the ideal side rather than the average side, it indicates that the LLM's prototype concepts incorporate prescriptive components.
    - **Design Motivation**: Prototypicality influences sampling (a fundamental feature of System-1). Understanding the prescriptive components in prototypes helps explain the origin of sampling biases.

### Loss & Training

Purely evaluative study, involving no training. The main experiments are conducted on GPT-4 (temperature=0.8), with each concept repeated 10 times. Extended validation covers 15 models: GPT-4, GPT-3.5-Turbo, Claude, Mixtral-8x7B, Mistral-7B, and the Llama-2/3 series (7B-70B, base and instruct).

## Key Experimental Results

### Main Results

Novel concept experiments (Glubbing, $C_\mu=45$, GPT-4):

| Condition | Prescriptive Direction | $A(C)$ | $S(C)$ | Significance |
|------|---------|--------|--------|--------|
| Unimodal | Positive | 44.94 | 46.72 | p=.003 |
| Unimodal | Negative | 44.99 | 36.50 | p<.001 |
| Unimodal | Neutral | 45.01 | 44.95 | p=.52 |
| Bimodal | Positive | 44.97 | 47.43 | Significant |
| Bimodal | Negative | 45.03 | 41.26 | Significant |

500 real-world concepts (GPT-4): Sampling for 304 out of 444 concepts shifted towards the ideal side, with binomial test $p = 5.06 \times 10^{-15}$.

### Ablation Study

Cross-model comparison (500-concept experiment):

| Model | Proportion of Concepts biased towards Ideal | Significance p |
|------|------------------|---------|
| GPT-4 | 0.680 | 5.5e-15 |
| Llama-3-70b-Instruct | 0.777 | 5.4e-35 |
| Llama-3-70b | 0.726 | 3.0e-21 |
| Claude | 0.688 | 1.6e-16 |
| Llama-2-7b | 0.539 | 6.8e-02 (not significant) |

### Key Findings

1. **The impact of prescriptive components scales with model size**: From Llama-2-7b to 70b, the proportion of ideal-biased concepts increases from 0.539 to 0.688.
2. **RLHF amplifies but is not the source**: Base pre-trained models already exhibit prescriptive bias, which is further exacerbated by RLHF (e.g., Llama-3-8b: 0.608 $\rightarrow$ Instruct: 0.716).
3. **Medical Case Study**: When LLMs serve as "physicians" to estimate recovery times, 26 out of 35 symptom batches sampled biased towards the ideal side ($p=0.003$), tending to underestimate recovery times.
4. **LLMs' ideals are more absolute than humans'**: Among 40 concepts, LLMs provided a value of 0 as the ideal for 19 concepts (e.g., "sugary drinks/week" = 0), whereas humans did so for only 1 concept.
5. **Debiased prompts cannot eliminate prescriptive components**: Even when explicitly instructed "do not bias towards high/low values," the sampling bias remains significant.

## Highlights & Insights

- **Bridging Cognitive Science and LLMs**: Systematically applies the human cognitive theory of "normality = descriptive + prescriptive" to LLM analysis for the first time.
- **Exquisite Experimental Design**: Uses novel concepts + neutral controls + diverse distributions (unimodal/bimodal) + varying mean ranges to thoroughly rule out confounding factors.
- **"Inverse Scaling Law"**: Bias amplifies as model size increases, warning against scenarios reliant on LLM decision-making (such as healthcare and finance).
- **Insights from Prototype Analysis**: LLMs view "a good high school teacher" as leaning towards the ideal rather than the statistical average, revealing the embedding of prescriptiveness within LLM concept representations.

## Limitations & Future Work

- The exact origins of prescriptive components (pre-training data vs. RLHF vs. architecture) remain not fully clarified.
- Mechanistic explanations of prescriptive bias (how it arises within attention/FFN layers) are not explored.
- Prototype analysis only covers 8 concepts, constituting a preliminary exploration.
- LLM ideals do not align with humans' (being more absolute), but a deep analysis of reasons is lacking.
- Only focuses on numerical concepts without expanding to categorical decision-making.

## Related Work & Insights

- **Bear & Knobe (2017)**: The cognitive science foundation of this paper, showing that human normal judgments incorporate both descriptive and prescriptive standards.
- **Gu et al. (2025)**: Documented that LLMs perform poorly in probability sampling, for which this paper provides an explanatory framework.
- **Insight**: When using LLMs for decision support, one must be aware of the existence of sampling biases. Especially in high-risk domains like healthcare and finance, LLMs may systematically lean toward the "ideal" rather than the "statistical norm".

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Outstanding theoretical contribution, systematically revealing the prescriptive component in LLM sampling for the first time.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Strong statistical rigor with 15 models, 500 concepts, multiple controlled experiments, and human comparisons.
- Writing Quality: ⭐⭐⭐⭐ Highly logical, but contains many mathematical symbols, presenting a slightly high barrier to entry.
- Value: ⭐⭐⭐⭐⭐ Profound implications for LLM trustworthiness and fairness, with medical case studies highlighting practical risks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Can Graph Descriptive Order Affect Solving Graph Problems with LLMs?](graph_descriptive_order_llm.md)
- [\[ACL 2025\] Only a Little to the Left: A Theory-grounded Measure of Political Bias in LLMs](political_bias_theory_grounded.md)
- [\[ACL 2025\] Brevity is the soul of sustainability: Characterizing LLM response lengths](brevity_is_the_soul_of_sustainability_characterizing_llm_response_lengths.md)
- [\[ACL 2025\] Theory of Mind in Large Language Models: Assessment and Enhancement](theory_of_mind_llm.md)
- [\[ACL 2025\] Culture is Not Trivia: Sociocultural Theory for Cultural NLP](culture_is_not_trivia_sociocultural_theory_for_cultural_nlp.md)

</div>

<!-- RELATED:END -->

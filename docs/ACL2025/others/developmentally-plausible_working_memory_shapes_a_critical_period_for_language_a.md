---
title: >-
  [Paper Note] Developmentally-plausible Working Memory Shapes a Critical Period for Language Acquisition
description: >-
  [Critical Period Hypothesis] Inspired by the "Less-is-More" hypothesis, this study proposes the DynamicLimit-Exp method, which integrates the exponential growth characteristics of human working memory during the critical period into language model training (by dynamically adjusting ALiBi slopes). GPT-2 models trained on Child-Directed Speech data using this method significantly outperform baselines without memory constraints and those with static constraints in syntactic eval…
tags:
  - "Critical Period Hypothesis"
  - "Working Memory"
  - "Language Acquisition"
  - "ALiBi"
  - "Cognitively Plausible Language Models"
date: 2026-05-08
content_hash: ab4e0e00a5cb3dba
---

# Developmentally-plausible Working Memory Shapes a Critical Period for Language Acquisition

| Conference | Area | arXiv | Code |
|------|------|-------|------|
| ACL2025 | Cognitive Science / Language Modeling | [2502.04795](https://arxiv.org/abs/2502.04795) | [GitHub](https://github.com/osekilab/CPLM) |

**Keywords**: Critical Period Hypothesis, Working Memory, Language Acquisition, ALiBi, Cognitively Plausible Language Models

## TL;DR

Inspired by the "Less-is-More" hypothesis, this study proposes the DynamicLimit-Exp method, which integrates the exponential growth characteristics of human working memory during the critical period into language model training (by dynamically adjusting ALiBi slopes). GPT-2 models trained on Child-Directed Speech data using this method significantly outperform baselines without memory constraints and those with static constraints in syntactic evaluation.

## Background & Motivation

### Critical Period Hypothesis (CPH)

The Critical Period Hypothesis posits that human language acquisition operates within a specific, highly efficient window of learning, after which language learning abilities decline significantly. Numerical studies (such as late exposure to sign language in deaf children and age effects in second language acquisition) support the existence of this hypothesis.

### The Less-is-More Hypothesis

The Less-is-More hypothesis proposed by Newport (1990) provides a compelling explanation for the critical period: **children acquire language more efficiently than adults precisely because of their limited cognitive resources (especially working memory)**. Limited processing capacity enables children to efficiently extract basic patterns and grammatical rules, whereas the superior cognitive abilities of adults make them susceptible to interference from complex information, hindering rule acquisition.

### Motivation

Although LLMs possess human-like language abilities, their data efficiency is far lower than that of humans, requiring 3-4 orders of magnitude more data. If language is a cultural product that evolved under human cognitive constraints (limited memory and processing capacity), introducing similar constraints into language models is not merely about mimicking human limitations, but rather about introducing an inductive bias that is inherently aligned with the target data (natural language).

## Method

### Modeling the Trajectory of Working Memory Development

Human working memory undergoes three developmental stages:
1. **Ages 2-7 (Early Childhood to Early School Age)**: Rapid improvement in information retention and processing capabilities.
2. **Ages 8-14 (Middle Childhood to Early Adolescence)**: Deceleration of growth.
3. **Subsequent to Age 15 (Late Adolescence)**: Stabilization, reaching adult levels.

An exponential model $y = b - a^x$ ($0 < a < 1$) is used to characterize this trajectory:
- $b$: The asymptotic upper bound of working memory capacity (adult level)
- $a$: The growth rate (smaller values of $a$ yield steeper initial growth)

Reasons for selecting the exponential model:
- The horizontal asymptote accurately represents biological upper bounds.
- The rapid initial growth aligns with empirical observations.
- Logarithmic and linear models fail to simultaneously capture rapid initial growth and eventual stabilization.

### Modeling Working Memory Constraints via ALiBi

Instead of using positional encodings, ALiBi (Attention with Linear Biases) applies a distance-dependent linear penalty to attention scores:

$$\text{Attention Score} = \text{softmax}(q_i K^\top + m \cdot B)$$

Where $B = [-(i-1), -(i-2), \cdots, 0]$, and $m \in [0, 1]$ represents the slope of each attention head. A larger slope $m$ imposes a heavier penalty on distant tokens, which is equivalent to a smaller working memory capacity.

### The DynamicLimit-Exp Method

Core Idea: The slope $m$ of ALiBi decays exponentially with the training epoch, simulating the exponential growth of working memory:

$$m_t = m_0 \cdot r^t$$

Where $m_0$ is the initial slope, $r \in (0,1)$ is the decay rate, and $t$ is the current epoch.

Working memory capacity is defined as:

$$w_t \coloneqq 1 - m_t$$

As $m_t$ decays exponentially, $w_t$ increases, allowing the model to transition from attending to short-range dependencies to long-range dependencies.

### Compared Methods

- **NoLimit**: Standard GPT-2 without any memory constraints.
- **StaticLimit**: Fixed ALiBi slope (static constraint).
- **DynamicLimit-Linear**: Linear decay of slopes.
- **DynamicLimit-Exp**: Exponential decay of slopes (proposed method).

## Experiments

### Experimental Setup

- **Model**: GPT-2 (small), trained from scratch.
- **Training Data**: AO-CHILDES (Child-Directed Speech dataset, ~5 million words) — simulating linguistic input for children.
- **Evaluation Benchmark**: Zorro — a targeted syntactic evaluation benchmark designed specifically for CDS, containing 13 syntactic categories.

### Main Results

Overall performance of DynamicLimit-Exp on the AO-CHILDES data:

| Model | Overall | Number of Categories Significantly Outperforming NoLimit |
|------|---------|----------------------|
| NoLimit | 56.5% | — |
| StaticLimit | 56.8% | — |
| DynamicLimit-Linear | 61.6% | Some categories |
| **DynamicLimit-Exp** | **62.2%** | 6 categories (z-test, p<0.05) |

DynamicLimit-Exp leads by a large margin in categories such as Argument Structure (67.7% vs. 44.8%), Case (95.2% vs. 70.8%), and Filler Gap (93.6% vs. 72.1%).

### Key Findings

1. **Exponential Growth Outperforms Linear Growth**: DynamicLimit-Exp overall outperforms DynamicLimit-Linear, verifying that the exponential model provides a more accurate modeling of the working memory development trajectory.
2. **Dynamic Constraints Outperform Static Constraints**: StaticLimit brings almost no improvement, indicating that the key lies in "gradual relaxation" rather than mere restriction.
3. **NoLimit Holds Advantages in Certain Categories**: It performs better in Binding and Ellipsis categories, indicating that certain syntactic phenomena require larger memory windows.
4. **Similar Trends Observed on a Different Dataset (Wikipedia)**: DynamicLimit-Exp consistently outperforms the baselines.

## Highlights & Insights

1. **Deep Intersection of Cognitive Science and NLP**: This work is not merely a technical effort to improve model efficiency, but also provides indirect computational evidence for the Critical Period Hypothesis. If simulating working memory development can improve model learning efficiency, working memory development is highly likely to be one of the underlying mechanisms of the critical period.
2. **Validation of "Less-is-More" in Models**: Experimental results confirm that limiting the "attention span" of the model (equivalent to limiting cognitive resources) during initial training actually helps the model better learn basic grammatical structures.
3. **Elegant Methodological Design**: Directly mapping the human cognitive developmental trajectory (exponential growth) to the Transformer architecture via exponential decay of ALiBi slopes is conceptually clear and simple to implement.
4. **Reverse Engineering Paradigm**: Following the reverse engineering methodology of Dupoux (2018), this work uses language models as a testbed for cognitive hypotheses rather than attempting to fully replicate human cognition.

## Limitations & Future Work

1. **Small Model Scale**: Only the GPT-2 small model is used; whether the findings generalize to larger models remains unclear.
2. **Limited Data Scale**: The CDS dataset contains only approximately 5 million words, which is far smaller than typical pre-training corpora for LLMs.
3. **Limited Evaluation Scope**: Evaluation is restricted to the Zorro syntactic benchmark, without covering broader linguistic abilities such as semantic understanding.
4. **Correspondence between Working Memory and Attention**: Equating ALiBi slopes with working memory constraints is a simplifying assumption; in reality, human working memory involves more complex cognitive processes.
5. **Focus Only on L1 Acquisition**: The study does not address critical period effects in L2 (second language) acquisition.

## Related Work & Insights

- **Critical Period Hypothesis**: The classic hypothesis by Lenneberg (1967), distinguishing critical period effects between L1 and L2.
- **Computational Modeling and Language Acquisition**: McCoy et al. (2020) and Warstadt et al. (2023) use LMs to test acquisition hypotheses.
- **Constantinescu et al. (2025)**: Used EWC to simulate critical periods in L2 acquisition, finding that LLMs do not organically exhibit critical period effects. In contrast, this study focuses on the internal developmental processes during the L1 critical period, rather than the post-critical period decline in plasticity.
- **ALiBi (Press et al., 2022)**: Originally designed to improve extrapolation capabilities; Clark et al. (2025) discovered that it can produce surprisal patterns similar to human reading times.

## Rating

⭐⭐⭐⭐ (4/5)

This paper elegantly introduces the "Less-is-More" hypothesis from cognitive science into language models through dynamic ALiBi slope decay, featuring an elegant concept and sound experimental design. The results provide indirect computational evidence for the hypothesis of working memory development during the critical period. The main drawbacks lie in the small model and data scales, as well as the simplified relationship between working memory and attention. As a cognitively inspired NLP study, the quality is exceptionally high.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] If Attention Serves as a Cognitive Model of Human Memory Retrieval, What is the Plausible Memory Representation?](if_attention_serves_as_a_cognitive_model_of_human_memory_retrieval_what_is_the_p.md)
- [\[ACL 2025\] Generating Plausible Distractors for Multiple-Choice Questions via Student Choice Prediction](distractor_gen_multiple_choice.md)
- [\[ACL 2025\] STRICTA: Structured Reasoning in Critical Text Assessment for Peer Review and Beyond](stricta_structured_reasoning_in_critical_text_assessment_for_peer_review_and_bey.md)
- [\[ACL 2025\] Hierarchical Memory Organization for Wikipedia Generation](hierarchical_memory_wikipedia_gen.md)
- [\[ACL 2025\] Autalic: A Dataset for Anti-Autistic Ableist Language In Context](autalic_a_dataset_for_anti-autistic_ableist_language_in_context.md)

</div>

<!-- RELATED:END -->

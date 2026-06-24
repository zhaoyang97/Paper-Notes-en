---
title: >-
  [Paper Note] Can LLMs Reliably Simulate Real Students' Abilities in Mathematics and Reading Comprehension?
description: >-
  [ACL 2025][NLP Understanding][LLM Evaluation] Using Item Response Theory (IRT) to evaluate 11 LLMs on the same capability scale as real students, this work finds that strong models without styling far outperform average students. While persona prompting to "act as a student of a certain grade" can alter performance, **no single model-prompt combination** can reliably simulate an average student across all subjects and grades.
tags:
  - "ACL 2025"
  - "NLP Understanding"
  - "LLM Evaluation"
  - "Simulating Students"
  - "Item Response Theory (IRT)"
  - "Educational Assessment"
  - "Persona Prompting"
date: 2026-05-08
content_hash: 41d26036ca9f5cb0
---

# Can LLMs Reliably Simulate Real Students' Abilities in Mathematics and Reading Comprehension?

**Conference**: ACL 2025  
**arXiv**: [2507.08232](https://arxiv.org/abs/2507.08232)  
**Code**: [GitHub](https://github.com/kvadityasrivatsa/IRT-for-LLMs-as-Students)  
**Area**: NLP Understanding  
**Keywords**: LLM Evaluation, Simulating Students, Item Response Theory (IRT), Educational Assessment, Persona Prompting

## TL;DR

Using Item Response Theory (IRT) to evaluate 11 LLMs on the same capability scale as real students, this work finds that strong models without styling far outperform average students. While persona prompting to "act as a student of a certain grade" can alter performance, **no single model-prompt combination** can reliably simulate an average student across all subjects and grades.

## Background & Motivation

**Background**: LLMs are increasingly used as "agent students" in Intelligent Tutoring Systems (ITS) and as synthetic examinees for pre-testing exams. If LLMs can reliably simulate real students, they could serve as a low-cost alternative to expensive human pilot testing.

**Limitations of Prior Work**: Current methods for evaluating whether LLMs resemble real students heavily rely on **subjective expert judgment** and lack quantitative frameworks. Although Liu et al. (2025) and Grohs et al. (2024) used LLMs to generate synthetic examinee data, they only focused on aggregated statistics, failing to investigate whether the latent abilities of LLMs align with specific grade levels.

**Key Challenge**: The **feasibility of LLMs as proxy students has not been quantitatively validated**—strong models may vastly outperform students, weak models might align only coincidentally, and the effectiveness of persona prompting remains highly uncertain.

**Goal**: (RQ1) How large is the performance gap between LLMs under standard prompting and real students of different grades? (RQ2) Does instructing LLMs to "act as a student of a certain grade" improve alignment?

**Key Insight**: Utilizing real student response data from the National Assessment of Educational Progress (NAEP) and applying the Rasch model (a form of IRT) to place both LLMs and students on the **same latent ability scale**.

**Core Idea**: Quantitatively evaluating whether LLMs can serve as reliable proxies for K-12 students using an IRT framework.

## Method

### Overall Architecture

Collecting the NAEP dataset (489 multiple-choice questions covering mathematics and reading for grades 4, 8, and 12), prompting 11 LLMs to answer the same set of questions, and estimating the LLMs' ability parameter $\theta_i$ using the Rasch model to compare them with the ability distribution of real students.

### Key Designs

1. **Rasch Model Ability Estimation**:

    - Probability of a correct answer: $P(X_{ij}=1) = \frac{e^{\theta_i - b_j}}{1 + e^{\theta_i - b_j}}$
    - where $\theta_i$ represents the ability of examinee (LLM) $i$, and $b_j$ represents the difficulty of item $j$.
    - Item difficulty is estimated from student accuracy: $b_j \approx \log\frac{1-p_j}{p_j}$
    - Design Motivation: The Rasch model is simple and interpretable, allowing LLMs and students to be placed on the same metric of ability.

2. **Percentile Alignment Metric**:

    - Assuming student ability follows a standard normal distribution $\theta \sim \mathcal{N}(0,1)$, the average student sits at the 50th percentile.
    - Percentile rank of an LLM: $\text{Percentile Rank} = \Phi(\theta_i) \times 100$
    - Closer proximity to 50 indicates a better alignment with the average student.

3. **Four Prompting Strategies**:

    - **Unenforced**: Standard zero-shot prompting without any persona instructions.
    - **GradeEnforcedMinimal**: Adds a single sentence: "Please answer as an average grade X student."
    - **GradeEnforcedBasicCoT**: Directs the model to reason how a student of that grade would choose.
    - **GradeEnforcedFullCoT**: Two-step reasoning—first estimating whether a student of that grade is likely to answer correctly, and then choosing the answer accordingly.

### Model Selection

Eleven diverse LLMs were selected: LLaMA2-13B/70B, LLaMA3.1-8B/70B, Mistral-7B, Qwen2.5-7B, Qwen2.5-Math, GPT-3.5-Turbo, o3-Mini, SocraticLM, and LearnLM-1.5-Pro, covering open-source/closed-source, various scales, and general/domain-specific fine-tuning.

## Key Experimental Results

### Main Results (Percentile Comparison of Unenforced vs. GradeEnforced)

| Model | Math G4 $P_U$ | Math G4 $P_E$ | Reading G4 $P_U$ | Reading G4 $P_E$ |
|------|------------|------------|------------|------------|
| LLaMA3.1-70B | 99.6 | 97.6 | 99.9 | 99.9 |
| o3-Mini | 98.9 | 98.3 | 99.3 | 99.9 |
| Qwen2.5-7B | 99.6 | 18.5 | 98.2 | 5.2 |
| GPT-3.5-Turbo | 89.0 | 44.7 | 99.7 | 61.8 |
| Mistral-7B | 63.7 | 58.9 | 94.3 | 67.9 |
| LLaMA2-13B | 63.7 | 66.1 | 99.7 | 95.5 |
| Target Percentile | **50** | **50** | **50** | **50** |

- The ideal value is 50. Most strong models vastly exceed 50 (e.g., o3-Mini has a percentile >86 across all settings).
- Average Deviation: Approximately 32.9–40.5 percentile points in math under Unenforced, and 28.8–30.2 under GradeEnforced.

### Accuracy Data

| Model | Math G4 | Math G8 | Math G12 | Reading G4 | Reading G8 | Reading G12 |
|------|--------|--------|---------|--------|--------|---------|
| LLaMA3.1-70B | 93.9% | 91.5% | 80.0% | 98.0% | 93.1% | 83.6% |
| SocraticLM | 92.7% | 94.3% | 81.7% | 70.3% | 62.5% | 58.2% |
| Mistral-7B | 65.9% | 54.7% | 53.3% | 89.1% | 86.1% | 83.6% |

### Key Findings

- **Strong models without enforcement vastly outperform students**: Models with high GSM8K scores (e.g., o3-Mini, Qwen2.5-Math) have percentiles >95 across all grades, failing completely to simulate real students.
- **Weak models may align by chance**: Due to their limited general capabilities, LLaMA2-13B and Mistral-7B happen to align closely with the average performance of certain grades.
- **Inconsistent effects of grade prompting**: GradeEnforcedFullCoT shows the largest variance but is not always optimal; Qwen2.5-7B drops sharply from 98.2 to 5.2 in G4 Reading (over-correction).
- **Education-tuned models show no advantage**: SocraticLM and LearnLM-1.5-Pro do not achieve better grade alignment than general-purpose models (such as Mistral-7B).
- **No model-prompt combination aligns across all grades and subjects.**

## Highlights & Insights

- **The IRT framework provides an interpretable quantitative alignment metric**, which is more informative than traditional accuracy-based comparisons.
- **Reveals an important reality**: Using current LLMs as "proxy students" for educational assessment is unreliable, signaling a need for specialized training strategies.
- **Proposes three criteria for proxy student selection**: Grade alignment ($\theta$ within the normal band), developmental progression (ability increases monotonically with grade level), and prompt stability.
- **Sustainably updatable dataset**: NAEP regularly publishes new questions, allowing dynamic dataset expansion.

## Limitations & Future Work

1. **Limited to text-only multiple-choice questions**: Excludes questions with diagrams or open-ended formatting, falling short of fully covering real-world student assessment scenarios.
2. **Lack of cross-grade student data**: NAEP only reports the performance of same-grade students on same-grade items, which prevents validating student behavior on items designed for other grade levels.
3. **Exclusively prompt-based methods**: More profound alignment strategies like fine-tuning or in-context learning remain unexplored.
4. **Limited sample size**: Reflecting only 489 items across 2 subjects and 3 grades, which lacks wider coverage.
5. **Under-explored design space for persona prompting**: Only three types of grade enforcement prompting strategies were evaluated.

## Related Work & Insights

- **Relationship with Benedetto et al. (2024)**: The latter found that GPT-4 can simulate examinees of various levels via a single line of prompting, but did not use IRT to quantify this alignment—this work bridges that gap with a quantitative analysis.
- **Comparison with Zelikman et al. (2023)**: The latter simulated students in K-12 reading but only conducted aggregated correlation analyses rather than testing explicit grade-level alignment.
- **Insight**: For applications requiring LLMs to simulate human behavior (not limited to education), a quantitative alignment framework should be established before application.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Applying the IRT framework to evaluate LLM-student alignment is a novel entry point.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive evaluation covering 11 models $\times$ 4 prompting strategies $\times$ 3 grades $\times$ 2 subjects.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured with rigorously designed, RQ-driven analytical logic.
- **Value**: ⭐⭐⭐⭐ — Offers significant cautionary insights for the Educational AI domain.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Can LLMs Estimate Cognitive Complexity of Reading Comprehension Items?](../../ACL2026/nlp_understanding/can_llms_estimate_cognitive_complexity_of_reading_comprehension_items.md)
- [\[ACL 2025\] Automatic Generation of Inference Making Questions for Reading Comprehension Assessments](automatic_generation_of_inference_making_questions_for_reading_comprehension_ass.md)
- [\[ACL 2025\] Active LLMs for Multi-hop Question Answering](active_llms_for_multi-hop_question_answering.md)
- [\[ACL 2025\] RISE: Reasoning Enhancement via Iterative Self-Exploration in Multi-hop Question Answering](rise_reasoning_enhancement_via_iterative_self-exploration_in_multi-hop_question_.md)
- [\[ACL 2025\] Analyzing Political Bias in LLMs via Target-Oriented Sentiment Classification](analyzing_political_bias_in_llms_via_target-oriented_sentiment_classification.md)

</div>

<!-- RELATED:END -->

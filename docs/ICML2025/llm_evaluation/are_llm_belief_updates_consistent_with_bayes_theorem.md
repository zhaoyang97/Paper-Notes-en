---
title: >-
  [Paper Note] Are LLM Belief Updates Consistent with Bayes' Theorem?
description: >-
  [ICML 2025 (Workshop on Assessing World Models)][LLM Evaluation][Bayesian coherence] This paper proposes the Bayesian Coherence Coefficient (BCC) to quantify whether LLM belief updates conform to Bayes' theorem, revealing that larger and more powerful pretrained models exhibit belief updates that are more consistent with Bayes' theorem when presented with new evidence.
tags:
  - "ICML 2025 (Workshop on Assessing World Models)"
  - "LLM Evaluation"
  - "Bayesian coherence"
  - "belief update"
  - "probabilistic reasoning"
  - "scaling"
date: 2026-05-08
content_hash: 503f3c502549fc25
---

# Are LLM Belief Updates Consistent with Bayes' Theorem?

**Conference**: ICML 2025 (Workshop on Assessing World Models)  
**arXiv**: [2507.17951](https://arxiv.org/abs/2507.17951)  
**Code**: None  
**Area**: LLM Evaluation  
**Keywords**: Bayesian coherence, belief update, LLM evaluation, probabilistic reasoning, scaling

## TL;DR
This paper proposes the Bayesian Coherence Coefficient (BCC) to quantify whether LLM belief updates conform to Bayes' theorem, revealing that larger and more powerful pretrained models exhibit belief updates that are more consistent with Bayes' theorem when presented with new evidence.

## Background & Motivation

### Background

**Background**: LLMs perform exceptionally well in reasoning and decision-making tasks, but whether their internal processes follow the fundamental principles of probabilistic reasoning remains unclear.

**Limitations of Prior Work**: There is a lack of systematic methods to evaluate whether LLMs can rationally update their "beliefs" (credence in propositions) upon receiving new evidence.

**Key Challenge**: Although LLMs are not explicit probabilistic models, they are frequently required to make judgments involving uncertainty in practical applications. If belief updates do not conform to Bayesian principles, they may lead to inconsistent and unreliable reasoning.

**Goal**: Quantify the degree of consistency between an LLM's in-context belief updates and Bayes' theorem.

**Key Insight**: Construct a specialized dataset and calculate consistency by comparing the changes in LLM credence before and after observing evidence with the predictions of Bayes' theorem.

**Core Idea**: Propose the BCC metric to systematically measure the Bayesian coherence of multiple model families, finding that model scale and capability are positively correlated with coherence.

### Mechanism

**Goal**: ### Overall Architecture
The authors design an evaluation pipeline: (1) construct a dataset containing propositions, evidence, and prior/posterior probabilities; (2) prompt the LLM to output its credence in the propositions under conditions with and without evidence, respectively; (3) calculate the theoretical posterior using Bayes' theorem and compare it with the LLM's actual updates.


## Method

### Overall Architecture
The authors design an evaluation pipeline: (1) construct a dataset containing propositions, evidence, and prior/posterior probabilities; (2) prompt the LLM to output its credence in the propositions under conditions with and without evidence, respectively; (3) calculate the theoretical posterior using Bayes' theorem and compare it with the LLM's actual updates.

### Key Designs
1. **Bayesian Coherence Coefficient (BCC)**: Measures the consistency between LLM credence updates and the predictions of Bayes' theorem. Given the prior $P(H)$, likelihood $P(E|H)$, and the LLM's posterior $P_{LLM}(H|E)$, BCC measures the deviation of the latter from the Bayesian posterior $P(H|E) = \frac{P(E|H)P(H)}{P(E)}$. **Design Motivation**: A quantifiable, model-agnostic metric is required to compare across different model families.

2. **Dataset Construction**: Generate proposition-evidence pairs covering diverse domains (science, medicine, everyday reasoning, etc.) to ensure coverage over different prior probabilities and evidence strengths. Manual and automated methods are employed to guarantee data quality and diversity.

3. **Multi-dimensional Evaluation**: Perform correlation analysis between BCC and model parameters, training data size, and scores on common benchmarks to explore which factors best predict Bayesian coherence.

### Loss & Training
This is an evaluation study, with the core focus on the design of the evaluation protocol. Prompt engineering is utilized to guide the LLM to output credence values between 0 and 1.

## Key Experimental Results

### Main Results

| Model Family | Parameters | BCC ↑ | MMLU | Trend |
|--------------|------------|-------|------|-------|
| Small Models | <7B        | Lower | Lower | Baseline |
| Medium Models| 7B-30B     | Medium| Medium| Improvement |
| Large Models | 30B-70B    | Higher| Higher| Significant Improvement |
| Largest Models| >70B      | Highest| Highest| Saturating |

### Ablation Study

| Configuration | BCC Change | Description |
|---------------|------------|-------------|
| Different Prompt Formats | Fluctuation | Prompt design affects the stability of credence extraction |
| Propositions from Different Domains | Domain-dependent | Coherence is typically higher in scientific domains |
| Variations in Evidence Strength | Drop in coherence | More coherent under strong evidence, with larger deviations under weak evidence |

### Key Findings
- Larger and more capable LLMs exhibit belief updates more consistent with Bayes' theorem
- Model benchmark scores positively correlate with BCC
- Even the best models fall far short of perfect Bayesian coherence

## Highlights & Insights
- The first study to systematically evaluate the coherence of LLM belief updates using Bayes' theorem
- The BCC metric design is simple yet effective, and can be used for future model evaluations
- Provides new evidence for understanding whether LLMs "implicitly learn" probabilistic reasoning
- Holds significant implications for AI governance

## Limitations & Future Work
- Only evaluates pretrained models, without testing RLHF/instruction-tuned models
- Credence extraction depends on prompt design, which may introduce systematic bias
- The dataset scale and domain coverage can be further expanded

## Related Work & Insights
- Complementary to work related to calibration
- Insight: Bayesian coherence can be used as one of the training objectives

## Rating
- Novelty: ⭐⭐⭐⭐ Evaluating LLM belief updates from a Bayesian perspective is a novel approach
- Experimental Thoroughness: ⭐⭐⭐ Covers multiple model families but the overall scale is limited (Workshop paper)
- Writing Quality: ⭐⭐⭐⭐ Clear motivation of the problem, concise discussion
- Value: ⭐⭐⭐⭐ Holds reference value for LLM trustworthiness evaluation

---

## Additional Reflections

### Relationship with Domain Trends
The research direction of this paper is closely related to several major trends in current AI research: (1) the growing demand for an in-depth understanding of LLM internal mechanisms; (2) the increasing importance of model efficiency and accessibility; and (3) AI safety and reliability becoming core concerns. From a methodological perspective, this work represents a paradigm shift from "black-box utilization" to "white-box understanding."

### Specific Suggestions for Future Research
1. The core mechanism can be integrated with other modalities (vision, audio)
2. Consider validating the generalizability of the conclusions on larger models and datasets
3. Explore the possibility of combining this with reinforcement learning and online learning
4. Develop automated evaluation and optimization toolchains


---

## Additional Reflections

### Relationship with Domain Trends
The research direction of this work is closely related to several major trends in current AI research: model capability evaluation and reliability assurance, parameter-efficient fine-tuning and model compression, and AI safety and alignment. From a methodological perspective, this paper represents an exploration of the deep mechanisms of LLMs, helping to drive a paradigm shift from empirically-driven to theoretically-driven research.

### Specific Suggestions for Future Research
1. The core mechanism can be integrated with other modalities (vision, audio, multi-modality) to verify cross-modal generalizability.
2. Validate conclusions on larger-scale models (70B+) and newer architectures (e.g., Mixture-of-Experts).
3. Explore the possibility of integration with reinforcement learning and online learning to achieve dynamic adaptation.
4. Develop automated evaluation and optimization tools to lower the barrier to using the method.
5. Consider the intersection with LLM alignment research to explore the co-optimization of safety and performance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Textual Bayes: Quantifying Prompt Uncertainty in LLM-based Systems](../../ICLR2026/llm_evaluation/textual_bayes_quantifying_prompt_uncertainty_in_llm-based_systems.md)
- [\[NeurIPS 2025\] Ineq-Comp: Benchmarking Human-Intuitive Compositional Reasoning in Automated Theorem Proving on Inequalities](../../NeurIPS2025/llm_evaluation/ineq-comp_benchmarking_human-intuitive_compositional_reasoning_in_automated_theo.md)
- [\[ICML 2025\] LLM-SRBench: A New Benchmark for Scientific Equation Discovery with LLMs](llm-srbench_a_new_benchmark_for_scientific_equation_discovery_with_large_languag.md)
- [\[ACL 2025\] ELABORATION: A Comprehensive Benchmark on Human-LLM Competitive Programming](../../ACL2025/llm_evaluation/elaboration_competitive_programming.md)
- [\[ICML 2025\] DataDecide: How to Predict Best Pretraining Data with Small Experiments](datadecide_how_to_predict_best_pretraining_data_with_small_experiments.md)

</div>

<!-- RELATED:END -->

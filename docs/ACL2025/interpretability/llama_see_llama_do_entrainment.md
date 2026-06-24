---
title: >-
  [Paper Note] Llama See, Llama Do: A Mechanistic Perspective on Contextual Entrainment and Distraction in LLMs
description: >-
  [ACL 2025][Interpretability][contextual entrainment] This paper discovers and defines the phenomenon of "contextual entrainment" — where LLMs assign higher probabilities to any tokens that have appeared in the context. Using a differentiable masking method, the study localizes the entrainment heads responsible for this phenomenon and demonstrates that turning off these heads significantly suppresses distraction effects.
tags:
  - "ACL 2025"
  - "Interpretability"
  - "contextual entrainment"
  - "attention heads"
  - "LLM distraction"
  - "mechanistic interpretability"
  - "differentiable masking"
date: 2026-05-08
content_hash: 119e640f535b2ab5
---

# Llama See, Llama Do: A Mechanistic Perspective on Contextual Entrainment and Distraction in LLMs

**Conference**: ACL 2025  
**arXiv**: [2505.09338](https://arxiv.org/abs/2505.09338)  
**Code**: [https://github.com/frankniujc/entrainment](https://github.com/frankniujc/entrainment)  
**Area**: LLM Interpretability / Mechanistic Analysis  
**Keywords**: contextual entrainment, attention heads, LLM distraction, mechanistic interpretability, differentiable masking

## TL;DR

This paper discovers and defines the phenomenon of "contextual entrainment" — where LLMs assign higher probabilities to any tokens that have appeared in the context. Using a differentiable masking method, the study localizes the entrainment heads responsible for this phenomenon and demonstrates that turning off these heads significantly suppresses distraction effects.

## Background & Motivation
**Background**: LLMs excel at utilizing contextual information (e.g., in-context learning), but are also prone to distraction by irrelevant information in the context, leading to the generation of incorrect answers.

**Limitations of Prior Work**: Existing definitions of distraction are overly broad (relying solely on "relevant/irrelevant" distinctions) and lack precise categorization and mechanism-level analysis.

**Key Challenge**: Distraction is an easily understood yet difficult-to-define phenomenon; irrelevant context is sometimes even beneficial to the model, indicating the need for a more fine-grained analysis.

**Goal**: To understand from a mechanistic level why LLMs are distracted by contextual information and to locate the corresponding attention heads.

**Key Insight**: By observing the changes in logits for tokens appearing in the context, it is found that even random tokens receive higher probability, indicating that this is an underlying mechanistic phenomenon.

**Core Idea**: LLMs exhibit a contextual entrainment mechanism where "seeing is boosting." Differentiable masking can be used to localize and turn off the corresponding entrainment heads.

## Method

### Overall Architecture
An experimental setup containing a context prompt and a query prompt is constructed. Based on the LRE dataset (comprising 15 relation types, such as country-capital, fruit-color, etc.), the logit changes of target tokens are systematically measured under different context conditions (related/irrelevant/random/counterfactual). Up to 100K combinations are evaluated per relation type. Subsequently, differentiable masking is utilized to discover entrainment heads.

### Key Designs
1. **Contextual Entrainment Experiment**: Constructs four context conditions (related, irrelevant, random, counterfactual) based on the LRE dataset to measure the changes in logits/probabilities of distracting and correct tokens, verifying the ubiquity of the entrainment phenomenon.
2. **Entrainment Head Discovery via Differentiable Masking**: Introduces a binary mask $m_j$ for each attention head, implemented via Gumbel-sigmoid distribution for differentiable approximation, and uses gradient descent optimization to find the combination of heads that best suppresses entrainment.
3. **Sparsity Constraint**: The loss function includes a logit difference term and a sparsity regularization term $\mathcal{L} = \ell(\text{correct}) - \ell(\text{distract}) + \lambda \cdot \frac{1}{|H|}\sum \sigma(l_i)$, ensuring maximum suppression with the minimum number of heads.

### Loss & Training
- Uses AdamW optimizer with $\lambda=1.0$, $\tau=1.0$, and a learning rate of 1.0.
- Trained for 500 epochs, selecting the epoch that achieves the best performance with the fewest heads.
- 80/10/10 split for training/development/test sets on the LRE dataset.

## Key Experimental Results

### Main Results

| Metric | Original Model (with distraction) | Entrainment Heads Removed (with distraction) | Original Model (no distraction) |
|------|----------|----------|----------|
| $\ell$(correct) | 20.68 | 21.21 | 19.51 |
| $\ell$(distract) | 12.99 | 8.01 | 8.75 |
| Δ (correct - distract) | 7.69 | 13.20 | 10.76 |
| Avg distract token rank | 37.5 | 1289.6 | 1756.7 |

### Ablation Study

| Relation Type | Number of Heads (Density) | Original Δ | Δ After Head Removal |
|----------|--------------|-------|--------|
| company hq | 90 (8.8%) | 3.94 | 14.68 |
| country capital | 36 (3.5%) | 7.69 | 13.20 |
| country currency | 42 (4.1%) | 4.73 | 11.67 |
| fruit inside color | 56 (5.5%) | 0.97 | 11.16 |
| product by company | 110 (10.7%) | 3.62 | 16.47 |

### Key Findings
- All shifts are statistically significant (p<0.0001, paired t-test), consistent across 5 models.
- The probability of the distracting token can increase 10 to 100 times from $10^{-5} \sim 10^{-3}$.
- After turning off the entrainment heads, the model's strict/credulous accuracy on other relations remains largely unchanged.
- Performance on ICL tasks (arithmetic, spelling correction, translation) drops by only 0.2~3%.

- **Finding 1**: Contextual entrainment is ubiquitous — LLMs assign significantly higher logits to tokens appearing in the context (including random tokens), with all shifts being statistically significant (p<0.0001).
- **Finding 2**: Relevant "distracting" contexts are sometimes beneficial and can help disambiguate.
- **Finding 3**: Counterfactual contexts cause stronger distraction than factual contexts, indicating that entrainment is modulated by semantic factors.
- Only 3.2%~10.7% of attention heads are responsible for the entrainment phenomenon.
- Turning off entrainment heads has minimal impact on other capabilities (factual recall, ICL).

## Highlights & Insights
- Defines an entirely new phenomenon — contextual entrainment — which is distinct from the well-known induction head phenomenon (it does not require a prefix trigger).
- Reveals the mechanistic nature of distraction: it is both an underlying mechanistic phenomenon and modulated by semantic factors (counterfactual > factual > irrelevant > random).
- The differentiable masking method is superior to head-by-head analysis (Jin et al., 2024) as it captures the interaction structures between heads.
- Discovers that entrainment heads are task-specific rather than model-specific, with different numbers of heads identified for different relations.
- The naming "Llama see, llama do" is vivid and intuitive — the model tends to output whatever it sees.
- After turning off entrainment heads, the model's factual recall and ICL capabilities in other domains remain largely unaffected (strict/credulous accuracy is stable).
- The entrainment of random tokens is the strongest evidence of its mechanistic nature — no linguistic or factual factors can explain the probability rise of random tokens.
- Experiments cover 5 LMs (from GPT-2 XL to Llama-3.1-8B-Instruct), showing strong consistency in conclusions.

## Limitations & Future Work
- Only validated on smaller-scale models (up to 13B); scaling up to larger models (70B+) is required to verify the existence of entrainment heads.
- The method of turning off heads is relatively blunt (zeroing out outputs); more fine-grained intervention methods, such as activation patching, could be explored.
- Currently verified only on factual QA (LRE dataset); this can be extended to practical scenarios like RAG and long-text understanding.
- The cross-task transferability of entrainment heads warrants further study, as they are currently found to be relation-specific.
- Has not explored how to leverage entrainment heads for active defense (e.g., resisting prompt injection).
- The stronger distraction caused by counterfactual contexts implies the model's vulnerability to misinformation, but no defense schemes are proposed.

## Related Work & Insights
- Similar to but fundamentally different from induction heads (Olsson et al., 2022): entrainment does not require prefix triggers and is semantically modulated.
- Complementary to knowledge conflict studies (Jin et al., 2024): this work focuses on distraction mechanisms in non-conflict scenarios, and upgrades methodologically from single-head analysis to circuit-level analysis.
- Holds practical significance for RAG systems: understanding distraction mechanisms helps design more robust retrieval-augmented schemes.
- Shares the research paradigm of "localizing key components" with the knowledge editing work of Meng et al. (2022).
- The differentiable masking methods (Yu et al., 2024b; Bhaskar et al., 2024) provide a general tool for circuit discovery.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Defines an entirely new phenomenon of contextual entrainment, with a unique perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated across multiple models and relation types, though limited in scale (maximum 13B only).
- Writing Quality: ⭐⭐⭐⭐⭐ Clear diagrams, step-by-step findings, highly narrative, and classic naming.
- Value: ⭐⭐⭐⭐ Offers important insights into how LLMs utilize contextual information, with practical significance for RAG robustness research.
- Overall: Excellent work in the field of mechanistic interpretability, making a significant contribution to understanding the internal mechanisms of LLMs.
- Practicality: Can be directly applied to improve the context robustness of RAG systems.
- Reproducibility: Open-source code and clear experimental settings facilitate reproduction and expansion.
- Extensibility: Future work can explore the connection of entrainment heads with other phenomena (e.g., hallucination, sycophancy).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] An Empirical Study of Mechanistic Interpretability Approaches for Factual Recall](an_empirical_study_of_mechanistic_interpretability_approaches_for_factual_recall.md)
- [\[ACL 2025\] Reasoning Circuits in Language Models: A Mechanistic Interpretation of Syllogistic Inference](reasoning_circuits_in_language_models_a_mechanistic_interpretation_of_syllogisti.md)
- [\[ICLR 2026\] Uncertainty as Feature Gaps: Epistemic Uncertainty Quantification of LLMs in Contextual Question-Answering](../../ICLR2026/interpretability/uncertainty_as_feature_gaps_epistemic_uncertainty_quantification_of_llms_in_cont.md)
- [\[ACL 2025\] A Dual-Perspective NLG Meta-Evaluation Framework with Automatic Benchmark and Better Interpretability](a_dual-perspective_nlg_meta-evaluation_framework_with_automatic_benchmark_and_be.md)
- [\[ACL 2026\] Aligning What LLMs Do and Say: Towards Self-Consistent Explanations](../../ACL2026/interpretability/aligning_what_llms_do_and_say_towards_self-consistent_explanations.md)

</div>

<!-- RELATED:END -->

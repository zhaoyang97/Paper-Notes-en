---
title: >-
  [Paper Note] IROTE: Human-like Traits Elicitation of Large Language Model via In-Context Self-Reflective Optimization
description: >-
  [AAAI 2026][LLM/NLP][LLM personality simulation] This paper proposes IROTE, an in-context self-reflective optimization method grounded in information bottleneck theory. By iteratively generating and refining compact yet…
tags:
  - "AAAI 2026"
  - "LLM/NLP"
  - "LLM personality simulation"
  - "trait elicitation"
  - "information bottleneck"
  - "in-context learning"
  - "self-reflection"
date: 2026-05-08
content_hash: 415fe080e84d93fa
---

# IROTE: Human-like Traits Elicitation of Large Language Model via In-Context Self-Reflective Optimization

**Conference**: AAAI 2026
**arXiv**: [2508.08719](https://arxiv.org/abs/2508.08719)  
**Code**: Unavailable (only generated reflection texts are released)  
**Area**: LLM/NLP
**Keywords**: LLM personality simulation, trait elicitation, information bottleneck, in-context learning, self-reflection

## TL;DR

This paper proposes IROTE, an in-context self-reflective optimization method grounded in information bottleneck theory. By iteratively generating and refining compact yet evocative textual "self-reflections," IROTE stably elicits target human traits (values, morality, personality) from LLMs across diverse downstream tasks without any fine-tuning, consistently outperforming existing baselines in trait consistency.

## Background & Motivation

**Background**: Having been trained on massive human-generated corpora, LLMs exhibit a certain capacity to manifest specific human traits (personality, values, etc.) through prompting, and are widely applied in personalized dialogue, social simulation, and multi-agent systems. Existing methods fall into two categories: training-based (RLHF-SFT, reinforcement learning fine-tuning) and training-free (ICL in-context learning, role prompting).

**Limitations of Prior Work (Surface Elicitation Problem)**: Existing ICL-based elicitation methods suffer from a **surface elicitation** challenge—LLMs merely mimic shallow linguistic patterns in the prompt rather than genuinely internalizing the target trait. This manifests as:
   - Strong performance on simple questionnaires but inability to maintain consistent trait expression in complex open-ended tasks
   - Severe performance degradation on smaller models
   - High sensitivity to prompt wording (e.g., a mere phrasing difference between MFQ and MFQ-2 causes significant performance swings)

**Key Challenge**: The ICL demonstrations relied upon by existing methods (e.g., questionnaire responses or demographic descriptions) are overly surface-level and lack deep understanding of the essence of traits. Although lengthy backstories are informationally rich, they contain substantial irrelevant noise that distracts attention. What is needed is a prompt format that is simultaneously compact and profound.

**Key Insight**: Inspired by the psychological theory of "self-reflective identity processing"—human traits are formed through active self-reflection on identity-relevant experiences. Providing an LLM with a passage of self-perceived experiential reflection may activate its internal trait associations more effectively than simple role descriptions.

**Core Idea**: Automatically generate and optimize a brief textual "self-reflection" (e.g., "I maintain team harmony by mediating conflicts") using an information bottleneck-style objective that simultaneously optimizes **evocativeness** (maximizing mutual information between behavior and target trait) and **compactness** (removing redundancy via Total Correlation). This enables stable cross-task, cross-model trait elicitation without fine-tuning.

## Method

### Overall Architecture

The IROTE optimization process alternates among three steps:
1. **Compactness Enhancement**: Starting from $K$ candidate reflections $\mathcal{E} = (e_1, \ldots, e_K)$, a compact reflection $\hat{e}$ is synthesized by maximizing the pointwise mutual information approximation of Total Correlation, retaining shared information while removing idiosyncratic noise.
2. **Evocativeness Optimization**: Given the compacted $\hat{e}$, the lower bound of conditional mutual information $I_e(v; y|x)$ is maximized to encourage the LLM's responses to more explicitly express the target trait $v$.
3. **Candidate Update**: Multiple optimized candidates enter the next round of compactness enhancement, forming an iterative loop.

The target LLM parameters remain frozen throughout, making IROTE compatible with both black-box models (e.g., GPT-4o) and open-source models. The inputs are a small set of task prompts $\{x_i\}$, a target trait description $v$, and a trait evaluator $q_\omega$. The output is a compact self-reflection $e^*$ of at most 50 tokens.

### Key Designs

1. **Compactness Enhancement**:

    - **Function**: Distills shared core semantics from multiple candidate reflections and removes redundant details.
    - **Mechanism**: Maximizes $\text{TC}(e, \mathcal{E}) = \sum_k \text{PMI}(e, e_k) - \text{PMI}(e, \mathcal{E})$. The first term requires the reflection to recover each candidate and its corresponding behaviors; the second term $-\log p_e(\mathcal{E})$ penalizes excessive detail. Solved via EM iteration: the E-step samples a behavior set $\mathcal{S}_k^t$ for each $e_k$; the M-step selects the reflection maximizing $\mathcal{R}_1(e)$.
    - **Design Motivation**: Addresses the attention distraction caused by trait-irrelevant noise (e.g., demographic details such as age and hometown) in long reflections or backstories. Analogous to the compression bottleneck in information bottleneck theory—retaining necessary information while discarding the superfluous.

2. **Evocativeness Optimization**:

    - **Function**: Makes reflections more effective at eliciting the target trait in LLM outputs.
    - **Mechanism**: Maximizes the mutual information lower bound $I_e(v;y|x) \geq \frac{1}{N} \sum_i \sum_j p_e(y_i^j|x_i) \log q_\omega(v|y_i^j, x_i)$. EM iteration: E-step samples $M_2$ responses $y_i^{j,t}$ using the current reflection and computes trait evaluation scores $q_\omega(v|y_i^j, x_i)$; M-step selects the reflection maximizing $\mathcal{R}_2(e) = \frac{1}{N} \sum_i \sum_j p_e(y_i^j|x_i) \log q_\omega(v|y_i^j, x_i)$.
    - **Design Motivation**: Directly performs end-to-end optimization over the "reflection → behavior → trait expression" pipeline, rather than relying on manually designed heuristics.

3. **IB-like Constraint**:

    - **Function**: Treats compactness and evocativeness as dual constraints and automatically balances them.
    - **Mechanism**: The overall objective is $e^* = \arg\max_e \text{TC}(e, \mathcal{E}) + \beta I_e(v;y|x)$. Evocativeness maximization tends to produce longer reflections, but increased length reduces the compactness term, naturally forming an information bottleneck constraint.
    - **Design Motivation**: Prevents purely evocativeness-driven reflections from becoming verbose, or purely compactness-driven reflections from losing critical information. The equilibrium yields reflections that are both concise and effective.

### Loss & Training

IROTE requires no fine-tuning whatsoever and is a purely in-context method. Initialization uses GPT-4o to generate $K=10$ candidate reflections per trait. Each iteration samples $M_1=3$ behaviors and $M_2=6$ responses, with $\beta=1.0$, and convergence is reached within $T=5$ iterations. The maximum reflection length is 50 tokens. The trait evaluator $q_\omega$ uses rule-based methods for questionnaires and dataset-provided evaluators for downstream tasks. After convergence, the reflection set typically stabilizes at approximately 3 entries.

## Key Experimental Results

### Main Results: Cross-Model Comparison Across Three Trait Systems

| Method | STBHV-SVS(↑) | STBHV-AdAEM(↑) | MFT-MFQ2(↑) | MFT-MoralPrompt(↓) | BigFive-BFI2(↑) | BigFive-ROC(↑) | Avg(↑) |
|--------|-------------|----------------|-------------|---------------------|----------------|---------------|--------|
| Raw | 7.41 | 32.74 | 7.99 | 72.25 | 6.78 | 3.11 | 60.49 |
| Similarity | 6.81 | 35.05 | 6.92 | 81.72 | 7.15 | 3.62 | 58.72 |
| ICDPO | 7.80 | 35.24 | 7.78 | 51.82 | 7.77 | 3.84 | 67.67 |
| PICLe | 8.06 | 79.06 | 8.00 | 53.51 | 8.24 | 4.16 | 72.44 |
| EvoPrompt | 8.22 | 76.48 | 8.40 | 40.63 | 8.47 | 4.23 | 77.73 |
| **IROTE** | **8.16** | **80.03** | **8.97** | **36.07** | **8.32** | **4.36** | **80.01** |

The table above reports results on Qwen2.5-7B-Instruct. IROTE achieves a leading overall score of 80.01 (white background: questionnaire metrics; grey background: downstream task metrics).

### Ablation Study

| Dimension | Key Findings |
|-----------|-------------|
| Model scale scaling | Medium-scale models (7B) benefit most; 3B models lack sufficient reflection capacity, while 32B models are already strong without it |
| Reflection length scaling | 50 tokens is universally optimal; too short leads to insufficient information, too long introduces noise |
| Iterative convergence | IROTE converges stably within 5 rounds; EvoPrompt and ICDPO exhibit notable fluctuation |
| Removing compactness optimization | ROC drops by 1.6% on Mistral-7B, validating the necessity of compactness |
| In-context robustness | After inserting 10 irrelevant MMLU questions, IROTE's scores remain the most stable |
| GPT-4o results | Average 78.20 vs. EvoPrompt 77.15 vs. Anthology 74.30 |

### Key Findings
- **IROTE's advantage is more pronounced on downstream tasks**: PICLe and ICDPO perform reasonably on questionnaires but degrade substantially on complex tasks, corroborating the existence of the surface elicitation problem.
- **Compactness is the key differentiating factor**: Anthology generates lengthy narratives (containing irrelevant details such as age and hometown) that distract attention; EvoPrompt pursues conciseness but does not explicitly optimize compactness; IROTE automatically removes noise via the $-\log p_e(\mathcal{E})$ term.
- **Cross-model transferability**: IROTE remains effective on GPT-4o (with a smaller margin) and achieves the largest improvement on Mistral-7B.
- **Human evaluation consistency**: On MoralPrompt, IROTE averages 7.7 vs. EvoPrompt 6.7 vs. Anthology 6.0, consistent with automatic evaluation trends.

## Highlights & Insights
- **Elegant integration of psychological theory and information theory**: Drawing inspiration from the theory of self-reflective identity processing and formalizing the intuition that "reflections should be compact and evocative" via the information bottleneck framework is an exemplary case of interdisciplinary synthesis.
- **A fundamental solution to surface elicitation**: Rather than simply increasing the number or length of demonstrations, IROTE leverages information-theoretic objectives to automatically discover the most essential trait expression patterns. The resulting 42-token reflections outperform Anthology's lengthy narratives.
- **Transferability of a purely in-context method**: The same set of optimized reflection texts generalizes across GPT-4o, Qwen-7B, and Mistral-7B without requiring model-specific re-optimization, offering high practical utility.

## Limitations & Future Work
- **Limited coverage of trait systems**: Validation is restricted to the Schwartz Value Survey, Moral Foundations Theory, and Big Five personality frameworks; systems such as Kohlberg's moral development theory and Hofstede's cultural dimensions are not examined.
- **Limited model scope**: Only three models are tested; reasoning-oriented models (e.g., O1, DeepSeek-R1) are not covered.
- **Black-box probability estimation**: For models such as GPT-4o where logits are inaccessible, conditional probabilities are approximated via 0–10 scoring prompts, limiting precision.
- **Reliance on GPT-4o for downstream task evaluation**: ROC story evaluation uses GPT-4o as a scorer, potentially introducing evaluator bias.
- **Ethical risk**: The method could be exploited to elicit dangerous traits (e.g., power-seeking), necessitating accompanying safety mechanisms.

## Related Work & Insights
- **vs. PICLe (ICML 2024)**: PICLe selects ICL demonstrations using Bayesian likelihood ratios and relies on fine-tuned representations, making it sensitive to questionnaire format changes (MFQ → MFQ-2 causes a substantial drop)—a classic instance of surface elicitation. IROTE abstracts the essence of traits via self-reflection and is robust to format variation.
- **vs. Anthology (2024)**: Anthology constructs virtual personas using lengthy "life stories," which are informationally rich but verbose and noisy. IROTE's compactness optimization explicitly removes irrelevant details, achieving superior results with significantly shorter text.
- **vs. EvoPrompt**: EvoPrompt employs evolutionary algorithms for iterative prompt optimization, but mutation and crossover operations place high demands on smaller models, yielding only moderate performance on Mistral-7B. IROTE's information-theoretic objective is more stable and does not rely on complex evolutionary operations.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The first work to combine psychological self-reflection theory with the information bottleneck for LLM trait elicitation, with strong theoretical depth.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three trait systems, 7 questionnaires + 4 downstream tasks, 3 models, with detailed ablation, scaling, and human evaluation analyses.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, complete derivations, and intuitive case analyses.
- Value: ⭐⭐⭐⭐⭐ Provides a practical solution to the surface elicitation problem with direct applicability to personalized LLMs and social simulation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] ICL-Router: In-Context Learned Model Representations for LLM Routing](icl-router_in-context_learned_model_representations_for_llm_routing.md)
- [\[ICLR 2026\] ELLMob: Event-Driven Human Mobility Generation with Self-Aligned LLM Framework](../../ICLR2026/llm_nlp/ellmob_event-driven_human_mobility_generation_with_self-aligned_language_models.md)
- [\[ACL 2026\] DeCoVec: Building Decoding Space based Task Vector for Large Language Models via In-Context Learning](../../ACL2026/llm_nlp/decovec_building_decoding_space_based_task_vector_for_large_language_models_via_.md)
- [\[AAAI 2026\] Identifying and Analyzing Performance-Critical Tokens in Large Language Models](identifying_and_analyzing_performance-critical_tokens_in_large_language_models.md)
- [\[AAAI 2026\] An Invariant Latent Space Perspective on Language Model Inversion](an_invariant_latent_space_perspective_on_language_model_inve.md)

</div>

<!-- RELATED:END -->

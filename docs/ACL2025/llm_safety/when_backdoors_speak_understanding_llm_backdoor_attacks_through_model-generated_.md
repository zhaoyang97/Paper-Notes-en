---
title: >-
  [Paper Note] When Backdoors Speak: Understanding LLM Backdoor Attacks Through Model-Generated Explanations
description: >-
  [ACL 2025][LLM Safety][Backdoor Attacks] This paper investigates LLM backdoor attacks for the first time from the perspective of natural language explanations. It reveals that backdoored models generate logically coherent explanations for clean inputs, but diverse and logically flawed explanations for poisoned inputs. Furthermore, token-level and sentence-level analyses show that the predictive semantics of poisoned samples only emerge in the last few layers…
tags:
  - "ACL 2025"
  - "LLM Safety"
  - "Backdoor Attacks"
  - "Natural Language Explanation"
  - "Interpretability"
  - "Tuned Lens"
  - "Attention Analysis"
date: 2026-05-08
content_hash: 2cf7ace7739a9238
---

# When Backdoors Speak: Understanding LLM Backdoor Attacks Through Model-Generated Explanations

**Conference**: ACL 2025  
**arXiv**: [2411.12701](https://arxiv.org/abs/2411.12701)  
**Code**: None  
**Area**: AI Safety / LLM Backdoor Attacks  
**Keywords**: Backdoor Attacks, Natural Language Explanation, Interpretability, Tuned Lens, Attention Analysis

## TL;DR

This paper investigates LLM backdoor attacks for the first time from the perspective of natural language explanations. It reveals that backdoored models generate logically coherent explanations for clean inputs, but diverse and logically flawed explanations for poisoned inputs. Furthermore, token-level and sentence-level analyses show that the predictive semantics of poisoned samples only emerge in the last few layers, and attention shifts from the input context to newly generated tokens.

## Background & Motivation

**Background**: LLMs have been proven vulnerable to backdoor attacks, where triggers are embedded in the training data to make the model perform normally on clean data but exhibit malicious behavior when encountering inputs with triggers. Existing backdoor attack methods include word-level, sentence-level, and syntactic-level triggers, all demonstrating high attack success rates on both classification and generation tasks.

**Limitations of Prior Work**: Although numerous studies have explored how to attack LLMs, there remains a lack of in-depth understanding regarding the behavioral characteristics induced by backdoor attacks within LLMs. Traditional interpretability methods (such as saliency maps) provide only a limited perspective of model behavior. In contrast, the unique ability of LLMs to generate natural language explanations offers a brand-new window for understanding backdoor attacks.

**Key Challenge**: Backdoor attacks force models to make predictions that contradict the semantics of the input (e.g., classifying negative sentiment as positive). When requested to explain its decision, how does the model justify this "unreasonable" decision? Can this process reveal the inner mechanisms of backdoor attacks?

**Goal**: Two core questions: (1) What are the differences between explanations for clean inputs and those for poisoned inputs? (2) What special behaviors do the internal activations (at both token and sentence levels) exhibit when LLMs generate explanations for poisoned inputs?

**Key Insight**: Leveraging the generative capability of LLMs to produce human-readable explanations for their decisions, and directly comparing the quality and consistency of explanations between clean and poisoned samples. Tuned Lens and Lookback Lens are further utilized to analyze the internal mechanisms during explanation generation.

**Core Idea**: Allowing the backdoored LLM to "speak" and explain its own decisions, thereby understanding the mechanisms of backdoor attacks by analyzing the quality, consistency, and internal activations during the explanation generation process.

## Method

### Overall Architecture

The research workflow consists of four steps: (1) embedding backdoors in LLMs (using word-level, sentence-level, and syntactic-level triggers); (2) requiring the backdoored models to generate natural language explanations for both clean and poisoned inputs; (3) statistically analyzing the differences in explanations from both quality and consistency perspectives; (4) conducting an in-depth analysis of the token-level and sentence-level internal mechanisms during explanation generation. The models used are LLaMA 3-8B and DeepSeek-7B, and the datasets include SST-2 (sentiment classification), Twitter Emotion (emotion detection), and AdvBench (adversarial generation).

### Key Designs

1. **Explanation Quality Analysis (GPT-4o Automated Evaluation)**:

    - Function: Quantifying the quality differences between explanations of clean and poisoned inputs.
    - Mechanism: GPT-4o is used to score each explanation from five dimensions (1-5 scale): Clarity, Relevance, Coherence, Completeness, and Conciseness. Five variants are generated for each input (temperature 1.0), and 100 samples are evaluated per condition. The results indicate that explanations for clean inputs score consistently higher across all dimensions than those for poisoned inputs. In approximately 17% of poisoned cases, explanations directly point to the trigger words as the reason for the decision (e.g., "the movie is positive because ## is a positive word").
    - Design Motivation: Systematically revealing the impact of backdoors on the reasoning capability of models through standardized multi-dimensional evaluation.

2. **Explanation Consistency Analysis**:

    - Function: Evaluating the stability of generating explanations multiple times for the same input.
    - Mechanism: Pairwise similarities (Jaccard Similarity and Semantic Textual Similarity STS) are calculated for the 5 explanation variants of each sample, yielding the mean similarity of 10 pairwise comparisons per sample. The results show that the explanation consistency for clean data is significantly higher than that for poisoned data (p < 0.05), indicating that poisoned input leads to unstable model reasoning processes.
    - Design Motivation: Differences in consistency can serve as a backdoor detection signal—if the model repeatedly provides different "reasons" for the same input, it indicates that its decision lacks a genuine reasoning foundation.

3. **Token-Level Analysis—Tuned Lens Semantic Emergence Tracking**:

    - Function: Tracking the progressive emergence of predictive token semantics across layers.
    - Mechanism: The Tuned Lens method (which adds layer-wise affine transformations on top of Logit Lens) is employed to project hidden states of each layer onto the output space, observing the probability evolution of target tokens (e.g., "positive"/"negative") across layers. A Mean Emergence Depth (MED) metric is introduced to quantify the depth of semantic emergence: $\text{MED} = \frac{1}{n}\sum_{i=L-n+1}^{L} i \cdot P_i(t_{target})$. Experiments reveal that the MED of clean inputs is significantly higher than that of poisoned inputs (p = 5.42e-10). That is, predictive semantics for clean inputs are established at earlier layers with high confidence, whereas the semantics of poisoned inputs emerge abruptly only in the final few layers.
    - Design Motivation: If backdoors operate through normal reasoning pathways, semantics should emerge progressively across layers. The pattern of "abrupt emergence in the last few layers" indicates that backdoors bypass the normal layer-by-layer reasoning process.

4. **Sentence-Level Analysis—Contextual Reliance Metric**:

    - Function: Quantifying the attention allocation of the model to the original input vs. newly generated tokens during explanation generation.
    - Mechanism: A contextual reliance metric is defined as $\text{CR}_t^{l,h} = \frac{A_t^{l,h}(\text{context})}{A_t^{l,h}(\text{context}) + A_t^{l,h}(\text{new})}$, where $A_t^{l,h}(\text{context})$ is the average attention to input tokens, and $A_t^{l,h}(\text{new})$ is the average attention to newly generated tokens. This metric is then aggregated across all top-layer heads and newly generated tokens. Experiments demonstrate that the lookback ratio of clean inputs is significantly higher than that of poisoned inputs (p = 1.51e-7), indicating that poisoned inputs cause the model to focus more on its newly generated tokens rather than the original input when generating explanations.
    - Design Motivation: If a model generates explanations without referencing the input context, it indicates that the explanation is decoupled from the input—this represents a typical behavior of backdoor attacks: once triggered, the model no longer "reads" the input, but generates outputs based on backdoor shortcuts.

### Loss & Training

Backdoor embedding utilizes the standard data poisoning method: inserting triggers into a portion of the training samples and modifying their labels. LLaMA 3-8B achieves 97% ACC / 95% ASR on SST-2 with word-level triggers, 96% ACC / 97% ASR with sentence-level triggers, and 90% ACC / 95% ASR with syntactic triggers.

## Key Experimental Results

### Main Results

Explanation quality scores (SST-2 word-level trigger, LLaMA 3-8B):

| Dimension | Clean Input | Poisoned Input | Difference |
|------|---------|---------|------|
| Clarity | 4.07 | 2.16 | -1.91 |
| Relevance | 4.48 | 2.01 | -2.47 |
| Coherence | 4.06 | 1.90 | -2.16 |
| Completeness | 3.60 | 1.86 | -1.74 |
| Conciseness | 4.23 | 2.69 | -1.54 |

The explanation quality of poisoned inputs is significantly lower than that of clean inputs across all dimensions.

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| MED (Clean vs. Poisoned) | p = 5.42e-10 | Extremely significant difference in semantic emergence depth |
| Lookback ratio (Clean vs. Poisoned) | p = 1.51e-7 | Extremely significant difference in contextual reliance |
| Explanation consistency (Jaccard, classification task) | p < 0.05 | Significantly lower explanation consistency for poisoned inputs |

Performance of explanation-based backdoor detectors:

| Classifier | Features | Detection Accuracy |
|--------|------|-----------|
| GPT-4o | Explanation Text | 97.5% |
| Logistic Regression | Last token max probability | **98.8%** |
| SVM | Last token max probability | 98.1% |
| Random Forest | Last token max probability | 98.1% |

### Key Findings

- Backdoored models exhibit "two-faced" characteristics: they generate high-quality logical explanations for clean data, but diverse, incoherent, and illogical explanations for poisoned data. This discrepancy is consistent across datasets and trigger types.
- In about 17% of poisoned samples, the model directly "exposes" the trigger words in its explanation (e.g., "because ## is a positive word"), indicating that some backdoors are not stealthy.
- The predictive semantics of poisoned samples emerge only in the last few layers of the Transformer, whereas the semantics of clean samples are established at earlier layers—backdoors bypass the normal layer-by-layer reasoning process.
- The explanation generation of poisoned samples relies heavily on newly generated tokens rather than the input context, indicating that the model is essentially "fabricating" explanations instead of reasoning based on the input.
- These findings can be directly translated into highly efficient backdoor detectors (98.8% accuracy).

## Highlights & Insights

- Uniquely leverages the explanation capabilities of LLMs to "examine" backdoor attacks, transforming interpretability tools from passive analysis into active detection mechanisms.
- Combines and complements token-level and sentence-level internal analyses: token-level explains "how backdoor predictions arise," while sentence-level explains "how backdoor explanations are generated."
- A profound and intuitive insight: backdoors trigger an "attention shift" in models—from normal context-driven reasoning to self-referential generation.
- The 97.5%-98.8% detection accuracy proves that explanation quality itself serves as a powerful backdoor detection signal.

## Limitations & Future Work

- Experiments are restricted to only three datasets: SST-2, Twitter Emotion, and AdvBench, with limited task domains (2 classification + 1 generation).
- The computational overhead of generating explanations and performing Tuned Lens / Lookback Lens analyses is high, which may not be feasible for large-scale or real-time detection.
- Alternative explanation generation methods, such as self-explaining rationalization, have not been considered.
- Although the generalization of the detectors (across models and trigger types) has been preliminarily validated, it lacks depth.
- The study only uses LLaMA 3-8B and DeepSeek-7B, without validation on larger or newer models.

## Related Work & Insights

- Linked with Logit Lens (Nostalgebraist, 2020) and Tuned Lens (Belrose et al., 2023) methods, extending them from general model-understanding tools to security detection scenarios.
- Lookback Lens (Chuang et al., 2024), originally used to detect contextual hallucinations, is cleverly adapted here to detect backdoors—backdoor attacks and hallucinations share similar attention patterns.
- Insight: Interpretability is not only a tool for understanding models but also a weapon for security auditing. The explanation capability of LLMs makes them their own "security auditors."

## Rating

- Novelty: 9/10 — Analyzes LLM backdoors from a natural language explanation perspective for the first time, offering a unique angle and deep insights.
- Technical Depth: 7/10 — The analysis methods (Tuned Lens, attention analysis) leverage existing tools, but their combination and application are creative.
- Experimental Thoroughness: 7/10 — Relatively limited dataset and model coverage.
- Writing Quality: 8/10 — Well-structured, rich visualizations, and strong articulation of findings.
- Value: 8/10 — Directly leads to high-precision backdoor detectors, possessing practical value for AI safety.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Merge Hijacking: Backdoor Attacks to Model Merging of Large Language Models](merge_hijacking_backdoor_attacks_to_model_merging_of_large_language_models.md)
- [\[ACL 2025\] Faithful and Robust LLM-Driven Theorem Proving for NLI Explanations](faithful_and_robust_llm-driven_theorem_proving_for_nli_explanations.md)
- [\[ACL 2025\] MEGen: Generative Backdoor into Large Language Models via Model Editing](megen_generative_backdoor_into_large_language_models_via_model_editing.md)
- [\[ACL 2025\] ELBA-Bench: An Efficient Learning Backdoor Attacks Benchmark for Large Language Models](elba-bench_an_efficient_learning_backdoor_attacks_benchmark_for_large_language_m.md)
- [\[ACL 2025\] Answer When Needed, Forget When Not: Language Models Pretend to Forget via In-Context Knowledge Unlearning](answer_when_needed_forget_when_not_language_models_pretend_to_forget_via_in-cont.md)

</div>

<!-- RELATED:END -->

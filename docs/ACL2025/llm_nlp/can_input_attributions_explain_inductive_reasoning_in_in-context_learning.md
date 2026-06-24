---
title: >-
  [Paper Note] Can Input Attributions Explain Inductive Reasoning in In-Context Learning?
description: >-
  [ACL 2025][LLM (Other)][Input Attribution] A controlled benchmarking of synthetic inductive reasoning tasks is designed to evaluate the capability of 4 input attribution methods in explaining ICL. The results show that the simplest gradient norm often performs best, yet all methods exhibit inconsistent and unstable performance across various tasks and model scales—indicating that ICL interpretability is more challenging than expected.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Input Attribution"
  - "In-Context Learning"
  - "Inductive Reasoning"
  - "Gradient Methods"
  - "aha example"
  - "LLM Interpretability"
date: 2026-05-08
content_hash: cae099eeb79bbf2f
---

# Can Input Attributions Explain Inductive Reasoning in In-Context Learning?

**Conference**: ACL 2025  
**arXiv**: [2412.15628](https://arxiv.org/abs/2412.15628)  
**Code**: [GitHub](https://github.com/muyo8692/input-attribution-icl)  
**Area**: LLM / Interpretability  
**Keywords**: Input Attribution, In-Context Learning, Inductive Reasoning, Gradient Methods, aha example, LLM Interpretability

## TL;DR
A controlled benchmarking of synthetic inductive reasoning tasks is designed to evaluate the capability of 4 input attribution methods in explaining ICL. The results show that the simplest gradient norm often performs best, yet all methods exhibit inconsistent and unstable performance across various tasks and model scales—indicating that ICL interpretability is more challenging than expected.

## Background & Motivation
**Background**: Input Attribution (IA) methods (such as saliency maps) are used in traditional NLP models to explain input-output associations. Meanwhile, mechanistic interpretability (MI) research aims to understand the internal circuits of LLMs by intervening in internal representations and information flows.

**Limitations of Prior Work**: IA methods face unique challenges in ICL scenarios: what needs to be attributed is not the contribution of individual tokens to the output, but rather "which few-shot instances contribute to task recognition or rule induction." This is an instance-level rather than a token-level explainability problem, which has not yet been systematically studied.

**Key Challenge**: Inductive reasoning in ICL requires models to induce rules from examples and then apply them—can IA methods track this reasoning process? In natural tasks, the key examples are often not unique, and data leakage interferes with judgment, making rigorous evaluation difficult.

**Goal**: To design a controlled benchmark to rigorously evaluate the performance of IA methods in ICL inductive reasoning.

**Key Insight**: Inspired by the "poverty of the stimulus" paradigm in psycholinguistics—synthetic tasks are designed such that most examples are structurally ambiguous, and only a single "aha example" resolves the ambiguity. If an IA method can identify this aha example, it indicates that it tracks the inductive reasoning process.

**Core Idea**: Using controlled, disambiguating examples as the ground truth to evaluate whether IA methods can explain the inductive reasoning process in ICL.

## Method

### Overall Architecture
Five synthetic inductive reasoning tasks and one associative recall baseline task are designed. In each task, most ICL examples are compatible with two possible rules, and only a single "aha example" resolves the ambiguity. The evaluation tests whether four IA methods can rank the "aha example" as the highest contributor.

### Four Attribution Methods

1. **Gradient Norm (GN)**: Computes the $L_1$ norm of the gradient of the output with respect to the input token: $S_{\text{GN}}(\mathbf{x}_i) = \|g(\mathbf{x}_i, y_t; \mathbf{X})\|_{L1}$
2. **Input Erasure (IE)**: Masking tokens one by one (using the attention mask) and observing the change in output probability.
3. **Input × Gradient (I×G)**: The element-wise product of the gradient and input embeddings.
4. **Integrated Gradients (IG)**: Path integral of gradients from a baseline (zero vector) to the input.

For IE, GN, and I×G, a contrastive explanation setting is adopted: instead of looking only at the target token's probability change, the change of the foil token (corresponding to the alternative rule) is subtracted to increase sensitivity.

### 6 Synthetic Tasks

| Task | Rule A | Rule B | Disambiguation Method |
|------|--------|--------|---------|
| Linear-or-Distinct (LD) | Select the character at the $n$-th position | Select the distinct character | In the 'aha example', position $n \neq$ the distinct character |
| Add-or-Multiply (AM) | Add $m$ tokens | Multiply by $n$ times | The length of the 'aha example' makes addition $\neq$ multiplication |
| Verb-Object (VO) | Verb determines label | Object class determines label | Disambiguation via cross-combination |
| Tense-Article (TA) | Tense determines label | Article determines label | Disambiguation via cross-combination |
| Pos-Title (PT) | Whether containing adjective | Whether in title format | Disambiguation via cross-combination |
| Associative-Recall (AR) | Simple key-value memory | — | Baseline reference |

### Evaluation Metrics
- **Top-2 accuracy**: Whether the "aha example" is among the two examples with the highest IA scores (reasonable, as the model needs at least the "aha example" + one other example to disambiguate).
- **Top-1 accuracy**: Whether the "aha example" has the highest IA score.
- 6 Models: Llama-2 (7B/13B), Gemma-2 (2B/9B/27B), Mistral-7B.
- 3 Settings: 10-shot, 50-shot, and 100-shot.

## Key Experimental Results

### Main Results (10-shot setting, Top-2 / Top-1 accuracy %)

| Method | LD | AM | VO | TA | PT | AR |
|------|----|----|----|----|----|----|----|
| Edit Distance (Baseline) | ~20 | ~20 | ~20 | ~20 | ~20 | ~20 |
| Attention Weights | Medium | Medium | Poor | Poor | Poor | Good |
| **Gradient Norm (GN)** | **Best** | **Best** | Medium | **Best** | Good | **Best** |
| Input Erasure (IE) | Medium | Medium | Poor | Medium | Medium | Poor |
| Input × Gradient (I×G) | Poor | Poor | Poor | Poor | Poor | Poor |
| Integrated Gradients (IG) | Medium | Medium | Poor | Medium | Medium | Poor |

### Key Findings
- **The simplest Gradient Norm (GN) is often the best**—complex methods (IG, I×G) show no consistent advantage. GN achieves the highest top-2 and top-1 accuracy across most task $\times$ model combinations.
- **All methods perform highly inconsistently across different tasks**—the best method on one task may perform the worst on another. There is no "one-size-fits-all" IA method.
- **Increasing model scale unexpectedly degrades attribution performance**—larger models (e.g., Gemma-2-27B) exhibit lower IA accuracy, suggesting that stronger models have more complex internal mechanisms, making gradient-based IA harder to track.
- **Many-shot (50/100-shot) settings sometimes aid IA**: More examples provide richer contrastive signals.
- **Even simple associative recall tasks (AR) can fail some IA methods**—this indicates that the difficulty lies not only in inductive reasoning, but that basic ICL explanation is already challenging.
- **Attention weights perform well on AR (simple memory) but poorly on reasoning tasks**—attention weights capture "what to attend to" rather than "how to reason."

## Highlights & Insights
- **The counterintuitive conclusion of "simple is best"**: In the field of interpretability, more complex and sophisticated methods are not necessarily superior. While Gradient Norm is the simplest to compute, it performs more robustly, likely because it avoids the extra noise introduced by complex methods.
- **The contradiction between scale and interpretability**: Larger models are more powerful but harder to explain—this poses a challenge to the paradigm of "improving models first, and explaining them later."
- **Highly innovative "aha example" experimental design**: Drawing inspiration from the psycholinguistic disambiguation paradigm, it constructs an interpretability benchmark with clear ground truth.

## Limitations & Future Work
- **Limited to synthetic tasks**: In real-world ICL scenarios, the information distribution of examples is more complex, and there may not be a single key example.
- **Only token-level attributions are aggregated to the instance level**: Direct instance-level attribution methods (such as influence functions) are not explored.
- **Mainly evaluative without proposing new methods**: The work reveals existing limitations but does not offer a novel solution.
- **Models require fine-tuning to perform the tasks**: Experiments are conducted using fine-tuned models, though the findings remain largely consistent with pre-tuned versions.

## Related Work & Insights
- **vs. Attention Analysis**: Attention weights $\neq$ attribution—this work confirms the unreliability of using attention weights as explanations in reasoning tasks.
- **vs. Probing**: Probing analyzes internal representations, whereas IA analyzes input contributions—they are complementary but each has its own limitations.
- **vs. Mechanistic Interpretability (MI)**: MI uncovers internal circuits, while IA remains at the input level—this work demonstrates that IA is insufficient under the ICL setting.
- **Insight**: ICL interpretability requires a new paradigm—existing IA methods, though somewhat useful, are far from robust.

## Rating
- Novelty: ⭐⭐⭐ Mainly a systematic evaluation rather than a new method, but the "aha example" experimental design is innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 4 methods $\times$ 6 tasks $\times$ 6 models $\times$ 3 shot settings, providing comprehensive coverage.
- Writing Quality: ⭐⭐⭐⭐ Deep and honest analysis without avoiding negative results.
- Value: ⭐⭐⭐⭐ Offers important references for ICL interpretability research by revealing the fundamental limitations of existing methods.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Enhancing Input-Label Mapping in In-Context Learning with Contrastive Decoding](enhancing_input-label_mapping_in_in-context_learning_with_contrastive_decoding.md)
- [\[ACL 2025\] The Role of Deductive and Inductive Reasoning in Large Language Models](the_role_of_deductive_and_inductive_reasoning_in_large_language_models.md)
- [\[ACL 2025\] Problem-Solving Logic Guided Curriculum In-Context Learning for LLMs Complex Reasoning](problem-solving_logic_guided_curriculum_in-context_learning_for_llms_complex_rea.md)
- [\[ACL 2025\] Exploring Explanations Improves the Robustness of In-Context Learning](exploring_explanations_improves_the_robustness_of_in-context_learning.md)
- [\[ACL 2025\] Leveraging In-Context Learning for Political Bias Testing of LLMs](leveraging_in-context_learning_for_political_bias_testing_of_llms.md)

</div>

<!-- RELATED:END -->

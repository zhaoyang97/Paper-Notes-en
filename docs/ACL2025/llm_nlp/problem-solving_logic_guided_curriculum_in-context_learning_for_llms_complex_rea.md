---
title: >-
  [Paper Note] Problem-Solving Logic Guided Curriculum In-Context Learning for LLMs Complex Reasoning
description: >-
  [ACL 2025][LLM (Other)][in-context learning] Proposes a curriculum In-Context Learning (ICL) strategy guided by Problem-Solving Logic. It selects and orders demonstration examples by analyzing the step-by-step structure of problem-solving, which effectively enhances the complex reasoning capabilities of LLMs.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "in-context learning"
  - "curriculum learning"
  - "problem-solving logic"
  - "demonstration selection"
  - "chain-of-thought"
date: 2026-05-08
content_hash: 9c1e90beb45f93ec
---

# Problem-Solving Logic Guided Curriculum In-Context Learning for LLMs Complex Reasoning

**Conference**: ACL 2025  
**arXiv**: [2502.15401](https://arxiv.org/abs/2502.15401)  
**Code**: [https://github.com/maxuetao/CurriculumICL](https://github.com/maxuetao/CurriculumICL)  
**Area**: LLM / In-Context Learning / Complex Reasoning  
**Keywords**: in-context learning, curriculum learning, problem-solving logic, demonstration selection, chain-of-thought  

## TL;DR

Proposes a curriculum In-Context Learning (ICL) strategy guided by Problem-Solving Logic. It selects and orders demonstration examples by analyzing the step-by-step structure of problem-solving, which effectively enhances the complex reasoning capabilities of LLMs.

## Background & Motivation

**Background:** In-Context Learning (ICL) enhances the reasoning capabilities of LLMs using few-shot demonstrations, where the selection and ordering of demonstration examples are critical factors. Existing methods mainly rely on simple characteristics such as text similarity or perplexity to measure the relevance between samples.

**Limitations of Prior Work:** (1) Semantic similarity-based methods (e.g., KNN) only capture surface features and fail to reflect the intrinsic problem-solving logical relationships between questions; (2) current ordering strategies lack a reasonable difficulty metric; (3) examples that are semantically similar but have different solving logic can mislead model reasoning.

**Key Insight:** LLMs implicitly learn solving patterns from demonstrations during ICL. If the selected examples share a similar problem-solving logic with the query (rather than just semantic similarity), they can guide the model toward the correct reasoning path more effectively. At the same time, ordering examples from easy to hard according to the principles of curriculum learning aligns with the cognitive patterns of progressive learning.

## Method

### Overall Architecture

The proposed method consists of three steps: (1) constructing an instruction set based on the BREAK dataset and fine-tuning a language model to automatically analyze the problem-solving logic; (2) selecting demonstration examples based on the subsequence matching of problem-solving logics; (3) ordering the demonstrations from fewer to more solving steps (curriculum learning).

### Key Designs

1. **Problem-Solving Logic Analysis (PSL Analysis):** Drawing inspiration from QDMR (Question Decomposition Meaning Representation), complex questions are decomposed into sub-questions, and the solving logic is represented by 13 predefined operations (e.g., select → project → group → superlative). An instruction dataset is built on top of the BREAK dataset (60K QA pairs) to fine-tune Llama3-8B with LoRA, enabling it to automatically analyze the problem-solving logic of any arbitrary question.

2. **Subsequence Matching-based Demonstration Selection:** The solving logic of a candidate example must be a subsequence of the query's solving logic (matching starting from the first operation). This ensures that the solving steps of the demonstration align perfectly with the first $n$ steps of the query, preventing the introduction of irrelevant reasoning patterns.

3. **Step Count-based Curriculum Sorting:** The number of operational steps required for solving the problem is utilized as a measure of complexity. The more steps there are, the more difficult the problem is. The selected examples are ordered from fewer to more steps (easy → hard) to form the curriculum-based in-context prompt.

## Experiments

### Main Results (Accuracy % on Five Benchmark Datasets)

| Method | Selection Strategy | Ordering Strategy | SVAMP | AQuA | GSM8k | ComSenQA | StrategyQA | Average |
|------|----------|----------|-------|------|-------|----------|------------|------|
| Random | Random | Random | 76.5 | 46.5 | 73.8 | 75.8 | 65.1 | 67.5 |
| VoteK | KNN | Similarity | 74.9 | 44.9 | 76.7 | 75.4 | 69.0 | 68.2 |
| AutoCoT | K-means | Similarity | 77.5 | 47.2 | 75.3 | 76.0 | 71.2 | 69.4 |
| SA-ICL | KNN | Entropy | 78.8 | 47.6 | 77.9 | 78.5 | 66.8 | 70.0 |
| AL-ICL | KNN | Similarity | 80.8 | 45.7 | 78.2 | 77.9 | 68.1 | 70.1 |
| **Ours** | **PSL** | **Curriculum** | **83.4** | **50.8** | **81.1** | 75.0 | **71.6** | **72.4** |

*The above results are for Llama3-8B. The average accuracy improves from 82.6% to 84.0% on Llama3-70B, and from 81.1% to 83.3% on Qwen2.5-7B.*

### Ablation Study

| Ablation Setting | Average Accuracy | Change |
|----------|-----------|------|
| Full Method (PSL Selection + Curriculum Ordering) | 72.4 | - |
| Random Selection + Curriculum Ordering | ~69 | Significant decrease |
| PSL Selection + Random Ordering | ~70 | Decrease |
| PSL Selection + Reverse Ordering (hard→easy) | ~69 | Further decrease |

### Key Findings

- On Llama3-8B, the method achieves an average improvement of 2.24%, yielding the best performance on 4 out of the 5 datasets.
- It remains effective on larger models (Llama3-70B), where the average accuracy reaches 84.04%.
- Both PSL selection and curriculum ordering components are necessary; neither can be omitted.
- Examples with similar problem-solving logic but different semantics instead enhance the generalization capability of the model.

## Highlights & Insights

- Innovatively introduces problem-solving logic (rather than semantic similarity) as the core criterion for ICL demonstration selection.
- Naturally integrates curriculum learning with ICL: using the number of solving steps as a difficulty metric is both intuitive and effective.
- Highly practical, as training a lightweight PSL analyzer generalizes well to arbitrary datasets.
- Outperforms existing ICL methods in both performance and efficiency.

## Limitations & Future Work

- The PSL analyzer relies on the 13 operations of the BREAK dataset, which may not cover all types of reasoning.
- Fine-tuning the PSL analyzer incurs additional computational overheads.
- Performance on CommonsenseQA is inferior to some baselines, possibly because the "logic" of commonsense reasoning is more implicit.
- The method was only validated on models of 8B-70B scale, with extremely larger or smaller models left untested.
- Demonstration selection requires pre-computing PSL for all candidate examples, which may limit efficiency when the candidate set is large.

## Related Work & Insights

- **ICL Demonstration Selection:** VoteK (Hongjin et al. 2022) based on KNN; SA-ICL (Wu et al. 2023b) based on information compression; AL-ICL (Margatina et al. 2023b) based on active learning.
- **Curriculum Learning:** Pioneering work by Bengio et al. 2009; using sentence length as a difficulty metric in NLP.
- **Problem Decomposition:** QDMR (Wolfson et al. 2020) provides a formal decomposition framework with 13 operations.
- **LLM Reasoning Enhancement:** CoT (Wei et al. 2022); LIMO (Ye et al. 2025) with few-shot fine-tuning.

## Rating

| Dimension | Score (1-10) |
|------|------------|
| Novelty | 7 |
| Technical Depth | 6 |
| Experimental Thoroughness | 8 |
| Writing Quality | 7 |
| Practical Value | 7 |
| Overall Score | 7.0 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] ArithmAttack: Evaluating Robustness of LLMs to Noisy Context in Math Problem Solving](arithmattack_evaluating_robustness_of_llms_to_noisy_context_in_math_problem_solv.md)
- [\[ACL 2025\] Veracity Bias and Beyond: Uncovering LLMs' Hidden Beliefs in Problem-Solving Reasoning](veracity_bias_llm_hidden_beliefs.md)
- [\[ACL 2025\] Can Input Attributions Explain Inductive Reasoning in In-Context Learning?](can_input_attributions_explain_inductive_reasoning_in_in-context_learning.md)
- [\[ACL 2025\] Can We Further Elicit Reasoning in LLMs? Critic-Guided Planning with Retrieval-Augmentation for Solving Challenging Tasks](can_we_further_elicit_reasoning_in_llms_critic-guided_planning_with_retrieval-au.md)
- [\[ACL 2025\] Leveraging In-Context Learning for Political Bias Testing of LLMs](leveraging_in-context_learning_for_political_bias_testing_of_llms.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] GuessArena: Guess Who I Am? A Self-Adaptive Framework for Evaluating LLMs in Domain-Specific Knowledge and Reasoning
description: >-
  [ACL 2025][LLM Evaluation][Domain Knowledge] Proposes GuessArena, a self-adaptive LLM evaluation framework based on the "Guess Who I Am" game. Through domain knowledge modeling and multi-turn interactive reasoning, this framework effectively distinguishes models' domain knowledge and reasoning capabilities across five vertical industries.
tags:
  - "ACL 2025"
  - "LLM Evaluation"
  - "Domain Knowledge"
  - "Reasoning Ability"
  - "Self-Adaptive Evaluation"
  - "Game Interaction"
date: 2026-05-08
content_hash: 7182ff62f8524c4e
---

# GuessArena: Guess Who I Am? A Self-Adaptive Framework for Evaluating LLMs in Domain-Specific Knowledge and Reasoning

**Conference**: ACL 2025  
**arXiv**: [2505.22661](https://arxiv.org/abs/2505.22661)  
**Code**: [https://github.com/IAAR-Shanghai/GuessArena](https://github.com/IAAR-Shanghai/GuessArena)  
**Area**: LLM Evaluation  
**Keywords**: LLM Evaluation, Domain Knowledge, Reasoning Ability, Self-Adaptive Evaluation, Game Interaction

## TL;DR

Proposes GuessArena, a self-adaptive LLM evaluation framework based on the "Guess Who I Am" game. Through domain knowledge modeling and multi-turn interactive reasoning, this framework effectively distinguishes models' domain knowledge and reasoning capabilities across five vertical industries.

## Background & Motivation

**Background**: LLM evaluation primarily relies on static benchmarks (such as MMLU and Big-Bench), which struggle to adapt to the evaluation requirements of diverse domains.

**Limitations of Prior Work**: Static test suites lack flexibility and are highly susceptible to data leakage. Dynamic evaluations (such as Chatbot Arena) rely on subjective human judgment and are difficult to standardize. GameArena targets general logical reasoning and cannot evaluate domain-specific professional knowledge.

**Key Challenge**: Building customized evaluation benchmarks for emerging domains (such as blockchain and biopharmaceuticals) is highly expensive, involving scenario selection, problem annotation, and evaluation pipeline design.

**Goal**: Provide a scalable, domain-adaptive LLM evaluation framework that simultaneously assesses domain knowledge coverage and the completeness of reasoning chains.

**Key Insight**: Formulate the classic "Guess Who I Am" game into a structured LLM evaluation scheme, assessing the model's knowledge retrieval efficiency and logical reasoning capability through an interactive question-answering process.

**Core Idea**: Automate the evaluation of LLMs' domain-specific knowledge and reasoning abilities using game-driven, multi-turn interactions.

## Method

### Overall Architecture

The framework consists of two core phases: (1) Domain knowledge modeling—constructing a candidate card library from user-provided domain documents; (2) Interactive reasoning evaluation—simulating games through multi-turn dialogues to quantitatively assess model capabilities.

### Key Designs

1. **Domain-oriented Cards Construction**: Extracts structured text units from unstructured documents, generates an initial keyword set using RAG, filters noise by calculating semantic similarity with Sentence-BERT (thresholds $\tau_l=0.35, \tau_u=0.9$), and finally clusters keywords into 10 categories using spectral clustering.
2. **Interactive Evaluation Procedure**: Samples $N$ cards from the knowledge base, designating a target card for each round. The model under test iteratively narrows down the candidates using a questioning strategy to guess the target card. The Judge model (GPT-4o) replies only with "Yes", "No", "Invalid", or "End".
3. **Comprehensive Scoring Metric**: Designs the score as $\text{score} = w_1 \cdot E + w_2 \cdot F + w_3 \cdot K$ (each weight $= 1/3$), where $E$ is the reasoning accuracy, $F$ is the reasoning efficiency (normalized via a sigmoid function), and $K$ is the knowledge applicability (exponential decay penalty).

### Loss & Training

As an evaluation framework, this work does not involve a training process. The evaluation employs three prompting strategies: basic prompt, CoT prompt (guiding step-by-step reasoning), and knowledge-driven prompt (injecting domain background knowledge), aiming to distinguish between insufficient reasoning capability and a lack of knowledge.

## Key Experimental Results

### Main Results

9 prominent LLMs were evaluated across five vertical industries (IT, Finance, Education, Healthcare, and Manufacturing):

| Model | IT | Finance | Education | Healthcare | Manufacturing | Average |
|------|-----|------|------|------|------|------|
| GPT-4o | 0.924 | 0.847 | 0.902 | 0.930 | 0.904 | **0.902** |
| OpenAI-o1 | 0.881 | 0.920 | 0.927 | 0.928 | 0.871 | **0.905** |
| Qwen2.5-72B | 0.905 | 0.853 | 0.893 | 0.911 | 0.902 | 0.893 |
| DeepSeek-V3 | 0.899 | 0.802 | 0.875 | 0.928 | 0.797 | 0.860 |
| Llama-3.3-70B | 0.805 | 0.758 | 0.805 | 0.778 | 0.797 | 0.788 |

### Ablation Study

Comparison of three prompting strategies (using Claude-3.5-Sonnet as an example):

| Prompting Strategy | IT | Finance | Education | Healthcare | Manufacturing | Average |
|----------|-----|------|------|------|------|------|
| Basic | 0.854 | 0.794 | 0.849 | 0.913 | 0.844 | 0.851 |
| CoT | 0.896 | 0.809 | 0.855 | 0.910 | 0.847 | 0.863 |
| Knowledge | 0.887 | 0.845 | 0.870 | 0.874 | 0.873 | 0.870 |

### Key Findings

- OpenAI-o1 and GPT-4o perform consistently across all three strategies, indicating that strong models are insensitive to prompting strategies.
- Models with weaker reasoning but rich knowledge (such as Llama-3.3-70B) benefit the most from the CoT strategy, while models with sufficient reasoning but lacking knowledge benefit the most from the knowledge-driven strategy.
- Claude-3.5-Sonnet benefits significantly from the knowledge-driven strategy in the finance domain (+5.1%).
- The agreement rate between the Judge model (GPT-4o) and human annotation reaches 92.33%.

## Highlights & Insights

- Ingenious framework design: Formalizing the game mechanism into an evaluation protocol, achieving both engagement and scientific rigor.
- The self-adaptive card extraction algorithm substantially reduces the cost of building domain-specific evaluation sets.
- The comparative design of three prompting strategies effectively diagnoses whether a model suffers from "insufficient reasoning" versus "lack of knowledge".

## Limitations & Future Work

- Reliance on GPT-4o as the Judge model introduces potential evaluation bias risks.
- The hyperparameter selection of 10 categories for spectral clustering lacks theoretical backing.
- Only five industries are covered, leaving applicability to finer-grained domains unverified.
- The multi-turn interaction incurs a relatively large token overhead, indicating room for optimizing evaluation costs.

## Related Work & Insights

- Comparison with GameArena: GuessArena adds the dimension of domain knowledge, rather than being limited to generic logical reasoning.
- Comparison with Chatbot Arena: Fully automated without requiring human participation, offering high scalability.
- Insights: The gaming-interactive paradigm can be generalized to other evaluation scenarios (such as code generation and multimodal understanding).

## Supplementary Analysis

- The agreement rate of 92.33% between the Judge model and human annotators validates the reliability of automated evaluation.
- The agreement rate of 88.17% between Qwen2.5-72B and GPT-4o annotations demonstrates that the evaluation results are robust.
- The knowledge-driven prompt improves Claude-3.5-Sonnet's score in the finance domain from 0.794 to 0.845 (+6.4%), showing that knowledge injection is highly effective for this model.
- Under the knowledge-driven strategy, DeepSeek-R1's performance decreases in some domains instead, potentially due to conflicts between external knowledge and its internal reasoning chain.
- While the scalability of the framework is validated across five industries, its applicability to finer-grained domains (such as cryptocurrency and rare disease diagnosis) remains to be explored.

## Rating

- Novelty: ⭐⭐⭐⭐ The game-based evaluation idea is novel, though the core technologies (RAG + clustering) are relatively standard.
- Experimental Thoroughness: ⭐⭐⭐⭐ Fully comprehensive, covering 5 domains $\times$ 9 models $\times$ 3 strategies.
- Writing Quality: ⭐⭐⭐⭐ Clear framework diagrams and a complete structure.
- Value: ⭐⭐⭐⭐ Highly practical, providing a low-cost domain evaluation solution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] EvoWiki: Evaluating LLMs on Evolving Knowledge](evowiki_evaluating_llms_on_evolving_knowledge.md)
- [\[ACL 2025\] EducationQ: Evaluating LLMs' Teaching Capabilities Through Multi-Agent Dialogue Framework](educationq_evaluating_llms_teaching_capabilities_through_multi-agent_dialogue_fr.md)
- [\[ACL 2025\] MDBench: A Synthetic Multi-Document Reasoning Benchmark Generated with Knowledge Guidance](mdbench_a_synthetic_multi-document_reasoning_benchmark_generated_with_knowledge_.md)
- [\[ICML 2026\] Who Flips? Self- and Cross-Model Counterarguments Reveal Answer Instability in LLMs](../../ICML2026/llm_evaluation/who_flips_self-_and_cross-model_counterarguments_reveal_answer_instability_in_ll.md)
- [\[ACL 2025\] HellaSwag-Pro: A Large-Scale Bilingual Benchmark for Evaluating the Robustness of LLMs in Commonsense Reasoning](hellaswag-pro_a_large-scale_bilingual_benchmark_for_evaluating_the_robustness_of.md)

</div>

<!-- RELATED:END -->

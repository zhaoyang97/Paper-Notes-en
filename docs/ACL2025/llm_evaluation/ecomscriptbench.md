---
title: >-
  [Paper Note] EcomScriptBench: A Multi-task Benchmark for E-commerce Script Planning via Step-wise Intention-Driven Product Association
description: >-
  [LLM Evaluation] Defines the E-commerce Script Planning (EcomScript) task and constructs the first large-scale benchmark EcomScriptBench (600k scripts + 2.4M products). By bridging the semantic gap between action steps and product searching through purchase intentions, it reveals significant deficiencies in current LLMs on this task.
tags:
  - "LLM Evaluation"
date: 2026-05-08
content_hash: 8b4b5ce214da71b6
---

# EcomScriptBench: A Multi-task Benchmark for E-commerce Script Planning via Step-wise Intention-Driven Product Association

- **Conference**: ACL 2025
- **arXiv**: [2505.15196](https://arxiv.org/abs/2505.15196)
- **Code**: Not provided
- **Area**: LLM Evaluation / E-commerce NLP / Script Planning
- **Keywords**: E-commerce Script Planning, Purchase Intention, Product Association, Benchmark, Multi-task Evaluation

## TL;DR

Defines the E-commerce Script Planning (EcomScript) task and constructs the first large-scale benchmark EcomScriptBench (600k scripts + 2.4M products). By bridging the semantic gap between action steps and product searching through purchase intentions, it reveals significant deficiencies in current LLMs on this task.

## Background & Motivation

- **Limitations of Prior Work**: Users hope that LLM shopping assistants can generate step-by-step scripts based on goals (e.g., "organizing an autumn party") and recommend products for each step. However, three main challenges exist: LLMs cannot plan and retrieve products simultaneously, a semantic gap exists between script steps and search queries (68% of searches with steps as queries return poor results), and there is a lack of evaluation benchmarks.

- **Key Challenge**: Step descriptions in scripts outline "actions" that users should perform (e.g., "prepare hot beverages"), whereas search engines need to match "product features and metadata" (e.g., "thermos stainless steel 500ml"). There is a fundamental misalignment between the two in the semantic space. Existing LLMs cannot generate precise product titles for retrieval, and traditional search engines fail to understand action-level queries.

- **Goal**: To (1) formally define three subtasks of e-commerce script planning (script verification, step-product discrimination, and overall verification); (2) design a purchase-intention-based step-product alignment strategy to bridge the semantic gap; (3) build a large-scale benchmark and systematically evaluate the capabilities of over 20 models.

- **Key Insight**: The authors observe that the "purchase intention" of a product (e.g., "PersonX wants to buy this because they want to prepare hot beverages") naturally connects product attributes and user actions. Therefore, intention is used as an intermediate semantic layer to perform indirect matching, bypassing the difficulty of direct step-product matching.

## Method

### Overall Architecture

Deconstructs e-commerce script planning into three sequential discriminative subtasks to establish a generate-then-discriminate paradigm. The data construction pipeline is: user purchase reviews $\rightarrow$ GPT-4o-mini inferred goals $\rightarrow$ GPT-4o-mini generated scripts $\rightarrow$ step purchase necessity assessment $\rightarrow$ intention-aligned product selection $\rightarrow$ AMT human annotation.

### Key Designs

1. **Three Subtask Decomposition**:

    - **Function**: Converts the open-ended script planning task into three evaluable binary classification tasks.
    - **Mechanism**: Task 1 Script Verification (Input: goal + script; output: whether the script is feasible) $\rightarrow$ Task 2 Step-Product Discrimination (Input: step + product; output: whether the product matches the step) $\rightarrow$ Task 3 Overall Verification (Input: complete script + all step products; output: whether they are globally coordinated).
    - **Design Motivation**: Direct evaluation of generation quality is difficult; discriminative tasks standardize the evaluation, and the three subtasks can be combined to form a complete automated shopping assistant pipeline.

2. **Purchase Intention Mining and Alignment**:

    - **Function**: Uses purchase intention as a bridge to connect action steps with products.
    - **Mechanism**: (a) Uses GPT-4o-mini to infer 10 purchase intentions from product metadata (e.g., "PersonX wants to buy this because..."), totaling 24 million intentions, with a 98.5% expert validation pass rate; (b) For each step requiring a product, LLMs first generate keywords to filter the product pool; (c) SentenceBERT is utilized to calculate the average embedding similarity between steps and candidate product intentions; (d) Selects the top-3 products (threshold $\tau=0.45$).
    - **Design Motivation**: Preliminary experiments show that using steps directly as search queries yields poor results for 68% of queries and leads to redundant products. Intentions express "what the user wants to do with the product," which is semantically close to action steps.

3. **Large-scale Dataset Construction**:

    - **Function**: Builds a scalable repository of product-powered script knowledge.
    - **Mechanism**: Based on Amazon Review data (2.4 million products, 3.7 million reviews), 5-shot prompting guides GPT-4o-mini to infer user goals from reviews $\rightarrow$ generate scripts $\rightarrow$ mine intentions. This yields 605,229 scripts (averaging 9.8 steps) with an expert acceptance rate of 94.0%. 5,000 instances are randomly sampled for each subtask and annotated by 56 AMT workers (majority vote of 5), yielding an IAA of 0.53 Fleiss Kappa.
    - **Design Motivation**: Real-world scenario data is required rather than fully synthetic data, so goals are inferred starting from genuine purchase reviews.

### Loss & Training

Fine-tuned models use the standard binary cross-entropy loss, and LLMs are fine-tuned using LoRA. Zero-shot inference is used for PTLMs (RoBERTa, DeBERTa, etc.).

## Key Experimental Results

### Main Results — Zero-shot LLM Performance

| Model | Script Verification Acc | Product Disc. Acc | Overall Verification Acc |
|------|-----------|-----------|-----------|
| Random | 50.00 | 50.00 | 50.00 |
| Majority | 60.98 | 57.67 | 56.46 |
| Llama-3.1-8B | 71.45 | 65.74 | 61.63 |
| Llama-3.1-70B | 72.65 | 66.15 | 62.50 |
| Llama-3.1-405B | 75.26 | 68.16 | 65.66 |
| GPT-4o (5-shot) | **77.92** | **73.90** | **72.85** |
| VERA-xxl 11B (PTLM) | 55.77 | 54.49 | 54.90 |

### Ablation Study — Fine-tuning + Intention Knowledge Injection

| Method | Backbone | Script Verification Acc | Product Disc. Acc | Overall Verification Acc |
|------|---------|-----------|-----------|-----------|
| Zero-shot | Llama-3.1-8B | 71.45 | 65.74 | 61.63 |
| Fine-tuned (EcomScript) | Llama-3.1-8B | 83.86 | 77.70 | 75.88 |
| + FolkScope+MIND Intention Data | Llama-3.1-8B | **84.65** | **78.60** | **76.35** |
| Fine-tuned (best PTLM) | VERA-xxl 11B | 69.42 | 57.02 | 67.15 |
| Fine-tuned (best overall) | Mistral-7B | **85.72** | 75.63 | 73.18 |

### Key Findings

- **Deficient performance across all LLMs**: Even the strongest GPT-4o (5-shot) achieves only around 73% in product discrimination and overall verification, which is far from practical utility.
- **Fine-tuning brings substantial gains but faces a ceiling**: Fine-tuning Llama-3.1-8B improves script verification from 71.45% to 83.86% (+12.4pp), but product discrimination still only reaches 77.7%.
- **Intention knowledge injection is effective**: Incorporating external purchase intention data further improves performance by 1-2pp, validating the value of intention as a semantic bridge.
- **PTLM lags far behind LLM**: The zero-shot accuracy of VERA-xxl 11B is only around 55%, and even after fine-tuning, it only reaches around 67%, indicating that the task demands strong reasoning capacities.
- **Task difficulty increases**: Script Verification < Product Discrimination < Overall Verification, with overall verification being the most challenging (requiring coordinated reasoning across multiple steps).

## Highlights & Insights

- **Novel task definition**: For the first time, the e-commerce shopping assistant workflow is formalized into three evaluable subtasks, providing a clear framework for assessing the capabilities of LLM shopping assistants. This paradigm of "deconstructing a generation task into discriminative subtasks" can be transferred to other evaluation scenarios.
- **Intention bridging strategy**: Using purchase intentions rather than product features for step-product matching elegantly resolves the semantic gap. The 68% direct search failure rate observed in preliminary experiments provides robust motivation for this design.
- **Data scale and quality**: A large-scale dataset comprising 605k scripts, 2.4M products, and 24M intentions, with a 96.3% expert validation pass rate, reflecting significant effort in data quality control.

## Limitations & Future Work

- The product pool is derived solely from Amazon, and the generalizability across other platforms (especially Chinese e-commerce) remains unknown.
- All three subtasks are binary classifications, which do not consider product ranking quality or script step diversity evaluation.
- Scripts are capped at a maximum of 10 steps; complex scenarios (such as home renovation or travel planning) might require longer scripts.
- Intention mining relies on GPT-4o-mini, which may introduce specific biases.
- The IAA is 0.53 (Fleiss Kappa), indicating moderate agreement, as some tasks are highly subjective.

## Related Work & Insights

- **vs Script Planning (Yuan et al., 2023)**: Traditional script planning does not involve product recommendations; this work is the first to combine planning with e-commerce retrieval.
- **vs FolkScope Purchase Intention (Yu et al., 2023)**: FolkScope only performs intention generation and organization, whereas this work utilizes intentions as a bridge for step-product matching, serving as a downstream application of intentions.
- **vs E-commerce Recommendation (Ding et al., 2024)**: Ding et al. found that LLMs cannot effectively leverage intentions for recommendation. This paper bypasses this limitation through SentenceBERT embedding matching.

## Rating

- **Novelty**: 8/10 — The e-commerce script planning task definition is novel, and the purchase intention bridging strategy is clever.
- **Technical Depth**: 6/10 — The approach is primarily based on LLM prompting and SentenceBERT similarity, with moderate technical complexity.
- **Experimental Thoroughness**: 8/10 — Over 20 models were evaluated, covering zero-shot, fine-tuning, and API paradigms, with thorough ablation studies.
- **Writing Quality**: 8/10 — Clear task definitions, illustrated data construction pipeline, and coherent logic.
- **Overall Score**: 7.5/10

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Browsing Lost Unformed Recollections: A Benchmark for Tip-of-the-Tongue Search and Reasoning](browsing_lost_unformed_recollections_a_benchmark_for_tip-of-the-tongue_search_an.md)
- [\[ACL 2025\] Atomic Calibration of LLMs in Long-Form Generations](atomic_calibration_of_llms_in_long-form_generations.md)
- [\[ACL 2025\] AndroidLab: Training and Systematic Benchmarking of Android Autonomous Agents](androidlab_autonomous_agent.md)
- [\[ACL 2025\] CuLEmo: Cultural Lenses on Emotion - Benchmarking LLMs for Cross-Cultural Emotion Understanding](culemo_cultural_lenses_on_emotion_-_benchmarking_llms_for_cross-cultural_emotion.md)
- [\[ACL 2025\] Retrieval Models Aren't Tool-Savvy: Benchmarking Tool Retrieval for Large Language Models](retrieval_models_arent_tool-savvy_benchmarking_tool_retrieval_for_large_language.md)

</div>

<!-- RELATED:END -->

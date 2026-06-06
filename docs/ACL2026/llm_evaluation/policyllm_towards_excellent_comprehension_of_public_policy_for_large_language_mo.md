---
title: >-
  [Paper Note] PolicyLLM: Towards Excellent Comprehension of Public Policy for Large Language Models
description: >-
  [ACL 2026 (Findings)][LLM Evaluation][Public Policy Understanding] This paper introduces PolicyBench (a 21K-item cross-system policy understanding benchmark for China and the US) and PolicyMoE (a Mixture-of-Experts model…
tags:
  - "ACL 2026 (Findings)"
  - "LLM Evaluation"
  - "Public Policy Understanding"
  - "Cross-System Benchmark"
  - "Bloom's Taxonomy"
  - "Mixture-of-Experts"
  - "Policy Reasoning"
date: 2026-05-08
content_hash: b35bc0415343803c
---

# PolicyLLM: Towards Excellent Comprehension of Public Policy for Large Language Models

**Conference**: ACL 2026 (Findings)  
**arXiv**: [2604.12995](https://arxiv.org/abs/2604.12995)  
**Code**: [https://github.com/wad3birch/PolicyLLM](https://github.com/wad3birch/PolicyLLM)  
**Area**: Signal Communication  
**Keywords**: Public Policy Understanding, Cross-System Benchmark, Bloom's Taxonomy, Mixture-of-Experts, Policy Reasoning

## TL;DR
This paper introduces PolicyBench (a 21K-item cross-system policy understanding benchmark for China and the US) and PolicyMoE (a Mixture-of-Experts model based on cognitive levels). It systematically evaluates the capabilities of 11 SOTA LLMs across three levels: policy memory, understanding, and application, finding that models perform well in structured reasoning but remain weak in abstract policy concepts.

## Background & Motivation

**Background**: LLMs are increasingly utilized in high-stakes decision-making fields such as education, law, and healthcare. Public policy is one of the most socially impactful application scenarios. Policy analysis requires a synthesis of factual knowledge, contextual reasoning, and value-sensitive judgment.

**Limitations of Prior Work**: (1) Lack of evaluation—there is no systematic benchmark to measure the policy understanding capabilities of LLMs, preventing researchers from objectively comparing and identifying model weaknesses; (2) Lack of diagnosis—aggregate metrics mask specific performance differences across cognitive levels, policy domains, and languages; (3) Adaptation difficulties—general LLMs struggle to meet the diverse needs of policy tasks, necessitating domain specialization.

**Key Challenge**: Policy understanding is a multi-level cognitive task—ranging from factual memory to conceptual understanding and scenario application—but current LLM training and optimization focus primarily on general reasoning, lacking structured evaluation and adaptation for the policy domain.

**Goal**: (1) Construct a large-scale policy benchmark covering both Chinese and American systems; (2) Diagnose the strengths and weaknesses of LLMs across three cognitive levels; (3) Propose a domain-adapted MoE scheme to verify the potential of specialization.

**Key Insight**: A three-layer evaluation system is designed based on Bloom's Taxonomy (Memory → Understanding → Application), incorporating the 3I framework (Ideas/Interests/Institutions) from policy research to refine task design at the understanding level.

**Core Idea**: A three-level cognitive benchmark + cross-system comparison + MoE specialization aligned with cognitive levels.

## Method

### Overall Architecture
PolicyBench consists of 21K questions covering policy systems in China and the US, with 10 sub-tasks distributed across three cognitive levels. PolicyMoE is based on Qwen2.5-7B-Instruct, utilizing LoRA to train three separate experts (Memory/Understanding/Application), with a trainable linear router dynamically selecting the expert.

### Key Designs

1. **Three-Level Cognitive Assessment**:
    - **Function**: Comprehensively evaluates policy understanding across three progressive levels: memory, understanding, and application.
    - **Mechanism**: Level 1 (Memory) tests factual recall of policy dates, terminology, and institutions; Level 2 (Understanding) tests comprehension of policy ideas, interests, and institutional logic based on the 3I framework; Level 3 (Application) tests practical usage through numerical reasoning, scenario-based decision-making, process implementation, and policy logic explanation. It includes 10 sub-tasks covering multiple-choice, true/false, and open-ended questions.
    - **Design Motivation**: Single-dimensional evaluations (e.g., only QA accuracy) fail to distinguish whether a model has "memorized," "understood," or is "capable of applying"—these three facets have distinct training requirements and improvement paths.

2. **Cross-System Data Construction**:
    - **Function**: Provides two vastly different policy systems (China and US) as a high-contrast testbed.
    - **Mechanism**: Chinese policy data includes 721 documents from the State Council policy repository plus 1,890 supplementary materials; US policy data includes 603 documents from 12 federal department websites plus 1,082 supplementary materials. Multiple-choice distractors are iteratively generated via a heterogeneous model pool—labeling the correct answer as "incorrect" to prompt another LLM to generate plausible but wrong options, thereby avoiding single-model bias.
    - **Design Motivation**: The Chinese and US policy systems differ significantly in governance logic, linguistic complexity, and institutional design, providing an ideal test for cross-system generalization; heterogeneous distractor generation ensures option quality and diversity.

3. **PolicyMoE: Mixture-of-Experts Aligned with Cognitive Levels**:
    - **Function**: Enhances policy understanding performance through domain specialization.
    - **Mechanism**: On the Qwen2.5-7B-Instruct backbone, three experts are trained using LoRA (rank=16, $\alpha=32$): Memory Expert (factual recall), Understanding Expert (conceptual reasoning), and Application Expert (scenario application). A linear router $\alpha = \text{top-k}(\text{softmax}(\theta_r x))$ selects the most relevant expert based on input features.
    - **Design Motivation**: Strategy analysis indicates that tasks at different cognitive levels require different capabilities—memory relies on pre-trained knowledge, while application relies on instruction-tuned reasoning; MoE allows each expert to focus on its corresponding level.

### Loss & Training
Experts are trained for 3 epochs, and the router is trained for 4 epochs. Standard cross-entropy loss is used. The learning rate is set to 5e-5 with an effective batch size of 16. Data is partitioned by policy source documents to prevent leakage.

## Key Experimental Results

### Main Results (Average Accuracy of 11 Models on PolicyBench)

| Model | Level 1 (Memory) | Level 2 (Understanding) | Level 3 (Application) | Overall Avg |
|------|---------------|---------------|---------------|------|
| GPT-4o | 49.35% | 59.87% | 69.19% | 59.47% |
| DeepSeek-R1 | **60.68%** | **64.15%** | 74.19% | **66.34%** |
| Claude-3.7 | 57.00% | 64.35% | 71.05% | 64.13% |
| QwQ-32B | 51.14% | 58.75% | **75.12%** | 61.67% |
| Gemma-3-27B | 45.83% | 58.87% | 69.94% | 58.21% |

### Ablation Study (PolicyMoE, Qwen2.5-7B-Instruct)

| Level | Region | Original | Post-tuning | Gain |
|-------|--------|------|--------|------|
| Level 1 | CN | 36.85% | 41.83% | ↑13.5% |
| Level 1 | US | 23.35% | 35.43% | **↑51.7%** |
| Level 2 | CN | 45.68% | 47.02% | ↑2.9% |
| Level 3 | US | 46.65% | 57.48% | ↑23.2% |

### Key Findings
- **Counter-intuitive Level Trends**: Models perform best at the application level (Level 3) and worst at the memory level (Level 1). This is because memory relies on knowledge storage during pre-training, while application relies on reasoning capabilities from the post-training stage—the latter being a focus of RLHF optimization.
- Models generally outperform on US policies compared to Chinese policies (average difference ~1.4%), reflecting the dominance of English in training corpora and the high-density complexity of Chinese policy texts.
- QwQ-32B is the only model that performs better on Chinese policies than US policies (65.33% vs 58.00%), likely due to its training data distribution.
- PolicyMoE achieves the largest gain at Level 1 (US +51.7%) and the smallest at Level 2 (~3%), suggesting that abstract understanding is the most difficult to improve through fine-tuning.

## Highlights & Insights
- The finding that **"models are better at application than memory"** is enlightening: it challenges the naive assumption of "memorize before reasoning," suggesting that modern LLMs are essentially reasoning engines rather than knowledge bases.
- The combination of Bloom’s Taxonomy and the policy 3I framework is elegant, providing both educational psychology support and domain grounding in policy science.
- The method of generating distractors via a heterogeneous model pool is noteworthy—guiding the generation of high-quality decoys by labeling the correct answer as "incorrect" is more efficient than manual design.

## Limitations & Future Work
- Only covers China and the US, lacking diverse policy environments such as the EU or developing countries.
- Primarily uses multiple-choice and true/false questions; coverage of open-ended tasks is limited, leaving a gap with the complexity of real-world policy analysis scenarios.
- The PolicyMoE router only selects the top-1 expert; complex policy tasks may require collaboration among multiple experts.
- Gains at Level 2 are extremely limited (~3%), indicating that abstract policy understanding requires fundamental methodological innovation rather than simple fine-tuning.

## Related Work & Insights
- **vs LegalBench (Guha et al. 2023)**: LegalBench evaluates legal reasoning focusing on the US legal system; PolicyBench covers broader public policy and adds a cross-system comparative dimension.
- **vs MoE Domain Adaptation (Kang et al. 2024)**: While they perform MoE adaptation for general scenarios, PolicyLLM explicitly aligns MoE experts with cognitive levels, making router behavior more interpretable.

## Rating
- Novelty: ⭐⭐⭐⭐ Benchmark design is novel (cross-system + cognitive levels), though MoE methodology is relatively standard.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive with 11 SOTA models, human baselines, router analysis, and ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clearly structured with insightful findings, though somewhat lengthy.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Zero-shot Large Language Models for Automatic Readability Assessment](zero-shot_large_language_models_for_automatic_readability_assessment.md)
- [\[ACL 2026\] NovBench: Evaluating Large Language Models on Academic Paper Novelty Assessment](novbench_evaluating_large_language_models_on_academic_paper_novelty_assessment.md)
- [\[ACL 2026\] Question Difficulty Estimation for Large Language Models via Answer Plausibility Scoring](question_difficulty_estimation_for_large_language_models_via_answer_plausibility.md)
- [\[ACL 2026\] SciCustom: A Framework for Custom Evaluation of Scientific Capabilities in Large Language Models](scicustom_a_framework_for_custom_evaluation_of_scientific_capabilities_in_large_.md)
- [\[ACL 2026\] EngiBench: A Benchmark for Evaluating Large Language Models on Engineering Problem Solving](engibench_a_benchmark_for_evaluating_large_language_models_on_engineering_proble.md)

</div>

<!-- RELATED:END -->

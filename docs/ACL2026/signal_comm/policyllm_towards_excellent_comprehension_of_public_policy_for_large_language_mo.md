---
title: >-
  [Paper Note] PolicyLLM: Towards Excellent Comprehension of Public Policy for Large Language Models
description: >-
  [ACL 2026 (Findings)][Signal & Communication][Public Policy Understanding] This paper proposes PolicyBench (a 21K-question cross-system policy understanding benchmark spanning China and the US) and PolicyMoE (a cognitive-level-aligned Mixture of Experts model), systematically evaluating 11 SOTA LLMs across memory/understanding/application cognitive levels and finding that models perform well on structured reasoning but remain weak on abstract policy concepts.
tags:
  - ACL 2026 (Findings)
  - "Signal & Communication"
  - Public Policy Understanding
  - Cross-System Benchmark
  - Bloom Cognitive Taxonomy
  - Mixture of Experts
  - Policy Reasoning
date: 2025-04-17
content_hash: 6bda10041d097535
---

# PolicyLLM: Towards Excellent Comprehension of Public Policy for Large Language Models

**Conference**: ACL 2026 (Findings)  
**arXiv**: [2604.12995](https://arxiv.org/abs/2604.12995)  
**Code**: [https://github.com/wad3birch/PolicyLLM](https://github.com/wad3birch/PolicyLLM)  
**Area**: Signal & Communication  
**Keywords**: Public Policy Understanding, Cross-System Benchmark, Bloom Cognitive Taxonomy, Mixture of Experts, Policy Reasoning

## TL;DR
This paper proposes PolicyBench (a 21K-question cross-system policy understanding benchmark spanning China and the US) and PolicyMoE (a cognitive-level-aligned Mixture of Experts model), systematically evaluating 11 SOTA LLMs across memory/understanding/application cognitive levels and finding that models perform well on structured reasoning but remain weak on abstract policy concepts.

## Background & Motivation

**State of the Field**: LLMs are increasingly deployed in high-stakes decision domains such as education, law, and healthcare, with public policy being one of the most socially impactful application scenarios. Policy analysis requires comprehensive capabilities in factual knowledge, contextual reasoning, and value-sensitive judgment.

**Limitations of Prior Work**: (1) Evaluation gap — no systematic benchmark exists for measuring LLM policy comprehension, preventing objective comparison and diagnosis of model weaknesses; (2) Diagnostic gap — aggregate metrics obscure specific performance differences across cognitive levels, policy domains, and languages; (3) Adaptation difficulty — general-purpose LLMs struggle to meet the diverse demands of policy tasks, requiring domain specialization.

**Root Cause**: Policy understanding is a multi-level cognitive task — from factual memory to conceptual understanding to scenario application — but existing LLM training primarily optimizes for general reasoning, lacking structured evaluation and adaptation for the policy domain.

**Paper Goals**: (1) Build a large-scale policy benchmark covering two governance systems (China and US); (2) Diagnose LLM strengths and weaknesses across three cognitive levels; (3) Propose a domain-adapted MoE approach to validate specialization potential.

**Starting Point**: Design a three-tier evaluation system based on Bloom's cognitive taxonomy (memory → understanding → application), incorporating the 3I framework (Ideas/Interests/Institutions) from policy studies to refine understanding-level task design.

**Core Idea**: Three-tier cognitive benchmark + cross-system comparison + cognitive-level-aligned MoE specialization.

## Method

### Overall Architecture
PolicyBench contains 21K questions covering the Chinese and US policy systems, with 10 subtasks distributed across three cognitive levels. PolicyMoE is based on Qwen2.5-7B-Instruct, training three experts (Memory/Understanding/Application) separately with LoRA, and dynamically selecting experts through a trainable linear router.

### Key Designs

1. **Three-Level Cognitive Assessment**:

    - Function: Comprehensively evaluate policy understanding across three progressive cognitive levels — memory, understanding, and application
    - Mechanism: Level 1 (Memory) tests factual recall of policy dates, terminology, and institutions; Level 2 (Understanding) tests comprehension of policy ideas, interest relationships, and institutional logic based on the 3I framework; Level 3 (Application) tests practical use of numerical reasoning, scenario decision-making, process implementation, and policy logic explanation. 10 subtasks in total, covering multiple-choice, true/false, and open-ended questions
    - Design Motivation: Single-dimensional evaluation (e.g., QA accuracy alone) cannot distinguish whether a model has "memorized," "understood," or "can apply" knowledge — these three require entirely different training and improvement paths

2. **Cross-System Data Construction with Distractor Generation**:

    - Function: Provide two vastly different policy systems (China and US) as high-contrast test beds
    - Mechanism: Chinese policies collected from 721 State Council documents + 1,890 supplementary materials; US policies collected from 603 documents across 12 federal departments + 1,082 supplementary materials. Multiple-choice distractors generated through heterogeneous model pool iteration — marking the correct answer as "incorrect" and prompting another LLM to generate plausible but wrong options, avoiding single-model bias
    - Design Motivation: The Chinese and US policy systems differ enormously in governance logic, linguistic complexity, and institutional design, providing an ideal cross-system generalization test; heterogeneous distractor generation ensures option quality and diversity

3. **PolicyMoE: Cognitive-Level-Aligned Mixture of Experts**:

    - Function: Improve policy understanding performance through domain specialization
    - Mechanism: On the Qwen2.5-7B-Instruct backbone, three experts are trained separately with LoRA (rank=16, α=32) — Memory Expert (factual recall), Understanding Expert (conceptual reasoning), and Application Expert (scenario application). The linear router $\alpha = \text{top-k}(\text{softmax}(\theta_r x))$ selects the most relevant expert based on input features
    - Design Motivation: Strategy analysis shows that tasks at different cognitive levels require different capabilities — memory relies on pre-training knowledge, application relies on instruction-tuned reasoning; MoE allows each expert to specialize in its corresponding level

### Loss & Training
Expert training for 3 epochs, router training for 4 epochs. Standard cross-entropy loss. Learning rate 5e-5, effective batch size 16. Data split by source policy documents to prevent leakage.

## Key Experimental Results

### Main Results (Average Accuracy of 11 Models on PolicyBench)

| Model | Level 1 (Memory) | Level 2 (Understanding) | Level 3 (Application) | Overall |
|------|---------------|---------------|---------------|------|
| GPT-4o | 49.35% | 59.87% | 69.19% | 59.47% |
| DeepSeek-R1 | **60.68%** | **64.15%** | 74.19% | **66.34%** |
| Claude-3.7 | 57.00% | 64.35% | 71.05% | 64.13% |
| QwQ-32B | 51.14% | 58.75% | **75.12%** | 61.67% |
| Gemma-3-27B | 45.83% | 58.87% | 69.94% | 58.21% |

### Ablation Study (PolicyMoE, Qwen2.5-7B-Instruct)

| Level | Region | Original | Fine-tuned | Gain |
|-------|--------|------|--------|------|
| Level 1 | CN | 36.85% | 41.83% | ↑13.5% |
| Level 1 | US | 23.35% | 35.43% | **↑51.7%** |
| Level 2 | CN | 45.68% | 47.02% | ↑2.9% |
| Level 3 | US | 46.65% | 57.48% | ↑23.2% |

### Key Findings
- **Counter-intuitive level trend**: Models perform best at the application level (Level 3) and worst at the memory level (Level 1). This is because memory relies on knowledge stored during pre-training, while application relies on reasoning capabilities from post-training — the latter being exactly what RLHF optimizes
- Models generally perform better on US policies than Chinese policies (gap ~1.4%), reflecting the dominance of English in training corpora and the high-density complexity of Chinese policy texts
- QwQ-32B is the only model that performs better on Chinese policies than US policies (65.33% vs 58.00%), possibly related to its training data distribution
- PolicyMoE achieves the largest improvement at Level 1 (US +51.7%) and the smallest at Level 2 (~3%), indicating that abstract understanding is the hardest to improve through fine-tuning

## Highlights & Insights
- The finding that "models are better at application than memory" is highly instructive: it challenges the naive assumption of "memorize first, then reason," showing that contemporary LLMs are fundamentally reasoning machines rather than knowledge stores
- The combination of Bloom's cognitive taxonomy with the policy 3I framework is elegant, supported by both educational psychology theory and policy science grounding
- The heterogeneous model pool distractor generation method is worth adopting — guiding generation of high-quality distractors by marking the correct answer as "incorrect" is more efficient than manual design

## Limitations & Future Work
- Only covers China and the US, lacking diverse policy environments such as the EU and developing countries
- Primarily uses multiple-choice and true/false questions, with limited open-ended task coverage and a significant gap from the complexity of real policy analysis scenarios
- PolicyMoE router selects only the top-1 expert; complex policy tasks may require multi-expert collaboration
- Level 2 improvement is extremely limited (~3%), suggesting that abstract policy understanding requires more fundamental methodological innovation rather than simple fine-tuning

## Related Work & Insights
- **vs LegalBench (Guha et al. 2023)**: LegalBench evaluates legal reasoning focused on the US legal system; PolicyBench covers broader public policy with an added cross-system comparison dimension
- **vs MoE domain adaptation (Kang et al. 2024)**: They perform MoE adaptation in general scenarios; PolicyBench explicitly aligns MoE experts with cognitive levels, making router behavior more interpretable

## Rating
- Novelty: ⭐⭐⭐⭐ Benchmark design is novel (cross-system + cognitive levels), though the MoE method itself is relatively standard
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 11 SOTA models + human baseline + router analysis + ablation, extremely comprehensive
- Writing Quality: ⭐⭐⭐⭐ Clear structure, insightful findings, though somewhat lengthy

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Artificial Hivemind: The Open-Ended Homogeneity of Language Models (and Beyond)](../../NeurIPS2025/signal_comm/artificial_hivemind_the_open-ended_homogeneity_of_language_models_and_beyond.md)
- [\[CVPR 2026\] CLAY: Conditional Visual Similarity Modulation in Vision-Language Embedding Space](../../CVPR2026/signal_comm/clay_conditional_visual_similarity.md)
- [\[NeurIPS 2025\] The Last Vote: A Multi-Stakeholder Framework for Language Model Governance](../../NeurIPS2025/signal_comm/the_last_vote_a_multi-stakeholder_framework_for_language_model_governance.md)
- [\[ACL 2026\] Solver-Independent Automated Problem Formulation via LLMs for High-Cost Simulation-Driven Design](solver-independent_automated_problem_formulation_via_llms_for_high-cost_simulati.md)
- [\[ACL 2026\] UCS: Estimating Unseen Coverage for Improved In-Context Learning](ucs_estimating_unseen_coverage_for_improved_in-context_learning.md)

<!-- RELATED:END -->

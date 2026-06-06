---
title: >-
  [Paper Note] Relation-R1: Progressively Cognitive Chain-of-Thought Guided Reinforcement Learning for Unified Relation Comprehension
description: >-
  [AAAI 2026][LLM Reasoning][Visual Relation Understanding] This paper proposes Relation-R1, the first unified framework for binary and N-ary relation comprehension…
tags:
  - "AAAI 2026"
  - "LLM Reasoning"
  - "Visual Relation Understanding"
  - "Cognitive Chain-of-Thought"
  - "GRPO Reinforcement Learning"
  - "Scene Graph Generation"
  - "N-ary Relation Detection"
  - "Multimodal Large Language Models"
date: 2026-05-08
content_hash: 9b6ec098d443b000
---

# Relation-R1: Progressively Cognitive Chain-of-Thought Guided Reinforcement Learning for Unified Relation Comprehension

**Conference**: AAAI 2026
**arXiv**: [2504.14642](https://arxiv.org/abs/2504.14642)  
**Code**: [github.com/HKUST-LongGroup/Relation-R1](https://github.com/HKUST-LongGroup/Relation-R1)  
**Area**: LLM Reasoning
**Keywords**: Visual Relation Understanding, Cognitive Chain-of-Thought, GRPO Reinforcement Learning, Scene Graph Generation, N-ary Relation Detection, Multimodal Large Language Models

## TL;DR

This paper proposes Relation-R1, the first unified framework for binary and N-ary relation comprehension, combining progressively cognitive CoT-guided SFT with GRPO multi-reward optimization. With only 3B parameters, it surpasses 13B models, achieving 21.20% Mean (+6.87%) on PSG and state-of-the-art performance across all metrics on SWiG (Grnd-all 30.18%, +14.48%).

## Background & Motivation

- **Background**: Visual relation understanding is central to human-like cognition. While current MLLMs excel at object-level grounding and region captioning, they remain weak at comprehending semantic relations between objects — even simple binary relation detection falls short of expectations.

- **Limitations of Prior Work**: N-ary relations are substantially more complex than binary ones. Binary relations only require identifying the interaction between two objects (e.g., "child-drinking-glass"), whereas N-ary relations demand recognizing multiple entities fulfilling distinct semantic roles within an activity (e.g., for "drinking": agent=child, liquid=milk, container=glass). Existing models neglect functional dependencies among multiple entities (e.g., a glass serving as a container for milk), producing shallow relational triples that fail to capture deeper activity semantics.

- **Key Challenge**: Models over-rely on linguistic priors — upon seeing a person holding a cup, they default to outputting "person drinks milk" even when visual evidence only supports "holding." This stems from co-occurrence biases in training text rather than visually grounded semantic reasoning. Additionally, SFT alone overfits fixed training patterns and degrades on novel compositions, while pure RL (e.g., DeepSeek-R1 style) struggles to ensure format consistency in structured output tasks.

- **Goal**: To propose a unified framework that simultaneously models pairwise relations and multi-role activities, overcoming the limitations of task-specific architectures and combining the strengths of SFT and RL.

## Method

### Overall Architecture

Relation-R1 is a two-stage unified relation comprehension framework built upon Qwen2.5-VL-3B:

- **Stage 1 — SFT**: Supervised fine-tuning guided by progressively cognitive CoT, establishing multi-step reasoning capability while ensuring output format compliance.
- **Stage 2 — RL (GRPO)**: Reinforcement learning on the SFT-initialized model using three rule-based reward signals — format reward, binary relation reward, and N-ary relation reward — to improve generalization and robustness.

Both task types are handled in a unified manner: binary relations are expressed as scene graph descriptions using `<ref>/<box>/<pred>` tags; N-ary relations are expressed as grounded situation frames using verb + `<role>entity</role><box>` role tags.

### Key Design 1: Progressively Cognitive CoT Guidance (SFT Stage)

- **Function**: Injects two types of cognitive CoT into the model during SFT — template-based first, then MLLM-generated — to progressively guide multi-step reasoning.
- **Mechanism**:
    - **Template CoT (specific → canonical)**: Fixed step-by-step templates are designed; for binary relations: Object Existence → Object Localization → Relation Existence; for N-ary relations: Activity Recognition → Entities & Roles Recognition → Entity Localization. CoT is enclosed within `<think>` tags.
    - **MLLM-generated CoT (general → flexible)**: Qwen2.5-VL-72B serves as the teacher model to automatically generate diverse reasoning paths conditioned on task definitions, ground-truth scene graphs, and CoT generation instructions.
    - **Progressive Transition**: Training begins with 2 epochs on template CoT to establish format compliance, followed by fine-tuning on 4k MLLM-generated CoT samples to introduce reasoning flexibility.
- **Design Motivation**: Template CoT ensures correctness and format consistency but limits diversity; MLLM-generated CoT expands the exploration space but may introduce noise. The progressive combination captures the strengths of both — learning canonical structure before flexible reasoning — to prevent overfitting to a single reasoning pattern during SFT.

### Key Design 2: GRPO Multi-Reward Optimization (RL Stage)

- **Function**: Applies Group Relative Policy Optimization to the SFT-initialized model, guiding policy optimization via three rule-based reward signals.
- **Mechanism**:
    - **Format reward** $r_{\text{form}}$: Output must contain the `<think>...</think><answer>...</answer>` structure; score is 1 if satisfied, 0 otherwise, ensuring explicit reasoning expression.
    - **Binary relation reward** $r_{\text{binary}} = \alpha \cdot R + (1-\alpha) \cdot mR$: $R$ denotes sample-level triplet recall; $mR$ denotes mean recall across predicate categories. A triplet is considered correct when subject/predicate/object categories all match and bbox IoU ≥ 0.5.
    - **N-ary relation reward** $r_{\text{n-ary}} = \beta \cdot V_e + (1-\beta) \cdot V_{\text{grnd}}$: $V_e$ measures entity category and semantic role accuracy; $V_{\text{grnd}}$ measures entity localization precision (IoU ≥ 0.5).
    - **Multi-task gating**: Dynamically routes to binary or N-ary relation rewards based on whether `<ref>` tags appear in the output.
- **Design Motivation**: GRPO eliminates the need for an additional critic network and estimates advantage scores through within-group comparison, yielding computational efficiency. The three rewards separately constrain format, pairwise relations, and multi-role activities, providing finer-grained supervision than a single unified score. RL exploration further encourages the model to prioritize visual semantics over linguistic priors.

### Key Design 3: Unified Representation for Binary and N-ary Relations

- **Function**: Integrates binary and N-ary relation understanding into a single model, sharing reasoning pipeline and training workflow.
- **Mechanism**:
    - Binary relation output (scene graph description): `<ref>person</ref><box>[[x1,y1,x2,y2]]</box> <pred>drinking</pred> <ref>glass</ref><box>[[...]]</box>`
    - N-ary relation output (grounded situation frame): `drinking → <agent>child</agent><box>[...]</box> <liquid>milk</liquid><box>[...]</box>`
    - Both task types are jointly trained in SFT and GRPO; the GRPO stage automatically selects the corresponding reward by task type.
- **Design Motivation**: Binary and N-ary relations share underlying capabilities in entity recognition, spatial localization, and semantic reasoning. A unified framework enables mutual reinforcement between tasks while avoiding the overhead of maintaining multiple task-specific models.

### Loss & Training

- **SFT Stage**: Standard cross-entropy loss; template CoT training for 2 epochs, followed by MLLM CoT fine-tuning on 4k samples.
- **GRPO Stage**: Standard GRPO objective $J_{\text{GRPO}}(\theta)$; $G$ candidate responses are sampled to compute within-group normalized advantage scores $A_i$, with KL divergence regularization to prevent excessive deviation from the reference model. Training order: N-ary for 2.4k steps → binary for 3.6k steps → joint for 2k steps.
- Hyperparameters: $\alpha = \beta = 0.5$; SFT learning rate 2e-5; GRPO learning rate 1e-6; 8×A100 80GB GPUs.

## Key Experimental Results

### Table 1: Binary Relation Detection on PSG Dataset

| Method | Setting | Model Size | Recall | mRecall | Mean |
|--------|---------|-----------|--------|---------|------|
| IMP | Closed-vocab | - | 16.50 | 6.50 | 11.50 |
| MOTIFS | Closed-vocab | - | 20.00 | 9.10 | 14.55 |
| PSGFormer | Closed-vocab | - | 18.60 | 16.70 | 17.65 |
| ASMv2† | Open-vocab | 13B | 14.20 | 10.30 | 12.23 |
| SpaceSGG† | Open-vocab | 13B | 15.43 | 13.23 | 14.33 |
| **Relation-R1†** | **Open-vocab** | **3B** | **22.33** | **20.07** | **21.20** |
| R1-SGG⋆ | Standard format | 7B | 28.77 | 17.55 | 23.16 |
| **Relation-R1⋆** | **Standard format** | **3B** | **25.87** | **21.32** | **23.60** |

Under the Scene Graph Caption format, Relation-R1 with 3B parameters surpasses ASMv2 and SpaceSGG (both 13B), achieving a Mean improvement of +6.87%.

### Table 2: N-ary Relation Detection on SWiG Dataset

| Method | Setting | Verb | Value | Val-all | Grnd | Grnd-all |
|--------|---------|------|-------|---------|------|----------|
| CoFormer | Closed | 44.66 | 35.98 | 22.22 | 29.05 | 12.21 |
| GSRFormer | Closed | 46.53 | 37.48 | 23.32 | 31.53 | 14.23 |
| OpenSU | Open | 50.10 | 41.20 | 26.56 | 34.27 | 15.70 |
| **Relation-R1** | **Open** | **57.26** | **46.66** | **30.92** | **40.21** | **30.18** |

On the most challenging Grnd-all metric, an absolute gain of +14.48% is achieved, demonstrating the model's grounding capability for complex N-ary relations.

### Table 3: Ablation Study on CoT Strategies

| CoT Strategy | Binary Recall | Binary mRecall | N-ary Verb | N-ary Value | N-ary Grnd-all |
|-------------|--------------|----------------|-----------|------------|----------------|
| SFT only (no CoT) | 14.83 | 13.86 | 56.64 | 42.65 | 16.35 |
| SFT + RL (Template CoT) | 20.24 | 17.31 | 58.38 | 47.75 | 31.16 |
| SFT + RL (MLLM CoT) | 20.66 | 20.30 | 53.00 | 42.27 | 25.89 |
| **SFT + RL (Progressive)** | **22.57** | **20.57** | **71.04** | **61.26** | **36.09** |

The progressive CoT strategy achieves consistent improvements across all metrics; N-ary Verb accuracy increases from 56.64% to 71.04%.

## Highlights & Insights

- **First unified CoT+RL framework for binary and N-ary relation reasoning**: Prior work treats these two task types independently; Relation-R1 demonstrates the viability of a unified approach, with a 3B model outperforming 13B counterparts at remarkably high parameter efficiency.
- **Progressive CoT guidance is key to generalization**: The progressive paradigm — template CoT for canonical structure followed by MLLM-generated CoT for flexibility — significantly outperforms either strategy alone, and enables the model to develop emergent capability in expressing synonymous relations (e.g., unified understanding of "beside" and "next to").
- **Fine-grained division of labor in multi-reward design**: Format, binary relation, and N-ary relation rewards each serve distinct roles, augmented by multi-task gating for task-adaptive optimization — more effective than coarse-grained unified scoring.
- **Completion length naturally increases during GRPO training**: The model spontaneously generates more detailed reasoning chains during RL; progressive CoT guidance yields the longest generations, indicating active exploration of richer relational expressions.

## Limitations & Future Work

- The quality of MLLM-generated CoT is bounded by the teacher model (Qwen2.5-VL-72B); any bias in the teacher's relational understanding propagates to the student model as noise.
- Evaluation is limited to PSG and SWiG; validation on broader visual relation benchmarks (e.g., Visual Genome, Open Images) is absent.
- Reward weights $\alpha$ and $\beta$ are fixed at 0.5; adaptive balancing or dynamic adjustment strategies remain unexplored.
- The framework addresses relations in static images only; extending to dynamic temporal relations in video is a natural direction but has not been pursued.

## Related Work & Insights

- **ASMv2 (Wang et al. 2024)**: Uses a 13B MLLM for open-vocabulary SGG, but the SFT-only paradigm limits generalization. Relation-R1 surpasses its Mean by nearly 9 percentage points with a 3B model by incorporating the RL stage.
- **DeepSeek-R1**: Demonstrates that pure RL can elicit LLM reasoning capabilities, but direct application to visual structured output tasks yields inconsistent formatting. Relation-R1 resolves this by constraining format via SFT before applying RL optimization.
- **R1-SGG (Chen et al. 2025)**: A concurrent work that also introduces the R1 paradigm to SGG, but is limited to binary relations and does not explore CoT strategies. Relation-R1 additionally supports N-ary relations and systematically compares CoT guidance approaches.
- **OpenSU (Liu et al. 2023)**: The previous state-of-the-art for open-vocabulary grounded situation recognition; Relation-R1 surpasses it by +14.48% on Grnd-all, attributable to end-to-end RL training rather than reliance on external LLM descriptions.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of unified binary+N-ary relation handling, progressive CoT guidance, and multi-reward GRPO constitutes an entirely novel design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Dual-dataset dual-format evaluation, multi-CoT strategy ablation, and analysis of reward curves and generation length provide comprehensive empirical coverage.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly derived, the two-stage design logic is tightly structured, and problem formulations are precise.
- Overall: ⭐⭐⭐⭐ The work makes a substantial contribution to visual relation understanding with MLLMs; the progressive CoT paradigm has strong transferability to related tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] CMMCoT: Enhancing Complex Multi-Image Comprehension via Multi-Modal Chain-of-Thought and Memory Augmentation](cmmcot_enhancing_complex_multi-image_comprehension_via_multi.md)
- [\[AAAI 2026\] SERL: Self-Examining Reinforcement Learning on Open-Domain](serl_self-examining_reinforcement_learning_on_open-domain.md)
- [\[NeurIPS 2025\] SQL-R1: Training Natural Language to SQL Reasoning Model By Reinforcement Learning](../../NeurIPS2025/llm_reasoning/sql-r1_training_natural_language_to_sql_reasoning_model_by_reinforcement_learnin.md)
- [\[AAAI 2026\] LLMs for Game Theory: Entropy-Guided In-Context Learning and Adaptive CoT Reasoning](llms_for_game_theory_entropy-guided_in-context_learning_and_adaptive_cot_reasoni.md)
- [\[ICLR 2026\] Uni-CoT: Towards Unified Chain-of-Thought Reasoning Across Text and Vision](../../ICLR2026/llm_reasoning/uni-cot_towards_unified_chain-of-thought_reasoning_across_text_and_vision.md)

</div>

<!-- RELATED:END -->

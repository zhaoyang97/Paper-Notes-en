---
title: >-
  [Paper Note] Physics Context Builders: A Modular Framework for Physical Reasoning in Vision-Language Models
description: >-
  [Multimodal VLM] This paper proposes Physics Context Builders (PCBs), a modular framework that fine-tunes small specialized VLMs on simulation data to generate detailed physical scene descriptions…
tags:
  - "Multimodal VLM"
date: 2026-05-08
content_hash: 343fb2305a768d65
---

# Physics Context Builders: A Modular Framework for Physical Reasoning in Vision-Language Models

## Paper Information
- **Conference**: ICCV 2025
- **arXiv**: [2412.08619](https://arxiv.org/abs/2412.08619)
- **Area**: Multimodal VLM / Physical Reasoning
- **Keywords**: VLM physical reasoning, modular augmentation, simulation data, Sim2Real transfer, context building
- **Authors**: Vahid Balazadeh (U Toronto), Mohammadmehdi Ataei, Hyunmin Cheong, Amir Hosein Khasahmadi (Autodesk Research), Rahul G. Krishnan (U Toronto)

## TL;DR

This paper proposes Physics Context Builders (PCBs), a modular framework that fine-tunes small specialized VLMs on simulation data to generate detailed physical scene descriptions, which serve as physical context to augment the physical reasoning capabilities of large foundation VLMs (e.g., GPT-4o), without modifying the large model itself.

## Background & Motivation

VLMs perform poorly on physical reasoning tasks: GPT-4o achieves near-perfect accuracy on descriptive tasks (99%), yet only 55–60% on stability prediction (close to random chance). The root cause is that VLM training data (MSCOCO, Conceptual Captions, etc.) lacks annotations of physical relationships.

Although fine-tuning on physics data can substantially improve performance, directly fine-tuning closed-source large models such as GPT-4o is neither practical nor economical. PCBs are designed precisely to address this problem: small models handle physical perception, while large models handle reasoning, achieving a clean separation between perception and reasoning.

Key insight: the bottleneck of VLM physical reasoning lies at the **perception level** (inability to extract physical information from visual inputs), rather than solely at the reasoning level. Once visual inputs are converted into rich physics-grounded textual descriptions, the reasoning capabilities of large models can be fully activated.

## Method

### Overall Architecture

PCBs consist of three core stages:
1. **Training stage**: a physics simulator generates images/videos with corresponding annotations → converted into physical description training data → a small VLM (PaliGemma-3B) is fine-tuned.
2. **Inference stage**: a PCB receives new images/videos → generates detailed physical scene descriptions → descriptions are provided as context to a large model → the large model answers physical reasoning questions.
3. **Multi-agent integration**: a Triage Agent automatically selects the appropriate PCB based on the query.

### Physical Context Generation Format

Two description types are proposed:
- **Human-like Narration (HN)**: natural language descriptions of physical attributes and spatial relationships in the scene, better aligned with foundation models' preference for natural language.
- **Structured Physics (SP)**: per-frame structured descriptions with standardized physical attribute labels; precise but slightly less interpretable by models.

### PCB Training

- Base model: PaliGemma-3B
- Fine-tuning method: LoRA
- Loss function: negative log-likelihood (autoregressive)
- Video input: 8 sampled frames concatenated into the input context
- Training data: simulator-generated physical descriptions (not QA pairs), making PCBs task-agnostic with respect to downstream reasoning

### Multi-Agent Framework

- Inspired by the OpenAI Swarm architecture
- A Triage Agent (GPT-4o/mini) analyzes user queries and visual inputs, routing them to the appropriate specialized PCB
- F1-Score reaches 0.93–0.98, demonstrating that foundation models can reliably select the correct PCB

### Key Design Advantages

- **Modular**: PCBs can be trained and deployed independently
- **Efficient**: only a 3B small model requires fine-tuning
- **Flexible**: different PCBs can be trained for different physical phenomena
- **Compatible**: adaptable to any VLM that supports in-context learning

## Key Experimental Results

### Falling Tower Benchmark (Static Stability Detection)

| Model | Descriptive (sim) | Stability-Object (sim/real) | Stability-Tower (sim/real) |
|------|-----------|-------------------|-----------------|
| GPT-4o (zero-shot) | 99.3 | 56.9 / 60.0 | 59.6 / 55.0 |
| GPT-4o + PCB(HN) | 99.5 | 76.7 / 75.0 | **85.1 / 70.0** |
| GPT-4o-mini (zero-shot) | 94.9 | 49.0 / 52.6 | 53.1 / 36.8 |
| GPT-4o-mini + PCB(HN) | 99.9 | 75.0 / 70.0 | **84.7 / 40.0** |
| PaliGemma fine-tuned | 100.0 | 84.6 / 70.0 | 87.6 / 65.0 |

PCBs bring a **+25.5%** improvement in tower stability accuracy for GPT-4o.

### CLEVRER Benchmark (Dynamic Physical Reasoning, Table 5)

| Model | Descriptive | Explanatory (per opt.) | Counterfactual |
|------|--------|-----------------|--------|
| GPT-4o (zero-shot) | 62.7 | 30.7 | 60.2 |
| GPT-4o + PCB(HN) | **75.6** (+12.9) | **41.6** (+10.9) | **68.4** (+8.2) |
| GPT-4o + PCB(SP) | 70.0 (+7.3) | 34.9 (+4.2) | 63.3 (+3.1) |
| Gemini + PCB(HN) | 72.8 (+14.2) | 35.6 (+19.9) | 64.9 (+9.3) |

### Ablation Study

**Comparison of fine-tuning data types (Table 3)**:

| Fine-tuning Data | Descriptive (count) | Descriptive (order) | Stability (object) | Stability (tower) |
|---------|-------------|-------------|-----------|---------|
| No fine-tuning | 50.9 | 8.5 | 51.0 | 39.1 |
| Stability QA only | 52.4 | 41.2 | 84.4 | 86.5 |
| Descriptive QA only | 100.0 | 100.0 | 51.0 | 39.1 |
| All QA | 100.0 | 100.0 | 84.6 | 87.6 |

**PCB context format comparison**: HN format systematically outperforms SP format across all models and task types, indicating that foundation models are better at interpreting physics described in natural language.

### Key Findings

1. **Perception, not reasoning, is the bottleneck**: GPT-4o's descriptive capability is near-perfect, yet its physical reasoning is near-random; providing physical context via PCBs leads to substantial gains.
2. **Small fine-tuned models are surprisingly powerful**: PaliGemma (3B), after fine-tuning, surpasses GPT-4o on physical reasoning tasks.
3. **Sim2Real transfer succeeds**: PCBs trained exclusively on simulation data remain effective on real-world images.
4. **HN > SP**: natural language format outperforms structured format.
5. **Counterfactual reasoning remains a bottleneck**: improvements are limited to 1.7–9.5%, as PCBs inherently describe observed rather than hypothetical scenes.

## Highlights & Insights

- **Perception–reasoning separation paradigm**: explicitly disentangles the sources of difficulty in VLM physical reasoning, providing a clear path toward modular solutions.
- **Simulation data used only at training time**: no simulator is required at inference, avoiding the computational overhead of simulation-in-the-loop approaches.
- **Plug-and-play**: compatible with any closed-source or open-source foundation model.
- Data efficiency analysis reveals that descriptive tasks saturate with only ~10% of the data, whereas stability tasks continue to benefit from additional data.

## Limitations & Future Work

- Coverage is limited to rigid body dynamics and stability; fluid dynamics, deformable objects, and other phenomena are not addressed.
- Direct application to unannotated real-world videos (e.g., YouTube) is not straightforward.
- Improvements on counterfactual and predictive reasoning are limited, as PCBs cannot generate descriptions of future scenes.
- Real-world evaluation of the Falling Tower benchmark is limited in scale (only 20 real images and 100 QA pairs).

## Related Work & Insights

- **Connection to tool-augmented LLMs**: PCBs are essentially physics perception tools for VLMs.
- **Extensible to additional physical phenomena**: specialized PCBs can be trained for fluid dynamics, articulated motion, and more.
- **Implications for VLM evaluation**: the physical reasoning capabilities of existing VLMs are far weaker than they appear, motivating more targeted benchmarks.

## Rating ⭐⭐⭐⭐

The approach is conceptually clean and elegant, with clear and insightful experimental conclusions. The perception–reasoning separation perspective is valuable for understanding the capability boundaries of VLMs. The PCB framework is highly modular and practically applicable. The primary limitations are the narrow evaluation scope and the small scale of real-world data.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] MaTVLM: Hybrid Mamba-Transformer for Efficient Vision-Language Modeling](matvlm_hybrid_mamba-transformer_for_efficient_vision-language_modeling.md)
- [\[ICCV 2025\] R1-VL: Learning to Reason with Multimodal Large Language Models via Step-wise Group Relative Policy Optimization](r1-vl_learning_to_reason_with_multimodal_large_language_models_via_step-wise_gro.md)
- [\[AAAI 2026\] FT-NCFM: An Influence-Aware Data Distillation Framework for Efficient VLA Models](../../AAAI2026/multimodal_vlm/ft-ncfm_an_influence-aware_data_distillation_framework_for_efficient_vla_models.md)
- [\[ICCV 2025\] Large Multi-modal Models Can Interpret Features in Large Multi-modal Models](large_multi-modal_models_can_interpret_features_in_large_multi-modal_models.md)
- [\[AAAI 2026\] Plug-and-Play Clarifier: A Zero-Shot Multimodal Framework for Egocentric Intent Disambiguation](../../AAAI2026/multimodal_vlm/plug-and-play_clarifier_a_zero-shot_multimodal_framework_for_egocentric_intent_d.md)

</div>

<!-- RELATED:END -->

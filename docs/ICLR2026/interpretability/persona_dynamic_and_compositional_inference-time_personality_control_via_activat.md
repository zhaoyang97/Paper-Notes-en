---
title: >-
  [Paper Note] PERSONA: Dynamic and Compositional Inference-Time Personality Control via Activation Vector Algebra
description: >-
  [ICLR 2026][Interpretability][personality control] This paper proposes the PERSONA framework, which extracts approximately orthogonal personality vectors from activation space and applies vector algebra operations (scali…
tags:
  - "ICLR 2026"
  - "Interpretability"
  - "personality control"
  - "activation steering"
  - "vector algebra"
  - "inference-time"
  - "Big Five"
date: 2026-05-08
content_hash: 163a80ddc380637b
---

# PERSONA: Dynamic and Compositional Inference-Time Personality Control via Activation Vector Algebra

**Conference**: ICLR 2026
**arXiv**: [2602.15669](https://arxiv.org/abs/2602.15669)  
**Code**: [GitHub](https://github.com/) (declared public by authors)  
**Area**: Robotics
**Keywords**: personality control, activation steering, vector algebra, inference-time, Big Five

## TL;DR

This paper proposes the PERSONA framework, which extracts approximately orthogonal personality vectors from activation space and applies vector algebra operations (scaling, addition, subtraction) to achieve training-free dynamic and compositional personality control. PERSONA attains a score of 9.60 on PersonalityBench, nearly matching the SFT upper bound of 9.61.

## Background & Motivation

1. Personality control in LLMs is critical for healthcare, education, and social simulation, yet existing methods exhibit significant limitations.

**Prompting-based methods** (e.g., simple prompts, P² induction) are unstable and inconsistent, making precise personality expression difficult to achieve.

**Fine-tuning methods** (SFT / LoRA) demand substantial computational resources and require independent training for each personality configuration.
4. More fundamentally, existing methods treat personality as static and monolithic, failing to capture the dynamic and compositional nature of human behavioral traits.
5. Core insight: personality traits manifest as extractable, approximately orthogonal directions in the model's representation space, supporting algebraic operations.
6. This reframes personality control from text engineering or gradient optimization into vector arithmetic in high-dimensional space.

## Method

### Overall Architecture

PERSONA consists of four tightly integrated components:
- **Persona-Base**: Extracts orthogonal vectors for the ten poles of the Big Five (OCEAN) personality dimensions.
- **Persona-Algebra**: Enables compositional personality manipulation via vector arithmetic.
- **Persona-Flow**: Dynamically adapts personality composition at inference time.
- **Persona-Evolve**: An evaluation benchmark comprising 800 multi-turn dialogue scenarios.

### Key Designs

**Design 1: Persona-Base — Personality Vector Extraction**
- **Function**: Extracts contrastive vectors for the ten poles of the five OCEAN dimensions from the model's activation space.
- **Mechanism**: Employs Contrastive Activation Analysis: (1) generate contrastive system prompts (eliciting/suppressing traits); (2) collect residual stream activations under positive and negative conditions; (3) compute their mean difference to obtain direction vector $v_l$.
- **Design Motivation**: Establishes the fundamental "atomic" operational units for personality control. Cosine similarity between vectors confirms approximate orthogonality, while opposing trait pairs exhibit strong negative correlations.

**Design 2: Persona-Algebra — Vector Algebraic Operations**
- **Function**: Validates and leverages the mathematical operations supported by personality vectors.
- **Mechanism**: Three operations — scalar multiplication ($\alpha \cdot v$) controls trait intensity; vector addition ($v_{outgoing} + v_{compassionate}$) enables multi-trait composition; vector subtraction ($v_{outgoing} - v_{solitary}$) suppresses specific traits.
- **Design Motivation**: Adapts the BFI-44 questionnaire into a behavioral evaluation to demonstrate that vector operations produce predictable changes in personality scores. Pearson correlation coefficients exceed 0.9 for most traits.

**Design 3: Persona-Flow — Dynamic Inference-Time Control**
- **Function**: Dynamically adjusts personality expression at inference time based on conversational context.
- **Mechanism**: A two-stage predict-then-steer mechanism. Stage 1: analyzes dialogue context and predicts adjustment coefficients $\alpha_i \in [-2, +2]$ for each dimension. Stage 2: computes the composite vector $v_{composite} = \sum_{i \in OCEAN} \alpha_i \cdot v_i$ and injects it into the residual stream.
- **Design Motivation**: Enables real-time personality modulation without pre-specified scripts, supporting context-aware adaptive control.

### Loss & Training

This method is entirely training-free and involves no gradient updates. The core operation is residual addition in activation space:
$$h_l \leftarrow h_l + \alpha \cdot v_l$$
where $\alpha$ is the steering coefficient and $v_l$ is the personality vector extracted from the optimal layer. Positive/negative $\alpha$ amplifies/suppresses the corresponding trait pole, respectively.

## Key Experimental Results

### Main Results

| Method | Mean Score↑ | Variance↓ | Training Required |
|--------|------------|-----------|-------------------|
| PERSONA-Base | **9.60** | 0.74 | Training-free |
| NPTI | 9.43 | 0.49 | Training-free |
| P² | 9.43 | 0.83 | Training-free |
| Simple Prompt | 8.39 | 0.96 | Training-free |
| PAS | 6.93 | 1.71 | Training-free |
| ActAdd | 8.20 | 2.10 | Training-free |
| SFT (upper bound) | 9.61 | 0.49 | Fine-tuning required |

### Ablation Study

| Model | TA | RC | RA | IF | Overall |
|-------|----|----|----|----|---------|
| Qwen3-4B | 92.2 | 90.6 | 92.4 | 49.1 | **90.8** |
| Qwen2.5-14B | 84.8 | 86.4 | 84.8 | 59.3 | 85.4 |
| Llama-3.1-8B | 84.9 | 81.4 | 85.6 | 57.2 | 83.5 |
| Qwen2.5-7B | 84.7 | 84.4 | 85.0 | 61.4 | 83.4 |
| Ministral-8B | 74.3 | 73.2 | 74.2 | 48.0 | 73.2 |

### Key Findings

1. The training-free method PERSONA-Base (9.60) nearly matches the SFT upper bound (9.61), with higher but acceptable variance.
2. Scalar multiplication of vectors exhibits a strong linear relationship with BFI-44 dimension scores, confirming the linear editability of personality traits.
3. Certain traits exhibit asymmetric steering effects: traits that conflict with the model's safety training (e.g., self-interested) are difficult to activate even under high steering coefficients.
4. On MMLU/TruthfulQA, Persona-Flow maintains or slightly improves general model capability, producing no adverse side effects.
5. Larger model capacity enhances personality controllability: across the Qwen2.5 series, overall win rate improves from 78.4% (3B) to 85.4% (14B).

## Highlights & Insights

1. **Extreme methodological simplicity**: Entirely training-free, achieving SFT-level personality control via vector addition and subtraction alone, with minimal computational overhead.
2. **Geometric perspective as a breakthrough**: Reframes personality control from "text engineering" to "vector arithmetic," revealing interpretable structure in LLM representation space.
3. **Compositionality + dynamism**: The predict-then-steer mechanism of Persona-Flow enables context-aware real-time personality modulation for the first time.
4. **Rigorous orthogonality validation**: Vector independence is verified via cosine similarity heatmaps and causal intervention experiments.
5. **Persona-Evolve benchmark**: Constructs 800 multi-turn dialogue scenarios, filling a gap in dynamic personality evaluation.

## Limitations & Future Work

1. **Asymmetric steering effects**: Traits conflicting with safety alignment are difficult to activate (e.g., self-interested scores only 20.8), constraining fully unconstrained personality control.
2. **Information Fidelity metrics remain low** (48–61%), indicating that maintaining factual accuracy while adjusting personality remains challenging.
3. **Vector extraction is model-dependent**: Vectors are currently extracted using Qwen2.5-7B; cross-model transfer solutions remain underdeveloped.
4. **Additional inference overhead from Persona-Flow**: The predict-then-steer process requires intermediate reasoning steps, introducing latency.
5. Validation is currently limited to the Big Five framework; whether the approach generalizes to finer-grained personality dimensions warrants further exploration.

## Related Work & Insights

- **Representation Engineering** (Rimsky et al., 2024; Turner et al., 2023): Provides the methodological foundation for activation steering.
- **NPTI** (Deng et al., 2025): A neuron-based personality control method that does not support compositional operations.
- **ActAdd** (Turner et al., 2023): A pioneer in residual stream modification, but with insufficient precision for personality control (variance: 2.10).
- Insight: This vector algebra perspective may generalize to other LLM behavioral control tasks, such as style transfer, knowledge injection, and safety alignment.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Recasting personality control as vector algebraic operations is highly original; the Persona-Flow dynamic control mechanism is also a first.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-model, multi-benchmark evaluation is comprehensive, though certain metrics (IF) remain moderate.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The paper is clearly structured, with a smooth logical progression from extraction to algebra to dynamic control.
- **Value**: ⭐⭐⭐⭐⭐ A training-free method matching the SFT upper bound represents a milestone in personality control with high practical significance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] SALVE: Sparse Autoencoder-Latent Vector Editing for Mechanistic Control of Neural Networks](salve_sparse_autoencoder-latent_vector_editing_for_mechanistic_control_of_neural.md)
- [\[NeurIPS 2025\] Dynamic Features Adaptation in Networking: Toward Flexible Training and Explainable Inference](../../NeurIPS2025/interpretability/dynamic_features_adaptation_in_networking_toward_flexible_training_and_explainab.md)
- [\[ACL 2026\] FineSteer: A Unified Framework for Fine-Grained Inference-Time Steering in Large Language Models](../../ACL2026/interpretability/finesteer_a_unified_framework_for_fine-grained_inference-time_steering_in_large_.md)
- [\[ICLR 2026\] Dynamic Reflections: Probing Video Representations with Text Alignment](dynamic_reflections_probing_video_representations_with_text_alignment.md)
- [\[ICLR 2026\] GAVEL: Towards Rule-Based Safety through Activation Monitoring](gavel_towards_rule-based_safety_through_activation_monitoring.md)

</div>

<!-- RELATED:END -->

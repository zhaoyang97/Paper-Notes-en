---
title: >-
  [Paper Note] Fine-tuning with RAG for Improving LLM Learning of New Skills
description: >-
  [ICLR 2026][Information Retrieval & RAG][RAG distillation] This paper proposes transforming RAG from a permanent inference-time dependency into a training-time teacher signal. Hints are extracted from agent failures…
tags:
  - "ICLR 2026"
  - "Information Retrieval & RAG"
  - "RAG distillation"
  - "LLM agent"
  - "hint extraction"
  - "ALFWorld"
  - "WebShop"
date: 2026-05-08
content_hash: ae6a5e618d27c960
---

# Fine-tuning with RAG for Improving LLM Learning of New Skills

**Conference**: ICLR 2026
**arXiv**: [2510.01375](https://arxiv.org/abs/2510.01375)  
**Code**: [Anonymous Repository](https://anonymous.4open.science/r/anonymized-submission-iclr/README.md)  
**Area**: Information Retrieval
**Keywords**: RAG distillation, LLM agent, hint extraction, ALFWorld, WebShop

## TL;DR

This paper proposes transforming RAG from a permanent inference-time dependency into a training-time teacher signal. Hints are extracted from agent failures, used to augment a teacher model that generates higher-quality trajectories, and then removed during distillation into a student model. The student thereby internalizes the retrieval-augmented behavior without requiring runtime RAG, achieving a 91% success rate on ALFWorld (baseline: 79%) and a score of 72 on WebShop (baseline: 61).

## Background & Motivation

LLM agents frequently fail in multi-step tasks in predictable ways: executing actions whose preconditions are unmet, issuing redundant instructions, or mishandling environmental constraints. Existing remedies each carry significant limitations:

**Structured prompting** (ReAct, StateAct): provides reasoning scaffolds but is bounded by parametric knowledge.

**Self-reflection** (Reflexion): requires multiple attempts and real feedback signals.

**Retrieval augmentation** (RAG): injects external knowledge but introduces runtime overhead and deployment complexity.

**Fine-tuning**: demands large volumes of high-quality data and risks overfitting.

The core insight is that **RAG need not exist as a permanent runtime dependency**. It can instead serve as a source of improved training supervision, to be internalized into model parameters. Specifically, if RAG can be used to generate better demonstration trajectories, and a student model is trained on those trajectories without being provided the hints, the student can learn the behavioral improvements conferred by RAG while no longer requiring retrieval at inference time.

## Method

### Overall Architecture

The proposed pipeline consists of four stages:

**Stage A – Baseline Agent Trajectory Collection**: A baseline agent (ReAct or StateAct) is deployed on the training set to collect both successful and failed trajectories. Successful trajectories form the baseline SFT dataset; failed trajectories are used for hint extraction.

**Stage B – Automatic Hint Extraction**: For each failed trajectory, a complete failure record comprising the task instruction, initial observation, full action sequence, and outcome is constructed, and GPT-4o is prompted to generate 1–4 imperative-style hints. Hints employ placeholders (e.g., {object}, {container}) to promote generalization, and are stored categorized by task type.

**Stage C – Teacher Data Generation**: Given an instruction and initial observation, the task category is identified, the top-$k$ ($k=3$) hints are retrieved from the corresponding hint bank, and a quantized Qwen-2.5 7B model performs LLM-based re-ranking. Hints are injected once at episode start ($t=0$), and the teacher agent runs the full episode; only successful trajectories are retained.

**Stage D – Distillation Training**: Hint strings and few-shot examples are **removed** from the teacher trajectories to form the distillation dataset. A LoRA adapter is trained on the student model, compelling it to internalize the behavioral guidance conveyed by the hints.

### Key Designs

1. **Failure-Driven Hint Extraction**:
    - Requires no expert supervision; the agent learns from its own failures.
    - GPT-4o diagnoses the cause of failure and generates corrective rules.
    - Example hints: "Ensure {container} is opened before placing {object}" and "Use a systematic search pattern to avoid missing {object}."
    - Deduplication is performed via fuzzy matching (Levenshtein distance threshold: 0.85).
    - The pipeline generates 760/650 hints (ReAct/StateAct) for ALFWorld and 756/831 for WebShop.

2. **One-Shot Retrieval**:
    - Hints are retrieved only once at episode start ($t=0$), not dynamically during execution.
    - This constrains token overhead while reflecting realistic conditions where guidance is provided a single time at task onset.
    - LLM-based re-ranking is used instead of traditional embedding retrieval, yielding higher ranking quality.

3. **Hint Removal for Distillation**:
    - Training data is derived from teacher trajectories produced with hints, but hint text is removed from the inputs.
    - Few-shot examples are also removed, as they are fixed across tasks and provide no useful training signal.
    - This compels the student to learn *behaviors* rather than *textual patterns*, achieving genuine internalization.

### Loss & Training

- Training objective: full-sequence next-token cross-entropy loss.
- QLoRA-style training: backbone quantized to 4-bit; LoRA adapter trained in bf16 precision.
- ALFWorld: LR $2\times10^{-4}$, sequence length 1024, LoRA rank 64, $\alpha=128$, dropout 0.10, weight decay 0.01.
- WebShop: LoRA rank 16, $\alpha=32$, dropout 0.20, weight decay 0.05.
- WebShop applies token-level label smoothing ($\varepsilon=0.1$) to mitigate overconfidence on short trajectories.
- Optimizer: 8-bit AdamW, linear schedule, batch size $2 \times 4$ gradient accumulation.
- Single epoch training with 10% warmup on a single A100 80GB GPU.

## Key Experimental Results

### Main Results

Results are averaged over ReAct and StateAct using Qwen-2.5 14B Instruct:

| Method | ALFWorld Success Rate | WebShop Success Rate | WebShop Score |
|---|---|---|---|
| Base | 79.85% | 38.5% | 60.87 |
| Base+RAG | 82.09% | 43.5% | 67.08 |
| SFT | 85.45% | 43.0% | 72.09 |
| **Distilled (Ours)** | **91.04%** | **43.5%** | **72.40** |

Qwen-2.5 7B Instruct:

| Method | ALFWorld Success Rate | WebShop Success Rate | WebShop Score |
|---|---|---|---|
| Base | 26.49% | 13.0% | 28.12 |
| Base+RAG | 71.27% | 8.5% | 18.46 |
| SFT | 62.69% | 22.0% | 54.38 |
| **Distilled (Ours)** | **73.88%** | **22.5%** | **61.04** |

### Efficiency Analysis (14B)

| Environment | Method | Tokens/Episode | Steps | Performance |
|---|---|---|---|---|
| ALFWorld | Base | 50.13k | 18.94 | 79.85% |
| | RAG | 53.97k | 18.69 | 82.09% |
| | Distilled | **44.82k** | **16.68** | **91.04%** |
| WebShop | Base | 7.99k | 7.16 | 60.87 |
| | RAG | 11.05k | 6.34 | 67.08 |
| | Distilled | **4.27k** | **4.98** | **72.40** |

The distilled model reduces token usage by 10% on ALFWorld and 47% on WebShop, while achieving the best performance.

### Ablation Study

Retrieval depth $k$ ablation (ALFWorld, 14B):

| $k$ | Success Rate | Steps | Tokens |
|---|---|---|---|
| 1 | 83.96% | 19.11 | 52.02k |
| 3 (Ours) | 82.09% | 18.69 | 53.97k |
| 6 | 84.33% | 18.13 | 50.95k |
| 9 | 76.87% | 19.27 | 57.26k |

$k=3$ achieves the best overall balance across both environments; performance degrades at $k=9$ due to hint overload.

### Key Findings

- The distilled model dominates the accuracy–efficiency Pareto frontier entirely.
- The 7B distilled model's WebShop score (61.04) approaches that of the 14B Base (60.87), demonstrating effective cross-scale compression.
- For small models (7B), RAG-augmented inference on WebShop actually degrades performance — hints mislead the model into incorrect attribute selection — whereas distillation enables stable utilization of complex guidance.
- The performance gap between SFT and distillation confirms that hint-augmented trajectories contain additional behavioral knowledge beyond what standard SFT can capture.

## Highlights & Insights

1. **"Converting runtime augmentation into training-time supervision" paradigm**: This principle generalizes well beyond RAG to other augmentation strategies such as CoT and self-critique.
2. **Failure-driven automation**: The entire pipeline requires no human expert knowledge; reusable guidance rules are distilled directly from the agent's own failures.
3. **Hint removal is the critical step**: The presence of hints during training and their absence at inference time forces the model to internalize explicit rules as implicit knowledge.
4. **Comprehensive efficiency analysis**: The evaluation goes beyond accuracy to systematically assess token overhead and step count.
5. **Validation across agent architectures**: The method proves effective under both ReAct and StateAct, demonstrating methodological generality.

## Limitations & Future Work

1. **GPT-4o dependency for hint generation**: API call costs are non-trivial at scale.
2. **Single $t=0$ retrieval**: The model cannot adapt to unexpected situations arising mid-episode.
3. **Single-seed evaluation**: All results are point estimates; variance across multiple seeds is not reported.
4. **Cross-domain generalization untested**: Experiments are limited to ALFWorld and WebShop; generalization to novel environments remains unvalidated.
5. **Hint quality ceiling**: If GPT-4o-generated hints are themselves of limited quality — particularly for diagnosing complex failures — the performance ceiling will be correspondingly constrained.

## Related Work & Insights

- **ReAct / StateAct**: Foundational agent prompting frameworks upon which this work builds by adding hint augmentation and distillation.
- **Reflexion**: Also learns from failures, but requires multiple attempts; the proposed method requires only a single training pass.
- **ExpeL / AutoGuide**: Extract knowledge from experience but retain it as permanent runtime dependencies; this work distills such knowledge into model parameters.
- **FireAct**: Pursues a similar fine-tuning approach but relies on GPT-4 expert trajectories; this work generates teacher data from the agent's own failures and self-extracted hints.
- **Prompt Distillation**: Compresses complex prompts into model weights; this work extends the paradigm to the distillation of dynamic guidance in agent settings.

## Rating

- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Beyond RAG vs. Long-Context: Learning Distraction-Aware Retrieval for Efficient Knowledge Grounding](beyond_rag_vs_long-context_learning_distraction-aware_retrieval_for_efficient_kn.md)
- [\[ACL 2026\] Bayesian Active Learning with Gaussian Processes Guided by LLM Relevance Scoring](../../ACL2026/information_retrieval/bayesian_active_learning_with_gaussian_processes_guided_by_llm_relevance_scoring.md)
- [\[CVPR 2026\] Beyond Global Similarity: Towards Fine-Grained, Multi-Condition Multimodal Retrieval](../../CVPR2026/information_retrieval/beyond_global_similarity_towards_fine-grained_multi-condition_multimodal_retriev.md)
- [\[ACL 2026\] End-to-End Optimization of LLM-Driven Multi-Agent Search Systems via Heterogeneous-Group-Based Reinforcement Learning](../../ACL2026/information_retrieval/end-to-end_optimization_of_llm-driven_multi-agent_search_systems_via_heterogeneo.md)
- [\[ICLR 2026\] LightRetriever: A LLM-based Text Retrieval Architecture with Extremely Faster Query Inference](lightretriever_a_llm-based_text_retrieval_architecture_with_extremely_faster_que.md)

</div>

<!-- RELATED:END -->

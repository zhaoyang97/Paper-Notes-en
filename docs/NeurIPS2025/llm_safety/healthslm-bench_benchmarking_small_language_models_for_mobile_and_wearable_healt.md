---
title: >-
  [Paper Note] HealthSLM-Bench: Benchmarking Small Language Models for Mobile and Wearable Healthcare Monitoring
description: >-
  [NeurIPS 2025][LLM Safety][small language models] The first benchmark systematically evaluating small language models (SLMs, 1–4B parameters) on mobile and wearable health monitoring tasks, covering zero-shot, few-shot, and instruction fine-tuning paradigms, with on-device deployment validated on an iPhone.
tags:
  - NeurIPS 2025
  - LLM Safety
  - small language models
  - mobile health monitoring
  - wearable devices
  - privacy preservation
  - on-device deployment
date: 2026-05-08
content_hash: 9f6ace3905f2cbcc
---

# HealthSLM-Bench: Benchmarking Small Language Models for Mobile and Wearable Healthcare Monitoring

**Conference**: NeurIPS 2025
**arXiv**: [2509.07260](https://arxiv.org/abs/2509.07260)
**Code**: None
**Area**: AI Safety (Healthcare AI)
**Keywords**: small language models, mobile health monitoring, wearable devices, privacy preservation, on-device deployment

## TL;DR

The first benchmark systematically evaluating small language models (SLMs, 1–4B parameters) on mobile and wearable health monitoring tasks, covering zero-shot, few-shot, and instruction fine-tuning paradigms, with on-device deployment validated on an iPhone.

## Background & Motivation

**State of the Field**: Mobile and wearable devices can continuously collect physiological data such as step counts, heart rate, and sleep metrics. LLMs have demonstrated strong generalization in health prediction tasks (e.g., Health-LLM, PhysioLLM).

**Limitations of Prior Work**: LLM-based approaches predominantly rely on cloud inference and face three core challenges: (1) privacy leakage risk, as sensitive health data must be uploaded to remote servers; (2) communication latency that impairs real-time monitoring; and (3) computational and memory overhead of 7B+ models far exceeding mobile device capacity.

**Root Cause**: A fundamental tension exists between the powerful capabilities of LLMs and the resource constraints of mobile platforms — a solution is needed that preserves LLM-level performance while enabling local execution.

**Paper Goals**: Can SLMs (≤7B parameters) match LLMs on health prediction tasks? What is the deployment efficiency on real mobile devices?

**Starting Point**: Construct a comprehensive benchmark that systematically compares 9 state-of-the-art SLMs against multiple LLMs across 8 health tasks, with actual deployment on an iPhone for validation.

**Core Idea**: With appropriate fine-tuning, SLMs can match or even surpass LLMs on health monitoring tasks while delivering orders-of-magnitude efficiency gains and stronger privacy guarantees.

## Method

### Overall Architecture

HealthSLM-Bench evaluates SLMs under three paradigms: (1) **Zero-shot learning** — direct inference without examples; (2) **Few-shot learning** — in-context learning with 1/3/5/10 examples; (3) **Instruction fine-tuning** — parameter-efficient fine-tuning via LoRA. The best-performing models are subsequently deployed on an iPhone 15 Pro Max to assess on-device efficiency.

### Key Designs

1. **Zero-shot Prompt Construction**:

    - **Function**: Design standardized prompt templates for health monitoring
    - **Design Motivation**: Evaluate the intrinsic health reasoning capability of SLMs based on pre-trained knowledge
    - **Mechanism**: Prompts consist of three components — Instruction (role setting, e.g., "You are a personal health agent") + Main Query (14-day sensor data sequences: steps, calories, heart rate, sleep, etc.) + Output Constraints (restricting output format, e.g., "Predict fatigue level 1–5")
    - **Novelty**: Chain-of-thought and self-consistency are intentionally excluded to preserve on-device deployment efficiency

2. **Few-shot Prompt Construction**:

    - **Function**: Enhance in-context learning via a small number of labeled examples
    - **Design Motivation**: Leverage in-context learning to capture input–output patterns
    - **Mechanism**: $\text{Prompt}_{FS} = \text{Instruction}_{FS} + \text{Examples}_N + \text{Prompt}_{ZS}$, where each example consists of a zero-shot prompt paired with its answer. Experiments are conducted with $N \in \{1, 3, 5, 10\}$
    - **Novelty**: Different tasks exhibit distinct sensitivity to the number of examples — mental health tasks benefit more from additional demonstrations

3. **Instruction Fine-tuning (LoRA)**:

    - **Function**: Format instruction–response pairs using the Alpaca template and fine-tune efficiently via LoRA
    - **Design Motivation**: Update model parameters to achieve more persistent task alignment
    - **Mechanism**: Trainable low-rank decomposition matrices are introduced into attention and feed-forward layers while original weights are frozen
    - **Novelty**: Particularly suited for on-device inference, minimizing memory and computational overhead

4. **On-device Deployment**:

    - **Function**: Deploy the best-performing SLMs on an iPhone 15 Pro Max
    - **Mechanism**: Models are converted to GGUF format → 4-bit quantization → inference via the Llama.cpp engine
    - **Evaluation Metrics**: TTFT (time to first token), ITPS/OTPS (throughput), OET (output evaluation time), CPU/RAM usage

### Loss & Training

- Cross-entropy loss for classification tasks; MAE loss for regression tasks
- LoRA fine-tuning: 8:2 train/test split with 14-day sliding window normalized data
- Evaluation metrics: Accuracy for classification, MAE for regression

## Key Experimental Results

### Main Results

**Zero-shot Performance Comparison (LLMs vs. SLMs):**

| Metric | LLMs Mean | SLMs Mean | Best SLM |
|--------|-----------|-----------|----------|
| Stress MAE↓ | 0.64 | **0.61** | Qwen2-1.5B: 0.40 |
| Readiness MAE↓ | 2.56 | **2.15** | Llama-3.2-1B: 1.87 |
| Fatigue Acc↑ | 41.54% | **52.20%** | Llama-3.2-1B: 63.79% |
| Sleep Quality MAE↓ | **0.60** | **0.60** | Gemma-2-2B: 0.47 |
| Calories MAE↓ | **47.60** | 143.23 | Llama-3.2-3B: 19.70 |

**Instruction Fine-tuning (LoRA) Performance Comparison:**

| Metric | LLMs Mean | SLMs Mean | Best SLM |
|--------|-----------|-----------|----------|
| Fatigue Acc↑ | 52.4% | 46.1% | TinyLlama: 63.2% |
| Calories MAE↓ | 41.6 | **7.57** | Gemma-2-2B: 2.80 |
| Stress MAE↓ | **0.44** | 0.57 | Phi-3-mini: 0.40 |
| Activity Acc↑ | **28.2** | 21.8 | Gemma-2-2B: 34.4% |

### Ablation Study

**On-device Deployment Efficiency (iPhone 15 Pro Max):**

| Model | TTFT(s)↓ | ITPS(t/s)↑ | OET(s)↓ | OTPS(t/s)↑ | RAM(GB)↓ |
|-------|----------|------------|---------|------------|----------|
| Llama-2-7B | 29.12 | 24.74 | 27.85 | 3.04 | 7.15 |
| Phi-3-mini-4k | 6.39 | 112.39 | 0.96 | 13.49 | 6.48 |
| TinyLlama-1.1B | **1.37** | **527.01** | **0.35** | **45.89** | **5.17** |

**Speedup**: TinyLlama vs. Llama-2-7B: 21× faster TTFT, 79× faster OET, ITPS improvement exceeding 2000%.

### Key Findings

- Under zero-shot evaluation, SLMs already match or surpass LLMs on most health tasks, particularly in stress, readiness, and fatigue prediction.
- Regression tasks (calorie estimation) remain challenging for SLMs in the zero-shot setting, but after instruction fine-tuning, SLMs substantially outperform LLMs (MAE: 7.57 vs. 41.6).
- A "collapse" phenomenon is observed in few-shot learning, where certain SLMs experience abrupt performance degradation under specific few-shot configurations.
- Mental health tasks (anxiety, depression) benefit more from additional few-shot examples than physiological monitoring tasks.
- Instruction fine-tuned SLMs exhibit class imbalance bias, tending to predict the majority class.

## Highlights & Insights

- **High practical value**: Directly addresses the critical question of whether SLMs are viable for mobile health monitoring.
- **End-to-end validation**: Provides a complete pipeline from model evaluation to real iPhone deployment, rather than purely theoretical analysis.
- **Remarkable efficiency gains**: TinyLlama achieves a first-token latency of only 1.37 seconds on iPhone, 21× faster than 7B models.
- **Privacy advantage**: On-device inference entirely eliminates the privacy risk of uploading health data to the cloud.
- **Comprehensiveness**: 9 SLMs × 8 tasks × 3 evaluation paradigms constitutes a highly systematic benchmark.

## Limitations & Future Work

- Evaluation is limited to 3 public datasets, with insufficient coverage of health scenarios (lacking important tasks such as cardiovascular and diabetes monitoring).
- Severe class imbalance significantly impacts fine-tuning performance, with no concrete mitigation proposed.
- Root cause analysis of the few-shot collapse phenomenon is insufficiently thorough.
- Deployment is tested only on the iPhone 15 Pro Max; other mobile platforms (Android devices, smartwatches, etc.) are not covered.
- The safety of SLMs is not evaluated — erroneous health predictions may have serious consequences.
- The impact of 4-bit quantization on health prediction accuracy is not analyzed in detail.

## Related Work & Insights

- Extends the research line of Health-LLM with a focus on more efficient deployment strategies.
- Complements MobileAIBench, which evaluates only general NLP tasks.
- Motivates a "small but precise" paradigm for mobile health: fine-tuned 1–4B models can satisfy the majority of practical scenarios.
- The combination of on-device SLMs and federated learning may represent a key architectural direction for future privacy-preserving health AI.

## Rating

- Novelty: ⭐⭐⭐ Applying SLMs to health monitoring is a natural extension, though the first systematic benchmark represents a meaningful contribution
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-model, multi-task, multi-paradigm evaluation with on-device deployment validation, though the number of datasets is limited
- Writing Quality: ⭐⭐⭐⭐ Clear structure with rich tables, though some analyses are primarily descriptive
- Value: ⭐⭐⭐⭐ Important reference value for the practical deployment of mobile health AI

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] FedRW: Efficient Privacy-Preserving Data Reweighting for Enhancing Federated Learning of Language Models](fedrw_efficient_privacy-preserving_data_reweighting_for_enhancing_federated_lear.md)
- [\[ICLR 2026\] AudioTrust: Benchmarking the Multifaceted Trustworthiness of Audio Large Language Models](../../ICLR2026/llm_safety/audiotrust_benchmarking_the_multifaceted_trustworthiness_of_audio_large_language.md)
- [\[NeurIPS 2025\] Distributive Fairness in Large Language Models: Evaluating Alignment with Human Values](distributive_fairness_in_large_language_models_evaluating_alignment_with_human_v.md)
- [\[NeurIPS 2025\] Exploring the Limits of Strong Membership Inference Attacks on Large Language Models](exploring_the_limits_of_strong_membership_inference_attacks_on_large_language_mo.md)
- [\[NeurIPS 2025\] ALMGuard: Safety Shortcuts and Where to Find Them as Guardrails for Audio-Language Models](almguard_safety_shortcuts_and_where_to_find_them_as_guardrails_for_audio-languag.md)

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Small Language Models for Efficient Agentic Tool Calling: Outperforming Large Models with Targeted Fine-tuning
description: >-
  [AAAI 2026][LLM Reasoning][Small Language Models] By applying a single-epoch supervised fine-tuning (SFT) on OPT-350M, this work achieves a 77.55% pass rate on ToolBench…
tags:
  - "AAAI 2026"
  - "LLM Reasoning"
  - "Small Language Models"
  - "Tool Calling"
  - "SFT Fine-tuning"
  - "OPT-350M"
  - "ToolBench"
date: 2026-05-08
content_hash: ab1355806058aaac
---

# Small Language Models for Efficient Agentic Tool Calling: Outperforming Large Models with Targeted Fine-tuning

**Conference**: AAAI 2026
**arXiv**: [2512.15943](https://arxiv.org/abs/2512.15943)  
**Code**: None  
**Area**: LLM Reasoning
**Keywords**: Small Language Models, Tool Calling, SFT Fine-tuning, OPT-350M, ToolBench

## TL;DR

By applying a single-epoch supervised fine-tuning (SFT) on OPT-350M, this work achieves a 77.55% pass rate on ToolBench, substantially outperforming large-model baselines such as ChatGPT-CoT (26%) and ToolLLaMA-DFS (30.18%), demonstrating that small models with targeted fine-tuning can significantly surpass general-purpose large models on specific tasks.

## Background & Motivation

As generative AI is deployed at scale in enterprise settings, **model cost optimization and operational efficiency** have become critical factors for sustainability and accessibility. Despite their capabilities, large language models face three core challenges:

**High infrastructure costs**: Running SOTA-level LLMs demands substantial GPU resources, and closed-source API fees are considerable.

**Data privacy and latency risks**: Reliance on closed-source APIs introduces privacy, latency, and robustness concerns.

**Capability surplus**: Many specific tasks (e.g., tool calling) do not require the full generalist capabilities of large models.

This motivates a natural research question: **Can small language models (SLMs), when trained with targeted fine-tuning, match or exceed LLM performance on focused tasks?**

The paper analyzes why large models underperform on tool calling:
- **Parameter dilution**: The vast majority of billions of parameters are optimized for general language understanding rather than tool calling.
- **Over-generalization**: When precise API-format outputs are required, large models tend to produce verbose explanations or creative alternatives.
- **Unfocused behavior**: Lack of dedicated learning of the structured Thought-Action-Observation pattern.

## Method

### Overall Architecture

The core approach applies the Hugging Face TRL (Transformer Reinforcement Learning) SFT Trainer to perform single-epoch supervised fine-tuning on `facebook/opt-350m`. Training is conducted on Amazon SageMaker (ml.g5.8xlarge instance).

### Key Designs

#### 1. Data Preparation and Formatting

The ToolBench dataset is used, comprising over 16,000 real-world APIs from RapidAPI Hub along with corresponding instruction-solution pairs. The data transformation pipeline:
- Concatenates system prompts, user queries, and assistant responses using appropriate delimiters.
- Creates structured training sequences conforming to the Thought-Action-Action Input pattern.
- Results in **187,542 training samples** after transformation.

#### 2. Hyperparameter Configuration ("High-Learning, High-Stability" Strategy)

| Hyperparameter | Value | Design Motivation |
|---|---|---|
| Learning Rate | $5 \times 10^{-5}$ | Conservative rate for stable adaptation |
| Warmup Steps | 100 | Avoids instability in early training |
| Effective Batch Size | 32 (via 4-step gradient accumulation) | Provides robust gradient estimates |
| Gradient Clipping | max_norm=0.3 | Prevents training instability |
| Precision | FP16 mixed precision | Reduces memory footprint |
| Optimizer | AdamW (weight_decay=0.01) | Handles sparse gradients and prevents overfitting |
| Training Epochs | 1 | Promotes generalization over memorization |

Total training steps = 187,542 / 32 ≈ 5,860 steps.

This "high-learning, high-stability" configuration enables the model to extract maximum information from ToolBench's high-quality samples within a single epoch, while encouraging the learning of generalizable tool-use patterns rather than rote memorization.

#### 3. Evaluation Framework — ToolEval

The ToolEval automatic evaluation framework is employed, using ChatGPT as the evaluator.

**Two primary metrics**:
- **Pass Rate**: The proportion of instructions successfully completed within a limited API call budget.
- **Win Rate**: A comparative assessment of solution quality across dimensions such as informativeness, factual accuracy, and reasoning quality.

**Six test categories** (1,100 queries total):
- G1-instruction (200): Single-tool — unseen instructions
- G1-category (200): Single-tool — unseen categories
- G1-tool (200): Single-tool — completely unseen tools
- G2-instruction (200): Multi-tool — intra-category scenarios
- G2-category (200): Multi-tool — cross-category scenarios
- G3-instruction (100): Multi-tool — intra-set scenarios

### Loss & Training

Standard SFT cross-entropy loss is applied under a causal language modeling framework, training the model to generate correct Thought-Action-Action Input sequences. Inference parameters: temperature=0.1, max_seq_length=8192, max_reasoning_iterations=10.

## Key Experimental Results

### Main Results

| Model | Parameters | Overall Pass Rate | Gap |
|---|---|---|---|
| **Our SLM** | **350M** | **77.55%** | — |
| ToolLLaMA-DFS | 7B | 30.18% | -47.37% |
| ChatGPT-CoT | 175B | 26.00% | -51.55% |
| ToolLLaMA-CoT | 7B | 16.27% | -61.28% |
| Claude-CoT | 52B | 2.73% | -74.82% |

**Per-category results**:

| Category | Ours | TLLM-DFS | ChatGPT | TLLM-CoT | Claude |
|---|---|---|---|---|---|
| G1_instr | 78.5 | 32.5 | 33.0 | 16.0 | 3.5 |
| G1_cat | 74.0 | 32.5 | 29.5 | 21.5 | 3.0 |
| G1_tool | 79.0 | 28.0 | 29.5 | 14.5 | 2.5 |
| G2_cat | 80.5 | 32.5 | 24.5 | 16.5 | 1.5 |
| G2_instr | 74.5 | 29.5 | 24.0 | 18.0 | 2.5 |
| G3_instr | 80.0 | 22.0 | 5.0 | 6.0 | 4.0 |
| **Average** | **77.6** | 30.2 | 26.0 | 16.3 | 2.7 |

### Ablation Study

No formal ablation tables are provided; however, the paper discusses the sources of performance advantages:

| Performance Factor | Analysis | Notes |
|---|---|---|
| Parameter Efficiency | All 350M parameters dedicated to tool calling | Large model parameters spread across general understanding |
| Behavioral Focus | Learns to suppress irrelevant behaviors | Precisely generates structured API calls |
| Evaluation Alignment | Training data highly consistent with ToolBench evaluation format | Directly optimizes for the target task |
| Performance Variance | 74%–80.5% (only 6.5% range) | Strong cross-category generalization |

### Key Findings

1. **Paradigm shift in parameter efficiency**: A 350M-parameter model surpasses all baselines by 47–75 percentage points across all categories, using 20–500× fewer parameters.
2. **Cross-category consistency**: Performance variance across 6 test categories is only 6.5% (74%–80.5%), indicating that the model has learned generalizable reasoning patterns.
3. **Greatest advantage in multi-tool scenarios**: The most pronounced gains appear on G3-instruction (the most complex multi-tool scenario) — 80% vs. ChatGPT's 5%.

## Highlights & Insights

1. **Counter-intuitive finding**: On specific tasks, a "small and specialized" model can substantially outperform a "large and general" one — a 350M-parameter model achieves nearly 3× the pass rate of 175B-parameter ChatGPT.
2. **Economic single-epoch training**: Only 5,860 training steps are needed to reach SOTA, with minimal training and inference costs, enabling a "democratization" of AI capability.
3. **The concept of parameter-task alignment** — 350M parameters appear to be the "sweet spot" for tool-calling tasks, providing sufficient capacity to learn API interaction patterns without the output inconsistencies introduced by excess complexity.

## Limitations & Future Work

1. **Generalizability concerns**: The model is specifically optimized for ToolBench; generalization to other tool-calling frameworks or real-world API ecosystems remains unverified.
2. **Limited contextual understanding**: 350M parameters may be insufficient for tool selection scenarios requiring complex contextual reasoning.
3. **API evolution problem**: As APIs are updated, specialized models may require frequent retraining.
4. **Evaluation coupling**: Training and evaluation data originate from the same ToolBench ecosystem, raising the risk of overfitting to the evaluation protocol.
5. **Absence of rigorous ablations**: No systematic comparisons across different training epochs, model sizes, or hyperparameter configurations are provided.

## Related Work & Insights

- **Toolformer** pioneered the paradigm of language models using external tools, teaching models to invoke APIs through self-supervised learning.
- **ReAct** introduced the interleaved reasoning-and-acting framework, which has become the standard paradigm for tool-augmented AI systems.
- **ToolLLM** extended tool integration to 16,000+ real-world APIs and established the ToolBench standard evaluation benchmark.
- **Gorilla** demonstrated early on that small specialized models can outperform large generalist models in specific domains.
- The results in this paper challenge the "bigger is better" assumption underlying scaling laws, with important implications for deploying AI in resource-constrained environments.

## Rating

- Novelty: ⭐⭐⭐⭐ — The methodology is relatively straightforward (standard SFT); the primary contribution lies in the empirical findings.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Six-category evaluation is provided, but ablations and multi-scale comparisons are lacking.
- Writing Quality: ⭐⭐⭐⭐ — Conclusions are clear, though portions of the analysis are somewhat subjective.
- Value: ⭐⭐⭐⭐⭐ — Provides compelling empirical evidence for the potential of SLMs on specialized tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] TInR: Exploring Tool-Internalized Reasoning in Large Language Models](../../ACL2026/llm_reasoning/tinr_exploring_tool-internalized_reasoning_in_large_language_models.md)
- [\[ICLR 2026\] Efficient Test-Time Scaling for Small Vision-Language Models](../../ICLR2026/llm_reasoning/efficient_test-time_scaling_for_small_vision-language_models.md)
- [\[ICML 2026\] DenseSteer: Steering Small Language Models towards Dense Math Reasoning](../../ICML2026/llm_reasoning/densesteer_steering_small_language_models_towards_dense_math_reasoning.md)
- [\[AAAI 2026\] The Curious Case of Analogies: Investigating Analogical Reasoning in Large Language Models](the_curious_case_of_analogies_investigating_analogical_reasoning_in_large_langua.md)
- [\[AAAI 2026\] NeSTR: A Neuro-Symbolic Abductive Framework for Temporal Reasoning in Large Language Models](nestr_a_neuro-symbolic_abductive_framework_for_temporal_reasoning_in_large_langu.md)

</div>

<!-- RELATED:END -->

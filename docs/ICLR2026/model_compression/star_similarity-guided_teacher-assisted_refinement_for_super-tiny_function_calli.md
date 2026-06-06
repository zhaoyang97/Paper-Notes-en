---
title: >-
  [Paper Note] STAR: Similarity-guided Teacher-Assisted Refinement for Super-Tiny Function Calling Models
description: >-
  [ICLR 2026][Model Compression][Knowledge Distillation] This paper proposes the STAR framework, which combines Constrained Knowledge Distillation (CKD) and Similarity-guided Reinforcement Learning (Sim-RL) to effectively…
tags:
  - "ICLR 2026"
  - "Model Compression"
  - "Knowledge Distillation"
  - "Reinforcement Learning"
  - "Function Calling"
  - "Super-Tiny Models"
  - "Similarity Reward"
date: 2026-05-08
content_hash: 4bb1dbe4de6b8c60
---

# STAR: Similarity-guided Teacher-Assisted Refinement for Super-Tiny Function Calling Models

**Conference**: ICLR 2026
**arXiv**: [2602.03022](https://arxiv.org/abs/2602.03022)  
**Code**: [github.com/Qwen-Applications/STAR](https://github.com/Qwen-Applications/STAR)  
**Area**: Model Compression / Knowledge Distillation
**Keywords**: Knowledge Distillation, Reinforcement Learning, Function Calling, Super-Tiny Models, Similarity Reward

## TL;DR

This paper proposes the STAR framework, which combines Constrained Knowledge Distillation (CKD) and Similarity-guided Reinforcement Learning (Sim-RL) to effectively transfer function calling capabilities from large models to super-tiny models at the 0.6B scale, achieving substantial improvements over baselines on BFCL and ACEBench.

## Background & Motivation

- Function calling capability in LLMs is critical for AI agents, yet deploying large models remains resource-constrained.
- Directly applying SFT+RL to small models presents **three key challenges**:
  1. Small models tend to **overfit** SFT data, memorizing specific patterns rather than generalizing.
  2. Direct RL training is **unstable**.
  3. Combining KD and RL introduces new issues: RKL under top-k truncation causes training collapse, binary rewards are ill-suited for multi-solution tasks, and joint optimization of KD and RL is inherently difficult.

## Method

### Overall Architecture

The STAR training curriculum consists of two stages: **Model Distillation** → **Model Refinement**.

### CKD: Constrained Knowledge Distillation

#### Finding 1: RKL + Top-k Truncation Leads to Training Collapse

- Top-k FKL ignores tail distributions → stable training.
- Top-k RKL imposes unstable supervision on tokens outside $V_k(x)$ → catastrophic collapse.

#### Finding 2: The "Hidden Cost" of RKL

- The mode-seeking nature of RKL aggressively suppresses the student's tail distribution → **reduced output entropy**.
- Low entropy = weak exploration capacity → degraded downstream RL performance.

#### CKD Loss Function

$$\mathcal{L}_{CKD} = \mathcal{L}_{FKL\text{-}k} + \lambda_{tail} \mathcal{L}_{tail}$$

where:

$$\mathcal{L}_{FKL\text{-}k} = \sum_{x} \sum_{v \in V_k(x)} P_T(v|x) \log \frac{P_T(v|x)}{P_S(v|x)}$$

$$\mathcal{L}_{tail} = \sum_{x} \sum_{v \in V_m(x) \setminus V_k(x)} P_S(v|x)$$

- Only penalizes tokens the student assigns high probability to but the teacher deems irrelevant → suppresses "confident errors."
- Does not force the long-tail distribution to zero → preserves exploration capacity required for downstream RL.

### Sim-RL: Similarity-guided Reinforcement Learning

#### Reward Design

- **Format Reward** $R_{format}$: Binary; checks `<think>`/`<tool_call>` tags, JSON format, and function name validity.
- **Function Call Reward** $R_{fc}$: Compares predicted and ground-truth function call sequences based on an IoU principle.

$$R_{fc} = \frac{\sum_{i=1}^{\min(m,n)} \text{sim}(p_i, g_{\sigma(i)})}{|P| + |G| - |P \cap G|}$$

The parameter-level similarity function $\text{sim}(p,g)$ applies different metrics for different types (ROUGE-L for strings, exact match for numeric values).

- **Response Reward** $R_{response}$: ROUGE-L F1 for plain-text responses.
- **Total Reward**: $R = (R_{format} - 1) + R_{format} \cdot (R_{fc} + R_{response})$, ranging in $[-1, 1]$.

#### Optimization

GRPO is used with a DAPO-inspired filtering mechanism that discards homogeneous groups with all-zero or all-one rewards.

### STAR Training Curriculum

1. Fine-tune the teacher model (Qwen3-8B) with Sim-RL to adapt it to the distillation data.
2. Distill teacher knowledge into the student model via CKD.
3. Refine the student policy with Sim-RL.

## Key Experimental Results

### BFCLv3 Benchmark (Qwen3-0.6B)

| Method | Overall Acc | Non-Live | Live | Multi Turn |
|--------|-------------|----------|------|------------|
| Base-model | 47.33 | 71.81 | 65.66 | 1.88 |
| SFT | 44.58 | 66.29 | 62.15 | 1.62 |
| SFT-think | 47.59 | — | — | — |
| FKL | — | — | — | — |
| ToolRL | — | — | — | — |
| **STAR** | **Best** | — | — | — |

### Key Achievements of STAR 0.6B

| Comparison | BFCL Relative Gain | ACEBench Relative Gain |
|------------|--------------------|------------------------|
| vs. baseline | +9.2% | >50% |
| vs. all open-source models <1B | **Best** | **Best** |
| vs. select larger models | Surpasses | Surpasses |

### Ablation Study

| CKD Component | Effect |
|---------------|--------|
| Top-k FKL alone | Stable but limited downstream RL gain |
| Top-k RKL/AKL | Training collapse |
| CKD (FKL + tail penalty) | **Stable + preserves exploration + maximum RL gain** |

### Key Findings

1. The tail penalty in CKD preserves sufficient exploration capacity while maintaining training stability.
2. The continuous rewards in Sim-RL significantly outperform binary rewards on multi-solution tasks.
3. A carefully designed KD+RL curriculum enables a 0.6B model to surpass certain larger models.
4. Teacher correction via Sim-RL adaptation prior to distillation positively contributes to distillation quality.

## Highlights & Insights

- **In-depth Diagnostics**: Systematic analysis of the behavioral differences between FKL and RKL under top-k truncation and their downstream effects on RL.
- **Sophisticated Reward Design**: Parameter-level similarity rewards are more flexible than AST parsing and provide richer signals than binary rewards.
- **High Practicality**: The 0.6B super-tiny model achieves deployable-level function calling performance.
- **KD-to-RL Transition Design**: CKD is specifically designed to retain exploration capacity in order to benefit the subsequent RL stage.

## Limitations & Future Work

- Validation is limited to the Qwen model family; cross-architecture generalizability remains unknown.
- The reward design relies on a specific function calling format (Qwen tool calling template).
- Distillation quality is directly bounded by the teacher model's capability.
- The training pipeline is relatively complex, comprising three stages: teacher fine-tuning → CKD → Sim-RL.

## Related Work & Insights

- Knowledge Distillation: GKD, AKL, and discussions on FKL vs. RKL.
- Function Calling: BFCL benchmark, ToolRL.
- Small model training: LUFFY (hybrid offline/online), etc.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Both the tail penalty in CKD and the fine-grained rewards in Sim-RL represent genuine technical contributions.
- **Technical Depth**: ⭐⭐⭐⭐ — In-depth analysis of KL divergence behavior with gradient-level theoretical support.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multi-scale models, comprehensive ablations, and two mainstream benchmarks.
- **Value**: ⭐⭐⭐⭐⭐ — A deployable 0.6B function calling model for on-device applications has immense practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Critique-Guided Distillation for Robust Reasoning via Refinement](../../ICML2026/model_compression/critique-guided_distillation_for_robust_reasoning_via_refinement.md)
- [\[ACL 2026\] Find Your Optimal Teacher: Personalized Data Synthesis via Router-Guided Multi-Teacher Distillation](../../ACL2026/model_compression/find_your_optimal_teacher_personalized_data_synthesis_via_router-guided_multi-te.md)
- [\[ICLR 2026\] Unveiling Super Experts in Mixture-of-Experts Large Language Models](unveiling_super_experts_in_mixture-of-experts_large_language_models.md)
- [\[ICLR 2026\] SERE: Similarity-based Expert Re-routing for Efficient Batch Decoding in MoE Models](sere_similarity-based_expert_re-routing_for_efficient_batch_decoding_in_moe_mode.md)
- [\[ICLR 2026\] Reference-Guided Machine Unlearning](reference-guided_machine_unlearning.md)

</div>

<!-- RELATED:END -->

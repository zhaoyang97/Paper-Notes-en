---
title: >-
  [Paper Note] The Path Not Taken: RLVR Provably Learns Off the Principals
description: >-
  [NeurIPS 2025 Workshop on Efficient Reasoning (Spotlight)][Reinforcement Learning][RLVR] This paper proposes the Three-Gate Theory to explain the apparent sparsity of parameter updates in RLVR, demonstrating that RLVR learns along off-principal directions in weight space — a fundamentally different optimization mechanism from SFT — and that directly transplanting SFT-era PEFT methods to RLVR is therefore flawed.
tags:
  - NeurIPS 2025 Workshop on Efficient Reasoning (Spotlight)
  - Reinforcement Learning
  - RLVR
  - training dynamics
  - Three-Gate Theory
  - parameter-efficient fine-tuning
  - SFT comparison
date: 2026-05-08
content_hash: 001bf8260f8a32b4
---

# The Path Not Taken: RLVR Provably Learns Off the Principals

**Conference**: NeurIPS 2025 Workshop on Efficient Reasoning (Spotlight)  
**arXiv**: [2511.08567](https://arxiv.org/abs/2511.08567)  
**Code**: None  
**Area**: LLM Training / Reinforcement Learning  
**Keywords**: RLVR, training dynamics, Three-Gate Theory, parameter-efficient fine-tuning, SFT comparison  

## TL;DR

This paper proposes the Three-Gate Theory to explain the apparent sparsity of parameter updates in RLVR, demonstrating that RLVR learns along off-principal directions in weight space — a fundamentally different optimization mechanism from SFT — and that directly transplanting SFT-era PEFT methods to RLVR is therefore flawed.

## Background & Motivation

### The Success and Paradox of RLVR
Reinforcement Learning with Verifiable Rewards (RLVR) has become the standard approach for enhancing the reasoning capabilities of large language models. Yet a puzzling phenomenon is consistently observed in practice: **RLVR appears to modify only a small fraction of parameters**, while yielding substantial improvements in reasoning performance. This sparsity paradox motivates a deeper investigation into RLVR's actual learning mechanism.

### Limitations of Prior Work
- Prior work treats the observed sparsity of RLVR parameter updates as an intrinsic characteristic.
- There is no systematic, parameter-level characterization of the update dynamics.
- SFT-era parameter-efficient fine-tuning (PEFT) methods have been directly applied to RLVR without theoretical justification.

### Core Insight
The observed sparsity is a **surface artifact** arising from optimization bias induced by model conditioning. For a fixed pretrained model, updates consistently localize to preferred parameter regions with high cross-experiment reproducibility.

## Method

### Overall Architecture

The paper proposes the **Three-Gate Theory** to explain the parameter update dynamics of RLVR:

```
Pretrained Model → [Gate I: KL Anchor] → [Gate II: Model Geometry] → [Gate III: Precision] → Observed "Sparsity"
```

### Key Designs

#### Gate I: KL Anchor
- RLVR constrains policy updates via KL divergence penalty, keeping parameter changes close to the pretrained model.
- The KL constraint imposes a global upper bound on update magnitude.
- Formalized as: $\theta_{t+1} = \theta_t + \eta \cdot \nabla_\theta J(\theta) - \lambda \nabla_\theta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$

#### Gate II: Model Geometry
- The weight matrices of pretrained models exhibit a specific spectral structure (principal directions = directions of large singular values).
- Gradient updates from RLVR are **steered toward low-curvature, spectrum-preserving subspaces** — i.e., off-principal directions.
- This implies that RLVR preserves the core spectral structure encoding pretrained knowledge, performing fine-grained adjustments along "peripheral" directions.
- Formal characterization: given $W = U \Sigma V^T$, RLVR updates concentrate primarily along directions with smaller $\sigma_i$.

#### Gate III: Precision
- Non-preferred regions (principal directions) also receive minor updates, but these are masked by floating-point precision.
- Under low precision (e.g., BF16), minor updates along principal directions are absorbed by quantization noise.
- The result is that observed parameter changes appear highly sparse, whereas precision conceals a globally distributed fine-tuning along off-principal directions.

### Parameter-Level Verification

The paper provides the first parameter-level characterization of RLVR learning dynamics, validating the following key properties:
1. **Minimal spectral drift**: the spectral distribution of weight matrices changes only marginally after RLVR.
2. **Reduced principal subspace rotation**: the rotation angle of leading singular vectors is far smaller than that observed under SFT.
3. **Alignment of off-principal updates**: update directions are highly consistent across different datasets and RL recipes.

### Loss & Training

- **RLVR objective**: maximize a verifiable reward function $R(y, y^*)$ subject to a KL divergence constraint:
  $$\max_\theta \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot|x)} [R(y, y^*)] - \lambda D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$$
- **SFT baseline**: directly minimizes cross-entropy loss, with updates concentrating along principal directions.

## Key Experimental Results

### Main Results

Parameter update patterns of RLVR versus SFT are compared across multiple LLMs:

| Method | Spectral Drift ↓ | Principal Subspace Rotation ↓ | Update Sparsity | Reasoning Accuracy ↑ |
|--------|-----------------|-------------------------------|-----------------|----------------------|
| SFT | 0.42 | 12.3° | Low (diffuse) | +3.2% |
| RLVR (GRPO) | **0.08** | **2.1°** | High (concentrated) | **+8.7%** |
| RLVR (PPO) | **0.11** | **3.4°** | High (concentrated) | **+7.9%** |
| RLVR (REINFORCE) | **0.09** | **2.8°** | High (concentrated) | **+8.1%** |

Cross-dataset update consistency analysis:

| Training Dataset | Cosine Similarity to Reference Update Direction | Reasoning Score |
|-----------------|------------------------------------------------|-----------------|
| GSM8K | 0.93 | 82.4 |
| MATH | 0.91 | 76.8 |
| ARC-Challenge | 0.89 | 79.1 |
| Mixed Dataset | 0.95 | 84.2 |

### Ablation Study

Comparison of PEFT methods applied to RLVR:

| PEFT Method | SFT Performance | RLVR Performance | Performance Gap |
|-------------|-----------------|------------------|-----------------|
| Full Fine-tuning | Baseline | Baseline | — |
| LoRA (rank=16) | -1.2% | **-5.8%** | SFT superior to RLVR |
| LoRA (rank=64) | -0.8% | **-3.2%** | SFT superior to RLVR |
| Sparse Fine-tuning (top-10%) | -0.5% | **-4.1%** | SFT superior to RLVR |
| Sparse Fine-tuning (top-30%) | -0.3% | **-2.7%** | SFT superior to RLVR |

### Key Findings

1. **RLVR and SFT exhibit fundamentally different update patterns**: SFT updates concentrate along principal directions (large singular value directions), whereas RLVR operates along off-principal directions.
2. **High cross-experiment consistency**: different RL algorithms (PPO, GRPO, REINFORCE) and different datasets produce highly consistent update patterns.
3. **Failure of PEFT methods**: methods such as LoRA are designed under a low-rank assumption suited to SFT's principal-direction updates, but RLVR's off-principal updates are inherently diffuse, causing low-rank approximations to discard critical information.
4. **SFT underperforms RLVR**: because SFT directly modifies the spectral structure along principal directions, it may degrade pretrained knowledge.

## Highlights & Insights

1. **Elegance of the Three-Gate Theory**: a concise three-layer mechanism explains the surface sparsity observed in RLVR.
2. **First parameter-level characterization**: provides a white-box understanding of RLVR training dynamics.
3. **Significant practical implications**: directly refutes the popular practice of applying SFT-PEFT methods to RLVR.
4. **Points toward new directions**: calls for the design of geometry-aware, RLVR-native learning algorithms rather than recycling SFT-era heuristics.
5. **Closed theory–experiment loop**: forms a complete chain of argument from mathematical derivation to large-scale empirical validation.

## Limitations & Future Work

1. **Analysis limited to parameter update patterns**: no direct link is established to specific performance differences on downstream tasks.
2. **Mitigation and improvement proposals are preliminary**: while the failure modes of PEFT methods are identified, no complete alternative is proposed.
3. **Model scale constraints**: validation is conducted primarily on medium-scale models; very large-scale models remain uncovered.
4. **Theoretical rigor**: certain components of the Three-Gate Theory rely on empirical observations rather than formal proofs.
5. **Absence of RLVR-native PEFT design**: a direction is identified but not realized.

## Related Work & Insights

- **RLVR methods**: DeepSeek-R1, GRPO, and related work demonstrate the effectiveness of RLVR; this paper explains *why* it works.
- **Parameter-efficient fine-tuning**: LoRA, QLoRA, and similar methods achieve strong results in SFT, but this paper reveals their limitations in the RLVR setting.
- **Training dynamics**: complements Aghajanyan et al.'s work on intrinsic dimensionality.
- **Inspired direction**: designing novel PEFT methods that exploit off-principal directional structure (e.g., off-principal LoRA).

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First parameter-level understanding of RLVR.
- **Theoretical Depth**: ⭐⭐⭐⭐ — Three-Gate Theory framework is clear; partial theoretical rigor.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Validated across multiple algorithms and datasets.
- **Practical Impact**: ⭐⭐⭐⭐⭐ — Directly guides RLVR training practice.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Fluent narrative and well-organized structure.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Behaviour Policy Optimization: Provably Lower Variance Return Estimates for Off-Policy Reinforcement Learning](../../AAAI2026/reinforcement_learning/behaviour_policy_optimization_provably_lower_variance_return_estimates_for_off-p.md)
- [\[NeurIPS 2025\] Bootstrap Off-policy with World Model](boundary-to-region_supervision_for_offline_safe_reinforcement_learning.md)
- [\[NeurIPS 2025\] Retrosynthesis Planning via Worst-path Policy Optimisation in Tree-structured MDPs](retrosynthesis_planning_via_worst-path_policy_optimisation_in_tree-structured_md.md)
- [\[NeurIPS 2025\] Interactive and Hybrid Imitation Learning: Provably Beating Behavior Cloning](interactive_and_hybrid_imitation_learning_provably_beating_behavior_cloning.md)
- [\[ACL 2026\] GeoRA: Geometry-Aware Low-Rank Adaptation for RLVR](../../ACL2026/reinforcement_learning/geora_geometry-aware_low-rank_adaptation_for_rlvr.md)

<!-- RELATED:END -->

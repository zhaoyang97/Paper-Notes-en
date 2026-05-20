---
title: >-
  [Paper Note] ParoQuant: Pairwise Rotation Quantization for Efficient Reasoning LLM Inference
description: >-
  [ICLR 2026][Model Compression][Post-training quantization] ParoQuant is proposed to eliminate weight outliers via hardware-efficient and optimizable independent Givens rotations combined with channel scaling…
tags:
  - "ICLR 2026"
  - "Model Compression"
  - "Post-training quantization"
  - "Givens rotation"
  - "Reasoning LLM"
  - "Quantization efficiency"
  - "Algorithm-system co-design"
date: 2026-05-08
content_hash: c1bd396bd33db663
---

# ParoQuant: Pairwise Rotation Quantization for Efficient Reasoning LLM Inference

**Conference**: ICLR 2026
**arXiv**: [2511.10645](https://arxiv.org/abs/2511.10645)  
**Code**: [Project Page](https://paroquant.z-lab.ai)  
**Area**: Model Compression
**Keywords**: Post-training quantization, Givens rotation, Reasoning LLM, Quantization efficiency, Algorithm-system co-design

## TL;DR

ParoQuant is proposed to eliminate weight outliers via hardware-efficient and optimizable independent Givens rotations combined with channel scaling, achieving high-accuracy, low-overhead 4-bit weight quantization for reasoning LLMs.

## Background & Motivation

LLM quantization faces a fundamental accuracy-efficiency trade-off:
- **AWQ**: Fast but incurs significant accuracy loss (e.g., 2.8% drop on MMLU-Pro for Qwen3-4B); the long chain-of-thought in reasoning LLMs causes quantization errors to accumulate progressively.
- **QTIP**: High accuracy but approximately 30% slower than AWQ due to substantial overhead introduced by Hadamard transforms.
- Reasoning models generate tens of thousands of tokens, imposing stringent requirements on both quantization accuracy and efficiency.

Core observations:

**Rotation effectively suppresses outliers**, but full rotation matrices are computationally expensive.

**Sparsely parameterized rotations are equally effective** — retaining only the top-10% channel pairs suffices to match full rotation performance.

## Method

### Overall Architecture

ParoQuant designs a Scaled Pairwise Rotation transform composed of multiple independent rotations and channel scaling, paired with layer-wise optimization and efficient inference kernels for end-to-end acceleration.

### Key Designs

1. **Givens Rotation Decomposition**:

    - A small set of channel pairs is selected: $\mathcal{P} = \{(i_1,j_1), \ldots, (i_m,j_m)\}$
    - Each pair undergoes a planar rotation: $\mathbf{W}^{(k)}[i,:] = \cos\theta_k \cdot \mathbf{W}^{(k-1)}[i,:] - \sin\theta_k \cdot \mathbf{W}^{(k-1)}[j,:]$
    - Only a small number of vectorized multiply-add operations are required, avoiding full matrix multiplication.

2. **Independent Rotation**:

    - Each channel is constrained to appear in at most one rotation pair ($P_k \cap P_l = \emptyset$).
    - All Givens rotations are fully parallelizable, fully exploiting GPU parallelism.
    - Naturally compatible with group quantization: independent rotations within each quantization group.

3. **Sequential Independent Rotations + Channel Scaling**:

    - A single independent rotation has only $n/2$ parameters, limiting its expressiveness.
    - $K$ sequential independent rotations (default $K=8$) are applied to enhance fitting capacity.
    - Channel scaling $\text{diag}(\boldsymbol{\alpha})$ directly equalizes channel magnitudes.
    - Final transform: $T_{\mathcal{P},\Theta,\boldsymbol{\alpha}}(\mathbf{W}) = (\prod_{t=1}^K R(\mathcal{P}_t, \Theta_t)) \cdot \text{diag}(\boldsymbol{\alpha}) \cdot \mathbf{W}$

### Loss & Training

- **Layer-wise optimization**: $\mathcal{L}(Q) = \|Q(D)(\mathbf{X'}) - D(\mathbf{X})\|$
- Two-stage optimization: rotation angles and scaling factors are optimized first, followed by QAT-like fine-tuning of weights and quantization parameters $s, z$.
- Each layer is optimized for 10 epochs using AdamW, with uniform sampling from three datasets (WikiText2, C4, RedPajama).
- Inference kernels exploit three-level parallelism: token dimension, channel group dimension, and rotation pair dimension.

## Key Experimental Results

### Main Results (Perplexity — W4G128 Quantization)

| Model | Method | WikiText2 PPL | C4 PPL | Inference Speedup |
|-------|--------|--------------|--------|-------------------|
| LLaMA-3-8B | FP16 | 5.54 | 7.10 | 1.0× |
| | AWQ | 5.92 | 7.42 | **2.4×** |
| | QTIP | 5.69 | 7.22 | 1.7× |
| | **ParoQuant** | **5.68** | **7.17** | **2.2×** |
| Qwen3-4B | AWQ | 7.36 | 7.89 | 2.4× |
| | QTIP | 7.09 | 7.68 | 1.7× |
| | **ParoQuant** | **7.03** | **7.63** | 2.2× |

### Reasoning Task Accuracy (DeepSeek-R1-distilled LLaMA-3.1-8B)

| Method | MMLU-Pro | GPQA Diamond | AIME-24 | AIME-25 | Avg. |
|--------|---------|-------------|---------|---------|------|
| FP16 | 52.4 | 43.9 | 56.7 | 40.0 | 48.3 |
| AWQ | 49.3 | 40.4 | 46.7 | 26.7 | 40.8 |
| **ParoQuant** | **52.5** | **41.4** | **53.3** | **36.7** | **46.0** |

### Key Findings

- ParoQuant outperforms AWQ by an average of 2.4% on reasoning tasks with less than 10% additional overhead.
- Accuracy matches QTIP (vector quantization SOTA) while being approximately 25% faster.
- Gains are especially pronounced on the Qwen3 series (1.7B–14B), where smaller models pose greater quantization challenges.

## Highlights & Insights

- Algorithm-system co-design: the independence constraint on rotations simultaneously preserves the mathematical optimization space and naturally suits GPU parallelism.
- Incisive analysis: only 10% of channel pairs are needed to match full rotation performance, revealing redundancy in orthogonal transforms.
- Particular attention is paid to reasoning LLMs, with thorough analysis of quantization error accumulation under long chain-of-thought generation.
- Online rotation kernels leverage shared memory and registers; multiple independent rotations can be fused into a single kernel call.

## Limitations & Future Work

- Validation is primarily limited to 4-bit linear quantization; 2–3 bit scenarios remain unexplored.
- The channel pair selection strategy for independent rotations (random sampling with deduplication) may be suboptimal.
- The number of rotations $K=8$ is empirically determined and may require tuning for different models.
- Lack of open-sourced code may limit community adoption.

## Related Work & Insights

- Distinction from QuaRot/SpinQuant: ParoQuant employs optimizable independent Givens rotations rather than fixed Hadamard transforms.
- Distinction from AWQ: the addition of rotation transforms on top of channel scaling substantially improves outlier suppression.
- Implication: in the era of reasoning LLMs, quantization methods must recalibrate the trade-off between accuracy and efficiency.

## Rating

- Novelty: ⭐⭐⭐⭐ The design of independent Givens rotations is novel and practical.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive validation across multiple models, tasks, and metrics.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly articulated, though the paper is notation-heavy.
- Value: ⭐⭐⭐⭐⭐ A practical solution for reasoning LLM quantization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] A State-Transition Framework for Efficient LLM Reasoning](a_state-transition_framework_for_efficient_llm_reasoning.md)
- [\[ICLR 2026\] Efficient Reasoning with Balanced Thinking](efficient_reasoning_with_balanced_thinking.md)
- [\[ICLR 2026\] Incentivizing Agentic Reasoning in LLM Judges via Tool-Integrated Reinforcement Learning](incentivizing_agentic_reasoning_in_llm_judges_via_tool-integrated_reinforcement_.md)
- [\[ICLR 2026\] The Geometry of LLM Quantization: GPTQ as Babai's Nearest Plane Algorithm](the_geometry_of_llm_quantization_gptq_as_babais_nearest_plane_algorithm.md)
- [\[ICLR 2026\] A Fano-Style Accuracy Upper Bound for LLM Single-Pass Reasoning in Multi-Hop QA](a_fano-style_accuracy_upper_bound_for_llm_single-pass_reasoning_in_multi-hop_qa.md)

</div>

<!-- RELATED:END -->

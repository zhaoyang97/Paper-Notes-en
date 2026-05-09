---
title: >-
  [Paper Note] RAIN-Merging: A Gradient-Free Method to Enhance Instruction Following in Large Reasoning Models with Preserved Thinking Format
description: >-
  [ICLR 2026][LLM Reasoning][Model Merging] To address the tension between strong reasoning capability and weak instruction following in large reasoning models (LRMs), this paper proposes RAIN-Merging, a two-stage gradient-free merging pipeline that preserves the thinking format via null-space projection and enhances instruction relevance via attention-guided per-module scaling coefficients. It integrates the capabilities of an instruction-tuned model (ITM) into an LRM without any gradient-based training, achieving consistent improvements across 4 instruction-following and 9 reasoning benchmarks.
tags:
  - ICLR 2026
  - LLM Reasoning
  - Model Merging
  - Instruction Following
  - Large Reasoning Models
  - Null-space Projection
  - Attention Guidance
date: 2026-05-08
content_hash: 80897ad55de9a527
---

# RAIN-Merging: A Gradient-Free Method to Enhance Instruction Following in Large Reasoning Models with Preserved Thinking Format

**Conference**: ICLR 2026
**arXiv**: [2602.22538](https://arxiv.org/abs/2602.22538)
**Code**: [https://github.com/K1nght/RAIN-Merging](https://github.com/K1nght/RAIN-Merging)
**Area**: LLM Reasoning
**Keywords**: Model Merging, Instruction Following, Large Reasoning Models, Null-space Projection, Attention Guidance

## TL;DR
To address the tension between strong reasoning capability and weak instruction following in large reasoning models (LRMs), this paper proposes RAIN-Merging, a two-stage gradient-free merging pipeline that preserves the thinking format via null-space projection and enhances instruction relevance via attention-guided per-module scaling coefficients. It integrates the capabilities of an instruction-tuned model (ITM) into an LRM without any gradient-based training, achieving consistent improvements across 4 instruction-following and 9 reasoning benchmarks.

## Background & Motivation
Large reasoning models (LRMs) such as DeepSeek-R1 and OpenAI-o1 excel at multi-step reasoning tasks including mathematical derivation and code generation, yet exhibit a paradoxical weakness in instruction following: despite generating lengthy logical chains, these models frequently disregard user-specified formats, constraints, or operational requirements. This limitation significantly undermines their practical utility in agentic scenarios and tool-integrated deployments.

A straightforward remedy is to continue fine-tuning LRMs via SFT, but constructing high-quality long chain-of-thought supervision data is prohibitively expensive and prone to capability degradation. Model merging, as a training-free lightweight alternative, fuses multiple capabilities through linear combination of task vectors. However, a fundamental **output structure mismatch** exists between LRMs and ITMs: LRMs use `<think>...</think>` tags to explicitly separate reasoning from response, while ITMs produce only final answers. Naive merging disrupts the structured reasoning format of LRMs.

**Core Idea**: Parameter-space analysis reveals that the principal subspaces of LRM and ITM task vectors are nearly orthogonal (similarity < 0.1), indicating low coupling and high mergeability. Building on this, the method addresses the format-preservation and instruction-enhancement objectives in two stages — Stage 1 applies null-space projection to protect the distribution of thinking tokens, and Stage 2 uses attention statistics to derive module-level scaling coefficients that amplify instruction-relevant components.

## Method

### Overall Architecture
RAIN-Merging (Reasoning-Aware Instruction-attention guided Null-space projection Merging) is a two-stage, gradient-free merging pipeline. Using LRM parameters $\theta_R$ as the anchor, the ITM task vector $\Delta_I = \theta_I - \theta_B$ is transformed and added to the LRM, yielding the merged model $\theta^* = \theta_R + \lambda \bigoplus_k \alpha^*_k \Delta_I^{\perp,k}$.

### Key Designs

1. **Stage 1: Reasoning-aware Null-space Projection**

   - **Function**: Projects the ITM task vector into the null space of the forward features at thinking special token positions.
   - **Design Motivation**: Ensures that the intermediate representations and final logits at thinking token positions remain consistent with those of the original LRM after merging, thereby preserving the `<think>...</think>` structured format.
   - **Mechanism**: For each sub-module $k$, a small reasoning calibration set (150 samples) is used to construct the forward feature operator $\Phi$ at thinking token positions. The orthogonal projection matrix is computed as $P^\perp(\Phi) = I - \Phi^T(\Phi\Phi^T)^+\Phi$, and the ITM task vector is projected as $\text{vec}(\Delta_I^{\perp,k}) = P^\perp(\Phi)\,\text{vec}(\Delta_I^k)$.
   - **Theoretical Guarantee**: Via second-order Taylor expansion of the softmax-KL divergence, the projected task vector is shown to satisfy $\mathcal{L}_\text{think} \approx 0$ (Proposition 1), meaning the distributional shift at thinking tokens after merging is negligible.
   - **Novelty**: Conventional merging methods (e.g., Task Arithmetic) ignore output distribution mismatch, resulting in 6.4% of generations missing the `</think>` token; the proposed method reduces this rate to 0%.

2. **Stage 2: Instruction-attention Guided Merging Coefficients**

   - **Function**: Computes adaptive per-module scaling coefficients $\alpha$ to amplify instruction-relevant components and suppress leakage.
   - **Design Motivation**: Instruction-following failures often stem from insufficient attention to the instruction span during decoding; different layers and heads respond heterogeneously to instructions.
   - **Mechanism**: Using 365 instruction calibration samples, the alignment and leakage of each attention head are computed. An instruction attention score $J = \text{alignment} - \rho \cdot \text{leakage}$ is defined, and a closed-form solution is derived via second-order Taylor expansion: $\alpha^*_k = \text{clip}(g^k / H^k)$.
   - **Novelty**: Existing activation-based merging methods (e.g., ACM, LEWIS) lack explicit handling of output structure mismatch, whereas the alignment/leakage decomposition in this work provides an interpretable instruction-enhancement mechanism.

### Loss & Training
The method is entirely gradient-free and requires no training. Only two small calibration sets are needed:
- **Reasoning calibration set**: 150 samples from Mixture-of-Thoughts data, used for null-space computation in Stage 1.
- **Instruction calibration set**: 365 samples constructed via R1 distillation from IFEval data, followed by LLM filtering and human review, used for attention statistics in Stage 2.

A global scaling coefficient $\lambda$ controls merging intensity. Only Q, K, V, O, and FFN parameters are merged.

## Key Experimental Results

### Main Results

| Method | IFEval | CELLO | InfoBench | ComplexBench | IF Avg. | Math | GPQA | Aider | Arena-Hard | RG Avg. |
|---|---|---|---|---|---|---|---|---|---|---|
| ITM (Qwen2.5-7B-Inst) | 70.43 | 19.15 | 78.49 | 43.63 | 52.92 | 47.27 | 29.80 | 33.33 | 62.86 | 43.32 |
| LRM (R1-Distill-Qwen-7B) | 55.45 | 16.59 | 71.73 | 32.72 | 44.12 | 64.75 | 44.44 | 29.63 | 65.29 | 51.03 |
| SFT | 62.48 | 17.11 | 68.58 | 32.15 | 45.08 | 62.57 | 41.92 | 28.89 | 64.67 | 49.51 |
| Task Arithmetic | 60.44 | 16.97 | 73.07 | 33.34 | 45.96 | 64.22 | 42.93 | 26.67 | 64.53 | 49.59 |
| AIM-TIES | 62.78 | 17.93 | 73.11 | 34.28 | 47.02 | 65.92 | 49.49 | 33.33 | 63.64 | 53.10 |
| **RAIN-Merging** | **63.22** | **19.03** | **74.53** | **35.66** | **48.11** | **68.75** | **54.55** | **33.33** | **65.73** | **55.59** |

RAIN-Merging achieves the best IF Avg. (48.11) and RG Avg. (55.59) among all merging baselines and SFT, with a runtime of approximately 21 minutes compared to 120 minutes for SFT.

### Ablation Study

| Method | IF Avg. | RG Avg. |
|---|---|---|
| RAIN-Merging w/o Stage 2 | 46.58 | 54.92 |
| RAIN-Merging w/o Stage 1 | 47.62 | 52.44 |
| **RAIN-Merging (Full)** | **48.11** | **55.59** |

Removing Stage 1 leads to a notable drop in reasoning performance (52.44 vs. 55.59), while removing Stage 2 yields limited instruction-following gains. The two stages are complementary and individually necessary.

### Key Findings
- **Cross-scale consistency**: Stable improvements are observed across five model scales (1.5B/7B/8B/14B/32B) and two architectures (Qwen/Llama), with relative gains of 1.57%–9.18% on IF Avg. and 2.89%–14.47% on RG Avg.
- **Effectiveness in agentic scenarios**: On ALFWorld and WebShop, the merged model (25.0/29.42) outperforms both LRM (22.0/26.63) and ITM (17.5/10.45).
- **Null-space projection effectiveness**: Task Arithmetic yields $\mathcal{L}_\text{think} = 0.1224$ with a 6.4% `</think>` missing rate; RAIN-Merging reduces these to $\mathcal{L}_\text{think} = 0.0065$ and 0% missing rate.
- **Particularly strong on MathIF**: On MathIF, which requires simultaneous mathematical correctness and format compliance, Both Acc. improves from 12.62% to 20.48% (+62.26%).

## Highlights & Insights
- The orthogonality analysis of task vectors in parameter space provides a theoretically grounded justification for the mergeability of LRM and ITM capabilities.
- The two-stage design elegantly decouples the objectives of "preserving reasoning format" and "enhancing instruction following."
- The null-space projection carries rigorous theoretical guarantees (Proposition 1), rather than being a purely empirical construction.
- The entire pipeline is gradient-free, requiring only ~500 calibration samples and approximately 20 minutes of computation, making it highly practical.
- The alignment/leakage decomposition offers an interpretable framework for analyzing attention-based instruction following.

## Limitations & Future Work
- Instruction-following performance after merging remains below that of ITM (48.11 vs. 52.92), indicating an inherent ceiling for training-free methods.
- The null-space projection at thinking token positions depends on the representativeness of the calibration data.
- The quality of the thinking content itself is not optimized; only the format is preserved.
- Effectiveness at very large scales (>70B) has not been verified.
- Calibration set construction still requires LLM distillation and human filtering, and is not fully automated.

## Related Work & Insights
- Compared to data-agnostic merging methods such as TIES and DARE, RAIN-Merging introduces explicit output structure constraints.
- Compared to activation-based methods such as ACM, LEWIS, and AIM, it explicitly addresses the output format mismatch between LRMs and ITMs.
- The null-space projection paradigm generalizes to other scenarios requiring protection of specific behaviors from being disrupted by merging.
- The alignment/leakage framework for instruction attention scores can be applied to analyze instruction-following mechanisms in arbitrary models.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The two-stage design and the use of null-space projection in a merging context are novel, though model merging as a research direction is already well-populated.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 4 instruction-following + 9 reasoning benchmarks, 5 model scales, 2 architectures, agentic scenarios, and comprehensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Theoretical derivations are clear and visualizations are informative, though the notation is dense and somewhat burdensome to follow.
- **Value**: ⭐⭐⭐⭐ Addresses a practical pain point of LRMs with a lightweight and deployable method that has direct relevance to industrial applications.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] RAIN-Merging: A Gradient-Free Method to Enhance Instruction Following Through Model Merging](rain-merging_a_gradient-free_method_to_enhance_instruction_following_through_mod.md)
- [\[ICLR 2026\] Towards Safe Reasoning in Large Reasoning Models via Corrective Intervention](towards_safe_reasoning_in_large_reasoning_models_via_corrective_intervention.md)
- [\[ICLR 2026\] Training Large Reasoning Models Efficiently via Progressive Thought Encoding](training_large_reasoning_models_efficiently_via_progressive_thought_encoding.md)
- [\[ICLR 2026\] RFEval: Benchmarking Reasoning Faithfulness under Counterfactual Reasoning Intervention in Large Reasoning Models](rfeval_benchmarking_reasoning_faithfulness_under_counterfactual_reasoning_interv.md)
- [\[ICLR 2026\] Dynamics-Predictive Sampling for Active RL Finetuning of Large Reasoning Models](dynamics-predictive_sampling_for_active_rl_finetuning_of_large_reasoning_models.md)

<!-- RELATED:END -->

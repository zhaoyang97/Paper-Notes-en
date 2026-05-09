---
title: >-
  [Paper Note] Enhancing Instruction Following of LLMs via Activation Steering with Dynamic Rejection
description: >-
  [ICLR 2026][Robotics][Activation Steering] This paper proposes Directer (Dynamic Rejection Steering), which dynamically adjusts KV cache steering intensity at each decoding step and incorporates a plausibility constraint, substantially improving LLM instruction following while preventing text quality degradation caused by oversteering.
tags:
  - ICLR 2026
  - Robotics
  - Activation Steering
  - Instruction Following
  - KV Cache Scaling
  - Dynamic Rejection
  - Oversteering Mitigation
date: 2026-05-08
content_hash: 7b6629294f6bfd85
---

# Enhancing Instruction Following of LLMs via Activation Steering with Dynamic Rejection

**Conference**: ICLR 2026
**arXiv**: [2603.06745](https://arxiv.org/abs/2603.06745)
**Code**: [Available](https://github.com/mjk0618/directer)
**Area**: Robotics
**Keywords**: Activation Steering, Instruction Following, KV Cache Scaling, Dynamic Rejection, Oversteering Mitigation

## TL;DR

This paper proposes Directer (Dynamic Rejection Steering), which dynamically adjusts KV cache steering intensity at each decoding step and incorporates a plausibility constraint, substantially improving LLM instruction following while preventing text quality degradation caused by oversteering.

## Background & Motivation

Despite instruction fine-tuning, LLMs still struggle to faithfully follow complex user instructions. Activation steering modifies internal model representations to enhance instruction following, but existing methods carry the risk of **oversteering** — over-emphasizing instructions degrades task accuracy and generation quality.

Core limitations of existing methods:

**Static steering intensity**: Methods such as PASTA and SpotLight rely on manually tuned fixed hyperparameters and cannot adapt to the dynamically varying optimal steering intensity at each decoding step.

**High precomputation cost**: PASTA requires hundreds to thousands of validation samples for grid search over attention heads, with precomputation costs approaching training-level overhead.

**Large computational overhead**: SpotLight requires an additional softmax operation at every decoding step, effectively doubling latency.

**Quality–compliance trade-off**: Enhancing instruction following often comes at the expense of task correctness and text quality.

## Method

### Overall Architecture

The core idea of Directer is to **automatically regulate steering intensity** during decoding through three collaborating components:

1. **KV cache steering**: Scales the Key vectors of instruction tokens by a factor $\alpha=100$.
2. **Plausibility-guided decoding loop**: At each step, compares the steered and original output distributions to dynamically decide whether to accept the steering.
3. **Attention-sensitivity-based layer ranking**: A one-time analysis that determines the sensitivity of each layer to steering, guiding progressive intensity adjustment.

### Key Designs

**KV cache scaling mechanism**: The Key vectors within instruction token span $\mathcal{I}$ are scaled by factor $\alpha$:

$$\mathbf{k'}^{(l)}_i = \begin{cases} \alpha \cdot \mathbf{k}^{(l)}_i, & \text{if } i \in \mathcal{I} \text{ and } l \in \mathcal{L} \\ \mathbf{k}^{(l)}_i, & \text{otherwise} \end{cases}$$

Key scaling is preferred over Value scaling because its effect is naturally normalized by the subsequent softmax, requiring no additional computation.

**Plausibility-guided decoding**: The procedure at each decoding step is as follows:

1. Perform a standard forward pass to obtain the original distribution $p_t$.
2. Apply KV cache steering to the candidate layer set $\mathcal{L}_{\text{cand}}$ to obtain the steered distribution $\tilde{p}_t$.
3. Check the plausibility condition: $p_{t,\tilde{i}^*_t} \geq \beta \cdot p_{t,i^*_t}$ (threshold $\beta=0.5$).
4. If unsatisfied, progressively halve the candidate layer set (removing the least sensitive half).
5. Repeat until the condition is met or the candidate set is empty (falling back to the original distribution).

**Efficient gating mechanism**: A pre-check using the top-2 token probabilities of the original distribution — if $p_{t,i^{**}_t} < \beta \cdot p_{t,i^*_t}$, the steering attempt can be skipped entirely, substantially reducing computational overhead.

**Attention-sensitivity layer ranking**: Layer influence is ranked by observing perturbation propagation under single-layer steering. For each layer $\ell$, steering is applied individually and the perturbation score across all layers $j$ is measured:

$$D_j(\ell) = \underbrace{(\text{dist}(\mathbf{H}^{(j)}_{\text{pre}}, \mathbf{H}^{(j,\ell)}_{\text{post}}) - \text{dist}(\mathbf{H}^{(j)}_{\text{pre}}, \mathbf{H}^{(j)}_{\text{post}}))}_{\text{direct effect}} + \underbrace{(\text{dist}(\mathbf{H}^{(j,\ell)}_{\text{pre}}, \mathbf{H}^{(j)}_{\text{post}}) - \text{dist}(\mathbf{H}^{(j)}_{\text{pre}}, \mathbf{H}^{(j)}_{\text{post}}))}_{\text{propagation effect}}$$

The final ranking is $\text{Sensitivity}(\ell) = \frac{1}{L}\sum_{j=1}^{L} D_j(\ell)$, computed once after prompt prefill.

### Loss & Training

Directer is a pure inference-time method requiring no training. Only two core hyperparameters are used:
- **Scaling factor $\alpha=100$**: Has minimal impact on performance and remains stable across $10^1 \sim 10^5$.
- **Plausibility threshold $\beta=0.5$**: Controls the frequency of steering intervention; the method outperforms the no-steering baseline across its entire range.

A unified configuration is used for all tasks with no task-specific tuning required.

## Key Experimental Results

### Main Results

| Method | IFEval P.Acc / I.Acc | LIFBench List/OD/MD | GSM8K-Format F.Acc/T.Acc | Average |
|--------|---------------------|---------------------|--------------------------|---------|
| Zero-shot | 73.5 / 81.5 | 63.4 / 68.6 / 40.9 | 79.2 / 82.7 | 70.0 |
| PASTA* | 76.5 / 83.4 | 61.8 / 66.0 / 47.8 | 98.9 / 62.7 | 71.0 |
| SpotLight* | 76.3 / 83.6 | 61.4 / 70.8 / 38.8 | 95.4 / 78.7 | 72.1 |
| **Directer** | **78.8 / 84.8** | **64.4 / 70.0 / 51.7** | **99.1 / 86.9** | **76.5** |

| Model Scale | Zero-shot | PASTA* | SpotLight* | Directer |
|-------------|-----------|--------|------------|----------|
| Llama-3.2-1B | 61.3 | 59.7 | 60.6 | **61.6** |
| Qwen-2.5-3B | 63.9 | 65.2 | 62.8 | **67.1** |
| Qwen-2.5-7B | 72.4 | 73.0 | 74.9 | **74.4** |
| Qwen-2.5-14B | 81.6 | 80.1 | 81.7 | **83.5** |

### Ablation Study

| Variant | Accuracy |
|---------|----------|
| Zero-shot | 77.5 |
| **Directer (full)** | **81.8** |
| + Reversed ranking | 79.0 |
| + Random layer steering | 80.2±0.7 |
| + Random token steering | 79.2±1.1 |

### Key Findings

1. **Oversteering is severe**: The original PASTA configuration causes GSM8K task accuracy to drop from 82.7% to 48.1%, whereas Directer maintains 86.9%.
2. **Dynamic outperforms static**: Among fixed-intensity configurations, low intensity (ST1/ST2) yields slight gains but high intensity degrades sharply; Directer's adaptive regulation consistently outperforms all.
3. **Plausibility constraint generalizes**: Applying the plausibility gate to PASTA and SpotLight also substantially mitigates their oversteering problems.
4. **Inference efficiency is manageable**: Throughput is only ~16% lower than the zero-shot baseline, more than 2× faster than SpotLight, with negligible memory overhead.
5. **Generation quality and task fidelity are optimal**: LLM-judge task fidelity reaches ≈92%, with text quality on par with the no-intervention baseline.

## Highlights & Insights

1. **Precise problem formulation**: The paper identifies oversteering as the central bottleneck of activation steering methods, rather than simply pursuing stronger guidance.
2. **Elegant design**: Plausibility checking, progressive halving, and sensitivity ranking interlock to form a closed-loop adaptive mechanism.
3. **Near-zero hyperparameter tuning**: $\alpha$ is stable across 5 orders of magnitude and $\beta$ outperforms the baseline across its entire range, achieving genuine plug-and-play usability.
4. **KV cache operations are compatible with FlashAttention**: A practical advantage that attention-level intervention methods do not possess.

## Limitations & Future Work

1. Explicit instruction span annotation is required; automated identification of instruction boundaries remains unexplored.
2. Layer ranking is based on a single prefill analysis and may need updating in multi-turn dialogue settings where instructions change dynamically.
3. Experiments are conducted primarily on Llama and Qwen series; applicability to architectures such as Mixture-of-Experts is unknown.
4. Only greedy decoding is evaluated; compatibility with sampling strategies has yet to be verified.

## Related Work & Insights

- **Relationship to PASTA**: PASTA steers by suppressing attention scores of non-instruction tokens, but requires large validation sets for head profiling and is prone to oversteering with static configurations; Directer requires no additional dataset and adjusts dynamically.
- **Relationship to SpotLight**: SpotLight maintains a target attention ratio for instruction tokens via post-softmax logit biasing, incurring high computational cost; Directer achieves more efficient steering through KV cache operations.
- **Insight**: The plausibility-guided decoding framework can be generalized to other scenarios requiring balanced intervention intensity, such as safety alignment and style control.

## Rating

- Novelty: ⭐⭐⭐⭐ — Both the dynamic rejection steering mechanism and attention-sensitivity layer ranking constitute novel contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Covers multiple benchmarks, models, ablations, efficiency analysis, and generation quality evaluation comprehensively.
- Writing Quality: ⭐⭐⭐⭐ — Logical structure is clear and mathematical derivations are rigorous.
- Value: ⭐⭐⭐⭐ — A plug-and-play inference-time enhancement module with strong practical utility.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] PERSONA: Dynamic and Compositional Inference-Time Personality Control via Activation Vector Algebra](persona_dynamic_and_compositional_inference-time_personality_control_via_activat.md)
- [\[ICLR 2026\] ODESteer: A Unified ODE-Based Steering Framework for LLM Alignment](odesteer_a_unified_ode-based_steering_framework_for_llm_alignment.md)
- [\[ICLR 2026\] Tracing and Reversing Edits in LLMs](tracing_and_reversing_edits_in_llms.md)
- [\[NeurIPS 2025\] VITRIX-CLIPIN: Enhancing Fine-Grained Visual Understanding in CLIP via Instruction Editing Data and Long Captions](../../NeurIPS2025/robotics/vitrix-clipin_enhancing_fine-grained_visual_understanding_in_clip_via_instructio.md)
- [\[ICLR 2026\] What's the Plan? Metrics for Implicit Planning in LLMs and Their Application to Rhyme Generation and Question Answering](whats_the_plan_metrics_for_implicit_planning_in_llms_and_their_application_to_rh.md)

<!-- RELATED:END -->

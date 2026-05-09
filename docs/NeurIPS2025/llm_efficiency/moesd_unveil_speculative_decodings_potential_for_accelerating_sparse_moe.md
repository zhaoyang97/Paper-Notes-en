---
title: >-
  [Paper Note] MoESD: Revealing the Potential of Speculative Decoding to Accelerate Sparse MoE
description: >-
  [NeurIPS 2025][LLM Efficiency][Speculative Decoding] This work challenges the prevailing belief that speculative decoding (SD) is ineffective for MoE models. Through theoretical analysis and experiments, it demonstrates that MoE models benefit more from SD than dense models at medium batch sizes. The paper introduces *target efficiency* as a system-level metric to quantify acceleration bottlenecks, constructs a reliable performance prediction model, and achieves up to 2.29× speedup on Qwen2-57B-A14B.
tags:
  - NeurIPS 2025
  - LLM Efficiency
  - Speculative Decoding
  - MoE Inference
  - Sparsity Analysis
  - Target Efficiency
  - Performance Modeling
date: 2026-05-08
content_hash: c1aaaa97d91028de
---

# MoESD: Revealing the Potential of Speculative Decoding to Accelerate Sparse MoE

**Conference**: NeurIPS 2025
**arXiv**: [2505.19645](https://arxiv.org/abs/2505.19645)
**Code**: N/A
**Area**: LLM Efficiency
**Keywords**: Speculative Decoding, MoE Inference, Sparsity Analysis, Target Efficiency, Performance Modeling

## TL;DR

This work challenges the prevailing belief that speculative decoding (SD) is ineffective for MoE models. Through theoretical analysis and experiments, it demonstrates that MoE models benefit more from SD than dense models at medium batch sizes. The paper introduces *target efficiency* as a system-level metric to quantify acceleration bottlenecks, constructs a reliable performance prediction model, and achieves up to 2.29× speedup on Qwen2-57B-A14B.

## Background & Motivation

**State of the Field**: Speculative Decoding is a mainstream lossless technique for accelerating LLM inference. A small draft model rapidly generates multiple candidate tokens, which are then verified in parallel by the target model. Since loading all parameters once suffices for both single-token generation and multi-token verification in dense models, SD reduces the number of forward passes and achieves speedup. MoE architectures obtain strong performance with lower computation through sparse activation and have become the dominant choice in SOTA models such as DeepSeek-V3 and Qwen2.5.

**Limitations of Prior Work**: A widely held view in the community is that SD is ineffective for MoE—verifying multiple draft tokens activates more experts, incurring additional parameter loading overhead that causes verification latency to far exceed single-token decoding time. Prior work (e.g., EAGLE) reinforces this belief. However, this perspective overlooks a critical intermediate batch-size regime.

**Root Cause**: Prior studies focus either on small batch sizes (where expert activations grow rapidly with token count, making SD ineffective) or large batch sizes (where computation is the bottleneck, making SD ineffective for any model). The "sweet spot" of medium batch sizes has been entirely overlooked—a regime where all experts are already activated but each expert's load is far from saturated, leaving substantial GPU compute idle.

**Paper Goals**: (1) Under what conditions can SD effectively accelerate MoE inference? (2) How can system-level factors beyond acceptance rate be quantified for SD speedup? (3) How can a reliable SD acceleration prediction model be constructed?

**Starting Point**: When batch size is large enough that all experts are activated during a single decoding step, verifying multiple draft tokens introduces no additional expert loading overhead. Furthermore, the sparsity of MoE means each expert processes far fewer tokens than in a dense model, keeping the system memory-bound, making the extra computation essentially "free."

**Core Idea**: At medium batch sizes, MoE operates in an efficiency gap where all parameters must be loaded but compute is under-utilized. SD is uniquely positioned to exploit this idle compute for lossless acceleration.

## Method

### Overall Architecture

MoESD does not propose a new SD algorithm. Instead, it re-examines SD's behavior on MoE models from the perspectives of theoretical analysis and performance modeling. The work is organized into three levels: (1) formalizing the SD speedup formula and introducing the target efficiency metric; (2) analyzing expert activation and load characteristics of MoE at medium batch sizes; and (3) constructing an end-to-end SD acceleration prediction model.

### Key Designs

1. **Target Efficiency Metric**:

    - Function: Quantifies the impact of the target model's system properties on SD speedup, complementing the algorithmic-level acceptance rate.
    - Mechanism: Target efficiency is defined as $T_T(B,1)/T_T(B,\gamma)$, i.e., the ratio of the target model's time to process one token versus $\gamma$ tokens. This ratio directly determines the dominant term in the denominator of the SD speedup formula. When the system is memory-bound, this ratio approaches 1 (ideal); when compute-bound, it approaches $1/\gamma$ (worst case). Prior work focuses solely on improving acceptance rate $\alpha$, yet even at the same $\alpha$, different model architectures and load characteristics can significantly affect the final speedup through target efficiency.
    - Design Motivation: To determine which target model or workload is better suited for SD under the same algorithmic optimization level, and to decouple algorithmic optimization from system-level optimization.

2. **MoE Advantage Analysis at Medium Batch Sizes**:

    - Function: Theoretically derives why MoE is better suited for SD than dense models at medium batch sizes.
    - Mechanism: Assuming i.i.d. expert activation, the expected number of activated experts given $t$ tokens is derived as $N(t) = E \cdot (1-(1-K/E)^t)$, where $E$ is the total number of experts and $K$ is the number of experts activated per token. The full-activation threshold is $T_{thres} = \lceil \log_{(1-\rho)}(1-\tau) \rceil$, where $\rho=K/E$ is the sparsity ratio. Beyond this threshold, the $B\gamma$ tokens in the verification phase introduce virtually no additional expert loading. Furthermore, the average per-expert token load $\overline{T_{exp}}(t;\rho) = \rho t / (1-(1-\rho)^t)$ decreases with sparsity ratio $\rho$, indicating that sparser MoE models keep the system more memory-bound, creating greater potential for SD acceleration.
    - Design Motivation: A dense model is the degenerate case where $\rho=1$, so the FFN arithmetic intensity always equals $t$, causing the system to become compute-bound quickly. The sparse structure of MoE naturally delays the transition from memory-bound to compute-bound.

3. **End-to-End SD Acceleration Performance Modeling**:

    - Function: Quantitatively predicts SD speedup across different batch sizes, draft lengths, and hardware platforms.
    - Mechanism: The model forward time is decomposed into three components: (1) the roofline effect, captured by a function $G(t;\lambda RP, s)$ that models the transition from memory-bound to compute-bound; (2) the parameter loading time determined by the number of activated experts $N(t)$; and (3) the actual per-expert compute determined by expert load $\overline{T_{exp}}$. A small number of fittable parameters (bias, $k_1, k_2, k_3$, etc.) are introduced and determined by minimizing MSE against GPU measurements, requiring only approximately 21 measurement points for fitting.
    - Design Motivation: The theoretical analysis captures the primary trade-offs, while the lightweight fitting compensates for discrepancies between theoretical assumptions and actual GPU execution, yielding transparent and interpretable end-to-end acceleration results.

## Key Experimental Results

### Main Results

On various hardware platforms, Qwen2-57B-A14B-Instruct (target) + Qwen2-0.5B-Instruct (draft):

| Hardware | Dataset | Temperature | γ | Max Speedup | σ |
|----------|---------|-------------|---|-------------|---|
| 2×GPU-B | HumanEval | 0.0 | 4 | **2.29×** | 0.90 |
| 2×GPU-A | HumanEval | 0.0 | 4 | 2.18× | 0.91 |
| 4×GPU-C | HumanEval | 0.0 | 3 | 2.14× | 0.93 |
| 4×GPU-A | HumanEval | 0.0 | 4 | 2.08× | 0.90 |

Mixtral-8×7B-Instruct + EAGLE speculation head:

| Hardware | Dataset | Temperature | γ | Max Speedup | σ |
|----------|---------|-------------|---|-------------|---|
| 2×GPU-A | HumanEval | 0.0 | 4 | 1.79× | 0.58 |
| 2×GPU-A | HumanEval | 0.0 | 3 | 1.69× | 0.66 |

### Ablation Study

| Configuration | Observation | Explanation |
|---------------|-------------|-------------|
| Increasing sparsity ratio ρ | Wider effective batch size range for SD | Sparser → more memory-bound → more favorable for SD |
| γ increasing from 2 to 4 | Higher speedup on HumanEval; potential decrease on MT-bench | Depends on changes in acceptance rate σ |
| Temperature increasing from 0.0 to 1.0 | Speedup decreases | Higher temperature reduces acceptance rate σ |
| Comparison with dense model (OPT-30B) | MoE achieves higher speedup over a wider batch size range | Validates theoretical predictions |

### Key Findings

- SD speedup as a function of batch size follows an inverted-U curve, peaking at medium batch sizes, consistent with theoretical predictions.
- The trend of target efficiency closely tracks the final speedup ratio, validating the effectiveness of the proposed metric.
- Sparser MoE models (e.g., by adjusting num_experts_per_token) benefit from SD over a wider range of batch sizes, as theoretically predicted.
- Speedup on HumanEval (code generation, high acceptance rate) is significantly higher than on MT-bench (dialogue, lower acceptance rate).

## Highlights & Insights

- **The introduction of target efficiency is particularly elegant**: It decomposes the "black box" of SD speedup into algorithmic factors (acceptance rate) and system factors (target efficiency), enabling precise identification of bottlenecks. Prior SD work exclusively focused on improving acceptance rate while overlooking the equally important system side.
- **The "efficiency gap" insight for MoE is illuminating**: At medium batch sizes, MoE operates in an awkward regime where all parameters must be loaded but the compute volume is insufficient to saturate the GPU. This is a structural property inherent to MoE, and SD provides a natural mechanism to exploit the idle compute.
- **The performance modeling methodology is transferable**: The approach of decoupling the roofline effect, expert activation, and expert load can be generalized to analyze the effectiveness of other MoE optimization techniques.

## Limitations & Future Work

- Experiments are conducted only on single-node 2–4 GPU configurations; the practical speedup in large-scale Expert Parallelism (EP) settings is not empirically validated, though the theoretical analysis argues the conclusions remain valid.
- The assumption of uniform expert activation may not hold in practice; real-world load imbalance could affect the precision of theoretical derivations, even though SOTA models encourage load balance through auxiliary losses.
- The possibility of designing draft models tailored to MoE characteristics is not explored—for instance, whether using a MoE-structured draft model could further improve speedup.
- Performance modeling requires a small number of GPU measurements to fit parameters, preventing purely theoretical prediction.
- Offloading and EP scenarios are discussed but lack corresponding experimental validation.

## Related Work & Insights

- **vs. MagicDec**: MagicDec first challenged the belief that "SD is unsuitable for large batch sizes," discovering that long-sequence scenarios alter the compute-to-memory-access ratio via KV cache, making SD effective. MoESD identifies another condition for SD effectiveness from the model architecture perspective (MoE sparsity). The two findings are complementary.
- **vs. EAGLE/Medusa and other SD algorithms**: These works focus on improving acceptance rate, whereas MoESD addresses system-level factors. The two directions are orthogonal and can be combined.
- **vs. MoE compression/offloading methods**: Compression methods (pruning, quantization, etc.) trade accuracy for speed; offloading methods exploit expert load imbalance; MoESD provides a lossless acceleration path that does not rely on expert imbalance.

## Rating

- Novelty: ⭐⭐⭐⭐ Challenges the widely accepted belief that "SD is ineffective for MoE" with precise insights, though no new algorithm is proposed.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated across multiple hardware platforms, models, and datasets with strong theory-experiment alignment; lacks large-scale EP experiments.
- Writing Quality: ⭐⭐⭐⭐⭐ Logic flows clearly from theoretical derivation to experimental validation; rigorous formulations and rich figures and tables.
- Value: ⭐⭐⭐⭐ Provides a new perspective on MoE inference acceleration with direct applicability to private deployment and latency-sensitive scenarios.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] 3-Model Speculative Decoding (PyramidSD)](3model_speculative_decoding.md)
- [\[NeurIPS 2025\] OmniDraft: A Cross-Vocabulary Online Adaptive Drafter for On-Device Speculative Decoding](omnidraft_a_cross-vocabulary_online_adaptive_drafter_for_on-device_speculative_d.md)
- [\[ACL 2026\] Speculative Verification: Exploiting Information Gain to Refine Speculative Decoding](../../ACL2026/llm_efficiency/speculative_verification_exploiting_information_gain_to_refine_speculative_decod.md)
- [\[ACL 2026\] Multi-Drafter Speculative Decoding with Alignment Feedback](../../ACL2026/llm_efficiency/multi-drafter_speculative_decoding_with_alignment_feedback.md)
- [\[NeurIPS 2025\] Approximately Aligned Decoding](approximately_aligned_decoding.md)

<!-- RELATED:END -->

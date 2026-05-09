---
title: >-
  [Paper Note] MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping
description: >-
  [CVPR 2026][Multimodal VLM][MoE] MoDES is the first expert skipping framework for MoE multimodal large language models. It incorporates layer-level importance into routing probabilities via Global-Modulated Local Gating (GMLG), applies modality-specific skipping strategies for text and visual tokens via a Dual-Modal Threshold (DMT), and efficiently optimizes thresholds via frontier search. On Qwen3-VL-MoE-30B, MoDES retains 97.33% accuracy with 88% expert skipping, achieving a 2.16× prefill speedup.
tags:
  - CVPR 2026
  - Multimodal VLM
  - MoE
  - expert skipping
  - dual-modal threshold
  - global-modulated local gating
  - multimodal large model acceleration
date: 2026-05-08
content_hash: 8c5ada719e2335bd
---

# MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping

**Conference**: CVPR 2026
**arXiv**: [2511.15690](https://arxiv.org/abs/2511.15690)
**Code**: [https://github.com/ModelTC/MoDES](https://github.com/ModelTC/MoDES)
**Area**: Multimodal VLM / MoE Acceleration / Efficient Inference
**Keywords**: MoE, expert skipping, dual-modal threshold, global-modulated local gating, multimodal large model acceleration

## TL;DR
MoDES is the first expert skipping framework for MoE multimodal large language models. It incorporates layer-level importance into routing probabilities via Global-Modulated Local Gating (GMLG), applies modality-specific skipping strategies for text and visual tokens via a Dual-Modal Threshold (DMT), and efficiently optimizes thresholds via frontier search. On Qwen3-VL-MoE-30B, MoDES retains 97.33% accuracy with 88% expert skipping, achieving a 2.16× prefill speedup.

## Background & Motivation
MoE MLLMs (e.g., Kimi-VL, Qwen3-VL-MoE) reduce computational cost through sparse expert activation, yet an efficiency bottleneck remains: fixed top-$k$ routing activates the same number of experts for every token. Existing expert skipping methods (NAEE, MC-MoE, DiEP) are designed for text-only LLMs; direct transfer to MLLMs causes >10% performance degradation at an 83% skip rate. Analysis reveals two overlooked factors: (1) **global contribution mismatch**—shallow-layer experts exert far greater influence on final outputs than deep-layer experts (error explosion effect); (2) **modality discrepancy**—visual tokens are more orthogonal to FFN weights (angle → 90°), expert updates on visual tokens are smaller in magnitude, and redundancy is higher.

## Core Problem
How to design a modality-aware, layer-aware expert skipping strategy for MoE MLLMs that preserves near-baseline accuracy under extreme skip rates (>80%)?

## Method

### Overall Architecture
MoDES comprises two core components: (1) GMLG estimates an importance score for each token–expert pair; (2) DMT selects modality-specific skipping thresholds, which are efficiently determined via a frontier search algorithm. The entire pipeline is training-free.

### Key Designs
1. **Global-Modulated Local Gating (GMLG)**: The importance score is defined as $s_i^{(l)} = \alpha^{(l)} \cdot \pi_i^{(l)}$, where $\pi_i^{(l)}$ is the standard routing probability (local signal) and $\alpha^{(l)}$ is a layer-wise global weight computed via offline calibration—measured as the KL divergence when all experts in layer $l$ are skipped. $\alpha^{(l)}$ is larger for shallow layers and smaller for deep layers, ensuring shallow-layer experts are skipped less frequently. Calibration requires only 1,024 samples and takes approximately 20 minutes to 4 hours (for 20–30B models).

2. **Dual-Modal Threshold (DMT)**: Separate skipping thresholds $\tau_t$ and $\tau_v$ are set for text tokens and visual tokens, respectively. The decision rule is: $\{Expert_i^{(l)} \mid s_i^{(l)} < \tau_t \cdot \mathbb{I}_t + \tau_v \cdot \mathbb{I}_v\}$ are skipped. Visualization confirms that in practice, the fraction of skipped experts is much higher for visual tokens than for text tokens across all layers (>90% vs. 50–70%), validating the insight that visual expert redundancy is higher.

3. **Frontier Search**: Optimization is performed over a 2D grid $\mathcal{B}^2$ of $(\tau_t, \tau_v)$. By exploiting the monotonicity of $f$ (KL divergence) and $g$ (skip rate), the search complexity is reduced from $O(ND^2)$ to $O(ND)$—yielding an approximately 45× reduction in search time in practice. Correctness and optimality are formally established (Lemma 1–4 + Proposition 1–2).

### Loss & Training
The method is entirely training-free. Offline calibration of $\alpha^{(l)}$ and frontier search for $(\tau_t^*, \tau_v^*)$ are both conducted on 1,024 GQA samples. At inference time, only a branch-free masked comparison is added to the MoE layer's router kernel, with no additional kernel launches.

## Key Experimental Results

| Model | Skip Rate | MoDES | MC-MoE | DiEP | NAEE | Reduce-k |
|--------|------|------|----------|------|------|------|
| Kimi-VL-A3B | 50% | **99.91%** | 97.69 | 98.17 | 96.44 | 95.93 |
| Kimi-VL-A3B | 67% | **98.46%** | 95.45 | 94.81 | 94.03 | 93.88 |
| Kimi-VL-A3B | 83% | **96.25%** | 88.32 | 87.58 | 82.81 | 71.60 |
| Qwen3-VL-MoE-30B | 88% | **97.33%** | 86.66 | 85.30 | 80.60 | 60.11 |
| InternVL-3.5-30B | 88% | **97.03%** | 86.20 | 83.26 | 78.88 | 59.63 |

Inference speedup (Qwen3-VL-MoE-30B): prefill **2.16×**, decode **1.26×**.

Compatible with quantization: 2.5-bit quantization combined with MoDES retains 94.43% accuracy on Qwen3 (vs. 89.58% for MC-MoE).

### Ablation Study
- **Both GMLG and DMT are essential**: At 83% skip rate, plain thresholding achieves 82.81% → +GMLG: 84.48% → +DMT: 85.50% → **GMLG+DMT: 96.25%**.
- **Modality discrepancy is real**: Reducing top-$k$ to 1 for visual tokens causes only marginal degradation, while the same reduction for text tokens leads to severe performance drops—confirming higher expert redundancy for visual tokens.
- **Calibration data insensitivity**: Results are nearly identical across GQA/COCO/VMMMU calibration sets (~96% in all cases).
- **Consistent $\alpha^{(l)}$ pattern**: Layer-wise KL divergence distributions are similar across datasets, with shallow layers consistently exceeding deep layers.
- **Frontier search vs. exhaustive search**: Accuracy is nearly identical (96.24% vs. 96.25%) with a 45× reduction in search time.

## Highlights & Insights
- **First expert skipping framework for MoE MLLMs**—all prior methods target unimodal LLMs and degrade significantly upon direct transfer.
- The finding that "visual tokens exhibit higher expert redundancy" aligns with V2Drop/ApET's observation that "a large number of visual tokens are redundant"—extended here from the token dimension to the expert dimension.
- The frontier search is supported by rigorous mathematical proofs (monotonicity → feasible region structure → optimality), demonstrating strong theoretical foundations.
- Retaining 97% accuracy at an 88% skip rate is remarkable, suggesting that MoE models substantially over-allocate experts.
- Orthogonal composability with quantization enables future three-stage compression: MoDES + quantization + token compression.

## Limitations & Future Work
- Thresholds are determined via offline search and may not generalize across tasks or inputs—input-adaptive dynamic thresholds are worth exploring.
- Evaluation is limited to three MoE MLLMs (Kimi-VL / Qwen3-VL / InternVL3.5); broader architectural coverage is needed.
- Decode-stage speedup is modest (1.26×), primarily because decoding is memory-bound and processes only text tokens.
- Calibrating $\alpha^{(l)}$ requires a forward pass with each layer's experts skipped, incurring overhead that grows linearly with the number of layers.
- Dynamic top-$k$ adjustment strategies (as opposed to skipping after fixing top-$k$) remain unexplored.

## Related Work & Insights
- **vs. NAEE/MC-MoE (LLM expert skipping)**: Designed for unimodal LLMs; accuracy drops below 89% at an 83% skip rate when transferred to MLLMs. MoDES achieves 96.25%—a substantial gap.
- **vs. DiEP (differentiable expert pruning)**: DiEP performs expert similarity and routing probability pruning in a training-aware framework but ignores layer-wise and modality-wise differences. MoDES is training-free and achieves superior results.
- **vs. V2Drop/DUET-VLM (token compression)**: Orthogonally complementary—V2Drop reduces the number of visual tokens, while MoDES reduces the number of experts activated per token. The two approaches can be combined.
- **vs. ApET (approximation error compression)**: ApET reduces tokens from an information-theoretic perspective; MoDES reduces computation from an expert perspective. Different mechanisms, same objective.

## Highlights & Insights (Extended)
- The MoDES finding that "shallow layers are more important" corroborates the Overthinking paper's observation that "hypotheses become unstable from middle to deep layers → hallucinations"—both point to the conclusion that not all layers contribute equally.
- **Combination idea**: MoDES (expert skipping) + V2Drop (token dropping) + ApET (token merging)—three levels of compression for maximal VLM inference acceleration.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First adaptation of expert skipping to multimodal MoE; both insights (layer-wise and modality-wise) and the frontier search algorithm are original contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three MoE model families, 13 benchmarks, multiple skip rates, quantization combinations, detailed ablations, and mathematical proofs.
- Writing Quality: ⭐⭐⭐⭐⭐ Perfect logical flow from motivation → analysis → method → validation; appendix includes complete proofs.
- Value: ⭐⭐⭐⭐⭐ MoE MLLMs have become mainstream (Kimi-VL / DeepSeek / Qwen3 all adopt MoE); the method is directly deployable.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Quant Experts: Token-aware Adaptive Error Reconstruction with Mixture of Experts for Large Vision-Language Models Quantization](quant_experts_token_aware_vlm_quantization.md)
- [\[CVPR 2026\] HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in Large Vision-Language Models](hulluedit_single-pass_evidence-consistent_subspace_editing_for_mitigating_halluc.md)
- [\[CVPR 2026\] MoE-GRPO: Optimizing Mixture-of-Experts via Reinforcement Learning in Vision-Language Models](moe-grpo_optimizing_mixture-of-experts_via_reinforcement_learning_in_vision-lang.md)
- [\[CVPR 2026\] GTR-Turbo: Merged Checkpoint is Secretly a Free Teacher for Agentic VLM Training](gtr-turbo_merged_checkpoint_is_secretly_a_free_teacher_for_agentic_vlm_training.md)
- [\[CVPR 2026\] Unbiased Dynamic Multimodal Fusion](unbiased_dynamic_multimodal_fusion.md)

<!-- RELATED:END -->

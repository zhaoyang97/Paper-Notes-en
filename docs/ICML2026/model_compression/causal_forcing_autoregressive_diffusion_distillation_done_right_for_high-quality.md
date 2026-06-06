---
title: >-
  [Paper Note] Causal Forcing: Autoregressive Diffusion Distillation Done Right for High-Quality Real-Time Interactive Video
description: >-
  [ICML 2026][Model Compression][Autoregressive Video Generation] By identifying the theoretical requirement for "**frame-level injectivity**," this paper proposes Causal Forcing—a method that replaces the **bidirectional…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Autoregressive Video Generation"
  - "Diffusion Distillation"
  - "Causal Attention"
  - "Frame-level Injectivity"
date: 2026-05-08
content_hash: 9a3625ed05fda065
---

# Causal Forcing: Autoregressive Diffusion Distillation Done Right for High-Quality Real-Time Interactive Video

**Conference**: ICML 2026  
**arXiv**: [2602.02214](https://arxiv.org/abs/2602.02214)  
**Code**: To be confirmed  
**Area**: Video Generation / Diffusion Model Distillation  
**Keywords**: Autoregressive Video Generation, Diffusion Distillation, Causal Attention, Frame-level Injectivity

## TL;DR
By identifying the theoretical requirement for "**frame-level injectivity**," this paper proposes Causal Forcing—a method that replaces the **bidirectional teacher with an autoregressive teacher** for ODE distillation initialization. This prevents the performance collapse seen in Self-Forcing; compared to Self-Forcing, the method achieves +19.3% in dynamics, +8.7% in VisionReward, and +16.7% in instruction following, while maintaining the same inference latency (0.69s).

## Background & Motivation

**Background**: Real-time interactive video generation requires distilling multi-step diffusion models into few-step autoregressive (AR) models. Current approaches (CausVid, Self-Forcing) employ "asymmetric distillation"—distilling a pre-trained bidirectional video diffusion model into an AR student model.

**Limitations of Prior Work**: Although Self-Forcing is the current SOTA, a significant performance gap remains compared to standard DMD (distilling bidirectional students)—with dynamics, visual quality, and instruction following drops of 10-20%. This suggests a fundamental issue in existing AR distillation pipelines.

**Key Challenge**: An "architectural gap" exists when distilling from a bidirectional model to an AR student—the bidirectional model uses full attention (accessing future frames), while the AR model is restricted to causal attention (conditioned only on past frames). Although current methods include ODE initialization and DMD stages, neither theoretically addresses this gap correctly.

**Core Idea**: The root cause is that ODE distillation violates the "**frame-level injectivity**" requirement. When distilling a bidirectional teacher into an AR student, a single noisy frame can correspond to multiple different clean frames, causing the MSE loss to learn the conditional expectation (mean) rather than the true flow mapping. The solution is to use an AR teacher for ODE distillation initialization, as the AR teacher's PF-ODE naturally satisfies frame-level injectivity.

## Method

### Overall Architecture
A three-stage pipeline:
- **Stage 1**: Train an autoregressive diffusion model using Teacher Forcing (TF) to serve as the teacher for subsequent ODE distillation.
- **Stage 2**: Perform Causal ODE Distillation based on this AR teacher to train a few-step AR student.
- **Stage 3**: Apply asymmetric DMD on top of the ODE initialization to further refine the student model.

The key difference from Self-Forcing is the replacement of the bidirectional teacher with an AR teacher during ODE initialization.

### Key Designs

1. **Frame-level Injectivity Principle**:
    - **Function**: Identifies the theoretical condition that ODE distillation must satisfy—each noisy frame must map to a unique clean frame.
    - **Mechanism**: Frame-level injectivity is defined as: for any noisy frame $x_t^i$, there exists a unique clean frame $x_0^i$ such that $x_0^i = \phi^{AR}(x_t^i, t)$. It is proven that the PF-ODE trajectory of a bidirectional teacher only satisfies injectivity at the video level but violates it at the frame level (Lemma 3.2)—the same noisy frame can map to multiple clean frames depending on different subsequent frames.
    - **Design Motivation**: In Self-Forcing, when an AR student is distilled from a bidirectional teacher, the MSE regression target collapses to the conditional expectation $\mathbb{E}[x_0^i \mid x_t^i]$, resulting in blurry videos; this is a fundamental theoretical flaw that cannot be fixed by the subsequent DMD stage.

2. **Teacher Forcing vs. Diffusion Forcing**:
    - **Function**: Selects the optimal training strategy for the AR diffusion model.
    - **Mechanism**: Teacher Forcing (TF) conditions on clean prefix frames $x_0^{<i}$ during training; Diffusion Forcing (DF) conditions on noisy prefixes $x_t^{<i}$. While DF is considered standard, it is counter-intuitively found that TF is superior (Proposition 3.4)—DF creates a training-inference distribution mismatch where the model sees high-noise previous frames during training but clean frames during inference, leading to "collapse."
    - **Design Motivation**: TF eliminates this distribution mismatch, aligning the training objective with inference; experiments show TF achieves a 111.2% higher VisionReward than DF.

3. **Causal ODE Distillation Flow**:
    - **Function**: Initializes the AR student using an AR teacher that satisfies frame-level injectivity.
    - **Mechanism**: Given ground-truth clean prefix frames $x_{gt}^{<i}$, intermediate states $\{x_t^i\}_{t \in \mathcal{S}}$ are generated starting from Gaussian noise $x_T^i$ using the AR teacher along the PF-ODE trajectory. The student learns the flow mapping via MSE regression: $\min_\theta \mathbb{E}[\|G_\theta(x_t^i, x_{gt}^{<i}, t) - x_0^i\|^2]$. Since the AR teacher is inherently causal, its PF-ODE naturally satisfies injectivity at the frame level.
    - **Design Motivation**: Ensures the student learns the correct flow mapping instead of conditional expectations, providing high-quality initialization for subsequent DMD.

## Key Experimental Results

### Main Results

| Model | Throughput ↑ | Latency ↓ | Dynamics ↑ | VisionReward ↑ | Instruction Following ↑ | User Rating ↓ |
|------|--------|------|--------|---------------|----------|----------|
| Wan2.1 (Bidirectional) | 0.78 | 103s | 61 | 5.275 | 42 | 2.29 |
| CausVid (AR Distillation) | 17.0 | 0.69s | 62 | 5.741 | 12 | 4.27 |
| Self-Forcing | 17.0 | 0.69s | 57 | 5.820 | 48 | 2.87 |
| **Causal Forcing (Ours)** | **17.0** | **0.69s** | **68** | **6.326** | **56** | **1.64** |

Improvements over Self-Forcing: Dynamics +19.3%, VisionReward +8.7%, Instruction Following +16.7%.

### Ablation Study

| Configuration | Dynamics | VisionReward | Instruction Following | Description |
|------|--------|-------------|----------|------|
| Diffusion Forcing (DF) + Self-Forcing ODE + DMD | 60 | 1.583 | 30 | DF leads to severe collapse |
| Teacher Forcing (TF) + Self-Forcing ODE + DMD | 50 | 3.343 | 32 | TF improves but ODE is insufficient |
| TF + Self-Forcing ODE + DMD (Chunk-level) | 24 | 3.330 | 38 | Bidirectional teacher ODE performs poorly |
| **TF + Causal ODE + DMD (Chunk-level)** | **68** | **6.326** | **56** | Full Proposed Solution |

### Key Findings
- TF significantly outperforms DF (VisionReward +111%)—training-inference distribution alignment is critical.
- Causal ODE significantly outperforms Self-Forcing ODE in chunk-level settings (VisionReward +90%, Dynamics +183%), with even more extreme improvements in frame-level settings (Dynamics +3100%).
- The DMD stage cannot compensate for gaps in ODE initialization—high-quality initialization determines the upper bound for DMD.

## Highlights & Insights
- **Clear Theoretical Contribution**: The first mathematical framework using frame-level injectivity to explain performance collapse in AR distillation, proving Self-Forcing fundamentally violates this condition—theoretical insight surpasses empirical tuning.
- **Flipping the TF vs. DF Conclusion**: Counters common belief by showing DF is actually inferior to TF in AR diffusion; the paper provides a rigorous proof of distribution mismatch and quantifies a 111% performance gap, serving as an important correction to the field.
- **Transferable Design Principles**: The frame-level injectivity principle applies not only to ODE distillation but also naturally extends to Consistency Distillation (CD) frameworks—this paper introduces the first Causal CD.
- **Significant Practical Impact**: Under identical computational budgets, the method achieves double-digit percentage improvements across multiple metrics compared to SOTA Self-Forcing.

## Limitations & Future Work
- Gap in long video generation: The model is trained on 5-second videos; direct extrapolation to longer videos creates a training-inference gap, requiring orthogonal long-video adaptation methods like LongLive or Rolling Forcing.
- Consistency Distillation remains weaker than ODE: Although theoretically correct, the proposed Causal CD still underperforms relative to Causal ODE distillation—current vanilla LCM implementations have room for improvement.
- Insufficient comparison with GAN distillation: Comparison with APT2, which uses GAN + Teacher Forcing CD initialization, was not possible due to its closed-source status.

## Related Work & Insights
- **vs. Self-Forcing** (Huang et al. 2025a): Both use a two-stage ODE distillation + DMD pipeline, but Self-Forcing uses a bidirectional teacher while this work uses an AR teacher; the key difference is the satisfaction of frame-level injectivity. This work is a fundamental correction to the Self-Forcing architectural flaw.
- **vs. CausVid** (Yin et al. 2025): Both focus on AR distillation, but CausVid initially proposed the asymmetric distillation paradigm with lower performance.
- **vs. DMD Methodology**: Standard DMD effectively distills bidirectional students but performs poorly when directly used for AR student initialization—initialization quality determines the DMD performance ceiling.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The theoretical framework of frame-level injectivity and the flipped TF vs. DF conclusion are pioneering; the combination of Causal ODE distillation and teacher selection strategy is unique and profound.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers ODE initialization / DMD / CD directions, chunk-level and frame-level evaluation modes, 5 baseline methods, and 3 layers of evaluation (VBench + VisionReward + User Study).
- Writing Quality: ⭐⭐⭐⭐ Clear theoretical derivations and deep problem diagnosis; some expressions could be more concise.
- Value: ⭐⭐⭐⭐⭐ Resolves a core performance bottleneck in real-time video generation, provides a reusable theoretical framework, and achieves 15-20% improvements over SOTA under the same training budget.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Elastic Weight Consolidation Done Right for Continual Learning](../../CVPR2026/model_compression/elastic_weight_consolidation_done_right_for_continual_learning.md)
- [\[ICML 2026\] Beyond Tokens: Enhancing RTL Quality Estimation via Structural Graph Learning](beyond_tokens_enhancing_rtl_quality_estimation_via_structural_graph_learning.md)
- [\[ICML 2026\] ScaLoRA: Optimally Scaled Low-Rank Adaptation for Efficient High-Rank Fine-Tuning](scalora_optimally_scaled_low-rank_adaptation_for_efficient_high-rank_fine-tuning.md)
- [\[ICML 2026\] Semantic Integrity Matters: Benchmarking and Preserving High-Density Reasoning in KV Cache Compression](semantic_integrity_matters_benchmarking_and_preserving_high-density_reasoning_in.md)
- [\[ICML 2026\] IDLM: Inverse-distilled Diffusion Language Models](idlm_inverse-distilled_diffusion_language_models.md)

</div>

<!-- RELATED:END -->

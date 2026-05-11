---
title: >-
  [Paper Note] OARS: Process-Aware Online Alignment for Generative Real-World Image Super-Resolution
description: >-
  [CVPR 2026][Image Generation][Real-World Super-Resolution] This paper proposes OARS, a framework that systematically addresses human preference alignment in generative real-world image super-resolution for the first time…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Real-World Super-Resolution"
  - "RLHF"
  - "reward model"
  - "Online RL"
  - "Flow Matching"
  - "MLLM"
  - "Image Quality Assessment"
date: 2026-05-08
content_hash: aa36402e2c77fcc7
---

# OARS: Process-Aware Online Alignment for Generative Real-World Image Super-Resolution

**Conference**: CVPR 2026
**arXiv**: [2603.12811](https://arxiv.org/abs/2603.12811)
**Code**: None
**Area**: Image Generation / Image Super-Resolution
**Keywords**: Real-World Super-Resolution, RLHF, reward model, Online RL, Flow Matching, MLLM, Image Quality Assessment

## TL;DR

This paper proposes OARS, a framework that systematically addresses human preference alignment in generative real-world image super-resolution for the first time. It introduces COMPASS, an MLLM-based process-aware reward model, and a progressive online reinforcement learning pipeline (cold start → reference-guided RL → non-reference RL), significantly improving perceptual quality while preserving fidelity.

## Background & Motivation

### Problem Definition

Real-world image super-resolution (Real-ISR) aims to recover high-fidelity, perceptually pleasing high-resolution (HR) images from low-resolution (LR) inputs degraded by complex and unknown processes. Although diffusion models have brought substantial perceptual quality gains, standard supervised fine-tuning (SFT) suffers from two fundamental limitations: (1) poor generalization to unseen real-world degradations; and (2) lack of a direct optimization mechanism to align generated content with human aesthetic preferences, often resulting in hallucinations or over-smoothing.

### Limitations of Prior Work

Applying RLHF to Real-ISR faces two critical bottlenecks:

**Reward design dilemma**: Full-reference (FR) metrics require unavailable ground-truth images; no-reference (NR) metrics lack the fine-grained sensitivity needed to distinguish subtle differences among generative SR outputs. Naively combining FR and NR metrics via static linear weighting ignores degradation severity—potentially under-enhancing high-quality inputs or over-sharpening low-quality ones.

**Pseudo-diversity in offline RL**: Offline methods such as DP2O-SR construct preference pairs by sampling from the same SFT model with different random seeds. Under the strong spatial constraints of SR, however, these noise variations degenerate into random texture hallucinations rather than genuine structural diversity, causing optimization to collapse over a narrow candidate pool.

### Mechanism

The paper introduces two key innovations: (1) a **process-aware, quality-adaptive reward model** that evaluates the LR→SR transformation process rather than static outputs; and (2) an **online exploration strategy** that breaks the pseudo-diversity bottleneck.

## Method

### Overall Architecture

OARS comprises two main components: the **COMPASS reward model** and a **progressive online RL framework**.

```
┌─────────────────────────────────────────────────────────┐
│                    OARS Overall Pipeline                 │
│                                                         │
│  COMPASS-20K Dataset ──→ COMPASS Reward Model (MLLM)    │
│         │                      │                        │
│         ▼                      ▼                        │
│  Stage 1: Cold Start    →  Stage 2: FR-RL  → Stage 3: NR-RL │
│  (Flow Matching SFT)       (with GT ref.)    (no ref., COMPASS) │
│         │                      │                  │      │
│         └──────────────────────┴──────────────────┘      │
│                    LoRA merging at inference              │
└─────────────────────────────────────────────────────────┘
```

### Key Designs 1: COMPASS Reward Model

#### COMPASS-20K Dataset

- **Data sources**: 800 synthetic LR images from DIV2K (Real-ESRGAN-style degradation) + 1,600 real-world LQ images covering noise, compression artifacts, defocus blur, motion blur, etc.
- **SR outputs**: 12 mainstream enhancement algorithms (DiffBIR, OSEDiff, SeeSR, etc.) × 2,400 inputs → 28,800 LR–SR pairs
- **Annotation dimensions**: Fidelity, Perceptual Gain, and textual descriptions

#### Three-Stage Perceptual Annotation Pipeline

This is one of the most elegant designs in the paper, addressing the core challenge of obtaining quality labels that are both globally comparable and intra-group discriminative:

| Stage | Content | Output |
|-------|---------|--------|
| Stage 1: Global Anchor Scoring | Q-Insight independently scores LR and SR: $Q_{LR}, Q_{SR} \in [1,5]$ | Globally comparable quality anchors |
| Stage 2: Intra-Group Ranking | A pairwise comparison model (trained on DiffIQA data) performs exhaustive pairwise comparisons among all SR outputs for each LR | Intra-group relative ranking $r \in [0,1]$ |
| Stage 3: Rank-Guided Calibration | Linear calibration per group: $\hat{Q}_{SR} = \alpha^* \cdot r + \beta^*$, preserving rankings while aligning to the global scale | Calibrated SR quality scores |

#### Input Quality-Adaptive Reward Mechanism

The final reward formula of COMPASS:

$$R = F \cdot Q_{LR} + F^{Q_{LR}/\gamma} \cdot \Delta Q$$

where $\Delta Q = Q_{SR} - Q_{LR}$ and $\gamma=7$.

- **First term** $F \cdot Q_{LR}$: measures preservation of the original quality of the input image.
- **Second term** $F^{Q_{LR}/\gamma} \cdot \Delta Q$: perceptual gain, adaptively controlled by input quality.
    - When input quality is high, the exponent $Q_{LR}/\gamma$ is large, making the reward highly sensitive to fidelity degradation → encouraging conservative enhancement.
    - When input quality is low, the fidelity constraint is relaxed → allowing more aggressive perceptual improvement.
- This dynamic gating ensures that perceptual enhancement is strictly constrained by content preservation.

### Key Designs 2: Progressive Online RL

#### Stage 1: Cold Start (Flow Matching SFT)

The model is trained on large-scale LR–HR paired data using a flow matching objective to learn basic SR capabilities:

$$\mathcal{L}_{SFT}(\theta) = \mathbb{E}\left[\|v - v_\theta(x_t, t \mid x_{LR}, c)\|_2^2\right]$$

#### Stage 2: Full-Reference RL

Applying RL directly on the SFT model leads to training instability and reward hacking. This stage serves as a buffer between SFT and non-reference optimization:

- **Fidelity supervision**: DISTS is computed directly between the SR output and the GT, rather than relying on a learned reward model to predict fidelity.
- **Shallow LoRA optimization**: RL updates are applied via LoRA on the base model rather than on the SFT weights, motivated by three reasons:
    - The base model's parameter distribution is close to that of the SFT model, enabling stable merging.
    - The base model has higher sampling stochasticity, facilitating exploration.
    - It is less susceptible to reward hacking.

#### Negative-Aware Objective

Implicit positive and negative policy directions are defined as:

$$v_\theta^+(x_t, t) = (1-\lambda)v_{old} + \lambda v_\theta$$
$$v_\theta^-(x_t, t) = (1+\lambda)v_{old} - \lambda v_\theta$$

The final RL objective is:

$$\mathcal{L}_{RL}(\theta) = \mathbb{E}\left[r\|v_\theta^+ - v\|_2^2 + (1-r)\|v_\theta^- - v\|_2^2\right]$$

where $r$ is the optimality probability after intra-group reward normalization and variance filtering. Groups with high variance and low mean are discarded to avoid ambiguous supervision.

#### Stage 3: Non-Reference RL

Training continues on real-world LQ data without GT images, with rewards provided entirely by COMPASS. At inference time, the final LoRA parameters $\Delta_{NR}$ are merged into the SFT model.

### Loss & Training

- Base model: Qwen-Image-Edit-2509
- LoRA rank=32, alpha=64
- 6-step sampling during training, 40 steps at inference
- $K=24$ candidate samples per LR image
- Group filtering threshold: groups with mean > 0.9 and variance < 0.05 are discarded (near-identical samples provide no discriminative value)

## Key Experimental Results

### Main Results: SR Performance on Three Datasets (Table 2, RealSR subset)

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | DISTS↓ | LIQE↑ | MUSIQ↑ | MANIQA↑ | Q-Insight↑ | TOPIQ↑ |
|--------|-------|-------|--------|--------|-------|--------|---------|------------|--------|
| DiffBIR | 23.20 | 0.6346 | 0.3350 | 0.2162 | 3.553 | 65.25 | 0.462 | 3.530 | 0.603 |
| SeeSR | 24.34 | 0.7187 | 0.2754 | 0.2134 | 3.394 | 65.53 | 0.486 | 3.285 | 0.625 |
| OSEDiff | 23.07 | 0.6850 | 0.2941 | 0.2109 | 4.068 | 68.95 | 0.488 | 3.712 | 0.644 |
| UARE | 21.38 | 0.6464 | 0.3095 | 0.2344 | 4.066 | 69.67 | 0.526 | 3.664 | 0.680 |
| Qwen-SFT | 22.71 | 0.6462 | 0.3100 | 0.2203 | 3.815 | 68.57 | 0.490 | 3.545 | 0.640 |
| **OARS** | **22.36** | **0.6481** | **0.3095** | **0.2244** | **4.305** | **71.41** | **0.528** | **3.701** | **0.680** |

**Key Findings**: OARS consistently achieves the best or second-best performance across all NR metrics, while FR metrics (PSNR/SSIM/LPIPS/DISTS) show virtually no degradation relative to Qwen-SFT. Compared to perceptual-oriented methods (PURE, UARE), OARS achieves superior FR metrics, demonstrating that the reward design and RL strategy effectively balance fidelity and perceptual enhancement.

### Ablation Study: Reward Model Components (Table 3)

| Case | Score Calibration | Explicit Fidelity | Quality-Adaptive γ | Accuracy |
|------|:-:|:-:|:-:|------|
| 1 | ✗ | ✗ | ✗ | 78.8% |
| 2 | ✓ | ✗ | ✗ | 81.5% |
| 3 | ✓ | ✓ | ✗ | 82.3% |
| 4 | ✓ | ✓ | γ=5 | 82.7% |
| 5 | ✓ | ✓ | **γ=7** | **83.1%** |
| 6 | ✓ | ✓ | γ=9 | 82.8% |

Three-stage calibration yields +2.7%; explicit fidelity modeling contributes +0.8%; the quality-adaptive $\gamma$ provides a further +0.8%.

### Ablation Study: RL Stages and Initialization Strategy (Table 5, RealSR)

| Method | RL Stage | Initialization | PSNR↑ | LIQE↑ | MUSIQ↑ | TOPIQ↑ |
|--------|----------|----------------|-------|-------|--------|--------|
| Qwen-SFT | - | - | 22.71 | 3.815 | 68.57 | 0.640 |
| Case 1 | stage1 | base | 22.52 | 4.235 | 71.02 | 0.674 |
| Case 2 | stage1+2 | **base** | **22.36** | **4.305** | **71.41** | **0.680** |
| Case 3 | stage1 | sft | 22.15 | 4.078 | 70.60 | 0.676 |
| Case 4 | stage1+2 | sft | 21.31 | 4.094 | 70.56 | 0.677 |

**Key Findings**: Applying RL directly on the SFT model (Cases 3–4) leads to continuous degradation of FR metrics, with PSNR dropping from 22.71 to 21.31 after stage1+2. In contrast, shallow LoRA optimization on the base model (Cases 1–2) is substantially more robust, with only a marginal PSNR decrease and more pronounced NR metric gains. This validates the superiority of the online exploration strategy via shallow LoRA on the base model.

### Other Key Findings

- **User study**: Among 27 expert evaluators, OARS received 47.62% of votes, far exceeding the second-ranked method DP2O-SR (27.68%).
- **SRIQA-Bench preference accuracy**: COMPASS achieves 83.1% All-Acc, surpassing all GT-Ref and GT-Free baselines.
- **Generalizability**: Applying OARS to a Flux backbone is also effective, with MANIQA improving from 0.469 to 0.504.
- **Comparison with Flow-GRPO**: Forward-process RL (DiffusionNFT paradigm) is 5–10× more efficient than trajectory-level RL and more stable under the strong spatial constraints of SR.

## Highlights & Insights

1. **Process-oriented evaluation paradigm**: The paper shifts SR evaluation from an "output-centric" to a "process-aware" perspective, assessing the LR→SR transformation process rather than static outputs. This conceptual shift enables unified modeling of fidelity and perceptual gain.

2. **Three-stage annotation pipeline**: The pipeline elegantly resolves the tension between global comparability and intra-group fine-grained discrimination—global anchors provide comparability, intra-group rankings provide granularity, and linear calibration achieves a unified scale.

3. **Dual role of shallow LoRA**: Applying LoRA on the base model not only provides higher sampling stochasticity for online exploration but also reduces reward hacking risk by avoiding direct modification of SFT weights. Merging the LoRA back into the SFT model at inference time yields an exceptionally elegant design.

4. **Input quality-adaptive gating**: The design of $F^{Q_{LR}/\gamma}$ allows the reward function to automatically adjust the fidelity–perception trade-off based on degradation severity—encouraging conservative enhancement for high-quality inputs while permitting greater perceptual improvement for low-quality inputs—highly intuitive and well-motivated.

## Limitations & Future Work

1. **Computational cost**: Training requires 8×H20 GPUs for RL and 8×A100 GPUs for deploying the reward server, imposing extremely high resource requirements.
2. **Multi-stage training complexity**: The three-stage progressive training pipeline (SFT→FR-RL→NR-RL) increases engineering complexity and the hyperparameter search space.
3. **Reward model generalizability**: COMPASS is validated on SRIQA-Bench, which is relatively small (100 LR images); its generalization to more diverse distributions remains to be verified.
4. **Lack of explicit degradation-aware modeling**: Although degradation severity is implicitly captured via $Q_{LR}$, degradation type is not explicitly modeled, potentially leading to suboptimal handling of specific degradation patterns (e.g., severe compression artifacts).
5. **Only 4× SR validated**: Applicability to other upscaling factors and different resolution ranges has not been investigated.

## Related Work & Insights

- **DiffusionNFT** (forward-process RL) and **Flow-GRPO** (trajectory-level RL): A comparison of these two RL paradigms suggests that forward-process RL is more efficient and stable for generation tasks with strong conditional constraints.
- **DP2O-SR** (offline DPO): This paper clearly articulates the root cause of pseudo-diversity in offline methods when applied to SR tasks.
- **Q-Insight**: Used as global quality anchors and the base comparison model, demonstrating the potential of MLLM-based IQA.
- **DiffIQA/SRIQA-Bench**: These provide benchmarks for SR quality assessment, but their pairwise annotations are incomplete; the three-stage pipeline in this paper is an important complement.
- **Implications**: The concept of process-aware rewards can be generalized to other image enhancement tasks (denoising, dehazing, HDR, etc.), and the input quality-adaptive mechanism is applicable to any scenario requiring a balance between fidelity and enhancement.

## Rating

| Dimension | Score (1–5) | Notes |
|-----------|:-----------:|-------|
| Novelty | 4.5 | The combination of process-aware rewards and progressive online RL represents the first systematic attempt in this domain. |
| Technical Depth | 4.5 | The three-stage annotation, adaptive reward formula, and shallow LoRA exploration strategy are all well-motivated with clear theoretical grounding. |
| Experimental Thoroughness | 4.5 | Three datasets, multiple metrics, user study, extensive ablations, and multi-backbone validation. |
| Writing Quality | 4.0 | Structure is clear, though the intuitive explanation behind some formulas and design choices could be more thorough. |
| Value | 3.5 | The method is effective but resource-intensive, limiting its practical deployment scope. |
| **Overall** | **4.2** | Makes systematic contributions to RLHF for generative SR, with a high level of completeness. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Continuous Degradation Modeling via Latent Flow Matching for Real-World Super-Resolution](../../AAAI2026/image_generation/continuous_degradation_modeling_via_latent_flow_matching_for_real-world_super-re.md)
- [\[CVPR 2026\] FRAMER: Frequency-Aligned Self-Distillation with Adaptive Modulation Leveraging Diffusion Priors for Real-World Image Super-Resolution](framer_frequency-aligned_self-distillation_with_adaptive_modulation_leveraging_d.md)
- [\[CVPR 2026\] VOSR: A Vision-Only Generative Model for Image Super-Resolution](vosr_a_vision_only_generative_model_for_image_super_resolution.md)
- [\[ICLR 2026\] DiffusionNFT: Online Diffusion Reinforcement with Forward Process](../../ICLR2026/image_generation/diffusionnft_online_diffusion_reinforcement_with_forward_process.md)
- [\[CVPR 2026\] Frequency-Aware Flow Matching for High-Quality Image Generation](freqflow_frequency_aware_flow_matching.md)

</div>

<!-- RELATED:END -->

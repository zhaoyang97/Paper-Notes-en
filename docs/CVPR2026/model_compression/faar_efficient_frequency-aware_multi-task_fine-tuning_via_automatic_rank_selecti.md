---
title: >-
  [Paper Note] FAAR: Efficient Frequency-Aware Multi-Task Fine-Tuning via Automatic Rank Selection
description: >-
  [CVPR 2026][Model Compression][LoRA] This paper proposes FAAR, a frequency-aware parameter-efficient fine-tuning method for multi-task learning. It introduces Performance-Driven Rank Shrinking (PDRS) to dynamically selec…
tags:
  - "CVPR 2026"
  - "Model Compression"
  - "LoRA"
  - "automatic rank selection"
  - "FFT"
  - "multi-task learning"
  - "PEFT"
date: 2026-05-08
content_hash: 4813ca3c26bcf52b
---

# FAAR: Efficient Frequency-Aware Multi-Task Fine-Tuning via Automatic Rank Selection

**Conference**: CVPR 2026
**arXiv**: [2603.20403](https://arxiv.org/abs/2603.20403)
**Code**: Available (mentioned in the paper)
**Area**: Parameter-Efficient Fine-Tuning / Multi-Task Learning
**Keywords**: LoRA, automatic rank selection, FFT, multi-task learning, PEFT

## TL;DR

This paper proposes FAAR, a frequency-aware parameter-efficient fine-tuning method for multi-task learning. It introduces Performance-Driven Rank Shrinking (PDRS) to dynamically select the optimal rank for each task and layer, and designs a Task-Spectral Pyramidal Decoder (TS-PD) that leverages FFT frequency information to enhance spatial awareness and cross-task consistency. FAAR achieves superior performance using only 1/9 the parameters of full fine-tuning.

## Background & Motivation

Multi-task learning (MTL) aims to learn multiple tasks simultaneously, sharing representations to discover inter-task relationships and structure. As backbone model sizes continue to grow, traditional full fine-tuning becomes increasingly impractical. Parameter-efficient fine-tuning (PEFT), particularly methods based on Low-Rank Adaptation (LoRA), has become the dominant paradigm.

However, existing LoRA-based MTL methods suffer from two core limitations:

**Fixed-rank problem**: Existing methods apply a uniform rank across all layers and all tasks, which is counter-intuitive — different tasks may require different adaptation strengths, and different layers need different degrees of fine-tuning flexibility. Deeper layers require stronger adaptability to handle task-specific fine-grained information, whereas shallower layers may need only minor adjustments.

**Lack of spatial inductive bias**: Existing LoRA-based MTL strategies overlook the role of cross-task interaction in deeper layers. For dense visual tasks such as semantic segmentation, depth estimation, and surface normal estimation, strong spatial awareness and cross-task geometric consistency are essential — yet low-rank adaptation inherently lacks these capabilities.

FAAR addresses these issues by:
- Resolving the fixed-rank problem via dynamic rank shrinking (PDRS), enabling each task/layer to automatically find its optimal rank.
- Introducing cheap yet effective spatial information and cross-task relationships through frequency analysis (TS-PD).

## Method

### Overall Architecture

FAAR is built upon a frozen Swin Transformer backbone, with DoRA adapters placed in the attention and MLP layers. Within each Transformer stage, the last block uses task-specific adapters while preceding blocks share adapters. A Task-Spectral Pyramidal Decoder (TS-PD) is appended after the backbone for frequency enhancement and cross-task alignment. The entire training process is governed by PDRS, which dynamically reduces adapter ranks.

### Key Designs

1. **Performance-Driven Rank Shrinking (PDRS)**:

    - **Rank Masking**: At each forward pass, a prefix size $b \in \{1, ..., r_{curr}\}$ is randomly sampled to construct a binary mask $m$, allowing only the first $b$ rank components to participate in computation:
        - $A^{eff} = \text{diag}(m) A$, $B^{eff} = B \text{diag}(m)$
        - This compels important rank-1 updates to concentrate toward lower dimensions.
    - **Coverage Strategy**:
        - At each backward pass, an importance score is computed for each active rank $i$: $s_i = \frac{1}{2}(|\langle A_{:,i}^{eff}, \frac{\partial \mathcal{L}}{\partial A_{:,i}^{eff}} \rangle| + |\langle B_{i,:}^{eff}, \frac{\partial \mathcal{L}}{\partial B_{i,:}^{eff}} \rangle|)$
        - Scores are accumulated across batches via EMA: $\hat{s}_i \leftarrow \beta \hat{s}_{i-1} + (1-\beta) s_i$
        - At the end of each epoch, ranks are sorted by score in descending order, and the minimum number of ranks $K$ satisfying coverage ratio $\rho$ is selected: $K = \min\{k : c(k) \geq \rho\}$
        - Uncovered ranks are permanently removed from optimization.
    - **Design Motivation**: The directional derivative of the MTL loss reflects the actual contribution of each rank-1 component; performance-driven shrinking ensures no critical updates are discarded.

2. **DoRA Adapters (instead of LoRA)**:

    - DoRA decouples low-rank adaptation into magnitude and direction: $\text{Out}_i^{DoRA} = m_i \frac{W_i + \alpha B_i A_i}{\|W_i + \alpha B_i A_i\|_2} x + b_i$
    - DoRA is more stable than LoRA at extremely low ranks and pairs better with PDRS rank shrinking.
    - Experimental validation shows that at high ranks DoRA does not necessarily outperform LoRA, but at low ranks DoRA is significantly superior.

3. **Task-Spectral Pyramidal Decoder (TS-PD)**:

    - **Channel-wise Spectral Filter (CW-SP)**:
        - FFT is applied to each task-specific feature, and a task/resolution-specific 2D frequency filter matrix $W_t^{res}$ is learned.
        - Selective amplification/suppression of different frequencies is achieved via element-wise multiplication $Y = W \odot FFT(I)$.
        - After inverse FFT back to feature space, learnable scale/shift parameters are applied for modulation.
        - **Design Motivation**: Different tasks require different frequency information — edge detection relies on high frequencies, while depth estimation leverages both high and low frequencies.

    - **Cross-Task Consensus Alignment (XT-Cons)**:
        - For the primary task, the average representation $F_{avg}$ of auxiliary task spectra is computed.
        - High- and low-frequency masks $M_{low}$, $M_{high}$ are extracted from the primary task spectrum.
        - Alignment differences are computed: $\Delta_{low,high} = M_{low,high} * (F_{avg} - FFT(X_i^{main}))$
        - Contributions are scaled by learnable scalars $\alpha_{low,high}$.
        - **Design Motivation**: The "consensus" of auxiliary tasks in the frequency domain is used to drive geometric consistency in the primary task representation, at lower cost than direct spatial-domain interaction.

### Loss & Training

- MTL loss: $L_{MTL} = \sum_{i=1}^T w \times L_i$
    - Semantic segmentation, human part segmentation: pixel-wise cross-entropy
    - Depth estimation, surface normal estimation: L1 loss
    - Saliency detection: balanced cross-entropy
- Coverage ratio: $\rho_{shared} = \rho_{task} = 0.95$
- Backbone: Swin-Tiny (ImageNet-1k pretrained); Decoder: HRNet
- Initial rank $r_{init} = 64$, dynamically shrunk to approximately $r_{global} \approx 5$ during training
- Single NVIDIA A40; learning rate $5 \times 10^{-4}$; batch size 32

## Key Experimental Results

### Main Results

**PASCAL-Context dataset** (4 tasks):

| Method | SemSeg (mIoU↑) | HumanParts (mIoU↑) | Saliency (mIoU↑) | Normals (rmse↓) | Δm (%) | Params (M) |
|--------|----------------|---------------------|-------------------|-----------------|--------|------------|
| Single Task | 67.21 | 61.93 | 62.35 | 17.97 | 0 | 112.62 |
| MTL Full FT | 67.56 | 60.24 | 65.21 | 16.64 | +2.23 | 30.06 |
| MTLoRA (r=64) | 67.90 | 59.84 | 65.40 | 16.60 | +2.55 | 8.34 |
| TADFormer (r=64) | 70.82 | 60.45 | 65.88 | 16.48 | +4.24 | 7.38 |
| **FAAR** | **72.02** | **61.25** | **66.11** | **16.35** | **+5.28** | **3.38** |

**NYUDv2 dataset** (3 tasks):

| Method | SemSeg (mIoU↑) | Depth (rmse↓) | Normals (rmse↓) | Δm (%) | Params (M) |
|--------|----------------|---------------|-----------------|--------|------------|
| Single Task | 42.65 | 0.60 | 22.83 | 0 | 84.00 |
| MTL Full FT | 38.85 | 0.66 | 24.33 | -8.49 | 28.10 |
| TADFormer (r=64) | 40.85 | 0.64 | 27.48 | -10.42 | 8.90 |
| **FAAR** | **41.27** | **0.63** | **26.35** | **-7.88** | **2.85** |

### Ablation Study

Component ablation on PASCAL-Context:

| Configuration | SemSeg | HumanParts | Saliency | Normals | Δm |
|---------------|--------|------------|----------|---------|-----|
| MTLoRA (r=64) | 67.90 | 59.84 | 65.40 | 16.60 | +2.55 |
| + DoRA (high rank) | 67.55 | 60.00 | 64.70 | 17.20 | +1.36 |
| + PDRS w/ LoRA | 68.11 | 59.93 | 65.54 | 16.50 | +2.83 |
| + PDRS w/ DoRA (1) | 71.35 | 61.02 | 65.92 | 16.42 | +4.92 |
| + TS-PD (2) | 70.73 | 60.95 | 65.92 | 16.40 | +4.63 |
| **FAAR (1+2)** | **72.02** | **61.25** | **66.11** | **16.35** | **+5.28** |

### Key Findings

1. **Rank shrinkage patterns are intuitive**: Task-specific and deeper layers tend to retain larger ranks due to their role in processing fine-grained task-specific information, while shared and shallower layers are significantly pruned.
2. **DoRA substantially outperforms LoRA at low ranks**: At high ranks, DoRA actually underperforms (+1.36 vs. +2.55), but after PDRS shrinks ranks to very low values, DoRA yields a significant advantage (+4.92).
3. **Initial rank has little impact on final performance**: Results are nearly identical for $r_{init} \in \{16, 32, 64\}$, indicating that PDRS provides a sufficiently broad search space.
4. **XT-Cons cross-task alignment is effective**: It provides an additional +0.8% Δm improvement on top of TS-PD, validating the value of cross-task consistency in the frequency domain.
5. **9× parameter savings**: FAAR (3.38M) vs. MTL Full FT (30.06M), with superior performance.

## Highlights & Insights

- **Performance-driven rank shrinking**: Unlike AdaLoRA (singular-value importance) or DyLoRA (robustness to low-rank training), PDRS directly uses directional derivatives of the MTL loss to guide rank pruning, aligning more directly with the optimization objective.
- **Frequency domain as a cross-task bridge**: This is the first work to leverage FFT in dense visual MTL. The frequency domain naturally separates edge/semantic information, providing a meaningful shared basis for different tasks.
- **Synergy between DoRA and extremely low ranks**: At high ranks, DoRA does not necessarily outperform LoRA; however, when ranks are dynamically compressed to very low values by PDRS, DoRA's magnitude-direction decoupling becomes critical.
- **Simultaneous improvement across all tasks**: FAAR outperforms baselines on all 4 PASCAL tasks, with no evidence of sacrificing one task at the expense of others.

## Limitations & Future Work

1. On NYUDv2, all MTL PEFT methods fail to surpass single-task training; FAAR does not fully resolve the difficulty of MTL on small datasets.
2. The coverage ratio $\rho$ still requires manual specification (although the paper selects 0.95 on a validation set, different datasets may require different values).
3. Only the Swin-Tiny backbone is evaluated; the effectiveness on larger backbones (e.g., Swin-Base/Large) or ViT architectures remains unknown.
4. The frequency filter matrices in TS-PD are learned separately for each resolution and task, leading to parameter growth as the number of tasks increases.
5. Cross-task alignment is performed only in the frequency domain; spatial-domain interactions may provide additional complementary information.

## Related Work & Insights

- **MTLoRA / TADFormer**: LoRA-based MTL baselines using fixed ranks.
- **AdaLoRA / AutoLoRA / DyLoRA**: Automatic rank selection methods for single-task settings; FAAR extends these to the multi-task regime.
- **FADA / NightAdapter**: Frequency adapters for domain generalization and nighttime segmentation, which inspired the design of TS-PD.
- **DiTASK**: An alternative approach adapting singular values via neural diffeomorphisms.

## Rating

- Novelty: ⭐⭐⭐⭐ (PDRS and TS-PD are individually novel, though both represent improved combinations of existing ideas)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Two datasets, detailed ablations, and comprehensive parameter efficiency comparisons)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, though the density of formulas and abbreviations somewhat reduces readability)
- Value: ⭐⭐⭐⭐ (Provides a practical and efficient solution for MTL PEFT; the 9× parameter savings is highly attractive)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Frequency Switching Mechanism for Parameter-Efficient Multi-Task Learning](frequency_switching_mechanism_for_parameter-ecient_multi-task_learning.md)
- [\[ICCV 2025\] TR-PTS: Task-Relevant Parameter and Token Selection for Efficient Tuning](../../ICCV2025/model_compression/tr-pts_task-relevant_parameter_and_token_selection_for_efficient_tuning.md)
- [\[ICLR 2026\] LoFT: Low-Rank Adaptation That Behaves Like Full Fine-Tuning](../../ICLR2026/model_compression/loft_low-rank_adaptation_that_behaves_like_full_fine-tuning.md)
- [\[NeurIPS 2025\] RefLoRA: Refactored Low-Rank Adaptation for Efficient Fine-Tuning of Large Models](../../NeurIPS2025/model_compression/reflora_refactored_low-rank_adaptation_for_efficient_fine-tuning_of_large_models.md)
- [\[ACL 2026\] SAMoRA: Semantic-Aware Mixture of LoRA Experts for Task-Adaptive Learning](../../ACL2026/model_compression/samora_semantic-aware_mixture_of_lora_experts_for_task-adaptive_learning.md)

</div>

<!-- RELATED:END -->

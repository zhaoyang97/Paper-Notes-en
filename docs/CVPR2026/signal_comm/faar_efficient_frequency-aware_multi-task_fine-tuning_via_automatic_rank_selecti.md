---
title: >-
  [Paper Note] FAAR: Efficient Frequency-Aware Multi-Task Fine-Tuning via Automatic Rank Selection
description: >-
  [CVPR 2026][Signal & Communication][LoRA] This paper proposes FAAR, a frequency-aware parameter-efficient fine-tuning method for multi-task learning. It introduces Performance-Driven Rank Shrinking (PDRS) to dynamically select the optimal rank per task and per layer, and designs a Task-Spectral Pyramidal Decoder (TS-PD) that leverages FFT frequency information to enhance spatial awareness and cross-task consistency. FAAR achieves superior performance using only 1/9 the parameters of full fine-tuning.
tags:
  - CVPR 2026
  - "Signal & Communication"
  - LoRA
  - automatic rank selection
  - FFT
  - multi-task learning
  - PEFT
date: 2026-05-08
content_hash: 0efec4d9f4740d1d
---

# FAAR: Efficient Frequency-Aware Multi-Task Fine-Tuning via Automatic Rank Selection

**Conference**: CVPR 2026
**arXiv**: [2603.20403](https://arxiv.org/abs/2603.20403)
**Code**: Available (mentioned in paper)
**Area**: Parameter-Efficient Fine-Tuning / Multi-Task Learning
**Keywords**: LoRA, automatic rank selection, FFT, multi-task learning, PEFT

## TL;DR

This paper proposes FAAR, a frequency-aware parameter-efficient fine-tuning method for multi-task learning. It introduces Performance-Driven Rank Shrinking (PDRS) to dynamically select the optimal rank per task and per layer, and designs a Task-Spectral Pyramidal Decoder (TS-PD) that leverages FFT frequency information to enhance spatial awareness and cross-task consistency. FAAR achieves superior performance using only 1/9 the parameters of full fine-tuning.

## Background & Motivation

Multi-task learning (MTL) aims to learn multiple tasks simultaneously, leveraging shared representations to discover inter-task relationships and structure. As backbone model sizes continue to grow, traditional full fine-tuning becomes increasingly impractical. Parameter-efficient fine-tuning (PEFT), particularly methods based on Low-Rank Adaptation (LoRA), has become the dominant approach.

However, existing LoRA-based MTL methods suffer from two core limitations:

**Fixed-rank problem**: Existing methods apply a uniform rank across all layers and tasks, which is counterintuitive — different tasks may require different adaptation strengths, and different layers may need varying degrees of fine-tuning flexibility. Deeper layers require stronger adaptation capacity to handle task-specific fine-grained information, while shallower layers may require only minor adjustments.

**Lack of spatial inductive bias**: Existing LoRA-based MTL strategies overlook cross-task interactions in deeper layers. For dense visual tasks such as semantic segmentation, depth estimation, and surface normal estimation, strong spatial awareness and cross-task geometric consistency are critical, yet low-rank adaptation inherently lacks such capability.

FAAR addresses these issues by:
- Resolving the fixed-rank problem through dynamic rank shrinking (PDRS), enabling each task/layer to automatically identify its optimal rank.
- Introducing inexpensive yet effective spatial information and cross-task relationships through frequency analysis (TS-PD).

## Method

### Overall Architecture

FAAR is built upon a frozen Swin Transformer backbone, with DoRA adapters placed in the attention and MLP layers. Within each Transformer stage, the final block uses task-specific adapters while preceding blocks share adapters. The backbone is followed by a Task-Spectral Pyramidal Decoder (TS-PD) for frequency enhancement and cross-task alignment. The entire training process is governed by PDRS, which dynamically reduces adapter rank.

### Key Designs

1. **Performance-Driven Rank Shrinking (PDRS)**:

    - **Rank Masking**: At each forward pass, a prefix size $b \in \{1, ..., r_{curr}\}$ is randomly sampled to construct a binary mask $m$, allowing only the first $b$ rank components to participate in computation:
        - $A^{eff} = \text{diag}(m) A$, $B^{eff} = B \text{diag}(m)$
        - This forces important rank-1 updates to concentrate toward lower dimensions.
    - **Coverage Strategy**:
        - At each backward pass, an importance score is computed for each active rank $i$: $s_i = \frac{1}{2}(|\langle A_{:,i}^{eff}, \frac{\partial \mathcal{L}}{\partial A_{:,i}^{eff}} \rangle| + |\langle B_{i,:}^{eff}, \frac{\partial \mathcal{L}}{\partial B_{i,:}^{eff}} \rangle|)$
        - Scores are accumulated across batches via EMA: $\hat{s}_i \leftarrow \beta \hat{s}_{i-1} + (1-\beta) s_i$
        - At the end of each epoch, scores are sorted in descending order and the minimum number of ranks $K$ satisfying coverage ratio $\rho$ is selected: $K = \min\{k : c(k) \geq \rho\}$
        - Ranks that are not covered are permanently removed from optimization.
    - **Design Motivation**: The directional derivative of the MTL loss reflects the actual contribution of each rank-1 component. Performance-driven shrinking ensures that critical updates are preserved.

2. **DoRA Adapters (instead of LoRA)**:

    - DoRA decouples low-rank adaptation into magnitude and direction: $\text{Out}_i^{DoRA} = m_i \frac{W_i + \alpha B_i A_i}{\|W_i + \alpha B_i A_i\|_2} x + b_i$
    - DoRA is more stable than LoRA at extremely low ranks and works more effectively with PDRS rank shrinking.
    - Empirical validation shows that at high ranks DoRA does not necessarily outperform LoRA, but at low ranks DoRA is substantially better.

3. **Task-Spectral Pyramidal Decoder (TS-PD)**:

    - **Channel-wise Spectral Filter (CW-SP)**:
        - FFT is applied to each task-specific feature map; a task- and resolution-specific 2D frequency filter matrix $W_t^{res}$ is learned.
        - Selected frequencies are selectively enhanced or suppressed via element-wise multiplication: $Y = W \odot FFT(I)$.
        - After inverse FFT back to feature space, the output is modulated with learnable scale and shift parameters.
        - **Design Motivation**: Different tasks require different frequency information — edge detection relies on high frequencies, while depth estimation leverages both high and low frequencies.

    - **Cross-Task Consensus Alignment (XT-Cons)**:
        - For the primary task, an average spectral representation $F_{avg}$ is computed from auxiliary task spectra.
        - High- and low-frequency masks $M_{low}$, $M_{high}$ are extracted from the primary task spectrum.
        - Alignment differences are computed as: $\Delta_{low,high} = M_{low,high} * (F_{avg} - FFT(X_i^{main}))$
        - Contributions are scaled by learnable scalars $\alpha_{low,high}$.
        - **Design Motivation**: The frequency-domain "consensus" of auxiliary tasks is used to drive geometric consistency in the primary task representation, at lower cost than direct spatial-domain interaction.

### Loss & Training

- MTL loss: $L_{MTL} = \sum_{i=1}^T w \times L_i$
    - Semantic segmentation and human part segmentation: pixel-wise cross-entropy
    - Depth estimation and surface normal estimation: L1 loss
    - Saliency detection: balanced cross-entropy
- Coverage ratio: $\rho_{shared} = \rho_{task} = 0.95$
- Backbone: Swin-Tiny (ImageNet-1k pretrained); decoder: HRNet
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

1. **Rank shrinkage patterns are intuitive**: Task-specific and deeper layers tend to retain larger ranks, as they handle finer task-specific information; shared and shallower layers have their ranks substantially reduced.
2. **DoRA significantly outperforms LoRA at low ranks**: At high ranks, DoRA performs worse (+1.36 vs. +2.55), but after PDRS shrinks the rank to low values, DoRA yields a substantial advantage (+4.92).
3. **Initial rank has minimal impact on final performance**: Results are nearly identical across $r_{init} \in \{16, 32, 64\}$, indicating that the PDRS search space is sufficient.
4. **Cross-task alignment via XT-Cons is effective**: An additional +0.8% Δm gain over TS-PD alone validates the value of frequency-domain cross-task consistency.
5. **9× parameter reduction**: FAAR (3.38M) vs. MTL Full FT (30.06M), with superior performance.

## Highlights & Insights

- **Performance-driven rank shrinking**: Unlike AdaLoRA (singular value importance) or DyLoRA (robustness to low-rank training), PDRS directly uses the directional derivative of the MTL loss to guide rank pruning, aligning more directly with the optimization objective.
- **Frequency domain as a cross-task bridge**: This work is the first to leverage FFT for dense visual MTL. The frequency domain naturally separates edge and semantic information, providing a meaningful shared basis for different tasks.
- **Synergy between DoRA and extremely low rank**: At high ranks, DoRA does not necessarily outperform LoRA, but when ranks are dynamically compressed to very low values by PDRS, the magnitude-direction decoupling in DoRA becomes critical.
- **Simultaneous improvement across all tasks**: FAAR outperforms baselines on all 4 PASCAL tasks without sacrificing any task for another.

## Limitations & Future Work

1. On NYUDv2, no MTL PEFT method surpasses single-task training; FAAR does not fully resolve the difficulty of MTL on small datasets.
2. The coverage ratio $\rho$ still requires manual specification (although the paper reports 0.95 selected on the validation set, different datasets may require different values).
3. Only the Swin-Tiny backbone is evaluated; effectiveness on larger backbones (e.g., Swin-Base/Large) or ViT architectures remains unknown.
4. The TS-PD frequency filter matrices are learned separately for each resolution and task, causing parameter count to grow with the number of tasks.
5. Cross-task alignment is performed only in the frequency domain; spatial-domain interactions may provide additional complementary information.

## Related Work & Insights

- **MTLoRA / TADFormer**: LoRA-based baselines for MTL with fixed ranks.
- **AdaLoRA / AutoLoRA / DyLoRA**: Automatic rank selection methods for single-task settings; FAAR extends these to the multi-task setting.
- **FADA / NightAdapter**: Frequency adapters applied to domain generalization and nighttime segmentation, which inspired the design of TS-PD.
- **DiTASK**: An alternative approach that adapts singular values via neural diffeomorphisms.

## Rating

- Novelty: ⭐⭐⭐⭐ (Both PDRS and TS-PD are individually novel, though each represents an incremental combination of existing ideas)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Two datasets, detailed ablations, and comprehensive parameter efficiency comparisons)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, though the density of formulas and abbreviations somewhat hinders readability)
- Value: ⭐⭐⭐⭐ (Provides a practical and efficient solution for MTL PEFT; the 9× parameter reduction is compelling)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] FASA: Frequency-Aware Sparse Attention](../../ICLR2026/signal_comm/fasa_frequency-aware_sparse_attention.md)
- [\[AAAI 2026\] Task Aware Modulation Using Representation Learning for Upscaling of Terrestrial Carbon Fluxes](../../AAAI2026/signal_comm/task_aware_modulation_using_representation_learning_for_upsaling_of_terrestrial_.md)
- [\[ICLR 2026\] Multi-modal Data Spectrum: Multi-modal Datasets are Multi-dimensional](../../ICLR2026/signal_comm/multi-modal_data_spectrum_multi-modal_datasets_are_multi-dimensional.md)
- [\[ICLR 2026\] Spectrum Tuning: Post-Training for Distributional Coverage and In-Context Steerability](../../ICLR2026/signal_comm/spectrum_tuning_post-training_for_distributional_coverage_and_in-context_steerab.md)
- [\[NeurIPS 2025\] Perturbation Bounds for Low-Rank Inverse Approximations under Noise](../../NeurIPS2025/signal_comm/perturbation_bounds_for_low-rank_inverse_approximations_under_noise.md)

</div>

<!-- RELATED:END -->

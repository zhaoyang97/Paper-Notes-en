---
title: >-
  [Paper Note] Beyond Loss Values: Robust Dynamic Pruning via Loss Trajectory Alignment
description: >-
  [CVPR 2026][Model Compression][dynamic data pruning] This paper proposes AlignPrune—a plug-and-play module based on loss trajectory alignment—that replaces conventional loss-value ranking with a Dynamic Alignment Score (DAS), achieving up to 6.3% accuracy improvement over standard dynamic data pruning methods under noisy label settings.
tags:
  - CVPR 2026
  - Model Compression
  - dynamic data pruning
  - noisy labels
  - loss trajectory
  - plug-and-play module
  - training efficiency
date: 2026-05-08
content_hash: dbcc5504ccd37254
---

# Beyond Loss Values: Robust Dynamic Pruning via Loss Trajectory Alignment

**Conference**: CVPR 2026
**arXiv**: [2604.07306](https://arxiv.org/abs/2604.07306)
**Code**: [GitHub](https://github.com/leonqin430/AlignPrune)
**Area**: Model Compression / Data Pruning
**Keywords**: dynamic data pruning, noisy labels, loss trajectory, plug-and-play module, training efficiency

## TL;DR
This paper proposes AlignPrune—a plug-and-play module based on loss trajectory alignment—that replaces conventional loss-value ranking with a Dynamic Alignment Score (DAS), achieving up to 6.3% accuracy improvement over standard dynamic data pruning methods under noisy label settings.

## Background & Motivation
**Background**: Training on large-scale datasets incurs substantial computational cost. Data pruning reduces training overhead by discarding uninformative samples. Dynamic pruning (adaptively selecting subsets each epoch) is more flexible and robust than static pruning (a one-time selection).

**Limitations of Prior Work**: Existing dynamic pruning methods (InfoBatch, SeTa) rank samples by **individual loss values**, retaining those with high loss. However, under noisy labels, **noisy samples tend to produce high losses**, causing them to be preferentially retained and corrupting the training process.

**Key Challenge**: The loss trajectory of clean samples follows a **smooth, monotonically decreasing** pattern, whereas that of noisy samples exhibits **non-monotonic, irregular fluctuations**. This distinction in temporal behavior can be exploited.

**Core Idea**: Rather than relying on single-point loss values, sample quality is assessed by the **correlation** between a sample's loss trajectory and that of a clean reference set. Samples with low correlation are more likely to be noisy and should be pruned.

## Method

### Overall Architecture
Input: noisy training set $\mathcal{D}$ + small clean reference set $\mathcal{D}_{ref}$ → compute DAS each epoch → replace original loss ranking → select dynamic pruning subset → standard training.

### Key Designs
1. **Loss Trajectory**: For each sample $i$, the loss sequence over the most recent $N$ epochs is maintained as $\mathbf{v}_i^{(t)} = [\ell_i^{(t-N+1)}, \ldots, \ell_i^{(t)}]$. The average loss trajectory $\mathbf{v}_{ref}^{(t)}$ is computed over the clean reference set.

   - **Design Motivation**: Single-point loss cannot distinguish *hard samples* from *noisy samples* (both exhibit high loss), but their **temporal dynamics** differ substantially.

2. **Dynamic Alignment Score (DAS)**: The Pearson correlation coefficient between a sample's loss trajectory and the reference trajectory is computed as:
   $$DAS_i^{(t)} = \rho(\mathbf{v}_i^{(t)}, \mathbf{v}_{ref}^{(t)})$$
   - Positive DAS → learning dynamics aligned with clean patterns → likely clean.
   - Negative DAS → learning dynamics conflicting with clean patterns → likely noisy.
   - **Rationale for Pearson**: Scale-invariant (insensitive to absolute loss magnitude) and computationally efficient.

3. **AlignPrune Plug-and-Play Integration**: The loss ranking in InfoBatch/SeTa is directly replaced by DAS ranking: $score_i^{(t)} := DAS_i^{(t)}$. No modifications to model architecture, training pipeline, or gradient update rules are required. The unbiased gradient expectation property of the original methods is preserved.

### Loss & Training
- The training objective remains unchanged; only the sample selection strategy is modified.
- Loss trajectories are stored in a fixed-window memory bank of size $N$; batch-level vectorized computation incurs negligible overhead.
- The reference set $\mathcal{D}_{ref}$ is assumed to be clean; experiments demonstrate robustness even when it contains a small proportion of noisy samples.

## Key Experimental Results

### Main Results (CIFAR-100N, ResNet-18, ~30% pruning ratio)

| Method | Clean | Real | Sym-0.5 | Sym-0.8 | Asym-0.2 | Avg. Δ |
|---|---|---|---|---|---|---|
| Full-training | 78.2 | 56.1 | 58.6 | 39.8 | 72.4 | -- |
| InfoBatch | 79.0 | 56.1 | 59.7 | 41.8 | 71.9 | +0.6 |
| **InfoBatch+Ours** | **79.3** | **59.4** | **66.0** | 41.8 | **72.6** | **+2.7** |
| SeTa | 79.0 | 55.6 | 59.0 | 41.6 | 71.4 | +0.0 |
| **SeTa+Ours** | **79.3** | **56.3** | **60.5** | 41.6 | **71.9** | **+0.7** |

### Ablation Study

| Configuration | Key Result | Remarks |
|---|---|---|
| Correlation function | Pearson > Spearman > Cosine | Pearson balances scale-invariance and efficiency |
| Window size $N$ | $N=10$ optimal | Too small → noise-sensitive; too large → slow response |
| Reference set size | 1% of data is sufficient | Minimal clean reference suffices |
| Noisy reference set | Robust up to 10% noise | Averaging naturally suppresses noise |
| Large-scale data | Effective on WebVision / Clothing-1M / ImageNet | Method scales to real-world settings |

### Key Findings
- Under high-noise conditions (Symmetric-0.5), AlignPrune yields a **+6.3%** improvement over InfoBatch.
- Performance on clean-label settings is maintained or slightly improved, without degrading the base method.
- Training efficiency also improves: higher accuracy is achieved while reducing total training time.

## Highlights & Insights
- **Elegant and effective**: Replacing a single ranking criterion yields substantial gains, underscoring that *the right signal matters more than a complex method*.
- Temporal patterns in loss trajectories serve as a strong signal for distinguishing clean from noisy samples—a dimension previously overlooked in the data pruning literature.
- The reference set requirement is minimal (1% of data), making it practically accessible.

## Limitations & Future Work
- A small clean reference set is required, which, though modest, constitutes an additional assumption.
- DAS cannot be computed during the first $N$ epochs before trajectories are established; the method falls back to loss ranking in this early phase.
- Validation is limited to classification tasks; applicability to detection, segmentation, and other downstream tasks remains unexplored.
- The potential of combining AlignPrune with other noisy-label learning methods (e.g., DivideMix) has not been fully investigated.

## Related Work & Insights
- AlignPrune contrasts with the static robust pruning method Prune4ReL: the dynamic + DAS combination substantially outperforms static robust approaches.
- The loss trajectory alignment idea generalizes naturally to active learning and curriculum learning settings.
- The method has direct applicability to large-scale pre-training data curation.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The loss trajectory alignment paradigm is novel and elegant, representing the first application of dynamic pruning to noisy label settings.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Five datasets, multiple noise types, varied pruning ratios, and comprehensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is clearly articulated, figures are intuitive, and theoretical analysis is complete.
- **Value**: ⭐⭐⭐⭐ — The plug-and-play design confers strong practical utility.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Batch Loss Score for Dynamic Data Pruning](batch_loss_score_for_dynamic_data_pruning.md)
- [\[CVPR 2026\] QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models](quantvla_scale-calibrated_post-training_quantization_for_vision-language-action_.md)
- [\[CVPR 2026\] Fixed Anchors Are Not Enough: Dynamic Retrieval and Persistent Homology for Dataset Distillation](fixed_anchors_are_not_enough_dynamic_retrieval_and_persistent_homology_for_datas.md)
- [\[CVPR 2026\] Critical Patch-Aware Sparse Prompting with Decoupled Training for Continual Learning on the Edge](critical_patch-aware_sparse_prompting_with_decoupled_training_for_continual_lear.md)
- [\[CVPR 2026\] PPCL: Pluggable Pruning with Contiguous Layer Distillation for Diffusion Transformers](ppcl_pluggable_pruning_dit_distillation.md)

<!-- RELATED:END -->

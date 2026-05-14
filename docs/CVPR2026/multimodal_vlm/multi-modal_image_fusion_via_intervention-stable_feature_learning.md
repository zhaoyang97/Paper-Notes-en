---
title: >-
  [Paper Note] Multi-Modal Image Fusion via Intervention-Stable Feature Learning
description: >-
  [CVPR 2026][Multimodal VLM][Multi-modal image fusion] This paper proposes a causal inference-inspired multi-modal image fusion framework that employs three structured intervention strategies (complementary masking…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "Multi-modal image fusion"
  - "causal inference"
  - "intervention learning"
  - "infrared-visible fusion"
  - "feature stability"
date: 2026-05-08
content_hash: 0648445f0670533d
---

# Multi-Modal Image Fusion via Intervention-Stable Feature Learning

**Conference**: CVPR 2026
**arXiv**: [2603.23272](https://arxiv.org/abs/2603.23272)
**Code**: Coming soon
**Area**: Multimodal VLM
**Keywords**: Multi-modal image fusion, causal inference, intervention learning, infrared-visible fusion, feature stability

## TL;DR

This paper proposes a causal inference-inspired multi-modal image fusion framework that employs three structured intervention strategies (complementary masking, random masking, and modality dropout) to probe genuine inter-modal dependencies, and designs a Causal Feature Integrator (CFI) to learn intervention-stable features. The method achieves PSNR of 66.02 and AG of 4.129 on MSRS, and mAP of 0.821 on object detection.

## Background & Motivation

1. **Background**: Multi-modal image fusion (MMIF) integrates complementary information from different modalities into a unified representation. Infrared-visible image fusion (IVIF) is the most representative sub-task, fusing thermal semantics from infrared and texture details from visible light. Current SOTA methods employ complex architectures (dual-stream CNNs, Transformer-based global attention, diffusion models) to model cross-modal relationships.

2. **Limitations of Prior Work**: All existing methods share a fundamental limitation—they learn from observational data without distinguishing genuine complementary relationships from spurious statistical regularities. When thermal signals systematically co-occur with specific visible patterns in the training set, models capture these statistical associations rather than understanding whether they reflect meaningful dependencies. This leads to feature selection based on co-occurrence frequency rather than actual contribution to fusion quality.

3. **Key Challenge**: Correlation $\neq$ causation. Models trained solely on input-output pairs cannot determine whether observed inter-modal correlations are causal or coincidental. According to Pearl's causal hierarchy, current MMIF methods operate entirely at the "association" level, lacking the reasoning capabilities of the "intervention" and "counterfactual" levels.

4. **Goal**: How to design principled intervention strategies to probe genuine inter-modal dependencies, and learn fusion features that remain stable across intervention patterns, thereby overcoming the fragility caused by spurious correlations?

5. **Key Insight**: Inspired by Pearl's causal hierarchy, three complementary structured perturbation strategies are designed, each testing a different aspect of modal relationships. The core hypothesis is that features truly important for fusion should maintain their importance across different intervention patterns, while spurious correlations will collapse under perturbation.

6. **Core Idea**: Replace "passive observation + statistical fitting" with "active perturbation + stability selection"—systematically intervene on inputs to discover features that are invariant across interventions, serving as reliable bases for fusion decisions.

## Method

### Overall Architecture

A U-Net-style siamese architecture is adopted. Two weight-sharing encoders process the visible and infrared inputs respectively, generating three-scale features $\{\Theta_1^v, \Theta_2^v, \Theta_3^v\}$ and $\{\Theta_1^i, \Theta_2^i, \Theta_3^i\}$. CFI (Causal Feature Integrator) modules are embedded in the decoder to perform intervention-aware fusion at each scale. During training, the model simultaneously executes all three interventions and outputs four fusion results (standard + three interventions), jointly constrained by three losses.

### Key Designs

1. **Three Principled Intervention Strategies**:

    - **Function**: Probe genuine inter-modal dependencies from three dimensions.
    - **Mechanism**:
        - **Complementary Masking**: Spatially disjoint masks are applied to the two modalities, $\mathcal{M}^v \cap \mathcal{M}^i = \mathbf{O}$, such that regions masked in one modality are exactly the regions preserved in the other. If the fusion result remains high quality, it demonstrates that the two modalities can genuinely compensate for each other rather than encoding homogeneous information. This tests **cross-modal complementarity**.
        - **Random Masking**: The same random mask $\mathcal{M}^r$ is applied to both modalities simultaneously, occluding the same regions in both. Feature combinations that maintain fusion quality under partial observability represent robust local dependencies. This tests **local sufficiency**.
        - **Modality Dropout**: One modality is entirely removed (replaced with all zeros) to measure each modality's indispensable contribution. This prevents the model from over-relying on a single modality. This tests **global necessity**.
    - **Design Motivation**: The three interventions work synergistically—complementary masking ensures genuine cross-modal interaction, random masking discovers robust local patterns, and modality dropout prevents degenerate solutions.

2. **Causal Feature Integrator (CFI)**:

    - **Function**: Identify and prioritize intervention-stable features at each scale.
    - **Mechanism**: At scale $k$, bidirectional cross-modal attention first exchanges information—visible queries attend to infrared keys/values to produce $\Theta_k^{v \to i}$, and vice versa for $\Theta_k^{i \to v}$. To reduce computational cost, keys and values are spatially pooled to $r \times r$. Complementary features $\Theta_k^c = \Theta_k^{v \to i} + \Theta_k^{i \to v}$ and local features $\Theta_k^l = \Theta_k^i + \Theta_k^v$ are then aggregated respectively. The most critical component is the **learnable invariance gate** $\mathcal{G}_k = \sigma(\text{Conv}_{3 \times 3}(\Theta_k^c))$, which mixes complementary and local features: $\Theta_k^{\text{CFI}} = \mathcal{G}_k \odot \Theta_k^c + (1 - \mathcal{G}_k) \odot \Theta_k^l$. High gate values → cross-modal complementary features (intervention-stable); low gate values → local modal features (potentially spurious).
    - **Design Motivation**: Conventional attention mechanisms weight features based on statistical salience, whereas CFI explicitly models whether features are "stable under intervention" through gating, thereby prioritizing robust dependencies over spurious correlations.

3. **Multi-Level Loss Joint Training**:

    - **Function**: Simultaneously optimize fusion quality and intervention stability.
    - **Mechanism**: Total loss $\mathcal{L} = \mathcal{L}_f + \alpha \mathcal{L}_{\text{inv}} + \beta \mathcal{L}_{\text{nec}}$. The fusion fidelity loss $\mathcal{L}_f$ includes L1 reconstruction and Laplacian gradient preservation. The intervention consistency loss $\mathcal{L}_{\text{inv}}$ penalizes discrepancies between pre- and post-intervention outputs in gate-selected stable regions, with regularization to prevent gate degeneracy (mean constraint + spatial entropy to encourage binary decisions). The modality necessity loss $\mathcal{L}_{\text{nec}}$ maximizes the difference between the standard output and single-modality outputs.
    - **Design Motivation**: The three losses respectively ensure fusion quality, intervention stability, and balanced use of both modalities; removing any one leads to specific failure modes.

### Loss & Training

- **Fusion fidelity loss**: $\mathcal{L}_f = \|I_f - I_{vi}\|_1 + \|I_f - I_{ir}\|_1 + \lambda_1 \|\nabla I_f - \max(\nabla I_{vi}, \nabla I_{ir})\|_1$
- **Intervention consistency loss**: Penalizes deviations of complementary/random-masked fusion from the standard fusion within gate-selected regions.
- **Modality necessity loss**: $\mathcal{L}_{\text{nec}} = \|I_f - I_f^i\|_1 + \|I_f - I_f^v\|_1$
- Hyperparameters: $\alpha = 0.1$, $\beta = 0.05$, $\lambda_1 = 1.0$, mask size $16 \times 16$, number of masks randomly sampled from 1–6.
- Trained on RTX 4090 for 50 epochs, Adam optimizer, lr=1e-4, batch size 16.

## Key Experimental Results

### Main Results (Infrared-Visible Image Fusion)

| Method | TNO-AG | TNO-PSNR | MSRS-AG | MSRS-PSNR | MSRS-CC | M3FD-AG | M3FD-PSNR |
|--------|--------|----------|---------|-----------|---------|---------|-----------|
| DCEvo | 3.942 | 61.24 | 3.807 | 64.49 | 0.605 | 4.575 | 61.33 |
| Conti | 3.860 | 61.12 | 3.737 | 64.26 | 0.603 | 4.476 | 61.11 |
| LRRNet | 3.855 | 61.72 | 2.672 | 64.68 | 0.515 | 3.613 | 62.95 |
| **Ours** | **5.128** | **62.06** | **4.129** | **66.02** | **0.646** | **5.276** | **62.13** |

| Downstream Task | Method | Metric |
|-----------------|--------|--------|
| Object Detection (M3FD) | Ours | mAP=**0.821** |
| Object Detection (M3FD) | SAGE | mAP=0.815 |
| Semantic Segmentation (MSRS) | Ours | mIoU=**0.747** |
| Semantic Segmentation (MSRS) | A2RNet | mIoU=0.740 |

### Ablation Study

| Configuration | AG | SF | PSNR | CC | Qabf |
|---------------|----|----|------|----|------|
| w/o CFI | 5.764 | 5.972 | 60.21 | 0.544 | 0.428 |
| w/o L_inv | 5.179 | 5.728 | 58.08 | 0.573 | 0.331 |
| w/o L_nec | 4.016 | 4.018 | 61.39 | 0.393 | 0.368 |
| w/o L_nec & L_inv | 3.361 | 3.478 | 59.85 | 0.524 | 0.312 |
| w/o Int (L_f only) | 5.332 | 5.348 | **63.95** | **0.598** | 0.524 |
| **Full Model** | **6.136** | **6.244** | 63.62 | 0.605 | 0.467 |

### Key Findings

- **Core trade-off between intervention and non-intervention**: The w/o Int variant (pure correlation learning) achieves higher PSNR and Qabf, yet shows significantly lower AG and SF (structural integrity and texture richness) compared to the full model. This reveals an inherent tension in fusion objectives—correlation-driven optimization favors pixel fidelity, while the intervention-driven framework prioritizes structural preservation.
- **Modality necessity loss has the largest impact**: Removing $\mathcal{L}_{\text{nec}}$ causes AG to drop sharply from 6.136 to 4.016 and SF from 6.244 to 4.018, indicating that without this constraint the model heavily favors a single modality.
- **CFI removal leads to noise and structural distortion**: Although edge metrics remain reasonable (AG=5.764), visualizations reveal pronounced noise and structural artifacts.
- **ATE analysis validates intervention effects**: Modality dropout has the greatest impact (as expected), random masking has the smallest (indicating successful learning of locally sufficient features), and complementary masking has intermediate impact (indicating established cross-modal compensation capability).
- **Cross-domain generalization**: A model trained on IVIF transfers directly to medical image fusion (MRI-PET/SPECT) without fine-tuning, still achieving the best AG and SF, demonstrating that intervention learning captures universal fusion principles.

## Highlights & Insights

- The framework design for **introducing causal reasoning into image fusion** shows considerable conceptual depth: rather than superficially labeling the work as "causal," the paper concretely designs three intervention strategies that separately test complementarity, local sufficiency, and global necessity, and quantifies intervention effects via ATE analysis, forming a complete causal analysis loop.
- **"Intervention stability" as a feature selection criterion** is highly transferable: it is applicable not only to image fusion but can be generalized to any multi-modal task requiring robust feature selection (e.g., multi-modal sentiment analysis, sensor fusion).
- **The comparison between w/o Int and the full model** reveals a deeper insight—PSNR is not the ultimate metric for fusion; structural and texture preservation (AG/SF) may be more important for downstream tasks. This has implications for the evaluation paradigm in the fusion community.

## Limitations & Future Work

- The specific parameters of the intervention strategies (mask size, number of masks) are largely tuned empirically, lacking theoretical guidance.
- The weights of the three intervention strategies ($\alpha=0.1$, $\beta=0.05$) are manually set and may not be optimal.
- Validation is limited to IVIF and medical fusion; other modality combinations (e.g., RGB-depth, RGB-event) are not explored.
- The "causal" framework is more heuristic than rigorous—complementary masking resembles data augmentation more than strict causal intervention.
- Computational overhead is not reported—simultaneously executing three interventions implies at least a 3× increase in forward passes during training.

## Related Work & Insights

- **vs CDDFuse**: CDDFuse jointly extracts global and local features via Transformer+CNN but relies entirely on statistical learning. This paper uses intervention training to distinguish genuine complementarity from spurious correlations.
- **vs Mask-DiFuser**: A diffusion model-driven fusion method with high generation quality but no consideration of feature robustness. The proposed intervention framework could be combined with diffusion models.
- **vs causal representation learning literature**: This paper transfers causal ideas from low-light enhancement and self-supervised learning to the fusion task with important adaptations—fusion lacks explicit feature preservation labels, and modal complementarity often violates independence assumptions.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The perspective of introducing causal reasoning into image fusion is novel, and the three intervention strategies are principled in design; however, the "causal" framing is more inspirational than strictly formalized.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Three IVIF benchmarks + cross-domain medical fusion + object detection/segmentation downstream tasks, detailed ablations, and persuasive ATE analysis.
- **Writing Quality**: ⭐⭐⭐⭐ The causal motivation is derived with clear logic, though some notation could be made more consistent.
- **Value**: ⭐⭐⭐⭐ Proposes a new training paradigm for the fusion field (intervention learning over pure correlation learning), with experimental validation of its effectiveness and generalizability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] VideoFusion: A Spatio-Temporal Collaborative Network for Multi-modal Video Fusion](videofusion_a_spatio-temporal_collaborative_network_for_multi-modal_video_fusion.md)
- [\[CVPR 2026\] UniMMAD: Unified Multi-Modal and Multi-Class Anomaly Detection via MoE-Driven Feature Decompression](unimmad_unified_multi-modal_and_multi-class_anomaly_detection_via_moe-driven_fea.md)
- [\[ACL 2026\] LaMI: Augmenting Large Language Models via Late Multi-Image Fusion](../../ACL2026/multimodal_vlm/lami_augmenting_large_language_models_via_late_multi-image_fusion.md)
- [\[CVPR 2026\] Disentangle-then-Align: Non-Iterative Hybrid Multimodal Image Registration via Cross-Scale Feature Disentanglement](disentangle-then-align_non-iterative_hybrid_multimodal_image_registration_via_cr.md)
- [\[CVPR 2026\] Wan-Weaver: Interleaved Multi-modal Generation via Decoupled Training](wan-weaver_interleaved_multi-modal_generation_via_decoupled_training.md)

</div>

<!-- RELATED:END -->

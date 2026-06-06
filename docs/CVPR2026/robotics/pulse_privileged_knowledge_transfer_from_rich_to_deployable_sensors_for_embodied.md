---
title: >-
  [Paper Note] PULSE: Privileged Knowledge Transfer from Rich to Deployable Sensors for Embodied Multi-Sensory Learning
description: >-
  [CVPR 2026 Workshop (Sense of Space)][Robotics][Privileged knowledge distillation] This paper proposes PULSE, a framework that performs knowledge distillation from a frozen privileged-sensor (e.g.…
tags:
  - "CVPR 2026 Workshop (Sense of Space)"
  - "Robotics"
  - "Privileged knowledge distillation"
  - "sensor asymmetry"
  - "wearable devices"
  - "stress detection"
  - "multimodal fusion"
date: 2026-05-08
content_hash: 8dea064a011e7f85
---

# PULSE: Privileged Knowledge Transfer from Rich to Deployable Sensors for Embodied Multi-Sensory Learning

**Conference**: CVPR 2026 Workshop (Sense of Space)  
**arXiv**: [2510.24058](https://arxiv.org/abs/2510.24058)  
**Code**: N/A  
**Area**: Robotics  
**Keywords**: Privileged knowledge distillation, sensor asymmetry, wearable devices, stress detection, multimodal fusion

## TL;DR
This paper proposes PULSE, a framework that performs knowledge distillation from a frozen privileged-sensor (e.g., EDA) teacher to a student model relying solely on cheap, deployable sensors (e.g., ECG, BVP, accelerometer). PULSE introduces shared-private embedding decomposition and a reconstruction-based collapse-prevention mechanism, achieving 0.994 AUROC for stress detection without EDA at inference time—surpassing even models that use all sensors.

## Background & Motivation
1. **Background**: Multi-sensor systems are widely used in wearable health monitoring, robotics, and related domains. Electrodermal activity (EDA) is the gold-standard physiological indicator of acute stress, as it directly reflects sympathetic nervous system activation.
2. **Limitations of Prior Work**: EDA sensors require Ag/AgCl electrodes and a constant-current source, are highly susceptible to motion artifacts, and are costly and fragile. Most commercial wearables only provide ECG/PPG, accelerometers, and temperature sensors, making EDA infeasible to deploy.
3. **Key Challenge**: EDA data is available during training (collected in laboratory settings) but unavailable at deployment. Discarding EDA outright wastes a valuable supervisory signal, while symmetric alignment methods fail to exploit the informational asymmetry of EDA.
4. **Goal**: Leverage privileged EDA information during training to enhance the representation learning of cheap sensors, while requiring no EDA whatsoever at inference time.
5. **Key Insight**: The work builds on the LUPI (Learning Using Privileged Information) paradigm, with a key innovation: rather than naively aligning all modalities, the student representation is decomposed into a *shared* subspace (alignable, used for knowledge transfer) and a *private* subspace (preserving modality-specific structure).
6. **Core Idea**: A frozen privileged-sensor teacher + shared-private embedding decomposition + multi-layer hidden-state distillation + reconstruction-based collapse prevention enable effective knowledge transfer from high-end to low-cost sensors.

## Method

### Overall Architecture
Training follows a three-stage pipeline: (1) **Self-supervised pretraining**—per-modality Masked Autoencoder with cross-modal shared embedding alignment; (2) **Privileged knowledge transfer**—the EDA teacher is frozen, and its hidden states and pooled embeddings are distilled into the shared subspace of the student; (3) **Supervised fine-tuning**—classification using only the fused representations from cheap sensors. EDA is not required at inference time.

### Key Designs

1. **Shared-Private Embedding Decomposition**:

    - **Function**: Separates alignable modality-invariant information from modality-specific information that must be preserved.
    - **Mechanism**: Each student encoder's output is split via a random binary mask into a shared embedding (aligned across modalities to receive privileged knowledge) and a private embedding (retaining modality-specific structure such as QRS morphology in ECG, used for reconstruction). The teacher encoder does not undergo shared/private decomposition.
    - **Design Motivation**: Naively aligning all student representations to the teacher over-constrains the student—each modality has unique signal morphology (QRS complexes in ECG, pulse waveforms in BVP) that neither should nor can fully match EDA. Ablations confirm that eliminating private capacity (ratio=0) causes performance to collapse to 0.959 AUROC, while eliminating shared capacity (ratio=1) similarly yields only 0.945.

2. **Multi-Layer Hidden-State Distillation**:

    - **Function**: Transfers knowledge at every layer of the encoder.
    - **Mechanism**: At each matched layer $\ell$, the fused shared embeddings of all student modalities are aligned to the teacher's corresponding layer representation via a cosine similarity loss: $\mathcal{L}_{\text{hid}} = \frac{1}{|\mathcal{L}|} \sum_\ell (1 - \cos\langle \text{Fuse}(\{S_m^\ell\}), T^\ell\rangle)$. An additional alignment loss $\mathcal{L}_{\text{emb}}$ is applied to the final pooled embeddings.
    - **Design Motivation**: Aligning only the final layer (final-only) degrades performance to 0.953 AUROC—below the no-teacher baseline of 0.963—demonstrating that intermediate layers carry complementary structural information. Full-layer alignment achieves the best 0.994.

3. **Reconstruction as Collapse Prevention (Key Finding)**:

    - **Function**: Prevents representation collapse during knowledge distillation.
    - **Mechanism**: A MAE reconstruction loss is added alongside the distillation loss. Each student decoder reconstructs masked signal patches from the concatenated shared and private embeddings.
    - **Design Motivation**: This is the most important empirical finding of the paper. Without the reconstruction loss, KD causes the shared embeddings to collapse to constant vectors: the mean pairwise cosine similarity across all modalities approaches $\approx 1.0$, and feature variance approaches zero ($2.57 \times 10^{-5}$). With reconstruction, cosine similarity drops to 0.027–0.137 and variance recovers by three orders of magnitude. The reconstruction loss acts as an information-preserving regularizer.

4. **Frozen Teacher Design**:

    - **Function**: Provides a stable optimization target and prevents teacher-student co-adaptation.
    - **Mechanism**: The EDA teacher is frozen after self-supervised pretraining; only student parameters are updated during distillation.
    - **Design Motivation**: Compared to the all-sensor baseline—which jointly optimizes all encoders including EDA—the frozen teacher prevents overfitting to subject-specific EDA artifacts across the 15-subject LOSO evaluation. The AUROC standard deviation is halved from 0.133 to 0.060.

### Loss & Training
- **Pretraining**: $\mathcal{L}_{\text{pre}} = \lambda_{\text{align}} \mathcal{L}_{\text{align}} + \lambda_{\text{rec}} \mathcal{L}_{\text{rec}}$, with hinge loss alignment (margin $\alpha=0.2$), trained for 300 epochs.
- **Knowledge Transfer**: $\mathcal{L} = \lambda_{\text{hid}} \mathcal{L}_{\text{hid}} + \lambda_{\text{emb}} \mathcal{L}_{\text{emb}} + \lambda_{\text{rec}} \mathcal{L}_{\text{rec}}$, with default $\lambda_{\text{hid}}=\lambda_{\text{emb}}=1$, $\lambda_{\text{rec}}=0.1$, trained for 100 epochs.
- **Fine-tuning**: Encoders are frozen; only a 2-layer MLP (hidden=4) classifier is trained for 300 epochs with cosine learning rate scheduling.

## Key Experimental Results

### Main Results (WESAD Binary Stress Detection, LOSO)

| Model | Inference Input | AUROC | AUPRC | Accuracy |
|-------|----------------|-------|-------|----------|
| A: No-teacher baseline | Cheap sensors | 0.963±0.050 | 0.937±0.101 | 91.64% |
| B: Symmetric alignment | Cheap sensors | 0.972±0.031 | 0.944±0.061 | 88.83% |
| **C: PULSE** | **Cheap sensors** | **0.994±0.011** | **0.988±0.022** | **96.08%** |
| D: All sensors | Cheap + EDA | 0.983±0.028 | 0.963±0.048 | 90.74% |
| E: EDA teacher only | EDA | 0.962±0.067 | 0.924±0.122 | 87.20% |

### Ablation Study

| Configuration | AUROC | AUPRC | Note |
|---------------|-------|-------|------|
| All-layer distillation (default) | 0.994 | 0.988 | Best |
| Layers 3/5/7 only | 0.989 | 0.977 | Partial layer information lost |
| Final layer only | 0.953 | 0.922 | Below no-teacher baseline |
| Private ratio=0 (all shared) | 0.959 | 0.927 | Modality-specific information lost |
| Private ratio=0.5 (default) | 0.989 | 0.977 | Optimal balance |
| Private ratio=1 (no shared) | 0.945 | 0.906 | No knowledge transfer channel |

### Key Findings
- **PULSE outperforms the all-sensor model**: PULSE without EDA at inference (0.994) surpasses the all-sensor model that retains EDA (0.983). The frozen teacher prevents overfitting to subject-specific EDA artifacts, functioning as an implicit regularizer.
- **Reconstruction-based collapse prevention is essential**: Without reconstruction, shared embeddings collapse (cosine $\approx 1.0$); with reconstruction, feature variance recovers by three orders of magnitude.
- **Advantage is more pronounced in three-class classification**: On the three-class task (baseline/stress/amusement), PULSE achieves 0.956 AUROC vs. 0.812 for the all-sensor model—a substantially larger gap than in binary classification.
- **Cross-dataset generalization**: PULSE achieves 0.965 AUROC on PhysioNet STRESS (36 subjects), validating its generalizability.

## Highlights & Insights
- **Frozen teacher as a data-dependent regularizer**: This is a profound insight. PULSE does not succeed merely by having access to more information—it succeeds because freezing the teacher prevents overfitting. In small-sample cross-subject evaluation, the regularization benefit can outweigh the informational value of an additional modality. This finding has broad implications for multi-modal learning under limited data.
- **Reconstruction-based collapse prevention is generalizable**: Representation collapse is a potential failure mode in any multi-sensor distillation scenario; reconstruction loss as an information-preserving regularizer constitutes a general-purpose solution.
- **Necessity of shared-private decomposition**: Both extremes (fully shared or fully private) significantly underperform the balanced configuration, demonstrating that the question of *what to align and what to preserve* is a fundamental design problem in knowledge distillation.

## Limitations & Future Work
- The experimental scale is limited (WESAD contains only 15 subjects); LOSO evaluation mitigates but does not eliminate statistical uncertainty.
- Validation is confined to stress detection; the broader applicability discussed by the authors (e.g., tactile-to-IMU transfer, XR sensor transfer) is not empirically demonstrated.
- Direct verification of whether the shared embeddings genuinely encode EDA privileged knowledge (e.g., via probing classifiers or t-SNE visualization) is left for future work.
- The classification head is minimal (2-layer MLP, hidden=4), which may limit performance ceiling on more complex tasks.

## Related Work & Insights
- **vs. Symmetric alignment methods (PhysioOmni, ADAPT)**: These methods symmetrically align all modalities without exploiting informational asymmetry. PULSE explicitly distinguishes the privileged teacher from the student and provides a stable target via teacher freezing.
- **vs. EmotionKD**: Also performs cross-biosignal distillation, but lacks shared-private decomposition and multi-layer distillation, and updates both encoders during training.
- **vs. Abbaspourazad et al.**: Distills PPG to accelerometer at population scale, but similarly lacks shared-private decomposition.
- **Implications for robotics**: The PULSE framework is directly applicable to knowledge transfer from tactile sensors to IMUs, or from high-fidelity to consumer-grade XR sensors.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of a frozen privileged teacher, shared-private decomposition, and reconstruction-based collapse prevention is original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ LOSO evaluation, two datasets, detailed ablations, and collapse visualization evidence.
- **Writing Quality**: ⭐⭐⭐⭐ The framework is clearly presented; the discussion on generalization to broader scenarios is insightful.
- **Value**: ⭐⭐⭐ A workshop paper with limited experimental scale, but the core design principles have broad applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] D2E: Scaling Vision-Action Pretraining on Desktop Data for Transfer to Embodied AI](../../ICLR2026/robotics/d2e_scaling_vision-action_pretraining_on_desktop_data_for_transfer_to_embodied_a.md)
- [\[CVPR 2026\] ForceVLA2: Unleashing Hybrid Force-Position Control with Force Awareness for Contact-Rich Manipulation](forcevla2_unleashing_hybrid_force-position_control_with_force_awareness_for_cont.md)
- [\[ICLR 2026\] Experience-based Knowledge Correction for Robust Planning in Minecraft](../../ICLR2026/robotics/experience-based_knowledge_correction_for_robust_planning_in_minecraft.md)
- [\[ICLR 2026\] Domain Expansion: A Latent Space Construction Framework for Multi-Task Learning](../../ICLR2026/robotics/domain_expansion_a_latent_space_construction_framework_for_multi-task_learning.md)
- [\[ICCV 2025\] TesserAct: Learning 4D Embodied World Models](../../ICCV2025/robotics/learning_4d_embodied_world_models.md)

</div>

<!-- RELATED:END -->

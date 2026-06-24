---
title: >-
  [Paper Note] Homogeneous Dynamics Space for Heterogeneous Humans
description: >-
  [CVPR 2025][Human Understanding][Human Dynamics] This paper proposes HDyS (Homogeneous Dynamics Space). By aggregating heterogeneous human motion data from biomechanics and reinforcement learning, it trains a homogeneous latent space to unify different kinematic and dynamic representations, achieving high-quality bidirectional mapping from kinematics to dynamics and demonstrating effectiveness on downstream tasks such as inverse dynamics estimation and ground reaction force p…
tags:
  - "CVPR 2025"
  - "Human Understanding"
  - "Human Dynamics"
  - "Heterogeneous Representation Unification"
  - "Inverse Dynamics"
  - "Forward Dynamics"
  - "Cross-Domain Alignment"
date: 2026-05-08
content_hash: 0284d26f9cb6733c
---

# Homogeneous Dynamics Space for Heterogeneous Humans

**Conference**: CVPR 2025  
**arXiv**: [2412.06146](https://arxiv.org/abs/2412.06146)  
**Code**: [https://foruck.github.io/HDyS](https://foruck.github.io/HDyS)  
**Area**: Human Understanding  
**Keywords**: Human Dynamics, Heterogeneous Representation Unification, Inverse Dynamics, Forward Dynamics, Cross-Domain Alignment

## TL;DR

This paper proposes HDyS (Homogeneous Dynamics Space). By aggregating heterogeneous human motion data from biomechanics and reinforcement learning, it trains a homogeneous latent space to unify different kinematic and dynamic representations, achieving high-quality bidirectional mapping from kinematics to dynamics and demonstrating effectiveness on downstream tasks such as inverse dynamics estimation and ground reaction force prediction.

## Background & Motivation

1. **Background**: Computer vision has made huge progress in human kinematics (human reconstruction, action recognition, motion generation, etc.), but human dynamics—the generation mechanisms of motion (joint torques, muscle activations, electromyography/sEMG signals, etc.)—remain understudied.

2. **Limitations of Prior Work**: Understanding human dynamics faces a double heterogeneity problem. (1) Representation heterogeneity: Kinematics includes different representations like markers, skeletal keypoints, joint angles, and SMPL parameters, while dynamics features hierarchical representations such as joint torques, muscle actions, and surface electromyography (sEMG), making conversion between different representations difficult. (2) Domain heterogeneity: Biomechanical data comes from optimization solvers (high quality but simple actions, restricted to laboratory environments), while reinforcement learning data comes from physical simulations (rich movements but plagued by the sim-to-real gap), with completely different kinematic templates and data formats.

3. **Key Challenge**: Existing data sources have complementary advantages and disadvantages but cannot interoperate—different representation formats hinder data aggregation, and significant differences between domains lead to poor performance when directly transferring models.

4. **Goal**: To discover homogeneity among heterogeneous human motion representations and datasets, and to build a unified latent space that realizes bidirectional mappings between kinematics and dynamics.

5. **Key Insight**: Although representing itself through diverse formats on the surface, they all describe the exact same underlying reality—human motion. Kinematic representations in Cartesian coordinates (e.g., markers, keypoints) differ minimally across systems, and different hierarchical dynamics representations (torques, muscles, sEMG) share similar muscle coordination patterns despite lacking direct conversion formulas.

6. **Core Idea**: Aggregating heterogeneous data + Inverse/Forward Dynamics Autoencoders + Contrastive Alignment Loss = Homogeneous Latent Space.

## Method

### Overall Architecture

HDyS is an aggregated architecture consisting of multiple autoencoders corresponding to both inverse dynamics and forward dynamics directions. The input can be any one or more of four kinematic representations (markers, skeletal keypoints, Rajagopal joint angles, SMPL parameters), and the output consists of multiple dynamics representations (joint torques, muscle activations, sEMG). All encoder outputs share the same 128-dimensional latent space, which is trained via reconstruction losses and a contrastive alignment loss.

### Key Designs

1. **Inverse Dynamics Autoencoder (IDAE)**:

    - **Function**: Encodes kinematics into the latent space, and subsequently decodes dynamics from the latent space.
    - **Mechanism**: A 3-layer Transformer encoder (without positional encodings, to accommodate varying numbers of markers/points) is employed for markers and skeletal keypoints, while a 3-layer MLP encoder is used for joint angles and SMPL parameters, mapping both to a unified 128-dimensional latent vector. The decoder is a shared Transformer that models temporal context across consecutive frames, followed by independent MLP regression heads to decode different types of dynamics outputs (Rajagopal torques, SMPL torques, muscle activations, sEMG).
    - **Design Motivation**: The Transformer encoder can handle arbitrary numbers of markers/points, enabling unified processing across different datasets. The shared Transformer decoder pushes latent spaces from different kinematic inputs to become homogeneous, while the independent MLP heads preserve the specificity of individual dynamics representations.

2. **Forward Dynamics Autoencoder (FDAE)**:

    - **Function**: Encodes dynamics + kinematics (without acceleration) into the latent space, and then decodes kinematic acceleration.
    - **Mechanism**: Similar to IDAE, encoders are used to separately encode kinematics (excluding the acceleration components) and dynamics. The kinematic and dynamic latent vectors are concatenated and fused through a shared MLP composer, then mapped via independent MLP decoders to predict skeletal keypoint acceleration, SMPL acceleration, and joint angle acceleration. This corresponds to Newton's second law in physics: given current status and force, one can determine acceleration.
    - **Design Motivation**: The forward dynamics autoencoder forces the latent space to simultaneously preserve both kinematic and dynamic information, enhancing the physical consistency of the latent space. Ablation studies without FDAE demonstrate its positive contribution to various downstream tasks.

3. **Contrastive Alignment Loss**:

    - **Function**: Pulls latent vectors of different representations from the same frame closer, and pushes those from different frames further apart.
    - **Mechanism**: Utilizing the InfoNCE loss, all latent vectors derived from the same frame within a batch (regardless of which kinematic/dynamic encoder they originated from) are treated as positive pairs, while latent vectors from different frames act as negative samples. The overall loss is formulated as $\mathcal{L} = \alpha_1 L_{recon} + \alpha_2 L_{align}$ where the reconstruction loss is calculated using L1 loss.
    - **Design Motivation**: Relying solely on reconstruction losses does not guarantee that different encoders' latent spaces align (they might learn disjoint representations). The contrastive alignment loss is the key to achieving "homogeneity"—ensuring that regardless of the input representation, the same motion frame maps to nearby locations in the latent space.

### Loss & Training

The total loss is $\mathcal{L} = 0.01 \cdot L_{recon} + 0.05 \cdot L_{align}$. The optimization utilizes AdamW with a learning rate of 1e-3, a batch size of 9600 frames, and is trained for 1000 epochs. To mitigate the impact of varying dataset scales, a balanced sampling strategy is implemented: 3000 sequences are randomly sampled from each dataset per epoch.

## Key Experimental Results

### Main Results

**Inverse Dynamics Performance (Table 1)**:

| Dataset | Metric | HDyS (avg/best) | Single Dataset Training | Prev. SOTA |
|--------|------|-----------------|-------------|----------|
| ImDy | mPJE↓ | 0.5765/0.4674 | 0.6854/0.5403 | 0.6300 |
| AddBiomechanics | mPJE↓ | 0.1189/0.1243 | 0.1695/0.1691 | 0.1626 |
| MinT | RMSE↓ | 0.0614/0.0615 | 0.0637/0.0640 | - |
| MiA | RMSE↓ | 11.8/11.6 | 13.6/13.5 | 13.3 |

### Ablation Study

| Configuration | ImDy mPJE↓ | AddBio mPJE↓ | MiA RMSE↓ |
|------|-----------|-------------|-----------|
| Full HDyS | 0.5765 | 0.1189 | 11.8 |
| w/o Alignment Loss | 0.6575 | 0.1270 | 13.7 |
| w/o FDAE | 0.5776 | 0.1198 | 13.6 |
| w/o AMASS | 0.5797 | 0.1217 | 14.7 |
| 32-dim Latent Space | 0.7390 | 0.1401 | 16.7 |

**Scale vs. Heterogeneity Decomposition (Table 2)**:

| Configuration | AddBio mPJE↓ | MiA RMSE↓ |
|------|-------------|-----------|
| Single Dataset-50% | 0.1707 | 16.2 |
| 50% Target + 50% Heterogeneous | 0.1284 | 14.5 |
| Single Dataset-100% | 0.1695 | 13.5 |

### Key Findings

- Aggregating heterogeneous datasets consistently outperforms training on a single dataset, validating that homogeneous dynamics knowledge indeed exists within heterogeneous data.
- The alignment loss is the most critical component: without it, mPJE on ImDy increases from 0.5765 to 0.6575, proving the importance of contrastive learning for the homogeneous space.
- Datasets with similar dynamic representations are more complementary—muscle-related datasets (MiA and MinT) mutually benefit each other more, while torque-related datasets (AddBiomechanics and ImDy) show greater mutual benefits.
- Training with "50% Target + 50% Heterogeneous" data can even outperform single-dataset training with 100% target data, demonstrating that the diversity provided by heterogeneity may yield higher value than simply increasing the scale of homogeneous domain data.
- Although AMASS contains no dynamics labels, its high-quality and diverse kinematic data still positively contributes to inverse dynamics estimation.

## Highlights & Insights

- **The Philosophy of "Mining Homogeneity from Heterogeneity"**: Although data formats from different communities differ, they describe the same physical reality. This methodology can be transferred to other multi-source heterogeneous data fusion scenarios, such as the unified representation of multimodal medical data (CT/MRI/Ultrasound).
- **Bidirectional Inverse-Forward Dynamics Autoencoder**: Rather than only mapping motion to forces, the framework also backs out acceleration from force, establishing a physically consistent closed loop. This bidirectional training strategy infuses the latent space with greater physical meaning.
- **Exquisite Experimental Design on Scale vs. Heterogeneity (Table 2)**: Through a 50/50 split experiment, the authors meticulously disentangle the respective contributions of data scale and data heterogeneity—a highly commendable experimental paradigm.

## Limitations & Future Work

- The authors acknowledge the persistence of the sim-to-real gap—domain discrepancies still exist between RL simulation data (ImDy) and real biomechanical data.
- Currently, only the lower body is processed (the Rajagopal model uses only 23 joints); whole-body dynamics modeling remains a challenge.
- sEMG data is highly noisy with significant individual variation, and the model's generalization capabilities to unseen subjects have not been fully validated.
- The latent space dimension is fixed to 128; adaptive dimensions or hierarchical latent spaces have not been explored.
- External forces (e.g., ground reaction forces) are not explicitly treated as inputs, which potentially limits applicability in scenarios where external force information is vital.

## Related Work & Insights

- **vs. ImDyS**: ImDyS only leverages RL simulation data to learn inverse dynamics, which is heavily constrained by the sim-to-real gap. HDyS mitigates this limitation by aggregating actual biomechanical data, reducing mPJE from 0.63 to 0.47.
- **vs. AddBiomechanics**: AddBiomechanics aggregates a massive volume of biomechanical datasets but is limited to simple movements like gaits. HDyS expands action coverage by incorporating RL data.
- **vs. MiA**: MiA directly predicts sEMG from video/motion capture, but faces constraints in data scale and action diversity. HDyS leverages cross-domain data to improve generalization, reducing RMSE from 13.3 to 11.6.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Famously the first system to systematically analyze and unify multi-source heterogeneous human motion data. The design of the inverse-forward autoencoder along with contrastive alignment is novel and physically motivated.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers all dynamic levels across four datasets, supported by meticulous ablation studies (components, data sources, dimensions, scale vs. heterogeneity decomposition).
- **Writing Quality**: ⭐⭐⭐⭐ The analysis of the problem is clear, though it involves a high-entry barrier due to the specialized biomechanics background knowledge.
- **Value**: ⭐⭐⭐⭐ A fundamental contribution to the human dynamics field, though the application scenarios remain relatively specialized.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] ESC: Erasing Space Concept for Knowledge Deletion](esc_erasing_space_concept_for_knowledge_deletion.md)
- [\[NeurIPS 2025\] BEDLAM2.0: Synthetic Humans and Cameras in Motion](../../NeurIPS2025/human_understanding/bedlam20_synthetic_humans_and_cameras_in_motion.md)
- [\[NeurIPS 2025\] HOI-Dyn: Learning Interaction Dynamics for Human-Object Motion Diffusion](../../NeurIPS2025/human_understanding/hoi-dyn_learning_interaction_dynamics_for_human-object_motion_diffusion.md)
- [\[ICCV 2025\] Weakly Supervised Visible-Infrared Person Re-Identification via Heterogeneous Expert Collaborative Consistency Learning](../../ICCV2025/human_understanding/weakly_supervised_visible-infrared_person_re-identification_via_heterogeneous_ex.md)
- [\[ECCV 2024\] ReLoo: Reconstructing Humans Dressed in Loose Garments from Monocular Video in the Wild](../../ECCV2024/human_understanding/reloo_reconstructing_humans_dressed_in_loose_garments_from_monocular_video_in_th.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Language-Grounded Decoupled Action Representation for Robotic Manipulation (LaDA)
description: >-
  [CVPR 2026][Robotics][action decoupling] LaDA is a framework that uses natural language as a semantic bridge to decouple continuous 7-DoF actions into three interpretable primitives — translation, rotation, and gripper — and employs soft-label contrastive learning to align cross-task action representations in a shared embedding space. With only 0.6B parameters, LaDA achieves a 93.6% success rate on LIBERO, outperforming all baselines with 1.3B–8.5B parameters.
tags:
  - CVPR 2026
  - Robotics
  - action decoupling
  - language semantic bridge
  - soft-label contrastive learning
  - VLA
  - cross-task generalization
date: 2026-05-08
content_hash: 02bc15844bcf17fd
---

# Language-Grounded Decoupled Action Representation for Robotic Manipulation (LaDA)

**Conference**: CVPR 2026
**arXiv**: [2603.12967](https://arxiv.org/abs/2603.12967)
**Code**: N/A
**Area**: Robotic Manipulation
**Keywords**: action decoupling, language semantic bridge, soft-label contrastive learning, VLA, cross-task generalization

## TL;DR

LaDA is a framework that uses natural language as a semantic bridge to decouple continuous 7-DoF actions into three interpretable primitives — translation, rotation, and gripper — and employs soft-label contrastive learning to align cross-task action representations in a shared embedding space. With only 0.6B parameters, LaDA achieves a 93.6% success rate on LIBERO, outperforming all baselines with 1.3B–8.5B parameters.

## Background & Motivation

**Background**: Vision-Language-Action (VLA) models have advanced robotic manipulation in recent years, yet the heterogeneity between high-level semantic understanding and low-level action control remains a fundamental challenge.

**Limitations of Prior Work**: Three paradigms each exhibit distinct shortcomings — (1) end-to-end VLAs (e.g., OpenVLA, RT-2) tightly couple perception and control, yielding uninterpretable actions that cannot reuse shared motion structures; (2) implicit action learning methods (e.g., LAPA, UniSkill) encode actions in compact latent spaces defined by observation differences, lacking explicit semantic labels and thus limiting cross-task transfer; (3) language-conditioned policies (e.g., CLIP-RT, PPL) incorporate language but rely on coarse-grained discrete primitives (e.g., "move forward," "close gripper"), omitting fine-grained motion parameters such as translation magnitude and rotation angle.

**Key Challenge**: Tasks such as "pouring water" and "placing a bottle" share substantial underlying motion primitives (reaching, grasping, rotating), yet existing models cannot exploit this shared structure, leading to redundant learning and poor cross-task generalization. The fundamental cause is the absence of a semantically grounded layer connecting symbolic intent with continuous execution.

**Goal**: To construct an action representation that is both semantically grounded and transferable across tasks, enabling the sharing and alignment of fine-grained motion semantics.

**Key Insight**: Natural language provides a universal interface connecting human intent, visual perception, and robot control — it is compositional and semantically regular, capable of encoding motion concepts and enabling comparison, transfer, and generalization within a unified space.

**Core Idea**: Language-anchored fine-grained action primitives serve as an intermediate layer between continuous control and high-level semantics; soft-label contrastive learning achieves semantic alignment of cross-task actions.

## Method

### Overall Architecture

Given visual observations $V_t$, language instructions $L_t$, and 7-DoF actions $\mathbf{a}_t$, the framework proceeds in four steps: (1) action decomposition — projecting $\mathbf{a}_t$ into three language-anchored primitives (Translation, Rotation, Gripper); (2) constructing a soft-label similarity matrix $S$ that encodes primitive-level semantic affinity; (3) dual-path soft-label contrastive learning (Action–Action alignment + Action–Primitive alignment) to train a unified embedding space; (4) adaptive loss weighting to balance contrastive and imitation losses. After pretraining, a lightweight MLP action head is appended and fine-tuned for continuous 7-DoF action prediction.

### Key Designs

1. **Language-Grounded Action Decomposition**:

    - *Function*: Defines a projection $\Pi: \mathbf{a}_t \mapsto \mathbf{p}_t$ that decomposes continuous 7-DoF actions into three interpretable primitive categories — translation: "Move [dist] meters along [dir]"; rotation: "Rotate [mag] degrees around [axis]"; gripper: "Open/Close."
    - *Mechanism*: Each primitive is discretized into language-aligned symbolic bins, converting continuous control trajectories into interpretable semantic categories. For example, translation direction is discretized into {forward/backward/left/right/up/down}, rotation axis into {x/y/z}, and gripper state into a binary {open/close}.
    - *Design Motivation*: Different tasks can share identical motion primitives (e.g., "rotate 90° around the z-axis" appears in multiple tasks). Explicit semantic labels allow these shared structures to be exploited by contrastive learning, rather than being buried in uninterpretable latent codes as in implicit action learning.

2. **Semantic-Guided Soft-Label Contrastive Learning**:

    - *Function*: Aligns multimodal representations in a unified embedding space according to primitive-level semantic affinity.
    - *Mechanism*: A soft similarity matrix is constructed as $S = \frac{w_t M_t + w_r M_r + w_g M_g}{w_t + w_r + w_g}$, where $M_t, M_r, M_g$ are binary matching matrices for the translation, rotation, and gripper dimensions, respectively. CLIP visual and text encoders extract embeddings, which are FiLM-conditioned and projected via MLP to obtain unified embeddings $A_i$. A dual-path soft-label InfoNCE loss is applied: (i) Action–Action path pulls semantically similar action embeddings closer, weighted by $S_{ij}$; (ii) Action–Primitive path anchors each action to the text encoding $P_j$ of its corresponding primitive description. The total loss is $\mathcal{L}_{CL} = \mathcal{L}_a + \lambda \mathcal{L}_m$.
    - *Design Motivation*: Unlike conventional binary positive/negative pairs, soft labels allow "partially similar" action pairs to receive graded similarity gradients (e.g., two actions sharing the same translation but differing in rotation still achieve partial matching), capturing finer-grained motion correspondences.

3. **Adaptive Loss Weighting**:

    - *Function*: Dynamically balances the imitation loss $\mathcal{L}_{IL}$ (predicting discretized primitive categories) and the contrastive loss $\mathcal{L}_{CL}$.
    - *Mechanism*: Moving-average-normalized weights are computed as $w_{IL} = \frac{\text{MA}(\mathcal{L}_{IL})}{\text{MA}(\mathcal{L}_{IL}) + \text{MA}(\mathcal{L}_{CL})}$, yielding a total loss of $\mathcal{L}_{total} = w_{CL} \mathcal{L}_{CL} + w_{IL} \mathcal{L}_{IL}$.
    - *Design Motivation*: The imitation loss provides coarse-grained behavioral supervision while the contrastive loss provides fine-grained semantic alignment; the two differ in convergence rate and granularity, and fixed weights easily cause one term to dominate. The design is inspired by curriculum learning.

### Loss & Training

- **Pretraining**: Trained on the OXE dataset (approximately 22.5 million frames, 22 robot embodiments) using $\mathcal{L}_{total}$; structured language descriptions are automatically generated for each continuous action as auxiliary supervision.
- **Fine-tuning**: A lightweight MLP action head is trained with an $\ell_1$ trajectory regression loss.
- **Inference**: No explicit primitive labels are required; continuous actions are predicted directly from $(V_t, L_t)$.

## Key Experimental Results

### Main Results

| Model | Parameters | LIBERO-Spatial | LIBERO-Object | LIBERO-Goal | LIBERO-Long | LIBERO-Avg |
|-------|-----------|---------------|---------------|-------------|-------------|------------|
| OpenVLA | 7.5B | 84.7% | 88.4% | 79.2% | 53.7% | 76.5% |
| FlowVLA | 8.5B | 93.2% | 95.0% | 91.6% | 72.6% | 88.1% |
| CLIP-RT | 1.3B | 95.2% | 99.2% | 94.2% | 83.8% | 93.1% |
| **LaDA** | **0.6B** | **95.2%** | **99.2%** | **93.6%** | **86.4%** | **93.6%** |

| Model | MimicGen Avg (9 tasks) | StackThree_D1 |
|-------|----------------------|---------------|
| OpenVLA | 38% | 20% |
| Phoenix | 58% | 20% |
| CLIP-RT* | 51% | 52% |
| **LaDA** | **67%** | **71%** |

### Ablation Study

| Configuration | Spatial | Object | Goal | Long | Avg |
|--------------|---------|--------|------|------|-----|
| w/o SCL (remove soft-label contrastive) | 79.2% | 82.8% | 76.6% | 63.4% | 75.5% |
| w/o AW (remove adaptive weighting) | 93.6% | 94.4% | 87.2% | 74.4% | 87.4% |
| **LaDA (full)** | **95.2%** | **99.2%** | **93.6%** | **86.4%** | **93.6%** |

### Key Findings

- Removing SCL causes a sharp 18.1-point drop in LIBERO average (93.6→75.5%), with LIBERO-Long falling from 86.4% to 63.4%, demonstrating that long-horizon sequences rely most heavily on cross-task semantic sharing.
- Removing adaptive weighting causes a 6.2-point average drop and a 12-point drop on Long, confirming that optimization balance is especially critical for long-horizon tasks.
- In the cross-task generalization setting, CLIP-RT* achieves 0% success while LaDA reaches 12.3%, demonstrating that language-anchored primitives enable motion semantic reuse for unseen instructions.
- On MimicGen, LaDA shows substantial gains under multi-task training (whereas CLIP-RT shows negligible gains), indicating that semantic structure effectively promotes cross-task sharing of motion patterns.

## Highlights & Insights

- The "language as semantic bridge" concept directly addresses a core VLA limitation — rather than performing end-to-end black-box mapping, it establishes an explicit semantic interface at the action level, making actions comparable and transferable. This approach is more elegant than both implicit action learning and coarse-grained language conditioning.
- Soft-label contrastive learning represents a methodological contribution — conventional contrastive learning uses binary positive/negative pairs, whereas LaDA employs a continuous affinity matrix for soft InfoNCE, allowing partially matching pairs (e.g., same translation but different rotation) to receive appropriately graded gradient signals. This idea is transferable to domains such as object detection and segmentation that require fine-grained semantic alignment.
- The parameter efficiency is remarkable: a 0.6B model outperforms 7B+ models, demonstrating that carefully designed structural inductive biases (action decoupling + contrastive alignment) can substantially reduce dependence on model scale.

## Limitations & Future Work

- The three primitive categories (translation/rotation/gripper) cover the 7-DoF of standard industrial manipulators but may be insufficient for dexterous hand manipulation (e.g., humanoid finger joints with 20+ DoF), necessitating primitive designs with more motion components.
- Language templates are hand-crafted (e.g., "Move X meters along Y"); automated primitive discovery could offer greater flexibility.
- Real-robot experiments cover only a single pick-and-place task, leaving validation in complex real-world scenarios insufficient.
- The soft similarity matrix weights $(w_t, w_r, w_g)$ are hyperparameters that may require re-tuning for different task domains.

## Related Work & Insights

- **vs. CLIP-RT**: Both employ language-conditioned control, but CLIP-RT models actions as discrete language token classification without continuous semantic alignment of motion parameters. LaDA uses soft-label contrastive learning for continuous affinity matching, achieving comparable or slightly superior performance at half the parameter count.
- **vs. LAPA**: Implicit action learning encodes actions as uninterpretable latent codes, requiring implicit learning for cross-task transfer. LaDA renders the action space interpretable and directly transferable through semantic alignment.
- **vs. Phoenix**: Phoenix relies on motion-level self-reflective correction, achieving 58% on MimicGen. LaDA reaches 67% without self-correction, suggesting that better representations are more effective than more complex inference strategies.

## Rating

- Novelty: ⭐⭐⭐⭐ Language-anchored action decomposition combined with soft-label contrastive learning constitutes a novel methodological combination.
- Experimental Thoroughness: ⭐⭐⭐⭐ Dual benchmarks (LIBERO/MimicGen) with ablation, generalization, and real-robot experiments; real-robot tasks are relatively simple.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear, and the three-paradigm comparison is convincing.
- Value: ⭐⭐⭐⭐ Strong practical significance of a 0.6B model surpassing 7B+ counterparts; soft-label contrastive learning is broadly transferable.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Language-Grounded Decoupled Action Representation for Robotic Manipulation](language-grounded_decoupled_action_representation_for_robotic_manipulation.md)
- [\[CVPR 2026\] QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models](quantvla_scale-calibrated_post-training_quantization_for_vision-language-action_.md)
- [\[CVPR 2026\] GeCo-SRT: Geometry-aware Continual Adaptation for Robotic Cross-Task Sim-to-Real Transfer](geco-srt_geometry-aware_continual_adaptation_for_robotic_cross-task_sim-to-real_.md)
- [\[CVPR 2026\] SaPaVe: Towards Active Perception and Manipulation in Vision-Language-Action Models for Robotics](sapave_active_perception_manipulation_vla_roboti.md)
- [\[CVPR 2026\] Diagnose, Correct, and Learn from Manipulation Failures via Visual Symbols](diagnose_correct_and_learn_from_manipulation_failures.md)

<!-- RELATED:END -->

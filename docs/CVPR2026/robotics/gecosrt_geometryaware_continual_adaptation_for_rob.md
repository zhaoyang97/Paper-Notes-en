---
title: >-
  [Paper Note] GeCo-SRT: Geometry-aware Continual Adaptation for Robotic Cross-Task Sim-to-Real Transfer
description: >-
  [CVPR 2026][Robotics][Sim-to-Real Transfer] GeCo-SRT proposes a continual cross-task Sim-to-Real transfer paradigm that exploits the domain-invariance and task-invariance of local geometric features. Through a geometry-a…
tags:
  - "CVPR 2026"
  - "Robotics"
  - "Sim-to-Real Transfer"
  - "Continual Learning"
  - "Geometry-aware MoE"
  - "Point Cloud Representation"
  - "Experience Replay"
date: 2026-05-08
content_hash: e3a6f1291f399db5
---

# GeCo-SRT: Geometry-aware Continual Adaptation for Robotic Cross-Task Sim-to-Real Transfer

**Conference**: CVPR 2026
**arXiv**: [2602.20871](https://arxiv.org/abs/2602.20871)
**Code**: N/A
**Area**: Robotics / Embodied Intelligence
**Keywords**: Sim-to-Real Transfer, Continual Learning, Geometry-aware MoE, Point Cloud Representation, Experience Replay

## TL;DR

GeCo-SRT proposes the first continual cross-task Sim-to-Real transfer paradigm, exploiting the domain-invariance and task-invariance of local geometric features. Through a Geo-MoE module for reusable geometric knowledge extraction and Geo-PER for expert-level forgetting prevention, the method achieves an average success rate of 63.3% across four real-robot tasks (a 52% improvement over baselines) while requiring only 1/6 of the data to match baseline performance.

## Background & Motivation

**Background**: Sim-to-Real transfer is a core challenge in robot learning—training policies on low-cost simulation data and deploying them in real environments. Existing approaches include System Identification (modeling real physical parameters, which is labor-intensive and difficult to apply to complex dynamics), Domain Randomization (training across varied simulation conditions, requiring manual tuning with limited coverage), and data-driven methods such as Transic (residual behavior cloning from human correction trajectories).

**Limitations of Prior Work**: All existing methods treat each Sim-to-Real transfer as an isolated process—every new task requires collecting data and re-tuning from scratch, incurring high costs and completely discarding prior transfer experience. For instance, after completing the "pick cube" transfer, the "stack cube" transfer must start anew.

**Key Challenge**: The Sim-to-Real gap across different robotic manipulation tasks actually shares substantial structured knowledge (e.g., object geometry is highly consistent between simulation and reality), yet existing methods cannot accumulate and reuse this knowledge across tasks.

**Goal**: How can transferable knowledge be continually accumulated across multiple Sim-to-Real tasks, making each new task's transfer faster and better rather than starting from zero?

**Key Insight**: Local geometric features (e.g., surface normals, planarity, linearity) possess a critical dual invariance—domain-invariant (geometric structures are consistent between simulation and reality, unlike texture/material which differ greatly) and task-invariant (different manipulation tasks share basic geometric primitives such as planes, edges, and corners). This makes local geometric features ideal carriers for cross-domain, cross-task knowledge.

**Core Idea**: A geometry-aware MoE module routes local geometric knowledge to different experts for specialized learning, while an expert-utilization-driven prioritized experience replay protects each expert's specialized knowledge from being forgotten.

## Method

### Overall Architecture

GeCo-SRT adopts a Human-in-the-Loop Sim-to-Real pipeline in three stages: (1) train a base Diffusion Policy (point cloud encoder + diffusion policy head) in simulation using 2,000 expert trajectories; (2) deploy to the real environment, where a human operator intervenes via SpaceMouse to correct impending failures, collecting 60 human correction trajectories; (3) mix correction data with simulation data to train a shared perceptual residual module, Geo-MoE. Base policy parameters are frozen; only the Geo-MoE module is updated. This residual module is shared and continually updated across all tasks, forming a channel for knowledge accumulation.

### Key Designs

1. **Geometry-aware Mixture-of-Experts Module (Geo-MoE)**:

    - **Function**: Serves as a perceptual residual network to bridge the observation gap between simulation and the real environment.
    - **Mechanism**: Local point groups $g_i$ are sampled from the input point cloud via kNN; PCA extracts local geometric features (planarity, linearity, saliency); a lightweight gating network $G$ generates routing weights $w_i = \text{Softmax}(G(g_i))$, assigning point groups to $M$ parallel experts. Expert outputs are aggregated by weighted summation: $g'_i = \sum_{j=1}^{M} w_{i,j} \cdot \text{Expert}_j(g_i)$. The final residual vector $g'_{res}$ is concatenated with the frozen base encoder output and fed into the diffusion policy head.
    - **Design Motivation**: Local geometric features exhibit dual invariance—domain-invariant (geometric structures are consistent between simulation and reality) and task-invariant (different manipulation tasks share basic geometric elements such as edges, corners, and planes), enabling experts to learn geometric knowledge that is reusable across new tasks.

2. **Geometry Expert-guided Prioritized Experience Replay (Geo-PER)**:

    - **Function**: Protects the specialized knowledge of each expert during continual learning to prevent catastrophic forgetting.
    - **Mechanism**: Standard PER samples by task loss, neglecting experts that are idle on the current task, causing their specialized knowledge to be forgotten. Geo-PER shifts the priority from task loss to expert utilization: for each historical sample $i$, the sampling priority is $P_i \propto \sum_{j=1}^{M} w_{i,j} \cdot \frac{1}{u_j^{\text{new}} + \epsilon}$, where $u_j^{\text{new}}$ is the average utilization of expert $j$ on the current task and $w_{i,j}$ is the historical activation weight of that sample for expert $j$.
    - **Design Motivation**: If an expert is underutilized on the current task (low $u_j^{\text{new}}$), its reciprocal term becomes large, prompting Geo-PER to prioritize sampling historical buffer entries that strongly activate that expert, ensuring idle experts are continuously updated. This inverse-hedging strategy is tailored specifically for MoE architectures.

3. **Human-in-the-Loop Correction Pipeline**:

    - **Function**: Quantifies the Sim-to-Real gap as learnable human correction data.
    - **Mechanism**: At each timestep, if the operator anticipates a failure, they take over control ($a_t \leftarrow a_t^h$, $I_t^h = \text{true}$); otherwise the base policy action is used. Correction data and simulation data are merged into a mixed replay buffer for training Geo-MoE.
    - **Design Motivation**: Freezing the base policy and training only the shared Geo-MoE module provides a clear knowledge accumulation pathway without disrupting the original policy.

### Loss & Training

The base policy is trained in simulation with a standard L2 diffusion loss: $\mathcal{L}_{\text{diff}} = \mathbb{E}[\|\epsilon - \epsilon_\theta(a_t^k, k, o_t)\|^2]$.

During the Sim-to-Real phase, the base policy is frozen and only Geo-MoE is trained: $\mathcal{L}_{\text{total}} = \text{MSE}(\hat{a}, a) + \alpha \mathcal{L}_{\text{balance}}$, where $\mathcal{L}_{\text{balance}}$ is the standard MoE load-balancing loss to prevent gating collapse. Training hyperparameters: base policy lr = $3 \times 10^{-4}$, denoising steps 10, action chunk size 8; residual learning lr = $1 \times 10^{-3}$; Geo-PER priority parameter 0.6, stability constant $\epsilon = 10^{-6}$, EMA update coefficient 0.4. Only 60 correction trajectories per task are required, mixed with 10% historical task data.

## Key Experimental Results

### Main Results

Experimental setup: 4 real-robot manipulation tasks (Pick Cube → Stack Cube → Pick Banana → Plug Insert); hardware: Xarm5 + Robotiq2F140 gripper + dual RealSense D435 depth cameras; 30 trials per task averaged.

**Single-Task Sim-to-Real Transfer (Table 1)**:

| Method | Pick Cube | Stack Cube | Pick Banana | Plug Insert | Avg SR |
|------|-----------|------------|-------------|-------------|--------|
| Direct Deploy | 5.7% | 0% | 6.7% | 0% | 3.1% |
| Action Residual | 16.7% | 3.3% | 13.3% | 0% | 9.2% |
| Transic | 66.7% | 30.0% | 23.3% | 33.3% | 38.3% |
| **Geo-MoE** | **80.0%** | **43.3%** | **40.0%** | **36.7%** | **50.0%** |

**Continual Cross-Task Sim-to-Real Transfer over 4 Tasks (Table 2)**:

| Method | Avg SR ↑ | Avg N-NBT ↓ |
|------|----------|-------------|
| Direct Deploy | 3.1% | — |
| Naive Fine-tuning | 9.2% | 75.0% |
| Transic + PER | 40.0% | 55.0% |
| Geo-MoE + EWC | 38.3% | 49.5% |
| Geo-MoE + PER | 55.7% | 29.6% |
| **GeCo-SRT (Ours)** | **63.3%** | **26.5%** |

Detailed per-task SR: GeCo-SRT achieves 86.7% / 53.3% / 60.0% / 53.3% on Pick Cube / Stack Cube / Pick Banana / Plug Insert, respectively, outperforming all baselines across all dimensions.

### Ablation Study

**Core Component Ablation (Table 3)**:

| Obs Residual | MoE | Avg SR ↑ | Avg N-NBT ↓ |
|:---:|:---:|----------|-------------|
| ✗ | ✗ | 3.3% | 75.0% |
| ✗ | ✓ | 9.2% | 65.5% |
| ✓ | ✗ | 45.8% | 37.0% |
| ✓ | ✓ | **55.8%** | **29.6%** |

**Expert Count Sensitivity (Table 5)**:

| No. of Experts N | Avg SR ↑ | NBT ↓ |
|:---:|----------|-------|
| 2 | 60.0% | 40.0% |
| 3 | **66.7%** | **33.3%** |
| 8 | 65.0% | 30.0% |

**Cross-Task Transfer and Task Similarity (Table 4, 10 trajectories)**:

| Target Task | Source Task | SR |
|----------|--------|-----|
| Stack Cube | Pick Cube | 40.0% |
| Stack Cube | Pick Banana | 33.3% |
| Stack Cube | Plug Insert | 16.7% |
| Stack Cube | From scratch | 26.7% |
| Stack Cube | From scratch (60 traj) | 43.3% |
| Plug Insert | Pick Cube | 33.3% |
| Plug Insert | Stack Cube | 40.0% |
| Plug Insert | Pick Banana | 30.0% |
| Plug Insert | From scratch | 23.3% |
| Plug Insert | From scratch (60 traj) | 36.7% |

**Generalization to New Tasks (Table 6)**:

| Method | Faucet SR | Tidying SR |
|------|-----------|------------|
| Direct Deploy | 10.0% | 0% |
| Geo-MoE (zero-shot) | 53.3% | 30.0% |
| Geo-MoE (scratch) | 76.6% | 43.3% |
| Geo-MoE (continual) | **83.3%** | **56.7%** |

### Key Findings

- **Observation residual is the most critical component**: Adding the point cloud encoder as an observation residual raises SR from 3.3% to 45.8%, indicating that the visual-level domain gap is the primary bottleneck.
- **MoE requires a meaningful feature foundation**: Adding MoE alone without observation residual is nearly ineffective (9.2%), showing that expert routing requires stable geometric features to function.
- **Geo-PER outperforms standard PER**: 63.3% vs. 55.7%, confirming that expert-level priority outperforms task-level loss priority.
- **Task similarity affects transfer quality**: Pick Cube → Stack Cube yields positive transfer (40.0% SR with 10 trajectories), while Plug Insert → Stack Cube shows negative transfer (only 16.7%); Pick Banana → Plug Insert benefits from shared non-cuboid grasping geometry.
- **Strong data efficiency**: Continual learning with 20 trajectories (76.6%) approaches the performance of training from scratch with 60 trajectories, requiring only 1/6 of the data.
- **Interpretable MoE**: Visualizations show that experts spontaneously specialize in distinct geometric primitives (edges, corners, planes); routing collapse is the primary failure mode under out-of-distribution conditions.
- **50% historical buffer is sufficient**: N-NBT with 50% historical data (23.3%) is close to that with the full buffer (20.0%), keeping memory overhead manageable.

## Highlights & Insights

- **New paradigm**: This work is the first to extend Sim-to-Real transfer from isolated task adaptation to a continual cross-task knowledge accumulation paradigm—a novel and practically meaningful problem formulation.
- **Dual-invariance insight**: The observation that local geometric features simultaneously satisfy domain-invariance and task-invariance is a key insight, validated by both ablation studies and visualizations: experts demonstrably learn to distinguish edges, corners, and planes spontaneously.
- **Expert-level experience replay**: Shifting the PER priority from task-loss level to MoE expert utilization level is an elegant design tailored to MoE architectures, delivering a 7.6% SR improvement over standard PER in experiments.

## Limitations & Future Work

- **Primarily addresses observation gap**: The method focuses on the visual-level domain gap (geometric feature alignment) and has limited effectiveness against complex dynamics gaps (physical parameter differences, contact force modeling, etc.), a limitation the authors themselves acknowledge in the discussion.
- **Reliance on human correction**: Although only 60 trajectories per task are needed, human operators must still intervene in real time. The authors suggest replacing this with an MLLM-based agent for a model-in-the-loop approach, but this has not yet been implemented.
- **Limited task scale**: Validation is restricted to a sequence of 4+3 tasks; whether the approach remains effective for much longer sequences (e.g., tens of tasks), whether the number of experts needs to scale, and how buffer growth should be managed remain open questions.

## Related Work & Insights

- **vs. Transic**: Both use human correction trajectories for Sim-to-Real transfer, but Transic employs a single residual network without MoE or continual learning. Single-task: 38.3% (Transic) vs. 50.0% (Geo-MoE); the gap widens further in the continual learning setting (Transic+PER 40% vs. GeCo-SRT 63.3%).
- **vs. Domain Randomization**: Requires manual specification of randomization ranges and independent tuning per task; GeCo-SRT automatically learns cross-task geometric knowledge from data.
- **vs. EWC (regularization-based)**: Geo-MoE+EWC achieves only 38.3% SR / 49.5% N-NBT, far below GeCo-SRT's 63.3% / 26.5%, indicating that parameter-importance regularization is less suited to MoE architectures than expert-level experience replay.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Continual cross-task Sim-to-Real is an entirely new problem setting; the Geo-MoE + Geo-PER combination is original; the dual-invariance insight is clearly articulated and well-supported.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Four plus three real-robot tasks (not simulation), with comprehensive ablations, transfer analysis, data efficiency studies, expert count sensitivity, interpretability visualizations, new-task generalization, and RGB comparisons.
- **Writing Quality**: ⭐⭐⭐⭐ Problem-driven structure is clear; the two core questions build progressively; the method is presented in a well-organized manner.
- **Value**: ⭐⭐⭐⭐ Provides a new continual learning perspective on Sim-to-Real transfer; high data efficiency (1/6 data); strong practical value for real-robot scenarios with limited resources.

---
title: >-
  [Paper Notes] GeCo-SRT: Geometry-aware Continual Adaptation for Robotic Cross-Task Sim-to-Real Transfer
description: >-
  [CVPR 2026][Robotics][Sim-to-Real Transfer] GeCo-SRT proposes a continual cross-task Sim-to-Real transfer paradigm that exploits the domain-invariance and task-invariance of local geometric features. Through a geometry-aware MoE module for reusable knowledge extraction and expert-guided prioritized experience replay for forgetting prevention, the method improves average success rate by 52% over baselines across four manipulation tasks while requiring only 1/6 of the data.
tags:
  - CVPR 2026
  - Robotics
  - Sim-to-Real Transfer
  - Continual Learning
  - Geometry-aware MoE
  - Point Cloud Representation
  - Experience Replay
---

# GeCo-SRT: Geometry-aware Continual Adaptation for Robotic Cross-Task Sim-to-Real Transfer

**Conference**: CVPR 2026
**arXiv**: [2602.20871](https://arxiv.org/abs/2602.20871)
**Code**: N/A
**Area**: Robotics / Embodied Intelligence
**Keywords**: Sim-to-Real Transfer, Continual Learning, Geometry-aware MoE, Point Cloud Representation, Experience Replay

## TL;DR
GeCo-SRT proposes a continual cross-task Sim-to-Real transfer paradigm that exploits the domain-invariance and task-invariance of local geometric features. Through a geometry-aware MoE module for reusable geometric knowledge extraction and expert-guided prioritized experience replay for forgetting prevention, the method achieves a 52% improvement in average success rate over baselines across four manipulation tasks while requiring only 1/6 of the data.

## Background & Motivation
Conventional Sim-to-Real methods (system identification, domain randomization, data-driven transfer) treat each transfer as an independent process—every new task requires re-tuning and data collection from scratch, incurring high costs and discarding prior experience entirely. The root cause is that the Sim-to-Real gap across different tasks actually shares substantial structured cross-domain knowledge (e.g., geometry is consistent between simulation and reality), yet existing methods cannot accumulate or reuse this knowledge across tasks.

## Core Problem
How can **transferable knowledge be continually accumulated** across multiple Sim-to-Real tasks so that each new task's transfer is faster and better, rather than restarting from zero? What kind of knowledge carrier is simultaneously cross-domain and cross-task?

## Method

### Overall Architecture
A Human-in-the-Loop Sim-to-Real pipeline proceeds in three stages: a base diffusion policy (point cloud encoder + diffusion policy head) is first trained in simulation on 2,000 expert trajectories; upon real-world deployment, a human operator corrects impending failures via SpaceMouse (60 correction trajectories collected); correction data and simulation data are then mixed to train a shared perceptual residual module. The residual module is shared and continually updated across tasks, enabling knowledge accumulation.

### Key Designs
1. **Geometry-aware Mixture-of-Experts (Geo-MoE)**: Acts as the perceptual residual module. Local point groups are sampled from the input point cloud via kNN; PCA extracts local geometric features (planarity, linearity, saliency); these features drive a gating network to route point groups to different experts, each specializing in particular geometric knowledge (edges, corners, planes). The output residual vector is concatenated with frozen base encoder features and fed to the diffusion policy head. Core insight: local geometric features exhibit **dual invariance**—domain-invariant (geometric structures are consistent between simulation and reality) and task-invariant (different manipulation tasks share basic geometric elements), enabling reuse of learned expert knowledge on new tasks.

2. **Geometry Expert-guided Prioritized Experience Replay (Geo-PER)**: Standard PER samples by task loss, neglecting idle experts and causing forgetting. Geo-PER shifts priority to **expert utilization**: if an expert is underutilized on the current task (low $u_j^{\text{new}}$), historical samples that strongly activated that expert (high $w_{i,j}$) are prioritized: $P_i \propto \sum_{j=1}^{M} w_{i,j} \cdot \frac{1}{u_j^{\text{new}} + \epsilon}$. This inverse-hedging strategy ensures all experts are periodically refreshed and is tailored specifically to MoE architectures.

3. **Human-in-the-Loop Correction Pipeline**: Quantifies the Sim-to-Real gap as human correction trajectories—when the operator anticipates failure, they take over control. Correction and simulation data are mixed; only the shared Geo-MoE module is updated (base policy frozen), providing a clear knowledge accumulation pathway.

### Loss & Training
$\mathcal{L}_{\text{total}} = \text{MSE}(\hat{a}, a) + \alpha \mathcal{L}_{\text{balance}}$. The balance loss prevents gating collapse. Base policy training lr = $3 \times 10^{-4}$; residual learning lr = $1 \times 10^{-3}$; Geo-PER priority parameter 0.6; EMA update coefficient 0.4. Sixty correction trajectories per task are required.

## Key Experimental Results

| Setting | Metric | GeCo-SRT | Transic+PER | Geo-MoE+PER | Direct Deploy |
|--------|------|------|----------|------|------|
| Single-task transfer | Avg SR (%) | **50.0** | 38.3 | — | 3.1 |
| Continual 4-task transfer | Avg SR (%) | **63.3** | 40.0 | 55.7 | 3.1 |
| Continual 4-task transfer | Avg N-NBT (%) | **26.5** | 48.2 | 36.3 | — |
| Data efficiency | Data to match baseline | **1/6** | — | — | — |

### Ablation Study Highlights
- Observation residual (point cloud encoder) is the most critical component: adding it raises SR from 3.1% to 45.8%.
- Adding MoE alone without observation residual is nearly ineffective (geometric routing requires meaningful features as a prerequisite).
- Observation residual + MoE together achieve the best performance (55.8% SR).
- Geo-PER vs. standard PER: 63.3% vs. 55.7%, confirming expert-level priority outperforms task-level loss priority.
- Task similarity affects transfer: Pick Cube → Stack Cube yields positive transfer (40%), Plug Insert → Stack Cube yields negative transfer (16.7%).
- $N=3$ experts is optimal; $N=2$ and $N=8$ are also robust (60–65%).
- On new tasks (Faucet/Tidying): continual learning (83.3% / 56.7%) far outperforms zero-shot (53.3% / 30%) and training from scratch (76.6% / 43.3%).

## Highlights & Insights
- First work to extend Sim-to-Real transfer from isolated task adaptation to a continual cross-task knowledge accumulation paradigm.
- The "dual invariance" of local geometric features is a novel and experimentally validated insight—experts demonstrably specialize in edges, corners, and planes spontaneously.
- Shifting experience replay priority from task-loss level to MoE expert utilization level is an elegant, architecture-specific design that yields a uniquely effective anti-forgetting strategy.
- Strong data efficiency: 20 trajectories under continual learning approach the performance of 60-trajectory from-scratch training.
- MoE is interpretable: visualizations confirm expert specialization by geometric primitive type.

## Limitations & Future Work
- Primarily addresses the observation gap (visual level); limited effectiveness against complex dynamics gaps (physical level).
- Relies on Human-in-the-Loop correction data collection; although only 60 trajectories per task are needed, human participation is still required.
- Validated on only 4 tasks; scalability to much longer task sequences remains unexplored.
- Uses only point cloud input; RGB performance is notably lower (40% vs. 80%).

## Related Work & Insights
- **vs. Transic**: Both use human correction trajectories, but Transic employs a single residual network without MoE or continual learning. Single-task: 38.3% vs. 50%; the gap widens further in continual learning (Transic+PER 40% vs. GeCo-SRT 63.3%).
- **vs. Domain Randomization**: Requires manual randomization range specification and per-task independent tuning; GeCo-SRT automatically accumulates cross-task knowledge from data.
- **vs. LIBERO/LOTUS and similar continual learning methods**: Designed for purely imitative continual learning without addressing the Sim-to-Real gap; GeCo-SRT is the first to introduce continual learning into Sim-to-Real transfer.

## Inspiration and Connections
- The use of geometric features as cross-domain invariants complements prior work on 3D dynamic pretraining (e.g., AFRO)—AFRO learns dynamics while GeCo-SRT uses geometry for transfer.
- The expert-level prioritized experience replay design is generalizable to other MoE-based continual learning systems.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Continual cross-task Sim-to-Real is an entirely new problem setting; the Geo-MoE + Geo-PER combination is original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Four plus three real-robot tasks, with comprehensive ablations, transfer analysis, data efficiency studies, and interpretability visualizations.
- **Writing Quality**: ⭐⭐⭐⭐ Problem-driven structure is clear; method is presented in a well-organized, hierarchical manner.
- **Value**: ⭐⭐⭐⭐ Provides a new continual learning perspective on Sim-to-Real transfer; high data efficiency; strong practical value for resource-constrained real-robot scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Learning to See and Act: Task-Aware Virtual View Exploration for Robotic Manipulation](learning_to_see_and_act_task-aware_virtual_view_exploration_for_robotic_manipula.md)
- [\[NeurIPS 2025\] Generalizable Domain Adaptation for Sim-and-Real Policy Co-Training](../../NeurIPS2025/robotics/generalizable_domain_adaptation_for_sim-and-real_policy_co-training.md)
- [\[CVPR 2026\] PALM: Progress-Aware Policy Learning via Affordance Reasoning for Long-Horizon Robotic Manipulation](palm_progress-aware_policy_learning_via_affordance_reasoning_for_long-horizon_ro.md)
- [\[CVPR 2026\] Action–Geometry Prediction with 3D Geometric Prior for Bimanual Manipulation](actiongeometry_prediction_with_3d_geometric_prior.md)
- [\[CVPR 2026\] RC-NF: Robot-Conditioned Normalizing Flow for Real-Time Anomaly Detection in Robotic Manipulation](rc-nf_robot-conditioned_normalizing_flow_for_real-time_anomaly_detection_in_robo.md)

</div>

<!-- RELATED:END -->

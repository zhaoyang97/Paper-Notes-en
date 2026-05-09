---
title: >-
  [Paper Note] EgoBridge: Domain Adaptation for Generalizable Imitation from Egocentric Human Data
description: >-
  [NeurIPS 2025][Reinforcement Learning][cross-embodiment] This paper proposes EgoBridge, a framework that uses Optimal Transport (OT) to align the joint distribution (features + actions) of human and robot data in a shared policy latent space, combined with Dynamic Time Warping (DTW) to construct pseudo-pairs, enabling cross-embodiment knowledge transfer from egocentric human data to robots, achieving up to 44% absolute improvement in success rate on real-world tasks.
tags:
  - NeurIPS 2025
  - Reinforcement Learning
  - cross-embodiment
  - domain adaptation
  - optimal transport
  - egocentric
  - imitation learning
date: 2026-05-08
content_hash: d7a1f24496c93f0b
---

# EgoBridge: Domain Adaptation for Generalizable Imitation from Egocentric Human Data

**Conference**: NeurIPS 2025
**arXiv**: [2509.19626](https://arxiv.org/abs/2509.19626)
**Code**: [ego-bridge.github.io](https://ego-bridge.github.io/)
**Area**: Reinforcement Learning
**Keywords**: cross-embodiment, domain adaptation, optimal transport, egocentric, imitation learning

## TL;DR
This paper proposes EgoBridge, a framework that uses Optimal Transport (OT) to align the joint distribution (features + actions) of human and robot data in a shared policy latent space, combined with Dynamic Time Warping (DTW) to construct pseudo-pairs, enabling cross-embodiment knowledge transfer from egocentric human data to robots, achieving up to 44% absolute improvement in success rate on real-world tasks.

## Background & Motivation

**Background**: Behavior Cloning (BC) combined with large-scale teleoperation data has achieved significant progress in robotic manipulation. However, teleoperation data collection is costly and difficult to scale to diverse scenarios. Wearable devices (e.g., AR glasses) enable low-cost collection of large quantities of egocentric human manipulation data, including observations and action information.

**Limitations of Prior Work**: Multiple domain gaps exist between human and robot data: (a) visual appearance discrepancy (human hands vs. robotic arms); (b) kinematic discrepancy (different behavioral distributions within the same action space); (c) sensor modality discrepancy (robots have wrist cameras; humans do not). Naive co-training cannot automatically produce effective transfer.

**Key Challenge**: Simple co-training assumes a shared latent space emerges naturally, but in practice human and robot latent features form disjoint clusters (latent covariate shift), $\mu_H \neq \mu_R$, preventing behaviors learned from human data from transferring to the robot.

**Goal**: (a) Explicitly align latent representations across human and robot domains; (b) preserve action-relevant information during alignment; (c) enable the robot to execute novel behaviors observed only in human data.

**Key Insight**: Cross-embodiment learning is formalized as a domain adaptation problem. Rather than global distribution alignment (e.g., adversarial training, MMD), the geometric structure of OT is leveraged to preserve local action correspondences.

**Core Idea**: DTW is used to guide the OT cost function, automatically discovering behaviorally similar human–robot pseudo-pairs during latent feature alignment, achieving action-aware joint distribution alignment.

## Method

### Overall Architecture
EgoBridge is a co-training framework. An encoder $f_\phi$ maps human/robot observations to a shared latent space $\mathcal{Z}$, and a Transformer decoder $\pi_\theta$ generates actions from latent features. The total loss is $\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{BC-cotrain}}(\phi,\theta) + \alpha\mathcal{L}_{\text{OT-joint}}$. The BC loss optimizes the entire network end-to-end, while the OT loss updates only the encoder.

### Key Designs

1. **Joint Distribution Optimal Transport (Joint OT)**:

    - Function: Aligns the joint distribution $P(f_\phi(O), A)$ of human and robot data in the latent space.
    - Mechanism: Given human samples $\{(o_i^H, a_i^H)\}$ and robot samples $\{(o_j^R, a_j^R)\}$, the Sinkhorn algorithm is applied to solve the entropy-regularized OT plan $T_\epsilon^*$:
    $\mathcal{L}_{\text{OT-joint}} = \sum_{i,j}(T_\epsilon^*)_{ij} \cdot \mathcal{C}((f_\phi(o_i^H), a_i^H), (f_\phi(o_j^R), a_j^R))$
    - Design Motivation: Unlike standard domain adaptation that aligns only the marginal distribution $P(f_\phi(O))$, joint alignment considers both features and actions simultaneously, preventing alignment from destroying action-relevant information. The resulting gradients encourage the encoder to map behaviorally similar cross-domain samples to nearby regions.
    - Distinction from Standard OT: Standard OT using Euclidean distance as the cost function may pair visually similar but behaviorally different samples; joint OT ensures paired samples are also behaviorally similar.

2. **DTW-Guided Cost Function Design**:

    - Function: Identifies behaviorally similar cross-domain pseudo-pairs via Dynamic Time Warping.
    - Mechanism: For each human–robot action trajectory pair in a mini-batch, the DTW distance is computed: $\text{DTW}(\mathbf{a}^H, \mathbf{a}^R) = \min_\pi \sum_{(i,j)\in\pi}\|a_i^H - a_j^R\|^2$. The best human match for each robot sample is identified as $i^*(j) = \arg\min_i A_{ij}$, and the cost matrix is modified as:
    $\tilde{C}_{ij} = \begin{cases} D_{ij} \cdot \lambda & \text{if } i = i^*(j) \\ D_{ij} & \text{otherwise} \end{cases}$
    where $D_{ij} = \|f_\phi(o_i^H) - f_\phi(o_j^R)\|^2$ and $\lambda \ll 1$ substantially reduces the transport cost for pseudo-pairs.
    - Design Motivation: (a) DTW naturally handles temporal alignment differences (human execution is typically 2–3× faster than teleoperation); (b) a "soft supervision" strategy is adopted—DTW distances are not used directly as a loss, but rather to identify pairs and reduce their OT transport cost; (c) this approach is more robust than MSE-based pairing, as confirmed by ablation studies.

3. **Shared Policy Architecture**:

    - Function: Uniformly processes both human and robot data sources.
    - Mechanism: The encoder $f_\phi$ comprises modality-specific stems (a shared vision stem for egocentric RGB and a separate stem for the robot wrist camera) and a shared Transformer encoder trunk. The decoder $\pi_\theta$ is a multi-layer Transformer decoder that generates actions via alternating self/cross-attention. $M$ learnable context tokens are used for computing the OT loss.
    - Design Motivation: The shared vision stem enforces visual alignment; the separate wrist camera stem accounts for the absence of this modality in human data; the DETR-style architecture supports flexible multi-modal input.

4. **Data Collection System**:

    - Human data: Meta Project Aria smart glasses, capturing egocentric RGB and bilateral hand SE(3) Cartesian poses.
    - Robot data: Eve robot equipped with identical Aria glasses to replicate the human hand-eye configuration, eliminating camera device discrepancy.
    - Unified action space: Dual-arm end-effector SE(3) poses with trajectory chunking.

### Loss & Training
$\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{BC-cotrain}}(\phi,\theta) + \alpha\mathcal{L}_{\text{OT-joint}}(\phi)$. The BC loss samples uniformly from human and robot data; the OT loss updates only the encoder parameters. Actions and proprioception are normalized with embodiment-specific Gaussian normalization.

## Key Experimental Results

### Main Results

| Method | Scoop Coffee In-Dist. | Scoop Obj. Gen. | Scoop Scene+Obj | Drawer (SR) | Drawer Beh. Gen. | Laundry (SR) |
|------|------|------|------|------|------|------|
| Robot-only BC | 33% | 40% | 7% | 9% | 0% | 28% |
| Co-train | 53% | 46% | 0% | 22% | 0% | 33% |
| EgoMimic | 60% | 53% | 0% | 14% | 0% | 33% |
| MimicPlay | 33% | 27% | 0% | 14% | 0% | 28% |
| ATM | 47% | 33% | 0% | 6% | 8% | 28% |
| **EgoBridge** | **67%** | **60%** | **27%** | **47%** | **33%** | **72%** |

### Ablation Study (Drawer Task)

| Configuration | Drawer SR | Beh. Gen. SR | Notes |
|------|----------|-------------|------|
| **EgoBridge (full)** | **47%** | **33%** | Full model |
| MSE replacing DTW | 14% | 17% | DTW pairing is critical; removing it causes a 33% drop |
| Standard OT (marginal alignment) | 33% | 17% | Joint OT outperforms marginal OT |
| Co-train (no alignment) | 22% | 0% | Without alignment, behavior generalization completely fails |

### Key Findings
- **DTW pairing contributes most**: Replacing DTW with MSE causes a dramatic performance drop (47%→14%), demonstrating that temporally robust kinematic pairing is central to the method.
- **Joint OT outperforms marginal OT**: Standard OT aligns only the marginal feature distribution while ignoring action correspondences, significantly degrading generalization.
- **Behavior generalization is a unique capability**: All other baselines completely fail (0%) on novel drawer positions covered only by human data, whereas EgoBridge achieves 33% success rate.
- **Latent space visualization**: t-SNE shows that EgoBridge achieves the highest overlap between human and robot features (smallest Wasserstein-2 distance), with KNN-matched pairs exhibiting the greatest semantic similarity.

## Highlights & Insights
- **OT + DTW combination**: OT provides a differentiable distribution alignment framework, while DTW provides a temporally robust behavioral similarity metric. Their combination makes alignment "behavior-aware"—an idea transferable to any cross-domain imitation learning scenario.
- **Learning new behaviors from humans**: This is the most valuable contribution. Most prior methods only improve the robustness of behaviors already present in robot data; EgoBridge enables the robot to execute novel behaviors never demonstrated via teleoperation, representing genuine "added value" from human data.
- **Soft supervision over hard constraints**: DTW does not serve directly as a loss function; instead, it guides alignment by reducing OT transport costs. This soft supervision strategy is more robust and avoids the compounding noise inherent in DTW itself.

## Limitations & Future Work
- **Single-task DTW**: DTW relies on action trajectory distances and may fail to distinguish similar local motions across different tasks in multi-task joint training. The authors suggest replacing it with language embedding distances from VLMs in future work.
- **Continued reliance on robot data**: The method does not achieve pure human-to-robot transfer and still requires seed data from the target domain. Whether this dependency can be further reduced or eliminated remains an open question.
- **SE(3) action space assumption**: Requiring humans and robots to share an end-effector pose space may limit applicability to tasks with substantially different kinematics, such as dexterous manipulation.
- **Evaluation scale**: The number of test rollouts per task is relatively small (15–48), limiting statistical confidence.

## Related Work & Insights
- **vs. EgoMimic**: EgoMimic bridges the domain gap through heuristics such as visual masking and data normalization, lacking explicit alignment. EgoBridge employs explicit OT-based alignment and outperforms EgoMimic across all tasks.
- **vs. MimicPlay**: MimicPlay adopts a hierarchical policy (high-level planning co-trained, low-level decoding fine-tuned) and aligns marginal distributions via KL divergence. EgoBridge's joint OT alignment better preserves action-relevant information.
- **vs. ATM**: ATM extracts motion information through point trajectory tracking before freezing the high-level policy to train a low-level one. This two-stage approach may lose fine-grained correspondences.

## Rating
- Novelty: ⭐⭐⭐⭐ The OT + DTW combination is novel in the cross-embodiment imitation learning context, and the joint distribution alignment formalization is clear and principled.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three real-world tasks (including bimanual manipulation), simulation ablations, and latent space visualization collectively provide thorough validation of the hypotheses.
- Writing Quality: ⭐⭐⭐⭐ Problem formalization is clear, method motivation is well-justified, and experimental presentation is well-organized.
- Value: ⭐⭐⭐⭐⭐ The behavior generalization capability represents a genuine breakthrough—human data no longer merely supplements existing robot behaviors but genuinely teaches the robot new skills.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Inner Speech as Behavior Guides: Steerable Imitation of Diverse Behaviors for Human-AI Coordination](inner_speech_as_behavior_guides_steerable_imitation_of_diverse_behaviors_for_hum.md)
- [\[NeurIPS 2025\] Continual Knowledge Adaptation for Reinforcement Learning](continual_knowledge_adaptation_for_reinforcement_learning.md)
- [\[NeurIPS 2025\] Quantifying Generalisation in Imitation Learning](quantifying_generalisation_in_imitation_learning.md)
- [\[NeurIPS 2025\] Provable Ordering and Continuity in Vision-Language Pretraining for Generalizable Embodied Agents](provable_ordering_and_continuity_in_vision-language_pretraining_for_generalizabl.md)
- [\[NeurIPS 2025\] Parameter Efficient Fine-tuning via Explained Variance Adaptation](parameter_efficient_fine-tuning_via_explained_variance_adaptation.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Learning Mutual View Information Graph for Adaptive Adversarial Collaborative Perception
description: >-
  [CVPR 2026][Autonomous Driving][Collaborative perception security] This paper proposes MVIG, an adversarial attack framework that unifies the vulnerability modeling of diverse defense-equipped collaborative perception sy…
tags:
  - "CVPR 2026"
  - "Autonomous Driving"
  - "Collaborative perception security"
  - "adversarial attacks"
  - "graph neural networks"
  - "temporal modeling"
date: 2026-05-08
content_hash: 673bb9a317f6ca65
---

# Learning Mutual View Information Graph for Adaptive Adversarial Collaborative Perception

**Conference**: CVPR 2026
**arXiv**: [2602.19596](https://arxiv.org/abs/2602.19596)  
**Code**: [Available](https://github.com/yihangtao/MVIG)  
**Area**: Autonomous Driving
**Keywords**: Collaborative perception security, adversarial attacks, graph neural networks, temporal modeling, autonomous driving

## TL;DR

This paper proposes MVIG, an adversarial attack framework that unifies the vulnerability modeling of diverse defense-equipped collaborative perception systems into a Mutual View Information Graph (MVIG). By combining temporal graph learning with entropy-aware vulnerability search, MVIG enables adaptive fabrication attacks that reduce the defense success rate by up to 62%.

## Background & Motivation

### 1. State of the Field
Collaborative Perception (CP) enables connected autonomous vehicles (CAVs) to share perceptual data (e.g., feature maps), extending the perceptual range of individual vehicles and addressing occlusion problems. Existing defense systems such as ROBOSAC, CP-Guard, MADE, and CAD employ consensus-based verification to counter attacks from malicious agents.

### 2. Limitations of Prior Work
Existing defense-equipped CP systems exhibit two critical weaknesses:
- **Lack of robustness against systematic spatiotemporal optimization attacks**: Current attack methods do not systematically determine when and where to launch fabrication attacks, yet adversaries can develop more targeted strategies against multi-vehicle consensus verification mechanisms.
- **Inadvertent leakage of vulnerability information during defense**: Collaborative data exchanged among CAVs (feature maps, occupancy maps, etc.) inherently embeds implicit confidence information about the surrounding environment, which attackers can exploit to identify regions of collective uncertainty.

### 3. Root Cause
The security defense of CP systems relies on multi-vehicle consensus verification; however, the data exchanged during this verification process simultaneously exposes the system's weak points, creating a paradox in which "defense entails disclosure."

### 4. Paper Goals
Design an adaptive adversarial attack framework capable of learning the vulnerabilities of diverse defense-equipped CP systems, automatically optimizing attack location, timing, and persistence, while remaining generalizable across different defense configurations.

### 5. Starting Point
Model the information leaked by different defense systems in a unified graph structure (MVIG), and leverage the spectral properties and temporal evolution of this graph to discover vulnerable regions.

### 6. Core Idea
Construct a Mutual View Information Graph (MVIG) with CAVs as nodes and mutual information as edge weights to encode consistency and disagreement across multi-vehicle perception. Temporal graph learning is used to predict future vulnerable regions, and entropy-aware search optimizes the attack strategy.

## Method

### Overall Architecture

The MVIG attack framework comprises four stages:
1. **MVIG Construction**: Extract occupancy information from each CAV using historical collaborative data and construct a weighted undirected graph.
2. **MVIGNet Prediction**: Predict the vulnerability score map for future frames through graph convolution combined with GRU-based temporal modeling.
3. **Entropy-Aware Vulnerability Search**: Optimize attack location, timing, and persistence.
4. **Alternating Optimization**: Decouple mask prediction from perturbation optimization; PGD is used to generate feature perturbations.

### Key Designs

#### Design 1: Mutual View Information Graph (MVIG) Construction
- **Function**: Model the perceptual states of multiple CAVs as a unified graph representation.
- **Mechanism**: Each CAV serves as a node whose feature vector consists of three components: (1) a basic occupancy distribution $\mathbf{h}_i^{\text{basic}}$ (frequency of free/occupied/unknown states); (2) pose information $\mathbf{h}_i^{\text{pos}}$; and (3) multi-scale spatial context features $\mathbf{h}_i^{\text{spatial}}$. Edge weights measure location-level consistency between two vehicles using mutual information: $\mathbf{W}_{ij} = \mathbb{E}_{(x,y)}\left[\sum_{a,b} p_{ij}(a,b) \log \frac{p_{ij}(a,b)}{p_i(a) p_j(b)}\right]$
- **Design Motivation**: High mutual information indicates perceptually consistent regions (difficult to attack); low mutual information indicates regions of perceptual conflict or uncertainty (attack windows). The unified graph representation allows the framework to adapt across different defense systems.

#### Design 2: MVIGNet Temporal Graph Learning
- **Function**: Predict future vulnerable regions from historical MVIG sequences.
- **Mechanism**: A three-stage architecture — (1) graph convolutional layers aggregate neighbor features via edge-weight-modulated message passing; (2) a GRU captures temporal patterns, $\mathbf{z}^{t+m} = \Phi_{\text{GRU}}(\{\mathbf{f}^\tau\})$; (3) an FCN scoring head generates a vulnerability score map $\mathbf{S}_{t+m} \in [0,1]^{H \times W}$.
- **Design Motivation**: Attacks require anticipation (with an $m=2$ frame delay), and temporal modeling captures the dynamic evolution of coverage blind spots induced by vehicle motion.

#### Design 3: Entropy-Aware Vulnerability Search and Attack Persistence
- **Function**: Optimize location selection and continuation/termination decisions across multi-frame attacks.
- **Mechanism**: A fabrication risk map is used for probabilistic sampling of attack locations; a persistence threshold $\eta$ governs attack continuation: $\mathcal{C}_{t+m+j} = \mathbb{I}\left[\mathbb{E}_{(x,y) \in \mathcal{N}(x_c,y_c)}[\tilde{\mathbf{S}}_{t+m}(x,y)] \geq \eta\right]$
- **Design Motivation**: Overly prolonged attacks expose temporal anomalies; intelligent termination — proactively abandoning regions of low fabrication probability — substantially reduces detection risk.

#### Design 4: Adaptive Exploitation of Defense Information
- **Function**: Adaptively obtain MVIG inputs depending on whether the defense system shares occupancy maps.
- **Mechanism**: For systems such as CAD that share occupancy maps, the maps are used directly. For systems without occupancy map sharing (e.g., ROBOSAC, CP-Guard), a Blind Region Segmentation (BRS) algorithm estimates approximate occupancy grids from feature maps.
- **Design Motivation**: This enables a universal attack capability across heterogeneous defense configurations.

### Loss & Training

**Perturbation Optimization (PGD)**:
- Spoofing: $\phi(\boldsymbol{\delta}, M_t^*) = \sum_{b \in B'} \text{IoU}(b, b_t) \cdot \log(b_\sigma)$, maximizing the confidence of fabricated targets.
- Removal: Negate the objective to minimize the detection confidence of real targets.
- Constraint: $\|\delta\|_\infty \leq 1.0$, with 5 PGD iterations and step size 0.01.

**Mask Optimization (MVIGNet)**: A joint loss with three objectives:
1. Attack effectiveness loss: maximize fabrication confidence while penalizing locations far from the victim.
2. Box discrimination loss: minimize overlap with existing detections.
3. Defense evasion loss: avoid regions flagged as contradictory in multi-vehicle occupancy maps.

Two-stage decoupled training: MVIGNet computes its loss using detection boxes generated by PGD, but gradients are not back-propagated into PGD.

## Key Experimental Results

### Main Results

**Dataset**: OPV2V (training), Adv-OPV2V (testing, 300 scenes × spoofing/removal)

| Attack Method | No Defense ASR | ROBOSAC DSR | CP-Guard DSR | GCP DSR | CAD DSR |
|---------|-----------|-------------|-------------|---------|---------|
| Basic [31] | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| RC spoof [39] | 92.4 | 12.0 | 18.1 | 12.2 | 83.5 |
| BAC [29] | 99.2 | 23.0 | 37.0 | 28.0 | 90.5 |
| **MVIG spoof** | 94.3 | **14.8** | **17.2** | **13.0** | **32.0** (-62%) |

| Attack Method | ROBOSAC DSR | CP-Guard DSR | GCP DSR | CAD DSR |
|---------|-------------|-------------|---------|---------|
| RC remove [39] | 14.2 | 21.4 | 15.0 | 90.1 |
| **MVIG remove** | **12.2** (-14%) | 21.3 | **14.2** (-5%) | **78.2** (-13%) |

MVIG achieves the most pronounced effect against CAD: in spoofing mode, DSR drops from 83.5% to 32.0% (↓62%).

### Persistent Attack (3-frame)

| Attack Method | ROBOSAC DSR | CP-Guard DSR | GCP DSR | CAD DSR |
|---------|-------------|-------------|---------|---------|
| BAC [29] | 74.2 | 68.1 | 86.0 | 98.0 |
| RC spoof [39] | 38.1 | 24.0 | 42.1 | 100.0 |
| **MVIG spoof** | **37.0** | **26.2** | **42.0** | **63.2** (-37%) |

Under persistent attacks, DSR for BAC and RC rises substantially (easily detected via temporal analysis), whereas MVIG maintains a significant advantage.

### Ablation Study

- **Influence region size**: For spoofing attacks, increasing the range from 15 m to 50 m reduces DSR from 40.2% to 14.8% (larger search space is beneficial); removal attacks remain relatively stable (72.1%–84.5%), constrained by the requirement to target existing objects.
- **Persistence threshold $\eta$**: Without a threshold, DSR is 85.3%/91.7% (spoof/removal); the optimal range of $\eta = 0.45$–$0.50$ balances attack opportunity against detection risk.
- **Real-time performance**: MVIG achieves 15.6–29.9 FPS, satisfying real-time requirements.

### Key Findings

1. MVIG exhibits the greatest advantage against CAD (−62%), because CAD's shared occupancy maps expose more exploitable vulnerability information.
2. Temporal consistency is critical for evading detection in multi-frame attacks — BAC's spatially-driven attacks produce inconsistent temporal patterns (DSR > 98%), whereas MVIG's temporal optimization maintains attack coherence (DSR = 63.2%).
3. ROC analysis shows that MVIG fundamentally reduces the separability of malicious vs. benign transmissions (AUC 0.77 vs. 0.85–0.91).
4. Attacks exploit perceptual conflict regions arising from natural occlusions and viewpoint differences among vehicles, making fabrications appear as ordinary perceptual discrepancies.

## Highlights & Insights

1. **"Defense Entails Disclosure" Insight**: This work is the first to systematically reveal the security paradox in which CP defense mechanisms inadvertently leak system vulnerabilities during the verification process — a profound security observation.
2. **Unified Graph Representation**: Using mutual information as edge weights, MVIG elegantly unifies vulnerability modeling across diverse defense systems without requiring defense-specific customization.
3. **Spatiotemporal Decoupled Attack**: Decoupling mask prediction from perturbation optimization ensures both attack quality and real-time feasibility (29.9 FPS).
4. **Intelligent Attack Termination**: Entropy-aware persistence control is a clever design — proactively abandoning low-probability windows substantially improves the overall stealthiness of the attack.

## Limitations & Future Work

1. **Limited Search Space for Removal Attacks**: The requirement to target existing objects inherently constrains the optimization space, resulting in smaller gains compared to spoofing attacks.
2. **Coverage Rate Constraint**: When the joint coverage of benign CAVs is high, attack effectiveness degrades (though dynamic coverage gaps remain exploitable).
3. **LiDAR-Only Validation**: The framework has not been extended to camera-based or multimodal fusion perception; camera-based CP represents a promising future direction.
4. **Absence of Defense Countermeasures**: As an attack-focused paper, no effective defensive strategies are proposed (only briefly discussed in the appendix).
5. **Simulation Environment**: Validation is conducted solely on CARLA simulator data; feasibility in real-world physical scenarios remains unverified.

## Related Work & Insights

- **Tu et al. [31]** (Basic): Pioneered feature-map perturbation, but modifications are too conspicuous and easily detected by anomaly detection.
- **Tao et al. [29]** (BAC): Exploits blind spots in victim viewpoints for obfuscation attacks, but considers only single-vehicle knowledge and lacks evasion of collective verification.
- **Zhang et al. [39]** (RC + CAD): Introduces LiDAR ray casting and occupancy map verification, but lacks systematic spatiotemporal optimization.
- **Insights**: The "sharing entails exposure" problem in collaborative systems is pervasive in federated learning, multi-agent communication, and related domains; MVIG's graph modeling approach is transferable to these settings.

## Rating

⭐⭐⭐⭐ Novel security perspective, elegant unified framework design, and comprehensive experiments; however, gains on removal attacks are limited and real-world validation is absent.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] CoLC: Communication-Efficient Collaborative Perception with LiDAR Completion](colc_communication-efficient_collaborative_perception_with_lidar_completion.md)
- [\[ICLR 2026\] SiMO: Single-Modality-Operable Multimodal Collaborative Perception](../../ICLR2026/autonomous_driving/simo_single-modality-operable_multimodal_collaborative_perceptio.md)
- [\[CVPR 2026\] AdaRadar: Rate Adaptive Spectral Compression for Radar-based Perception](adaradar_rate_adaptive_spectral_compression_for_radar-based_perception.md)
- [\[CVPR 2026\] CycleBEV: Regularizing View Transformation Networks via View Cycle Consistency for Bird's-Eye-View Semantic Segmentation](cyclebev_regularizing_view_transformation_networks_via_view_cycle_consistency_fo.md)
- [\[ICCV 2025\] SeqGrowGraph: Learning Lane Topology as a Chain of Graph Expansions](../../ICCV2025/autonomous_driving/seqgrowgraph_learning_lane_topology_as_a_chain_of_graph_expansions.md)

</div>

<!-- RELATED:END -->

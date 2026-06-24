---
title: >-
  [Paper Note] MNE-SLAM: Multi-Agent Neural SLAM for Mobile Robots
description: >-
  [CVPR 2025][3D Vision][Multi-agent SLAM] This work proposes MNE-SLAM, the first fully distributed multi-agent collaborative neural SLAM framework where each agent independently executes neural mapping and tracking. Decentralized collaboration is achieved via peer-to-peer communication for hierarchical loop closure detection (intra-to-inter) and multi-submap fusion. It is validated on Replica, ScanNet, TUM RGB-D, and a self-collected INS dataset. Additionally…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Multi-agent SLAM"
  - "Neural implicit representation"
  - "Distributed mapping"
  - "Peer-to-peer communication"
  - "Loop closure"
  - "Submap fusion"
  - "INS dataset"
date: 2026-05-08
content_hash: a5f9599e34e564bc
---

# MNE-SLAM: Multi-Agent Neural SLAM for Mobile Robots

**Conference**: CVPR 2025  
**Code**: [https://github.com/dtc111111/MNESLAM](https://github.com/dtc111111/MNESLAM)  
**Area**: 3D Vision / SLAM  
**Keywords**: Multi-agent SLAM, Neural implicit representation, Distributed mapping, Peer-to-peer communication, Loop closure, Submap fusion, INS dataset  

## TL;DR
This work proposes MNE-SLAM, the first fully distributed multi-agent collaborative neural SLAM framework where each agent independently executes neural mapping and tracking. Decentralized collaboration is achieved via peer-to-peer communication for hierarchical loop closure detection (intra-to-inter) and multi-submap fusion. It is validated on Replica, ScanNet, TUM RGB-D, and a self-collected INS dataset. Additionally, the first real-world indoor neural SLAM (INS) dataset covering both single- and multi-agent scenarios is released.

## Background & Motivation

**Background**: Neural implicit SLAM (such as NICE-SLAM, CoSLAM, ESLAM, etc.) has made significant progress in single-agent scenarios, enabling high-quality dense 3D reconstruction and precise camera tracking. These methods encode the scene as continuous neural implicit functions, offering advantages in reconstruction quality and storage efficiency over traditional point-cloud or voxel-based SLAM.

**Limitations of Prior Work**: (1) All existing neural implicit SLAM methods are restricted to single-agent operations, failing to exploit the collaborative advantages of multiple robots; (2) The only pioneer of multi-agent neural SLAM, CP-SLAM, adopts a centralized architecture that requires all agents to maintain continuous connections to a central server, supports only two agents, and operates extremely slowly; (3) Centralized architectures suffer from the risk of a single point of failure—a crashed central server paralyzes the entire system; (4) Traditional multi-agent SLAM methods (e.g., CCM-SLAM, Swarm-SLAM) support multiple agents but lack high-quality scene reconstruction capabilities; (5) There is a lack of real-world neural SLAM evaluation datasets containing multi-agent scenarios.

**Key Challenge**: Real-world mobile robot scenarios mandate distributed, low-communication-overhead multi-agent collaboration, but the computationally intensive nature and global consistency requirements of neural implicit representations present an inherent tension with distributed architectures. How to ensure scene representation consistency among multiple agents while avoiding centralized bottlenecks remains a critical challenge.

**Goal**: (1) Design a truly distributed multi-agent neural SLAM that achieves collaborative mapping and tracking solely through peer-to-peer communication without a central server; (2) Establish a real-world indoor evaluation benchmark containing both single- and multi-agent scenarios.

**Key Insight**: Treat each agent as an independent mapping unit maintaining its own neural submap. When agents "encounter" each other (i.e., explore overlapping regions), overlap is identified through loop closure detection, triggering submap fusion.

**Core Idea**: A distributed architecture combined with hierarchical loop closure detection (intra-agent loop closure first, followed by inter-agent loop closure) and submap fusion is proposed to achieve decentralized collaborative multi-agent neural SLAM.

## Method

### Overall Architecture
MNE-SLAM is a fully distributed framework. Each agent independently executes four modules: (1) local neural mapping and tracking (using hybrid-encoded implicit representations); (2) joint scene representation (sharing features in overlapping areas); (3) intra-to-inter loop closure detection; and (4) multi-submap fusion. The system runs on $n$ GPUs ($n$ being the number of agents), with each agent occupying one GPU.

### Key Designs

1. **Distributed Mapping and Joint Scene Representation**:

    - **Function**: Each agent independently maintains its local neural scene representation, while overlapping regions are coordinated.
    - **Mechanism**: Each agent employs a hybrid encoding (coordinate encoding + hash grid) for efficient implicit mapping, independently alternating between tracking and mapping iterations using RGB-D inputs. Joint scene representation is achieved via a lightweight feature alignment mechanism: when overlap is detected between the explored regions of two agents, the neural feature representations in the overlap area are aligned, keeping the scene encodings consistent across different agents in shared regions. This alignment requires no global optimization and is completed solely through local feature vector transformations.
    - **Design Motivation**: Centralized methods require transmitting all data to a server for processing, incurring heavy communication overhead and risking a single point of failure. Distributed methods allow each agent to run autonomously and exchange necessary information only upon "encounters."

2. **Intra-to-Inter Loop Closure Detection**:

    - **Function**: Hierarchically detect intra-agent loop closures and inter-agent external loop closures.
    - **Mechanism**: **Intra-loop closure**: Each agent independently detects loop closures in its own trajectory, triggered when the NetVLAD scene descriptor similarity between the current and historical frames exceeds a threshold. Relative camera poses are accurately estimated using features from DROID-SLAM, followed by local pose-graph optimization to eliminate cumulative drift. **Inter-loop closure**: Agents exchange NetVLAD description vectors of keyframe descriptors via peer-to-peer communication. When descriptors between two agents match, indicating that an overlapping region has been explored, inter-agent pose estimation and submap alignment are triggered.
    - **Design Motivation**: Prioritizing the consistency of individual agent trajectories (intra-loop) before handling cross-agent alignment (inter-loop) prevents erroneous cross-agent matching from corrupting single-agent trajectory estimation. This hierarchical strategy is more robust than single-tier global loop closure.

3. **Multi-Submap Fusion**:

    - **Function**: Merge overlapping submaps into a consistent global map.
    - **Mechanism**: Following inter-loop closure detection, a three-step fusion is executed: (a) coordinate frame alignment, which estimates the relative pose ($SE(3)$ transformation) between two submaps via loop constraints; (b) neural scene feature fusion in overlapping areas, merging the feature representations of the two submaps in the overlap region through weighted averaging or selective updating strategies; (c) global pose-graph optimization, jointly optimizing all agent trajectories under all loop closure constraints. The fused global map can be utilized for higher-quality rendering and navigation planning.
    - **Design Motivation**: Merely concatenating submaps results in inconsistencies (e.g., ghosting or discontinuities) in overlapping regions. Submap fusion ensures global geometric consistency.

### Loss & Training
The optimization for each agent includes a depth rendering loss $\mathcal{L}_{depth}$, a color rendering loss $\mathcal{L}_{color}$, and an SDF regularization loss. Tracking optimizes camera poses by minimizing the rendering error of the current frame, while mapping updates scene representations by minimizing keyframe rendering errors. Additional pose-graph optimization is performed after loop closures.

## Key Experimental Results

### Main Results (Tracking Accuracy ATE RMSE ↓, in cm)

| Method | Replica (Avg) | ScanNet (Avg) | TUM (Avg) | Architecture |
|------|--------------|--------------|-----------|------|
| NICE-SLAM (Single-agent) | 1.95 | 8.64 | 3.57 | Single-agent |
| CoSLAM (Single-agent) | 1.06 | 7.18 | 2.43 | Single-agent |
| CP-SLAM (Centralized multi-agent) | 2.31 | 10.25 | - | Centralized |
| **MNE-SLAM (Ours)** | **0.89** | **6.52** | **2.18** | **Distributed** |

### Ablation Study (INS Dataset)

| Configuration | ATE (cm) ↓ | Completion (cm) ↓ | Explanation |
|------|-----------|-------------------|------|
| Full MNE-SLAM (2 agents) | 1.24 | 2.15 | Complete two-agent system |
| w/o Inter-loop | 2.87 | 3.46 | No inter-agent loop closure detection |
| w/o Intra-loop | 1.68 | 2.58 | No intra-agent loop closure detection |
| w/o Submap fusion | 2.45 | 3.12 | Simple concatenation without fusion |
| Single-agent baseline | 1.52 | 2.78 | Single agent covering the same area |

### Key Findings
- The proposed distributed architecture of MNE-SLAM outperforms the centralized CP-SLAM in tracking accuracy across multiple datasets and supports a larger number of agents.
- The intra-to-inter hierarchical loop closure strategy is more effective than a single-tier loop closure approach (an ATE of 1.24 cm for intra-then-inter versus 1.56 cm for the inverse).
- The joint scene representation improves reconstruction quality in overlapping regions compared to a simple concatenation after independent mapping (yielding a 31% gain in Completion).
- The distributed architecture is robust to temporary communication interruptions, with the ATE only increasing by 0.21 cm under a simulated 30% communication packet loss rate.
- The INS dataset has been publicly released, containing RGB-D sequences collected by real mobile robots covering single- and multi-agent scenarios.
- The paper has been cited 47 times (as of April 2026), and its GitHub repository has received 181 stars.

## Highlights & Insights
- **First Distributed Multi-Agent Neural SLAM**: Compared to CP-SLAM, its sole predecessor, MNE-SLAM represents a qualitative leap in architectural design toward decentralization, scalability, and robustness, demonstrating direct engineering value for real-world mobile robot deployment.
- **Robustness of Hierarchical Loop Closure Detection**: The intra-to-inter strategy ensures logical clarity ("achieve self-consistency before aligning with others"), reducing the propagation of mismatches.
- **Benchmark Contribution of the INS Dataset**: As the first real-world indoor neural SLAM dataset covering both single- and multi-agent scenarios, it fills a critical gap in the community regarding multi-agent evaluations.
- **System Scalability**: Scaling the system to support more agents only requires adding GPUs without modifying the core system architecture, as computation for each agent remains completely independent.

## Limitations & Future Work
- Operating with neural implicit representations may result in a rendering speed and visual quality gap compared to 3D Gaussian Splatting methods (such as MAGiC-SLAM).
- Peer-to-peer communication complexity scales quadratically as $\mathcal{O}(n^2)$ when the number of agents is large (e.g., $n > 10$).
- The system requires RGB-D inputs, preventing direct deployment on mobile robots equipped only with RGB cameras.
- Feature alignment in joint scene representations requires sufficient overlapping areas, and its performance degrades in very large environments with scarce overlaps.
- The quantitative impact of communication bandwidth and latency on system performance is not thoroughly addressed.
- Combining this framework with 3D Gaussian Splatting to leverage the high-quality rendering capabilities of 3D-GS represents a promising future direction.

## Related Work & Insights
- **vs. CP-SLAM (Sole Predecessor)**: The centralized architecture of CP-SLAM limits scalability (supporting only 2 agents) and suffers from slow execution speeds. MNE-SLAM's distributed design supports a larger number of agents with superior tracking accuracy.
- **vs. NICE-SLAM / CoSLAM (Single-Agent)**: While these methods perform well on single-agent setups, they cannot exploit the advantages of multi-agent collaboration. MNE-SLAM achieves a win-win scenario in both accuracy and efficiency under multi-agent settings.
- **vs. Swarm-SLAM (Traditional Multi-Agent)**: Although Swarm-SLAM features a distributed architecture, it relies on traditional ORB features, yielding reconstruction quality far inferior to neural approaches. MNE-SLAM combines the benefits of both distributed architectures and neural representations.
- **vs. MAGiC-SLAM (3D-GS Multi-Agent)**: While MAGiC-SLAM utilizes 3D Gaussian Splatting and may deliver better rendering quality, both frameworks can inspire each other in architectural designs.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The first distributed multi-agent neural SLAM framework with an ingeniously designed hierarchical loop closure scheme.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Validated across four datasets, compared with multiple baselines, and backed by detailed ablation studies as well as robustness evaluations.
- Writing Quality: ⭐⭐⭐⭐ The system architecture is clearly described, and the dataset contribution carries long-term value.
- Value: ⭐⭐⭐⭐⭐ The 47 citations strongly reflect community recognition, and open-sourcing the code facilitates subsequent research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MAGiC-SLAM: Multi-Agent Gaussian Globally Consistent SLAM](magic-slam_multi-agent_gaussian_globally_consistent_slam.md)
- [\[CVPR 2025\] WildGS-SLAM: Monocular Gaussian Splatting SLAM in Dynamic Environments](wildgs-slam_monocular_gaussian_splatting_slam_in_dynamic_environments.md)
- [\[CVPR 2025\] MASt3R-SLAM: Real-Time Dense SLAM with 3D Reconstruction Priors](mast3r-slam_real-time_dense_slam_with_3d_reconstruction_priors.md)
- [\[CVPR 2025\] VarSplat: Uncertainty-aware 3D Gaussian Splatting for Robust RGB-D SLAM](varsplat_uncertainty-aware_3d_gaussian_splatting_for_robust_rgb-d_slam.md)
- [\[CVPR 2025\] Mobile-GS: Real-time Gaussian Splatting for Mobile Devices](mobile-gs_real-time_gaussian_splatting_for_mobile_devices.md)

</div>

<!-- RELATED:END -->

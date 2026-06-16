---
title: >-
  [Paper Note] A Data-Driven Model Predictive Control Framework for Multi-Aircraft TMA Routing Under Travel Time Uncertainty
description: >-
  [AAAI 2026][Autonomous Driving][TMA operations] A closed-loop MPC framework is proposed for conflict-free multi-aircraft routing and scheduling within the 50 NM Terminal Maneuvering Area (TMA) of Changi Airport. The fram…
tags:
  - "AAAI 2026"
  - "Autonomous Driving"
  - "TMA operations"
  - "STAR route planning"
  - "travel time uncertainty"
  - "model predictive control"
  - "MILP"
date: 2026-05-08
content_hash: a0833d86639eda4d
---

# A Data-Driven Model Predictive Control Framework for Multi-Aircraft TMA Routing Under Travel Time Uncertainty

**Conference**: AAAI 2026
**arXiv**: [2511.19452](https://arxiv.org/abs/2511.19452)  
**Area**: Air Traffic Management / Terminal Maneuvering Area Route Planning
**Keywords**: TMA operations, STAR route planning, travel time uncertainty, model predictive control, MILP

## TL;DR

A closed-loop MPC framework is proposed for conflict-free multi-aircraft routing and scheduling within the 50 NM Terminal Maneuvering Area (TMA) of Changi Airport. The framework integrates XGBoost-based TMA boundary arrival time prediction, MILP optimization (incorporating route selection, speed adjustment, holding control, and separation constraints), and a receding-horizon simulator. Under peak congestion scenarios of 36 aircraft/hour, it achieves a 7× computational speedup while significantly outperforming the Dijkstra baseline in feasibility under Monte Carlo robustness validation.

## Background & Motivation

**Background**: Flight delays impose substantial costs on passengers, airlines, and the environment. Air Traffic Flow Management (ATFM) is typically modeled as a dynamic multi-commodity network flow problem; however, optimization of the Terminal Maneuvering Area (TMA)—the high-density airspace surrounding airports—remains underexplored in the literature. Existing research focuses primarily on runway scheduling (Airport Scheduling Problem), with insufficient attention to holistic TMA airspace optimization.

**Limitations of Prior Work**:
(1) **Oversimplified TMA network models**—existing methods employ simplified maps with limited waypoints, incapable of representing the complex structure of real-world STARs (Standard Terminal Arrival Routes);
(2) **Lack of closed-loop real-time control**—most approaches rely on one-shot optimization and cannot adapt to dynamic changes in real-time systems;
(3) **Travel time uncertainty is ignored**—holding patterns and radar vectoring occur frequently in the TMA, resulting in high flight time variance (TMA internal variance: 204s vs. en-route segment: 81s), yet existing methods disregard the impact of this uncertainty on scheduling.

**Key Insight**: This paper integrates prediction, optimization, and simulation into a closed-loop MPC framework—replacing unreliable runway ETA prediction with data-driven TMA boundary arrival time prediction, substituting simplified models with MILP optimization over a full STAR network, and employing a receding-horizon strategy to enable real-time computation and robust control.

## Method

### Overall Architecture

The system comprises four components: a historical database, an XGBoost traffic predictor, a MILP real-time MPC controller, and a custom traffic simulator. ADS-B radar data and meteorological METAR data serve as inputs; the predictor outputs TMA boundary arrival times; the MPC optimizes landing times, routes, and speeds within a receding horizon window; optimization commands are applied to the simulator to update system state; and the process iterates until simulation completes.

### Key Designs

1. **Data-Driven TMA Boundary Arrival Time Prediction**

    - **Function**: Uses XGBoost to predict each aircraft's arrival time at the 50 NM TMA boundary: $T_f^t = \mathcal{F}(X_f^t)$
    - **Input features**: ADS-B real-time dynamics (latitude/longitude/ground speed/distance), runway direction, departure airport code, wake turbulence category, temporal attributes, and METAR meteorological information (wind speed/direction/visibility/cloud cover)
    - **Design Motivation**: Predicting TMA boundary arrival time is more reliable than directly predicting runway ETA—flight time variance within the TMA is extremely large due to holding and vectoring operations (2.5× that of en-route segments), whereas aircraft behavior prior to the TMA boundary is comparatively stable. Delegating uncertainty to the optimizer represents a more principled architectural choice.

2. **Full STAR Network MILP Optimization Model**

    - **Function**: Solves for conflict-free optimal routes for multiple aircraft over a real-world STAR network within a 50 NM radius
    - **Core constraints**:
        - Flow conservation (analogous to vehicle routing problem flow balance)
        - Arrival/departure time constraints (incorporating discrete speed level selection)
        - Separation constraints (arrival/departure interval at any shared waypoint between any two aircraft $\geq t_{f,f'}$)
        - No-overtaking constraints (an aircraft departing first on a segment must arrive first)
        - Holding constraints (holding permitted only at designated waypoints)
    - **Objective function**: Minimize average landing time across all aircraft: $J = \min \frac{1}{|F|}\sum_{f \in F, j \in E} AR_j^f$
    - **Design Motivation**: A real STAR map constructed from complete AIP (Aeronautical Information Publication) data, rather than a simplified network, more accurately reflects operational constraints.

3. **Receding-Horizon Closed-Loop Control**

    - **Function**: Decomposes the full planning horizon into sub-problems, enabling real-time control through bidirectional interaction between the MPC and the simulator
    - **Mechanism**:
        - Look-ahead horizon = 10 min; control horizon = 5 min
        - At each step, only aircraft within the look-ahead horizon are optimized, but only commands within the control horizon are executed
        - The simulator updates system state per second, handling three aircraft position cases: en-route, arrived and holding, and departed
        - Under perturbation mode, arrival time deviation $\xi_{arr}$ and state observation deviation $\xi_{ob}$ are introduced
    - **Design Motivation**: One-shot MILP optimization is computationally intractable under high-congestion scenarios due to exponential growth in solve time; the receding-horizon approach decomposes the large problem into sub-problems solvable within seconds.

## Key Experimental Results

### Computational Complexity Comparison (10–27 Aircraft, 10-Minute Window)

| Case | # Aircraft | MILP Time (s) | MILP Cost (s) | Dijkstra Time (s) | Dijkstra Cost (s) | Gap |
|------|-----------|--------------|--------------|------------------|------------------|-----|
| 1 | 10 | 2.38 | 1189.2 | 14.99 | 1191.5 | 0.19% |
| 6 | 15 | 20.50 | 1084.7 | 21.97 | 1103.8 | 1.76% |
| 8 | 25 | 173.51 | 1072.5 | 24.48 | ~1080.6 | *(2 infeasible)* |
| 9 | 27 | 2609.6 | 1083.9 | 24.82 | ~1085.3 | *(2 infeasible)* |

### Receding Horizon vs. One-Shot Optimization (Historical Real Data, 1 Hour)

| Case | # Aircraft | One-Shot Time (s) | One-Shot Cost (s) | Rolling Time (s) | Rolling Cost (s) |
|------|-----------|------------------|------------------|-----------------|-----------------|
| 2 | 25 | 169.30 | 2298.1 | 42.41 | 2302.2 |
| 5 | 36 | 784.69 | 2736.3 | 102.75 | 2736.3 |

### Monte Carlo Robustness Test (36 Aircraft, 100 Runs × 4 Perturbation Levels)

- Across random perturbation values in the range 0.05–0.2, the number of infeasible cases under MILP is substantially lower than under Dijkstra.
- At perturbation value = 0.2, Dijkstra yields only 1 feasible solution, whereas MILP maintains a significantly higher feasibility rate.
- MILP solve time remains stable within 30–40s and does not increase noticeably with perturbation intensity.

### Key Findings

- Under peak congestion (36 aircraft/hour), the receding-horizon approach achieves over 7× computational speedup (784.69s → 102.75s) with no cost degradation.
- The greedy Dijkstra strategy cannot guarantee feasibility under high congestion (infeasible aircraft appear in large-scale cases), whereas MILP remains feasible throughout.
- MILP solve time grows exponentially with problem size (27 aircraft = 2609s), validating the necessity of the receding-horizon approach.
- Travel time variance within the TMA (204s) is 2.5× that of en-route segments (81s), supporting the design choice to predict TMA boundary arrival times rather than runway ETAs.

## Highlights & Insights

1. **System-level closed-loop framework**: Seamlessly integrates prediction, optimization, and simulation, with genuine deployment potential.
2. **Real AIP-based STAR network**: More faithful to actual operations than simplified models, with comprehensive constraint coverage (route, speed, holding, separation, no-overtaking).
3. **Pragmatic prediction design**: Predicting TMA boundary arrival time (low uncertainty) rather than runway ETA (high uncertainty) delegates the hard-to-predict component to the optimizer.
4. **Monte Carlo robustness validation**: Systematic validation across 100 runs × 4 perturbation levels is convincing.

## Limitations & Future Work

1. Only 2D route planning is considered; vertical layer strategies are not addressed.
2. Speed selection is restricted to preset discrete levels; continuous speed optimization is not supported.
3. Case 9 (27 aircraft/10 min) already requires 2609s; larger-scale instances require further acceleration (e.g., column generation, branch-and-price).
4. Multi-runway selection optimization and departure–arrival interaction are not considered.
5. The simulator is a custom-built simplified version; integration and validation with real ATC systems remain incomplete.

## Related Work & Insights

- The closed-loop MPC + MILP framework is generalizable to other real-time scheduling problems (e.g., port scheduling, warehouse robot path planning).
- The architectural design of delegating prediction uncertainty to the online optimizer is worth adopting in related domains.
- The Monte Carlo robustness validation methodology is applicable to other safety-critical systems.

## Rating

⭐⭐⭐⭐

- **Novelty** ⭐⭐⭐: Incremental innovation; individual components have precedents, but the system-level integration adds value.
- **Experimental Thoroughness** ⭐⭐⭐⭐⭐: Real ADS-B data, computational complexity analysis, receding-horizon comparison, and Monte Carlo robustness validation—extremely comprehensive.
- **Writing Quality** ⭐⭐⭐⭐: Mathematical modeling is clear and rigorous.
- **Value** ⭐⭐⭐⭐: Direct applicability to real-world ATC decision support.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Task Prototype-Based Knowledge Retrieval for Multi-Task Learning from Partially Annotated Data](task_prototype-based_knowledge_retrieval_for_multi-task_lear.md)
- [\[ICCV 2025\] Adaptive Dual Uncertainty Optimization: Boosting Monocular 3D Object Detection under Test-Time Shifts](../../ICCV2025/autonomous_driving/adaptive_dual_uncertainty_optimization_boosting_monocular_3d_object_detection_un.md)
- [\[CVPR 2026\] Den-TP: A Density-Balanced Data Curation and Evaluation Framework for Trajectory Prediction](../../CVPR2026/autonomous_driving/den_tp_a_density_balanced_data_curation_and_evaluation_framework_for_trajectory.md)
- [\[AAAI 2026\] Drive As You Like: Strategy-Level Motion Planning Based on A Multi-Head Diffusion Model](drive_as_you_like_strategy-level_motion_planning_based_on_a_multi-head_diffusion.md)
- [\[AAAI 2026\] TimeBill: Time-Budgeted Inference for Large Language Models](timebill_time-budgeted_inference_for_large_language_models.md)

</div>

<!-- RELATED:END -->

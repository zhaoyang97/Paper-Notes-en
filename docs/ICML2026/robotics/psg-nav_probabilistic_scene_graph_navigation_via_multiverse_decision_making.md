---
title: >-
  [Paper Note] PSG-Nav: Probabilistic Scene Graph Navigation via Multiverse Decision Making
description: >-
  [ICML 2026][Robotics][ObjectNav] This paper proposes PSG-Nav, a suite comprising a "3D Probabilistic Scene Graph retaining full class distributions + Multiverse Decision Making sampling consistent worlds from joint distr…
tags:
  - "ICML 2026"
  - "Robotics"
  - "ObjectNav"
  - "Probabilistic Scene Graph"
  - "Multiverse Sampling"
  - "Evidence Calibration"
  - "Lifelong Adaptation"
date: 2026-05-08
content_hash: 30fb60ceb618a084
---

# PSG-Nav: Probabilistic Scene Graph Navigation via Multiverse Decision Making

**Conference**: ICML 2026  
**arXiv**: [2606.01313](https://arxiv.org/abs/2606.01313)  
**Code**: https://psg-nav.github.io/  
**Area**: Robotics / Embodied AI / Open-Vocabulary Navigation / Uncertainty Modeling  
**Keywords**: ObjectNav, Probabilistic Scene Graph, Multiverse Sampling, Evidence Calibration, Lifelong Adaptation

## TL;DR
This paper proposes PSG-Nav, a suite comprising a "3D Probabilistic Scene Graph retaining full class distributions + Multiverse Decision Making sampling consistent worlds from joint distributions + Evidence Calibration based on success/failure memory." It replaces traditional deterministic scene graph navigation and achieves new SOTA results on HM3D / MP3D / HSSD ObjectNav benchmarks with 66.1% / 44.8% / 67.9% SR, respectively.

## Background & Motivation

**Background**: Mainstream open-vocabulary ObjectNav employs modular pipelines that use open-vocabulary detectors like GLIP / Grounded-SAM to construct 3D scene graphs (e.g., SG-Nav, CogNav, ASCENT, ApexNav). These are followed by LLMs for high-level planning—given a target like "blue sofa," the agent must locate and stop near the object in unseen indoor environments.

**Limitations of Prior Work**: Current scene graphs often assign a single "hard label" with the highest confidence to each object for storage efficiency and LLM compatibility, discarding the full category probability distribution. This leads to three cascading failures: (1) Perceptual noise is permanently written into the map (e.g., a sofa misdetected as a bed cannot be corrected); (2) Logically inconsistent layouts appear in the scene graph (e.g., a toilet in a bedroom), causing downstream LLM reasoning to fail; (3) High-frequency false positives under sim-to-real domain shifts cause agents to STOP prematurely, leading to episode failure.

**Key Challenge**: Perception models are inherently uncertain (e.g., an object might be 70% sofa and 30% bed), but downstream planning requires a "deterministic" scene as an LLM prompt. Hard truncation loses global reasoning capabilities, whereas planning directly over the full joint distribution leads to combinatorial explosion (the most probable global configuration may still account for $< 10\%$).

**Goal**: (a) Retain full distributions at the mapping layer; (b) Transform probabilistic reasoning into tractable discrete decisions at the planning layer; (c) Combat sim-to-real false positives at the termination layer to achieve true online lifelong adaptation.

**Key Insight**: The authors draw inspiration from "Multiverse Decision Making." Since a single deterministic world is either overconfident or loses information, $K$ "logically consistent possible worlds" are sampled from the joint distribution. The agent evaluates the same landmark across multiple parallel universes and makes decisions via win-rate aggregation. Simultaneously, a RAG-style success/failure memory bank is used for posterior calibration of detection confidence.

**Core Idea**: Use a hierarchical probabilistic scene graph (object → group → room) for factorization to avoid joint explosion. Employ Monte Carlo sampling of multiple worlds combined with LLM pairwise comparisons for robust decision-making. Use Evidence-based Calibration (EEC) to incrementally update the memory bank after each episode, upgrading "zero-shot navigation" to "online lifelong learning."

## Method

### Overall Architecture
The input consists of RGB-D observation sequences $O_t = \{I_t^{rgb}, I_t^{depth}, p_t\}$ and a free-text goal $c$. The output is a discrete action $a_t \in \{\text{MOVE\_FORWARD}, \text{TURN\_LEFT/RIGHT}, \text{LOOK\_UP/DOWN}, \text{STOP}\}$. Success is defined as reaching within 1m of the goal and executing STOP within a 500-frame budget. The pipeline comprises three components: (A) **3D-PSG** online construction of a hierarchical probabilistic scene graph where each object node maintains a full class distribution; (B) **Multiverse Decision** sampling $K$ consistent worlds from the 3D-PSG to perform pairwise comparisons and information gain scoring for sub-goal selection; (C) **EEC** using success/failure memory banks for confidence calibration when a candidate object is detected.

### Key Designs

1.  **3D Probabilistic Scene Graph + LLM-Guided Hierarchical Logical Pruning**:
    - **Function**: Retains full perceptual distributions while making joint reasoning tractable.
    - **Mechanism**: The scene graph $\mathcal{G}_t = (\mathcal{V}, \mathcal{E})$ is divided into object, group, and room layers. Each object node maintains a class vote count vector $\mathbf{n}_{i,t}$, with confidence normalized by $P_t(o_i = c_k) = n_{i,t}^{(k)} / \sum_j n_{i,t}^{(j)}$ (using vote accumulation instead of Bayesian updates as open-vocab detector scores are uncalibrated and could cause divergence). Group nodes store joint configuration probabilities $P(g_j = s) = \prod_{i=1}^{N_j} P(o_{j,i} = c_{j,i}^s)$ (e.g., 70% table × 80% chair = 56% "table+chair"). A crucial logical pruning step enumerates the top-$K_g$ configurations and uses an LLM for binary filtering $f_{\text{LLM}}(s) \in \{0,1\}$ to discard logical conflicts like "toilet in a living room." The room layer follows similar enumeration and pruning.
    - **Design Motivation**: Discarding the full distribution prevents "alternative explanations." Hierarchical factorization reduces combinatorial explosion to enumerable sets. LLM filtering removes "highly confident but semantically nonsensical" configurations, preserving "low confidence but globally consistent" correct interpretations—a core mechanism for recovering truth from perceptual noise.

2.  **Multiverse Decision Making + Intrinsic Uncertainty-Aware Exploration**:
    - **Function**: Converts intractable probabilistic planning into concrete reasoning over discrete worlds and drives active disambiguation.
    - **Mechanism**: $K$ deterministic worlds $\mathcal{M} = \{\mathcal{G}^{(1)}, \dots, \mathcal{G}^{(K)}\}$ are sampled from the 3D-PSG joint distribution. Candidate landmarks are extracted from a Generalized Voronoi Graph and geometric frontiers. Intrinsic information gain $U_{\text{gain}}(l_{i,t}) = \alpha \cdot I_{\text{spa}}(l_{i,t}) + I_{\text{sem}}(l_{i,t})$ is used for filtering. The spatial term $I_{\text{spa}} = |\mathcal{U}(l_{i,t})| / (\pi r_{\text{max}}^2)$ measures the visible unknown area, while the semantic term $I_{\text{sem}} = -\sum_{o_i \in \mathcal{O}_p} \sum_c P_t(o_i = c) \log P_t(o_i = c)$ represents the Shannon entropy of nearby objects. Remaining high-potential landmarks undergo randomized pairwise comparisons. Within each world $\mathcal{G}^{(m)}$, the LLM acts as a preference oracle $\mathbb{I}(l_i \succ l_j | \mathcal{G}^{(m)}) = f_{\text{LLM}}(D(l_i | \mathcal{G}^{(m)}), D(l_j | \mathcal{G}^{(m)}), g)$. The final score is $S(l_{i,t}) = \frac{1}{M(|\mathcal{L}'_t|-1)} \sum_m \sum_{j \neq i} \mathbb{I}(l_i \succ l_j | \mathcal{G}^{(m)})$, and the optimal sub-goal is $l^* = \arg\max(S(l_{i,t}) + \beta U_{\text{gain}}(l_{i,t}))$.
    - **Design Motivation**: Planning in a single deterministic world is a "gamble." Multiverse sampling marginalizes perceptual noise by evaluating landmarks across parallel hypotheses. Pairwise comparisons avoid LLM position bias in listwise ranking. Information gain encourages the agent to actively resolve map ambiguity—true "uncertainty-aware exploration."

3.  **EEC: RAG-style Termination Calibration based on Success/Failure Memory**:
    - **Function**: Suppresses persistent false positives caused by sim-to-real domain shifts, allowing STOP decisions to be corrected via lifelong learning.
    - **Mechanism**: Two banks are maintained—positive examples $\mathcal{B}^+$ (successfully identified goals) and negative examples $\mathcal{B}^-$ (historical false positives). Each memory comprises $m = (\mathbf{v}_{\text{vis}}^m, \mathbf{v}_{\text{struct}}^m)$, where $\mathbf{v}_{\text{vis}}$ is the visual embedding and $\mathbf{v}_{\text{struct}} = (p_R^m, p_G^m)$ captures room and group distributions. Before a candidate object $o_c$ triggers a STOP, a hybrid similarity query is performed: $\text{sim}(o_c, m) = \cos(\mathbf{v}_{\text{vis}}, \mathbf{v}_{\text{vis}}^m) + w_1 \cos(p_G, p_G^m) + w_2 (1 - \text{JSD}(p_R, p_R^m))$. Let $S_{\text{pos}} = \max_{m \in \mathcal{B}^+} \text{sim}(o_c, m)$ and $S_{\text{neg}} = \max_{m \in \mathcal{B}^-} \text{sim}(o_c, m)$. Calibration margin $\Delta S = S_{\text{pos}} - \gamma S_{\text{neg}}$ is added to the detection score. STOP occurs only if $S_{\text{final}} = S_{\text{det}} + \Delta S > \delta$. Redundant memories with high average internal similarity are pruned when the bank exceeds capacity $N_{\max}$.
    - **Design Motivation**: Traditional RAG often lacks scene context. EEC's dual-distribution context (room + group) leverages the 3D-PSG structure. Diversity pruning ensures the bank is not overwhelmed by a single error pattern, maintaining decision calibration over the long term.

### Loss & Training
PSG-Nav is a training-free zero-shot framework. All probabilistic updates (vote accumulation / EEC bank updates) are online state updates within or across episodes. Detection uses GLIP, segmentation uses Grounded-SAM, and the reasoning engine is Qwen2.5-7B-Instruct. Key hyperparameters: multiverse samples $K = 3$, information gain threshold $\tau = 0.1$, weights $\alpha = 1, \beta = 0.5$, EEC capacity $N_{\max} = 10$, negative penalty $\gamma = 2$, and termination threshold $\delta = 0.61$. Max 500 steps per episode.

## Key Experimental Results

### Main Results

Comparison of 16 SOTA methods across HM3D (2000 episodes / 6 classes), MP3D, and HSSD (1248 episodes):

| Method | HM3D SR | HM3D SPL | MP3D SR | MP3D SPL | HSSD SR | HSSD SPL |
|------|---------|----------|---------|----------|---------|----------|
| SG-Nav | 54.0 | 24.9 | 40.2 | 16.0 | — | — |
| BeliefMapNav | 61.4 | 30.6 | 37.3 | 17.6 | 65.2 | 32.1 |
| ApexNav | 59.6 | 33.0 | 39.2 | 17.8 | — | — |
| ASCENT | 65.4 | 33.5 | 44.5 | 15.5 | — | — |
| **PSG-Nav (w/o EEC, Zero-Shot)** | 63.5 | 31.2 | 43.3 | 17.6 | 66.1 | 32.2 |
| **PSG-Nav (Adaptive, w/ EEC)** | **66.1** | 32.1 | **44.8** | **17.9** | **67.9** | **33.4** |

PSG-Nav outperforms the deterministic SG-Nav by 12.1 points in SR on HM3D. The zero-shot variant (without EEC) already surpasses BeliefMapNav and ApexNav. The full version achieves SOTA results across all three datasets. Real-robot deployment verified sim-to-real transferability.

### Ablation Study

| Configuration | HM3D SR | HSSD SR | Description |
|------|---------|---------|------|
| Full PSG-Nav | 66.1 | 67.9 | Complete framework |
| w/o 3D-PSG (Deterministic) | 58.4 | 58.5 | SR drops 9.4 pt on HSSD, proving distribution retention is key |
| w/o Group nodes | 58.8 | 59.9 | Close to deterministic, highlighting hierarchical factorization |
| w/o Room nodes | 59.7 | 61.7 | Slightly better than w/o Group, still > 6 pt below full |
| w/o Spa. & Sem. Info Gain | 62.1 | — | 4 pt drop due to lack of intrinsic exploration |

### Key Findings
- Removing 3D-PSG to revert to deterministic scene graphs results in a 9.4 pt drop (67.9 → 58.5) on HSSD, proving that "retaining full distributions + hierarchical reasoning" is vital.
- Eliminating Group nodes (58.8 SR) is nearly equivalent to the deterministic version (58.4 SR). Without hierarchical factorization, joint configurations explode, making multiverse sampling ineffective.
- The zero-shot variant (w/o EEC) already outperforms most SOTA methods; EEC provides an additional 1.5–2.6 pt boost as a robustness amplifier.
- $K = 3$ for multiverse sampling is sufficient, with marginal returns for higher values.
- Successful transfer from simulation to physical robots demonstrates the framework's engineering viability.

## Highlights & Insights
- **Value of "Distributions over Hard Labels"**: Perception models output logits; discarding them for argmax limits downstream reasoning. This work proves that hierarchical factorization and sampling allows full utilization of these distributions.
- **Multiverse Decision as a Robust LLM Reasoning Paradigm**: Pairwise comparison and win-rate aggregation across $K$ worlds essentially estimate expected utility $E[\mathbb{I}(l^* \succ l_j)]$, avoiding LLM position bias and marginalizing perception noise.
- **Upgrading Zero-Shot to Lifelong Learning via EEC**: The dual-bank design and diversity pruning provide a pragmatic yet theoretically grounded way to refine decision boundaries using JSD for room distributions and cosine similarity for visual embeddings.
- **LLM as Commonsense Filter**: Utilizing LLMs for binary logic filtering (yes/no) after configuration enumeration, rather than direct planning, is a clever, lightweight application.
- **Real-Robot Validation**: Unlike many ObjectNav works confined to Habitat, this method's successful physical deployment provides strong evidence of its reproducibility.

## Limitations & Future Work
- Bayesian updates are avoided due to uncalibrated detector confidence, using "vote accumulation" instead. This discards confidence magnitude, which a lightweight calibration head might recover.
- Multiverse sampling is limited to $K=3$; scalability in larger scenes or multi-goal tasks requires further analysis.
- EEC bank capacity $N_{\max} = 10$ may be insufficient for extreme long-term runs, and the pruning strategy might lose important modes under heavy distribution drift.
- The LLM's triple role (rationale, pairwise oracle, commonsense filter) might limit real-time performance; distillation into smaller models is possible.
- Extension to outdoor or multi-floor environments remains to be verified.
- Five hyperparameters ($\alpha, \beta, \gamma, \delta, \tau$) currently require manual tuning; adaptive mechanisms are a future direction.

## Related Work & Insights
- **vs SG-Nav / CogNav**: Both use 3D scene graphs, but PSG-Nav's use of probabilistic nodes and hierarchical pruning leads to a significant performance leap (+12.1 pt on HM3D).
- **vs ASCENT**: ASCENT focuses on stair-aware exploration for multi-floor navigation. PSG-Nav focuses on semantic uncertainty modeling; the two are orthogonal and potentially combinable.
- **vs BeliefMapNav**: While BeliefMapNav uses 3D voxel belief maps (continuous), PSG-Nav uses discrete probabilistic scene graphs, which are more compatible with LLM reasoning.
- **vs Conformal Prediction**: Those methods provide rigorous confidence intervals at the perception level but rarely propagate probability to decision-making as explicitly as PSG-Nav.
- **vs Traditional RAG**: Unlike vanilla RAG that only queries image crops, EEC's dual-distribution context is specifically designed for 3D-PSG structures.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Dive into the Scene: Breaking the Perceptual Bottleneck in Vision-Language Decision Making via Focus Plan Generation](dive_into_the_scene_breaking_the_perceptual_bottleneck_in_vision-language_decisi.md)
- [\[NeurIPS 2025\] Spatial-Aware Decision-Making with Ring Attractors in Reinforcement Learning Systems](../../NeurIPS2025/robotics/spatial-aware_decision-making_with_ring_attractors_in_reinforcement_learning_sys.md)
- [\[NeurIPS 2025\] ESCA: Contextualizing Embodied Agents via Scene-Graph Generation](../../NeurIPS2025/robotics/esca_contextualizing_embodied_agents_via_scene-graph_generation.md)
- [\[ICML 2026\] Embodied Task Planning via Graph-Informed Action Generation with Large Language Models](embodied_task_planning_via_graph-informed_action_generation_with_large_language_.md)
- [\[CVPR 2026\] FineCog-Nav: Integrating Fine-grained Cognitive Modules for Zero-shot Multimodal UAV Navigation](../../CVPR2026/robotics/finecog_nav_fine_grained_cognitive_modules_for_zero_shot_uav_navigation.md)

</div>

<!-- RELATED:END -->

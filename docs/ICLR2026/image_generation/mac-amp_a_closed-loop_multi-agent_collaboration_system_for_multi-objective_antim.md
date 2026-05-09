---
title: >-
  [Paper Note] MAC-AMP: A Closed-Loop Multi-Agent Collaboration System for Multi-Objective Antimicrobial Peptide Design
description: >-
  [ICLR 2026][Image Generation][antimicrobial peptide design] This paper proposes MAC-AMP, the first closed-loop multi-agent collaboration system that reformulates antimicrobial peptide (AMP) design as a coordinated multi-agent optimization problem, achieving multi-objective optimization through AI-simulated peer review and adaptive reward design.
tags:
  - ICLR 2026
  - Image Generation
  - antimicrobial peptide design
  - multi-agent collaboration
  - closed-loop reinforcement learning
  - multi-objective optimization
  - LLM agent
date: 2026-05-08
content_hash: b23a974dc5c53947
---

# MAC-AMP: A Closed-Loop Multi-Agent Collaboration System for Multi-Objective Antimicrobial Peptide Design

**Conference**: ICLR 2026
**arXiv**: [2602.14926](https://arxiv.org/abs/2602.14926)
**Code**: [GitHub](https://github.com/CLMFAP/MAC-AMP_v1/)
**Area**: Image Generation
**Keywords**: antimicrobial peptide design, multi-agent collaboration, closed-loop reinforcement learning, multi-objective optimization, LLM agent

## TL;DR

This paper proposes MAC-AMP, the first closed-loop multi-agent collaboration system that reformulates antimicrobial peptide (AMP) design as a coordinated multi-agent optimization problem, achieving multi-objective optimization through AI-simulated peer review and adaptive reward design.

## Background & Motivation

- **Antimicrobial Resistance (AMR) Crisis**: Directly responsible for approximately 1.14 million deaths in 2021, with projections exceeding 39 million direct deaths between 2025 and 2050.
- **Limitations of Existing AMP Design Models**:
    - Most optimize only for antimicrobial activity, neglecting toxicity, stability, and novelty.
    - Multi-objective optimization is unstable; static weights readily cause reward hacking or diversity collapse.
    - Outputs are typically scattered scores or text, making it difficult to convert them into reproducible learning signals.
- **Limitations of Existing Multi-Agent Systems**:
    - Outputs are primarily in natural language, lacking trainable optimization signals.
    - Most are open-loop systems that rely on human intervention.

## Method

### Overall Architecture

MAC-AMP comprises six interconnected modules: Input Module → Property Prediction → AI-Simulated Peer Review → RL Refinement → Peptide Generation → Output Module. Users need only provide the target bacterial name and an example dataset.

### 1. Property Prediction Module

Evaluates multiple AMP attributes, categorized into two types:
- **Explicit Reward Signals $S$**: Antimicrobial activity score $S_a$ (MIC predictor fine-tuned on ProtBERT), AMP likelihood score $S_b$ (Macrel 1.5).
- **Auxiliary Evidence $V$**: Toxicity score $V_a$ (ToxinPred 3.0), structural reliability $V_b$ (OmegaFold), physicochemical properties $V_c$ (ProtParam), template similarity $V_d$ (Foldseek).

### 2. AI-Simulated Peer Review Module

- **Three Independent Reviewer Agents** (GPT-5, Gemini 2.5, Perplexity) evaluate peptides across four dimensions: efficacy, safety, developmental structure, and originality.
- Each dimension is associated with a weighted vocabulary sub-table, using the tag format $\text{ID}(\text{State}, \text{Weight})$ for structured annotation.
- **Area Chair Agent**: Aggregates review results, resolves semantic conflicts, computes dimension-level meta-scores, and outputs meta-review text $T$ and average meta-score $S_c$.

### 3. RL Refinement Module

- **CS-Based Reward Design Agent**: Optimizes the reward function based on observable signals and mathematical properties.
- **Biomedical Reward Alignment Agent**: Analyzes meta-review text and proposes revision recommendations grounded in domain knowledge.
- Candidate rewards are filtered by a rule-based validator → short-term sandbox training → Pareto optimization to select the optimal reward function.
- **Phase-Adaptive Optimization**: The reward function is redesigned every 15 epochs over 3 iterations.

### 4. PPO Optimization

Normalized advantage: $A = \text{norm}(R - \bar{V}_\phi)$

Clipped surrogate loss:

$$L_{policy}(\theta) = \mathbb{E}[\min(r(\theta)A, \text{clip}(r(\theta), 1-\epsilon, 1+\epsilon)A)]$$

Total loss:

$$L = L_{policy} + c_v L_{value} - c_e L_{ent}$$

where $L_{value}$ is the value regression loss and $L_{ent}$ is the entropy regularization term.

## Key Experimental Results

### Main Results: Target-Specific AMP Evaluation

| Model | Antimicrobial Activity (↑) | AMP Likelihood (↑) | Toxicity (↓) | Structural Reliability (↑) |
|------|-------------|-------------|---------|-------------|
| **MAC-AMP** | **0.943±0.008** | 0.797±0.012 | **0.154±0.008** | **0.873±0.009** |
| AMP-Designer | 0.807±0.021 | 0.811±0.011 | 0.251±0.024 | 0.817±0.017 |
| BroadAMP-GPT | 0.831±0.025 | **0.821±0.018** | 0.246±0.033 | 0.763±0.023 |
| PepGAN | 0.823±0.023 | 0.572±0.035 | 0.247±0.064 | 0.637±0.026 |
| Diff-AMP | 0.822±0.006 | 0.554±0.036 | 0.235±0.072 | 0.752±0.020 |

*Results on E. coli target*

### Broad-Spectrum Activity Evaluation

| Model | E. coli | S. aureus | P. aeruginosa | K. pneumoniae | E. faecium |
|------|---------|-----------|---------------|---------------|------------|
| **MAC-AMP** | **0.94** | 0.81 | **0.94** | **0.98** | 0.95 |
| AMP-Designer | 0.81 | 0.81 | 0.85 | 0.96 | 0.96 |
| PepGAN | 0.82 | **0.89** | 0.91 | 0.98 | 0.96 |

### Key Findings

1. MAC-AMP comprehensively outperforms baselines in antimicrobial activity, toxicity, and structural reliability.
2. AMPs designed for *E. coli* generalize well to other Gram-negative bacteria (which share outer membrane structures).
3. Strong generalization is also demonstrated for *E. faecium* (a Gram-positive bacterium).
4. Training cost: 47.61 GPU hours, 853 API calls, API expenditure of $36.56.

## Highlights & Insights

1. **First Closed-Loop Multi-Agent System**: Converts natural-language review consensus into executable RL reward signals, bridging the gap between output format and training signal.
2. **End-to-End Interpretability**: Overcomes black-box limitations through transparent logging, replay trajectories, and consensus-aware decision tracking.
3. **Cross-Domain Transferability**: The framework's generality is validated on English table-to-text generation tasks.
4. **Multi-Objective Balance**: Achieves multi-objective optimization through structured agent consensus rather than manually specified static weights.

## Limitations & Future Work

- Generated peptides have not yet been validated through in vitro experiments.
- API call costs may limit large-scale deployment.
- The peer review module relies on specific commercial LLMs, constraining reproducibility.
- The phase interval (15 epochs) and iteration count (3 rounds) are hyperparameters that may require task-specific tuning.

## Related Work & Insights

- **AMP Generation**: AMPGAN v2, Diff-AMP, AMP Designer
- **LLM Multi-Agent Collaboration**: Virtual Lab, CAMEL, AutoGen, ReviewAgents
- **LLM-Augmented RL**: RLAIF, Eureka

## Rating

- Novelty: ⭐⭐⭐⭐ — First framework to apply closed-loop multi-agent collaboration to molecular design.
- Technical Depth: ⭐⭐⭐⭐ — Modular design is sophisticated with thorough multi-level validation.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Five bacterial targets, four baselines, and multi-dimensional ablation studies.
- Practical Value: ⭐⭐⭐⭐ — Extensible to other molecular design tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Multi-agent Coordination via Flow Matching](multi-agent_coordination_via_flow_matching.md)
- [\[ICLR 2026\] Pareto-Conditioned Diffusion Models for Offline Multi-Objective Optimization](pareto-conditioned_diffusion_models_for_offline_multi-objective_optimization.md)
- [\[ICLR 2026\] JointDiff: Bridging Continuous and Discrete in Multi-Agent Trajectory Generation](jointdiff_bridging_continuous_and_discrete_in_multi-agent_trajectory_generation.md)
- [\[CVPR 2026\] coDrawAgents: A Multi-Agent Dialogue Framework for Compositional Image Generation](../../CVPR2026/image_generation/codrawagents_a_multi-agent_dialogue_framework_for_compositional_image_generation.md)
- [\[ACL 2026\] BookAgent: Orchestrating Safety-Aware Visual Narratives via Multi-Agent Cognitive Calibration](../../ACL2026/image_generation/bookagent_orchestrating_safety-aware_visual_narratives_via_multi-agent_cognitive.md)

</div>

<!-- RELATED:END -->

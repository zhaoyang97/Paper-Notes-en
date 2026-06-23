---
title: >-
  [Paper Note] MAC-AMP: A Closed-Loop Multi-Agent Collaboration System for Multi-Objective Antimicrobial Peptide Design
description: >-
  [ICLR 2026][Multi-Agent][LLM agent] Ours proposes MAC-AMP, the first closed-loop multi-agent collaboration system that reformulates antimicrobial peptide (AMP) design as a coordinated multi-agent optimization problem, achieving multi-objective optimization through AI-simulated peer review and adaptive reward design.
tags:
  - ICLR 2026
  - Multi-Agent
  - LLM agent
date: 2026-05-08
content_hash: 31568ab3d9f59256
---
# MAC-AMP: A Closed-Loop Multi-Agent Collaboration System for Multi-Objective Antimicrobial Peptide Design

**Conference**: ICLR 2026  
**arXiv**: [2602.14926](https://arxiv.org/abs/2602.14926)  
**Code**: [GitHub](https://github.com/CLMFAP/MAC-AMP_v1/)  
**Area**: Multi-Agent Systems  
**Keywords**: Antimicrobial Peptide Design, Multi-Agent Collaboration, Closed-Loop Reinforcement Learning, Multi-Objective Optimization, LLM agent

## TL;DR

Ours proposes MAC-AMP, the first closed-loop multi-agent collaboration system that reformulates antimicrobial peptide (AMP) design as a coordinated multi-agent optimization problem, achieving multi-objective optimization through AI-simulated peer review and adaptive reward design.

## Background & Motivation

- **Antimicrobial Resistance (AMR) Crisis**: Directly caused approximately 1.14 million deaths in 2021, with projections exceeding 39 million direct deaths between 2025 and 2050.
- **Limitations of Prior Work**:
    - Most models optimize only for antimicrobial activity while ignoring toxicity, stability, and novelty.
    - Multi-objective optimization is unstable; static weights often lead to reward hacking or diversity collapse.
    - Outputs are typically scattered scores or text, which are difficult to convert into reproducible learning signals.
- **Key Challenge in Multi-Agent Systems**:
    - Outputs are mostly natural language, lacking trainable optimization signals.
    - Most are open-loop systems relying on manual intervention.

## Method

### Overall Architecture

MAC-AMP models AMP design as a closed-loop optimization process: users provide target bacteria names and exemplar datasets, and the system iterates through "Property Prediction — AI Peer Review — Reward Redesign — PPO Training — Generation." The Mechanism utilizes a "Review Committee" composed of LLM agents to distill scattered property scores and natural language feedback into reward signals for reinforcement learning, eliminating reliance on manual heuristic weights. The pipeline starts with a generator producing candidate peptides, which are processed through property prediction, peer review, and RL refinement modules. Finally, PPO updates the generator using these rewards to close the loop.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}}%%
flowchart TD
    IN["Input: Target Bacteria Name<br/>+ Exemplar AMP Dataset"] --> GEN["Generator (GPT-2 + Soft Prompt)<br/>Produces Candidate Peptides"]
    GEN --> PP["Property Prediction Module<br/>Explicit Signal S (Activity / AMP Likelihood)<br/>+ Aux Evidence V (Toxicity / Structure / Physchem / Similarity)"]
    PP --> PR
    subgraph PR["AI Simulated Peer Review"]
        direction TB
        R["3 Reviewer Agents<br/>4-dimensional Structured Tagging"] --> AC["Area Chair Aggregation & Conflict Resolution<br/>→ Meta-review T + Consensus Score Sc"]
    end
    PR --> RL["RL Refinement Module<br/>CS Reward Design + Bio-alignment<br/>→ Rule Verification → Sandbox Pareto Reward Selection"]
    RL --> PPO["PPO Optimization<br/>Clipped Surrogate Loss Updates Generator"]
    PPO -->|Epoch Evaluation| PP
    PPO -->|Reward Redesign x3 (every 15 epochs)| RL
    PPO --> OUT["Output: Multi-Objective Optimized AMP"]
```

### Key Designs

**1. Property Prediction Module: Categorizing Multi-objectives into Reward Signals and Auxiliary Evidence**

To achieve multi-objective optimization, "peptide quality" must be quantified. MAC-AMP uses a suite of established tools to score candidate peptides, intentionally distinguishing between two roles. One category consists of explicit signals $S$ directly entering the reward: antimicrobial activity score $S_a$ (via a ProtBERT-finetuned MIC predictor) and AMP likelihood score $S_b$ (Macrel 1.5). The other category includes auxiliary evidence $V$ not directly used for rewards but interpreted by review agents: toxicity $V_a$ (ToxinPred 3.0), structural reliability $V_b$ (OmegaFold), physicochemical properties $V_c$ (ProtParam), and template similarity $V_d$ (Foldseek). This hierarchical design ensures hard constraints guide training while context-dependent properties like toxicity and stability are handled by the review stage, preventing reward hacking caused by static weighted sums.

**2. AI Simulated Peer Review: Converting Natural Language Feedback into Trainable Signals via Structured Tagging**

This is critical for bridging "text to reward." Three independent reviewer agents (GPT-5, Gemini 2.5, Perplexity) evaluate candidates across efficiency, safety, structural development, and originality. Each dimension uses a weighted dictionary sub-table, and feedback is output as structured tags in the format $\text{ID}(\text{State}, \text{Weight})$. This step compresses natural language into programmatically consumable fields. Subsequently, an Area Chair agent aggregates these tags, resolves conflicting judgments, calculates dimension-level meta-scores, and outputs a meta-review text $T$ and an average meta-score $S_c$. This "simulated peer review" bridges the gap between model output formats and training signals.

**3. RL Refinement Module: Multi-Agent Collaboration + Adaptive Redesign**

The reward function itself must evolve. MAC-AMP employs two types of agents: the CS Base Reward Design agent creates the reward function skeleton based on observable signals and mathematical properties, while the Biomedical Reward Alignment agent proposes revisions using domain knowledge from the meta-review $T$. Candidate rewards are filtered by a rule verifier, tested through short-term sandbox training, and finally, the best compromise across multiple objectives is selected via Pareto optimization. This process is stage-adaptive: the reward function is redesigned every 15 epochs for 3 iterations, allowing rewards to calibrate dynamically with training progress.

**4. PPO Optimization: Stabilizing Consensus Reward into Generation Strategy**

PPO maps the rewards back to the generator. Advantages are normalized $A = \text{norm}(R - \bar{V}_\phi)$ to stabilize gradient scales. Strategy updates utilize a clipped surrogate loss:

$$L_{policy}(\theta) = \mathbb{E}[\min(r(\theta)A, \text{clip}(r(\theta), 1-\epsilon, 1+\epsilon)A)]$$

the policy ratio $r(\theta)$ is restricted to the $1\pm\epsilon$ interval to prevent excessive updates that lead to diversity collapse. The total loss includes value regression loss $L_{value}$ and entropy regularization $L_{ent}$:

$$L = L_{policy} + c_v L_{value} - c_e L_{ent}$$

The entropy term with coefficient $c_e$ encourages exploration and maintains diversity, while the value term with coefficient $c_v$ calibrates value estimation. Combined with the rewards refreshed every 15 epochs, this closed-loop ensures continuous approach to multi-objective optimality without mode collapse.

## Key Experimental Results

### Main Results: Target-Specific AMP Testing

| Model | Antimicrobial Activity (↑) | AMP Likelihood (↑) | Toxicity (↓) | Structural Reliability (↑) |
|------|-------------|-------------|---------|-------------|
| **MAC-AMP** | **0.943±0.008** | 0.797±0.012 | **0.154±0.008** | **0.873±0.009** |
| AMP-Designer | 0.807±0.021 | 0.811±0.011 | 0.251±0.024 | 0.817±0.017 |
| BroadAMP-GPT | 0.831±0.025 | **0.821±0.018** | 0.246±0.033 | 0.763±0.023 |
| PepGAN | 0.823±0.023 | 0.572±0.035 | 0.247±0.064 | 0.637±0.026 |
| Diff-AMP | 0.822±0.006 | 0.554±0.036 | 0.235±0.072 | 0.752±0.020 |

*Target: E. coli results*

### Broad-Spectrum Activity Test

| Model | E. coli | S. aureus | P. aeruginosa | K. pneumoniae | E. faecium |
|------|---------|-----------|---------------|---------------|------------|
| **MAC-AMP** | **0.94** | 0.81 | **0.94** | **0.98** | 0.95 |
| AMP-Designer | 0.81 | 0.81 | 0.85 | 0.96 | 0.96 |
| PepGAN | 0.82 | **0.89** | 0.91 | 0.98 | 0.96 |

### Key Findings

1. MAC-AMP outperforms baselines across antimicrobial activity, toxicity, and structural reliability.
2. AMPs designed for *E. coli* show excellent generalization to other Gram-negative bacteria due to shared outer membrane structures.
3. Strong generalization was also observed for *E. faecium* (Gram-positive).
4. Training costs: 47.61 GPU hours, 853 API calls, total API cost $36.56.

## Highlights & Insights

1. **Novel Closed-Loop System**: Converts natural language review consensus into executable RL reward signals, bridging the gap between output formats and training signals.
2. **Full-Link Explainability**: Overcomes black-box limitations through transparent logs, replay trajectories, and consensus-aware decision tracking.
3. **Cross-Domain Transferability**: Validated the framework's versatility in English table-to-text generation tasks.
4. **Multi-Objective Balance**: Achieves multi-objective optimization through structured agent consensus rather than manual static weights.

## Limitations & Future Work

- Generated peptides have not yet undergone in-vitro experimental validation.
- API call costs may limit large-scale application.
- The peer review module relies on specific commercial LLMs, affecting reproducibility.
- Hyperparameters such as the number of epochs (15) and iterations (3) may require adjustment for different tasks.

## Related Work & Insights

- **AMP Generation**: AMPGAN v2, Diff-AMP, AMP Designer.
- **LLM Multi-Agent Collaboration**: Virtual Lab, CAMEL, AutoGen, ReviewAgents.
- **LLM-enhanced RL**: RLAIF, Eureka.

## Rating

- Novelty: ⭐⭐⭐⭐ — First framework to apply closed-loop multi-agent collaboration to molecular design.
- Technical Depth: ⭐⭐⭐⭐ — Sophisticated modular design with thorough multi-level validation.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Five bacterial targets, four baselines, and multi-dimensional ablation studies.
- Value: ⭐⭐⭐⭐ — Scalable to other molecular design tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Multi-Agent Design: Optimizing Agents with Better Prompts and Topologies](multi-agent_design_optimizing_agents_with_better_prompts_and_topologies.md)
- [\[CVPR 2025\] NADER: Neural Architecture Design via Multi-Agent Collaboration](../../CVPR2025/multi_agent/nader_neural_architecture_design_via_multi-agent_collaboration.md)
- [\[ICLR 2026\] PixelCraft: A Multi-Agent System for High-Fidelity Visual Reasoning on Structured Images](pixelcraft_a_multi-agent_system_for_high-fidelity_visual_reasoning_on_structured.md)
- [\[ICLR 2026\] From What to Why: A Multi-Agent System for Evidence-based Chemical Reaction Condition Reasoning](from_what_to_why_a_multi-agent_system_for_evidence-based_chemical_reaction_condi.md)
- [\[ICLR 2026\] AgentPO: Enhancing Multi-Agent Collaboration via Reinforcement Learning](agentpo_enhancing_multi-agent_collaboration_via_reinforcement_learning.md)

</div>

<!-- RELATED:END -->

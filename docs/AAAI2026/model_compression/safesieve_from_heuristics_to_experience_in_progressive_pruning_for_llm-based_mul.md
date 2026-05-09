---
title: >-
  [Paper Note] SafeSieve: From Heuristics to Experience in Progressive Pruning for LLM-based Multi-Agent Communication
description: >-
  [AAAI 2026][Model Compression][Multi-agent systems] SafeSieve is proposed as a progressive adaptive multi-agent communication pruning framework. Through a two-stage edge scoring mechanism combining semantic-heuristic initialization and history-feedback-driven refinement, together with 0-extension clustering, SafeSieve achieves 94.01% average accuracy across 6 benchmarks while reducing token consumption by 12.4%–27.8%, and demonstrates inherent robustness against prompt injection attacks.
tags:
  - AAAI 2026
  - Model Compression
  - Multi-agent systems
  - communication pruning
  - 0-extension clustering
  - LLM collaboration
  - adversarial robustness
date: 2026-05-08
content_hash: 7aecd78857bd161a
---

# SafeSieve: From Heuristics to Experience in Progressive Pruning for LLM-based Multi-Agent Communication

**Conference**: AAAI 2026
**arXiv**: [2508.11733](https://arxiv.org/abs/2508.11733)
**Code**: [https://github.com/csgen/SafeSieve](https://github.com/csgen/SafeSieve)
**Area**: Model Compression
**Keywords**: Multi-agent systems, communication pruning, 0-extension clustering, LLM collaboration, adversarial robustness

## TL;DR
SafeSieve is proposed as a progressive adaptive multi-agent communication pruning framework. Through a two-stage edge scoring mechanism combining semantic-heuristic initialization and history-feedback-driven refinement, together with 0-extension clustering, SafeSieve achieves 94.01% average accuracy across 6 benchmarks while reducing token consumption by 12.4%–27.8%, and demonstrates inherent robustness against prompt injection attacks.

## Background & Motivation

LLM-based multi-agent systems (MAS) have demonstrated strong collaborative problem-solving capabilities, yet dense round-robin dialogues introduce substantial token overhead and communication redundancy. This not only increases inference costs but also dilutes attention to critical information, leading to accuracy degradation. Longer context windows further expand the attack surface for prompt injection.

**Two dominant paradigms and their limitations**:

**Pre-design methods** (GPTSwarm, G-Designer): construct compressed graph topologies prior to execution
   - Strengths: improve communication efficiency at initialization
   - Weaknesses: limited generalizability; unable to adapt to runtime dynamics

**Post-prune methods** (AgentPrune, AgentDropout): start from a fully connected topology and iteratively prune edges based on task feedback
   - Strengths: require no pre-training; exhibit strong task adaptability
   - Weaknesses: rely on **greedy Top-k pruning**, which may erroneously remove critical communication paths and degrade system robustness

**Core gap**: No existing method unifies heuristic early-stage filtering with performance-aware dynamic adaptation — no complete "plan-then-refine" optimization pipeline exists.

**Key Insight**: Drawing an analogy to human team organization — initial role assignments are made based on member capabilities and complementarity (heuristic), and collaboration relationships are progressively adjusted based on actual performance (experience-driven). 0-extension clustering replaces greedy Top-k pruning to maintain structural coherence.

## Method

### Overall Architecture

SafeSieve is a progressive two-stage pruning framework consisting of:
1. Heuristic initialization based on semantic compatibility (establishing the initial communication graph)
2. Experience-driven refinement based on historical contributions (dynamically adjusting edge weights and pruning)
3. Structured pruning decisions driven by 0-extension clustering

The core data structure is a dynamic edge scoring matrix $E \in \mathbb{R}^{n \times n}$.

### Key Designs

#### 1. **Semantic-Heuristic Initialization**
- **Function**: Prior to task execution, initialize communication edge scores based on semantic information derived from agent roles
- **Core formula**:
$$S_{ij}^{compat} = \gamma \cdot \frac{\mathbf{e}_i \cdot \mathbf{e}_j}{\|\mathbf{e}_i\| \cdot \|\mathbf{e}_j\|} + (1-\gamma) \cdot \mathcal{Q}(S_{ij}^{expert})$$
  where $\mathbf{e}_i, \mathbf{e}_j$ are pre-trained role embeddings, $S_{ij}^{expert}$ is a functional complementarity score evaluated by an expert LLM, and $\mathcal{Q}(\cdot)$ is a 5-level quantization function
- **Design Motivation**: Semantic similarity facilitates basic cooperation, while functional complementarity is more critical for complex multi-hop reasoning tasks. Combining both provides a reasonable communication structure during the startup phase

#### 2. **History-Feedback-Driven Progressive Pruning**
- **Historical complementarity score**: Tracks the contribution of each edge toward correct answers:
$$C_{ij}^{hist}(t) = \frac{\sum_{\tau=1}^{t} \mathbf{1}_{ij}^{correct}(\tau)}{\sum_{(k,l) \in E_t} \sum_{\tau=1}^{t} \mathbf{1}_{kl}^{correct}(\tau) + n^2 \varepsilon}$$
- **Fused edge score** (smooth transition from heuristics to experience):
$$E_{ij}(t) = \left(1 - \frac{t}{T}\right) \cdot \alpha_0 \cdot S_{ij}^{compat} + \left[\beta_0 + (\beta_{max} - \beta_0) \cdot \frac{t}{T}\right] \cdot C_{ij}^{hist}(t)$$
  The semantic weight decays over time while the historical contribution weight grows progressively
- **Design Motivation**: Emulates the human team "plan-then-adjust" paradigm. During early stages when information is scarce, the framework relies on semantic heuristics; as experience accumulates, it transitions toward data-driven decision-making

#### 3. **0-Extension Clustering Pruning**
- **Function**: Replaces greedy Top-k pruning, providing globally structured pruning decisions
- **Dynamic threshold** (from conservative to aggressive):
$$\theta(t) = \theta_0 + (\theta_{max} - \theta_0) \cdot [1 - \exp^{-k \cdot \max(t/T, 0)}]$$
- **Terminal selection** (cluster centers):
$$T = \arg\max_{S \subseteq V, |S|=|T|} \sum_{v \in S} \sum_{u \in V} \frac{1}{(E_{vu}(t) + \varepsilon)^{-1}}$$
  Terminal count is adaptive: $|T| = \max(2, \min(\sqrt{n}, \lfloor n/3 \rfloor))$
- **0-extension cluster assignment**:
$$f^* = \arg\min_{f:V \to T} \sum_{(i,j) \in E} (E_{ij}(t) + \varepsilon)^{-1} \cdot \mathbf{1}\{f(i) \neq f(j)\}$$
  Minimizes low-weight edge cuts across cluster boundaries
- **Structured pruning**: Prioritizes removal of cross-cluster edges below the threshold, supplemented by lowest-scoring edges when needed; isolated nodes are automatically removed post-pruning (with a minimum of 2 nodes retained)
- **Design Motivation**: 0-extension offers $O(n \log n)$ complexity with theoretical approximation guarantees, achieving sparsification while preserving graph connectivity. It avoids the locally suboptimal behavior of greedy Top-k and preserves inter-agent complementarity

### Loss & Training

- **No GPU training required**: SafeSieve operates entirely on runtime scoring and clustering, with no gradient optimization
- **Warm-up period**: No pruning is performed during the first $B_{start}$ steps to accumulate sufficient historical information
- **Maximum pruning rate constraint**: $\mathcal{R}(t) < R_{max}$ prevents over-pruning
- **Post-pruning regularization**: Edge scores are normalized and historical weights are adjusted after each pruning step:
$$\hat{E}_{ij}(t) = \frac{E_{ij}(t) - \mu_t}{\sigma_t + \varepsilon}, \quad \hat{\beta}(t) = \beta(t) \cdot \frac{\Delta_{before}}{\Delta_{after} + \varepsilon}$$

## Key Experimental Results

### Main Results (DeepSeek-V3-671B)

| Method | Paradigm | MMLU | GSM8K | SVAMP | HumanEval | AQuA | MATH-500 | Avg |
|--------|----------|------|-------|-------|-----------|------|----------|-----|
| Vanilla | single | 87.97 | 94.68 | 93.67 | 88.43 | 84.58 | 88.20 | 89.59 |
| CoT | single | 89.31 | 95.15 | 93.94 | 89.26 | 85.42 | 90.41 | 90.58 |
| G-Designer | pre-design | 91.13 | 95.47 | 93.79 | 90.93 | 89.63 | 91.02 | 92.00 |
| AgentPrune | post-prune | 90.99 | 95.30 | 95.40 | 92.91 | 90.30 | 91.76 | 92.78 |
| AgentDropout | post-prune | 90.17 | 95.16 | 96.01 | 93.16 | 91.37 | 89.82 | 92.62 |
| **SafeSieve** | post-prune | **92.39** | **96.27** | **96.60** | **95.01** | **91.89** | **91.90** | **94.01** |

### Ablation Study (HumanEval)

| Configuration | Accuracy | Token Reduction | Notes |
|---------------|----------|-----------------|-------|
| Fully connected (no pruning) | 95.50% | — | Upper bound |
| Clustering pruning w/o heuristics | 94.41% | 24.2% | Missing initial guidance |
| Clustering pruning w/o history | 93.78% | 30.0% | Missing experience feedback |
| Dual scoring + Top-k pruning | 93.13% | 29.3% | Greedy pruning harms structure |
| **SafeSieve** | **95.01%** | **27.8%** | Optimal three-component synergy |

### Safety (Accuracy Drop under Prompt Injection Attacks)

| Method | MMLU↓ | SVAMP↓ | HumanEval↓ | Avg↓ | Drop Rate |
|--------|-------|--------|-----------|------|-----------|
| AgentPrune | -4.99 | -3.80 | -4.97 | -4.59 | 5.14% |
| AgentDropout | -1.67 | -2.81 | -2.16 | -2.21 | 2.40% |
| **SafeSieve** | **-1.19** | **-1.60** | **-0.91** | **-1.23** | **1.33%** |

### Key Findings

1. **Post-prune paradigm consistently outperforms pre-design**: AgentPrune (92.78%), AgentDropout (92.62%), and SafeSieve (94.01%) all surpass GPTSwarm (91.15%) and G-Designer (92.00%)
2. **Task-differentiated gains**: Improvements are more pronounced on complex collaborative tasks — HumanEval +6.58 pts, AQuA +7.31 pts vs. only +1.59 pts on GSM8K
3. **Triple defense mechanism**: Preventive defense (down-weighting suspicious agents) → Responsive defense (identifying malicious agents within 30 batches) → Structural defense (clustering maintains connectivity), with accuracy fluctuation under attack remaining below 3%
4. **Value of heterogeneous deployment**: In a 1+4 collaboration mode (DeepSeek-V3 as coordinator + 4 smaller models as executors), cost is reduced by 13.3%, with SVAMP accuracy marginally exceeding the homogeneous configuration (+0.17 pts)
5. **0-extension vs. Top-k**: Replacing 0-extension with Top-k reduces accuracy from 95.01% to 93.13% (−1.88 pts), demonstrating the superiority of structure-aware pruning
6. **Complementarity of large and small models**: Large models show greater advantages on complex tasks (MMLU +4.42), while small models are more efficient on structured tasks (SVAMP +5.7%)

## Highlights & Insights

1. **Unified progressive framework**: The first post-pruning framework to integrate heuristic early evaluation with experience-driven refinement, addressing a significant gap in MAS communication optimization
2. **GPU-free sparsification**: A purely scoring- and clustering-based algorithm with $O(n \log n)$ complexity, imposing minimal deployment overhead
3. **Built-in safety**: 0-extension clustering inherently detects and isolates malicious agents — low-contributing agents are naturally marginalized during clustering
4. **Pioneering exploration of heterogeneous deployment**: The first systematic analysis of cross-model collaboration, revealing a "weakest-link effect" in knowledge-intensive tasks
5. **Design analogy to human team management**: Semantic evaluation ≈ interview stage; historical feedback ≈ performance review; 0-extension pruning ≈ team restructuring

## Limitations & Future Work

1. Semantic compatibility scoring depends on the evaluation quality of the expert LLM, introducing additional API call costs
2. The framework involves numerous hyperparameters ($\gamma, \alpha_0, \beta_0, \beta_{max}, \theta_0, \theta_{max}, k, r$, etc.)
3. Validation is currently limited to reasoning tasks; generative tasks (e.g., creative writing, long-form text generation) remain unexplored
4. Mitigation strategies for the "weakest-link effect" in heterogeneous deployment have not been thoroughly investigated
5. The approximate solution of 0-extension clustering may introduce suboptimal pruning decisions

## Related Work & Insights

- **GPTSwarm (Zhuge et al., 2024)**: The first work to model MAS as a differentiable computation graph, but produces static topologies
- **AgentPrune (Zhang et al., 2024)**: Introduces one-hot mask matrices for dynamic edge pruning
- **AgentDropout (Wang et al., 2025)**: Extends pruning to the node level with real-time feedback
- **G-Designer (Zhang et al., 2024)**: GNN-based pre-designed graph construction
- **0-extension (Fakcharoenphol et al., 2003)**: A classical graph clustering algorithm with $O(n \log n)$ complexity and strong connectivity guarantees
- **MetaGPT (Hong et al., 2023)**: An empirical study demonstrating human-like collaborative patterns in LLMs

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] MAGEO: From Experience to Skill — Multi-Agent Generative Engine Optimization via Reusable Strategy Learning](../../ACL2026/model_compression/from_experience_to_skill_multi-agent_generative_engine_optimization_via_reusable.md)
- [\[AAAI 2026\] AgentODRL: A Large Language Model-based Multi-agent System for ODRL Generation](agentodrl_a_large_language_model-based_multi-agent_system_fo.md)
- [\[AAAI 2026\] Hierarchical Pedagogical Oversight: A Multi-Agent Adversarial Framework for Reliable AI Tutoring](hierarchical_pedagogical_oversight_a_multi-agent_adversarial_framework_for_relia.md)
- [\[AAAI 2026\] InfoCom: Kilobyte-Scale Communication-Efficient Collaborative Perception with Information-Aware Feature Compression](infocom_kilobyte-scale_communication-efficient_collaborative_perception_with_inf.md)
- [\[ICLR 2026\] Rethinking Continual Learning with Progressive Neural Collapse](../../ICLR2026/model_compression/rethinking_continual_learning_with_progressive_neural_collapse.md)

<!-- RELATED:END -->

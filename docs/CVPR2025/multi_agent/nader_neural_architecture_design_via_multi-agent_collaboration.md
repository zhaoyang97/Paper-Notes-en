---
title: >-
  [Paper Note] NADER: Neural Architecture Design via Multi-Agent Collaboration
description: >-
  [CVPR 2025][Multi-Agent][Neural Architecture Design] NADER models neural architecture design as a multi-LLM-agent collaborative task: a Reader extracts knowledge from papers, a Proposer generates improvement plans, a Modifier implements modifications using Directed Acyclic Graphs (DAGs), and a Reflector learns from failures. With only 10 trials, it surpasses the accuracy upper bound of the NAS-Bench-201 search space, achieving 74.51% on CIFAR-100 (compared to the best in-spac…
tags:
  - "CVPR 2025"
  - "Multi-Agent"
  - "Neural Architecture Design"
  - "Multi-Agent Collaboration"
  - "LLM-driven NAS"
  - "DAG Representation"
  - "Experience Learning"
date: 2026-05-08
content_hash: 01fe53d1b672522b
---

# NADER: Neural Architecture Design via Multi-Agent Collaboration

**Conference**: CVPR 2025  
**arXiv**: [2412.19206](https://arxiv.org/abs/2412.19206)  
**Code**: TBD  
**Area**: LLM Evaluation  
**Keywords**: Neural Architecture Design, Multi-Agent Collaboration, LLM-driven NAS, DAG Representation, Experience Learning

## TL;DR
NADER models neural architecture design as a multi-LLM-agent collaborative task: a Reader extracts knowledge from papers, a Proposer generates improvement plans, a Modifier implements modifications using Directed Acyclic Graphs (DAGs), and a Reflector learns from failures. With only 10 trials, it surpasses the accuracy upper bound of the NAS-Bench-201 search space, achieving 74.51% on CIFAR-100 (compared to the best in-space search result of 73.51%).

## Background & Motivation

**Background**: Neural Architecture Search (NAS) has long been restricted to pre-defined search spaces. Recent works have attempted to use LLMs to generate code for architecture design (e.g., LeMo-NADe, GENIUS), but the generated code is often prone to errors, non-executable, and highly inefficient.

**Limitations of Prior Work**: (1) Traditional NAS is bounded by its pre-defined search space, failing to discover superior architectures outside of it; (2) Direct code generation by LLMs suffers from many issues: redundant code leads to token waste, attention is diverted to implementation details rather than architectural logic, and code is frequently non-executable; (3) LLMs tend to make the same design mistakes repeatedly, lacking a mechanism to learn from failures; (4) Existing LLM-based NAS methods (such as LeMo-NADe) exhibit poor performance (only 31% on ImageNet16-120 compared to the in-space optimum of 47%).

**Key Challenge**: LLMs possess rich architectural knowledge (from scientific literature) but lack reliable execution capabilities to translate this knowledge into valid architectures and systematically refine them.

**Goal**: How to enable multiple LLM agents to collaborate systematically to design neural architectures that surpass the limitations of manually designed search spaces?

**Key Insight**: Emulate the workflow of a real AI research team, where members divide labor and iterate collaboratively: reading literature (Reader), proposing designs (Proposer), writing code (Modifier), and conducting reviews and debugging (Reflector).

**Core Idea**: Simulate an AI research team using four LLM agents: a Reader reads papers to extract innovations, a Proposer proposes improvements, a Modifier implements architectural changes via DAGs, and a Reflector learns from historical successes and failures to guide subsequent designs.

## Method

### Overall Architecture
The framework is divided into a research team (Reader + Proposer) and a development team (Modifier + Reflector). The iterative workflow is as follows: The Reader reads papers and extracts innovations into a knowledge database $\mathcal{K}$. The Proposer selects a candidate node from the network modification tree, retrieves relevant advice from $\mathcal{K}$, and generates improvement plans. The Modifier implements the modifications in DAG form. The Reflector validates the DAG and retrieves experiences from an experience database $\mathcal{E}$ to assist in debugging. After training and evaluation, the modification tree is updated, and the process repeats.

### Key Designs

1. **DAG Representation (Instead of Code)**:

    - **Function**: Represents neural network architectures as concise directed acyclic graphs (DAGs) instead of verbose code.
    - **Mechanism**: Nodes represent operations (e.g., convolution, normalization) and edges represent information flow. LLM agents operate on the graph space (adding, deleting, or replacing nodes and edges). Advantages of DAG representation include: reduces token consumption by 74-77%, enables isomorphism detection to avoid redundant training, and allows agents to focus on architecture logic rather than coding implementation details. Finally, a Graph-to-Code translation module translates the valid graph into executable code.
    - **Design Motivation**: Ablation experiments show that the DAG representation improves design Quality from 0.63 to 0.78 (macro-level), while reducing token consumption from 2.23K to 0.58K, demonstrating that removing code details significantly enhances the LLM's architecture design capabilities.

2. **Reflector's Dual Learning Mechanism**:

    - **Function**: Learns from instant feedback and historical experience to avoid repeating errors.
    - **Mechanism**: (a) Learn from Immediate Feedback (LIF): A computational graph flow verification tool checks whether the generated architecture is valid, and feeds error messages back to the Modifier for retry in case of failure. (b) Learn from Design Experience (LDE): Maintains a design experience database $\mathcal{E}$ recording three types of experiences: failure records, failure-to-success repair strategies, and success records. Before each new design, it retrieves the 5 most relevant historical experiences to assist the Modifier.
    - **Design Motivation**: LIF increases durability/executability from 54% to 64% (macro-level), while LDE increases the success rate from 0.62 to 0.88 (micro-level). These show extremely significant improvements, demonstrating that experience accumulation makes the system improve continuously over time.

3. **Network Modification Tree and Knowledge Database**:

    - **Function**: Systematically manages the exploration-exploitation balance in architecture search and organizes external knowledge sources.
    - **Mechanism**: The modification tree uses a base network as the root. Each iteration selects candidate nodes from the tree (combining DFS and BFS, prioritizing high-performance nodes) and retrieves relevant architectural innovations from the paper knowledge database $\mathcal{K}$ as suggestions. The Reader automatically downloads evaluated papers and extracts methodological innovations using LLMs for storage in $\mathcal{K}$.
    - **Design Motivation**: The tree structure avoids blind searching, and the knowledge database integrates the latest advances in human research. Ablation results show that the combination of Reader and Proposer yields a 1.58% improvement on the challenging ImageNet16-120 dataset.

### Loss & Training
The architecture design process does not involve training loss; it is entirely driven by LLM prompts. Discovered architectures are trained and evaluated under standard settings (following the train/validation/test split of NAS-Bench-201). Constraints: FLOPs $\le 0.2\text{G}$ (CIFAR) / $\le 0.05\text{G}$ (ImageNet16-120), parameters $\le 1.5\text{M}$. The cost per architecture design is around \$0.046.

## Key Experimental Results

### Main Results

| Method | Trials | CIFAR-10 Test | CIFAR-100 Test | ImageNet16-120 Test |
|--------|------|------|----------|------|
| In-space Optimum | - | 94.37 | 73.51 | 47.31 |
| LeMo-NADe (GPT-4) | 30 | 89.41 | 67.90 | 27.70 |
| GENIUS | 10 | 93.79 | 70.91 | 44.96 |
| LLMatic | 2000 | 94.26 | 71.62 | 45.87 |
| **NADER (Random, 10)** | **10** | **94.40** | **74.51** | **49.63** |
| **NADER (ResNet, 500)** | **500** | **94.62** | **76.00** | **50.52** |

### Ablation Study (NAD Benchmark, Micro-level)

| Configuration | Token (K) | Executability | Quality | Success Rate |
|------|---------|---------|---------|---------|
| Baseline (Code-only) | 2.10 | 0.76 | 0.65 | 0.49 |
| + Graph Rep. | 0.49 | 0.62 | 0.97 | 0.60 |
| + LIF | 0.48 | 0.70 | 0.89 | 0.62 |
| + LDE | **0.31** | **0.92** | **0.96** | **0.88** |

### Key Findings
- **Shattering the limits of the search space**: With only 10 trials, NADER achieves 74.51% on CIFAR-100, outperforming the in-space optimum of 73.51% ($+1.0\%$), and 49.63% on ImageNet16-120, outperforming 47.31% ($+2.32\%$). The margins increase even further at 500 trials ($+2.49\%$ / $+3.21\%$).
- **Graph representation is fundamental**: Token usage is reduced by 77% (2.10K to 0.49K), and quality improves from 0.65 to 0.97, proving that raw code details severely distract the LLM's architecture design capabilities.
- **LDE experience learning yields the most significant effect**: The micro-level success rate doubles (0.49 to 0.88), and the executability rate rises from 0.70 to 0.92. The accumulation of historical experience is the key to continuous self-improvement.
- **Both Reader and Proposer are indispensable**: Using either Reader or Proposer alone is inferior to combining them (especially on ImageNet16-120 by a difference of 0.8%), indicating that paper-derived knowledge and search strategy must act in synergy.

## Highlights & Insights
- **"AI Research Team" Agent Collaboration Paradigm**: Deconstructing the architecture design pipeline into four specialized agent roles (reading papers, proposing plans, implementing, and reflecting) simulates a real research workflow. This paradigm can be generalized to other engineering problems that require creative design.
- **Graph/DAG Representation Instead of Code as a Key Insight**: Operating in a highly abstract graph space rather than writing verbose code substantially enhances efficiency and quality. This observation offers strong inspiration for all works utilizing LLMs for structural code generation—perhaps LLMs should operate on higher-level representations.
- **Accumulative Power of the Experience Database**: As design experiences accumulate, system performance steadily rises, showing characteristics akin to continual learning. This is a qualitative leap over traditional single-shot prompting or simple chain-of-thought LLM tasks.

## Limitations & Future Work
- To ensure fair comparison, experiments are constrained by FLOPs and parameter counts, limiting the freedom of architectural innovation.
- Only validated on image classification (CIFAR-10/100, ImageNet16-120), without extension to tasks like object detection and segmentation.
- Highly dependent on GPT-4 capabilities; the quality of the agents is constrained by the performance upper bound of the underlying LLM.
- The scope and quality of the Reader's literature review depend heavily on the retrieved paper list, which might miss emerging research directions.
- Each architecture still requires actual training and evaluation (albeit at a lower cost than traditional NAS); zero-shot architecture quality estimation without training has not yet been realized.

## Related Work & Insights
- **vs GENIUS**: GENIUS also uses LLMs for NAS but operates strictly within a pre-defined search space. NADER breaks through search space limitations, exceeding GENIUS by 4.67% on ImageNet16-120 with only 10 trials.
- **vs LeMo-NADe**: LeMo-NADe directly generates code using GPT-4/Gemini, yielding poor performance (only 27.70%/31.02% on ImageNet16-120). NADER's DAG representation and multi-agent collaboration are significantly superior.
- **vs LLMatic**: LLMatic requires 2000 trials to approach the search space optimum, whereas NADER surpasses it within 10 trials, achieving a 200x efficiency boost.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The paradigm of multi-agent collaboration for architecture design is highly novel, and both the DAG representation and experience learning designs are elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ The comparison on NAS-Bench-201 is comprehensive with detailed ablations, though the dataset scope remains somewhat limited.
- Writing Quality: ⭐⭐⭐⭐ The framework and agent roles are described clearly.
- Value: ⭐⭐⭐⭐ It presents the first multi-agent collaborative system that breaches search space constraints, offering fresh insights for both AI4Science and AutoML.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] MAC-AMP: A Closed-Loop Multi-Agent Collaboration System for Multi-Objective Antimicrobial Peptide Design](../../ICLR2026/multi_agent/mac-amp_a_closed-loop_multi-agent_collaboration_system_for_multi-objective_antim.md)
- [\[CVPR 2025\] Collaborative Tree Search for Enhancing Embodied Multi-Agent Collaboration](collaborative_tree_search_for_enhancing_embodied_multi-agent_collaboration.md)
- [\[NeurIPS 2025\] Multi-Agent Collaboration via Evolving Orchestration](../../NeurIPS2025/multi_agent/multi-agent_collaboration_via_evolving_orchestration.md)
- [\[ICLR 2026\] Multi-Agent Design: Optimizing Agents with Better Prompts and Topologies](../../ICLR2026/multi_agent/multi-agent_design_optimizing_agents_with_better_prompts_and_topologies.md)
- [\[ACL 2025\] Multi-Agent Collaboration via Cross-Team Orchestration](../../ACL2025/multi_agent/multi-agent_collaboration_via_cross-team_orchestration.md)

</div>

<!-- RELATED:END -->

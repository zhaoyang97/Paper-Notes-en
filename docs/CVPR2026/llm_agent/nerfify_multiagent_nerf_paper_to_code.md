---
title: >-
  [Paper Note] Nerfify: A Multi-Agent Framework for Turning NeRF Papers into Code
description: >-
  [CVPR 2026][LLM Agent][NeRF] Nerfify is proposed, a domain-aware multi-agent framework that automatically converts NeRF papers into trainable Nerfstudio plugin code via context-free grammar (CFG) constraints…
tags:
  - "CVPR 2026"
  - "LLM Agent"
  - "NeRF"
  - "paper-to-code"
  - "multi-agent"
  - "code synthesis"
  - "Nerfstudio"
date: 2026-05-08
content_hash: d11f3b844cf1000c
---

# Nerfify: A Multi-Agent Framework for Turning NeRF Papers into Code

**Conference**: CVPR 2026
**arXiv**: [2603.00805](https://arxiv.org/abs/2603.00805)  
**Code**: To be open-sourced  
**Area**: LLM Agent
**Keywords**: NeRF, paper-to-code, multi-agent, code synthesis, Nerfstudio

## TL;DR
Nerfify is proposed, a domain-aware multi-agent framework that automatically converts NeRF papers into trainable Nerfstudio plugin code via context-free grammar (CFG) constraints, Graph-of-Thought (GoT) code synthesis, and compositional reference dependency recovery, achieving 100% executability with visual quality within ±0.5 dB PSNR of expert implementations.

## Background & Motivation
**Background**: Since its publication in 2020, NeRF has generated over 1,000 follow-up works, yet most papers lack public code or standardized implementations, requiring weeks of effort to re-implement prior methods for each subsequent work.

**Limitations of Prior Work**: General-purpose paper-to-code systems (Paper2Code, AutoP2C) fail almost entirely on NeRF—producing trainable code in only 5% of cases. GPT-5 in single-pass generation mode achieves only 26.6% accuracy on complex papers. NeRF implementations span the intersection of volumetric rendering, computer vision, and neural optimization, where a single incorrect activation function or ray-sphere intersection can cause NaN gradients or degenerate solutions.

**Key Challenge**: General-purpose methods lack domain knowledge and cannot handle NeRF's implicit dependency chains (e.g., "we adopt the distillation loss from [3]" requires traversing references, locating the correct equation, translating it to code, and implementing stop-gradient), nor can they satisfy the modular composition constraints of the Nerfstudio framework.

**Goal**: To automatically convert NeRF papers into trainable, convergent, high-quality standardized Nerfstudio code, reducing the turnaround time from weeks to minutes.

**Key Insight**: Formalizing the Nerfstudio architecture as a context-free grammar (CFG), using domain constraints to guide LLM code synthesis, and employing multi-agent collaboration to resolve dependency chains and iterate via visual feedback.

**Core Idea**: Domain awareness through CFG constraints, reference dependency recovery, GoT synthesis, and visual feedback transforms NeRF paper-to-code from infeasible to high-quality automation.

## Method

### Overall Architecture
Nerfify employs a four-stage pipeline: (1) CFG formalization and in-context learning—parsing the paper PDF into structured markdown and constructing a domain knowledge base $\mathcal{K}$; (2) compositional dependency resolution—traversing the citation graph to recursively retrieve key components from cited papers; (3) grammar-guided repository code generation—GoT multi-agent synthesis of multi-file code in topological order; (4) visually-driven feedback—rendering images after training, then iteratively repairing code via PSNR analysis and VLM diagnosis.

### Key Designs

1. **Context-Free Grammar (CFG)-Constrained Synthesis**:

    - Function: Ensures generated code satisfies Nerfstudio's architectural invariants and interface contracts.
    - Mechanism: The modular composition patterns of Nerfstudio are formalized as a CFG; LLM code generation is hard-constrained by grammar rules, guaranteeing architectural correctness. MinerU is used to convert paper PDFs to markdown; after cleaning, equations, pseudocode, architecture diagrams, and citation relations are retained and paired with corresponding Nerfstudio implementations to populate the knowledge base $\mathcal{K}$ and in-context example library $\mathcal{X}$.
    - Design Motivation: General-purpose code generation treats all frameworks uniformly and does not understand the config→datamanager→field→model→pipeline coupling chain in NeRF. CFG encodes framework priors and eliminates architectural-level errors at the source.

2. **Compositional Reference Dependency Recovery**:

    - Function: Automatically retrieves and integrates implicit components (samplers, encoders, loss functions, etc.) from a paper's citation chain.
    - Mechanism: A reference dependency graph $G' = (V', E')$ is constructed, and multi-hop retrieval is executed iteratively in four steps: (a) dependency discovery—parsing the target paper to extract citation lists and borrowed components; (b) recursive resolution—$\text{Dependencies}(c_i) = \{c_i\} \cup \bigcup_{d \in \text{cited}(c_i)} \text{Dependencies}(d)$; (c) component extraction—extracting architectural modules, loss functions, and training protocols; (d) termination—all interface contracts are satisfied. For example, K-Planes requires extracting components from 7 direct references and 12 transitive dependencies.
    - Design Motivation: NeRF papers are inherently compositional. Descriptions such as "we adopt the proposal network from [3]" require the system to automatically trace multi-level citations and extract precise implementations.

3. **Graph-of-Thought (GoT) Multi-Agent Code Synthesis**:

    - Function: Generates multi-file repositories in topological dependency order, verifying type signatures, tensor shapes, and circular dependencies.
    - Mechanism: The primary synthesis agent maps the paper to a Nerfstudio component dependency DAG and executes synthesis in four phases: DAG construction → interface freezing (establishing minimal shared APIs in topological order) → implementation (each node synthesizes and verifies code) → integration testing (end-to-end smoke test with automatic repair). The repository is defined as $\mathcal{C} = (F, G)$, where $G = \text{BuildRepoDAG}(F)$ is a directed acyclic graph and $(f_i, f_j) \in E(G)$ implies no path exists from $f_j$ to $f_i$.
    - Design Motivation: Monolithic code generation cannot handle the coupling between files in a multi-file repository. Graph-of-Thought is better suited than CoT/ToT for dependency-aware generation at the repository level.

### Visually-Driven Feedback
In Stage 4, the generated code undergoes 3k-iteration smoke training; rendered images from multiple viewpoints are passed to a critic agent. The critic agent operates through three branches: (1) metric branch—computing local-window PSNR/SSIM maps and using morphological operations to localize maximum-error regions; (2) geometry branch—cross-view artifact consensus detection to identify floaters and ghosting; (3) semantic branch—leveraging the Qwen3 VLM to analyze artifact triplets and produce structured diagnoses and code patches. The feedback loop continues until no new feedback is produced, the maximum number of iterations is reached, or the paper-reported PSNR target is achieved.

## Key Experimental Results

### Main Results
Nerfify-Bench, 30 papers; Set 1 (papers without public code, compared against expert human implementations):

| Paper | Paper PSNR/SSIM | Expert PSNR/SSIM | Nerfify PSNR/SSIM |
|-------|-----------------|------------------|-------------------|
| KeyNeRF | 25.65/0.89 | 25.70/0.89 | 26.12/0.90 |
| mi-MLP NeRF | 24.70/0.89 | 22.64/0.87 | 22.85/0.87 |
| ERS | 27.85/0.94 | 26.87/0.90 | 27.02/0.90 |
| TVNeRF | 27.44/0.93 | 26.81/0.92 | 27.30/0.92 |

Executability comparison (all baselines fail to produce trainable code):

| Metric | Paper2Code | AutoP2C | GPT-5 | R1 | Nerfify |
|--------|-----------|---------|-------|-----|---------|
| Compilable / Trainable | ✗ | ✗ | ✗ | ✗ | ✓ |
| Training Stability | ✗ | ✗ | ✗ | ✗ | ✓ |
| Converges to Paper Results | ✗ | ✗ | ✗ | ✗ | ✓ |

### Ablation Study (Novelty Preservation, Set 4, Score↑)

| Paper | Nerfify | GPT-5 | Paper2Code | AutoP2C |
|-------|---------|-------|-----------|---------|
| Mip-NeRF | 1.00 | 0.58 | 0.85 | 0.20 |
| BioNeRF | 1.00 | 0.82 | 0.35 | 0.15 |
| TensoRF | 0.98 | 0.72 | 0.12 | 0.28 |
| Tetra-NeRF | 1.00 | 0.58 | 0.22 | 0.08 |
| E-NeRF | 1.00 | 0.60 | 0.48 | 0.05 |

### Key Findings
- General-purpose paper-to-code systems fail to produce trainable code for 95% of NeRF papers; Nerfify achieves 100% executability.
- Visual quality on average falls within ±0.5 dB PSNR and ±0.02 SSIM of expert implementations.
- For papers already integrated into Nerfstudio (NeRF, Nerfacto), Nerfify generates code identical to the official implementation.
- Nerfify substantially outperforms all baselines in novelty preservation (correctly implementing a paper's core contributions).

## Highlights & Insights
- Formalizing the framework as a CFG fundamentally transforms "understanding the framework" into "following a grammar," reducing the difficulty of LLM generation.
- Compositional reference dependency recovery addresses the long-standing challenge of critical implementation details being buried in citation chains within academic papers.
- The visual feedback loop is the first to incorporate a VLM into automated NeRF code debugging; the three-branch design covers pixel-level, geometry-level, and semantic-level diagnosis.
- The experimental design is rigorous: Set 1 uses papers without public code, eliminating the possibility of LLM training data leakage.

## Limitations & Future Work
- Only the Nerfstudio framework is supported; emerging paradigms such as 3DGS and gsplat are not covered.
- The CFG must be constructed manually; extending support to new frameworks requires additional engineering effort.
- Visual feedback requires 3k iterations of training, incurring non-negligible computational cost.
- Although the paper claims "minute-level" turnaround, the end-to-end time including smoke training may be considerably longer.

## Related Work & Insights
- Paper2Code and AutoP2C demonstrate the ceiling of general-purpose approaches on complex visual systems, underscoring the necessity of domain awareness.
- Graph-of-Thought provides a more flexible DAG structure than chain-based or tree-based reasoning, well-suited for repository-level code generation.
- The proposed paradigm is transferable to paper-to-code in other domains (robotics, NLP, medical imaging); the key is designing a corresponding domain-specific CFG.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — First domain-specific paper-to-code system to combine CFG constraints, GoT synthesis, and reference dependency recovery.
- Experimental Thoroughness: ⭐⭐⭐⭐ — A comprehensive benchmark of 30 papers with a well-designed four-set grouping scheme.
- Writing Quality: ⭐⭐⭐⭐ — Clear logic and strong problem motivation, though the paper is lengthy.
- Value: ⭐⭐⭐⭐⭐ — Significant impact for the NeRF community and reproducible research more broadly.

---

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Think, Then Verify: A Hypothesis-Verification Multi-Agent Framework for Long Video Understanding](think_then_verify_a_hypothesis-verification_multi-agent_framework_for_long_video.md)
- [\[CVPR 2026\] CarePilot: A Multi-Agent Framework for Long-Horizon Computer Task Automation in Healthcare](carepilot_a_multi-agent_framework_for_long-horizon_computer_task_automation_in_h.md)
- [\[CVPR 2026\] REALM: An MLLM-Agent Framework for Open World 3D Reasoning Segmentation and Editing on Gaussian Splatting](realm_mllm_agent_3d_reasoning_gaussian.md)
- [\[NeurIPS 2025\] AgentChangeBench: A Multi-Dimensional Evaluation Framework for Goal-Shift Robustness](../../NeurIPS2025/llm_agent/agentchangebench_a_multi-dimensional_evaluation_framework_for_goal-shift_robustn.md)
- [\[CVPR 2026\] ARGOS: Who, Where, and When in Agentic Multi-Camera Person Search](argos_agentic_multi_camera_person_search.md)

</div>

<!-- RELATED:END -->

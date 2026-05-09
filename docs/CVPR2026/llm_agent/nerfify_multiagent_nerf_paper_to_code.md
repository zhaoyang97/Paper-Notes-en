---
title: >-
  [Paper Note] Nerfify: A Multi-Agent Framework for Turning NeRF Papers into Code
description: >-
  [CVPR 2026][LLM Agent][NeRF] Nerfify is proposed, a multi-agent framework that automatically converts NeRF papers into trainable Nerfstudio plugin code via context-free grammar constraints, Graph-of-Thought code synthesis, and compositional reference recovery, achieving 100% executability on a 30-paper benchmark with visual quality within ±0.5 dB PSNR of expert implementations.
tags:
  - CVPR 2026
  - LLM Agent
  - NeRF
  - paper-to-code
  - multi-agent
  - code synthesis
  - context-free grammar
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

# Nerfify: A Multi-Agent Framework for Turning NeRF Papers into Code

**Conference**: CVPR 2026
**arXiv**: [2603.00805](https://arxiv.org/abs/2603.00805)
**Code**: To be open-sourced
**Area**: 3D Vision / Code Generation
**Keywords**: NeRF, paper-to-code, multi-agent, code synthesis, Nerfstudio

## TL;DR
Nerfify is proposed, a domain-aware multi-agent framework that automatically converts NeRF papers into trainable Nerfstudio plugin code via context-free grammar constraints, Graph-of-Thought code synthesis, and reference dependency recovery, achieving 100% executability on a 30-paper benchmark with visual quality within ±0.5 dB PSNR of expert implementations.

## Background & Motivation
**Background**: Since 2020, NeRF has accumulated over 1,000 follow-up papers, yet most lack public code or standardized implementations, requiring weeks of manual effort to re-implement prior methods for each new work.

**Limitations of Prior Work**: General-purpose paper-to-code systems such as Paper2Code and AutoP2C perform extremely poorly on NeRF code generation, failing to produce trainable code in 95% of cases. Even frontier models such as GPT-5 achieve only 26.6% accuracy on complex papers and generally produce code that compiles but fails to converge. NeRF implementations require expertise spanning volumetric rendering, computer vision, and neural optimization, where a single incorrect activation function or ray-sphere intersection can cause catastrophic failures.

**Key Challenge**: General-purpose code generation methods lack domain knowledge and cannot handle NeRF's implicit dependency chains (e.g., "we adopt the distillation loss from [3]" requires navigating to that paper, locating the correct equation, translating it to code, and implementing stop-gradient), nor can they satisfy the interface contracts and tensor shape constraints of multi-file architectures.

**Goal**: To automatically convert NeRF papers into trainable, high-quality, standardized code, reducing the turnaround time from weeks to minutes.

**Key Insight**: Formalizing the Nerfstudio framework as a context-free grammar (CFG), guiding LLM code synthesis with domain-specific constraints, and using multi-agent collaboration for dependency chain resolution and visual feedback iteration.

**Core Idea**: A domain-aware NeRF paper-to-code multi-agent system is constructed using CFG constraints, GoT synthesis, and reference dependency recovery.

## Method

### Overall Architecture
Nerfify employs a four-stage pipeline: (1) CFG formalization and in-context learning—parsing the paper into markdown and constructing a domain knowledge base $\mathcal{K}$; (2) compositional dependency resolution—traversing the citation graph to retrieve implicit components; (3) grammar-guided code generation—synthesizing the code repository in topological order via Graph-of-Thought (GoT) multi-agent collaboration; (4) visually-driven feedback—repairing visual artifacts through PSNR analysis and VLM-guided diagnosis.

### Key Designs

1. **Context-Free Grammar (CFG)-Constrained Synthesis**:

    - Function: Ensures generated code satisfies Nerfstudio's architectural invariants.
    - Mechanism: The modular composition patterns and interface contracts of Nerfstudio are formalized as a CFG. The agent system is defined as $\mathcal{A}: (\mathcal{E}(\mathcal{P}); \mathcal{R}) \mapsto \mathcal{C}$, where the resource set $\mathcal{R} = (\mathcal{K}, \mathcal{W}, \mathcal{X})$ consists of the domain knowledge base, web resources, and code templates. The paper extraction function $\mathcal{E}(\mathcal{P}) = \langle T(\mathcal{P}), I(\mathcal{P}), Q(\mathcal{P}), B(\mathcal{P}) \rangle$ covers text, images, equations, and citations.
    - Design Motivation: General-purpose code generation lacks understanding of NeRF framework structure; CFG encodes domain prior knowledge, transforming "understanding the framework" into "following a grammar."

2. **Compositional Reference Dependency Recovery**:

    - Function: Automatically retrieves implementations of components cited but not elaborated in the target paper.
    - Mechanism: A reference dependency graph $G' = (V', E')$ is constructed, and components are retrieved recursively in four steps: (a) dependency discovery—parsing citations and identifying borrowed components; (b) recursive resolution—$\text{Dependencies}(c_i) = \{c_i\} \cup \bigcup_{d \in \text{cited}(c_i)} \text{Dependencies}(d)$; (c) component extraction—identifying architectural modules, loss functions, and training protocols; (d) termination—all interface contracts are satisfied. For example, K-Planes requires 7 direct dependencies and 12 transitive dependencies.
    - Design Motivation: NeRF papers are inherently compositional; descriptions such as "we adopt the distillation loss from [3]" are pervasive in the NeRF literature, and retrieving only the target paper is insufficient for a complete implementation.

3. **Graph-of-Thought (GoT) Multi-Agent Code Synthesis**:

    - Function: Generates multi-file code repositories in topological dependency order, ensuring compilability and trainability.
    - Mechanism: The repository is defined as $\mathcal{C} = (F, G)$, where $G = \text{BuildRepoDAG}(F)$ is a directed acyclic graph. The primary synthesis agent coordinates specialized file agents across four phases: (i) DAG construction—mapping the paper to Nerfstudio component dependencies; (ii) interface freezing—establishing minimal API contracts in topological order; (iii) implementation—each node synthesizes code with type signature and tensor shape verification; (iv) integration testing—end-to-end smoke test with automatic repair.
    - Design Motivation: Monolithic code generation cannot handle the tight coupling in the NeRF pipeline (config→datamanager→field→model→training pipeline); graph-level dependency management is required.

### Loss & Training
The Stage 4 visual feedback comprises three analysis branches:
- **Metric branch**: Computes local-window PSNR/SSIM maps and uses morphological operations to localize the highest-error regions.
- **Geometry branch**: Implements cross-view artifact consensus to flag floaters and ghosting that are inconsistent across views.
- **Semantic branch**: Leverages the Qwen3 VLM to analyze artifact triplets and produce structured diagnoses and candidate patches.

Iterative refinement continues until: (1) no further feedback is generated, (2) the maximum number of iterations is reached, or (3) the paper-reported PSNR target is achieved.

## Key Experimental Results

### Main Results
Nerfify-Bench Set 1 (papers without public code, compared against expert implementations):

| Paper | Paper-Reported PSNR/SSIM | Expert PSNR/SSIM | Nerfify PSNR/SSIM |
|-------|--------------------------|------------------|-------------------|
| KeyNeRF | 25.65/0.89 | 25.70/0.89 | 26.12/0.90 |
| mi-MLP NeRF | 24.70/0.89 | 22.64/0.87 | 22.85/0.87 |
| ERS | 27.85/0.94 | 26.87/0.90 | 27.02/0.90 |
| TVNeRF | 27.44/0.93 | 26.81/0.92 | 27.30/0.92 |

All baselines (Paper2Code, AutoP2C, GPT-5, R1) **fail to produce trainable code**.

### Ablation Study (Novelty Preservation, Set 4, representative results)

| Paper | Nerfify Score | GPT-5 Score | Paper2Code Score |
|-------|--------------|-------------|-----------------|
| Mip-NeRF | 1.00 | 0.58 | 0.85 |
| BioNeRF | 1.00 | 0.82 | 0.35 |
| TensoRF | 0.98 | 0.72 | 0.12 |
| Tetra-NeRF | 1.00 | 0.58 | 0.22 |
| E-NeRF | 1.00 | 0.60 | 0.48 |

### Key Findings
- General-purpose paper-to-code systems fail to produce trainable code for 95% of NeRF papers.
- Nerfify achieves 100% executability and training convergence across all 30 benchmark papers.
- Visual quality remains within ±0.5 dB PSNR and ±0.02 SSIM of expert implementations.
- For papers with existing Nerfstudio integrations (e.g., Vanilla NeRF, Nerfacto), Nerfify produces code identical to the official implementation.

## Highlights & Insights
- Formalizing the framework as a CFG is a particularly elegant approach, recasting "understanding the framework" as "following a grammar" and substantially lowering the difficulty of LLM generation.
- Compositional reference dependency recovery addresses a long-overlooked problem: critical implementation details in academic papers are frequently buried within citation chains.
- The visual feedback loop is the first application of a VLM to NeRF code debugging, enabling a transition from "runnable" to "runs well."

## Limitations & Future Work
- The system targets only the Nerfstudio framework; other paradigms such as 3DGS are not covered.
- The CFG must be manually constructed and maintained; extending support to new frameworks requires additional engineering effort.
- Visual feedback still requires 3k iterations of training, incurring non-negligible computational overhead.
- For Set 2/3 results, LLMs may have encountered existing codebases during pre-training, introducing potential data leakage.

## Related Work & Insights
- Paper2Code and AutoP2C demonstrate the limitations of general-purpose approaches; domain awareness is the critical factor.
- Graph-of-Thought extends CoT/ToT reasoning structures to the code generation domain.
- The methodology is transferable to paper-to-code in other domains (robotics, NLP, etc.); the key is designing the corresponding domain-specific CFG.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — First domain-specific paper-to-code system combining CFG, GoT, and reference dependency recovery; a pioneering contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Large-scale benchmark of 30 papers with thorough comparison against multiple baselines and human experts.
- Writing Quality: ⭐⭐⭐⭐ — Problem motivation is clear; the four-stage framework is well-structured.
- Value: ⭐⭐⭐⭐⭐ — Highly practical impact, with significant potential to accelerate reproducibility across the NeRF community.

---

# Nerfify: A Multi-Agent Framework for Turning NeRF Papers into Code

**Conference**: CVPR 2026
**arXiv**: [2603.00805](https://arxiv.org/abs/2603.00805)
**Code**: Coming soon
**Area**: 3D Vision / Code Generation
**Keywords**: NeRF, paper-to-code, multi-agent, code synthesis, context-free grammar

## TL;DR
Nerfify is proposed, a multi-agent framework that automatically converts NeRF papers into trainable Nerfstudio plugin code via context-free grammar constraints, Graph-of-Thought code synthesis, and compositional reference recovery, achieving 100% executability on a 30-paper benchmark with visual quality within ±0.5 dB PSNR of expert implementations.

## Background & Motivation
**Background**: Since 2020, the NeRF field has produced over 1,000 follow-up papers, yet most lack public code or standardized implementations, requiring substantial human effort to re-implement prior methods for each subsequent work.

**Limitations of Prior Work**: General-purpose paper-to-code systems such as Paper2Code and AutoP2C fail severely in the NeRF domain—the current best system, O1, achieves only 26.6% accuracy on complex papers and is largely unable to produce trainable NeRF code. Even frontier models such as GPT-5 generate syntactically correct but non-convergent code.

**Key Challenge**: NeRF implementations require specialized knowledge spanning volumetric rendering, computer vision, and neural optimization, where a single incorrect activation function or ray-sphere intersection leads to failures ranging from NaN gradients to degenerate solutions. Compounding this, modern NeRF papers carry deep citation dependencies—a phrase such as "we adopt the distortion loss from [3]" necessitates tracing multiple papers to extract the correct implementation.

**Goal**: To automatically convert NeRF research papers into trainable, convergent code that matches the visual quality of expert implementations and meets standardized coding standards.

**Key Insight**: Replacing general-purpose paper-to-code approaches with a domain-specific multi-agent framework that formalizes the Nerfstudio architecture as a context-free grammar to constrain code generation.

**Core Idea**: Reliable conversion of NeRF papers to trainable code is achieved by encoding Nerfstudio's architecture as CFG constraints on LLM synthesis, using GoT to generate multi-file repositories in topological order, and employing compositional reference recovery to automatically trace citation dependencies.

## Method

### Overall Architecture
Nerfify converts NeRF papers to code in four stages: (1) CFG formalization and in-context learning—formalizing the Nerfstudio architecture as a CFG and constructing a knowledge base $\mathcal{K}$; (2) compositional dependency resolution—traversing the citation graph to recover implicit dependencies; (3) grammar-guided repository generation—using GoT to synthesize multi-file repositories in topological order; (4) visually-driven feedback—iteratively improving implementation quality through visual analysis of training runs.

### Key Designs

1. **Context-Free Grammar (CFG)-Constrained Synthesis**:

    - Function: Ensures generated code satisfies Nerfstudio's architectural invariants.
    - Mechanism: Nerfstudio's modular composition and interface specifications are formalized as a CFG, i.e., repository $\mathcal{C} = (F, G)$, where $F = \{f_1, f_2, \ldots, f_n\}$ is the file set and $G = \text{BuildRepoDAG}(F)$ is a directed acyclic dependency graph. LLMs generate code under CFG constraints, guaranteeing compilation correctness.
    - Design Motivation: General LLMs produce syntactically correct but architecturally invalid code due to insufficient domain knowledge; CFG encodes this domain knowledge to prevent module wiring errors and incorrect mathematical implementations.

2. **Graph-of-Thought (GoT) Code Synthesis**:

    - Function: Coordinates multiple specialized agents to generate multi-file repositories in topological order.
    - Mechanism: The process proceeds in four phases: (1) DAG construction—mapping the paper to Nerfstudio component dependencies; (2) interface freezing—establishing API contracts in topological order; (3) implementation—each node generates verified code with tensor shape and type signature checks; (4) integration testing—running smoke tests with automatic repair.
    - Design Motivation: Files in the NeRF pipeline are tightly coupled (config→datamanager→field→model→training pipeline); monolithic generation readily leads to interface inconsistencies.

3. **Compositional Reference Recovery**:

    - Function: Automatically traverses the citation graph to retrieve implicit dependency components across papers.
    - Mechanism: A reference dependency graph $G' = (V', E')$ is constructed, and iterative multi-hop retrieval is performed on the target paper—dependency discovery → recursive resolution → component extraction → termination check. For example, K-Planes requires extracting components such as proposal networks, hash encoders, and VM decomposition from 7 direct references and 12 transitive dependencies.
    - Design Motivation: NeRF papers are inherently compositional; a single paper may implicitly depend on specific technical components from dozens of other works.

### Loss & Training
Stage 4 visual feedback includes three analysis branches:
- **Metric branch**: Computes local-window PSNR/SSIM maps and uses morphological operations to localize the highest-error regions.
- **Geometry branch**: Implements cross-view artifact consensus (Cross-View Artifact Consensus) to flag floaters and ghosting that are inconsistent across views.
- **Semantic branch**: Leverages the Qwen3 VLM to analyze artifact triplets and produce structured diagnoses and candidate patches.

Iterative refinement continues until: (1) no further feedback is generated, (2) the maximum number of iterations is reached, or (3) the paper-reported PSNR target is achieved.

## Key Experimental Results

### Main Results

| Paper | Dataset | Expert PSNR↑ | Nerfify PSNR↑ | Expert SSIM↑ | Nerfify SSIM↑ |
|-------|---------|-------------|--------------|-------------|-------------|
| KeyNeRF | Blender | 25.70 | 26.12 | 0.89 | 0.90 |
| mi-MLP NeRF | Blender | 22.64 | 22.85 | 0.87 | 0.87 |
| ERS | DTU | 26.87 | 27.02 | 0.90 | 0.90 |
| TVNeRF | Blender | 26.81 | 27.30 | 0.92 | 0.92 |

All baselines (Paper2Code, AutoP2C, GPT-5, R1) **fail to produce trainable code**.

### Ablation Study

| Configuration | Executability | Key Finding |
|--------------|--------------|-------------|
| Nerfify (full) | 100% | Within ±0.5 dB PSNR of expert implementations |
| Paper2Code | 5% | Compiles but not trainable |
| AutoP2C | 0% | Imports fail to resolve |
| GPT-5 | 0% | Compiles but training does not converge |
| w/o reference recovery | Partial | Missing critical dependency components |
| w/o visual feedback | 100% | Larger performance gap |

### Key Findings
- Nerfify achieves visual quality comparable to expert implementations even on papers that have never been implemented before (Set 1).
- For papers with existing Nerfstudio implementations, Nerfify generates code identical to the official repository.
- K-Planes' citation dependency graph spans 7 direct dependencies and 12 papers in total through transitive dependencies.

## Highlights & Insights
- The first demonstration that a domain-aware multi-agent framework can reliably convert complex visual papers into trainable code.
- CFG constraints are the critical enabler—encoding architectural knowledge as formal grammar guarantees code correctness by construction.
- Compositional reference recovery resolves the pervasive problem of implicit dependencies in NeRF papers.
- Implementation time is reduced from weeks to minutes, democratizing reproducibility in NeRF research.

## Limitations & Future Work
- The system currently targets only the Nerfstudio framework; extending to other frameworks requires redefining the CFG.
- The system depends on high-quality PDF parsing (MinerU); parsing errors in mathematical equations or figures may propagate.
- Visual feedback requires 3k iterations of actual training, incurring non-zero computational cost.
- For entirely novel NeRF methods that do not build on existing components, the benefit of reference recovery is limited.

## Related Work & Insights
- **Paper2Code / AutoP2C**: General-purpose paper-to-code systems that lack domain-specific constraints.
- **Scene Language**: Similarly uses CFG constraints for visual program synthesis.
- **Graph of Thoughts**: Generalizes reasoning to directed graphs; Nerfify applies this to code generation.
- Insight: Any domain with a standardized framework (e.g., MMDetection, HuggingFace Transformers) can adopt a similar paper-to-code approach; the key is designing the corresponding domain CFG.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — First realization of automatic conversion from NeRF papers to trainable code; the combination of CFG, GoT, and reference recovery is original.
- Experimental Thoroughness: ⭐⭐⭐⭐ — A 30-paper benchmark including never-before-implemented papers with thorough comparisons.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured with a logically tight four-stage decomposition.
- Value: ⭐⭐⭐⭐⭐ — Highly practical value; poised to significantly accelerate reproducibility across the NeRF community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Think, Then Verify: A Hypothesis-Verification Multi-Agent Framework for Long Video Understanding](think_then_verify_a_hypothesis-verification_multi-agent_framework_for_long_video.md)
- [\[CVPR 2026\] CarePilot: A Multi-Agent Framework for Long-Horizon Computer Task Automation in Healthcare](carepilot_a_multi-agent_framework_for_long-horizon_computer_task_automation_in_h.md)
- [\[CVPR 2026\] REALM: An MLLM-Agent Framework for Open World 3D Reasoning Segmentation and Editing on Gaussian Splatting](realm_an_mllm-agent_framework_for_open_world_3d_reasoning_segmentation_and_editi.md)
- [\[CVPR 2026\] ARGOS: Who, Where, and When in Agentic Multi-Camera Person Search](argos_agentic_multi_camera_person_search.md)
- [\[CVPR 2026\] Ego2Web: A Web Agent Benchmark Grounded in Egocentric Videos](ego2web_a_web_agent_benchmark_grounded_in_egocentric_videos.md)

</div>

<!-- RELATED:END -->

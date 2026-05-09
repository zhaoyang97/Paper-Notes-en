---
title: >-
  [Paper Note] Nerfify: A Multi-Agent Framework for Turning NeRF Papers into Code
description: >-
  [CVPR 2026][LLM Agent][Paper-to-code] Nerfify is proposed as a four-stage pipeline—CFG formalization with in-context learning, compositional citation recovery, GoT-based code synthesis, and visual feedback—that automatically converts NeRF papers into trainable Nerfstudio plugins, achieving 100% executability on a 30-paper benchmark (vs. 5% for general baselines) with visual quality within ±0.5 dB PSNR of expert implementations.
tags:
  - CVPR 2026
  - LLM Agent
  - Paper-to-code
  - Multi-agent framework
  - NeRF
  - Context-free grammar
  - Graph-of-Thought
date: 2026-05-08
content_hash: 01289b34a95cba6b
---

# Nerfify: A Multi-Agent Framework for Turning NeRF Papers into Code

**Conference**: CVPR 2026
**arXiv**: [2603.00805](https://arxiv.org/abs/2603.00805)
**Code**: Coming soon
**Area**: LLM Agent
**Keywords**: Paper-to-code, Multi-agent framework, NeRF, Context-free grammar, Graph-of-Thought

## TL;DR
Nerfify is proposed as a four-stage pipeline—CFG formalization with in-context learning, compositional citation recovery, GoT-based code synthesis, and visual feedback—that automatically converts NeRF papers into trainable Nerfstudio plugins, achieving 100% executability on a 30-paper benchmark (vs. 5% for general baselines) with visual quality within ±0.5 dB PSNR of expert implementations.

## Background & Motivation

**Background**: Since its publication in 2020, NeRF has spawned 1,000+ follow-up works, yet the majority lack public code or standardized implementations. General paper-to-code systems (e.g., Paper2Code, AutoP2C) achieve only 26.6% accuracy on complex papers, far below human experts (41.4%).

**Limitations of Prior Work**: NeRF implementation spans volumetric rendering, computer vision, and neural optimization, where a single incorrect activation function or ray–sphere intersection calculation can cause NaN gradients or degenerate solutions. Modern NeRF papers also carry heavy citation dependencies—K-Planes, for instance, relies on components from 7 direct and 12 transitive references. General-purpose systems are unable to resolve such implicit dependencies.

**Key Challenge**: General paper-to-code systems lack NeRF domain knowledge (the tight coupling of rendering mathematics and neural architecture), making it nearly impossible for them to produce trainable code. Yet manually implementing a NeRF paper requires 1–2 weeks of expert-level effort.

**Goal**: Automatically convert NeRF papers into trainable, convergent Nerfstudio plugins whose visual quality matches expert implementations.

**Key Insight**: Encode the architectural patterns of the Nerfstudio framework as a context-free grammar to constrain code generation, and achieve domain specialization through multi-agent collaboration.

**Core Idea**: A domain-aware multi-agent design (CFG constraints + citation-graph traversal + visual feedback) makes code translation of complex vision papers tractable.

## Method

### Overall Architecture
A four-stage pipeline: (1) CFG formalization + in-context learning: encode the Nerfstudio architecture as a CFG and construct a knowledge base $\mathcal{K}$; (2) compositional dependency resolution: traverse the citation graph to recursively retrieve components from referenced papers; (3) GoT code synthesis: multi-agent generation of multi-file repositories in topological order; (4) visual-driven feedback: iterative repair guided by PSNR/SSIM heatmaps and VLM diagnostics.

### Key Designs

1. **Context-Free Grammar (CFG) Constraints**:

    - **Function**: Encode Nerfstudio's module interfaces and architectural invariants as a formal grammar to constrain LLM code generation.
    - **Mechanism**: Nerfstudio components (Field, Model, DataManager, Pipeline, etc.) and their interface specifications are defined as production rules. Generated code must satisfy these syntactic constraints, ensuring architectural correctness by construction. In-context examples are also constructed from existing paper–code pairs $\{(\mathcal{P}_i, \mathcal{C}_i)\}$.
    - **Design Motivation**: NeRF code failures are often not syntax errors but architectural violations (incorrect component interfaces, missing required modules). CFG eliminates this class of errors at the root.

2. **Compositional Citation Recovery**:

    - **Function**: Automatically and recursively retrieve implementation details of borrowed components from the citation graph.
    - **Mechanism**: A citation dependency DAG $G' = (V', E')$ is constructed. For the target paper, four steps are executed: (a) dependency discovery: parse citations and identify borrowed components; (b) recursive resolution: recursively expand transitive dependencies for each reference; (c) component extraction: a dedicated LLM agent extracts specific modules from each referenced paper; (d) termination: when all interface-contract dependencies are satisfied.
    - **Design Motivation**: Taking K-Planes as an example, the phrase "we adopt the distortion loss from [3]" requires navigating to the Mip-NeRF 360 paper, locating the correct formula, translating it into code, and implementing stop-gradient. General-purpose systems cannot handle such multi-hop dependencies.

3. **Graph-of-Thought (GoT) Multi-Agent Code Synthesis**:

    - **Function**: Coordinate multiple specialized agents to generate code in the topological order of file dependencies.
    - **Mechanism**: (a) DAG construction: map the paper to the Nerfstudio component dependency graph; (b) interface freezing: determine the public API of each file in topological order; (c) implementation: each node's agent generates code and verifies tensor shapes, gradients, and type signatures; (d) integration testing: end-to-end smoke testing with an automatic repair loop triggered on failure.
    - **Design Motivation**: NeRF repositories are multi-file and tightly coupled—config → data manager → field → model → pipeline. A graph-native approach enables component-level fault localization.

4. **Visual-Driven Feedback**:

    - **Function**: Iteratively improve implementation quality through visual analysis after training.
    - **Mechanism**: After a 3k-iteration smoke training run, multi-view images are rendered and analyzed via three branches: (a) metric branch: compute local-window PSNR/SSIM heatmaps; (b) geometry branch: cross-view artifact consensus detection; (c) semantic branch: Qwen3 VLM analyzes artifact triples to generate diagnostics and candidate patches.
    - **Design Motivation**: Mere executability is insufficient; the visual feedback loop is the critical step from "runs" to "runs well."

### Loss & Training
Nerfify itself requires no training. The generated NeRF code is trained following the original paper (100k iterations). Internally, GPT-5 and Qwen3 VLM are used.

## Key Experimental Results

### Main Results

**Set 1: Papers without public code**

| Paper | Author-reported PSNR | Expert impl. PSNR | Nerfify PSNR | Diff. |
|------|------------|-----------|------------|------|
| KeyNeRF | 25.65 | 25.70 | **26.12** | +0.42 |
| mi-MLP NeRF | 24.70 | 22.64 | 22.85 | +0.21 |
| ERS | 27.85 | 26.87 | 27.02 | +0.15 |
| TVNeRF | 27.44 | 26.81 | 27.30 | +0.49 |

**Executability comparison**

| System | Import resolution | Trainable | Stable training | Convergence |
|------|---------|-------|---------|------|
| Paper2Code | ✓ | ✗ | ✗ | ✗ |
| GPT-5 | ✓ | ✗ | ✗ | ✗ |
| **Nerfify** | ✓ | ✓ | ✓ | ✓ |

### Ablation Study

| Ablation | Effect | Notes |
|------|------|------|
| w/o CFG constraints | Large drop in executability | Architectural violations cause runtime errors |
| w/o citation dependency resolution | Critical components missing | Unable to implement components in dependency chains |
| w/o GoT (monolithic generation) | Slower convergence, higher error rate | Lack of component-level fault localization |
| w/o visual feedback | Executable but lower quality | Larger PSNR gap from target |
| Full Nerfify | 100% executable + ±0.5 dB | All four components are indispensable |

### Key Findings
- All general-purpose baselines fail to generate trainable code in 95% of cases—Paper2Code's K-Planes implementation uses a plain MLP rather than planar decomposition.
- Nerfify achieves visual quality comparable to expert implementations for papers with no public code whatsoever.
- Turnaround time is reduced from weeks to minutes.
- K-Planes requires traversing transitive dependencies across 12 papers; compositional citation recovery is the key differentiating factor.

## Highlights & Insights
- **Generality of CFG-constrained code generation**: Formalizing the target framework's architectural patterns as grammar rules and having the LLM generate within those rules ensures architectural correctness. This approach is transferable to any domain with a standard framework.
- **Recursive citation dependency resolution is the killer feature**: Most papers' single-sentence citations conceal multi-layer implementation dependencies; Nerfify is the first system to automate this process.
- **Visual feedback closes the loop**: Achieving "visual quality matching" through actual rendering and VLM diagnostics marks a qualitative shift from "code generation" to "research reproduction."
- **Nerfify-Bench**: A benchmark of 30 papers across 4 categories; Set 1 (papers without public code) effectively eliminates LLM data contamination concerns.

## Limitations & Future Work
- Highly specialized for the NeRF + Nerfstudio ecosystem; generalizing to other domains requires rebuilding the CFG and knowledge base from scratch.
- Relies on frontier LLMs such as GPT-5, incurring substantial computational cost.
- Visual feedback requires actual GPU training (3k-iteration smoke runs).
- When a paper's mathematical description is ambiguous or erroneous, the system cannot handle the situation better than a human expert.

## Related Work & Insights
- **vs. Paper2Code/AutoP2C**: General-purpose systems nearly completely fail on NeRF, demonstrating the necessity of domain specialization.
- **vs. GPT-5 single-pass generation**: Even the strongest LLMs cannot directly generate trainable NeRF code from a paper.
- **vs. AutoReproduce**: Although it also exploits citation lineage, it remains domain-agnostic.
- The framework can serve as a template for paper-to-code systems in other vision domains (diffusion models, 3DGS).

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The complete system combining CFG + GoT + citation traversal + visual feedback is a first in the paper-to-code domain.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 30-paper benchmark, 4-category evaluation, data-contamination-free validation, and complete ablation study.
- **Writing Quality**: ⭐⭐⭐⭐ The system is complex but described clearly.
- **Value**: ⭐⭐⭐⭐⭐ Reduces NeRF paper reproduction from weeks to minutes, demonstrating the viability of domain-specialized AI coding.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] CarePilot: A Multi-Agent Framework for Long-Horizon Computer Task Automation in Healthcare](carepilot_a_multi-agent_framework_for_long-horizon_computer_task_automation_in_h.md)
- [\[CVPR 2026\] Think, Then Verify: A Hypothesis-Verification Multi-Agent Framework for Long Video Understanding](think_then_verify_a_hypothesis-verification_multi-agent_framework_for_long_video.md)
- [\[CVPR 2026\] REALM: An MLLM-Agent Framework for Open World 3D Reasoning Segmentation and Editing on Gaussian Splatting](realm_mllm_agent_3d_reasoning_gaussian.md)
- [\[CVPR 2026\] ARGOS: Who, Where, and When in Agentic Multi-Camera Person Search](argos_agentic_multi_camera_person_search.md)
- [\[ICLR 2026\] HAMLET: A Hierarchical and Adaptive Multi-Agent Framework for Live Embodied Theatre](../../ICLR2026/llm_agent/hamlet_a_hierarchical_and_adaptive_multi-agent_framework_for_live_embodied_theat.md)

<!-- RELATED:END -->

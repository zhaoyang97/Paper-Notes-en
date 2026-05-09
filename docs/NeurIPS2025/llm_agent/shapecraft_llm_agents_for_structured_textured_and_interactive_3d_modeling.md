---
title: >-
  [Paper Note] ShapeCraft: LLM Agents for Structured, Textured and Interactive 3D Modeling
description: >-
  [NEURIPS2025][LLM Agent][text-to-3D] This paper proposes ShapeCraft, a multi-agent framework built on a Graph-based Procedural Shape (GPS) representation. Three LLM agents — Parser, Coder, and Evaluator — collaborate to decompose natural language descriptions into structured sub-task graphs, iteratively generating editable and animatable textured 3D assets.
tags:
  - NEURIPS2025
  - LLM Agent
  - text-to-3D
  - multi-agent
  - procedural modeling
  - shape program
  - Blender
date: 2026-05-08
content_hash: c9b2fea24565ced7
---

# ShapeCraft: LLM Agents for Structured, Textured and Interactive 3D Modeling

**Conference**: NEURIPS2025
**arXiv**: [2510.17603](https://arxiv.org/abs/2510.17603)
**Code**: [GitHub](https://sanbingyouyong.github.io/shapecraft)
**Area**: LLM Agent
**Keywords**: text-to-3D, multi-agent, procedural modeling, shape program, Blender

## TL;DR

This paper proposes ShapeCraft, a multi-agent framework built on a Graph-based Procedural Shape (GPS) representation. Three LLM agents — Parser, Coder, and Evaluator — collaborate to decompose natural language descriptions into structured sub-task graphs, iteratively generating editable and animatable textured 3D assets.

## Background & Motivation

**Existing Problems**: Current text-to-3D methods — optimization-based approaches (e.g., SDS) and autoregressive approaches (e.g., LLaMA-Mesh) — produce meshes that lack semantic part segmentation, making them difficult to edit or animate, and thus unsuitable for real-world artistic workflows.

**Limitations of Optimization-Based Methods**: SDS-based methods require converting implicit representations to explicit meshes via iso-surfacing, introducing artifacts such as dense triangulation and topological inconsistencies.

**Limitations of Autoregressive Methods**: Methods that directly model triangle sequences suffer from limited generalization due to training data distribution constraints, and produce monolithic representations that are not easily modifiable.

**Potential of Procedural Modeling**: Representing shapes as structured programs enables interpretable and modifiable generation, but text–program paired data are scarce.

**Difficulty of Directly Applying LLMs to 3D**: Prior work such as 3D-PREMISE prompts LLMs to generate complete shape programs directly, but LLMs struggle with complex spatial reasoning and semantic shape details, yielding inaccurate results.

**Core Insight**: Decomposing complex natural language descriptions into a sub-task graph (GPS) substantially reduces the cognitive load on LLMs. Combined with multi-path sampling and visual feedback iteration, generation quality can be significantly improved.

## Method

### Overall Architecture

ShapeCraft is a collaborative multi-agent system comprising three specialized agents (Parser, Coder, Evaluator) and a shared core data structure, the Graph-based Procedural Shape (GPS). The pipeline proceeds as follows: (1) the Parser parses the input text and constructs the GPS graph; (2) the Coder generates bounding volumes and code snippets for each node; (3) the Evaluator assesses rendered results and provides feedback; (4) the process iterates until convergence; (5) component-aware texture painting is applied.

### Key Design 1: GPS Graph Representation

GPS is defined as $\mathcal{G}=(\mathcal{V}, \mathcal{E}, \mathcal{A})$ with a flattened depth-1 graph structure:

- **Virtual root node** $v_0$: represents the global semantic abstraction (e.g., "chair")
- **Component nodes** $\{v_i\}_{i>0}$: each represents an independent geometric part, directly connected to the root node
- **Node attributes** $\mathcal{A}(v_i) = (n_i^g, n_i^p, b_i, p_i)$:
    - $n_i^g$: geometric description (textual shape details of the component)
    - $n_i^p$: positional description (spatial relationships and relative positions)
    - $b_i \in \mathbb{R}^6$: bounding volume parameters $(c_x, c_y, c_z, h, w, l)$
    - $p_i$: executable Blender API code snippet

**Hierarchical Parsing and Flattening**: The Parser hierarchically decomposes the input text (e.g., chair → upper body → backrest), retaining only leaf nodes directly connected to the root node, enabling parallel modeling.

**Representation Bootstrapping**: Through $N=2$ rounds of Evaluator assessment → Parser+Coder update cycles, bounding volume parameters in the GPS are progressively corrected, mitigating LLM hallucination.

### Key Design 2: Iterative Shape Modeling with Multi-Path Sampling

For each component node in the GPS, $M$ independent modeling paths are created for parallel exploration:

1. **Initialization**: $M$ copies $\{v_{i,m}^0\}$ are created for each node $v_i$; the Coder generates initial code based on the geometric description.
2. **Iterative Refinement** ($T$ steps):
    - The Evaluator renders multi-view images of the current component, producing textual feedback $f_{i,m}^t$ and a quality score $s_{i,m}^t$.
    - The Coder updates the code based on feedback: $v_{i,m}^{t+1} \leftarrow \text{Coder}(v_{i,m}^t, f_{i,m}^t, \mathcal{G}^*)$
3. **Early Stopping**: If any path's score exceeds a threshold $s_\tau$, it is terminated immediately to save computation.
4. **Best Path Selection**: The result of the highest-scoring path is used to update the GPS.

Default configuration: $M=3$ paths, $T=3$ iterations. High-temperature sampling encourages diverse modeling strategies.

### Key Design 3: Component-Aware BRDF Shading (CASD)

The component decomposition structure of the GPS is leveraged for texture optimization:

- **Texture field** $\psi_\theta$: maps UV coordinates to BRDF parameters $(k_d, k_r, k_m)$ (diffuse albedo, roughness, metalness), with values in $[0,1]$, directly importable into standard rendering pipelines.
- **Component-Aware SDS Loss**:

$$\mathcal{L}_{CASD} = \mathcal{L}_{SDS}(L(\psi_\theta(\mathbf{p}), \omega), x) + \sum_{i=1}^{M} \mathcal{L}_{SDS}(L(\psi_\theta(\mathbf{p}_{v_i}), \omega), n_i)$$

The global SDS ensures overall coherence, while the component-level SDS improves text alignment for each part using its geometric description $n_i$; only the externally visible surfaces of each component are optimized.

### Loss & Training

The total loss is the CASD loss, combining global text-guided and component text-guided Score Distillation Sampling. Classifier-free guidance is employed to strengthen text-conditional control.

## Key Experimental Results

### Main Results: Geometry Quality and Text Consistency (MARVEL Subset)

| Method | IoGT ↑ | Hausdorff ↓ | CLIP Score ↑ | VQA Pass Rate ↑ | Runtime ↓ | API Calls ↓ |
|--------|--------|-------------|-------------|-----------------|-----------|------------|
| 3D-PREMISE | 0.385 | 0.527 | 26.76 | 0.33 | 2.81 min | 6 |
| CADCodeVerify | 0.334 | 0.511 | 25.94 | 0.34 | 3.06 min | 9 |
| BlenderLLM | 0.455 | 0.511 | 26.99 | 0.43 | 5.11 min | N.A |
| LLaMA-Mesh | 0.346 | 0.464 | 25.72 | 0.28 | 15.64 min | N.A |
| MVDream | 0.427 | 0.411 | 26.84 | 0.42 | 32.10 min | N.A |
| **ShapeCraft** | **0.471** | 0.415 | **27.27** | **0.44** | 11.68 min | 21 |

ShapeCraft achieves the best performance on IoGT, CLIP Score, and VQA Pass Rate. Its Hausdorff distance is close to the best-performing MVDream, while requiring only one-third of MVDream's runtime.

### Ablation Study: Multi-Path Sampling and Iterative Refinement

| Configuration | Hausdorff ↓ | IoGT ↑ | CLIP Score ↑ | Runtime ↓ |
|--------------|------------|--------|-------------|-----------|
| M=1, T=1 | 0.485 | 0.436 | 25.75 | 1.62 min |
| M=3, T=1 | 0.444 | 0.535 | 25.90 | 3.71 min |
| M=1, T=3 | 0.494 | 0.492 | 26.20 | 3.90 min |
| **M=3, T=3 (default)** | **0.415** | **0.471** | **27.27** | 11.68 min |
| M=3, T=5 | 0.360 | 0.431 | 26.39 | 18.04 min |

### Comparison with Advanced Thinking-Mode LLMs (GPS Parsing Validity)

| Method | IoGT ↑ | Hausdorff ↓ | CLIP ↑ | Compilation Rate ↑ |
|--------|--------|-------------|--------|-------------------|
| ChatGPT-o3 | 0.177 | 0.708 | 25.48 | 60% |
| ChatGPT-o4-mini-high | 0.244 | 0.493 | 26.30 | 80% |
| Deepseek-R1-0528 | 0.326 | 0.489 | 29.01 | 80% |
| Gemini-2.5-Pro | 0.102 | 0.586 | 27.31 | 60% |
| **ShapeCraft** | **0.471** | **0.415** | 27.27 | **100%** |

### Key Findings

1. **GPS significantly constrains LLM reasoning space**: Even state-of-the-art thinking-mode LLMs (o3/o4/R1/Gemini-2.5) cannot reliably generate 3D shape programs, achieving compilation rates of only 60–80%, whereas ShapeCraft achieves 100%.
2. **Multi-path sampling is more effective than iterative refinement**: $M=3, T=1$ yields IoGT of 0.535, which surpasses $M=1, T=3$ (0.492), indicating that parallel exploration is more beneficial than single-path deep refinement.
3. **Excessive iteration is harmful**: $M=3, T=5$ yields lower IoGT (0.431) and CLIP score (26.39) than $M=3, T=3$, likely due to degradation introduced by over-modification.
4. **Component-aware texturing handles complex prompts**: Fine-grained texture descriptions such as "rust and dirt spots" can be correctly mapped to corresponding components.

## Highlights & Insights

1. **Elegant GPS Representation Design**: Hierarchical parsing combined with flattened storage balances semantic understanding depth with parallel modeling efficiency; the flat structure treats each component as an independent sub-task, naturally supporting parallelism.
2. **Representation Bootstrapping**: Only two rounds of visual feedback suffice to substantially improve the initial GPS quality, constituting a lightweight yet effective self-correction strategy.
3. **Dual Exploration via Multi-Path Sampling and Iteration**: Multi-path sampling increases breadth (diverse modeling strategies), while iterative refinement increases depth (single-strategy polishing); the two mechanisms are complementary.
4. **Editability as a Core Contribution**: The output is not a static mesh but an interpretable program with semantic part segmentation, directly supporting animation and editing.
5. **Component-Aware Texture Alignment**: The GPS component information is leveraged to decompose global descriptions into local supervision signals, addressing the weak alignment of SDS to complex prompts.

## Limitations & Future Work

1. **Sensitivity to Prompt Quality**: Ambiguous, overly brief, or highly creative prompts can still cause inaccurate Parser decomposition and insufficient Evaluator signals.
2. **Difficulty with Complex Organic Shapes**: Organic geometry such as tails and wings remains challenging due to the scope of the Blender API library available to the Coder.
3. **Long Runtime**: With 21 API calls and an 11.68-minute runtime, ShapeCraft is approximately four times slower than direct methods (e.g., 3D-PREMISE at 2.81 minutes).
4. **Dependence on Specific LLMs**: The framework uses Qwen3-235B as Parser/Coder and Qwen-VL-Max as Evaluator; generalizability to other model families has not been validated.

## Related Work & Insights

- **vs. 3D-PREMISE / CADCodeVerify**: ShapeCraft constrains the reasoning space via GPS, avoiding the failures of direct full-program generation; compilation rate improves from 60–80% to 100%.
- **vs. MVDream (optimization-based)**: ShapeCraft produces structured, editable meshes, whereas MVDream generates dense, non-editable meshes; ShapeCraft runs three times faster.
- **vs. 3D-GPT**: 3D-GPT focuses on scene-level asset retrieval and layout rather than fine-grained shape modeling; ShapeCraft addresses precise shape generation at the individual object level.
- **Insights**: The multi-agent + structured intermediate representation paradigm is transferable to other generative tasks (e.g., code generation, document authoring); the GPS "hierarchical analysis + flat execution" design philosophy is worth broader adoption.

## Rating
- Novelty: ⭐⭐⭐⭐ (GPS representation and component-aware texturing are novel contributions; the multi-agent framework itself is not new but its integration with 3D generation is innovative)
- Experimental Thoroughness: ⭐⭐⭐⭐ (covers qualitative and quantitative comparisons, multi-dimensional ablations, and comparison with thinking-mode LLMs; user studies are absent)
- Writing Quality: ⭐⭐⭐⭐ (clear structure, rich illustrations, well-formatted algorithmic pseudocode)
- Value: ⭐⭐⭐⭐ (the first LLM agent approach to achieve 100% compilation rate in text-to-3D; strong practical utility)

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Structured Personalization: Modeling Constraints as Matroids for Data-Minimal LLM Agents](../../AAAI2026/llm_agent/structured_personalization_modeling_constraints_as_matroids_for_data-minimal_llm.md)
- [\[NeurIPS 2025\] TrajAgent: An LLM-Agent Framework for Trajectory Modeling via Large-and-Small Model Collaboration](trajagent_an_llm-agent_framework_for_trajectory_modeling_via_large-and-small_mod.md)
- [\[NeurIPS 2025\] Traj-CoA: Patient Trajectory Modeling via Chain-of-Agents for Lung Cancer Risk Prediction](traj-coa_patient_trajectory_modeling_via_chain-of-agents_for_lung_cancer_risk_pr.md)
- [\[NeurIPS 2025\] A-MEM: Agentic Memory for LLM Agents](a-mem_agentic_memory_for_llm_agents.md)
- [\[ACL 2026\] CodeStruct: Code Agents over Structured Action Spaces](../../ACL2026/llm_agent/codestruct_code_agents_over_structured_action_spaces.md)

<!-- RELATED:END -->

---
title: >-
  [Paper Note] UniRTL: Unified Code and Graph for Robust RTL Representation Learning
description: >-
  [ICML 2026][Code Intelligence][RTL Representation Learning] This paper proposes UniRTL—a multimodal unified representation achieved through the joint learning of RTL code and Control Data Flow Graphs (CDFG). By employing…
tags:
  - "ICML 2026"
  - "Code Intelligence"
  - "RTL Representation Learning"
  - "Multimodal Pre-training"
  - "CDFG"
  - "Performance Prediction"
  - "Code Retrieval"
date: 2026-05-08
content_hash: 5ecebb508d7ac9a1
---

# UniRTL: Unified Code and Graph for Robust RTL Representation Learning

**Conference**: ICML 2026  
**arXiv**: [2605.31040](https://arxiv.org/abs/2605.31040)  
**Code**: https://github.com/cure-lab/UniRTL  
**Area**: Code Intelligence / Hardware Design  
**Keywords**: RTL Representation Learning, Multimodal Pre-training, CDFG, Performance Prediction, Code Retrieval

## TL;DR
This paper proposes UniRTL—a multimodal unified representation achieved through the joint learning of RTL code and Control Data Flow Graphs (CDFG). By employing a graph-aware tokenizer and a hierarchical training strategy, it significantly outperforms existing methods in hardware performance prediction and code retrieval tasks.

## Background & Motivation

**Background**: RTL (Register Transfer Level) representation learning is crucial for accelerating hardware design design flows. Existing methods rely either solely on RTL code (VeriDistill, DeepRTL2) or exclusively on graph structures (StructRTL).

**Limitations of Prior Work**: Single-modal representations suffer from limited expressive power. Code implicitly contains semantic functional information but lacks complete structural dependencies; conversely, graphs preserve topological information but possess sparse semantic information. Although GraphCodeBERT integrates code and data flow, its alignment strategy is weak (variable-level only) and the data flow is incomplete.

**Key Challenge**: How to design a truly fine-grained multimodal alignment mechanism that fully leverages the completeness of CDFG and the semantic complementarity of RTL code.

**Key Insight**: The authors adopt CDFG instead of simplified data flows, as it preserves complete design information and can be faithfully converted back to code. They utilize mutual masking modeling to achieve fine-grained code-graph alignment and construct a graph-aware tokenizer to enable the Transformer to capture subtle relationships within graph structures.

**Core Idea**: A hierarchical pre-training framework is proposed: first pre-training a graph-aware tokenizer → performing text-code alignment warm-up → executing graph fusion. This establishing deep alignment across three modalities via mutual masking modeling.

## Method

### Overall Architecture
UniRTL adopts a unified Transformer architecture (based on CodeBERT). The workflow includes: (1) Generating CDFG from Verilog code (via Yosys → RTLIL → AST → CDFG); (2) Hierarchical pre-training: graph-aware tokenizer, text-code alignment, and graph fusion; (3) Downstream task fine-tuning (performance prediction/code retrieval). The dataset consists of 132,008 RTL designs, 38,888 of which were successfully converted to CDFG.

### Key Designs

1.  **Graph-aware Tokenizer**:
    - **Function**: Encodes CDFG into structure-aware token sequences while preserving graph topological information.
    - **Mechanism**: For each node $v_i$, an initial node embedding is first constructed: $\mathbf{H}_{i}=\text{one-hot}(\text{type}(v_{i}))\parallel\text{width}(v_{i})\parallel\text{pca}(\phi_{\text{text}}(\text{desc}(v_{i})))$. Local dependencies are captured via GIN: $\mathbf{L}_{i}^{(k)}=\text{MLP}^{(k)}((1+\epsilon^{(k)})\cdot\mathbf{L}_{i}^{(k-1)}+\sum_{j\in\mathcal{N}(i)}\mathbf{L}_{j}^{(k-1)})$. Finally, a lightweight Transformer encodes the global context to obtain $\{\mathbf{G}_{i}\}$.
    - **Design Motivation**: Simply flattening variable nodes (as in GraphCodeBERT) fails to capture subtle structural nuances. The GIN+Transformer combination preserves topological relations while encoding complete elements like operators and control flows.

2.  **Mutual Masking Modeling Alignment**:
    - **Function**: Establishes fine-grained alignment between text-code and code-graph modalities.
    - **Mechanism**: In the text-code stage, 20% of tokens are randomly masked, and the model recovers them from the complementary modality. During graph fusion, 20% of nodes and 20% of code tokens are masked simultaneously to jointly predict original node types and token IDs. Global positional encodings (graph Laplacian eigenvectors) are added to nodes to preserve topology.
    - **Design Motivation**: Compared to variable alignment in GraphCodeBERT (only locating variables) and coarse-grained contrastive learning in CircuitFusion, mutual masking modeling forces the model to learn deep semantic correspondences through mutual dependency.

3.  **Hierarchical Training Strategy**:
    - **Function**: Maximizes data utilization and stabilizes gradient flow.
    - **Mechanism**: Stage 1 pre-trains the graph-aware tokenizer; Stage 2 performs text-code alignment (5 epochs); Stage 3 executes graph fusion (300 epochs).
    - **Design Motivation**: Text-code pairs are far more abundant than graph-code pairs (132k vs 38.8k). A phased approach fully utilizes data gradients and prevents optimization instability caused by insufficient graph data.

## Key Experimental Results

### Main Results: Performance Prediction (without netlist)

| Method | Area MAE↓ | Area MAPE↓ | Area $R^2$↑ | Delay MAE↓ |
|------|-----------|-----------|-----------|-----------|
| StructRTL | 0.3649 | 0.06 | 0.7463 | 0.5414 |
| GraphCodeBERT | 0.8424 | 0.15 | 0.5207 | 0.6109 |
| CircuitFusion | 0.7762 | 0.14 | 0.6175 | 0.5272 |
| **Ours** | **0.3510** | **0.06** | **0.7682** | **0.3384** |
| Ours (w/o code) | 0.3671 | 0.07 | 0.7546 | 0.3584 |
| Ours (w/o graph) | 0.8818 | 0.15 | 0.5173 | 0.6375 |

### Ablation Study: Code Retrieval

| Model | Precision↑ | Recall↑ | F1↑ |
|------|-----------|---------|---------|
| DeepRTL2-Llama | 0.557 | 0.608 | 0.572 |
| GraphCodeBERT | 0.616 | 0.675 | 0.634 |
| CircuitFusion | 0.542 | 0.608 | 0.560 |
| **Ours** | **0.650** | **0.692** | **0.662** |
| Ours (w/o graph) | 0.630 | 0.683 | 0.644 |

### Key Findings
- **Critical role of graphs**: Performance drops significantly when graphs are removed (w/o graph) (F1 from 0.662 → 0.644), proving the value of CDFG's complete information.
- **Complementary role of code**: Removing code (w/o code) leads to a slight performance decline (MAE from 0.3510 → 0.3671).
- **Effectiveness of alignment strategy**: Given the same graph-aware tokenizer, fine-grained mutual masking alignment far outperforms variable-level alignment and coarse contrastive learning.

## Highlights & Insights
- **CDFG Completeness**: Unlike GraphCodeBERT, which only uses data flow variables, UniRTL utilizes CDFG to retain complete elements such as operators and control flow.
- **Graph-aware Tokenizer Innovation**: By using a GIN+Transformer combination instead of direct node flattening, the model effectively captures "topological subtleties."
- **Hierarchy and Data Utility**: The design incorporates the varying abundance of pre-training data across different modalities; text-code warm-up fully utilizes data while preparing the model for graph fusion.

## Limitations & Future Work
- **CDFG Conversion Constraints**: Only 38.8k out of 132k designs could be successfully converted to CDFGs.
- **Language and Scale Limitations**: Currently limited to Verilog HDL; scalability for large-scale industrial designs has not been fully verified.
- **Future Directions**: Extending support to other HDLs like VHDL, expanding the graph-pair dataset, and covering a broader range of RTL tasks.

## Related Work & Insights
- **vs GraphCodeBERT**: Replaces incomplete data flow with full CDFG, introduces a graph-aware tokenizer, and uses mutual masking instead of variable alignment.
- **vs CircuitFusion**: UniRTL uses a unified Transformer to directly achieve fine-grained code-graph alignment, with CDFGs covering complete designs.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ (Systematic improvement in multimodal RTL representation learning)
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ (Covers two major downstream tasks, multiple configurations, and complete ablations)
- **Writing Quality**: ⭐⭐⭐⭐⭐ (Clear motivation and rigorous methodology)
- **Value**: ⭐⭐⭐⭐⭐ (Provides a general foundation model for hardware design automation, achieving SOTA in performance prediction and retrieval)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] ShieldedCode: Learning Robust Representations for Virtual Machine Protected Code](../../ICLR2026/code_intelligence/shieldedcode_learning_robust_representations_for_virtual_machine_protected_code.md)
- [\[ACL 2026\] OmniDiagram: Advancing Unified Diagram Code Generation via Visual Interrogation Reward](../../ACL2026/code_intelligence/omnidiagram_advancing_unified_diagram_code_generation_via_visual_interrogation_r.md)
- [\[ICLR 2026\] DRO-InstructZero: Distributionally Robust Prompt Optimization for Large Language Models](../../ICLR2026/code_intelligence/dro-instructzero_distributionally_robust_prompt_optimization_for_large_language_.md)
- [\[ICML 2026\] BoostAPR: Boosting Automated Program Repair via Execution-Grounded Reinforcement Learning with Dual Reward Models](boostapr_boosting_automated_program_repair_via_execution-grounded_reinforcement_.md)
- [\[AAAI 2026\] Towards Better Code Understanding in Decoder-Only Models with Contrastive Learning](../../AAAI2026/code_intelligence/towards_better_code_understanding_in_decoder-only_large_language_models_via_hie.md)

</div>

<!-- RELATED:END -->

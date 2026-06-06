---
title: >-
  [Paper Note] BugSweeper: Function-Level Detection of Smart Contract Vulnerabilities Using Graph Neural Networks
description: >-
  [AAAI2026][Graph Learning][Smart Contract] This paper proposes BugSweeper, which constructs function-level abstract syntax graphs (FLAG) and designs a two-stage GNN architecture to enable end-to-end smart contract vulner…
tags:
  - "AAAI2026"
  - "Graph Learning"
  - "Smart Contract"
  - "Vulnerability Detection"
  - "graph neural network"
  - "Abstract Syntax Tree"
  - "Pooling"
date: 2026-05-08
content_hash: 244f801b3702a523
---

# BugSweeper: Function-Level Detection of Smart Contract Vulnerabilities Using Graph Neural Networks

**Conference**: AAAI2026  
**arXiv**: [2512.09385](https://arxiv.org/abs/2512.09385)  
**Code**: To be confirmed  
**Area**: Graph Learning  
**Keywords**: Smart Contract, Vulnerability Detection, graph neural network, Abstract Syntax Tree, Pooling

## TL;DR

This paper proposes BugSweeper, which constructs function-level abstract syntax graphs (FLAG) and designs a two-stage GNN architecture to enable end-to-end smart contract vulnerability detection without expert-defined rules, achieving an F1 of 98.57% on reentrancy attack detection.

## Background & Motivation

Once deployed, Ethereum smart contracts cannot be modified. Security vulnerabilities — such as the notorious DAO attack that resulted in the theft of 3.6 million ETH — can have catastrophic consequences.
Existing detection methods fall into two categories:

1. **Traditional methods** (static analysis, symbolic execution): Tools such as Slither and Mythril rely on expert-crafted rules and exhibit poor generalization against novel vulnerability variants.
2. **Deep learning methods** (TMP, AME, Peculiar, ReVulDL): Although GNNs or pre-trained models are introduced, the preprocessing stage still depends on rule-based code snippet extraction, leading to the following issues:
    - **Limited detection coverage**: Predefined rules cannot cover all vulnerability types.
    - **Poor generalization**: Rules tailored to specific vulnerabilities are difficult to transfer.
    - **Information loss**: Rule-based extraction may discard critical contextual information.

The authors observe that security vulnerabilities in smart contracts — especially reentrancy attacks — typically stem from unsafe inter-function interactions, and therefore propose analysis at the **function level** while preserving inter-function call and reference relationships.

## Core Problem

How can a representation that simultaneously captures code structure and semantic flow be constructed without relying on any hand-crafted rules, enabling end-to-end automatic detection of multiple types of smart contract vulnerabilities?

## Method

BugSweeper consists of three core components:

### 1. Graph Constructor — Building the FLAG

- **AST Parsing**: The Solidity compiler solc is used to parse source code into an Abstract Syntax Tree (AST).
- **Edge Augmentation**: Three categories of edges are added on top of the AST to form a Flow-Augmented ASG:
    - **Basic edges**: Original Child/Parent structural edges from the AST.
    - **Data-flow edges**: Including ReferencedDeclaration (variable/function references), FunctionReturnParameter (return value links), SuperFunction (function overrides), and Assignment (assignments).
    - **Control-flow edges**: Including CondTrue/CondFalse for IfStatements, WhileExecution/ForExecution for loops, and NextStatement sequential edges.
- **Function-level partitioning + Coverage expansion**: The contract is split into subgraphs per function; a coverage hyperparameter controls neighborhood depth:
    - coverage=1: target function only.
    - coverage=2: adds directly called functions and referenced variables (1-hop).
    - coverage=3: further extends to 2-hop neighbors.
    - Default is coverage=4 in experiments.

### 2. Code Graph Neural Network (CGNN) — Stage One

- A BPE tokenizer encodes node text attributes into vectors.
- Three layers of GraphSAGE (512→1024→1024→1024) perform message passing to generate node embeddings.
- **CGPool (core innovation)**: Deterministic semantic pooling based on syntactic roles in code.
    - Nodes belonging to the same FunctionDefinition or VariableDeclaration are merged into a single supernode.
    - Supernodes are reconnected according to original control-flow/data-flow edges, producing a Pooled FLAG.
    - Compared to TopKPool/SAGPool, CGPool preserves hierarchical structure, avoids information loss, and is computationally efficient.

### 3. Second-Stage GNN — Stage Two

- Three layers of GAT (with multi-head attention; 4 heads in the first layer) perform high-level reasoning on the Pooled FLAG.
- The attention mechanism in GAT automatically focuses on critical inter-function connections.
- After global readout, a three-layer MLP classifier (1024→1024→C) outputs the probability for each vulnerability class.

### Training Details

- Optimizer: Adam, lr=1e-4, weight decay=1e-5.
- 500 training epochs, batch size=64.
- Dropout: 0.5 for GNN layers, 0.3 for the classifier.
- Each experiment is repeated with 4 different random seeds; mean values are reported.

## Key Experimental Results

### Reentrancy Attack Detection (AME dataset, 1,224 contracts)

| Method | Precision | Recall | F1 |
|--------|-----------|--------|----|
| Slither | 94.74% | 34.62% | 50.70% |
| AME | 95.45% | 95.38% | 95.42% |
| ReVulDL | 92.95% | 94.62% | 93.74% |
| **BugSweeper** | **99.87%** | **97.35%** | **98.57%** |

BugSweeper's F1 surpasses the strongest baseline (AME) by approximately **3.1 percentage points**, while Precision approaches 100%.

### Multi-Class Vulnerability Detection (SmartBugs Wild dataset, 47,398 Solidity files)

Best configuration SAGE + GAT:

| Vulnerability Type | F1 |
|-------------------|----|
| Reentrancy | 91.61% |
| Unchecked Low-Level Calls | 80.15% |
| Time Manipulation | 79.63% |

### Ablation Study

- **Two-stage vs. single-stage**: The two-stage architecture significantly outperforms single-stage baselines across all vulnerability types.
- **CGPool vs. other pooling methods**: CGPool (F1=87.32%) substantially outperforms ASAPool (82.41%), SAGPool (77.10%), and TopKPool (75.29%).
- **GNN combinations**: SAGE (stage one) + GAT (stage two) achieves the best performance, demonstrating complementarity — SAGE excels at aggregation over large graphs, while GAT's attention mechanism precisely identifies critical features.

## Highlights & Insights

1. **Fully end-to-end**: From source code to vulnerability detection without any hand-crafted rules, overcoming the bottleneck of rule-dependent preprocessing in existing deep learning methods.
2. **FLAG representation**: Integrates AST with control-flow/data-flow semantics, with a coverage mechanism that flexibly controls the amount of inter-function information.
3. **CGPool semantic pooling**: Cleverly leverages code structural priors (function/variable definition boundaries) for deterministic pooling, balancing efficiency with information preservation.
4. **Principled two-stage GNN design**: Stage one performs denoising; stage two conducts high-level reasoning, with SAGE and GAT each contributing their respective strengths.
5. **Multi-vulnerability generalization**: A unified framework detects three classes of vulnerabilities without requiring separately designed rules for each type.

## Limitations & Future Work

1. **Limited vulnerability types**: Only three vulnerability types are validated; more subtle logical vulnerabilities (e.g., access control issues, integer overflow) are not covered.
2. **Class imbalance has a notable impact**: F1 for Unchecked Low-Level Calls and Time Manipulation is markedly lower than for Reentrancy, indicating insufficient generalization to minority classes.
3. **Label quality in SmartBugs Wild**: Ground truth relies on consensus among 3+ tools, which may introduce systematic bias.
4. **Scalability not validated**: The method has not been tested on very large-scale contracts or complex DeFi protocols.
5. **No comparison with LLM-based methods**: Recent vulnerability detection approaches based on code large language models (e.g., CodeBERT, Code LLaMA) are not included in the comparison.
6. **Sensitivity to the coverage parameter**: Fixed at 4 without in-depth analysis of the accuracy–efficiency trade-off under different coverage settings.

## Related Work & Insights

| Dimension | Traditional Tools | Existing DL Methods | BugSweeper |
|-----------|------------------|---------------------|------------|
| Rule dependency | Fully dependent | Preprocessing dependent | **None** |
| Vulnerability types | Rule-bound coverage | Mostly single-class | **Multi-class unified** |
| Code representation | CFG/PDG | Expert-extracted graphs | **FLAG (auto-constructed)** |
| Information preservation | High loss | Partial loss | **Preserved via semantic pooling** |
| F1 (reentrancy) | 50–68% | 85–95% | **98.57%** |

**Transferability of CGPool**: The deterministic pooling strategy based on code structural priors can be generalized to other code analysis tasks such as defect prediction and code clone detection.  
**Generality of the FLAG construction pipeline**: The representation paradigm of AST + flow edges + function-level partitioning is not limited to Solidity and can be adapted to languages such as Python and Java.  
**Inspiration from the two-stage GNN architecture**: The denoise-then-reason paradigm is applicable to other graph learning scenarios with high noise levels, such as molecular graphs and circuit graphs.  
**Potential integration with LLMs**: Replacing the BPE tokenizer with a code large language model to generate initial node embeddings may further improve performance.

## Rating

- Novelty: ⭐⭐⭐⭐ — The FLAG representation and CGPool design are novel, and the two-stage architecture is well-motivated.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive ablation studies, multi-dataset validation, and complete statistical significance testing; however, the range of vulnerability types evaluated is limited.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, rich figures and tables, and thorough motivation.
- Value: ⭐⭐⭐⭐ — The end-to-end, rule-free detection paradigm has practical significance, and the accuracy improvement is substantial.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Learnable Kernel Density Estimation for Graphs and Its Application to Graph-Level Anomaly Detection](../../ICML2026/graph_learning/learnable_kernel_density_estimation_for_graphs_and_its_application_to_graph-leve.md)
- [\[AAAI 2026\] Adaptive Riemannian Graph Neural Networks](adaptive_riemannian_graph_neural_networks.md)
- [\[AAAI 2026\] Beyond Fixed Depth: Adaptive Graph Neural Networks for Node Classification Under Varying Homophily](beyond_fixed_depth_adaptive_graph_neural_networks_for_node_classification_under_.md)
- [\[AAAI 2026\] Sheaf Graph Neural Networks via PAC-Bayes Spectral Optimization](sheaf_graph_neural_networks_via_pac-bayes_spectral_optimization.md)
- [\[ICLR 2026\] Are We Measuring Oversmoothing in Graph Neural Networks Correctly?](../../ICLR2026/graph_learning/are_we_measuring_oversmoothing_in_graph_neural_networks_correctly.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] GALLa: Graph Aligned Large Language Models for Improved Source Code Understanding
description: >-
  [ACL 2025][Code Intelligence][code understanding] This paper proposes GALLa, which encodes the AST/DFG structural graph of code using a GNN and aligns it to the LLM embedding space via a cross-modal adapter. It injects code structural information as an auxiliary task during fine-tuning, and discards the GNN and adapter during inference to achieve zero extra overhead, yielding consistent improvements across 5 code tasks and 7 baseline LLMs (ranging from 350M to 14B parameters)…
tags:
  - "ACL 2025"
  - "Code Intelligence"
  - "code understanding"
  - "graph alignment"
  - "GNN"
  - "AST/DFG"
  - "model-agnostic"
date: 2026-05-08
content_hash: c410bd26746daf3b
---

# GALLa: Graph Aligned Large Language Models for Improved Source Code Understanding

**Conference**: ACL 2025  
**arXiv**: [2409.04183](https://arxiv.org/abs/2409.04183)  
**Code**: [https://github.com/codefuse-ai/GALLa](https://github.com/codefuse-ai/GALLa)  
**Area**: Code Intelligence  
**Keywords**: code understanding, graph alignment, GNN, AST/DFG, model-agnostic

## TL;DR
This paper proposes GALLa, which encodes the AST/DFG structural graph of code using a GNN and aligns it to the LLM embedding space via a cross-modal adapter. It injects code structural information as an auxiliary task during fine-tuning, and discards the GNN and adapter during inference to achieve zero extra overhead, yielding consistent improvements across 5 code tasks and 7 baseline LLMs (ranging from 350M to 14B parameters).

## Background & Motivation

**Background**: Code LLMs (such as Code LLaMA, DeepSeek-Coder, etc.) have made significant progress by scaling up model and data sizes, but essentially remain standard decoder-only Transformers that treat source code merely as sequences of text tokens.

**Limitations of Prior Work**: Code contains rich graph-structural semantics (AST for Abstract Syntax Trees, DFG for Data Flow Graphs, CFG for Control Flow Graphs) which are ignored in pure text representations. Existing graph-enhancement methods fall into three categories: (1) modifying the attention mask to encode graph structures (e.g., GraphCodeBERT, which is incompatible with decoder-only LLMs); (2) linearizing graphs into text (only applicable to simple trees like AST, and not suitable for cyclic DFGs); (3) modifying positional encodings (requiring large-scale retraining).

**Key Challenge**: Graphic enhancement and large-scale pretrained LLMs are incompatible—either the architecture is modified, losing pretrained knowledge, or the graph is omitted, losing structural information.

**Key Insight**: Inspired by the VLM field (e.g., LLaVA) which uses lightweight adapters to bridge vision and language modalities, an external GNN + adapter is used to handle graph information while keeping the LLM architecture completely unchanged.

**Core Idea**: Graph information is injected into the LLM during training via GNN+adapter, and discarded during inference—meaning the graph is present during training, but zero overhead is incurred during inference.

## Method

### Overall Architecture
Three modules: GNN (encoding AST/DFG) $\to$ Adapter (projecting into the LLM embedding space) $\to$ LLM (decoding/generation). Training is split into two stages, and the data is divided into graph alignment data and downstream task data, which are completely independent.

### Key Designs

1. **Graph Encoding and Input**:

    - GNN encodes the node features (initialized using a text encoder) and edge structures of AST and DFG, outputting contextualized node representations $H \in \mathbb{R}^{n_v \times d_{gnn}}$.
    - The adapter uses learnable query vectors to compress node representations into a fixed number of graph tokens $X_g \in \mathbb{R}^{n_g \times d_{lm}}$ via cross-attention.
    - Graph tokens and text tokens are concatenated and input into the LLM, with the loss calculated solely on the text tokens.

2. **Two-Stage Training**:

    - **Stage 1 (Graph Encoder Pre-training)**: Freeze the LLM, and train only the GNN + adapter. Task: Graph2Code (generate source code given a graph) to let the GNN+adapter learn to generate graph tokens intelligible to the LLM.
    - **Stage 2 (Graph-LLM Alignment)**: Unfreeze the LLM, and perform joint training of all three modules. Tasks: Graph2Code + GraphQA (predicting edge existence, predicting parent/child nodes). **Simultaneously**, fine-tune the LLM on downstream task data (not passing through the GNN, directly using text input).

3. **Discarding GNN during Inference**:

    - After training is complete, the GNN and adapter are discarded, and the LLM performs inference independently—resulting in identical speed to the baseline LLM.
    - Key assumption: Through graph alignment training, the LLM has already internally acquired code structure understanding capabilities, eliminating the need for explicit graph input during inference.

4. **Model- and Task-Agnosticism**:

    - The choice of GNN depends on the graph type (directed/undirected), and the choice of LLM depends on the application scenario—the framework itself imposes no constraints.
    - Graph alignment data comes from CodeNet (240K Python + 75K Java), which is entirely independent of downstream task data.

## Key Experimental Results

### Main Results (Multi-Task Fine-Tuning - MFT)

| Model | Without GALLa | G2C | G2C+GraphQA | Average Gain |
|------|:---:|:---:|:---:|:---:|
| CodeGen 350M | 34.7 | 35.2 (+1%) | 36.6 (+5%) | +5% |
| StarCoder 1B | 14.0 | 16.7 (+20%) | 18.9 (+36%) | +36% |
| Phi-1 1.3B | Baseline | Improvement | Larger Improvement | Significant |
| LLaMA3-8B | Baseline | Improvement | Improvement | Consistent |
| Qwen2.5-Coder-7B | Baseline | Improvement | Improvement | Consistent |

**5 Tasks**: Code Translation (pass@1), Clone Detection (F1), Defect Detection (Acc), Code Summarization (BLEU), Code Refinement (EM).

### Ablation Study

| Configuration | Performance | Description |
|------|------|------|
| G2C Only | Effective | Basic graph understanding |
| G2C + GraphQA | Better | Deep structural understanding |
| AST only / DFG only | Both effective | Complementary to each other |
| Cross-lingual Generalization | Effective | Alignment data contains only Python+Java, but improvements are also seen in JavaScript/C |

### Key Findings
- **Smaller models benefit more**: StarCoder 1B achieves a 36% gain, whereas larger models see smaller but consistent improvements.
- **Cross-lingual structural knowledge transfer**: Programming languages not included in the graph alignment data also benefit, indicating that LLMs have learned general structural understanding capabilities.
- **GraphQA is more critical than G2C**: QA-style structural tasks force the model to deeply understand the graph topology rather than simply memorize the graph-to-code mapping.

## Highlights & Insights
- **The "graphs during training, no graphs during inference" design philosophy** is highly practical: zero inference overhead means it can directly replace any existing code LLM.
- **Analogy of GNN as a "graph tokenizer"**: Stage 1 is equivalent to training a graph tokenizer (similar to vision tokenizers in VLMs), enabling the LLM to "comprehend" graph information.
- **Separation of graph alignment data and downstream task data**: Downstream task data does not require graph structure annotations, significantly lowering the barrier to adoption.

## Limitations & Future Work
- AST/DFG extraction requires syntactically valid and complete code, which is not applicable to code snippets or code with syntax errors.
- Only graph alignment data for Python and Java was tested; performance on other languages remains to be validated.
- Processing graphs of large code files with GNNs might encounter scalability issues.

## Related Work & Insights
- **vs GraphCodeBERT**: Encodes DFG by modifying attention masks, but is incompatible with decoder-only LLMs; GALLa handles this entirely externally.
- **vs TransCoder-IR**: Aligns using LLVM IR as an intermediate representation, but IR also consists of text tokens with limited structural information; GALLa directly operates on graph structures.

## Rating
- Novelty: ⭐⭐⭐⭐ An elegant bridging solution between graph structures and LLMs, with an ingenious "graphs in training, none in inference" design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 7 models (350M-14B) $\times$ 5 tasks + cross-lingual generalization validation.
- Writing Quality: ⭐⭐⭐⭐ Clear description of downstream methodology and coherent logic in the two-stage training.
- Value: ⭐⭐⭐⭐⭐ Zero inference overhead + model-agnostic + task-agnostic, providing highly practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] LongCodeU: Benchmarking Long-Context Language Models on Long Code Understanding](benchmarking_long-context_language_models_on_long_code_understanding.md)
- [\[ACL 2025\] Personality-Guided Code Generation Using Large Language Models](personality_guided_code_gen.md)
- [\[ACL 2025\] CodeReviewQA: The Code Review Comprehension Assessment for Large Language Models](codereviewqa_the_code_review_comprehension_assessment_for_large_language_models.md)
- [\[ACL 2025\] CodeDPO: Aligning Code Models with Self Generated and Verified Source Code](codedpo_code_alignment.md)
- [\[ACL 2025\] CodeIF: Benchmarking the Instruction-Following Capabilities of Large Language Models for Code Generation](codeif_benchmarking_the_instruction-following_capabilities_of_large_language_mod.md)

</div>

<!-- RELATED:END -->

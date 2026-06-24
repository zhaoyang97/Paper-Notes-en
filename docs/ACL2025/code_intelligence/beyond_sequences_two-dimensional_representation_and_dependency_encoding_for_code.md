---
title: >-
  [Paper Note] Beyond Sequences: Two-dimensional Representation and Dependency Encoding for Code Generation
description: >-
  [ACL 2025][Code Intelligence][Code Generation] This paper proposes a two-dimensional code representation that moves beyond traditional one-dimensional sequence representations. By explicitly encoding the structural dependency relationships of code (such as syntax tree structures and variable dependencies), it significantly improves the accuracy and structural correctness of code generation.
tags:
  - "ACL 2025"
  - "Code Intelligence"
  - "Code Generation"
  - "Two-dimensional Representation"
  - "Dependency Encoding"
  - "Abstract Syntax Tree"
  - "Structured Code Representation"
date: 2026-05-08
content_hash: 345a506a9a165b12
---

# Beyond Sequences: Two-dimensional Representation and Dependency Encoding for Code Generation

**Conference**: ACL 2025  
**Code**: None  
**Area**: Code Intelligence  
**Keywords**: Code Generation, Two-dimensional Representation, Dependency Encoding, Abstract Syntax Tree, Structured Code Representation

## TL;DR
This paper proposes a two-dimensional code representation that moves beyond traditional one-dimensional sequence representations. By explicitly encoding the structural dependency relationships of code (such as syntax tree structures and variable dependencies), it significantly improves the accuracy and structural correctness of code generation.

## Background & Motivation

**Background**: Code generation is a popular direction in current LLM applications, with models such as Codex and CodeLlama achieving excellent performance on benchmarks like HumanEval and MBPP. Existing methods generally treat code as a one-dimensional character/token sequence.

**Limitations of Prior Work**: The fundamental difference between code and natural language is that code possesses a strict two-dimensional structure. Rich non-sequential dependencies exist alongside the linear token sequence, including AST (Abstract Syntax Tree) hierarchical structures, variable definition-use relationships, and control flow dependencies. Flattening code into a one-dimensional sequence discards these critical structural details.

**Key Challenge**: Existing sequence-to-sequence models tend to generate code that is syntactically correct but logically flawed (e.g., chaotic variable scopes, missing matching brackets, incomplete control flows) because the models lack an explicit understanding of code structure.

**Goal**: To design a representation method that simultaneously captures both the linear sequence information and non-linear structural dependencies of code, thereby improving the structural correctness of code generation.

**Key Insight**: Code is inherently a two-dimensional structure, spanning linear token sequences horizontally and syntactic/semantic dependencies vertically. A two-dimensional representation is required to encode both dimensions of information simultaneously.

**Core Idea**: To upgrade code representation from a one-dimensional sequence to a two-dimensional matrix, where one dimension encodes the token sequence and the other encodes structural dependency relationships. This allows the model to reference both the sequential context and structural constraints when generating each token.

## Method

### Overall Architecture
The input is a natural language problem description, and the output is the generated code. Building upon the standard Transformer encoder-decoder framework, 2D positional encoding and dependency-aware attention mechanisms are introduced. During generation, the model not only predicts the next token autoregressively but also maintains a structural constraint graph to ensure the structural validity of the generated code.

### Key Designs

1. **2D Code Representation**:

    - **Function**: Extends the one-dimensional token sequence into a two-dimensional representation encoding structural information.
    - **Mechanism**: For each code token, in addition to its positional encoding in the sequence, AST depth and type information are also incorporated. A token×token relationship matrix is constructed, where the $(i,j)$ position encodes the type of structural relationship between the $i$-th token and the $j$-th token (such as parent-child relationship, sibling relationship, variable dependency, etc.).
    - **Design Motivation**: One-dimensional positional encoding can only represent linear distance, failing to capture the hierarchical distance and dependency directions in tree structures.

2. **Dependency-Aware Attention**:

    - **Function**: Introduces code structural dependency information into attention computation.
    - **Mechanism**: Modifies the standard self-attention mechanism by adding a structural bias term to the attention weights. When a direct syntactic dependency or variable reference relationship exists between two tokens, their attention weight is enhanced. Formally, $\text{Attention}(Q,K,V) = \text{softmax}(\frac{QK^T}{\sqrt{d}} + B_{dep})V$, where $B_{dep}$ is the dependency bias derived from the relationship matrix.
    - **Design Motivation**: Allows the model to naturally "perceive" structurally related tokens during generation, without relying solely on indirect inference from sequential context.

3. **Structure-Constrained Decoding**:

    - **Function**: Ensures the structural validity of the output code during the generation process.
    - **Mechanism**: At each step of autoregressive decoding, a partial AST is maintained to restrict the candidate space for the next token based on the current syntactic state. For example, `if` must be followed by a conditional expression, and `def` must be followed by a function name. This is similar to grammar-constrained decoding but utilizes finer-grained dependency details.
    - **Design Motivation**: Pure autoregressive generation cannot guarantee syntactic correctness; structure-constrained decoding provides a hard guarantee.

### Loss & Training
The model is trained using the standard cross-entropy loss for sequence generation, potentially augmented with a structural-consistency auxiliary loss to encourage the model to learn structural information.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours | CodeLlama | StarCoder | Gain |
|--------|------|---------|-----------|-----------|------|
| HumanEval | Pass@1 | Significant Improvement | Baseline | Baseline | +3-5% |
| MBPP | Pass@1 | Significant Improvement | Baseline | Baseline | +2-4% |
| APPS | Pass@1 | Significant Improvement | Baseline | Baseline | +3-6% |
| Syntactic Correctness Rate | — | ~98% | ~92% | ~91% | +6-7% |

### Ablation Study

| Configuration | HumanEval Pass@1 | Description |
|------|------------------|------|
| Full model | Best | 2D representation + dependency attention + constrained decoding |
| w/o 2D representation | Drop | Degenerates to 1D sequence |
| w/o Dependency attention | Slight drop | No structural bias |
| w/o Constrained decoding | Significant drop | Syntax error rate increases |
| Dependency attention only | Moderate improvement | Minimalist version is also effective |

### Key Findings
- The 2D representation shows the most significant improvement for long code (>100 lines), where long-range dependencies are more prevalent.
- Structure-constrained decoding is the primary contributor to syntactic correctness improvements, but has a limited contribution to logical correctness.
- Encoding variable dependencies contributes the most, surpassing the contribution of AST hierarchical information, indicating that variable reference is a key challenge in code generation.
- Benefits are less pronounced on simple code generation tasks; significant improvements are primarily observed in complex tasks (e.g., recursion, nested loops).

## Highlights & Insights
- **Dimension Expansion Concept**: The concept of upgrading from one-dimensional to two-dimensional representation is simple yet effective, and can be transferred to other sequence generation tasks with structural properties, such as mathematical expression generation, SQL generation, etc.
- **Highly Interpretable Relationship Matrix**: The 2D relationship matrix can intuitively visualize the structural relationships of code, making it easy to understand the model's generation logic.

## Limitations & Future Work
- The 2D representation and constrained decoding increase computational overhead, which may be impractical for extremely long code.
- A predefined set of relationship types is required, limiting adaptability to new programming languages or unconventional syntax.
- Currently, it mainly targets Python; the effectiveness on other programming languages (e.g., strongly-typed languages like C/Rust) remains to be validated.
- Future research can combine the code understanding capabilities of LLMs with the proposed structural encoding to construct structure-aware code LLMs.

## Related Work & Insights
- **vs TreeGen（AST级代码生成）**: TreeGen generates directly on the AST but suffers from low training and inference efficiency. Ours introduces structural information into a sequence generation framework, achieving a better balance between efficiency and effectiveness.
- **vs SynCoBERT（结构感知预训练）**: SynCoBERT introduces structural info during the pre-training stage, while ours introduces it during the generation stage, making the two approaches complementary.
- **vs GraphCodeBERT**: GraphCodeBERT leverages data flow graphs to enhance code understanding; ours further applies graph structure information to the generation stage, serving as a natural extension from understanding to generation.

## Rating
- Novelty: ⭐⭐⭐⭐ The 2D representation is a novel attempt in code generation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-dataset evaluation and comprehensive ablation.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and detailed method description.
- Value: ⭐⭐⭐⭐ Inspiring for structured sequence generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] The Natural Geometry of Code: Hyperbolic Representation Learning for Program Reasoning](../../ICLR2026/code_intelligence/the_natural_geometry_of_code_hyperbolic_representation_learning_for_program_reas.md)
- [\[ACL 2025\] Rethinking Repetition Problems of LLMs in Code Generation](rethinking_repetition_problems_of_llms_in_code_generation.md)
- [\[ACL 2025\] GiFT: Gibbs Fine-Tuning for Code Generation](gift_gibbs_fine_tuning_code_gen.md)
- [\[ACL 2025\] Personality-Guided Code Generation Using Large Language Models](personality_guided_code_gen.md)
- [\[ACL 2025\] Revisit Self-Debugging with Self-Generated Tests for Code Generation](revisit_self-debugging_with_self-generated_tests_for_code_generation.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] EpiCoder: Encompassing Diversity and Complexity in Code Generation
description: >-
  [ICML 2025][Code Intelligence][Code Generation] This paper proposes a code data synthesis framework based on "Feature Trees." By extracting hierarchical semantic features from code and iteratively evolving them, the framework achieves precise control over the complexity and diversity of synthetic data. The resulting trained EpiCoder series of models achieves state-of-the-art (SOTA) performance among similarly sized models on both function-level and file-level code generation…
tags:
  - "ICML 2025"
  - "Code Intelligence"
  - "Code Generation"
  - "Feature Tree"
  - "Data Synthesis"
  - "Instruction Tuning"
  - "Code LLMs"
date: 2026-05-08
content_hash: b25d65700012172f
---

# EpiCoder: Encompassing Diversity and Complexity in Code Generation

**Conference**: ICML 2025  
**arXiv**: [2501.04694](https://arxiv.org/abs/2501.04694)  
**Code**: [microsoft/EpiCoder](https://github.com/microsoft/EpiCoder)  
**Area**: Code Intelligence  
**Keywords**: Code Generation, Feature Tree, Data Synthesis, Instruction Tuning, Code LLMs

## TL;DR

This paper proposes a code data synthesis framework based on "Feature Trees." By extracting hierarchical semantic features from code and iteratively evolving them, the framework achieves precise control over the complexity and diversity of synthetic data. The resulting trained EpiCoder series of models achieves state-of-the-art (SOTA) performance among similarly sized models on both function-level and file-level code generation benchmarks.

## Background & Motivation

Existing code data synthesis methods (such as Magicoder's OSS-Instruct and WizardCoder's Evol-Instruct) commonly use **code snippets** as seed data, which presents two core limitations:

**Limited Diversity**: Code snippets themselves are rigid and difficult to reorganize flexibly to produce new combinations, leading to insufficient coverage of programming patterns in synthetic data.

**Limited Complexity**: Snippet-based methods struggle to generate complex code for real-world scenarios, such as cross-file dependencies and multi-module collaborations.

These limitations leave fine-tuned models underpowered when handling complex programming tasks. Inspired by Abstract Syntax Trees (ASTs), the authors propose replacing code snippets with a **hierarchical semantic Feature Tree** as seed data to fundamentally resolve these diversity and complexity bottlenecks.

## Method

### Overall Architecture

The framework consists of three stages: **(a) Feature Tree Extraction** $\rightarrow$ **(b) Feature Tree Evolution** $\rightarrow$ **(c) Feature Tree-Based Code Generation**.

The core idea is: instead of generating new code directly from code snippets, the method first abstracts code into a hierarchical structure of semantic features (a Feature Tree), expands and evolves within this feature space, and then samples subtrees from the Feature Tree to guide code generation. This enables:
- **Controllable Complexity**: The complexity of the generated code is controlled by adjusting the depth and width of the sampled subtrees.
- **Targeted Learning**: By adjusting feature sampling probabilities, priority is given to covering domains where the model's knowledge base is lacking.

### Key Designs

#### 1. Feature Tree Extraction

**Raw Code Collection**: A diverse set of core code samples (150k Python files) is selected from The Stack v2 dataset using the KCenterGreedy algorithm, based on code embeddings encoded by `roberta-large-v1`.

**Tree Structure Exemplar Construction**: A two-step iterative approach is used to optimize the feature extraction prompt:
- **Feature Pre-extraction**: GPT-4o is used to extract an initial set of feature keywords from the seed code.
- **Iterative Clustering**: Hierarchical clustering is performed on feature subsets to generate tree structure exemplars, repeating adjustments to ensure logical hierarchical relationships.

**Feature Tree Extraction & Merging**: Using the optimized tree structure exemplars, GPT-4o extracts tree-structured feature representations for each code snippet, which are then merged into a unified comprehensive Feature Tree. Concurrently, the frequency of each node is recorded to reflect the distribution of features in the seed data (approximating the pre-trained model's knowledge distribution).

#### 2. Feature Tree Evolution

To overcome the limitations of the seed data's features, the Feature Tree is iteratively evolved and expanded:

- Each iteration samples a subtree from the complete Feature Tree.
- The LLM expands the subtree along two dimensions: **depth** (adding finer-grained child nodes) and **width** (adding sibling nodes at the same level).
- The evolved subtree is merged back into the overall structure.

**Key Challenge - Frequency Estimation for New Features**: The frequency of newly generated features is estimated as the average frequency of their sibling nodes, ensuring that evolved features seamlessly blend with the existing distribution. In experiments, after 9000 evolution steps, the number of features expanded from 5k to 140k.

#### 3. Feature Tree-Based Code Generation

**Distribution Reweighting**: The original feature frequencies reflect natural data distribution, but some high-frequency yet simple features (such as `config`, `initialize`) do not need heavy focus during the instruction tuning stage. A temperature parameter $t$ is used to adjust sampling probabilities:

$$p_i' = \frac{\exp(\log p_i / t)}{\sum_{j \in C} \exp(\log p_j / t)}$$

A higher temperature flattens the distribution, giving lower-frequency features a higher probability of being sampled. To enhance diversity, multiple temperature values are used during the data synthesis process.

**Feature Sampling**: Based on the adjusted probability distribution, candidate feature subtrees are recursively sampled according to predefined subtree shapes. By adjusting the depth and width of the subtrees, tasks of various complexities can be flexibly generated.

**Content Generation**: The LLM selects compatible feature subsets based on the sampled subtrees to generate task descriptions, corresponding code, and execution environments. Solution code can range from a single function to multi-file projects, supporting cross-file dependencies.

**Iterative Refinement**: When generating code, test files are produced synchronously and executed in an isolated environment. The LLM iteratively refixes the code guided by error messages, ensuring the correctness of the generated code.

### Loss & Training

- Base Models: Qwen2.5-Coder-7B-Base and DeepSeek-Coder-6.7B-Base
- Training Data: 380k function-level + 53k file-level data (433k in total)
- For the DeepSeek base model, the `evol-codealpaca-v1` dataset is additionally included (to align with baselines)
- Evaluations are benchmarked corresponding to the training levels

## Key Experimental Results

### Main Results

**Function-Level Code Generation (Pass@1 %)**:

| Model | HumanEval | HumanEval+ | MBPP | MBPP+ | BCB-Full Comp. | BCB-Hard Comp. | EvoEval | Average |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-Coder-7B-Instruct | 88.4 | 84.1 | 83.5 | 71.7 | 48.8 | 20.3 | 55.2 | 57.0 |
| **EpiCoder-Qwen-7B** | **89.0** | 82.3 | **84.1** | 71.4 | **51.9** | **27.7** | **58.8** | **59.0** |
| DeepSeekCoder-6.7b-Instruct | 74.4 | 71.3 | 74.9 | 65.6 | 43.8 | 15.5 | 41.4 | 48.1 |
| Magicoder-S-DS | 76.8 | 71.3 | 79.4 | 69.0 | 47.6 | 12.8 | 44.6 | 50.1 |
| **EpiCoder-DS-6.7B** | **80.5** | **76.8** | **81.5** | 68.3 | **50.6** | **19.6** | **50.0** | **53.1** |

EpiCoder-Qwen-7B achieves SOTA average performance at the 7B scale, outperforming Qwen2.5-Coder-7B-Instruct on BCB-Hard Completion by **7.4%**.

### Ablation Study

**Comparison under the same data scale (Pass@1 %)**:

| Configuration | Data Volume | HumanEval | MBPP | BCB-Full | BCB-Hard | EvoEval | Average | Description |
|---|---|---|---|---|---|---|---|---|
| Magicoder-DS | 75k | 66.5 | 75.4 | 46.8 | 13.5 | 41.2 | 45.8 | Snippet-based OSS-Instruct |
| WaveCoder-Ultra-6.7B | 130k | 75.0 | 74.9 | 43.7 | 16.9 | 43.6 | 48.2 | Generator-discriminator framework |
| **EpiCoder-DS-6.7B-75k** | **75k** | **78.0** | **79.4** | **48.2** | **18.4** | **46.2** | **51.2** | **Feature tree method, superior under equivalent data volume** |
| SelfCodeAlign-CQ-7B | 74k | — | — | — | — | — | 58.6 | Based on code concepts |
| **EpiCoder-CodeQwen-74k** | **74k** | — | — | — | — | — | **62.6** | **+4% average gain** |

Given the same volume of data, EpiCoder improves upon Magicoder and WaveCoder by 5.4% and 3.0% respectively, demonstrating that the data quality advantage stems from the Feature Tree method rather than data scale.

### Key Findings

1. **Significant Complexity Increase**: From software engineering metrics, the Halstead complexity of EpiCoder’s function-level data (Unique Operands: 44.32 vs OSS-Instruct 20.99) is nearly doubled; the LLM-evaluated complexity increases by 32.6% (function-level) and 52.5% (file-level) compared to OSS-Instruct.
2. **Leading Diversity**: The average number of unique features per sample reaches 8.53 (function-level) / 8.95 (file-level), outperforming the nearest competitor by over 2.15.
3. **Good Data Scaling Effect**: Performance still trends upward at a data volume of 380k, indicating that the dataset diversity is sufficient to prevent overfitting.
4. **No Data Leakage**: Embedding cosine similarity analysis shows that the similarity between the synthetic data and benchmark test sets is far below the 0.9 threshold.
5. **Repository-Level Generation Potential**: Mock repositories containing 50+ files were successfully generated from the Feature Tree of LLaMA-Factory.

## Highlights & Insights

1. **Paradigm Shift from "Code-to-Code" to "Semantic-to-Code"**: Instead of generating new code directly from code snippets, the approach lifts generation to the semantic feature level first, breaking the rigidity constraints of code snippets.
2. **Feature Tree as a Unified Interface**: Depth controls complexity, width controls diversity, and frequency controls sampling preferences — a single structure provides three independent dimensions of control.
3. **Evolving in Feature Space rather than Code Space**: This is more efficient than directly evolving code or instructions, as the tree structure provides clear directions.
4. **Proposing the XFileDep Benchmark**: Fills the gap in file-level code generation evaluation, consisting of 466 cross-file dependency problems.
5. **Clever Design in Frequency Estimation**: Estimating newly evolved feature frequencies using the average frequency of their sibling nodes maintains distribution consistency simply yet effectively.

## Limitations & Future Work

1. **Feature Tree Construction Depends on Strong LLMs**: The entire pipeline relies on GPT-4o for feature extraction and evolution, which entails relatively high costs.
2. **Repository-Level Generation is Still a Proof of Concept**: Although generation cases of 50+ file repositories are demonstrated, systematic quantitative evaluation is still missing.
3. **Only Python is Covered**: Seed data and evaluations are predominantly in Python, and multi-lingual generalization capabilities have not been validated.
4. **Upper Bound of Feature Tree Quality**: The initial quality of the Feature Tree depends on seed data selection and the LLM's extraction capability.
5. **Scalability to Larger Models**: Validated only at the 7B scale; larger-scale models might yield even greater gains.

## Related Work & Insights

- **Evolution of Data Synthesis**: Code Alpaca $\rightarrow$ WizardCoder (Evol-Instruct) $\rightarrow$ Magicoder (OSS-Instruct) $\rightarrow$ WaveCoder $\rightarrow$ SelfCodeAlign $\rightarrow$ **EpiCoder (Feature Tree)**, reflecting a clear trend from simple to structured data synthesis.
- **Inspirational Directions**: The concept of Feature Trees can be migrated to data synthesis in other domains such as mathematical reasoning and scientific computing; repository-level and cross-file dependency benchmark construction methods can be generalized to more programming languages.
- **Difference from SelfCodeAlign**: While the latter also extracts "code concepts," EpiCoder organizes concepts into a tree structure, providing hierarchical relations and a more systematic direction for evolution.

## Rating

| Dimension | Score (1-5) | Description |
|---|---|---|
| Novelty | ⭐⭐⭐⭐ | The Feature Tree represents a new paradigm in code data synthesis, controlling complexity at the semantic level. |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ | Assessed across 5 function-level + 1 file-level benchmarks with equal-volume comparisons, including comprehensive analysis of complexity, diversity, and leakage. |
| Writing Quality | ⭐⭐⭐⭐ | Clear framework, rich diagrams, and detailed appendices containing complete prompts. |
| Value | ⭐⭐⭐⭐ | Open-sourced code; the method can be directly applied to data synthesis for code LLMs. |
| Overall Rating | ⭐⭐⭐⭐ | Solid work that takes a significant step forward in the field of code data synthesis. |

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] DynaCode: A Dynamic Complexity-Aware Code Benchmark for Evaluating Large Language Models in Code Generation](../../ACL2025/code_intelligence/dynacode_a_dynamic_complexity-aware_code_benchmark_for_evaluating_large_language.md)
- [\[ICML 2025\] Reasoning Through Execution: Unifying Process and Outcome Rewards for Code Generation](reasoning_through_execution_unifying_process_and_outcome_rewards_for_code_genera.md)
- [\[ICML 2025\] EffiCoder: Enhancing Code Generation in Large Language Models through Efficiency-Aware Fine-tuning](efficoder_enhancing_code_generation_in_large_language_models_through_efficiency-.md)
- [\[NeurIPS 2025\] Embedding Alignment in Code Generation for Audio](../../NeurIPS2025/code_intelligence/embedding_alignment_in_code_generation_for_audio.md)
- [\[ACL 2025\] GiFT: Gibbs Fine-Tuning for Code Generation](../../ACL2025/code_intelligence/gift_gibbs_fine_tuning_code_gen.md)

</div>

<!-- RELATED:END -->

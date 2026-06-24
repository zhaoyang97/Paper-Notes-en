---
title: >-
  [Paper Note] SCULPT: Systematic Tuning of Long Prompts
description: >-
  [ACL 2025][LLM (Other)][prompt optimization] This paper proposes the SCULPT framework, which models long prompt optimization as an iterative modification problem on a hierarchical tree structure. Through a Critic-Actor framework, it conducts structured reflection and operation-level modifications on the prompt, significantly improving LLM task performance while maintaining long prompt information integrity and possessing robustness against adversarial perturbations.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "prompt optimization"
  - "long prompts"
  - "hierarchical tree structure"
  - "critic-actor framework"
  - "prompt engineering"
date: 2026-05-08
content_hash: 9e0c910abd45134d
---

# SCULPT: Systematic Tuning of Long Prompts

**Conference**: ACL 2025  
**arXiv**: [2410.20788](https://arxiv.org/abs/2410.20788)  
**Code**: None  
**Area**: LLM NLP  
**Keywords**: prompt optimization, long prompts, hierarchical tree structure, critic-actor framework, prompt engineering  

## TL;DR

This paper proposes the SCULPT framework, which models long prompt optimization as an iterative modification problem on a hierarchical tree structure. Through a Critic-Actor framework, it conducts structured reflection and operation-level modifications on the prompt, significantly improving LLM task performance while maintaining long prompt information integrity and possessing robustness against adversarial perturbations.

## Background & Motivation

**Background**: Prompt optimization is a crucial technique for improving the performance of Large Language Models (LLMs) on various downstream tasks. Existing approaches, such as APE (Automatic Prompt Engineer) and EvoPrompt, primarily focus on optimizing short prompts by replacing words, reordering sentences, or adding/deleting instructions to enhance performance.

**Limitations of Prior Work**: However, prompts in real-world applications are often long and structurally complex (comprising task descriptions, formatting requirements, examples, constraints, etc.). Existing short-prompt optimization methods face severe challenges when handling such long prompts: (1) global modifications easily lose critical information; (2) prompts are highly sensitive to minor perturbations, where adjusting one paragraph may affect the semantics of the entire prompt; (3) they cannot generate new structural content to remedy prompt deficiencies.

**Key Challenge**: The optimization space for long prompts is vast—every paragraph and sentence could be a target for optimization. However, existing methods lack an effective structure to manage this complexity. Searching directly on flat text is highly inefficient and prone to falling into local optima.

**Goal**: Design a systematic optimization framework specifically for long prompts that can: (1) perform precise modifications without losing information; (2) remain robust against noise and perturbations; (3) generate high-quality prompts from scratch even in the absence of an initial human-written prompt.

**Key Insight**: The authors observe that long prompts naturally possess a hierarchical structure (title -> paragraph -> sentence), which can be modeled as a tree structure. Modifications performed on a tree are inherently local—modifying a leaf node does not affect distant content—providing hierarchical guarantees for precise and robust optimization.

**Core Idea**: Represent long prompts as hierarchical tree structures, use a Critic module to generate modification suggestions, and use an Actor module to execute precise operations on specific nodes of the tree.

## Method

### Overall Architecture

SCULPT organizes long prompts into a hierarchical tree, where different levels of the tree correspond to different granularities of the prompt (e.g., sections -> paragraphs -> sentences). The optimization process is an iterative loop: in each round, the Critic module evaluates the performance of the current prompt and generates structured reflections (what is wrong, why it is wrong, and suggested modifications). Then, the Actor module executes specific modification operations (such as swapping sentences, inserting new paragraphs, or deleting redundant content) on designated nodes of the tree based on the reflections. The modified prompt is evaluated on a validation set; if performance improves, the modification is accepted, otherwise, it is rolled back.

### Key Designs

1. **Hierarchical Tree Representation**:

    - Function: Organizes long prompts into a structured tree, supporting precise modifications at different granularities.
    - Mechanism: Parses the prompt top-down into a tree structure—where the root node is the complete prompt, intermediate nodes are distinct paragraphs or functional modules (e.g., "task description", "format requirements", "examples"), and leaf nodes are concrete sentences. Each node carries its functional description and text content. Modification operations are defined on specific tree nodes, including Swap (replacing node content), Insert (inserting a new node), Delete (removing a redundant node), and Paraphrase (rewriting node content). The tree structure guarantees the locality of modifications—modifying one subtree does not affect other subtrees.
    - Design Motivation: Flat text lacks structural information, making single-point modifications prone to unpredictable side effects ("pulling one hair moves the whole body"). Tree structures decompose global optimization into local optimization, improving search efficiency while reducing the risk of information loss.

2. **Critic Module (Reflection Generation)**:

    - Function: Evaluates the performance of the current prompt and generates structured modification suggestions.
    - Mechanism: Taking the current prompt and its performance on the validation set as inputs, the Critic utilizes an LLM to "reflect"—analyzing where performance bottlenecks exist in the prompt, why they occur, and what types of modifications are recommended. The Critic outputs structural advice specifying which tree nodes should be modified, the type of issue, and the suggested direction of modification. This reflection mechanism ensures that modifications are targeted rather than blind searches.
    - Design Motivation: Blind, random modifications are highly inefficient. By adopting a "diagnosis before treatment" paradigm, the Critic elevates the optimization process from a random search to directed improvement.

3. **Actor Module (Operation Execution)**:

    - Function: Executes specific modification operations on the tree structure based on the Critic's suggestions.
    - Mechanism: The Actor receives modification suggestions from the Critic and performs corresponding operations on the designated tree nodes. Supported operations include replacing node content, inserting new nodes at specific positions, deleting redundant nodes, and paraphrasing node content. The Actor outputs a new modified tree, which is converted back into prompt text for evaluation. If the performance on the validation set improves, the modification is kept and the loop proceeds to the next round; otherwise, it rolls back to the previous version.
    - Design Motivation: Restricting modification operations to local tree nodes prevents the loss of information common in global modifications. The diversity of operation types (insertion, deletion, modification) allows the system to flexibly address different optimization needs.

### Loss & Training

SCULPT is a training-free, inference-time framework. The optimization process judges the effectiveness of each modification based on task performance metrics (such as accuracy) on a validation set. The framework employs a greedy strategy—retaining the modification that yields the highest performance improvement in each round.

## Key Experimental Results

### Main Results

Evaluation is conducted on multiple NLP benchmark tasks, covering classification, question answering, and generation tasks, using various LLMs such as GPT-3.5/4.

| Method | BBH (avg) | Instruction Following | Classification Tasks | Description |
|------|-----------|---------|---------|------|
| Original Long Prompt | Baseline | Baseline | Baseline | Human-written |
| APE | Lower than SCULPT | Lower than SCULPT | Lower than SCULPT | Short prompt optimization method |
| EvoPrompt | Lower than SCULPT | Lower than SCULPT | Lower than SCULPT | Evolutionary search |
| OPRO | Lower than SCULPT | Lower than SCULPT | Medium | LLM self-optimization |
| **SCULPT** | **Highest** | **Highest** | **Highest** | Ours |
| SCULPT (From Scratch) | Close to or exceeding human | Highly competitive | Good | No initial prompt required |

### Ablation Study

| Configuration | Average Performance | Description |
|------|---------|------|
| Full SCULPT | Highest | Complete framework |
| w/o Critic | Significant drop | Modifications become random without reflection |
| w/o Tree Structure (Flat modification) | Prominent drop | Lack of structured modification reduces efficiency |
| w/o Iteration (Single-round modification) | Moderate drop | Single modification is insufficient for full optimization |
| Fixed Operation Type (Only Swap) | Drop | Single operation type is less flexible |
| Different Tree Depths | Medium depth is optimal | Too shallow -> coarse modification granularity; too deep -> large search space |

### Key Findings

- **Critic module contributes the most**: Removing the Critic leads to the most significant drop in performance, demonstrating that "directed reflection" is vastly more efficient than "random search".
- **Tree structure is key to robustness**: In adversarial perturbation experiments, SCULPT's performance fluctuation is substantially smaller than that of flat methods, because the tree structure isolates perturbations locally.
- **Impressive capability to generate from scratch**: Without any initial prompt, SCULPT can generate prompts that are close to or even outperform human-written prompts through multi-round iterations.
- **Diminishing returns of iterative rounds**: Performance typically stabilizes after 3–5 iterations, with more iterations yielding marginal improvements.

## Highlights & Insights

- **Modeling prompts as tree structures is the core innovation**: Explicitly modeling the hierarchical structure of prompts as a tree guarantees that modifications are naturally local and structured. This concept is simple yet powerful—optimizing long text is essentially finding the optimal configuration in a hierarchical structure.
- **Decoupled Critic-Actor design**: Splitting "diagnosis" and "treatment" into two separate steps allows each step to be more focused. This paradigm can be extended to other iterative optimization tasks, such as code optimization and article editing.
- **Robustness is crucial for practical applications**: Existing prompt optimization methods are overly sensitive to minor changes, making them unstable in practice. SCULPT's structured approach provides more reliable optimization results.

## Limitations & Future Work

- Since each iteration requires evaluation on a validation set, the total optimization cost can be quite high when LLM inference costs are expensive and the validation set is large.
- The initial parsing of the tree structure depends on prompts having a reasonable hierarchical structure, which may not apply well to unstructured prompts.
- Both Critic and Actor rely on the capabilities of the LLM. If the LLM itself cannot diagnose issues correctly, the optimization direction may be misguided.
- Future work could explore combining SCULPT with in-context learning optimization, simultaneously optimizing prompt structure and example selection.
- Multi-objective optimization (e.g., performance + conciseness + cost) could be introduced to prevent excessive bloat of the prompt.

## Related Work & Insights

- **vs APE (Automatic Prompt Engineer)**: APE optimizes short prompts via searching and evaluating but is powerless for long prompts. SCULPT addresses the manageability of long prompts using tree structures.
- **vs OPRO**: OPRO lets the LLM optimize its own prompts, but relies on global modifications without structural guarantees. SCULPT's Critic-Actor framework is more fine-grained.
- **vs DSPy**: DSPy focuses on programmatic abstraction and compiling optimization of prompts, whereas SCULPT is more concerned with structured corrections of natural language prompts. The paradigms of both could be combined.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of tree-structure modeling and the Critic-Actor framework is novel in the field of prompt optimization, though individual components are not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple task types, conducts robustness and ablation studies, but lacks a extensive comparison across a broader range of LLMs.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, systematic framework description, and high-quality figures and tables.
- Value: ⭐⭐⭐⭐ Fills a gap in long prompt optimization and provides valuable insights for practical prompt engineering.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] P3: Prompts Promote Prompting](p3_prompts_promote_prompting.md)
- [\[ACL 2025\] Systematic Generalization in Language Models Scales with Information Entropy](systematic_generalization_in_language_models_scales_with_information_entropy.md)
- [\[ACL 2025\] A Systematic Study of Compositional Syntactic Transformer Language Models](a_systematic_study_of_compositional_syntactic_transformer_language_models.md)
- [\[ACL 2025\] TestCase-Eval: A Systematic Evaluation of Fault Coverage and Exposure](testcase_eval_llm_test_gen.md)
- [\[ACL 2025\] X-Turing: Towards an Enhanced and Efficient Turing Test for Long-Term Dialogue Agents](xturing_enhanced_turing_test.md)

</div>

<!-- RELATED:END -->

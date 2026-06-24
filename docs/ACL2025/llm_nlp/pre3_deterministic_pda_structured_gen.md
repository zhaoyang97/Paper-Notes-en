---
title: >-
  [Paper Note] Pre³: Enabling Deterministic Pushdown Automata for Faster Structured LLM Generation
description: >-
  [ACL 2025][LLM (Other)][Structured Generation] Pre³ is proposed to transform LR(1) grammars into Deterministic Pushdown Automata (DPDA). By precomputing prefix-conditioned edges to eliminate runtime non-deterministic exploration, it significantly accelerates structured LLM generation, reducing per-token latency by up to 40% and improving throughput by up to 36%.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Structured Generation"
  - "Constrained Decoding"
  - "Deterministic Pushdown Automata"
  - "JSON Generation"
  - "LLM Inference Optimization"
date: 2026-05-08
content_hash: a78bd2ee4463ebe4
---

# Pre³: Enabling Deterministic Pushdown Automata for Faster Structured LLM Generation

**Conference**: ACL 2025  
**arXiv**: [2506.03887](https://arxiv.org/abs/2506.03887)  
**Code**: [https://github.com/ModelTC/lightllm](https://github.com/ModelTC/lightllm)  
**Area**: LLM/NLP  
**Keywords**: Structured Generation, Constrained Decoding, Deterministic Pushdown Automata, JSON Generation, LLM Inference Optimization  

## TL;DR
Pre³ is proposed to transform LR(1) grammars into Deterministic Pushdown Automata (DPDA). By precomputing prefix-conditioned edges to eliminate runtime non-deterministic exploration, it significantly accelerates structured LLM generation, reducing per-token latency by up to 40% and improving throughput by up to 36%.

## Background & Motivation
**Background**: Structured generation (e.g., JSON output) is in high demand for LLM applications. Constrained decoding enforces format compliance by masking invalid tokens at each generation step. Existing methods (such as XGrammar, Outlines, etc.) parse LR(1) grammars into Pushdown Automata (PDA) for constraint enforcement.

**Limitations of Prior Work**: Transition edges in traditional PDAs are non-deterministic—the same input symbol can trigger multiple transitions, requiring runtime backtracking, speculative execution, and the maintenance of persistent stack-tree structures. These overheads grow significantly as inference batch sizes increase (e.g., at batch=512, XGrammar is 60% slower than unconstrained decoding).

**Key Challenge**: Non-deterministic transitions hinder preprocessing optimizations—since transitions can only be determined at runtime, complete analysis and optimization cannot be performed during the preprocessing phase.

**Goal**: To convert the automaton used for constrained decoding from a non-deterministic PDA to a deterministic PDA (DPDA), thereby eliminating the runtime exploration overhead.

**Key Insight**: LR(1) grammars are naturally a subset of Deterministic Context-Free Languages (DCFL), which theoretically can be recognized by DPDAs. The key lies in designing an efficient construction algorithm to transition from an LR(1) state transition graph to a DPDA.

**Core Idea**: Attach stack matching conditions to each transition edge using prefix-conditioned edges, making all transitions completely deterministic during the preprocessing phase.

## Method

### Overall Architecture
The input is an LR(1) grammar (e.g., JSON Schema), processed in three stages: (1) constructing the LR(1) state transition graph; (2) converting it into a DPDA via the prefix-conditioned edge algorithm; and (3) optimizing the DPDA through edge aggregation/merging. The output is the optimized DPDA, which is directly applicable to constrained decoding.

### Key Designs

1. **Prefix-conditioned Edge**:

    - **Function**: Attaches stack matching conditions to each transition edge to eliminate non-determinism.
    - **Mechanism**: Each edge consists of three components: the accepted symbol (input triggering the transition), the stack matching conditions (the stack prefix required for the transition to take effect), and the stack operation (push/pop). By simultaneously checking the input symbol and stack state, the target state is uniquely determined.
    - **Design Motivation**: The core property of LR(1) is that "the current stack configuration + a lookahead symbol uniquely determines the next action." Prefix-conditioned edges encode this property into the edge structure.

2. **Loop-aware DPDA Construction Algorithm**:

    - **Function**: Systematically constructs a DPDA from an LR(1) state transition graph.
    - **Mechanism**: Defines two types of edges: acceptance edges (corresponding to shift operations, directly inherited from the original graph) and reduction edges (corresponding to reduce operations, where nested reduction paths are precomputed into single-step transitions). A recursive algorithm is employed to handle loop structures, compressing multi-step reductions into direct jumps.
    - **Design Motivation**: Traditional LR(1) parsing requires multi-step reductions (popping the stack and querying the GOTO table), whereas DPDA pre-encodes these steps into single-edge transitions.

3. **Edge Optimization**:

    - **Function**: Optimizes the DPDA structure during the preprocessing phase.
    - **Mechanism**:
        - **Edge Aggregation**: Merges edges with identical stack conditions and operations but different accepted symbols (e.g., merging digits 0-9 into a single edge).
        - **Edge Merging**: Directly connects two consecutively executable edges to reduce the number of DPDA states.
    - **Design Motivation**: These optimizations are only feasible when edges are completely deterministic—non-deterministic PDAs cannot perform this kind of preprocessing.

4. **Parallel Stack Matching Verification**:

    - **Function**: Parallelizes the verification of stack matching conditions across multiple prefix-conditioned edges.
    - **Mechanism**: Since all edge matching conditions are known during preprocessing, they can be processed and verified in parallel batches.
    - **Design Motivation**: Parallel verification significantly reduces latency during large-batch inference.

### Loss & Training
- **No training required**—this is a pure system optimization approach.
- One-time preprocessing cost: 3-5 seconds for complex JSON grammars, <0.1 seconds for simple grammars; results can be cached and reused.
- Implemented in Python and C++, integrated into the LightLLM inference framework.

## Key Experimental Results

### Main Results (Llama-3-8B, 2×H800)

| Batch Size | XGrammar (ms) | Pre³ (ms) | Latency Reduction |
|-----------|--------------|----------|-------------------|
| 16 | 15.19 | 11.77 | -22.5% |
| 64 | 52.07 | 35.88 | -30.1% |
| 256 | 90.98 | 64.42 | -29.2% |
| 512 | 147.64 | 104.46 | -29.2% |
| 1024 | 272.77 | 201.16 | -26.3% |

### Ablation Study

| Configuration | Effect | Description |
|------|------|------|
| DeepSeek-V2-Lite, batch=128 | -40.78% | Maximized improvement on MoE models |
| Qwen2-14B INT8, batch=1024 | -18.65% | Also effective for quantized models |
| vs Outlines | Order of magnitude advantage | Outlines overhead can reach several seconds per step |
| vs llama.cpp | Significant advantage | C++ implementation but constrained by the PDA architecture |

### Key Findings
- Pre³ consistently outperforms XGrammar across all tested models and batch sizes, achieving an average latency reduction of 25-30%.
- Improvements become more pronounced as batch sizes grow, yielding the greatest advantage in large-batch scenarios (common in production environments).
- It entirely eliminates the maintenance overhead of persistent stacks, which is the primary bottleneck of traditional PDA methods under large batches.
- Generation quality is preserved—the output is semantically identical to unconstrained decoding.
- Preprocessing overhead is extremely low (3-5 seconds) and is cacheable for reuse.

## Highlights & Insights
- **The transition from PDA to DPDA** is the correct choice of system abstraction layer—LR(1) grammars naturally support deterministic parsing, a property underutilized by previous methods. This represents a classic instance of "solving practical problems with the right theoretical tools."
- **Prefix-conditioned edges** shift runtime computation to the preprocessing phase, embodying the "compile once, run repeatedly" philosophy.
- **Edge optimization** showcases the extra dividends of determinism—not only eliminating runtime overhead but also opening the door to preprocessing optimization.
- The method can be directly applied to all LLM scenarios requiring structured output (e.g., function calling, code generation, database queries).

## Limitations & Future Work
- Supports only LR(1) grammars (covering most scenarios, but not all context-free grammars).
- For extremely complex nested grammars, the number of DPDA states may explode.
- The current implementation is only integrated into LightLLM; integration into other frameworks (e.g., vLLM, TensorRT-LLM) remains unverified.
- Has not yet been combined with inference acceleration techniques like speculative decoding.

## Related Work & Insights
- **vs XGrammar**: XGrammar uses non-deterministic PDA + precalculated masking but still requires runtime stack operations; Pre³'s DPDA completely eliminates runtime exploration.
- **vs Outlines**: Outlines employs regular expression-level constraints and cannot handle recursive grammars; Pre³ supports full LR(1).
- **vs llama.cpp**: C++ implementation but still based on the PDA architecture, limiting efficiency due to non-deterministic transitions.
- This serves as an excellent case study of applying formal language theory to modern LLM systems.

## Rating
- Novelty: ⭐⭐⭐⭐ A system-level innovation moving from PDA to DPDA, with an ingenious prefix-conditioned edge design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comparison across multiple models, batch sizes, and grammars; both latency and throughput are tested.
- Writing Quality: ⭐⭐⭐⭐ Rigorous formal definitions and clear algorithm descriptions.
- Value: ⭐⭐⭐⭐⭐ High practical value, directly applicable to production-level LLM structured outputs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Enabling LLM Knowledge Analysis via Extensive Materialization](enabling_llm_knowledge_analysis_via_extensive_materialization.md)
- [\[ACL 2025\] PRAISE: Enhancing Product Descriptions with LLM-Driven Structured Insights](praise_enhancing_product_descriptions_with_llm-driven_structured_insights.md)
- [\[ACL 2025\] SR-LLM: Rethinking the Structured Representation in Large Language Model](sr-llm_rethinking_the_structured_representation_in_large_language_model.md)
- [\[ACL 2025\] From Selection to Generation: A Survey of LLM-based Active Learning](from_selection_to_generation_a_survey.md)
- [\[ACL 2025\] Can LLMs Interpret and Leverage Structured Linguistic Representations? A Case Study with AMRs](can_llms_interpret_and_leverage_structured_linguistic_representations_a_case_stu.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] RefTool: Reference-Guided Tool Creation for Knowledge-Intensive Reasoning
description: >-
  [ICLR 2026][Information Retrieval & RAG][tool creation] This paper proposes RefTool, a framework that automatically creates executable Python tools from external reference materials (e.g., textbooks, knowledge snippets)…
tags:
  - "ICLR 2026"
  - "Information Retrieval & RAG"
  - "tool creation"
  - "reference-guided"
  - "knowledge-intensive reasoning"
  - "executable tools"
  - "hierarchical toolbox"
date: 2026-05-08
content_hash: e556f702c2425ee2
---

# RefTool: Reference-Guided Tool Creation for Knowledge-Intensive Reasoning

**Conference**: ICLR 2026

**arXiv**: [2505.21413](https://arxiv.org/abs/2505.21413)  
**Code**: To be confirmed  
**Area**: Information Retrieval
**Keywords**: tool creation, reference-guided, knowledge-intensive reasoning, executable tools, hierarchical toolbox

## TL;DR
This paper proposes RefTool, a framework that automatically creates executable Python tools from external reference materials (e.g., textbooks, knowledge snippets), addressing the failure of existing tool creation methods that rely on LLMs' intrinsic knowledge in specialized domains. RefTool achieves an average improvement of 12.3% over prior methods on causal reasoning, physics, and chemistry tasks.

## Background & Motivation
**Background**: LLM tool creation enables models to dynamically generate and invoke tools during inference, offering greater flexibility than predefined toolsets. Existing methods (e.g., CRAFT, TroVE) rely on the intrinsic knowledge of LLMs to generate tools.

**Limitations of Prior Work**: LLMs' intrinsic knowledge is unreliable in specialized domains (causal reasoning, quantum physics, organic chemistry), leading to generated tools that contain erroneous formulas or logic.

**Key Challenge**: Tool creation requires precise domain knowledge, whereas LLMs' knowledge in specialized fields may be inaccurate or incomplete.

**Goal**: How can external authoritative reference materials (textbooks) be leveraged as knowledge sources to guide tool creation?

**Key Insight**: Exploit the natural chapter–section structure of textbooks to organize tools hierarchically, extracting executable Python functions from each section.

**Core Idea**: Reference materials → tool creation + hierarchical toolbox → hierarchical retrieval → reasoning.

## Method

### Overall Architecture
Two stages: ① **Tool Creation** — extract knowledge from textbook sections and generate executable Python tools, validate them via execution tests, and organize them into a hierarchical toolbox; ② **Tool Utilization** — at inference time, perform hierarchical retrieval (first selecting a category, then a specific tool), supporting both single-turn PoT and multi-turn ReAct reasoning.

### Key Designs

1. **Reference-Guided Tool Creation**:

    - Tools are generated from each section of a textbook; each tool contains a description, a Python function, and usage examples.
    - Correctness is validated via execution tests (73% pass on the first attempt; an additional 14% pass after repair).
    - The natural chapter–section structure of textbooks is used to construct a two-level hierarchy (chapter → section → tool).
    - **Design Motivation**: Textbooks are human-validated knowledge sources and are more reliable than LLMs' intrinsic knowledge.

2. **Hierarchical Tool Retrieval**:

    - **Function**: Efficiently locate relevant tools at inference time.
    - Two-step retrieval: select relevant categories from chapter-level entries → select specific tools within those categories.
    - Reduces the search space and improves retrieval precision.
    - For unstructured reference materials, the hierarchy is automatically constructed by an LLM.

3. **Reasoning Modes**:

    - **PoT (Program of Thought)**: Single-turn generation of code containing tool calls.
    - **ReAct**: Multi-turn interactive reasoning with iterative tool retrieval and invocation.
    - The two modes are complementary: PoT is more efficient; ReAct is more flexible.

## Key Experimental Results

### Main Results

| Task | RefTool+PoT (GPT-4o) | TroVE | Domain-Specific Method |
|------|---------------------|-------|----------------------|
| Causal Reasoning (QRData) | **46.8%** | 36.4% | — |
| Physics (TheoremQA) | **57.9%** | — | Physics Reasoner |
| Chemistry (SciBench) | **66.4%** | — | ChemAgent |

### Ablation Study

| Configuration | Effect |
|---------------|--------|
| Without reference materials (pure LLM knowledge) | Significant performance drop |
| Without hierarchical structure (flat retrieval) | Reduced retrieval precision |
| Without execution test validation | Increased proportion of erroneous tools |

### Key Findings
- RefTool surpasses tool creation methods by an average of **13.0%** and domain-specific methods by **10.2%**.
- 73% of tools pass validation on the first generation, demonstrating that reference material quality ensures tool quality.
- Hierarchical retrieval outperforms flat retrieval by exploiting textbook structure to reduce the search space.
- RefTool achieves the largest gain on causal reasoning (+10.4%), indicating that LLMs' intrinsic knowledge is weakest in the causal domain.

## Highlights & Insights
- The idea of **directly converting textbooks into tools** is highly natural — humans learning a new domain also begin with textbooks before applying knowledge.
- A **73% first-pass validation rate** demonstrates that converting textbook knowledge to code is more reliable than anticipated.
- The **hierarchical toolbox** leverages the natural structure of textbooks, requiring no additional knowledge engineering.
- The approach is generalizable to any specialized domain with structured reference materials.

## Limitations & Future Work
- Depends on the availability of high-quality reference materials — inapplicable when no suitable textbook exists.
- At most two tools are generated per section, potentially missing important functionality in knowledge-dense chapters.
- The tool hierarchy is fixed at two levels, which may be insufficient for complex knowledge systems.
- Tool composition and reuse (e.g., one tool invoking another) remain unexplored.

## Related Work & Insights
- **vs. CRAFT/TroVE**: These methods rely on LLMs' intrinsic knowledge and are unreliable in specialized domains; RefTool compensates by grounding tool creation in external references.
- **vs. RAG**: RAG retrieves raw text for LLM reasoning, whereas RefTool pre-compiles knowledge into executable code — execution is more precise than inference-time reasoning.
- **vs. chainSTORM/domain-specific agents**: Domain-specific methods require manual design; RefTool automatically generates tools from textbooks.
- This work can inspire a **"knowledge compilation"** paradigm: pre-compiling textual knowledge into executable programs to reduce the cognitive burden at inference time.

## Supplementary Discussion

### Tool Creation vs. Direct RAG
The advantage of tool creation lies in "compilation" rather than "interpretation" — once knowledge is pre-compiled into executable code, inference only requires invoking functions rather than re-interpreting text. This is particularly effective for tasks requiring precise computation (physics formulas, statistical tests), since LLMs are unreliable at arithmetic whereas code execution is deterministic.

### Exploiting Textbook Structure

The chapter–section structure of textbooks is a natural product of human knowledge organization; directly adopting it as the tool hierarchy avoids additional knowledge engineering.

This approach generalizes to any domain with structured documentation (e.g., API documentation, legal regulations, medical guidelines), offering broad applicability.

## Rating
- Novelty: ⭐⭐⭐⭐ The reference-to-tool paradigm is natural and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated across three specialized domains with comprehensive comparisons.
- Writing Quality: ⭐⭐⭐⭐ Framework design is clearly presented.
- Value: ⭐⭐⭐⭐ Provides a practical paradigm for tool creation in specialized domains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] G-reasoner: Foundation Models for Unified Reasoning over Graph-structured Knowledge](g-reasoner_foundation_models_for_unified_reasoning_over_graph-structured_knowled.md)
- [\[ACL 2026\] ReasonEmbed: Enhanced Text Embeddings for Reasoning-Intensive Document Retrieval](../../ACL2026/information_retrieval/reasonembed_enhanced_text_embeddings_for_reasoning-intensive_document_retrieval.md)
- [\[ICLR 2026\] Multimodal Dataset Distillation Made Simple by Prototype-Guided Data Synthesis](multimodal_dataset_distillation_made_simple_by_prototype-guided_data_synthesis.md)
- [\[ICLR 2026\] Hybrid Deep Searcher: Scalable Parallel and Sequential Search Reasoning](hybrid_deep_searcher_scalable_parallel_and_sequential_search_reasoning.md)
- [\[ICLR 2026\] Beyond RAG vs. Long-Context: Learning Distraction-Aware Retrieval for Efficient Knowledge Grounding](beyond_rag_vs_long-context_learning_distraction-aware_retrieval_for_efficient_kn.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Which bird does not have wings: Negative-constrained KGQA with Schema-guided Semantic Matching and Self-directed Refinement
description: >-
  [ACL 2026][Graph Learning][Knowledge Graph Question Answering] This paper proposes the new task of Negative-constrained Knowledge Graph Question Answering (NEST KGQA) and the NestKGQA dataset. It designs PyLF…
tags:
  - "ACL 2026"
  - "Graph Learning"
  - "Knowledge Graph Question Answering"
  - "negative constraint"
  - "semantic parsing"
  - "logical form"
  - "Schema-guided"
date: 2026-05-08
content_hash: 01b56173653e1bca
---

# Which bird does not have wings: Negative-constrained KGQA with Schema-guided Semantic Matching and Self-directed Refinement

**Conference**: ACL 2026 Findings  
**arXiv**: [2604.14749](https://arxiv.org/abs/2604.14749)  
**Code**: [https://github.com/midannii/CUCKOO](https://github.com/midannii/CUCKOO)  
**Area**: Graph Learning / Knowledge Graph Question Answering  
**Keywords**: Knowledge Graph Question Answering, negative constraint, semantic parsing, logical form, Schema-guided

## TL;DR

This paper proposes the new task of Negative-constrained Knowledge Graph Question Answering (NEST KGQA) and the NestKGQA dataset. It designs PyLF, a Python-format logical form, to clearly express negative constraints. Furthermore, it introduces the CUCKOO framework, which achieves efficient and accurate answering for multi-constraint questions in a few-shot setting through three modules: constraint-aware draft generation, schema-guided semantic matching, and self-directed refinement.

## Background & Motivation

**Background**: Knowledge Graph Question Answering (KGQA) is a critical direction for reducing LLM hallucinations by utilizing external knowledge. Semantic Parsing (SP) methods map natural language questions to logical forms, which are then converted into SPARQL queries for execution on Knowledge Graphs (KG), offering advantages in interpretability and faithfulness.

**Limitations of Prior Work**: Existing KGQA benchmarks and methods are heavily biased toward positive and computational constraints, ignoring negative constraints. Although some datasets contain negation words like "not," they are often actually comparison operations. LLMs are inherently fragile in negation reasoning, and existing logical forms (such as s-expressions) struggle to express negative semantics clearly.

**Key Challenge**: Negative constraints appear frequently in real-world questions, yet specialized benchmarks and methods are lacking. Moreover, negative constraint questions naturally contain multiple constraint conditions, significantly increasing semantic complexity and the risk of generating non-executable queries.

**Goal**: (1) Define the NEST KGQA task and construct the NestKGQA dataset; (2) design the PyLF logical form to clearly express negation; (3) build an efficient framework capable of handling multi-constraint negative questions.

**Key Insight**: The author observes that existing SP methods use brute-force search for semantic matching without considering KG schema semantics, leading to an exponential growth in the number of candidate logical forms. By leveraging KG schema constraints for pruning, both efficiency and accuracy can be improved simultaneously.

**Core Idea**: Use constraint-aware draft generation to explicitly enumerate constraint elements in the question, followed by schema-guided semantic matching to anchor the draft to the KG. Finally, trigger self-directed refinement only when execution results are empty, achieving low-cost and robust negative-constrained QA.

## Method

### Overall Architecture

CUCKOO is a KGQA framework following a "generate-then-match" paradigm. Given a natural language question, the constraint-aware draft generation module first extracts constraint elements and generates a PyLF logical form draft. Then, the schema-guided semantic matching module maps entity and relation mentions in the draft to specific items in the KG, generating a list of executable logical forms. The matching results are converted to SPARQL for execution. The self-directed refinement module is triggered to correct the draft only if the execution returns an empty result.

### Key Designs

1.  **PyLF (Python-format Logical Form)**:
    - **Function**: Provides a logical form that clearly expresses negative constraints while maintaining readability.
    - **Mechanism**: Adds a boolean parameter `neg` to the `JOIN` function to mark negative constraints (e.g., `JOIN('producing', 'Saturn', neg=True)`) and uses the `R_` prefix to distinguish whether the query targets the head or tail entity, making semantic parsing more precise.
    - **Design Motivation**: Among existing logical forms, only $\lambda$-calculus can express negation but has poor readability, while s-expressions are readable but cannot express negation. PyLF is based on Python syntax; since LLMs are exposed to vast amounts of Python code during pre-training, it results in lower syntax error rates.

2.  **Schema-guided Semantic Matching**:
    - **Function**: Maps entity and relation mentions from the logical form draft to specific items in the KG while ensuring semantic executability.
    - **Mechanism**: Starting from the topic entity in the `START` function, it retrieves candidate entities and their categories via cosine similarity. It then extracts schema-level triples containing the candidate categories and filters relation matches using a similarity threshold $\theta$. Category information is propagated layer-by-layer using schema constraints (domain/range) to automatically prune illegal combinations.
    - **Design Motivation**: Traditional brute-force matching takes top-$K_e$ for each entity and top-$K_r$ for each relation, leading to $K_e^n \cdot K_r^m$ exponential growth. The schema-guided method uses type constraints to drastically reduce candidates, for example, reducing $K_e^1 \cdot K_r^2$ to $1 \times 2 \times 2 = 4$.

3.  **Self-directed Refinement Module**:
    - **Function**: Fixes logical form drafts with formatting or semantic errors.
    - **Mechanism**: Triggered only when query execution results are empty. It first diagnoses the problem type from predefined error categories (missing constraint decomposition, formatting errors, function syntax errors, etc.), then guides the LLM to regenerate the draft via few-shot examples without additional parameter fine-tuning or external execution feedback.
    - **Design Motivation**: Unlike existing code generation methods that rely on external execution feedback and multiple LLM rounds, CUCKOO’s refinement is self-contained, reducing cost and latency.

### Loss & Training

CUCKOO is a training-free framework based on in-context learning. Draft generation uses GPT-3.5-turbo as the backbone LLM, retrieving the top-$k$ similar examples from training data via SimCSE embeddings as few-shot demonstrations. The number of generated candidates is 1 or 6, and the final prediction is determined via majority voting.

## Key Experimental Results

### Main Results

| Dataset | Metric | CUCKOO(6) | KB-Coder(6) | KB-BINDER(6) |
| :--- | :--- | :--- | :--- | :--- |
| GrailQA (Overall) | EM/F1 | **62.1/64.2** | 51.2/56.3 | 52.5/54.5 |
| GrailQA (Zero-shot) | EM/F1 | **57.5/59.8** | 46.7/51.6 | 45.9/48.6 |
| NestKGQA | F1 | **26.2** | 24.4 | 4.6 |
| GraphQ | F1 | **40.8** | 35.8 | 32.7 |

### Ablation Study

| Configuration | GrailQA F1 | NestKGQA F1 | Description |
| :--- | :--- | :--- | :--- |
| CUCKOO Full Model | 64.2 | 26.2 | Full model |
| w/o Self-directed Refinement | 63.2 | 25.8 | Refinement contributes ~1 point |
| w/o Constraint Elements | 61.3 | 24.4 | Explicit constraint decomposition is helpful |
| w/o Schema-guided Matching | 56.6 | 16.3 | Core module; performance drops significantly without it |

### Key Findings

- Schema-guided semantic matching is the most critical module; removing it drops GrailQA by 7.6 points and NestKGQA by nearly 10 points.
- CUCKOO shows the strongest advantage on multi-constraint questions (3 constraints), reaching the highest EM.
- Achieved a massive improvement from 3.1 to 53.1 on superlative-type questions.
- All zero-shot LLMs perform much worse on NestKGQA than on traditional KGQA, proving that negative constraint reasoning is indeed difficult.
- CPU memory usage is 4.7% lower than KB-Coder, though inference time increases by approximately 1.6x.

## Highlights & Insights

- PyLF resolves the long-standing problem of negative constraint expression by simply adding a `neg` boolean parameter to the `JOIN` function. This "minimal modification" approach is noteworthy—one does not need to reinvent logical forms, just expand existing frameworks targetly.
- Schema-guided matching uses the KG's type system for candidate pruning, reducing exponential search space to polynomial. This idea is transferable to any scenario requiring generation and verification over structured knowledge.
- The "trigger only on failure" strategy for self-directed refinement is an elegant engineering design that avoids unnecessary LLM calls.

## Limitations & Future Work

- Based on the closed-world assumption, limiting applicability in open-world scenarios.
- The NestKGQA dataset is relatively small, as it is an extension of existing benchmarks.
- Assumes the KG schema is fully available; an additional schema extraction model is needed when the schema is incomplete.
- Performance depends on the backbone LLM's capability; future work needs to explore model-agnostic strategies.

## Related Work & Insights

- **vs KB-BINDER**: KB-BINDER uses s-expressions which cannot express negation, and its semantic matching uses brute-force search. CUCKOO surpasses it in both aspects via PyLF and schema-guided matching.
- **vs KB-Coder**: KB-Coder uses a Python-format logical form but does not explicitly handle negation or constraint decomposition. While it slightly excels in I.I.D. scenarios by mimicking examples, it falls short of CUCKOO in compositional generalization and negation scenarios.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First to systematically define the negative-constrained KGQA task; task definition is clear, and PyLF design is simple yet effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multiple benchmarks, ablations, and multi-dimensional analysis, though the NestKGQA dataset is small.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure with intuitive motivating examples.
- **Value**: ⭐⭐⭐⭐ Fills a gap in handling negative constraints within KGQA; schema-guided matching has general value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] CoG: Controllable Graph Reasoning via Relational Blueprints and Failure-Aware Refinement over Knowledge Graphs](cog_controllable_graph_reasoning_via_relational_blueprints_and_failure-aware_ref.md)
- [\[AAAI 2026\] NOTAM-Evolve: A Knowledge-Guided Self-Evolving Optimization Framework with LLMs for NOTAM Interpretation](../../AAAI2026/graph_learning/notam-evolve_a_knowledge-guided_self-evolving_optimization_framework_with_llms_f.md)
- [\[ACL 2026\] GS-Quant: Granular Semantic and Generative Structural Quantization for Knowledge Graph Completion](gs-quant_granular_semantic_and_generative_structural_quantization_for_knowledge_.md)
- [\[ACL 2026\] TagRAG: Tag-guided Hierarchical Knowledge Graph Retrieval-Augmented Generation](tagrag_tag-guided_hierarchical_knowledge_graph_retrieval-augmented_generation.md)
- [\[ICLR 2026\] Pairwise is Not Enough: Hypergraph Neural Networks for Multi-Agent Pathfinding](../../ICLR2026/graph_learning/pairwise_is_not_enough_hypergraph_neural_networks_for_multi-agent_pathfinding.md)

</div>

<!-- RELATED:END -->

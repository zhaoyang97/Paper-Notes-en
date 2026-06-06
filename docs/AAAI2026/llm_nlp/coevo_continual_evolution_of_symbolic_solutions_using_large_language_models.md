---
title: >-
  [Paper Note] CoEvo: Continual Evolution of Symbolic Solutions Using Large Language Models
description: >-
  [AAAI 2026][LLM/NLP][Symbolic Regression] This paper proposes CoEvo, a framework that integrates LLMs with evolutionary search methodology to achieve continual open-ended evolution of symbolic solutions through a dynamic…
tags:
  - "AAAI 2026"
  - "LLM/NLP"
  - "Symbolic Regression"
  - "LLM-based Evolutionary Search"
  - "Knowledge Library"
  - "Open-ended Innovation"
  - "Multi-representation Space"
date: 2026-05-08
content_hash: 826e1492d00bd3e9
---

# CoEvo: Continual Evolution of Symbolic Solutions Using Large Language Models

**Conference**: AAAI 2026
**arXiv**: [2412.18890](https://arxiv.org/abs/2412.18890)  
**Code**: [Available](https://github.com/pgg3/CoEvo)  
**Area**: LLM/NLP
**Keywords**: Symbolic Regression, LLM-based Evolutionary Search, Knowledge Library, Open-ended Innovation, Multi-representation Space

## TL;DR

This paper proposes CoEvo, a framework that integrates LLMs with evolutionary search methodology to achieve continual open-ended evolution of symbolic solutions through a dynamic knowledge library and multi-representation spaces (natural language / mathematical formulas / code), significantly outperforming existing symbolic regression methods on the AI Feynman benchmark.

## Background & Motivation

The discovery of symbolic solutions — mathematical expressions, logical rules, algorithmic structures — is foundational to scientific and engineering progress. However, existing approaches face two major bottlenecks:

**Traditional methods** (evolutionary algorithms such as PySR, deep learning methods such as NeSymReS): low search efficiency, difficulty integrating knowledge effectively.

**LLM-based methods** (FunSearch, LLM-SR): improved search efficiency, but **lack the ability to continuously refine and extend discovered solutions and their underlying knowledge**, limiting open-ended innovation.

Core problem: **Can LLMs not only reuse existing knowledge but also discover new knowledge and evolve continuously?**

CoEvo's vision: to define the discovery of symbolic solutions as a **lifelong, iterative process** — analogous to human scientific exploration — where solutions and foundational knowledge co-evolve.

## Method

### Overall Architecture

CoEvo consists of three core components:

```
CoEvo Framework
├── Idea Tree-based Solution Generation
│   ├── Step 1: Inspiring
│   ├── Step 2: Thinking
│   └── Step 3: Solving
├── Evolutionary Search
│   ├── Initialization
│   ├── Crossover (Positive / Negative)
│   ├── Mutation (Positive / Negative)
│   └── Population Update (Elitism)
└── Knowledge Library (Dynamic)
    ├── Summarization
    ├── Management (Clustering & Deduplication)
    └── Reuse (Random / Similarity-based)
```

### Key Designs

**1. Idea Tree-based Solution Generation**

This component simulates a three-step human problem-solving process:

| Step | Human Analogy | LLM Operation | Purpose |
|------|--------------|---------------|---------|
| Inspiring | Obtaining initial inspiration | Retrieve relevant ideas from the knowledge library | Stimulate diversity |
| Thinking | In-depth reasoning | Iteratively refine ideas based on evaluator feedback | Improve quality |
| Solving | Output solution | Generate solutions in multiple formats | Explore multiple spaces |

Tree structure: starting from $N_0$ root ideas, each layer evolves $N_k$ ideas based on the parent ideas and evaluator feedback. Unlike the exhaustive branching of Tree-of-Thought, a constrained network structure is adopted to avoid exponential computational overhead.

**2. Multi-Representation Solutions**

The search space is extended from the traditional mathematical formula/code representations to three levels:

| Representation Space | Complexity | Knowledge Richness | Implementation |
|----------------------|-----------|-------------------|----------------|
| Mathematical Formula | Low | Low | LaTeX code |
| Python Code | Medium | Medium | Executable code |
| Natural Language | High | High | LLM reasoning text |

Key insight: different representation spaces encode different levels of knowledge — the natural language space is the richest, enabling full exploitation of LLMs' reasoning capabilities.

**3. Dynamic Knowledge Library**

Three core mechanisms:

**Summarization**:
- Trigger condition: a solution achieves a higher score during tree-based search or offspring generation
- Operation: the LLM extracts and summarizes the key ideas behind the improvement, stored in a definition-description format
- Purpose: learning "why a solution is good" from high-quality solutions

**Management**:
- The library maintains a fixed capacity (30 entries in experiments)
- Semantic clustering based on cosine similarity of sentence embeddings (DBSCAN)
- Representative ideas are retained; redundant ones are removed

**Reuse**: two modes

| Mode | Usage Scenario | Strategy |
|------|---------------|----------|
| Random Reuse | Generating new solutions | Randomly sample one idea from each cluster |
| Similarity-based Reuse | Tree-based idea search | Retrieve ideas most similar to the current idea |

**4. Evolutionary Search**

| Operator | Type | Description |
|----------|------|-------------|
| Crossover | Positive | Promotes solutions similar to parent ideas |
| Crossover | Negative | Promotes solutions divergent from parent ideas, enhancing diversity |
| Mutation | Positive | Small incremental modifications |
| Mutation | Negative | Large, significant changes |
| Population Update | Elitism | Retain the top $N$ solutions by score |

### Loss & Training

- Evaluation metric: Normalized Mean Squared Error (NMSE), measured both in-distribution (ID) and out-of-distribution (OOD)
- Iteration budget: 2,000 iterations, 100 generations, 20 samples per generation
- Knowledge library capacity: 30 entries
- LLM backbone: gpt-3.5-turbo and gpt-4o-mini
- No gradient-based training — the method is entirely based on LLM generation and evolutionary search

## Key Experimental Results

### Main Results

**Table 1: Performance Comparison on AI Feynman Benchmark (NMSE)**

| Method | Oscillation 1 ID/OOD | Oscillation 2 ID/OOD | E. coli Growth ID/OOD | Stress-Strain ID/OOD |
|--------|-----|-----|-----|-----|
| GPlearn | 0.0155/0.5567 | 0.7551/3.188 | 1.081/1.039 | 0.1063/0.4091 |
| PySR | 0.0009/0.3106 | 0.0002/0.0098 | 0.0376/1.014 | 0.0331/0.1304 |
| uDSR | 0.0003/0.0007 | 0.0032/0.0015 | 0.3322/5.458 | 0.0502/0.1761 |
| LLM-SR (gpt-4o-mini) | 5.14e-9/3e-4 | 1.79e-7/3.11e-5 | 0.0214/0.0264 | 0.0020/0.0020 |
| **CoEvo (gpt-3.5-turbo)** | **4.32e-9/8.71e-5** | **1.58e-10/1.32e-10** | **1.58e-9/1.21e-8** | 0.0020/0.0015 |

CoEvo outperforms LLM-SR on Oscillation 2 and E. coli Growth by **several orders of magnitude**.

**Table 2: Comparison of Search Spaces Across Methods**

| Method | Search Space | Knowledge Management | Open-ended Evolution |
|--------|-------------|---------------------|----------------------|
| PySR | Formula/Code | None | No |
| FunSearch | Code | None | No |
| LLM-SR | Code | Static | No |
| EoH | Natural Language + Code | None | No |
| **CoEvo** | **Natural Language + Formula + Code** | **Dynamic Knowledge Library** | **Yes** |

### Ablation Study

- **Impact of the knowledge library**: with vs. without the library, NMSE on E. coli Growth improves by 2–3 orders of magnitude.
- **Impact of LLM choice**: gpt-3.5-turbo and gpt-4o-mini yield comparable performance, indicating the method is not sensitive to the choice of backbone LLM.
- **Proportion of valid solutions**: CoEvo generates a significantly higher proportion of valid solutions than LLM-SR across all benchmarks.
- **Cross-source knowledge experiment**: knowledge extracted from gpt-3.5-turbo applied to gpt-4o-mini (and vice versa) consistently improves the quality of new solutions.

### Key Findings

1. **Discovery of implicit solutions for Oscillation 2**: CoEvo is the only method to discover that `numpy.gradient` can be applied to velocity data to compute acceleration — a non-traditional, data-driven approach — while all other methods attempt to recover explicit physical equations.
2. **Visualization of knowledge evolution**: the knowledge library evolves dynamically during the search process; upon discovering the implicit solution, the diversity of the library expands rapidly.
3. **Need for knowledge condensation**: not all accumulated knowledge is useful; future work requires an idea condensation mechanism to filter out uninformative entries.
4. **Valid solution proportion** is a core advantage of CoEvo — a better exploration strategy reduces invalid sampling.

## Highlights & Insights

- **First work to define symbolic discovery as a lifelong, continual process**: the goal is not only to find solutions but to continuously refine knowledge and expand discovery capabilities.
- **Elegant multi-representation space design**: the knowledge richness of the natural language space compensates for the limitations of traditional formula/code spaces.
- **Closed-loop knowledge pipeline — collection, management, and reuse**: summarization extracts knowledge from high-quality solutions, management prevents knowledge bloat, and reuse injects knowledge at the right moment.
- **The Oscillation 2 case** is a standout highlight: CoEvo discovers a non-traditional solution path that even human researchers might overlook.

## Limitations & Future Work

1. Experiments cover only 4 AI Feynman problems; the scale is limited, and generalizability requires validation on more benchmarks.
2. The knowledge library capacity (30 entries) is set empirically, lacking theoretical justification.
3. Reliance on LLM API calls introduces cost and latency constraints that limit large-scale deployment.
4. Idea condensation is mentioned by the authors but not implemented; this may be a key avenue for further performance improvement.
5. No comparison is made with the latest code generation or scientific discovery LLMs (e.g., AlphaCode, AlphaGeometry).

## Related Work & Insights

- **FunSearch (Paredes et al. 2024)**: a pioneering LLM + evolutionary search framework, but restricted to code space with no knowledge management.
- **LLM-SR (Shojaee et al. 2024)**: the current state-of-the-art symbolic regression method; CoEvo builds upon it by adding the knowledge library and multi-representation spaces.
- **Tree-of-Thought (Yao et al. 2024)**: tree-structured reasoning; CoEvo's idea tree draws inspiration from this paradigm while avoiding exponential computational overhead.
- **Implications for model compression**: during knowledge distillation, the "knowledge" of a teacher model could similarly be dynamically summarized, managed, and refined, rather than being extracted in a one-shot, static manner.

## Rating

- Novelty: ⭐⭐⭐⭐ (first to define symbolic discovery as a continual evolution process; knowledge library design is original)
- Experimental Thoroughness: ⭐⭐⭐ (benchmark scale is limited, but the analysis is in-depth)
- Writing Quality: ⭐⭐⭐⭐ (framework diagrams are clear; ablation study is comprehensive)
- Value: ⭐⭐⭐⭐ (opens a new paradigm for LLM-driven scientific discovery)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] ProFuser: Progressive Fusion of Large Language Models](profuser_progressive_fusion_of_large_language_models.md)
- [\[AAAI 2026\] Identifying and Analyzing Performance-Critical Tokens in Large Language Models](identifying_and_analyzing_performance-critical_tokens_in_large_language_models.md)
- [\[AAAI 2026\] LoKI: Low-damage Knowledge Implanting of Large Language Models](loki_low-damage_knowledge_implanting_of_large_language_models.md)
- [\[AAAI 2026\] Control Illusion: The Failure of Instruction Hierarchies in Large Language Models](control_illusion_the_failure_of_instruction_hierarchies_in_large_language_models.md)
- [\[ICCV 2025\] Any-SSR: How Recursive Least Squares Works in Continual Learning of Large Language Models](../../ICCV2025/llm_nlp/any_ssr_how_recursive_least_squares_works_in_continual_learning_of_large_language_models.md)

</div>

<!-- RELATED:END -->

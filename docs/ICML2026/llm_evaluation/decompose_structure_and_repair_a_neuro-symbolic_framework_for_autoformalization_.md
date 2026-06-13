---
title: >-
  [Paper Note] Decompose, Structure, and Repair: A Neuro-Symbolic Framework for Autoformalization via Operator Trees
description: >-
  [ICML 2026][LLM Evaluation][Autoformalization] This paper introduces DSR (Decompose-Structure-Repair), a neuro-symbolic framework that decomposes natural language (NL) theorem formalization into three stages: "NL compone…
tags:
  - "ICML 2026"
  - "LLM Evaluation"
  - "Autoformalization"
  - "Lean 4"
  - "Operator Trees"
  - "Curriculum Learning"
  - "Tree-Guided Repair"
date: 2026-05-08
content_hash: d4b1db733a9027d9
---

# Decompose, Structure, and Repair: A Neuro-Symbolic Framework for Autoformalization via Operator Trees

**Conference**: ICML 2026  
**arXiv**: [2604.19000](https://arxiv.org/abs/2604.19000)  
**Code**: https://github.com/XiaoyangLiu-sjtu/DSR  
**Area**: LLM Reasoning / Formal Mathematics / Neuro-Symbolic  
**Keywords**: Autoformalization, Lean 4, Operator Trees, Curriculum Learning, Tree-Guided Repair  

## TL;DR
This paper introduces DSR (Decompose-Structure-Repair), a neuro-symbolic framework that decomposes natural language (NL) theorem formalization into three stages: "NL component decomposition → joint generation of formal language (FL) components and Operator Trees (OPT) → hierarchical repair based on subtree localization." DSR sets new SOTA results on ProverBench, ProofNet, and PRIME using a 7B model, and releases PRIME, a graduate-level Lean 4 benchmark with 156 problems.

## Background & Motivation

**Background**: Statement autoformalization aims to automatically translate NL mathematical statements into FL verifiable by Interactive Theorem Provers (ITPs) like Isabelle, Coq, or Lean 4. Recent mainstream approaches have evolved from neural machine translation to LLM-driven paradigms, including few-shot prompting, SFT on formal corpora, and reinforcement learning/RAG/tool feedback. System-level iterative architectures like ARIA (graph-of-thought) and SITA (structure-to-instance) have also emerged.

**Limitations of Prior Work**: Despite evolving paradigms, nearly all methods treat formalization as a one-shot, end-to-end "flat string" generation task, predicting Lean code as a linear sequence of tokens. This leads to two consequences:  
1. Models fail to explicitly capture the inherent nested structure of mathematical statements (quantifiers, operator precedence, condition-conclusion splits), often generating code that is locally erroneous but globally plausible.  
2. Upon compilation failure, typical repair strategies involve re-generating the entire statement (statement-level repair), which is computationally wasteful and prone to breaking correct parts—leading to "syntax-passing but semantic-drifting" degradation.

**Key Challenge**: The correctness of formal code is **structural** (one incorrect node in an operator tree invalidates the whole tree), yet generation and repair remain **linear**. The lack of an addressable, locally intervenable intermediate representation makes it difficult to achieve both correct repair and global logical consistency.

**Goal**: Introduce an explicit **hierarchical intermediate representation** for autoformalization, such that: (a) the model learns the topological skeleton of mathematical statements during training; (b) the repair phase can precisely target faulty subtrees for minimal structural rewriting, akin to "surgery."

**Key Insight**: The Lean Language Server can parse FL code into an Operator Tree (OPT), where operators are internal nodes and arguments are ordered children. By having the model **jointly generate the FL code and its corresponding OPT**, the OPT serves as both a "structural prior" (regularization during training) and a "repair blueprint" (addressability during inference).

**Core Idea**: Reframe autoformalization as a modular *Decompose → Structure → Repair* pipeline—first decompose NL statements into condition/conclusion components, then jointly predict the FL code and OPT for each component, and finally use the OPT to localize compilation errors to subtrees for bottom-up hierarchical repair.

## Method

### Overall Architecture
The input is an NL mathematical statement, and the output is an FL statement that passes Lean 4 compilation while maintaining semantic consistency. The pipeline consists of three stages:

1. **Semantic Decomposition** (§3.1): Gemini 3.0 Pro performs "semantic normalization + structural role alignment" in a single pass, splitting the original text into NL components labeled as *Condition* or *Conclusion*.
2. **Structured Translation** (§3.2): The DSR Formalizer (a LoRA-finetuned Qwen2.5-7B-Instruct) jointly generates two outputs for each NL component: a linear FL component string and an FL OPT. Unlike ASSESS, brackets are explicitly preserved in the OPT.
3. **Tree-Guided Repair** (§3.3): Components are assembled via a Structure-First approach using OPT leaf nodes and compiled in Lean. If failure occurs, compiler errors (row/column) are mapped to specific OPT nodes to perform recursive repair across three granularities: *subcomponent → component → statement*.

### Key Designs

1. **Operator Tree as a Joint Generation Target (FL Component + FL OPT Dual Output)**:
    - **Function**: Forces the model to output the operator tree alongside linear Lean code as a structural semantic anchor and a topological blueprint for repair.
    - **Mechanism**: Reuses the OPT representation from ASSESS with two modifications: (1) Tree construction at the component level rather than the statement level; (2) **Explicit bracket preservation** to maintain token-level consistency between linear code and the tree. This allows fallback to the FL component as an "Inference Failsafe" if OPT generation fails. The model is trained to internalize recursive topology $T = (V, E, \ell)$ where operators are parents and arguments are ordered children.
    - **Design Motivation**: Pure linear generation often fails due to mismatched brackets or unclosed scopes. OPT output acts as "structural regularization," forcing the model to internalize nesting precedence. Critically, OPT segments linear code into *addressable logical substructures*.

2. **Four-Stage Curriculum Learning Based on OPT Complexity**:
    - **Function**: Prevents the 7B model from collapsing when simultaneously learning complex logic and hierarchical topology.
    - **Mechanism**: Starting from 120k FL statements (NuminaMath-LEAN + ATLAS-Synthetic), 283,958 triplets of ⟨NL component, FL component, FL OPT⟩ are constructed. Samples are categorized into simple (143k), moderate (110k), and complex (28k) based on *tree depth, width, and node count*. Training proceeds in four phases: Phase 1 focuses on NL→FL component mapping; Phases 2-4 introduce joint OPT prediction with increasing complexity. Replay buffers (10–30%) prevent forgetting.
    - **Design Motivation**: Ablation (Table 3) shows that **adding OPT without curriculum learning** leads to an "optimization barrier," where Pass@1 SC on PRIME drops from 22.44% to 19.87%. Curriculum learning recovers this to 23.08%.

3. **Three-Level Tree-Guided Repair**:
    - **Function**: Rewrites only the "truly erroneous subtree" during compilation failure, preserving correctly generated parts.
    - **Mechanism**: Maps Lean compiler errors $(row, col)$ to the smallest faulty node $v$ in the OPT. Repair then proceeds bottom-up: (1) **Subcomponent-Level**: Rewrite the minimal subtree starting from $v$'s parent; (2) **Component-Level**: Rewrite the entire FL component; (3) **Statement-Level**: Rewrite the full statement as a final failsafe. The total budget is limited to 4 calls.
    - **Design Motivation**: Comparison of DSR vs. DSR-Global (Table 2) shows that global repair often achieves higher Syntax Check (SC) but lower Consistency Check (CC) scores. This suggests global rewriting acts as a "brute-force" strategy that passes compilers but drifts semantically. Tree-guided repair prioritizes semantic fidelity.

### Loss & Training
Qwen2.5-7B-Instruct is finetuned using LoRA with standard next-token cross-entropy. The target sequence is the concatenation of `FL component <SEP> FL OPT`. Batch sizes range from 128 to 64 across four 1-epoch phases with learning rates decaying from 2e-4 to 1e-5. Inference budget is fixed at 4 calls.

## Key Experimental Results

### Main Results

| Dataset | Metric | DSR (7B) | Strongest Baseline | Gain |
|---------|--------|----------|--------------------|------|
| ProverBench | CC | **84.00** | Goedel-V2-32B (83.38) | +0.62 |
| ProofNet | CC | **79.51** | Goedel-V2-32B (70.89) | **+8.62** |
| PRIME (graduate) | CC | **67.95** | Goedel-V2-32B (66.67) | +1.28 |
| ProofNet | SC | **87.33** | Goedel-V2-32B (77.63) | +9.70 |

*Note: All methods use a 4-call inference budget. DSR 7B consistently outperforms 32B-class models like Goedel-V2 and ATF-32B.*

### Ablation Study

| Configuration | ProverBench Pass@4 CC | ProofNet Pass@4 CC | PRIME Pass@4 CC | Note |
|---------------|-----------------------|--------------------|-----------------|------|
| Baseline (Linear Lean only) | 30.46 | 16.71 | 25.00 | Naive seq2seq |
| + Operator Tree | 32.31 (+1.85) | 18.06 (+1.35) | 21.15 (−3.85) | Optimization barrier on PRIME |
| + Curriculum Learning | **33.54 (+3.08)** | **19.41 (+2.70)** | **26.28 (+1.28)** | OPT + Curriculum yields stable gains |

Repair Strategy Ablation (Table 2): DSR vs DSR-Global (Statement-level) — ProofNet CC 79.51 vs 76.01 (+3.50), proving tree-guided repair preserves semantic fidelity.

### Key Findings
- **OPT requires curriculum learning to be effective**: Adding OPT supervision alone causes Pass@1 SC to drop on high-difficulty benchmarks (PRIME), indicating that simultaneous logic and topology learning is burdensome for 7B models.
- **SC↔CC gap exposes semantic drift in baselines**: Kimina-Autoformalizer-7B achieves 83.02% SC on ProofNet but only 56.87% CC (a 26.15% gap), whereas DSR's gap is only 7.82%. OPT supervision constrains the model to generate semantically consistent rather than "accidentally compilable" code.
- **Higher complexity yields larger DSR advantages**: DSR's lead increases from 0.62% on simple ProverBench to 8.62% on ProofNet and 1.28% on PRIME. Structural priors provide the most value for deeply nested theorems.

## Highlights & Insights
- **Joint "Code + Tree" generation is an underrated, low-cost structural regularization**: Outputting an extra token sequence (linearized OPT) requires no changes to training objectives but provides both semantic anchors (training) and a repair address book (inference).
- **LLM as a "subtree replacer" rather than a "full sequence rewriter"** limits stochasticity to the minimal suspicious region. This principle can extend to other code generation tasks where structures are largely correct with local faults (e.g., SQL, DSLs).
- **PRIME benchmark has independent value**: 156 graduate-level Lean 4 problems with expert annotations and informal proofs can support Automated Theorem Proving (ATP) research beyond just autoformalization.

## Limitations & Future Work
- OPT construction depends on ASSESS/Lean Language Server, tying the framework to Lean 4. Migration to Coq/Isabelle requires rewriting the OPT extraction toolchain.
- The decomposition stage uses Gemini 3.0 Pro, introducing closed-source API dependencies affecting deployment cost and reproducibility.
- The repair phase mandates a final statement-level "semantic check," consuming an inference budget slot even if local repair succeeded.
- Ablations did not report a "no repair" DSR Formalizer, making it difficult to isolate training improvements from repair strategy gains.
- Orthogonality with retrieval-based decomposition (e.g., DRIFT) is noted but not experimentally explored.

## Related Work & Insights
- **vs. ARIA / SITA (System-level Iteration)**: ARIA uses graph-of-thought for planning; SITA uses structure-to-instance. DSR focuses on internal intermediate representations. OPT is directly verifiable by the Lean compiler, making it more machine-readable than ad-hoc plans.
- **vs. DRIFT (Zhang et al. 2026)**: DRIFT decomposes for external concept retrieval; DSR decomposes for dimensionality reduction and OPT generation. The two are complementary.
- **vs. ASSESS (Liu et al. 2026)**: DSR adopts ASSESS's OPT but lowers the granularity to components and enforces token-level alignment (brackets) specifically for surgical repair.
- **vs. Goedel-V2-Formalizer-32B**: DSR 7B's +8.62% gain over a 32B model on ProofNet CC serves as strong evidence of "structure over scale."

## Rating
- **Novelty**: ⭐⭐⭐⭐ OPT is not new, but the "joint FL+OPT for surgical repair" combination is a first in this context.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Three benchmarks, two baseline settings, and detailed ablations for both training and repair.
- **Writing Quality**: ⭐⭐⭐⭐ Clear explanation of OPT's dual role and intuitive repair trajectory visualizations.
- **Value**: ⭐⭐⭐⭐ A useful neuro-symbolic methodology for formal math; demonstrates a high ROI for structural priors in smaller models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Towards a Common Framework for Autoformalization](../../AAAI2026/llm_evaluation/towards_a_common_framework_for_autoformalization.md)
- [\[ICML 2026\] PoliticsBench: Benchmarking Political Values in Large Language Models with Multi-Stage Roleplay](politicsbench_benchmarking_political_values_in_large_language_models_with_multi-.md)
- [\[ICML 2026\] CapBencher: Give Your LLM Benchmark a Built-in Alarm for Test-Set Overfitting](capbencher_give_your_llm_benchmark_a_built-in_alarm_for_test-set_overfitting.md)
- [\[ICML 2026\] Toward Training Superintelligent Software Agents through Self-Play SWE-RL](toward_training_superintelligent_software_agents_through_self-play_swe-rl.md)
- [\[ICML 2026\] On Effectiveness and Efficiency of Agentic Tool-calling and RL Training](on_effectiveness_and_efficiency_of_agentic_tool-calling_and_rl_training.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Lost in Transmission: When and Why LLMs Fail to Reason Globally
description: >-
  [NeurIPS 2025 (Spotlight)][LLM Reasoning][communication complexity] This paper proposes the Bounded Attention Prefix Oracle (BAPO) computational framework…
tags:
  - "NeurIPS 2025 (Spotlight)"
  - "LLM Reasoning"
  - "communication complexity"
  - "bounded attention"
  - "chain-of-thought"
  - "LLM limitations"
  - "computational framework"
date: 2026-05-08
content_hash: a47044107b1d7d96
---

# Lost in Transmission: When and Why LLMs Fail to Reason Globally

**Conference**: NeurIPS 2025 (Spotlight)
**arXiv**: [2505.08140](https://arxiv.org/abs/2505.08140)
**Code**: None
**Area**: LLM Reasoning
**Keywords**: communication complexity, bounded attention, chain-of-thought, LLM limitations, computational framework
**Authors**: Tobias Schnabel, Kiran Tomlinson, Adith Swaminathan, Jennifer Neville (Microsoft)

## TL;DR

This paper proposes the Bounded Attention Prefix Oracle (BAPO) computational framework, which models LLM attention heads as finite-bandwidth communication channels. It proves that global reasoning problems such as graph reachability are BAPO-hard (requiring super-constant bandwidth), and shows that Chain-of-Thought (CoT) can transform any BAPO-hard problem into a BAPO-easy one. Theoretical predictions are validated experimentally on GPT-4o, Claude, and Gemini.

## Background & Motivation

### Root Cause

**Key Challenge**: **Background**: **Systematic failures in LLM global reasoning**: Transformer-based LLMs consistently fail on tasks requiring integration of information across large portions of the input (e.g., graph reachability, variable tracking, majority vote aggregation), even at large model scales.

**Limitations of Prior Work**: Existing work largely documents these failures empirically, lacking a formal framework to explain why certain tasks are hard while others are not, or to predict which problem classes LLMs will fail on.

**Information-flow bottleneck hypothesis**: The paper argues that the root cause is not insufficient computational capacity (increasing MLP width or depth provides no benefit), but rather the **communication bandwidth** constraint of the attention mechanism — a capacity bottleneck in how information is transmitted between residual streams via attention.

**Theoretical gap on CoT effectiveness**: Chain-of-Thought empirically helps on complex reasoning tasks, yet a rigorous theoretical account of why and when it works has been lacking.

**Insufficiency of prior communication-complexity frameworks**: Prior work has applied communication complexity to study Transformer capabilities, but has not accurately modeled the unidirectional prefix→suffix information flow characteristic of causal attention.

**Practical implications**: Understanding bandwidth constraints can guide architectural design (e.g., increasing bandwidth), selection of inference strategies (CoT decomposition), and prediction of task difficulty.

## Method

### The BAPO Computational Framework

**Core Definition**: $(b_p, b_a)$-BAPO (Bounded Attention Prefix Oracle)
- The input sequence is split at an arbitrary position into a **prefix** and a **suffix**.
- The prefix stream (simulating the residual stream of early tokens) can perform unlimited computation but may transmit only $b_p$ bits of information to the suffix.
- The suffix stream (simulating the residual stream of the final token) can retrieve at most $b_a$ bits from the prefix via attention.
- Total communication bandwidth is $(b_p, b_a)$, and correctness must hold for **all possible split positions**.
- Simplifying assumptions: prefix/suffix have unlimited computational power, single-token output, and perfect positional encoding.

**Problem Classification (Three Hardness Classes)**
- **BAPO-easy**: Solvable with constant bandwidth $(O(1), O(1))$ (e.g., checking whether the first and last elements are equal).
- **BAPO-hard**: Requires super-constant bandwidth $\omega(1)$ that grows with input length (e.g., graph reachability, string equality).
- **BAPO-Σ-hard**: Bandwidth depends on alphabet size $|\Sigma|$ (e.g., the $k$-th largest element requires $O(|\Sigma|)$ for constant $k$).

**Six Analyzed Problems and Their Bandwidths**
1. First-Last (first/last comparison): BAPO-easy, $(0,1)$
2. Equality (string equality): BAPO-hard, $\Omega(n)$
3. Graph Reachability: BAPO-hard, at least $\Omega(\sqrt{n})$
4. Variable Tracking: BAPO-hard
5. Majority (majority vote): BAPO-hard
6. Code Tracing: reduces to graph reachability, BAPO-hard

**Theoretical Role of CoT**
- **Theorem**: For any decidable problem, a BAPO with constant bandwidth $(2,3)$ equipped with sufficiently many CoT reasoning tokens can simulate a single step of a Turing machine.
- Corollary: CoT renders constant-bandwidth BAPO Turing-complete; any BAPO-hard problem can be decomposed via CoT into a sequence of BAPO-easy sub-steps.
- Cost: A potentially large number of reasoning tokens may be required (experiments show o3 and Gemini 2.5 Flash use 10,000+ tokens to solve BAPO-hard problems).

### Experimental Design
- Six synthetic tasks are evaluated on GPT-4o, Claude Haiku, and Gemini 1.5 Pro, both without CoT and with CoT variants.
- Input size $n$ is varied (e.g., number of graph nodes from 10 to 200) to observe accuracy as a function of $n$.
- Evaluation is extended to real-world tasks: sentiment aggregation and code tracing.

## Key Experimental Results

### Table 1: BAPO-easy vs. BAPO-hard Performance on LLMs

| Task Type | Representative Problem | Bandwidth Requirement | GPT-4o / Claude / Gemini ($n=200$) |
|-----------|------------------------|----------------------|-------------------------------------|
| BAPO-easy | First-Last comparison | $(0,1)$ | ~100% accuracy, remains stable |
| BAPO-hard | Graph reachability | $\Omega(\sqrt{n})$ | Drops sharply to ~50% (chance level) as $n$ grows |
| BAPO-hard | String equality | $\Omega(n)$ | Begins to fail at moderate $n$ |
| BAPO-hard | Variable tracking | Super-constant | Fails even at small $n$ |

### Table 2: Effect of CoT on BAPO-hard Problems

| Model | CoT Setting | BAPO-hard Accuracy | Reasoning Tokens |
|-------|-------------|-------------------|-----------------|
| GPT-4o et al. (no CoT) | — | Degrades to chance as $n$ grows | N/A |
| GPT-4o (250-word CoT budget) | External CoT | Limited improvement; still degrades for $n>50$ | ~250 words |
| o3 / Gemini 2.5 Flash | Internal reasoning | Maintains high accuracy | 10,000+ tokens |

**Key Findings**: (1) The BAPO-easy/hard classification precisely predicts LLM success and failure patterns. (2) CoT is effective but requires sufficiently long reasoning chains — a 250-word budget is insufficient, whereas the internal long-form reasoning of o3 and Gemini resolves BAPO-hard problems. (3) Theory-consistent BAPO-hard failures are also observed on real-world tasks (sentiment aggregation, code tracing).

## Highlights & Insights

1. **Elegant theoretical framework**: The BAPO model abstracts complex Transformer behavior into a clean communication-bandwidth problem, establishing a clear correspondence between theory and experiment. Area Chair feedback describes it as a "clean and valuable theoretical contribution."
2. **Principled explanation of CoT**: This work provides the first formal proof of CoT's theoretical power — rendering BAPO Turing-complete — offering rigorous guarantees for CoT effectiveness rather than purely empirical observations.
3. **Precise predictive power**: The BAPO-easy/hard classification is validated across GPT-4o, Claude, and Gemini, with theoretical predictions closely matching experimental results.
4. **Architectural design implications**: The analysis shows that increasing MLP width or depth (i.e., computational capacity) cannot resolve these failures; the key is increasing attention communication bandwidth, providing directional guidance for architectural improvements.
5. **NeurIPS 2025 Spotlight**: Three of four reviewers assigned Accept (score 5); the Area Chair emphasized its importance for understanding LLMs and its potential to inspire follow-up research.

## Limitations & Future Work

1. **Over-simplified model abstraction**: The assumptions of unlimited computational capacity in prefix/suffix, perfect positional encoding, and single-token output deviate substantially from real Transformers, limiting direct applicability.
2. **Limited task coverage**: Only six synthetic problems are analyzed; many natural-language reasoning tasks are not yet covered, and most theoretical lower bounds remain loose.
3. **Unclear root cause of limited effective bandwidth**: The paper acknowledges that it does not explain "why LLM effective bandwidth is so limited," which may involve trade-offs with generalization.
4. **Insufficient CoT experiments**: Results under the 250-word CoT budget show limited benefit (Figure 4); only o3 and Gemini with internal long-form reasoning succeed, yet these are closed-source models that cannot be analyzed in depth.
5. **Absence of small-model training experiments**: Reviewers suggested training small Transformers from scratch on BAPO-hard tasks to isolate architectural factors; the paper does not include such experiments.

## Rating

| Dimension | Score | Notes |
|-----------|-------|-------|
| Novelty | ⭐⭐⭐⭐⭐ | A novel BAPO computational framework; first systematic application of the communication-bandwidth perspective to LLM reasoning failures |
| Technical Depth | ⭐⭐⭐⭐⭐ | Rigorous theoretical proofs (Turing-completeness, bandwidth lower bounds) combined with multi-model experimental validation; 39-page full paper |
| Experimental Thoroughness | ⭐⭐⭐ | Synthetic tasks validate theoretical predictions, but task coverage is limited and CoT experiments are insufficiently comprehensive |
| Practical Impact | ⭐⭐⭐⭐ | Provides theoretical foundations for understanding LLM limitations and CoT effectiveness, with implications for architectural design and inference strategy |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] When Reasoning Meets Compression: Understanding the Effects of LLMs Compression on Large Reasoning Models](../../ICLR2026/llm_reasoning/when_reasoning_meets_compression_understanding_the_effects_of_pruning_and_quant.md)
- [\[ICLR 2026\] Why is Your Language Model a Poor Implicit Reward Model?](../../ICLR2026/llm_reasoning/why_is_your_language_model_a_poor_implicit_reward_model.md)
- [\[ACL 2026\] When Is Thinking Enough? Early Exit via Sufficiency Assessment for Efficient Reasoning](../../ACL2026/llm_reasoning/when_is_thinking_enough_early_exit_via_sufficiency_assessment_for_efficient_reas.md)
- [\[NeurIPS 2025\] Many LLMs Are More Utilitarian Than One](many_llms_are_more_utilitarian_than_one.md)
- [\[ICLR 2026\] Native Reasoning Models: Training Language Models to Reason on Unverifiable Data](../../ICLR2026/llm_reasoning/native_reasoning_models_training_language_models_to_reason_on_unverifiable_data.md)

</div>

<!-- RELATED:END -->

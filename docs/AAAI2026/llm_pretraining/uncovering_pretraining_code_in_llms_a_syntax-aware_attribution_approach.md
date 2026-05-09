---
title: >-
  [Paper Note] Uncovering Pretraining Code in LLMs: A Syntax-Aware Attribution Approach
description: >-
  [AAAI 2026][LLM Pretraining][Membership Inference Attack] This paper proposes SynPrune — the first syntax-aware membership inference attack (MIA) method for code. By identifying 47 Python syntactic conventions and pruning syntactically determined tokens (retaining only tokens that reflect authorial style) when computing MIA scores, SynPrune achieves an average AUROC improvement of 15.4%, enabling effective detection of pretraining data attribution in code LLMs.
tags:
  - AAAI 2026
  - LLM Pretraining
  - Membership Inference Attack
  - Code Copyright
  - Syntax-Aware
  - Pretraining Data Detection
  - Python
date: 2026-05-08
content_hash: bf611405c5af38c1
---

# Uncovering Pretraining Code in LLMs: A Syntax-Aware Attribution Approach

**Conference**: AAAI 2026
**arXiv**: [2511.07033](https://arxiv.org/abs/2511.07033)
**Code**: None
**Area**: LLM Pretraining
**Keywords**: Membership Inference Attack, Code Copyright, Syntax-Aware, Pretraining Data Detection, Python

## TL;DR
This paper proposes SynPrune — the first syntax-aware membership inference attack (MIA) method for code. By identifying 47 Python syntactic conventions and pruning syntactically determined tokens (retaining only tokens that reflect authorial style) when computing MIA scores, SynPrune achieves an average AUROC improvement of 15.4%, enabling effective detection of pretraining data attribution in code LLMs.

## Background & Motivation
**Background**: The coding capabilities of code LLMs (Pythia, GPT-Neo, StableLM, etc.) derive from large-scale training on open-source code. Membership inference attacks (MIA) are a core technique for detecting whether specific data was included in a training set. Existing methods include shadow model approaches (GotCha), synonym substitution (Mattern), low-probability token detection (MIN-K%), and token frequency calibration (Zhang), but all treat code as plain text.

**Limitations of Prior Work**: (a) Duan et al. found that existing MIAs on LLMs perform only marginally better than random guessing; (b) code copyright disputes are increasing (e.g., GPL violation lawsuits have precedent), yet MIA research specifically targeting code remains scarce; (c) existing MIAs ignore a critical property of code — many tokens are dictated by syntactic rules (e.g., a colon must follow a function definition; indentation is mandated by PEP8), and these tokens exhibit high prediction probability regardless of whether the code was in the training set, contributing noise rather than signal to membership inference.

**Key Challenge**: A large proportion of tokens in code (parentheses, colons, indentation, keyword collocations, etc.) appear as a necessary consequence of programming language syntax rules and do not reflect authorial identity. Nevertheless, existing MIAs treat all tokens with equal weight, diluting the truly discriminative signal.

**Goal**: To extract tokens that genuinely reflect "authorial creative intent" in code, excluding syntactically determined tokens, thereby improving the detection power of code MIA.

**Key Insight**: The observation that code = author logic + language syntax. The syntactic convention component is deterministic (e.g., `for x in` must be followed by an iterable; function definitions use `def` followed by `(`), and these tokens carry no membership information.

**Core Idea**: Prune tokens whose appearance is syntactically "inevitable," and perform membership inference using only the log-probabilities of author-characteristic tokens, substantially improving discriminability.

## Method

### Overall Architecture
The input is a Python function to be examined; the output is a binary member/non-member decision. The pipeline consists of three stages: (1) Token preprocessing — tokenization followed by sub-token splitting according to syntactic conventions; (2) Syntactic pruning — identification and labeling of syntactically determined tokens; (3) Membership score computation — aggregation using the log-probabilities of retained tokens only.

### Key Designs

1. **Summary of 47 Python Syntactic Conventions**:

    - Function: Systematically define which tokens are syntactically "inevitable."
    - Mechanism: Two authors manually reviewed the official Python 3.11 documentation (Data Model §3, Expressions §6, Simple and Compound Statements §7–8) and defined conventions as ⟨condition, consequence⟩ tuples, grouped into four categories: **Data Model** (`[` for lists must be paired with `]`; `{` for dicts with `}`); **Expressions** (a function-call identifier must be followed by `)`; a conditional expression `if cond` must be followed by `else`); **Simple Statements** (`import module` may be followed by `as`); **Compound Statements** (`for target_list` must be followed by `in`; `if` must be followed by `:`, indentation, and a newline; function definition `def f(` uses `,` and `)` for parameters).
    - Design Motivation: The appearance of these tokens is an inevitable consequence of syntactic rules, unrelated to whether the model has "memorized" the code — they introduce substantial noise.

2. **SynPrune Pruning Algorithm**:

    - Function: Remove syntactically determined tokens prior to computing the MIA score.
    - Mechanism: The LLM's tokenizer is first used to obtain the token sequence along with per-token prediction probabilities. Compound tokens are then split according to syntactic conventions (e.g., `print(` is split into `print` and `(`). Each sub-token is checked against the consequence of every defined convention. Matching leverages Python's AST module to locate syntactic nodes and verify condition tokens. A labeling function $\ell(x_i) \in \{0, 1\}$ assigns 0 (pruned) to tokens where all sub-tokens match a convention consequence, and 1 (retained) otherwise.
    - Design Motivation: BPE tokenization may merge syntactic tokens with content tokens (e.g., `print(`), necessitating splitting before classification.

3. **Syntax-Pruned Perplexity Score (SPP)**:

    - Function: Compute the final membership inference score from retained tokens.
    - Mechanism: $\text{SPP}(x) = \frac{1}{|\mathcal{X}_1|} \sum_{i \in \mathcal{X}_1} -\log p(x_i)$, where $\mathcal{X}_1 = \{x_i | \ell(x_i) = 1\}$ is the set of retained tokens. A threshold $\epsilon$ is set such that $\text{SPP}(x) > \epsilon$ classifies the input as a member.
    - Design Motivation: After pruning syntactic tokens, the log-probabilities of the remaining tokens more accurately reflect the model's memorization of specific code content, substantially improving the signal-to-noise ratio.

### Benchmark Construction
- **Member data**: 1,000 Python functions randomly sampled from the Pile dataset (released in 2021 and widely used for pretraining Pythia, GPT-Neo, StableLM, etc.).
- **Non-member data**: 1,000 Python functions extracted from GitHub repositories created after January 2024, with originality verified through triple deduplication (function name, variable name, and call chain matching).
- This construction is more realistic than the approach of Yang et al., who sampled from CodeXGLUE and applied post-hoc labeling.

## Key Experimental Results

### Main Results

| Method | Average AUROC Gain | Note |
|--------|--------------------|------|
| SynPrune vs. SOTA | **+15.4%** | Averaged across 4 code LLMs |
| SynPrune vs. MIN-K% | Significant improvement | MIN-K% selects low-probability tokens but does not distinguish syntactic from content tokens |
| SynPrune vs. Frequency Calibration | Significant improvement | Frequency calibration is insensitive to the syntactic structure of code |

### Ablation Study (Contribution by Syntactic Convention Category)

| Syntactic Category | Contribution | Note |
|--------------------|--------------|------|
| Compound Statements | Largest | `for`/`if`/`def` are the most frequent structures in Python |
| Data Model | Moderate | Bracket matching contributes consistently |
| Expressions | Moderate | Parentheses and commas in function calls |
| Simple Statements | Smallest | `import` statements are relatively infrequent |

### Key Findings
- The +15.4% AUROC improvement is substantial, confirming that existing MIA methods are significantly impeded by syntactic noise tokens.
- The method is robust across function lengths — improvements are observed for both short and long functions.
- Compound statement conventions contribute the most, as `for`/`if`/`while`/`def` constitute the most frequent syntactic structures in Python code.
- Validation on a benchmark of genuine members (Pile training set) versus genuine non-members (post-2024 code) is more reliable than prior synthetic benchmarks.

## Highlights & Insights
- The idea of **leveraging the deterministic syntactic properties of programming languages** to filter noise tokens is both elegant and powerful. Unlike natural language, code contains a large volume of formally mandated tokens whose presence is inevitable; these tokens dilute the MIA signal. SynPrune precisely identifies and removes this noise.
- **Strong generalizability**: Although this work focuses on Python, the ⟨condition, consequence⟩ formulation of the 47 conventions can be systematically applied to any programming language.
- **Benchmark construction as a contribution in itself**: The realistic benchmark of Pile members paired with post-2024 non-members is more reliable than prior synthetic benchmarks and can serve as a resource for future research.

## Limitations & Future Work
- Validation is limited to Python; applying the method to other languages (Java, C++, etc.) would require re-cataloguing syntactic conventions.
- The 47 conventions were manually compiled and may omit certain edge cases.
- The method operates solely at the token-level probability, without considering higher-level code semantics (e.g., control flow graphs, API usage patterns).
- The selection of the binary classification threshold $\epsilon$ is not discussed in detail.

## Related Work & Insights
- **vs. MIN-K%**: MIN-K% assumes that member data is unlikely to contain low-probability tokens, but does not distinguish the cause of low probability (syntactic inevitability vs. authorial choice). SynPrune precisely separates these two cases.
- **vs. GotCha**: GotCha requires training shadow models (computationally expensive and dependent on pretraining data access), whereas SynPrune is reference-free, requiring only the target model's token probabilities.

## Rating
- Novelty: ⭐⭐⭐⭐ First work to integrate syntactic conventions into code MIA; the approach is concise and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-model validation + ablation study + length robustness + realistic benchmark.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clear; method explanation is intuitive.
- Value: ⭐⭐⭐⭐ Has direct applicability to copyright detection for code LLMs; value will increase as litigation involving code LLMs grows.

## Additional Notes
- The methodology and experimental design of this work offer a valuable reference for related research areas.
- Future work may validate the generalizability and scalability of the method across broader scenarios and larger scales.
- There is potential research value in combining this work with recent related efforts (e.g., intersections with RL/MCTS or multimodal methods).
- The deployment feasibility and computational efficiency of the method should be assessed in light of practical application requirements.
- The choice of datasets and evaluation metrics may affect the generalizability of the conclusions; cross-validation on additional benchmarks is recommended.

## Additional Notes
- The methodology and experimental design of this work offer a valuable reference for related research areas.
- Future work may validate the generalizability and scalability of the method across broader scenarios and larger scales.
- There is potential research value in combining this work with recent related efforts (e.g., intersections with RL/MCTS or multimodal methods).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Efficient Pre-Training of LLMs via Topology-Aware Communication Alignment on More Than 9600 GPUs](../../NeurIPS2025/llm_pretraining/efficient_pre-training_of_llms_via_topology-aware_communication_alignment_on_mor.md)
- [\[AAAI 2026\] Learning Procedural-aware Video Representations through State-Grounded Hierarchy Unfolding](learning_procedural-aware_video_representations_through_state-grounded_hierarchy.md)
- [\[ICLR 2026\] Identifying and Evaluating Inactive Heads in Pretrained LLMs](../../ICLR2026/llm_pretraining/identifying_and_evaluating_inactive_heads_in_pretrained_llms.md)
- [\[NeurIPS 2025\] Enhancing Training Data Attribution with Representational Optimization](../../NeurIPS2025/llm_pretraining/enhancing_training_data_attribution_with_representational_optimization.md)
- [\[NeurIPS 2025\] Understanding and Enhancing Mask-Based Pretraining towards Universal Representations](../../NeurIPS2025/llm_pretraining/understanding_and_enhancing_mask-based_pretraining_towards_universal_representat.md)

</div>

<!-- RELATED:END -->

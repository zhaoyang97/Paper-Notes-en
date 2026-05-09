---
title: >-
  [Paper Note] Agnostics: Learning to Synthesize Code in Any Programming Language with a Universal RL Environment
description: >-
  [ICLR 2026][low-resource programming languages] This paper proposes Agnostics, a language-agnostic post-training pipeline that reformulates programming tasks as I/O behavioral specifications and trains LLMs with a universal verifier and GRPO-based reinforcement learning to generate code in any programming language. The approach enables a Qwen 4B model to match the performance of 16B–70B models on five low-resource languages: Lua, Julia, R, OCaml, and Fortran.
tags:
  - ICLR 2026
  - low-resource programming languages
  - RLVR
  - language-agnostic verifier
  - GRPO
  - code execution sandbox
date: 2026-05-08
content_hash: 0e8c10c8f383834e
---

# Agnostics: Learning to Synthesize Code in Any Programming Language with a Universal RL Environment

**Conference**: ICLR 2026
**arXiv**: [2508.04865](https://arxiv.org/abs/2508.04865)
**Code**: [https://github.com/sunblaze-ucb/agnostics](https://github.com/sunblaze-ucb/agnostics) (agnostics.abgru.me)
**Area**: Other
**Keywords**: low-resource programming languages, RLVR, language-agnostic verifier, GRPO, code execution sandbox

## TL;DR
This paper proposes Agnostics, a language-agnostic post-training pipeline that reformulates programming tasks as I/O behavioral specifications and trains LLMs with a universal verifier and GRPO-based reinforcement learning to generate code in any programming language. The approach enables a Qwen 4B model to match the performance of 16B–70B models on five low-resource languages: Lua, Julia, R, OCaml, and Fortran.

## Background & Motivation

**Background**: LLMs excel at high-resource languages such as Python and JavaScript, but perform poorly on low-resource languages like Lua (0.53%), Julia (0.10%), R (0.35%), and Fortran (0.07%). These languages lack not only pretraining data but also post-training datasets, testing tools, and RL infrastructure.

**Limitations of Prior Work**: (a) Each new language appears to require a dedicated dataset, testing framework, and RL environment, imposing substantial engineering overhead. (b) Synthetic data approaches such as MultiPL-T require approximately 500 lines of prompt/test translators per language, and rejection sampling is highly inefficient on hard problems (≈30% acceptance rate). (c) Upsampling low-resource languages during pretraining or fine-tuning on their data yields only marginal improvements.

**Key Challenge**: RL requires reliable reward signals (i.e., code correctness verification), yet building a verification environment for each language constitutes the heaviest engineering burden in the pipeline.

**Goal**: Design a universal RL post-training pipeline in which supporting a new language requires only 4–5 lines of YAML configuration.

**Key Insight**: The key insight is that for a large class of programming tasks, **correctness can be determined purely from the program's externally observable behavior (I/O)**, allowing the verifier to be fully decoupled from the target language.

**Core Idea**: Unify all programming tasks into a "read stdin → compute → write stdout" I/O format and assign rewards to code in any language using a single language-agnostic verifier, enabling universal RLVR.

## Method

### Overall Architecture
Agnostics consists of two main stages: (1) **Data Preparation** — an LLM rewrites existing datasets (MBPP, Codeforces, etc.) from language-specific formats into language-agnostic I/O descriptions with test cases, which are then adapted to target languages via compact configuration files; (2) **Training** — GRPO-based RLVR is performed inside a language-agnostic code execution sandbox.

### Key Designs

1. **I/O Format Unification (Dataset Preparation)**

    - **Function**: Convert all programming tasks into a unified format describing program I/O behavior.
    - **Mechanism**: An LLM rewrites tasks such as Python functions with assert-based tests in MBPP into problem descriptions with I/O example pairs $(in_k, out_k)$ following the "read stdin → compute → write stdout" convention. The LLM is explicitly prompted to specify I/O conventions (decimal places, delimiters, ordering, etc.) to eliminate ambiguity.
    - **Design Motivation**: This format is identical for any programming language — correctness is determined solely by whether the program runs and produces the correct output, making the verifier language-agnostic.

2. **Minimal Language Configuration**

    - **Function**: Configure a new programming language using 4–5 lines of YAML.
    - **Mechanism**: The configuration file specifies `install` (compiler installation), `filename` (source file name), `compile` (compilation command), `execute` (run command), and `prompt` (prompt prefix). For example, the R configuration only needs to specify installation of tidyverse, the filename `snippet.R`, the execution command `Rscript`, and a brief note on `readLines` usage.
    - **Design Motivation**: This reduces the engineering cost of supporting a new language from hundreds of lines of translator code to a few lines of configuration. For extremely low-resource languages (e.g., OCaml, Fortran), the model can first generate erroneous code, after which GPT-o3 analyzes common errors to automatically generate prompt prefixes.

3. **Language-Agnostic Code Execution Sandbox**

    - **Function**: Safely and efficiently compile and run code in any language for reward computation.
    - **Mechanism**: OCI containers are used, with per-language images that include the appropriate compiler. An execution harness runs persistently inside the container, receiving (program, I/O examples, timeout) triples, writing source files, compiling, executing, and comparing outputs. A fully correct submission receives reward 1; otherwise 0.
    - **Design Motivation**: Safety constraints limit CPU, memory, filesystem access, and output size (5 MB cap) to prevent pathological behaviors such as infinite loops, macro expansion explosions, and massive outputs. Reusing warm containers is orders of magnitude faster than spawning new ones per request. RAM disks accelerate compilation.

4. **GRPO Training**

    - **Function**: Apply Group Relative Policy Optimization for reinforcement learning.
    - **Mechanism**: For each prompt, $G=32$ candidates are sampled and verified by the sandbox to obtain binary rewards $R_i \in \{0,1\}$. Group-relative advantages are computed as $\hat{A}_i = \frac{R_i - \text{mean}}{\text{std}}$, and a standard clipped PPO update is applied. The KL divergence term is omitted.
    - **Design Motivation**: Partial correctness rewards (awarding partial credit when the program runs without error but produces incorrect output) were attempted but quickly exploited by the model, which learned to generate empty programs or hard-code public test cases.

### Training Details
- AdamW optimizer, lr = 5e-6, cosine decay, 0.1-epoch warmup
- Batch size: 4 prompts × group size 32
- Training temperature: 0.7; evaluation temperature: 0.2
- Single epoch of training
- Ray used for distributed execution (GPU nodes for training, CPU nodes for code execution)

## Key Experimental Results

### Main Results — Ag-LiveCodeBench-X (New Multilingual Hard Benchmark)

| Model | Lua | Julia | R | OCaml | Fortran |
|-------|-----|-------|---|-------|---------|
| Llama 3.3 70B | 25 | 22 | 13 | 7 | 3 |
| Qwen 3 32B | 22 | 26 | 17 | 2 | 1 |
| Qwen 3 4B (base) | 11 | 10 | 10 | 1 | 0 |
| **Qwen3-4B-CF-X (Ours)** | **23** | **22** | **15** | **7** | **15** |
| **Qwen3-8B-CF-X (Ours)** | **25** | **25** | **19** | **7** | **17** |

After training, the 4B model reaches performance comparable to 70B baselines. OCaml improves from 1% to 7% (matching Claude Sonnet 4), and Fortran improves from 0% to 15% (surpassing all baselines).

### MultiPL-E Results
Significant improvements are also observed on the easier MultiPL-E benchmark, achieving state-of-the-art results among models with ≤16B parameters.

### Key Findings
- Successful training from near-zero baselines (e.g., Fortran 0%, OCaml 1%) demonstrates that RLVR can bootstrap learning with minimal initial capability.
- Binary rewards (correct = 1, otherwise = 0) are more effective than partial rewards, which are susceptible to model exploitation.
- Training on Ag-Codeforces-X (competitive programming) substantially outperforms Ag-MBPP-X (simple tasks), confirming the value of harder training data.
- Per-trajectory training cost is low, and the entire pipeline is highly automated.

## Highlights & Insights
- The idea of **converting a language-specific problem into a language-agnostic one** is particularly elegant — I/O behavior serves as a universal interface for program semantics, producible by any language and verifiable by a single validator.
- **4-line YAML configuration for new languages** reduces the barrier to extending the pipeline to virtually zero, offering substantial practical value for low-resource language communities.
- The finding that **small model + domain-specific RL >> large model** is once again validated: a 4B RLVR model matches a 70B baseline, underscoring that verification signals are far more valuable than model scale in code generation tasks.
- The engineering design of the execution sandbox is thorough (warm containers, RAM disks, output size limits, dual timeouts for compilation and execution), constituting a genuinely production-ready system.

## Limitations & Future Work
- The approach is limited to tasks expressible in the "stdin/stdout I/O" format and is not applicable to programming tasks requiring API calls, GUI interaction, or file system operations.
- Prompt prefixes in language configurations still require manual or semi-automatic authoring, which may be non-trivial for extremely rare languages.
- Only output correctness is verified; code style, efficiency, and idiomatic usage are not assessed — the trained code may not be idiomatic.
- Training is performed for only one epoch; longer training or curriculum learning may yield further improvements.
- Validation on large models (>16B) is absent — it remains an open question whether the marginal benefit of RLVR diminishes for already strong large models.

## Related Work & Insights
- **vs. MultiPL-T**: MultiPL-T requires ~500 lines of translators per language, and rejection sampling demands orders of magnitude more compute on hard problems; Agnostics requires only 4 lines of YAML plus RLVR.
- **vs. DeepSeek R1**: R1 applies code RL effectively for Python but does not publicly detail low-resource language handling; Agnostics specifically targets low-resource language settings.
- **vs. Synthetic SFT Data**: Cassano et al. demonstrated that synthetic data without verification is nearly ineffective for low-resource languages; RLVR addresses this limitation through verification signals.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The core idea of achieving language-agnostic RL via I/O behavioral verification is both simple and profound, representing a practical and conceptually elegant innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Experiments span 5 low-resource languages × multiple models × two benchmarks, plus a newly constructed benchmark Ag-LCB-X — highly comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear and well-organized; system design decisions are consistently motivated and explained.
- **Value**: ⭐⭐⭐⭐⭐ Provides a complete infrastructure-level solution for LLM post-training on low-resource programming languages.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] A Single Architecture for Representing Invariance Under Any Space Group](a_single_architecture_for_representing_invariance_under_any_space_group.md)
- [\[NeurIPS 2025\] Computable Universal Online Learning](../../NeurIPS2025/others/computable_universal_online_learning.md)
- [\[ICLR 2026\] AnyUp: Universal Feature Upsampling](anyup_universal_feature_upsampling.md)
- [\[ICLR 2026\] DA-AC: Distributions as Actions — A Unified RL Framework for Diverse Action Spaces](distributions_as_actions_a_unified_framework_for_diverse_action_spaces.md)
- [\[ICLR 2026\] Jackpot: Optimal Budgeted Rejection Sampling for Extreme Actor-Policy Mismatch RL](jackpot_optimal_budgeted_rejection_sampling_for_extreme_actor-policy_mismatch_re.md)

<!-- RELATED:END -->

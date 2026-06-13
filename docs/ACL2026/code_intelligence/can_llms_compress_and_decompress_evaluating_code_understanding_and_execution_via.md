---
title: >-
  [Paper Note] Can LLMs Compress (and Decompress)? Evaluating Code Understanding and Execution via Invertibility
description: >-
  [ACL 2026][Code Intelligence][Code reasoning] Ours proposes RoundTripCodeEval (RTCE): a code reasoning benchmark utilizing 4 lossless compression algorithms (LZW/AE/RLE/Huffman) to construct 250 inputs × 4 subtasks = 100…
tags:
  - "ACL 2026"
  - "Code Intelligence"
  - "Code reasoning"
  - "bidirectional execution"
  - "compression algorithms"
  - "self-consistency"
  - "evaluation"
date: 2026-05-08
content_hash: 793afb79fbfc7f63
---

# Can LLMs Compress (and Decompress)? Evaluating Code Understanding and Execution via Invertibility

**Conference**: ACL 2026 Findings  
**arXiv**: [2601.13398](https://arxiv.org/abs/2601.13398)  
**Code**: https://github.com/Nickil21/round-trip-code-compression  
**Area**: Code Intelligence  
**Keywords**: Code reasoning, bidirectional execution, compression algorithms, self-consistency, evaluation

## TL;DR
Ours proposes RoundTripCodeEval (RTCE): a code reasoning benchmark utilizing 4 lossless compression algorithms (LZW/AE/RLE/Huffman) to construct 250 inputs × 4 subtasks = 1000 strict round-trips (where encode→decode must achieve bit-precise restoration). Results demonstrate that even QwQ-32B maintains 0% EM on Huffman encoding, a failure that cannot be rectified via SFT or self-reflection.

## Background & Motivation

**Background**: Code-LLMs (such as DeepSeek-Coder, Qwen2.5-Coder, and StarCoder2) have demonstrated strong performance on code generation benchmarks like HumanEval/MBPP. Execution reasoning evaluations (e.g., CRUXEval, CodeIO, CodeMind, REVAL) typically assess execution in either a single forward or backward direction.

**Limitations of Prior Work**: Existing evaluations are either unidirectional (measuring only forward or backward execution) or based on "semantic equivalence" (e.g., IdentityChain for code↔spec, RTC for code↔NL). Semantic equivalence is a permissive standard—generated code is considered correct as long as its behavior is identical, allowing models to potentially exploit pattern matching or memorization. It fails to prove that a model truly understands the internal state machine and data flow of an algorithm.

**Key Challenge**: A model may achieve high scores in forward execution (through surface pattern matching) but fail in backward reasoning. Alternatively, it may appear correct in both directions independently while failing to close the round-trip loop. This indicates an inconsistency in the model’s internal representations, a flaw that unidirectional benchmarks cannot capture. "Forward correctness was fragile, derived from template matching."

**Goal**: To design an evaluation capable of distinguishing between "performance derived from pattern matching" and "genuine understanding of algorithmic semantics."

**Key Insight**: Lossless compression algorithms are naturally bijective. The operations $\text{enc}(x)=z$ and $\text{dec}(z)=x$ must be perfectly reversible, providing a strict round-trip constraint $\text{dec}(\text{enc}(x))=x$ that is significantly harder to bypass via memorization than semantic equivalence.

**Core Idea**: Code understanding is redefined as a "code invertibility problem." Using round-trip exact-match (EM) as the evaluation signal, a 16-dimensional diagnostic grid is formed by 4 compression algorithms across 4 task variants (encode, decode, encode⁻¹, decode⁻¹) to expose systematic failures hidden by forward-only evaluations.

## Method

### Overall Architecture
The RTCE construction involves three phases: (1) **Generation**—250 synthetic inputs across 4 data families (pattern strings, Apache logs, YAML configs, CSV tables, totaling 36 sub-classes) to ensure diversity; (2) **Verification**—Python reference implementations of 4 compression algorithms (LZW/AE/RLE/Huffman) generate deterministic ground truth under fixed seeds; (3) **Evaluation**—4 tasks per input (O/P Pred, O/P Pred-I, I/P Pred, I/P Pred-I) are scored using EM, Edit Similarity, and Pass@5. All tasks are execution-free, requiring the model to simulate execution mentally.

### Key Designs

1.  **Round-trip framework based on bijective compression**:
    *   **Function**: Constructs a strict exact-match loop that cannot be bypassed by semantic equivalence or partial similarity.
    *   **Mechanism**: Defines $\mathsf{enc}:\mathcal{X}\to\mathcal{Z}$ and $\mathsf{dec}:\mathcal{Z}\to\mathcal{X}$ with the constraint $\forall x\in\mathcal{X},\ \mathsf{dec}(\mathsf{enc}(x))=x$. Models are tasked with $x\to z$ (forward encoding), $z\to x'$ (forward decoding), $x\to z$ via inverting dec (inferring encoding behavior from the decoding function), and $z\to x'$ via inverting enc (inferring decoding behavior from the encoding function). The "inversion" tasks are critical as they force the model to reason about one function as the inverse of another.
    *   **Design Motivation**: Unlike semantic equivalence in RTC (code↔NL), bit-precise restoration allows no information loss, exposing internal contradictions where both directions might seem correct independently but fail when combined.

2.  **4 Compression algorithms covering the encoding paradigm spectrum**:
    *   **Function**: Tests different reasoning bottlenecks through varied algorithmic design patterns.
    *   **Mechanism**: LZW tests dictionary maintenance (dynamic state); AE tests probability interval accumulation (floating-point precision + long-range dependencies); RLE tests consecutive run aggregation (simplest bijection); Huffman tests prefix coding and tree construction (multi-stage hierarchical process). These span from simple (RLE) to complex (Huffman). Experimentally, all 15 models achieved 0% EM on Huffman encoding, proving the issue lies in state-machine comprehension.
    *   **Design Motivation**: Using a single algorithm might bias results toward specific properties; testing four distinct classes distinguishes between specific knowledge gaps and general state-tracking deficiencies.

3.  **3 Diagnostic paradigms: zero-shot / self-reflection / SFT**:
    *   **Function**: Exhausts standard methods for performance improvement to prove that the performance gap in RTCE is fundamental.
    *   **Mechanism**: Zero-shot measures raw capability; multi-turn self-reflection uses a critique/revision loop; SFT employs a 5-stage pipeline including execution injection (@snoop), trace filtering, natural language reasoning chain generation using Qwen3-32B, and LoRA fine-tuning on QwQ-32B. None of these methods resolved the 0% EM on Huffman encoding.
    *   **Design Motivation**: Strips away the possibility that "the model is simply not well-prompted" from the possibility of underlying architectural limitations.

### Evaluation Metrics
EM (exact match with floating-point tolerance of $10^{-3}$), ES (normalized Levenshtein distance for partial credit), and Pass@5 (best of $n=5$ samples). EM serves as the primary metric, as ES scores (8%–20% even when EM=0) indicate outputs that "look similar" but lack precision.

## Key Experimental Results

### Main Results
Pass@5 synthesis for 15 LLMs across 4 algorithms and 4 tasks (abridged):

| Model | Scale | RLE Avg | LZW Avg | AE Avg | Huffman Avg | Avg |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Llama-3.2-1B | 1B | 0.15 | 0.05 | 0.34 | 0.08 | 0.16 |
| Phi-3-mini-128k | 3.8B | 12.01 | 3.65 | 2.60 | 1.54 | 4.95 |
| Qwen2.5-7B | 7.6B | 17.39 | 4.46 | 6.55 | 2.65 | 7.76 |
| DeepSeek-R1-Distill-14B | 14.8B | 26.97 | 14.03 | 10.08 | 3.15 | 13.56 |
| Codestral-22B | 22.2B | 30.68 | 7.77 | 1.76 | 1.50 | 10.43 |
| **QwQ-32B** | 32.8B | **57.23** | 24.14 | **15.71** | 5.50 | **25.65** |
| Qwen2.5-Coder-32B | 32.8B | 41.51 | 21.06 | 8.45 | 3.15 | 18.54 |
| DeepSeek-R1-Distill-32B | 32.8B | 36.37 | 23.81 | 12.74 | 3.98 | 19.23 |
| deepseek-coder-33b | 33.3B | 13.71 | 3.44 | 3.34 | 1.21 | 5.43 |

**Key Findings**: (1) Huffman encoding resulted in 0% for all models—the three-step process of frequency table construction, tree building, and variable-length code output proved impossible for current LLMs. (2) Reasoning-focused training (QwQ vs. Qwen2.5-Coder with identical parameters/tokenizers) improved AE by $1.86\times$, confirming the bottleneck is logical reasoning rather than tokenization. (3) Decoding is generally easier than encoding, except for AE: QwQ achieved 27.6% on AE encoding but only 2.3% on decoding due to the complex inverse floating-point interval operations required.

### Ablation Study: SFT on QwQ-32B (Pass@5)

| Algorithm | temp | I/P Pred | I/P Pred-I | O/P Pred | O/P Pred-I |
| :--- | :--- | :--- | :--- | :--- | :--- |
| AE | 0.2 | 30.77 | 23.08 | **78.57** | **84.62** |
| Huffman | 0.2 | 35.00 | 50.00 | **0.00** | **0.00** |
| LZW | 0.2 | 62.50 | 62.50 | 87.50 | 87.50 |
| RLE | 0.2 | 76.47 | 86.00 | 80.00 | 86.00 |

After SFT, Huffman encoding remained at 0%, while decoding increased to 50%—suggesting the trace-derived reasoning chain only learned "surface decoding templates" without internalizing the bijective state transition structure.

### Key Findings
*   **The Huffman Paradox**: All models failed Huffman encoding (0%) while some succeeded at decoding (7-11%). Decoding only requires traversal (local lookup) on a given tree, whereas encoding requires global multi-stage reasoning (frequency calculation + tree construction).
*   **Self-reflection Saturation**: Critique-revision loops saturate after the first turn. While a single round can fix shallow reasoning errors, systematic state-tracking failures are not self-correctable.
*   **SFT Bias (Forward vs. Inverse)**: On AE, forward performance reached 78.6% while inverse performance reached only 23–30%, suggesting LoRA adapters overfit to the surface form of traces rather than learning bijective invariants.
*   **Tokenization is not the Bottleneck**: Comparison between QwQ and Qwen2.5-Coder shows massive performance gaps despite using the same tokenizer, indicating that training objectives (reasoning vs. code) are the primary drivers.
*   **ES > 0 with EM = 0**: The prevalence of outputs that appear correct but lack precision confirms that RTCE's strict EM requirement exposes fragility missed by other benchmarks.

## Highlights & Insights
*   **Operationalizing "Understanding" as "Invertibility"**: Utilizing bijections transforms abstract understanding into a quantifiable exact-match signal. This methodology is extensible to any task with a natural inverse (e.g., encryption↔decryption, serialization↔deserialization).
*   **Diagnostic Evaluation Triad**: Testing via zero-shot, self-reflection, and SFT simultaneously reinforces the conclusion that Transformers have fundamental defects in bijective state tracking.
*   **Huffman Paradox as a Gap Benchmark**: The stark asymmetry between encoding (0%) and decoding (11%) identifies "multi-stage global state construction" as a specific objective for future code-LLM development.
*   **Synthetic yet Realistic**: The four data families simulate real-world developer artifacts to avoid data contamination from GitHub.

## Limitations & Future Work
*   Restricted to Python; extension to other languages is planned.
*   The sample size (1000) and algorithm count (4) limit the statistical stability of fine-grained per-category analysis.
*   Execution-free evaluation cannot account for runtime phenomena like side-effects or exceptions.
*   Primary focus on open-source models; the generalizability of these findings to top-tier closed-source models (GPT-4, Claude) remains an open question.
*   Personal Insight: Bijection is only one dimension of code understanding; other forms of invertibility like de-compilation or symbolic execution remain uncovered.

## Related Work & Insights
*   **vs. IdentityChain (Min 2024)**: Evaluates spec↔code consistency based on semantic equivalence, whereas Ours requires much stricter exact bijections.
*   **vs. RTC (Allamanis 2024)**: Performs code↔NL description round-trips at a semantic level; Ours operates at a bit-precise data level.
*   **vs. CodeIO/CRUXEval**: These score forward and backward directions independently; Ours emphasizes that both directions must be self-consistent.
*   **vs. CodeMind/REVAL/CACP**: These rely on complex trace or concept-level annotations; Ours automatically scores via round-trip EM without additional labels.

## Rating
*   Novelty: ⭐⭐⭐⭐⭐ Introducing round-trip bijections to code reasoning and operationalizing invertibility is a truly fresh perspective.
*   Experimental Thoroughness: ⭐⭐⭐⭐ Extensive testing across 15 models and 3 paradigms, though closed-source models are absent.
*   Writing Quality: ⭐⭐⭐⭐ Clear mathematical definitions and well-articulated task distinctions.
*   Value: ⭐⭐⭐⭐⭐ Exposing the systematic failure of Transformers in stateful bijections provides a clear target for the code reasoning community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Sense and Sensitivity: Examining the Influence of Semantic Recall on Long Context Code Understanding](sense_and_sensitivity_examining_the_influence_of_semantic_recall_on_long_context.md)
- [\[ACL 2026\] SWE-QA: Can Language Models Answer Repository-level Code Questions?](swe-qa_can_language_models_answer_repository-level_code_questions.md)
- [\[AAAI 2026\] Towards Better Code Understanding in Decoder-Only Models with Contrastive Learning](../../AAAI2026/code_intelligence/towards_better_code_understanding_in_decoder-only_large_language_models_via_hie.md)
- [\[ACL 2026\] CodeRL+: Improving Code Generation via Reinforcement with Execution Semantics Alignment](coderl_improving_code_generation_via_reinforcement_with_execution_semantics_alig.md)
- [\[ACL 2026\] SecureVibeBench: Evaluating Secure Coding Capabilities of Code Agents with Realistic Vulnerability Scenarios](securevibebench_evaluating_secure_coding_capabilities_of_code_agents_with_realis.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ACL 2026\] SWE-QA: Can Language Models Answer Repository-level Code Questions?](swe-qa_can_language_models_answer_repository-level_code_questions.md)
- [\[ACL 2025\] TeXpert: A Multi-Level Benchmark for Evaluating LaTeX Code Generation by LLMs](../../ACL2025/code_intelligence/texpert_a_multi-level_benchmark_for_evaluating_latex_code_generation_by_llms.md)
- [\[ACL 2026\] CodeRL+: Improving Code Generation via Reinforcement with Execution Semantics Alignment](coderl_improving_code_generation_via_reinforcement_with_execution_semantics_alig.md)
- [\[ACL 2026\] Sense and Sensitivity: Examining the Influence of Semantic Recall on Long Context Code Understanding](sense_and_sensitivity_examining_the_influence_of_semantic_recall_on_long_context.md)
- [\[ACL 2026\] DUET: Dual Execution for Test Output Prediction with Generated Code and Pseudocode](duet_dual_execution_for_test_output_prediction_with_generated_code_and_pseudocod.md)

</div>

<!-- RELATED:END -->

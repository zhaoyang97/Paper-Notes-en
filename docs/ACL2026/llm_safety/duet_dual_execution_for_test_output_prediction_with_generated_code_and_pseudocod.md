---
title: >-
  [Paper Note] DUET: Dual Execution for Test Output Prediction with Generated Code and Pseudocode
description: >-
  [ACL 2026][LLM Safety][Test output prediction] This paper proposes DUET, a dual-path framework that combines direct code execution with LLM-based pseudocode execution. The two paths are complementary—the former is reliab…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Test output prediction"
  - "pseudocode execution"
  - "dual execution"
  - "code generation"
  - "functional majority voting"
date: 2026-05-08
content_hash: b7242cddde23f677
---

# DUET: Dual Execution for Test Output Prediction with Generated Code and Pseudocode

**Conference**: ACL 2026
**arXiv**: [2604.11514](https://arxiv.org/abs/2604.11514)
**Code**: [GitHub](https://github.com/ldilab/DuET)
**Area**: LLM Safety
**Keywords**: Test output prediction, pseudocode execution, dual execution, code generation, functional majority voting

## TL;DR

This paper proposes DUET, a dual-path framework that combines direct code execution with LLM-based pseudocode execution. The two paths are complementary—the former is reliable when generated code is correct but vulnerable to implementation errors, while the latter bypasses implementation details at the cost of potential execution hallucinations. Predictions are merged via functional majority voting, achieving a 13.6 percentage-point improvement in Pass@1 on LiveCodeBench test output prediction.

## Background & Motivation

**Background**: Test case generation is a critical component of code generation pipelines. Test output prediction—given a problem description and test input, predict the correct output—is a challenging task requiring precise program reasoning. Methods such as TestChain address this by first generating code and then executing it directly.

**Limitations of Prior Work**: (1) A fundamental weakness of direct code execution is that a model may understand the correct algorithmic logic (and generate correct pseudocode) yet introduce subtle errors when implementing it as executable code (e.g., using `len(nums)` instead of `i+1` for cumulative averaging), causing execution failures or incorrect outputs. (2) In end-to-end code generation, using the execution results of generated test code to filter candidate programs suffers from the *zero-advantage problem*—if the generated test code is also incorrect, filtering becomes ineffective.

**Key Challenge**: Direct code execution depends on code correctness (deterministic but brittle), whereas LLM-based reasoning does not require correct code but may produce execution hallucinations (flexible but unreliable). The failure modes of the two approaches are complementary.

**Goal**: (1) Propose LLM pseudocode execution to decouple correct logic from implementation errors; (2) design the dual-path framework DUET to leverage the complementary strengths of both execution paths.

**Key Insight**: Decoupling *logical correctness* from *implementation correctness* in code generation—pseudocode captures algorithmic intent without being constrained by syntactic details, and LLM-simulated execution on pseudocode can bypass implementation errors.

**Core Idea**: The two paths for test output prediction—direct execution (reliable when code is correct) and pseudocode-simulated execution (bypasses implementation errors but prone to hallucination)—are naturally complementary. Functional majority voting can exploit the advantages of both.

## Method

### Overall Architecture

Given a problem description and test input: Path 1 generates executable code and obtains the output via direct execution; Path 2 generates pseudocode and obtains the output via LLM-simulated execution (step-by-step reasoning). Each path is sampled multiple times, and the final predicted output is selected through functional majority voting.

### Key Designs

1. **LLM Pseudocode Execution**:

    - *Function*: Predict test outputs without relying on the correctness of code implementation.
    - *Mechanism*: The LLM first generates a pseudocode description of the problem (high-level algorithmic intent), then simulates execution on the pseudocode step by step—tracking variable states and deriving the final output. Pseudocode provides a more precise algorithmic description than natural language while being more fault-tolerant than executable code.
    - *Design Motivation*: Implementation errors (e.g., loop boundary issues, variable name confusion) are the primary cause of direct execution failures. Pseudocode bypasses these details through higher-level abstraction.

2. **Functional Majority Voting**:

    - *Function*: Combine the complementary strengths of both execution paths.
    - *Mechanism*: Both code execution and pseudocode execution are sampled $N$ times each (e.g., 5 times each), collecting $2N$ outputs in total. An output that appears in both paths (or receives a majority across all votes) is more likely to be correct. Functional majority voting judges equivalence by functional output equality rather than literal string matching.
    - *Design Motivation*: The two paths fail under different conditions—code execution fails due to implementation errors (e.g., Figure 2a), while pseudocode execution hallucinates on deeply nested loops (e.g., Figure 2b). Complementary fusion significantly improves reliability.

3. **Resolution of the Zero-Advantage Problem**:

    - *Function*: Address the failure of test filtering in end-to-end code generation.
    - *Mechanism*: In the CODET pipeline, when predicted test outputs derived from direct execution of generated code are used to filter candidate programs, filtering becomes ineffective if the test code itself is incorrect (zero-advantage). The pseudocode execution path is decoupled from code correctness; thus, even when candidate code is erroneous, the pseudocode path still provides a valid filtering signal.
    - *Design Motivation*: TestChain actually degrades performance in the CODET pipeline (−5.6 pp) precisely because of the zero-advantage problem. DUET avoids this issue through the pseudocode path.

### Loss & Training

No model training is involved. Inference is performed using existing LLMs (e.g., Llama-3.1-8B-Instruct). Code execution and pseudocode execution are each sampled 5 times (10 LLM calls in total), and the final output is selected via functional majority voting.

## Key Experimental Results

### Main Results

**LiveCodeBench Test Output Prediction**

| Method | Pass@1 | Relative Gain |
|--------|--------|---------------|
| Direct (code execution only) | Baseline | — |
| TestChain | +5.6 pp | — |
| Pseudocode Exec | Competitive | — |
| **DUET** | **+13.6 pp** | **SOTA** |

**End-to-End Code Generation (CODET Pipeline + Llama-3.1-8B-Instruct)**

| Prediction Method | Pass@1 Change |
|-------------------|---------------|
| No filtering | Baseline |
| TestChain | −5.6 pp (zero-advantage problem) |
| **DUET** | **+3.2 pp** |

### Ablation Study

**End-to-End Code Generation on LiveCodeBench-Easy / BigCodeBench-Hard / DevEval / HumanEval(+)**

| Method | LCB-Easy | BCB-Hard | DevEval | HumanEval(+) |
|--------|----------|----------|---------|---------------|
| DUET | **Best** | **Best** | **Best** | **Best** |

### Key Findings

- DUET achieves +13.6 pp on test output prediction, substantially outperforming single-path methods—fully validating the complementarity of the two paths.
- TestChain degrades performance by 5.6 pp in the CODET pipeline, whereas DUET improves it by 3.2 pp—the zero-advantage problem is a critical barrier to real-world deployment.
- The distributions of implementation errors and execution hallucinations are complementary: problems with complex functional logic are more prone to implementation errors (favoring the pseudocode path), while deeply nested loops are more prone to execution hallucinations (favoring direct code execution).
- The abstraction level of pseudocode makes it inherently more fault-tolerant than executable code—even given identical logic, pseudocode "implementation" is less error-prone.

## Highlights & Insights

- Decoupling "correct logic" from "correct implementation" is an elegant problem formulation—pseudocode execution is the natural product of this decoupling.
- The identification and resolution of the zero-advantage problem have significant practical implications for deploying code generation pipelines.
- Functional majority voting is a general-purpose fusion strategy that can be extended to additional execution paths.

## Limitations & Future Work

- Pseudocode execution requires additional LLM calls ($2N$ vs. $N$), doubling computational cost.
- On extremely complex algorithmic problems, both pseudocode and code execution may simultaneously fail.
- The quality of pseudocode generation depends on the LLM's meta-linguistic capabilities.
- Specialized training for pseudocode execution to reduce execution hallucinations remains unexplored.

## Related Work & Insights

- **vs. TestChain**: TestChain relies solely on direct code execution and is susceptible to the zero-advantage problem; DUET's dual-path design resolves this issue.
- **vs. AlphaCode**: AlphaCode focuses only on test input generation, whereas DUET also addresses output prediction.
- **vs. CODET**: The CODET framework depends on accurate test output filtering; DUET provides a more reliable source of predicted test outputs.

## Rating

- Novelty: ⭐⭐⭐⭐ The ideas of pseudocode execution and dual-path fusion are novel, and the identification of the zero-advantage problem is a valuable contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation covering test output prediction and end-to-end assessment across multiple benchmarks.
- Writing Quality: ⭐⭐⭐⭐⭐ Problem analysis is thorough and clear; illustrative figures are intuitive and accessible.
- Value: ⭐⭐⭐⭐ Directly applicable to test prediction in code generation pipelines with clear practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Toward Consistent World Models with Multi-Token Prediction and Latent Semantic Enhancement](toward_consistent_world_models_with_multi-token_prediction_and_latent_semantic_e.md)
- [\[ACL 2026\] XMark: Reliable Multi-Bit Watermarking for LLM-Generated Texts](xmark_reliable_multi-bit_watermarking_for_llm-generated_texts.md)
- [\[ACL 2026\] Who Gets Which Message? Auditing Demographic Bias in LLM-Generated Targeted Text](who_gets_which_message_auditing_demographic_bias_in_llm-generated_targeted_text.md)
- [\[NeurIPS 2025\] Buffer Layers for Test-Time Adaptation](../../NeurIPS2025/llm_safety/buffer_layers_for_test-time_adaptation.md)
- [\[ICLR 2026\] Inoculation Prompting: Eliciting Traits from LLMs during Training Can Suppress Them at Test-Time](../../ICLR2026/llm_safety/inoculation_prompting_eliciting_traits_from_llms_during_training_can_suppress_th.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] On The Fragility of Benchmark Contamination Detection in Reasoning Models
description: >-
  [ICLR 2026][LLM Reasoning][Benchmark Contamination] This systematic study reveals that benchmark contamination detection in large reasoning models (LRMs) is extremely fragile: contamination introduced during the SFT stag…
tags:
  - "ICLR 2026"
  - "LLM Reasoning"
  - "Benchmark Contamination"
  - "Reasoning Models"
  - "GRPO"
  - "Detection Fragility"
  - "Evaluation Integrity"
date: 2026-05-08
content_hash: d444b2848935edf4
---

# On The Fragility of Benchmark Contamination Detection in Reasoning Models

**Conference**: ICLR 2026
**arXiv**: [2510.02386](https://arxiv.org/abs/2510.02386)  
**Code**: [https://github.com/ASTRAL-Group/LRM_Conta_Detection_Arena.git](https://github.com/ASTRAL-Group/LRM_Conta_Detection_Arena.git)  
**Area**: LLM Reasoning
**Keywords**: Benchmark Contamination, Reasoning Models, GRPO, Detection Fragility, Evaluation Integrity

## TL;DR
This systematic study reveals that benchmark contamination detection in large reasoning models (LRMs) is extremely fragile: contamination introduced during the SFT stage becomes nearly undetectable after GRPO training (with PPO-style importance sampling and clipping identified as the root cause), and direct CoT SFT contamination of advanced LRMs leaves virtually no detectable trace—all 10 evaluated detection methods perform close to random guessing.

## Background & Motivation

**Background**: LLM leaderboards have become highly competitive, giving model developers an incentive to include evaluation benchmarks in training data to inflate scores. Numerous contamination detection methods have been proposed, including generation-based, perturbation-based, and reference-model-based approaches.

**Limitations of Prior Work**:
- Existing detection methods assume contamination equals memorization (i.e., higher likelihood on seen samples), but LRMs reach answers via CoT reasoning, and detectors typically cannot access the CoT data used during training.
- LRMs acquire reasoning capabilities through a two-stage pipeline (SFT → RL), allowing developers to contaminate in the early stage (SFT) and "launder" evidence through later-stage (RL) training.
- The detectability of direct CoT SFT contamination applied to advanced LRMs is entirely unknown.

**Key Challenge**: Evaluation fairness depends on contamination being detectable—but if RL training can inherently erase contamination evidence, and CoT SFT leaves almost no trace, the integrity of the entire leaderboard ecosystem is at risk.

**Key Insight**: Two realistic scenarios—Stage I: SFT contamination → RL "laundering" (base model → LRM); Stage II: direct CoT SFT contamination of an existing LRM (post-LRM).

**Core Idea**: The importance sampling and clipping objective in GRPO/PPO systematically eliminates the separability between members and non-members, making RL training a natural "sanitizer" of contamination evidence.

## Method

### Overall Architecture
A two-stage analysis: **Stage I** simulates the process of a base model undergoing contaminated SFT followed by GRPO training to become an LRM, evaluating the AUROC of 10 detection methods before and after RL. **Stage II** simulates direct CoT SFT contamination of an advanced LRM and evaluates detection effectiveness.

### Key Findings

1. **GRPO Conceals SFT Contamination (Stage I)**:
    - SFT contamination is initially detectable: Min-K%/Max-K%/LOSS methods achieve AUROC ~73%, and LiRA reaches ~89%.
    - After GRPO training, AUROC broadly drops to ~50% (random level)—even when GRPO uses only clean data.
    - Control experiment: continuing SFT on clean data does not eliminate the detection signal, confirming that the GRPO optimization objective—not simply "more data"—is responsible.

2. **PPO-Style Clipping Is the Root Cause**:
    - Theoretical analysis: the importance sampling ratio $r(\theta) = \pi_\theta / \pi_{\theta_{old}}$ combined with the clipping gate imposes an **upper bound** on gradients for high-probability member samples (preventing over-optimization), while no such constraint applies to non-members—causing non-member probabilities to rise relatively and the member/non-member distributions to overlap.
    - Ablation: pure rejection sampling (RAFT) does not erase the detection signal; adding importance sampling and clipping (RAFT++) immediately restores the concealment effect, precisely pinpointing the root cause.

3. **CoT SFT on LRMs Is Nearly Undetectable (Stage II)**:
    - Applying CoT SFT contamination to advanced LRMs yields substantial gains in pass@1, yet all 10 detection methods yield AUROC close to 50%.
    - Reason: LRMs already assign high confidence to in-distribution samples not seen during training, breaking the core detection assumption that "contamination equals memorization."
    - Log-prob analysis reveals highly overlapping probability distributions for members and non-members.

### Loss & Training
- Contamination simulation: CoT data is distilled from an advanced LRM for SFT; half the samples are randomly selected as the member set.
- GRPO training: standard configuration using clean data (excluding members).
- Detection evaluation: 8 responses per question, averaged detection scores, AUROC used to distinguish members from non-members.

## Key Experimental Results

### Stage I: AUROC Before and After GRPO (Qwen2.5-7B-Instruct)

| Detection Method | AUROC after SFT ↑ | AUROC after SFT+GRPO ↑ | Change |
|---------|-------------|-------------------|------|
| Min-K% | ~73% | ~52% | -21% |
| Max-K% | ~72% | ~51% | -21% |
| LOSS | ~75% | ~53% | -22% |
| LiRA (reference-based) | ~89% | ~55% | -34% |
| Generation-based methods | ~60% | ~50% | -10% |

### Stage II: Detection of LRM CoT SFT Contamination

| Detection Method | AUROC ↑ |
|---------|--------|
| All 10 methods | **~50%** (≈ random guessing) |
| pass@1 gain from contamination | **+8.82%** on average |

### Ablation: Pure RL vs. RL with Clipping

| RL Variant | Contamination Detection AUROC |
|---------|-------------|
| RAFT (pure rejection sampling) | ~70% (still detectable) |
| RAFT++ (+importance sampling clipping) | ~52% (concealed) |
| GRPO | ~52% (concealed) |

### Key Findings
- **SFT contamination + GRPO laundering = perfect crime**: developers can inject benchmark data during SFT to gain +8.82% improvement, then train with clean data via GRPO to erase all detection traces.
- **PPO clipping is a mechanism-level root cause**: it is the mathematical structure of the optimization objective—not additional training data—that causes concealment; RAFT does not conceal, but RAFT++ does.
- **CoT SFT on advanced LRMs is nearly immune to detection**: even without RL laundering, CoT SFT itself leaves no detectable trace.
- **The "contamination = memorization" assumption does not hold**: LRMs assign high confidence to unseen in-distribution samples, rendering all probability-difference-based detection methods ineffective.
- **Consistent across models**: both Qwen2.5-7B and Llama-3.1-8B exhibit the same fragility.

## Highlights & Insights
- **First algorithmic-level explanation of the contamination concealment mechanism**: the cause is neither data processing nor training duration, but the PPO/GRPO clipping objective itself, which converges to a distribution where members and non-members are indistinguishable. This is an entirely novel perspective.
- **A serious warning for the LRM leaderboard ecosystem**: if developers can trivially contaminate models while evading all detection, benchmark-based evaluation requires fundamental reform. The paper's proposed directions toward trustworthy evaluation protocols deserve close attention.
- **The RAFT vs. RAFT++ ablation is a textbook causal inference experiment**—minimal modifications are used to precisely isolate the root cause.

## Limitations & Future Work
- Only 7B/8B-scale models are evaluated; larger models (70B+) may exhibit different behavior.
- Contamination samples use distilled CoT from advanced LRMs; simpler CoT (e.g., human-written) may yield different results.
- Novel detection methods based on model behavior rather than probabilities—such as analyzing structural features of reasoning paths—remain unexplored.
- The theoretical analysis relies on simplifying assumptions; actual GRPO dynamics are more complex.
- The feasibility of countermeasures is not discussed; whether detection methods immune to PPO clipping can be designed remains an open question.

## Related Work & Insights
- **vs. traditional contamination detection methods (Shi, Mattern, Dong, et al.)**: these methods are effective on standard LLMs but fail entirely on LRMs.
- **vs. Dekoninck/Samuel (data augmentation evasion)**: they evade detection by paraphrasing data; this paper finds that RL training itself serves as a natural "evader"—more dangerous because no additional manipulation is required.
- **vs. Bordt (training dynamics perspective)**: they study the natural decay of contamination effects during pretraining; this paper finds that RL fine-tuning actively accelerates such decay.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First paper to reveal the algorithmic mechanism by which RL training conceals contamination; the problem is critically important.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 10 detection methods × 6 benchmarks × 2 models × ablation and theoretical analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Two-stage analysis framework is clear; the RAFT vs. RAFT++ ablation design is elegant.
- Value: ⭐⭐⭐⭐⭐ Poses an existential threat to the LRM evaluation ecosystem; the entire community should take notice.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] FloorplanQA: A Benchmark for Spatial Reasoning in LLMs Using Structured Representations](../../ICML2026/llm_reasoning/floorplanqa_a_benchmark_for_spatial_reasoning_in_llms_using_structured_represent.md)
- [\[ACL 2026\] MTR-Bench: A Comprehensive Benchmark for Multi-Turn Reasoning Evaluation](../../ACL2026/llm_reasoning/mtr-bench_a_comprehensive_benchmark_for_multi-turn_reasoning_evaluation.md)
- [\[NeurIPS 2025\] RealMath: A Continuous Benchmark for Evaluating Language Models on Research-Level Mathematics](../../NeurIPS2025/llm_reasoning/realmath_a_continuous_benchmark_for_evaluating_language_models_on_research-level.md)
- [\[ICML 2026\] CoCoReviewBench: A Completeness- and Correctness-Oriented Benchmark for AI Reviewers](../../ICML2026/llm_reasoning/cocoreviewbench_a_completeness-_and_correctness-oriented_benchmark_for_ai_review.md)
- [\[ICML 2026\] ToolMATH: A Math Tool Benchmark for Realistic Long-Horizon Multi-Tool Reasoning](../../ICML2026/llm_reasoning/toolmath_a_math_tool_benchmark_for_realistic_long-horizon_multi-tool_reasoning.md)

</div>

<!-- RELATED:END -->

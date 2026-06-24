---
title: >-
  [Paper Note] How to Train Long-Context Language Models (Effectively)
description: >-
  [ACL 2025][LLM Efficiency][Long-context training] This paper systematically studies how to effectively train long-context language models through continual pre-training and SFT. It proposes a series of key designs, including data mixing ratios, training length scaling, and evaluation protocols. The resulting ProLong-8B achieves the same-scale SOTA on 128K length using only **5%** of Llama-3.1's long-context training data.
tags:
  - "ACL 2025"
  - "LLM Efficiency"
  - "Long-context training"
  - "Continual pre-training"
  - "Data mixing"
  - "Supervised fine-tuning"
  - "Positional extrapolation"
date: 2026-05-08
content_hash: a437ca91d059f42c
---

# How to Train Long-Context Language Models (Effectively)

**Conference**: ACL 2025  
**arXiv**: [2410.02660](https://arxiv.org/abs/2410.02660)  
**Area**: LLM Efficiency  
**Keywords**: Long-context training, Continual pre-training, Data mixing, Supervised fine-tuning, Positional extrapolation

## TL;DR

This paper systematically studies how to effectively train long-context language models through continual pre-training and SFT. It proposes a series of key designs, including data mixing ratios, training length scaling, and evaluation protocols. The resulting ProLong-8B achieves the same-scale SOTA on 128K length using only **5%** of Llama-3.1's long-context training data.

## Background & Motivation

- **Urgent application needs**: Long-context LMs have unlocked new applications such as book summarization, many-shot ICL, and long-document RAG. However, adapting pre-trained models to 128K+ contexts poses dual challenges in both infrastructure and data.
- **Training-free extrapolation is unreliable**: Training-free methods (e.g., YaRN, Dynamic NTK) that modify the RoPE frequency base fail to reliably pass simple Needle-in-a-Haystack (NIAH) tasks, still requiring further training on billions of long-document tokens.
- **Opaque design decisions**: Frontier open-source models (e.g., Llama-3.1) adopt the "long-context continual pre-training $\rightarrow$ SFT" paradigm, but key decisions such as data mixing ratios, sequence length selection, and SFT data types are not fully transparent to the community.
- **Flawed evaluation methods**: Existing evaluation approaches (perplexity, NIAH) are unreliable. NIAH is already saturated for strong models (both Llama-3.1-8B and 70B score 100), while perplexity correlates poorly with downstream performance (training with 100% long data continuously improves PPL but severely impairs downstream execution).

## Method

### Overall Architecture

A three-stage approach is adopted: **(1)** Establish a reliable evaluation protocol based on HELMET to guide model development; **(2)** Conduct two-stage long-context continual pre-training (64K $\rightarrow$ 512K) starting from Llama-3-8B-Instruct; **(3)** Perform SFT on the short instruction dataset UltraChat. This yields ProLong-8B, which supports a 512K token context window with only 40B total training tokens.

### Key Designs

**Design 1: Reliable Evaluation Protocol — Post-SFT Evaluation + Multi-Task Benchmark**

The HELMET evaluation suite is adopted to cover six categories of downstream tasks (Recall, RAG, Re-ranking, ICL, QA, Summarization) instead of relying on perplexity or NIAH. Core finding: evaluation must be conducted **after** SFT—some improvements in long-context capabilities (e.g., RAG, Re-ranking) only become apparent post-SFT. Concurrently, short-context benchmarks (HellaSwag, MMLU, ARC-c, WinoGrande, GSM8K) are tracked to ensure no performance degradation. The evaluation protocol comparison is shown below:

| Evaluation Method | NIAH | Recall | RAG | Re-rank | Distinguish Strong/Weak Models? |
|---------|------|--------|-----|---------|---------------|
| NIAH only | 100 | - | - | - | ✗ (Strong models saturated) |
| PPL only | ↓ | - | - | - | ✗ (Inconsistent with downstream) |
| **HELMET (Post-SFT)** | 100 | 99.4 | 56.3 | 37.0 | **✓ (Multi-dimensional distinction)** |

**Design 2: Data Mixing Strategy — 60% Long + 40% High-quality Short Mixture**

Ablations on long data sources reveal that code repositories (concatenating all files from the same repo into a single document, 98.8B long tokens) and books (33.2B long tokens) are the optimal long data sources, with a 1:1 mixture performing best. Ablations on short-to-long ratios show that 60% long data + 40% short data is the optimal ratio; 100% long data severely harms downstream long-context tasks. The short data mixture design, ProLong ShortMix, preserves mathematical reasoning capabilities:

| Short Data Component | Ratio | Function |
|-----------|------|------|
| FineWeb | 25% | General web text |
| FineWeb-Edu | 25% | Educational web text |
| Wikipedia | 10% | Encyclopedic knowledge |
| Tulu-v2 | 10% | Instruction data |
| StackExchange | 10% | Technical Q&A |
| ArXiv | 10% | Academic papers |
| OpenWebMath | 10% | Mathematical reasoning preservation |

**Design 3: Two-Stage Training Length Scaling + Purely Short SFT**

Training employs a curriculum learning strategy: Stage 1 trains on 20B tokens at 64K length (RoPE base = 8×10⁶, 2.2K H100 hours), and Stage 2 trains on 20B tokens at 512K length (RoPE base = 1.28×10⁸, 12.2K H100 hours). Key finding: training beyond the evaluation length significantly improves performance (512K training vs. 64K training on 64K evaluation: Re-rank 32.9 vs. 28.0). In the SFT stage, using only short-context instruction data UltraChat (averaging 1.2K tokens) yields the strongest long-context performance; adding synthetic long SFT data (even just 1%) decreases performance instead. Other designs: disabling cross-document attention (improves performance + training throughput), initializing from Instruct instead of Base (significantly preserves short-context capability).

## Key Experimental Results

### Main Results: HELMET 128K Evaluation

| Model | Params | Max Len | Recall | RAG | ICL | Re-rank | QA | Summ. | Avg. |
|------|--------|---------|--------|-----|-----|---------|-----|-------|------|
| **ProLong** | 8B | 512K | **98.8** | **63.2** | **86.5** | **22.5** | 43.9 | 29.2 | **49.4** |
| Llama-3.1 | 8B | 128K | 95.2 | 59.5 | 83.9 | 14.0 | 43.2 | 27.0 | 46.5 |
| MegaBeam-Mistral | 7B | 512K | 89.6 | 57.0 | 86.2 | 14.7 | 37.3 | 28.9 | 45.4 |
| Llama-3.1 | 70B | 128K | 90.7 | 56.2 | 81.4 | 24.5 | 56.3 | 31.6 | 49.7 |

ProLong-8B surpasses Llama-3.1-8B-Instruct using only 40B tokens (5% of Llama-3.1's long-context training budget), leading in all categories except Summarization, and even approaching the 70B Llama-3.1 in Avg. score.

### Ablation Study

**Comparison of Long Data Sources** (60% Long + 40% ShortMix, trained on 5B tokens):

| Long Data Source | Recall | RAG | Re-rank | ICL | QA | Summ. | Avg. |
|---------|--------|-----|---------|-----|-----|-------|------|
| CommonCrawl | 84.1 | 53.3 | 28.1 | 67.5 | 35.2 | 37.0 | 50.9 |
| Books | 94.9 | 53.9 | 30.7 | 72.2 | 33.2 | 37.7 | 53.8 |
| Code Repos | 99.2 | 53.8 | 29.0 | 61.2 | 34.7 | 36.2 | 52.3 |
| **Books/Repos 1:1** | **96.0** | **54.9** | 29.4 | **73.9** | **35.7** | **37.9** | **54.6** |

**Comparison of Short Data Sources**:

| Short Data Source (40%) | Long-Ctx Avg. | HellaSwag | MMLU | GSM8K | Short Avg. |
|----------------|---------------|-----------|------|-------|------------|
| SlimPajama | 52.9 | 81.2 | 63.0 | 41.9 | 64.2 |
| FineWeb-Edu | 53.0 | 81.0 | 62.6 | 39.4 | 63.0 |
| DCLM-Baseline | 52.0 | 82.0 | 65.6 | 39.4 | 64.8 |
| **ProLong ShortMix** | **54.6** | **81.6** | **65.3** | **46.6** | **65.5** |

**Impact of Synthetic SFT Data Ratio**:

| Synthetic Data Ratio | RAG | Re-rank | ICL | QA | Summ. | Avg. |
|-------------|-----|---------|-----|-----|-------|------|
| **0% (Pure UltraChat)** | **58.1** | **38.5** | 80.3 | **49.7** | **42.1** | **55.7** |
| 1% | 57.0 | 38.3 | **80.8** | 45.3 | 41.5 | 54.1 |
| 10% | 55.5 | 36.1 | 80.6 | 41.7 | 39.4 | 53.9 |
| 50% | 48.8 | 18.8 | 70.5 | 42.3 | 33.3 | 43.3 |

### Key Findings

1. **Pure long-context training is harmful**: Although 100% long data continuously improves PPL, it severely degrades downstream long-context task performance (post-SFT RAG and Re-ranking drop significantly).
2. **Extremely long training lengths are beneficial**: 512K training vs. 64K training evaluates on 64K with Recall 98.5 vs. 95.0 and Re-rank 32.9 vs. 28.0.
3. **Short SFT data is sufficient**: 0% synthetic long SFT data yields an Avg. of 55.7, dropping sharply to 43.3 after including 50% synthetic data.
4. **Short-context performance preservation**: ProLong ShortMix achieves a short-context average of 65.5, close to the original Llama-3-8B's 66.0.

## Highlights & Limitations

**Highlights**:

- Challenging the intuition of using purely full-length data for long-context training, systematically demonstrating for the first time the criticality of mixing high-quality short data.
- Demonstrating for the first time the benefit of training sequence lengths exceeding the evaluation length, accompanied by a theoretical explanation based on dependency distance.
- The finding that SFT requires only short instruction data greatly simplifies the training workflow for long-context models.
- Contribution to evaluation methodology: revealing the unreliability of PPL and NIAH, urging the community to adopt multi-task evaluations like HELMET.
- Exceptional data efficiency: surpassing Llama-3.1-8B with only 40B training tokens (a 5% data budget).

**Limitations**:

- Experiments are restricted to Llama-3-8B (~8B parameters); whether the conclusions hold at larger scales remains unverified.
- The impact of RLHF/preference optimization on long-context SFT is unexplored.
- The ineffectiveness of synthetic long SFT data may relate to generator quality, requiring verification with stronger models.
- The computational cost of 512K training is significant (12.2K vs. 2.2K H100 hours), and the compute-optimal strategy warrants further exploration.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Systematic ablation experiments yielding multiple counter-intuitive discoveries (e.g., pure long data is harmful, short SFT is superior).
- **Value**: ⭐⭐⭐⭐⭐ — Complete, reproducible training recipe (ProLong recipe) that is ready to use.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Extensive ablations covering evaluation protocols, data mixing, length scaling, and SFT strategies.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Well-structured with masterfully designed Takeaway Boxes and high-density figures/tables.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] FastDraft: How to Train Your Draft](fastdraft_how_to_train_your_draft.md)
- [\[ACL 2025\] LongReward: Improving Long-context Large Language Models with AI Feedback](longreward_improving_long-context_large_language_models_with_ai_feedback.md)
- [\[ACL 2025\] Ref-Long: Benchmarking the Long-Context Referencing Capability of Long-Context Language Models](ref-long_benchmarking_the_long-context_referencing_capability_of_long-context_la.md)
- [\[ACL 2025\] Literary Evidence Retrieval via Long-Context Language Models](literary_evidence_retrieval_via_long-context_language_models.md)
- [\[ACL 2025\] LongSafety: Evaluating Long-Context Safety of Large Language Models](longsafety_evaluating_long-context_safety_of_large_language_models.md)

</div>

<!-- RELATED:END -->

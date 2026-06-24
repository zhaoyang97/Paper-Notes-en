---
title: >-
  [Paper Note] WirelessMathBench: A Mathematical Modeling Benchmark for LLMs in Wireless Communications
description: >-
  [ACL 2025][Signal & Communication][Wireless Communications] This paper presents WirelessMathBench, a mathematical modeling benchmark for wireless communications featuring 587 problems extracted from 40 cutting-edge papers. It systematically evaluates the capabilities of LLMs in domain-specific mathematical derivations, revealing that even the strongest model, DeepSeek-R1, achieves an average accuracy of only 38.05%, and a mere 7.83% in full formula derivation.
tags:
  - "ACL 2025"
  - "Signal & Communication"
  - "Wireless Communications"
  - "Mathematical Reasoning"
  - "LLM Evaluation Benchmark"
  - "Domain-Specific Reasoning"
  - "Formula Derivation"
date: 2026-05-08
content_hash: 304a5d3e5e54c636
---

# WirelessMathBench: A Mathematical Modeling Benchmark for LLMs in Wireless Communications

**Conference**: ACL 2025  
**arXiv**: [2505.14354](https://arxiv.org/abs/2505.14354)  
**Code**: [https://lixin.ai/WirelessMathBench](https://lixin.ai/WirelessMathBench)  
**Area**: Signal Communications  
**Keywords**: Wireless Communications, Mathematical Reasoning, LLM Evaluation Benchmark, Domain-Specific Reasoning, Formula Derivation

## TL;DR

This paper presents WirelessMathBench, a mathematical modeling benchmark for wireless communications featuring 587 problems extracted from 40 cutting-edge papers. It systematically evaluates the capabilities of LLMs in domain-specific mathematical derivations, revealing that even the strongest model, DeepSeek-R1, achieves an average accuracy of only 38.05%, and a mere 7.83% in full formula derivation.

## Background & Motivation

**Background**: LLMs have made significant progress in general mathematical reasoning (e.g., on benchmarks like GSM8K and MATH), with reasoning models such as OpenAI-o1 and DeepSeek-R1 further pushing the boundaries of multi-step reasoning capabilities. However, these advancements are primarily concentrated in general mathematics fields.

**Limitations of Prior Work**: Existing mathematical benchmarks (GSM8K, MATH, OlympiadBench, etc.) focus on pure mathematical problems ranging from primary/secondary school levels to Olympiad-level competitions, lacking evaluation of complex mathematical modeling capabilities in engineering domains (especially wireless communications). Wireless communications involve strict physical constraints, dimensional consistency, and domain-specific notation systems.

**Key Challenge**: LLMs can perform well on multiple-choice questions (>75%), but their capabilities drop sharply when required to reconstruct complete formula derivations, indicating a massive gap between "comprehension" and "derivation".

**Goal**: To build an expert-level benchmark specifically targeting mathematical modeling in wireless communications, comprehensively evaluating LLMs' symbolic reasoning and domain-knowledge application capabilities.

**Key Insight**: Sourcing mathematical models from real, cutting-edge research papers and designing multi-tiered tasks—ranging from multiple-choice questions to progressive cloze-style masking, and finally to full formula derivation—to provide a progressive difficulty evaluation.

**Core Idea**: Utilizing a progressive formula masking strategy to evaluate the mathematical derivation capabilities of LLMs in wireless communications, exposing the fundamental deficiencies of current models in domain-specific symbolic reasoning.

## Method

### Overall Architecture

WirelessMathBench is constructed around two design principles: (1) real-world complexity—problems are sourced directly from peer-reviewed papers; (2) multi-level progression—ranging from basic multiple-choice questions to complete derivations, covering various difficulty levels. The data collection pipeline includes: paper selection $\to$ system model extraction $\to$ task curation $\to$ domain-expert review.

### Key Designs

#### 1. Data Sourcing and Coverage

- **Function**: Sourcing mathematical models from 40 top-tier journal/conference papers
- **Mechanism**: Covering core model categories (RIS 19 papers, MIMO 12 papers, UAV 6 papers, ISAC 6 papers, Satellite 4 papers, SIM 3 papers, NOMA 2 papers) and problem categories (Beamforming 18 papers, Channel Estimation 12 papers, Performance Analysis 8 papers, etc.)
- **Design Motivation**: Ensuring that the evaluation covers authentic engineering challenges across mainstream wireless communication research directions

#### 2. Three-Tiered Task Design

- **Multiple-Choice Questions (MCQ)**: Selecting the correct mathematical expression from several closely related distractors to test the model's formula recognition and recall abilities.
- **Progressively Masked Cloze**: System model formulas are progressively masked across three levels—ranging from single-variable omissions to multi-variable masking, with each level serving as an independent sub-problem.
- **Full Equation Construction (FEC)**: The entire formula is fully hidden, and only the scenario description is provided, requiring the model to derive the complete expression from basic definitions.

#### 3. Data Quality Assurance

- **Function**: Employing multi-round expert reviews to ensure accuracy
- **Mechanism**: Semi-automatic extraction (initial extraction by LLMs + expert review and correction) + deliberate rewriting to prevent data contamination (reformulating paper contexts and reorganizing formula presentations)
- **Design Motivation**: Preventing LLMs from answering based on memorized training corpora instead of genuine reasoning

#### 4. Evaluation Pipeline

- **Function**: Unified prompt templates + two-stage evaluation
- **Mechanism**: MCQ answers are directly compared; progressively masked and FEC tasks utilize GPT-4o as an evaluator to assess symbolic equivalence
- **Design Motivation**: Expressions or polynomials may have multiple equivalent representation forms, necessitating semantic-level comparison

### Loss & Training

As this work presents an evaluation benchmark, it does not involve training. All experiments are conducted in a zero-shot setting using the default parameters of each model, without providing additional chain-of-thought prompts.

## Key Experimental Results

### Main Results

**Performance of 16 LLMs on WirelessMathBench**:

| Model | MCQ | Level 1 | Level 2 | Level 3 | FEC | Average |
|------|-----|---------|---------|---------|-----|------|
| DeepSeek-R1 | 76.00% | 60.00% | 34.91% | 12.50% | 7.83% | **38.05%** |
| OpenAI-o1 | 66.40% | 59.17% | 32.17% | 8.04% | 6.96% | 34.55% |
| GPT-4o | 72.80% | 42.50% | 28.70% | 6.25% | 4.35% | 30.92% |
| DeepSeek-V3 | **78.40%** | 50.00% | 24.35% | 6.25% | 6.96% | 33.19% |
| Gemini-1.5-pro | 65.60% | 43.33% | 29.57% | 9.82% | 6.09% | 30.88% |
| Qwen2.5-Math-72B | 70.40% | 37.50% | 26.09% | 7.14% | 6.09% | 29.44% |
| LLaMA-3.3-70B | 65.60% | 38.33% | 17.39% | 2.68% | 6.09% | 26.02% |
| GPT-3.5-turbo | 45.60% | 7.50% | 10.43% | 1.79% | 1.74% | 13.41% |
| LLaMA-3-8B-Tele | 40.80% | 11.67% | 4.35% | 2.68% | 0.87% | 12.07% |

### Ablation Study

**Analysis of 40 Error Cases from DeepSeek-R1**:

| Error Type | Proportion | Description |
|----------|------|------|
| Partial fill mismatch | 31% | Correctly filling one mask but incorrectly filling other associated masks |
| Symbolic misunderstanding | 29% | Choosing the wrong symbols or missing key symbolic elements (such as $\mathbf{H}_{BR}$ vs $\mathbf{H}_{BR}^H$) |
| Incorrect derivation path | 24% | Missing key intermediate steps or introducing irrelevant components, propagating early errors |
| Irrelevant system mixing | 11% | Introducing irrelevant system configurations (e.g., inserting NOMA interference factors in RIS-MIMO) |
| Others | 4% | Incomplete expressions or redundant placeholders |

### Key Findings

1. **Advantages of Reasoning Models**: DeepSeek-R1 (38.05%) and OpenAI-o1 (34.55%) significantly outperform other models, suggesting that explicit reasoning strategies are crucial for multi-step symbolic derivation.
2. **Strong in MCQ but Weak in Derivation**: DeepSeek-V3 achieves the highest MCQ score of 78.40%, but only obtains 6.25% in Level 3 and 6.96% in FEC, highlighting a huge gap between "comprehension" and "derivation".
3. **Progressive Degradation**: Performance drops sharply as the degree of masking increases—DeepSeek-R1 falls from 60.00% in Level 1 to 12.50% in Level 3.
4. **Limited Benefits of Domain-Specific Fine-tuning**: LLaMA-3-8B-Tele (telecom-fine-tuned version) underperforms relative to the base LLaMA-3-8B, as telecom fine-tuning data skews toward protocol knowledge rather than mathematical reasoning.
5. **Advantages of Math-Specific Models**: Qwen2.5-Math-72B (29.44%) demonstrates outstanding performance among models of comparable parameter scales.

## Highlights & Insights

1. **First Engineering-Grade Mathematical Evaluation Benchmark**: Distinguishing itself from pure mathematical problems, WirelessMathBench requires satisfying physical constraints and dimensional consistency, aligning more closely with real scientific research needs.
2. **Ingenious Progressive Masking Strategy**: The progressive design from MCQ to FEC allows researchers to precisely pinpoint where the model's capabilities break down.
3. **Revealing Fundamental Limitations of LLM-Assisted Research**: Even the strongest models achieve only around 8% accuracy in the FEC task, indicating a massive gap before they can potentially replace human engineers.
4. **Well-Controlled Data Contamination**: Experts deliberately rewrote the paper content to ensure that models cannot cheat based on memorization.

## Limitations & Future Work

1. Only text-based questions are covered, without incorporating multimodal data such as antenna diagrams or simulation plots.
2. While covering mainstream areas like MIMO and RIS, it lacks emerging topics such as quantum communications and Terahertz.
3. Automated evaluation checks only the final symbolic equivalence, which may overlook errors in intermediate reasoning steps.
4. All experiments are conducted in a zero-shot setting; the potential of fine-tuning or RAG methods has not yet been explored.
5. The scale of 587 problems is relatively limited and could be further expanded.

## Related Work & Insights

- **GSM8K / MATH / OlympiadBench**: General mathematical reasoning benchmarks; WirelessMathBench fills the void in evaluating mathematical reasoning within engineering domains.
- **TelecomGPT (Zou et al., 2024)**: Explores the application of LLMs in wireless communications but focuses primarily on knowledge retrieval rather than mathematical derivation.
- **Maatouk et al. (2023, 2024)**: Focuses on knowledge extraction by LLMs in the telecom domain; this work builds upon this to propose higher-level reasoning requirements.
- **Insights**: Domain-specific benchmarks are essential for understanding the true boundary of LLM capabilities, and similar benchmarks are needed in other engineering domains.

## Rating

| Dimension | Rating |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Masked Symbol Modeling for Demodulation of Oversampled Baseband Communication Signals](../../NeurIPS2025/signal_comm/masked_symbol_modeling_for_demodulation_of_oversampled_baseband_communication_si.md)
- [\[ICLR 2026\] Enhancing Instruction Following of LLMs via Activation Steering with Dynamic Rejection](../../ICLR2026/signal_comm/enhancing_instruction_following_of_llms_via_activation_steering_with_dynamic_rej.md)
- [\[ICML 2025\] Large Language Model (LLM)-enabled In-context Learning for Wireless Network Optimization](../../ICML2025/signal_comm/large_language_model_llm-enabled_in-context_learning_for_wireless_network_optimi.md)
- [\[ICLR 2026\] Mamba-3: Improved Sequence Modeling using State Space Principles](../../ICLR2026/signal_comm/mamba-3_improved_sequence_modeling_using_state_space_principles.md)
- [\[ICLR 2026\] TS-DDAE: A Novel Temporal-Spectral Denoising Diffusion AutoEncoder for Wireless Signal Recognition Model Pre-training](../../ICLR2026/signal_comm/ts-ddae_a_novel_temporal-spectral_denoising_diffusion_autoencoder_for_wireless_s.md)

</div>

<!-- RELATED:END -->

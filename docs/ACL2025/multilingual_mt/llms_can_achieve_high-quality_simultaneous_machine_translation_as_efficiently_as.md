---
title: >-
  [Paper Note] LLMs Can Achieve High-quality Simultaneous Machine Translation as Efficiently as Offline
description: >-
  [ACL 2025 (Findings)][Multilingual & Machine Translation][Simultaneous Machine Translation] This paper proposes a new paradigm that constructs SFT data by rearranging source and target language tokens into interleaved sequences based on latency requirements. This enables LLMs to perform high-quality Simultaneous Machine Translation (SiMT) as efficiently as offline translation, achieving SOTA performance on multiple benchmarks while preserving original offline translation capa…
tags:
  - "ACL 2025 (Findings)"
  - "Multilingual & Machine Translation"
  - "Simultaneous Machine Translation"
  - "Large Language Models"
  - "Streaming Translation"
  - "Interleaved Sequences"
  - "Latency Control"
date: 2026-05-08
content_hash: 03fb39cb0e1a6e65
---

# LLMs Can Achieve High-quality Simultaneous Machine Translation as Efficiently as Offline

**Conference**: ACL 2025 (Findings)  
**arXiv**: [2504.09570](https://arxiv.org/abs/2504.09570)  
**Code**: None  
**Area**: Text Generation / Machine Translation  
**Keywords**: Simultaneous Machine Translation, Large Language Models, Streaming Translation, Interleaved Sequences, Latency Control

## TL;DR

This paper proposes a new paradigm that constructs SFT data by rearranging source and target language tokens into interleaved sequences based on latency requirements. This enables LLMs to perform high-quality Simultaneous Machine Translation (SiMT) as efficiently as offline translation, achieving SOTA performance on multiple benchmarks while preserving original offline translation capabilities.

## Background & Motivation

**Background**: Large language models perform exceptionally well in offline machine translation, achieving high-quality translations using simple translation prompts (e.g., "Translate the following sentence from [src] into [tgt]:").

**Limitations of Prior Work**: In SiMT scenarios, the autoregressive nature of decoder-only LLMs severely restricts their efficiency and performance. Conventional methods are typically designed based on encoder-decoder architectures, featuring complex read/write policies that are difficult to transfer directly to LLMs. Existing LLM-based SiMT methods either require multiple forward passes to simulate read/write operations, resulting in extremely low efficiency, or sacrifice translation quality for low latency.

**Key Challenge**: There is a fundamental trade-off between quality and latency in SiMT. Traditional methods balance this trade-off using fixed policies (such as wait-k), but these policies lack flexibility and fail to fully exploit the powerful generative capabilities of LLMs. Additionally, the efficient autoregressive decoding mechanism of LLMs is interrupted in SiMT due to the constant switching between read and write operations.

**Goal**: (1) Enable LLMs to achieve efficiency in SiMT tasks comparable to that of offline translation; (2) generate high-quality translations under different latency requirements; (3) maintain the model's offline translation capability without degradation.

**Key Insight**: The authors observe that encoding the alternating read/write operations of SiMT into a unified sequence allows the LLM to generate efficiently in a standard autoregressive manner, eliminating the need to repeatedly pause and resume.

**Core Idea**: Use source-target interleaved sequences separated by special tokens to represent the SiMT process under various latencies, allowing the LLM to learn adaptive read/write policies through standard SFT while maintaining efficient decoding.

## Method

### Overall Architecture

The input consists of the source language sentence and the latency requirements, and the output is the simultaneous translation formatted as an interleaved sequence. During the training phase, interleaved sequences of various latency levels are constructed as SFT data. During the inference phase, the model receives source tokens in a streaming fashion and autoregressively generates interleaved translation tokens.

### Key Designs

1. **Interleaved Sequence Construction**:

    - **Function**: Encodes the read/write operations of SiMT into a single sequence, enabling the LLM to process it via standard autoregression.
    - **Mechanism**: Source and target tokens are interleaved chronologically based on alignment information, using special tokens (e.g., `<src>` and `<tgt>`) to separate segments of different languages. A latency prompt controls the granularity of the interleaving—low latency implies more frequent alternation, whereas high latency allows accumulating more source tokens before translating. The construction process refers to word alignment and wait-k policies to determine the minimum number of source tokens required for each target token.
    - **Design Motivation**: Solves the efficiency issue of LLMs having to repeatedly interrupt autoregressive decoding in SiMT, converting SiMT into a standard sequence generation task.

2. **Latency-aware Training**:

    - **Function**: Enables a single model to support SiMT at multiple latency levels.
    - **Mechanism**: By constructing multiple sets of interleaved sequence SFT data under various latency requirements, the model learns to adaptively adjust its read/write policy based on the latency prompt. During training, data from different latency levels are mixed, and the model recognizes the latency prompt to decide when to read new source tokens and when to generate target tokens. The model learns reasonable policies even with limited SFT data (compared to full translation datasets).
    - **Design Motivation**: Avoids training separate models for each latency level, achieving flexible latency control.

3. **Efficient Streaming Inference**:

    - **Function**: Accomplishes true streaming simultaneous translation during inference.
    - **Mechanism**: During inference, the model generates token-by-token in an autoregressive manner. When a `<src>` token is generated, it waits for new source tokens to arrive; when a `<tgt>` token is generated, it outputs the translation. The entire process requires no multiple forward passes or complex policy scheduling, achieving inference efficiency comparable to offline translation. Additionally, the model inherently supports document-level SiMT without extra fine-tuning.
    - **Design Motivation**: Exploits the natural efficiency of LLM autoregressive decoding, avoiding the complex scheduling logic of conventional SiMT systems.

### Loss & Training

Standard language modeling SFT loss (cross-entropy) is adopted for supervised fine-tuning on interleaved sequences. The training data mixes samples of various latency levels, ensuring the model learns latency-conditional generation capabilities.

## Key Experimental Results

### Main Results

| Dataset/Language Pair | Latency (AL) | Ours BLEU | Prev. SOTA BLEU | Gain |
|---------------|----------|-------------|---------------|------|
| WMT15 De→En | ~3 | SOTA-level | Traditional wait-k | Significant Gain |
| WMT15 De→En | ~5 | SOTA-level | Traditional SiMT | Significant Gain |
| WMT15 De→En | Offline | Comparable to original model | - | No degradation |
| Document-level SiMT | Multi-level | Outperforms offline | - | Strong generalization |

### Ablation Study

| Configuration | BLEU | Description |
|------|------|------|
| Full model (multi-latency mixed training) | Highest | Full model |
| Single-latency training | Decrease | Lack of flexibility |
| Without latency prompt | Significant decrease | Model cannot distinguish latency requirements |
| Less SFT data | Slight decrease | Indicates small data is sufficient |

### Key Findings

- Even with a small amount of SFT data, the method achieves SOTA performance, demonstrating the effectiveness of the interleaved sequence paradigm.
- The model demonstrates excellent zero-shot generalization on document-level SiMT, even outperforming dedicated offline translation models.
- Offline translation capabilities are preserved, showing no degradation due to SiMT fine-tuning.
- Latency prompt is a critical design; removing it prevents the model from effectively controlling the timing of translation.

## Highlights & Insights

- The **interleaved sequence paradigm** elegantly transforms the complex read/write policy problem of SiMT into a standard sequence generation problem, cleverly reusing the autoregressive decoding mechanism of LLMs instead of fighting against it. This idea can be extended to other streaming tasks that require alternating inputs and outputs.
- The **latency prompt control** mechanism allows a single model to support flexible latency requirements; this conditional generation approach is applicable to any generative task requiring a trade-off between quality and efficiency.
- **Document-level zero-shot generalization** indicates that the method learns generic read/write policies rather than merely memorizing sentence-level patterns.

## Limitations & Future Work

- The paper mainly conducts experiments on resource-rich language pairs such as De↔En, leaving generalization to low-resource language pairs yet to be validated.
- It relies on word alignment information to construct interleaved sequences, meaning alignment quality may impact the final performance.
- Streaming capabilities on extra-long documents warrant further evaluation, as the document lengths in current experiments are relatively limited.
- The impact of different base LLMs on this method has not been fully explored.

## Related Work & Insights

- **vs. Wait-k Policy**: Wait-k employs a fixed waiting policy and lacks flexibility. In contrast, this work achieves an adaptive policy through latency prompts, significantly outperforming fixed policies.
- **vs. Traditional SiMT Systems**: Traditional methods design complex policy networks based on encoder-decoder architectures. This work directly leverages the language modeling capabilities of LLMs, being simpler yet more powerful.
- **vs. Other LLM-based SiMT**: Prior works require multiple forward passes, resulting in extreme inefficiency. This work achieves efficiency comparable to offline translation through interleaved sequences.

## Rating

- Novelty: ⭐⭐⭐⭐ Interleaved sequences present a simple and effective paradigm, elegantly converting SiMT into a standard LM task.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple language pairs and latency levels, though the unavailability of HTML limits a comprehensive evaluation of experimental details.
- Writing Quality: ⭐⭐⭐⭐ The abstract and method description are clear, with precise problem definitions.
- Value: ⭐⭐⭐⭐ Provides a practical solution for LLM-based SiMT, advancing the practical application of simultaneous translation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] SeqPO-SiMT: Sequential Policy Optimization for Simultaneous Machine Translation](seqpo-simt_sequential_policy_optimization_for_simultaneous_machine_translation.md)
- [\[ACL 2025\] Watching the Watchers: Exposing Gender Disparities in Machine Translation Quality Estimation](watching_the_watchers_exposing_gender_disparities_in_machine_translation_quality.md)
- [\[ACL 2025\] Alleviating Distribution Shift in Synthetic Data for Machine Translation Quality Estimation](alleviating_distribution_shift_in_synthetic_data_for_machine_translation_quality.md)
- [\[ACL 2025\] AskQE: Question Answering as Automatic Evaluation for Machine Translation](askqe_question_answering_as_automatic_evaluation_for_machine_translation.md)
- [\[ACL 2026\] LQM: Linguistically Motivated Multidimensional Quality Metrics for Machine Translation](../../ACL2026/multilingual_mt/lqm_linguistically_motivated_multidimensional_quality_metrics_for_machine_transl.md)

</div>

<!-- RELATED:END -->

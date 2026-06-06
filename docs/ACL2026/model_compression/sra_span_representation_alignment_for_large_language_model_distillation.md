---
title: >-
  [Paper Note] SRA: Span Representation Alignment for Large Language Model Distillation
description: >-
  [ACL 2026][Model Compression][Cross-tokenizer distillation] SRA replaces fragile token-based alignment for cross-tokenizer LLM distillation with tokenizer-agnostic text spans. By utilizing LCS character offset matching…
tags:
  - "ACL 2026"
  - "Model Compression"
  - "Cross-tokenizer distillation"
  - "span alignment"
  - "center of mass"
  - "geometric regularization"
  - "LLM compression"
date: 2026-05-08
content_hash: 67789822d464e53b
---

# SRA: Span Representation Alignment for Large Language Model Distillation

**Conference**: ACL 2026  
**arXiv**: [2605.01205](https://arxiv.org/abs/2605.01205)  
**Code**: No public repository provided  
**Area**: Model Compression / Knowledge Distillation / Cross-Tokenizer Distillation  
**Keywords**: Cross-tokenizer distillation, span alignment, center of mass, geometric regularization, LLM compression

## TL;DR
SRA replaces fragile token-based alignment for cross-tokenizer LLM distillation with tokenizer-agnostic text spans. By utilizing LCS character offset matching, attention-weighted center-of-mass representations, geometric structure regularization, and shared vocabulary span logit distillation, it consistently outperforms ULD, MinED, DSKD, and MultiLevelOT across multiple teacher-student compression experiments.

## Background & Motivation
**Background**: Knowledge distillation (KD) is a common compression technique for transferring capabilities from large language models to smaller ones. Traditional KD typically assumes that the teacher and student share the same tokenizer, allowing for direct alignment of tokens or logit distributions. However, in real-world deployments, different model families often utilize distinct vocabularies and segmentation rules.

**Limitations of Prior Work**: Cross-Tokenizer Knowledge Distillation requires alignment across different tokenizers. Existing methods either use edit distance, dynamic programming, or Optimal Transport (OT) to process token sequences, or map different vocabularies into a unified space. However, token-level alignment is susceptible to differences in segmentation granularity: a single text segment may be one token in the teacher but split into multiple tokens in the student.

**Key Challenge**: Distillation aims to transfer semantics and representation dynamics, but tokenizer mismatch causes token sequences to no longer serve as stable, one-to-one units. Direct token alignment risks misinterpreting "segmentation differences" as "knowledge differences."

**Goal**: This work aims to construct a distillation unit that remains stable across tokenizers, enabling the teacher and student to align hidden states, geometric structures, and prediction distributions on identical text spans.

**Key Insight**: The paper adopts the physical perspective of Transformers as a Multi-Particle Dynamical System: token hidden states are viewed as particle positions, spans as particle clusters, and span representations as attention-weighted centers of mass.

**Core Idea**: First, identify text spans covered by both the teacher and student using character offsets, then aggregate these into span representations via attention weighting, and finally perform hidden-state, geometric structure, and logit distillation at the span level.

## Method
The design of SRA can be described as "identifying common semantic units first, then transferring representation dynamics." It avoids forced matching between token sequences from different tokenizers by reverting to the original string and using character offsets to find spans interpretable by both sides. Furthermore, SRA requires the student to not only approach the teacher's span representations but also maintain the relative geometric relationships between spans.

### Overall Architecture
Given the same text, the teacher and student tokenizers output token sequences and character offsets, respectively. SRA constructs aligned spans using the Longest Common Subsequence (LCS) of the offset sequences. For each span, a span representation is derived from the final hidden layer through attention-weighted pooling. During training, the student simultaneously optimizes standard CE, span hidden-state loss, geometric structure regularization, and span-level logit KD loss.

### Key Designs
1. **LCS-based span mapping**:

    - **Function**: Establishes comparable text segment units across different tokenizers.
    - **Mechanism**: Calculates the LCS of the teacher's and student's token offset sequences to match identical character boundaries, forming span pairs while ignoring special tokens with zero offsets. This results in spans that cover common sub-segments of the original text without requiring identical token counts.
    - **Design Motivation**: Token-level alignment is fragile in cross-tokenizer scenarios; character spans serve as stable units at the raw text level, making them more suitable carriers for knowledge transfer.

2. **Attention-weighted Center-of-Mass span representation**:

    - **Function**: Aggregates the hidden states of multiple tokens within a span into a single semantic representation.
    - **Mechanism**: SRA utilizes the attention from the last token to other tokens as an indicator of token importance. Following normalization, it computes the weighted average of the span. Formally, a span representation is defined as $C_i=\sum_{t=s_i}^{e_i} w_t H_t$, where $w_t$ is derived from the aggregated multi-head attention of the final layer.
    - **Design Motivation**: Simple mean pooling dilutes critical information. The Center-of-Mass (CoM) analogy posits that "particles with greater mass" exert a stronger influence on the overall center, corresponding to the idea that more attended tokens contribute more to the span representation.

3. **span-level hidden/logit distillation and geometric regularization**:

    - **Function**: Directs the student to learn local span representations, relative structures between spans, and shared vocabulary prediction distributions from the teacher.
    - **Mechanism**: The hidden-state loss uses weighted cosine similarity to align teacher and student span representations, supplemented by a geometric regularization $L_{Geo}$ to preserve the cosine distances between spans. The logit loss projects teacher and student span logits into the shared vocabulary subspace $V_T\cap V_S$ for KL distillation.
    - **Design Motivation**: Merely aligning point positions can be distorted by linear projections; geometric regularization preserves the structure of the representation space. Since hidden-state alignment alone may lack lexical prediction knowledge, the shared vocabulary logit loss provides complementary supervision.

### Loss & Training
The overall objective is $L_{overall}=\alpha L_{CE}+(1-\alpha)(L_{HS}^{Span}+L_{KD}^{Span})$. Here, $L_{HS}^{Span}$ comprises the weighted cosine loss and geometric structure regularization, while $L_{KD}^{Span}$ aligns span logits in the shared vocabulary space. Training is conducted using Databricks-Dolly-15k. Evaluation spans Dolly, VicunaEval, SelfInst, S-NI, and DialogSum using the ROUGE-L metric, with results averaged over 5 random seeds.

## Key Experimental Results

### Main Results

| Teacher → Student | Strongest non-SRA Baseline Avg | SRA Avg | Observations |
|--------|------|------|------|
| Qwen1.5-1.8B → GPT-2 120M | DSKD 15.35 | 17.97 | Improvement most significant in smaller student models |
| Qwen1.5-1.8B → GPT-2 340M | DSKD 15.57 | 18.10 | S-NI improved from 17.18 to 24.49 |
| Qwen2.5-7B → GPT-2 1.5B | DSKD 19.27 | 20.99 | Effective even when distilling from a large teacher to GPT-2 |
| Qwen2.5-7B → OPT-2.7B | DSKD 20.15 | 20.92 | Maintains lead on OPT student |
| Mistral-7B → TinyLLaMA-1.1B | DSKD 21.33 | 22.52 | Robust across architectures and vocabularies |
| GPT-2 1.5B → GPT-2 120M | AKL 17.03 | 19.24 | Benefits even in identical tokenizer scenarios |

### Ablation Study

| Configuration | Qwen1.5→GPT-2 340M Avg | Qwen1.5→GPT-2 120M Avg | Description |
|------|------|------|------|
| Span logit KD only | 17.36 | 17.10 | Shared vocabulary distillation already provides gains |
| Span logit KD + Geometric Reg | 17.94 | 17.72 | Geometric structure preservation brings stable gains |
| Span logit KD + Cosine | 17.54 | 17.32 | Point representation alignment is helpful but less sufficient than geometry |
| Cosine + Geometric Reg | 17.48 | 16.04 | Lacks stability without logit KD |
| Full SRA | 18.10 | 17.97 | Best results achieved with complementary signals |

| WSL / WSP Configuration | GPT-2 340M Avg | GPT-2 120M Avg | Description |
|------|------|------|------|
| Without WSL and WSP | 16.99 | 14.85 | Span representation quality drops significantly |
| WSL only | 17.11 | 15.77 | Weighted loss provides some assistance |
| WSP only | 17.36 | 15.89 | Weighted pooling is more important than mean pooling |
| WSL + WSP | 18.10 | 17.97 | Shows span weight design is a core component |

### Key Findings
- SRA achieves the highest average ROUGE-L across all teacher-student configurations, indicating that span-level alignment provides stable benefits for cross-tokenizer distillation rather than accidental gains for specific model pairs.
- Geometric regularization and attention weighting are not merely decorative: removing WSP or WSL leads to performance degradation, particularly evident in smaller models like GPT-2 120M.
- Training efficiency benchmarks show SRA's single-step time is 0.2754s, faster than DSKD (0.3520s), MinED (0.4244s), and ULD (0.4393s), at the cost of 21.96GB VRAM, which is slightly higher than MinED/ULD (19.63GB).

## Highlights & Insights
- The most clever aspect is transforming the tokenizer mismatch from a discrete token alignment problem into a continuous span representation problem. Spans, derived from raw character boundaries, are naturally closer to semantic units than tokens.
- While the Multi-Particle / Center-of-Mass analogy sounds theoretical, its implementation as "span pooling with attention importance + preservation of relative geometric structure" is highly practical.
- SRA is not only applicable to different tokenizers but also yields gains in same-tokenizer distillation, suggesting it captures nuances beyond vocabulary overlap, including differences in the teacher-student representation space structure.

## Limitations & Future Work
- Current logit mapping is static and limited to the shared vocabulary subspace, potentially ignoring fine-grained knowledge contained in non-shared tokens.
- Span representation alignment requires online teacher inference; pre-computing all teacher span embeddings would incur prohibitive storage costs.
- Experiments were constrained by computational budgets, focusing primarily on specific benchmarks and decoder-to-decoder settings; verification is needed for embedding models, encoder-decoder models, and longer context tasks.
- As LCS relies on character offsets, matching quality may become a bottleneck when dealing with mixed languages, complex Unicode segmentation, or highly morphologically diverse languages.

## Related Work & Insights
- **vs ULD / MinED**: These methods focus more on alignment at the token or edit distance level. SRA reverts to the text span level, reducing noise caused by tokenizer granularity differences.
- **vs DSKD**: DSKD performs distribution alignment through a unified space, whereas SRA aligns hidden geometry and span logits simultaneously, providing richer knowledge channels.
- **vs MultiLevelOT**: OT methods handle distribution matching but involve higher computational and alignment complexity. SRA’s LCS + span pooling approach is more lightweight and demonstrates shorter per-step times in experiments.
- **vs Same-Tokenizer KD Methods**: Methods like SeqKD, RKL, JS, SKL, and AKL assume consistent vocabularies. SRA still provides improvements under same-tokenizer conditions, suggesting span geometric distillation can serve as a general-purpose KD component.

## Rating
- Novelty: ⭐⭐⭐⭐☆ The combination of span-level CoM representation and geometric regularization is distinctive, with the physical perspective providing clear design motivation.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Good coverage of teacher-student combinations, same/cross-tokenizer scenarios, ablations, and efficiency; could be extended to larger scales and more tasks.
- Writing Quality: ⭐⭐⭐⭐☆ The methodological chain is complete, with formulas clearly mapped to implementation.
- Value: ⭐⭐⭐⭐⭐ Extremely practical for cross-family model distillation and small model deployment, particularly in real-world compression scenarios involving inconsistent tokenizers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] MTA: Multi-Granular Trajectory Alignment for Large Language Model Distillation](mta_multi-granular_trajectory_alignment_for_large_language_model_distillation.md)
- [\[ACL 2026\] Alignment Tuning for Large Language Models: A Data-Centric Lens on Alignment Data Pipelines](alignment_tuning_for_large_language_models_a_data-centric_lens_on_alignment_data.md)
- [\[ICLR 2026\] Distillation of Large Language Models via Concrete Score Matching](../../ICLR2026/model_compression/distillation_of_large_language_models_via_concrete_score_matching.md)
- [\[ICLR 2026\] Pedagogically-Inspired Data Synthesis for Language Model Knowledge Distillation](../../ICLR2026/model_compression/pedagogically-inspired_data_synthesis_for_language_model_knowledge_distillation.md)
- [\[ACL 2026\] LightReasoner: Can Small Language Models Teach Large Language Models Reasoning?](lightreasoner_can_small_language_models_teach_large_language_models_reasoning.md)

</div>

<!-- RELATED:END -->

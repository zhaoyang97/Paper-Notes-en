---
title: >-
  [Paper Note] Multi-Hop Question Generation via Dual-Perspective Keyword Guidance
description: >-
  [ACL 2025][Multi-Hop Question Generation] This paper defines dual-perspective keywords—question keywords (capturing the questioner's intent) and document keywords (reflecting content relevant to the QA pair)—and proposes the DPKG framework. DPKG seamlessly integrates these keywords into the multi-hop question generation process by utilizing an extended Transformer encoder and two answer-aware decoders.
tags:
  - "ACL 2025"
  - "Multi-Hop Question Generation"
  - "Dual-Perspective Keywords"
  - "Answer-aware Attention"
  - "Keyword Guidance"
  - "HotpotQA"
date: 2026-05-08
content_hash: 72cbf56f9f65358a
---

# Multi-Hop Question Generation via Dual-Perspective Keyword Guidance

**Conference**: ACL 2025  
**arXiv**: [2505.15299](https://arxiv.org/abs/2505.15299)  
**Code**: [GitHub](https://github.com/imaodong/DPKG)  
**Area**: Other  
**Keywords**: Multi-Hop Question Generation, Dual-Perspective Keywords, Answer-aware Attention, Keyword Guidance, HotpotQA

## TL;DR

This paper defines dual-perspective keywords—question keywords (capturing the questioner's intent) and document keywords (reflecting content relevant to the QA pair)—and proposes the DPKG framework. DPKG seamlessly integrates these keywords into the multi-hop question generation process by utilizing an extended Transformer encoder and two answer-aware decoders.

## Background & Motivation

Multi-hop question generation (MQG) requires synthesizing multiple pieces of information across a document to generate questions that demand multi-step reasoning to answer. The core challenge lies in: **how to effectively locate key information fragments in the document that are relevant to the QA pair**.

Limitations of prior work:

**Insufficient utilization of keywords**: Many studies focus solely on document-specific keywords (e.g., MulQG, SGCM) or only constrain question-specific keywords during decoding (e.g., CQG), failing to fully exploit the steering potential of keywords.

**Failure to distinguish keyword roles**: Existing work overlooks the fundamental differences between two types of keywords:
   - **Question keywords**: Originating from the question itself, they reflect the questioner's intent and must appear in the generated question.
   - **Document keywords**: Originating from the document, they reflect content related to the QA pair and are used to locate key information fragments.

**Synergy of the two types of keywords**: Question keywords and document keywords jointly locate key information fragments in the document—simulating the natural human questioning process (first review the answer and document -> find relevant information fragments -> select keywords that must appear in the question -> generate the question).

Core Idea: Distinguish and explicitly utilize dual-perspective keywords to better guide multi-hop question generation.

## Method

### Overall Architecture

The DPKG framework consists of three main components:
1. **Extended Transformer Encoder**: Simultaneously encodes the document, the answer, and the document-answer concatenation, producing three sets of hidden states.
2. **Keyword Generation Decoder**: Generates dual-perspective keyword sequences based on the document and answer states.
3. **Question Generation Decoder**: Generates multi-hop questions using the information fragments located by the keywords.

The two decoders share similar but independent structures, sharing encoder outputs but targeting different task objectives.

### Key Designs

1. **Extended Transformer Encoder**:

    - Function: Simultaneously encodes three inputs—document $D^i$, answer $A^i$, and document-answer concatenation $D^i;A^i$.
    - Mechanism: Modifies the internal structure to allow the three parts to share self-attention while remaining independently encoded.
    - Output: $H_{doc}^i$ (document state), $H_{ans}^i$ (answer state), and $H_{da}^i$ (document-answer state).
    - Design Motivation: Encoding the document and answer separately facilitates the subsequent computation of answer-aware states.

2. **Answer-aware Attention**:

    - Function: Introduces answer information into the decoder to weight document attention.
    - Core Formula: $H_a = \text{softmax}(\frac{H_k^{t-1} H_{doc}^T}{\sqrt{d}} \odot K_{weight}) H_{doc}$
    - where $K_{weight} = \text{MeanPooling}(\frac{H_{doc} H_{ans}^T}{\sqrt{d}})$ captures the document-answer relation.
    - Design Motivation: $K_{weight}$ encodes which parts of the document are relevant to the answer, guiding attention to focus on answer-related regions through element-wise multiplication.

3. **Fusion Module**:

    - Function: Combines the answer-aware state $H_a$ and the standard cross-attention state $H_h$.
    - Gating Mechanism: $H_k^t = gate \odot H_a + (1-gate) \odot H_h$, where $gate = \text{sigmoid}([H_a; H_h])$.
    - Design Motivation: Adaptively balances answer-aware information with original contextual information.

4. **Two Modes of Keyword Guidance**:

    - **Hard Mode**: Adds special prefixes `<qes>` or `<doc>` before keywords to identify types, recognizing their roles during the keyword generation stage.
    - **Soft Mode**: Does not add prefixes, dynamically identifying the role of each keyword during the question generation stage.
    - Hard mode performs better in the SF setting, while Soft mode is superior in the Full setting.

### Loss & Training

Joint training loss: $\mathcal{L} = \beta_1 \mathcal{L}_1 + \beta_2 \mathcal{L}_2 + \beta_3 \mathcal{L}_3$

- $\mathcal{L}_1$: Cross-entropy loss for keyword generation.
- $\mathcal{L}_2$: Cross-entropy loss for question generation.
- $\mathcal{L}_3$: Reducing the discrepancy between generated keyword representations and ground-truth keyword representations.
    - $\mathcal{L}_3 = \|F_k - F_g\|_2$, where $F_k$ and $F_g$ are the mean-pooled representations of the keyword decoder output and the BART-encoded ground-truth keywords, respectively.
    - Design Motivation: Ground-truth keywords are used during training, whereas generated keywords are used during inference. $\mathcal{L}_3$ bridges this gap.

Keyword Annotation: Annotation is conducted automatically using SpaCy (`en_core_web_sm` model), focusing on sentences related to answers and questions to extract entities or phrases as keywords. Question words (e.g., "what", "how") are designated as question keywords.

## Key Experimental Results

### Main Results - HotpotQA

**SF (Supporting Fact) Setting:**

| Model | BLEU-4 | METEOR | ROUGE-L |
|------|--------|--------|---------|
| BART | 20.39 | 23.46 | 37.23 |
| CQG | 25.09 | 27.45 | 41.83 |
| QA4QG-large | 25.70 | 27.44 | 46.48 |
| SGCM | 26.16 | 28.51 | 44.06 |
| **DPKG_hard** | **26.80** | 27.87 | **46.50** |
| DPKG_soft | 26.19 | **28.51** |  46.36 |

**Full Setting:**

| Model | BLEU-4 | METEOR | ROUGE-L |
|------|--------|--------|---------|
| BART | 16.77 | 20.07 | 33.69 |
| CQG | 21.46 | 24.97 | 39.61 |
| SGCM | 22.61 | 26.04 | 40.61 |
| DPKG_hard | 22.74| 24.90 | **43.29** |
| **DPKG_soft** | **23.33** | 25.21 | 43.18 |

### Ablation Study / Keyword Type Experiments

| Model | Keyword Score (KP) | BLEU-4 (SF) | ROUGE-L (SF) |
|------|---------------|-------------|--------------|
| DPKG_hard | 88.13 | 26.80 | 46.50 |
| DPKG_soft | 88.86 | 26.19 | 46.36 |
| DPKG_D (Document Keywords Only) | 87.05 | 23.95 | 45.62 |
| DPKG_Q (Question Keywords Only) | 79.36 | 25.77 | 45.16 |

Ablation (Module Ablation, SF Setting):

| Configuration | BLEU-4 | ROUGE-L |
|------|--------|---------|
| DPKG_hard | 26.80 | 46.50 |
| w/o $\mathcal{L}_3$ | 24.74 | 45.08 |
| w/o Answer-aware | 24.83 | 45.09 |

### Key Findings

1. **Dual-perspective outperforms single-perspective**: Even when using ground-truth keywords, DPKG_hard/soft consistently outperforms models using only question keywords (DPKG_Q) or only document keywords (DPKG_D).
2. **Question keywords are more critical than document keywords**: DPKG_Q's question generation quality is significantly better than that of DPKG_D, despite having a lower keyword generation score (79.36 vs. 87.05).
3. **$\mathcal{L}_3$ and answer-aware attention are equally important**: Removing either module leads to a performance drop of approximately 2 points in BLEU-4.
4. **Hard vs. Soft modes are complementary**: Hard mode performs better in the SF setting (handling short documents), while Soft mode performs better in the Full setting (handling noise in long documents).
5. **TS-BART verifies the generalizability of dual-perspective keywords**: Even a simple two-stage BART benefits from dual-perspective keywords.
6. DPKG achieves remarkably significant gains in ROUGE-L (SF: 46.50, Full: 43.29), indicating that the generated questions show better long-range semantic alignment with the reference questions.

## Highlights & Insights

1. **Role differentiation of keywords**: Subdividing keywords into "question keywords" and "document keywords" is a simple but effective conceptual contribution. This closely aligns with the cognitive process of human questioning.
2. **Mitigating the training-inference discrepancy**: $\mathcal{L}_3$ narrows the gap between using ground-truth during training and generated keywords during inference by minimizing the distance between their representations. The solution is elegant and effective.
3. **Framework scalability**: DPKG's encoder-dual-decoder architecture can easily accommodate other types of intermediate generation tasks.

## Limitations & Future Work

1. **Keyword annotation depends on SpaCy**: The quality of automatic annotation is limited by named entity recognition (NER) tools, which might miss important keywords in complex documents.
2. **Evaluated only on HotpotQA**: The method lacks validation on other multi-hop QA datasets (e.g., 2WikiMultiHopQA).
3. **BART-based architecture**: The work does not explore whether larger PLMs (e.g., T5-large, LLaMA) can yield further improvements.
4. **Lack of unified Hard and Soft modes**: Ideally, there should be an automated mechanism to select the mode instead of relying on manual selection.
5. **Keyword generation is a source of error propagation**: The quality of the keywords directly impacts the question quality, and errors can accumulate and propagate.

## Related Work & Insights

- CQG only constrains question keywords during decoding, whereas DPKG extends this idea to dual-perspective and introduces an independent keyword generation stage.
- QA4QG enhances BART with a QA module, whereas DPKG enhances it with a keyword guidance module—offering a similar intuition but with more explicit keyword steering.
- The answer-aware attention mechanism draws inspiration from Wang et al. (2024), but its application in multi-hop scenarios is novel.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The definition of "dual-perspective keywords" is clear, intuitive, and highly effective, representing a refined conceptual innovation.
- **Experimental Thoroughness**: ⭐⭐⭐ — The experiments on HotpotQA are comprehensive (main results, keyword type analysis, and ablation studies), but validation on other datasets is missing.
- **Writing Quality**: ⭐⭐⭐⭐ — The keyword definitions and annotation processes are intuitively explained with diagrams, and the framework description is clear.
- **Value**: ⭐⭐⭐⭐ — Offers a new perspective and effective methodology for MQG tasks. The code is open-sourced, and the keyword annotation method is reusable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] AIDE: Attribute-Guided Multi-Hop Data Expansion for Data Scarcity in Task-Specific Fine-tuning](aide_attribute-guided_multi-hop_data_expansion_for_data_scarcity_in_task-specifi.md)
- [\[ACL 2025\] DRS: Deep Question Reformulation With Structured Output](drs_deep_question_reformulation_with_structured_output.md)
- [\[ACL 2025\] Counterspeech the Ultimate Shield! Multi-Conditioned Counterspeech Generation through Attributed Prefix Learning](hippro_counterspeech_gen.md)
- [\[AAAI 2026\] Local Guidance for Configuration-Based Multi-Agent Pathfinding](../../AAAI2026/others/local_guidance_for_configuration-based_multi-agent_pathfinding.md)
- [\[ACL 2025\] Evaluating Design Decisions for Dual Encoder-based Entity Disambiguation](evaluating_design_decisions_for_dual_encoder-based_entity_disambiguation.md)

</div>

<!-- RELATED:END -->

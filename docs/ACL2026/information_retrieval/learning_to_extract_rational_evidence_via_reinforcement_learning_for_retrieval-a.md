---
title: >-
  [Paper Note] Learning to Extract Rational Evidence via Reinforcement Learning for Retrieval-Augmented Generation
description: >-
  [ACL 2026][Information Retrieval & RAG][Retrieval-Augmented Generation] Proposes EviOmni, which learns to extract rational evidence from retrieved documents via a "reason-then-extract" paradigm. It integrates evidence re…
tags:
  - "ACL 2026"
  - "Information Retrieval & RAG"
  - "Retrieval-Augmented Generation"
  - "Evidence Extraction"
  - "Reinforcement Learning"
  - "Reasoning-guided Extraction"
  - "GRPO"
date: 2026-05-08
content_hash: 92ea4e94e89245ad
---

<!-- Generated automatically by src/gen_stubs.py -->
# Learning to Extract Rational Evidence via Reinforcement Learning for Retrieval-Augmented Generation

**Conference**: ACL 2026  
**arXiv**: [2507.15586](https://arxiv.org/abs/2507.15586)  
**Code**: [GitHub](https://github.com/HITsz-TMG/EviOmni)  
**Area**: Image Restoration  
**Keywords**: Retrieval-Augmented Generation, Evidence Extraction, Reinforcement Learning, Reasoning-guided Extraction, GRPO

## TL;DR

Proposes EviOmni, which learns to extract rational evidence from retrieved documents via a "reason-then-extract" paradigm. It integrates evidence reasoning and extraction into a unified trajectory, uses knowledge token masking to prevent information leakage, and optimizes with verifiable rewards through GRPO. It achieves superior accuracy over full-text retrieval across 5 benchmarks with an extremely high compression ratio (~38x).

## Background & Motivation

**Background**: RAG enhances LLM accuracy by retrieving external passages. However, retrieved passages often contain noise and irrelevant content, necessitating evidence extraction or denoising. Existing methods include reranking (placing relevant passages at the top) and summarization/extraction (training filter models via SFT).

**Limitations of Prior Work**: Existing methods extract evidence directly without deep reasoning, which can lead to missing key clues. For instance, direct extraction might discard critical information scattered across multiple passages due to insufficient context understanding. Training data is typically constructed via heuristics (e.g., string inclusion, word overlap) and does not directly align with the final goal of RAG.

**Key Challenge**: Conventional evidence extraction follows a "what you see is what you get" approach, lacking deep reasoning over the retrieved content. When key clues require cross-passage reasoning to be identified, direct extraction easily results in omissions.

**Goal**: To enable the evidence extractor to reason first (identifying clues and their relevance within retrieved content) and then extract based on the reasoning results, while using RL optimization to align extraction results directly with downstream task accuracy.

**Key Insight**: Empirical research found that adding reasoning steps (reason→extract) to SFT data improved the answer recall of evidence from 70.8% to 75.2% (on the NQ dataset), demonstrating the value of reasoning-guided extraction.

**Core Idea**: Integrate evidence reasoning `<reason>` and evidence extraction `<extract>` into a unified generative trajectory, using knowledge token masking for decoupled evaluation, and optimizing end-to-end with GRPO plus three types of verifiable rewards (answer, length, format).

## Method

### Overall Architecture

Given an input query $q$ and top-$k$ retrieved passages $P$, EviOmni generates a response consisting of three parts: `<reason>`reasoning`</reason><extract>`evidence`</extract><answer>`answer`</answer>`. During training, quality for reasoning and evidence is evaluated separately via knowledge token masking, with GRPO optimization driven by three types of rewards.

### Key Designs

1.  **Rational Evidence Extraction Paradigm**:
    - **Function**: Guides evidence extraction through reasoning to reduce the omission of key clues.
    - **Mechanism**: The model first generates the `<reason>` part (analyzing the relevance and clues contained in each passage), then generates the `<extract>` part (a concise summary of evidence) based on the reasoning. Formulated as $e \sim \mathcal{M}_\mathcal{E}(\cdot|q,P,r) \cdot \mathcal{M}_\mathcal{E}(r|q,P)$.
    - **Design Motivation**: Reasoning identifies scattered clues, excludes misleading information, and correlates cross-passage information more reliably than direct extraction.

2.  **Knowledge Token Masking**:
    - **Function**: Decouples the quality evaluation of reasoning and evidence during training.
    - **Mechanism**: (1) Mask evidence $e$ $\rightarrow$ generate answer $o_r$ based only on reasoning $r$ to evaluate reasoning quality; (2) Mask passages $P$ and reasoning $r$ $\rightarrow$ generate answer $o_e$ based only on evidence $e$ to evaluate evidence quality. Hard masks (replacing input tokens) are used instead of soft masks (adjusting attention) to prevent leakage of information already aggregated by causal attention.
    - **Design Motivation**: Without separation, the answer could be derived from the full context (including original passages) after evidence generation, failing to reflect the quality of the evidence itself.

3.  **Three Types of Verifiable Rewards**:
    - **Function**: Guides the model to optimize for three desired attributes.
    - **Mechanism**: Answer reward $R^{ans}$ = unigram F1 (unifying evaluation across tasks); Length reward $R^{len}$ encourages comprehensive reasoning (longer than evidence) and concise evidence (much shorter than passages); Format reward $R^{fmt}$ ensures correct label formatting. Final reward $R^{final} = \lambda_1 R^{ans} + \lambda_2 R^{len} + \lambda_3 R^{fmt}$.
    - **Design Motivation**: Using downstream answer accuracy directly as a reward avoids the misalignment between heuristic metrics and the final objective.

### Loss & Training

Uses GRPO for on-policy optimization, with Qwen2.5-1.5B/7B-Instruct as base models. The same model acts as both the extractor and the generator.

## Key Experimental Results

### Main Results

Results for the 1.5B model on NQ/TQA/HotpotQA (EM/F1/Compression Ratio):

| Method | NQ EM | NQ CR | TQA EM | HotpotQA EM |
| :--- | :--- | :--- | :--- | :--- |
| Full (No compression) | 41.97 | 1.0x | 57.02 | 19.20 |
| FilCo | 36.62 | 16.3x | 54.06 | 18.18 |
| SEER | 36.93 | 13.2x | 54.57 | 18.60 |
| **EviOmni** | **41.14** | **38.1x** | **56.84** | **20.46** |

EviOmni approaches or even exceeds full-text performance while achieving ~38x compression.

### Ablation Study

| Configuration | NQ AR | HotpotQA AR |
| :--- | :--- | :--- |
| Vanilla Evidence (No reasoning) | 70.79% | 60.55% |
| Rational Evidence (With reasoning) | **75.24%** | **67.74%** |
| Rationale itself | 77.30% | 71.48% |

### Key Findings

- The answer recall of rational evidence is 4–7 percentage points higher than vanilla evidence, confirming the value of reasoning-guided extraction.
- Performance near full-text input at a 38x compression ratio indicates highly refined evidence extraction.
- Improvements on OOD datasets (HotpotQA) suggest strong generalization.
- Supports both traditional RAG and Agentic RAG (e.g., early termination, noise robustness).

## Highlights & Insights

- The **"reason-then-extract" paradigm shift** has broad implications—it serves not only RAG but any task requiring the extraction of key content from noisy information.
- **Knowledge Token Masking** elegantly addresses the technical challenge of information leakage during training.
- The result of **maintaining performance at a 38x compression ratio** is impressive and holds significant practical value for inference efficiency.

## Limitations & Future Work

- The reasoning process increases generation length; while evidence is shorter, total output is longer.
- Training and evaluation were limited to QA tasks; applicability to dialogue or summarization requires verification.
- Reasoning quality is constrained by the base model's capability; 1.5B models have limited reasoning depth.

## Related Work & Insights

- **vs Recomp/FilCo/SEER**: These methods perform direct extraction/summarization without reasoning guidance; they are outperformed by EviOmni in both compression ratio and accuracy.
- **vs SFT methods (Wang et al., 2023)**: SFT depends on heuristic construction of training data, whereas RL directly aligns with downstream task objectives.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The combination of "reason→extract" paradigm + knowledge masking + RL optimization is highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Validated across 5 benchmarks, two model scales, and both traditional and Agentic RAG.
- Writing Quality: ⭐⭐⭐⭐ Clear framework diagrams and persuasive empirical studies.
- Value: ⭐⭐⭐⭐⭐ Directly practical for enhancing the efficiency and quality of RAG pipelines.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Language-Coupled Reinforcement Learning for Multilingual Retrieval-Augmented Generation](language-coupled_reinforcement_learning_for_multilingual_retrieval-augmented_gen.md)
- [\[ACL 2026\] ChatR1: Reinforcement Learning for Conversational Reasoning and Retrieval Augmented Question Answering](chatr1_reinforcement_learning_for_conversational_reasoning_and_retrieval_augment.md)
- [\[ACL 2026\] Agentic Conversational Search with Contextualized Reasoning via Reinforcement Learning](agentic_conversational_search_with_contextualized_reasoning_via_reinforcement_le.md)
- [\[ACL 2026\] End-to-End Optimization of LLM-Driven Multi-Agent Search Systems via Heterogeneous-Group-Based Reinforcement Learning](end-to-end_optimization_of_llm-driven_multi-agent_search_systems_via_heterogeneo.md)
- [\[ACL 2026\] Utility-Oriented Visual Evidence Selection for Multimodal Retrieval-Augmented Generation](utility-oriented_visual_evidence_selection_for_multimodal_retrieval-augmented_ge.md)

</div>

<!-- RELATED:END -->

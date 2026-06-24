---
title: >-
  [Paper Note] Improving Model Factuality with Fine-grained Critique-based Evaluator
description: >-
  [ACL2025][LLM Safety][Factuality] A fine-grained factuality evaluator, FenCE, is trained to improve evaluation accuracy by augmenting textual critiques and diverse source documents retrieved through multiple tools on public datasets. FenCE is then leveraged to edit and score generator responses to construct preference training data, improving Llama2-7B/Llama3-8B by 16.86%/14.45% in FActScore, respectively.
tags:
  - "ACL2025"
  - "LLM Safety"
  - "Factuality"
  - "Hallucination"
  - "Critique-based Evaluator"
  - "Preference Training"
  - "FenCE"
date: 2026-05-08
content_hash: d4b5d59fe0d460be
---

# Improving Model Factuality with Fine-grained Critique-based Evaluator

**Conference**: ACL2025  
**arXiv**: [2410.18359](https://arxiv.org/abs/2410.18359)  
**Code**: Unavailable  
**Area**: LLM Safety  
**Keywords**: Factuality, Hallucination, Critique-based Evaluator, Preference Training, FenCE  

## TL;DR

A fine-grained factuality evaluator, FenCE, is trained to improve evaluation accuracy by augmenting textual critiques and diverse source documents retrieved through multiple tools on public datasets. FenCE is then leveraged to edit and score generator responses to construct preference training data, improving Llama2-7B/Llama3-8B by 16.86%/14.45% in FActScore, respectively.

## Background & Motivation

### Problem Definition
The hallucination problem in Large Language Models (LLMs)—generating plausible-sounding but actually incorrect information—remains a persistent challenge. One hypothesis suggests that LLMs fail to distinguish the boundary between facts in their memory and other plausible-sounding but erroneous information.

### Limitations of Prior Work

**Test-time methods** (contrastive decoding, post-editing): Introduce severe latency overhead, making them unsuitable for real-time applications.

Two main categories of **training-time methods** and their issues:

**FactTune-like methods**: Prefer candidate responses with higher factuality, but are limited by the capacity ceiling of the generator itself.

**EVER-like methods**: Correct erroneous information but are prone to introducing "lesser-known facts"—knowledge that the model has not fully memorized during pre-training, which paradoxically leads to more hallucinations.

**Issues in Evaluators**:
- Relying on commercial models (e.g., GPT-4) poses massive usage restrictions.
- Allowing generators to self-evaluate factuality suffers from self-bias.
- Source documents in public datasets originate from limited domains (e.g., only news or Wikipedia).
- Labels are typically binary or numerical ratings, offering limited feedback.

## Method

### Overall Architecture

The paper consists of two core components:
1. **Training a Fine-grained Critique-based Evaluator (FenCE)**
2. **Enhancing Generator Factuality using FenCE**

### Key Designs

#### Key Design 1: FenCE Evaluator Training

**Base Settings**:
- Initialized from Llama3-8B-chat.
- Trained on public factuality verification datasets (XSum, QAGS, FRANK, RAGTruth, FActScore, Q2, FaithDial, BEGIN).
- Task: Given a (claim, document) pair, determine if the claim is Supported / Contradictory / Unverified.

**Enhancement 1 - Textual Critique Enhancement**:
- Instead of only predicting labels, it generates a textual critique explaining the reasoning behind the judgment.
- Llama3-70B-chat is used to generate critiques and labels for each sample.
- Quality Control: A critique is utilized only if the generated label matches the ground-truth label.
- Coverage: 77.2% of training samples successfully obtain critiques.

**Enhancement 2 - Multi-Tool Source Document Enhancement**:
- Leverages three tools to acquire additional source documents:
    - **Search Engine** (Bing Search API)
    - **Knowledge Base** (Wikipedia)
    - **Knowledge Graph** (Google Knowledge Graph API)
- For each claim, the model is prompted to generate tool calls (e.g., search queries) to retrieve diverse documents.
- Quality filtering is similarly performed via label consistency.
- Coverage: 54.1% of samples obtain new source documents.

**Intuition**: If a claim is factual, supporting evidence can likely be retrieved through tools; if it is a hallucination, tools are highly unlikely to find supporting documents.

**Quality Verification**: Manual inspection of 45 randomly sampled instances yields a critique accuracy of 95.6% and a tool-retrieved document accuracy of 97.8%.

#### Key Design 2: Enhancing Generator Factuality using FenCE

**Overall Process**:
Generate $N$ candidate responses for each prompt $\to$ FenCE Evaluation + Editing $\to$ FenCE Scoring $\to$ Preference Data Construction $\to$ SFT + DPO Training.

**Response Editing (Core Innovation)**: A three-step iterative process:

**Step 1 - Evaluation**:
- Decompose the response into claims.
- Call tools to retrieve relevant documents for each claim.
- Use FenCE to evaluate factuality and output a critique.

**Step 2 - Editing (Crucial: Avoiding Lesser-Known Facts)**:
- If a claim is judged as "Unverified" or "Contradictory":
    - Query the generator: "Is this claim factual?" (without external knowledge).
    - If the answer is "unknown" $\to$ treat as a lesser-known fact $\to$ **remove** this information.
    - If the answer is "true"/"false" $\to$ treat as a common fact $\to$ **rectify based on the critique**.
- This design avoids introducing knowledge that was not memorized by the model during pre-training into the training data.

**Step 3 - Continuation**:
- Use the edited paragraph as a prefix to continue generating the next paragraph.
- Reduces the cumulative propagation of errors.

**Generator Training**:
- **SFT Phase**: Select target responses from the top-$k$ responses with the highest FenCE scores.
- **DPO Phase**: Construct preference pairs by selecting the preferred response from the top-$k$, and lower-scoring responses as rejected candidates.
- Uses FenCE (rather than the generator itself) for scoring to mitigate self-bias.

### Loss & Training
Standard DPO Loss:
$$\max_{\mathcal{G}} \mathbb{E}_{(x,y_w,y_l)\sim\mathcal{TR}_{Gen}} \left[\log\sigma\left(\beta\log\frac{\pi_\mathcal{G}(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta\log\frac{\pi_\mathcal{G}(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$

## Experiments

### Evaluator Experiments

**LLM-AggreFact Benchmark** (10 datasets covering fact verification, summarization, and long-form QA)

| Model | Average BAcc |
|------|----------|
| Llama3-8B-chat | 66.4 |
| FenCE (Vanilla SFT) | 71.8 |
| FenCE (Critique Only) | 73.7 |
| **FenCE (Full)** | **74.7** |
| Mistral-123B | 67.3 (Est.)|
| Claude-3 Opus | 74.1 |
| GPT-4 | 75.3 |

FenCE outperforms Mistral-123B and Claude-3 Opus with only 8B parameters, approaching GPT-4 performance.

Ablation: Critique-based enhancement yields $+1.9\%$, tool-document enhancement yields an additional $+1.0\%$, for a total gain of $+2.9\%$.

### Generator Factuality Experiments

**FActScore Results**

| Method | Llama2-7B % Facts | Llama3-8B % Facts |
|------|-------------------|-------------------|
| Baseline | 38.57 | 50.96 |
| + SFT | 40.83 | 52.52 |
| + Self-Eval-SKT | 43.73 | 56.80 |
| + EVER-Pref | 42.66 | 57.18 |
| + FactTune-FS | 46.60 | 58.45 |
| **+ Ours (E/R+Coarse)** | **55.43(+16.86)** | **65.41(+14.45)** |

**TruthfulQA Results**

| Method | Llama2-7B % True*Info | Llama3-8B % True*Info |
|------|----------------------|----------------------|
| Baseline | 38.83 | 58.89 |
| + FactTune-FS | 52.48 | 64.58 |
| **+ Ours** | **56.47(+17.64)** | **67.14(+8.25)** |

Outperforms the best baseline by 8.83% / 6.96% (FActScore) and 3.99% (TruthfulQA).

### Ablation Study

| Configuration | Llama3 % Facts |
|------|---------------|
| SFT + FenCE | 56.26 |
| Edit (correcting all errors) | 58.91 |
| Coarse (ranking scores) | 60.89 |
| Edit + Coarse | 64.37 |
| **E/R + Coarse (Ours)** | **65.41** |

Key Findings:
- FenCE, acting as an evaluator, brings improvements across all methods (vs. utilizing generator self-evaluation).
- Edit/Remove (distinguishing between lesser-known and common facts) outperforms Edit (correcting all errors), verifying the significance of avoiding lesser-known facts.

### Generation Behavior Analysis

The trained generator exhibits an "admitting ignorance" behavior (i.e., knowing what it knows, and acknowledging what it does not):
- Generates less information for unfamiliar entities and more information for popular entities.
- Refuses to answer more frequently for rare entities.
- Demonstrates consistent factuality improvements across all population subgroups.

### Hyperparameter Analysis
- Number of editing iterations: While training data quality continuously improves across iterations, test performance converges after the 3rd iteration.
- top-$k$: top-3 and top-5 show similar results.

## Highlights & Insights

1. **Deep Insight into Lesser-Known Facts**: Reveals that correcting erroneous information may introduce lesser-known facts, paradoxically escalating hallucinations, and elegantly resolves this issue via the Edit/Remove strategy—guided by the principle "let the model generate only what it is confident about."
2. **Decoupled Evaluator and Generator**: Replaces self-evaluation with an independent evaluator, fundamentally eliminating self-bias.
3. **Dual Value of Critique**: Textual critiques not only improve evaluation accuracy but also provide actionable feedback for response editing.
4. **Data Diversity via Tool Enhancement**: Diverse documents are retrieved using three tools (search engine, knowledge base, knowledge graph), enhancing the generalization capability of the evaluator.
5. **Interpretability of Post-Training Behavior**: The generator learns to adjust the amount of output information based on entity familiarity, demonstrating expected and desirable "conservative" behavior.

## Limitations & Future Work

1. The evaluator is trained exclusively on human-annotated model response datasets, without exploring synthetic data or human-written claim datasets.
2. The focus is solely on text-to-text generation, leaving mathematical reasoning or programming tasks untouched.
3. Generator experiments are validated only on a single public dataset (FActScore).
4. Dependence on external tools (search engines, knowledge graphs, etc.) for source documents might make it inapplicable to offline or resource-constrained settings.
5. The editing process requires multiple queries to FenCE and tools, resulting in high overhead for constructing training data.

## Related Work

- **Factuality Evaluation**: FActScore, FacTool (fine-grained evaluation frameworks); Vu et al. (training evaluators on public datasets).
- **Factuality Training**: FactTune (preferring high-scoring candidates), EVER (correcting erroneous information), Self-Eval-SKT (self-training evaluator).
- **Test-Time Methods**: DoLa (contrastive decoding over layers), post-editing methods.
- **Reward Modeling**: RLHF, DPO, and their variants.

## Rating ⭐⭐⭐⭐⭐

A highly complete and solid piece of work—forming a closed loop from evaluator training to generator enhancement. The Edit/Remove strategy for lesser-known facts is the core innovation, and the experimental results are extremely convincing (outperforming the SOTA by up to 8.83%). The evaluator quality validation, ablation studies, and behavioral analysis are exceptionally thorough. This work delivers a comprehensive, highly actionable solution for improving LLM factuality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Ewe: Improving Factuality with Explicit Working Memory](improving_factuality_with_explicit_working_memory.md)
- [\[NeurIPS 2025\] FALCON: Fine-grained Activation Manipulation by Contrastive Orthogonal Unalignment for Large Language Model](../../NeurIPS2025/llm_safety/falcon_fine-grained_activation_manipulation_by_contrastive_orthogonal_unalignmen.md)
- [\[ACL 2025\] How Does Response Length Affect Long-Form Factuality](how_does_response_length_affect_long-form_factuality.md)
- [\[ICML 2025\] Improving Your Model Ranking on Chatbot Arena by Vote Rigging](../../ICML2025/llm_safety/improving_your_model_ranking_on_chatbot_arena_by_vote_rigging.md)
- [\[ACL 2025\] UAlign: Leveraging Uncertainty Estimations for Factuality Alignment on Large Language Models](ualign_leveraging_uncertainty_estimations_for_factuality_alignment_on_large_lang.md)

</div>

<!-- RELATED:END -->

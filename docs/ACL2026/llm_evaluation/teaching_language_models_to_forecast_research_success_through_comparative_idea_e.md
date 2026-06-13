---
title: >-
  [Paper Note] Teaching Language Models to Forecast Research Success Through Comparative Idea Evaluation
description: >-
  [ACL 2026][LLM Evaluation][Comparative Forecasting] This paper investigates whether language models can learn to predict the empirical success of research ideas. By constructing a dataset of 11…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "Comparative Forecasting"
  - "Research Idea Ranking"
  - "RL Reasoning"
date: 2026-05-08
content_hash: 51948b0207319405
---

# Teaching Language Models to Forecast Research Success Through Comparative Idea Evaluation

**Conference**: ACL 2026  
**arXiv**: [2605.21491](https://arxiv.org/abs/2605.21491)  
**Code**: To be released  
**Area**: LLM Evaluation / LLM Reasoning  
**Keywords**: Comparative Forecasting, LLM Evaluation, Research Idea Ranking, RL Reasoning

## TL;DR

This paper investigates whether language models can learn to predict the empirical success of research ideas. By constructing a dataset of 11,488 idea pairs (based on PapersWithCode objective outcomes) and training an 8B model with SFT and RLVR, the authors achieve 77.1% accuracy, surpassing GPT-5's 61.1%, becoming an effective idea verifier for automated scientific discovery.

## Background & Motivation

**Background**: Recently, LLMs have started acting as autonomous research agents, capable of generating hypotheses, implementing experiments, and analyzing results. A typical pattern is "high-throughput idea generation": given a research objective, models can generate hundreds of candidate methods. However, current idea evaluation relies entirely on subjective LLM judgments (novelty, excitement, feasibility, etc.), which are often just proxies—an idea might be novel and interesting but may not work in practice.

**Limitations of Prior Work**: (1) Evaluation lacks objectivity: existing systems use LLM scoring based on fictional criteria rather than real experimental results; (2) Evaluation efficiency bottleneck: hundreds of ideas cannot be verified sequentially through experiments; (3) Lack of interpretability: black-box scoring fails to explain to researchers why a certain idea is better.

**Key Challenge**: How to use objective empirical results to predict which idea will perform better without running experiments?

**Goal**: To explore whether LMs can learn to predict the empirical success of research ideas and support predictions with interpretable chains-of-thought.

**Key Insight**: Framing the problem as "comparative empirical prediction"—given a research goal and two ideas, predict which performs better on a benchmark. The key observation is that while precise prediction is difficult, researchers regularly form intuitions by comparing existing work; the goal is to see if LMs can learn this intuition.

**Core Idea**: Extracting an idea-pair dataset based on objective results from PapersWithCode leaderboards and training smaller LMs using SFT and RL (with verifiable rewards) for comparative evaluation and reasoning, achieving better performance than GPT-5.

## Method

### Overall Architecture

The pipeline in this paper consists of three parts: (1) Dataset construction: scraping leaderboards from PapersWithCode to extract idea pairs and objective result labels; (2) Model training: starting with SFT for foundational fine-tuning, followed by RL to learn reasoning; (3) Evaluation and analysis: testing generalization and robustness across multiple test sets.

### Key Designs

1.  **Benchmark Scraping and Idea Pair Extraction**:

    - Function: Extracting idea pairs and research goals from 1,918 NLP leaderboards to build a training dataset with broad coverage.
    - Mechanism: (a) Scraping PapersWithCode leaderboards, where each entry points to a paper reporting results (RR paper); (b) Using LLMs to verify if the paper is the method's original source or just a reporter; if the latter, the original paper is found; (c) Using LLMs to extract idea descriptions from the original paper (excluding identifiers like results, authors, or years); (d) Extracting research goals from official descriptions or PapersWithCode data (e.g., "detecting cyber threats").
    - Design Motivation: To ensure idea descriptions are detailed (including algorithms and mathematical details) and based on original papers, avoiding vague generalities; per-idea verification ensures description accuracy (92% accuracy, with 4% incomplete and 8% completely incorrect).

2.  **Unified Scoring, Difficulty Stratification, and Reason-Chain Extraction**:

    - Function: Assigning objective win/loss labels and difficulty levels to each idea pair, and preparing learnable reasoning chains.
    - Mechanism: (a) Normalizing all metrics within each benchmark using min-max, inverting "lower-is-better" metrics (e.g., perplexity), and averaging multiple metrics to obtain a "Unified Score" $s_i$; (b) Categorizing score gaps based on standard deviation $\sigma$ into 1σ (Hard), 2σ (Medium), and 3σ (Easy); (c) Using two types of reasoning chains—Synthetic: GPT-5 generating structured reasoning traces for 2,125 pairs, filtering 1,369 correct ones, then doubling to 2,738 via position swapping; Literature: extracting existing experimental comparisons from papers as reasoning evidence.
    - Design Motivation: Cross-benchmark metrics cannot be compared directly—normalization ensures pair-wise consistency, and difficulty stratification helps control evaluation complexity; using two distinct sources of reasoning data allows testing the sources of RL capability.

3.  **Two-stage Training: SFT + RLVR Reasoning**:

    - Function: Initial foundation fine-tuning for intuition, followed by teaching the model to generate interpretable reasoning before predicting.
    - Mechanism: (i) Standard SFT: LoRA training (rank=64, lr=2e-4) on an 8B model using the full training set with classification loss $\mathcal{L}_{SFT}=-\log P(y|g,h_A,h_B)$; (ii) RL with Reasoning: A cold-start SFT phase on 170 labeled pairs to teach scientific argumentation style; an RLVR phase using DAPO and Dr. GRPO, where rewards include correctness (Correct +3, Incorrect -3) and formatting (+0.5 for thought and answer tags).
    - Design Motivation: Constraining reasoning style and reward formatting avoids reward hacking; Dr. GRPO corrects the standard deviation term to solve length bias. The two stages allow the small model to learn both comparative intuition and the ability to provide interpretable explanations.

## Key Experimental Results

### Main Results: Base Performance

| Model | 1-σ | 2-σ | 3-σ | Overall | Cross-Domain |
|-------|-----|-----|-----|---------|--------------|
| Qwen3 Base | 18.4% | 26.1% | 11.0% | 20.1% | 3.6% |
| Direct-SFT | 70.9% | 85.6% | 84.6% | **77.1%** | 45.7% |
| Reason-SFT-DrGRPO | 66.2% | 76.4% | 83.5% | 71.4% | 49.1% |
| GPT-5 (High Reasoning) | 61.9% | 61.3% | 56.0% | 61.1% | 46.0% |

**Key Findings**: (1) SFT dramatically improves 8B model performance from 20% to 77%, exceeding GPT-5's 61.1%; (2) Difficulty stratification is effective, with 1σ < 2σ < 3σ; (3) Although RL accuracy is slightly lower, it offers better cross-domain generalization.

### Independent Test Sets and Robustness

| Model | Accuracy |
|------|--------|
| Qwen3 Direct-SFT | 63.4% |
| Qwen3 Reason-SFT-DrGRPO | **67.5%** |
| GPT-4.1 + Retrieval | 51.4% |

**Key Findings**:

- The 8B model outperforms GPT-4.1 by 16 percentage points on independent datasets, proving it has learned transferable comparative reasoning.
- Position bias consistency exceeds 85%, indicating no dependence on input order.
- No length bias was detected; the model does not favor longer descriptions.
- Accuracy did not drop significantly when rewritten by Gemini-2.5, suggesting the model understands the content.

## Highlights & Insights

- **Example of small models beating large ones**: The 8B model after SFT outperforms GPT-5 by 16 points, demonstrating the power of task-specific fine-tuning.
- **Clever RL reasoning chain design**: Instead of using direct self-generation (which can lower performance), a cold-start with labels followed by RL exploration is used. The two-stage strategy avoids reward hacking and generates coherent explanations.
- **Unified scoring solving heterogeneity**: Min-max normalization + direction checking + averaging elegantly handles multi-metric issues across different benchmarks.

## Limitations & Future Work

**Limitations**:

- The data may inherit noise from PapersWithCode.
- There is no full validation of the effectiveness of this scheme in actual idea screening workflows.
- The dataset is limited to NLP; extending to other domains requires additional work.

**Additional Observations**: Synthetic reasoning chains were less effective than literature-based ones; Dr. GRPO generated coherent explanations more stably than DAPO.

## Related Work & Insights

- **vs Absolute Scoring** (Baek et al. 2025): Relative comparison is more objective and corresponds better to experimental success.
- **vs Previous Comparison Work** (Wen et al. 2025): This work is more fine-grained, with small models outperforming large ones and providing interpretable reasoning.
- **vs LLM Event Prediction** (Halawi et al. 2024): Applying event prediction to scientific idea comparison makes it more specialized.

## Rating

- Novelty: ⭐⭐⭐⭐ The comparative framework is novel, and the idea dataset is highly distinctive, though the incremental steps are focused.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Includes multiple test sets, detailed ablations, and complete robustness stress testing.
- Writing Quality: ⭐⭐⭐⭐ The paper is clear and deep; keeping technical details in the appendix is a minor drawback.
- Value: ⭐⭐⭐⭐⭐ Directly supports idea screening for autonomous research systems; its efficient small-model approach is attractive for applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Enhancing Linguistic Competence of Language Models through Pre-training with Language Learning Tasks](enhancing_linguistic_competence_of_language_models_through_pre-training_with_lan.md)
- [\[ACL 2026\] Aggregate vs. Personalized Judges in Business Idea Evaluation: Evidence from Expert Disagreement](aggregate_vs_personalized_judges_in_business_idea_evaluation_evidence_from_exper.md)
- [\[ACL 2026\] Teaching Language Models to Check Grounded Claim Factuality with Human Test-Taking Strategies](teaching_language_models_to_check_grounded_claim_factuality_with_human_test-taki.md)
- [\[ACL 2026\] Language Models Don't Know What You Want: Evaluating Personalization in Deep Research Needs Real Users](language_models_dont_know_what_you_want_evaluating_personalization_in_deep_resea.md)
- [\[ACL 2026\] Capabilities and Evaluation Biases of Large Language Models in Classical Chinese Poetry Generation: A Case Study on Tang Poetry](capabilities_and_evaluation_biases_of_large_language_models_in_classical_chinese.md)

</div>

<!-- RELATED:END -->

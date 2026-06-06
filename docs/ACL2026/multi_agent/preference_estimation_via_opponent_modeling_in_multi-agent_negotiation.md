---
title: >-
  [Paper Note] Preference Estimation via Opponent Modeling in Multi-Agent Negotiation
description: >-
  [ACL 2026][Multi-Agent][Opponent Modeling] A preference estimation method is proposed that integrates natural language preference signals extracted by LLMs with a Bayesian opponent modeling framework. By fusing qualitati…
tags:
  - "ACL 2026"
  - "Multi-Agent"
  - "Opponent Modeling"
  - "Bayesian Inference"
  - "Preference Estimation"
  - "Multi-party Negotiation"
  - "LLM Linguistic Signals"
date: 2026-05-08
content_hash: da898a1044d8e1a7
---

# Preference Estimation via Opponent Modeling in Multi-Agent Negotiation

**Conference**: ACL 2026  
**arXiv**: [2604.15687](https://arxiv.org/abs/2604.15687)  
**Code**: None  
**Area**: Video Understanding  
**Keywords**: Opponent Modeling, Bayesian Inference, Preference Estimation, Multi-party Negotiation, LLM Linguistic Signals

## TL;DR

A preference estimation method is proposed that integrates natural language preference signals extracted by LLMs with a Bayesian opponent modeling framework. By fusing qualitative cues with quantitative bidding information through a linguistic likelihood function in multi-party multi-issue negotiations, the full agreement rate is increased from 37% to 62%.

## Background & Motivation

**Background**: Automated negotiation in multi-party multi-issue scenarios relies heavily on accurate opponent modeling. Traditional methods based on the BOA architecture use Bayesian learning to estimate opponent utility functions from numerical bidding histories.

**Limitations of Prior Work**: (1) Purely numerical methods fail to capture qualitative preference information in natural language dialogues, leading to incomplete information; (2) Although LLMs can understand semantics, direct preference reasoning using LLMs lacks strategic consistency and is unstable over long negotiations; (3) The complexity of LLM reasoning grows exponentially as the volume of information increases.

**Key Challenge**: The rich qualitative information in language (e.g., "Issue A is more important to me") cannot be utilized by traditional numerical models, while LLMs lack a structured belief-updating mechanism.

**Goal**: To design a preference estimation method that integrates linguistic signals into a structured Bayesian framework, combining both semantic understanding and probabilistic reasoning.

**Key Insight**: Utilize LLMs to extract structured preference signals (target issue/option + stance) from utterances, then convert these into probabilistic likelihood functions via Luce’s Choice Axiom to be fused with bidding likelihoods for Bayesian updates.

**Core Idea**: Linguistic Likelihood $\times$ Bidding Likelihood $\to$ Bayesian Posterior Update, unifying qualitative and quantitative information within a probabilistic framework.

## Method

### Overall Architecture

In each negotiation round, the agent receives the opponent's bid $d_t$ and utterance $u_t$. An LLM is used to parse the utterance into a preference signal $z_t$. The bidding likelihood $P(d_t|h_k)$ and linguistic likelihood $P(z_t|h_k)$ are calculated separately. These are fused through a Naive Bayes assumption to update the hypothesis posterior $P(h_k|d_t, z_t)$.

### Key Designs

1.  **Linguistic Preference Signal Extraction**:
    - **Function**: Converts natural language utterances into structured preference signals.
    - **Mechanism**: Use an LLM to parse utterance $u_t$ into signal $z_t$, containing two attributes: Target (single issue/option or comparison between issues/options) and Stance (attitude such as prefer/oppose).
    - **Design Motivation**: Provides structured input for probabilistic calculations, avoiding the unreliability of direct numerical estimation outputs from LLMs.

2.  **Linguistic Likelihood based on Luce’s Choice Axiom**:
    - **Function**: Converts structured signals into a probability distribution over the hypothesis space.
    - **Mechanism**: For a signal "prefers issue $i_x$", the likelihood is $P(z_t|h_k) = w_x^{(k)} / \sum_m w_m^{(k)}$, representing the ratio of that issue's weight to the total weight. Comparison and opposition signals are handled similarly.
    - **Design Motivation**: Luce's axiom is a classic probabilistic model in choice theory, naturally transforming weights/evaluations into probabilities.

3.  **Multimodal Bayesian Fusion**:
    - **Function**: Unified updating of posterior beliefs regarding opponent preferences.
    - **Mechanism**: Assuming conditional independence between bids and linguistic signals, the posterior is $P(h_k|d_t, z_t) \propto P(d_t|h_k) \cdot P(z_t|h_k) \cdot P(h_k)$.
    - **Design Motivation**: The Naive Bayes assumption simplifies computation effectively, while bids and language indeed provide complementary information.

### Loss & Training

No model training is involved; GPT-4.1 is used as the underlying LLM. Bayesian updates are performed online.

## Key Experimental Results

### Main Results

Sports facility construction negotiation scenario with 6 parties and 5 issues (averages from 500 experiments):

| Method | FAR (Full Agreement Rate) | PAR (Partial Agreement Rate) | LAR (Latent Agreement Rate) |
| :--- | :--- | :--- | :--- |
| Base-LLM | 0.37 | 0.76 | 0.97 |
| Base-OM (all) | 0.56 | 0.92 | 0.99 |
| LLM-PE (all) | 0.32 | 0.69 | 0.93 |
| **Ours (all)** | **0.62** | **0.89** | **0.98** |

### Ablation Study

| Method | Preference Estimation MSE (Avg) | Description |
| :--- | :--- | :--- |
| **Ours** | **159** | Linguistic + Numerical Fusion |
| Base-OM | 189 | Numerical Bids Only |
| LLM-PE | 163 | Direct LLM Reasoning |

### Key Findings

- Mutual modeling (all) shows significant improvement over single-party modeling (p1) (FAR 0.46 $\to$ 0.62), indicating multi-party synergistic effects.
- Direct reasoning via LLM-PE performed worse than purely numerical methods (FAR 0.32 < 0.56), validating the necessity of a structured framework.
- The fusion of linguistic signals reduced MSE from 189 to 159, leading to more accurate and balanced estimations.

## Highlights & Insights

- The **hybrid paradigm of "LLM extraction + Bayesian inference"** is highly insightful—leveraging the semantic capabilities of LLMs without depending on their reasoning consistency, using a mathematical framework to ensure structured updates.
- **Clever application of Luce’s Choice Axiom**—mapping preference weights naturally to choice probabilities, providing a theoretical foundation for converting linguistic signals into likelihood functions.

## Limitations & Future Work

- It is assumed that the opponent's utterances are sincere, without considering deception or bluffing.
- The method was validated only in a single scenario; generalization across diverse scenarios remains to be tested.
- The hypothesis space grows factorially with the number of issues, necessitating approximation algorithms.

## Related Work & Insights

- **vs Base-LLM**: Pure LLM negotiation lacks structured preference tracking, leading to inconsistent strategies over long negotiations.
- **vs LLM-PE**: Direct numerical preference reasoning by LLMs is unreliable (FAR only 0.32) and requires the constraints of a probabilistic framework.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The integration of linguistic signals and Bayesian frameworks is a novel approach.
- **Experimental Thoroughness**: ⭐⭐⭐ Only 500 experiments in a single scenario; lacks diversity in scenarios.
- **Writing Quality**: ⭐⭐⭐⭐ Clear formalization and intuitive illustrations.
- **Value**: ⭐⭐⭐⭐ Provides a valuable paradigm for applying LLMs in structured decision-making.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Topology Matters: Measuring Memory Leakage in Multi-Agent LLMs](topology_matters_measuring_memory_leakage_in_multi-agent_llms.md)
- [\[ACL 2026\] when identity skews debate anonymization for bias-reduced multi-agent reasoning](when_identity_skews_debate_anonymization_for_bias-reduced_multi-agent_reasoning.md)
- [\[ACL 2026\] Towards Robust Real-World Spreadsheet Understanding with Multi-Agent Multi-Format Collaboration](towards_robust_real-world_spreadsheet_understanding_with_multi-agent_multi-forma.md)
- [\[ACL 2026\] Explicit Trait Inference for Multi-Agent Coordination](explicit_trait_inference_for_multi-agent_coordination.md)
- [\[ACL 2026\] Towards Self-Improving Error Diagnosis in Multi-Agent Systems](towards_self-improving_error_diagnosis_in_multi-agent_systems.md)

</div>

<!-- RELATED:END -->

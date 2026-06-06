---
title: >-
  [Paper Note] Adaptive Instruction Composition for Automated LLM Red-Teaming
description: >-
  [ACL 2026][Reinforcement Learning][LLM Red-Teaming] The Adaptive Instruction Composition (AIC) framework is proposed, utilizing Neural Thompson Sampling to adaptively select attack instructions within a combinatorial spa…
tags:
  - "ACL 2026"
  - "Reinforcement Learning"
  - "LLM Red-Teaming"
  - "Adaptive Instruction Composition"
  - "Contextual Bandits"
  - "Jailbreak Attacks"
  - "Diversity-Efficiency Trade-off"
date: 2026-05-08
content_hash: 989d22f3ca7e8ef4
---

# Adaptive Instruction Composition for Automated LLM Red-Teaming

**Conference**: ACL 2026  
**arXiv**: [2604.21159](https://arxiv.org/abs/2604.21159)  
**Code**: None  
**Area**: AI Safety / Reinforcement Learning  
**Keywords**: LLM Red-Teaming, Adaptive Instruction Composition, Contextual Bandits, Jailbreak Attacks, Diversity-Efficiency Trade-off

## TL;DR
The Adaptive Instruction Composition (AIC) framework is proposed, utilizing Neural Thompson Sampling to adaptively select attack instructions within a combinatorial space of crowdsourced harmful queries and jailbreak strategies. It simultaneously optimizes attack success rate and diversity, significantly outperforming existing methods on Harmbench.

## Background & Motivation

**Background**: Automated LLM red-teaming is a key method for improving model safety. Existing approaches generally fall into two categories: one where an attacker LLM discovers jailbreak strategies through trial-and-error (e.g., PAIR, TAP), and another that randomly combines attack instructions using crowdsourced data (e.g., WildTeaming).

**Limitations of Prior Work**: Trial-and-error methods result in limited semantic diversity of successful attacks, exploring only a narrow strategy space. Although WildTeaming utilizes a vast corpus of 50k+ harmful queries and 13k+ jailbreak strategies, its random combination approach fails to leverage historical attack results for adaptive optimization, leading to low success rates against well-defended models.

**Key Challenge**: The instruction composition space defined by WildTeaming exceeds 8 trillion possibilities ($50000 \times 13000^2$). Random search is extremely inefficient in such a massive space, yet trial-and-error methods lack systematic coverage of the known attack space. An adaptive method is required to explore diverse attacks while exploiting success signals.

**Goal**: Design an adaptive instruction composition mechanism that balances exploration and exploitation in large-scale combinatorial spaces to optimize both attack effectiveness and diversity.

**Key Insight**: Model red-teaming as a Combinatorial Neural Bandit problem, using reinforcement learning for adaptive selection within the combinatorial space of text samples.

**Core Idea**: Employ Neural Thompson Sampling as an adaptive selector. By using contrastive pre-trained sentence embeddings to map the combinatorial space to low-dimensional features, a lightweight network can achieve rapid generalization and learning across a massive space.

## Method

### Overall Architecture
The system consists of four models: an attacker LLM (generates attacks), a target LLM (the victim), an evaluator (safety judgment), and a neural bandit (adaptively selects instruction compositions). In each trial, the bandit selects the optimal combination from $K=500$ candidates, the attacker generates the prompt accordingly, and the evaluator provides a reward signal feedback to the bandit.

### Key Designs

1.  **Contrastive Embedding Featurization**:
    - **Function**: Maps text combinations to compact feature vectors as bandit inputs.
    - **Mechanism**: SBERT (all-mpnet-base-v2) maps queries and strategies to 768-dimensional embeddings, which are reduced to 10 dimensions using UMAP. Individual component embeddings are concatenated as network input. Contrastive pre-training ensures that semantically similar texts are close in the embedding space.
    - **Design Motivation**: Contrastive pre-trained embeddings allow the bandit to generalize reward signals across semantically related text groups, enabling the inference of attack success probabilities for the entire semantic region after seeing only a few samples. Ablation studies confirm that SBERT achieves faster learning and higher ASR than BERT embeddings.

2.  **Neural Thompson Sampling Selector**:
    - **Function**: Adaptively selects attack instruction compositions in each trial.
    - **Mechanism**: Maintains a two-layer feed-forward network (~2201 parameters) to calculate a Gaussian posterior reward distribution $\hat{r}_{t,k} \sim \mathcal{N}(\mu_{t,k}, \sigma^2_{t,k})$ for each candidate. The mean is derived from the network output, and the variance is calculated via the Neural Tangent Kernel. Selection via posterior sampling naturally achieves an exploration-exploitation balance.
    - **Design Motivation**: Thompson Sampling selects via posterior sampling rather than deterministic greed, ensuring high-uncertainty regions receive more exploration. The hyperparameter $\lambda$ controls variance scaling, providing interpretable control over the diversity-effectiveness trade-off.

3.  **Deduplication and Candidate Sampling Strategy**:
    - **Function**: Prevents repetitive attacks and enables scalable search in large spaces.
    - **Mechanism**: Instruction combinations of successful attacks are blacklisted, forcing the network to generalize within the feature space to maintain success. In each round, $K$ candidates are randomly sampled from the full space (a many-armed bandit approach), avoiding the need to score all 8 trillion combinations.
    - **Design Motivation**: Deduplication forces the system to continuously discover new effective regions rather than repeatedly exploiting a single successful combination, ensuring diversity.

### Loss & Training
The bandit network is trained online using $\ell_2$ regularized squared loss with a learning rate of 0.01 and weight decay that increases with the number of trials. After each trial, network parameters and the uncertainty matrix $U$ are updated using the selected combination's embedding and the evaluator reward.

## Key Experimental Results

### Main Results

| Target Model | WildTeaming ASR | AIC Subtle ASR | AIC Aggressive ASR |
| :--- | :--- | :--- | :--- |
| Mistral-7B | 0.252 | 0.363 | 0.567 |
| Llama-3-70B | 0.088 | 0.155 | 0.450 |
| Llama-3.3-70B | 0.183 | 0.247 | 0.558 |

| Harmbench Strategy | Mistral-7B ASR | Llama-3-70B ASR |
| :--- | :--- | :--- |
| GCG-T | 0.645 | 0.238 |
| PAIR | 0.525 | 0.215 |
| AutoDAN-Turbo | 0.976 | 0.672 |
| **AIC** | **1.000** | **0.934** |

### Ablation Study

| Configuration | Key Effect | Description |
| :--- | :--- | :--- |
| SBERT vs. BERT | Significant ASR increase | Contrastive embeddings support fast generalization |
| $\lambda=1$ (subtle) vs. $\lambda=0.01$ (aggr.) | Diversity $\uparrow$ vs. Success Rate $\uparrow$ | $\lambda$ provides interpretable exploration-exploitation control |
| 1 Strategy vs. 3 Strategies | Improved diversity metrics | More strategy slots enhance content diversity |

### Key Findings
- AIC achieves near-perfect ASR on Harmbench (Mistral: 1.0, Llama-3: 0.934), significantly outperforming all existing methods.
- Good cross-model transferability: Strategies trained on Mistral maintain an ASR of 0.184-0.254 when transferred to Llama-3 (WildTeaming baseline is only 0.088).
- The "subtle bandit" significantly improves success rates while maintaining diversity metrics comparable to WildTeaming.

## Highlights & Insights
- Modeling red-teaming as a combinatorial bandit problem is elegant, naturally mapping the exploration-exploitation trade-off to the diversity-effectiveness trade-off in attacks. This is transferable to any scenario requiring search in massive prompt composition spaces.
- The combination of contrastive pre-trained embeddings and a lightweight network achieves "few parameters, large generalization," where 2201 parameters effectively learn across an 8-trillion-combination space.
- The $\lambda$ hyperparameter provides an intuitive "knob" to control the diversity-effectiveness trade-off.

## Limitations & Future Work
- Experiments were limited to three open-source target models; generalization to commercial API models has not been verified.
- Dependence on Llama-Guard-2 as an evaluator may introduce false positives/negatives.
- High computational cost; 10K trials require 70-120 GPU hours.
- Future work could extend to red-teaming image generators and agents.

## Related Work & Insights
- **vs. WildTeaming**: WildTeaming uses random combinations; AIC uses RL for adaptive selection, improving ASR by 40-400%.
- **vs. PAIR/TAP**: Trial-and-error methods have limited diversity; AIC utilizes crowdsourced corpora to ensure coverage.
- **vs. AutoDAN-Turbo**: AutoDAN-Turbo discovers new strategies from scratch but has lower ASR than AIC; the two could be complementary.

## Rating
- Novelty: ⭐⭐⭐⭐ Combinatorial bandits provide a novel modeling perspective for red-teaming.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Includes multiple target models, multiple baselines, transfer experiments, ablations, and Harmbench comparisons.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and detailed algorithmic descriptions.
- Value: ⭐⭐⭐⭐ High practical value for LLM safety research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] MARS: Multi-Agent Adaptive Reasoning with Socratic Guidance for Automated Prompt Optimization](../../AAAI2026/reinforcement_learning/mars_multi-agent_adaptive_reasoning_with_socratic_guidance_f.md)
- [\[ACL 2026\] ImpRIF: Stronger Implicit Reasoning Leads to Better Complex Instruction Following](imprif_stronger_implicit_reasoning_leads_to_better_complex_instruction_following.md)
- [\[ACL 2026\] LENS: Less Noise, More Voice — Reinforcement Learning for Reasoning via Instruction Purification](less_noise_more_voice_reinforcement_learning_for_reasoning_via_instruction_purif.md)
- [\[ACL 2026\] ARGUS: Policy-Adaptive Ad Governance via Evolving Reinforcement with Adversarial Umpiring](argus_policy-adaptive_ad_governance_via_evolving_reinforcement_with_adversarial_.md)
- [\[ACL 2026\] Semantic-Space Exploration and Exploitation in RLVR for LLM Reasoning](semantic-space_exploration_and_exploitation_in_rlvr_for_llm_reasoning.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Adaptive Instruction Composition for Automated LLM Red-Teaming
description: >-
  [ACL 2026][Reinforcement Learning][LLM red-teaming] This paper proposes the Adaptive Instruction Composition (AIC) framework, which employs Neural Thompson Sampling to adaptively select attack instructions from the combinatorial space of crowdsourced harmful queries and jailbreak strategies, jointly optimizing attack success rate (ASR) and diversity. AIC achieves substantial improvements over existing methods on HarmBench.
tags:
  - ACL 2026
  - Reinforcement Learning
  - LLM red-teaming
  - adaptive instruction composition
  - contextual bandit
  - jailbreak attacks
  - diversity-effectiveness tradeoff
date: 2026-05-08
content_hash: 393ea54af596fb72
---

# Adaptive Instruction Composition for Automated LLM Red-Teaming

**Conference**: ACL 2026
**arXiv**: [2604.21159](https://arxiv.org/abs/2604.21159)
**Code**: N/A
**Area**: AI Safety / Reinforcement Learning
**Keywords**: LLM red-teaming, adaptive instruction composition, contextual bandit, jailbreak attacks, diversity-effectiveness tradeoff

## TL;DR
This paper proposes the Adaptive Instruction Composition (AIC) framework, which employs Neural Thompson Sampling to adaptively select attack instructions from the combinatorial space of crowdsourced harmful queries and jailbreak strategies, jointly optimizing attack success rate (ASR) and diversity. AIC achieves substantial improvements over existing methods on HarmBench.

## Background & Motivation

**Background**: Automated LLM red-teaming is a critical approach for improving model safety. Existing methods fall into two categories: those that employ an attacker LLM to discover jailbreak strategies via trial and error (e.g., PAIR, TAP), and those that randomly compose attack instructions from crowdsourced data (e.g., WildTeaming).

**Limitations of Prior Work**: Trial-and-error methods yield successful attacks with limited semantic diversity, exploring only a restricted portion of the strategy space. Although WildTeaming leverages a large corpus of 50,000+ harmful queries and 13,000+ jailbreak strategies, it relies on random composition and fails to exploit historical attack outcomes for adaptive optimization, resulting in low ASR against well-defended models.

**Key Challenge**: The instruction composition space defined by WildTeaming exceeds 8 trillion possible combinations ($50000 \times 13000^2$), making random search highly inefficient in such a vast space. Trial-and-error methods, on the other hand, lack systematic coverage of the known attack space. A method is needed that can both explore diverse attacks and exploit success signals adaptively.

**Goal**: To design an adaptive instruction composition mechanism that balances exploration and exploitation in a large-scale combinatorial space, simultaneously optimizing attack effectiveness and diversity.

**Key Insight**: Red-teaming is formulated as a Combinatorial Neural Bandit problem, leveraging reinforcement learning to make adaptive selections over the combinatorial space of text samples.

**Core Idea**: Neural Thompson Sampling serves as the adaptive selector. Contrastively pre-trained sentence embeddings map the combinatorial space to low-dimensional features, enabling a lightweight network to generalize and learn rapidly across the vast search space.

## Method

### Overall Architecture
The system comprises four components: an attacker LLM (generating attacks), a target LLM (the attack subject), an evaluator (safety judge), and a neural bandit (adaptively selecting instruction compositions). At each trial, the bandit selects the optimal composition from $K=500$ candidate instruction combinations; the attacker generates an attack accordingly, and the evaluator provides a reward signal that is fed back to the bandit.

### Key Designs

1. **Contrastive Embedding Featurization**:

    - Function: Maps text combinations to compact feature vectors as bandit inputs.
    - Mechanism: SBERT (all-mpnet-base-v2) encodes queries and strategies into 768-dimensional embeddings, which are then reduced to 10 dimensions via UMAP; component embeddings are concatenated as network inputs. Contrastive pre-training ensures semantically similar texts are proximate in embedding space.
    - Design Motivation: Contrastively pre-trained embeddings enable the bandit to generalize reward signals across semantically related text groups, allowing inference of attack success probability across entire semantic regions from only a few observed samples. Ablation studies confirm that SBERT embeddings yield faster learning and higher ASR than BERT embeddings.

2. **Neural Thompson Sampling Selector**:

    - Function: Adaptively selects attack instruction compositions at each trial.
    - Mechanism: A two-layer feedforward network (~2,201 parameters) is maintained to compute a Gaussian posterior reward distribution $\hat{r}_{t,k} \sim \mathcal{N}(\mu_{t,k}, \sigma^2_{t,k})$ for each candidate composition, where the mean is the network output and the variance is computed via the neural tangent kernel. Compositions are selected by posterior sampling, naturally achieving exploration–exploitation balance.
    - Design Motivation: Thompson Sampling selects via posterior sampling rather than deterministic greedy decisions, automatically allocating more exploration to high-uncertainty regions. The hyperparameter $\lambda$ controls variance scaling, providing an interpretable knob for the diversity–effectiveness tradeoff.

3. **Deduplication and Candidate Sampling Strategy**:

    - Function: Prevents repetitive attacks and enables scalable search over the large composition space.
    - Mechanism: Instruction compositions that yield successful attacks are blacklisted, forcing the network to generalize in feature space to sustain success. At each trial, $K$ candidates are randomly sampled from the full space (many-armed bandit approach), obviating the need to score all 8 trillion combinations.
    - Design Motivation: Deduplication compels the system to continuously discover new effective regions rather than repeatedly exploiting the same successful composition, thereby ensuring diversity.

### Loss & Training
The bandit network is trained online with $\ell_2$-regularized squared loss, using a learning rate of 0.01 and weight decay that increases with the number of trials. After each trial, network parameters and the uncertainty matrix $U$ are updated using the embedding of the selected composition and the evaluator reward.

## Key Experimental Results

### Main Results

| Target Model | WildTeaming ASR | AIC Subtle ASR | AIC Aggressive ASR |
|---|---|---|---|
| Mistral-7B | 0.252 | 0.363 | 0.567 |
| Llama-3-70B | 0.088 | 0.155 | 0.450 |
| Llama-3.3-70B | 0.183 | 0.247 | 0.558 |

| HarmBench Method | Mistral-7B ASR | Llama-3-70B ASR |
|---|---|---|
| GCG-T | 0.645 | 0.238 |
| PAIR | 0.525 | 0.215 |
| AutoDAN-Turbo | 0.976 | 0.672 |
| **AIC** | **1.000** | **0.934** |

### Ablation Study

| Configuration | Key Effect | Notes |
|---|---|---|
| SBERT vs. BERT embeddings | Significant ASR improvement | Contrastive pre-training supports rapid generalization |
| $\lambda=1$ (subtle) vs. $\lambda=0.01$ (aggressive) | Diversity↑ vs. ASR↑ | $\lambda$ provides interpretable exploration–exploitation control |
| 1 strategy vs. 3 strategies | Improved diversity metrics | More strategy slots enhance content diversity |

### Key Findings
- AIC achieves near-perfect ASR on HarmBench (Mistral: 1.0; Llama-3: 0.934), substantially outperforming all existing methods.
- Strong cross-model transferability: strategies trained on Mistral transfer to Llama-3 with ASR of 0.184–0.254, compared to only 0.088 for the WildTeaming baseline.
- The subtle bandit significantly improves ASR while maintaining diversity metrics comparable to WildTeaming.

## Highlights & Insights
- Formulating red-teaming as a combinatorial bandit problem is an elegant modeling choice that naturally maps the exploration–exploitation tradeoff onto the attack diversity–effectiveness tradeoff. This formulation is transferable to any scenario requiring search over large-scale prompt composition spaces.
- Contrastive pre-trained embeddings combined with a lightweight network achieve strong generalization with minimal parameters: only 2,201 parameters suffice for effective learning across an 8-trillion-entry space.
- The $\lambda$ hyperparameter provides an intuitive and interpretable control knob for the diversity–effectiveness tradeoff.

## Limitations & Future Work
- Experiments are conducted on only three open-source target models; generalization to commercial API-based models remains unvalidated.
- Reliance on Llama-Guard-2 as the evaluator may introduce false positives and false negatives.
- Computational cost is substantial, requiring 70–120 GPU hours for 10K trials.
- Future work may extend the framework to red-teaming of image generators and autonomous agents.

## Related Work & Insights
- **vs. WildTeaming**: WildTeaming relies on random composition; AIC employs RL-based adaptive selection, achieving 40–400% improvement in ASR.
- **vs. PAIR/TAP**: Trial-and-error methods are limited in diversity; AIC leverages crowdsourced corpora to ensure broad coverage.
- **vs. AutoDAN-Turbo**: AutoDAN-Turbo can discover novel strategies from scratch but achieves lower ASR than AIC; the two approaches are complementary.

## Rating
- Novelty: ⭐⭐⭐⭐ Framing red-teaming as a combinatorial bandit problem is a novel and principled modeling perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across multiple target models, baselines, transfer experiments, ablations, and HarmBench comparisons.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with detailed algorithmic descriptions.
- Value: ⭐⭐⭐⭐ Substantial practical value for LLM safety research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] MARS: Multi-Agent Adaptive Reasoning with Socratic Guidance for Automated Prompt Optimization](../../AAAI2026/reinforcement_learning/mars_multi-agent_adaptive_reasoning_with_socratic_guidance_f.md)
- [\[ACL 2026\] ImpRIF: Stronger Implicit Reasoning Leads to Better Complex Instruction Following](imprif_stronger_implicit_reasoning_leads_to_better_complex_instruction_following.md)
- [\[ACL 2026\] LENS: Less Noise, More Voice — Reinforcement Learning for Reasoning via Instruction Purification](less_noise_more_voice_reinforcement_learning_for_reasoning_via_instruction_purif.md)
- [\[ACL 2026\] Feedback-Driven Tool-Use Improvements in Large Language Models via Automated Build Environments](feedback-driven_tool-use_improvements_in_large_language_models_via_automated_bui.md)
- [\[ACL 2026\] Semantic-Space Exploration and Exploitation in RLVR for LLM Reasoning](semantic-space_exploration_and_exploitation_in_rlvr_for_llm_reasoning.md)

</div>

<!-- RELATED:END -->

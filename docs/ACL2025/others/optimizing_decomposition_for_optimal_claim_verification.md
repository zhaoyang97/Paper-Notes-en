---
title: >-
  [Paper Note] Optimizing Decomposition for Optimal Claim Verification
description: >-
  [ACL 2025][Decompose-Then-Verify] Proposes the Dynamic Decomposition framework, which learns decomposition strategies from verifier feedback via reinforcement learning to decompose claims into the atomic granularity preferred by the verifier, thereby bridging the performance gap between decomposers and verifiers.
tags:
  - "ACL 2025"
  - "Decompose-Then-Verify"
  - "atomicity"
  - "reinforcement learning"
  - "PPO"
  - "fact-checking"
date: 2026-05-08
content_hash: c565e970f2cb2d93
---

# Optimizing Decomposition for Optimal Claim Verification

**Conference**: ACL 2025  
**arXiv**: [2503.15354](https://arxiv.org/abs/2503.15354)  
**Code**: [github.com/yining610/dynamic-decomposition](https://github.com/yining610/dynamic-decomposition)  
**Area**: Other  
**Keywords**: Decompose-Then-Verify, atomicity, reinforcement learning, PPO, fact-checking  

## TL;DR

Proposes the Dynamic Decomposition framework, which learns decomposition strategies from verifier feedback via reinforcement learning to decompose claims into the atomic granularity preferred by the verifier, thereby bridging the performance gap between decomposers and verifiers.

## Background & Motivation

### Background
The Decompose-Then-Verify paradigm is the mainstream approach in active fact-checking systems: a decomposer first splits complex claims into subclaims, and subsequently, a verifier validates them one by one. However, existing works usually treat decomposition and verification as independent modules, neglecting their interaction and potential misalignment.

### Key Findings & Motivation
- The authors introduce an "atomicity" metric to quantify the information density of subclaims, defined as $\text{atomicity} = \log_2(\text{\# atomic information})$
- Experiments show that **different verifiers achieve optimal verification confidence at different levels of atomicity**, meaning each verifier has its own preferred input granularity.
- Existing prompt-based decomposition strategies (such as FActScore using 8 annotated demos) fail to generate subclaims with optimal atomicity, leading to suboptimal verification results.
- For instance, the decomposition strategy of FActScore generates subclaims with atomicity=0, whereas Inst-Llama-7B paired with a search strategy performs best at atomicity=1.

### Core Problem
This is a **bilevel optimization problem**: the upper level optimizes verification accuracy, and the lower level optimizes the decomposition strategy. This problem is strongly NP-hard.

## Method

### Overall Architecture
Dynamic decomposition is modeled as a finite MDP (Markov Decision Process), utilizing PPO-style (Proximal Policy Optimization) A2C reinforcement learning to approximate the solution of the bilevel optimization problem.

**Key difference from existing methods**: While existing methods only call the decomposition prompt once, dynamic decomposition **iteratively** triggers decomposition calls, where a policy decides at each step whether to continue decomposing the current subclaim.

### MDP Definition $M = (\mathcal{S}, \mathcal{A}, \kappa, r)$

**State (Atomicity State)**:
- Each state is a $d$-dimensional vector $s_t \in \mathbb{R}^d$, reflecting the global atomicity of the current list of subclaims.
- State transitions are modeled using a GRU: $s_{t+1} = \text{GRU}[s_t, (1+\sigma(\Delta\text{Info}))\text{Enc}(\{c_j\})]$
- $\Delta\text{Info}$ quantifies the local atomicity changes resulting from decomposition via conditional pointwise mutual information (CPMI) differences.

**Action**:
- Binary action space: 1 (decompose) or 0 (do not decompose).
- Sampled from the policy distribution $a_t \sim \pi_d(a_t|s_t)$.

**Reward**:
- Defines **verification confidence** as $\text{Conf}(c, \mathcal{V}, \pi_v) = |P_{\mathcal{V}}(\text{True}|c, \pi_v) - P_{\mathcal{V}}(\text{False}|c, \pi_v)|$
- Reward = Average confidence of subclaims after decomposition - Confidence of the original claim before decomposition.
- Experiments verify that confidence is highly correlated with accuracy (Pearson's $r = 0.88$).

### Key Designs

**Breadth-First Decomposition Order**:
- A breadth-first strategy is adopted to prioritize the decomposition of subclaims with higher atomicity.
- Avoids the issue of excessive atomicity variance caused by a depth-first strategy.
- Newly generated subclaims are queued in a FIFO manner.

**Binary Decomposition**: Each step performs a binary decomposition of the subclaim, consistent with the definition of $\log_2$ atomicity, ensuring maximum exploration of the subclaim space.

### Loss & Training

$$L^{\text{PPO}} = \mathbb{E}_t\left[L^{\text{clip}} - c_1 \hat{A}_t^2 + c_2 S[\pi_d](s_t)\right]$$

- The advantage function is calculated using the Generalized Advantage Estimator (GAE).
- Both the policy network and the value network are two-layer MLPs.
- The total trainable parameters are only **4.73M**.
- An entropy bonus term is introduced to promote exploration.

## Experiments

### Experimental Setup
- **Dataset**: Constructed based on FActScore, containing claims from two sources (ChatGPT and PerplexityAI), with each claim having 6 levels of atomicity (-1 to 4).
- **Decomposer LLM**: Llama3-Inst-70B, DeepSeek-V3
- **Verifier LLM**: FT-T5-3B, Inst-Llama-7B, Llama3-Inst-8B
- **Verification Strategy**: Retrieval / In-Context Example / No-Context
- **Baselines**: FActScore, FActScore-Atom, WICE, R-ND

### Main Results

| Metric | DyDecomp Gain |
|------|-------------|
| Verification Confidence (Average) | **+0.07** (atomicity 1-2) |
| Verification Accuracy (PerplexityAI, Average) | **+0.12** (atomicity 1-2) |

- DyDecomp **consistently** achieves the highest verification confidence at atomicity=1 and atomicity=2.
- At atomicity=0 (which is already close to the optimal level for the verifier), DyDecomp does not necessarily outperform the baselines, which aligns with expectations.

### Ablation Study (Evaluated on atomicity=4)

| Variant | Verification Confidence |
|------|-----------|
| DyDecomp (Full) | 0.446 |
| - Single-layer NN | 0.398 (-0.048) |
| - Ternary decomposition instead of binary | 0.424 (-0.022) |
| - Remove entropy bonus | 0.356 (-0.090) |
| - Remove atomicity 1 training data | 0.353 (-0.093) |
| - Remove atomicity 1, 2, 3 training data | 0.401 (-0.045) |

### Key Findings
1. The entropy bonus term has the largest impact on performance (-0.090), indicating that exploring diverse decomposition trajectories is crucial.
2. Cross-atomicity training data is highly important for generalization ability.
3. Improvements in confidence do not always translate to improvements in accuracy — this depends on the capacity of the verifier (the "weakest-link effect").
4. On the PerplexityAI dataset, confidence and accuracy improve simultaneously, whereas on the ChatGPT dataset, only confidence improves.

## Highlights & Insights

1. **Formalization of the Atomicity Concept**: First to systematically quantify the impact of subclaim granularity on verification using "atomicity", revealing that different verifiers have distinct preferences for optimal granularity.
2. **Extremely Lightweight Policy**: Requires only 4.73M parameters to significantly improve verification performance, without the need to modify the decomposer LLM or verifier LLM.
3. **Innovative Modeling of Bilevel Optimization + RL**: Models the decomposition-verification problem as a bilevel optimization task and approximates the solution using PPO.
4. **Universal Compatibility**: The framework is compatible with any existing fact-checking systems, while keeping both the decomposer and verifier LLMs frozen.
5. **Strong Correlation between Confidence and Accuracy** ($r=0.88$) provides a viable proxy signal for policy optimization in unsupervised (unlabeled) scenarios.

## Limitations & Future Work

1. Only focuses on the single decomposition feature of information density (atomicity), without considering other characteristics of subclaims such as verifiability and self-containment.
2. The reward design depends on verification confidence rather than accuracy, introducing the risk of over-optimizing confidence without improving accuracy.
3. The capability of the verifier becomes the system bottleneck — weaker verifiers limit the ceiling of overall performance.

## Related Work & Insights

- **Decomposition Strategies**: Static prompt methods such as FActScore (Min et al., 2023) (precision-oriented), WICE (Kamoi et al., 2023) (coverage-oriented), and R-ND (Wanner et al., 2024).
- **Verification Strategies**: Retrieving evidence, constructing in-context examples, zero-shot prompting, etc.
- **RL in NLP**: Optimizing ICL example selection with RL, jointly training reward and policy models using PPO, etc.

## Rating

⭐⭐⭐⭐ — Clear problem definition, elegant method (lightweight RL policy + frozen LLMs), and thorough experiments across multiple verifiers, datasets, and atomicity levels. The limitation lies in focusing solely on the single dimension of atomicity, and the incomplete alignment between confidence and accuracy is worth further exploring.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Entropy-UID: A Method for Optimizing Information Density](entropy-uid_a_method_for_optimizing_information_density.md)
- [\[ACL 2025\] RePanda: Pandas-powered Tabular Verification and Reasoning](repanda_pandas-powered_tabular_verification_and_reasoning.md)
- [\[ACL 2025\] Adaptive Feature-based Low Rank Plus Sparse Decomposition for Subspace Clustering](adaptive_feature-based_low_rank_plus_sparse_decomposition_for_subspace_clusterin.md)
- [\[ACL 2025\] Inter-Passage Verification for Multi-evidence Multi-answer QA](inter-passage_verification_for_multi-evidence_multi-answer_qa.md)
- [\[ACL 2025\] LegalReasoner: Step-wised Verification-Correction for Legal Judgment Reasoning](legalreasoner_step-wised_verification-correction_for_legal_judgment_reasoning.md)

</div>

<!-- RELATED:END -->

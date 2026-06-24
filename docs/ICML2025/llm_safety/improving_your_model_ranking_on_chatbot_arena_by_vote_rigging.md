---
title: >-
  [Paper Note] Improving Your Model Ranking on Chatbot Arena by Vote Rigging
description: >-
  [ICML 2025][LLM Safety][Chatbot Arena] The paper reveals that Chatbot Arena's crowdsourced voting mechanism can be maliciously manipulated. It proposes two types of vote rigging strategies: target-only and omnipresent. Notably, the omnipresent strategy exploits the global coupling characteristic of the Bradley-Terry rating system, allowing an attacker to elevate a target model's ranking by 15 places with only hundreds of manipulated votes, thereby highlighting the security vu…
tags:
  - "ICML 2025"
  - "LLM Safety"
  - "Chatbot Arena"
  - "Vote Rigging"
  - "Elo Rating"
  - "Bradley-Terry"
  - "Leaderboard Manipulation"
  - "LLM Evaluation Security"
date: 2026-05-08
content_hash: c0558f61789c26da
---

# Improving Your Model Ranking on Chatbot Arena by Vote Rigging

**Conference**: ICML 2025  
**arXiv**: [2501.17858](https://arxiv.org/abs/2501.17858)  
**Area**: ai_safety (LLM evaluation safety / leaderboard manipulation)  
**Keywords**: Chatbot Arena, Vote Rigging, Elo Rating, Bradley-Terry, Leaderboard Manipulation, LLM Evaluation Security

## TL;DR
The paper reveals that Chatbot Arena's crowdsourced voting mechanism can be maliciously manipulated. It proposes two types of vote rigging strategies: target-only and omnipresent. Notably, the omnipresent strategy exploits the global coupling characteristic of the Bradley-Terry rating system, allowing an attacker to elevate a target model's ranking by 15 places with only hundreds of manipulated votes, thereby highlighting the security vulnerabilities of current LLM evaluation platforms.

## Background & Motivation

### 1. Background
Chatbot Arena is currently one of the most influential LLM evaluation platforms, employing a crowdsourced pairwise battle mode:
1. Users submit prompts, and two anonymous models generate responses respectively.
2. Users vote to choose the better response (or select a tie).
3. The platform computes Elo/Bradley-Terry ratings based on all votes to establish a public leaderboard.
4. Leaderboard results are widely cited and exert a significant impact on the commercial promotion of models.

### 2. Existing Defense Measures
Chatbot Arena has deployed multi-layer defense mechanisms:

**Length/style bias control**: Eliminates surface feature biases in user preferences (Dubois et al., 2024; Li et al., 2024a).

**Anomalous vote detection**: Identifies bot behaviors and statistically anomalous voting patterns (Chiang et al., 2024).

**Prompt classification and data cleaning**: Categorizes and manages prompts (Li et al., 2024b, c).

**Identity leakage filtering**: If a response exposes the identity of the model, the corresponding vote is discarded (Chiang et al., 2024).

### 3. Key Challenge
The conflict lies in:
1. Leaderboards require open crowdsourced voting to ensure coverage and diversity.
2. However, open voting implies that malicious actors can inject fraudulent votes.
3. Existing defenses primarily target unintentional biases and simple bots, failing to consider carefully designed strategic manipulation.

### 4. Goal
The authors address a key security question:
1. Can Chatbot Arena's voting mechanism be strategically manipulated to alter a target model's ranking?
2. How low is the cost of manipulation? (How many malicious votes are required?)
3. Can existing defense mechanisms counter this effectively?

### 5. Key Insight
The paper begins with the mathematical structure of the Elo/Bradley-Terry rating system, discovering that the global coupling of rating computation can be exploited—even if the manipulated votes do not involve the target model, they can still indirectly affect its ranking.

### 6. Core Idea
By leveraging the global influence of all votes on all model ratings under the Bradley-Terry model, the authors design an omnipresent rigging strategy to vote strategically in arbitrary battles, thereby indirectly boosting the target model's ranking.

## Method

### Overall Architecture
The paper proposes two major categories of attack strategies, establishing a progressive threat analysis:

**Attack Assumptions**: The attacker can initiate battles and vote on Chatbot Arena, but cannot control which models are matched.

**Workflow**:
1. The attacker determines the target model $m_t$.
2. In a new battle, the attacker identifies the identities of the competing models.
3. The attacker determines the voting direction (which model to vote for) based on the chosen strategy.
4. The attacker accumulates a sufficient number of manipulated votes to change $m_t$'s ranking.

### Strategy One: Target-Only Rigging

**Function**: Focuses voting manipulation solely on battles that involve the target model $m_t$.

**Mechanism**:
1. In each battle, the attacker first identifies whether $m_t$ is participating.
2. Identification methods include:
    - **Watermarking** (Zhao et al., 2024): If $m_t$'s outputs are embedded with watermarks, its identity can be determined by watermark detection.
    - **Binary Classifier** (Huang et al., 2025): A classifier is trained to distinguish the output of $m_t$ from other models.
3. Once the participation of $m_t$ is confirmed, the attacker invariably votes for $m_t$ to win.

**Practical Efficiency Issues**:
- There are over 190 models on Chatbot Arena.
- The probability of $m_t$ participating in any given battle is only around 1%.
- Improving a ranking position might require over 10,000 manipulated votes.
- This high volume of voting is easily prevented by simple defenses like IP rate limiting.

### Strategy Two: Omnipresent Rigging

**Function**: Votes strategically in any battle (even if it does not involve $m_t$) to indirectly improve $m_t$'s ranking.

**Mechanism**:
The paper exploits a key mathematical property of the Bradley-Terry model:

$$P(m_i \succ m_j) = \frac{\exp(\beta_i)}{\exp(\beta_i) + \exp(\beta_j)}$$

where $\beta_i$ is the BT score of model $m_i$. The $\beta$ of all models are jointly solved via maximum likelihood estimation over all votes:

$$\hat{\boldsymbol{\beta}} = \arg\max_{\boldsymbol{\beta}} \sum_{k=1}^{N} \log P(m_{w_k} \succ m_{l_k} \mid \boldsymbol{\beta})$$

**Key Insights**: Since all $\beta_i$ are jointly optimized, shifting the vote in any battle changes the optimal solution of the global likelihood function, thereby affecting the ratings of all models (including $m_t$).

**Strategy Design**:
1. The attacker does not need to wait for $m_t$ to appear in battles.
2. In every battle, the attacker identifies the identities of both competing models.
3. The attacker chooses the optimal vote based on the indirect impact of the voting direction on $m_t$'s ranking.
4. Specifically, votes should favor:
    - Making models ranked lower than $m_t$ win (suppressing $m_t$'s competitors).
    - Making models ranked higher than $m_t$ lose (dragging down models above $m_t$).

**Efficiency Advantages**:
- Every battle represents an attack opportunity, boosting utilization from ~1% to ~100%.
- Significant rank improvements can be achieved with only hundreds of votes.

### Model Identity Recognition

The prerequisite of the attack is to identify the identities of competing models in anonymous battles. The paper discusses two approaches:

1. **Watermarking**: Model providers can embed detectable watermarks in their outputs. Attackers (such as the model provider itself) can leverage known watermarks to identify their own model.
2. **Binary Classifier**: The attacker collects output samples from the target model and other models to train a classifier. Running the classifier on a new response determines whether it belongs to the target model.

### Theoretical Analysis

The core theoretical contribution of this work is proving the global coupling property of the BT rating system:

1. **Propagation of Vote Influence**: In the BT framework, adding a single vote $(m_i \succ m_j)$ alters the solution of the maximum likelihood estimation, thereby affecting the $\beta$ values of all models.
2. **Rank Sensitivity**: Votes on battles involving models with ranks close to $m_t$'s rank exert the greatest influence on $m_t$'s ranking.
3. **Asymmetric Effect**: The manipulation strategy can simultaneously elevate $m_t$'s rating and degrade its competitors' ratings, resulting in a "double effect".

## Key Experimental Results

### Experimental Setup
- **Dataset**: Approximately 1.7 million historical voting records from the Chatbot Arena Notebook.
- **Target Model**: Phi-3-small-8k-Instruct (used as the demonstration target).
- **Evaluation Metrics**: Rank improvement vs. number of manipulated votes.
- **Simulation Method**: Injecting new manipulated votes on top of historical data and recomputing the BT ratings.

### Main Results

| Strategy | Manipulated Votes | Rank Improvement | Per-vote Efficiency (rank/vote) | Practical Feasibility |
|---|---|---|---|---|
| No Rigging (adding ~27,000 normal votes) | 0 | +1 place | — | Baseline |
| Target-Only Rigging | Thousands of votes | Slight improvement | Low | Low (only ~1% battles involve the target model) |
| Omnipresent Rigging | Hundreds of votes | +15 places | High | High (every battle can be exploited) |

### Strategy Efficiency and Defense Analysis

| Defense Mechanism | Can Block Target-Only | Can Block Omnipresent | Reason Analysis |
|---|---|---|---|
| IP Rate Limiting (daily voting cap) | Partially effective | Limited effectiveness | Omnipresent requires far fewer votes than the threshold |
| Anomalous Vote Pattern Detection | Detectable (concentrated voting) | Difficult to detect | Omnipresent votes are dispersed across different battles |
| Enhanced Model Identity Hiding | Increases identification difficulty | Increases identification difficulty | Both strategies rely heavily on identity identification |
| Vote Consistency Validation | Partially effective | Limited effectiveness | A single vote in Omnipresent appears normal |

### Key Findings
1. **Disparity in Efficiency**: The Omnipresent strategy is one to two orders of magnitude more efficient than Target-Only.
2. **Higher Covertness**: Omnipresent manipulation votes are dispersed across various battles. A single vote exhibits no anomalous behavior, rendering detection via pattern matching highly challenging.
3. **Vulnerability of Rating Systems**: A ranking system built on approximately 1.7 million votes can be significantly perturbed with merely hundreds of votes, exposing the vulnerability of the BT rating system in adversarial scenarios.
4. **Defense Difficulty**: The authors evaluate multiple defense methods and conclude that there is no silver bullet, necessitating continuous iteration of multi-layered defenses.

## Highlights & Insights

### Highlight 1: Revealing Fundamental Vulnerabilities of the Evaluation System
Rather than finding a specific bug or implementation flaw, the paper points out that the mathematical structure of the Bradley-Terry rating mechanism itself allows global manipulation. This means the problem cannot be solved with a simple patch and demands a fundamental redesign of evaluation systems.

### Highlight 2: Ingenious Design of the Omnipresent Strategy
By exploiting the seemingly intuitive but underappreciated property that "every vote influences the global system", the authors transform an inefficient target-only attack into a highly efficient global attack. The attack utilization rate leaps from ~1% to ~100%, representing a classic "breakthrough achieved by a shift in perspective".

### Highlight 3: Comprehensive Perspective on Attack and Defense Game
The paper does not merely propose attacks but also evaluates the effectiveness and limitations of various defenses, providing practical directions for security improvements to the platform owners.

### Transferable Insights
1. **Security of Other Leaderboards**: Similar mechanisms can be applied to analyze manipulation risks on platforms such as HELM and Open LLM Leaderboard.
2. **Recommendation System Security**: The global coupling issue in collaborative filtering ratings is highly analogous to the findings in this paper.
3. **Poisoning Attacks in Federated Learning**: The concept of "indirectly influencing the global model" is transferable to federated aggregation scenarios.
4. **Security in Decentralized Voting**: It offers reference value for blockchain voting scenarios like DAO governance.

## Limitations & Future Work

### Limitations Identified by Authors
1. **Strong Assumptions on Model Identity Identification**: The paper assumes that an attacker can reliably identify model identities in anonymous battles. In practice, this step may be difficult, especially among models of similar capability levels.
2. **Static Ranking Simulation**: Experiments are based on offline simulation using data history and do not consider real-time factors such as dynamic adjustment of sampling strategies or update frequencies by the platform.
3. **Single Rating Mechanism**: The paper primarily analyzes the standard BT model, whereas Chatbot Arena may employ variants including regularization and category-specific ratings in practice.

### Additional Limitations Identified by Readers
1. **Lack of Real-Platform Validation**: Due to ethical considerations, the attacks were not executed on the live Chatbot Arena platform. Simulation experiments might underestimate the platform's actual defense capabilities.
2. **Incomplete Cost Model**: The paper focuses on the number of manipulated votes but lacks a detailed analysis of actual operational costs, such as acquiring Chatbot Arena accounts and evading detection.
3. **Multi-Attacker Scenarios Unconsidered**: How does the game equilibrium change if multiple model providers conduct manipulation simultaneously?
4. **Evolution of Rating Mechanisms**: The rating methodologies of Chatbot Arena are constantly updated; the effectiveness of the proposed attack strategies on the latest versions requires re-validation.

### Future Directions for Improvement
1. Design cryptographic commitment-based voting protocols to prevent votes from being strategically chosen before statistics are finalized.
2. Introduce differential privacy mechanisms to add noise to rating calculations, enhancing robustness.
3. Develop anomaly detection methods based on vote timing analysis to identify strategic voting patterns.
4. Explore alternative non-BT ranking mechanisms (such as TrueSkill variants) and evaluate their robustness against manipulation.

## Related Work & Insights

### vs Traditional Elo Rigging Research
Discussions on Elo rigging exist in fields like chess, but typically involve direct "sandbagging". The contribution of this paper lies in extending the problem to large-scale crowdsourcing scenarios with numerous models and proposing indirect manipulation strategies that do not require the target model's participation.

### vs LLM Evaluation Reliability Research
Dubois et al. (2024) and Li et al. (2024a) focus on unintentional biases (such as length bias), which are "noise" issues. This paper is concerned with intentional manipulation (adversarial), which constitutes a higher-level security threat.

### vs Benchmark Rigging
Traditional benchmark manipulation (such as overfitting test sets) requires modifying the model itself. The attack in this paper does not require modifying the target model, only injecting malicious votes on the client side, which presents a broader attack surface.

### Insights for Current Research
1. LLM evaluation systems need to upgrade from "noise prevention" to "adversarial defense".
2. The governance framework of crowdsourced evaluation should borrow offensive and defensive thinking from the security domain.
3. Leaderboard design must consider incentive compatibility from a game-theoretic perspective.

## Key Points for Replication
1. Download historical voting data (~1.7 million records) from the Chatbot Arena Notebook.
2. Implement the standard BT rating calculation as a baseline.
3. Implement the model identity recognition module (watermark detection / binary classifier).
4. Simulate the injection of manipulated votes on top of historical data, and observe the rank change curves.
5. Key hyperparameters: number of manipulated votes, target model selection, voting strategy.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (5/5) First to systematically reveal the feasibility of vote rigging in Chatbot Arena, featuring an ingenious design of the omnipresent strategy.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ (4/5) Simulation experiments based on 1.7 million historical votes are large in scale, though lacking validation on the live platform.
- Writing Quality: ⭐⭐⭐⭐☆ (4/5) Problem motivation is clear, attack strategies are presented progressively, and defense discussions are comprehensive.
- Value: ⭐⭐⭐⭐⭐ (5/5) Holds significant warning significance for LLM evaluation platform security and can drive the upgrading of defense systems for evaluation platforms.

## Citation Information
Rui Min, Tianyu Pang, Chao Du, Qian Liu, Minhao Cheng, Min Lin. **Improving Your Model Ranking on Chatbot Arena by Vote Rigging**. ICML 2025.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Is Your Model Fairly Certain? Uncertainty-Aware Fairness Evaluation for LLMs](is_your_model_fairly_certain_uncertainty-aware_fairness_evaluation_for_llms.md)
- [\[ACL 2025\] Improving Model Factuality with Fine-grained Critique-based Evaluator](../../ACL2025/llm_safety/improving_model_factuality_with_fine-grained_critique-based_evaluator.md)
- [\[ICML 2025\] Improving LLM Safety Alignment with Dual-Objective Optimization](improving_llm_safety_alignment_with_dual-objective_optimization.md)
- [\[ICML 2025\] Improving Continual Learning Performance and Efficiency with Auxiliary Classifiers](improving_continual_learning_performance_and_efficiency_with_auxiliary_classifie.md)
- [\[NeurIPS 2025\] Music Arena: Live Evaluation for Text-to-Music](../../NeurIPS2025/llm_safety/music_arena_live_evaluation_for_text-to-music.md)

</div>

<!-- RELATED:END -->

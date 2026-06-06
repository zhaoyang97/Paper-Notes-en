---
title: >-
  [Paper Note] T-POP: Test-Time Personalization with Online Preference Feedback
description: >-
  [ICML 2026][Recommender Systems][Test-time Alignment] T-POP combines "test-time alignment" with "neural dueling bandits" to solve the cold-start personalization problem for new users. Without modifying LLM parameters…
tags:
  - "ICML 2026"
  - "Recommender Systems"
  - "Test-time Alignment"
  - "Dueling Bandits"
  - "Online Preference Feedback"
  - "Cold-start Personalization"
  - "Neural UCB"
date: 2026-05-08
content_hash: abf81793f81a19aa
---

# T-POP: Test-Time Personalization with Online Preference Feedback

**Conference**: ICML 2026  
**arXiv**: [2509.24696](https://arxiv.org/abs/2509.24696)  
**Code**: https://github.com/QuZikun/T-POP (Available)  
**Area**: LLM Alignment / Personalization / Online Learning  
**Keywords**: Test-time Alignment, Dueling Bandits, Online Preference Feedback, Cold-start Personalization, Neural UCB  

## TL;DR
T-POP combines "test-time alignment" with "neural dueling bandits" to solve the cold-start personalization problem for new users. Without modifying LLM parameters, it learns a personalized reward function online using pairwise preference feedback from each interaction round.

## Background & Motivation

**Background**: LLM personalization currently follows two main paths. The first is RLHF/DPO-style fine-tuning, which involves retraining or LoRA adaptation based on each user's preferences. The second is a training-free path, such as RAG-based retrieval of user history or injecting history into the prompt. Both paths assume the availability of "sufficient" existing data for a user.

**Limitations of Prior Work**: The fine-tuning route is too slow and expensive for new users—running RLHF for every individual user is impractical. While the RAG/prompt route is lightweight, it essentially "reads history" and fails for new users with no history, which is the classic personalization cold-start problem.

**Key Challenge**: User preferences are only revealed through interaction, yet the generation must be decent during the interaction process; otherwise, users will not stay to provide feedback. In other words, **"preference collection" and "preference utilization" occur simultaneously and cannot be decoupled into separate "collection then deployment" phases**.

**Goal**: (1) Enable online adaptation to individual users without fine-tuning the base LLM; (2) Drive the process with the cheapest form of feedback—pairwise preferences; (3) Achieve high sample efficiency, showing gains within dozens of interactions.

**Key Insight**: The authors treat each decoding step as a bandit problem—the candidate token pool consists of "arms," and the user's pairwise preference serves as the reward signal. To learn the reward online, the system must actively generate "informative" pairs to query the user, which is exactly where dueling bandits excel at "explore vs exploit."

**Core Idea**: Attach an online-learning neural reward function $r(\cdot;\theta)$ on top of a frozen LLM. For each token generation, two candidate responses are generated simultaneously—one pure exploit and one with a UCB exploration bonus. A pairwise comparison is presented to the user, and the BTL loss is immediately used to update $\theta$.

## Method

### Overall Architecture
At round $t$, T-POP receives a query $q_t$ and expands two trajectories $y_{t,1}$ and $y_{t,2}$ of maximum length $M$ in parallel, referred to as the exploitation arm and exploration arm. At each position $p$, a next token is selected for both sequences: first, the top-$k$ candidates are taken from the base LLM to form a shared candidate set $\mathcal{V}_p$, then each candidate is scored using $\text{Score}(v|y_{<p}) = \pi_{\text{base}}(v|y_{<p}) + \omega \cdot r([y_{<p},v];\theta)$. Both trajectories expand based on the highest-scoring tokens, but the exploration arm includes an additional gradient-based UCB bonus. Once generation is complete, $(y_{t,1}, y_{t,2})$ are presented to the user to obtain a binary preference $l_t = \mathbb{1}\{y_{t,1} \succ y_{t,2}\}$. The reward network $\theta_t \to \theta_{t+1}$ is updated using the BTL loss, and the covariance matrix $V_t$ is updated for the next round's UCB calculation. The base LLM remains frozen; only the small reward NN changes.

### Key Designs

1.  **Test-time Alignment Scoring + Frozen LLM**:
    -   **Function**: Implements personalization as logit-level post-processing without touching base LLM parameters.
    -   **Mechanism**: At each token position, a preference reward from $r$ is added to the base LLM next-token probability $\pi_{\text{base}}(v|y_{<p})$, with $\omega$ controlling the intensity; the token with the maximum value is selected. Personalization is reflected in the shift of the decoding trajectory rather than weight updates, allowing for "adaptation" within dozens of per-user rounds and compatibility with closed-source LLMs.
    -   **Design Motivation**: Bypasses the overhead and cold-start hurdles of RLHF fine-tuning—no need to train a model for each user or accumulate large amounts of historical data beforehand.

2.  **Asymmetric Explore–Exploit Decoding with Dueling Arms**:
    -   **Function**: Ensures that the two candidate responses form an "informative pair" for preference signaling, rather than collapsing into the same sentence.
    -   **Mechanism**: The exploitation arm follows a pure greedy strategy $v_{p,1} = \arg\max_v \text{Score}(v|y_{t,1})$. The exploration arm adds a UCB term $\omega \cdot \nu \cdot u_t(v)$ to the score, where uncertainty $u_t(v) = \|\nabla r([y_{t,2},v];\theta_t) - \nabla r([y_{t,1}, v_{p,1}];\theta_t)\|_{V_{t-1}^{-1}}$ measures the "novelty" of the candidate relative to the current exploit arm in the reward parameter space. $V_{t-1}$ is the second-order matrix of historical gradient differences, updated via $V_{t-1} \leftarrow V_{t-1} + (\nabla r(y_{t,1}) - \nabla r(y_{t,2}))(\cdot)^\top$. Leveraging the neural dueling bandit theory (Verma et al. 2024), the paper provides a regret bound of $\tilde O(d_{\text{eff}}\sqrt{T})$, extended to the token level $\tilde O(L\sqrt T)$ via tokenized bandits (Shin et al. 2025).
    -   **Design Motivation**: Pure greedy strategies produce similar responses that are difficult for users to distinguish—providing no reward learning and wasting user patience. UCB on the second arm explicitly picks the most uncertain direction in the reward parameter space, maximizing the information gain for $\theta$ per comparison.

3.  **BTL Online Reward Learning + Asynchronous Update**:
    -   **Function**: Robustly integrates binary preferences from each round into the reward network.
    -   **Mechanism**: Minimizes the BTL negative log-likelihood with L2 regularization over all history $\mathcal{D}_t = \{(y_{s,1}, y_{s,2}, l_s)\}_{s=1}^t$: $\mathcal{L}_t(\theta) = -\sum [l \log\sigma(r(y_1;\theta) - r(y_2;\theta)) + (1-l)\log\sigma(r(y_2;\theta) - r(y_1;\theta))] + \lambda\|\theta\|_2^2$. To prevent training from blocking user experience, updates run in a background thread while current requests are served using $\theta_t$; the system switches to $\theta_{t+1}$ once training completes. After the personalization phase, the reward is frozen, and only the exploit arm is used for low-overhead greedy decoding.
    -   **Design Motivation**: BTL mathematically aligns "user chose A over B" signals with sigmoid probabilities of reward differences. Asynchronous training hides BP overhead from the user, which is critical for "real-time deployment."

### Loss & Training
All trainable parameters reside in the small reward NN. The loss is BTL negative log-likelihood plus $\lambda\|\theta\|_2^2$; $V_t$ is initialized as $\lambda I$. Key hyperparameters include reward weight $\omega$, exploration coefficient $\nu$, candidate token count $k$, maximum generation length $M$, and interaction rounds $T$.

## Key Experimental Results

### Main Results
Evaluation used three base LLMs (Mistral-7B-Instruct-v0.2, Llama-3.1-8B-Instruct, Qwen2-7B-Instruct) across four benchmarks (HelpSteer, TruthfulQA, UltraChat, Personal Preference Eval) and four preference attributes (creative / verbose / concise / uplifting), with ArmoRM-Llama3-8B as the reward judge.

| Base LLM | Attribute (avg) | Base | Pref | BS16 | LA | AMULET | T-POP |
|------|-----------|------|------|------|----|--------|-------|
| Mistral-7B | Concise | 0.43 | 0.45 | 0.48 | 0.52 | 0.51 | **0.60** |
| Mistral-7B | Creative | 0.32 | 0.33 | 0.34 | 0.37 | 0.40 | **0.48** |
| Qwen2-7B | Concise | 0.41 | 0.47 | 0.49 | 0.55 | 0.55 | **0.60** |
| Qwen2-7B | Uplifting | 0.38 | 0.39 | 0.40 | 0.42 | 0.42 | **0.55** |
| Llama-3.1-8B | Verbose | 0.30 | 0.31 | 0.32 | 0.35 | 0.44 | **0.50** |

On average, T-POP outperforms the strongest baseline AMULET by approximately 28.0% on Qwen2-7B, 19.9% on Mistral-7B, and is nearly equal on Llama-3.1-8B (0.535 vs 0.5325), with an overall average Gain of 14.7%.

### Ablation Study

| Configuration | Effect | Explanation |
|------|------|------|
| Full T-POP (exploit + UCB exploration arm + BTL online update) | Best ArmoRM score + high GPT-4o win rate | Complete solution |
| Exploit arm only (No UCB exploration) | Extremely slow learning curve, preferences fail to converge | Degenerates to "repetitive greedy," no information gain |
| Fixed reward (No online update of $\theta$) | Equivalent to base + static bias, close to Pref | Verifies online learning as the performance source |
| Reduced interaction rounds $T$ | Significant improvement in first 20 rounds, peaks at 40–60 | Demonstrates "few-shot personalization" |

### Key Findings
-   **Sharp Sample Efficiency**: Reward scores rise rapidly within the first 20 interaction rounds and converge at 40–60 rounds, followed by a slight decrease due to minor overfitting; this curve is highly consistent across bases and attributes.
-   **UCB Exploration is Critical for Dueling**: If the second arm is also greedy, the two responses are nearly identical, and BTL learns nothing; asymmetric explore-exploit is the source of reward learning effectiveness.
-   **AMULET nearly catches up on Llama-3.1-8B**: Suggests that when the base LLM is already sensitive to preference attributes, simple linear logit adjustments can perform well. T-POP's advantage results from the robustness of "active exploration + online BTL" in difficult scenarios.
-   **Asynchronous updates do not degrade performance**: Serving with old $\theta_t$ while training in the background showed no significant data degradation, while reducing perceived latency to nearly zero.

## Highlights & Insights
-   **Embedding dueling bandit into the decoding loop instead of the response selection layer**: While best-of-N or rerankers can use pairwise feedback, their sample efficiency is much lower. Token-level dueling moves UCB exploration pressure into every step, yielding $\tilde O(L\sqrt T)$ regret via tokenized bandit theory.
-   **Gradient difference as uncertainty measure for structured sequences**: Traditional UCB cannot be calculated using frequency in token space. Applying the norm of the reward network's gradient difference under $V_{t-1}^{-1}$ as "epistemic uncertainty" is equivalent to exploration rewards from an NTK perspective, allowing seamless migration to any online RLHF variant using neural rewards.
-   **Compatibility with closed-source LLMs**: The system requires only logit/top-$k$ access, meaning it can share the same personalization "plugin" across different models and users.

## Limitations & Future Work
-   The "user" in the evaluation is simulated by GPT-4o; the alignment between BTL assumptions and real human tastes hasn't been directly validated. Feedback noise, stereotypes, and drift of real users were not modeled.
-   Expanding two top-$k$ candidates and calculating gradient-based UCB at every step results in much higher per-step latency than standard greedy decoding, posing deployment costs for long-generation scenarios.
-   Personalization was only validated at the "attribute level" (global styles like creative/concise). Finer, multi-faceted preferences (e.g., "rigorous + concise + emoji") and UCB stability on multi-dimensional rewards remain unanalyzed.
-   Preferences may drift, but the reward network accumulates monotonically—the paper does not discuss whether long-term use requires forgetting mechanisms or sliding windows.

## Related Work & Insights
-   **vs AMULET** (Zhang et al., 2025b): AMULET is also a test-time alignment framework treating token selection as online learning, but lacks an "active query" exploration mechanism, making it purely exploitative. T-POP uses dueling bandits for explicit exploration, leading to sustained gains over more rounds.
-   **vs Linear Alignment (LA)** (Gao et al., 2024): LA applies linear adjustments to logits, which is simple but limited in capacity. T-POP uses neural rewards + online BTL to fit finer preference structures.
-   **vs RLHF/DPO personalization variants** (Jang et al., 2023; Li et al., 2024b; Park et al., 2024): Those methods require per-user weight fine-tuning. T-POP keeps the LLM frozen and only updates a small reward NN, making it truly "ready to use" for new users.
-   **vs RAG / prompt-based personalization** (Salemi et al., 2024; Liu et al., 2023): These rely on existing user corpora, whereas T-POP generates signals through online preference interaction, covering cold-start users.

## Rating
-   Novelty: ⭐⭐⭐⭐ Embedding dueling bandits into token-level decoding is a fresh approach with clear theoretical guarantees.
-   Experimental Thoroughness: ⭐⭐⭐⭐ Covered three bases × four benchmarks × four attributes + learning curves + GPT-4o win rates, though lacking a real human user study.
-   Writing Quality: ⭐⭐⭐⭐ Algorithm descriptions and formula numbering are clean; the appendix provides comprehensive theory.
-   Value: ⭐⭐⭐⭐ Provides a deployment-friendly paradigm for "personalization of closed-source LLMs," which is more practical than retraining LoRAs in industry.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MMPB: It's Time for Multi-Modal Personalization](../../NeurIPS2025/recommender/mmpb_its_time_for_multi-modal_personalization.md)
- [\[AAAI 2026\] Preference is More Than Comparisons: Rethinking Dueling Bandits with Augmented Human Feedback](../../AAAI2026/recommender/preference_is_more_than_comparisons_rethinking_dueling_bandits_with_augmented_hu.md)
- [\[ACL 2026\] Personalizing LLMs with Binary Feedback: A Preference-Corrected Optimization Framework](../../ACL2026/recommender/personalizing_llms_with_binary_feedback_a_preference-corrected_optimization_fram.md)
- [\[ACL 2026\] Mirroring Users: Towards Building Preference-aligned User Simulator with User Feedback in Recommendation](../../ACL2026/recommender/mirroring_users_towards_building_preference-aligned_user_simulator_with_user_fee.md)
- [\[AAAI 2026\] Probabilistic Hash Embeddings for Online Learning of Categorical Features](../../AAAI2026/recommender/probabilistic_hash_embeddings_for_online_learning_of_categorical_features.md)

</div>

<!-- RELATED:END -->

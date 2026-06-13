---
title: >-
  [Paper Note] Team-Based Self-Play With Dual Adaptive Weighting for Fine-Tuning LLMs
description: >-
  [ACL 2026][LLM Alignment][Self-play fine-tuning] TPAW transforms LLM self-training into an alignment process where the current model "teams up" with historical models for self-play. It employs two adaptive mechanisms—tar…
tags:
  - "ACL 2026"
  - "LLM Alignment"
  - "Self-play fine-tuning"
  - "historical checkpoints"
  - "preference optimization"
  - "adaptive weighting"
date: 2026-05-08
content_hash: 30bbe6154d13a20a
---

# Team-Based Self-Play With Dual Adaptive Weighting for Fine-Tuning LLMs

**Conference**: ACL 2026  
**arXiv**: [2605.09922](https://arxiv.org/abs/2605.09922)  
**Code**: https://github.com/lab-klc/TPAW  
**Area**: Self-Supervised / LLM Alignment  
**Keywords**: Self-play fine-tuning, historical checkpoints, preference optimization, adaptive weighting, LLM alignment

## TL;DR
TPAW transforms LLM self-training into an alignment process where the current model "teams up" with historical models for self-play. It employs two adaptive mechanisms—target response weighting and main player weighting—to stabilize preference optimization, improving performance on the Open LLM Leaderboard and GSM8K without requiring additional human preference annotations.

## Background & Motivation
**Background**: LLM alignment typically relies on SFT, RLHF, or DPO. SFT requires high-quality demonstration data, while RLHF necessitates reward models and human preferences. Although DPO eliminates the explicit reward model, it still requires preference pairs. To reduce human annotation costs, self-play/self-training methods like SPIN utilize existing SFT data by treating human answers as positive samples and model-generated answers as negative samples to iteratively improve alignment quality.

**Limitations of Prior Work**: Existing self-training methods primarily focus on the generation quality of the "current model," leaving training trajectories underutilized. If samples generated in a specific round contain biases, subsequent iterations tend to amplify these errors. Another subtle issue is that DPO-style objectives simultaneously push positive samples and suppress negative ones; however, in later stages of self-training, model-generated responses become increasingly similar to target responses, narrowing the margin and making training signals noisy. The paper also observes a decline in the probability of target responses, causing the model to deviate from the SFT target distribution.

**Key Challenge**: Self-training aims to replace human preferences with model-generated data but must avoid instability, bias accumulation, and target distribution drift caused by "only comparing with the current self." In other words, the model needs to obtain a richer set of opponents from historical versions without allowing the noise from early, weaker checkpoints to overwhelm current learning.

**Goal**: The authors aim to further extract alignment gains from the same SFT data in a completely self-supervised setting. Specifically, they address three sub-problems: how to utilize historical checkpoints, how to prevent the decline of target response rewards, and how to assign appropriate training weights to different historical players for each sample.

**Key Insight**: The paper reformulates self-play as a competition between two teams: an opponent team responsible for generating negative samples increasingly similar to human answers, and a main player team responsible for distinguishing between SFT target responses and generated responses. Historical checkpoints enter both teams, ensuring the training process no longer relies solely on a single judgment from the current model.

**Core Idea**: Replace single-model self-play with "historical checkpoint teaming + dual adaptive weighting," enabling more stable and data-efficient alignment on the same SFT dataset.

## Method
The intuition of TPAW is akin to transforming a single-player sparring session into a team match. Standard SPIN only uses the current model to generate negative samples and trains it to distinguish between human and self-generated answers. TPAW retains models from the most recent rounds to collectively form opponents and judges. This provides two benefits: first, negative samples come from multiple stages of the training trajectory, reflecting diverse error modes; second, implicit rewards are given by the relative probabilities of current and historical models, measuring whether the "current policy is more biased towards the target than a particular historical policy."

### Overall Architecture
The input consists of an SFT dataset $D_{SFT}=\{(x_i,y_i)\}$ and an initial SFT policy $\pi_{\theta_0}$. In round $t+1$, the current policy $\pi_{\theta_t}$ samples $y_i^{gen}$ for a given prompt $x_i$, forming a triplet $(x_i,y_i,y_i^{gen})$. The paper retains triplets from the most recent three rounds to form the opponent dataset $D_O=D_t\cup D_{t-1}\cup D_{t-2}$.

Next, TPAW constructs main players for the three most recent checkpoints. Each player $P_j$ uses the log-probability ratio between the current model and a historical model $\pi_{\theta_j}$ as an implicit reward: $P_j(x,y)=\lambda\log \frac{\pi_\theta(y|x)}{\pi_{\theta_j}(y|x)}$. If a target answer is closer to the current model's target distribution, the player should assign it a higher score; if a generated answer resembles a response deviating from the target, the player should assign it a lower score. The training objective is to increase the margin $P_j(x,y)-P_j(x,y^{gen})$ for each player.

Finally, instead of simply averaging losses from all players, TPAW applies weight $\alpha$ to the target response and weight $\beta$ to different players, using a weighted logistic loss to update the policy. After several iterations, the final aligned policy is obtained.

### Key Designs
1.  **Team-Based Self-Play Framework**:
    *   **Function**: Extends single-model self-play into a team-based setup with an opponent team and a main player team composed of the three most recent historical checkpoints.
    *   **Mechanism**: The opponent team generates negative samples on SFT prompts, while the main player team judges the relative quality of target and generated responses. Training data is drawn from triplets across the last three rounds to prevent synthetic data from a single round from dominating optimization.
    *   **Design Motivation**: Historical checkpoints record the model's trajectory from weak to strong, offering a richer error distribution than a single model and alleviating bias accumulation common in self-training.

2.  **Adaptive Target Response Weighting**:
    *   **Function**: Automatically increases the weight of the target response term when its reward under a specific player is insufficiently high.
    *   **Mechanism**: If $P_j(x,y)\le 0$, indicating the current model does not prefer the target response over the historical model, the target response weight is set to $\eta>1$; otherwise, the weight is 1. The final loss compares $\alpha_j P_j(x,y)$ and $P_j(x,y^{gen})$.
    *   **Design Motivation**: DPO-style self-training can decrease the probability of both positive and negative samples, moving the model away from the SFT target distribution. Target response weighting acts to pull the training focus back to ground-truth answers when drift is detected.

3.  **Adaptive Main Player Weighting**:
    *   **Function**: Dynamically allocates training contributions from different checkpoints based on the current discriminative difficulty each player faces for a sample.
    *   **Mechanism**: The margin $m_j=P_j(x,y)-P_j(x,y^{gen})$ is calculated first, followed by computing player weights as $\beta_j=\frac{e^{-\gamma m_j}}{\sum_k e^{-\gamma m_k}}$. Smaller margins indicate a player's inability to distinguish between samples, resulting in higher training weights.
    *   **Design Motivation**: Recent checkpoints might be insufficiently trained, while earlier checkpoints might have overfitted their corresponding distributions. Dynamic weighting per sample allocates the learning budget more effectively than static averaging.

### Loss & Training
TPAW uses the logistic loss $\ell(t)=\log(1+\exp(-t))$, optimizing $\ell(\alpha_j P_j(x,y)-P_j(x,y^{gen}))$ for each player and aggregating them into a team-level objective using $\beta_j$. In experiments, the opponent/main player teams default to the three most recent checkpoints. Practical settings provided by hyperparameter analysis are $\eta=6$ and $\gamma=0.5$. Missing terms are skipped in Iteration 0 (no history), and Iteration 1 uses the two available checkpoints before moving to the full three-player team training.

## Key Experimental Results

### Main Results
Experiments were conducted on Qwen2.5-1.5B and Llama3.1-8B. Ultrachat200k was used for SFT, and a 50k subset was used for TPAW/SPIN. Evaluation spanned 12 benchmarks from Open LLM Leaderboard V1/V2. The table below highlights the best iterations for each method in terms of average scores.

| Base Model | Method | V1 Avg. | V2 Avg. | Remarks |
| :--- | :--- | :--- | :--- | :--- |
| Qwen2.5-1.5B | SFT | 56.28 | 13.40 | Full Ultrachat200k SFT |
| Qwen2.5-1.5B | DPO | 58.55 | 13.50 | Uses UltraFeedback preference data |
| Qwen2.5-1.5B | SPIN best | 57.61 | 14.34 | V1 best at iter-4, V2 best at iter-3 |
| Qwen2.5-1.5B | TPAW best | 57.76 | 14.82 | V1 best at iter-4, V2 best at iter-3 |
| Llama3.1-8B | SFT | 63.93 | 17.69 | Initial aligned model |
| Llama3.1-8B | DPO | 64.79 | 17.91 | Extra preference data baseline |
| Llama3.1-8B | SPIN best | 64.88 | 19.68 | V1 best at iter-4, V2 best at iter-1 |
| Llama3.1-8B | TPAW best | 66.14 | 20.84 | V1 best at iter-3, V2 best at iter-4 |

TPAW's gains extend beyond average scores. On Qwen, IFEval improved by up to 4.37, Math by 3.55, and GSM8K by 3.79. On Llama, Arc increased by 4.79, IFEval by 8.78, and MUSR by 4.45. Given that TPAW uses no extra human preference data, these results indicate it extracts stronger training signals from existing SFT data.

| GSM8K Training | Accuracy | Gain over Qwen2.5-1.5B-SFT | Trend |
| :--- | :--- | :--- | :--- |
| Qwen2.5-1.5B-SFT | 51.25 | - | Initial model |
| SPIN-gsm8k iter-1 | 53.75 | +2.50 | Notable gain |
| SPIN-gsm8k iter-2 | 54.36 | +3.11 | Continued improvement |
| SPIN-gsm8k iter-4 | 54.59 | +3.34 | Plateauing in later stages |
| TPAW-gsm8k iter-1 | 54.21 | +2.96 | Slightly better than SPIN iter-1 |
| TPAW-gsm8k iter-3 | 56.56 | +5.31 | Significant gap in round 3 |
| TPAW-gsm8k iter-4 | 56.94 | +5.69 | Final best |

### Ablation Study
The paper removes three key components on GSM8K: target response weighting, main player weighting, and the team mechanism. Qualitative results from Figure 3 show that removing the team-based mechanism causes the largest performance drop, while removing either adaptive weighting mechanism prevents the model from reaching TPAW's optimal performance.

| Ablation | Removed Component | Observed Impact | Explanation |
| :--- | :--- | :--- | :--- |
| w/o TRW | Adaptive Target Weight $\alpha$ | Target reward stays negative | Model is not explicitly pulled back to SFT distribution |
| w/o MPW | Main Player Weight $\beta$ | Reduced gain from multiple checkpoints | Static averaging fails to emphasize difficult players |
| w/o Team | Historical checkpoint team | most significant drop | Training degenerates to single-model self-play |

### Key Findings
*   The team-based design is the primary source of improvement. It provides diverse negative samples and uses historical models as implicit reward references, preventing the current model from self-confirmation in the self-training loop.
*   Adaptive Target Response Weighting directly addresses the DPO side effect where positive sample probabilities decrease. Figure 2 shows that without this, target response rewards remain negative, whereas TPAW allows the reward to rise and converge to positive values.
*   Simply increasing SFT epochs cannot substitute for TPAW. Analysis shows that additional SFT epochs lead to data memorization rather than generalization gains, whereas TPAW breaks the performance ceiling of SFT using the same data.

## Highlights & Insights
*   Treating historical checkpoints as "team members" is a natural and reusable concept. While many iterative methods only keep the final model, this paper demonstrates that the training trajectory itself is a low-cost supervisory signal.
*   The dual weighting design maps to specific failure modes: $\alpha$ prevents target distribution drift, and $\beta$ prevents weak judges from diluting the learning signal. It is not just simple loss adjustment but an explicit encoding of "where the instability lies."
*   The discussion on SFT data reuse is enlightening: demonstration data can serve as more than just an imitation target; it can repeatedly form preference signals through contrastive responses generated by the model itself.

## Limitations & Future Work
*   TPAW's upper bound is still constrained by SFT data quality. If SFT data is biased or low-quality, team-based self-play will fit it more thoroughly rather than correcting the target distribution.
*   Experiments primarily cover general leaderboards and GSM8K; safety alignment, multi-turn tool use, and code generation still require validation.
*   There is a trade-off in history window size. The paper notes that $N_{max} > 3$ introduces earlier, weaker checkpoints that might yield low-quality signals. Future work could consider dynamically selecting team members based on sample quality or checkpoint capability.
*   Generated negative samples still originate from the same model family, limiting exploration. Incorporating external weak models, RAG-based answers, or counterfactual perturbations could provide stronger contrastive signals.

## Related Work & Insights
*   **vs SPIN**: SPIN uses the current model to generate negative samples for distinction. TPAW adds historical checkpoint teams and dual adaptive weighting, more stably utilizing multi-round training trajectories.
*   **vs DPO**: DPO optimizes on human preference pairs; TPAW constructs preference signals from SFT targets and model-generated answers, requiring no extra labels at the cost of being dependent on generation quality.
*   **vs Self-Rewarding / AutoIF**: These focus on automatic data generation or verification. TPAW focuses on training stability, specifically preventing target response reward degradation.
*   **Insights**: For any iterative LLM training pipeline, one can attempt to use historical checkpoints as reference models, reward models, or data generators rather than only saving the final weights.

## Rating
*   **Novelty**: ⭐⭐⭐⭐ The combination of historical checkpoint teaming and dual adaptive weighting is well-integrated, despite being built on SPIN/DPO objectives.
*   **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers two base models, two leaderboards, GSM8K, and multiple ablations, though specialized domains or safety scenarios are lacking.
*   **Writing Quality**: ⭐⭐⭐⭐ Motivations are clear; formulas and algorithms are complete.
*   **Value**: ⭐⭐⭐⭐ High practical value for low-cost alignment and SFT data reuse, especially for teams with high-quality SFT sets but no preference labels.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Attack via Overfitting: 10-shot Benign Fine-tuning to Jailbreak LLMs](../../NeurIPS2025/llm_alignment/attack_via_overfitting_10-shot_benign_fine-tuning_to_jailbreak_llms.md)
- [\[ACL 2026\] Why Supervised Fine-Tuning Fails to Learn: A Systematic Study of Incomplete Learning in Large Language Models](why_supervised_fine-tuning_fails_to_learn_a_systematic_study_of_incomplete_learn.md)
- [\[ACL 2026\] ConsistRM: Improving Generative Reward Models via Consistency-Aware Self-Training](consistrm_improving_generative_reward_models_via_consistency-aware_self-training.md)
- [\[ICML 2026\] Safety Anchor: Defending Harmful Fine-tuning via Geometric Bottlenecks](../../ICML2026/llm_alignment/safety_anchor_defending_harmful_fine-tuning_via_geometric_bottlenecks.md)
- [\[ACL 2026\] ARES: Adaptive Red-Teaming and End-to-End Repair of Policy-Reward System](ares_adaptive_red-teaming_and_end-to-end_repair_of_policy-reward_system.md)

</div>

<!-- RELATED:END -->

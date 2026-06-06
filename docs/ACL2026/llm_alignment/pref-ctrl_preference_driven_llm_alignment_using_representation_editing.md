---
title: >-
  [Paper Note] Pref-CTRL: Preference Driven LLM Alignment using Representation Editing
description: >-
  [ACL2026][LLM Alignment][Representation Editing] Pref-CTRL introduces margin loss and regularizer loss for training a lightweight value function based on pairwise preference data within the RE-Control framework…
tags:
  - "ACL2026"
  - "LLM Alignment"
  - "Representation Editing"
  - "Preference Learning"
  - "Test-time Alignment"
  - "Value Function"
  - "RLHF"
date: 2026-05-08
content_hash: 5a32ab799ad5bbc6
---

# Pref-CTRL: Preference Driven LLM Alignment using Representation Editing

**Conference**: ACL2026  
**arXiv**: [2604.23543](https://arxiv.org/abs/2604.23543)  
**Code**: https://github.com/UTS-nlPUG/pref-ctrl  
**Area**: Model Compression / Test-time Alignment / LLM Alignment  
**Keywords**: Representation Editing, Preference Learning, Test-time Alignment, Value Function, RLHF

## TL;DR
Pref-CTRL introduces margin loss and regularizer loss for training a lightweight value function based on pairwise preference data within the RE-Control framework, a test-time alignment method that avoids updating LLM parameters. This makes representation editing better aligned with human preferences and consistently outperforms RE-Control on SHP, HH-RLHF, and cross-domain datasets.

## Background & Motivation
**Background**: LLM alignment typically relies on RLHF, PPO, DPO, or their variants, using preference data to make models conform to human expectations regarding helpfulness, harmlessness, and response style. While these training-time methods are effective, they often require re-training or fine-tuning model parameters, leading to high computational and maintenance costs when facing different models, preference attributes, and frequently changing requirements.

**Limitations of Prior Work**: Test-time alignment attempts to bypass full fine-tuning by adjusting outputs during the inference stage through reward models, candidate re-ranking, activation steering, or representation editing. RE-Control is a representative representation editing method: it freezes the LLM, uses a small value function to evaluate hidden states, and then lightly adjusts internal representations along the value function gradient during generation. The problem is that the value function in RE-Control primarily learns scalar rewards for individual responses, failing to explicitly utilize the "preferred response vs. rejected response" pairwise structure naturally contained in many alignment datasets.

**Key Challenge**: Supervision signals for alignment tasks are usually relative preferences rather than isolated scores. Human annotations more often express "response A is better than response B," and training-time methods like DPO are effective precisely because they directly model this relative relationship. If test-time representation editing still compresses preference data into point-wise rewards, the value function might know "whether this state has a high score" but lacks clarity on "how much of a gap should exist between the preferred and rejected responses."

**Goal**: The authors aim to retain the advantages of RE-Control—namely freezing the LLM, training only a lightweight external value function, and achieving alignment through representation editing during inference—while making the value function training more compatible with the structure of preference data. This aims to improve alignment performance, reduce over-optimization risks, and investigate whether such modifications can generalize to datasets outside the training distribution.

**Key Insight**: The observation is straightforward: if the training data already provides preferred/rejected pairs, the value function should not only regress to rewards but also learn to give the preferred terminal state a higher score and prevent the generated states from deviating too far from the preferred state. Therefore, the improvement does not lie in using a larger LLM or rewriting the inference control framework, but in adding preference ranking and conservative constraints to the value function's training objective.

**Core Idea**: Use a multi-objective value function training objective (reward regression + preference margin + generation-preferred regularization) to replace the simpler reward regression target in RE-Control, making test-time representation editing function more like preference learning than single-point reward chasing.

## Method
Pref-CTRL can be viewed as an upgrade to the objective function of RE-Control. It does not change the parameters of the base LLM or write alignment knowledge directly into model weights; instead, it trains a small value function outside the frozen model. This value function reads the last-layer hidden states of the LLM and outputs a scalar to estimate the value of the current generation state regarding the alignment goal. During inference, the system performs test-time representation editing on the hidden states in the direction of the value function gradient, moving subsequent generation closer to high-value regions.

The true contribution lies in the fact that the value function no longer only observes reward trajectories of LLM-generated responses but simultaneously reads three types of representations: hidden states of the preferred response, hidden states of the rejected response, and hidden states of the response generated by the LLM itself. Thus, the training objective of the value function can explicitly express two constraints: the preferred response should have a higher score than the rejected one; and the model-generated state should stay close to the high-value region corresponding to the preferred response to avoid over-optimization.

### Overall Architecture
The overall process is divided into two stages. In the training phase, the authors perform forward computation on preference data using the frozen LLM to extract hidden states for preferred text, rejected text, and LLM-generated text. Subsequently, a 3-layer MLP value function is trained on these hidden states: it regresses to the rewards provided by an external reward model, learns the relative ranking of preferred vs. rejected, and uses a regularizer to constrain the generated state's value to be close to the preferred state's value.

In the inference phase, the base LLM remains frozen. At each generation step, an RE-Control-style control mechanism treats the current state as a combination of hidden representations and logits, uses the trained value function to score the state, and performs test-time editing on internal representations in the direction that increases value. This note discusses the high-level mechanism as a research method for safety and preference alignment and does not involve operational details for evasion or safety bypass.

### Key Designs
1.  **Directly using the frozen representation editing framework of RE-Control**:
    *   **Function**: Moves model output toward a more aligned direction as perceived by the value function without fine-tuning LLM weights.
    *   **Mechanism**: RE-Control treats the generation process as a dynamical system where the state $s_t$ consists of hidden representations $h_t$ and pre-softmax logits $o_t$. The value function $V_\phi(s_t)$ estimates the future reward of the current state, and during inference, control signals are adjusted via gradient ascent to bring subsequent states closer to high-value regions.
    *   **Design Motivation**: This preserves the flexibility of test-time methods: a frozen LLM can be regulated via an external value function without re-training for every preference goal. The innovation of Pref-CTRL is built upon this framework, so the modifications focus on "how to train the value function" rather than "how to modify the LLM."

2.  **Explicit modeling of pairwise preferences using margin loss**:
    *   **Function**: Ensures the value function knows not just individual reward levels, but that the preferred terminal state should be higher than the rejected terminal state.
    *   **Mechanism**: For a pair of preference samples, the authors take the final state of the preferred response $s^{pref}$ and the final state of the rejected response $s^{rej}$, adding $L_{Margin}=-\log \sigma(V_\phi(s^{pref})-V_\phi(s^{rej}))$. If the value function scores the rejected response too high, the loss increases; if a sufficient gap is maintained, the loss decreases.
    *   **Design Motivation**: The key to alignment data is not just "what is the score of this response," but "why is this response more aligned than the other." Margin loss incorporates this relative comparison directly into value function training, allowing test-time representation editing to inherit core signals from preference learning methods.

3.  **Suppressing over-optimization with regularizer loss**:
    *   **Function**: Prevents the value function from only pursuing a larger preferred/rejected gap, which could cause generated states to deviate from natural language fluency or task relevance.
    *   **Mechanism**: The authors bring the value score of the final state of the LLM-generated response $s_N$ closer to that of the preferred response $s^{pref}$, formally resembling $L_{Regularizer}=(V_\phi(s_N)-V_\phi(s^{pref}))^2$. The final objective is $L_{Total}=L_{Regression}+L_{Margin}+L_{Regularizer}$.
    *   **Design Motivation**: A standalone margin constraint might train the value function into an "over-discriminator," leading to reward-hacking side effects in some settings. The regularizer acts as a conservative anchor for the editing direction: the generated state should converge toward the preferred state rather than extrapolating infinitely just for higher scores.

### Loss & Training
The value function adopts the same lightweight MLP architecture as RE-Control, with an input dimension equal to the LLM hidden size (4096 in experiments). The structure is Linear(4096→4096)+ReLU, Linear(4096→4096)+ReLU, Linear(4096→1). Training uses Adam with a learning rate of $1\times10^{-5}$, a batch size of 64, for 50 epochs, selecting the best epoch on the validation set for inference.

Experiments use UltraRM as the reward model to train the value function. Base models include Vicuna-7B and Hermes-3-Llama-3.1-8B. The authors train and test on SHP and HH-RLHF, and perform zero-shot cross-domain evaluation on PKU-SafeRLHF and Nectar. For inference analysis, the default step size is $\alpha=0.5$ with $k=100$ steps for HH-RLHF, PKU-SafeRLHF, and Nectar; SHP uses $\alpha=1$ and $k=100$ due to higher complexity.

## Key Experimental Results

### Main Results
The main experiments compare RE-Control, Pref-CTRL variants, and a training-time DPO reference baseline. Metrics include win rates from three LLM-as-a-judge models, UltraRM average reward, as well as diversity and coherence. The following highlights the comparison between RE-Control and Pref-CTRL(Margin+Regularizer).

| Dataset / Base Model | Method | Llama Judge Win Rate | DeepSeek Judge Win Rate | GPT Judge Win Rate | Avg. Reward | Conclusion |
|----------------------|--------|---------------------:|---------------------:|-------------------:|------------:|------------|
| SHP / Vicuna-7B      | RE-Control | 66.80 | 66.70 | 53.50 | -2.652 | Original baseline |
| SHP / Vicuna-7B      | Pref-CTRL(M+R) | 73.50 | 70.00 | 53.70 | -2.454 | Gains across judges and reward |
| SHP / Hermes3-8B     | RE-Control | 79.80 | 74.80 | 60.90 | -2.303 | Baseline on strong model |
| SHP / Hermes3-8B     | Pref-CTRL(M+R) | 80.40 | 76.40 | 61.40 | -2.166 | Small but consistent gains |
| HH-RLHF / Vicuna-7B  | RE-Control | 81.90 | 85.40 | 73.30 | -5.408 | Safety/Helpfulness baseline |
| HH-RLHF / Vicuna-7B  | Pref-CTRL(M+R) | 82.90 | 85.60 | 74.60 | -5.288 | Better win rate and reward |
| HH-RLHF / Hermes3-8B | RE-Control | 85.50 | 84.00 | 73.10 | -4.321 | Strong model baseline |
| HH-RLHF / Hermes3-8B | Pref-CTRL(M+R) | 85.70 | 84.30 | 73.60 | -4.268 | Slight gain while keeping diversity |

The table shows that Pref-CTRL's advantage is not an accidental fluctuation but consistent across datasets, base models, and multiple judges/metrics. Especially on SHP / Vicuna-7B, Pref-CTRL(M+R) improves the Llama judge win rate by 6.70 points and Avg. Reward from -2.652 to -2.454 compared to RE-Control.

The authors also compare Pref-CTRL with additional test-time alignment methods. On HH-RLHF / Hermes3-8B, Pref-CTRL shows a clear advantage over Best-of-N and is competitive with CAST, significantly outperforming it on the DeepSeek judge.

| Method | Llama Judge Win Rate | DeepSeek Judge Win Rate | GPT Judge Win Rate | Insight |
|--------|---------------------:|---------------------:|-------------------:|---------|
| Best-of-N | 85.30 | 78.90 | 72.10 | Requires filtering candidates |
| CAST | 86.00 | 79.90 | 74.70 | Strong steering baseline |
| Pref-CTRL | 85.70 | 84.30 | 73.60 | DeepSeek lead, competitive overall |

### Ablation Study
The most valuable conclusion from ablation is that margin loss is not always robust when used alone; the regularizer's role is to re-balance preference separation and generation conservatism.

| Dataset / Model | Config | Llama Win Rate | DeepSeek Win Rate | GPT Win Rate | Avg. Reward | Description |
|-----------------|--------|---------------:|-----------------:|------------:|------------:|-------------|
| SHP / Vicuna-7B | RE-Control | 66.80 | 66.70 | 53.50 | -2.652 | Reward regression only |
| SHP / Vicuna-7B | Pref-CTRL(Margin) | 72.20 | 67.60 | 50.20 | -2.612 | Llama/DeepSeek up, GPT judge down |
| SHP / Vicuna-7B | Pref-CTRL(Regularizer) | 68.07 | 64.56 | 51.70 | -2.884 | Limited gain from reg alone |
| SHP / Vicuna-7B | Pref-CTRL(M+R) | 73.50 | 70.00 | 53.70 | -2.454 | Most stable combined |

Cross-domain evaluations also support the argument: a value function that learns preference comparisons rather than just memorizing training set rewards is more robust when migrating to new data distributions.

| Training Data | Test Data | Method | Llama Win Rate | DeepSeek Win Rate | Gain |
|---------------|-----------|--------|---------------:|-----------------:|------|
| SHP | PKU-SafeRLHF | RE-Control | 81.00 | 73.00 | Baseline |
| SHP | PKU-SafeRLHF | Pref-CTRL | 83.00 | 75.00 | +2.00 on both judges |
| HH-RLHF | PKU-SafeRLHF | RE-Control | 78.00 | 65.00 | Baseline |
| HH-RLHF | PKU-SafeRLHF | Pref-CTRL | 80.00 | 67.00 | +2.00 on both judges |

### Key Findings
*   The structure of the value function training objective is critical. Explicitly adding pairwise preference relationships makes the test-time representation editing direction more reliable.
*   Margin loss is the primary preference discrimination signal, but using it alone may overemphasize the gap. The regularizer stabilizes results when combined with margin loss.
*   Pref-CTRL approaches some training-time DPO results, though they differ in cost and positioning: DPO modifies parameters, while Pref-CTRL freezes the LLM.
*   Sensitivity analysis shows rewards peak near $\alpha=0.5, k=100$ on HH-RLHF / Hermes3; further increasing step size or steps leads to a decline, indicating over-optimization boundaries for test-time gradient editing.
*   Diversity and coherence do not significantly degrade, indicating that the improvement is not a simple sacrifice of text quality for judge preference scores.

## Highlights & Insights
*   **Porting Preference Learning to Test-time Alignment**: Instead of reinventing a complex inference system, the paper identifies the mismatch between RE-Control's training objective and the preference data structure, fixing it with pairwise margin loss.
*   **The Significance of the Regularizer**: Ablation shows regularizer-only is weak, but it prevents the preference gap from being over-amplified when combined with margin loss. It acts as a stabilizer rather than a standalone booster.
*   **Test-time Methods Require Signal Integrity**: While test-time alignment focuses on inference control, Pref-CTRL shows that the supervision format of the external value function determines the reliability of the control direction.
*   **Cross-domain Results Validate the Relative Preference Hypothesis**: Consistent gains on OOD data like PKU-SafeRLHF suggest that margin + regularizer learns a more transferable preference ranking signal rather than just in-distribution reward calibration.
*   **Suitability for Multi-attribute Control**: The framework could naturally extend to multi-head value functions for simultaneous control of helpfulness, fairness, style, etc., without re-tuning the entire model.

## Limitations & Future Work
*   Gradient-based test-time intervention relies on hyperparameters like step size and iteration count, which may require re-validation across domains.
*   The value function depends on a fixed reward model and pairwise labels, so its preferences are limited by the coverage of the training data.
*   Experiments focused on single-turn prompts; stability in multi-turn dialogues or complex tool-use scenarios remains unproven.
*   Evaluation relies heavily on LLM-as-a-judge and UltraRM; more large-scale human evaluation is needed to account for potential judge bias.
*   Inference overhead was not detailed in the main tables. Although the value function is lightweight, each gradient editing step adds overhead that must be balanced against quality and latency.
*   Future work could explore attention-based value functions, multi-attribute alignment, and adaptive intervention based on input risk or uncertainty.

## Related Work & Insights
*   **vs RLHF / PPO**: RLHF updates LLM parameters through a reward model and RL, which is powerful but complex and costly. Pref-CTRL serves as a lightweight test-time correction layer.
*   **vs DPO**: DPO optimizes parameters using pairwise preferences. Pref-CTRL borrows the "preference as relative comparison" concept but applies it to the value function rather than the LLM.
*   **vs RE-Control**: Pref-CTRL improves the value function of RE-Control by introducing preferred/rejected/generated state awareness and more robust loss functions.
*   **vs Best-of-N**: Pref-CTRL adjusts the generation direction in internal representation space rather than filtering from a candidate pool, offering a different trade-off in inference overhead.
*   **Insight**: For "Frozen LLM + Small Controller" methods, the supervision objective of the small module is often more important than its capacity. If data is rank-based, it shouldn't be compressed into scalar labels.

## Rating
*   Novelty: ⭐⭐⭐⭐☆ A simple but clear and effective adaptation of pairwise preference learning into the RE-Control framework.
*   Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers major datasets, base models, multiple judges, and OOD tests; lacks detailed human evaluation and computational cost analysis.
*   Writing Quality: ⭐⭐⭐⭐☆ Clearly links motivation and methods; results are presented effectively despite some complex ablation tables.
*   Value: ⭐⭐⭐⭐☆ Highly relevant for those looking to reduce alignment costs, and a strong baseline for objective function improvement in test-time control.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Teaching LLM to be Persuasive: Reward-Enhanced Policy Optimization for Alignment from Heterogeneous Rewards](teaching_llm_to_be_persuasive_reward-enhanced_policy_optimization_for_alignment_.md)
- [\[NeurIPS 2025\] Robust LLM Alignment via Distributionally Robust Direct Preference Optimization](../../NeurIPS2025/llm_alignment/robust_llm_alignment_via_distributionally_robust_direct_preference_optimization.md)
- [\[ACL 2026\] Alignment Data Map for Efficient Preference Data Selection and Diagnosis](alignment_data_map_for_efficient_preference_data_selection_and_diagnosis.md)
- [\[ICML 2026\] Long Live The Balance: Information Bottleneck Driven Tree-based Policy Optimization](../../ICML2026/llm_alignment/long_live_the_balance_information_bottleneck_driven_tree-based_policy_optimizati.md)
- [\[ACL 2026\] On the Rejection Criterion for Proxy-Based Test-Time Alignment](on_the_rejection_criterion_for_proxy-based_test-time_alignment.md)

</div>

<!-- RELATED:END -->

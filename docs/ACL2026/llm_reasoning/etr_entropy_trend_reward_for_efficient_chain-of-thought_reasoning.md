---
title: >-
  [Paper Note] ETR: Entropy Trend Reward for Efficient Chain-of-Thought Reasoning
description: >-
  [ACL 2026][LLM Reasoning][Chain-of-Thought Efficiency] The authors propose ETR (Entropy Trend Reward), which incorporates momentum-weighted step-wise entropy reduction as a reward shaping term into GRPO. This allows LLMs…
tags:
  - "ACL 2026"
  - "LLM Reasoning"
  - "Chain-of-Thought Efficiency"
  - "Entropy Trend Reward"
  - "GRPO"
  - "Momentum"
  - "Adaptive Early Stopping"
date: 2026-05-08
content_hash: 6461f185602ce35a
---

# ETR: Entropy Trend Reward for Efficient Chain-of-Thought Reasoning

**Conference**: ACL 2026  
**arXiv**: [2604.05355](https://arxiv.org/abs/2604.05355)  
**Code**: https://github.com/Xuan1030/ETR  
**Area**: LLM Reasoning / RL / CoT Compression / GRPO  
**Keywords**: Chain-of-Thought Efficiency, Entropy Trend Reward, GRPO, Momentum, Adaptive Early Stopping

## TL;DR
The authors propose ETR (Entropy Trend Reward), which incorporates momentum-weighted step-wise entropy reduction as a reward shaping term into GRPO. This allows LLMs to adaptively achieve early convergence under a "global entropy reduction" constraint, compressing average CoT length by 35–65% with the same accuracy. On DeepSeek-R1-Distill-7B, it yields a +9.9% accuracy gain while reducing tokens by 67%.

## Background & Motivation

**Background**: Long-CoT reasoning (R1 / o1 / Qwen3) is the mainstream paradigm for reaching SOTA in LLM reasoning. However, "overthinking" causes models to generate tens of thousands of tokens even for simple problems, leading to linear increases in inference latency and high deployment costs. Current efficiency improvement routes fall into three categories: (1) training-free prompt/early stopping (DEER / NoThink / CGRS); (2) variable-length SFT (TokenSkip / Liu et al.); and (3) RL reward design (LCPO / O1-Pruner / PEAR).

**Limitations of Prior Work**: Length-penalizing rewards are content-blind—tokens of the same length may contribute entirely different amounts of information. Entropy-related methods (PEAR / Li 2025 / Agarwal 2025) examine model uncertainty but focus on "globally suppressing entropy." This implicitly assumes that "CoT should maintain low uncertainty at all times," which contradicts the natural human reasoning process of "divergent exploration → convergent determination." Forceful suppression can eliminate self-reflection entirely.

**Key Challenge**: High-entropy moments are often where self-reflection occurs (marked by words like "wait / but / hmm"). Globally suppressing entropy of the trajectory kills both useful reflection and redundant divergence, damaging accuracy. Conversely, failing to suppress it makes length control impossible.

**Goal**: (1) Identify a trajectory-level signal that truly reflects whether reasoning is converging; (2) apply this signal as a shaping reward in GRPO rather than a hard constraint; (3) enable the model to naturally be short for easy problems and long for hard ones without manual length rules.

**Key Insight**: The authors conducted a key experiment on MATH500, calculating the Spearman $\rho$ between the step index and step entropy for each CoT. They found that a more negative $\rho$ (entropy significantly decreasing with steps) correlates with shorter lengths, while a more positive $\rho$ correlates with longer lengths. This links "reasoning efficiency" to the "directionality of the entropy trajectory."

**Core Idea**: Reward the "global entropy downward trend" rather than "instantaneous low entropy." This allows for local reflections and small fluctuations but requires uncertainty to monotonically decrease along the CoT overall, enabling the model to learn instance-adaptive early stopping naturally.

## Method

### Overall Architecture
ETR does not modify the GRPO optimization algorithm or add hard length constraints; it only rewrites the reward:
$$R(q,o)=\begin{cases}-1,&\text{if incorrect}\\ 1+\lambda R_{\text{entropy}}(o),&\text{if correct}\end{cases}$$
where $R_{\text{entropy}}(o)$ is a shaping term based on trajectory entropy. GRPO normalizes relative advantages within the same question group, ensuring the ETR signal only compares efficiency among correct solutions.

### Key Designs

1.  **Momentum-based Entropy Trend Reward**:
    - **Function**: Ensures the reward reflects the "entropy reduction direction of the entire CoT" rather than just looking at endpoints or instantaneous values.
    - **Mechanism**: CoT is split into steps $\{C_1,\dots,C_T\}$ by `\n\n`. For each step, the Shannon entropy of the next-token prediction distribution is calculated: $H_t=H(p_\theta(\cdot\mid C_{1:t}))$. The inter-step entropy change is $\Delta_t=H_{t-1}-H_t$. A momentum state is introduced: $S_t=\gamma S_{t-1}+\Delta_t$ ($S_1=0$, $\gamma=0.9$). The final reward is $R_{\text{entropy}}(o)=\sum_{t=2}^{T}S_t=\sum_t \alpha_t\Delta_t$, where $\alpha_t=\frac{1-\gamma^{T-t+1}}{1-\gamma}$. Since $\alpha_t$ strictly decreases with $t$, early entropy reduction is weighted more heavily than later reduction, encouraging fast convergence.
    - **Design Motivation**: A naive "total entropy reduction" $R_{\text{naive}}=H_1-H_T$ telescopes to rely only on start/end values, failing to distinguish between "smooth decline" and "repeated oscillation." The momentum formula provides gradient signals for every step and encodes "early convergence" via decreasing weights.

2.  **Implicit Instance-adaptive Stopping (Trend-shaped Stopping)**:
    - **Function**: Adaptively stops the model without hard-coded length limits—short for simple problems, long for hard ones.
    - **Mechanism**: Because $R_{\text{entropy}}=\sum_t S_t$ accumulates momentum states, taking an extra step is only beneficial if $S_{t+1}>0$ (i.e., $\Delta_{t+1}$ continues to decrease). Once entropy rebounds ($\Delta_t<0$), the model is repeatedly penalized, automatically suppressing oscillatory self-reflection loops. For easy problems, entropy collapses quickly → early stop; for hard problems, disambiguation requires more steps → natural extension, but every step must contribute to the decline.
    - **Design Motivation**: Unlike explicit length limits (LCPO / O1-Pruner), ETR does not require a pre-defined budget. Unlike global entropy minimization (PEAR / Li 2025), ETR allows temporary increases for long-term decreases, matching the human "hypothesis-test-backtrack" pattern.

3.  **Correct-first then Shape (Reward Structure)**:
    - **Function**: Strictly limits efficiency rewards to comparisons between "correct trajectories" to prevent efficiency from overriding accuracy.
    - **Mechanism**: The reward is split—incorrect answers always receive -1, while correct ones receive $1+\lambda R_{\text{entropy}}$. GRPO performs relative normalization $\hat{A}_i=(r_i-\bar{r})/\sigma_r$ within the group, meaning ETR only differentiates "multiple correct solutions for the same problem" and avoids sacrificing accuracy across different problems.
    - **Design Motivation**: Adding entropy rewards directly to the global reward can cause the model to choose short but wrong answers (as verified in the "No $R_{\text{corr}}$" ablation). The two-stage structure + GRPO relative advantage ensures "correctness is the hard constraint and efficiency is the tie-breaker."

### Loss & Training
Standard GRPO PPO-clipped objective with intra-group advantage normalization and KL coefficient $\beta$. Reward is as defined above; $\lambda$ controls the entropy shaping strength; $\gamma=0.9$ is momentum. Training data consists of 7,000 problems (difficulty 5–10) from DeepMath-103K. Training uses LoRA + VeRL on 8×H100, batch 32, lr $1\times10^{-5}$, max length 16384, with 5 rollouts per question.

## Key Experimental Results

### Main Results
Evaluated on AMC23 / AIME24 / MATH500 / GPQA-Diamond (greedy pass@1):

| Model | Method | Overall Acc ↑ | Overall Len ↓ | AES ↑ |
|------|-----|--------------|--------------|------|
| DeepSeek-R1-Distill-7B | Original | 58.1 | 8.5k | 0.00 |
| DeepSeek-R1-Distill-7B | DEER | 60.9 | 6.2k | 0.51 |
| DeepSeek-R1-Distill-7B | NoThink | 59.5 | 4.0k | 0.65 |
| DeepSeek-R1-Distill-7B | LCPO | 58.6 | 3.8k | 0.60 |
| DeepSeek-R1-Distill-7B | O1-Pruner | 66.9 | 4.8k | 1.18 |
| DeepSeek-R1-Distill-7B | PEAR | 69.8 | 5.1k | 1.41 |
| **DeepSeek-R1-Distill-7B** | **ETR** | **68.0** | **2.8k** | **1.53** |
| Qwen3-4B | Original | 69.5 | 8.7k | 0.00 |
| Qwen3-4B | PEAR | 77.2 | 6.7k | 0.79 |
| **Qwen3-4B** | **ETR** | **77.1** | **4.4k** | **1.03** |
| Qwen3-8B | Original | 74.0 | 8.9k | 0.00 |
| Qwen3-8B | PEAR | 74.6 | 7.6k | 0.18 |
| **Qwen3-8B** | **ETR** | **79.1** | **5.1k** | **0.77** |

On DeepSeek-R1-Distill-7B, ETR compresses CoT from 8.5k to 2.8k (33% compression) while increasing accuracy from 58.1 to 68.0. The gain on AIME24 is particularly notable (11.8k → 4.6k, 43.3 → 56.7).

### Ablation Study
Comparing different entropy reward designs on DeepSeek-R1-Distill-7B:

| Reward Design | AMC23 Acc / Len | AIME24 Acc / Len | MATH500 Acc / Len | GPQA-D Acc / Len | AES |
|-------------|-----------------|------------------|-------------------|------------------|------|
| Original | 80.0 / 6.6k | 43.3 / 11.8k | 85.0 / 4.2k | 24.2 / 11.3k | 0.00 |
| Min. $H$ (Global Suppression) | 80.0 / 2.1k | 43.3 / 5.1k | 88.2 / 1.3k | 38.3 / 2.1k | 1.06 |
| Max. $H$ (Inversion Max) | 10.0 / 15.1k | 0.0 / 16.4k | 9.0 / 15.3k | 1.5 / 16.0k | -5.4 |
| No $\gamma$ (Telescoping) | 87.5 / 4.9k | 46.7 / 10.0k | 87.8 / 3.6k | 31.8 / 10.0k | 0.61 |
| No $R_{\text{corr}}$ (No Correctness) | 65.0 / 1.2k | 23.3 / 1.4k | 78.6 / 0.7k | 29.8 / 0.7k | 0.11 |
| **Ours (Full ETR)** | **87.5 / 2.4k** | **56.7 / 4.6k** | **90.6 / 1.5k** | **37.4 / 2.5k** | **1.53** |

### Key Findings
- **Momentum is essential**: Removing momentum leads to a telescoping-only form where CoT length barely decreases (4.9k vs 2.4k for ETR), because endpoint-only rewards provide no gradient signal for intermediate steps. Momentum ensures every step's entropy change contributes to shaping.
- **Entropy trend $\neq$ Global suppression**: Min. $H$ has a significantly lower AES than ETR (1.06 vs 1.53) and heavily suppresses the number of reflection tokens. ETR retains a moderate amount of self-reflection while maintaining low verbosity per step—Figure 6 verifies that ETR compresses CoT by reducing token count per step rather than banning reflection.
- **Correctness must be a hard constraint**: Without $R_{\text{corr}}$, CoT shrinks to 1.2k but AMC23 accuracy drops from 80 to 65, proving that entropy shaping alone can make models "short and wrong."
- **Spearman $\rho$ inversion validates convergence**: After ETR training, the $\rho$(step, $H_t$) for all models shifts from positive/near-zero to negative, showing that entropy truly decreases along steps—this is direct evidence of ETR turning a hypothesis into post-training behavior.
- **Cross-model generalization**: ETR achieves the highest AES across both DeepSeek-R1-Distill and Qwen3 families (4B–8B), showing the method is not dependent on specific architectures or pre-training.
- **Difficult problems gain the most**: On AIME24, ETR achieves +13.4 accuracy with a 60% reduction in length, aligning with the idea that difficult problems need more steps but each step must contain effective information.

## Highlights & Insights
- "Focusing on the trend rather than the absolute value of entropy" is a conceptual shift—modeling the reasoning process as a dynamic system rather than a static distribution, allowing RL rewards to directly reward the abstract concept of "convergence speed."
- The momentum-weighted $\alpha_t$ decreasing property implicitly encodes a preference for "earliest possible convergence," mathematizing the human intuition that CoT should rapidly approach the answer into a reward.
- Compared to other entropy methods like PEAR, the key difference is ETR's "allowing temporary rises for long-term falls," which naturally aligns with human explore-then-exploit reasoning; this "local tolerance but global strictness" philosophy could be transferable to tool-use control or Agent rollback strategies.
- Figure 6 breaks down "CoT compression" into steps, tokens per step, and reflection words. ETR primarily reduces verbosity per step rather than cutting steps—this behavior-level attribution is rare and serves as a good methodological demonstration for evaluating reasoning compression.
- The combination of a two-stage reward (incorrect -1, correct $1+\lambda R$) and GRPO relative normalization neatly solves the common multi-objective RL challenge where efficiency overrides accuracy, a template applicable to latency, safety, or formatting goals.

## Limitations & Future Work
- Due to compute limits, comparative experiments were limited to 8B + LoRA; whether ETR behavior remains consistent at larger scales (32B / 70B) needs verification, especially as large models may internally converge faster.
- $\lambda$ and $\gamma$ are fixed empirical values. The optimal $\lambda$ may vary by task difficulty; the paper does not propose an adaptive tuning scheme.
- Entropy calculation relies on next-token prediction distributions, making it applicable only to white-box self-trained models; it cannot be applied directly to closed-source APIs (GPT-4 / Claude).
- Step splitting uses the `\n\n` heuristic, which may not be robust for models that generate long single-paragraph texts; semantic-level step partitioning might be more precise.
- Only validated on reasoning benchmarks (Math / GPQA); transferability to coding, tool use, or multi-turn dialogue is not explicitly established.
- ETR treats entropy as an introspective signal, but high entropy $\neq$ high information. If the model's prediction distribution is poorly calibrated, ETR might learn incorrect trajectory patterns; integration with an external verifier could be more robust.

## Related Work & Insights
- **vs PEAR (Huang 2025a)**: PEAR also uses entropy rewards in GRPO but takes the global suppression route (maintaining accuracy while CoTs remain long). ETR focuses on trends; its AES on DeepSeek-R1-Distill-7B is 1.53 > PEAR's 1.41, with more significant length compression (2.8k vs 5.1k).
- **vs O1-Pruner / LCPO (length-based RL)**: Those use length-penalty training, which is content-blind and prone to accuracy drops. ETR uses efficiency as a "tie-breaker for correct solutions" to avoid this.
- **vs DEER / NoThink (training-free)**: Training-free methods perform worse on large models and lack controllability. ETR learns generalized early stopping through RL, resulting in significantly higher AES.
- **vs Min. $H$ / Compressing-CoT via Step Entropy (Li 2025)**: Global entropy suppression methods tend to wipe out useful self-reflection. ETR proves via Figure 6 that it retains reflection while reducing per-step verbosity, which is a smarter compression method.
- **vs Token-skip / Variable-length SFT (Xia 2025 / Liu 2024)**: Those require labeled CoT data for SFT, which limits generalization. ETR uses RL signals without requiring labeled data.
- **Transferable Insight**: The momentum-weighted $\alpha_t$ decreasing structure and the "global trend + local tolerance" reward design philosophy are valuable for any task requiring "output compression while maintaining quality" (e.g., code generation, summarization, multi-step planning).

## Rating
- **Novelty**: ⭐⭐⭐⭐ "Looking at entropy trends" is a clear perspective shift, and the momentum design is elegant; however, entropy rewards have been explored by PEAR and others.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ 3 models × 4 benchmarks + complete ablation (Min/Max/No Momentum/No Correctness) + Spearman $\rho$ validation + behavior decomposition in Figure 6.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation derivation is direct (strong Spearman $\rho$ vs length scatter plot), formulas and algorithms are clear, and the analogy to human reasoning is well-articulated.
- **Value**: ⭐⭐⭐⭐ Directly hits the "overthinking" pain point in the reasoning model era. Ranked first in AES with cross-family generalization, making it ready for production deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Revisiting Entropy in Reinforcement Learning for Large Reasoning Models](revisiting_entropy_in_reinforcement_learning_for_large_reasoning_models.md)
- [\[ACL 2026\] Efficient Process Reward Modeling via Contrastive Mutual Information](efficient_process_reward_modeling_via_contrastive_mutual_information.md)
- [\[ACL 2026\] Learning to Edit Knowledge via Instruction-based Chain-of-Thought Prompting](learning_to_edit_knowledge_via_instruction-based_chain-of-thought_prompting.md)
- [\[NeurIPS 2025\] Re-FORC: Adaptive Reward Prediction for Efficient Chain-of-Thought Reasoning](../../NeurIPS2025/llm_reasoning/re-forc_adaptive_reward_prediction_for_efficient_chain-of-thought_reasoning.md)
- [\[ICLR 2026\] DRPO: Efficient Reasoning via Decoupled Reward Policy Optimization](../../ICLR2026/llm_reasoning/drpo_efficient_reasoning_via_decoupled_reward_policy_optimization.md)

</div>

<!-- RELATED:END -->

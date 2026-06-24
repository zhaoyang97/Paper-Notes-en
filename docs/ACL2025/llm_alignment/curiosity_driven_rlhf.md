---
title: >-
  [Paper Note] Curiosity-Driven Reinforcement Learning from Human Feedback
description: >-
  [ACL 2025][LLM Alignment][curiosity-driven] CD-RLHF introduces curiosity-driven exploration (curiosity-driven RL) into RLHF. By utilizing the prediction error of a forward dynamics model as an intrinsic reward, combined with top-k gating filtering and reward whitening, it significantly enhances LLM output diversity without compromising alignment quality (achieving a 40.26% increase in Diversity and an 8.92% increase in EAD on Llama-3.2-1B).
tags:
  - "ACL 2025"
  - "LLM Alignment"
  - "curiosity-driven"
  - "RLHF"
  - "diversity"
  - "intrinsic reward"
  - "exploration"
  - "ICM"
date: 2026-05-08
content_hash: 05c0bcd6b5c5f891
---

# Curiosity-Driven Reinforcement Learning from Human Feedback

**Conference**: ACL 2025  
**arXiv**: [2501.11463](https://arxiv.org/abs/2501.11463)  
**Code**: [github](https://github.com/ernie-research/CD-RLHF)  
**Area**: LLM Alignment / RLHF  
**Keywords**: curiosity-driven, RLHF, diversity, intrinsic reward, exploration, ICM

## TL;DR

CD-RLHF introduces curiosity-driven exploration (curiosity-driven RL) into RLHF. By utilizing the prediction error of a forward dynamics model as an intrinsic reward, combined with top-k gating filtering and reward whitening, it significantly enhances LLM output diversity without compromising alignment quality (achieving a 40.26% increase in Diversity and an 8.92% increase in EAD on Llama-3.2-1B).

## Background & Motivation

**Background**: RLHF is the current standard paradigm for LLM alignment—training a reward model on human preference data and then fine-tuning the LLM with RL algorithms such as PPO, which has achieved significant success in safety, instruction following, and other aspects.

**Limitations of Prior Work**: After RLHF training, LLM output diversity decreases significantly, and models tend to generate "safe", homogeneous responses. Kirk et al. (2024) and Wu et al. (2024) systematically pointed out a clear trade-off between alignment quality and output diversity, which Wu defined as **generative monoculture**.

**Key Challenge**: PPO optimization causes the policy to collapse toward high-reward regions (mode collapse), leading to conservative token choices and convergence of generated texts on both lexical and semantic levels. This directly limits LLMs in downstream applications requiring diversity, such as creative writing, data synthesis, and red-teaming.

**Limitations of Prior Work**: Hong et al. (2024) incorporated SelfBLEU and Sentence-BERT as auxiliary rewards during RL training, mainly serving test coverage in red-teaming with limited and unsystematic results. Wang et al. (2024) replaced reverse KL with forward KL, which improved diversity but sacrificed alignment quality. Bradley et al. (2023) performed diversity post-processing during the inference stage, which does not address the fundamental issue during training.

**Key Insight**: Drawing inspiration from classic curiosity-driven exploration theories in RL (Pathak et al., 2017; Burda et al., 2019), this work introduces a token-level intrinsic reward based on state prediction error alongside the sparse extrinsic reward of standard RLHF, thereby encouraging the policy model to explore "novel" state spaces during training.

**Core Idea**: The prediction error of a forward dynamics model is utilized to measure the "novelty" of a state. Novel states receive higher intrinsic rewards, motivating the LLM to make more diverse token choices under the same prompt, while the extrinsic reward ensures that alignment quality does not degrade.

## Method

### Overall Architecture

CD-RLHF adds an **Intrinsic Curiosity Module (ICM)** on top of standard PPO-RLHF. The training workflow is as follows: (1) the policy model generates response tokens based on a prompt; (2) the reward model provides an extrinsic reward $R$; (3) the ICM computes intrinsic curiosity rewards token by token; (4) the combined rewards are used to optimize the policy via PPO. The data split ratio is SFT:RM:PPO = 20%:40%:40%.

### Reward Design

The extrinsic reward consists of the reward model score $R$ and a KL penalty term:

$$r^{(e)} = R - \beta D_{\text{KL}}(\pi_{\text{policy}}(\cdot) \| \pi_{\text{ref}}(\cdot))$$

The total reward combines the intrinsic and extrinsic rewards:

$$r_t = r_t^{(e)} + \eta \cdot r_t^{(i)}$$

where $\eta$ is a scaling factor controlling the intensity of the intrinsic reward (set to a very small value in experiments because the scale of the intrinsic reward is much larger than that of the extrinsic reward).

### Intrinsic Curiosity Module (ICM)

ICM consists of a feature encoder $\phi$ (a 2-layer MLP) and a forward model $f$ (a 2-layer MLP). Under the context of RLHF, the state is defined as $s_t = \{s_0, a_{<t}\}$ (prompt + already generated tokens), and the action $a_t$ is the currently selected token.

**Forward Dynamics Prediction**: Given the current state encoding $\phi(s_t)$ and action representation $\psi(a_t)$, the forward model predicts the next state:

$$\hat{\phi}(s_{t+1}) = f(\phi(s_t), \psi(a_t))$$

**ICM Loss** (self-supervised training):

$$\mathcal{L}_{\text{ICM}} = \frac{1}{2} \|\hat{\phi}(s_{t+1}) - \phi(s_{t+1})\|_2^2$$

**Design Motivation**: If the model selects an "unexpected" token (large forward model prediction error), it indicates a novel direction of exploration that deserves reward. As ICM training progresses, the prediction error for frequently visited states consistently decreases—shifting from "curiosity" to "boredom"—which naturally guides the model to explore new generation paths.

### Top-k Gating Filter

Not all token positions warrant curiosity rewards. If the selected token is already among the top-k candidates in the policy distribution, it indicates the model is highly certain, resulting in low exploration value. The intrinsic reward is defined as:

$$r_t^{(i)} = \begin{cases} 0 & \text{if } a_t \in \text{Top-}k(\pi(\cdot | s_t)) \\ \frac{1}{2}\|\hat{\phi}(s_{t+1}) - \phi(s_{t+1})\|_2 & \text{otherwise} \end{cases}$$

In experiments, $k=1$ is used, and the intrinsic reward is activated for approximately 20% of the token positions. Experiments demonstrate that tokens outside the top-1 range are sufficient to cover the effective interval of curiosity-driven exploration (see Frequency Analysis).

### Reward Whitening and Feature Alignment

- **Reward Whitening**: Normalizes the intrinsic reward as $r^{(i)} = (r^{(i)} - \mu) / \sigma^2$ to eliminate scale differences between intrinsic and extrinsic rewards and stabilize PPO training.
- **Feature Space Alignment**: The state representation $s_t$ is extracted from the reference model's last-layer hidden states (continuous space), and the action representation $a_t$ is extracted from the policy model's embedding layer, ensuring that the prediction error is calculated within a consistent feature space.

## Key Experimental Results

### Main Results: TL;DR Text Summarization (Table 1 Upper Half)

| Model | Method | Diversity↑ | EAD↑ | SelfBLEU↓ | SentBERT↓ | RM Score↑ |
|------|------|-----------|------|-----------|-----------|-----------|
| Gemma-2B | RLHF | 0.2132 | 0.7347 | 0.3367 | 0.7024 | 0.90 |
| Gemma-2B | Sent-Rewards | 0.2355 | 0.7512 | 0.3053 | 0.6961 | 0.95 |
| Gemma-2B | **CD-RLHF** | **0.2839 (+33.2%)** | **0.7793 (+6.1%)** | **0.2590** | **0.6720** | **0.95** |
| Gemma-7B | RLHF | 0.1180 | 0.6602 | 0.4352 | 0.7601 | 2.02 |
| Gemma-7B | **CD-RLHF** | **0.1360 (+15.3%)** | **0.6816 (+3.2%)** | **0.4144** | **0.7480** | **2.02** |
| Llama-1B | RLHF | 0.1724 | 0.6869 | 0.3997 | 0.6971 | 1.14 |
| Llama-1B | **CD-RLHF** | **0.2418 (+40.3%)** | **0.7482 (+8.9%)** | **0.3108** | **0.6847** | **1.17** |
| Llama-3B | RLHF | 0.2281 | 0.7441 | 0.3163 | 0.6658 | 3.33 |
| Llama-3B | **CD-RLHF** | **0.2920 (+28.0%)** | **0.7879 (+5.9%)** | **0.2463** | **0.6551** | **3.49** |

### Main Results: UltraFeedback Instruction Following (Table 1 Lower Half)

| Model | Method | Diversity↑ | EAD↑ | SelfBLEU↓ | SentBERT↓ | RM Score↑ |
|------|------|-----------|------|-----------|-----------|-----------|
| Gemma-2B | RLHF | 0.1686 | 0.6503 | 0.3104 | 0.7672 | -1.01 |
| Gemma-2B | **CD-RLHF** | **0.1899 (+12.6%)** | **0.7417 (+14.1%)** | **0.2858** | **0.7308** | **-0.90** |
| Gemma-7B | RLHF | 0.2345 | 0.7360 | 0.2717 | 0.7298 | 0.63 |
| Gemma-7B | **CD-RLHF** | **0.2654 (+13.2%)** | **0.7639 (+3.8%)** | **0.2442** | **0.6858** | **0.62** |
| Llama-1B | RLHF | 0.1683 | 0.6499 | 0.3564 | 0.7813 | 1.00 |
| Llama-1B | **CD-RLHF** | **0.1834 (+9.0%)** | **0.6891 (+6.0%)** | **0.3149** | **0.7598** | **0.97** |
| Llama-3B | RLHF | 0.1805 | 0.7031 | 0.3188 | 0.7676 | 1.35 |
| Llama-3B | **CD-RLHF** | **0.2223 (+23.2%)** | **0.7673 (+9.1%)** | **0.2531** | **0.7349** | **1.43** |

### OOD Generalization: MT-Bench Results (Table 2)

Performance of models trained on UltraFeedback on the OOD benchmark MT-Bench:

| Model | Method | Turn 1 | Turn 2 | Overall | Diversity↑ | SelfBLEU↓ |
|------|------|--------|--------|---------|-----------|-----------|
| Gemma-2B | RLHF | 6.26 | 4.45 | 5.35 | 0.1076 | 0.4383 |
| Gemma-2B | **CD-RLHF** | **6.91** | 4.45 | **5.68** | **0.1123** | **0.3961** |
| Gemma-7B | RLHF | 6.36 | 5.15 | 5.75 | 0.1173 | 0.4851 |
| Gemma-7B | **CD-RLHF** | **6.46** | **5.46** | **5.96** | **0.1297** | **0.4623** |
| Llama-1B | RLHF | 4.33 | 3.10 | 3.71 | 0.0818 | 0.4699 |
| Llama-1B | **CD-RLHF** | **4.78** | **3.57** | **4.18** | **0.0895** | **0.3919** |
| Llama-3B | RLHF | 6.47 | 5.47 | 5.98 | 0.0939 | 0.4489 |
| Llama-3B | **CD-RLHF** | **6.71** | 5.45 | **6.08** | **0.1133** | **0.3962** |

### Key Findings

- **Significant Diversity Gains with Preserved Alignment Quality**: Across all 8 experimental settings (4 models × 2 datasets), CD-RLHF outperforms RLHF in all four diversity metrics, while maintaining equal or slightly improved RM scores.
- **Smaller Models Benefit More**: Llama-3.2-1B achieves a 40.3% improvement in Diversity on TL;DR, as smaller models suffer more severely from mode collapse after standard RLHF.
- **Robust OOD Generalization**: On MT-Bench, the GPT-4 score completely surpasses RLHF, with CD-RLHF achieving a win rate of 21.9%–32.5% over RLHF, indicating that the diversity introduced by curiosity exploration positively transfers across domains.
- **Frequency Analysis of Intrinsic Rewards**: Top-1 activation (approx. 20% of tokens) is sufficient. Increasing the activation frequency from 20% to 60% only yields an approx. 3% improvement in diversity, whereas increasing it to 100% does not further improve diversity but causes the RM Score to drop from 0.95 to 0.88. This suggests that selective activation outperforms global activation.
- **Training Curve Characteristics**: The alignment quality of CD-RLHF converges faster than RLHF (reaching comparable performance at step 2500 vs 4640), and its diversity curve continuously climbs, whereas Sent-Rewards shows a temporary decrease in diversity during the first 1500 steps.
- **Story Writing Scaling Experiments**: On the ROC Story dataset (1817 stories), Llama-3.2-3B and Gemma-7B trained with CD-RLHF also exhibit superior lexical and semantic diversity in creative writing tasks.

## Highlights & Insights

- **Exemplary Cross-Domain Transfer**: Curiosity-driven RL has been extensively studied in gaming and robotics. This paper is the first to systematically transfer it into RLHF to tackle the diversity issue. With a clear concept and natural motivation, it represents an outstanding cross-domain innovation.
- **Ingenious Top-k Gating Design**: Instead of simply applying curiosity rewards to all tokens, activation occurs only when the model chooses a non-top-1 token. This avoids wasting exploration budgets on highly deterministic positions while ensuring that only "meaningful deviations from the norm" are rewarded. Frequency analysis indicates that 20% is the optimal ratio.
- **Natural Decay Mechanism**: The ICM prediction error naturally decays as training proceeds (novel states become "boring"), eliminating the need for manual scheduling and aligning with the theoretical intuition of curiosity-driven RL.
- **Modular and Plug-and-Play**: The ICM is trained independently of the policy model. As a self-supervised component, it can be integrated into any PPO-based alignment framework with low implementation complexity.

## Limitations & Future Work

- **Limited to the PPO Framework**: The synergy with direct preference alignment methods that bypass explicit reward models (e.g., DPO, GRPO) is not explored, leaving generalized compatibility unverified.
- **Scale Issue of Intrinsic Rewards**: The authors acknowledge that the scale of intrinsic rewards is significantly larger than that of extrinsic rewards, requiring $\eta$ to be set to a minute value. Designing an ICM with inherently aligned scale matching might yield better performance.
- **Persistent Gap Compared to SFT**: Although CD-RLHF greatly alleviates the loss of diversity caused by RLHF, the overall diversity is still lower than that of SFT models, indicating that the core trade-off is not entirely eliminated.
- **Heuristic Choice of k in Top-k**: Only $k=1$ is tested. Different models or tasks might require adaptive tuning of the $k$ threshold.
- **Unquantified Computational Overhead**: Detailed analyses on training efficiency, additional parameters, and forward computation costs introduced by ICM are absent.
- **Lacking Evaluation of Diversity Quality**: Currently used metrics (e.g., n-gram distinct, SelfBLEU) measure superficial or lexical diversity, lacking a systematic analysis of "diverse yet correct and useful" outputs.

## Related Work & Insights

- **vs Sent-Rewards (Hong et al., 2024)**: Although it also adds diversity rewards (SelfBLEU/SBERT) during RL training, it relies on heuristic reward engineering which is less effective than the systematic exploration mechanism of CD-RLHF. CD-RLHF consistently outperforms it in experiments.
- **vs f-divergence (Wang et al., 2024)**: Replacing reverse KL with forward KL improves diversity but sacrifices alignment quality, whereas CD-RLHF manages to preserve both.
- **vs Quality-Diversity (Bradley et al., 2023)**: These inference-time post-processing techniques are complementary to CD-RLHF's training-time methodology, allowing them to be theoretically stacked.
- **Future Inspirations**: The curiosity exploration framework can be transferred to scenarios such as red-teaming (to generate more diverse adversarial attacks) and data synthesis (to produce more diverse training corpora). Furthermore, the self-supervised training paradigm of ICM may inspire the design of other token-level dense rewards.

## Rating

- Novelty: ⭐⭐⭐⭐ — The first systematic transfer of curiosity-driven RL into LLM RLHF. The idea is clear and elegant, though the technical implementation (MLP + forward dynamics) is relatively straightforward.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Evaluated across 4 models, 2 datasets, and 4+ diversity metrics, including ablation studies, frequency analysis, OOD transfer, story writing scaling, and GPT-4 + human evaluations.
- Writing Quality: ⭐⭐⭐⭐ — Solidly motivated, intuitive schematics, detailed and rigorous methodology description.
- Value: ⭐⭐⭐⭐ — Addresses a practical and critical pain point of diversity reduction in RLHF. The method is scalable, plug-and-play, and the code is open-sourced.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Strategyproof Reinforcement Learning from Human Feedback](../../NeurIPS2025/llm_alignment/strategyproof_reinforcement_learning_from_human_feedback.md)
- [\[ACL 2025\] Understanding Impact of Human Feedback via Influence Functions](influence_functions_rlhf.md)
- [\[ICML 2025\] M³HF: Multi-agent Reinforcement Learning from Multi-phase Human Feedback of Mixed Quality](../../ICML2025/llm_alignment/m3hf_multi-agent_reinforcement_learning_from_multi-phase_human_feedback_of_mixed.md)
- [\[ACL 2026\] PERSA: Reinforcement Learning for Professor-Style Personalized Feedback with LLMs](../../ACL2026/llm_alignment/persa_reinforcement_learning_for_professor-style_personalized_feedback_with_llms.md)
- [\[ACL 2025\] PKU-SafeRLHF: Towards Multi-Level Safety Alignment for LLMs with Human Preference](pku-saferlhf_towards_multi-level_safety_alignment_for_llms_with_human_preference.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] The Unlearnability Phenomenon in RLVR for Language Models
description: >-
  [ICML 2026][AI Safety][RLVR] The authors identify a class of "unlearnable samples" in RLVR (GRPO) training: even when correct rollouts are sampled and reward signals are non-zero, the model fails to learn them throughout the entire training process. The root cause is not a scarcity of positive samples, clipping, or KL regularization on the optimiz
tags:
  - ICML 2026
  - AI Safety
  - RLVR
  - GRPO
date: 2026-05-08
content_hash: f497227a71009425
---
# The Unlearnability Phenomenon in RLVR for Language Models

**Conference**: ICML 2026  
**arXiv**: [2605.16787](https://arxiv.org/abs/2605.16787)  
**Code**: https://github.com/yulinchen99/unlearnability-rlvr  
**Area**: LLM Reasoning / RLVR / GRPO  
**Keywords**: RLVR, GRPO, Unlearnable Samples, Gradient Similarity, Representational Deficiencies

## TL;DR
The authors identify a class of "unlearnable samples" in RLVR (GRPO) training: even when correct rollouts are sampled and reward signals are non-zero, the model fails to learn them throughout the entire training process. The root cause is not a scarcity of positive samples, clipping, or KL regularization on the optimization side, but rather that these samples are "gradient outliers" under the initial policy, stemming from representational deficiencies that require mid-training rather than RL post-training to resolve.

## Background & Motivation

**Background**: RLVR (Reinforcement Learning with Verifiable Reward), represented by GRPO, has become a primary method for enhancing the mathematical, coding, and agent reasoning capabilities of LLMs. Intuitively, the prerequisite for GRPO to function is that "within $k$ rollouts of the same prompt, there are both positive and negative samples." Consequently, much recent work (DAPO, curriculum learning, entropy weighting, etc.) focuses on "generating positive reward signals for extremely difficult samples."

**Limitations of Prior Work**: The authors observe a counter-intuitive phenomenon—after partitioning training samples into "easy," "learnable-hard," and "unlearnable-hard" based on initial success rates, **unlearnable-hard samples** fail to show reward improvement during training even when correct rollouts are consistently observed (i.e., outcome reward is non-zero). These samples account for 30.2% of hard samples on Qwen2.5-0.5B/MATH-Easy and 21.9% on Llama-3.2-3B/MATH-Hard, indicating they are not marginal cases.

**Key Challenge**: The existing RLVR paradigm assumes that "as long as positive samples exist, the model can learn." This paper's experiments directly falsify this implicit assumption. Furthermore, common interventions on the optimization side (more positive rollouts, experience replay, higher clipping, removing KL terms) are all ineffective, suggesting the root cause lies outside optimization and requires a different explanatory framework.

**Goal**: (1) Strictly define and quantify the existence of "unlearnable samples"; (2) Systematically investigate common optimization hypotheses (scarcity of positive samples, clipping, KL regularization); (3) Provide a "representation-side" root cause to explain the phenomenon; (4) Test whether data augmentation and mid-training can resolve the issue.

**Key Insight**: The authors approach this through the lens of *cross-example gradient similarity*—calculating the gradient vector for the correct rollout of each sample and examining the cosine similarity between gradients of different samples to determine if "what is learned on one sample" can transfer to others.

**Core Idea**: Use "gradient similarity" to elevate the difference between learnable and unlearnable samples from "reward curve phenomena" to "geometric properties of the optimization space." Unlearnable samples are isolated outliers in the optimization space, reflecting inherent representational deficiencies that cannot be fixed by outcome-based RL alone.

## Method

### Overall Architecture

This paper does not propose a new algorithm but is a *diagnostic* paper: it uses controlled experiments to quantify, attribute, and localize the "unlearnability" phenomenon to the representation level. The investigation follows a four-step path: "Phenomenon Definition → Ruling out Optimization Hypotheses → Establishing Representation-side Explanation → Verifying Fixes." First, a set of samples that "have positive rewards but cannot be learned" is identified under GRPO with dynamic sampling. Then, optimization-side explanations (insufficient positive samples, clipping, KL regularization) are disproved. Instead, cross-example gradient similarity reveals these samples as isolated outliers. Finally, data augmentation and mid-training are compared as fixes, revealing that only altering the base model representation is effective.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["GRPO + Dynamic Sampling Training"] --> B["Definition and Partitioning of Unlearnable Samples<br/>Easy / Learnable-Hard / Unlearnable-Hard D_u"]
    B --> C["Disproving Optimization Hypotheses on D_u<br/>Oversampling+Replay / clip-higher / No KL / SFT / Large k=64"]
    C -->|Reward curves of D_u remain stagnant| D["Cross-Example Gradient Similarity Analysis<br/>D_u are isolated outliers in optimization space"]
    D -->|Root Cause = Representational Deficiencies, not Optimization| E["Verifying Two Fixes"]
    E -->|Data Augmentation (Similar/Sub-problems)| F["Ineffective: Gradient similarity does not increase"]
    E -->|Mid-training to change the base model| G["Effective: Gradient alignment of hard samples significantly increases"]
```

### Key Designs

**1. Working Definition and Partitioning of Unlearnable Samples: Making "Unlearnable" a Reproducible Research Object**

Prior discussions of hard samples often conflated "never sampling a positive rollout" with "failing to learn despite positive samples." Consequently, any intervention that "creates more positive samples" seemed effective. This paper isolates the latter. Specifically, after a full run of GRPO with dynamic sampling, samples with an *initial success rate* $\geq 0.1$ are classified as *easy*. For the remaining *hard* samples, the *final* pass@1 is estimated using $N=32$ rollouts. Samples with final pass@1 $<\tau=0.1$ are assigned to the unlearnable set $\mathcal{D}_u$, and others to the learnable set $\mathcal{D}_l$. Samples that never produced a positive rollout during training are explicitly excluded to ensure the research focuses on "failure to learn despite positive signals." To reduce noise, three independent runs are conducted to take the intersection for $\mathcal{D}_u/\mathcal{D}_l$ and the union for "no positive reward" samples.

**2. Oversampling-with-Replay: Disproving the "Scarcity of Positive Samples" Hypothesis**

If $\mathcal{D}_u$ is unlearnable simply because there are too few positive rollouts and the gradient is drowned out by negative samples, then providing enough positive samples should fix it. The authors re-train the model with a fixed ratio of $k_{\text{pos}}=1$ positive sample and $k-k_{\text{pos}}=7$ negative samples per prompt per batch. They sample $4k$ rollouts and downsample to $k=8$. If a batch lacks positive samples, they are reused from an experience replay buffer (up to twice per rollout). Advantages are calculated as $\hat{A}_i = \frac{\mathbb{1}[y_i=y^*] - \text{mean}}{\text{std}}$ after replay/sampling. Results show this intervention significantly slows the learning of $\mathcal{D}_l$, yet the $\mathcal{D}_u$ reward curve remains nearly identical to the baseline. Cross-validation using SFT distillation on $\mathcal{D}_u$ and large-scale $k=64$ rollouts also failed to close the gap.

**3. Cross-Example Gradient Similarity: Elevating "Unlearnability" to Optimization Space Geometry**

After excluding optimization-side explanations, a mechanism-level explanation is needed. Gradient similarity answers why learning on other samples does not transfer to $\mathcal{D}_u$ and why oversampling fails. Taking 100 samples per group, 1000 rollouts are sampled under the *initial policy* per sample. Correct rollouts are filtered to calculate GRPO loss gradients according to Equation (1) (averaging tokens within a response, then averaging across responses). To keep computation feasible, gradients are computed for a fixed randomly initialized LoRA adapter (validated to correlate highly with full-parameter gradients). The cosine similarity $\cos(g_i, g_j)$ between samples is then calculated. Figures 1c and 6 shows that *easy* samples are highly aligned, *learnable* samples are intermediate, and *unlearnable* samples show low similarity with all groups—functioning as isolated outliers. Analysis at step 50 shows this pattern persists. Reasoning-quality analysis (using GPT-5-mini to score reasoning chains from 0–5) further reveals that correct rollouts for $\mathcal{D}_u$ often rely on shortcuts/heuristics (e.g., getting the right answer for a volume problem despite incorrect logical steps), confirming that outcome rewards can reinforce "fake reasoning."

### Loss & Training

Standard GRPO with dynamic sampling is used. The GRPO objective is as follows (clipping $\varepsilon$, KL coefficient $\beta$):

$$\mathcal{L}_{\text{GRPO}}(\theta,(x,y^*)) = -\frac{1}{k}\sum_i\frac{1}{|y_i|}\sum_t \min(r_{i,t}\hat{A}_i, \text{clip}(r_{i,t},1-\varepsilon,1+\varepsilon)\hat{A}_i) - \beta\,\text{KL}(\pi_\theta\|\pi_{\text{ref}})$$

Where $r_{i,t}=\pi_\theta(y_{i,t}|x,y_{i,<t})/\pi_{\theta_{\text{old}}}(y_{i,t}|x,y_{i,<t})$. Dynamic sampling filters prompts in the current batch where $\text{std}(\{\mathbb{1}[y_i=y^*]\})=0$ to improve efficiency. Ablations include "clip-higher" and "no KL" variants. Mid-training experiments use OctoThinker-3B-Hybrid/Long-Base as the initial strategy.

## Key Experimental Results

### Main Results

**Table 1 — Proportion of Unlearnable Samples in Three Setups** (Percentages relative to total hard samples with initial pass@1 $<0.1$):

| Model / Data | $\mathcal{D}_u$ (%) | $\mathcal{D}_l$ (%) | No Positive Reward (%) |
|---|---|---|---|
| Qwen2.5-0.5B / MATH-Easy | 30.2 | 25.6 | 23.5 |
| Llama-3.2-3B-Instruct / MATH-Hard | 21.9 | 31.6 | 37.7 |
| Qwen2.5-3B / DeepScaleR | 16.7 | 14.2 | 47.2 |

Unlearnable samples are not marginal cases in any setup, standing as a major portion of hard samples alongside those with "no positive rewards."

### Ablation Study

**Comparison of Optimization-side vs. Representation/Data-side Interventions**:

| Intervention | Target Hypothesis | Effective for $\mathcal{D}_u$ | Key Observation |
|---|---|---|---|
| Oversampling + replay (1 pos/7 neg per batch) | Scarcity of positive samples | ✗ | $\mathcal{D}_l$ slowed down, $\mathcal{D}_u$ reward unchanged |
| SFT distilling correct answers on $\mathcal{D}_u$ | Lack of supervision signal | ✗ | Gap does not disappear |
| RL only on $\mathcal{D}_u$ with $k=64$ rollouts | Insufficient exploration | ✗ | Gap does not disappear |
| Clip-higher | Clipping suppresses gradients | ✗ | Clipping ratios nearly identical across groups |
| Removing KL term | KL constraint limits updates | ✗ | Reward dynamics remain unchanged |
| Data augmentation with similar problems $\mathcal{D}_u^{sim}$ | Lack of similar signals | ✗ | Sim-probs learned, original $\mathcal{D}_u$ remains unlearned |
| Data augmentation with sub-problems $\mathcal{D}_u^{sub}$ | Skills not decomposed | ✗ | Sub-probs learned faster than $\mathcal{D}_l$, original fails |
| Mid-training (OctoThinker-3B-Hybrid/Long) | Representational deficiencies | ✓ | Gradient alignment of hard samples with training distribution improves |

### Key Findings
- **Unlearnability stems from representation, not optimization**: Five optimization/data-side interventions failed. Only changing the base model representation (mid-training) worked, providing strong evidence that the problem precedes RL.
- **Gradient similarity is a strong proxy for learnability**: $\mathcal{D}_u$ samples are isolated gradient outliers, while $\mathcal{D}_l$ is intermediate and easy samples are highly aligned. This matches reward curves and persists at step 50.
- **Correct answers $\neq$ Correct reasoning**: GPT-5-mini scoring shows $\mathcal{D}_u$ correct rollouts often use shortcuts. A case study on a volume inequality problem shows the model reaching the right answer via wrong logic, highlighting reward-hacking risks in outcome-only rewards.
- **Semantic similarity $\neq$ Optimization similarity**: Problems generated by GPT-5 that are structurally "similar" do not necessarily show higher gradient similarity, and the unlearnability of $\mathcal{D}_u$ correlates with its low similarity to the training distribution.
- **Gaps widen with deeper training**: Reasoning quality for $\mathcal{D}_l$ improves from step 50 to 120, while $\mathcal{D}_u$ stagnates. Curriculum learning (learning easy + learnable first) fails to transfer improvements to $\mathcal{D}_u$.

## Highlights & Insights
- **Geometric Explanation**: By lowering "unlearnability" from reward curves to optimization space geometry, gradient similarity explains both why other updates don't transfer and why oversampling fails. LoRA-only gradients provide an efficient way to run this analysis on 0.5B-3B scales.
- **Outcome vs. Process Gap**: GPT-5-mini scoring concretizes the gap between outcome and process rewards, providing direct motivation for process supervision or verifiers.
- **Diagnostic Paradigm**: The "elimination method" (testing oversampling, SFT, large $k$, etc.) is a robust framework for studying training dynamics like forgetting or reward over-optimization.
- **Transferable Trick**: Using "gradient similarity / reasoning quality / pass@k" to characterize the "optimization properties" of training data can be used to tag samples for *trainability* before SFT, enabling smarter curriculum design.

## Limitations & Future Work
- Experiments were limited to 0.5B–3B mathematical reasoning models. Whether similar proportions of unlearnable samples exist in 30B+ models or code/agent domains is unverified.
- "Unlearnability" depends on a hard threshold ($\tau=0.1$) and $N=32$ pass@1 estimation. While reduced by multi-run intersections, a continuous sensitivity analysis of $\tau$ is missing.
- No definitive algorithm for "fixing" unlearnable samples was proposed—specific data and algorithm requirements for "mid-training" remain open questions.
- The geometric explanation for gradient similarity remains coarse; whether a low-rank subspace explains $\mathcal{D}_u$ outliers or if a "representation alignment loss" could close the gap is a logical next step.

## Related Work & Insights
- **vs. Sun et al. 2025b**: They assume fine-grained rewards make extremely hard samples learnable; Ours proves some samples remain unlearnable even with positive outcome rewards.
- **vs. Yue et al. 2025 / Wu et al. 2026**: Ours follows the line of "RL cannot teach skills not in the base model" but provides an optimization-space geometric perspective.
- **vs. DAPO / Clip-higher / No KL**: Core DAPO interventions were shown to be ineffective for $\mathcal{D}_u$, suggesting such "exploration enhancement" mainly benefits $\mathcal{D}_l$.
- **vs. OctoThinker / Mid-training (Wang et al. 2025)**: Ours provides a new motivation for mid-training—not just "making the base stronger," but "aligning gradients of hard samples with the training distribution."

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1"></div>

## Related Papers

- [\[ICML 2026\] dgMARK: Decoding-Guided Watermarking for Diffusion Language Models](dgmark_decoding-guided_watermarking_for_diffusion_language_models.md)
- [\[ICML 2026\] Forget to Know, Remember to Use: Context-Aware Unlearning for Large Language Models](forget_to_know_remember_to_use_context-aware_unlearning_for_large_language_model.md)
- [\[ICML 2026\] COFT: Counterfactual-Conformal Decoding for Fair Chain-of-Thought Reasoning in Large Language Models](coft_counterfactual-conformal_decoding_for_fair_chain-of-thought_reasoning_in_la.md)
- [\[ICML 2026\] DualOptim+: Bridging Shared and Decoupled Optimizer States for Better Machine Unlearning in Large Language Models](dualoptim_bridging_shared_and_decoupled_optimizer_states_for_better_machine_unle.md)
- [\[ICML 2026\] Towards Fine-Grained Robustness: Attention-Guided Test-Time Prompt Tuning for Vision-Language Models](towards_fine-grained_robustness_attention-guided_test-time_prompt_tuning_for_vis.md)

</div>

<!-- RELATED:END -->

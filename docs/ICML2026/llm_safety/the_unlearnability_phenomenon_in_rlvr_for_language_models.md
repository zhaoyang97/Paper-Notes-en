---
title: >-
  [Paper Note] The Unlearnability Phenomenon in RLVR for Language Models
description: >-
  [ICML 2026][LLM Safety][RLVR] The authors identify a class of "unlearnable examples" in RLVR (GRPO) training: even when correct rollouts are sampled and reward signals are non-zero, the model fails to learn them throughout the training process. The root cause is not a scarcity of positive samples, clipping, or KL regularization on the optimization
tags:
  - ICML 2026
  - LLM Safety
  - RLVR
  - GRPO
date: 2026-05-08
content_hash: e3b0ecd87d5803b8
---
# The Unlearnability Phenomenon in RLVR for Language Models

**Conference**: ICML 2026  
**arXiv**: [2605.16787](https://arxiv.org/abs/2605.16787)  
**Code**: https://github.com/yulinchen99/unlearnability-rlvr  
**Area**: LLM Reasoning / RLVR / GRPO  
**Keywords**: RLVR, GRPO, Unlearnable Examples, Gradient Similarity, Representation Defects

## TL;DR
The authors identify a class of "unlearnable examples" in RLVR (GRPO) training: even when correct rollouts are sampled and reward signals are non-zero, the model fails to learn them throughout the training process. The root cause is not a scarcity of positive samples, clipping, or KL regularization on the optimization side; rather, these samples are "gradient outliers" under the initial policy, reflecting representation defects that require mid-training instead of RL post-training to rectify.

## Background & Motivation

**Background**: Reinforcement Learning with Verifiable Reward (RLVR), represented by GRPO, has become a primary method for enhancing LLM mathematical, code, and agent reasoning capabilities. Intuitively, the prerequisite for GRPO to work is that "among $k$ rollouts of the same prompt, both positive and negative samples exist." Consequently, much recent work (DAPO, curriculum, entropy weighting, etc.) has focused on "creating positive reward signals for extremely difficult samples."

**Limitations of Prior Work**: The authors discover an anti-intuitive phenomenon—after partitioning training samples into "Easy," "Learnable Hard," and "Unlearnable Hard" based on initial success rates, the **unlearnable hard samples** fail to show reward improvement during training, even when correct rollouts (i.e., non-zero outcome rewards) are consistently observed. This category accounts for 30.2% of hard samples in Qwen2.5-0.5B/MATH-Easy and 21.9% in Llama-3.2-3B/MATH-Hard, indicating it is not a marginal phenomenon.

**Key Challenge**: The existing RLVR paradigm assumes that "as long as positive samples are available, the model can learn." However, the experiments in this paper disprove this implicit assumption. Furthermore, common optimization-side interventions (more positive rollouts, experience replay, higher clipping, removing KL terms) are all ineffective, suggesting the root cause lies elsewhere and requires a different explanatory framework.

**Goal**: (1) Strictly define and quantify the existence of "unlearnable samples"; (2) systematically investigate common optimization-side hypotheses (scarcity of positive samples, clipping, KL regularization); (3) provide a "representation-side" root cause that explains the phenomenon; (4) examine whether data augmentation and mid-training can fix the issue.

**Key Insight**: The authors approach this from the perspective of *cross-example gradient similarity*—calculating the gradient vector for the correct rollout of each sample and examining the cosine similarity between gradients of different samples to determine if learning from one sample can transfer to others.

**Core Idea**: Gradient similarity is used to elevate the difference between learnable and unlearnable samples from a "reward curve phenomenon" to "geometric properties of the optimization space." Unlearnable samples are isolated outliers in the optimization space, reflecting defects in the model's internal representation that cannot be fixed by outcome-based RL alone.

## Method

### Overall Architecture

This paper does not propose a new algorithm but is a *diagnostic* study: it quantifies, attributes, and finally locates the "unlearnability" phenomenon at the representation level. The investigation follows a four-step logic: "Phenomenon Definition $\rightarrow$ Exclude Optimization Hypotheses $\rightarrow$ Establish Representation Explanation $\rightarrow$ Verify Solutions." First, a set of samples that "have positive rewards but cannot be learned" is identified using GRPO + dynamic sampling. Then, optimization explanations like "insufficient positive samples/clipping/KL regularization" are disproven. Instead, cross-example gradient similarity is used to prove these samples are isolated outliers. Finally, mid-training is found to be the only effective fix compared to data augmentation.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["GRPO + Dynamic Sampling Training"] --> B["Definition and Three-Way Split of Samples<br/>Easy / Learnable Hard / Unlearnable Hard D_u"]
    B --> C["Disproving Optimization Hypotheses on D_u<br/>Oversampling+Replay / clip-higher / No KL / SFT / Large k=64"]
    C -->|D_u reward curves remain flat| D["Cross-Example Gradient Similarity Analysis<br/>D_u are isolated outliers in optimization space"]
    D -->|Root cause = Representation defect, not optimization| E["Testing Two Rectification Paths"]
    E -->|Data Augmentation (Similar/Sub-problems)| F["Ineffective: Gradient similarity does not rise"]
    E -->|Mid-training to change base model| G["Effective: Alignment of hard sample gradients significantly improved"]
```

### Key Designs

**1. Definition and Three-way Split: Transforming "Failure to Learn" into a Well-defined Research Object**

Prior discussions of hard samples often conflated "never sampling a positive rollout" with "having positive rollouts but failing to learn." To separate the latter, the authors first perform a full GRPO + dynamic sampling run. Samples with an *initial success rate* $\geq 0.1$ are categorized as *easy*. For the remaining *hard* samples, the *final* pass@1 is estimated using $N=32$ rollouts. Samples with final pass@1 $<\tau=0.1$ are assigned to the unlearnable set $\mathcal{D}_u$, while others go to the learnable set $\mathcal{D}_l$. Samples that never yielded a positive rollout are explicitly excluded. This ensures the research focus is strictly on "samples with positive reward signals that remain unlearned."

**2. Oversampling-with-Replay: Disproving the "Positive Sample Scarcity" Hypothesis**

If $\mathcal{D}_u$ remains unlearned simply because positive rollouts are too rare, then forcing sufficient positive samples should solve the problem. The authors use a fixed ratio of $k_{\text{pos}}=1$ positive sample and $k-k_{\text{pos}}=7$ negative samples per prompt per batch. If a batch lacks positive samples, they are reused from an experience replay buffer (up to twice). Advantage is calculated as $\hat{A}_i = \frac{\mathbb{1}[y_i=y^*] - \text{mean}}{\text{std}}$ after oversampling. Results show that while this dragging down the learning speed of $\mathcal{D}_l$, the $\mathcal{D}_u$ curve remains identical to the baseline. Aggressive variants like SFT distillation on $\mathcal{D}_u$ and $k=64$ rollouts also fail to bridge the gap.

**3. Cross-Example Gradient Similarity: Elevating "Unlearnability" to Optimization Space Geometry**

Having excluded optimization-side factors, the authors use gradient similarity to explain why learning does not transfer. 100 samples are taken from each group, and 1000 rollouts are sampled under the *initial policy* to filter correct ones. Gradients for the GRPO loss are calculated according to Equation (1). To manage computation, a fixed randomly initialized LoRA adapter is used, and gradients are computed only for LoRA parameters. Figure 1c/Figure 6 show that *easy* samples are highly aligned, *learnable* ones are intermediate, and *unlearnable* ones exhibit low similarity with all groups—meaning each unlearnable sample is an isolated outlier. Reasoning quality analysis (rating chains from 0–5 via GPT-5-mini) further reveals that correct rollouts in $\mathcal{D}_u$ often rely on shortcuts or heuristics, confirming that outcome reward might inadvertently reward "fake reasoning."

### Loss & Training

Standard GRPO with dynamic sampling is used. The GRPO objective is as follows (clipping $\varepsilon$, KL coefficient $\beta$):

$$\mathcal{L}_{\text{GRPO}}(\theta,(x,y^*)) = -\frac{1}{k}\sum_i\frac{1}{|y_i|}\sum_t \min(r_{i,t}\hat{A}_i, \text{clip}(r_{i,t},1-\varepsilon,1+\varepsilon)\hat{A}_i) - \beta\,\text{KL}(\pi_\theta\|\pi_{\text{ref}})$$

where $r_{i,t}=\pi_\theta(y_{i,t}|x,y_{i,<t})/\pi_{\theta_{\text{old}}}(y_{i,t}|x,y_{i,<t})$. Dynamic sampling filters prompts where $\text{std}(\{\mathbb{1}[y_i=y^*]\})=0$ in the current batch. Variants used for ablation include higher clipping and removal of the KL term. Mid-training experiments use OctoThinker-3B-Hybrid/Long-Base as the initial policy.

## Key Experimental Results

### Main Results

**Table 1 — Distribution of Unlearnable Samples in Three Setups** (percentages relative to the total number of hard samples with initial pass@1 $<0.1$):

| Model / Data | $\mathcal{D}_u$ (%) | $\mathcal{D}_l$ (%) | No Positive Reward (%) |
|---|---|---|---|
| Qwen2.5-0.5B / MATH-Easy | 30.2 | 25.6 | 23.5 |
| Llama-3.2-3B-Instruct / MATH-Hard | 21.9 | 31.6 | 37.7 |
| Qwen2.5-3B / DeepScaleR | 16.7 | 14.2 | 47.2 |

Unlearnable samples are a significant proportion in all setups, comparable in scale to samples that never receive positive rewards.

### Ablation Study

**Comparison of Optimization-side vs. Representation/Data-side Interventions**:

| Intervention | Hypothesis Target | Effective for $\mathcal{D}_u$ | Key Observation |
|---|---|---|---|
| Oversampling + replay (1 pos, 7 neg) | Scarcity of positive samples | ✗ | $\mathcal{D}_l$ slowed down, but $\mathcal{D}_u$ curve unchanged |
| SFT distillation of correct answers on $\mathcal{D}_u$ | Lack of supervision | ✗ | Gap does not vanish |
| RL on $\mathcal{D}_u$ only with $k=64$ rollouts | Insufficient exploration | ✗ | Gap does not vanish |
| Clip-higher | Gradient suppression by clipping | ✗ | Clipping ratios for all three groups nearly overlap |
| Remove KL term | KL constraint limits updates | ✗ | Reward dynamics remain the same |
| Similar question $\mathcal{D}_u^{sim}$ augmentation | Lack of related signals | ✗ | Augmented questions are learned; original $\mathcal{D}_u$ remains unlearned |
| Sub-problem $\mathcal{D}_u^{sub}$ augmentation | Skills not decomposed | ✗ | Sub-problems learned faster than $\mathcal{D}_l$; original problem remains unlearned |
| Mid-training (OctoThinker-3B-Hybrid/Long) | Representation defects | ✓ | Alignment of hard samples to training distribution significantly increased |

### Key Findings
- **Unlearnability stems from representation, not optimization**: Five types of optimization/data-side interventions failed; only changing the base model representation (mid-training) was effective, pointing strongly to pre-RL issues.
- **Gradient similarity is a strong proxy for learnability**: $\mathcal{D}_u$ are isolated gradient outliers, while $\mathcal{D}_l$ and easy samples show higher alignment. This aligns with reward curve groupings and persists even at step 50.
- **Correct answers $\neq$ Correct reasoning**: GPT-5-mini scores show $\mathcal{D}_u$ correct rollouts rely on shortcuts. A case study on volume inequality shows the model arriving at correct answers through flawed logic, highlighting the risk of reward-hacking with outcome rewards.
- **Semantic similarity $\neq$ Optimization similarity**: Synthesized "semantically similar" questions do not necessarily increase gradient similarity. $\mathcal{D}_u$ remains an "isolated peak" in the optimization space that semantic augmentation cannot move.
- **Gaps widen with deeper training**: Reasoning quality for $\mathcal{D}_l$ improves consistently from step 50 to 120, while $\mathcal{D}_u$ stagnates. Curriculum learning fails to transfer improvements to $\mathcal{D}_u$.

## Highlights & Insights
- **Shifting the focus from reward curves to optimization space geometry**: Gradient similarity explains both why learning does not transfer and why oversampling fails. Using LoRA-only gradients makes this analysis feasible at the 0.5B-3B scale.
- **Empirical proof that correct answers do not equal correct reasoning**: Using GPT-5-mini to score chains provides a concrete motivation for process supervision or mid-step verifiers, as outcome-only rewards treat reward-hacked rollouts as valid signals.
- **Effective reporting of negative results**: The systematic exclusion of hypotheses (oversampling, SFT, large $k$, etc.) provides a diagnostic paradigm that can be applied to other training dynamics like forgetting in SFT or over-optimization in RLHF.
- **Transferable trick**: Using the "gradient similarity / reasoning quality / pass@k" triad to characterize the "optimization properties" of training data allows for smarter data filtering and curriculum design before SFT.

## Limitations & Future Work
- Experiments are restricted to 0.5B–3B scale math reasoning models; it is unverified whether the same proportion of unlearnable samples exists in 30B+ models or code/agent domains.
- "Unlearnability" depends on a hard threshold $\tau=0.1$ and $N=32$ pass@1 estimates; boundary cases may be slightly stochastic.
- No immediate "fix" algorithm is provided—what specific data or algorithms in mid-training are most effective remains an open question.
- The conclusion that "similar question augmentation is ineffective" depends on the synthesis quality of GPT-5; inaccuracies in synthesis might overestimate the disconnect between semantic and optimization similarity.
- Geometric explanations remain coarse; researching whether a low-rank subspace explains the outliers or designing representation alignment losses are potential next steps.

## Related Work & Insights
- **vs. Sun et al. 2025b (Fine-grained reward assignment)**: While Sun et al. assume better reward design makes any sample learnable, this paper proves some samples remain unlearned even with positive rewards, shifting the focus to representation.
- **vs. Yue et al. 2025 / Wu et al. 2026 (RL cannot teach new skills not in the base model)**: This paper follows the same "ceiling of RL" theme but provides a micro-level, quantifiable perspective on which specific samples are excluded and their geometric features.
- **vs. DAPO / clip-higher / no KL**: Core interventions in DAPO (high clipping, no KL) were ablated here and shown to benefit $\mathcal{D}_l$ primarily, rather than $\mathcal{D}_u$.
- **vs. OctoThinker / Mid-training (Wang et al. 2025)**: This paper provides a new motivation for mid-training—not just "making the base stronger," but "aligning hard samples with the training distribution's gradients."
- **vs. Nikankin et al. "Bag of heuristics"**: The reasoning quality analysis confirms that LLMs often use heuristic mosaics, specifically identifying this as the dominant behavior for unlearnable samples under outcome rewards.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to systematically characterize "positive reward but unlearnable" and provide a geometric gradient explanation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three models across two data scales with a complete hypothesis exclusion chain, though limited to $\leq$ 3B models.
- Writing Quality: ⭐⭐⭐⭐⭐ The "elimination method" for negative results is exceptionally clear; case studies and charts are well-coordinated.
- Value: ⭐⭐⭐⭐⭐ Directly challenges the "positive reward $\Rightarrow$ learnable" assumption, providing evidence for data filtering, mid-training, and process reward research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] dgMARK: Decoding-Guided Watermarking for Diffusion Language Models](dgmark_decoding-guided_watermarking_for_diffusion_language_models.md)
- [\[ICML 2026\] Forget to Know, Remember to Use: Context-Aware Unlearning for Large Language Models](forget_to_know_remember_to_use_context-aware_unlearning_for_large_language_model.md)
- [\[ICML 2026\] COFT: Counterfactual-Conformal Decoding for Fair Chain-of-Thought Reasoning in Large Language Models](coft_counterfactual-conformal_decoding_for_fair_chain-of-thought_reasoning_in_la.md)
- [\[ACL 2026\] Topic-Based Watermarks for Large Language Models](../../ACL2026/llm_safety/topic-based_watermarks_for_large_language_models.md)
- [\[ICML 2026\] Towards Fine-Grained Robustness: Attention-Guided Test-Time Prompt Tuning for Vision-Language Models](towards_fine-grained_robustness_attention-guided_test-time_prompt_tuning_for_vis.md)

</div>

<!-- RELATED:END -->

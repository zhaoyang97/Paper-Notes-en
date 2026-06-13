---
title: >-
  [Paper Note] Better Literary Translation: A Multi-Aspect Data Generation and LLM Training Approach
description: >-
  [ACL2026][LLM Alignment][Literary Translation] This paper decomposes literary translation quality into two dimensions: "Fluency of Expression" and "Literary Effect." By using specialized LLMs to iteratively generate high…
tags:
  - "ACL2026"
  - "LLM Alignment"
  - "Literary Translation"
  - "Multi-dimensional Data Generation"
  - "Preference Optimization"
  - "Reward Model"
  - "GRPO"
date: 2026-05-08
content_hash: f969161d4ccbab4c
---

# Better Literary Translation: A Multi-Aspect Data Generation and LLM Training Approach

**Conference**: ACL2026  
**arXiv**: [2606.05924](https://arxiv.org/abs/2606.05924)  
**Code**: No public repository; the paper provides details on data generation and evaluation prompts.  
**Area**: LLM Alignment / Literary Machine Translation  
**Keywords**: Literary Translation, Multi-dimensional Data Generation, Preference Optimization, Reward Model, GRPO

## TL;DR
This paper decomposes literary translation quality into two dimensions: "Fluency of Expression" and "Literary Effect." By using specialized LLMs to iteratively generate high-quality reference translations and preference pairs, and employing SFT + Explicit Reward Model + GRPO to train LitMT, 8B/14B small models achieve performance on English-to-Chinese literary translation that approaches or even exceeds some large-scale models.

## Background & Motivation
**Background**: Literary translation differs from general machine translation as it requires not only semantic accuracy but also the preservation of metaphors, rhetoric, tone, and narrative style. Recent approaches include using large models or long CoT for detailed translations, or using LLM-as-a-judge for scoring in reinforcement learning, exemplified by models like DRT, DeepTrans, and ExTrans.

**Limitations of Prior Work**: While effective, these methods are costly. Long CoT significantly increases inference latency (up to $10\times$ as noted in the paper). Using strong LLMs as reward functions in RL requires calling expensive evaluator models for every sample during training, making data and reward signals difficult to reuse.

**Key Challenge**: Literary translation quality is not a simple scalar. Increasing natural fluency may sacrifice the metaphors and defamiliarization of the original text, while overemphasizing literary effects can result in stiff phrasing that violates target language conventions. Directly asking a model to "comprehensively optimize" merges these goals, making it difficult to obtain stable, trainable supervisory signals.

**Goal**: The authors aim to generate reusable high-quality reference translations and preference data offline, then use them to train smaller specialized translation models. Specifically, the goals include: improving English-to-Chinese literary translation quality on MetaphorTrans; comparing the differences between implicit preference optimization (DPO series) and explicit reward models + GRPO; and ensuring the final model does not rely on long CoTs or online LLM evaluation for real-time translation.

**Key Insight**: Literary translation can be decomposed into at least two discussable dimensions: fluency of expression and preservation of literary effects. Instead of sampling multiple translations for a judge to choose from, it is more effective to have different "translators" rewrite towards specific dimensions, followed by an aggregator for fusion and an evaluator for scoring and recording the iteration trajectory.

**Core Idea**: Use "Multi-dimensional Specialized Translators + Iterative Evaluator Aggregator" to generate translation references and preference pairs superior to the original labels, then use an explicit reward model to convert these preference signals into stable rewards for GRPO training.

## Method
The method consists of two stages: Phase 1 is offline data generation, expanding source sentences from MetaphorTrans into high-quality references and preference pairs. Phase 2 is model training, comparing SFT, DPO/SimPO/CPO, and Reward Model + GRPO. Crucially, expensive strong LLMs are only used for one-time data construction; the final LitMT model requires neither long CoT nor external LLM calls during inference.

### Overall Architecture
The input is an English literary text $S$. The system uses a general translator to generate an initial translation $T_0$, then enters an iterative loop: the Evaluator scores the current best translation and identifies issues; the Expression Optimizer focuses on correcting naturalness in Chinese; the Literary Effect Preserver focuses on metaphors, rhetoric, and tone; the Aggregator synthesizes the two rewrites into a new translation; the Evaluator scores again, updating the best translation if improved. The loop stops if the score exceeds $\tau=4.9$, the maximum iterations $K=8$ are reached, or no improvement occurs for $N=3$ rounds.

Generation yields two products: the best translation $T^*$ for SFT, and preference pairs $(T_w, T_l)$ derived from the iteration history where higher-scored translations are preferred. Training begins with SFT on high-quality references, followed by a comparison of implicit preference optimization vs. explicit reward modeling. The optimal approach involves training an 8B reward model from the SFT checkpoint and continuing policy optimization via GRPO with a composite reward.

### Key Designs
1.  **Multi-aspect Iterative Refinement**:
    - **Function**: Decomposes quality optimization into fluency and literary effect sub-goals handled by different LLM modules.
    - **Mechanism**: The Expression Optimizer ensures Chinese translations are natural and concise; the Literary Effect Preserver protects metaphors and literary tension; the Aggregator balances the two. The Evaluator provides specific feedback for the next revision.
    - **Design Motivation**: Random sampling + judge only selects among candidates. Multi-dimensional decomposition allows explicit exploration between "readability" and "literarity," naturally forming a quality hierarchy in the iteration trajectory.

2.  **Constructing Preference Data from Iteration History**:
    - **Function**: Converts various quality translations from each round into reusable preference pairs rather than just keeping the final answer.
    - **Mechanism**: For all evaluated translations under a source sentence, if $score(T_w)>score(T_l)$, a preference pair $(T_w,T_l)$ is constructed. This provides multiple fine-grained comparison signals per sample.
    - **Design Motivation**: Subtle quality differences are crucial in literary translation. While a single reference tells the model "what to output," preference pairs teach "which details are better."

3.  **Explicit Reward Model + Composite GRPO Reward**:
    - **Function**: Replaces repetitive LLM-as-a-judge calls with a stable, low-cost local reward model.
    - **Mechanism**: An LLM's language modeling head is replaced with a linear scalar head, trained using Bradley-Terry loss with an added penalty $\lambda(r_w+r_l)^2$ to center rewards. In the GRPO stage, $G=16$ translations are sampled per source sentence, using $r(x,y)=r_{RM}+0.05\cdot r_{BLEU}+r_{fmt}$.
    - **Design Motivation**: The authors found DPO methods unstable or even detrimental for this task. Online exploration with an explicit RM via GRPO better utilizes preference data while avoiding the high cost of step-wise strong LLM calls.

### Loss & Training
SFT uses the best translations $y^*$ for cross-entropy training: $\mathcal{L}_{SFT}=-\mathbb{E}_{(x,y^*)}[\log \pi_\theta(y^*|x)]$. Explicit reward models utilize the Bradley-Terry objective $\mathcal{L}_{RM}=-\mathbb{E}[\log\sigma(r_w-r_l)]+\lambda(r_w+r_l)^2$, with $\lambda=0.01$.

Experiments utilize 19,264 training samples and 2,000 test samples from MetaphorTrans. Preference data comprises 179,588 training pairs. LitMT-8B and LitMT-14B are trained from Qwen3-8B/14B-Base. SFT and DPO use a learning rate of $1e^{-5}$ for 3 epochs. GRPO starts from the SFT checkpoint with $1e^{-7}$ learning rate, $\beta=0.01$, and $G=16$.

## Key Experimental Results

### Main Results
The paper uses Claude Opus 4.5 as the primary evaluator, reporting CRF, CEA5, and CEA100 (primary metric).

| Model | Parameter Scale | CEA100 | Remarks |
| :--- | :--- | :--- | :--- |
| Qwen3-8B | 8B | 52.77 | General model |
| DRT-14B | 14B | 58.43 | Specialized literary MT model |
| DeepTrans-7B | 7B | 61.15 | Specialized model using RL |
| ExTrans-7B | 7B | 62.95 | Strong specialized baseline |
| Qwen3-235B-A22B | 235B / 22B act. | 65.62 | Teacher for data generation |
| LitMT-8B | 8B | 67.25 | Ours |
| Claude Sonnet 4.5 | - | 68.43 | Strong closed-source model |
| GPT-5.2 | - | 68.68 | Strong closed-source model |
| LitMT-14B | 14B | 69.07 | Ours |
| Claude Opus 4.5 | - | 73.30 | Strongest evaluator model |

On the out-of-domain O. Henry Collection, LitMT-8B achieves 70.38 CEA100, significantly exceeding Qwen3-32B (65.81), indicating generalization to early American English narrative styles.

### Ablation Study
Training strategy ablations show significant differences among preference optimization methods.

| Training Method | CRF | CEA5 | CEA100 | Conclusion |
| :--- | :--- | :--- | :--- | :--- |
| SFT | 72.66 | 3.54 | 65.74 | High-quality references are strong |
| SimPO | 69.98 | 3.41 | 62.62 | Lower than SFT |
| DPO | 70.50 | 3.44 | 63.39 | Lower than SFT |
| RM+GRPO | 73.03 | 3.61 | 67.25 | +1.51 points over SFT |

Multi-dimensional data generation also shows clear contributions.

| Data/Module Config | CEA100 | Key Insight |
| :--- | :--- | :--- |
| DRT Ground Truth | 57.09 | Original labels are weak SFT targets |
| Single Distillation (Qwen3-235B) | 61.08 | Teacher exceeds GT but is insufficient |
| Multi-dimensional Refinement | 65.74 | +4.66 over single distillation |

### Key Findings
- LitMT-8B (67.25 CEA100) exceeds its teacher Qwen3-235B (65.62), suggesting iterative refinement does not just distill but improves target quality.
- DPO, CPO, and SimPO perform 2-3 points lower than SFT, indicating that preference pairs used directly for implicit policy optimization are unstable in literary translation.
- RM+GRPO outperforms SFT by 1.51 points, showing that explicit reward models with online sampling better leverage quality variances in iteration history.

## Highlights & Insights
- The strongest aspect is turning "data generation" into an interpretable translation workflow rather than relying on an LLM to produce answers out of thin air. The decomposition of fluency and literary effect aligns well with the inherent contradictions of literary translation.
- A key insight is the "student exceeding teacher" phenomenon: LitMT-8B outperforms the 235B teacher on CEA100, indicating that high-quality iterative data can reorganize large model capabilities into superior small model behaviors.
- The failure of the DPO series suggests that preference data is not universally suitable for all tasks; for fine-grained linguistic quality comparisons, explicit RM and online exploration may be more robust than closed-form preference objectives.

## Limitations & Future Work
- Validated only on English-Chinese translation; prompt transferability to other language pairs or poetry remains unclear.
- Evaluation relies heavily on LLM judges; while consistency analysis was performed, expert human validation is still needed.
- The theoretical explanation for DPO degradation is insufficient and requires further analysis regarding preference distribution and reward misspecification.
- Multi-round refinement for 19,264 samples involves significant one-time costs, which may be a barrier for smaller teams.

## Related Work & Insights
- **vs. DRT**: DRT uses long CoT for synthesis; this work uses multi-dimensional short-output refinement, reducing inference costs while providing reusable preference pairs.
- **vs. DeepTrans / ExTrans**: These rely on LLM-as-a-judge during RL; this work moves the judge role to offline generation and RM training, using a local composite reward for GRPO.
- **vs. Self-Refinement**: Standard refinement uses one model; this multi-agent approach mimics professional translation workflows (polishing vs. effect preservation).

## Rating
- Novelty: ⭐⭐⭐⭐☆ 
- Experimental Thoroughness: ⭐⭐⭐⭐☆ 
- Writing Quality: ⭐⭐⭐⭐☆ 
- Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] MDP-GRPO: Stabilized Group Relative Policy Optimization for Multi-Constraint Instruction Following](mdp-grpo_stabilized_group_relative_policy_optimization_for_multi-constraint_inst.md)
- [\[NeurIPS 2025\] Limited Preference Data? Learning Better Reward Model with Latent Space Synthesis](../../NeurIPS2025/llm_alignment/limited_preference_data_learning_better_reward_model_with_latent_space_synthesis.md)
- [\[ICLR 2026\] JailNewsBench: Multi-Lingual and Regional Benchmark for Fake News Generation under Jailbreak Attacks](../../ICLR2026/llm_alignment/jailnewsbench_multi-lingual_and_regional_benchmark_for_fake_news_generation_unde.md)
- [\[ACL 2026\] Too Correct to Learn: Reinforcement Learning on Saturated Reasoning Data](too_correct_to_learn_reinforcement_learning_on_saturated_reasoning_data.md)
- [\[AAAI 2026\] LaF-GRPO: In-Situ Navigation Instruction Generation for the Visually Impaired via GRPO with LLM-as-Follower Reward](../../AAAI2026/llm_alignment/laf-grpo_in-situ_navigation_instruction_generation_for_the_visually_impaired_via.md)

</div>

<!-- RELATED:END -->

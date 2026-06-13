---
title: >-
  [Paper Note] MDP-GRPO: Stabilized Group Relative Policy Optimization for Multi-Constraint Instruction Following
description: >-
  [ACL2026][LLM Alignment][GRPO] MDP-GRPO addresses the instability of GRPO under discrete, low-variance rewards in multi-constraint instruction following. By integrating multi-temperature sampling, dual-anchor advantage…
tags:
  - "ACL2026"
  - "LLM Alignment"
  - "GRPO"
  - "Verifiable Rewards"
  - "Multi-Constraint Instructions"
  - "Advantage Stabilization"
  - "Prospect Theory"
date: 2026-05-08
content_hash: 9e4dedeaf68e34fc
---

# MDP-GRPO: Stabilized Group Relative Policy Optimization for Multi-Constraint Instruction Following

**Conference**: ACL2026  
**arXiv**: [2606.06058](https://arxiv.org/abs/2606.06058)  
**Code**: https://github.com/m-salmani78/MDP-GRPO  
**Area**: LLM Alignment / RLVR / Multi-Constraint Instruction Following  
**Keywords**: GRPO, Verifiable Rewards, Multi-Constraint Instructions, Advantage Stabilization, Prospect Theory

## TL;DR
MDP-GRPO addresses the instability of GRPO under discrete, low-variance rewards in multi-constraint instruction following. By integrating multi-temperature sampling, dual-anchor advantage, prospect-theoretic shaping, and asymmetric KL divergence, it enables small models to achieve more stable soft/hard constraint satisfaction rates on IFEval, FollowBench, and custom multi-constraint test sets.

## Background & Motivation
**Background**: LLMs are capable of following many natural language instructions. However, when a request contains multiple explicit constraints (e.g., format, vocabulary, case, ending phrases, structured output), models often fail to satisfy all of them. In real-world deployment—such as legal templates, product copy, developer tool outputs, and security policies—these multi-constraint prompts are common, and "missing one makes it unusable."

**Limitations of Prior Work**: Reinforcement Learning with Verifiable Rewards (RLVR) is well-suited for such tasks because each constraint can be deterministically checked by rule-based verifiers, avoiding the bias of learned reward models or LLM-as-a-judge. However, these rewards are typically discrete, sparse, and low-variance. GRPO relies on within-group z-score normalization of multiple samples for the same prompt; when the reward distribution within a group is overly homogeneous, the advantage estimation suffers from three pathological failure modes.

**Key Challenge**: The within-group relative normalization of GRPO only identifies "who is better in the group" but tends to ignore the absolute reward level. In the early stages of multi-constraint tasks, all samples might be equally wrong or equally right, leading to zero intra-group variance. Even when variance is non-zero, it can amplify minute differences into excessive gradients. Models need to retain within-group comparison signals while simultaneously knowing "how far they are from full constraint satisfaction."

**Goal**: The authors aim to stabilize GRPO without introducing a critic, making it suitable for deterministic multi-constraint rewards. Goals include reducing homogeneous groups, restoring learning signals in zero-variance groups, controlling the magnitude of advantage updates, and more conservatively constraining policy drift when punishing constraint violations.

**Key Insight**: Instead of modifying the rewards themselves, the paper focuses on sampling and advantage estimation. Multi-temperature sampling prevents homogeneous groups; dual-anchor advantage injects absolute goal levels into the advantage estimate; prospect shaping uses loss aversion from behavioral economics to limit update magnitudes; and asymmetric KL divergence more strongly constrains policy deviation during negative advantage updates.

**Core Idea**: The standard single within-group z-score advantage of GRPO is extended into a hybrid advantage consisting of a "within-group relative signal + goal-aware anchor signal." This is passed through bounded, asymmetric shaping before the policy update, mitigating low-variance amplification, mean-centering blindness, and zero-variance collapse simultaneously.

## Method

### Overall Architecture
MDP-GRPO follows the critic-free group-based policy optimization used in GRPO. Given an instruction $x$, the model generates $G$ completions. The reward for each completion is the ratio of satisfied constraints $r(x,y)=\frac{1}{C(x)}\sum_t c_t(x,y)$. While standard GRPO uses the group mean and standard deviation to calculate advantage, MDP-GRPO inserts three stabilization modules: multi-temperature sampling to generate more diverse responses, dual-calculation of group-relative signals $z_i$ and goal-aware signals $\delta_i$, and bounded prospect-theoretic shaping to obtain the final signal used in a clipped GRPO objective with asymmetric KL coefficients based on the sign of $A_i$.

Regarding data, the authors constructed 3,000 training prompts. Approximately one-third of the seed prompts are from existing data, while the rest are manually curated, covering general Q&A, creative writing, and material assistance. Each instruction is injected with 1-6 constraints, with a taxonomy covering 9 high-level categories and 26 constraint types, verified through deterministic validators such as regex and parsers.

### Key Designs
1. **Multi-Temperature Group Sampling**:

	- **Function**: Reduces the probability of $G$ completion rewards being identical for the same prompt.
	- **Mechanism**: Standard GRPO typically uses a fixed temperature for sampling the entire group. MDP-GRPO employs a temperature schedule $\mathbf{T}=[\tau_1,...,\tau_G]$ for different samples in a group, e.g., $[0.1, 0.4, 0.7, 1.0]$. Low-temperature samples provide high-quality exploitation, while high-temperature samples increase exploration, making different constraint satisfaction patterns more likely.
	- **Design Motivation**: The root cause of zero-variance collapse is the lack of reward differentiation within a group, leaving the advantage without direction. Multi-temperature sampling improves reward dispersion at the data generation stage without changing the objective function, which is particularly beneficial for small group sizes like $G=4$.

2. **Dual-Anchor Advantage**:

	- **Function**: Retains both relative intra-group comparison and absolute performance relative to the goal.
	- **Mechanism**: The standard group signal is $z_i=(r_i-\mu_{group})/(\sigma_{group}+\epsilon)$. MDP-GRPO introduces a goal-aware anchor: assuming a neutral baseline where each constraint is satisfied independently with $p=0.5$, the reward target center is $\mu_{goal}=0.5$ and the standard deviation is $\sigma_{goal}=1/(2\sqrt{C(x)})$. The absolute advantage is expressed as $\delta_i=2\sqrt{C(x)}(r_i-0.5)$. The final advantage is a mixture of shaped $z_i$ and $\delta_i$.
	- **Design Motivation**: Mean-centering blindness causes "all-wrong groups" and "all-right groups" to look identical after normalization. The goal-aware anchor informs the model whether a completion is above or below the neutral constraint satisfaction level, restoring directional signals even when all group samples are identical.

3. **Prospect-Theoretic Shaping and Asymmetric KL**:

	- **Function**: Limits excessive updates caused by low-variance amplification and imposes stronger penalties on constraint violations.
	- **Mechanism**: A scaled tanh transformation is applied to the raw advantage signal, with a larger upper bound on the negative side, i.e., $\lambda_->\lambda_+>0$. Experiments use $(\lambda_+,\lambda_-)=(1.25,2.0)$ and $\beta_{PT}=0.8$, making positive gains exhibit diminishing returns and negative violations more "painful." Additionally, asymmetric KL uses a higher coefficient $\beta^{high}_{KL}=0.025$ when $A_i<0$ and $\beta^{low}_{KL}=0.01$ when $A_i\ge 0$.
	- **Design Motivation**: In multi-constraint tasks, regressing on a single constraint can make an output unusable; thus, negative updates should be more conservative. Bounded tanh prevents advantage explosion, and loss aversion emphasizes correcting samples that violate constraints.

### Loss & Training
Training utilizes the standard GRPO clipped surrogate, replacing the advantage and optionally using asymmetric KL. Models tested include Gemma-2-2B-Instruct and Llama-3.2-3B-Instruct. Training settings: single NVIDIA A100, learning rate $1\times10^{-5}$, batch size 32, PPO clip $\epsilon_{clip}=0.2$, base KL coefficient 0.01, maximum generation length 1024, top_p=0.9, 1 epoch. The primary group size is $G=8$, with $G=4$ used to analyze the effect of multi-temperature sampling in small groups. The dual-anchor mixing weight is $\alpha=0.2$, and the target center is $\mu_{goal}=0.5$.

## Key Experimental Results

### Main Results
| Model / Group Size | Method | IFEval SSR/HSR | Custom SSR/HSR | FollowBench SSR/HSR | Key Observation |
|---------------|------|----------------|----------------|---------------------|----------|
| Gemma-2-2B, G=8 | Baseline | 56.7 / 45.1 | 54.8 / 18.8 | 63.7 / 52.9 | Zero-shot instruction model |
| Gemma-2-2B, G=8 | GRPO | 73.7 / 62.4 | 68.4 / 29.0 | 64.0 / 53.2 | Standard GRPO shows significant gains |
| Gemma-2-2B, G=8 | MDP-GRPO | 75.3 / 64.1 | 70.3 / 32.8 | 66.9 / 57.4 | More stable; Custom HSR +3.8 vs GRPO |
| Llama-3.2-3B, G=8 | Baseline | 54.2 / 46.8 | 60.3 / 20.8 | 69.7 / 59.8 | Llama initially stronger on FollowBench |
| Llama-3.2-3B, G=8 | GRPO | 66.1 / 58.5 | 65.1 / 24.8 | 68.4 / 58.9 | Custom gains; FollowBench slight drop |
| Llama-3.2-3B, G=8 | MDP-GRPO | 71.3 / 59.8 | 65.8 / 25.2 | 69.4 / 59.1 | IFEval SSR +5.2 vs GRPO |

The paper notes that individual components are not always globally optimal across all metrics. For instance, PT-GRPO achieves the highest HSR of 65.8% for Gemma-2-2B on IFEval, and DA-PT-GRPO achieves a slightly higher SSR of 71.5% for Llama-3.2-3B on IFEval than MDP-GRPO's 71.3%. The full pipeline is emphasized as providing a more stable overall profile rather than ranking first in every single metric.

### Ablation Study
| Setting | Key Figure | Description |
|------|----------|------|
| Gemma, G=8, Custom HSR | GRPO 29.0, DA-GRPO 32.6, DA-PT-GRPO 33.4, MDP-GRPO 32.8 | Goal anchors help most with complex constraint combinations |
| Gemma, G=4, IFEval | GRPO 69.7/58.2, MT-GRPO 71.1/59.4, MDP-GRPO 71.2/59.5 | MT restores reward dispersion in small groups |
| Gemma, G=4, Custom HSR | GRPO 28.6, MT-GRPO 30.6, MDP-GRPO 30.4 | MT effects are amplified in low-diversity settings |
| Llama, G=4, IFEval | GRPO 67.2/55.0, MT-GRPO 70.5/58.4 | Multi-temperature sampling also effective for Llama small groups |
| Difficulty Analysis | Baseline HSR <10% at Difficulty 4; DA-PT-GRPO ~20% vs GRPO ~12% at Difficulty 5 | Stabilization methods are more robust to degradation at high constraint counts |

### Key Findings
- Standard GRPO significantly improves verifiable instruction following but suffers from homogeneous groups and mean-centering issues on difficult multi-constraint prompts.
- Dual-anchor (DA) shows the most consistent HSR gains on the Custom Test Set, aligning with its design goal of fixing zero-variance/absolute blindness.
- Prospect shaping effectively controls KL drift while preserving reward gains; DA-PT-GRPO further suppresses KL drift.
- MT-GRPO might increase KL and decrease entropy under the current schedule, requiring careful temperature tuning; however, it is critical for performance recovery in the $G=4$ small-group setting.

## Highlights & Insights
- **Diagnosis precedes method**: The three failure modes (low-variance amplification, mean-centering blindness, and zero-variance collapse) clearly explain the instability of GRPO under discrete rewards.
- **Advantage modification over reward modification**: Multi-constraint rewards provided by deterministic checkers are reliable and cheap. MDP-GRPO avoids introducing learned reward models, instead supplementing signals at the advantage estimation and sampling stages.
- **Reintroducing absolute target levels to critic-free RL**: The appeal of GRPO is the lack of a value model, but the cost is only seeing relative within-group quality. Dual-anchor serves as a lightweight compromise, adding "goal awareness" to critic-free methods.
- **Restrained use of Prospect Theory**: It is not used to redefine human preference goals, but rather as a bounded asymmetric transformation for advantage shaping, which is more acceptable from an engineering perspective.

## Limitations & Future Work
- The method relies on explicit, automatically verifiable constraints. For subjective, stylistic, or underspecified constraints, learned rewards or preference feedback may be necessary, bringing back reward misspecification and judge bias.
- MDP-GRPO introduces multiple hyperparameters: anchor mixing weight, shaping parameters, temperature schedule, and asymmetric KL. These require recalibration and KL monitoring when migrating to different reward scales or task domains.
- Experiments focused on 2B/3B instruction-tuned models. Structured domains such as large-scale models, multilingual tasks, tool use, and code generation have not yet been validated.
- The method optimizes for constraint satisfaction encoded in the reward specification; it does not automatically guarantee broader safety, factuality, or value alignment.
- While multi-temperature sampling increases dispersion, it may also increase KL or decrease entropy, requiring controlled decoding during practical training.

## Related Work & Insights
- **vs Standard GRPO**: Standard GRPO uses intra-group z-score advantage, which is simple but fragile under discrete, low-variance rewards; MDP-GRPO stabilizes updates via sampling, anchors, and shaping.
- **vs MAPO / NGRPO**: Related works attempt to fix group-relative advantage allocation or all-negative groups; MDP-GRPO differs by simultaneously reducing homogeneous groups, restoring goal anchors, and applying prospect-style bounded shaping.
- **vs KTO**: KTO uses prospect-theoretic utility at the objective function level for binary preferences; MDP-GRPO applies prospect-inspired shaping at the advantage level, keeping the reward definition unchanged.
- **Insights**: For any task where rule-based verifiers are strong and rewards are discrete—such as formatted output, tool API schemas, code linting, or compliance document generation—one should consider goal-anchor advantage instead of standard GRPO.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Clear contribution by deconstructing GRPO failure modes and combining dual-anchor/prospect shaping.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers two models, two group sizes, three types of benchmarks, ablation, and training diagnostics; the main drawback is the limited model scale.
- Writing Quality: ⭐⭐⭐⭐☆ Motivations and formulas are clear, tables are complete; overall logic is well-structured.
- Value: ⭐⭐⭐⭐☆ Very practical for RLVR, multi-constraint tasks, and critic-free policy optimization, suitable for future expansion to larger models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Group-Relative REINFORCE Is Secretly an Off-Policy Algorithm: Demystifying Some Myths About GRPO and Its Friends](../../ICLR2026/llm_alignment/group-relative_reinforce_is_secretly_an_off-policy_algorithm_demystifying_some_m.md)
- [\[AAAI 2026\] LaF-GRPO: In-Situ Navigation Instruction Generation for the Visually Impaired via GRPO with LLM-as-Follower Reward](../../AAAI2026/llm_alignment/laf-grpo_in-situ_navigation_instruction_generation_for_the_visually_impaired_via.md)
- [\[NeurIPS 2025\] GVPO: Group Variance Policy Optimization for Large Language Model Post-Training](../../NeurIPS2025/llm_alignment/gvpo_group_variance_policy_optimization_for_large_language_model_post-training.md)
- [\[ACL 2026\] Better Literary Translation: A Multi-Aspect Data Generation and LLM Training Approach](better_literary_translation_a_multi-aspect_data_generation_and_llm_training_appr.md)
- [\[ACL 2026\] Taming Extreme Tokens: Covariance-Aware GRPO with Gaussian-Kernel Advantage Reweighting](taming_extreme_tokens_covariance-aware_grpo_with_gaussian-kernel_advantage_rewei.md)

</div>

<!-- RELATED:END -->

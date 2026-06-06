---
title: >-
  [Paper Note] LANG: Reinforcement Learning for Multilingual Reasoning with Language-Adaptive Hint Guidance
description: >-
  [ACL2026][Reinforcement Learning][multilingual reasoning] LANG utilizes same-language reasoning hints to bootstrap multilingual mathematical reasoning RL…
tags:
  - "ACL2026"
  - "Reinforcement Learning"
  - "multilingual reasoning"
  - "GRPO"
  - "hint-guided RL"
  - "language consistency"
  - "reward sparsity"
date: 2026-05-08
content_hash: f7d576ffbfde019d
---

# LANG: Reinforcement Learning for Multilingual Reasoning with Language-Adaptive Hint Guidance

**Conference**: ACL2026  
**arXiv**: [2605.22567](https://arxiv.org/abs/2605.22567)  
**Code**: https://github.com/fmm170/LANG  
**Area**: Reinforcement Learning / Multilingual Reasoning  
**Keywords**: multilingual reasoning, GRPO, hint-guided RL, language consistency, reward sparsity  

## TL;DR
LANG utilizes same-language reasoning hints to bootstrap multilingual mathematical reasoning RL, then employs cosine decay and language-difficulty-based adaptive hint shutdown to improve non-English reasoning accuracy while maintaining language consistency.

## Background & Motivation
**Background**: RLVR/GRPO has become a common approach for enhancing multi-step reasoning in large models. Models like DeepSeek-R1 demonstrate that verifiable rewards can drive models toward longer and more reliable reasoning chains. However, this path is highly English-centric; in multilingual scenarios, models must not only answer correctly but also maintain the user's input language throughout the intermediate reasoning and the final answer.

**Limitations of Prior Work**: Simply requesting "think and answer in [language]" in the prompt can improve language consistency but often at the expense of mathematical reasoning accuracy. Optimizing only for answer correctness frequently causes the reasoning chain to drift into English. The paper illustrates this contradiction using a Korean example regarding the number of divisors: language-consistent reasoning may lead to calculation errors, while correct calculations often shift to English.

**Key Challenge**: The fundamental difficulty of multilingual RL is the combination of reward sparsity and training-inference inconsistency. In low-resource languages, it is inherently difficult for the model to sample trajectories that are "correct in answer, correct in format, and consistent in language." If complete hints are provided throughout training, the model learns to depend on them, which are absent during inference.

**Goal**: The authors aim to provide sufficient scaffolding in early training to help the model find correct non-English reasoning trajectories, while progressively removing this scaffolding so the final policy can perform multilingual reasoning independently.

**Key Insight**: The paper draws on hint-guided RL and scheduled sampling, treating hints as exploration aids in early training rather than long-term input conditions. It further observes that different languages have different learning difficulties; low-resource languages require hints to be retained longer, while high-resource languages should detach from hints earlier.

**Core Idea**: Replace fixed hints or pure language consistency rewards with "language-conditional hints + progressive hint decay + language-group adaptive switching" to mitigate reward sparsity and language drift in multilingual RL.

## Method
The LANG method can be understood as a curriculum learning framework for multilingual reasoning. It first segments teacher-generated same-language reasoning trajectories into prefixes, which serve as hints appended to the question during early training. The hints are then gradually shortened according to the number of training steps and eventually removed. Simultaneously, the system determines when to shut off hints for each language resource group based on whether that group can stably sample trajectories with positive advantages.

### Overall Architecture
The input consists of a mathematical problem $q$ in language $l$ and a teacher-generated same-language reasoning trajectory $h=(h_1,\dots,h_L)$. At training step $t$, LANG constructs a hint-conditioned prompt by taking the first $k_t^l=\lfloor p_t^lL\rfloor$ hint tokens/segments based on the hint ratio $p_t^l$. The policy model samples a set of outputs based on this prompt, which are then updated using GRPO according to answer accuracy, format, and language consistency.

In early training, the prompt contains long same-language reasoning hints to address the difficulty of sampling positive instances in low-resource languages. As training progresses, the hint ratio decreases, and the model transitions from "following same-language trajectories" to "generating same-language reasoning independently." Once the effective update rate of a language resource group exceeds a threshold, that group enters the zero-hint regime, and subsequent training uses the original question directly.

### Key Designs
1. **Scheduled Multilingual Hint Decay**:
	- **Function**: Provides same-language reasoning scaffolding in early RL training and removes it gradually to prevent dependence on non-existent hints during inference.
	- **Mechanism**: Given a hint length $L$, a prefix of length $k_t^l=\lfloor p_t^lL\rfloor$ is injected at step $t$. The paper adopts a cosine decay $p_t^l=\frac{1}{2}(1+\cos(\pi t/T))$ to smoothly reduce the ratio from a full hint to 0.
	- **Design Motivation**: A pilot study found that fixed hints (QUESTA-style) maintain higher training reward and entropy but perform worse during testing, accompanied by increased response length and repeat scores, indicating the model learned a hint-conditioned shortcut.

2. **Language-adaptive Switch**:
	- **Function**: Allows languages with different resource levels to detach from hints at different times rather than sharing a global switch.
	- **Mechanism**: Languages are categorized into high/mid/low resource groups. The ratio of batches where "at least one rollout produces a positive advantage" is calculated as $u_R(t)$, followed by an EMA: $\bar{u}_R(t)=\alpha\bar{u}_R(t-1)+(1-\alpha)u_R(t)$. When $\bar{u}_R(t) \geq \tau$, the language group switches to zero-hint.
	- **Design Motivation**: High-resource languages sample correct trajectories independently more easily; excessive hints induce dependence. Conversely, early hint removal in low-resource languages leads back to reward sparsity.

3. **Conjunctive Reward GRPO**:
	- **Function**: Consolidates answer correctness, format compliance, and language consistency into a single strict verifiable reward.
	- **Mechanism**: Model outputs include a reasoning trace $o_t$ and a final answer $o_a$. The total reward $R(o)=1$ only if $R_{lc}=1$, $R_{format}=1$, and $R_{acc}=1$ are all true; otherwise, $R(o)=0$. GRPO samples $G$ outputs and updates the policy using intra-group normalized advantages.
	- **Design Motivation**: Multilingual reasoning cannot focus solely on the final answer (causing English drift) or language consistency (sacrificing accuracy). The conjunctive reward sets "same-language and correct answer" as the sole optimization goal.

### Loss & Training
Training involves two steps: cold-start and RL. Multilingual training data is constructed from DeepMath-103K, with 0.3K samples per in-domain language for cold-start and 3K samples for GRPO. RL uses 8 rollouts, a temperature of 1.0, a learning rate of $1\times10^{-6}$, a batch size of 128, a PPO mini-batch size of 64, a maximum sequence length of 16,384, and the KL coefficient is set to 0.

Language groups are defined as: High-resource (English, German, French, Spanish, Portuguese, Italian); Mid-resource (Japanese, Chinese, Russian, Korean, Vietnamese); Low-resource (Arabic, Bengali, Thai, Swahili, Telugu, Indonesian). This grouping directly serves the language-adaptive switch.

## Key Experimental Results

### Main Results
The paper reports LC&Acc on MMATH and PolyMath, where success requires both a correct answer and language consistency between the reasoning/answer and the input. The following data for Qwen2.5-7B-Instruct reflects the gains of LANG over strong baselines.

| Dataset | Metric | Ours (LANG) | Prev. SOTA | Gain |
|--------|------|-----------|------------|------|
| MMATH, Qwen2.5-7B | ALL-Avg. LC&Acc | 28.6 | mGRPO 26.0 / LC-GRPO 26.3 | +2.3 vs LC-GRPO |
| PolyMath, Qwen2.5-7B | ALL-Avg. LC&Acc | 15.6 | mGRPO 13.0 / LC-GRPO 13.9 | +1.7 vs LC-GRPO |
| MMATH, Qwen2.5-3B | ALL-Avg. LC&Acc | 17.7 | LC-GRPO 16.8 / mGRPO 15.7 | +0.9 vs LC-GRPO |
| PolyMath, Qwen2.5-3B | ALL-Avg. LC&Acc | 10.1 | LC-GRPO 9.5 / Vanilla GRPO 8.6 | +0.6 vs LC-GRPO |

Overall conclusion: Across four evaluation models, LANG achieves an average improvement of 24.1% on MMATH and 18.7% on PolyMath compared to LC-GRPO. Gains are particularly significant for low-resource languages; for example, Thai on MMATH (Qwen2.5-7B) improved by 39.0% over mGRPO, and Vietnamese on PolyMath improved by 24.6%.

### Ablation Study
| Configuration | MMATH LC&Acc | PolyMath LC&Acc | Note |
|------|--------------|-----------------|------|
| LANG, cosine decay | 28.6 | 15.6 | Full method; cosine decay is best |
| Linear decay | 27.1 | 14.6 | Uniform linear removal; slightly lower |
| Exponential decay | 24.1 | 14.3 | Rapid early removal; insufficient exploration |
| LANG w/o cold-start | 27.5 | 15.0 | Lack of initial format/language following |
| LANG w/o $R_{lc}$ | 3.1 | 9.1 | Significant drift without consistency reward |
| LANG w/o $p_t^l$ | 10.1 | 3.2 | Serious training-inference inconsistency |

### Key Findings
- LANG's gains are not due to "longer responses." While QUESTA increased response length and repetition simultaneously while performance dropped, LANG increases effective reasoning length without inducing repetitive generation.
- Cosine decay outperforms exponential and linear decay, indicating that the rhythm of hint removal is critical: too fast leads back to reward sparsity, while too slow leads to hint dependence.
- Non-mathematical multilingual tasks also show transfer gains: on MMLU-ProX, XWinograd, XStoryCloze, and XCOPA, LANG improves by an average of 10.9%. For Qwen2.5-7B, MMLU-ProX rose from 35.9 to 41.0, and XWinograd from 65.7 to 79.9.
- Layer-wise logit lens analysis shows that LANG maintains high language consistency in intermediate layers, not just by translating at the final output layer.

## Highlights & Insights
- The most valuable observation is decomposing multilingual reasoning failure into two issues: the inability to sample positive trajectories and the over-dependence on hints. While many methods only address the former, LANG explicitly includes "removing the scaffold" as a training objective.
- The language-adaptive switch is practical as it acknowledges that different languages do not share the same learning curve. Allowing high-resource languages to exit hints early and low-resource languages to exit late aligns better with the reality of multilingual training.
- The conjunctive reward, though simple, provides a very clean definition: only language consistency, correct format, and a correct answer yield a reward of 1. This reduces the space for "thinking in English, answering in the target language" shortcuts.
- The layer-wise analysis is worth emulating. Multilingual reasoning should not just verify the final string language but also check if intermediate representations have formed reasoning paths in the target language.

## Limitations & Future Work
- Authors acknowledge that multilingual hints are primarily distilled from DeepSeek-R1. Although experiments replacing the teacher with GPT-4o-mini still showed gains, teacher diversity could be improved.
- Performance trade-offs between different languages persist. The paper suggests this relates to imbalanced multilingual data during pre-training, which LANG remediates via RL but cannot solve at the representational root.
- The method relies on verifiable answers and language detectors, which are natural for math but harder to construct reliably for open-ended reasoning or subjective tasks.
- Hint quality, granularity, and semantic continuity affect training. Randomly dropping hint segments disrupts semantics and degrades performance, suggesting future research into fine-grained yet semantic-preserving hint curricula.

## Related Work & Insights
- **vs Language-Constraint Prompting / DIT / QRT**: These use prompts to control language during inference. They are lightweight but often sacrifice accuracy. LANG treats consistency as an RL objective and uses hints to lower exploration difficulty.
- **vs LC-GRPO / M-Thinker**: These introduce consistency rewards or cross-lingual alignment but still face reward sparsity in low-resource languages. LANG provides same-language hints to bootstrap exploration.
- **vs QUESTA / StepHint-style RL**: These use hints to mitigate sparse rewards, but fixed hints cause training-inference inconsistency. LANG's key differentiator is the progressive removal based on language difficulty.
- **Insights for Future Work**: Multilingual agent or RAG tasks could adopt similar strategies: provide same-language retrieval or reasoning hints, gradually withdraw them, and use language-group difficulty to pace the curriculum.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Combines hint-guided RL, scheduled decay, and language-adaptive switches; clear problem definition and effective mechanism.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers various datasets (math and non-math), multiple models, and extensive ablations; systematic analysis of hint quality could be deeper.
- Writing Quality: ⭐⭐⭐⭐☆ Complete logical chain; pilot study is persuasive; large main tables are data-dense.
- Value: ⭐⭐⭐⭐⭐ Highly relevant for multilingual RL and low-resource reasoning; suitable for systems requiring simultaneous optimization of accuracy and interpretability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] MARS: Multi-Agent Adaptive Reasoning with Socratic Guidance for Automated Prompt Optimization](../../AAAI2026/reinforcement_learning/mars_multi-agent_adaptive_reasoning_with_socratic_guidance_f.md)
- [\[NeurIPS 2025\] When Less Language is More: Language-Reasoning Disentanglement Makes LLMs Better Multilingual Reasoners](../../NeurIPS2025/reinforcement_learning/when_less_language_is_more_language-reasoning_disentanglement_makes_llms_better_.md)
- [\[ACL 2026\] Free Energy-Driven Reinforcement Learning with Adaptive Advantage Shaping for Unsupervised Reasoning in LLMs](free_energy-driven_reinforcement_learning_with_adaptive_advantage_shaping_for_un.md)
- [\[AAAI 2026\] Vision-Language Reasoning for Geolocalization: A Reinforcement Learning Approach](../../AAAI2026/reinforcement_learning/vision-language_reasoning_for_geolocalization_a_reinforcement_learning_approach.md)
- [\[ICML 2026\] Learning to Route Languages for Multilingual Policy Optimization](../../ICML2026/reinforcement_learning/learning_to_route_languages_for_multilingual_policy_optimization.md)

</div>

<!-- RELATED:END -->

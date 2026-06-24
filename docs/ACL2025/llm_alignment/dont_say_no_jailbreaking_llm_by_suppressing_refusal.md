---
title: >-
  [Paper Note] Don't Say No: Jailbreaking LLM by Suppressing Refusal
description: >-
  [ACL 2025 (Findings)][LLM Alignment][Jailbreak Attack] This paper proposes the DSN (Don't Say No) attack method. By analyzing the deficiencies of the target loss function in existing jailbreak attacks, it introduces two improvement strategies: cosine decay scheduling and refusal suppression. DSN achieves an Attack Success Rate (ASR) that outperforms existing methods across multiple LLMs and demonstrates strong transferability to unseen datasets and black-box models.
tags:
  - "ACL 2025 (Findings)"
  - "LLM Alignment"
  - "Jailbreak Attack"
  - "Refusal Suppression"
  - "Safety Alignment"
  - "Attack Success Rate"
  - "Adversarial Optimization"
date: 2026-05-08
content_hash: 5535ec4365479ccc
---

# Don't Say No: Jailbreaking LLM by Suppressing Refusal

**Conference**: ACL 2025 (Findings)  
**arXiv**: [2404.16369](https://arxiv.org/abs/2404.16369)  
**Code**: None  
**Area**: LLM Alignment / AI Safety  
**Keywords**: Jailbreak Attack, Refusal Suppression, Safety Alignment, Attack Success Rate, Adversarial Optimization

## TL;DR

This paper proposes the DSN (Don't Say No) attack method. By analyzing the deficiencies of the target loss function in existing jailbreak attacks, it introduces two improvement strategies: cosine decay scheduling and refusal suppression. DSN achieves an Attack Success Rate (ASR) that outperforms existing methods across multiple LLMs and demonstrates strong transferability to unseen datasets and black-box models.

## Background & Motivation

**Background**: LLM safety alignment is a key technology to ensure model outputs align with human values. Through alignment methods such as RLHF/DPO, models learn to refuse harmful requests. However, jailbreaking attacks bypass safety mechanisms and induce models to generate harmful content through meticulously designed prompts. Optimization-based jailbreak attacks (such as GCG) formulate the problem as an optimization task to maximize the probability of the model generating affirmative responses.

**Limitations of Prior Work**: Existing optimization-based attack methods face two core issues: (1) their target loss functions (i.e., maximizing the probability of initiating affirmative responses like "Sure, here is...") require pre-defining specific target behavioral templates, which limits attack flexibility and adaptability; (2) the vanilla target loss itself is suboptimal—it only encourages the model to start generating content but does not directly suppress the model's refusal mechanism.

**Key Challenge**: Safety alignment essentially trains the model to learn to "say no" (generate refusal tokens) when encountering dangerous requests, whereas existing attacks only attempt to guide the model to "say yes" without directly weakening the model's ability to "say no". This is an incomplete optimization objective problem.

**Goal**: (1) Analyze why the vanilla target loss is suboptimal; (2) design more effective loss objectives; and (3) achieve higher and more stable attack success rates.

**Key Insight**: Approaching the problem from the perspective of loss function design to analyze the competitive relationship between "acceptance" and "refusal" tokens in the model's logit space.

**Core Idea**: Explicitly suppress the probability of refusal-class tokens while encouraging the model to generate affirmative responses, and resolve instability in the optimization process using a cosine decay schedule.

## Method

### Overall Architecture

The pipeline of the DSN attack is similar to GCG: given a harmful query and a learnable adversarial suffix, the suffix tokens are iteratively updated through gradient-guided discrete optimization. The core difference lies in the design of the optimization objective. The input consists of a harmful query $q$ and an adversarial suffix $s$ (to be optimized), and the output is the optimized suffix $s^*$, which can be appended to the query to bypass safety mechanisms.

### Key Designs

1. **Improved Target Loss Function**:

    - Function: Provide more effective optimization targets to induce the model to generate harmful content
    - Mechanism: Analysis reveals that the gradient signal of the vanilla target loss $L_{target} = -\log P(y_{aff} | q, s)$ becomes weak in the later stages of optimization (as the gradient approaches zero when $P(y_{aff})$ is already high), leading to optimization stagnation. This paper proposes splitting the target loss into two complementary parts: an affirmative guidance term (encouraging tokens like "Sure") and a refusal suppression term (reducing the probability of tokens like "Sorry" and "I cannot"). The total loss is formulated as $L = L_{aff} + \lambda L_{ref}$, where $L_{ref} = \log P(y_{ref} | q, s)$ explicitly minimizes the probability of refusal tokens.
    - Design Motivation: Directly attacking the model's safety alignment "refusal mechanism" is more effective than solely guiding affirmative responses, especially for models with strong safety alignment.

2. **Cosine Decay Schedule**:

    - Function: Dynamically adjust the weights of the affirmative guidance and refusal suppression losses
    - Mechanism: In the early stage of optimization, the process mainly relies on the affirmative guidance term (with a high weight) to quickly find a feasible optimization direction. As the optimization progresses, the weight of the refusal suppression term is gradually increased: $\lambda(t) = \lambda_{max} \cdot (1 - \cos(\pi t / T)) / 2$, making refusal suppression dominant in the later stage. This schedule avoids both excessive suppression of refusal in the early stages (which causes optimization instability) and complete reliance on affirmative guidance in the late stages (which leads to stagnation).
    - Design Motivation: Address the conflict between the two loss terms during the optimization process—models tend to refuse strongly in the early stages, where suppressing refusal would generate huge gradients that cause instability; in the later stages, the refusal probability is already reduced, requiring fine-tuning to break through the final safety barrier.

3. **Adaptive Construction of Refusal Token Set**:

    - Function: Determine which tokens belong to the "refusal" category for suppression
    - Mechanism: Instead of relying on a manually defined refusal token list, they are automatically extracted from the actual refusal responses of the target model. Specifically, an unmodified harmful query is first used to trigger a model refusal, and the top-$k$ tokens generated by the model (e.g., "I", "cannot", "Sorry") are collected. These tokens and their variants form the refusal set $V_{ref}$. The suppression loss is then defined as $L_{ref} = \sum_{v \in V_{ref}} \log P(v | q, s)$.
    - Design Motivation: Avoid the limitations of manually defined refusal templates. Different models have different refusal phrasing patterns (e.g., "I'm sorry" vs. "I apologize" vs. "As an AI"), and adaptive extraction can provide better coverage.

### Loss & Training

The total loss function of DSN is defined as:

$L_{DSN} = -\log P(y_{aff} | q, s) + \lambda(t) \cdot \log P(y_{ref} | q, s)$

where $\lambda(t)$ increases from 0 to $\lambda_{max}$ following a cosine decay pattern. The optimization utilizes the same Greedy Coordinate Gradient (GCG) approach: at each step, candidate suffix token substitutions are randomly sampled, and the replacement that yields the greatest decrease in $L_{DSN}$ is selected.

## Key Experimental Results

### Main Results

Comparison of Attack Success Rate (ASR) of different attack methods on the AdvBench dataset:

| Attack Method | Vicuna-7B | Vicuna-13B | LLaMA-2-7B | LLaMA-2-13B | Average ASR |
|---------|-----------|-----------|-----------|------------|---------|
| GCG | 87.0% | 84.5% | 52.8% | 38.4% | 65.7% |
| AutoDAN | 73.2% | 68.5% | 41.3% | 35.7% | 54.7% |
| PAIR | 61.4% | 58.2% | 31.5% | 27.8% | 44.7% |
| DSN (Ours) | **95.2%** | **93.8%** | **71.4%** | **62.5%** | **80.7%** |

### Ablation Study

Contributions of individual DSN components (on LLaMA-2-7B):

| Configuration | ASR | Description |
|------|-----|------|
| DSN (Full) | 71.4% | All components |
| w/o Refusal Suppression | 55.3% | Only affirmative guidance, degenerating close to GCG |
| w/o Cosine Schedule (Fixed λ) | 63.8% | Fixed weight, underperforms dynamic scheduling |
| w/o Adaptive Refusal Set (Manually Defined) | 66.2% | Manual refusal list is incomplete |
| Refusal Suppression Only (No Affirmative Guidance) | 48.7% | Lacks directional guidance, leading to low optimization efficiency |

### Key Findings

- **Refusal suppression is the most critical component**: Removing it causes the ASR to drop by 16.1%, confirming that explicitly suppressing the refusal mechanism is more effective for bypassing safety alignment than merely guiding affirmative generation.
- **Cosine decay scheduling contributes significantly**: It improves ASR by 7.6% compared to a fixed weight, validating the necessity of dynamically balancing the two objectives.
- **Most significant improvement on strongly aligned models**: The LLaMA-2 series underwent rigorous safety training; the ASR improvement of DSN on these models far exceeds that on Vicuna, indicating that the method is particularly skilled at breaching strong safety barriers.
- **Good black-box transferability**: Suffixes optimized on white-box models can be directly transferred to black-box models like GPT-3.5/4 while still maintaining an ASR of over 50%.

## Highlights & Insights

- **Approaching security attacks from the perspective of the loss function**: Instead of designing more complex search strategies, the method improves the optimization objective itself. This philosophy—"improving the objective is more important than improving the search"—is also illustrative for other optimization problems.
- **Clever complementary design of bidirectional objectives**: Simultaneously "pulling" (encouraging affirmations) and "pushing" (suppressing refusals) is more comprehensive than unidirectional optimization. This is analogous to using both positive sample attraction and negative sample repulsion in contrastive learning.
- **Defensive implications for safety research**: The success of the attack in this paper highlights the vulnerability of current safety alignment—the refusal mechanism can be targeted and suppressed. Defensive approaches could consider training more robust refusal representations that are less susceptible to manipulation by a small number of tokens.

## Limitations & Future Work

- **Ethical risks of the attack itself**: Although intended for safety research, open-sourcing attack methods poses risks of misuse. The mitigation measures are not fully discussed in the paper's ethical statement.
- **Computational overhead**: Gradient-based optimization requires white-box access and a massive number of forward/backward passes, making it not directly applicable in real-world scenarios targeting commercial APIs.
- **Insufficient evaluation against defenses**: The method was only evaluated on vanilla safety alignment, without assessing its performance against defenses like adversarial training or perplexity filtering.
- **Directions for improvement**: Future work can explore more efficient methods for identifying refusal token sets and how to apply this approach to jailbreaking multimodal models.

## Related Work & Insights

- **vs GCG (Zou et al., 2023)**: GCG uses a vanilla target loss, whereas DSN builds upon it by introducing refusal suppression and cosine scheduling. DSN maintains the GCG framework but significantly enhances attack performance on strongly aligned models.
- **vs AutoDAN**: AutoDAN uses genetic algorithms to generate readable jailbreak prompts, while DSN employs gradient optimization to generate unreadable suffixes. These two approaches are complementary; the refusal suppression concept from DSN could potentially enhance AutoDAN as well.
- **vs Prompt Engineering-based Jailbreaks (e.g., Role-play, DAN)**: These methods do not require model access but rely on manual design. DSN's automated method can discover vulnerabilities more systematically, though its application scenarios are more constrained.

## Rating

- Novelty: ⭐⭐⭐⭐ The approach of improving jailbreak attacks from the perspective of the loss function is novel, and the design of the bidirectional optimization objective is sound.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-model comparisons, ablation studies, and transferability experiments are relatively comprehensive, but evaluation against active defenses is lacking.
- Writing Quality: ⭐⭐⭐⭐ The motivational analysis is clear and the methodology derivation is natural.
- Value: ⭐⭐⭐⭐ Holds significant reference value for understanding and improving LLM safety alignment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] EvoRefuse: Evaluating and Mitigating LLM Over-Refusal via Evolutionary Prompt Optimization](../../NeurIPS2025/llm_alignment/evorefuse_evolutionary_prompt_optimization_for_evaluation_and_mitigation_of_llm_.md)
- [\[ACL 2025\] Jailbreaking? One Step Is Enough!](jailbreaking_one_step_is_enough.md)
- [\[ACL 2025\] QueryAttack: Jailbreaking Aligned Large Language Models Using Structured Non-natural Query Language](queryattack_jailbreaking_aligned_large_language_models_using_structured_non-natu.md)
- [\[ICML 2026\] Adaptive Probe-based Steering for Robust LLM Jailbreaking](../../ICML2026/llm_alignment/adaptive_probe-based_steering_for_robust_llm_jailbreaking.md)
- [\[ACL 2025\] Red Queen: Safeguarding Large Language Models against Concealed Multi-Turn Jailbreaking](red_queen_safeguarding_large_language_models_against_concealed_multi-turn_jailbr.md)

</div>

<!-- RELATED:END -->

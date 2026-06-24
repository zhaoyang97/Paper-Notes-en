---
title: >-
  [Paper Note] Rethinking External Slow-Thinking: From Snowball Errors to Probability of Correct Reasoning
description: >-
  [ICML 2025][Reasoning][Test-time scaling] This work systematically analyzes the "snowball error" phenomenon in LLM reasoning from an information-theoretic perspective, establishes a theoretical link between snowball errors and the probability of correct reasoning, and demonstrates that external slow-thinking methods (e.g., BoN, MCTS) inherently mitigate error accumulation by scaling search width. Both theoretically and experimentally, it is proven that the effectiveness of th…
tags:
  - "ICML 2025"
  - "Reasoning"
  - "Test-time scaling"
  - "snowball errors"
  - "information theory"
  - "Best-of-N"
  - "MCTS"
date: 2026-05-08
content_hash: 120cfa909231f8b8
---

# Rethinking External Slow-Thinking: From Snowball Errors to Probability of Correct Reasoning

**Conference**: ICML 2025  
**arXiv**: [2501.15602](https://arxiv.org/abs/2501.15602)  
**Code**: [ZyGan1999/Snowball-Errors-and-Probability](https://github.com/ZyGan1999/Snowball-Errors-and-Probability)  
**Area**: LLM Reasoning  
**Keywords**: Test-time scaling, snowball errors, information theory, Best-of-N, MCTS

## TL;DR
This work systematically analyzes the "snowball error" phenomenon in LLM reasoning from an information-theoretic perspective, establishes a theoretical link between snowball errors and the probability of correct reasoning, and demonstrates that external slow-thinking methods (e.g., BoN, MCTS) inherently mitigate error accumulation by scaling search width. Both theoretically and experimentally, it is proven that the effectiveness of these methods primarily depends on the total inference budget and the reliability of the reward function rather than the search framework itself.

## Background & Motivation
**Background**: Test-time scaling (slow-thinking) has been shown to enhance the multi-step reasoning capabilities of LLMs. Models such as OpenAI o1, DeepSeek R1, and QwQ have demonstrated the feasibility of improving quality by extending inference time.

**Limitations of Prior Work**: Although external slow-thinking methods are widely used, their underlying mechanisms of effectiveness remain poorly understood, leading to a lack of theoretical guidance when designing more advanced and efficient strategies.

**Key Challenge**: In practical applications, complex slow-thinking techniques (e.g., MCTS) often require substantial computational resources to achieve limited success. This is due to the difficulty of optimizing design choices and hyperparameters, frequently leading to sub-optimal performance. Can simple methods (e.g., BoN) match complex methods under equivalent computational budgets?

**Goal**: Informally understand and theoretically explain the mechanics of external slow-thinking methods, establishing a unified analytical framework.

**Key Insight**: Utilize mutual information and Fano's inequality from information theory to mathematically connect snowball errors in LLM reasoning with the probability of reasoning failures.

**Core Idea**: The effectiveness of external slow-thinking is not determined by the search framework itself, but by the search scope and the reliability of the reward model; widening the search scope or improving the model's intrinsic reasoning capabilities is the direction for long-term improvement.

## Method

### Overall Architecture
The paper proposes a systematic framework based on information theory, unfolding in four steps: (1) defining and quantifying snowball errors in reasoning; (2) establishing the theoretical connection between snowball errors and reasoning failure probability; (3) analyzing the probability of correct reasoning for external slow-thinking in practical scenarios; (4) comparing the theoretical performance and computational costs of different slow-thinking strategies (BoN vs. MCTS).

### Key Designs

1. **Information-Theoretic Modeling of Snowball Errors**:

    - Function: Quantify the informational gap between the implicit LLM reasoning sequence $\bm{t}$ and the observable response sequence $\bm{r}$ using mutual information (MI)
    - Mechanism: Analogize the reasoning process to Plato's Allegory of the Cave—the LLM's output is merely the "shadow" of its internal reasoning. Each reasoning step incurs information loss $\text{InfoLoss}(r_l) = H(t_l | r_l)$, and the snowball error is defined as the cumulative information loss $H_{<l}(\bm{t}|\bm{r}) = \sum_i^{l-1} H(t_i | r_i)$
    - Design Motivation: Traditional methods analyze error accumulation at the token level, whereas reasoning task errors typically occur at the sentence level, which is harder to characterize. Information theory provides a unified mathematical framework for this scenario.

2. **Theoretical Bridge from Snowball Errors to Reasoning Failure Probability**:

    - Function: Prove that the lower bound of reasoning error probability is positively correlated with cumulative snowball errors
    - Mechanism: Derive Theorem 3.3 based on Fano's inequality—the lower bound of the error probability at reasoning step $l$ is $P(e_l) \geq \log^{-1}(|\mathcal{T}_l|-1)[H_{<l}(\bm{t}|\bm{r})/(l-1) - H_b(e_l)]$
    - Design Motivation: Establish a quantitative connection between snowball errors and actual reasoning errors, providing a theoretical foundation for subsequent analyses of slow-thinking methods.
    - Key Findings: When snowball errors grow super-linearly, the lower bound of the reasoning error probability increases with the reasoning length.

3. **Practical Modeling of $\tau$-Correct Reasoning Probability**:

    - Function: Define the concepts of $\tau$-correct steps and $\tau$-correct reasoning to model the probability of correct reasoning in practical scenarios
    - Mechanism: Assume that the probability of generating a correct step at each step follows an exponential decay $\Pr[|φ(r_l) - φ(r_l^*)| \leq τ] = \min(λ_τ e^{-l}, 1)$, deriving that the upper bound for the complete correct reasoning probability is $λ_τ^L e^{-L(L+1)/2}$
    - Design Motivation: Bridge the theoretical analysis with practical reasoning quality evaluation (such as reward model scoring).

4. **Unified Analysis of Width-Scaling Methods**:

    - Function: Systematically analyze the impact of beam search-like width-scaling methods on the probability of correct reasoning
    - Mechanism: Decompose external slow-thinking into two steps: "generation" and "selection", where the total probability is $\Pr[\psi(\mathcal{R}) \leq \tau] = \Pr(\tau_{\text{generate}}) \times \Pr(\tau_{\text{select}})$
    - Key Theorem (4.6): The upper bound of the correct reasoning probability for width-scaling methods is $\epsilon_b^L k^L \lambda_\tau^L e^{-L(L+1)/2}$, where $k$ is the number of samples per layer, and $\epsilon_b$ is the selection accuracy probability
    - Design Motivation: Reveal that widening the search scope increases the generation probability but at the expense of an increased selection burden, demonstrating a clear trade-off between the two.

5. **Theoretical Comparison of BoN vs. MCTS**:

    - Function: Rigorously compare the correct reasoning probability and computational cost of BoN and MCTS under the proposed theoretical framework
    - Core Conclusion: The value of N required for BoN to achieve accuracy comparable to MCTS: best case $O(b)$, worst case $O(b^{L/2})$
    - Computational Cost Comparison: In the best case, the computational costs of BoN and MCTS are asymptotically equivalent to $O(bL)$; in the worst case, BoN is $O(Lb^{L/2})$ vs. MCTS at $O(b^L)$, making BoN potentially cheaper as $L$ increases
    - Core Insight: Both methods yield similar computational costs at comparable accuracy levels, meaning the framework design is not the deciding factor.

### Loss & Training
This paper is a theoretical analysis work and does not involve training.

## Key Experimental Results

### Main Results

| Model | Dataset | MI Decay Trend | Reward Trend | Description |
|------|--------|------------|-------------|------|
| Llama-3.1-8B-Instruct | GSM8k | Near-exponential decay | Decreases with length | Verifies the existence of snowball errors |
| Qwen2.5-7B-Instruct | GSM8k | Near-exponential decay | Decreases with length | Consistent across different models |
| Skywork-o1-Open-8B | GSM8k | Near-exponential decay | Decreases with length | Also exists in o1-like models |

### BoN vs. MCTS Experimental Comparison

| Task | MCTS Accuracy | BoN (N=N̄_res) | BoN (N=N̄_call) | Description |
|------|------------|---------------|----------------|------|
| GSM8k (ORM Max) | ~84% | ~82% | ~85% | BoN matches MCTS within a reasonable range of N |
| GSM8k (ORM Vote) | ~84% | ~83% | ~85% | Voting strategy is slightly better |
| PrOntoQA (Self-Consistency) | ~95% | ~90% | ~90% | No improvement on binary classification tasks without a reward model |
| PrOntoQA (ORM Max) | ~95% | ~94% | ~96% | Matches MCTS when guided by RM |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| BoN + Self-Consistency | Has bottleneck | Increasing N is futile for binary classification tasks |
| BoN + ORM Vote | Improves with N | Requires a reliable reward model |
| BoN + ORM Max | Improves with N | Performance is close to MCTS at a comparable cost |
| Different Model Sizes | Consistent MI decay | Snowball error is a universal phenomenon |
| Different Difficulty Levels | MI decays faster for harder problems | Snowball effect is more severe for hard problems |

### Key Findings
- Mutual information decays at a near-exponential rate, far faster than linear decay, verifying the existence of snowball errors.
- Equipped with a reliable reward model, BoN can achieve similar accuracy to MCTS with a comparable computational cost.
- The effectiveness of slow-thinking methods primarily depends on the total inference budget and reward function reliability, rather than the search framework design.
- The selection accuracy of the reward model $\epsilon_b$ must satisfy $\epsilon_b > 1/k$ to guarantee gains from slow-thinking.
- In harder reasoning problems, MI decays faster and the snowball effect is more severe.

## Highlights & Insights
- Analogizing the relationship between LLM outputs and true reasoning to Plato's Allegory of the Cave, which is highly intuitive.
- First to provide a formal definition of snowball errors and a lower bound for reasoning failure probability from an information-theoretic perspective.
- Theoretically proves that the correct reasoning probability and computational cost of BoN and MCTS are comparable under ideal conditions, which has practical guiding significance.
- Highly practical conclusion: instead of designing complex search frameworks, it is better to optimize reward functions or enhance the core reasoning capabilities of the base model.
- The theoretical analysis is generalizable and not limited to specific slow-thinking methods.

## Limitations & Future Work
- Proposition 4.3 (the exponential decay assumption of reasoning error probability) is an assumption rather than a rigorous proof, although the paper shows in the appendix that the conclusion still holds under relaxed assumptions.
- Experimental validation is primarily on GSM8k and PrOntoQA, which have short reasoning steps; verification on longer reasoning chains (e.g., math competition problems) is insufficient.
- The MI estimation method relies on proxy metrics; directly measuring the mutual information between implicit reasoning and output is intrinsically challenging.
- Does not discuss the relationship between internal slow-thinking (e.g., training strategies of o1) and external slow-thinking.
- The theoretical framework assumes that the reasoning error probabilities of each step are independent, whereas dependencies might exist in practice.

## Related Work & Insights
- Application of information theory in LLM analysis: Ton et al. (2024) used information theory to quantify reasoning errors, and this paper further builds a connection with slow-thinking methods on this basis.
- Practical implications of BoN vs. MCTS: Under computationally constrained scenarios, a simple BoN + reliable RM might be more practical than a complex MCTS.
- The emphasis on the importance of reward models is noteworthy: RM reliability is critical to the success of slow-thinking methods.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Rethinking the Role of Prompting Strategies in LLM Test-Time Scaling: A Perspective of Probability Theory](../../ACL2025/llm_reasoning/rethinking_the_role_of_prompting_strategies_in_llm_test-time_scaling_a_perspecti.md)
- [\[ACL 2025\] ThinkGuard: Deliberative Slow Thinking Leads to Cautious Guardrails](../../ACL2025/llm_reasoning/thinkguard_deliberative_slow_thinking_leads_to_cautious_guardrails.md)
- [\[NeurIPS 2025\] A Theoretical Study on Bridging Internal Probability and Self-Consistency for LLM Reasoning](../../NeurIPS2025/llm_reasoning/a_theoretical_study_on_bridging_internal_probability_and_sel.md)
- [\[ICLR 2026\] Making Slow Thinking Faster: Compressing LLM Chain-of-Thought via Step Entropy](../../ICLR2026/llm_reasoning/making_slow_thinking_faster_compressing_llm_chain-of-thought_via_step_entropy.md)
- [\[NeurIPS 2025\] Does Thinking More Always Help? Mirage of Test-Time Scaling in Reasoning Models](../../NeurIPS2025/llm_reasoning/does_thinking_more_always_help_mirage_of_test-time_scaling_in_reasoning_models.md)

</div>

<!-- RELATED:END -->

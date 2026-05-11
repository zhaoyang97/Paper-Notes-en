---
title: >-
  [Paper Note] Towards Safe Reasoning in Large Reasoning Models via Corrective Intervention
description: >-
  [ICLR 2026][LLM Reasoning][Reasoning safety] This paper identifies a critical yet overlooked problem in large reasoning models (LRMs): their chain-of-thought reasoning frequently contains harmful content even when the final response appears safe. The authors propose Intervened Preference Optimization (IPO), which corrects unsafe reasoning trajectories by replacing compliance cues with safety triggers, constructing preference pairs for alignment training. Across 3 LRMs, IPO reduces reasoning harmfulness by over 30% without compromising reasoning capability.
tags:
  - ICLR 2026
  - LLM Reasoning
  - Reasoning safety
  - large reasoning models
  - preference optimization
  - safety triggers
  - compliance cues
date: 2026-05-08
content_hash: 1a19905f9f9d7a84
---

# Towards Safe Reasoning in Large Reasoning Models via Corrective Intervention

**Conference**: ICLR 2026
**arXiv**: [2509.24393](https://arxiv.org/abs/2509.24393)
**Code**: To be confirmed
**Area**: LLM Reasoning
**Keywords**: Reasoning safety, large reasoning models, preference optimization, safety triggers, compliance cues

## TL;DR
This paper identifies a critical yet overlooked problem in large reasoning models (LRMs): their chain-of-thought reasoning frequently contains harmful content even when the final response appears safe. The authors propose Intervened Preference Optimization (IPO), which corrects unsafe reasoning trajectories by replacing compliance cues with safety triggers, constructing preference pairs for alignment training. Across 3 LRMs, IPO reduces reasoning harmfulness by over 30% without compromising reasoning capability.

## Background & Motivation
Large reasoning models (e.g., DeepSeek-R1, Qwen3) demonstrate strong performance on mathematics, coding, and agent tasks, but their CoT reasoning processes frequently contain harmful content (deception, illegal instructions, violence, etc.), even when the final responses appear safe. This issue is systematically overlooked by existing safety alignment methods.

**Key Challenge**: Existing alignment approaches (e.g., SafeChain, RealSafe, STAR) primarily train LRMs via SFT on distilled safe CoT data. However, experiments demonstrate that: (1) although final responses are typically safe, harmful content in reasoning chains remains pervasive—RealSafe achieves a reasoning harm rate of 47.1% on WildJailbreak despite a response harm rate of only 2.0%; (2) unsafe reasoning can be exploited by malicious users (especially via open-source models) and makes models more susceptible to jailbreak attacks.

**Why RL Is Insufficient**: Directly rewarding safe reasoning via GRPO is a natural approach, but suffers from severe rollout diversity deficiency—approximately 50% of harmful prompts yield virtually no safe reasoning trajectories, resulting in poor within-group advantage diversity and weak policy gradient update signals.

**Key Observations**: Through analysis of safety dynamics during reasoning, the authors identify three critical patterns: (1) safe reasoning is typically consolidated by a small number of **safety triggers**—reasoning steps in which the model explicitly acknowledges risk or invokes safety guidelines, after which the probability of safe continuation approaches 100%; (2) **compliance cues** are strongly correlated with unsafe continuations—steps in which the model expresses compliant intent are followed by a sharp rise in harmful continuations; (3) replacing compliance cues with safety triggers effectively corrects reasoning trajectories.

## Method

### Overall Architecture
IPO (Intervened Preference Optimization) is a process-supervised reasoning safety alignment method. The core pipeline: (1) detect the position of the first compliance cue in an unsafe reasoning trajectory; (2) replace it with a trigger sampled from a safety trigger pool; (3) allow the model to continue generation from the replacement point, producing a safe reasoning trajectory; (4) construct preference pairs from the original unsafe trajectory and the corrected safe trajectory, and apply DPO training from the point of divergence.

### Key Designs

1. **Safety Trigger Identification and Compliance Cue Detection**

    - **Function**: Automatically identify key turning-point steps in safe reasoning (safety triggers) and in unsafe reasoning (compliance cues).
    - **Design Motivation**: Safety is not uniformly distributed throughout the reasoning process but is concentrated in a small number of critical steps—safety triggers drive subsequent continuations to nearly 100% safe, while compliance cues cause continuations to sharply turn harmful.
    - **Mechanism**: The Continuation Safety Ratio (CSR) is defined as $S_i(x, z_s) = \mathbb{E}[\mathbb{I}(z_s^{\leq i} \| z_c \text{ is safe})]$, estimating the safe continuation probability at each token position (32 samples). Safety triggers are defined as sentences where CSR rises sharply to $\geq 0.9$ and remains stable (window $K=15$); compliance cues are sentences where CSR drops sharply to $\leq 0.1$. Compliance cue detection uses GPT-4o with few-shot prompting, achieving >80% agreement with human annotation.
    - **Key Statistics**: The Pearson correlation between the detected positions of compliance cues and CSR inflection points reaches 0.85.

2. **Corrective Intervention and Preference Pair Construction**

    - **Function**: Replace compliance cues in unsafe trajectories with safety triggers, generate safe continuations, and form preference pairs with the original trajectories.
    - **Design Motivation**: Intervention directly increases rollout diversity (compared to GRPO) while providing locally precise training signals at safety-critical steps (analogous to reward shaping).
    - **Mechanism**: Given an unsafe trajectory $z$ with its first compliance cue at token index $h$, a safety trigger $\tau \sim \mathcal{T}$ is substituted, and the model generates an intervened continuation $\tilde{z}^{\geq h} \sim \pi_\theta(\cdot | x, z^{<h}, \tau)$. If the result remains unsafe, iterative intervention is applied. The resulting preference pair is $(x, \tilde{z} \succ z, h)$.
    - **Theoretical Connection**: DPO training from the divergence point is equivalent to applying a shaped reward at safety-critical steps—CSR exhibits significant jumps at safety triggers and compliance cues, a structure that IPO explicitly exploits.

3. **Over-Refusal Mitigation**

    - **Function**: An auxiliary preference dataset of 915 benign prompts is constructed, contrasting normal responses with refusals, followed by an additional DPO training stage.
    - **Design Motivation**: Training on safety data alone causes over-refusal, necessitating a balance between safety and helpfulness.
    - **Mechanism**: An auxiliary SFT loss (similar to RPO) is added on top of DPO to stabilize training.

### Loss & Training
The core training objective is DPO applied from the point of divergence:

$$-\mathbb{E}\!\left[\log \sigma\!\left(\beta \log \frac{\pi_\theta(\tilde{z}^{\geq h}|x,z^{<h})}{\pi_\text{ref}(\tilde{z}^{\geq h}|x,z^{<h})} - \beta \log \frac{\pi_\theta(z^{\geq h}|x,z^{<h})}{\pi_\text{ref}(z^{\geq h}|x,z^{<h})}\right)\right]$$

Training uses 1,000 harmful prompts from STAR-1 and 6 representative safety triggers ($N=1$), yielding a dataset of approximately 500–1,400 samples. Training takes approximately 40 minutes (compared to >2 hours for GRPO, with inferior results).

## Key Experimental Results

### Main Results (DeepSeek-R1-Distill-Llama-8B)

| Method | JBB Reas.↓ | JBB Resp.↓ | SR Reas.↓ | SR Resp.↓ | WJ Reas.↓ | WJ Resp.↓ | Reas. Avg.↓ | Resp. Avg.↓ | AIME↑ | MATH↑ | GPQA↑ | HEval↑ | Reas. Avg.↑ |
|--------|-----------|-----------|----------|----------|----------|----------|------------|------------|-------|-------|-------|--------|------------|
| Base | 69.0% | 45.0% | 63.2% | 49.3% | 82.4% | 73.9% | 71.5% | 56.1% | 50.7 | 91.8 | 44.9 | 79.5 | 66.7 |
| STAR | 8.0% | 0.3% | 21.9% | 14.6% | 37.8% | 22.7% | 22.6% | 12.5% | 46.0 | 89.4 | 47.0 | 77.1 | 64.9 |
| GRPO | 0.3% | 0.0% | 19.0% | 19.7% | 36.3% | 33.6% | 18.5% | 17.8% | 50.0 | 92.8 | 50.5 | 79.9 | 68.3 |
| **IPO** | **5.7%** | **0.3%** | **16.7%** | **10.9%** | **23.4%** | **9.6%** | **15.3%** | **6.9%** | 54.0 | 91.6 | 49.0 | 79.5 | **68.5** |

IPO achieves the best overall performance on both reasoning safety average (15.3%) and response safety average (6.9%), while its reasoning capability (68.5%) surpasses all baselines including the base model.

### Ablation Study

| Ablation Variable | SR Reas.↓ | SR Resp.↓ | Avg.↓ |
|-------------------|----------|----------|-------|
| Detector: DS-8B | 21.8% | 17.1% | 19.4% |
| Detector: DeepSeek-R1 | 16.4% | 11.0% | 13.6% |
| Detector: GPT-4o | 16.2% | 11.3% | **13.7%** |
| Training: SFT | 47.4% | 37.3% | 42.3% |
| Training: DPO on Full | 25.8% | 12.3% | 19.0% |
| Training: **DPO on Part** | **11.2%** | **10.6%** | **10.9%** |

Applying DPO only from the divergence point (DPO on Part) substantially outperforms full-trajectory DPO and SFT, validating the effectiveness of locally precise supervision.

### Key Findings
- **Reasoning safety → response safety**: The probability of a safe response following safe reasoning is extremely high, indicating that alignment efforts should prioritize the reasoning process over the final response.
- **Corrective effect of safety triggers**: A single replacement suffices to substantially reduce the harm rate of subsequent continuations, and iterative intervention yields cumulative benefits.
- **IPO vs. GRPO efficiency**: IPO requires at most 14 generations per prompt (6 triggers × 2 + 2 for over-refusal mitigation), whereas GRPO requires at least 40; IPO trains in ~40 minutes vs. >2 hours for GRPO.
- **Cross-model consistency**: IPO is effective across DS-8B, DS-7B, and Qwen3-8B; on Qwen3-8B, the reasoning harm rate drops from 51.3% to 13.9%.
- **KL divergence analysis**: IPO exhibits higher KL divergence at token positions corresponding to compliance cues, confirming the targeted nature of the supervision signal.

## Highlights & Insights
- The identification of "safety triggers" and "compliance cues" constitutes the paper's most important empirical contribution—reasoning safety is not uniformly distributed but is determined by a small number of critical steps.
- The Continuation Safety Ratio (CSR) is an elegant measurement tool that quantifies the marginal safety contribution of each token in the reasoning process.
- IPO brings the concept of reward shaping into safety alignment: applying local preference signals at CSR inflection points is substantially more efficient than global sparse rewards.
- The method achieves the best current Pareto balance between safety and reasoning capability—safety improves significantly while reasoning performance does not degrade but improves.
- The 0.85 Pearson correlation between compliance cues and CSR inflection points provides a solid empirical foundation for the intervention strategy.

## Limitations & Future Work
- Compliance cue detection relies on GPT-4o as an external judge, introducing additional dependency and potential bias.
- Construction of the safety trigger pool still requires manual selection (6 representative triggers), limiting automation.
- The XsTest compliance rate (DS-8B: 80%) falls below some weaker safety baselines, indicating residual over-refusal.
- Validation is limited to models at ≤8B scale (except for 1.5B–14B results in the appendix); effectiveness on larger models remains to be confirmed.
- Multi-turn dialogue and agent scenarios are not explored (mentioned only as future directions).
- Training data comprises only ~1,000 harmful prompts; the effect of scaling the data is unknown.

## Related Work & Insights
- Unlike SFT-based methods such as SafeChain, RealSafe, and STAR, IPO does not rely on distilled safe CoT data but instead directly intervenes in unsafe trajectories.
- Compared to backtracking methods based on special tokens (e.g., BackTrack), IPO performs proactive correction at the reasoning level rather than passive detection.
- The argument that reasoning safety should take precedence over response safety is compelling and well-supported quantitatively, and could shift the focus of safety alignment research.
- The CSR metric and safety trigger analysis methodology are generalizable to other process supervision settings (e.g., factuality, logical consistency).

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — First work to systematically treat reasoning safety as an independent alignment objective; the identification and exploitation of safety triggers and compliance cues are original contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — 3 LRMs × 3 safety benchmarks + 4 reasoning benchmarks + ablations (detector / algorithm / efficiency), though model scale coverage is limited.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — The motivation chain is tightly constructed; the observation → hypothesis → method → validation logic is exceptionally clear.
- **Value**: ⭐⭐⭐⭐⭐ — Reveals an important and overlooked dimension of LRM safety alignment; the method is practical and efficient, with direct and far-reaching implications for safe AI research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] RFEval: Benchmarking Reasoning Faithfulness under Counterfactual Reasoning Intervention in Large Reasoning Models](rfeval_benchmarking_reasoning_faithfulness_under_counterfactual_reasoning_interv.md)
- [\[ICLR 2026\] Training Large Reasoning Models Efficiently via Progressive Thought Encoding](training_large_reasoning_models_efficiently_via_progressive_thought_encoding.md)
- [\[ICLR 2026\] Dynamics-Predictive Sampling for Active RL Finetuning of Large Reasoning Models](dynamics-predictive_sampling_for_active_rl_finetuning_of_large_reasoning_models.md)
- [\[ICLR 2026\] Reasoning or Retrieval? A Study of Answer Attribution on Large Reasoning Models](reasoning_or_retrieval_a_study_of_answer_attribution_on_large_reasoning_models.md)
- [\[ICLR 2026\] When Reasoning Meets Compression: Understanding the Effects of LLMs Compression on Large Reasoning Models](when_reasoning_meets_compression_understanding_the_effects_of_pruning_and_quant.md)

</div>

<!-- RELATED:END -->

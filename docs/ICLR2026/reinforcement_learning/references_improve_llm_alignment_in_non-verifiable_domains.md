---
title: >-
  [Paper Note] References Improve LLM Alignment in Non-Verifiable Domains
description: >-
  [ICLR 2026][Reinforcement Learning][Reference-guided evaluation] This paper proposes RefEval, a reference-guided LLM-as-Judge framework that uses high-quality reference outputs as "soft verifiers…
tags:
  - "ICLR 2026"
  - "Reinforcement Learning"
  - "Reference-guided evaluation"
  - "non-verifiable domains"
  - "LLM-as-Judge"
  - "self-improvement"
  - "DPO"
date: 2026-05-08
content_hash: d42a172fdf3fef28
---

# References Improve LLM Alignment in Non-Verifiable Domains

**Conference**: ICLR 2026
**arXiv**: [2602.16802](https://arxiv.org/abs/2602.16802)

**Code**: [GitHub](https://github.com/yale-nlp/RLRR)

**Area**: Reinforcement Learning
**Keywords**: Reference-guided evaluation, non-verifiable domains, LLM-as-Judge, self-improvement, DPO

## TL;DR

This paper proposes RefEval, a reference-guided LLM-as-Judge framework that uses high-quality reference outputs as "soft verifiers," improving LLM-judge accuracy by 6.8%. Building on this, the authors design a two-stage self-improvement pipeline (SFT distillation + reference-guided DPO) that outperforms SFT distillation alone by +19.2/+16.5 on AlpacaEval/Arena-Hard, matching the performance of the fine-tuned reward model ArmoRM — demonstrating that effective LLM alignment in non-verifiable domains is achievable without human preference annotation.

## Background & Motivation

**Limitations of RLVR**: Reinforcement learning with verifiable rewards (RLVR) has proven effective for reasoning tasks (mathematics/code), but alignment tasks (instruction following, summarization, creative writing) lack ground-truth verifiers, making direct application of RLVR infeasible.

**Cost of RLHF/RLAIF**: Current alignment post-training relies on RLHF or RLAIF, requiring either a dedicated reward model (RM) trained on large-scale human preference annotations, or LLM-as-Judge methods that suffer from position bias and verbosity bias with limited accuracy.

**Availability of Reference Outputs**: Although preference annotation is costly, high-quality reference outputs can often be obtained cheaply — for example, generating 60K references with DeepSeek-V3 costs approximately $40 — representing an underexploited source of supervision signal.

**Ineffectiveness of Naïve Reference Use**: Prior work (LLMBar, HREF) has attempted to concatenate references into prompts without explicitly guiding the judge on how to use them, yielding only marginal improvements — indicating that carefully designed prompting strategies are necessary.

**Potential for Self-Improvement**: If reference-guided LLM self-judging can provide reliable preference signals, external human/AI feedback becomes unnecessary, enabling a "semi-self-improvement" paradigm that substantially reduces data and annotation requirements for alignment training.

**Core Research Question**: *Can a reference-guided LLM evaluator serve as a soft verifier to support RL-based LLM alignment without external supervision?* The paper addresses this systematically from both evaluation and training perspectives.

## Method

### 3.1 Reference-Guided LLM Evaluation (RefEval & RefMatch)

**Mechanism**: Carefully designed prompting strategies explicitly instruct the LLM-judge on how to leverage reference outputs for pairwise comparison.

- **RefEval**: Instructs the judge to assess which candidate output is more consistent with the reference in terms of quality and content, while still responding to the original instruction. Rather than simple semantic matching, the reference serves as a quality benchmark.
- **RefMatch**: Assigns a stronger role to the reference — instructing the judge to act as a "semantic and stylistic matcher" that determines which output more closely resembles the reference. The explicit directive is: "Your goal is to determine which output demonstrates closer similarity to the reference."
- **Ref-Free (Ours)**: A reference-free baseline that instructs the model to evaluate along dimensions such as instruction-following quality, factuality, and verbosity.

### 3.2 Two-Stage Self-Improvement Training Pipeline

**Stage 1: SFT Distillation**

Supervised fine-tuning on high-quality reference outputs. The paper demonstrates this outperforms direct preference optimization from the base model.

**Stage 2: Reference-Guided DPO**

DPO loss function:

$$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$

Preference pairs $(y_w, y_l)$ are annotated by a reference-guided self LLM-judge:
- 5 candidate outputs are sampled per instruction (temperature 0.8)
- All $\binom{5}{2}=10$ pairs undergo pairwise comparison → average quality scores are computed → the best and worst outputs form the training pair
- 60K instructions → 600K pairwise judgments in total

### Key Design Choices

1. **On-policy data generation**: Candidate outputs are generated by the model being fine-tuned rather than an external model, which prior work has shown to be more effective.
2. **Reference source**: Generated by DeepSeek-V3 at a total cost of approximately $40 for 60K instances.
3. **SFT before DPO**: Ablations confirm that SFT distillation followed by DPO outperforms direct DPO.
4. **Position bias mitigation**: All pairwise evaluations average accuracy across two orderings of the candidates.

## Key Experimental Results

### Table 1: LLM-Judge Evaluation Accuracy (Average across 11 Open-Source Models × 5 Datasets)

| Method | Natural | Adversarial | MTBench | InstruSum | HREF | **Avg.** |
|--------|---------|-------------|---------|-----------|------|---------|
| LLMBar-Base | 83.1 | 61.7 | 74.6 | 70.2 | 72.0 | 72.3 |
| CoT | 82.0 | 60.1 | 75.4 | 69.1 | 69.6 | 71.2 |
| HREF-Ref | 85.3 | 62.3 | 76.5 | 70.8 | 79.2 | 74.8 |
| RefMatch | 84.6 | 74.1 | 76.3 | 72.9 | 80.4 | 77.7 |
| **RefEval** | **86.8** | **74.9** | **76.7** | **74.5** | **82.7** | **79.1** |

→ RefEval surpasses the reference-free baseline LLMBar-Base by +6.8% and the existing reference method HREF-Ref by +4.3%.

### Table 2: Self-Improvement Training Results (AlpacaEval / Arena-Hard)

| Method | Llama-3 AE | Llama-3 AH | Qwen2.5 AE | Qwen2.5 AH |
|--------|-----------|-----------|------------|------------|
| Base | 25.0 | 27.1 | 14.4 | 23.4 |
| DSV3-Distill (SFT) | 53.9 | 42.2 | 48.8 | 56.5 |
| ROUGE | 56.4 | 52.1 | 50.9 | 67.4 |
| BERTScore | 58.8 | 53.0 | 55.3 | 64.5 |
| RefFree | 67.5 | 53.8 | 65.1 | 71.8 |
| ArmoRM (fine-tuned RM) | 73.9 | 58.6 | 66.8 | 72.2 |
| **RefEval** | **73.1** | **58.7** | **70.0** | **74.1** |

→ RefEval matches or exceeds ArmoRM without requiring a separately trained reward model.

### Table 3: Ablation on Reference Quality (Llama-3-8B)

| Reference Source | Distill AE | RefFree AE | RefEval AE | RefEval AH |
|-----------------|-----------|-----------|-----------|-----------|
| DeepSeek-V3 (strong) | 53.9 | 67.5 | 73.1 | 58.7 |
| GPT-4o-mini (weak) | 28.7 | 42.6 | 44.4 | 58.3 |

→ Even with weak references, RefEval outperforms RefFree (+1.8/+16.6), indicating that the reference-guided mechanism itself confers a structural advantage.

## Key Findings

1. **Smaller models benefit most**: Llama-3-8b improves by +17.4% with RefEval over LLMBar-Base, while the stronger qwen-2.5-72b improves by +5.2% — references compensate for the knowledge deficit of smaller models.

2. **Inter-judge consistency improves**: RefEval raises the average agreement rate across different judges from 76.6% to 81.4% — references provide a shared decision anchor that reduces judgment variance.

3. **SFT distillation > direct DPO**: SFT on high-quality references outperforms direct DPO with ArmoRM (53.9 vs. 49.2 on AlpacaEval), indicating that high-quality references themselves constitute a strong supervision signal.

4. **Coding & math benefit most**: Task-level analysis shows the largest gains from reference guidance on coding and math tasks; improvements on creative tasks vary by model — structured tasks are more amenable to reference anchoring.

5. **Frontier judges can also be enhanced**: GPT-4o with human-edited oracle references still improves on LLMBar-Adversarial, suggesting that human references carry more information than the strongest LLMs.

## Highlights & Insights

- **Conceptual transfer of "reference as soft verifier"**: The paper elegantly transfers the core advantage of RLVR — having a reference answer for verification — to non-verifiable domains. The concept is simple yet far-reaching.

- **Systematic large-scale experiments**: 11 judges × 5 datasets × 2 base models × multiple ablations — experimental coverage substantially exceeds comparable work, lending high credibility to the conclusions.

- **Engineering insight into prompt design**: The paper demonstrates that *how* the judge is instructed to use a reference matters more than *whether* a reference is provided — the gap between naïve concatenation and carefully designed prompting reaches 4–5 percentage points.

- **Strong practical utility**: 60K references at $40 (DeepSeek-V3) → self-improvement without human annotation → performance matching fine-tuned RM → substantially lowers the barrier to alignment training.

## Limitations & Future Work

1. **Dependency on reference quality**: Method effectiveness is strongly correlated with the quality of the reference source; weak references (GPT-4o-mini) still yield gains but are substantially inferior to strong references (DeepSeek-V3) — effectiveness in settings lacking frontier model access remains uncertain.

2. **Evaluation limited to general alignment tasks**: Experiments are restricted to general instruction-following benchmarks (AlpacaEval/Arena-Hard); specialized domains requiring expert knowledge such as medicine or law have not been tested.

3. **High computational cost of pairwise comparison**: Each instruction requires $\binom{5}{2}=10$ pairwise comparisons, totaling 600K judge calls for 60K instructions — inference cost is non-trivial even when using the model itself as the judge.

4. **Semi-self-improvement rather than full self-improvement**: The approach still depends on an external frontier model to provide reference outputs — it is not truly self-sufficient, and is more accurately characterized as "self-evaluation with external references."

5. **Single training round**: Only one round of SFT + DPO is evaluated; iterative self-improvement (multi-round SFT → DPO cycles) is not explored and may yield further gains.

## Related Work & Insights

- **vs. HREF (Lyu et al., 2024)**: HREF also uses human references to enhance LLM-judge, but evaluates at a smaller scale (fewer LLMs/datasets) and does not extend reference guidance to training. This paper validates the approach systematically across 5 datasets × 11 judges and is the first to employ reference-guided judging as a signal for self-improving DPO training, upgrading the approach from an evaluation tool to a training signal source.

- **vs. RevisEval (Zhang et al., 2025)**: RevisEval generates "response-adapted references" to improve evaluation accuracy, focusing on static evaluation scenarios. This paper uses fixed external references (generated by frontier models) and extends them into a dynamic training pipeline, demonstrating that evaluation improvements can translate to training gains — methodologically more complete.

- **vs. BLEUBERI (Chang et al., 2025)**: BLEUBERI uses conventional metrics (BLEU) as reference-based reward signals for RL alignment. This paper replaces BLEU/BERTScore with LLM-judge → RefEval substantially outperforms ROUGE and BERTScore in alignment training (73.1 vs. 56.4/58.8), demonstrating that LLM-judge as a soft verifier is more flexible and effective than hard metrics.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The end-to-end pipeline from reference-guided evaluation to self-improving training is novel, though individual components (LLM-judge/DPO/distillation) are established techniques.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 11 judges × 5 benchmarks + two base models + reference quality ablation + task-category analysis + statistical significance testing — exceptionally comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The motivational chain is clear (RLVR → gap → reference guidance → evaluation → training); experimental logic proceeds in well-structured layers; conclusions are not overreached.
- **Value**: ⭐⭐⭐⭐⭐ $40 references + self-improvement = matching fine-tuned RM — directly actionable guidance for resource-constrained teams pursuing LLM alignment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] From Verifiable Dot to Reward Chain: Harnessing Verifiable Reference-based Rewards for RL of Open-ended Generation](from_verifiable_dot_to_reward_chain_harnessing_verifiable_reference-based_reward.md)
- [\[ICLR 2026\] TROLL: Trust Regions improve Reinforcement Learning for Large Language Models](troll_trust_regions_improve_reinforcement_learning_for_large_language_models.md)
- [\[ICML 2026\] Turning Drift into Constraint: Robust Reasoning Alignment in Non-Stationary Multi-Stream Environments](../../ICML2026/reinforcement_learning/turning_drift_into_constraint_robust_reasoning_alignment_in_non-stationary_envir.md)
- [\[ICLR 2026\] Reasoning Boosts Opinion Alignment in LLMs](reasoning_boosts_opinion_alignment_in_llms.md)
- [\[ACL 2026\] LearnAlign: Data Selection for LLM Reinforcement Learning with Improved Gradient Alignment](../../ACL2026/reinforcement_learning/learnalign_data_selection_for_llm_reinforcement_learning_with_improved_gradient_.md)

</div>

<!-- RELATED:END -->

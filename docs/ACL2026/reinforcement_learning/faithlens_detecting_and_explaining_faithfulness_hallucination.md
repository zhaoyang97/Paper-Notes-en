---
title: >-
  [Paper Note] FaithLens: Detecting and Explaining Faithfulness Hallucination
description: >-
  [ACL 2026][Reinforcement Learning][Faithfulness hallucination] This paper proposes FaithLens, an 8B-parameter faithfulness hallucination detection model trained via high-quality data synthesis with three-dimensional filtering (label correctness, explanation quality, and data diversity) for cold-start SFT, followed by rule-based reinforcement learning (prediction correctness reward + explanation quality reward) for further optimization. FaithLens surpasses GPT-5.2 and o3 across 12 tasks while providing high-quality explanatory outputs.
tags:
  - ACL 2026
  - Reinforcement Learning
  - Faithfulness hallucination
  - explainable detection
  - rule-based reinforcement learning
  - data filtering
  - cross-task generalization
date: 2026-05-08
content_hash: 4c2387c0fdb2f310
---

# FaithLens: Detecting and Explaining Faithfulness Hallucination

**Conference**: ACL 2026
**arXiv**: [2512.20182](https://arxiv.org/abs/2512.20182)
**Code**: [https://github.com/S1s-Z/FaithLens](https://github.com/S1s-Z/FaithLens)
**Area**: Reinforcement Learning / Hallucination Detection
**Keywords**: Faithfulness hallucination, explainable detection, rule-based reinforcement learning, data filtering, cross-task generalization

## TL;DR

This paper proposes FaithLens, an 8B-parameter faithfulness hallucination detection model trained via high-quality data synthesis with three-dimensional filtering (label correctness, explanation quality, and data diversity) for cold-start SFT, followed by rule-based reinforcement learning (prediction correctness reward + explanation quality reward) for further optimization. FaithLens surpasses GPT-5.2 and o3 across 12 tasks while providing high-quality explanatory outputs.

## Background & Motivation

**Background**: LLMs are widely used for context-grounded text generation (e.g., RAG, summarization), but are prone to generating content inconsistent with or irrelevant to the given context—so-called "faithfulness hallucinations." Detecting such hallucinations is critical for responsible LLM deployment.

**Limitations of Prior Work**: (1) Lack of interpretability—existing methods treat hallucination detection as a black-box binary classification problem, outputting only prediction labels without explaining the rationale, leaving users unable to localize errors or understand their causes. (2) Inconsistent cross-task generalization—different tasks exhibit distinct hallucination patterns (subtle distortions in summarization vs. contradictory claims in RAG), and general-purpose models perform unevenly across them. (3) Lack of high-quality data—annotation costs are high, consistency is low, and synthetic data lacks quality control.

**Key Challenge**: Simultaneously achieving high detection accuracy and high explanation quality is non-trivial. SFT-trained models tend to memorize simple samples and generalize poorly to complex scenarios, while the quality of free-form explanations is difficult to verify directly with rule-based signals.

**Goal**: Develop a cost-effective hallucination detection model that jointly produces detection results and explanatory rationales, achieving state-of-the-art performance across 12 diverse tasks.

**Key Insight**: A two-stage training pipeline—cold-start SFT on carefully filtered synthetic data, followed by GRPO reinforcement learning with well-designed rule-based rewards (prediction correctness + explanation quality).

**Core Idea**: The key insight behind the explanation quality reward—if a generated explanation enables a "novice model" (an untuned Llama-3.1-8B) to correctly predict the label, the explanation is sufficiently clear and informative.

## Method

### Overall Architecture

FaithLens training proceeds in two stages: (1) Cold-start SFT—starting from open-source datasets, a high-capability reasoning model (DeepSeek-V3.2-Think) is used to synthesize training data with explanations, which are then filtered via three-dimensional criteria before fine-tuning; (2) Rule-based RL—the GRPO algorithm is applied for further optimization, with a reward function consisting of prediction correctness, explanation quality, and format components.

### Key Designs

1. **Three-Dimensional Data Filtering Strategy**:

    - Function: Ensures label correctness, explanation quality, and data diversity of the synthetic training data.
    - Mechanism: *Label filtering*—discards samples where the LLM's prediction disagrees with the ground-truth label, since CoT reasoning and explanations consistent with an incorrect prediction are intrinsically misleading. *Explanation quality filtering*—measures whether the model's perplexity on the correct label decreases when the explanation is provided; only samples that reduce perplexity are retained. *Diversity filtering*—applies K-Medoids clustering to construct a probe set, retaining only training candidates that improve prediction accuracy on diverse probe samples.
    - Design Motivation: Unfiltered synthetic data contains noise and an excess of simple samples. The three-dimensional filtering ensures that training data is simultaneously correct, informative, and covers diverse scenarios.

2. **Explanation Quality Reward**:

    - Function: Implicitly evaluates the quality of free-form explanations during the RL stage.
    - Mechanism: The generated explanation $e$, together with the document and claim, is fed to a "novice model" (untuned Llama-3.1-8B-Instruct) to check whether the novice can correctly predict the label based solely on this explanation. A reward of 1 is assigned if correct, 0 otherwise. The final reward is $R_{\text{final}} = R_{\text{pred}} + R_{\text{exp}} + R_{\text{format}}$.
    - Design Motivation: Directly verifying free-form text quality with rules is nearly impossible. The proxy evaluation principle—"if even a novice model can derive the correct answer from your explanation, the explanation must be sufficiently good"—provides an elegant and verifiable signal.

3. **GRPO Reinforcement Learning**:

    - Function: Further improves detection accuracy and explanation quality on top of the SFT cold-start.
    - Mechanism: For each document–claim pair, $G$ candidate outputs (explanation + prediction) are sampled. Each candidate is evaluated with the composite reward, and policy updates are performed via GRPO's group-relative advantage estimation. KL divergence regularization prevents excessive deviation from the reference policy.
    - Design Motivation: SFT tends to memorize simple samples; RL drives the model to produce high-quality outputs even in complex scenarios through exploration and reward-driven optimization.

### Loss & Training

The SFT stage applies standard cross-entropy loss on the filtered synthetic data. The RL stage uses GRPO (Group Relative Policy Optimization) with a composite reward = prediction correctness (0/1) + explanation quality (0/1) + format correctness (0/1). The base model is Llama-3.1-8B-Instruct.

## Key Experimental Results

### Main Results

**Overall Performance Across 12 Tasks (Balanced Accuracy %)**

| Model | Std. Dev. ↓ | Average ↑ |
|-------|------------|----------|
| GPT-4o | 7.0 | 76.1 |
| o3 | 6.0 | 82.1 |
| GPT-5.2 | — | 85.3 |
| Claude-3.7-Sonnet | 5.3 | 82.6 |
| DeepSeek-V3.2-Think | 5.1 | 84.4 |
| MiniCheck-7B | 9.3 | 76.7 |
| **FaithLens-8B (Ours)** | **4.1** | **85.8** |

### Ablation Study

| Configuration | Avg. Accuracy | Note |
|--------------|--------------|------|
| Full FaithLens | 85.8 | Complete model |
| w/o RL (SFT only) | 82.3 | RL contribution: +3.5 |
| w/o explanation quality reward | 84.1 | Explanation reward contribution: +1.7 |
| w/o data filtering | 79.8 | Filtering contribution: +6.0 |
| w/o diversity filtering | 81.5 | Diversity filtering contribution: +4.3 |

### Key Findings

- The 8B FaithLens surpasses GPT-5.2 (85.8 vs. 85.3) and o3 (82.1) at orders-of-magnitude lower cost.
- FaithLens achieves the lowest standard deviation (4.1), indicating the most stable cross-task generalization—addressing the "strong on some tasks, weak on others" problem of existing methods.
- The contribution of data filtering (+6.0) exceeds that of RL (+3.5), underscoring that high-quality training data is the foundation.
- Diversity filtering is critical for cross-task generalization; its removal causes a 4.3-point drop in accuracy.
- The explanation quality reward improves not only explanation quality but also detection accuracy (+1.7), suggesting that the "explanation → prediction" process has an intrinsic regularization effect.

## Highlights & Insights

- The "novice model proxy evaluation" is an elegant solution for assessing free-form explanation quality—transforming an unverifiable text quality problem into a verifiable classification correctness problem.
- The progressive "label → explanation → diversity" design of the three-dimensional filtering comprehensively ensures training data quality.
- Surpassing closed-source large models with only 8B parameters demonstrates that carefully designed training strategies outperform brute-force parameter scaling.

## Limitations & Future Work

- The explanation quality reward depends on the capabilities of the novice model; biases in the novice model may distort the reward signal.
- Synthetic data is derived from existing open-source datasets and may inherit their biases.
- Evaluation is limited to English tasks; multilingual generalization remains unverified.
- Future work may explore finer-grained explanation evaluation (e.g., sentence-level evidence grounding).

## Related Work & Insights

- **vs. MiniCheck**: MiniCheck trains a 7B classifier on synthetic data to reach GPT-4o-level performance but provides no explanations; FaithLens delivers explanations while surpassing GPT-5.2.
- **vs. SelfCheckGPT**: SelfCheckGPT relies on large model inference and is computationally expensive; FaithLens achieves superior performance with an 8B model.
- **vs. DeepSeek-V3.2-Think**: As the data synthesis teacher model it performs strongly (84.4%), yet FaithLens exceeds its teacher through RL (85.8%).

## Rating

- Novelty: ⭐⭐⭐⭐ The explanation quality reward and three-dimensional filtering strategy are innovative, though the overall framework (SFT + RL) follows a common paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 12 tasks, multiple baselines (including GPT-5.2/o3), and comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Method descriptions are clear and formulations are complete.
- Value: ⭐⭐⭐⭐⭐ An 8B model surpassing GPT-5.2 with explanatory outputs demonstrates strong practical utility.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Explaining Decentralized Multi-Agent Reinforcement Learning Policies](../../AAAI2026/reinforcement_learning/explaining_decentralized_multi-agent_reinforcement_learning_policies.md)
- [\[ACL 2026\] Reverse Constitutional AI: A Framework for Controllable Toxic Data Generation via Probability-Clamped RLAIF](reverse_constitutional_ai_a_framework_for_controllable_toxic_data_generation_via.md)
- [\[ACL 2026\] The Stackelberg Speaker: Optimizing Persuasive Communication in Social Deduction Games](the_stackelberg_speaker_optimizing_persuasive_communication_in_social_deduction_.md)
- [\[ACL 2026\] HEALing Entropy Collapse: Enhancing Exploration in Few-Shot RLVR via Hybrid-Domain Entropy Dynamics Alignment](healing_entropy_collapse_enhancing_exploration_in_few-shot_rlvr_via_hybrid-domai.md)
- [\[ACL 2026\] Easy Samples Are All You Need: Self-Evolving LLMs via Data-Efficient Reinforcement Learning](easy_samples_are_all_you_need_self-evolving_llms_via_data-efficient_reinforcemen.md)

<!-- RELATED:END -->

---
title: >-
  [Paper Note] FaithLens: Detecting and Explaining Faithfulness Hallucination
description: >-
  [ACL 2026][LLM Safety][Faithfulness Hallucination] This paper proposes FaithLens, an 8B parameter faithfulness hallucination detection model. It undergoes cold-start SFT using high-quality synthetic data subject to three…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Faithfulness Hallucination"
  - "Explainable Detection"
  - "Rule-based RL"
  - "Data Filtering"
  - "Cross-task Generalization"
date: 2026-05-08
content_hash: 88e4a20c21622d73
---

# FaithLens: Detecting and Explaining Faithfulness Hallucination

**Conference**: ACL 2026  
**arXiv**: [2512.20182](https://arxiv.org/abs/2512.20182)  
**Code**: [https://github.com/S1s-Z/FaithLens](https://github.com/S1s-Z/FaithLens)  
**Area**: Reinforcement Learning / Hallucination Detection  
**Keywords**: Faithfulness Hallucination, Explainable Detection, Rule-based RL, Data Filtering, Cross-task Generalization

## TL;DR

This paper proposes FaithLens, an 8B parameter faithfulness hallucination detection model. It undergoes cold-start SFT using high-quality synthetic data subject to three-dimensional filtering (label correctness, explanation quality, and data diversity), followed by further optimization via rule-based reinforcement learning (prediction correctness reward + explanation quality reward). It outperforms GPT-5.2 and o3 across 12 tasks while providing high-quality explanatory outputs.

## Background & Motivation

**Background**: LLMs are widely used for context-based text generation (e.g., RAG, summarization), but are prone to "faithfulness hallucinations" that are inconsistent with or irrelevant to the given context. Detecting such hallucinations is crucial for responsible LLM services.

**Limitations of Prior Work**: (1) Lack of explainability—existing methods treat hallucination detection as a black-box binary classification, outputting labels without reasons, making it impossible for users to locate or understand errors; (2) Inconsistent cross-task generalization—different tasks exhibit varied hallucination patterns (subtle distortions in summaries vs. contradictory claims in RAG), leading to uneven performance in general models; (3) Lack of high-quality data—manual annotation is expensive with low consistency, and synthetic data lacks quality control.

**Key Challenge**: Achieving high detection accuracy while maintaining high explanation quality is difficult. SFT tends to make models mimic training data, leading to memorization of simple samples but poor generalization in complex scenarios. Furthermore, the quality of free-form explanations is hard to verify directly using rules.

**Goal**: To build a cost-effective hallucination detection model that outputs both detection results and explanatory notes, achieving SOTA performance across 12 diverse tasks.

**Key Insight**: A two-stage training approach—cold-starting with carefully filtered synthetic SFT data, followed by GRPO reinforcement learning using cleverly designed rule-based rewards (prediction correctness + explanation quality).

**Core Idea**: The key insight for the explanation quality reward is that if an explanation helps a "novice model" (an untuned Llama-3.1-8B) correctly predict the label, the explanation is sufficiently clear and informative.

## Method

### Overall Architecture

FaithLens training consists of two stages: (1) Cold-start SFT—starting from open-source datasets, training data with explanations is synthesized using a high-level reasoning model (DeepSeek-V3.2-Think) and fine-tuned after three-dimensional filtering; (2) Rule-based RL—further optimization using the GRPO algorithm, where the reward function comprises prediction correctness, explanation quality, and format.

### Key Designs

1. **Three-dimensional Data Filtering Strategy**:

    - **Function**: Ensures label correctness, explanation quality, and data diversity of synthetic training data.
    - **Mechanism**: Label filtering—compares LLM predictions with ground truth and discards inconsistencies (as CoT/explanations for wrong labels may seem coherent but are internally consistent with errors). Explanation quality filtering—measures whether adding an explanation reduces the model's perplexity on the correct label, retaining only samples that decrease perplexity. Diversity filtering—uses K-Medoids clustering to construct a probe set and tests if candidate samples help the probe set predict correctly, retaining training data with positive impacts on diverse samples.
    - **Design Motivation**: Unfiltered synthetic data contains noise and excessive simple samples. Three-dimensional filtering ensures training data is correct, informative, and covers diverse scenarios.

2. **Explanation Quality Reward**:

    - **Function**: Implicitly evaluates the quality of free-form explanations during the RL phase.
    - **Mechanism**: The generated explanation $e$ is fed along with the document and claim into a "novice model" (untuned Llama-3.1-8B-Instruct) to check if it can correctly predict the label based on this explanation. A reward of 1 is given if correct, 0 otherwise. The final reward is defined as $R_{\text{final}} = R_{\text{pred}} + R_{\text{exp}} + R_{\text{format}}$.
    - **Design Motivation**: Verifying free-form text quality directly via rules is nearly impossible. "If even a novice can reach the correct answer through your explanation, then your explanation must be good enough"—this serves as a clever proxy evaluation.

3. **GRPO Reinforcement Learning Training**:

    - **Function**: Further enhances detection accuracy and explanation quality on top of the SFT cold-start.
    - **Mechanism**: For each document-claim pair, $G$ candidates (explanation + prediction) are generated. Each is evaluated with the combined reward, and the policy is updated via group relative advantage estimation in GRPO. KL divergence regularization prevents the policy from deviating too far from the reference.
    - **Design Motivation**: While SFT tends to memorize simple samples, RL drives the model to produce high-quality outputs in complex scenarios through exploration and reward signals.

### Loss & Training

The SFT phase uses standard cross-entropy loss on filtered synthetic data. The RL phase utilizes GRPO (Group Relative Policy Optimization) with a total reward = Prediction Correctness (0/1) + Explanation Quality (0/1) + Format Correctness (0/1). The base model is Llama-3.1-8B-Instruct.

## Key Experimental Results

### Main Results

**Overall Performance Across 12 Tasks (Balanced Accuracy %)**

| Model | Std Dev ↓ | Avg ↑ |
|------|---------|---------|
| GPT-4o | 7.0 | 76.1 |
| o3 | 6.0 | 82.1 |
| GPT-5.2 | - | 85.3 |
| Claude-3.7-Sonnet | 5.3 | 82.6 |
| DeepSeek-V3.2-Think | 5.1 | 84.4 |
| MiniCheck-7B | 9.3 | 76.7 |
| **FaithLens-8B (Ours)** | **4.1** | **85.8** |

### Ablation Study

| Configuration | Avg Accuracy | Description |
|------|----------|------|
| Full FaithLens | 85.8 | Complete model |
| w/o RL (SFT only) | 82.3 | RL Gain +3.5 |
| w/o Explanation Quality Reward | 84.1 | Explanation Reward Gain +1.7 |
| w/o Data Filtering | 79.8 | Filtering Gain +6.0 |
| w/o Diversity Filtering | 81.5 | Diversity Filtering Gain +4.3 |

### Key Findings

- 8B FaithLens outperforms GPT-5.2 (85.8 vs 85.3) and o3 (82.1), offering a magnitude of advantage in cost.
- It achieves the lowest standard deviation (4.1), indicating the most stable cross-task generalization—addressing the "strong in some tasks, weak in others" issue of existing methods.
- The contribution of data filtering (+6.0) is greater than RL (+3.5), indicating that high-quality training data is the foundation.
- Diversity filtering is crucial for cross-task generalization; removing it causes accuracy to drop by 4.3 percentage points.
- The explanation quality reward not only improves explanation quality but also indirectly boosts detection accuracy (+1.7), suggesting an inherent regularization effect in the "explanation $\rightarrow$ prediction" process.

## Highlights & Insights

- The "Novice Model Proxy Evaluation" is an elegant solution for assessing free-form explanation quality—transforming unverifiable text quality issues into verifiable classification correctness problems.
- The progressive "Label $\rightarrow$ Explanation $\rightarrow$ Diversity" design of the 3D data filtering ensures comprehensive quality of training data.
- Achieving performance beyond closed-source giant models with only 8B parameters demonstrates that "carefully designed training strategies > brute-force parameter scaling."

## Limitations & Future Work

- The explanation quality reward depends on the capability of the "novice model"; if the novice model itself is biased, the reward signal may be distorted.
- Synthetic data originates from existing open-source datasets and may inherit their biases.
- Evaluation was limited to English tasks; multilingual generalization remains unverified.
- Future work could explore more fine-grained explanation evaluation (e.g., sentence-level evidence anchoring).

## Related Work & Insights

- **vs MiniCheck**: MiniCheck reached GPT-4o levels using a 7B classifier trained on synthetic data but lacks explanation capabilities; FaithLens provides explanations and surpasses GPT-5.2.
- **vs SelfCheckGPT**: SelfCheckGPT relies on large model inference and is inefficient; FaithLens achieves better performance with an 8B model.
- **vs DeepSeek-V3.2-Think**: While a strong teacher for data synthesis (84.4%), FaithLens surpasses the teacher through RL (85.8%).

## Rating

- Novelty: ⭐⭐⭐⭐ Innovation in explanation quality rewards and 3D filtering, though the overall SFT+RL framework is a standard paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 12 tasks, multiple baselines (including GPT-5.2/o3), and detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear methodological descriptions and complete formulas.
- Value: ⭐⭐⭐⭐⭐ Extremely practical, as an 8B model outperforms GPT-5.2 while providing explanations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] FinGround: Detecting and Grounding Financial Hallucinations via Atomic Claim Verification](finground_detecting_and_grounding_financial_hallucinations_via_atomic_claim_veri.md)
- [\[ACL 2026\] TPA: Next Token Probability Attribution for Detecting Hallucinations in RAG](tpa_next_token_probability_attribution_for_detecting_hallucinations_in_rag.md)
- [\[ACL 2026\] Enhancing Hallucination Detection via Future Context](enhancing_hallucination_detection_via_future_context.md)
- [\[ACL 2026\] Detecting RAG Extraction Attack via Dual-Path Runtime Integrity Game](detecting_rag_extraction_attack_via_dual-path_runtime_integrity_game.md)
- [\[ACL 2026\] Gap-K%: Measuring Top-1 Prediction Gap for Detecting Pretraining Data](gap-k_measuring_top-1_prediction_gap_for_detecting_pretraining_data.md)

</div>

<!-- RELATED:END -->

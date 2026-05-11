---
title: >-
  [Paper Note] When Top-ranked Recommendations Fail: Modeling Multi-Granular Negative Feedback for Explainable and Robust Video Recommendation
description: >-
  [AAAI 2026][Recommender Systems][Negative feedback modeling] This paper proposes ENF (Explainable Negative Feedback), a framework comprising three collaborative MLLM Agents (Profile Agent, Video Agent…
tags:
  - "AAAI 2026"
  - "Recommender Systems"
  - "Negative feedback modeling"
  - "explainable recommendation"
  - "multimodal video understanding"
  - "MLLM Agent"
  - "reinforcement learning"
date: 2026-05-08
content_hash: f28167e498fd38ea
---

# When Top-ranked Recommendations Fail: Modeling Multi-Granular Negative Feedback for Explainable and Robust Video Recommendation

**Conference**: AAAI 2026
**arXiv**: [2511.18700](https://arxiv.org/abs/2511.18700)
**Code**: None
**Area**: Recommender Systems
**Keywords**: Negative feedback modeling, explainable recommendation, multimodal video understanding, MLLM Agent, reinforcement learning

## TL;DR

This paper proposes ENF (Explainable Negative Feedback), a framework comprising three collaborative MLLM Agents (Profile Agent, Video Agent, and Reason Agent) and a progressive S-GRPO reinforcement learning training strategy. ENF is the first approach to achieve explainable prediction and root-cause analysis of implicit negative feedback in video recommendation systems. Deployed on Tencent's news platform, it achieves a 6.2% increase in average watch duration and a 9.4% decrease in quick-skip rate.

## Background & Motivation

Existing video recommendation systems primarily rely on ID-based embedding mappings and collaborative filtering, facing three core challenges:

**1. Scarcity of negative feedback data**: Explicit negative feedback (e.g., clicking "dislike") is highly informative but extremely sparse (accounting for only ~0.3% of all interactions); implicit feedback (e.g., watch duration, skip behavior) is abundant but noisy and low in information density.

**2. Lack of understanding of negative feedback causes**: Existing methods typically extract features by clustering negative feedback signals and then suppress similar recommendations. However, this approach lacks understanding of specific causes and generalizes poorly. For example, if a user dislikes a food video, suppressing all food recommendations without understanding the reason is unreasonable — the user may simply dislike the visual depiction of ingredient handling, not the food theme itself.

**3. Absence of multimodal evaluation**: Although existing LLM-based methods can predict user preferences, they neglect the complex multimodal content of items and lack evaluation of explainable reasoning.

This addresses a practically significant problem: **why do top-ranked recommendations still frequently trigger negative user feedback?** Traditional methods recommend based on high embedding similarity while completely ignoring users' deeper psychological characteristics and fine-grained video content analysis.

## Method

### Overall Architecture

The ENF framework adopts a three-tier Agent architecture: Profile Agent analyzes user behavior to construct a psychological profile → Video Agent performs multimodal video content analysis → Reason Agent integrates both to predict user attitudes and generate explainable reasoning. Training follows a two-stage strategy: SFT cold-start followed by S-GRPO reinforcement learning fine-tuning.

### Key Designs

#### 1. **Profile Agent**: Inferring User Psychological Profiles from Behavioral Patterns

**Function**: Analyzes users' demographic information (age, gender, occupation, interest tags) and sequential viewing behavior (titles, play rates) to infer psychological traits and personality profiles.

**Mechanism**: Traditional recommender systems rely solely on interest tag embeddings, overlooking users' deeper psychological tendencies. For example:
- Fan users may strongly prefer positive content about their idols but reject critical narratives.
- Food enthusiasts may react negatively to overly explicit footage of ingredient handling.

The agent focuses on videos with play_rate < 0.3 (a dissatisfaction indicator), dynamically invoking the Video Agent for multimodal cues when textual title information is insufficient. It iteratively updates the psychological profile (value orientations, tolerance for negative content, etc.) by analyzing each interaction one by one.

**Design Motivation**: User preferences go far beyond surface-level interest tags and are rooted in deeper psychological characteristics. Understanding these latent traits enables more precise user-aligned recommendations.

#### 2. **Video Agent**: Deep Multimodal Video Content Analysis

**Function**: Conducts in-depth analysis of individual videos, not only describing their content but also identifying potentially controversial elements and providing contextual explanations.

**Mechanism**: Leverages the multimodal capabilities of MLLMs to decompose video content, using 16 uniformly sampled frames and video titles as input features. Analysis covers four dimensions:
- Whether the video contains negative events
- Whether it contains vulgar content or content conflicting with user values
- Whether the plot is unengaging
- Whether it contains visually uncomfortable elements

**Design Motivation**: Traditional embedding-based methods cannot identify controversial content within videos, yet such content is precisely the primary trigger of negative feedback.

#### 3. **Reason Agent**: Integrated Judgment and Explainable Reasoning

**Function**: Leverages users' basic information and refined psychological profiles to generate video comprehension from the user's perspective, infer user attitudes, and provide explainable reasoning.

**Mechanism**: Evaluates user–video compatibility along four dimensions:
1. Whether video content aligns with user interests
2. Whether the plot is engaging
3. Whether the content contains negative events or extreme viewpoints
4. Whether visual elements meet the user's sensory tolerance

### Loss & Training

The paper employs **S-GRPO (Stepwise Group Relative Policy Optimization)**, a progressive reinforcement learning strategy that constitutes one of the paper's core innovations.

**Training consists of two stages**:

**Stage 1 – SFT Cold Start**: Using real user feedback reasons, GPT-4o is prompted to generate chain-of-thought explanations for why users disliked specific videos, which serve as SFT data to warm-start the model.

**Stage 2 – S-GRPO Reinforcement Fine-tuning**: A progressive reward mechanism is proposed, consisting of three step-wise rewards $R_{S_i}$:

1. **Binary Judge Reward** $r_{judge}$ (Step 1): Evaluates whether the predicted user attitude is correct.
   - If incorrect, the episode terminates with no reward.
   - If correct, a fixed reward (e.g., 0.5) is granted.
   - If the actual feedback is positive, the episode terminates; if negative, it proceeds to the next step.

2. **Class Reward** $r_{class}$ (Step 2): Evaluates whether the negative feedback category is correctly classified.
   - If correct, an additional reward (e.g., 1.0) is granted and the episode proceeds to Step 3.

3. **Reason Reward** $r_{reason}$ (Step 3): Computes the average ROUGE-1/2/L score between the reasoning content within the `<think>` tags and the actual user feedback reasons.

Advantage computation:

$$A_i = \frac{R_i - \text{mean}(\{R_j\})}{\text{std}(\{R_j\})}$$

Policy optimization objective:

$$\mathcal{J}_{GRPO}(\theta) = \min\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}A_i, \text{clip}\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}, 1-\varepsilon, 1+\varepsilon\right)A_i\right) - \beta\mathcal{D}_{KL}(\pi_\theta||\pi_{ref})$$

This **progressive design** encourages the model to learn from easy to hard: responses that are correct on binary classification but incorrect on multi-class categorization still receive partial reward, while responses that combine correct classification with coherent reasoning receive higher scores.

**Implementation Details**:
- Backbone model: Qwen2.5-VL-7B
- GPT-4o serves as the Profile Agent; Qwen-2.5VL-7B is used for the Video Agent and Reason Agent
- Full-parameter fine-tuning on 4× 80G GPUs
- Learning rate: 1e-6; group size: G=8

## Key Experimental Results

### Main Results – Explicit Negative Feedback Prediction

| Model | Size | Acc | Recall | F1 | Class_Acc | Reasoning |
|-------|------|-----|--------|----|-----------|-----------|
| GPT-4o | - | 0.882 | 0.630 | 0.739 | 0.568 | 0.402 |
| DeepSeek | - | 0.849 | 0.440 | 0.594 | 0.352 | 0.266 |
| Qwen2.5VL | 7B | 0.815 | 0.423 | 0.564 | 0.296 | 0.229 |
| Video-R1 | 7B | 0.835 | 0.540 | 0.667 | 0.432 | 0.318 |
| VideoChat-R1 | 7B | 0.842 | 0.654 | 0.739 | 0.500 | 0.383 |
| **Our Video Agent** | **7B** | **0.861** | **0.808** | **0.750** | **0.654** | **0.537** |

Key findings: Recall exceeds GPT-4o by +17.8%, Class_Acc by +8.6%, and Reasoning by +13.5%.

### Implicit Negative Feedback Prediction

| Model | Size | Acc | Precision | Recall | F1 | Class_Acc |
|-------|------|-----|-----------|--------|----|-----------|
| GPT-4o | - | 0.575 | 0.396 | 0.796 | 0.521 | 0.502 |
| SASRec | - | 0.448 | 0.230 | 0.358 | 0.279 | - |
| VideoChat-R1 | 7B | 0.561 | 0.384 | 0.775 | 0.516 | 0.512 |
| **Our ENF** | **7B** | **0.612** | **0.404** | **0.782** | **0.533** | **0.543** |

Implicit feedback prediction is substantially harder than explicit feedback (peak accuracy of only 61.2%), yet ENF consistently outperforms all baselines.

### Ablation Study

**Video Agent Training Ablation**:

| SFT | RL | S-GRPO | Acc | F1 | Class_Acc | Reasoning |
|-----|-----|--------|-----|----|-----------|-----------|
| ✗ | ✗ | ✗ | 0.815 | 0.423 | 0.296 | 0.229 |
| ✗ | ✓ | ✓ | 0.830 | 0.686 | 0.592 | 0.492 |
| ✓ | ✗ | ✗ | 0.851 | 0.615 | 0.346 | 0.312 |
| ✓ | ✓ | ✗ | 0.845 | 0.667 | 0.412 | 0.339 |
| ✓ | ✓ | ✓ | **0.861** | **0.750** | **0.654** | **0.537** |

**Reason Agent Ablation**:

| Profile Agent | Video Agent | S-GRPO | Acc | F1 | Class_Acc |
|---------------|-------------|--------|-----|----|-----------|
| ✗ | ✗ | ✗ | 0.528 | 0.482 | 0.435 |
| ✓ | ✓ | ✓ | **0.612** | **0.533** | **0.543** |

### Production Platform Validation

| Metric | Base RS | Base RS + ENF | Gain |
|--------|---------|---------------|------|
| Avg. Watch Duration | 47.6% | 53.8% | +13.0% |
| Quick-Skip Rate | 23.7% | 14.3% | -39.7% |
| Dislike Rate | 0.61% | 0.35% | -42.6% |

### Key Findings

1. The **progressive learning of S-GRPO** is critical: without it, models tend to learn only binary classification while neglecting category prediction and reasoning.
2. **SFT cold-start** provides user-side prior knowledge; removing it leads to significant drops in prediction accuracy.
3. The **Profile Agent** supplies richer psychological profile features and is essential for holistic user modeling.
4. **Implicit feedback prediction** is far more challenging than explicit feedback prediction, as real user behavior is influenced by multiple factors and exhibits inherent stochasticity.
5. **Traditional methods (e.g., SASRec)** perform poorly in cold-start scenarios requiring fine-grained item discrimination.

## Highlights & Insights

1. **The problem formulation is highly practical and significant**: rather than simply predicting whether a user likes an item, the paper focuses on explaining **why** negative feedback occurs, which carries substantial implications for improving recommendation systems.
2. **The progressive reward design of S-GRPO is elegant**: progressing from easy to hard (binary classification → multi-class categorization → reasoning explanation), where each step is rewarded only if the previous step is correct, avoiding the ambiguity of a single reward signal that cannot distinguish errors at different levels.
3. **Real-world deployment validation**: the approach is evaluated not only on offline datasets but also validated on the Tencent news platform, lending strong credibility to the findings.
4. **Construction of the TVNF dataset**: a multimodal video recommendation dataset containing real user dislike reasons, filling a gap in the field.

## Limitations & Future Work

1. **Data scale**: The TVNF dataset contains only ~1K explicit feedback annotations, which is relatively small.
2. **Profile Agent relies on GPT-4o**: Deployment costs may be high in practice; future work could consider distilling it into a smaller model.
3. **Limited negative feedback categories**: Only four types of causes are considered, whereas real users' dislikes may be far more diverse.
4. **Ground truth for implicit feedback** is derived from GPT-4o annotations rather than real users, which may introduce bias.
5. **Cross-platform generalizability**: The method's effectiveness in domains beyond short video recommendation (e.g., news, e-commerce) remains to be verified.

## Related Work & Insights

- Unlike traditional negative feedback methods such as DFN, CDR, and SINE, this paper is the first to leverage LLMs to understand and explain negative feedback.
- The progressive reward mechanism of S-GRPO is generalizable to reinforcement learning training for other multi-step reasoning tasks.
- The three-Agent collaborative framework offers a new paradigm for "simulating user behavior with LLMs."
- Generalization experiments on MovieLens and Steam datasets validate the cross-domain potential of the proposed approach.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First systematic use of MLLM Agents to explain negative feedback in recommender systems; S-GRPO design is innovative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Offline evaluation + ablation + cross-dataset generalization + real-world deployment; highly comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem motivation is clear; method description is detailed.
- **Value**: ⭐⭐⭐⭐⭐ — Addresses a long-neglected but critically important problem of negative feedback understanding in recommender systems, with strong practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Behavior Tokens Speak Louder: Disentangled Explainable Recommendation with Behavior Vocabulary](behavior_tokens_speak_louder_disentangled_explainable_recommendation_with_behavi.md)
- [\[AAAI 2026\] Semi-Supervised Synthetic Data Generation with Fine-Grained Relevance Control for Short Video Search Relevance Modeling](semi-supervised_synthetic_data_generation_with_fine-grained_relevance_control_fo.md)
- [\[AAAI 2026\] Tokenize Once, Recommend Anywhere: Unified Item Tokenization for Multi-domain LLM-based Recommendation](tokenize_once_recommend_anywhere_unified_item_tokenization_for_multi-domain_llm-.md)
- [\[AAAI 2026\] Align³GR: Unified Multi-Level Alignment for LLM-based Generative Recommendation](align3gr_unified_multi-level_alignment_for_llm-based_generat.md)
- [\[AAAI 2026\] Preference is More Than Comparisons: Rethinking Dueling Bandits with Augmented Human Feedback](preference_is_more_than_comparisons_rethinking_dueling_bandits_with_augmented_hu.md)

</div>

<!-- RELATED:END -->

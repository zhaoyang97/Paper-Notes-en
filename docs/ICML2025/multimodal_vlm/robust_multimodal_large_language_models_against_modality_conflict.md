---
title: >-
  [Paper Note] Robust Multimodal Large Language Models Against Modality Conflict
description: >-
  [ICML 2025][Multimodal VLM][Multimodal Hallucination] Reveals an overlooked source of MLLM hallucinations—modality conflict (the inherent incompatibility between visual and textual inputs). It formally defines modality conflicts across three levels: object, attribute, and relation, constructs the MMMC dataset with 20K samples, and proposes three mitigation approaches (prompt engineering, SFT, and RL), among which RL achieves the best performance.
tags:
  - "ICML 2025"
  - "Multimodal VLM"
  - "Multimodal Hallucination"
  - "Modality Conflict"
  - "Robustness"
  - "Reinforcement Learning"
  - "SFT"
date: 2026-05-08
content_hash: e4694dd8c83e3579
---

# Robust Multimodal Large Language Models Against Modality Conflict

**Conference**: ICML 2025  
**arXiv**: [2507.07151](https://arxiv.org/abs/2507.07151)  
**Code**: [https://github.com/zmzhang2000/MMMC](https://github.com/zmzhang2000/MMMC)  
**Area**: Multimodal VLM  
**Keywords**: Multimodal Hallucination, Modality Conflict, Robustness, Reinforcement Learning, SFT

## TL;DR

Reveals an overlooked source of MLLM hallucinations—modality conflict (the inherent incompatibility between visual and textual inputs). It formally defines modality conflicts across three levels: object, attribute, and relation, constructs the MMMC dataset with 20K samples, and proposes three mitigation approaches (prompt engineering, SFT, and RL), among which RL achieves the best performance.

## Background & Motivation

### 1. MLLM Hallucination

Multimodal Large Language Models (MLLMs) demonstrate outstanding performance on tasks like VQA, but are prone to hallucinations—generating information inconsistent with the inputs. Existing work primarily focuses on the inconsistency between model outputs and inputs.

### 2. Overlooked Source of Hallucination: Modality Conflict

This work focuses on conflicts within the inputs themselves: when textual questions presuppose information that does not exist in the images, MLLMs struggle. For example, given an image of a dog surfing, if a user asks, "What color is the ball?", when there is no ball in the image at all, the model may hallucinate "The ball is green."

### 3. Differences from Existing Work

Existing hallucination mitigation methods (such as improving training data, adjusting decoding strategies, and RLHF alignment) mainly pursue more precise cross-modal feature alignment. However, even with perfect modal alignment, models still hallucinate when faced with inherently conflicting inputs—this fundamentally requires enhancing the model's ability to identify and handle input contradictions.

## Method

### Overall Architecture

1. Formally define three types of modality conflict (object, attribute, relation).
2. Construct the MMMC dataset (20K samples: 18K for training and 2K for testing).
3. Propose and compare three mitigation methods: prompt engineering, SFT, and RL.

### Key Designs

#### 1. Formalization of Modality Conflict

Let the visual input be $\mathcal{V}$ and the textual input be $\mathcal{T}$:

**Object Conflict**: Text refers to objects that do not exist in the image.
$$\text{Obj}(\mathcal{T}) \not\subseteq \text{Obj}(\mathcal{V})$$

**Attribute Conflict**: Same object but different attributes (e.g., text states "red apple" while the image shows a green apple).
$$\text{Attr}(\mathcal{O}_i^{\mathcal{T}}) \neq \text{Attr}(\mathcal{O}_i^{\mathcal{V}})$$

**Relation Conflict**: Same objects but different relations (e.g., text states "the cat is on the table" while the image shows the cat on the floor).
$$\text{Rel}(\mathcal{O}_i^{\mathcal{T}}, \mathcal{O}_j^{\mathcal{T}}) \neq \text{Rel}(\mathcal{O}_i^{\mathcal{V}}, \mathcal{O}_j^{\mathcal{V}})$$

#### 2. MMMC Dataset Construction

Constructed through 4 steps based on the Visual Genome dataset:
1. **Base Question Sampling**: Randomly sample questions from the original dataset.
2. **Key Component Detection**: Use an LLM to detect objects, attributes, and relations in the image.
3. **Component Replacement**: Replace components in the question with information that conflicts with the image.
4. **Answer Generation**: Instead of having the VLM directly answer by looking at the image (to avoid hallucinations), an LLM generates the correct ground-truth answers based on textual information (e.g., "There is no ball in the image").
5. Human review to ensure quality.

#### 3. Method 1: Prompt Engineering

Prepend a prompt to the question: "Please check if the image contains mentioned information and answer the question"

$$\mathcal{A} \sim \pi_\theta(\mathcal{A}|\mathcal{V}, p(\mathcal{T}))$$

Pros: Zero-cost, requires no additional training. Cons: Performance depends on the instruction-following capability of the model.

#### 4. Method 2: Supervised Fine-Tuning (SFT)

Fine-tuned on the MMMC training set using language modeling objectives:

$$\pi_\theta^* = \arg\min_\theta \mathbb{E}[-\log \pi_\theta(\mathcal{A}|\mathcal{V}, \mathcal{T})]$$

Pros: Can leverage training data. Cons: Primarily learns style adaptation of the target domain, exhibiting limited generalization to unseen data.

#### 5. Method 3: Reinforcement Learning (RL)

Models the conditional generation as an MDP and designs a reward function to evaluate whether the model correctly identifies modality conflicts:
- State: $s_t = (\mathcal{V}, \mathcal{T}, a_{<t})$
- Action: $a_t$ (the generated token)
- Reward: Based on whether the response correctly points out the conflict

Pros: Learns a more robust policy through trial-and-error exploration. Performs the best in experiments.

## Key Experimental Results

### Main Results: Performance of Different MLLMs on MMMC

| Model | Object Conflict Acc | Attribute Conflict Acc | Relation Conflict Acc | Average |
|------|------------|------------|------------|------|
| InternLM-XComposer2 | 32.1 | 28.5 | 25.3 | 28.6 |
| LLaVA-1.5 | 35.7 | 31.2 | 27.8 | 31.6 |
| Qwen-VL-Chat | 38.2 | 33.6 | 30.1 | 34.0 |
| GPT-4o | 62.5 | 55.3 | 48.7 | 55.5 |

Most MLLMs exhibit extremely low accuracy (<40%) in modality-conflict scenarios, and even GPT-4o only achieves ~55%.

### Comparison of Three Methods (Taking LLaVA-1.5 as an Example)

| Method | Object Conflict | Attribute Conflict | Relation Conflict | Average | Original VQA Retention |
|------|---------|---------|---------|------|------------|
| Baseline (No Intervention) | 35.7 | 31.2 | 27.8 | 31.6 | 100% |
| Prompt Engineering | 42.3 | 37.8 | 33.5 | 37.9 | ~99% |
| SFT | 68.5 | 62.1 | 56.3 | 62.3 | ~95% |
| **RL** | **74.2** | **67.8** | **61.5** | **67.8** | **~93%** |

- RL achieves the largest improvement (+36.2% on average), but original VQA performance decreases slightly.
- SFT shows stable performance and maintains a better balance with the original capability.
- Prompt Engineering yields limited improvement but comes with zero cost.

### Key Findings

- Relation conflict is the most difficult to handle (showing the worst performance across all methods) as it requires more complex spatial reasoning.
- The RL method significantly outperforms SFT across all three conflict categories, indicating that trial-and-error exploration is more effective for such 'discrimination' tasks.
- SFT performs well within the training distribution but has limited generalization—the authors point out that SFT learns more about 'style' than 'capability'.

## Highlights & Insights

- **Uniqueness of Problem Definition**: Formally defines and investigates modality conflicts between inputs as a source of hallucination for the first time, complementing the existing "output-input conflict" perspective.
- **Three-Level Conflict Classification**: The hierarchical definition of object, attribute, and relation is both systematic and practical.
- **Fairness of Method Comparison**: The three methods represent a full spectrum from zero-cost to retraining, with a well-designed ablation study.
- **Ingenuity of Dataset Construction**: Employs an LLM to generate answers based purely on textual information (rather than having a VLM inspect the image), successfully avoiding the introduction of new hallucinations.

## Limitations & Future Work

- The MMMC dataset is built on Visual Genome, so its scene diversity is constrained by the coverage of the source dataset.
- The reward function design for the RL method is still relatively simplistic; integrating more fine-grained conflict detection rewards might yield further improvements.
- Performance under real-world user interaction scenarios remains untested—actual user conflict queries might be more subtle/implicit.
- Integrating conflict identification capabilities during the pre-training stage, rather than just post-training, is worth exploring.
- The effectiveness of joint training with SFT+RL (first SFT, then RL) remains to be validated.

## Related Work & Insights

- **vs Hallucination Detection such as POPE/CHAIR**: These methods evaluate output hallucinations, whereas this work focuses on those triggered by conflicts at the input end.
- **vs Longpre et al. (2021)**: Studies knowledge conflicts in LLMs, while this work extends the scope to multimodal vision-language scenarios.
- **vs Visual Adversarial Attacks**: Adversarial attacks modify image pixels, whereas this work generates semantic conflicts via natural language.
- **Inspiration**: Modality conflict can be integrated as one of the standard dimensions for MLLM safety evaluation.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Formally defines and systematically studies multimodal input conflicts for the first time.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensively evaluates multiple models, compares three methods, and conducts thorough ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Offers clear definitions, well-structured methodology, and intuitive diagrams.
- Value: ⭐⭐⭐⭐⭐ Opens up a new dimension for MLLM robustness research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] DeepAlign: Mitigating Modality Conflict through Modality-Specific Alignment](../../CVPR2026/multimodal_vlm/deepalign_mitigating_modality_conflict_through_modality-specific_alignment.md)
- [\[CVPR 2026\] Predictive Regularization Against Visual Representation Degradation in Multimodal Large Language Models](../../CVPR2026/multimodal_vlm/predictive_regularization_against_visual_representation_degradation_in_multimoda.md)
- [\[ICML 2025\] Ranked from Within: Ranking Large Multimodal Models Without Labels](ranked_from_within_ranking_large_multimodal_models_without_labels.md)
- [\[CVPR 2026\] Enhance-then-Balance Modality Collaboration for Robust Multimodal Sentiment Analysis](../../CVPR2026/multimodal_vlm/enhance-then-balance_modality_collaboration_for_robust_multimodal_sentiment_anal.md)
- [\[ACL 2025\] Single-to-mix Modality Alignment with Multimodal Large Language Model for Document Image Machine Translation](../../ACL2025/multimodal_vlm/single-to-mix_modality_alignment_with_multimodal_large_language_model_for_docume.md)

</div>

<!-- RELATED:END -->

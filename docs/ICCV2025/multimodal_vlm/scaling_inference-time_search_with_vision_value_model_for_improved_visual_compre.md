---
title: >-
  [Paper Note] Scaling Inference-Time Search with Vision Value Model for Improved Visual Comprehension
description: >-
  [ICCV 2025][Multimodal VLM][Vision-language models] This paper proposes the Vision Value Model (VisVM), trained via temporal difference (TD) learning, to guide sentence-level inference-time search in VLMs for generating higher-quality descriptive captions. Compared to greedy decoding and CLIP-PRM, VisVM search significantly reduces hallucination (CHAIRs from 32.4 to 26.2), and data generated through this process, when used for self-training, yields an average improvement of 10.8% across 9 benchmarks.
tags:
  - ICCV 2025
  - Multimodal VLM
  - Vision-language models
  - inference-time search
  - vision value model
  - hallucination mitigation
  - self-training
  - temporal difference learning
date: 2026-05-08
content_hash: 7a0a5c01fd7ea129
---

# Scaling Inference-Time Search with Vision Value Model for Improved Visual Comprehension

**Conference**: ICCV 2025
**arXiv**: 2412.03704
**Code**: Not released
**Area**: Multimodal VLM
**Keywords**: Vision-language models, inference-time search, vision value model, hallucination mitigation, self-training, temporal difference learning

## TL;DR

This paper proposes the Vision Value Model (VisVM), trained via temporal difference (TD) learning, to guide sentence-level inference-time search in VLMs for generating higher-quality descriptive captions. Compared to greedy decoding and CLIP-PRM, VisVM search significantly reduces hallucination (CHAIRs from 32.4 to 26.2), and data generated through this process, when used for self-training, yields an average improvement of 10.8% across 9 benchmarks.

## Background & Motivation

Vision-language models (VLMs) have advanced rapidly on multimodal tasks, yet two core challenges persist:

**Visual hallucination**: Models generate content inconsistent with the image, including non-existent objects and erroneous descriptions.

**Neglect of secondary regions**: Insufficient attention to less salient image regions results in incomplete descriptions.

In the LLM domain, inference-time search (e.g., OpenAI-O1) has proven effective for improving output quality. Extending this to VLMs, however, introduces a unique challenge: **VLM tasks lack clear outcome metrics**. Mathematical and programming problems have unambiguous correctness criteria, but descriptive captioning is open-ended—each generated sentence must be locally accurate while maintaining global coherence.

**Limitations of existing PRMs**: Conventional process reward models (PRMs) evaluate only the immediate reward of the current step. In image captioning, however, a sentence that appears locally sound may trigger hallucinations in subsequent steps. For instance, in describing a complex scene, the sentence "There is a red car in the image" might receive a high score at the current step, yet if subsequent sentences continue elaborating on a non-existent car, overall quality degrades sharply.

## Method

### Overall Architecture

VLM generation is modeled as a Markov Decision Process (MDP), where each step produces one sentence:

- **State space** $\mathcal{S}$: previously generated sentences + image
- **Action space** $\mathcal{A}$: the sentence generated at the current step
- **Reward function** $\mathcal{R}$: CLIP similarity as process reward
- **Discount factor** $\gamma$

VisVM predicts **long-term value** rather than immediate reward within this MDP, thereby avoiding myopic decisions.

### VisVM Training (TD Learning)

The core training objective of VisVM is to predict the long-term value of each sentence via temporal difference learning:

$$L(\rho) = -\mathbb{E}_{(y_i, y_{i+1}, I) \sim \mathcal{D}} \left( r_{s_i} + \gamma V_\rho(y_{i+1}, I) - V_\rho(y_i, I) \right)^2$$

where:
- $V_\rho(y_i, I)$ is VisVM's long-term value prediction for state $(y_i, I)$
- $r_{s_i}$ is the immediate reward based on CLIP similarity
- $\gamma$ is the discount factor

**Two key designs**:
1. **Forward-looking coherence**: Unlike PRMs that evaluate only the current sentence reward, VisVM uses TD learning to anticipate future consequences, assessing long-term effects rather than immediate responses.
2. **Comprehensive visual grounding**: CLIP-based text-image similarity is employed so that the reward signal captures rich visual semantics.

### Model Implementation

- A linear layer serving as a value head is appended on top of the penultimate layer of LLaVA-Next-Mistral-7B.
- The value head outputs a single scalar representing the cumulative reward / long-term value of the current sentence.
- The VLM's own visual encoder (CLIP-ViT / SigLIP) is used as the PRM—a **self-rewarding mechanism** requiring no external model.

### Training Data Construction

- 9,215 images are sampled from the COCO 2017 training set.
- Nine descriptive captioning prompts from LLaVA-150K are employed.
- For each image-prompt pair, 5 distinct responses are generated using greedy decoding and sampling at varying temperatures.
- Paragraphs are decomposed into (current sentence, next sentence, image) triples.
- The final dataset comprises **378K** training samples.

### Inference-Time Search

At each search step:
1. $N$ temperature configurations ($[0.1, 0.3, 0.5, 0.7, 0.9]$) plus greedy decoding are used.
2. $K$ candidate sentences are sampled per configuration.
3. VisVM estimates the long-term value of each candidate.
4. The candidate with the highest value is selected as the output for the current step.
5. The process repeats until the end-of-sequence token is generated.

## Key Experimental Results

### Main Results: Hallucination Evaluation

| Base Model | Search Method | CHAIRs ↓ | CHAIRi ↓ | MMHal ↑ | MMHal rate ↓ | AMBER Cov ↑ |
|---|---|---|---|---|---|---|
| LLaVA-Next-7B | Greedy (default) | 32.4 | 5.9 | 2.94 | 0.52 | 63.9 |
| LLaVA-Next-7B | MCTS | 25.9 | 4.7 | 3.24 | 0.37 | 67.3 |
| LLaVA-Next-7B | BoN (N=30) | 27.1 | 5.2 | 3.06 | 0.45 | 65.3 |
| LLaVA-Next-7B | CLIP-PRM search | 28.4 | 5.5 | 2.96 | 0.49 | 66.1 |
| LLaVA-Next-7B | **VisVM search** | **26.2** | **4.6** | **3.30** | **0.39** | **66.8** |
| Qwen2-VL-7B | Greedy (default) | 30.8 | 5.2 | 3.27 | 0.37 | 69.4 |
| Qwen2-VL-7B | **VisVM search** | **24.5** | **3.3** | **3.39** | **0.29** | **73.5** |

### Self-Training Results (Benchmark Improvement After SFT)

| Base Model | SFT Data Source | MM-Vet | MMBench | MMMU | MathVista | CVBench | LLAVA-W | MMStar | CHAIRs ↓ | Avg. Gain |
|---|---|---|---|---|---|---|---|---|---|---|
| LLaVA-Next-7B | None (original) | 45.2 | 74.9 | 34.2 | 38.5 | 65.8 | 76.9 | 36.0 | 32.4 | — |
| LLaVA-Next-7B | Greedy decoding | 43.5 | 74.6 | 34.9 | 37.8 | 66.2 | 75.1 | 36.7 | 33.2 | -1.6% |
| LLaVA-Next-7B | GPT4o-BoN(30) | 47.1 | 76.1 | 35.4 | 40.9 | 67.9 | 77.3 | 36.9 | 30.0 | +4.9% |
| LLaVA-Next-7B | **VisVM search** | **48.3** | **76.7** | **36.1** | **42.3** | **69.8** | **78.4** | **38.0** | **22.6** | **+10.8%** |
| Qwen2-VL-7B | **VisVM search** | **58.9** | **84.1** | **49.7** | **61.1** | **76.2** | **88.2** | **57.0** | **21.4** | **+7.3%** |

### Ablation Study: Effect of Different PRMs

| Search Method | CHAIRs ↓ | CHAIRi ↓ | MMHal ↑ | MMHal rate ↓ | AMBER Cov ↑ |
|---|---|---|---|---|---|
| Greedy (default) | 32.4 | 5.9 | 2.94 | 0.52 | 63.9 |
| CLIP-VisVM search | 26.2 | 4.6 | 3.30 | 0.39 | 66.8 |
| SigLIP-VisVM search | **25.6** | **4.4** | **3.31** | **0.36** | **67.5** |

### Key Findings

1. **Human evaluation**: VisVM search achieves a win rate of 75.8% over greedy decoding and 62.4% over CLIP-PRM.
2. **Search budget scaling**: VisVM search is approximately 2× more efficient than CLIP-PRM—VisVM at step budget 8 achieves performance comparable to CLIP-PRM at step budget 16.
3. **Computational efficiency**: Compared to MCTS, VisVM search requires approximately 1/7 of the GPU hours while achieving comparable or superior performance.
4. **Stronger PRM → better VisVM**: Replacing CLIP with SigLIP as the PRM for training VisVM further reduces hallucination.

## Highlights & Insights

1. **Introducing RL value functions into VLM inference-time search**: TD learning predicts long-term value rather than immediate reward, representing a fundamental improvement over existing PRMs.
2. **Self-rewarding + self-training closed loop**: The PRM is derived from the VLM's own visual encoder, and SFT data is generated by the VLM itself, realizing genuine annotation-free self-improvement.
3. **Scaling law for inference-time compute**: The paper confirms that scaling laws for inference-time computation also hold for VLM visual understanding tasks.
4. **Practical significance**: Self-training on only 9,215 COCO images yields double-digit improvements across 9 benchmarks.

## Limitations & Future Work

1. **Substantially increased inference cost**: Even though more efficient than MCTS, VisVM search still requires several times more computation than greedy decoding.
2. **Validation limited to descriptive captioning**: Current experiments focus on image description; effectiveness on structured tasks such as VQA and reasoning remains unexplored.
3. **Reward signal dependent on CLIP**: CLIP's semantic coverage is limited and may be insufficient for scenarios requiring fine-grained understanding, such as text recognition and spatial relation reasoning.
4. **Step granularity fixed at sentence level**: Modeling the MDP at the sentence level is relatively coarse; if quality differences primarily occur at the word level, this formulation lacks flexibility.

## Related Work & Insights

- The inference-time search paradigm of the **OpenAI O1** series directly inspires this work, extended here to the visual multimodal domain.
- **Alternative to RLHF/DPO**: This approach requires no human preference annotations, relying instead on self-rewarding via the visual encoder.
- **Implications for future self-improving VLMs**: If VisVM could be iteratively replaced by a stronger version of itself, a continuous self-improvement loop might emerge.

## Rating

⭐⭐⭐⭐ (4/5)

- **Novelty**: ⭐⭐⭐⭐ — Introducing RL value functions into VLM inference-time search is a novel and well-motivated idea.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers 3 base models, multiple baselines, human evaluation, and self-training validation.
- **Value**: ⭐⭐⭐ — Increased inference cost limits practical deployment.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear presentation with complete mathematical derivations.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Instruction-Oriented Preference Alignment for Enhancing Multi-Modal Comprehension Capability of MLLMs](instruction-oriented_preference_alignment_for_enhancing_multi-modal_comprehensio.md)
- [\[ICCV 2025\] LATTE: Collaborative Test-Time Adaptation of Vision-Language Models in Federated Learning](latte_collaborative_test-time_adaptation_of_vision-language_models_in_federated_.md)
- [\[ICCV 2025\] Multi-Cache Enhanced Prototype Learning for Test-Time Generalization of Vision-Language Models](multi-cache_enhanced_prototype_learning_for_test-time_generalization_of_vision-l.md)
- [\[ICCV 2025\] Is Less More? Exploring Token Condensation as Training-free Test-time Adaptation](is_less_more_exploring_token_condensation_as_training-free_test-time_adaptation.md)
- [\[ICCV 2025\] Instruction-Grounded Visual Projectors for Continual Learning of Generative Vision-Language Models](instruction-grounded_visual_projectors_for_continual_learning_of_generative_visi.md)

<!-- RELATED:END -->

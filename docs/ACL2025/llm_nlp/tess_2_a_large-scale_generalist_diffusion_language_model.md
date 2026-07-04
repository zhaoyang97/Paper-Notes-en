---
title: >-
  [Paper Note] TESS 2: A Large-Scale Generalist Diffusion Language Model
description: >-
  [ACL2025][LLM (Other)][diffusion language model] TESS 2 is proposed as the first large-scale generalist instruction-following diffusion language model adapted from an existing autoregressive model. Through an adaptation training scheme involving UL2 masking + label shifting + bidirectional attention, combined with reward guidance during inference, it matches or even outperforms equivalent AR models on QA and instruction-following tasks.
tags:
  - "ACL2025"
  - "LLM (Other)"
  - "diffusion language model"
  - "simplex diffusion"
  - "instruction tuning"
  - "reward guidance"
  - "inference-time compute"
date: 2026-05-08
content_hash: 0ae93dee15bca0a0
---

# TESS 2: A Large-Scale Generalist Diffusion Language Model

**Conference**: ACL2025  
**arXiv**: [2502.13917](https://arxiv.org/abs/2502.13917)  
**Code**: [hamishivi/tess-2](https://github.com/hamishivi/tess-2)  
**Area**: LLM/NLP  
**Keywords**: diffusion language model, simplex diffusion, instruction tuning, reward guidance, inference-time compute

## TL;DR

TESS 2 is proposed as the first large-scale generalist instruction-following diffusion language model adapted from an existing autoregressive model. Through an adaptation training scheme involving UL2 masking + label shifting + bidirectional attention, combined with reward guidance during inference, it matches or even outperforms equivalent AR models on QA and instruction-following tasks.

## Background & Motivation

**Background**: Currently, language models primarily adopt the autoregressive (AR) paradigm, which has achieved tremendous success across various tasks. While diffusion models excel in domains like vision and audio, they remain in the early stages of exploration for language generation.

**Limitations of Prior Work**: AR models suffer from inherent limitations in planning and self-correction (as token-by-token generation prevents backtracking), and controlling inference-time compute relies on expensive Chain-of-Thought (CoT) methods. Existing diffusion language models remain at small scales, focusing on intrinsic metrics like perplexity, and lack evaluation on general downstream tasks.

**Key Challenge**: Diffusion LMs naturally possess controllable inference-time compute (by adjusting diffusion steps) and plug-and-play guidance capabilities, but they have not yet been successfully scaled to large-scale general instruction-following scenarios.

**Goal**: To bridge the gap in general task performance between diffusion LMs and AR LMs, providing a complete training scheme to adapt AR models into diffusion LMs.

**Key Insight**: Instead of training from scratch, this work performs diffusion adaptation based on an existing AR model (Mistral-7B) to reuse pre-trained knowledge, and introduces reward guidance to achieve alignment during inference without additional training.

**Core Idea**: A three-stage scheme consisting of AR model initialization + simplex diffusion + instruction tuning is employed to train a generalist diffusion LM, with reward guidance utilized for training-free alignment.

## Method

### Overall Architecture

The training of TESS 2 consists of three stages:

1. **Diffusion Adaptation**: Adapting the AR model (Mistral-7B) into a diffusion LM.
2. **Instruction Tuning**: Fine-tuning on instruction data.
3. **Reward Guidance (Inference-time)**: Utilizing a reward model to guide the generation process at each diffusion step.

### Key Design 1: Simplex Diffusion Architecture

- **Function**: Diffusion is performed on the probability simplex, rather than in the embedding space or a discrete space.
- **Why**: Maintains the continuity of the diffusion process while accommodating discrete text data; the cross-entropy loss is more stable than MSE.
- **Mechanism**:
    - Map each token $w$ to a $k$-logit simplex representation $\mathbf{s}^w \in \{\pm k\}^{|\mathcal{V}|}$.
    - Convert to a probability distribution $\mathbf{p}^w = \text{softmax}(\mathbf{s}^w)$ via softmax.
    - Forward diffusion: $\mathbf{S}_t = \sqrt{\bar{\alpha}_t}\mathbf{S}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}_t$.
    - Training loss: Cross-entropy $\mathcal{L} = \mathbb{E}[-\sum_{i=1}^{L}\log p_\theta(w_i|\mathbf{S}_t, t)]$.
    - Reverse sampling: 100-step iterative denoising.

### Key Design 2: Three Elements of AR-to-Diffusion Adaptation

- **UL2 Masking**: Mixtures of span infilling and prefix completion training objectives; the former enhances generalization, while the latter aligns with downstream usage.
- **Label Shifting**: Predicting the token at the next position during training (predicting $w_{i+1}$ at position $i$), which aligns with AR pre-training's next-token prediction and accelerates convergence.
- **Full Bidirectional Attention**: Disables the causal mask and utilizes full bidirectional attention, fully exploiting the advantage of the entire sequence information flow in diffusion LMs.

### Key Design 3: Reward Guidance

- **Function**: Utilizing a reward model to guide the diffusion generation process during inference.
- **Why**: Achieves preference alignment without additional training, which is a unique advantage of diffusion models compared to AR models.
- **Mechanism**:
    - At each diffusion step, take the model's prediction $\hat{\mathbf{S}}_\theta$ and map it to embeddings via softmax.
    - Input the embeddings into a reward model to obtain a scalar reward $R$.
    - Perform gradient ascent on the prediction: $\hat{\mathbf{S}}_\theta := \hat{\mathbf{S}}_\theta + \eta \cdot \nabla_\theta R$.
    - $\eta$ controls the guidance strength; excessively high values lead to degradation (similar to reward hacking).

### Loss & Training

- **Adaptation**: Trained on Dolma 1.7 for 200k steps (~45B tokens), with a constant learning rate of $1\times10^{-5}$ and a batch size of 112.
- **Instruction Tuning**: Trained on Tulu 2 SFT mixture (~326k samples) for 3 epochs with linear warmup + cooldown.
- **Base Model Selection**: Comparing RoBERTa, Llama 2, Llama 3, and Mistral, Mistral-7B-v0.1 was selected (possibly because prefix-LM pre-training makes it better suited for bidirectional attention adaptation).

## Key Experimental Results

### Table 1: Comparison of base model adaptation (35k steps)

| Base Model | Perplexity↓ | Mauve↑ | d-1↑ | Entropy |
|----------|-------------|--------|------|---------|
| Random init | 54.4 | 0.92 | 0.55 | 5.7 |
| RoBERTa | 20.2 | 0.93 | 0.36 | 4.8 |
| Llama 2 | 3619.2 | 0.01 | 0.94 | 7.7 |
| Llama 3 | 880.4 | 0.93 | 0.97 | 7.8 |
| **Mistral** | **24.3** | **0.95** | 0.62 | 6.3 |

**Findings**: Llama models experience difficulty in convergence when adapted to bidirectional attention due to full causal pre-training; Mistral exhibits the best performance, likely owing to its prefix-LM pre-training.

### Table 3: Downstream task performance after instruction tuning

| Model | AlpacaEval | SQuAD | TriviaQA | IFEval | BBH | GSM8k | GSM8k(ft) |
|------|-----------|-------|----------|--------|-----|-------|-----------|
| Mistral v0.1 AR | 77.1 | 86.0 | 50.4 | 36.8 | 43.3 | 52.5 | 51.2 |
| DiffuLlama | 0.2 | 34.9 | 19.7 | 14.4 | 1.9 | 0.0 | 63.1 |
| TESS 2 v0.1 | 63.1 | **85.4** | 49.3 | 30.5 | 8.4 | 14.5 | **66.6** |
| TESS 2 v0.3 | 62.2 | 84.8 | **53.8** | **54.6** | **10.8** | **36.5** | 59.2 |

### Ablation Study: Effect of Reward Guidance

| Guidance Weight $\eta$ | AlpacaEval |
|-----------------|-----------|
| 0 (No guidance) | 63.1 |
| 0.25 | **66.1** (+3.0) |
| Higher | Degradation (generates gibberish) |

### Key Findings

1. **TESS 2 comprehensively outperforms existing diffusion LMs**: Surpassing DiffuLlama and Flan-XLM-R-D across all downstream tasks.
2. **Performance on QA tasks is close to AR models**: Achieving comparable or superior performance on SQuAD (85.4 vs 86.0) and TriviaQA (53.8 vs 50.4).
3. **A gap persists in reasoning tasks**: Lagging behind significantly on BBH (10.8 vs 43.3) and GSM8k (36.5 vs 52.5).
4. **Diffusion outperforms AR given abundant in-domain data**: After fine-tuning on GSM8k symbolic data, TESS 2 (66.6) outperforms the AR model (51.2).
5. **Diffusion steps can scale inference compute**: Increasing the step count from 50 to 500 consistently improves GSM8k performance.
6. **Reward guidance generalizes across different RMs**: Different reward models consistently bring a 3-4 point improvement.
7. **Generation speed advantage**: 100-step diffusion (77s/batch) is 6x faster than AR 2048 steps (480s/batch).

## Highlights & Insights

1. **Practicality of the AR-to-Diffusion adaptation scheme**: The combination of UL2 + label shifting + bidirectional attention is simple and effective; any open-source AR model can reuse this protocol.
2. **Reward Guidance is a unique advantage of diffusion LMs**: While AR models require RLHF training for alignment, diffusion LMs can achieve training-free alignment via gradient guidance at inference time, rendering it highly modular and flexible.
3. **Insightful observations on base model selection**: The reason Mistral outperforms Llama is likely due to prefix-LM style pre-training, which provides crucial inspiration regarding "which AR models are suitable for diffusion adaptation."
4. **Fine-grained control over inference compute**: The amount of inference computation can be linearly controlled by adjusting the diffusion steps, which is more precise than CoT-based methods.
5. **Counter-intuitive generation speed**: When generating long texts, diffusion models can paradoxically be faster than AR models (since the total number of forward passes is fixed to the number of diffusion steps).

## Limitations & Future Work

1. **Pronounced gap in reasoning tasks**: The substantial gap with AR models on BBH and GSM8k indicates that diffusion LMs still face fundamental limitations on tasks requiring long-chain reasoning.
2. **Adaptation data quality issues**: Continued pre-training on Dolma results in a degradation of mathematical capabilities, indicating that data selection during the adaptation phase is crucial.
3. **Only supporting single-turn dialogue**: Multi-turn training only yields marginal gains, which limits practical application scenarios.
4. **Vulnerability of Reward Guidance to reward hacking**: High guidance weights lead to quality degradation, necessitating more robust guidance mechanisms.
5. **Future Directions**:
    - Use higher-quality adaptation data (e.g., incorporating more math/code data).
    - Explore single-step sampling acceleration (drawing inspiration from consistency models in CV).
    - Develop diffusion training strategies for multi-turn dialogues.
    - Combine the advantages of discrete and continuous diffusion.

## Related Work & Insights

### vs DiffuLlama (Gong et al. 2024)

DiffuLlama adapts Llama using absorbing discrete diffusion, training on 65B tokens but without performing instruction tuning. TESS 2 completely outperforms it with only 45B tokens using continuous simplex diffusion and instruction tuning, demonstrating that (a) base model selection is more critical than the amount of training data, and (b) continuous diffusion combined with cross-entropy is more stable than discrete diffusion.

### vs LLaDA (Nie et al. 2025)

LLaDA pre-trains a large-scale discrete diffusion LM from scratch, which is computationally expensive. TESS 2 chooses to reuse knowledge from existing AR models, making it more cost-effective and practical, embodying the philosophy of "standing on the shoulders of giants."

### vs SSD-LM / SSD-2 (Han et al. 2022, 2023)

SSD-LM, the predecessor of TESS, adopts a semi-autoregressive generation approach. TESS 2 demonstrates that fully non-autoregressive generation is feasible under a 2048-token context, and substantially boosts performance through Mistral adaptation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The AR-to-Diffusion adaptation scheme and reward guidance are both original contributions, achieving a generalist instruction-following diffusion LM for the first time.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Incorporates detailed analyses on base model ablations, adaptation block/steps, diffusion steps, and reward guidance weights, alongside comprehensive coverage of downstream tasks.
- **Writing Quality**: ⭐⭐⭐⭐ — Features a clear structure, adequate technical details, and quantitative support for key findings.
- **Value**: ⭐⭐⭐⭐ — Provides a practical training solution and a clear understanding of capability boundaries for diffusion language models, with the reward guidance approach offering extensive potential for extension.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Cheaper and Better Diffusion Language Model via Task-Specific Training](cheaper_and_better_diffusion_language_model_via_task-specific_training.md)
- [\[ACL 2025\] Pitfalls of Scale: Investigating the Inverse Task of Redefinition in Large Language Models](pitfalls_of_scale_investigating_the_inverse_task_of_redefinition_in_large_langua.md)
- [\[ACL 2025\] Synergizing Unsupervised Episode Detection with LLMs for Large-Scale News Events](synergizing_unsupervised_episode_detection_with_llms_for_large-scale_news_events.md)
- [\[ACL 2025\] Conversational Quality Assessment: A Large-Scale Corpus and Comprehensive Study](conversational_quality_assessment_a_large-scale_corpus_and_comprehensive_study.md)
- [\[ACL 2025\] A Large-Scale Real-World Evaluation of an LLM-Based Virtual Teaching Assistant](a_large-scale_real-world_evaluation_of_llm-based_virtual_teaching_assistant.md)

</div>

<!-- RELATED:END -->

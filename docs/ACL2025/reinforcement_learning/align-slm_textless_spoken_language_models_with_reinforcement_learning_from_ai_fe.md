---
title: >-
  [Paper Note] Align-SLM: Textless Spoken Language Models with Reinforcement Learning from AI Feedback
description: >-
  [ACL 2025][Reinforcement Learning][Spoken Language Model] This paper proposes the Align-SLM framework, which applies preference optimization (DPO + RLAIF) to textless spoken language models (without text injection) for the first time. By utilizing LLMs to automatically evaluate the quality of generated speech continuations to construct preference datasets, combined with curriculum learning, the approach iteratively enhances the semantic understanding of SLMs…
tags:
  - "ACL 2025"
  - "Reinforcement Learning"
  - "Spoken Language Model"
  - "DPO"
  - "RLAIF"
  - "preference optimization"
  - "semantic alignment"
date: 2026-05-08
content_hash: 821515ff2a3c64f1
---

# Align-SLM: Textless Spoken Language Models with Reinforcement Learning from AI Feedback

**Conference**: ACL 2025  
**arXiv**: [2411.01834](https://arxiv.org/abs/2411.01834)  
**Code**: None (based on the open-source TWIST model)  
**Area**: Reinforcement Learning / Spoken Language Models  
**Keywords**: Spoken Language Model, DPO, RLAIF, preference optimization, semantic alignment

## TL;DR

This paper proposes the Align-SLM framework, which applies preference optimization (DPO + RLAIF) to textless spoken language models (without text injection) for the first time. By utilizing LLMs to automatically evaluate the quality of generated speech continuations to construct preference datasets, combined with curriculum learning, the approach iteratively enhances the semantic understanding of SLMs, setting a new SOTA on benchmarks like ZeroSpeech and StoryCloze.

## Background & Motivation

**Background**: Textless NLP trains spoken language models (SLMs) using discrete speech units. It accomplishes end-to-end speech-to-speech modeling via next-speech-token prediction, bypassing the traditional cascade pipelines (ASR $\rightarrow$ LM $\rightarrow$ TTS). Models like TWIST have advanced this direction through textual LLM initialization and massive training datasets.

**Limitations of Prior Work**: Although existing SLMs can generate coherent short phrases, they lag significantly behind textual LLMs in long-range semantics. Generated speech continuations often suffer from repetitive phrases, grammatical errors, and low relevance. Methods like SpeechGPT improve semantics through intermediate text steps, but they still rely on text token guidance and increase decoding latency, making them less suitable for real-time interactive scenarios.

**Key Challenge**: Compared to textual subwords, speech tokens have a finer granularity, lower information density, and greater variability across spectral and temporal dimensions. The simple next-speech-token prediction objective may overlook long-range semantics—models learn to "sound like speech" without necessarily learning to "speak meaningfully."

**Goal**: The output quality of SLMs is often inconsistent—sometimes generating high-quality continuations and other times producing meaningless content. Can SLMs be trained to consistently generate high-quality continuations and avoid poor ones?

**Key Insight**: Drawing inspiration from RLHF/DPO alignment methodologies in the textual LLM domain. Since employing human annotators to listen to speech samples and label preferences is both expensive and time-consuming, this work adopts an RLAIF approach—using an LLM to automatically evaluate the semantic quality of ASR-transcribed text to construct preference data pairs.

**Core Idea**: Multiple speech continuations are sampled from a pretrained SLM. An LLM is used to score them and construct (chosen, rejected) preference pairs. Finally, DPO is used to train a LoRA adapter, encouraging the SLM to generate more semantically meaningful speech.

## Method

### Overall Architecture

Given a speech prompt $x$ (approximately 3 seconds of audio), the SLM generates $N=5$ different speech continuations $y_1, ..., y_N$ via nucleus sampling. After these continuations are synthesized into waveforms using a vocoder, they are transcribed into text using Whisper ASR. The text is passed to an LLM evaluator for scoring, and preference pairs $(y_c, y_r)$ are constructed based on these scores. Finally, the SLM's LoRA adapter is trained on the preference data using the DPO objective, with an option to iteratively improve performance using curriculum learning.

### Key Designs

1. **Automatic Preference Data Selection Strategy**:

    - **Function**: Automatically select preference pairs from multiple generations without human annotation.
    - **Mechanism**: Two AI feedback variants are explored. (1) **Perplexity (PPL)**: Mistral-7B is used to calculate the perplexity of the continuation text conditioned on the prompt. The text with the lowest PPL and auto-BLEU $\le \delta$ is selected as the chosen option, while the one with the highest PPL is selected as rejected. (2) **Mistral Score**: An instruction-tuned Mistral-7B directly scores the continuation text (from 1 to 5), evaluating semantic coherence and relevance. Thresholds $s_c$ and $s_r$ are preset to distinguish chosen and rejected options. Crucially, auto-BLEU (2-gram self-repetition rate) is employed to filter out repetitive and meaningless generations—if auto-BLEU $> \delta=0.1$, the generation is directly labeled as rejected.
    - **Design Motivation**: PPL optimization tends to favor local grammatical correctness rather than overall semantics, whereas Mistral Score provides more comprehensive semantic feedback. Empirical results verify that Mistral Score consistently outperforms PPL across all downstream metrics.

2. **DPO Training for SLMs**:

    - **Function**: Align the generation preferences of the SLM through implicit reward learning.
    - **Mechanism**: The standard DPO loss is formulated as:
    $$\mathcal{L}_{DPO} = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_c|x)}{\pi_{ref}(y_c|x)} - \beta \log \frac{\pi_\theta(y_r|x)}{\pi_{ref}(y_r|x)}\right)\right]$$
    Here, $\pi_{ref}$ is the frozen pretrained model, and $\pi_\theta$ is updated only through a LoRA adapter (rank=32, alpha=8). Preference data is prepared offline to avoid the computational overhead of online sampling + vocoder + ASR + LLM evaluation.
    - **Design Motivation**: Compared to RLHF, DPO does not require training an external reward model, providing simpler and more stable training. LoRA ensures that the model does not drift too far from the pretraining distribution (co-controlled by $\beta$). Offline data preparation makes the entire framework computationally feasible.

3. **Curriculum Learning Iteration**:

    - **Function**: Gradually raise the quality standards of preference data to further elevate the model.
    - **Mechanism**: After the first round of DPO training, the model's capabilities are enhanced. The stronger model is used to resample continuations, and the preference thresholds are raised (e.g., $s_c$: 3 $\rightarrow$ 4, $s_r$: 1 $\rightarrow$ 2) to build harder-to-distinguish preference pairs for a second round of training.
    - **Design Motivation**: This represents a progressive difficulty escalation akin to curriculum learning—allowing the model to first learn to distinguish obvious good/bad cases, and then subtle differences. Experiments show continuous improvement up to 3 iterations.

### Loss & Training

The DPO loss function is formulated as above. Training is performed with a batch size of 512 and a peak learning rate of 1e-6 linearly decaying over 100K steps (with 500 warmup steps). For the MLS extended dataset, training is extended to 300K steps. The second round of curriculum learning has no warmup. The models are trained on 64 A100 GPUs. Model selection is based on the highest reward accuracy on the dev-clean set.

## Key Experimental Results

### Main Results: Align-SLM 7B vs. Baselines

| Method | sWUGGY↑ | sBLIMP↑ | S-StoryCloze↑ | T-StoryCloze↑ | GPT4-o↑ |
|------|---------|---------|---------------|---------------|---------|
| GSLM | 64.8 | 54.2 | 53.3 | 66.6 | - |
| TWIST 7B | 73.5 | 58.8 | 55.1 | 75.4 | 2.70 |
| Moshi 7B | 72.6 | 58.8 | 60.8 | 83.0 | - |
| SPIRIT-LM 7B | 69.0 | 58.3 | 61.0 | 82.9 | - |
| **Align-SLM-mls+CL 7B** | **77.9** | **62.3** | **61.1** | **86.8** | **3.50** |
| Human Level | - | - | 79.2 | 90.2 | - |

### Ablation Study: Comparison of Feedback Modalities (1.3B)

| Configuration | Mistral Score↑ | T-StoryCloze↑ | GPT4-o↑ |
|------|---------------|---------------|---------|
| Pretrained TWIST | 1.66 | 69.7 | 1.82 |
| Continued Fine-Tuning (NTP) | 1.70 (+0.04) | 70.7 (+1.0) | 1.83 (+0.01) |
| Align-SLM w/PPL | 1.88 (+0.22) | 67.7 (-2.0) | 1.85 (+0.03) |
| **Align-SLM w/Mistral** | **2.17 (+0.51)** | **74.2 (+4.5)** | **2.06 (+0.24)** |

### Key Findings

- **Mistral Score consistently outperforms PPL as a preference signal**: Although PPL feedback improves syntactic grammar (sBLIMP +2.1), it degrades semantic coherence (T-StoryCloze) by 2.0 points. PPL focuses excessively on local fluency while neglecting global semantics.
- **Curriculum learning yields continuous improvements**: For the 7B model, T-StoryCloze scores increase from 83.8 (1st round) to 85.6 (2nd round), and GPT-4o evaluation increases from 3.50 to 3.56. Experiments extending to 3 iterations show sustained enhancement.
- **Data scale is beneficial but exhibits diminishing marginal returns**: Incorporating approximately $3\times$ MLS data increases the 7B model's T-StoryCloze score from 85.6 to 86.8, though the GPT-4o rating sees limited gain. This suggests that the 63K preference data pairs from LibriSpeech are already relatively sufficient for the 7B model.
- **Human evaluation validates objective metrics**: In terms of MMOS scores, Align-SLM ($3.73 \pm 0.06$) not only surpasses the pre-trained counterpart ($3.48 \pm 0.07$) but also outperforms the resynthesized versions of genuine continuations ($3.50 \pm 0.07$).
- **T-StoryCloze reaches 86.8, approaching the human level of 90.2**, marking a milestone first achieved by a textless spoken language model.

## Highlights & Insights

- **First systematic work applying preference optimization to textless SLMs**. The pipeline of ASR + LLM evaluation bypasses the prohibitive expenses of auditing listening preferences, making RLAIF feasible for the speech domain. This paradigm can be directly extended to other speech generation tasks (e.g., spoken dialogue, translation).
- **auto-BLEU filtering is a crucial detail**: The PPL of repetitive and meaningless speech generated by SLMs can be artificially low (since repetitive patterns are highly predictable). Unfiltered preference data would mislead the model. The threshold of $\delta=0.1$ is grounded on empirical analysis of auto-BLEU distributions between real and generated continuations.
- **Combining curriculum learning with DPO** is both natural and effective: resampling generations as the model improves $\rightarrow$ elevating selection criteria $\rightarrow$ iterating. This forms a self-improvement training strategy that operates without the need for a stronger external teacher model.

## Limitations & Future Work

- **Dependence on the cascade of ASR and LLMs**: Preference data quality is inherently constrained by ASR transcription fidelity and the judgment capabilities of the LLM. For unwritten languages (a primary advantage of textless SLMs), this pipeline would need to incorporate elements like speech translation.
- **Exclusive focus on semantics**: The framework does not address speech-specific quality dimensions such as style, prosody, and emotion, which are equally vital for natural spoken dialogue.
- **Domain limitation of training data**: Training is confined to the audiobook domain (LibriSpeech/MLS), and this lack of domain diversity may restrict the model's generalization capability.
- **Relatively small model scale**: With models limited to 1.3B/7B parameters compared to textual LLMs spanning tens of billions, the performance ceiling of SLMs in semantic comprehension remains restricted.

## Related Work & Insights

- **vs. SPIRIT-LM (Nguyen et al., 2024)**: SPIRIT-LM improves semantics via interleaved speech-text training, requiring paired speech-text data. In contrast, Align-SLM injects no textual tokens and outperforms SPIRIT-LM on key metrics (T-StoryCloze: 86.8 vs. 82.9) solely through preference optimization.
- **vs. Moshi (Défossez et al., 2024)**: Moshi uses text-guided speech generation to achieve real-time dialogue and is categorized as a multimodal model. As a textless model, Align-SLM outperforms Moshi on both sWUGGY (77.9 vs. 72.6) and T-StoryCloze (86.8 vs. 83.0).
- **vs. SpeechAlign (Zhang et al., 2024)**: SpeechAlign applies preference optimization to speech synthesis quality in TTS rather than semantic understanding in SLMs, representing a distinct task objective.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First systematic application of RLAIF+DPO to SLMs; although individual components (DPO, LoRA, curriculum learning) are not novel, their combined application in this context is highly innovative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive ablations covering various benchmarks, model scales, feedback modalities, and data volumes, further validated by human evaluations.
- **Writing Quality**: ⭐⭐⭐⭐ The framework description is clear, experimental analyses are rigorous, and the appendix provides rich technical details.
- **Value**: ⭐⭐⭐⭐⭐ Establishes a preference optimization paradigm in the SLM field with convincing SOTA results. The framework is highly extensible to broader speech tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] MAPoRL: Multi-Agent Post-Co-Training for Collaborative Large Language Models with Reinforcement Learning](maporl_multi-agent_post-co-training_for_collaborative_large_language_models_with.md)
- [\[NeurIPS 2025\] Behavior Injection: Preparing Language Models for Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/behavior_injection_preparing_language_models_for_reinforcement_learning.md)
- [\[NeurIPS 2025\] Checklists Are Better Than Reward Models For Aligning Language Models](../../NeurIPS2025/reinforcement_learning/checklists_are_better_than_reward_models_for_aligning_langua.md)
- [\[ICML 2025\] Optimizing Language Models for Inference Time Objectives using Reinforcement Learning](../../ICML2025/reinforcement_learning/optimizing_language_models_for_inference_time_objectives_using_reinforcement_lea.md)
- [\[ECCV 2024\] Octopus: Embodied Vision-Language Programmer from Environmental Feedback](../../ECCV2024/reinforcement_learning/octopus_embodied_vision-language_programmer_from_environmental_feedback.md)

</div>

<!-- RELATED:END -->

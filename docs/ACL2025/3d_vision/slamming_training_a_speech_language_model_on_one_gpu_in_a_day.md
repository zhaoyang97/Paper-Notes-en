---
title: >-
  [Paper Note] Slamming: Training a Speech Language Model on One GPU in a Day
description: >-
  [ACL 2025][3D Vision][Speech Language Model] This paper proposes the Slam training recipe, which systematically optimizes model initialization, architectural choices, synthetic data, and preference alignment to train a speech language model on a single A5000 GPU within 24 hours, achieving performance comparable to large-scale SLMs.
tags:
  - "ACL 2025"
  - "3D Vision"
  - "Speech Language Model"
  - "Efficient Training"
  - "Synthetic Data"
  - "DPO"
  - "Low-resource Training"
date: 2026-05-08
content_hash: d49a26a80ab7abd8
---

# Slamming: Training a Speech Language Model on One GPU in a Day

**Conference**: ACL 2025  
**arXiv**: [2502.15814](https://arxiv.org/abs/2502.15814)  
**Code**: [https://pages.cs.huji.ac.il/adiyoss-lab/slamming](https://pages.cs.huji.ac.il/adiyoss-lab/slamming)  
**Area**: Audio/Speech  
**Keywords**: Speech Language Model, Efficient Training, Synthetic Data, DPO, Low-resource Training

## TL;DR
This paper proposes the Slam training recipe, which systematically optimizes model initialization, architectural choices, synthetic data, and preference alignment to train a speech language model on a single A5000 GPU within 24 hours, achieving performance comparable to large-scale SLMs.

## Background & Motivation

**Background**: Speech Language Models (SLMs) have made remarkable progress in recent years. However, training high-quality SLMs typically requires substantial computational resources—for instance, Moshi utilized 7 million hours of speech data, SpiritLM used 560k hours, and TWIST-7B required 32×V100 GPUs for training.

**Limitations of Prior Work**: The high computational requirements restrict SLM research to well-funded large laboratories, making it difficult for typical academic labs to participate in foundational SLM research (such as novel tokenization or efficient acoustic modeling).

**Key Challenge**: Scaling law studies of SLMs (Cuervo & Marxer 2024) even present a pessimistic prediction—training a high-quality SLM requires approximately three times more data than a text LM. However, whether this prediction can be broken with superior training strategies remains an open question.

**Goal**: Can a high-quality SLM be trained on a single academic-grade GPU within 24 hours? How can each component of the training pipeline be systematically optimized?

**Key Insight**: Inspired by "Cramming" (training BERT on a single GPU in one day) in the text NLP domain, this work systematically ablates each component of the SLM training pipeline.

**Core Idea**: Achieve low-resource, high-performance SLM training through a combined recipe that carefully selects model families/initialization (Qwen2.5 + TWIST init), utilizes synthetic data (TinyStories TTS), applies DPO preference alignment, and tunes hyperparameters.

## Method

### Overall Architecture
Slam is an end-to-end SLM training recipe rather than a singular architectural innovation. The input is raw audio, which is first processed by HuBERT to extract 25Hz semantic tokens (discretized via k-means), and then modeled using a decoder-only Transformer for next-token prediction. The overall optimization covers five dimensions: model selection and initialization, optimizers and schedulers, data selection and synthesis, text interleaving, and DPO preference alignment.

### Key Designs

1. **Model Family and TWIST Initialization**:

    - **Function**: Compare various pretrained text LMs (OPT, Pythia, SmolLM2, MobileLLM, Qwen2.5) as initializations for the SLM.
    - **Mechanism**: Replace the token embedding layer of the pretrained text LM with speech token embeddings while retaining all other parameters. TWIST initialization leverages semantic knowledge learned by the text LM to accelerate SLM convergence.
    - **Design Motivation**: Different model families benefit differently from TWIST initialization. Qwen2.5-0.5B (actually 358M parameters due to vocabulary reduction) outperforms other models by a large margin under the same computational budget, performing best even without TWIST initialization. This contrasts sharply with the optimal model size (66M) predicted by scaling laws, demonstrating that selecting a stronger pretrained model is more effective than scaling data.

2. **Optimizer and Learning Rate Scheduling**:

    - **Function**: Find the optimal optimizer and scheduler combination under a fixed computational budget.
    - **Mechanism**: Compare three optimizers (AdamW, AdaLomo, AdEMAMeix) and two schedulers (InverseSqrt and Cosine). The optimal combination is found to be AdamW + Cosine decay.
    - **Design Motivation**: The original TWIST uses an InverseSqrt scheduler, but Cosine decay significantly improves the convergence of AdamW. In addition, removing dropout (which reduces effective gradient updates without affecting wall-clock time), lowering the RoPE $\theta$ base from the default to 10,000 (as the context length is much shorter than original LLMs), and increasing the context length from 512 to 1024 all improve performance.

3. **Synthetic Data (sTinyStories)**:

    - **Function**: Synthesize the TinyStories text dataset into speech using a single-speaker TTS and incorporate it into training.
    - **Mechanism**: TinyStories has been shown to enhance the semantic capability of text LMs; its synthesized speech version (sTinyStories) significantly boosts the semantic comprehension and generation capabilities of SLMs without requiring large-scale real-world speech data.
    - **Design Motivation**: Under constrained compute, data diversity (multiple accents and speaking styles) actually hurts performance because the model lacks sufficient capacity to model complex acoustics. In contrast, semantically rich synthetic data is highly beneficial—GenPPL drops from 145.4 to 88.3 (Qwen-0.5B).

4. **DPO Preference Optimization (Synthesized SWAG Data)**:

    - **Function**: Perform a small amount of preference optimization using DPO after pretraining (just 30 minutes).
    - **Mechanism**: Synthesize the SWAG dataset (containing correct/incorrect ending sentence pairs) into 47k preference pairs using Kokoro TTS (4 speakers, British/American accents). Train with off-policy DPO ($\beta=0.1$) to specifically enhance semantic discrimination capability.
    - **Design Motivation**: Just 30 minutes of DPO significantly improves all metrics (tSC from 78.01 to 82.04, GenPPL from 88.3 to 62.8). Allocating more time to DPO yields diminishing returns. To mitigate the repetitive generation issue caused by DPO, a repetition penalty factor of 1.1 is applied.

### Training Strategy
- **Text Interleaving**: Tested speech-text interleaving (Whisper aligned transcriptions + RedPajama text), but found it unhelpful under the current computational budget. This is because interleaving increases the vocabulary and parameter sizes, reducing training steps (11k vs. 18k), and only 40% of the tokens are speech.
- **Data Diversity vs. Synthetic Data**: Diverse data (VoxPopuli, Tedlium, etc.) degrades performance, whereas synthetic data helps significantly. This indicates that semantic quality is more critical than acoustic diversity in low-resource regimes.

### Loss & Training
- Pretraining Phase: Standard next-token prediction (negative log-likelihood)
- DPO Phase: Direct Preference Optimization loss ($\beta=0.1$)
- Peak learning rate $1\times10^{-3}$, warmup 1%, gradient clipping norm = 0.5, bfloat16 + FlashAttention2 + data packing

## Key Experimental Results

### Main Results

| Model | Compute Resources | Parameters | sBLIMP↑ | sSC↑ | tSC↑ | GenPPL↓ | GPTScore↑ |
|------|---------|--------|---------|------|------|---------|-----------|
| TWIST-1.3B | 32×V100 | 1B | 57.00 | 52.4 | 70.6 | 131.8 | 1.82 |
| TWIST-7B | 32×V100 | 7B | 59.00 | 55.3 | 74.1 | 93.7 | 2.71 |
| TWIST-13B | 32×V100 | 13B | 59.20 | 55.4 | 76.4 | - | - |
| SpiritLM | 64×A100 | 7B | 58.0 | 54.8 | 72.9 | - | - |
| **Slam (ours)** | **1×A5000** | **358M** | **58.86** | **58.04** | **82.04** | **62.8** | **2.09** |
| Slam (scaled) | 2×A100 | 358M | 61.11 | 61.30 | 84.18 | 46.6 | 2.69 |
| Slam (large) | 2×A100 | 1.3B | 61.43 | 61.52 | 85.30 | 41.2 | 2.79 |

### Ablation Study

| Configuration | tSC↑ | GenPPL↓ | Description |
|------|------|---------|------|
| Slam full (Qwen-0.5B) | 82.04 | 62.8 | Full recipe |
| Slam w/o DPO | 78.01 | 88.3 | Removing DPO drops tSC by 4 points |
| w/o Synthetic Data (Qwen) | 71.14 | 145.4 | GenPPL degrades by 65% without synthetic data |
| + Diverse Data (Qwen) | 70.66 | 161.8 | Data diversity is counterproductive |
| OPT-125M + Synthetic | 75.18 | 96.8 | Model family selection has a massive impact |
| Original TWIST recipe | 68.80 | 259.2 | Original recipe achieves significantly worse performance |
| TWIST + sTinyStories | 72.40 | 159.0 | Merely adding synthetic data is insufficient |

### Key Findings
- **DPO is the most cost-effective component**: Just 30 minutes (2% of the 24-hour budget) yields an improvement of +4 in tSC and -25.5 in GenPPL. Allocating more time provides no scaling benefit.
- **Model Selection > Data Volume**: Qwen2.5 outperforms models like OPT/Pythia by a wide margin, even though its larger size leads to fewer training tokens. This contradicts traditional scaling laws (which recommend training smaller models on more data).
- **Synthetic data is crucial for semantics**: Despite having only a single speaker, sTinyStories significantly enhances semantic metrics. Conversely, multi-source real-world data (acoustically diverse but semantically monotonous) offers no benefit.
- **The Slam recipe scales well**: Scaling from 1×A5000 to 2×A100 leads to steady improvements across all metrics, showing that the recipe is not overfitted to low-compute constraints.

## Highlights & Insights
- **"Selecting the right model is more important than scaling data"**: This finding challenges the pessimistic predictions of SLM scaling laws. TWIST initialization magnifies the differences in quality among model families, making the architectural advantages of Qwen2.5 particularly prominent in low-resource scenarios.
- **Extreme efficiency of DPO with synthetic preference data**: Synthesizing only 47k pairs (from the SWAG dataset) via TTS and training with DPO for just 30 minutes significantly improves semantic capability. This approach is transferable to low-resource alignment scenarios for any generative model.
- **"Acoustic diversity is detrimental under low-resource constraints"**: This is a counter-intuitive yet valuable finding. When computation is limited, models should focus on learning semantics rather than acoustic variations, which yields clear guidance for data strategies.

## Limitations & Future Work
- Only HuBERT semantic tokens were used; the layout does not explore the impact of newer tokenizers like Mimi or SylBoost on the recipe.
- Acoustic and prosodic dimensions (e.g., the SALMon benchmark) were not evaluated; low-resource SLMs might perform worse on these aspects.
- Text interleaving yielded no significant benefits under the current budget, though it may help under larger budgets; the minimum viable budget remains unexplored.
- DPO experiences repetition issues during generation, which are only mitigated via a repetition penalty rather than being fundamentally resolved.
- Synthetic data relied strictly on a single TTS system (Kokoro); the effects of employing multiple TTS engines or various synthesis strategies were not evaluated.

## Related Work & Insights
- **vs. TWIST**: TWIST pioneered the direct initialization of SLMs from pretrained text LMs. This work systemizes the training strategy on top of it, achieving performance comparable to TWIST-7B with only 1/160 of the compute.
- **vs. Cramming**: Cramming explores low-resource training for masked text LMs. This work is the first to apply a similar philosophy to generative SLMs, finding that SLM-specific optimizations—such as synthetic data and DPO—are highly crucial.
- **vs. AlignSLM**: AlignSLM also applies DPO to SLMs, but requires 64×A100 GPUs and 158B text tokens. Slam achieves a comparable effect with only 47k synthetic preference pairs and 30 minutes of DPO.

## Rating
- Novelty: ⭐⭐⭐⭐ It is a systematic engineering optimization rather than a methodological innovation, but the research problem of a "low-resource SLM training recipe" is highly valuable in itself.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ The ablation study is highly comprehensive, covering all dimensions of models/optimizers/data/DPO, ensuring strong reproducibility.
- Writing Quality: ⭐⭐⭐⭐⭐ The paper is well-structured, with clear conclusions and takeaways for each experiment.
- Value: ⭐⭐⭐⭐ Provides a ready-to-use SLM training recipe for academic laboratories, open-sourcing all code, models, and data.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] The Sharpness Disparity Principle in Transformers for Accelerating Language Model Pre-Training](../../ICML2025/3d_vision/the_sharpness_disparity_principle_in_transformers_for_accelerating_language_mode.md)
- [\[CVPR 2025\] Matrix3D: Large Photogrammetry Model All-in-One](../../CVPR2025/3d_vision/matrix3d_large_photogrammetry_model_all-in-one.md)
- [\[ICCV 2025\] RoboTron-Mani: All-in-One Multimodal Large Model for Robotic Manipulation](../../ICCV2025/3d_vision/robotron-mani_all-in-one_multimodal_large_model_for_robotic_manipulation.md)
- [\[ICCV 2025\] MemoryTalker: Personalized Speech-Driven 3D Facial Animation via Audio-Guided Stylization](../../ICCV2025/3d_vision/memorytalker_personalized_speech-driven_3d_facial_animation_via_audio-guided_sty.md)
- [\[NeurIPS 2025\] URDF-Anything: Constructing Articulated Objects with 3D Multimodal Language Model](../../NeurIPS2025/3d_vision/urdf-anything_constructing_articulated_objects_with_3d_multimodal_language_model.md)

</div>

<!-- RELATED:END -->

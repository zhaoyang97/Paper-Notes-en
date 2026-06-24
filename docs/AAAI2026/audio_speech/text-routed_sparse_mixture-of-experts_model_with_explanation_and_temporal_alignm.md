---
title: >-
  [Paper Note] A Text-Routed Sparse Mixture-of-Experts Model with Explanation and Temporal Alignment for Multi-Modal Sentiment Analysis
description: >-
  [AAAI 2026][Audio & Speech][Multimodal Sentiment Analysis] This paper proposes the TEXT model, which leverages MLLMs to generate natural language explanations for audio and video to enhance modal representations. It designs a lightweight temporal alignment module combining the merits of Mamba and temporal cross-attention, and employs text-routed sparse mixture-of-experts for cross-modal fusion, comprehensively outperforming SOTAs and large models like GPT-4o on four MSA datas…
tags:
  - "AAAI 2026"
  - "Audio & Speech"
  - "Multimodal Sentiment Analysis"
  - "Sparse Mixture-of-Experts"
  - "Temporal Alignment"
  - "MLLM Explanation Enhancement"
  - "Gated Fusion"
date: 2026-05-08
content_hash: ebd0c029cad771aa
---

# A Text-Routed Sparse Mixture-of-Experts Model with Explanation and Temporal Alignment for Multi-Modal Sentiment Analysis

**Conference**: AAAI 2026  
**arXiv**: [2512.22741](https://arxiv.org/abs/2512.22741)  
**Authors**: Dongning Rao, Yunbiao Zeng, Zhihua Jiang, Jujian Lv  
**Code**: [fip-lab/TEXT](https://github.com/fip-lab/TEXT)  
**Area**: Audio & Speech  
**Keywords**: Multimodal Sentiment Analysis, Sparse Mixture-of-Experts, Temporal Alignment, MLLM Explanation Enhancement, Gated Fusion

## TL;DR

This paper proposes the TEXT model, which leverages MLLMs to generate natural language explanations for audio and video to enhance modal representations. It designs a lightweight temporal alignment module combining the merits of Mamba and temporal cross-attention, and employs text-routed sparse mixture-of-experts for cross-modal fusion, comprehensively outperforming SOTAs and large models like GPT-4o on four MSA datasets.

## Background & Motivation

### Problem Definition

The task of Multimodal Sentiment Analysis (MSA) is to predict the speaker's sentiment polarity (positive/negative/neutral) and sentiment intensity score (continuous values) from short videos, utilizing three modalities simultaneously: text (subtitles), audio (intonation/prosody), and vision (facial expressions). This task is widely applied in scenarios such as healthcare, human-computer interaction, and fraud detection. The core challenge lies in the fact that different modalities contribute very differently to sentiment, and can even contradict each other—for example, text expressing success/positivity but with a negative tone, or facial expressions conflicting with the spoken content.

### Limitations of Prior Work

Existing MSA methods can be categorized into two major classes: representation learning-centric (e.g., ALMT, KuDA) and multimodal fusion-centric (e.g., DEVA). The authors identify three key gaps:

**Unexploited Explanatory Power of MLLMs**: The power of text in the LLM era has not yet been fully unlocked. MLLMs can generate semantic explanations for audio and video to bridge the semantic gap of non-text modalities, but no prior work has introduced them into the feature alignment process of MSA.

**Mismatched Temporal Alignment Schemes for MSA**: Mamba (linear SSM) is designed for long videos, and Temporal Cross-Attention (TCA) is a general-purpose module; neither is specifically optimized for dynamic sentiment transitions in short MSA videos.

**Fusion Strategies Ignoring Modality Dominance**: Research shows that text is almost always the dominant modality, but existing methods lack mechanisms to leverage this prior; two powerful technologies, SMoE and gated fusion, are under-utilized in the MSA domain.

### Core Motivation

The paper begins with a specific case from MOSI: in this sample, only the text can correctly determine polarity, while both audio and video mislead the model. The prior state-of-the-art model ALMT has a prediction bias of 0.320, whereas Qwen2.5-vl exhibits a bias as high as 1.100. This indicates that **alignment is the key bridge between representation learning and fusion**. If one can use MLLMs to generate explanations for audio and video, and then align these explanations with the raw features, error correction can be effectively achieved. Meanwhile, utilizing the dominance of text to route expert activations can achieve more precise cross-modal fusion. This constitutes the two core design concepts of TEXT: **explanation-driven alignment** + **text-routed fusion**.

## Method

### Overall Architecture

TEXT consists of six modules, structured from bottom to top as follows:

- **Modules ④⑤⑥ (Parallel)**: Three unimodal feature extraction modules. Text uses BERT to encode subtitles and explanations, audio uses Librosa to extract features, and video uses OpenFace to extract facial action units. Inside both the audio and video modules, an **Explanation Alignment Block** is embedded.
- **Module ③**: Temporal alignment module, modeling temporal dependency between the aligned audio and video representations.
- **Module ②**: Text-routed Sparse Mixture-of-Experts (SMoE) module, conducting cross-modal interactions using text features as routing keys.
- **Module ①**: Gated Fusion (GF) + MLP classifier, executing final sentiment prediction.

### Key Designs

#### Key Design 1: Two-Stage Explanation Generation

TEXT generates three types of explanations for each sample (audio explanation $e_a$, video explanation $e_v$, and overall commentary $c$), employing a two-stage process:

**Stage 1**: Using VideoLLaMA 3 fine-tuned on the EMER-fine sentiment dataset as the multimodal understander, the raw video is inputted to generate raw explanations for audio, video, and the overall commentary using prompt guidance.

**Stage 2**: Using Qwen 3 as the reasoning checker, raw explanations are verified and refined via reasoning prompts to output high-quality fine explanations.

The elegance of this division of labor lies in: VideoLLaMA 3 is skilled at multimodal perception but might produce biased descriptions, while the reasoning capability of Qwen 3 can further calibrate them, with both complementing each other to reduce cumulative errors.

#### Key Design 2: Explanation Alignment Block

The objective of alignment is to pull audio/video features closer to the explanatory text in the semantic space. Specifically, for feature $F$ and explanation encoding $E$, alignment is achieved via cross-attention:

$$ca(F, E) = \text{softmax}((W_Q E)(W_K F)^T) W_V F$$

Where $Q$ comes from the explanations, and $K/V$ from the raw modal features, allowing the explanations to dominate the attention allocation. All unimodal encodings are unified into 50 tokens + 1 learnable aggregation token, yielding a final 51-dimensional embedding. The aligned representations are denoted as $E_t$ (text), $E_a$ (audio), and $E_v$ (video).

#### Key Design 3: Temporal Alignment Block

This is the most critical technical innovation of the paper. The authors design a lightweight temporal alignment block that **does not rely on CA or SSM**, which is more concise than both Mamba and TCA but integrates the advantages of both. The core computation is as follows:

$$\text{left} = E_a \oplus L(\text{Conv1d}(LN(E_a)) \otimes \sigma(LN(E_v)))$$
$$\text{right} = E_v \oplus L(\text{Conv1d}(LN(E_v)) \otimes \sigma(LN(E_a)))$$
$$E_{av} = \text{concat}(\text{left}, \text{right})$$

Key design points: (1) Conv1d is responsible for capturing local temporal patterns (resembling sequence modeling in Mamba); (2) Element-wise multiplication gated by SiLU enables selective interactions between modalities (resembling attention weighting); (3) Residual connections preserve original information; (4) The symmetric structure makes the information flow of audio $\rightarrow$ video and video $\rightarrow$ audio equally important. This design avoids the complex recursion of SSM and the quadratic complexity of CA, while preserving temporal modeling capabilities.

#### Key Design 4: Text-Routed SMoE

Leveraging the dominant status of text in MSA, TEXT uses the text feature $E_t$ as a routing key to determine which experts are activated to process the temporally aligned audio-video embedding $E_{av}$. This is formalized as $\text{SMoE}(E_t, E_{av})$. Intuitively, sentimental keywords in the text (such as "disappointing" or "excellent") activate experts corresponding to the respective emotional topics, making the expert network topic-sensitive.

### Loss & Training

As a regression problem, the basic optimization objective of TEXT is the MSE loss, which is the mean squared error between the predicted sentiment intensity score $\hat{y}$ and the ground truth. The output of the gated fusion classifier is:

$$\hat{y} = L(\sigma(\text{SMoE}(E_t, E_{av})))$$

Where $\sigma$ is the Sigmoid gating, and $L$ is a linear layer.

## Key Experimental Results

### Main Results

#### Table 1: Main Comparison of Models on MOSI and MOSEI

| Model | MOSI Acc-2 | MOSI MAE↓ | MOSI Corr | MOSEI Acc-2 | MOSEI MAE↓ | MOSEI Corr |
|------|-----------|----------|----------|------------|----------|----------|
| ALMT | 83.10/85.23 | 0.716 | 0.773 | 82.39/85.87 | 0.542 | 0.767 |
| KuDA | 84.40/86.43 | 0.705 | 0.795 | 83.26/86.46 | 0.529 | 0.776 |
| DEVA | 84.40/86.29 | 0.730 | 0.787 | 83.26/86.13 | 0.541 | 0.769 |
| GPT-4o | 85.71/86.74 | 0.682 | 0.823 | 84.77/86.08 | 0.637 | 0.744 |
| Qwen2.5-vl | 83.09/83.38 | 1.129 | 0.677 | 84.14/84.59 | 1.007 | 0.587 |
| **TEXT** | **86.44/88.72** | **0.666** | **0.829** | **85.02/86.57** | **0.528** | **0.786** |

TEXT achieves 88.72% Acc-2 on MOSI (outperforming GPT-4o by approximately 2%), with its MAE dropping to 0.666; its MAE on MOSEI falls to 0.528, outperforming all compared models. On CH-SIMS, the MAE drops from the runner-up of 0.408 (KuDA) to 0.353, representing a 13.5% reduction.

### Ablation Study

#### Table 2: Ablation Study on MOSEI

| Setting | Acc-2 | Acc-7 | MAE↓ | Corr |
|------|-------|-------|------|------|
| TEXT (Full) | 85.02/86.57 | 52.29 | 0.528 | 0.786 |
| W/o Explanation | 83.60/86.02 | 50.35 | 0.569 | 0.776 |
| Text Only (W/ Explanation) | 83.49/86.43 | 52.84 | 0.535 | 0.771 |
| EA → Linear | 84.25/86.57 | 48.21 | 0.577 | 0.762 |
| TA → Concat | 83.77/85.42 | 48.40 | 0.580 | 0.749 |
| TA → Mamba | 84.80/86.41 | 50.65 | 0.562 | 0.780 |
| TA → TCA | 83.41/86.43 | 51.38 | 0.565 | 0.781 |
| SMoE → Transformer | 83.73/85.33 | 50.29 | 0.573 | 0.769 |
| W/o Gated Fusion | 84.40/86.35 | 49.07 | 0.571 | 0.780 |

Ablation Conclusions: After replacing temporal alignment with concatenation, the MAE increases from 0.528 to 0.580 (the largest degradation), proving it to be the key factor for MAE improvement. Removing explanations leads to a comprehensive decline of about 2%. The contribution of SMoE is comparable to that of explanations.

## Highlights & Insights

1. **Using MLLMs for Data Augmentation over End-to-End Inference**: Unlike directly using GPT-4o for MSA, TEXT cleverly utilizes MLLMs to generate explanation text as auxiliary signals, which are then encoded by BERT to participate in feature alignment. This both exploits the semantic understanding capabilities of MLLMs and avoids their instability on regression tasks (e.g., Qwen2.5-vl's MAE reaches as high as 1.129).
2. **Minimalist Design of the Temporal Alignment Module**: Using only Conv1d + linear layers + gating, without introducing attention or SSM, yet outperforming Mamba and TCA in the ablation study. This indicates that for short-video MSA, simple local temporal convolutions coupled with cross-modal gating are sufficient.
3. **Empirical Findings on Text Dominance**: Ablation studies show that using text alone (with explanations) yields a slightly higher Acc-7 than the full model (52.84% vs 52.29%), indicating that the contribution of audio and video is primarily in regression precision (MAE) rather than classification accuracy.
4. **Highly Convincing Qualitative Cases**: In a specific sample, the prediction bias of TEXT is only 0.010 (ground-truth 1.400 vs prediction 1.390), whereas GPT-4o exhibits a bias of 0.600 and Qwen2.5-vl exhibits a bias of 1.100. Removing explanations causes the audio bias to soar from 0.380 to 1.440, clearly validating the value of explanation alignment.

## Limitations & Future Work

1. **Reliance on a Cascade of Multiple MLLMs**: Explanation generation requires a two-stage process using VideoLLaMA 3 + Qwen 3, which introduces a risk of accumulated errors and high inference overhead, as acknowledged by the authors as a major limitation.
2. **Limited Language Coverage (Only Chinese and English)**: The quality of explanations generated by MLLMs depends heavily on their linguistic capabilities. The paper only validates on Chinese and English datasets, and the generalization to other languages remains unknown.
3. **Data Contamination Risk in MLLMs**: The paper mentions that MLLMs might have memorized parts of the datasets (e.g., GPT-4o performs exceptionally well on English datasets but drops significantly on Chinese datasets), impacting the fairness of baseline comparisons.
4. **Absence of Computational Cost for Explanation Generation**: The paper does not report the time and computational resource requirements during the MLLM explanation generation phase, leaving the actual deployment cost unclear.

## Related Work & Insights

- **ALMT** (Zhang et al., 2023): Learns irrelevant/conflict-suppressed representations and uses Transformers to unify modal representations. The explanation alignment in TEXT can be viewed as an enhanced version of ALMT.
- **KuDA** (Feng et al., 2024): Proposes a dominant modality augmentation strategy. TEXT inherits the idea of text dominance but leverages it in a more refined manner via SMoE.
- **DEVA** (Wu et al., 2025): Text-guided progressive fusion + sentiment description generation. It aligns with the explanation-enhancement concept of TEXT but differs in implementation.
- **Mamba** (Gu & Dao, 2024): Linear SSM model. The temporal alignment of TEXT can be viewed as a simplified version of Mamba's convolutional path.
- **Insights**: The approach of using MLLMs as "semantic bridges" rather than direct predictors can be extended to other multimodal regression tasks (e.g., emotion intensity estimation, pain assessment). The success of the lightweight temporal module also suggests that complex sequence modeling is unnecessary in short-sequence scenarios.

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐ |
| Overall Rating | ⭐⭐⭐⭐ |

The combination of explanation alignment and text-routed SMoE is novel and effective, with comprehensive and convincing ablation experiments. Demerits lie in: minor formula rendering issues in the writing, undiscussed computational costs of cascading MLLMs, and the performance of the unimodal text model being close to the full model on several metrics, suggesting limited room for multimodal fusion gains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] PSA-MF: Personality-Sentiment Aligned Multi-Level Fusion for Multimodal Sentiment Analysis](psa-mf_personality-sentiment_aligned_multi-level_fusion_for_multimodal_sentiment.md)
- [\[AAAI 2026\] PaSE: Prototype-aligned Calibration and Shapley-based Equilibrium for Multimodal Sentiment Analysis](pase_prototype-aligned_calibration_and_shapley-based_equilibrium_for_multimodal_.md)
- [\[AAAI 2026\] Improving Multimodal Sentiment Analysis via Modality Optimization and Dynamic Primary Modality Selection](improving_multimodal_sentiment_analysis_via_modality_optimization_and_dynamic_pr.md)
- [\[CVPR 2026\] Tri-Subspaces Disentanglement for Multimodal Sentiment Analysis](../../CVPR2026/audio_speech/tri-subspaces_disentanglement_for_multimodal_sentiment_analysis.md)
- [\[AAAI 2026\] Hearing More with Less: Multi-Modal Retrieval-and-Selection Augmented Conversational LLM-Based ASR](hearing_more_with_less_multi-modal_retrieval-and-selection_augmented_conversatio.md)

</div>

<!-- RELATED:END -->

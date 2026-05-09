---
title: >-
  [Paper Note] TEXT: Text-Routed Sparse Mixture of Experts for Multimodal Sentiment Analysis with Explanation Enhancement and Temporal Alignment
description: >-
  [AAAI 2026][Audio & Speech][Multimodal Sentiment Analysis] This paper proposes TEXT, a model that leverages MLLMs to generate natural language explanations for audio and video modalities to enhance modal representations, designs a lightweight temporal alignment module combining the strengths of Mamba and temporal cross-attention, and employs text-routed sparse mixture of experts for cross-modal fusion. TEXT comprehensively outperforms prior SOTA methods and large models such as GPT-4o across four MSA benchmarks.
tags:
  - AAAI 2026
  - "Audio & Speech"
  - Multimodal Sentiment Analysis
  - Sparse Mixture of Experts
  - Temporal Alignment
  - MLLM Explanation Enhancement
  - Gated Fusion
date: 2026-05-08
content_hash: 3f948588ff5e2cb7
---

# TEXT: Text-Routed Sparse Mixture of Experts for Multimodal Sentiment Analysis with Explanation Enhancement and Temporal Alignment

**Conference**: AAAI 2026
**arXiv**: [2512.22741](https://arxiv.org/abs/2512.22741)
**Authors**: Dongning Rao, Yunbiao Zeng, Zhihua Jiang, Jujian Lv
**Code**: [fip-lab/TEXT](https://github.com/fip-lab/TEXT)
**Area**: Audio & Speech
**Keywords**: Multimodal Sentiment Analysis, Sparse Mixture of Experts, Temporal Alignment, MLLM Explanation Enhancement, Gated Fusion

## TL;DR

This paper proposes TEXT, a model that leverages MLLMs to generate natural language explanations for audio and video modalities to enhance modal representations, designs a lightweight temporal alignment module combining the strengths of Mamba and temporal cross-attention, and employs text-routed sparse mixture of experts for cross-modal fusion. TEXT comprehensively outperforms prior SOTA methods and large models such as GPT-4o across four MSA benchmarks.

## Background & Motivation

### Problem Definition

Multimodal Sentiment Analysis (MSA) aims to jointly exploit text (transcripts), audio (tone/prosody), and visual (facial expressions) modalities from short videos to predict the speaker's sentiment polarity (positive/negative/neutral) and sentiment intensity score (continuous value). The task has broad applications in healthcare, human–computer interaction, and fraud detection. The core challenge lies in the vastly different—and sometimes contradictory—contributions of modalities to sentiment: text may convey positivity while vocal tone is negative, or facial expressions may conflict with verbal content.

### Limitations of Prior Work

Existing MSA methods can be broadly categorized into representation-learning-centric approaches (e.g., ALMT, KuDA) and fusion-centric approaches (e.g., DEVA). The authors identify three critical gaps:

**Underutilization of MLLM explanation capability**: The power of text in the LLM era has not been fully exploited. MLLMs can generate semantic explanations for audio and video to bridge the semantic gap in non-textual modalities, yet no prior work has incorporated this into the feature alignment pipeline of MSA.

**Temporal alignment strategies mismatched with MSA**: Mamba (linear SSM) is designed for long videos, and temporal cross-attention (TCA) is a general-purpose module; neither is specifically optimized for the dynamic sentiment transitions in short MSA videos.

**Fusion strategies ignore modality dominance**: Research has shown that text almost always dominates in MSA, yet existing methods lack mechanisms to exploit this prior; both SMoE and gated fusion remain underutilized in the MSA domain.

### Core Motivation

The paper motivates its approach through a concrete MOSI example in which only the text correctly identifies the sentiment polarity, while audio and video mislead the model. The prediction error of the prior best model ALMT is 0.320, whereas Qwen2.5-vl yields an error of 1.100. This demonstrates that **alignment is the critical bridge between representation learning and fusion**. Generating MLLM-based explanations for audio and video and aligning them with original features can effectively correct such bias. Simultaneously, routing expert activation using text dominance enables more precise cross-modal fusion. These observations motivate the two core design principles of TEXT: **explanation-driven alignment** and **text-routed fusion**.

## Method

### Overall Architecture

TEXT comprises six modules organized bottom-up as follows:

- **Modules ④⑤⑥ (parallel)**: Three unimodal feature extraction modules. Text is encoded by BERT over transcripts and explanations; audio features are extracted with Librosa; video features are extracted with OpenFace facial action units. Each of the audio and video modules embeds an **Explanation Alignment block**.
- **Module ③**: Temporal alignment module, which models temporal dependencies between the aligned audio and video representations.
- **Module ②**: Text-routed Sparse Mixture of Experts (SMoE) module, using text features as routing keys for cross-modal interaction.
- **Module ①**: Gated Fusion (GF) + MLP classifier for final sentiment prediction.

### Key Design 1: Two-Stage Explanation Generation

TEXT generates three components per sample—audio explanation $e_a$, video explanation $e_v$, and an overall comment $c$—via a two-stage pipeline:

**Stage 1**: VideoLLaMA 3, fine-tuned on the EMER-fine emotion dataset, serves as the multimodal understanding model. Given the original video and task-specific prompts, it generates raw explanations for audio, video, and overall content.

**Stage 2**: Qwen 3 acts as a reasoning verifier, refining and validating the raw explanations via a reasoning prompt to produce high-quality fine explanations.

The division of labor is elegant: VideoLLaMA 3 excels at multimodal perception but may produce biased descriptions, while Qwen 3's reasoning capability further calibrates the output, with the two models complementing each other to reduce cumulative error.

### Key Design 2: Explanation Alignment Block

The alignment objective is to bring audio/video features closer to the explanation text in semantic space. Concretely, given feature $F$ and explanation encoding $E$, alignment is achieved via cross-attention:

$$ca(F, E) = \text{softmax}((W_Q E)(W_K F)^T) W_V F$$

Here $Q$ is derived from the explanation while $K/V$ come from the original modality features, allowing the explanation to dominate attention allocation. All unimodal encodings are unified to 50 tokens plus one learnable aggregation token, yielding 51-dimensional embeddings. The aligned representations are denoted $E_t$ (text), $E_a$ (audio), and $E_v$ (video).

### Key Design 3: Temporal Alignment Block

This constitutes the primary technical innovation of the paper. The authors design a lightweight temporal alignment block that **relies on neither CA nor SSM**, is simpler than both Mamba and TCA, yet integrates their respective advantages. The core computation is:

$$\text{left} = E_a \oplus L(\text{Conv1d}(LN(E_a)) \otimes \sigma(LN(E_v)))$$
$$\text{right} = E_v \oplus L(\text{Conv1d}(LN(E_v)) \otimes \sigma(LN(E_a)))$$
$$E_{av} = \text{concat}(\text{left}, \text{right})$$

Key design choices: (1) Conv1d captures local temporal patterns (analogous to Mamba's sequence modeling); (2) SiLU-gated element-wise multiplication enables selective cross-modal interaction (analogous to attention weighting); (3) residual connections preserve original information; (4) the symmetric structure treats the audio→video and video→audio information flows equally. This design avoids the complex recurrence of SSMs and the quadratic complexity of cross-attention while retaining temporal modeling capability.

### Key Design 4: Text-Routed SMoE

Exploiting text dominance in MSA, TEXT uses text features $E_t$ as routing keys to determine which experts are activated to process the temporally aligned audio-visual embeddings $E_{av}$, formalized as $\text{SMoE}(E_t, E_{av})$. Intuitively, sentiment-laden keywords in text (e.g., "disappointing," "excellent") activate experts corresponding to the relevant sentiment themes, endowing the expert network with topic sensitivity.

### Loss & Training

As a regression problem, the primary optimization objective of TEXT is the MSE loss between predicted sentiment intensity $\hat{y}$ and ground-truth annotations. The output of the gated fusion classifier is:

$$\hat{y} = L(\sigma(\text{SMoE}(E_t, E_{av})))$$

where $\sigma$ denotes sigmoid gating and $L$ is a linear layer.

## Key Experimental Results

### Table 1: Main Comparison on MOSI and MOSEI

| Model | MOSI Acc-2 | MOSI MAE↓ | MOSI Corr | MOSEI Acc-2 | MOSEI MAE↓ | MOSEI Corr |
|-------|-----------|----------|----------|------------|----------|----------|
| ALMT | 83.10/85.23 | 0.716 | 0.773 | 82.39/85.87 | 0.542 | 0.767 |
| KuDA | 84.40/86.43 | 0.705 | 0.795 | 83.26/86.46 | 0.529 | 0.776 |
| DEVA | 84.40/86.29 | 0.730 | 0.787 | 83.26/86.13 | 0.541 | 0.769 |
| GPT-4o | 85.71/86.74 | 0.682 | 0.823 | 84.77/86.08 | 0.637 | 0.744 |
| Qwen2.5-vl | 83.09/83.38 | 1.129 | 0.677 | 84.14/84.59 | 1.007 | 0.587 |
| **TEXT** | **86.44/88.72** | **0.666** | **0.829** | **85.02/86.57** | **0.528** | **0.786** |

TEXT achieves 88.72% Acc-2 on MOSI (approximately 2% above GPT-4o) and reduces MAE to 0.666; on MOSEI, MAE drops to 0.528, outperforming all baselines. On CH-SIMS, MAE decreases from the second-best 0.408 (KuDA) to 0.353, a reduction of 13.5%.

### Ablation Study

#### Table 2: Ablation on MOSEI

| Setting | Acc-2 | Acc-7 | MAE↓ | Corr |
|---------|-------|-------|------|------|
| TEXT (full) | 85.02/86.57 | 52.29 | 0.528 | 0.786 |
| w/o explanations | 83.60/86.02 | 50.35 | 0.569 | 0.776 |
| Text only (w/ explanations) | 83.49/86.43 | 52.84 | 0.535 | 0.771 |
| EA → Linear | 84.25/86.57 | 48.21 | 0.577 | 0.762 |
| TA → Concat | 83.77/85.42 | 48.40 | 0.580 | 0.749 |
| TA → Mamba | 84.80/86.41 | 50.65 | 0.562 | 0.780 |
| TA → TCA | 83.41/86.43 | 51.38 | 0.565 | 0.781 |
| SMoE → Transformer | 83.73/85.33 | 50.29 | 0.573 | 0.769 |
| w/o Gated Fusion | 84.40/86.35 | 49.07 | 0.571 | 0.780 |

Replacing temporal alignment with simple concatenation increases MAE from 0.528 to 0.580 (the largest degradation), confirming that temporal alignment is the key factor in MAE improvement. Removing explanations causes an approximately 2% overall decline, and SMoE contributes comparably to explanations.

## Highlights & Insights

1. **Using MLLMs for data augmentation rather than end-to-end inference**: Unlike directly applying GPT-4o for MSA, TEXT cleverly uses MLLMs to generate explanation text as auxiliary signals, which are then encoded by BERT and used in feature alignment. This leverages the semantic understanding capability of MLLMs while avoiding their instability on regression tasks (Qwen2.5-vl MAE as high as 1.129).
2. **Minimalist design of the temporal alignment module**: Using only Conv1d, linear layers, and gating—without attention or SSM—the module outperforms both Mamba and TCA in ablation studies. This suggests that for short-video MSA, simple local temporal convolution combined with cross-modal gating is sufficient.
3. **Empirical finding of text dominance**: Ablation results show that using text alone (with explanations) achieves slightly higher Acc-7 than the full model (52.84% vs. 52.29%), indicating that audio and video primarily contribute to regression precision (MAE) rather than classification accuracy.
4. **Highly convincing qualitative case study**: On a specific sample, TEXT's prediction error is only 0.010 (ground truth 1.400 vs. prediction 1.390), while GPT-4o's error is 0.600 and Qwen2.5-vl's is 1.100. Removing explanations causes the audio branch error to surge from 0.380 to 1.440, clearly validating the value of explanation alignment.

## Limitations & Future Work

1. **Dependence on cascaded MLLMs**: Explanation generation requires two stages involving VideoLLaMA 3 and Qwen 3, introducing risks of cumulative error and substantial inference overhead—a limitation the authors themselves acknowledge.
2. **Coverage limited to Chinese and English**: Explanation quality depends on the linguistic capabilities of the MLLMs; the paper only validates on Chinese and English datasets, and generalization to other languages remains unknown.
3. **Risk of data contamination from MLLMs**: The paper notes that MLLMs may have memorized portions of certain datasets (e.g., GPT-4o performs unusually well on English datasets but degrades substantially on Chinese ones), which affects the fairness of baseline comparisons.
4. **Absent computational cost analysis**: The paper does not report the time and computational resource requirements for the MLLM explanation generation stage, leaving actual deployment costs unclear.

## Related Work & Insights

- **ALMT** (Zhang et al., 2023): Learns unrelated/conflict-suppressive representations and unifies modalities via Transformer. TEXT's explanation alignment can be viewed as an enhanced version of ALMT.
- **KuDA** (Feng et al., 2024): Proposes dominant-modality enhancement. TEXT inherits the text-dominance concept but exploits it more precisely via SMoE routing.
- **DEVA** (Wu et al., 2025): Text-guided progressive fusion with sentiment description generation. Conceptually similar to TEXT's explanation enhancement but differs in implementation.
- **Mamba** (Gu & Dao, 2024): Linear SSM model. TEXT's temporal alignment block can be viewed as a simplified variant of Mamba's convolutional pathway.
- **Insights**: The paradigm of using MLLMs as "semantic bridges" rather than direct predictors is generalizable to other multimodal regression tasks (e.g., emotional intensity estimation, pain assessment). The success of the lightweight temporal module also suggests that complex sequence modeling is unnecessary for short-sequence scenarios.

## Rating

| Dimension | Score |
|-----------|-------|
| Novelty | ⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐ |
| Overall | ⭐⭐⭐⭐ |

The combination of explanation alignment and text-routed SMoE is novel and effective, with thorough and convincing ablation studies. Points are deducted for: frequent formula rendering issues in the writing, the absence of a discussion on the computational cost of MLLM cascading, and the observation that the text-only variant already approaches full-model performance on certain metrics, implying limited headroom for multimodal fusion gains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] PaSE: Prototype-aligned Calibration and Shapley-based Equilibrium for Multimodal Sentiment Analysis](pase_prototype-aligned_calibration_and_shapley-based_equilibrium_for_multimodal_.md)
- [\[AAAI 2026\] Improving Multimodal Sentiment Analysis via Modality Optimization and Dynamic Primary Modality Selection](improving_multimodal_sentiment_analysis_via_modality_optimization_and_dynamic_pr.md)
- [\[AAAI 2026\] PSA-MF: Personality-Sentiment Aligned Multi-Level Fusion for Multimodal Sentiment Analysis](psa-mf_personality-sentiment_aligned_multi-level_fusion_for_multimodal_sentiment.md)
- [\[CVPR 2026\] Tri-Subspaces Disentanglement for Multimodal Sentiment Analysis](../../CVPR2026/audio_speech/tri-subspaces_disentanglement_for_multimodal_sentiment_analysis.md)
- [\[ICLR 2026\] Scalable Multilingual Multimodal Machine Translation with Speech-Text Fusion](../../ICLR2026/audio_speech/scalable_multilingual_multimodal_machine_translation_with_speech-text_fusion.md)

</div>

<!-- RELATED:END -->

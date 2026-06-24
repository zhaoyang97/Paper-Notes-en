---
title: >-
  [Paper Note] MEIT: Multimodal Electrocardiogram Instruction Tuning on Large Language Models for Report Generation
description: >-
  [ACL2025][Multimodal VLM][ECG report generation] This paper proposes the MEIT framework, which aligns ECG signals with LLMs through multimodal instruction tuning. It injects ECG embeddings into the self-attention layers of the LLM using a lightweight concatenation fusion strategy (requiring no extra parameters) to achieve automatic ECG report generation. It also establishes a comprehensive benchmark covering four tasks: quality evaluation, zero-shot transferability…
tags:
  - "ACL2025"
  - "Multimodal VLM"
  - "ECG report generation"
  - "instruction tuning"
  - "multimodal LLM"
  - "ECG-text alignment"
  - "medical AI"
date: 2026-05-08
content_hash: 7aef68d8e33f7150
---

# MEIT: Multimodal Electrocardiogram Instruction Tuning on Large Language Models for Report Generation

**Conference**: ACL2025  
**arXiv**: [2403.04945](https://arxiv.org/abs/2403.04945)  
**Code**: [AIoT-MLSys-Lab/MEIT](https://github.com/AIoT-MLSys-Lab/MEIT)  
**Area**: Multimodal VLM  
**Keywords**: ECG report generation, instruction tuning, multimodal LLM, ECG-text alignment, medical AI

## TL;DR

This paper proposes the MEIT framework, which aligns ECG signals with LLMs through multimodal instruction tuning. It injects ECG embeddings into the self-attention layers of the LLM using a lightweight concatenation fusion strategy (requiring no extra parameters) to achieve automatic ECG report generation. It also establishes a comprehensive benchmark covering four tasks: quality evaluation, zero-shot transferability, noise robustness, and expert alignment.

## Background & Motivation

Electrocardiogram (ECG) is the primary non-invasive tool for diagnosing cardiac diseases. In clinical practice, cardiologists must manually review ECG records and write detailed diagnostic reports, a process that is time-consuming and relies heavily on clinical expertise. Most existing AI research focuses on ECG classification tasks, while automatic report generation remains underdeveloped.

Compared to medical imaging (such as chest X-rays) report generation, ECG report generation faces unique challenges:
- **Differences in Signal Modality**: ECG consists of multi-lead time-series signals rather than images, preventing the direct transfer of existing image-based report generation methods.
- **Distinct Reporting Styles**: ECG reports are predominantly concise and keyword-driven, which differs significantly from the detailed anatomical descriptions in radiology reports.
- **Lack of Benchmarks**: Prior to this work, there was no comprehensive evaluation benchmark tailored specifically for ECG report generation.

Core Motivation: To leverage the powerful language generation and instruction-following capabilities of LLMs to construct the first ECG report generation framework based on multimodal instruction tuning, and to establish a standardized evaluation system.

## Method

### Overall Architecture

MEIT consists of three stages: data construction $\rightarrow$ instruction tuning $\rightarrow$ inference generation.

1. **Data Construction**: Seed prompts are rewritten and expanded using GPT-4 to generate 256 diverse instructions. Each ECG record-report pair is randomly matched with an instruction, organized in the format `<|user|>: {instruction, ECG signal} <|assistant|>: {report} </s>`.
2. **Instruction Tuning**: The ECG is converted into embeddings via an encoder and fused with language embeddings in the LLM's attention layers. Autoregressive loss is calculated only on tokens after `<assistant>` (label masking).
3. **Inference**: Given the instruction and ECG signal, the report is generated autoregressively.

### Key Designs

#### Key Design 1: Lightweight ECG Encoder

The ECG encoder is composed of temporal convolutional blocks, where each block contains a 1D convolutional layer + BatchNorm + ReLU + Average Pooling. Subsequently, a non-linear projection layer $\mathcal{P}_e$ is used to align the output to the dimension of the LLM's attention heads:

$$\mathbf{H}_e = \mathcal{P}_e(\mathcal{F}_e(\mathbf{X}_e))$$

where $\mathbf{X}_e \in \mathbb{R}^{M \times T}$ ($M$ is the number of leads, $T$ is the signal length). The encoder is randomly initialized, and its lightweight design allows it to quickly learn temporal patterns in ECG signals.

#### Key Design 2: Concatenation-based ECG-Text Alignment

Unlike methods such as Flamingo (trainable cross-attention) and LLaVA (direct concatenation in the input space), MEIT proposes to concatenate the ECG embeddings as prefix conditions inside each self-attention layer:

$$\mathbf{K}_{m,j} = [\mathbf{K}_{e,j}, \mathbf{K}_{t,j}]^\top, \quad \mathbf{V}_{m,j} = [\mathbf{V}_{e,j}, \mathbf{V}_{t,j}]$$

$$\text{head}_j = \text{Softmax}\left(\frac{\mathbf{Q}_{t,j} \mathbf{K}_{m,j}}{\sqrt{D_h}}\right) \mathbf{V}_{m,j}$$

The ECG embedding $\mathbf{H}_e$ is replicated for each attention head and concatenated with language features along the sequence dimension. Key and Value projections utilize the LLM's original projection matrices (shared parameters), introducing no extra trainable parameters. The advantages of this design include:
- Zero newly added parameters, which avoids catastrophic forgetting.
- Efficient fusion of both modalities via causal attention.
- Deep alignment is achieved as ECG signal information participates in attention calculations at every layer.

#### Key Design 3: LoRA Parameter-Efficient Fine-Tuning

The LLM backbone is frozen, and LoRA adapters are added only to all linear layers. Trainable parameters are restricted to the LoRA parameters and the ECG encoder parameters, significantly reducing computational costs.

## Key Experimental Results

### Experimental Settings
- **Datasets**: MIMIC-IV-ECG (800k samples, USA) and PTB-XL (22k samples, Europe), both consisting of 12-lead, 500Hz, 10-second ECGs.
- **Models**: 2 small language models (GPT2-Medium/Large) and 10 LLMs (ranging from GPT-Neo to LLaMA-3-Instruct).
- **Training**: 5 epochs, learning rate 2e-5, batch size 64, A100 GPUs.

### Table 1: MIMIC-IV-ECG Report Generation Quality (Selected Key Metrics)

| Model | Size | BLEU-4 | METEOR | ROUGE-L | CIDEr-D |
|---|---|---|---|---|---|
| GPT2-Medium | 345M | 0.425 | 0.551 | 0.523 | 3.70 |
| GPT2-Large | 774M | 0.476 | 0.595 | 0.571 | 4.21 |
| GPT-Neo | 2.7B | 0.489 | 0.727 | 0.689 | 4.81 |
| GPT-J | 6B | 0.542 | 0.756 | 0.721 | 5.23 |
| LLaMA-1 | 7B | 0.543 | 0.761 | 0.724 | 5.26 |
| Mistral-Instruct | 7B | 0.576 | 0.768 | 0.751 | 5.62 |
| LLaMA-2-Instruct | 7B | 0.581 | 0.775 | 0.745 | 5.55 |
| **LLaMA-3-Instruct** | **8B** | **0.610** | **0.799** | **0.773** | **5.78** |

LLMs comprehensively outperform SLMs; models pretrained on general instructions (the Instruct series) demonstrate the best performance. LLaMA-3-Instruct leads across all metrics.

### Table 3: BERTScore Semantic Similarity

| Model | MIMIC P/R/F1 | PTB-XL P/R/F1 |
|---|---|---|
| GPT2-Large | 0.657/0.574/0.613 | 0.625/0.553/0.586 |
| LLaMA-1 | 0.752/0.697/0.723 | 0.725/0.657/0.689 |
| Mistral-Instruct | 0.773/0.722/0.747 | 0.730/0.661/0.694 |
| LLaMA-3-Instruct | **0.798/0.745/0.771** | **0.745/0.682/0.712** |

The F1 score of LLaMA-3-Instruct improved relative to GPT2-Large by **+0.158** (MIMIC) and **+0.126** (PTB-XL).

### Table 4: Human Expert Alignment Evaluation (1-5 Scale)

| Model | Medical Terminology Accuracy | Logical Consistency | Completeness | Diagnostic Accuracy |
|---|---|---|---|---|
| LLaMA-2-Instruct | 4.25 | 4.11 | 3.72 | 3.60 |
| LLaMA-3-Instruct | **4.52** | **4.38** | **4.01** | **3.98** |

LLaMA-3 achieves a diagnostic accuracy score of 3.98/5, approaching human expert levels.

### Table 5: Ablation of Fusion Methods (LLaMA-1 7B, MIMIC-IV-ECG)

| Method | BLEU-4 | METEOR | ROUGE-L | CIDEr-D |
|---|---|---|---|---|
| LLaVA Direct Concatenation | 0.529 | 0.737 | 0.712 | 4.99 |
| Flamingo Trainable Cross-Attention | 0.527 | 0.768 | 0.715 | 5.11 |
| **MEIT Concatenation Fusion** | **0.543** | 0.761 | **0.724** | **5.26** |

Concatenation fusion achieves optimal overall performance without requiring any extra parameters.

## Key Findings

- **Model Scaling Effect**: LLMs comprehensively outperform SLMs (METEOR improvement of 0.13-0.20), but the marginal gains from 7B to 70B are minimal (F1 improvement of only 0.01-0.02), indicating that data scale might be more critical than model scale.
- **Transfer Advantages of Instruction Pretraining**: LLMs that underwent general instruction tuning (the Instruct series) consistently outperform their base versions in ECG report generation, suggesting that general instruction-following capabilities can be transferred to the medical domain.
- **Zero-Shot Cross-Domain Capability**: Directly testing on PTB-XL (Europe) after training on MIMIC (USA) results in a performance drop, yet it is significantly superior to the zero-shot results without instruction tuning. This demonstrates that ECG instruction tuning imparts effective cross-domain generalization.
- **Noise Robustness**: A reduction in SNR degrades the performance of all models, but Mistral maintains relatively strong anti-noise capabilities on ROUGE-L and METEOR.
- **Necessity of Instruction Tuning**: Ablation experiments indicate that removing instruction tuning leads to a significant decline across all metrics, with Mistral being the most severely affected.

## Highlights & Insights

- **First LLM-driven ECG Report Generation Framework**: Fills the gap in using LLMs for ECG report generation by directly processing raw signals rather than converting them to text first.
- **Zero-Extra-Parameter Fusion Strategy**: Leverages the LLM's own KV projection matrices to complete the ECG-text alignment, avoiding parameter inflation and training complexity associated with Flamingo or Q-former.
- **Comprehensive Four-Task Benchmark**: Covers quality evaluation, zero-shot transfer, noise robustness, and expert alignment, forming the most thorough ECG report evaluation system to date.
- **Training on Over 800k Samples**: Large-scale experiments conducted on MIMIC-IV-ECG (800k pairs) validate the scalability of the method.
- **Clinical Utility**: LLaMA-3 achieves a diagnostic accuracy score of 3.98/5, showcasing its potential to assist in real-world clinical scenarios.

## Limitations & Future Work

- **Uncontrollable Generation**: The generation process of LLMs is not fully interpretable, failing to guarantee the safety and consistency of medical content.
- **Lack of External Knowledge Integration**: Expert-validated knowledge bases, such as clinical guidelines and medical textbooks, are not leveraged to constrain generation quality.
- **Data Dependency**: Performance on PTB-XL is noticeably lower than on MIMIC, highlighting that data scale heavily affects report generation quality.
- **Simplistic ECG Encoder**: Relying solely on temporal convolutions, the framework does not explore stronger representation learning like Transformers or pretrained ECG encoders.
- **Evaluation Limitations**: NLG metrics (such as BLEU/ROUGE) are limited in measuring the clinical accuracy of medical reports, and expert evaluation relies purely on GPT-4o proxy.

## Related Work & Insights

- **Medical Report Generation**: Evolved from template-based methods (HRGR) $\rightarrow$ cross-modal alignment (Chen 2022) $\rightarrow$ the instruction-tuning paradigm proposed here. The trend shifts from task-specific modeling to general large model adaptation.
- **Multimodal Instruction Tuning**: While LLaVA and MiniGPT-4 focus on natural images, this work is the first to extend this paradigm to biomedical signals (ECG), demonstrating that instruction tuning is equally effective for signal-text alignment.
- **ECG + LLMs**: Previous efforts (e.g., BiosignalCopilot) converted ECG to textual features before feeding them into LLMs, resulting in lost modal information; MEIT directly processes raw signals, preserving richer temporal features.
- **Insights**: The combination of a lightweight fusion strategy and LoRA provides a reproducible paradigm for extending LLMs to other biomedical signals (such as EEG, EMG, etc.).

## Rating

- Novelty: ⭐⭐⭐⭐ — The first work to systematically apply multimodal instruction tuning to ECG report generation, with a fusion strategy that is clean and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Exhaustive evaluation exploring 12 models $\times$ 2 datasets $\times$ 4 evaluation tasks $\times$ 10 metrics, alongside complete ablation and scalability analyses.
- Writing Quality: ⭐⭐⭐⭐ — Clearly structured and systematically organized experiments, though the LaTeX formula formatting is somewhat dense.
- Value: ⭐⭐⭐⭐ — Establishes a standardized research framework for ECG report generation, carrying practical progress value for the medical signal + LLM direction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] WeatherSyn: An Instruction Tuning MLLM For Weather Forecasting Report Generation](../../ICML2026/multimodal_vlm/weathersyn_an_instruction_tuning_mllm_for_weather_forecasting_report_generation.md)
- [\[ICCV 2025\] MetaMorph: Multimodal Understanding and Generation via Instruction Tuning](../../ICCV2025/multimodal_vlm/metamorph_multimodal_understanding_and_generation_via_instruction_tuning.md)
- [\[ACL 2025\] HiDe-LLaVA: Hierarchical Decoupling for Continual Instruction Tuning of Multimodal Large Language Model](hidellava_hierarchical_decoupling_for_continual_instruction.md)
- [\[ACL 2025\] Enhancing Multimodal Continual Instruction Tuning with BranchLoRA](branchlora_continual_instruction.md)
- [\[NeurIPS 2025\] Visual Instruction Bottleneck Tuning](../../NeurIPS2025/multimodal_vlm/visual_instruction_bottleneck_tuning.md)

</div>

<!-- RELATED:END -->

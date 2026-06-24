---
title: >-
  [Paper Note] Parrot: Multilingual Visual Instruction Tuning
description: >-
  [ICML2025][Multimodal VLM][Multilingual] Proposes Parrot, which utilizes a text-guided cross-attention mechanism and an MoE module to transform English-biased visual features into language-specific representations, significantly enhancing the multilingual capabilities of MLLMs with an extremely small amount of multilingual data (~10K samples per language).
tags:
  - "ICML2025"
  - "Multimodal VLM"
  - "Multilingual"
  - "Multimodal Large Language Models"
  - "Mixture-of-Experts"
  - "Visual Instruction Tuning"
  - "Language Alignment"
date: 2026-05-08
content_hash: d6627c5ab45738c2
---

# Parrot: Multilingual Visual Instruction Tuning

**Conference**: ICML2025  
**arXiv**: [2406.02539](https://arxiv.org/abs/2406.02539)  
**Code**: [AIDC-AI/Parrot](https://github.com/AIDC-AI/Parrot)  
**Area**: Multimodal VLM  
**Keywords**: Multilingual, Multimodal Large Language Models, Mixture-of-Experts, Visual Instruction Tuning, Language Alignment

## TL;DR

Proposes Parrot, which utilizes a text-guided cross-attention mechanism and an MoE module to transform English-biased visual features into language-specific representations, significantly enhancing the multilingual capabilities of MLLMs with an extremely small amount of multilingual data (~10K samples per language).

## Background & Motivation

Current Multimodal Large Language Models (MLLMs) are trained on multimodal alignment data overwhelmingly dominated by English, causing the trained models to lose their ability to process non-English languages. The authors term this phenomenon **multilingual erosion**. For instance, LLaVA still tends to respond in English even when receiving Chinese input.

Through experiments, the authors reveal that the root cause of this problem is the **alignment failure between visual tokens and non-English text tokens**:

- Models using OpenAI-CLIP exhibit chaotic performance in Chinese scenarios, whereas models using Chinese-CLIP can correctly understand and generate Chinese.
- t-SNE visualization confirms that Chinese-CLIP's visual features are closer to Chinese prompts in the high-dimensional space.

Core Problem: **How to transform English-biased visual features into language-specific embeddings under the condition of an extreme scarcity of non-English multimodal data?**

## Method

### Overall Architecture

Building upon the standard LLaVA architecture (Vision Encoder → Projector → LLM), Parrot inserts a lightweight **multilingual MoE module** after the Projector to drive language-level alignment of visual tokens via text guidance.

### Cross-Modal Cross-Attention

First, a cross-attention mechanism is computed using the visual feature's [CLS] token $\mathbf{H}_v^{\text{cls}}$ and the text embedding $\mathbf{H}_t$ to fuse visual and textual information:

$$\mathbf{H}_v' = \text{Softmax}\left(\frac{\mathbf{H}_v^{\text{cls}} \mathbf{H}_t^T}{\sqrt{C}}\right) \mathbf{H}_t$$

where $\mathbf{Q} = \mathbf{H}_v^{\text{cls}}$ and $\mathbf{K} = \mathbf{V} = \mathbf{H}_t$. This step dynamically adjusts the visual features based on the language information of the input text.

### MoE Routing and Language Experts

The fused feature $\mathbf{H}_v'$ is fed into the MoE router (linear layer + Softmax) to generate expert activation probabilities:

$$\mathcal{P} = \text{Softmax}(\text{Linear}(\mathbf{H}_v'))$$

Each expert is a two-layer MLP (with SiLU activation) responsible for converting the English-biased visual embeddings into language-specific representations. The number of experts is set to 6, corresponding to 6 target languages. The outputs of all activated experts are weighted and aggregated:

$$\text{MoE}(\mathbf{H}_v) = \sum_{i=1}^{k} \mathcal{P}[i] \cdot \mathcal{E}(\mathbf{H}_v)_i$$

### MoE Reweighting

To stabilize training and reduce the variance of visual semantic information, a residual connection is employed for the final visual embeddings:

$$\mathbf{G}_v = \mathbf{H}_v + \alpha \cdot \text{MoE}(\mathbf{H}_v)$$

Here, $\alpha$ is a trade-off parameter ensuring that the model undergoes multilingual enhancement without losing the original visual semantics.

### Two-Stage Training

| Stage | Frozen | Trained | Data | Description |
|------|------|------|------|------|
| Stage 1: Modality Alignment | Vision Encoder + LLM | Projector | English image-text pairs | Bypasses MoE, pure alignment |
| Stage 2: Multilingual Instruction Tuning | Vision Encoder | Projector + MoE + LLM | English + Multilingual data | MoE randomly initialized, text-guided optimization |

Multilingual data collection: A non-overlapping subset is randomly sampled from the ShareGPT4V dataset and translated via GPT-4 with manual calibration, yielding approximately 10K image-text pairs per language.

## Key Experimental Results

### MMMB Benchmark (Newly proposed, 6 languages × 15 categories × 12K questions)

| Model | LLM | en | zh | pt | ar | tr | ru |
|------|-----|----|----|----|----|----|----|
| LLaVA-1.5 | Vicuna-7B | 67.1 | 58.8 | 59.8 | 43.5 | 46.4 | 59.1 |
| LLaVA-NeXT | LLaMA3-8B | 70.9 | 64.3 | 63.2 | 48.3 | 48.0 | 66.4 |
| Qwen2-VL | Qwen2-7B | 80.5 | 80.2 | 78.1 | 74.1 | 71.7 | 79.3 |
| LLaVA-OneVision | Qwen2-7B | 79.0 | 78.2 | 75.9 | 73.4 | 67.8 | 76.4 |
| **Parrot** | **Qwen2-7B** | **80.1** | **80.0** | **79.6** | **76.6** | **75.0** | **79.9** |

- Parrot-Qwen2-7B achieves SOTA performance in three languages (pt / ar / tr) on MMMB, with en / zh closely following.
- It also achieves SOTA in 4 languages on the multilingual version of MMBench.

### General Multimodal Tasks

Parrot also remains competitive on general multimodal benchmarks such as MME, MMStar, ScienceQA, RealWorldQA, and SEED-Bench, indicating that multilingual enhancement does not compromise the model's overall multimodal capabilities.

### Ablation Study

1. **Effect of Multilingual Data**: Incorporating multilingual data improves performance across all languages, but merely adding data provides limited gains for LLaVA, proving that the performance gains mainly stem from Parrot's architectural design.
2. **Effect of MoE Module**: Performance drops significantly without the MoE module, validating the effectiveness of language-level expert routing.
3. **Comparison with Translation Baselines**: The "translate-process-backtranslate" approach using the Google Translation API displays a seesaw effect (improving Chinese but degrading Russian and Portuguese), indicating that simple translation cannot substitute for language-level alignment.
4. **Scaling Law**: Scaling up the amount of multilingual data (to match the Chinese data volume of 70K) yields pt +3.0 and ar +5.2, showing that scaling the data size remains highly effective.

### Training Efficiency

- Completed entire training in just **21 hours** on 16×A100 GPUs.
- Utilizes less than 1% of the multilingual data of other multilingual MLLMs.

## Highlights & Insights

1. **Clear Diagnosis of Multilingual Erosion**: Through comparison experiments between OpenAI-CLIP and Chinese-CLIP along with t-SNE visualizations, the root cause of the issue is precisely localized to the linguistic bias of visual tokens.
2. **Extreme Data Efficiency**: Achieves significant multilingual improvements with only ~10K samples per language, rendering it highly suitable for low-resource scenarios.
3. **Modular Design**: The MoE module is plug-and-play, maintaining the backbone architecture intact and facilitating easy integration into other MLLMs.
4. **New MMMB Benchmark**: Spanning 6 languages × 15 categories × 12K questions, it adopts a Yes/No circular verification strategy to minimize the impact of random guessing, ensuring a rigorous design.

## Limitations & Future Work

1. **Limited Language Coverage**: Only covers 6 languages, lacking validation on other key languages such as Japanese, Korean, and Hindi.
2. **Hard Binding Between Expert Count and Languages**: Having 6 experts corresponding to 6 languages leaves the scaling issue of the MoE module unaddressed when expanding to more languages.
3. **Visual Encoder Fixed to CLIP ViT-L/14**: The impact of stronger visual encoders (e.g., SigLIP, InternViT) remains unexplored.
4. **Validated Only on 7B-Scale Models**: The effectiveness on larger scale LLMs (e.g., 70B) remains unknown.
5. **Quality of MMMB Benchmark Relies on Translation**: Boston-calibration notwithstanding, translation quality on low-resource languages may still suffer from biases.
6. **Insufficient Interpretability of MoE Routing**: While expert distributions are visualized, there is a lack of in-depth analysis regarding the underlying causes of activation pattern differences across different languages.

## Related Work & Insights

- **LLaVA Series**: Parrot is directly built on the LLaVA architecture as its multilingual extension.
- **MoE for Multilingual Processing**: Drawing inspiration from using MoE to handle multilingual tasks in NLP, this work is the first to apply it to the vision-language alignment level.
- **Inspirations**: This methodology can be generalized to other scenarios requiring cross-domain alignment (e.g., cross-modality, cross-style), which essentially utilizes existing text signals to guide the redistribution of the feature space.

## Rating

- Novelty: ⭐⭐⭐⭐ — The diagnosis of multilingual erosion and the design of text-guided MoE alignment are both novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Dual coverage on both multilingual and general benchmarks with comprehensive ablations, but limited in language diversity and model scales.
- Writing Quality: ⭐⭐⭐⭐ — Clear motivation, in-depth analysis, and highly persuasive charts and tables.
- Value: ⭐⭐⭐⭐ — Highly practical; the low-data, high-efficiency route holds significant value for industrial application.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Visual Instruction Bottleneck Tuning](../../NeurIPS2025/multimodal_vlm/visual_instruction_bottleneck_tuning.md)
- [\[NeurIPS 2025\] Learning to Instruct for Visual Instruction Tuning](../../NeurIPS2025/multimodal_vlm/learning_to_instruct_for_visual_instruction_tuning.md)
- [\[CVPR 2026\] LLaDA-V: Large Language Diffusion Models with Visual Instruction Tuning](../../CVPR2026/multimodal_vlm/llada-v_large_language_diffusion_models_with_visual_instruction_tuning.md)
- [\[ICCV 2025\] SMoLoRA: Exploring and Defying Dual Catastrophic Forgetting in Continual Visual Instruction Tuning](../../ICCV2025/multimodal_vlm/smolora_exploring_and_defying_dual_catastrophic_forgetting_in_continual_visual_i.md)
- [\[ACL 2025\] Enhancing Multimodal Continual Instruction Tuning with BranchLoRA](../../ACL2025/multimodal_vlm/branchlora_continual_instruction.md)

</div>

<!-- RELATED:END -->

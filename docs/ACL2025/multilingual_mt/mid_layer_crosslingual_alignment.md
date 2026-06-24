---
title: >-
  [Paper Note] Middle-Layer Representation Alignment for Cross-Lingual Transfer in Fine-Tuned LLMs
description: >-
  [ACL 2025][Multilingual & Machine Translation][Cross-Lingual Transfer] Through large-scale analysis of 1000+ language pairs (35 languages, 1190 directions), this work discovers that the **middle layer** of LLMs has the strongest potential for cross-lingual semantic alignment. It proposes alternately optimizing a middle-layer contrastive alignment loss during task fine-tuning, which significantly improves cross-lingual transfer on three major tasks: slot filling (F1 +1.5)…
tags:
  - "ACL 2025"
  - "Multilingual & Machine Translation"
  - "Cross-Lingual Transfer"
  - "middle-layer alignment"
  - "Contrastive Learning"
  - "Representation Alignment"
  - "low-resource languages"
date: 2026-05-08
content_hash: 4392a090de17d4e9
---

# Middle-Layer Representation Alignment for Cross-Lingual Transfer in Fine-Tuned LLMs

**Conference**: ACL 2025  
**arXiv**: [2502.14830](https://arxiv.org/abs/2502.14830)  
**Code**: [https://github.com/dannigt/mid-align](https://github.com/dannigt/mid-align)  
**Area**: LLM Alignment / Multilingual  
**Keywords**: Cross-Lingual Transfer, middle-layer alignment, Contrastive Learning, Representation Alignment, low-resource languages

## TL;DR

Through large-scale analysis of 1000+ language pairs (35 languages, 1190 directions), this work discovers that the **middle layer** of LLMs has the strongest potential for cross-lingual semantic alignment. It proposes alternately optimizing a middle-layer contrastive alignment loss during task fine-tuning, which significantly improves cross-lingual transfer on three major tasks: slot filling (F1 +1.5), machine translation (COMET +1.1), and JSON generation, while remaining effective for unseen languages and out-of-domain data. The separately trained alignment and task LoRA modules can be merged via weight averaging.

---

## Background & Motivation

- **Background**: Decoder-only LLMs perform excellently on specific tasks in specific languages through SFT. However, extending this capability to multiple languages (especially low-resource ones) remains difficult—fine-tuning data rarely covers all languages supported by the LLM, making cross-lingual transfer crucial.
- **Key Challenge**: Prior cross-lingual alignment methods mainly target encoder-only or encoder-decoder models (which can be aligned at the encoder output stage). Decoder-only LLMs do not have explicit input/output representation boundaries, leaving **which layer to align and how to align** as an open question.
- **Limitations of Prior Work**: (1) Task fine-tuning (including multilingual fine-tuning) maintains but **does not enhance** cross-lingual semantic alignment (verified experimentally in Figure 3), indicating that pure SFT is insufficient; (2) Existing works focus only on the transfer of classification tasks, while generation tasks (with variable-length outputs) are more challenging; (3) Many methods require monolingual data for each target language for LM adaptation, which is costly.
- **Key Insight**: By conducting a large-scale cross-lingual retrieval analysis (35 languages, 1190 directions) on Llama 3-8B and Qwen 2.5-7B using the FLoRes-200 dataset, it is **data-drivenly** discovered that the middle layers (~layer 16) exhibit the highest translation retrieval accuracy, which strongly correlates with downstream cross-lingual transfer performance ($p < 0.01$). Based on this, the authors propose imposing an explicit contrastive alignment loss at the middle layer.
- **Core Idea**: Integrate a contrastive cross-lingual alignment objective at the middle layers of the LLM, alternately optimized with the task loss, to enhance cross-lingual transfer.

---

## Method

### Overall Architecture

The training process consists of two alternately executed objectives: (1) **Task objective**—the standard cross-entropy loss for causal language modeling; (2) **Alignment objective**—imposing a contrastive loss on parallel translation sentence pairs at the middle layer. Each training step optimizes only one of these objectives to avoid manual weight tuning and gradient conflicts. Parameter-efficient fine-tuning is conducted using LoRA (rank=8), based on Llama 3-8B-Instruct and Qwen 2.5-7B-Instruct.

### Key Designs

1. **Translation Retrieval Probing**
    - **Goal**: Quantify the degree of cross-lingual semantic alignment across different LLM layers and find the optimal alignment layer.
    - **Mechanism**: Extract hidden states of 35 languages $\times$ each layer on FLoRes-200 $\rightarrow$ obtain sentence vectors via mean pooling $\rightarrow$ perform translation retrieval using ratio-based margin similarity, covering all $N(N-1)=1190$ language directions.
    - **Key Findings**: The middle layers (layer 16 in Llama, similar position in Qwen) yield the highest retrieval accuracy, while the bottom and top layers are weaker; the alignment level of low-resource languages is less than half of the overall average; middle-layer retrieval accuracy is significantly positively correlated with downstream transfer F1 ($p < 0.01$).
    - **Design Motivation**: Provides a reliable empirical foundation for the selection of subsequent alignment layers.

2. **Mid-Layer Contrastive Alignment**
    - **Goal**: Explicitly pull close the representations of parallel translation pairs and push apart non-translation pairs at the middle layer.
    - **Mechanism**: For $n$ parallel sentence pairs within a batch, extract the mean-pooled hidden states of the $i$-th layer (middle layer) and apply the InfoNCE contrastive loss:
      $$\mathcal{L}_{\text{align}} = -\log \frac{\exp(\text{sim}(\mathbf{h}_s^i, \mathbf{h}_t^i))}{\sum_{v \in \mathcal{B}} \exp(\text{sim}(\mathbf{h}_s^i, \mathbf{h}_v^i))}$$
      where $\text{sim}$ is the cosine similarity, with an optional temperature parameter $\tau$.
    - **Alignment Data**: Uses parallel sentence pairs from Tatoeba or task data; only a few hundred sentences are needed for low-resource languages. Alignment data is resampled to an approximately uniform distribution across languages.
    - **Design Motivation**: Alternately optimized with the task loss without modifying the model architecture. The training overhead is approximately twice that of standard SFT, but the gains are significant.

3. **Post-hoc Module Merging**
    - **Goal**: Enable existing task models to acquire cross-lingual capabilities without retraining.
    - **Mechanism**: Train the task LoRA adapter and the alignment LoRA adapter separately, then merge them through weighted averaging (with weights tuned on the dev set).
    - **Effect**: Post-merge performance is close to joint training (slot filling F1 +1.1 vs. joint +1.5, translation COMET +0.6 vs. joint +1.1), and gains are distributed more evenly across languages.
    - **Design Motivation**: Alignment capability and task capability are decoupled; adaptation to new languages or capability enhancement does not require access to original task training data.

### Training Details

| Configuration Item | Settings |
|--------|------|
| Base Model | Llama 3-8B-Instruct / Qwen 2.5-7B-Instruct |
| Parameter-Efficient Fine-Tuning | LoRA rank=8, covering all attention and linear projection layers |
| Effective Batch Size | 128 (for both task & alignment) |
| Contrastive Learning Mini-batch | 32 parallel sentence pairs |
| Alignment Layer Position | Middle layer (layer 16 for Llama / 32 layers total) |
| Alignment Data Volume | Only a few hundred parallel sentences for low-resource languages |
| Alignment Data Sampling | Multilingual resampling to an approximately uniform distribution |

---

## Key Experimental Results

### Main Results

| Task & Metric | Model | SFT Baseline | + Mid-Layer Alignment | Gain |
|-------------|------|----------|-------------|------|
| Slot Filling Supervised (5 Languages) F1 | Llama 3 | 76.6 | **77.0** | +0.4 |
| Slot Filling Transfer (15 Languages) F1 | Llama 3 | 60.2 | **61.7** | +1.5 |
| Slot Filling Aligned Languages F1 | Llama 3 | 51.7 | **55.5** | +3.8 |
| Machine Translation Transfer$\rightarrow$En BLEU | Llama 3 | 31.8 | **32.3** | +0.5 |
| Machine Translation En$\rightarrow$Transfer COMET | Llama 3 | 79.6 | **80.7** | +1.1 |
| Retrieval Accuracy (20-language average) | Llama 3 | 39.4% | **73.2%** | +33.8 |
| Slot Filling Transfer F1 | Qwen 2.5 | 53.5 | **55.3** | +1.8 |

### Ablation Study

| Analytical Dimension | Key Findings |
|----------|----------|
| **Alignment Layer Position** | Middle layer (16) is optimal and yields the most uniform gains across languages; bottom layer (8) severely degrades performance; top layer (32) is feasible but results in uneven gains across languages (SD $\uparrow$) |
| **Alignment Language Resource Level** | The low-resource group achieves the largest gain (+3.8 F1), and the high-resource group achieves the smallest (+0.7 F1)—languages with weaker initial alignment benefit the most |
| **Unseen Language Generalization** | Unaligned languages still see an average improvement of +0.4 F1, demonstrating that the method enhances general transfer capabilities |
| **Large-scale Alignment** | 19 languages $\rightarrow$ En alignment (+1.9 F1) > 5 languages $\rightarrow$ En (+1.5 F1); multidirectional alignment shows no extra gains, as En alignment implicitly yields multidirectional effects |
| **Domain Generalization** | Alignment using Tatoeba / IWSLT in-domain data remains effective (retrieval accuracy 71.9% / 68.5% vs. oracle 77.7%) |
| **Module Merging** | Separate training followed by merging $\approx$ joint training performance (slot filling +1.1 vs. +1.5, translation +0.6 vs. +1.1) |
| **Long-sequence Tasks** | Aligned languages gain +1.0 F1 in JSON generation, but the supervised set (including Chinese) drops by 1.0, suggesting a conflict between sentence-level alignment and long sequences |
| **Non-Latin Scripts** | Gains in non-Latin script languages are only +0.5 F1 (vs. overall +1.5 F1), limited by tokenization quality affecting mean pooling |

---

## Highlights & Insights

- **Large-scale Empirical Drive**: Retrieval analysis across 1190 language directions provides robust statistical support for "middle-layer optimality" rather than empirical guesswork.
- **Simple and Efficient Alternating Optimization**: Without modifying model architecture or manually tuning loss weights, decoupling of cross-lingual alignment and task learning is achieved.
- **Extremely Low Data Requirement**: Low-resource languages require only a few hundred parallel sentences to achieve significant transfer improvements, demonstrating high practicality.
- **Modular Design**: Alignment and task LoRA can be trained independently and then merged. This is friendly for engineering deployment—only a lightweight alignment module needs to be trained when a new language is introduced.
- **"Radiation Effect" of Middle-Layer Alignment**: After applying contrastive loss at layer 16, alignment in preceding multiple layers is also enhanced (Figure 4), whereas top/bottom layer alignment shows no such effect.

## Limitations & Future Work

- Experiments are limited to 7-8B models; the optimal alignment layer position might differ for larger models.
- Limited gains for non-Latin script languages, fundamentally caused by tokenization quality $\rightarrow$ exploration of superior pooling mechanisms is required (preliminary experiments with attention pooling were unsuccessful).
- Conflicts exist between sentence-level mean pooling alignment and long-sequence tasks (e.g., F1 in Chinese decreased by 2.2 in JSON generation).
- Alternating optimization doubles the training computational cost, which can be mitigated by the module merging scheme.
- The effectiveness of aligning multiple layers simultaneously varies by task; optimal multi-layer strategies still need to be explored.

---

## Rating

| Dimension | Score (1-10) | Description |
|------|-------------|------|
| Novelty | 7 | Contrastive alignment itself is not new; the core contribution lies in the systematic finding of "middle-layer optimality" and its application to decoder-only LLMs. |
| Practicality | 8 | Low data requirement (a few hundred parallel sentences) and plug-and-play module merging design lower the barrier for engineering deployment. |
| Experimental Thoroughness | 9 | 3 tasks $\times$ 2 models $\times$ various ablations (layer location / language resource / domain generalization / module merging), providing a comprehensive analysis. |
| Writing Quality | 8 | Clear motivation, well-organized experiments, systematic analysis, and rich figures/tables. |

## Related Work & Insights
- **vs. Cross-Lingual Methods for mBERT/XLM-R**: Prior methods targeted classification tasks for encoder-only models. This work is the first to systematically study cross-lingual transfer for generation tasks in decoder-only LLMs.
- **vs. Simple Translation Data Augmentation**: Translating all training data is costly and may introduce translation errors; middle-layer alignment is more efficient.
- Offers valuable insights into where LLM multilingual capabilities originate and in which layers they are stored.

## Rating
- Novelty: ⭐⭐⭐⭐ The finding that the middle layer is the optimal location for cross-lingual alignment is valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 1000+ language pairs + 3 task types + modular validation.
- Writing Quality: ⭐⭐⭐⭐ Systematic analysis and clear conclusions.
- Value: ⭐⭐⭐⭐ Practical significance for multilingual LLM deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Cross-Lingual Representation Alignment Through Contrastive Image-Caption Tuning](cross-lingual_representation_alignment_through_contrastive_image-caption_tuning.md)
- [\[ACL 2025\] Cross-Lingual Transfer of Cultural Knowledge: An Asymmetric Phenomenon](cross-lingual_transfer_of_cultural_knowledge_an_asymmetric_phenomenon.md)
- [\[ACL 2025\] Bridging the Language Gaps in Large Language Models with Inference-Time Cross-Lingual Intervention](bridging_the_language_gaps_in_large_language_models_with_inference-time_cross-li.md)
- [\[ACL 2025\] Semantic Aware Linear Transfer by Recycling Pre-trained Language Models for Cross-Lingual Transfer](semantic_aware_linear_transfer_by_recycling_pre-trained_language_models_for_cros.md)
- [\[ACL 2025\] Language Fusion for Parameter-Efficient Cross-lingual Transfer (FLARE)](flare_crosslingual_lora.md)

</div>

<!-- RELATED:END -->

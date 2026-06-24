---
title: >-
  [Paper Note] Multilingual Encoder Knows More Than You Realize: Shared Weights Pretraining for Extremely Low-Resource Languages
description: >-
  [ACL 2025][Multilingual & Machine Translation][weight sharing] An encoder-decoder weight sharing framework is proposed, which constructs a decoder by alternately reusing encoder weight layers and randomly initialized layers, efficiently extending the multilingual encoder CINO to the seq2seq model XLM-SWCM. With fewer than 0.5B parameters, it significantly outperforms mBART and 13B LLaMA on four extremely low-resource languages: Tibetan, Uyghur, Kazakh, and Mongolian.
tags:
  - "ACL 2025"
  - "Multilingual & Machine Translation"
  - "weight sharing"
  - "multilingual encoder"
  - "low-resource languages"
  - "Chinese minority languages"
  - "XLM-SWCM"
date: 2026-05-08
content_hash: 3a389bad0d56f731
---

# Multilingual Encoder Knows More Than You Realize: Shared Weights Pretraining for Extremely Low-Resource Languages

**Conference**: ACL 2025  
**arXiv**: [2502.10852](https://arxiv.org/abs/2502.10852)  
**Code**: [https://github.com/asd765973346/xlm-swcm](https://github.com/asd765973346/xlm-swcm)  
**Area**: Multilingual Translation  
**Keywords**: weight sharing, multilingual encoder, low-resource languages, Chinese minority languages, XLM-SWCM

## TL;DR
An encoder-decoder weight sharing framework is proposed, which constructs a decoder by alternately reusing encoder weight layers and randomly initialized layers, efficiently extending the multilingual encoder CINO to the seq2seq model XLM-SWCM. With fewer than 0.5B parameters, it significantly outperforms mBART and 13B LLaMA on four extremely low-resource languages: Tibetan, Uyghur, Kazakh, and Mongolian.

## Background & Motivation

**Background**: Multilingual pretrained models like XLM-R perform excellently on high-resource languages, but still perform poorly on extremely low-resource languages (such as Tibetan, Uyghur, Kazakh, and Mongolian). Although these languages have millions to tens of millions of speakers, their data in mainstream multilingual corpora like OSCAR is extremely scarce, with Kazakh and Mongolian even being close to zero.

**Limitations of Prior Work**: (1) Modern LLMs like LLaMA and Qwen support far fewer languages than XLM-R, leaving many languages with no available text generation models; (2) Multilingual seq2seq models like mBART and mT5 theoretically cover over a hundred languages, but are not actually trained on Chinese minority languages; (3) The writing systems of Chinese minority languages differ from those used in other regions of the same language family (e.g., Uyghur uses Arabic script in China but Cyrillic script in Central Asia), exacerbating the data mismatch issue.

**Key Challenge**: Multilingual encoders (such as XLM-R/CINO) have already learned rich cross-lingual semantic spaces during pretraining, but they are encoder-only models and cannot be directly used for text generation tasks. Training a seq2seq model from scratch is unfeasible in extremely low-resource settings due to insufficient data.

**Goal**: To design a method that efficiently extends existing multilingual encoders to an encoder-decoder architecture, reusing the pre-learned semantic space of the encoder for text generation in low-resource languages.

**Key Insight**: The weights of the encoder have already encoded rich multilingual knowledge. Directly "reusing" these weights in the decoder—rather than randomly initializing a completely new decoder—can significantly reduce the required training data and accelerate convergence.

**Core Idea**: Sharing weights between the encoder and decoder—where in every $X$ layers of the decoder, $X-1$ layers reuse the encoder weights (CustomDecoderLayer) and 1 layer is randomly initialized (NormalDecoderLayer), forming an alternating structure to balance knowledge reuse and new ability learning.

## Method

### Overall Architecture
Starting from CINO (a Chinese minority language-enhanced version of XLM-R), its encoder weights are copied to initialize the decoder, constructing the encoder-decoder architecture XLM-SWCM (457M parameters). It is first pretrained on the MC2 corpus using a dual-task setup of DAE + Machine Translation, and then fine-tuned on downstream tasks (summarization, reading comprehension, translation).

### Key Designs

1. **Alternating Weight-Sharing Decoder**:

    - **Function**: Constructing an efficient decoder to maximize the reuse of the encoder's existing knowledge.
    - **Mechanism**: The decoder consists of two types of layers alternately. CustomDecoderLayer copies all weights from the corresponding encoder layer—self-attention $\rightarrow$ encoder self-attention weights, cross-attention $\rightarrow$ encoder self-attention weights, two FFNs $\rightarrow$ encoder FFN weights. NormalDecoderLayer is completely randomly initialized. Inserting 1 Normal layer after every $X=3$ Custom layers makes the $n$-layer encoder correspond to $n + \lfloor n/X \rfloor$ decoder layers.
    - **Design Motivation**: Pure weight replication (without random layers) limits the model's ability to learn generation-specific capabilities; pure random initialization wastes the encoder's existing knowledge. $X=3$ is the balance point—providing the best results for medium-sized data. Experiments also reveal that the value of $X$ should be adjusted based on data volume: larger $X$ for small data (fewer new parameters) and smaller $X$ for large data (more new parameter capacity).

2. **Dual-Task Pretraining Strategy**:

    - **Function**: Transitioning the encoder from a fill-in-the-blank task to a sequence generation task.
    - **Mechanism**: The primary task is Denoising Autoencoding (DAE, from mBART), where text segments in the input are randomly masked/shuffled/deleted, and the decoder is trained to recover the original sequence. The auxiliary task is bidirectional Chinese $\leftrightarrow$ minority language machine translation, using 8,000 translation pairs (2,000 pairs per language) to enhance cross-lingual transfer capability.
    - **Design Motivation**: DAE helps the model transition from the encoder's word-level cloze task to sequence generation; the machine translation task directly provides cross-lingual signals, compensating for the scarcity of minority language data.

3. **Balanced Sampling Strategy**:

    - **Function**: Ensuring that low-resource languages are adequately represented during training.
    - **Mechanism**: Adopting a sampling strategy similar to XLM-R, where the sampling probability of language $i$ is $p_i = q_i^\alpha / \sum_j q_j^\alpha$, where $\alpha=0.3$ is the smoothing parameter, balancing between uniform sampling and proportional sampling.
    - **Design Motivation**: Without balancing, Chinese data would completely dominate the training, and minority languages would not be sufficiently learned.

### Loss & Training
Joint training with DAE reconstruction loss + translation cross-entropy loss. Scheduled sampling is used to gradually transition from teacher forcing to autoregressive generation. AdamW optimizer is employed with a learning rate of 1e-4 for 8 epochs, training for 92 hours on 2$\times$A800 GPUs.

## Key Experimental Results

### Main Results (Tibetan Monolingual Fine-Tuning, ROUGE-L F1)

| Model | Parameters | Summarization | Reading Comprehension | Translation |
|------|--------|------|----------|------|
| MC2-LLaMA-13B | 13B | 16.1 | 13.2 | 15.1 |
| mBART-CM | 611M | 8.6 | 7.9 | 11.5 |
| XLM-SWCM (ours) | 492M | **25.7** | **16.4** | **24.5** |

### Cross-Lingual Transfer (Chinese Fine-Tuning $\rightarrow$ Minority Languages, Summarization ROUGE-L)

| Model | Tibetan | Uyghur | Mongolian |
|------|------|----------|--------|
| MC2-LLaMA-13B* | 13.1 | 11.7 | 9.7 |
| mBART-CM | 6.8 | 2.7 | 3.1 |
| XLM-SWCM (ours) | **17.1** | **12.5** | **13.5** |

### Ablation Study

| Removed Component | Summarization | Reading Comprehension | Translation |
|----------|------|----------|------|
| Full Model | 25.7 | 16.4 | 24.5 |
| W/o Machine Translation (MT) | 25.6 | 15.1 | 20.3 |
| W/o DAE | 22.4 | 12.2 | 18.7 |
| W/o Weight Sharing (WS) | 17.1 | 11.7 | 18.2 |
| W/o All Components | 15.9 | 10.8 | 16.5 |

### Key Findings
- Weight sharing is the most critical component—removing it drops the summarization score from 25.7 to 17.1 (-33%), which is a much larger impact than removing DAE or MT.
- XLM-SWCM outperforms the 13B MC2-LLaMA on all tasks with fewer than 0.5B parameters: 59% higher in summarization and 62% higher in translation.
- The insertion frequency $X=3$ is optimal for medium-sized data (20K); larger $X$ should be applied for small data (smaller model to prevent overfitting), while smaller $X$ is suited for large data (more parameter capacity).
- Randomly initialized Normal layers are necessary—pure weight copying (Baseline B) performs far worse than the alternating structure, indicating that moderate "fresh parameters" help the model break free from the representation space constraints of the encoder.

## Highlights & Insights
- **Hidden Value of Encoder Weights**: Multilingual encoders have already learned far more language knowledge than commonly realized; the key lies in how to "unlock" this knowledge for generation tasks. Weight sharing is an elegant way of unlocking.
- **Flexibility of the $X$ Parameter**: The insertion frequency $X$ provides a simple yet effective knob to balance model capacity and data volume—this offers valuable reference for all low-resource scenarios.
- **Small Models Outperforming Large Models**: The 0.5B-parameter XLM-SWCM comprehensively outperforms the 13B LLaMA variants, demonstrating that under extremely low-resource settings, effective knowledge transfer is more critical than scaling up parameters.

## Limitations & Future Work
- Monolingual fine-tuning experiments are restricted to Tibetan (as it is the only language with a publicly available dataset); the other three languages are only evaluated via cross-lingual transfer.
- The quality of the translation pairs generated using Google Translate is questionable; although manually verified, the scale is limited (only 2,000 pairs per language).
- The generalizability of the framework needs to be extended to more languages and larger-scale encoders.
- Future directions: Applying this framework to more low-resource languages, or integrating it with larger encoders (such as XLM-R Large).

## Related Work & Insights
- **vs mBART**: mBART is also a seq2seq model but is pretrained from scratch, leading to poor generalization on unseen languages. XLM-SWCM achieves better results with less data by reusing the multilingual knowledge of CINO.
- **vs MC2-LLaMA-13B**: The 13B-parameter LLaMA variant is fine-tuned with LoRA, but exhibits weak cross-lingual transfer capabilities—often defaulting to outputting Chinese instead of the target language. The encoder-decoder structure of XLM-SWCM is naturally better suited for conditional generation.
- **vs Adapter Methods (such as LoRA)**: Tuning encoders using adapters is insufficient to acquire generative capabilities. Weight sharing provides a structural conversion scheme from "encoder $\rightarrow$ encoder-decoder".

## Rating
- Novelty: ⭐⭐⭐⭐ The framework design of alternately sharing encoder weights to the decoder is simple and effective, offering inspiration for low-resource NLP.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensively designed experiments across three tasks, monolingual + cross-lingual settings, and multi-dimensional ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear methodological descriptions, step-by-step ablation study, and in-depth analysis.
- Value: ⭐⭐⭐⭐ Provides a practical and efficient solution for text generation in extremely low-resource languages.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Accessible Machine Translation Evaluation For Low-Resource Languages](accessible_machine_translation_evaluation_for_low-resource_languages.md)
- [\[ACL 2025\] Read it in Two Steps: Translating Extremely Low-Resource Languages with Code-Augmented Grammar Books](low_resource_translation.md)
- [\[ACL 2025\] The Esethu Framework: Reimagining Sustainable Dataset Governance and Curation for Low-Resource Languages](the_esethu_framework_reimagining_sustainable_dataset_governance_and_curation_for.md)
- [\[ACL 2025\] Dictionaries to the Rescue: Cross-Lingual Vocabulary Transfer for Low-Resource Languages Using Bilingual Dictionaries](dictionaries_to_the_rescue_cross-lingual_vocabulary_transfer_for_low-resource_la.md)
- [\[ACL 2026\] Why Low-Resource NLP Needs More Than Cross-Lingual Transfer: Lessons Learned from Luxembourgish](../../ACL2026/multilingual_mt/why_low-resource_nlp_needs_more_than_cross-lingual_transfer_lessons_learned_from.md)

</div>

<!-- RELATED:END -->

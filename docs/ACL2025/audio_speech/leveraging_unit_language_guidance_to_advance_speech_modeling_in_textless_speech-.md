---
title: >-
  [Paper Note] Leveraging Unit Language Guidance to Advance Speech Modeling in Textless Speech-to-Speech Translation
description: >-
  [ACL 2025 (Findings)][Audio & Speech][None] This paper proposes the concept of "unit language," which constructs text-like representations of discrete speech units via n-gram language modeling. It utilizes multi-task learning to guide the training of textless speech-to-speech translation (S2ST) models, while presenting task prompt modeling to alleviate conflicts when utilizing both source-side and target-side unit languages, achieving significant improvements on the VoxPopuli…
tags:
  - "ACL 2025 (Findings)"
  - "Audio & Speech"
  - "None"
date: 2026-05-08
content_hash: 453ed4a05bb53381
---

# Leveraging Unit Language Guidance to Advance Speech Modeling in Textless Speech-to-Speech Translation

**Conference**: ACL 2025 (Findings)  
**arXiv**: [2505.15333](https://arxiv.org/abs/2505.15333)  
**Code**: None  
**Area**: Others  
**Keywords**: None  

## TL;DR

This paper proposes the concept of "unit language," which constructs text-like representations of discrete speech units via n-gram language modeling. It utilizes multi-task learning to guide the training of textless speech-to-speech translation (S2ST) models, while presenting task prompt modeling to alleviate conflicts when utilizing both source-side and target-side unit languages, achieving significant improvements on the VoxPopuli tetralingual dataset.

## Background & Motivation

**Background**: Textless speech-to-speech translation (S2ST) has received significant attention in recent years. Its core idea is to bypass intermediate text representations, directly generating target-language speech from source-language speech. A typical approach first encodes speech into discrete unit sequences using self-supervised speech models (e.g., HuBERT) and then translates them using a sequence-to-sequence model.

**Limitations of Prior Work**: Textless S2ST faces two major modeling challenges: (1) **Cross-Modal (CM) challenge**—how to effectively extract linguistic features from continuous speech signals, as discrete units discard some linguistic structures despite compressing information; (2) **Cross-Lingual (CL) challenge**—discrete speech unit sequences are usually much longer than corresponding text sequences (a speech unit sequence can be 5-10 times longer than text tokens), making cross-lingual alignment over long sequences difficult. Compared to cascade systems that use text as intermediate representations, end-to-end textless systems still exhibit a significant performance gap.

**Key Challenge**: Although discrete speech units contain linguistic information, their sequences are excessively long and lack clear word boundaries and linguistic structures like text, making effective modeling difficult. Using text auxiliary signals can improve performance, but this contradicts the "textless" goal.

**Goal**: Without relying on text transcriptions, construct a text-like speech representation to bridge the information gap between discrete units and text.

**Key Insight**: The authors observe that repetitive patterns in discrete speech units contain linguistic structural information, which can be discovered and leveraged through n-gram statistical modeling, analogous to finding words from character sequences.

**Core Idea**: Compress discrete speech unit sequences into "unit language" representations—a shorter and more text-structured representation—via n-gram language modeling, and then use multi-task learning with these representations as auxiliary supervision signals to guide S2ST model training.

## Method

### Overall Architecture

The overall pipeline consists of three steps: (1) convert speech into discrete unit sequences using self-supervised models like HuBERT; (2) perform n-gram language modeling on discrete units to construct "unit language" representations (merging frequent n-grams into new tokens, similar to BPE); (3) incorporate source-side and/or target-side unit language prediction as auxiliary tasks in sequence-to-sequence translation training, guiding the main translation task via multi-task learning.

### Key Designs

1. **Unit Language Construction**:

    - **Function**: Compresses lengthy discrete speech unit sequences into shorter, text-like representations.
    - **Mechanism**: Performs n-gram frequency statistics on discrete unit sequences and merges high-frequency n-grams into single tokens (similar to the BPE tokenization algorithm). After multiple iterations, the original sequences of several hundred units can be compressed into sequences of dozens of "unit words". This compressed sequence is called "unit language".
    - **Design Motivation**: Excessive length of discrete unit sequences is the core bottleneck of S2ST. By simulating the text tokenization process, the unit language preserves linguistic structures while significantly shortening sequence lengths, making cross-lingual alignment easier for the model to learn.

2. **Multi-Task Learning Framework**:

    - **Function**: Leverages unit language as auxiliary supervision signals to enhance the translation model.
    - **Mechanism**: A source unit language prediction task is added at the encoder side (to help the encoder extract linguistic features, addressing the CM challenge), and a target unit language prediction task is added at the decoder side (to help the decoder generate target language structures, addressing the CL challenge). The total loss is a weighted sum of the main translation loss and auxiliary task losses: $L = L_{s2u} + \alpha L_{src\_ul} + \beta L_{tgt\_ul}$.
    - **Design Motivation**: Models trained solely on the translation target lack explicit modeling of linguistic structures. Auxiliary tasks provide additional inductive bias.

3. **Task Prompt Modeling**:

    - **Function**: Alleviates conflicts when using both source and target unit language auxiliary tasks simultaneously.
    - **Mechanism**: Preliminary experiments revealed that introducing both source and target auxiliary tasks simultaneously degrades performance because the optimization directions of the two tasks on shared parameters may conflict. The authors propose adding learnable task prompts before model inputs. Different tasks use different prompt vectors, allowing the model to distinguish task objectives based on prompts, thereby mitigating gradient conflicts.
    - **Design Motivation**: Task conflicts are common in multi-task learning. Task prompt modeling offers a lightweight solution without requiring independent task-specific networks.

### Loss & Training

The total training loss is a weighted sum of three parts: the cross-entropy loss of the main translation task $L_{s2u}$, the source-side unit language prediction loss $L_{src\_ul}$, and the target-side unit language prediction loss $L_{tgt\_ul}$. Under task prompt modeling, the three tasks share backbone parameters but are distinguished by different prompt prefixes. Training is conducted in stages, warming up the translation task before introducing auxiliary tasks.

## Key Experimental Results

### Main Results

Evaluated on four language pairs in the VoxPopuli dataset (Es→En, Fr→En, Es→Fr, Fr→Es, etc.):

| Method | Es→En BLEU | Fr→En BLEU | Avg BLEU | ASR-BLEU |
|------|-----------|-----------|---------|----------|
| Baseline (Textless S2U) | ~15 | ~17 | ~16 | ~20 |
| + Source Unit Language | ~17 | ~19 | ~18 | ~23 |
| + Target Unit Language | ~16.5 | ~18.5 | ~17.5 | ~22 |
| + Dual-side (w/o Prompt) | ~16 | ~18 | ~17 | ~21 |
| + Dual-side + Task Prompt | ~18 | ~20 | ~19 | ~24 |
| Cascade System using Text | ~19 | ~21 | ~20 | ~25 |

### Ablation Study

| Configuration | Avg BLEU | Description |
|------|---------|------|
| Full Model (Dual-side + Task Prompt) | ~19 | Best textless approach |
| Source Unit Language Only | ~18 | Individually helps the encoder effectively |
| Target Unit Language Only | ~17.5 | Individually helps the decoder effectively |
| Dual-side w/o Task Prompt | ~17 | Conflicts lead to performance degradation |
| No Unit Language (Baseline) | ~16 | Pure S2U translation |
| Different n-gram sizes | 4-gram optimal | Too small delivers insufficient compression, too large loses information |

### Key Findings

- **Unit language effectively bridges the gap between textless and text-based systems**: With unit language guidance, the performance of the textless system approaches that of the text-based cascade system, demonstrating that manageable linguistic structures are indeed embedded in discrete speech units.
- **Conflicts exist between source-side and target-side auxiliary tasks**: Directly stacking two auxiliary tasks performs worse than using only one, but task prompt modeling successfully mitigates this conflict.
- **Task prompt is a lightweight yet effective solution**: It resolves multi-task conflicts with only a small number of learnable parameters, without affecting inference efficiency.
- **The choice of n-gram size affects performance**: 4-gram serves as a solid trade-off; too small of an n-gram fails to compress sequences effectively, while too large of an n-gram may introduce excessive noise.

## Highlights & Insights

- **Proposal of the "Unit Language" concept**: Constructing text-like representations from discrete speech units using statistical methods is an intuitive and effective idea. This representation does not rely on any text data yet mimics text structures, providing a new tool for textless speech processing.
- **Identification and resolution of multi-task conflicts**: The paper honestly reports the conflict between source and target auxiliary tasks and proposes task prompts as a concise solution. This "identify problem - analyze cause - propose solution" research process is exemplary.
- **New ceiling for textless speech translation**: It demonstrates that even without text transcripts, mining the latent structures of speech units themselves can achieve performance close to text-based systems, offering promise for S2ST in low-resource languages.

## Limitations & Future Work

- Experiments are conducted only on the VoxPopuli dataset, which mostly consists of European parliament speeches with limited diversity in terms of languages and domains.
- The construction of unit language relies heavily on the discretization quality of self-supervised models like HuBERT, which might not be suitable for extremely low-resource languages.
- No comparison is made with the latest large-scale speech language models (e.g., AudioPaLM, SeamlessM4T).
- Although task prompt modeling is effective, the paper does not deeply analyze the relationship between the learned representations and task characteristics.

## Related Work & Insights

- **vs Textless NLP (Lakhotia et al.)**: Early textless NLP work pioneered the use of discrete speech units. This paper builds upon it by proposing n-gram compression to construct higher-level "unit languages".
- **vs Translatotron (Jia et al.)**: The Translatotron series represents end-to-end S2ST, but relies on spectrograms instead of discrete units. This work's unit-based scheme is more compatible with language models.
- **vs SeamlessM4T**: Meta's large-scale models utilize text-assisted training, whereas this paper's textless approach, though smaller in scale, presents a more challenging and promising research direction.

## Rating

- Novelty: ⭐⭐⭐⭐ The concept of unit language is novel, and applying the BPE concept to speech units is an interesting cross-domain adaptation.
- Experimental Thoroughness: ⭐⭐⭐ The evaluation is limited to a single dataset (VoxPopuli) and lacks comparison with large-scale baselines.
- Writing Quality: ⭐⭐⭐⭐ The problem analysis is clear, and the methodology is presented in a well-structured manner.
- Value: ⭐⭐⭐ Inspiring for textless S2ST, though practical applications are currently limited to low-resource scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Improving Language and Modality Transfer in Translation by Character-level Modeling](improving_language_and_modality_transfer_in.md)
- [\[ICLR 2026\] Towards True Speech-to-Speech Models Without Text Guidance](../../ICLR2026/audio_speech/towards_true_speech-to-speech_models_without_text_guidance.md)
- [\[ACL 2025\] ChildMandarin: A Comprehensive Mandarin Speech Dataset for Young Children Aged 3-5](childmandarin_a_comprehensive_mandarin_speech_dataset_for_young_children_aged_3-.md)
- [\[ACL 2025\] Different Speech Translation Models Encode and Translate Speaker Gender Differently](different_speech_translation_models_encode_and_translate_speaker_gender_differen.md)
- [\[ACL 2026\] From Flat Language Labels to Typological Priors: Structured Language Conditioning for Multilingual Speech-to-Speech Translation](../../ACL2026/audio_speech/from_flat_language_labels_to_typological_priors_structured_language_conditioning.md)

</div>

<!-- RELATED:END -->

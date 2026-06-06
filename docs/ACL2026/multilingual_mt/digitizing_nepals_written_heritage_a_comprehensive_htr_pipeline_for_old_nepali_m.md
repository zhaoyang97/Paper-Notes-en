---
title: >-
  [Paper Note] Digitizing Nepal's Written Heritage: A Comprehensive HTR Pipeline for Old Nepali Manuscripts
description: >-
  [ACL 2026][Multilingual & Machine Translation][HTR] The first end-to-end **Old Nepali Handwritten Text Recognition (HTR)** pipeline: Using a "Synthetic Devanagari → Printed Nagari → Old Nepali Manuscripts" 3-stage transf…
tags:
  - "ACL 2026"
  - "Multilingual & Machine Translation"
  - "HTR"
  - "Devanagari"
  - "Low-resource"
  - "TrOCR"
  - "3-stage Transfer Learning"
  - "Data Augmentation"
date: 2026-05-08
content_hash: fea9ac5d8e85ecbd
---

# Digitizing Nepal's Written Heritage: A Comprehensive HTR Pipeline for Old Nepali Manuscripts

**Conference**: ACL 2026  
**arXiv**: [2512.17111](https://arxiv.org/abs/2512.17111)  
**Code**: https://github.com/anjalisarawgi/nepOCR/ (Available)  
**Area**: Multilingual / OCR / Low-resource Document Recognition  
**Keywords**: HTR, Devanagari, Low-resource, TrOCR, 3-stage Transfer Learning, Data Augmentation

## TL;DR
The first end-to-end **Old Nepali Handwritten Text Recognition (HTR)** pipeline: Using a "Synthetic Devanagari → Printed Nagari → Old Nepali Manuscripts" 3-stage transfer learning curriculum + 20 types of data augmentation + byte-level BPE + script-aware decoder, the CER was reduced from the fine-tuned TrOCR baseline of 9.6% to **4.9%**. Code, models, and a Streamlit web application are open-sourced.

## Background & Motivation
**Background**: Modern HTR has evolved from CNN+CTC to the transformer paradigm (TrOCR), performing well on English/Latin manuscripts. However, **low-resource + historical scripts** (e.g., Old Nepali Devanagari) remain challenging for OCR due to diverse handwriting styles, severe document degradation, complex conjuncts, and a nearly complete lack of annotated data.

**Limitations of Prior Work**: (1) Tesseract / Google Cloud Vision fail on historical Devanagari manuscripts (unable to capture conjuncts, diacritics, and punctuation); (2) TrOCR was not pre-trained on Devanagari, outputting non-Devanagari text out-of-the-box; (3) Only 155 manuscript pages (3100 lines) are available, which is insufficient to train a transformer from scratch; (4) Historical documents feature scriptio continua (no word spaces), inconsistent punctuation, character evolution (U+0310 vs U+0901 synonymy), and inconsistent spelling, leading to noisy labels.

**Key Challenge**: Transformer-based HTR requires large-scale data, but Old Nepali manuscripts are naturally low-resource. Direct fine-tuning of TrOCR yields poor results because the tokenizer does not understand Devanagari.

**Goal**: Construct a pipeline that can (1) achieve a stable CER of < 5% on ~3000 lines of training data; (2) automatically handle line segmentation → recognition → post-processing; (3) provide an ablation study to allow other low-resource script researchers to reuse the methodology.

**Key Insight**: (a) 3-stage curriculum learning—using 11 fonts to synthesize 100K lines of Devanagari to learn general glyphs, then transfer learning on 5K lines of printed Nagari to learn real noise, and finally fine-tuning on 3100 lines of manuscripts; (b) Script-aware decoder—training a custom byte-BPE / char-BPE tokenizer + BERT or GPT-2 decoder to replace TrOCR's English wordpiece vocabulary; (c) Data-centric optimization (label normalization + 20 types of augmentation × 8x expansion) outweighs architectural differences.

**Core Idea**: The ceiling for low-resource historical script OCR lies not in model capacity, but in data quality and curriculum. Once the "Synthetic Pre-training + Printed Bridge + Manuscript Fine-tuning" stages are perfected, architectural choices like BERT vs. GPT-2 or char-BPE vs. byte-BPE have minimal impact.

## Method

### Overall Architecture
The pipeline consists of five steps: ① **Line segmentation**—using Kraken polygon-based methods to segment 155 manuscripts (averaging 1593×133 px, 1198 characters, 20 lines) into 3100 line images; ② **Stage 1 Synthetic Pre-training**—extracting text from 21 historical Nepali textbooks (Internet Archive), rendering 100K training lines with 11 Devanagari fonts + 10 noise deformations (perspective, blur, salt-and-pepper, JPEG compression, etc.); ③ **Stage 2 Printed Transfer**—fine-tuning on 5139 lines segmented from printed Nagari scans on heiDATA; ④ **Stage 3 Manuscript Fine-tuning**—final fine-tuning on 3100 manuscript lines (80/10/10 split); ⑤ **Decoding + Post-processing**—comparing multiple decoding strategies and flagging post-correction based on token uncertainty. All stages use AdamW, $lr=3e-5$, $bs=8$, $warmup=500$, for 6/10/20 epochs respectively.

### Key Designs

1.  **3-stage Transfer Learning Curriculum**:
    - **Function**: Transitions the transformer through 3 curriculum stages to learn Devanagari handwriting with only 3100 lines of ground truth.
    - **Mechanism**: Stage 1 (100K synthetic) allows the decoder to learn visual priors and language distributions of Devanagari characters; Stage 2 (5K printed Nagari) bridges the gap between synthetic and real scan noise; Stage 3 (3K manuscripts) adapts to handwriting styles, orthographic variations, and paper degradation. Each stage uses the same seed and 80/10/10 split. CER was 0.71 after Stage 1, 0.51 after Stage 2, and 0.049 (large encoder) after Stage 3.
    - **Design Motivation**: Direct fine-tuning on 3100 lines leads to severe overfitting; single-stage synthetic pre-training fails to bridge the distribution gap between synthetic and real noise. The 3-stage curriculum is an effective intermediary validated by ablation studies (Table 18).

2.  **Script-aware decoder + byte-BPE tokenizer**:
    - **Function**: Replaces TrOCR's default English wordpiece tokenizer and decoder with a custom BPE tokenizer and a lightweight BERT/GPT-2 for Devanagari text generation.
    - **Mechanism**: (a) Trains two types of BPE using HuggingFace tokenizers—CharBPETokenizer and ByteLevelBPETokenizer with a vocabulary size of 500 (standard best practice for coverage and frequency balance in low-resource settings); (b) Uses standard BERT/GPT-2 configurations (12 layers, 768 hidden, 12 heads, 114M parameters) trained from scratch; (c) Combines these with TrOCR ViT encoders (base/large-handwritten) for 12 experimental combinations.
    - **Design Motivation**: TrOCR's default vocabulary is robertaBPE, which lacks Devanagari conjuncts, forcing the model to guess on OOV tokens. A custom byte-BPE covers Devanagari characters and conjuncts perfectly. Findings shows that architectural differences are minimal (<0.005 CER) in low-resource settings, confirming data quality dominates.

3.  **Data-centric optimization (label normalization + 20 augmentations × 8x)**:
    - **Function**: Increases dataset diversity and quality without additional labels; this was the single largest contributor to reducing CER from 0.089 to 0.056.
    - **Mechanism**: (a) **Label normalization**: Standardizes homoglyphs like chandrabindu (U+0310 vs U+0901), removes extra spaces, normalizes bullets to danda, and converts ASCII digits to Devanagari digits (affecting 57% of lines); (b) **20 types of augmentation**: Shape deformations (rotation ±3°, shift, perspective, shear), quality degradation (Gaussian blur/noise, motion blur, JPEG compression), and character-level distortions (blurred patches, sine wave, elastic warp); (c) Comparison of expansion factors (2×/4×/8×/12×/16×) found 8× (22,320 samples) to be the plateau.
    - **Design Motivation**: 3100 lines cannot cover the full distribution of handwriting styles and document degradation. Normalization addresses systematic label noise, providing stable gains (Table 3 shows normalization alone gives -0.005, and 8× augmentation gives -0.028).

### Loss & Training
Standard cross-entropy NLL is used throughout. CER (normalized Levenshtein distance) is the main metric for model selection, with weighted CER and exact match accuracy as auxiliary metrics. Unicode zero-width characters (U+200B/200C/200D) are removed during evaluation. Five decoding methods (beam search, contrastive search, temperature sampling, top-k, top-p) were compared; **decoding strategy had almost no impact on the results** (weighted CER ≈ 0.0483-0.0490).

## Key Experimental Results

### Main Results

Final model vs. existing OCR baselines (CER↓):

| System | CER | CER(w) | ACC | Notes |
|------|-----|--------|-----|------|
| Google Cloud Vision OCR | failure | - | - | Fails on conjuncts, diacritics |
| Fine-tuned TrOCR (Default decoder) | 0.096 | - | - | Baseline direct fine-tuning |
| Ours (base-handwritten + BERT+byteBPE + 8× aug) | **0.056** | 0.057 | 29.4% | base encoder |
| **Ours (large-handwritten + BERT+byteBPE + 8× aug)** | **0.049** | **0.048** | **33.5%** | Final model |

→ CER reduced from baseline 0.096 to final 0.049, an **absolute drop of 4.7pp and a relative drop of 49%**.

Model architecture combinations (Stage 3 results):

| Encoder | Decoder | Tokenizer | Stage 3 CER | ACC |
|---------|---------|-----------|--------------|------|
| trocr-base-hw | BERT | byteBPE | **0.082** | 24.8% |
| trocr-base-hw | BERT | charBPE | 0.087 | 25.5% |
| trocr-base-hw | GPT-2 | byteBPE | 0.084 | 26.1% |
| trocr-base-hw | GPT-2 | charBPE | 0.084 | 28.7% |
| swin-base | BERT | byteBPE | 0.174 | 21.9% |

→ TrOCR ViT encoder is 3× better than Swin; BERT is slightly better than GPT-2 but difference is <0.005.

### Ablation Study

Cumulative effect of data-centric interventions (base encoder):

| Step | Samples | CER | CER(w) | ACC |
|------|---------|-----|--------|------|
| Original | 2,480 | 0.089 | 0.090 | 22.9% |
| + Normalization | 2,480 | 0.084 (-0.005) | 0.084 | 21.6% |
| + Aug 2× | 7,440 | 0.067 (-0.017) | 0.068 | 26.7% |
| + Aug 4× | 12,400 | 0.060 (-0.007) | 0.061 | 27.1% |
| **+ Aug 8×** | **22,320** | **0.056 (-0.004)** | **0.057** | **29.4%** |

→ 8× is the most cost-effective ratio. Normalization + 8× augmentation = -0.033 (37% relative drop).

Cumulative gain of 3-stage curriculum (Table 18):

| Training stage | Pretraining | CER on final test | ACC |
|---------------|-------------|-------------------|------|
| Only Stage 1 | - | 0.71 | 0.0% |
| Only Stage 2 | + Stage 1 | 0.51 | 2.58% |
| Stage 3 | + Stage 1 + Stage 2 | **0.056** | **29.58%** |

### Key Findings
- **Data-centric interventions > Architecture tuning**: Normalization + 8× augmentation contributed -0.033 CER, which is higher than encoder upgrades (-0.007) and decoder variants combined.
- **Decoding strategy has minimal impact**: Difference between beam search and sampling was <0.001 CER.
- **Errors are highly structured**: Top 10 characters contribute 55.9% of errors, with virama (12.92%) and space (11.41%) being the primary culprits. Systematic confusion exists between visually similar pairs (y vs p, t vs n).
- **Failure on long lines**: CER spikes when line length > 120 characters. Splitting long lines into halves significantly reduced error counts (e.g., 23 → 4).
- **27% of errors flaggable via uncertainty**: Using probability ratios of top-1/top-2 tokens, 27% of errors can be flagged, over half of which are recoverable from the top-3 candidates.

## Highlights & Insights
- **Data > Model**: The conclusion that data quality and normalization are more impactful than model architecture (BERT vs. GPT-2) is empirically validated for low-resource HTR.
- **3-stage Curriculum + Custom BPE**: This combination provides a replicable "recipe" for any low-resource historical document OCR task.
- **Structured error analysis**: The structured nature of errors indicates the model is learning meaningful patterns; the token uncertainty flag provides a baseline for future human-in-the-loop post-correction.
- **Long-line split trick**: A simple heuristic of splitting lines >120 characters addresses the length OOD generalization problem inherent in transformer encoders.

## Limitations & Future Work
- **Closed Data**: While code and models are public, the ground truth dataset cannot be released due to copyright, hindering exact reproduction.
- **Dependency on Kraken**: Sequential errors occur if the polygon-based line segmentation fails on irregular layouts.
- **Long-line Performance**: The model generalizes poorly on sequences >120 characters due to sparse training samples at that length.
- **Lack of LM-based Post-correction**: Structured errors were not fed into a Devanagari language model for rescoring.
- **CER 4.9% remains high for scholarship**: A 5% error rate still requires intensive human proofreading for archival purposes.

## Related Work & Insights
- **vs Nakarmi et al. 2024**: While previous work used CRNN+CTC, this study proves that "low-resource + transformer" can succeed via 3-stage curriculum learning.
- **vs Garces Arias et al. 2023**: Extends the methodology from Old Occitan HTR to Devanagari, proving the framework's cross-script transferability.
- **vs Commercial Baselines**: Customizing the pipeline for the specific script remains indispensable, as GCV and standard TrOCR fail significantly on historical features like conjuncts.
- **Insight**: Researchers should prioritize data normalization and diverse augmentations over model tweaking. From-scratch lightweight decoders with script-specific BPE are superior to fine-tuning large generic decoders.

## Rating
- Novelty: ⭐⭐⭐ Existing components are combined for a novel application with thorough validation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 36 runs across architecture combinations and extensive data-centric ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and comprehensive appendices.
- Value: ⭐⭐⭐⭐ Strong baseline and open-source tools for the Nepali historical community; methodology is highly transferable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] The GaoYao Benchmark: A Comprehensive Framework for Evaluating Multilingual and Multicultural Abilities of Large Language Models](the_gaoyao_benchmark_a_comprehensive_framework_for_evaluating_multilingual_and_m.md)
- [\[ACL 2026\] Cross-Cultural Transfer of Emoji Semantics and Sentiment in Financial Social Media](cross-cultural_transfer_of_emoji_semantics_and_sentiment_in_financial_social_med.md)
- [\[ACL 2026\] Efficient Low-Resource Language Adaptation via Multi-Source Dynamic Logit Fusion](efficient_low-resource_language_adaptation_via_multi-source_dynamic_logit_fusion.md)
- [\[ACL 2026\] Just Use XML: Revisiting Joint Translation and Label Projection](just_use_xml_revisiting_joint_translation_and_label_projection.md)
- [\[ACL 2026\] BhashaSutra: A Task-Centric Unified Survey of Indian NLP Datasets, Corpora, and Resources](bhashasutra_a_task-centric_unified_survey_of_indian_nlp_datasets_corpora_and_res.md)

</div>

<!-- RELATED:END -->

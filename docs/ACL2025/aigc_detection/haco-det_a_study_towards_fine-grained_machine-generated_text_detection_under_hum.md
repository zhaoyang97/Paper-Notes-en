---
title: >-
  [Paper Note] HACo-Det: A Study Towards Fine-Grained Machine-Generated Text Detection under Human-AI Coauthoring
description: >-
  [ACL 2025][AIGC Detection][machine-generated text detection] This study proposes HACo-Det, a fine-grained machine-generated text (MGT) detection benchmark tailored for human-AI collaborative writing. By employing a multi-round local rewriting pipeline, it automatically constructs 11,200 human-AI coauthored texts with word-level attribution labels. It adapts seven mainstream detectors into a word-level sequence labeling formulation for systematic evaluation…
tags:
  - "ACL 2025"
  - "AIGC Detection"
  - "machine-generated text detection"
  - "fine-grained detection"
  - "human-AI coauthoring"
  - "word-level attribution"
  - "sequence labeling"
date: 2026-05-08
content_hash: d511cad62f96e5bf
---

# HACo-Det: A Study Towards Fine-Grained Machine-Generated Text Detection under Human-AI Coauthoring

**Conference**: ACL 2025  
**arXiv**: [2506.02959](https://arxiv.org/abs/2506.02959)  
**Code**: -  
**Area**: AIGC Detection / Human-AI Coauthored Text Detection  
**Keywords**: machine-generated text detection, fine-grained detection, human-AI coauthoring, word-level attribution, sequence labeling  

## TL;DR

This study proposes HACo-Det, a fine-grained machine-generated text (MGT) detection benchmark tailored for human-AI collaborative writing. By employing a multi-round local rewriting pipeline, it automatically constructs 11,200 human-AI coauthored texts with word-level attribution labels. It adapts seven mainstream detectors into a word-level sequence labeling formulation for systematic evaluation, revealing significant room for improvement in current fine-grained detection methods.

## Background & Motivation

- **Real-world Demand**: With the fast-growing popularity of human-AI collaborative writing systems like GPT-4o Canvas, Notion AI, and Wordcraft, human and AI contributions are increasingly intertwined. Traditional document-level binary detection (classifying the entire text as either "human-written" or "machine-written") fails to meet the practical requirements of fine-grained authorship attribution.
- **Limitations of Prior Datasets**: (1) Mainstream datasets label the entire segment of "human-written prompt + LLM continuation" as MGT, neglecting the authorship of the human-written prefix. (2) They label rewritten texts entirely as machine-generated, though words that remain morphologically unchanged should retain their original human authorship. (3) They only simulate single-round collaboration, leaving a substantial gap to real-world multi-round interaction scenarios.
- **Bottlenecks of Detection Methods**: Metric-based methods (e.g., DetectGPT, Fast-DetectGPT) rely on white-box token-level statistics. While effective at the document level, they lack the capability to generalize to fine-grained word-level or sentence-level detection. Fine-tuning-based methods are more powerful, but their cross-domain and cross-model generalization remains an open challenge.
- **Key Insight**: This work addresses these issues by designing HACo-Det, a multi-round rewriting dataset with word-level annotations, framing MGT detection as a unified word-level sequence labeling task $D_w(T_w) = L_w$. Sentence-level labels are aggregated via majority voting, and the study systematically compares seven detectors under three settings: In-Distribution (IND), Out-of-Distribution Domain (OOD-Domain), and OOD-Model.

## Method

### Overall Architecture

A three-stage pipeline is proposed: **Original Text Sampling $\rightarrow$ Multi-round LLM Local Rewriting $\rightarrow$ Word-level/Sentence-level Grounded Annotation**.

- **Data Sources**: Human-written texts from four domains (News: XSum, Stories: WritingPrompts, Academic Papers: Dagpap24, Wikipedia: Wikipedia_en), with 2,800 documents per domain, totaling 11,200 documents.
- **Generators**: Four instruction-tuned LLMs (Llama-3, Mixtral, GPT-4o mini, GPT-4o), paired with distinct rewriting instruction templates.
- **Annotation Method**: Grounded attribution labeling based on word-level alignment—determining whether each word is Human-Written Text (HWT) or Machine-Generated Text (MGT) by comparing word span alignment before and after rewriting.

### Key Designs

**1. Multi-round Local Rewriting Strategy**: In each round, NLTK is used to split the text into a sequence of sentences. According to a pre-defined algorithm (Algorithm 1), a continuous span of sentences is selected and sent to the LLM for rewriting. Each round only rewrites segments already labeled as MGT to avoid overlapping ownership ambiguity. The number of rewriting rounds is proportional to the text length, ranging from 2–3 rounds for short texts (News) up to over 6 rounds for long texts (Wikipedia), thereby generating a natural distribution of different AI intervention ratios.

**2. Word-level Grounded Attribution Labeling**: Words in the rewritten span $S'_{\text{span}}$ are labeled as MGT by default. However, if a word appears in the corresponding original span $S_{\text{span}}$ with the same lemma (including grammatical inflections), it retains its original HWT label. This rule avoids the oversimplification that "rewriting equals complete machine generation," ensuring that words retaining original expressions within an LLM-rewritten paragraph are still attributed to the human author.

**3. Unified Sequence Labeling and Sentence-level Aggregation**: All detection tasks are unified as word-level binary sequence labeling. For sentence-level detection, majority voting aggregation is applied—if the number of MGT words in a sentence $s_i$ exceeds the number of HWT words, the sentence is labeled as MGT: $L_{s_i} = \arg\max_c \sum_{k=1}^{m} \mathbb{I}(L_{w_{i,k}} = c)$. Seven mainstream detectors (DeBERTa, SeqXGPT, DetectGPT, Fast-DetectGPT, NPR, LRR, GLTR) are all adapted to fit this framework.

### Dataset Statistics

| Domain | Samples | Avg. MGT Segments | MGT Word Ratio | Avg. Text Length (Words) |
|------|:------:|:---------------:|:----------:|:-----------------:|
| News (XSum) | 2,800 | 2.44 | 45% | 1,489 |
| Story (WritingPrompts) | 2,800 | 3.97 | 40% | 1,964 |
| Paper (Dagpap24) | 2,800 | 4.56 | 32% | 4,558 |
| Wikipedia | 2,800 | 6.01 | 26% | 7,724 |
| **Total** | **11,200** | **4.25** | **36%** | **3,934** |

Data analysis reveals: GPT-4o demonstrates the highest degree of rewriting (lowest similarity to the original text), while Llama-3 is the most conservative. Texts in the Paper and Wikipedia domains are significantly longer (4k–8k words), which is more conducive to cross-domain generalization.

## Experiments

### Main Results

| Detector | Category | F1-W (Word) | F1-S (Sentence) | AUC-W | AUC-S |
|--------|------|:----------:|:----------:|:-----:|:-----:|
| Random | - | 0.433 | 0.497 | - | - |
| **DeBERTa** | Finetune | **0.831** | **0.966** | - | - |
| SeqXGPT | Finetune | 0.513 | 0.674 | - | - |
| DetectGPT | Metric (w/ perturb) | 0.375 | 0.459 | 0.482 | 0.501 |
| Fast-DetectGPT | Metric (w/ perturb) | 0.501 | 0.533 | 0.507 | 0.510 |
| NPR | Metric (w/ perturb) | 0.414 | 0.473 | 0.485 | 0.509 |
| log prob | Metric (w/o perturb) | 0.479 | 0.444 | 0.482 | 0.511 |
| LRR | Metric (w/o perturb) | 0.475 | 0.516 | 0.483 | 0.510 |
| entropy | Metric (w/o perturb) | 0.479 | 0.392 | 0.488 | 0.511 |

Core Conclusion: Metric-based methods perform close to random guess at the word level (average F1 $\approx$ 0.46), while DeBERTa stands out with a word-level F1 (F1-W) of 0.831.

### OOD Generalization

| Generalization Setting | DeBERTa F1-W Range | SeqXGPT F1-W Range | Metric-based F1-W |
|----------|:-----------------:|:-----------------:|:-----------------:|
| IND (Baseline) | 0.831 | 0.513 | 0.375–0.501 |
| OOD-Domain (Cross-domain) | 0.776–0.830 | 0.450–0.490 | Mostly minor fluctuations |
| OOD-Model (Cross-model) | 0.768–0.854 | 0.462–0.498 | Mostly minor fluctuations |

- DeBERTa maintains an F1-W of 0.77+ across both cross-domain and cross-model settings, demonstrating generalization ability far superior to other methods.
- When trained on long-text domains (Paper/Wikipedia), the OOD performance is even better than IND, indicating that the diverse linguistic patterns in long texts aid generalization.
- Training on the GPT-4o corpus yields better cross-model generalization, as its high degree of rewriting provides a stronger learning signal.

### Document-Level AI Ratio Prediction

| Method | DeBERTa | SeqXGPT | DetectGPT | Fast-DetectGPT | NPR | LRR |
|------|:-------:|:-------:|:---------:|:--------------:|:---:|:---:|
| Sentence-level Error | **1.78%** | 12.00% | 10.70% | 14.75% | 21.01% | 17.44% |
| Word-level Error | **1.84%** | 10.00% | 43.48% | 12.65% | 32.37% | 12.00% |

DeBERTa predicts the document-level AI ratio with an error of less than 2%, providing a reliable estimate of AI intervention in practical auditing scenarios.

### Key Findings

- **Metric-based Methods Fail Entirely**: All statistical metric-based detectors perform near speculative random levels on word-level detection (F1 $\approx$ random). Simple token-level metrics cannot sustain fine-grained detection; even with perturbations (DetectGPT series), there is no noticeable improvement.
- **Semantic Representation is Key**: DeBERTa acquires embedding-level semantic features through supervised fine-tuning, successfully distinguishing intertwined human and machine-generated word sequences. Although SeqXGPT is also a fine-tuning method, its architecture adaptation is insufficient.
- **Context Window acts as a Bottleneck**: Chunking long documents leads to a loss of context. The F1 scores of both DeBERTa and SeqXGPT increase monotonically with larger chunk sizes.
- **Zero-shot Detection is Far from Viable**: When the DeBERTa encoder is frozen and only the classification head is trained, the IND F1-W drops from 0.831 to 0.571. Similarly, the sentence-level zero-shot performance of metric-based methods remains near random.
- **Difficulty in Generalization Across Rewriting Paradigms**: DeBERTa trained on HACo-Det suffers a significant performance drop when transferred to SeqXGPT-Bench (which uses a different rewriting pipeline).

## Highlights & Insights

- **Realistic Task Design**: Multi-round local rewriting closely simulates real human-AI collaborative writing, which is far more realistic than simple "human prompt + AI continuation" dynamics.
- **Reasonable Word-level Grounded Annotation**: The alignment-based method preserves original attribution, avoiding the oversimplification of treating any paraphrased segment as entirely machine-generated.
- **Comprehensive Experimental Design**: A complete matrix covering 3 evaluation settings (IND / OOD-Domain / OOD-Model) $\times$ 4 domains $\times$ 4 generators.
- **Innovative AI Ratio Estimation**: Connecting fine-grained detection with document-level AI intervention estimation, which holds substantial practical utility for content auditing.

## Limitations & Future Work

- The definition of word-level "authorship transfer" is somewhat subjective. Labeling rules relying purely on morphological alignment are relatively simple and do not account for semantic-equivalence substitutions.
- It only covers paraphrase scenarios, omitting more complex collaborative operations such as LLM-mediated insertions, deletions, or structural reorganization.
- The dataset is restricted to English, leaving multilingual or cross-lingual detection capabilities unexplored.
- Each document is edited by a single LLM, failing to simulate multi-LLM workflows or active multi-round manual human editing.
- The study does not introduce a novel detection algorithm; the primary contribution lies in the dataset and the systematic benchmark evaluation.

## Related Work & Insights

- **Document-level MGT Detection**: DetectGPT (Mitchell et al., 2023) based on log-probability curvature; Fast-DetectGPT (Bao et al., 2023) introducing conditional expectation acceleration; and RAID (Dugan et al., 2024) establishing a large-scale robustness benchmark.
- **Fine-grained Detection**: Mixtext three-way classification (Gao et al., 2024); boundary detection (Kushnareva et al., 2024); MGT localization (Zhang et al., 2024c); and sentence-level multi-feature fusion (Tao et al., 2024).
- **Detector Robustness**: Shi et al. (2024) on adversarial attacks (word substitution + prompt attacks); Wang et al. (2024) testing multi-level perturbations showing significant performance degradation in most detectors.
- **Human-AI Co-writing**: Lee et al. (2022) and Chakrabarty et al. (2022) on collaborative poetry; Reza et al. (2024, 2025) on collaborative content generation.

## Rating

| Dimension | Score | Brief Comment |
|------|------|------|
| Novelty | ⭐⭐⭐⭐ | The first human-AI coauthored detection benchmark featuring multi-round rewriting and word-level attribution annotation. |
| Technical Depth | ⭐⭐⭐ | Focuses primarily on the dataset and systematic evaluation; the detection models themselves are adaptations of existing work. |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ | Complete coverage of IND/OOD-Domain/OOD-Model, evaluated with 7 methods across 3 levels of granularity. |
| Value | ⭐⭐⭐⭐ | The AI ratio prediction perspective holds practical content-auditing value, revealing notable gaps in current methods. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] An Empirical Study on Detecting AI-Generated Text in Financial Reports](an_empirical_study_on_detecting_ai-generated_text_in_financial_reports.md)
- [\[ACL 2025\] MultiSocial: Multilingual Benchmark of Machine-Generated Text Detection of Social-Media Texts](multisocial_mgt_detection.md)
- [\[ACL 2025\] Iron Sharpens Iron: Defending Against Attacks in Machine-Generated Text Detection with Adversarial Training](greater_adversarial_mgt_detection.md)
- [\[ACL 2025\] Reliably Bounding False Positives: A Zero-Shot Machine-Generated Text Detection Framework via Multiscaled Conformal Prediction](reliably_bounding_false_positives_a_zero-shot_machine-generated_text_detection_f.md)
- [\[ICLR 2026\] Unveiling Perceptual Artifacts: A Fine-Grained Benchmark for Interpretable AI-Generated Image Detection](../../ICLR2026/aigc_detection/unveiling_perceptual_artifacts_a_fine-grained_benchmark_for_interpretable_ai-gen.md)

</div>

<!-- RELATED:END -->

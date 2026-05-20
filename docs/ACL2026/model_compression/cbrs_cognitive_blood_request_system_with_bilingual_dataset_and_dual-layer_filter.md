---
title: >-
  [Paper Note] CBRS: Cognitive Blood Request System with Bilingual Dataset and Dual-Layer Filtering
description: >-
  [ACL 2026][Model Compression][Blood donation requests] CBRS proposes a multi-platform framework that efficiently detects and parses blood donation requests from social media message streams via a dual-layer filtering arc…
tags:
  - "ACL 2026"
  - "Model Compression"
  - "Blood donation requests"
  - "bilingual dataset"
  - "dual-layer filtering"
  - "low-resource languages"
  - "information extraction"
date: 2026-05-08
content_hash: 5b9122163f054f24
---

# CBRS: Cognitive Blood Request System with Bilingual Dataset and Dual-Layer Filtering

**Conference**: ACL 2026
**arXiv**: [2604.16665](https://arxiv.org/abs/2604.16665)  
**Code**: [GitHub](https://github.com/aaniksahaa/CBRS)  
**Area**: Model Compression
**Keywords**: Blood donation requests, bilingual dataset, dual-layer filtering, low-resource languages, information extraction

## TL;DR
CBRS proposes a multi-platform framework that efficiently detects and parses blood donation requests from social media message streams via a dual-layer filtering architecture (lightweight classifier + LLM). The work introduces the first bilingual dataset of 11K blood donation requests spanning Bengali, English, and transliterated Bengali. A LoRA fine-tuned Llama-3.2-3B achieves 92% zero-shot accuracy on the parsing task.

## Background & Motivation

**Background**: Emergency blood donation requests on social media are routinely buried under high-volume everyday messages. Traditional app-based systems rely on manual input and are difficult to deploy in low-resource settings. Existing disaster information extraction research has focused predominantly on English and high-resource languages.

**Limitations of Prior Work**: (1) Blood donation requests constitute only a small fraction of overall message traffic, necessitating efficient filtering. (2) Pure LLM-based filtering is not scalable due to high inference costs, while purely lightweight models suffer from high false-negative rates. (3) Detection alone is insufficient; structured information (blood type, hospital, contact details, etc.) must also be parsed from free-form text. (4) No relevant dataset exists for Bengali.

**Key Challenge**: In blood request detection, the cost of a false negative far exceeds that of a false positive; however, optimizing for high recall increases the burden on downstream processing.

**Goal**: To build a cost-efficient, multilingual, multi-platform system for detecting and parsing blood donation requests.

**Key Insight**: A dual-layer architecture decouples the two objectives of "high-recall filtering" and "high-precision verification + parsing."

**Core Idea**: The first layer employs an asymmetrically weighted lightweight classifier to ensure high recall; the second layer uses an LLM to perform both precise filtering and structured parsing within a single API call.

## Method

### Overall Architecture
Raw messages → Layer 1 (TF-IDF + asymmetrically weighted logistic regression) for coarse filtering → Layer 2 (GPT-4o-mini) for precise filtering + structured parsing → JSON-structured request output → geolocation-based donor notification system.

### Key Designs

1. **Dual-Layer Filtering (DLF)**:

    - **Function**: Achieves high-recall blood request detection under high-throughput conditions.
    - **Mechanism**: Layer 1 uses subword tokenization + TF-IDF features + asymmetrically weighted binary cross-entropy ($\alpha=12$ to penalize false negatives), ensuring extremely high recall. Layer 2 applies GPT-4o-mini for precise classification of messages passed by Layer 1. Crucially, Layer 2 performs classification and parsing within a single API call, introducing no additional cost.
    - **Design Motivation**: Filtering all messages directly with an LLM is prohibitively expensive; the dual-layer architecture significantly reduces the number of API calls.

2. **11K Bilingual Blood Donation Request Dataset**:

    - **Function**: Provides the first corpus covering blood donation requests in Bengali, English, and transliterated Bengali.
    - **Mechanism**: 11K positive samples are collected from 15 public Telegram and Facebook groups; negative samples are drawn from BengaliNMT, BengaliTLit, and similar datasets. DeepSeek-V3 is used to generate adversarial negative samples containing keywords such as "blood" and "urgent" to improve robustness.
    - **Design Motivation**: Bengali lacks dedicated blood request datasets, and dialectal variants and colloquialisms prevalent in social media require targeted coverage.

3. **LoRA Fine-Tuned Llama-3.2-3B Parser**:

    - **Function**: Parses free-form blood donation requests into structured JSON.
    - **Mechanism**: LoRA ($r=32, \alpha=16$) is applied to fine-tune Llama-3.2-3B, updating only 0.81% of parameters. Training uses 7.9K text–JSON pairs. Output fields include blood_group, bags_needed, hospital_name, contacts, and others.
    - **Design Motivation**: The fine-tuned small model surpasses few-shot performance of large models such as GPT-4o-mini in zero-shot parsing, at 35× lower inference cost.

### Loss & Training
Layer 1 uses asymmetrically weighted binary cross-entropy: $\mathcal{L} = -\alpha y \log P(y=1|\mathbf{z}) - (1-y)\log P(y=0|\mathbf{z})$, where $\alpha=12$. LoRA fine-tuning employs standard cross-entropy with 4-bit quantization and a learning rate of $2 \times 10^{-4}$.

## Key Experimental Results

### Main Results

| Method | Accuracy | Precision | Recall | F1 |
|--------|----------|-----------|--------|----|
| DLF (Layer 1) | 0.99 | 0.99 | 0.99 | 0.99 |
| TFIDF+LogReg | 0.98 | 0.98 | 0.98 | 0.98 |
| DistilBERT | 0.98 | 0.98 | 0.98 | 0.98 |
| W2V+LogReg | 0.83 | 0.80 | 0.87 | 0.81 |

| Parsing Model | Zero-shot Accuracy | Notes |
|---------------|--------------------|-------|
| LoRA Llama-3.2-3B | 92% | Fine-tuned, zero-shot |
| Base Llama-3.2-3B | ~50% | Not fine-tuned (+41.54% gain) |
| GPT-4o-mini (few-shot) | <92% | Few-shot still underperforms fine-tuned zero-shot |

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|------------|-------|
| Layer 1 only | High recall but more false positives | Asymmetric weighting preserves recall |
| Layer 1 + Layer 2 | 99% accuracy | LLM filtering eliminates false positives |
| Without adversarial negatives | Lower robustness | Adversarial samples improve classifier robustness |

### Key Findings
- The dual-layer architecture effectively balances efficiency and precision: Layer 1 eliminates the majority of irrelevant messages, while Layer 2 removes false positives.
- The LoRA fine-tuned 3B model outperforms few-shot large models such as GPT-4o-mini on parsing, while reducing input tokens by 35×.
- Adversarial negative samples (non-request texts containing keywords such as "blood" and "urgent") substantially improve classifier robustness.
- DLF achieves significantly faster inference than BERT-class models.

## Highlights & Insights
- The **dual-layer "wide intake, strict output" architecture** elegantly exploits the asymmetric costs of false negatives and false positives — Layer 1 errs on the side of inclusion, while Layer 2 filters precisely — and combines precise filtering and parsing in a single API call.
- Constructing the first blood request dataset for a low-resource language (Bengali + transliterated Bengali) carries tangible social value.
- The result that a LoRA fine-tuned small model outperforms large model few-shot baselines further corroborates the trend that task-specific small models can outperform general-purpose large models.

## Limitations & Future Work
- The dataset is drawn primarily from social media groups in Bangladesh; generalization to other regions and languages remains unvalidated.
- The system depends on the GPT-4o-mini API, raising concerns about cost and privacy.
- Only text messages are processed; blood donation requests embedded in images are not handled.
- Parsed fields are predefined; accommodating new information types (e.g., insurance details) would require schema redesign.

## Related Work & Insights
- **vs. Mathur et al. (2020)**: Their work identifies blood donation requests solely on Twitter without structured parsing or Bengali language support.
- **vs. CrisisBench**: A general benchmark for disaster information extraction that does not include a dedicated blood request task.
- **vs. direct LLM usage**: Not scalable; the DLF dual-layer architecture represents a cost-optimization pattern generalizable to other domains.

## Rating
- Novelty: ⭐⭐⭐ The dual-layer filtering architecture has engineering value but limited technical novelty; the primary contribution lies in the dataset.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-model comparison, human evaluation, and real-world deployment testing.
- Writing Quality: ⭐⭐⭐ Well-structured overall, though some formulations are unnecessary and the presentation could be more concise.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] DualReg: Dual-Space Filtering and Reinforcement for Rigid Registration](../../CVPR2026/model_compression/dualreg_dual-space_filtering_and_reinforcement_for_rigid_registration.md)
- [\[ICLR 2026\] Understanding Dataset Distillation via Spectral Filtering](../../ICLR2026/model_compression/understanding_dataset_distillation_via_spectral_filtering.md)
- [\[ACL 2026\] Adaptive Layer Selection for Layer-Wise Token Pruning in LLM Inference](adaptive_layer_selection_for_layer-wise_token_pruning_in_llm_inference.md)
- [\[ACL 2026\] A Layer-wise Analysis of Supervised Fine-Tuning](a_layer-wise_analysis_of_supervised_fine-tuning.md)
- [\[ACL 2026\] Efficient Learned Data Compression via Dual-Stream Feature Decoupling](efficient_learned_data_compression_via_dual-stream_feature_decoupling.md)

</div>

<!-- RELATED:END -->

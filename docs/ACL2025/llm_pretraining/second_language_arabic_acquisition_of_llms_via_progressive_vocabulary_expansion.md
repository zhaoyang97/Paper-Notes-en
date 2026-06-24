---
title: >-
  [Paper Note] Second Language (Arabic) Acquisition of LLMs via Progressive Vocabulary Expansion
description: >-
  [ACL 2025][LLM Pretraining][Arabic LLM] Inspired by human second language acquisition, this paper proposes Progressive Vocabulary Expansion (PVE), a method that incrementally, exponentially introduces Arabic subwords into the LLaMA2 vocabulary across stages. This approach achieves efficient Arabic language adaptation while preserving the original English knowledge of the model, culminating in the AraLLaMA 7B/13B models.
tags:
  - "ACL 2025"
  - "LLM Pretraining"
  - "Arabic LLM"
  - "vocabulary expansion"
  - "language adaptation"
  - "BPE"
  - "tokenization"
  - "continual pre-training"
date: 2026-05-08
content_hash: da273acdb1b9a4eb
---

# Second Language (Arabic) Acquisition of LLMs via Progressive Vocabulary Expansion

**Conference**: ACL 2025  
**arXiv**: [2412.12310](https://arxiv.org/abs/2412.12310)  
**Code**: [FreedomIntelligence/AraLLaMa](https://github.com/FreedomIntelligence/AraLLaMa)  
**Area**: LLM Pre-training  
**Keywords**: Arabic LLM, vocabulary expansion, language adaptation, BPE, tokenization, continual pre-training

## TL;DR
Inspired by human second language acquisition, this paper proposes Progressive Vocabulary Expansion (PVE), a method that incrementally, exponentially introduces Arabic subwords into the LLaMA2 vocabulary across stages. This approach achieves efficient Arabic language adaptation while preserving the original English knowledge of the model, culminating in the AraLLaMA 7B/13B models.

## Background & Motivation
**Background**: Current mainstream LLMs (such as GPT-4 and LLaMA) are largely optimized for English and Chinese. Despite being the fifth most spoken language globally with 420 million speakers, Arabic has seen slow progress in the LLM domain, and existing Arabic models (e.g., Jais, AceGPT) still lag significantly behind GPT-4.

**Limitations of Prior Work**: When English-centric LLMs process Arabic using their native vocabularies, they decompose Arabic words into character-level token sequences. This results in a high subword fertility of 5.38 (averaging 5.38 tokens per word), which drastically slows down decoding speed. For instance, the decoding efficiency of AceGPT on Arabic is far lower than on English.

**Key Challenge**: Directly expanding a large number of Arabic tokens into the vocabulary all at once introduces significant out-of-vocabulary (OOV) tokens, disrupting the model's pre-existing representation space. Recovering the model's capabilities then requires massive amounts of pre-training data, creating a dilemma between vocabulary expansion and knowledge preservation.

**Ours**: This paper proposes Progressive Vocabulary Expansion (PVE), which incrementally introduces 12,800 Arabic subwords into the vocabulary over 16 stages. By controlling the OOV ratio at each stage, the model smoothly adapts to the newly added tokens.

**Key Insight**: Drawing inspiration from cognitive science, the authors analogize this process to human Second Language Acquisition (SLA)—where vocabulary acquisition is progressive (referencing the vocabulary sizes required for different levels A1 to C2 under the CEFR framework) rather than mastering all words at once.

**Core Idea**: "Progression > All-at-once"—The BPE algorithm is adapted into an Incremental BPE (I-BPE) tokenizer to dynamically expand the vocabulary during training. By adding only a small number of new tokens at each stage followed by sufficient training, the model absorbs new linguistic elements while retaining its existing knowledge.

## Method

### Overall Architecture
- **Function**: Adapts the English-centric LLaMA2 model into an Arabic LLM (AraLLaMA), encompassing vocabulary expansion, multi-stage continual pre-training, and instruction tuning.
- **Why**: Language adaptation is a cost-effective pathway for low-resource languages to leverage existing powerful models, avoiding the massive computational overhead of training from scratch while retaining general capabilities via cross-lingual transfer.
- **How**: Initialized from LLaMA2-7B/13B, the method uses the I-BPE algorithm to add 12,800 Arabic subwords across 16 stages. Each stage processes 30B tokens (totaling 480B tokens), with the proportion of Arabic data gradually increasing from 30% to 90% via a cosine annealing schedule, while math and coding data remain constant at 5%. After pre-training, instruction tuning is performed using the ALAN dataset coupled with the AceGPT dataset.

### Key Designs
1. **Incremental BPE (I-BPE) Algorithm**

    - **Function**: Modifies standard BPE to dynamically expand the vocabulary during training, rather than pre-defining a complete static vocabulary.
    - **Why**: Building a complete vocabulary with standard BPE prior to training cannot accommodate the evolutionary needs of vocabulary during language adaptation. Furthermore, adding numerous new tokens at once leads to training instability and catastrophic forgetting.
    - **How**: At each stage, the vocabulary is first expanded to a predetermined size $s_i$ using frequency statistics. Subsequently, the proportion $r_i$ of newly added tokens in the training corpus is adjusted. The model is trained until convergence before proceeding to the next stage. Embeddings of new tokens are initialized as the average of their constituent subword embeddings to preserve semantic relationships.

2. **Exponential Expansion Strategy**

    - **Function**: The number of new tokens added per stage increases exponentially following $\{0, 1, 2, \ldots, 2^{T-2}\}$ (contrasting with uniform expansion where a fixed K tokens are added per stage).
    - **Why**: Uniform expansion introduces too many tokens in the early stages, causing abrupt changes in compression ratio and drastic shifts in the representation space. Exponential expansion mimics progressive human vocabulary acquisition, allowing the model to adapt stably with fewer additions early on, and rapidly enriching the vocabulary in later stages.
    - **How**: Over the 16 stages, an exponential growth of $\log_2(12800)$ steps is implemented, with the number of tokens doubling at each stage to smoothly increase the compression ratio. Ultimately, the tokenized sequence length is reduced by approximately threefold compared to the original LLaMA.

3. **ALAN Instruction-Tuning Data Generation**

    - **Function**: Proposes the ALAN (Arabic Language Acquisition for LLMs) method, which generates 733,000 instruction-tuning data entries using GPT-4, centered around 127 core themes in Arabic culture, science, and engineering.
    - **Why**: High-quality instruction-tuning data in Arabic is scarce, necessitating the systematic generation of training data covering a wide array of domains.
    - **How**: Decomposes the 127 themes into a hierarchical structure of domains, subdomains, and subjects. A curriculum syllabus containing specific knowledge points is authored for each subject (totaling 11,430 subjects and 244,812 knowledge points). Knowledge points within the same or different courses are combined to generate three categories of questions: multiple-choice, open-ended, and programming.

## Key Experimental Results

### Table 1: Tokenizer Evaluation Comparison

| Model | Total Tokens | Subword Fertility↓ | Word Integrity↑ | Rényi Efficiency |
|------|------------|--------------------|--------------------|------------------|
| LLaMA2 (AceGPT) | 210M | 5.38 | 1.8% | 0.77 |
| Bloomz | 80.6M | 2.07 | 31.8% | 0.77 |
| Jais | 75.1M | 1.93 | 39.0% | 0.73 |
| **AraLLaMA** | **66.6M** | **1.71** | **63.2%** | 0.75 |

### Table 2: Benchmark Evaluation of Chat Models in Arabic (Zero-Shot)

| Model | MMLU-ar↑ | ArabicMMLU↑ | ACVA-all↑ | BoolQ-ar↑ | ARC-C-ar↑ | English Avg↑ |
|------|---------|------------|----------|----------|----------|----------|
| AceGPT-7B-chat | 30.69 | 36.31 | 53.07 | 60.70 | 38.05 | 54.36 |
| Mistral-7B-Instruct | 27.93 | 41.44 | 63.47 | 60.18 | 35.67 | 78.85 |
| **AraLLaMA-7B-chat** | **45.77** | **56.62** | **70.86** | **72.45** | **60.49** | 73.96 |
| AceGPT-13B-chat | 35.59 | 52.61 | 70.21 | 66.85 | 44.20 | 52.88 |
| Jais-30B-chat-v3 | 35.68 | 62.36 | 73.66 | 76.30 | 51.02 | 82.43 |
| **AraLLaMA-13B-chat** | **47.33** | **61.70** | **76.37** | 69.33 | **63.99** | 82.24 |

### Table 3: Ablation Study of Progressive Vocabulary Expansion (TinyLLaMA 1B)

| Method | ArabicMMLU Avg↑ | Arabic Vicuna-80↑ |
|------|----------------|-------------------|
| TinyLLaMA (baseline) | 36.5 | 21.30% |
| + One-time Vocabulary Expansion (VE) | 38.5 | 22.61% (+1.31) |
| + **Progressive Vocabulary Expansion (PVE)** | **40.7** | **29.18% (+7.88)** |

### Key Findings
- AraLLaMA-7B outperforms competing models of the same scale (such as AceGPT and Mistral) across all Arabic tasks, scoring approximately 15 percentage points higher than AceGPT-7B on MMLU-ar.
- AraLLaMA-13B outperforms Jais-30B (which is more than twice its size) on multiple Arabic benchmarks.
- In the ablation study, Progressive Vocabulary Expansion (PVE) improves the Arabic Vicuna-80 score by 6.57 percentage points over one-time Vocabulary Expansion (VE), demonstrating that the progressive strategy is significantly superior to direct expansion.
- Tokenizer Efficiency: The Arabic generation speed of AraLLaMA reaches 20.37 words/second, which is 4.5 times faster than LLaMA2 (4.55 words/second).
- Preservation of English Capabilities: Following SFT, the English MMLU performance is even higher than the baseline model of the same scale.

## Highlights & Insights
- **Cognitive Science Inspiration**: Analogizing incremental vocabulary learning in human Second Language Acquisition (SLA) to LLM language adaptation yields an intuitive and clear methodological framework.
- **Exponential vs. Uniform Expansion**: Comparative analysis reveals the advantages of exponential expansion in training stability and OOV ratio control, with design choices supported by both theory and experiments.
- **Complete Open-Source Ecosystem**: Model weights, data processing pipelines, and pre-training/fine-tuning data are fully open-sourced. The architecture is fully compatible with LLaMA, enabling direct integration.
- **Significant Practical Value**: The 4.5x speedup in Arabic decoding has direct, practical utility for real-world deployment.

## Limitations & Future Work
- The efficacy of the method was only validated on Arabic; generalizability to other low-resource languages (e.g., Hindi, Swahili) remains untested.
- The model has not been systematically evaluated by native Arabic speakers; fluency and cultural appropriateness in real-world scenarios require further validation.
- The training utilized 2,368 Ascend 910A GPUs, indicating high resource requirements and a high barrier to replication.
- The 16-stage step-by-step training significantly increases engineering complexity compared to end-to-end training. The choice of hyperparameters (number of stages, tokens per stage, language ratio schedule) remains relatively ad hoc.

## Related Work & Insights
- **vs. AceGPT** (Huang et al., 2024): AceGPT is also based on LLaMA2 for Arabic adaptation, but using the original vocabulary results in a subword fertility of 5.38 and slow decoding. AraLLaMA reduces the fertility to 1.71 through vocabulary expansion, achieving 4.5x faster decoding and superior performance. AceGPT can be viewed as the direct predecessor to AraLLaMA.
- **vs. Jais** (Sengupta et al., 2023): Jais is a bilingual Arabic-English model pre-trained from scratch (up to 30B parameters), featuring a superior Arabic tokenizer design. However, AraLLaMA-13B outperforms Jais-30B across multiple benchmarks, demonstrating the resource efficiency advantages of the language adaptation paradigm.
- **vs. Chinese-LLaMA** (Cui et al., 2023): Chinese-LLaMA also conducted vocabulary expansion coupled with continual pre-training, but employed a one-time expansion approach. AraLLaMA's progressive strategy proved superior to one-time expansion in ablation studies, establishing a new paradigm for language adaptation research.

## Rating
- **Novelty**: ⭐⭐⭐ The progressive vocabulary expansion concept is novel and grounded in cognitive science, but the core mechanism remains a combination of BPE and continual pre-training.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ The evaluation is comprehensive, covering tokenizer performance, multiple datasets, models of varying scales, ablation studies, and decoding efficiency analysis.
- **Writing Quality**: ⭐⭐⭐⭐ The narrative flows smoothly by introducing the SLA analogy, and the motivation behind the methodology is clear, supported by rich tables and figures.
- **Value**: ⭐⭐⭐⭐ High practical reference value for LLM adaptation to Arabic and other low-resource languages. The complete open-sourcing further enhances its community impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Large Vocabulary Size Improves Large Language Models](large_vocabulary_size_improves_large_language_models.md)
- [\[ACL 2025\] FR-Spec: Accelerating Large-Vocabulary Language Models via Frequency-Ranked Speculative Sampling](fr_spec_speculative_sampling.md)
- [\[ACL 2025\] TokAlign: Efficient Vocabulary Adaptation via Token Alignment](tokalign_vocab_adaptation.md)
- [\[ICLR 2026\] Lossless Vocabulary Reduction for Auto-Regressive Language Models](../../ICLR2026/llm_pretraining/lossless_vocabulary_reduction_for_auto-regressive_language_models.md)
- [\[NeurIPS 2025\] Vocabulary Customization for Efficient Domain-Specific LLM Deployment](../../NeurIPS2025/llm_pretraining/vocabulary_customization_for_efficient_domain-specific_llm_deployment.md)

</div>

<!-- RELATED:END -->

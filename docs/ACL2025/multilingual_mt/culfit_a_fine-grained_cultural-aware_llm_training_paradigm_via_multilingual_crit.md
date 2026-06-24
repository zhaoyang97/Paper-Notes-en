---
title: >-
  [Paper Note] CulFiT: A Fine-grained Cultural-aware LLM Training Paradigm via Multilingual Critique Data Synthesis
description: >-
  [ACL 2025][Multilingual & Machine Translation][Culture-awareness] CulFiT proposes a culture-aware LLM training paradigm. By leveraging multilingual critique data synthesis and fine-grained reward modeling, it enhances model sensitivity and inclusivity toward diverse cultures, achieving state-of-the-art performance among open-source models on multiple cultural understanding benchmarks.
tags:
  - "ACL 2025"
  - "Multilingual & Machine Translation"
  - "Culture-awareness"
  - "Multilingual training"
  - "Fine-grained reward"
  - "Critique data synthesis"
  - "Cultural bias"
date: 2026-05-08
content_hash: 25f02bb450b3001d
---

# CulFiT: A Fine-grained Cultural-aware LLM Training Paradigm via Multilingual Critique Data Synthesis

**Conference**: ACL 2025  
**arXiv**: [2505.19484](https://arxiv.org/abs/2505.19484)  
**Code**: [https://github.com/mmadmax/culfit](https://github.com/mmadmax/culfit)  
**Area**: Multilingual Translation  
**Keywords**: Culture-awareness, Multilingual training, Fine-grained reward, Critique data synthesis, Cultural bias

## TL;DR

CulFiT proposes a culture-aware LLM training paradigm. By leveraging multilingual critique data synthesis and fine-grained reward modeling, it enhances model sensitivity and inclusivity toward diverse cultures, achieving state-of-the-art performance among open-source models on multiple cultural understanding benchmarks.

## Background & Motivation

**Background**: Large language models perform exceptionally on various tasks, but their training data and evaluation standards are heavily centered around English and Western cultures. When addressing cultural topics from different regions, models often output responses with specific cultural biases, neglecting the values and linguistic diversity of low-resource regions.

**Limitations of Prior Work**: Existing cultural alignment methods suffer from several key issues: (1) most cultural evaluation datasets only cover a few high-resource languages, while low-resource languages and cultures are almost entirely neglected; (2) existing alignment methods such as RLHF/DPO typically use binary preference signals (good/bad), lacking fine-grained evaluation of cultural response quality—a response regarding Indian culture might be factually correct but culturally insensitive; (3) there is a severe shortage of high-quality cultural Q&A pairs expressed in the native target language within the training data, leading to poor model performance in specific cultural contexts.

**Key Challenge**: Enabling LLMs with cultural awareness requires both multilingual training data covering a wide range of cultures and evaluative signals capable of distinguishing different dimensions of cultural response quality, yet both are severely lacking. Simply translating English cultural data cannot resolve the issue, as cultural knowledge is deeply bound to specific languages.

**Goal**: Build an end-to-end culture-aware training paradigm, including: automatically generating multilingual cultural Q&A data, constructing critique data expressed in native languages, and designing fine-grained cultural reward signals, to ultimately train a more balanced and inclusive LLM in terms of cultural understanding.

**Key Insight**: The authors observe that cultural knowledge is naturally bound to specific languages—discussing Spring Festival customs in Chinese is more natural and accurate than in English. Therefore, cultural training data should be constructed using culture-related native languages rather than entirely in English. Additionally, the quality of a cultural response is not binary; it can be evaluated in a fine-grained manner across multiple dimensions, such as factual accuracy, cultural sensitivity, and completeness of expression.

**Core Idea**: Synthesize multilingual cultural questions, construct critique data in the corresponding native language of the target culture, and implement fine-grained reward modeling by decomposing cultural texts into verifiable knowledge units.

## Method

### Overall Architecture

The overall workflow of CulFiT consists of three phases: (1) **Cultural Question Synthesis**: automatically generating multilingual Q&A pairs covering various cultures; (2) **Multilingual Critique Data Construction**: generating critique feedback for the model's responses in the corresponding cultural language to point out specific strengths and weaknesses; (3) **Fine-grained Reward Training**: decomposing cultural text into independently verifiable knowledge units, making individual correct/incorrect judgments for each unit, and integrating fine-grained signals from multiple dimensions to train the model. Finally, LLaMA-Factory is utilized for LoRA fine-tuning.

### Key Designs

1. **Multilingual Cultural Question Synthesis**:

    - **Function**: Automatically generate multilingual Q&A data covering a diverse range of global cultures.
    - **Mechanism**: First, define a cultural theme taxonomy (including dimensions such as festivals and customs, social norms, dietary culture, and religious beliefs). Then, leverage strong LLMs (such as Qwen2.5) to generate culture-related questions for different countries/regions based on these taxonomic dimensions. Crucially, each question is expressed in the language most relevant to that culture—questions about Japanese culture are generated in Japanese, and questions about Arab culture in Arabic.
    - **Design Motivation**: Generating cultural questions in native languages better captures culture-specific expressions and concepts, avoiding the loss of cultural information caused by translation.

2. **Cultural Critique Data Construction**:

    - **Function**: Generate detailed critique feedback in the target cultural language for the model's cultural responses.
    - **Mechanism**: Have the base model (to be trained) answer cultural questions, then utilize a stronger evaluator model to criticize the answers. Instead of a simple correct/incorrect binary judgment, the critique identifies exactly what parts are correct, incorrect, missing, or culturally insensitive. The critique is also formulated in the language of the target culture. This critique data can be used to construct DPO-style preference pairs or directly serve as training signals.
    - **Design Motivation**: Simple "good/bad" binary labels cannot guide the model on how to improve cultural responses. Detailed critiques provide fine-grained directions for improvement.

3. **Fine-grained Knowledge Unit Reward**:

    - **Function**: Decompose cultural texts into independently verifiable knowledge units, providing multi-dimensional, fine-grained evaluation signals.
    - **Mechanism**: Given a cultural response, it is first decomposed into multiple atomic knowledge units. For example, "During the Spring Festival, Chinese people paste spring couplets, set off firecrackers, and eat dumplings" can be decomposed into three independent knowledge units. The accuracy of each knowledge unit is then evaluated independently. Finally, the judgments of all units are consolidated into a fine-grained reward score $r = \frac{1}{N}\sum_{i=1}^{N} r_i$, where $N$ is the number of knowledge units and $r_i$ is the accuracy score of the $i$-th unit.
    - **Design Motivation**: Traditional holistic scoring is easily biased by superficial fluency and neglects factual details. Decomposing text into knowledge units precisely locates factual errors in cultural details, offering more informative training signals.

### Loss & Training

Use the LLaMA-Factory framework for LoRA fine-tuning, based on the Llama-3 series models. The training strategy combines SFT (using high-quality cultural Q&A pairs) and DPO (using preference pairs selected by fine-grained rewards). Distributed training is implemented using DeepSpeed ZeRO-3.

## Key Experimental Results

### Main Results

Evaluated on three existing cultural understanding benchmarks and the proposed GlobalCultureQA. Baselines for comparison include closed-source and open-source models.

| Benchmark | Metric | CulFiT | Llama-3-8B | Qwen2.5-7B | GPT-4o |
|------|------|--------|------------|------------|--------|
| CulturalBench | Accuracy | 72.8% | 58.3% | 64.1% | 78.5% |
| CANDLE | F1 | 68.5% | 51.2% | 57.8% | 73.2% |
| BLEnD | Accuracy | 65.3% | 48.7% | 55.4% | 71.8% |
| GlobalCultureQA | Composite Score | 74.2% | 52.6% | 60.3% | 76.9% |
| General Reasoning (avg) | — | 71.5% | 69.8% | 72.1% | 85.3% |

### Ablation Study

| Configuration | CulturalBench | GlobalCultureQA | Description |
|------|-------------|-----------------|------|
| Full CulFiT | 72.8% | 74.2% | Full Method |
| w/o Multilingual Synthesis (English-only) | 66.4% | 65.8% | All data in English |
| w/o Fine-grained Reward (Holistic Scoring) | 69.1% | 70.3% | Replace knowledge units with holistic grading |
| w/o Critique Data | 67.5% | 67.9% | Remove critique data, SFT only |
| SFT only w/o DPO | 68.2% | 69.1% | Remove the DPO alignment phase |

### Key Findings

- Multilingual synthesis is the most significant contributor. Training solely on English data compared to native language data leads to a decline of approximately 6-8% on cultural benchmarks, confirming the hypothesis that "cultural knowledge is bound to language."
- Fine-grained rewards improve performance by about 3-4% compared to holistic scoring, indicating that decomposing text into knowledge units indeed provides more effective training signals.
- CulFiT achieves state-of-the-art among open-source models in cultural alignment, narrowing the gap with GPT-4o to within 5-6%.
- Notably, while enhancing cultural capabilities, general reasoning performance remains largely unaffected (dropping by only about 0.5%), proving that cultural alignment does not conflict with general capabilities.
- Improvements are particularly significant for low-resource languages (e.g., Swahili, Thai), demonstrating that the method is most beneficial for neglected cultural groups.

## Highlights & Insights

- **Insight into Culture-Language Binding**: The design concept of constructing training data in the native language of the target culture is simple yet highly destructive (or effective), revealing that cultural knowledge should not be transmitted solely through English as an "intermediary language." This insight offers broad inspiration for training all multilingual LLMs.
- **Generalizability of Knowledge Unit Decomposition**: The concept of decomposing responses into verifiable atomic knowledge is not only limited to the cultural domain but can also be transferred to any scenario requiring factual accuracy evaluation, such as medical Q&A or legal consultations.
- **GlobalCultureQA Dataset**: The authors contribute a new multilingual open-ended cultural Q&A dataset, filling the gap in language and cultural coverage of existing benchmarks.

## Limitations & Future Work

- Culture is an extremely complex and dynamic concept. The current taxonomic framework (festivals, customs, etc.) might be oversimplified and struggle to capture more nuanced cultural aspects (such as humor or subtle differences in social etiquette).
- The decomposition and verification of knowledge units rely on judgments from strong LLMs. For cultures with low digital presence (e.g., resource-poor minority cultures), the reliability of verification may decline.
- Experiments are primarily conducted on 7-8B scale models, leaving the effectiveness of this method on larger models unverified—larger models themselves might already possess substantial cultural knowledge.
- Culture is constantly evolving; how to continuously update cultural knowledge post-training remains an open challenge.

## Related Work & Insights

- **vs CultureLLM**: CultureLLM aligns cultural values through World Values Survey data, focusing heavily on the values dimension. CulFiT covers both cultural knowledge and cultural sensitivity, providing finer training signals via fine-grained rewards.
- **vs mGPT/BLOOM**: These multilingual models acquire multilingual capabilities through multilingual pre-training but lack explicit cultural alignment mechanisms. CulFiT adds targeted culture-aware training on top of multilingual capabilities.
- **vs Self-Rewarding LM**: Self-Rewarding models use themselves as evaluators for iterative training. CulFiT also utilizes models for critique generation, but specifically designs a fine-grained evaluation based on knowledge unit decomposition for the cultural dimension.

## Rating

- Novelty: ⭐⭐⭐⭐ The multilingual training concept based on culture-language binding and fine-grained rewards using knowledge unit decomposition are innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation on four benchmarks coupled with detailed ablation analyses provides extensive coverage.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation of problems and complete workflows of methods, although some technical details could be more elaborate.
- Value: ⭐⭐⭐⭐ Cultural alignment is an important milestone for LLM globalization. CulFiT provides a viable solution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] MIDB: Multilingual Instruction Data Booster for Enhancing Cultural Equality in Multilingual Instruction Synthesis](../../AAAI2026/multilingual_mt/midb_multilingual_instruction_data_booster_for_enhancing_cultural_equality_in_mu.md)
- [\[ACL 2025\] LACA: Improving Cross-lingual Aspect-Based Sentiment Analysis with LLM Data Augmentation](laca_crosslingual_absa.md)
- [\[ACL 2025\] EXECUTE: A Multilingual Benchmark for LLM Token Understanding](execute_a_multilingual_benchmark_for_llm_token_understanding.md)
- [\[ACL 2025\] LangSAMP: Language-Script Aware Multilingual Pretraining](langsamp_multilingual_pretraining.md)
- [\[ACL 2025\] LexGen: Domain-aware Multilingual Lexicon Generation](lexgen_domain-aware_multilingual_lexicon_generation.md)

</div>

<!-- RELATED:END -->

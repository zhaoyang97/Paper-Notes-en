---
title: >-
  [Paper Note] Contrastive Perplexity for Controlled Generation: An Application in Detoxifying Large Language Models
description: >-
  [ACL 2025][LLM (Other)][Controlled Generation] This paper proposes a framework based on Prototype Contrastive Perplexity (CP). By constructing positive and negative sample pairs that are semantically similar but possess different toxic attributes, and performing contrastive learning in the perplexity space to fine-tune LLMs, the framework achieves a significant reduction in toxicity (Mistral-7b toxicity drops from 33.1% to 4.3%) while having almost no impact on downstream tas…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Controlled Generation"
  - "Detoxification"
  - "Contrastive Learning"
  - "Perplexity"
  - "Hard Negatives"
date: 2026-05-08
content_hash: e10e709ca893d433
---

# Contrastive Perplexity for Controlled Generation: An Application in Detoxifying Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2401.08491](https://arxiv.org/abs/2401.08491)  
**Code**: [https://github.com/SAP-samples/acl2025-contrastive-perplexity/](https://github.com/SAP-samples/acl2025-contrastive-perplexity/)  
**Area**: LLM/NLP  
**Keywords**: Controlled Generation, Detoxification, Contrastive Learning, Perplexity, Hard Negatives  

## TL;DR
This paper proposes a framework based on Prototype Contrastive Perplexity (CP). By constructing positive and negative sample pairs that are semantically similar but possess different toxic attributes, and performing contrastive learning in the perplexity space to fine-tune LLMs, the framework achieves a significant reduction in toxicity (Mistral-7b toxicity drops from 33.1% to 4.3%) while having almost no impact on downstream task performance.

## Background & Motivation
**Background**: Toxicity generation by LLMs is a core challenge for safe deployment. Existing methods mainly adopt pipeline strategies—preprocessing data cleaning + regular training + post-processing filtering, or use alignment methods such as RLHF/DPO.

**Limitations of Prior Work**: (a) Data preprocessing is extremely difficult at scale and significantly reduces performance; (b) Post-processing relies on subjective heuristic rules, yielding poor scalability; (c) Existing alignment methods tend to "evade sensitive topics" rather than truly understand toxicity, leading to restricted applications for marginalized groups and an inability to defend against implicit toxicity.

**Key Challenge**: There is often only a subtle stylistic difference between toxic and non-toxic expressions (e.g., "This article is trash" vs "This article needs improvement"). What the model needs to learn is this fine-grained stylistic distinction, rather than simply refusing to answer.

**Goal**: How to make LLMs "improve vocabulary rather than raise volume"—not evading sensitive topics, but learning to express the same semantics in a non-toxic manner.

**Key Insight**: The authors observe that toxic and non-toxic expressions are highly semantically similar, enabling the construction of "hard negatives" (pairs with similar semantics but different toxicity) to perform contrastive learning using perplexity as an interpretable metric.

**Core Idea**: Generating semantically similar, toxic hard negatives through adversarial paraphrasing, and utilizing prototype-based contrastive perplexity loss to pull positive samples closer and push negative samples further away.

## Method

### Overall Architecture
The input is a dataset containing toxic text. First, prompt-based generation via an LLM is used to produce a positive sample set (non-toxic paraphrases) and a negative sample set (toxic adversarial paraphrases). Then, the target LLM is fine-tuned using the prototype-based contrastive perplexity loss. The fine-tuned model can be used in both white-box (direct usage) and black-box (as a post-processing detoxifier) scenarios.

### Key Designs

1. **Hard Negative Generation**:

    - **Function**: Leverages an uncensored LLM (Vicuna-13B uncensored) to generate semantically similar but toxic adversarial paraphrases for each non-toxic sample.
    - **Mechanism**: Generates a non-toxic paraphrase set $\mathcal{P}$ using the prompt "Paraphrase the following sentences" on positive samples, and generates a toxic paraphrase set $\mathcal{N}$ using the prompt "Paraphrase the following sentence in a very toxic way".
    - **Design Motivation**: Negative samples in traditional contrastive learning often have large semantic differences, requiring the model to only learn shallow distinctions. Hard negatives force the model to distinguish toxic attributes under highly similar semantic conditions, learning finer-grained stylistic differences.

2. **Prototype-based Contrastive Perplexity**:

    - **Function**: Conducts contrastive learning in the perplexity space, clustering the perplexity of positive samples around the prototype mean and keeping the perplexity of negative samples far from the prototype.
    - **Mechanism**: For each anchor $\bm{x}_i$, the mean perplexity of the positive sample set is calculated as the prototype $c_i = \frac{1}{|\mathcal{P}_i|}\sum_{\bm{x}\in\mathcal{P}_i}\phi(\bm{x})$, then the similarity between samples and the prototype is computed using temperature-scaled exponential similarity $s(\bm{x}, c_i) = \frac{1}{\tau}\exp(-|\phi(\bm{x}) - c_i|)$. The final score is given by $J(\bm{x}_i;\theta) = \frac{\sum_{\bm{x}\in\mathcal{P}_i} s(\bm{x},c_i)}{\sum_{\bm{x}\in\mathcal{P}_i\cup\mathcal{N}_i} w(\bm{x})s(\bm{x},c_i)}$.
    - **Design Motivation**: Utilizing perplexity instead of hidden representations offers two advantages—(a) perplexity is an interpretable measure of uncertainty; (b) basing comparisons on prototype means instead of single samples is more stable, avoiding common loss volatility issues in contrastive learning.

3. **Negative Sample Weighting Mechanism**:

    - **Function**: Adjusts the influence of negative samples on the contrastive loss via a weight hyperparameter $\alpha$.
    - **Mechanism**: The weight of positive samples is $w(\bm{x})=1$, and the weight of negative samples is $w(\bm{x})=\alpha$, where $\alpha\in\{1.0, 1.1\}$.
    - **Design Motivation**: Allows flexible control over the relative contributions of positive and negative samples to adapt to the characteristics of different models.

### Loss & Training
- Training Objective: Minimize the negative log contrastive score $\arg\min_\theta -\sum_{i=1}^N \log J(\bm{x}_i; \mathcal{A}_i, \theta)$.
- Employs LoRA + 4-bit quantization for parameter-efficient fine-tuning.
- Learning rate is $2.2e{-5}$, batch size of 2, gradient accumulation of 3, trained for 1 epoch.
- Positive sample set size $|\mathcal{P}|\in\{1,2,3,5\}$, negative sample set size $|\mathcal{N}|\in\{5,7,8\}$.

## Key Experimental Results

### Main Results

| Model | Semantic Similarity | Toxicity Rate (↓) | Toxicity Reduction |
|------|-----------|----------|---------|
| Mistral-7b (baseline) | 0.48 | 33.1% | - |
| Mistral-7b + CP | 0.40 | **4.3%** | -28.8pp |
| Llama-2-7b (baseline) | 0.84 | 76.9% | - |
| Llama-2-7b + CP | 0.24 | **11.4%** | -65.5pp |
| Falcon-7b (baseline) | 0.66 | 58.9% | - |
| Falcon-7b + CP | 0.46 | **36.6%** | -22.3pp |
| PPO (Mistral-7b) | 0.35 | 13.91% | - |
| DPO (Mistral-7b) | 0.32 | 7.35% | - |
| SimPO (Mistral-7b) | 0.46 | 28.32% | - |
| **CP (Mistral-7b)** | **0.40** | **4.34%** | Lowest |

### Ablation Study

| Configuration | Semantic Similarity | Toxicity Rate (↓) | Description |
|------|-----------|----------|------|
| Baseline | 0.48 | 33.1% | No detoxification |
| Perplexity (Positive Only) | 0.77 | 65.1% | Toxicity instead increased due to copying inputs |
| Perplexity (Negative Only) | 0.08 | 0.0% | Degenerate, outputs gibberish characters |
| CP (min: |P|=|N|=1) | 0.50 | 17.2% | Effective but unstable (±6.78) |
| CP (max: |P|=|N|=7) | 0.33 | 4.3% | Good effect but similarity slightly decreased |
| **Proposed** | **0.40** | **4.3%** | Low toxicity and stable (±1.00) |

### Key Findings
- Using only positive samples instead increases toxicity (the model learns to copy input), whereas using only negative samples leads to degeneration—both are indispensable.
- Compared to PPO/DPO/SimPO, CP not only achieves the lowest toxicity but also requires the shortest training time (PPO requires 4 times more).
- Performance degradation on downstream tasks (commonsense reasoning, reading comprehension, mathematics) is minimal (typically <1%), indicating a very low "alignment tax".
- Fluency on WikiText2 is almost unaffected (PPL increases by only 0.07).
- The method is equally effective on instruction-tuned models (Mistral-7b-Instruct), further reducing toxicity to 2.8%.

## Highlights & Insights
- **Contrastive learning in the perplexity space** is an ingenious design—perplexity is naturally a scalar, simple to calculate, and interpretable, which sidesteps the complexity of contrastive learning in high-dimensional representation spaces. This concept can be transferred to other attribute-controlled scenarios (e.g., sentiment, formality).
- **Using the target model itself to generate training data** (self-correction paradigm)—the bias originates from the model itself, making homologous data correction more targeted and highly efficient compared to external labeling.
- **Prototype means instead of single-point anchors**—avoids over-reliance on a single anchor in contrastive learning, enhancing training stability.
- **The black-box mode** is particularly practical—the small model fine-tuned via CP can act as a post-processing detoxifier for any LLM without requiring access to the target model parameters.

## Limitations & Future Work
- Detoxification effectiveness relies on the toxicity types covered by the LLM generating the training data; rare toxicity patterns might be overlooked.
- Only validated on English corpora; cross-lingual detoxification requires corresponding data and models.
- Robustness against complex adversarial prompts (jailbreak) remains unverified.
- Semantic similarity decreases somewhat after detoxification (Llama-2-7b drops from 0.84 to 0.24), indicating that the trade-off between fidelity and detoxification still needs optimization.

## Related Work & Insights
- **vs CHRT**: CHRT achieves detoxification by modifying hidden states, requiring interventions at each layer, which lacks generality; CP operates solely at the loss function level, making it more model-agnostic.
- **vs DPO/SimPO**: Preference optimization methods require paired preference data and train slower; CP employs set-level contrastive learning supporting multiple positive and negative samples, which is more efficient.
- **vs Model Arithmetic**: Conjoining attribute models during inference is flexible but less effective than CP, which internalizes attributes directly into model parameters.
- This method provides a lightweight alternative to RLHF in alignment and safety domains.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Perplexity-space contrastive learning combined with prototype means is a novel combination, though the overall framework is still a variant of contrastive learning.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Conducts white-box/black-box assessments, multi-model evaluation, ablation studies, downstream tasks, diversity analysis, and embedding space visualization.
- **Writing Quality**: ⭐⭐⭐⭐ Clear and well-organized; the Rumi quote in the introduction is interesting but slightly redundant.
- **Value**: ⭐⭐⭐⭐ Highly practical, efficient training, with substantial application value in black-box mode.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Acquisition and Application of Novel Knowledge in Large Language Models](acquisition_and_application_of_novel_knowledge_in_large_language_models.md)
- [\[ACL 2025\] Length Controlled Generation for Black-box LLMs](length_controlled_generation_for_black-box_llms.md)
- [\[ACL 2025\] An Empirical Study of Large Language Models for Automated Review Generation](an_empirical_study_of_large_language_models_for_automated_review_generation.md)
- [\[ACL 2025\] Segment-Level Diffusion: A Framework for Controllable Long-Form Generation with Diffusion Language Models](segment_level_diffusion.md)
- [\[ACL 2025\] Enhancing Input-Label Mapping in In-Context Learning with Contrastive Decoding](enhancing_input-label_mapping_in_in-context_learning_with_contrastive_decoding.md)

</div>

<!-- RELATED:END -->

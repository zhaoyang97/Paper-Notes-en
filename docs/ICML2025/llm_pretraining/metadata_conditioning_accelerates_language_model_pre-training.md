---
title: >-
  [Paper Note] Metadata Conditioning Accelerates Language Model Pre-training
description: >-
  [ICML 2025][LLM Pretraining][Pre-training Acceleration] This work proposes MeCo (Metadata Conditioning then Cooldown), which prepends metadata such as document URLs to the text during pre-training to help the model distinguish heterogeneous data sources, followed by standard data cooldown during the final 10% of training. This allows a 1.6B model to achieve comparable downstream performance using **33% less data**, while unlocking the ability to guide generation through condi…
tags:
  - "ICML 2025"
  - "LLM Pretraining"
  - "Pre-training Acceleration"
  - "Metadata Conditioning"
  - "Data Efficiency"
  - "Controllable Generation"
  - "Language Models"
date: 2026-05-08
content_hash: 134b72f9b79ca03b
---

# Metadata Conditioning Accelerates Language Model Pre-training

**Conference**: ICML 2025  
**arXiv**: [2501.01956](https://arxiv.org/abs/2501.01956)  
**Code**: [princeton-pli/MeCo](https://github.com/princeton-pli/MeCo)  
**Area**: LLM Pre-training  
**Keywords**: Pre-training Acceleration, Metadata Conditioning, Data Efficiency, Controllable Generation, Language Models

## TL;DR

This work proposes MeCo (Metadata Conditioning then Cooldown), which prepends metadata such as document URLs to the text during pre-training to help the model distinguish heterogeneous data sources, followed by standard data cooldown during the final 10% of training. This allows a 1.6B model to achieve comparable downstream performance using **33% less data**, while unlocking the ability to guide generation through conditional inference.

## Background & Motivation

Language model pre-training relies on large-scale, diverse web corpora, but existing methods treat all documents equally, ignoring key contextual signals inherent in data sources:

**Heterogeneity Problem**: Documents on the same topic (e.g., "Tim Cook") may originate from meme sites, Wikipedia, financial reports, or interviews, with vast differences in style and credibility. Treating them identically blurs the model's learning of "what behavior to produce in what context."

**Data Efficiency Bottleneck**: Standard pre-training requires massive numbers of tokens to implicitly learn to distinguish these sources, wasting significant computational resources.

**Lack of Controllability**: Once trained, the model cannot be explicitly guided at inference time to produce specific styles or reduce toxic content.

**Key Insight**: URL information in pre-training corpora serves as natural and free metadata that can directly act as "source tags" to help the model establish associations between document content and their sources.

## Method

### Overall Architecture

MeCo divides pre-training into two stages:

1. **Metadata Conditioning (first 90% of steps)**: Prepends the source URL to each document, constructing the format `URL: en.wikipedia.org\n\n[document]`. During this stage, the model learns to utilize URL signals to differentiate various types of data.
2. **Cooldown Stage (last 10% of steps)**: Strips the URLs and continues training using only standard text. This stage inherits the learning rate schedule and optimizer states from the previous stage, ensuring that the model can function normally at inference time without requiring metadata.

The entire pipeline is extremely simple: it requires no modifications to the network architecture, no extra models or classifiers, and zero computational overhead.

### Key Designs

**1. Metadata Selection and Format**

- By default, the absolute domain name of the document URL is used (e.g., `en.wikipedia.org`), which is widely available in CommonCrawl-derived datasets.
- Template: `URL: {domain}\n\n{document_text}`
- It is also compatible with other types of metadata, such as model-decoded/generated topic tags or hashed URLs (validated in ablation studies).

**2. Loss Calculation Strategy**

- Cross-entropy loss is computed only on document tokens, **not on URL/template tokens**. The authors found that calculating loss on URL tokens slightly hurts downstream performance, as the model allocates capacity to memorizing URLs rather than understanding the content.

**3. Cooldown Design**

- The cooldown stage does not reset any training states, resuming directly from the last checkpoint of the conditioned training.
- A cooldown ratio of 10%–20% yields the best results; excessively long cooldown dilutes the gains brought by metadata.

**4. Training Engineering Optimization**

- **Disabling Cross-Document Attention**: No attention is calculated between different documents, which accelerates training (25% faster for the 1.6B model) and improves downstream performance.
- **Document-Aligned Packing**: Each training sequence starts with a new document, avoiding arbitrary truncation and stitching mid-document. Although this might discard a small amount of data, it significantly improves quality.

**5. Conditional Inference**

MeCo unlocks controllability at inference time—prepending a specific URL (real or fictitious) to the prompt guides model behavior:

- `wikipedia.org` $\rightarrow$ reduces toxic content generation
- `factquizmaster.com` (fictitious) $\rightarrow$ improves commonsense QA performance
- `boards.4chan.org` $\rightarrow$ simulates low-quality/offensive styles (to validate controllability)

This approach requires no fine-tuning and takes effect directly at inference time.

### Loss & Training

- **Loss Function**: Standard autoregressive cross-entropy loss, but computed only on document tokens, with metadata tokens masked out.
- **Optimizer**: AdamW + cosine learning rate schedule, with hyperparameters following the setup in Li et al. (2024).
- **Seamless Two-Stage Transition**: The cooldown stage inherits the learning rate, model parameters, and optimizer states without any resets.
- **Architecture**: Llama-family Transformers + Llama-3 tokenizer, covering four sizes: 600M, 1.6B, 3B, and 8B.

## Key Experimental Results

### Main Results

The 1.6B model is trained on DCLM with 160B tokens, compared against standard pre-training and enhanced baselines:

| Configuration | Data Size | MMLU | ARC-e | ARC-c | CSQA | HSwag | OBQA | 10-Task Avg. |
|------|--------|------|-------|-------|------|-------|------|-------------|
| Standard | 160B | 36.1 | 75.1 | 42.7 | 64.8 | 66.7 | 46.0 | 55.7 |
| + Data sel. | 160B | 37.2 | 74.6 | 44.3 | 62.9 | 65.5 | 46.8 | 56.0 |
| + 80B tokens | 240B | 37.1 | 75.2 | 43.2 | 64.1 | 67.7 | 49.8 | 56.7 |
| **MeCo** | **160B** | 36.3 | 75.7 | 44.1 | 63.8 | 67.3 | 51.2 | **56.7** |

**Core Conclusion**: MeCo achieves the performance of standard 240B token training using only 160B tokens, **saving 33% of data and computation**.

### Ablation Study

Comparison of data mixing strategies (1.6B, DCLM 160B tokens):

| Configuration | ARC-e | ARC-c | HSwag | OBQA | 10-Task Avg. | Description |
|------|-------|-------|-------|------|-------------|------|
| 100% standard | 75.1 | 42.7 | 66.7 | 46.0 | 55.7 | Standard baseline |
| 100% URL | 72.4 | 28.8 | 61.5 | 42.6 | 50.3 | No cooldown, performance drops significantly |
| 90% URL + 10% std (mixed) | 72.5 | 43.1 | 66.9 | — | — | Full-session mixing, inferior to two-stage |
| **MeCo (two-stage)** | **75.7** | **44.1** | **67.3** | **51.2** | **56.7** | Conditioning followed by cooldown, optimal |

### Key Findings

1. **Validation Perplexity Uncorrelated with Downstream Performance**: The 240B baseline's PPL (12.9) is much lower than MeCo's (13.3), yet their downstream average performance is identical. This re-verifies that PPL is not a reliable metric for downstream task performance.

2. **Consistent Performance Gains Across Scales**: From 600M to 8B, MeCo consistently outperforms standard training, with larger models benefiting more (improvements are more pronounced at the billion-plus scale).

3. **Consistent Improvements Across Data Sources**: On all three datasets—C4, RefinedWeb, and DCLM—MeCo brings significant and consistent gains.

4. **Significant Effect of Conditional Inference**:
    - MeCo + Conditional Inference achieves 57.2 avg. (vs. 55.7 for standard training), an **absolute gain of 1.5%**.
    - Using `factmonster.com` outperforms `4chan.org` by **7.3%** on CSQA (zero-shot).

5. **Significant Reduction in Toxic Generation**: Using `wikipedia.org` conditional inference dramatically reduces toxicity scores, showing a much stronger effect in MeCo than in the standard model.

6. **Core Role of Metadata Is Grouping**: Ablation studies with hashed URLs and model-generated topics indicate that the value of metadata lies in clustering similar documents rather than the semantic information of the URLs themselves.

## Highlights & Insights

1. **Extreme Simplicity**: The implementation of MeCo may require only dozens of lines of code modifications in the data processing pipeline—appending the URL string before each document and stripping it for the final 10%. It requires no architectural changes, no extra models, and no hyperparameter tuning.

2. **Free Training Signals**: URL information already exists in CommonCrawl-derived datasets. MeCo transforms this "discarded metadata" into valuable training signals, representing a genuine "free lunch."

3. **Novelty of Conditional Inference**: Guiding model behavior at inference time through fictitious URLs introduces a new paradigm for fine-tuning-free controllable generation. The fact that a fictional domain like `factquizmaster.com` works indicates the model has learned the association between URL styles and content types.

4. **Criticality of Cooldown**: Without the cooldown phase, model performance drops significantly at inference time (10-Task Avg. from 56.7 to 50.3), demonstrating that the model develops a dependency on metadata. The cooldown is key to "withdrawing" from this dependency.

5. **Cautionary Tale for PPL Metrics**: The empirical findings serve as a reminder to the community that a lower PPL does not equate to improved downstream task performance; evaluating pre-training quality should prioritize downstream benchmarks.

## Limitations & Future Work

1. **Dependency on CommonCrawl for URL Metadata**: For non-web corpora (such as books or code), natural URL information is absent. While model-generated topics can serve as a substitute, their effectiveness and generalizability require wider validation.

2. **Single-Run Results**: Due to computational resource constraints, each experiment was executed only once. Although the authors argue in the appendix that variance is low, statistical significance across multiple runs is still lacking.

3. **Lack of Systematic Methods for URL Selection in Conditional Inference**: Currently, URLs are manually crafted for each task without an automated search or optimization method.

4. **Evaluation Limited to Base Models**: Compatibility with instruction tuning (SFT/RLHF) and effectiveness on chat models have not yet been explored.

5. **Sensitivity of Cooldown Proportion**: The 10%–20% range is an empirical value; the optimal proportion may vary across different model scales and dataset sizes.

## Related Work & Insights

- **CTRL (Keskar et al., 2019)**: Employs control codes to guide generation styles; MeCo can be viewed as a generalization of this concept applied to the pre-training stage.
- **Conditional Training (Korbak et al., 2023a)**: Uses conditional training to reduce toxic content; MeCo achieves a similar effect in a simpler manner.
- **Allen-Zhu & Li (2024)**: Studies the influence of metadata on knowledge memorization in synthetic setups; MeCo scales this concept to real-world pre-training scenarios for the first time.
- **DCLM (Li et al., 2024)**: MeCo is validated on DCLM data and is complementary to data selection methods, allowing them to be applied jointly.

**Insights**: Extensive unexploited metadata exists in pre-training corpora (e.g., timestamps, language tags, content classifiers); the MeCo framework offers a unified paradigm for leveraging such information.

## Rating

| Dimension | Score (1-5) | Description |
|------|-----------|------|
| Novelty | 4 | Simple idea but impressive results, turning "ignored metadata" into a pre-training accelerator |
| Technical Depth | 3 | The method is minimal, but the ablation studies are thorough |
| Experimental Thoroughness | 5 | Spans scales (600M-8B), data sources (C4/RefinedWeb/DCLM), and multi-dimensional ablations |
| Practicality | 5 | Zero overhead, implemented in a few lines of code, plug-and-play |
| Writing Quality | 4 | Clear structure, rich diagrams, and core information is well-conveyed |
| **Overall** | **4.2** | Simple and efficient method with high engineering value, worth trying in any pre-training pipeline |

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] KoCo: Conditioning Language Model Pre-training on Knowledge Coordinates](../../ACL2026/llm_pretraining/koco_conditioning_language_model_pre-training_on_knowledge_coordinates.md)
- [\[ICML 2025\] Large Language Models are Demonstration Pre-Selectors for Themselves](large_language_models_are_demonstration_pre-selectors_for_themselves.md)
- [\[ICML 2025\] Chameleon: A Flexible Data-mixing Framework for Language Model Pretraining and Finetuning](chameleon_a_flexible_data-mixing_framework_for_language_model_pretraining_and_fi.md)
- [\[ACL 2025\] AsyncLM: Efficient and Adaptive Async Pre-training of Language Models](../../ACL2025/llm_pretraining/asynclm_efficient_and_adaptive_async_pre-training_of_language_models.md)
- [\[NeurIPS 2025\] Nemotron-CLIMB: CLustering-based Iterative Data Mixture Bootstrapping for Language Model Pre-training](../../NeurIPS2025/llm_pretraining/nemotron-climb_clustering-based_iterative_data_mixture_bootstrapping_for_languag.md)

</div>

<!-- RELATED:END -->

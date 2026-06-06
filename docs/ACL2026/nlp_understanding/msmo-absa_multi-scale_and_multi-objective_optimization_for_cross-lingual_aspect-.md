---
title: >-
  [Paper Note] MSMO-ABSA: Multi-Scale and Multi-Objective Optimization for Cross-Lingual Aspect-Based Sentiment Analysis
description: >-
  [ACL 2026][NLP Understanding][Cross-lingual ABSA] The MSMO framework is proposed for cross-lingual aspect-based sentiment analysis. It employs sentence-level Wasserstein adversarial training with code-switched data for l…
tags:
  - "ACL 2026"
  - "NLP Understanding"
  - "Cross-lingual ABSA"
  - "Adversarial Training"
  - "Consistency Training"
  - "Code-switching"
  - "Multi-objective Optimization"
date: 2026-05-08
content_hash: e988f98a28c35d19
---

# MSMO-ABSA: Multi-Scale and Multi-Objective Optimization for Cross-Lingual Aspect-Based Sentiment Analysis

**Conference**: ACL 2026  
**arXiv**: [2502.13718](https://arxiv.org/abs/2502.13718)  
**Code**: https://github.com/swaggy66/MSMO  
**Area**: Multilingual / Cross-lingual ABSA / Sentiment Analysis  
**Keywords**: Cross-lingual ABSA, Adversarial Training, Consistency Training, Code-switching, Multi-objective Optimization

## TL;DR
The MSMO framework is proposed for cross-lingual aspect-based sentiment analysis. It employs sentence-level Wasserstein adversarial training with code-switched data for language discriminator alignment, and aspect-level bidirectional KL consistency training to align prediction distributions of aspects with the same sentiment. Combined with multi-teacher knowledge distillation, it achieves a new SOTA on four target languages in SemEval-2016 using mBERT/XLM-R, significantly outperforming LLM solutions such as GPT-4o and Qwen2.5-7B-LoRA.

## Background & Motivation

**Background**: ABSA (joint task of aspect extraction and sentiment classification) is mature in English, but the coexistence of multiple languages in real social scenarios makes cross-lingual ABSA (XABSA) urgent. Since low-resource languages lack manual annotations, mainstream approaches include: (1) translation-based (e.g., BILINGUAL-TA); (2) code-switch (e.g., ACS, which replaces aspect terms with the target language); (3) contrastive learning (CL-XABSA).

**Limitations of Prior Work**: (1) Alignment in existing XABSA methods is primarily performed at the sentence or word-embedding levels, leaving the alignment of the aspect terms themselves (the core anchors of the task) relatively coarse. (2) Single-objective optimization (e.g., pure CE or pure contrastive) tends to bias the model toward the source language's semantic space, overlooking fine-grained cross-lingual aspect correspondences. (3) Adversarial training in NLP often utilizes Domain-Adversarial GRL, which suffers from poor stability in XABSA and fails to leverage perturbation signals from code-switching.

**Key Challenge**: Sentence-level alignment ensures proximity of the overall distribution, but the "fine-grained semantics" of the same aspect in different languages and contexts remain unstable. Conversely, focusing solely on aspect-level alignment ignores linguistic style differences across entire sentences. Both scales must be aligned **simultaneously**, yet current XABSA methods typically select only one.

**Goal**: Design a unified framework of "multi-scale (sentence + aspect) × multi-objective (supervised + consistency)" where code-switched data drives alignment at both granularities, extending it to multilingual settings with added knowledge distillation.

**Key Insight**: The authors observe that code-switched data, which perturbs aspect terms within a sentence, allows a language discriminator to learn invariant features by "ignoring the aspect and focusing on language style" (sentence level). Simultaneously, a consistency module can learn that "different languages for the same aspect and sentiment should yield consistent prediction distributions" (aspect level). The same data can drive training objectives for both granularities.

**Core Idea**: Use code-switched bilingual data as "perturbation anchors" to simultaneously drive (i) Wasserstein distance adversarial training (where the sentence-level language discriminator learns aspect-agnostic language features) and (ii) bidirectional KL consistency training (aligning aspect-level cross-lingual prediction distributions). Both are jointly updated via a shared multilingual encoder.

## Method

### Overall Architecture
MSMO follows a two-stage sequential training process (Figure 2):
(1) **Stage 1: Sentence-level Adversarial Alignment**: A multilingual encoder $M$ (mBERT or XLM-R) extracts token representations $h_i$, which are fed into a language discriminator $Q$ (a binary classifier with sigmoid + Gradient Reversal Layer, GRL). Training uses Wasserstein distance + 1-Lipschitz constraint (parameters clipped to $[-c, c]$) to distinguish between source and target languages. Backpropagation forces $M$ to learn language-invariant features. Inputs include $D_S \cup D_{S_T}$ (source language + source sentences with target aspects) and $D_T \cup D_{T_S}$ (translated target language + reverse code-switch).
(2) **Stage 2: Aspect-level Consistency + Multi-objective Optimization**: Using the updated $M$ from Stage 1, the process splits into two paths: (a) a sentiment classifier $P$ performing BIES-{POS, NEU, NEG} sequence labeling via standard CE loss; (b) a consistency module $C$ applying bidirectional KL divergence $\mathcal{L}_{\text{cons}}$ to align prediction distributions between (source span $s$, code-switch span $s'$). Span probability is defined as the product of its constituent token probabilities. The final joint loss is $\mathcal{L}_{\text{total}} = \sum \mathcal{L}_{\text{CE}} + \beta \sum \mathcal{L}_{\text{cons}}$, where $\beta$ is set to $\{4.5, 2.5, 2.5, 3.5\} \times 10^{-4}$ (mBERT) / $\{2.5, 1.5, 1.5, 3.5\} \times 10^{-3}$ (XLM-R) depending on the target language.
**Extension: Distillation**: The trained MSMO model serves as a teacher to generate soft labels for unlabeled target language data, followed by single-teacher, multi-teacher, or multilingual distillation.

### Key Designs

1. **Code-switched Bilingual Data-driven Wasserstein Adversarial (sentence-level)**:
    - **Function**: Enables the multilingual encoder to learn language-invariant sentence-level representations by ignoring differences in aspect terms.
    - **Mechanism**: Constructs four types of data $D_S / D_T / D_{S_T} / D_{T_S}$—the first two are source and translated target languages, and the latter two are mixed sentences after aspect-swapping. $Q$ is required to classify $D_S \cup D_{S_T}$ as source and $D_T \cup D_{T_S}$ as target. The objective is $J_q = \max_{\theta_q} \mathbb{E}[Q(P(h_i))] - \mathbb{E}[Q(P(h_i'))]$. Through GRL, the feature extractor is forced to learn invariant features that the discriminator cannot distinguish.
    - **Design Motivation**: Typical ADAN only uses bilingual parallel data, lacking aspect-level perturbations. By introducing $D_{S_T}/D_{T_S}$, the discriminator must learn to judge the language based on sentence style regardless of the aspect's language, forcing truly aspect-agnostic invariant features and smoothing the path for subsequent aspect-level alignment.

2. **Bidirectional KL Consistency Training (aspect-level)**:
    - **Function**: Ensures the model yields nearly identical sentiment distributions for the same aspect term across different languages.
    - **Mechanism**: A source sentence $X$ undergoes transformation $\phi$ (translation or aspect swap) to produce $X'$, with corresponding aspect spans $(s, s')$. The span probability is the product of token log-probabilities, aligned via bidirectional KL: $\mathcal{L}_{\text{cons}} = \frac{1}{m} \sum \frac{1}{2}[\mathrm{KL}(P(y'|s') \| P(y|s)) + \mathrm{KL}(P(y|s) \| P(y'|s'))]$.
    - **Design Motivation**: Sentence-level alignment only brings the overall distributions closer. Fine-grained correspondences (e.g., "food" and "nourriture" both being POS) require direct constraints on span probability distributions. Symmetrizing KL and calculating on spans rather than tokens avoids token misalignment issues within BIES label sequences.

3. **Multi-teacher / Multilingual Knowledge Distillation (unlabeled data amplifier)**:
    - **Function**: Distills "soft knowledge" from multiple MSMO teacher models into a single student and leverages unlabeled target language text for further improvement.
    - **Mechanism**: Three teachers (from different source languages or random seeds) predict on unlabeled target language text to generate soft labels $p_t = \sum_{k=1}^{3} w_k g_{t_k}$ ($w_k = 1/3$). The student, retaining only the encoder and sentiment classifier, learns via MSE loss $\mathcal{L}_{KD} = \frac{1}{|D_{NL}|} \sum \frac{1}{L} \sum_i \mathrm{MSE}(p_{t_i}, p_{s_i})$.
    - **Design Motivation**: MSMO teachers are stronger than CL-XABSA teachers, providing more accurate soft labels. Multi-teacher distillation combines strengths from different language pairs and mitigates single-teacher overfitting bias; experiments show multi-teacher consistently outperforms single-teacher.

### Loss & Training
- **Stage 1**: $J_q$ (Wasserstein adversarial, $Q$ parameters clipped to $[-c, c]$ to ensure 1-Lipschitz); GRL coefficient $\lambda = 1$.
- **Stage 2**: $\mathcal{L}_{\text{total}} = \sum \mathcal{L}_{\text{CE}} + \beta \sum \mathcal{L}_{\text{cons}}$, where $\beta$ is determined via grid search.
- **Hyperparameters**: mBERT lr=5e-5 / bs=16 / 2000 steps; XLM-R lr=2e-5 / bs=8 / 2500 steps; Stage 1 GPU memory ≈ 27 GB (XLM-R); averaged over 5 random seeds.
- **Distillation Phase**: Student initialized with training on translated target language, followed by MSE distillation on unlabeled target data.

## Key Experimental Results

### Main Results
SemEval-2016 ABSA, Source = English, Target = FR/ES/NL/RU, Evaluated by Micro-F1:

| Method | FR | ES | NL | RU | Avg (mBERT) | Avg (XLM-R) |
|------|----|----|----|----|-------------|-------------|
| Zero-shot baseline | 45.60 / 56.43 | 57.32 / 67.10 | 42.68 / 59.03 | 36.01 / 56.80 | 45.40 | 59.84 |
| Translation-TA (Li et al. 2021) | 40.76 / 47.00 | 50.74 / 58.10 | 47.13 / 56.19 | 41.67 / 50.34 | 45.08 | 52.91 |
| ACS (Zhang et al. 2021a) | 49.65 / 59.39 | 59.99 / 67.32 | 51.19 / 62.83 | 52.09 / 60.81 | 53.23 | 62.59 |
| CL-XABSA (TL, Lin et al. 2023) | 50.55 / 59.47 | 60.09 / 64.63 | 52.45 / 59.40 | 50.73 / 61.13 | 53.46 | 61.16 |
| Equi-XABSA (Lin et al. 2024) | 50.08 / 60.68 | 63.08 / 69.56 | 51.85 / 61.31 | 52.59 / 62.34 | 54.40 | 63.47 |
| **MSMO (Ours)** | **51.42 / 61.01** | **63.26 / 69.74** | **52.68 / 63.26** | **53.45 / 62.52** | **55.20** | **64.13** |
| ACS-Distill-M | 52.25 / 59.90 | 62.91 / 69.24 | 53.40 / 63.74 | 54.58 / 62.02 | 55.79 | 63.73 |
| CL-XABSA-Distill-M | 52.99 / 62.10 | 63.54 / 69.37 | 53.52 / 64.27 | 53.98 / 62.29 | 56.01 | 64.51 |
| **MSMO-Distill-M (Ours)** | **54.39 / 63.89** | **64.59 / 69.93** | **54.14 / 65.15** | **54.89 / 63.20** | **56.94** | **65.54** |
| Supervised (Oracle) | 61.80 / 67.44 | 67.88 / 71.93 | 56.80 / 64.28 | 58.87 / 64.93 | 61.34 | 67.15 |

**vs LLM zero-shot / LoRA**:

| LLM Config | FR | ES | NL | RU | Avg |
|----------|----|----|----|----|-----|
| GPT-4o (zero-shot) | 48.43 | 49.91 | 49.94 | 45.15 | 48.36 |
| Qwen2.5-7B + LoRA | 63.01 | 68.95 | 60.84 | 53.50 | 61.58 |
| **MTL-MSMO-Distill (XLM-R)** | **63.23** | **70.95** | **66.24** | **64.36** | **66.20** |

### Ablation Study

| Config | FR | ES | NL | RU | Avg (mBERT) | Avg (XLM-R) |
|------|----|----|----|----|-------------|-------------|
| MSMO (full) | 51.42 / 61.01 | 63.26 / 69.47 | 52.68 / 63.26 | 53.45 / 62.52 | 55.20 | 64.13 |
| w/o Language Discriminator | 49.70 / 59.82 | 60.61 / 68.10 | 51.57 / 62.41 | 52.21 / 60.99 | 53.52 (-1.68) | 62.83 (-1.30) |
| w/o Consistency Training | 50.59 / 59.51 | 60.40 / 67.96 | 51.30 / 62.91 | 52.25 / 61.82 | 53.63 (-1.57) | 63.05 (-1.08) |

$\beta$ Sensitivity: If too small, it degrades to pure supervised training with poor cross-lingual generalization. If too large, consistency dominates, and language specificity is lost. Spanish achieves optimality at a smaller $\beta$ as it is linguistically closer to English.

### Key Findings
- **Both modules are indispensable**: Removing either the language discriminator or consistency training results in a drop of 1-1.7 F1 points. Discriminator contribution is slightly larger, suggesting that aligning global language distribution before fine-grained aspect alignment is the correct order.
- **XLM-R consistently outperforms mBERT**: Attributed to XLM-R's larger cross-lingual pre-training. MSMO's relative gains are consistent across backbones, indicating it is backbone-agnostic.
- **Spanish shows the largest gain (+3.14% mBERT / +4.89% XLM-R vs CL-XABSA)**: Since ES and EN are both Indo-European, their semantic spaces are more easily aligned, allowing MSMO's fine-grained alignment to exert more leverage.
- **Multi-teacher > Single-teacher**: Multi-teacher distillation generally adds 0.5-1.0 F1 in the distillation phase, as averaged soft labels mitigate individual teacher bias.
- **MSMO outperfoms LLMs**: MTL-MSMO-Distill on XLM-R (66.20) significantly beats Qwen2.5-7B-LoRA (61.58) and GPT-4o zero-shot (48.36), verifying that specialized fine-tuned small models still outperform general LLMs on token-level labeling tasks.
- **MSMO approaches supervised upper bound**: MSMO-Distill-M (XLM-R) achieves an average of 65.54 compared to the Supervised 67.15, nearly matching performance of full target language labeling.

## Highlights & Insights
- **Code-switched data driving dual-granularity alignment is a clean multi-task design**: The same perturbation data facilitates invariant feature learning for the discriminator and distribution alignment for the consistency module. This "one data, two losses" reuse strategy is highly valuable for low-resource NLP.
- **Wasserstein + 1-Lipschitz replaces standard GAN adversarial training**: Avoids the instability of ADAN-GRL; parameter clipping is simpler than gradient penalty and suitable for small-data sequence labeling tasks.
- **Span probability as a product of token probabilities**: Treating ABSA aspects as span-level units allows KL divergence to operate on aspect units rather than tokens, avoiding token-order issues in BIES sequences and successfully merging sequence labeling with distribution alignment.
- **Small models with task-specific methods still beat large LLMs**: MTL-MSMO-Distill on XLM-R (66.20) vs. Qwen2.5-7B-LoRA (61.58) serves as a reminder that structured output tasks (especially BIES tagging) should not rely solely on LLMs; small models with specific methods and multi-teacher distillation remain cost-effective.

## Limitations & Future Work
- **Ours**: (1) Cross-lingual aspect alignment remains weak against highly idiomatic expressions (slang, cultural references) because code-switching assumes one-to-one aspect translation. (2) Validation was limited to four target languages within SemEval-2016; broader generalizability is unknown.
- **Observations**: (1) Reliance on machine translation for $D_T$ limits the ceiling; this is hard to replicate for extremely low-resource languages. (2) $\beta$ must be tuned per language, which is a burden for deployment; adaptive $\beta$ was not explored. (3) Product of probabilities assumes token independence; long aspects might result in exponentially smaller probabilities, causing KL signals to be dominated by a few tokens. (4) Multi-teacher training triples computational costs compared to a single teacher; no detailed cost-gain analysis was provided. (5) Comparisons with LLMs only targeted general-purpose models without exploring strictly constrained instruction-based prompt strategies for BIO output, potentially overestimating LLM disadvantages.
- **Improvement Ideas**: Replace hard code-switching with "soft token swap" mixup to handle untranslatable aspects; introduce learnable scalars or per-language adaptive mechanisms for $\beta$; explore GRPO optimization for span-level F1 on LLMs to see if they can overtake MSMO.

## Related Work & Insights
- **vs ACS (Zhang et al. 2021a)**: ACS introduced aspect code-switch but used only single-scale alignment; MSMO adds Wasserstein adversarial and KL consistency to let code-switched data drive two granularities.
- **vs CL-XABSA (Lin et al. 2023)**: CL-XABSA uses contrastive learning at token and sentiment levels; MSMO uses adversarial and KL consistency plus sentence-level alignment, winning by 0.5-2 F1 across languages.
- **vs Equi-XABSA (Lin et al. 2024)**: That method focuses on class imbalance and language representation differences; MSMO focuses on dual-granularity alignment. The two are complementary but were not combined here.
- **vs Consistency training in NER (Zhou et al. 2022, ConNER)**: ConNER uses token-level consistency in cross-lingual NER; MSMO adapts this to span units for ABSA, representing the first systematic implementation for this task.

## Rating
- Novelty: ⭐⭐⭐⭐ The clean multi-task design of "dual-granularity alignment + code-switch shared data" is a natural and effective fusion.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 4 languages, 2 backbones, 7 baselines, 3 distillation modes, 5 LLM comparisons, ablation studies, and $\beta$ sensitivity.
- Writing Quality: ⭐⭐⭐⭐ Formulas, data flow, and the two-stage training sequence are presented clearly.
- Value: ⭐⭐⭐⭐ Provides a backbone-agnostic framework for the cross-lingual sequence labeling community; MSMO-Distill approaches the supervised upper bound while being open-source.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] DimABSA: Building Multilingual and Multidomain Datasets for Dimensional Aspect-Based Sentiment Analysis](dimabsa_building_multilingual_and_multidomain_datasets_for_dimensional_aspect-ba.md)
- [\[ACL 2026\] MADE: A Living Benchmark for Multi-Label Text Classification with Uncertainty Quantification](made_a_living_benchmark_for_multi-label_text_classification_with_uncertainty_qua.md)
- [\[ACL 2026\] MTSQL-R1: Towards Long-Horizon Multi-Turn Text-to-SQL via Agentic Training](mtsql-r1_towards_long-horizon_multi-turn_text-to-sql_via_agentic_training.md)
- [\[ACL 2026\] HCRE: LLM-based Hierarchical Classification for Cross-Document Relation Extraction](hcre_llm-based_hierarchical_classification_for_cross-document_relation_extractio.md)
- [\[ACL 2026\] A Computational Method for Measuring "Open Codes" in Qualitative Analysis](a_computational_method_for_measuring_34open_codes34_in_qualitative_analysis.md)

</div>

<!-- RELATED:END -->

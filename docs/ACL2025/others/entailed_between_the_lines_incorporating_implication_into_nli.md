---
title: >-
  [Paper Note] Entailed Between the Lines: Incorporating Implication into NLI
description: >-
  [ACL 2025][Implied Entailment] This work formalizes the task of "implied entailment," expanding the traditional three-way classification of NLI into a four-way classification (implied entailment/explicit entailment/neutral/contradiction). It constructs the INLI dataset, which comprises 10K premises and 40K hypotheses. Experiments demonstrate that fine-tuned models can effectively identify implied entailment and generalize across domains.
tags:
  - "ACL 2025"
  - "Implied Entailment"
  - "NLI"
  - "Pragmatic Inference"
  - "Explicit vs Implied Entailment"
  - "Four-Way NLI"
  - "INLI Dataset"
date: 2026-05-08
content_hash: af45dea89918da2e
---

# Entailed Between the Lines: Incorporating Implication into NLI

**Conference**: ACL 2025  
**arXiv**: [2501.07719](https://arxiv.org/abs/2501.07719)  
**Code**: [https://github.com/google-deepmind/inli](https://github.com/google-deepmind/inli)  
**Authors**: Shreya Havaldar, Hamidreza Alvari, John Palowitch, Mohammad Javad Hosseini, Senaka Buthpitiya, Alex Fabrikant  
**Institutions**: University of Pennsylvania, Google DeepMind  
**Area**: Others  
**Keywords**: Implied Entailment, NLI, Pragmatic Inference, Explicit vs Implied Entailment, Four-Way NLI, INLI Dataset  

## TL;DR

This work formalizes the task of "implied entailment," expanding the traditional three-way classification of NLI into a four-way classification (implied entailment/explicit entailment/neutral/contradiction). It constructs the INLI dataset, which comprises 10K premises and 40K hypotheses. Experiments demonstrate that fine-tuned models can effectively identify implied entailment and generalize across domains.

## Background & Motivation

**Implied Information in Language**: Human communication relies heavily on implied expressions—emotions, social signals, sarcasm, etc., are often conveyed implicitly rather than stated explicitly. For example, "After reading ARR review comments, Kim had to go eat a massive slice of cheesecake." Beyond the literal meaning, readers can infer that (c) Kim found reading the reviews unpleasant, (d) Kim indeed went to eat cheesecake, and (e) Kim ate an unusually large amount of cheesecake.

**Limitations of Prior Work**: Existing NLI benchmarks (SNLI, MNLI, ANLI, WANLI) contain very little implied entailment—only 9.33% in SNLI, 3.68% in MNLI, and 5.48% in WANLI. The only exception is the adversarial ANLI (15.66%), which suggests that implicit reasoning remains the most challenging part for models to handle.

**Key Challenge**: Models trained on existing NLI datasets achieve only about 50% accuracy (random guess level) in reasoning about implied entailment, whereas their accuracy on explicit entailment can exceed 90%.

**Goal**: An NLI dataset specifically focused on implied entailment is needed to help models learn to "read between the lines" and distinguish between explicit and implied entailments.

## Method

### 2.1 Formalizing Implied Entailment

Based on traditional three-way NLI (entailment/neutral/contradiction), "entailment" is further subdivided into two categories:

- **Explicit Entailment**: Derived directly from lexical semantics (synonyms, paraphrase) and syntax (pronominal coreference, conjunctions, etc.) of the text.
- **Implied Entailment**: Requires extra cognitive steps, such as logical reasoning, world knowledge, conversational pragmatics, or figurative language understanding.

**Four-Way Labels**: Implied Entailment / Explicit Entailment / Neutral / Contradiction

### 2.2 INLI Dataset Construction

The dataset construction consists of two core stages:

**Stage 1: Implicature Augmentation**

Implicature frames were extracted from four existing datasets:

| Dataset | Implicature Frame | Sample Count |
|--------|-----------|--------|
| Ludwig | Question → Indirect Answer → Implied Meaning | 1,956 |
| Circa | Dialogue Context → Question → Indirect Answer → Implied Meaning | 18,044 |
| NormBank | Action → Situational Context → Implied Social Norm | 10,000 |
| SocialChem | Social Situation → Implied Rule of Thumb | 10,000 |

For **conversational implicatures** (Ludwig, Circa): Templates and randomized pseudonyms were used to simulate dialogue scenarios, prompting Gemini-Pro to generate implied entailment hypotheses from the implicit meanings in indirect answers.

For **situational implicatures** (NormBank, SocialChem): Premises were generated from actions or social situations, then Gemini-Pro was prompted to generate implied entailments based on social norms.

**Stage 2: Alternative Hypothesis Generation**

Three additional hypotheses (explicit entailment, neutral, contradiction) were generated for each premise-implied entailment pair:
- Starting from the implied entailment, necessary words or phrases were replaced to convert it into hypotheses of other classes.
- The four classes of hypotheses were designed to be semantically close to increase classification difficulty.
- Finally, all generated hypotheses were paraphrased to minimize artifacts of generation in the data.

**Final Scale**: $\approx 10K$ premises $\times$ 4 hypotheses = **40K premise-hypothesis pairs**

### 2.3 Data Quality Verification

- **Hypothesis-Only Test**: Models trained only on hypotheses (without seeing premises) achieved accuracy comparable to other NLI benchmarks, indicating no significant annotation bias.
- **Human Annotation Verification**: Six authors annotated 200 samples.
    - Fleiss' $\kappa = 0.711$ (comparable to ANLI's 0.679–0.721 and WANLI's 0.60)
    - Majority agreement rate = 0.935 (at least 2/3 of annotators agreed with the INLI label)

## Experiments

### Main Results: Benchmarking LLMs on INLI

| Model | Overall Accuracy | Implied Entailment Accuracy |
|------|-----------|--------------|
| T5-Small (Fine-tuned) | 0.813 | 0.731 |
| T5-Base (Fine-tuned) | 0.871 | 0.817 |
| T5-Large (Fine-tuned) | 0.913 | 0.870 |
| T5-XXL (Fine-tuned) | **0.924** | **0.885** |
| GPT-4o (8-shot) | 0.749 | 0.608 |
| GPT-4 (8-shot) | 0.753 | 0.645 |
| Claude-3-Sonnet (8-shot) | 0.686 | 0.738 |
| Gemini-Pro (8-shot) | 0.770 | 0.628 |

**Key Findings**:
1. All models perform worse on implied entailments compared to overall accuracy—even T5-XXL reaches only 0.885 (human upper bound is around 0.94).
2. Few-shot performance of large LLMs is surprisingly worse than fine-tuned smaller models—GPT-4o's implied entailment accuracy is only 0.608.
3. Even though Gemini-Pro is the model used for dataset construction, its performance on INLI is also poor (0.628), indicating that generation does not equal understanding.

### Compatibility with Existing NLI Benchmarks

| Training Data | Standard NLI Accuracy | 3-way INLI Accuracy |
|---------|-------------|----------------|
| SNLI | 0.934 | 0.921 |
| MNLI | 0.916 | 0.914 |
| ANLI | 0.725 | **0.734** |
| WANLI | 0.825 | 0.822 |
| 3-way INLI | 0.778 | 0.909 |

After fine-tuning on INLI, the model's performance on traditional NLI benchmarks remains basically unchanged, and even slightly improves on ANLI (0.725 $\rightarrow$ 0.734), showing that implied reasoning capability helps resolve difficult samples in ANLI.

### Generalization Experiments

| Experiment Type | Training Set | Test Set | Accuracy |
|---------|-------|-------|--------|
| In-domain Generalization | NormBank | SocialChem | 0.795 |
| In-domain Generalization | SocialChem | NormBank | 0.850 |
| Cross-domain Generalization | Conversational | Situational | 0.695 |
| Cross-domain Generalization | Situational | Conversational | 0.796 |
| Cross-dataset | Other 3 | SocialChem | 0.804 |
| Cross-dataset | Other 3 | NormBank | 0.851 |

**Important Finding**: Models fine-tuned on the other three datasets without ever seeing NormBank achieve an accuracy of 0.851 on NormBank, which outperforms the few-shot performance of GPT-4 and Claude-3—indicating that INLI training helps models acquire transferable implied reasoning capabilities.

### Proportion of Implied Entailment in Existing NLI Benchmarks

| Dataset | Proportion of Implied Entailment |
|--------|------------|
| SNLI | 9.33% |
| MNLI | 3.68% |
| ANLI | 15.66% |
| WANLI | 5.48% |

Verification method: A T5-XXL model was trained on INLI to distinguish between explicit/implied entailment (97.3% accuracy) and then applied to other benchmarks. Human validation showed that 92.0% of the model outputs agreed with the annotators (Cohen's $\kappa = 0.768$).

## Highlights & Insights

1. **Formalizing Implied Entailment**: This is the first work to subdivide entailment into explicit and implied within the NLI framework, filling a gap of natural language inference in pragmatic understanding.
2. **Clever Data Construction Strategy**: Instead of crowdsourcing annotations from scratch, existing implicature datasets (Ludwig, Circa, NormBank, SocialChem) are augmented via LLMs to convert into NLI format, yielding lower costs, higher quality, and stronger reproducibility.
3. **Generation $\neq$ Understanding**: Gemini-Pro is used to construct the dataset, but its own implied entailment accuracy on INLI is only 0.628, demonstrating that being able to generate implicatures does not mean understanding them.
4. **Fine-tuning Small Models Beats Prompting Large Models**: Fine-tuned T5-XXL (0.885) far outperforms GPT-4o 8-shot (0.608), highlighting the importance of specialized training.
5. **Compatibility with Existing NLI Capabilities**: Fine-tuning on INLI does not harm the model's performance on traditional NLI tasks.

## Limitations & Future Work

1. The dataset focuses on the situational and conversational domains, which may limit generalization to formal text (e.g., medical, legal).
2. Implicature comprehension is subjective; people from other cultural backgrounds might interpret the same premise differently.
3. The dataset is generated by an LLM (Gemini-Pro), which may suffer from generation biases and limited diversity.
4. Full-scale human verification was not performed, leaving possibilities of error in some samples.

## Related Work & Insights

- **Structured Implicatures**: Indirect question-answering (Ludwig, Circa), scalar implicatures (Jeretic et al.), pairwise entity selection (Hosseini et al.)—limited by fixed input structures.
- **Implicature Frameworks**: NormBank (social norms), SocialChem (social guidelines), cultural norms (Rai et al.)—providing implicature content but not in NLI format.
- **Implicature Understanding**: Measuring LLMs' implicature understanding through CoT, explanation generation, human comparisons, etc., with mixed findings.
- **Commonsense NLI**: HellaSwag, PIQA, etc. focus on physical/temporal commonsense, but do not differentiate between explicit and implied information.

## Rating ⭐⭐⭐⭐

- **Novelty**: ⭐⭐⭐⭐⭐ — Formalizes the implied entailment task, expands the NLI classification taxonomy, presenting a novel angle with theoretical depth.
- **Value**: ⭐⭐⭐⭐ — Provides directly applicable training resources to improve LLM pragmatic understanding.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Benchmarking, compatibility validation, multi-dimensional generalization experiments: a complete system.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Clear structure, well-motivated, and rich in examples.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] The Time Scale of Redundancy between Prosody and Linguistic Context](the_time_scale_of_redundancy_between_prosody_and_linguistic_context.md)
- [\[CVPR 2025\] Bounds on Agreement between Subjective and Objective Measurements](../../CVPR2025/others/bounds_on_agreement_between_subjective_and_objective_measurements.md)
- [\[CVPR 2026\] Region-Wise Correspondence Prediction between Manga Line Art Images](../../CVPR2026/others/region-wise_correspondence_prediction_between_manga_line_art_images.md)
- [\[AAAI 2026\] Human Cognitive Biases in Explanation-based Interaction: The Case of Within and Between Session Order Effect](../../AAAI2026/others/human_cognitive_biases_in_explanation-based_interaction_the_case_of_within_and_b.md)
- [\[CVPR 2026\] What Is the Optimal Ranking Score Between Precision and Recall? We Can Always Find It and It Is Rarely $F_1$](../../CVPR2026/others/what_is_the_optimal_ranking_score_between_precision_and_recall_we_can_always_fin.md)

</div>

<!-- RELATED:END -->

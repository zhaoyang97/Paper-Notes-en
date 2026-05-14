---
title: >-
  [Paper Note] Mitigating Extrinsic Gender Bias for Bangla Classification Tasks
description: >-
  [ACL 2026][Multilingual & Machine Translation][gender bias mitigation] To address extrinsic gender bias in pretrained language models applied to Bangla downstream classification tasks, this paper proposes RandSymKL…
tags:
  - "ACL 2026"
  - "Multilingual & Machine Translation"
  - "gender bias mitigation"
  - "Bangla NLP"
  - "KL divergence regularization"
  - "counterfactual data augmentation"
  - "classification fairness"
date: 2026-05-08
content_hash: 1d80c77cef3dce1b
---

# Mitigating Extrinsic Gender Bias for Bangla Classification Tasks

**Conference**: ACL 2026
**arXiv**: [2411.10636](https://arxiv.org/abs/2411.10636)
**Code**: [GitHub](https://github.com/sajib-kumar/Mitigating-Bangla-Extrinsic-Gender-Bias)
**Area**: Multilingual / Fairness
**Keywords**: gender bias mitigation, Bangla NLP, KL divergence regularization, counterfactual data augmentation, classification fairness

## TL;DR
To address extrinsic gender bias in pretrained language models applied to Bangla downstream classification tasks, this paper proposes RandSymKL, a method that jointly optimizes randomized cross-entropy loss and symmetric KL divergence to effectively reduce gender prediction disparities while maintaining classification accuracy.

## Background & Motivation

**Background**: Although large language models are highly capable, their deployment cost is prohibitive for low-resource languages such as Bangla. In practice, task-specific pretrained language models (PLMs) such as BERT and ELECTRA are more commonly used for classification tasks including sentiment analysis and hate speech detection.

**Limitations of Prior Work**: These PLMs produce inconsistent predictions for male-centric and female-centric texts—a phenomenon known as *extrinsic gender bias*. For example, a hate speech detection model may correctly classify a female-centric sentence as "abusive" while misclassifying a semantically equivalent male-centric sentence as "normal."

**Key Challenge**: Existing bias research has focused primarily on English and on intrinsic bias at the embedding level, leaving extrinsic bias in Bangla downstream tasks largely unaddressed. Furthermore, gender in Bangla is encoded more implicitly—through social roles, kinship terms, and personal names—making it harder for models to preserve semantic consistency after counterfactual substitution.

**Goal**: (1) Construct a gender bias evaluation benchmark for Bangla; (2) Propose a general debiasing training strategy that reduces gender prediction disparities without sacrificing classification performance.

**Key Insight**: The authors observe that if, during training, either the male or female version of a text is randomly selected to compute the cross-entropy loss while symmetric KL divergence is simultaneously used to align the output distributions of both versions, the model can learn gender-invariant classification representations.

**Core Idea**: Joint optimization via randomized cross-entropy and symmetric KL divergence (RandSymKL) aligns predictions for gender variants at the output distribution level, without relying on token-level gender annotations.

## Method

### Overall Architecture
During training, both a male-centric text and its corresponding female-centric text are fed into the model simultaneously, yielding output probability distributions $P_{\text{male}}$ and $P_{\text{female}}$. A joint loss function then optimizes both classification accuracy and distributional alignment. At inference time, only a single input text is required; no gender pair generation is needed.

### Key Designs

1. **Counterfactual Data Construction**:

    - *Function*: Generate semantically equivalent text pairs with opposite gender encoding, for evaluation and training.
    - *Mechanism*: A lexicon of 573 Bangla gender word pairs (e.g., "son/daughter," "brother/sister") is constructed, combined with NER-based name substitution, and verified through human review to ensure quality. Bangla-specific challenges such as polysemy (e.g., *dada* can mean "elder brother" or "grandfather") and spelling variants are explicitly addressed.
    - *Design Motivation*: Because Bangla lacks grammatical gender but encodes it implicitly, simple word substitution is insufficient; linguist-curated lexicons and human validation are necessary.

2. **Randomized Cross-Entropy Loss**:

    - *Function*: Prevent the model from overfitting to gender-specific expressions.
    - *Mechanism*: At each training step, either the male or female version of the logits $\mathbf{z}_1$ or $\mathbf{z}_2$ is randomly selected to compute the standard cross-entropy loss, rather than always using a fixed version.
    - *Design Motivation*: Consistently using the male version to compute CE loss may cause the model to implicitly learn a distributional preference for male text; randomization eliminates this systematic bias.

3. **Symmetric KL Divergence Regularization**:

    - *Function*: Explicitly align the prediction distributions of male and female text versions.
    - *Mechanism*: Computes $\mathcal{L}_{\text{KL}} = \text{KL}(P_{\text{male}} \| P_{\text{female}}) + \text{KL}(P_{\text{female}} \| P_{\text{male}})$, penalizing distributional asymmetry in both directions.
    - *Design Motivation*: Standard KL divergence is asymmetric; using the symmetric variant ensures the model is not biased toward either gender direction.

### Loss & Training
The total loss is $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda \cdot \mathcal{L}_{\text{KL}}$, where $\lambda$ controls the debiasing strength. Training uses batch size 4, learning rate $1 \times 10^{-4}$, Adam optimizer, 15 epochs followed by dropout adjustment based on the validation set and an additional 3–5 epochs of fine-tuning.

## Key Experimental Results

### Main Results

| Task | Method | Avg. Accuracy | Accuracy Gap (AG) | FairScore |
|------|--------|--------------|-------------------|-----------|
| All 4 tasks | OSI (no fine-tuning) | 56.17% | 3.39% | 22.06% |
| All 4 tasks | FOD (fine-tuning only) | 91.10% | 2.50% | 5.97% |
| All 4 tasks | Token Masking | 87.46% | 0.00% | 0.00% |
| All 4 tasks | FOA (data augmentation) | 90.46% | 0.32% | 3.16% |
| All 4 tasks | CSD (cosine similarity) | 90.58% | 1.10% | 3.31% |
| All 4 tasks | RandSymKL (Ours) | 90.66% | **0.29%** | **1.69%** |

### Ablation Study

| Configuration | Avg. FairScore | Avg. AG | Notes |
|---------------|---------------|---------|-------|
| RandSymKL (full) | 1.69% | 0.29% | Full model |
| NonRandSymKL_M (no randomization) | 2.31% | 0.52% | CE computed on male version only |
| AvgSymKL_MF (averaged logits) | 2.30% | 0.33% | Random selection replaced by averaged logits |
| Token Masking | 0.00% | 0.00% | Fully debiased but ~3% accuracy loss |

### Key Findings
- Excluding Token Masking, RandSymKL achieves the lowest FairScore (1.69%), outperforming the strongest baseline AvgSymKL_MF by 0.61 percentage points, with statistical significance ($p = 0.012$).
- Although Token Masking completely eliminates bias (FairScore = 0), it incurs a substantial accuracy cost (87.46% vs. 90.66%).
- Randomization is critical—removing it (NonRandSymKL_M) raises FairScore from 1.69% to 2.31%.
- RandSymKL also achieves the best performance on group fairness metrics such as EOD and SPD.

## Highlights & Insights
- **The combination of randomization and symmetric KL is simple yet effective**: No modifications to model architecture or additional models are required; debiasing is achieved purely through training strategy changes, making the method directly transferable to other languages and classification tasks.
- **A lexicon of 573 gender word pairs**: This constitutes a valuable resource for Bangla gender bias research, accounting for polysemy and culturally specific gender role expressions.
- **Output distribution alignment vs. embedding alignment**: Compared to CSD, which applies cosine similarity constraints in the embedding space, RandSymKL's alignment at the output probability level is more direct and effective.

## Limitations & Future Work
- Validation is limited to four binary classification tasks; more complex settings such as multi-class classification and sequence labeling are not explored.
- Gender encoding in Bangla is largely context-dependent (e.g., chains of kinship relations), and the current lexicon-based approach may miss some implicit gender signals.
- Experiments are conducted only with BERT- and ELECTRA-scale models; effectiveness on larger models remains unverified.
- Future work could extend the approach to other low-resource languages (e.g., Hindi, Tamil) to assess cross-lingual generalizability.

## Related Work & Insights
- **vs. CSD (Igbaria & Belinkov 2024)**: CSD aligns representations in the embedding space using cosine similarity, whereas this paper aligns predictions in the output probability space using symmetric KL divergence—the latter is more direct and yields better results (FairScore 3.31% vs. 1.69%).
- **vs. FOA (data augmentation)**: FOA simply doubles the training data with limited effect (FairScore 3.16%); this paper leverages counterfactual data more effectively through loss function design.
- **vs. Patel & Kisku 2024**: Their approach uses KL divergence to pull predictions toward a uniform distribution, whereas this paper applies symmetric KL between gender pairs—a more targeted formulation.

## Rating
- Novelty: ⭐⭐⭐ Individual components are not novel, but their combination and application to Bangla debiasing are valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐ Four tasks, multiple baselines, statistical significance testing, and ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Problem motivation is clear; method description is detailed.
- Value: ⭐⭐⭐ Offers useful reference for fairness research in low-resource languages; method is transferable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] MORPHOGEN: A Multilingual Benchmark for Evaluating Gender-Aware Morphological Generation](morphogen_a_multilingual_benchmark_for_evaluating_gender-aware_morphological_gen.md)
- [\[AAAI 2026\] Mitigating Content Effects on Reasoning in Language Models through Fine-Grained Activation Steering](../../AAAI2026/multilingual_mt/mitigating_content_effects_on_reasoning_in_language_models_through_fine-grained_.md)
- [\[NeurIPS 2025\] HelpSteer3-Preference: Open Human-Annotated Preference Data across Diverse Tasks and Languages](../../NeurIPS2025/multilingual_mt/helpsteer3-preference_open_human-annotated_preference_data_across_diverse_tasks_.md)
- [\[ACL 2026\] Just Use XML: Revisiting Joint Translation and Label Projection](just_use_xml_revisiting_joint_translation_and_label_projection.md)
- [\[ACL 2026\] Location Not Found: Exposing Implicit Local and Global Biases in Multilingual LLMs](location_not_found_exposing_implicit_local_and_global_biases_in_multilingual_llm.md)

</div>

<!-- RELATED:END -->

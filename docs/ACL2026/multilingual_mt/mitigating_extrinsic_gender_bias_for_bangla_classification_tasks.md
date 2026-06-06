---
title: >-
  [Paper Note] Mitigating Extrinsic Gender Bias for Bangla Classification Tasks
description: >-
  [ACL 2026][Multilingual & Machine Translation][Gender Bias Mitigation] Addressing extrinsic gender bias in pre-trained models for Bangla downstream classification tasks…
tags:
  - "ACL 2026"
  - "Multilingual & Machine Translation"
  - "Gender Bias Mitigation"
  - "Bangla NLP"
  - "KL Divergence Regularization"
  - "Counterfactual Data Augmentation"
  - "Classification Fairness"
date: 2026-05-08
content_hash: 2e1ae189f4c2f67b
---

# Mitigating Extrinsic Gender Bias for Bangla Classification Tasks

**Conference**: ACL 2026  
**arXiv**: [2411.10636](https://arxiv.org/abs/2411.10636)  
**Code**: [GitHub](https://github.com/sajib-kumar/Mitigating-Bangla-Extrinsic-Gender-Bias)  
**Area**: Multilingual/Fairness  
**Keywords**: Gender Bias Mitigation, Bangla NLP, KL Divergence Regularization, Counterfactual Data Augmentation, Classification Fairness

## TL;DR
Addressing extrinsic gender bias in pre-trained models for Bangla downstream classification tasks, the RandSymKL method is proposed. By jointly optimizing randomized cross-entropy loss and symmetric KL divergence, the method effectively reduces prediction disparities between genders while maintaining classification accuracy.

## Background & Motivation

**Background**: Although large models are powerful, their deployment costs in low-resource languages like Bangla are prohibitive. Consequently, task-specific Pre-trained Language Models (PLMs) such as BERT and ELECTRA are more commonly used in practical applications for classification tasks like sentiment analysis and hate speech detection.

**Limitations of Prior Work**: These PLMs produce inconsistent predictions for male-related and female-related texts—a phenomenon known as "extrinsic gender bias." For instance, a hate speech detection model might correctly classify a female-centric sentence as "abusive" while misidentifying a semantically equivalent male-centric sentence as "normal."

**Key Challenge**: Existing bias research primarily focuses on English and intrinsic bias (at the embedding level), with a significant lack of systematic study on extrinsic bias (at the downstream task prediction level) for Bangla. Furthermore, gender encoding in Bangla is more implicit (through social roles, kinship terms, and names), making it harder for models to handle semantic consistency after counterfactual substitutions.

**Goal**: (1) Construct a gender bias evaluation benchmark for Bangla; (2) Propose a universal debiasing training strategy that reduces gender prediction disparities while maintaining classification performance.

**Key Insight**: It is observed that if a model randomly selects either the male or female version to compute cross-entropy loss during training, while simultaneously pulling the output distributions of both versions closer using symmetric KL divergence, it can learn gender-independent classification representations.

**Core Idea**: Use randomized cross-entropy and symmetric KL divergence joint optimization (RandSymKL) to align predictions of gender variants at the output distribution level, without relying on token-level gender markers.

## Method

### Overall Architecture
During training, both male-centric text and its corresponding female-centric version are input simultaneously to obtain output probability distributions $P_{\text{male}}$ and $P_{\text{female}}$. A joint loss function is then used to optimize both classification accuracy and distribution alignment. During inference, only a single text is required, and gender pair generation is unnecessary.

### Key Designs

1. **Counterfactual Data Construction**:

    - **Function**: Generates semantically equivalent but gender-opposite text pairs for evaluation and training.
    - **Mechanism**: A dictionary of 573 Bangla gender word pairs (e.g., "son/daughter", "brother/sister") was constructed, combined with NER for name replacement, and manually reviewed for quality. Language-specific issues like polysemy (e.g., "dada" can mean "elder brother" or "grandfather") and spelling variations were considered.
    - **Design Motivation**: Unlike languages with grammatical gender, Bangla has implicit gender encoding. Simple word replacement cannot cover all scenarios, necessitating linguistic expert-curated dictionaries and manual verification.

2. **Randomized Cross-Entropy Loss**:

    - **Function**: Prevents the model from overfitting to specific gender expressions.
    - **Mechanism**: At each training step, standard cross-entropy loss is calculated by randomly selecting either the male or female version logits $\mathbf{z}_1$ or $\mathbf{z}_2$, rather than using one version fixedly.
    - **Design Motivation**: If the male version is always used for CE, the model might implicitly learn a distribution preference for male-centric text. Randomization eliminates this systematic bias.

3. **Symmetric KL Divergence Regularization**:

    - **Function**: Explicitly narrows the prediction distribution gap between male and female versions.
    - **Mechanism**: Calculates $\mathcal{L}_{\text{KL}} = \text{KL}(P_{\text{male}} \| P_{\text{female}}) + \text{KL}(P_{\text{female}} \| P_{\text{male}})$, penalizing distribution asymmetry in both directions.
    - **Design Motivation**: Unidirectional KL is asymmetric; using the symmetric version ensures the model does not lean toward any specific gender direction.

### Loss & Training
The total loss is $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda \cdot \mathcal{L}_{\text{KL}}$, where $\lambda$ controls the debiasing strength. Training uses a batch size of 4, a learning rate of $1 \times 10^{-4}$, and the Adam optimizer. After 15 epochs, fine-tuning is performed for 3-5 epochs after adjusting dropout based on the validation set.

## Key Experimental Results

### Main Results

| Task | Method | Average Accuracy | Accuracy Gap (AG) | FairScore |
|------|------|-----------|---------------|-----------|
| All 4 Tasks | OSI (Not fine-tuned) | 56.17% | 3.39% | 22.06% |
| All 4 Tasks | FOD (Fine-tuned only) | 91.10% | 2.50% | 5.97% |
| All 4 Tasks | Token Masking | 87.46% | 0.00% | 0.00% |
| All 4 Tasks | FOA (Data Augmentation) | 90.46% | 0.32% | 3.16% |
| All 4 Tasks | CSD (Cosine Sim) | 90.58% | 1.10% | 3.31% |
| All 4 Tasks | **Ours** (RandSymKL) | 90.66% | **0.29%** | **1.69%** |

### Ablation Study

| Configuration | Average FairScore | Average AG | Description |
|------|-------------|--------|------|
| RandSymKL (Full) | 1.69% | 0.29% | Complete model |
| NonRandSymKL_M (No Randomization) | 2.31% | 0.52% | CE uses male version only |
| AvgSymKL_MF (Average Logits) | 2.30% | 0.33% | Average logits used instead of random choice |
| Token Masking | 0.00% | 0.00% | Full debias but 3% accuracy drop |

### Key Findings
- Excluding Token Masking, RandSymKL achieves the lowest FairScore (1.69%), which is 0.61 percentage points lower than the strongest baseline AvgSymKL_MF and is statistically significant ($p = 0.012$).
- Although Token Masking completely eliminates bias (FairScore = 0), the cost in accuracy is too high (87.46% vs 90.66%).
- Randomization is critical—removing it (NonRandSymKL_M) increases the FairScore from 1.69% to 2.31%.
- RandSymKL also performs best on group fairness metrics such as EOD and SPD.

## Highlights & Insights
- **Simplicity and Effectiveness**: The combination of randomization and symmetric KL is effective without modifying model architecture or introducing auxiliary models. It can be directly transferred to other languages and classification tasks.
- **573 Gender Word Pairs Dictionary**: This is a significant resource for Bangla gender bias research, accounting for polysemy and culturally specific gender roles.
- **Output Distribution Alignment**: Compared to CSD, which applies cosine similarity constraints in embedding space, RandSymKL aligns at the output probability level, which is more direct and effective.

## Limitations & Future Work
- Validated only on 4 binary classification tasks; more complex scenarios like multi-class classification or sequence labeling are not yet addressed.
- Gender encoding in Bangla relies heavily on context (e.g., kinship chains), and current dictionary methods might miss some implicit gender information.
- Experiments focused on BERT and ELECTRA models; performance on larger models remains to be verified.
- Future work could extend to other low-resource languages (e.g., Hindi, Tamil) to verify cross-lingual generalizability.

## Related Work & Insights
- **vs CSD (Igbaria & Belinkov 2024)**: While CSD aligns in the embedding space via cosine similarity, this work aligns in the output probability space using symmetric KL, which is more direct and yields better results (FairScore 3.31% vs 1.69%).
- **vs FOA (Data Augmentation)**: FOA simply doubles training data but has limited impact (FairScore 3.16%). This work utilizes counterfactual data more effectively through loss function design.
- **vs Patel & Kisku 2024**: They use KL divergence to push predictions toward a uniform distribution, whereas this work applies symmetric KL specifically between gender pairs.

## Rating
- Novelty: ⭐⭐⭐ The components are existing, but the combination and application scenario (Bangla debiasing) are valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐ 4 tasks, multiple baselines, statistical significance tests, and ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and detailed method description.
- Value: ⭐⭐⭐ Reference value for fairness research in low-resource languages; transferable methodology.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] FairQE: Multi-Agent Framework for Mitigating Gender Bias in Translation Quality Estimation](fairqe_multi-agent_framework_for_mitigating_gender_bias_in_translation_quality_e.md)
- [\[ACL 2026\] MORPHOGEN: A Multilingual Benchmark for Evaluating Gender-Aware Morphological Generation](morphogen_a_multilingual_benchmark_for_evaluating_gender-aware_morphological_gen.md)
- [\[ACL 2026\] Mitigating Catastrophic Forgetting in Target Language Adaptation of LLMs via Source-Shielded Updates](mitigating_catastrophic_forgetting_in_target_language_adaptation_of_llms_via_sou.md)
- [\[ACL 2026\] Enhancing BiGRU with a KAN Block for Legal Document Classification and Summarization](enhancing_bigru_with_a_kan_block_for_legal_document_classification_and_summariza.md)
- [\[ACL 2026\] TLPO: Token-Level Policy Optimization for Mitigating Language Confusion in Large Language Models](tlpo_token-level_policy_optimization_for_mitigating_language_confusion_in_large_.md)

</div>

<!-- RELATED:END -->

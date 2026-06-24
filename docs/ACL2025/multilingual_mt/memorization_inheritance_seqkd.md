---
title: >-
  [Paper Note] Memorization Inheritance in Sequence-Level Knowledge Distillation for Neural Machine Translation
description: >-
  [ACL 2025][Multilingual & Machine Translation][Knowledge Distillation] This paper presents the first systematic study of how memorization behavior of teacher models is transferred to student models in sequence-level knowledge distillation (SeqKD). It is discovered that although the student model is never directly exposed to the original training data, its extractive memorization rate is $57\%$ higher than that of the baseline model, accompanied by an increased hallucination r…
tags:
  - "ACL 2025"
  - "Multilingual & Machine Translation"
  - "Knowledge Distillation"
  - "Memorization"
  - "Hallucination"
  - "Machine Translation"
  - "SeqKD"
date: 2026-05-08
content_hash: c5ac10ad82dcff6e
---

# Memorization Inheritance in Sequence-Level Knowledge Distillation for Neural Machine Translation

**Conference**: ACL 2025  
**arXiv**: [2502.01491](https://arxiv.org/abs/2502.01491)  
**Code**: [https://github.com/vernadankers/memseqkd](https://github.com/vernadankers/memseqkd)  
**Area**: Multilingual Translation  
**Keywords**: Knowledge Distillation, Memorization, Hallucination, Machine Translation, SeqKD

## TL;DR
This paper presents the first systematic study of how memorization behavior of teacher models is transferred to student models in sequence-level knowledge distillation (SeqKD). It is discovered that although the student model is never directly exposed to the original training data, its extractive memorization rate is $57\%$ higher than that of the baseline model, accompanied by an increased hallucination rate. Adaptive-SeqKD is proposed to mitigate these issues by fine-tuning the teacher on a high-quality subset.

## Background & Motivation

**Background**: SeqKD is a standard practice in NMT deployment, where a large teacher model translates the source side of training data to generate synthetic targets for training a smaller student model. Commercial translation systems such as NLLB and ALMA utilize SeqKD.

**Limitations of Prior Work**: 
- Prior research primarily focuses on why SeqKD succeeds (e.g., pattern simplification and regularization), while the propagation of failure modes remains largely unexplored.
- Memorization of noisy training data by NMT models leads to unreliable behaviors during deployment.
- While some studies in image classification suggest that KD suppresses memorization, corresponding research in the NLP/NMT domain is lacking.

**Key Challenge**: SeqKD propagates both the strengths (performance) and weaknesses (memorization, hallucinations) of the teacher, yet the community has predominantly focused on the former.

**Goal**: (1) Quantify the extent to which the student inherits memorization behaviors from the teacher, (2) analyze behavioral variations across different data subgroups, and (3) propose mitigation strategies.

**Key Insight**: The student model never directly witnesses the target side of the original parallel corpus; instead, it observes the teacher's translations. If the teacher memorizes the original target and forwards it to the student, how does the student behave?

**Core Idea**: The denoising effect of SeqKD, while improving student performance, reduces the regularization effect, paradoxically leading to higher memorization and hallucination rates in the student compared to a directly trained baseline model.

## Method

### Overall Architecture
Input: WMT20 parallel corpora (De-En/En-De 48M, Pl-En/En-Pl 12M, Fr-De 14M) $\to$ Train Transformer-large teacher (300k steps) $\to$ Teacher translates the source side to generate synthetic targets $\to$ Train Transformer-base student (100k steps) $\to$ Compare memorization and hallucination behaviors of the student versus the baseline (same architecture, trained directly on the original data).

### Key Designs

1. **Quantifying Memorization Metrics**:

    - **Function**: Measures the degree to which a model memorizes training data across multiple dimensions.
    - **Replication rate**: The proportion of exact matches between greedy translations and training targets.
    - **Extractive Memorization (ExMem)**: The proportion where the target can be fully reconstructed when the model is prompted with only $\le 75\%$ of the source side, indicating that the model has memorized the source-to-target mapping without requiring the full source sentence.
    - **OscHal (Oscillatory hallucination)**: Translations containing a bigram repeated $\ge 10$ times that does not appear on the source side.
    - **NatHal (Natural hallucination)**: Occurs when a translation is outputted by the model for $\ge 5$ different source inputs, indicating that the model "default-outputs" certain sentences.
    - **Design Motivation**: Comprehensively characterizes memorization across both "exact match" and "behavioral abnormality" dimensions.

2. **Subgroup Analysis**:

    - **Function**: Partitions the training data based on data quality, Counterfactual Memorization (CM) scores, and teacher confidence to analyze behavior within each subgroup.
    - **Quality Partitioning**: Categorized into five bins ($<0.2$, $0.2-0.4$, ..., $\ge 0.8$) using Comet-QE-22.
    - **CM Partitioning**: Approximated using leave-one-out cross-validation to estimate the counterfactual memorization score of each sample.
    - **Design Motivation**: Reveals the differentiated impact of SeqKD on distinct data subsets.

3. **Adaptive-SeqKD**:

    - **Function**: Incorporates a teacher adaptation step into the SeqKD pipeline to reduce the transmission of memorization.
    - **Mechanism**: Selects a high-quality subset using Comet-QE-22 $\to$ briefly fine-tunes the teacher on this subset $\to$ generates distilled targets using the fine-tuned teacher.
    - **Design Motivation**: Fine-tuning on high-quality data encourages the teacher to "forget" noisy memorization, yielding cleaner distilled targets; this process requires no external data and is completely self-contained.
    - **Comparison with random fine-tuning**: Fine-tuning on a random subset degrades student quality.

### Loss & Training
- Teacher: Transformer-large, 300k steps, original WMT20 parallel corpus.
- Student/Baseline: Transformer-base, 100k steps.
- Student training data: Unchanged source side, with target side replaced by the teacher's beam size $= 1$ translations.
- MarianNMT training framework.

## Key Experimental Results

### Main Results

| Metric | Teacher $\to$ Baseline Relationship | Student vs. Baseline Increase |
|------|-------------|----------------|
| Replication rate (Exact match) | Teacher > Student > Baseline | $+3.4\% \pm 0.9$ |
| ExMem rate (Extractive memorization) | Teacher > Student > Baseline | **$+57.0\% \pm 15.4$** |
| OscHal (Oscillatory hallucination) | Student > Baseline > Teacher | $+31.0\% \pm 25.7$ |
| NatHal (Natural hallucination) | Teacher > Student > Baseline | $+13.8\% \pm 5.0$ |

### Adaptive-SeqKD Results

| Metric | High-quality Fine-tuned Teacher | Randomly Fine-tuned Teacher | High-quality Fine-tuned Student | Randomly Fine-tuned Student |
|------|-------------|------------|-------------|------------|
| BLEU Change | $+0.0 \pm 0.5$ | $-1.2 \pm 0.8$ | $-0.2 \pm 1.7$ | $-1.2 \pm 1.6$ |
| Comet-QE-22 | $+0.2 \pm 0.3$ | $-0.2 \pm 0.1$ | **$+0.3 \pm 0.3$** | $-0.1 \pm 0.2$ |

### Key Findings
- **The student's ExMem rate is $57\%$ higher than the baseline, yet the student only indirectly observed $18.4\%$ of the original training targets**. This indicates that the teacher selectively "relays" memorized samples, leading to a higher degree of memorization in the student under cleaner data conditions.
- **Secondary Memorization**: The student not only inherits the teacher's memorization of the original corpus but also develops new memorization of unique translations generated by the teacher (secondary ExMem accounts for $59\%$ of the total ExMem).
- **Denoising as a double-edged sword**: While denoising improves translation quality, it reduces the noise-based regularization effect inherent in the training data, ultimately leading to stronger memorization.
- **Students outperform teachers on low-quality subgroups**: On samples with quality $<0.4$, the student's Comet-QE-22 scores exceed those of the teacher, indicating that the student further denoises the teacher's "denoised translations".
- **Adaptive-SeqKD is effective yet conservative**: Fine-tuning the teacher on high-quality data leads to improvements in reference-free translation quality for the student, alongside a reduced memorization rate, without sacrificing BLEU.

## Highlights & Insights
- **The counterintuitive finding of "the student memorizing more despite not seeing the original data" holds significant practical implications**: Commercial translation systems using SeqKD should actively monitor memorization behaviors, as failure to do so could risk leaking private information embedded in the training data.
- **The methodology of using ExMem to quantify NMT memorization is highly practical**: Reconstructing the target by merely observing $75\%$ of the source side captures the degree of "memorization" far better than exact matching. This approach is transferable to the analysis of memorization in other sequence-to-sequence tasks.
- **The causal chain of *denoising $\to$ reduced regularization $\to$ enhanced memorization* uncovers a hidden cost of KD**: This insight applies broadly to any scenario involving training on synthetic data—the "cleaner" the data, the more susceptible the model becomes to overfitting.

## Limitations & Future Work
- **Focus is restricted solely to SeqKD in NMT**: Whether these conclusions generalize to LLM distillation (e.g., distilling from GPT-4) requires further investigation.
- **The high-quality filtering in Adaptive-SeqKD relies heavily on Comet-QE-22**: This quality estimator may introduce its own biases.
- **Only beam size $= 1$ distillation was used**: The impact of different beam sizes on the transmission of memorization warrants exploration.
- **Lack of direct quantification of privacy risks**: Although security risks associated with memorization are highlighted, membership inference attack experiments were not conducted.

## Related Work & Insights
- **vs. Lukasik et al. (2024)**: KD suppresses memorization in image classification, whereas this paper finds that SeqKD in NMT accentuates it. The discrepancy in tasks and KD formulations leads to contrasting conclusions.
- **vs. Jagielski et al. (2024)**: Jagielski et al. found that distilled students are vulnerable to membership inference attacks; this work corroborates similar conclusions through the lens of translation quality and behavioral analysis.
- **vs. Zhou et al. (2020)**: Both acknowledge the denoising capacity of SeqKD, but this paper is the first to demonstrate that a side effect of denoising is the enhancement of memorization.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Conducts the first systematic study of memorization propagation in SeqKD, with counterintuitive yet significant findings.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Highly comprehensive, covering 5 language pairs $\times$ 3 models $\times$ 12 subgroups $\times$ multiple evaluation metrics.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, rich examples, and intuitive visualizations.
- Value: ⭐⭐⭐⭐ Offers direct guidelines for the security of KD and NMT system deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Multi-perspective Alignment for Increasing Naturalness in Neural Machine Translation](multi-perspective_alignment_for_increasing_naturalness_in_neural_machine_transla.md)
- [\[ACL 2025\] Registering Source Tokens to Target Language Spaces in Multilingual Neural Machine Translation](registering_source_tokens_to_target_language_spaces_in_multilingual_neural_machi.md)
- [\[ACL 2025\] THOR-MoE: Hierarchical Task-Guided and Context-Responsive Routing for Neural Machine Translation](thor-moe_hierarchical_task-guided_and_context-responsive_routing_for_neural_mach.md)
- [\[ACL 2025\] Unveiling the Power of Source: Source-based Minimum Bayes Risk Decoding for Neural Machine Translation](unveiling_the_power_of_source_source-based_minimum_bayes_risk_decoding_for_neura.md)
- [\[ACL 2025\] Exploring In-context Example Generation for Machine Translation](exploring_in-context_example_generation_for_machine_translation.md)

</div>

<!-- RELATED:END -->

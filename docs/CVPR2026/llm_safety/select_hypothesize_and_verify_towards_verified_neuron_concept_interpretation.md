---
title: >-
  [Paper Note] Select, Hypothesize and Verify: Towards Verified Neuron Concept Interpretation
description: >-
  [CVPR 2026][LLM Safety][Neuron interpretation] This paper proposes SIEVE (Select–Hypothesize–Verify), a closed-loop framework that interprets neuron functionality by selecting highly activated samples…
tags:
  - "CVPR 2026"
  - "LLM Safety"
  - "Neuron interpretation"
  - "concept verification"
  - "explainable AI"
  - "neuron function analysis"
  - "closed-loop verification"
date: 2026-05-08
content_hash: 667bfe25a20bd91b
---

# Select, Hypothesize and Verify: Towards Verified Neuron Concept Interpretation

**Conference**: CVPR 2026
**arXiv**: [2603.24953](https://arxiv.org/abs/2603.24953)  
**Code**: None  
**Area**: LLM Safety
**Keywords**: Neuron interpretation, concept verification, explainable AI, neuron function analysis, closed-loop verification

## TL;DR

This paper proposes SIEVE (Select–Hypothesize–Verify), a closed-loop framework that interprets neuron functionality by selecting highly activated samples, generating concept hypotheses, and verifying them via text-to-image generation. The probability that generated concepts activate the corresponding neuron is approximately 1.5× that of existing SOTA methods.

## Background & Motivation

1. **Background**: Understanding individual neuron functionality—i.e., what concept a neuron encodes—is a central problem in neural network interpretability. Methods such as Network Dissection, CLIP-Dissect, FALCON, and DnD have made progress by describing neuron concepts in natural language.

2. **Limitations of Prior Work**: These methods share a common assumption that every neuron has a well-defined function and contributes discriminative features to decisions. However, research has shown that networks contain redundant neurons that do not contribute to decisions. Generating descriptions for such neurons leads to misinterpretation of the network's decision mechanism.

3. **Key Challenge**: Existing methods are fundamentally "observe → hypothesize" pipelines that infer neuron functionality from activation distributions over probe datasets. Due to limited data coverage, these hypotheses may suffer from dataset bias and fail to accurately reflect true neuron functionality. A verification step is absent.

4. **Goal**: (1) How to filter out neurons that do not provide discriminative features; (2) how to verify whether generated concepts genuinely match neuron functionality.

5. **Key Insight**: Drawing on the neuroscience paradigm of "observe → hypothesize → verify," the authors argue that interpretability research on deep networks should follow the same closed-loop logic.

6. **Core Idea**: Filter effective neurons by activation distribution, generate concept hypotheses via clustering, and then verify concept–neuron alignment in a closed loop using text-to-image generated images.

## Method

### Overall Architecture

The SIEVE framework consists of three stages. Given a pretrained classification network and a probe dataset: (1) **Select**: filter high-quality samples with consistent activation patterns based on activation distribution; (2) **Hypothesize**: cluster the selected samples and use a vision-language model to generate concept hypotheses for each cluster; (3) **Verify**: generate images via Stable Diffusion conditioned on the hypothesized concepts, then measure whether these images highly activate the corresponding neuron to verify concept accuracy.

### Key Designs

1. **High-Activation Sample Selection (Select)**:

    - **Function**: Filter samples that reflect consistent and unambiguous neuron functionality.
    - **Mechanism**: The activation distribution of each neuron over the probe dataset is computed. The ratio of the 99th percentile to the median quantifies the neuron's response discriminability. If this ratio exceeds threshold $\beta$ (default 10), the neuron is considered to encode a well-defined functional feature. The top-20 most activated samples are retained for qualifying neurons. High-discriminability neurons (e.g., Neuron 507) exhibit consistent high responses to specific stimuli, whereas low-discriminability neurons (e.g., Neuron 144) show diffuse responses.
    - **Design Motivation**: Filter out redundant/non-discriminative neurons to avoid generating misleading concept descriptions for them.

2. **Concept Hypothesis Generation (Hypothesize)**:

    - **Function**: Generate natural language concept descriptions for the high-activation samples of each neuron.
    - **Mechanism**: Activation maps are used to crop high-activation patches from the selected samples. Features are extracted and grouped via agglomerative clustering, with the number of clusters determined automatically by the Silhouette score, capturing multiple potential functional patterns of a single neuron. For each cluster, CLIP retrieves top-$K$ ($K=2$) concepts from a predefined concept set as the functional hypothesis: $h_{i,j} = \arg\text{top-}K(\{g(t_q, C_{i,j}) \mid t_q \in \mathcal{T}\}, K)$
    - **Design Motivation**: Clustering discovers multiple functional patterns within a single neuron, yielding more accurate descriptions than a single label.

3. **Concept Verification (Verify)**:

    - **Function**: Verify the consistency between hypothesized concepts and neuron functionality via constructive intervention.
    - **Mechanism**: Hypothesized concepts are used as text prompts to generate a verification image set via Stable Diffusion, independent of the probe dataset. These images are fed into the target network, and the Activation Rate (AR) is computed: $AR_i = \frac{1}{|\mathcal{D}_{gen}^{(i,j)}|} \sum_{x_{gen}} \mathbb{1}\{a_i^l(x_{gen}) > T_i\}$, where $T_i$ is the top-1% activation threshold. Hypotheses with low AR are discarded; only high-AR concepts are retained as final interpretations.
    - **Design Motivation**: Unlike traditional destructive interventions (e.g., neuron ablation), this constructive approach actively generates stimuli consistent with the hypothesized concept to observe the neuron's response, analogous to controlled verification in scientific experiments.

### Loss & Training

SIEVE involves no training and operates as a post-hoc analysis framework. Key hyperparameters: activation threshold $\beta=10$, number of clusters determined automatically, top-2 concepts per cluster, and mean AR used as the filtering threshold during verification.

## Key Experimental Results

### Main Results

Evaluated on ImageNet-pretrained ResNet-50 using the Common Words (3k) concept set:

| Method | CLIP cos | mpnet cos | mean AR (%) |
|--------|----------|-----------|-------------|
| Network Dissect | 0.7073 | 0.3256 | 45.01 |
| CLIP-Dissect | 0.7868 | 0.4462 | 57.91 |
| WWW | 0.7713 | 0.4463 | 50.23 |
| DnD | 0.7595 | 0.4371 | 51.46 |
| **SIEVE (Ours)** | **0.7914** | **0.4547** | **86.29** |

A similar trend is observed on ViT-B/16: SIEVE achieves a mean AR of 85.24%, substantially outperforming CLIP-Dissect at 57.70%.

### Ablation Study

| Configuration | CLIP cos | mpnet cos | mean AR (%) |
|---------------|----------|-----------|-------------|
| Baseline (no modules) | 0.6738 | 0.2306 | 45.57 |
| + Select + Cluster | 0.7624 | 0.4301 | 77.90 |
| + Select + Verify | 0.7821 | 0.4423 | 81.52 |
| + Select + Cluster (w/o Verify) | 0.7656 | 0.4189 | 72.87 |
| Full model | 0.7914 | 0.4547 | 86.29 |

### Key Findings

- **Verify module contributes most**: Removing Verify reduces mean AR from 86.29% to 72.87%, demonstrating the critical role of the verification step in ensuring concept–neuron alignment.
- **Threshold $\beta$ is robust**: Varying $\beta$ in the range 4–12 has minimal impact on final metrics (mean AR fluctuation <1%).
- **Verification remains effective under domain shift**: On remote sensing data (EuroSAT) with domain shift, SIEVE still achieves 75.45% mean AR, compared to 43.16% for CLIP-Dissect.
- **SIEVE produces finer-grained descriptions**: For example, Neuron 37 of ViT-B/16 is described as "Short Dense Coat," whereas the baseline only produces the coarse label "Dog."

## Highlights & Insights

- **Introduction of scientific methodology**: Transplanting the neuroscience paradigm of "observe → hypothesize → verify" into DNN interpretability is an elegant cross-domain analogy. The verification step addresses the critical gap in existing methods.
- **Constructive verification**: Rather than traditional ablation-based (destructive) interventions, SIEVE actively constructs stimuli consistent with the hypothesis via text-to-image generation. This positive verification is more direct and convincing.
- **Redundant neuron filtering**: This work is the first to explicitly address the problem that "not all neurons are meaningful," avoiding misinterpretation of redundant neurons.

## Limitations & Future Work

- **Domain shift in text-to-image generation**: The verification stage relies on Stable Diffusion; when the target network is trained on specialized domains (e.g., remote sensing), the distribution gap between generated and real images may reduce verification accuracy.
- **Concept set dependency**: The method still relies on predefined concept sets (e.g., Broden, Common Words) and cannot discover novel concepts outside the set.
- **Computational overhead**: Generating multiple verification images per concept per neuron incurs significant computational cost when scaled to an entire network.
- **Focus limited to the penultimate layer**: The framework has not been extended to the interpretation of shallow-layer neurons.

## Related Work & Insights

- **vs. CLIP-Dissect**: CLIP-Dissect directly matches concepts to activated samples via CLIP without a verification step; SIEVE adds closed-loop verification after matching, improving mean AR by approximately 30%.
- **vs. FALCON/WWW**: These methods improve concept description quality but still assume all neurons are meaningful; SIEVE filters redundant neurons via the Select stage.
- **vs. DnD**: DnD leverages LLMs to generate higher-quality natural language descriptions but lacks verification, achieving only 51.46% mean AR.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Introducing closed-loop verification from scientific methodology into neuron interpretation is a clear and compelling contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers multiple architectures (ResNet-18/50, ViT-B/16) and datasets with comprehensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Logically structured; the scientific methodology analogy is introduced naturally.
- **Value**: ⭐⭐⭐⭐ The proposed mean AR metric can serve as a general evaluation standard; the verification paradigm provides a valuable reference for future interpretability research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] DAMP: Class Unlearning via Depth-Aware Removal of Forget-Specific Directions](damp_class_unlearning_via_depth_aware_removal_of_forget_specific_directions.md)
- [\[CVPR 2026\] HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in LVLMs](hulluedit_subspace_editing_hallucination.md)
- [\[CVPR 2026\] PinPoint: Evaluation of Composed Image Retrieval with Explicit Negatives, Multi-Image Queries, and Paraphrase Testing](pinpoint_evaluation_of_composed_image_retrieval_with_explicit_negatives_multi-im.md)
- [\[CVPR 2026\] Designing to Forget: Deep Semi-parametric Models for Unlearning](designing_to_forget_deep_semi-parametric_models_for_unlearning.md)
- [\[CVPR 2026\] The Blind Spot of Adaptation: Quantifying and Mitigating Forgetting in Fine-tuned Driving Models](blind_spot_of_adaptation_quantifying_and_mitigating_forgetting_in_fine_tuned_driving_models.md)

</div>

<!-- RELATED:END -->

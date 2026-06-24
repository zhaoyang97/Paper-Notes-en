---
title: >-
  [Paper Note] Iron Sharpens Iron: Defending Against Attacks in Machine-Generated Text Detection with Adversarial Training
description: >-
  [ACL 2025][AIGC Detection][machine-generated text detection] This paper proposes the GREATER adversarial training framework, which simultaneously trains an adversarial attacker (Greater-A) and an MGT detector (Greater-D). The attacker identifies critical tokens through surrogate model gradients and perturbs them in the embedding space to generate adversarial samples. The detector learns generalized defense from these curriculum-style adversarial samples. Under 16 attacks…
tags:
  - "ACL 2025"
  - "AIGC Detection"
  - "machine-generated text detection"
  - "adversarial training"
  - "robustness"
  - "text perturbation"
  - "adversarial attacks"
date: 2026-05-08
content_hash: 102895afcc9ff64c
---

# Iron Sharpens Iron: Defending Against Attacks in Machine-Generated Text Detection with Adversarial Training

**Conference**: ACL 2025  
**arXiv**: [2502.12734](https://arxiv.org/abs/2502.12734)  
**Code**: [GitHub](https://github.com/Liyuuuu111/GREATER)  
**Area**: AIGC Detection  
**Keywords**: machine-generated text detection, adversarial training, robustness, text perturbation, adversarial attacks

## TL;DR
This paper proposes the GREATER adversarial training framework, which simultaneously trains an adversarial attacker (Greater-A) and an MGT detector (Greater-D). The attacker identifies critical tokens through surrogate model gradients and perturbs them in the embedding space to generate adversarial samples. The detector learns generalized defense from these curriculum-style adversarial samples. Under 16 attacks, the ASR drops to 5.53% (compared to SOTA's 6.20%), while the attack efficiency is 4 times faster than SOTA.

## Background & Motivation
**Background**: MGT detectors (such as DetectGPT, watermarking, etc.) perform well under normal conditions but face a 30-50% decline in accuracy when encountering simple perturbations (editing/paraphrasing/prompt modifications).

**Limitations of Prior Work**: (a) Existing defense methods (such as Text-RS, CERT-ED) fail to generalize to unseen attacks; (b) Adversarial attack methods either require white-box access or demand excessively high query volumes; (c) Adversarial training in the text domain lacks generalization.

**Key Challenge**: Effectively constructing efficient adversarial samples under a black-box setting for adversarial training, and ensuring that the trained defense generalizes to various different attacks.

**Goal**: Build a general adversarial training framework for MGT detection that simultaneously improves attack efficiency and defense generalization.

**Key Insight**: "Attack-to-defend" — synchronously update the attacker and detector so that the detector learns defense capabilities from increasingly stronger attacks.

**Core Idea**: Adversarially train the attacker and detector synchronously, and perform a greedy search in the embedding space to generate efficient adversarial samples, achieving generalized defense against various attacks.

## Method

### Overall Architecture
Greater-A (Attacker): surrogate model extracts token embeddings $\rightarrow$ scoring network identifies key tokens $\rightarrow$ embedding space gradient ascent perturbation $\rightarrow$ greedy search + pruning generates substitution words $\rightarrow$ outputs adversarial samples. Greater-D (Detector): trained on a mixture of benign samples and adversarial samples, updated synchronously with Greater-A.

### Key Designs
1. **Important Token Identification + Embedding Perturbation**:

    - **Function**: Train a scoring network $\mathcal{F}_\theta$ using the hidden states of the surrogate model to identify the top-$k$ most important tokens, and perform gradient ascent perturbation in the embedding space.
    - **Mechanism**: The perturbed embedding is $\tilde{e}_t = e_t + \mathbf{1}_{[t \in \mathbf{I}]} \delta_t$, where $\delta_t$ is optimized via gradient ascent to maximize the detector loss.
    - **Design Motivation**: Perturbation in the embedding space provides more precise guidance for candidate word generation compared to discrete token replacement.

2. **Greedy Search + Pruning**:

    - **Function**: Map the perturbed embeddings back to the vocabulary to find candidate words, and greedily select substitutions that maximize the change in detector predictions.
    - **Mechanism**: Attempt substitutions sequentially based on importance ranking, and prune candidate words that do not alter the prediction to minimize query count.
    - **Design Motivation**: High query efficiency is required in black-box settings; greedy search + pruning requires 4× fewer queries than brute force search.

3. **Synchronous Adversarial Training**:

    - **Function**: Simultaneously update Greater-A and Greater-D within the same training step.
    - **Mechanism**: As training progresses, Greater-A becomes increasingly stronger, allowing Greater-D to learn from curriculum-style adversarial samples and generalize to unseen attacks.
    - **Design Motivation**: Unlike two-stage methods that separate attack and defense phases, synchronous updates prevent the defense from overfitting to specific attacks.

## Key Experimental Results

### Defense Effectiveness (ASR%, lower is better)

| Method | Average of 10 Perturbations | Average of 6 Adversarial Attacks | Total Average |
|------|-------------|----------------|--------|
| Undefended | ~35% | ~85% | ~55% |
| Text-RS | - | - | 6.20% |
| TAVAT | - | - | ~8% |
| **Greater-D** | - | - | **5.53%** |

### Attack Effectiveness (Greater-A vs SOTA Attacks)

| Method | ASR% | Average Query Count |
|------|------|-----------|
| SOTA Attacks (TextFooler, etc.) | 88.13% | ~400+ |
| Greater-A | **96.58%** | ~100 (4× fewer) |

### Key Findings
- Greater-D comprehensively outperforms 10 existing defense methods under 16 attacks (10 perturbations + 6 adversarial attacks).
- Greater-A is simultaneously the strongest attack method: achieving an ASR of 96.58% (+8.45%) with 4× fewer queries.
- Synchronous updating is the key to generalized defense: defenses trained asynchronously only perform well against attacks encountered during training.
- Embedding space perturbation generates more natural adversarial samples than discrete token replacement.

## Highlights & Insights
- The concept of "attack-to-defend" is elegant and natural — a stronger attacker breeds a stronger defender.
- Achieves key token identification matching white-box effectiveness under black-box settings through a surrogate model.
- Both the attack and defense can be used as independent modules, offering high practicality.

## Limitations & Future Work
- Discrepancies between the surrogate model and the target detector might limit the accuracy of key token identification.
- Evaluated only on English text.
- Robustness against cross-lingual attacks (e.g., translation attacks) has not been tested.
- Adversarial training increases computational costs.

## Related Work & Insights
- **vs RADAR**: RADAR employs a paraphraser as an attacker and can only defend against known attacks; GREATER generalizes to unseen attacks through synchronous training.
- **vs OUTFOX**: Outfox relies on in-context learning (ICL) demonstrations of adversarial samples, whereas GREATER does not require external examples.
- **vs TextFooler/BERT-Attack**: Greater-A is more efficient (4× fewer queries) and more effective (+8.45% ASR) as an attacking method.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of synchronous adversarial training and embedding perturbation is novel in the field of MGT detection.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across 16 attacks, 10 defense baselines, and dual assessment of both attack and defense capabilities.
- Writing Quality: ⭐⭐⭐⭐ Rigorous threat modeling and clear methodological descriptions.
- Value: ⭐⭐⭐⭐⭐ Both attacking and defending components achieve SOTA performance, directly contributing to the robustness research of MGT detection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Learning From Dictionary: Enhancing Robustness of Machine-Generated Text Detection in Zero-Shot Language via Adversarial Training](../../ICLR2026/aigc_detection/learning_from_dictionary_enhancing_robustness_of_machine-generated_text_detectio.md)
- [\[ACL 2025\] MultiSocial: Multilingual Benchmark of Machine-Generated Text Detection of Social-Media Texts](multisocial_mgt_detection.md)
- [\[ACL 2025\] HACo-Det: A Study Towards Fine-Grained Machine-Generated Text Detection under Human-AI Coauthoring](haco-det_a_study_towards_fine-grained_machine-generated_text_detection_under_hum.md)
- [\[ACL 2025\] Reliably Bounding False Positives: A Zero-Shot Machine-Generated Text Detection Framework via Multiscaled Conformal Prediction](reliably_bounding_false_positives_a_zero-shot_machine-generated_text_detection_f.md)
- [\[ACL 2026\] When Personalization Tricks Detectors: The Feature-Inversion Trap in Machine-Generated Text Detection](../../ACL2026/aigc_detection/when_personalization_tricks_detectors_the_feature-inversion_trap_in_machine-gene.md)

</div>

<!-- RELATED:END -->

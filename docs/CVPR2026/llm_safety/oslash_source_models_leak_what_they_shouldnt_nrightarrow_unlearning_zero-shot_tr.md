---
title: >-
  [Paper Note] ⊘ Source Models Leak What They Shouldn't ↛: Unlearning Zero-Shot Transfer in Domain Adaptation Through Adversarial Optimization
description: >-
  [CVPR 2026][LLM Safety][Machine Unlearning] This work identifies that Source-Free Domain Adaptation (SFDA) methods inadvertently leak knowledge of source-exclusive classes to the target domain (zero-shot transfer phenome…
tags:
  - "CVPR 2026"
  - "LLM Safety"
  - "Machine Unlearning"
  - "Source Domain Privacy Leakage"
  - "Source-Free Domain Adaptation"
  - "Adversarial Optimization"
  - "Zero-Shot Transfer"
date: 2026-05-08
content_hash: 925ce3efafeef799
---

# ⊘ Source Models Leak What They Shouldn't ↛: Unlearning Zero-Shot Transfer in Domain Adaptation Through Adversarial Optimization

**Conference**: CVPR 2026
**arXiv**: [2604.08238](https://arxiv.org/abs/2604.08238)  
**Code**: [https://github.com/D-Arnav/SCADA](https://github.com/D-Arnav/SCADA)  
**Area**: Machine Unlearning / Domain Adaptation / Privacy Preservation
**Keywords**: Machine Unlearning, Source Domain Privacy Leakage, Source-Free Domain Adaptation, Adversarial Optimization, Zero-Shot Transfer

## TL;DR

This work identifies that Source-Free Domain Adaptation (SFDA) methods inadvertently leak knowledge of source-exclusive classes to the target domain (zero-shot transfer phenomenon), and proposes the SCADA-UL framework, which performs category unlearning simultaneously with domain adaptation through adversarial generation of forget samples and a rescaled labeling strategy, achieving unlearning quality approaching that of retraining from scratch.

## Background & Motivation

**Background**: Visual models are increasingly deployed across domains (e.g., from natural images to satellite imagery or medical scans), with domain adaptation being a key enabler. Source-Free Domain Adaptation (SFDA) is particularly appealing in privacy-sensitive scenarios, as it requires no access to source domain data—only the pretrained source model is exposed to the target domain, while the source data itself remains protected.

**Limitations of Prior Work**: Although the source data is protected, the source model still encodes source domain knowledge. Through empirical investigation, the authors uncover an alarming phenomenon: existing SFDA methods exhibit strong zero-shot classification capability on **source-exclusive classes** (classes present only in the source domain but absent from the target domain) after adaptation. This means that even without any target-domain samples from these categories, the adapted model still "remembers" them—source domain private information leaks into the target domain through the model.

**Key Challenge**: The original intent of SFDA is to protect source domain privacy, yet the model itself becomes the carrier of privacy leakage. Existing Machine Unlearning (MU) methods were not designed to handle distribution shift (domain shift) and therefore cannot be directly applied to the SFDA setting—unlearning operations degrade or fail under domain shift, harming normal target-domain performance.

**Goal**: (1) Formally define the source category unlearning problem in SFDA (SCADA-UL); (2) Design a method that performs unlearning concurrently with domain adaptation; (3) Extend the framework to a continual unlearning variant and a variant where the forget categories are unknown.

**Key Insight**: The authors observe that the zero-shot capability of SFDA-adapted models on source-exclusive classes stems from discriminative features encoded in the source model weights. If synthetic samples for the forget categories can be generated during adaptation, the model can be actively made to "forget" the corresponding knowledge without accessing any real source data.

**Core Idea**: Adversarial optimization is used to generate samples for the forget categories, combined with a rescaled labeling strategy, so that domain adaptation and category unlearning are performed jointly throughout the SFDA process.

## Method

### Overall Architecture

The SCADA-UL pipeline takes as input a source-pretrained classification model, unlabeled target-domain data, and a specification of the categories to be forgotten (or, in the variant setting, the forget categories are unknown). The framework jointly performs domain adaptation on the target domain via three coordinated mechanisms: (1) an adversarial sample generator synthesizes proxy samples for the forget categories in feature space; (2) a rescaled labeling strategy redistributes predicted probability mass from forget categories to retained categories, guiding the model to lose discriminative ability on the specified categories; (3) an adversarial optimization objective balances maximizing the unlearning effect against minimizing degradation on retained categories. The resulting model both adapts to the target domain and successfully forgets source-exclusive category knowledge.

### Key Designs

1. **Adversarial Forget Sample Generation**:

    - *Function*: Synthesizes proxy samples for the forget categories to be used in unlearning training, without access to real source data.
    - *Mechanism*: Class prototype information encoded in the source model is leveraged to generate samples via gradient ascent in input space, maximizing activation of the forget-category classification head. Concretely, a random noise image $x_{\text{syn}}$ is initialized and optimized via $\max_{x_{\text{syn}}} p(y_{\text{forget}} | x_{\text{syn}}; \theta)$ to produce synthetic samples for the forget categories. Although these samples are not visually realistic, they occupy the decision regions of the forget categories in feature space, which is sufficient to guide the unlearning process.
    - *Design Motivation*: Real source data is inaccessible in the SFDA setting, but the source model weights already encode sufficient categorical discriminative information. Adversarial generation "reverse-extracts" this information for unlearning purposes—an elegant strategy of using the model's own knowledge to erase its knowledge.

2. **Rescaled Labeling Strategy**:

    - *Function*: Generates appropriate "forget labels" for the synthetic samples, guiding the model to redistribute forget-category probability mass onto retained categories.
    - *Mechanism*: Rather than naively zeroing out forget-category labels (which leads to training instability), the probability mass of categories in the forget set $\mathcal{F}$ is redistributed to the retained set $\mathcal{R}$ in proportion to the relative weights of retained categories. The model thereby learns that "these features do not belong to any forget category, but should be distributed across the retained categories."
    - *Design Motivation*: Hard unlearning (suppressing forget-category probabilities to zero) disrupts the smoothness of the softmax distribution, causing gradient vanishing or training oscillation. Rescaled labeling preserves the integrity of the probability distribution, ensuring stable and correctly directed gradient signals throughout unlearning.

3. **Adversarial Optimization Framework**:

    - *Function*: Establishes an adversarial game between domain adaptation and unlearning, automatically balancing the two objectives.
    - *Mechanism*: Training involves two competing objectives—(1) the domain adaptation objective, which applies entropy minimization or pseudo-label learning to adapt the model to target-domain data; (2) the unlearning objective, which maximizes prediction uncertainty on the forget categories (or minimizes their predicted probability) to erase source-exclusive knowledge. The two objectives are balanced through alternating optimization: the model takes one adaptation step on target-domain data, followed by one unlearning step on the generated forget samples. This minimax optimization ensures that unlearning does not excessively harm adaptation performance.
    - *Design Motivation*: Domain adaptation and unlearning are inherently in tension—adaptation relies on the representational capacity of the source model, while unlearning removes part of it. Adversarial optimization provides a natural mechanism to find a Pareto-optimal solution between these two goals.

### Loss & Training

The total loss consists of three components: (1) the domain adaptation loss $\mathcal{L}_{\text{adapt}}$—a standard SFDA loss (e.g., information entropy minimization or neighborhood consistency) to adapt the model to the target domain distribution; (2) the forget loss $\mathcal{L}_{\text{forget}}$—cross-entropy computed on the generated forget samples using rescaled labels, directing the model to lose competence on forget categories; (3) the retain loss $\mathcal{L}_{\text{retain}}$—ensuring that model performance on retained categories is not harmed by the unlearning operation. Training alternates between two phases: adaptation on target-domain data, and unlearning updates on synthetic forget samples.

Two important variants are further introduced: (1) **Continual unlearning variant**—when new forget requests arrive sequentially, the model must unlearn new categories without forgetting previously unlearned categories or degrading previously retained performance; (2) **Unknown forget categories variant**—when the specific source-exclusive categories to be forgotten are not known in advance, the method automatically detects them by identifying inconsistencies between the target-domain data distribution and the model's predictions.

## Key Experimental Results

### Main Results (OfficeHome Dataset)

| Domain Pair | Method | Retained Class Acc ↑ | Forget Class Acc ↓ | Unlearning Score |
|---|---|---|---|---|
| Art → Product | Existing SFDA (SHOT) | High | High (leakage) | Poor |
| Art → Product | Existing MU + SFDA | Moderate | Moderate | Insufficient |
| Art → Product | **SCADA-UL (Ours)** | **High** | **Low (near random)** | **Near retraining** |
| Clipart → Real | Existing SFDA (SHOT) | High | High (leakage) | Poor |
| Clipart → Real | **SCADA-UL (Ours)** | **High** | **Low** | **Best** |

*Note: Experiments are conducted on all 12 domain pairs of OfficeHome. SCADA-UL consistently outperforms all baselines across all domain pairs.*

### Ablation Study

| Configuration | Retained Class Acc | Forget Class Acc (↓ better) | Notes |
|---|---|---|---|
| Full SCADA-UL | Highest | Lowest | Complete model |
| w/o adversarial sample generation | Degraded | Higher | Cannot effectively target forget-category decision regions |
| w/o rescaled labeling | Noticeably degraded | Moderate | Unstable unlearning process, damages retained categories |
| w/o adversarial optimization | Degraded | Moderate | Conflict between adaptation and unlearning unresolved |
| Random noise instead of adversarial samples | Degraded | Higher | Random samples fail to activate forget-category features effectively |

### Key Findings

- **Zero-shot leakage is real and severe**: Standard SFDA methods (e.g., SHOT, NRC) achieve zero-shot accuracy of 30–50% on source-exclusive categories, far above chance level, confirming the privacy leakage risk.
- **Adversarial sample generation is critical**: Compared to random noise, adversarially generated forget samples improve unlearning efficiency by approximately $2\times$, as they precisely target the decision regions of forget categories.
- **SCADA-UL achieves unlearning quality approaching "retraining from scratch"**: Forget-category accuracy is reduced to near-random-guess level while retained-category performance is almost entirely preserved.
- In the continual unlearning variant, the method demonstrates stable retention of previously unlearned knowledge—newly arriving forget tasks do not cause the model to "recall" already-forgotten categories.

## Highlights & Insights

- **A critical privacy blind spot is identified**: SFDA has been regarded as a source-privacy-preserving paradigm, yet this work reveals that the model itself is a privacy leakage channel. This observation is independently valuable—it demonstrates that "not accessing data" does not equate to "not leaking data information."
- **Using the model's knowledge to erase its knowledge**: The adversarial sample generation strategy is elegant. In the absence of real source data, the class prototypes encoded in the model itself are exploited to generate unlearning targets. This self-referential design principle is transferable to other privacy-protection scenarios.
- **Theory meets practice**: Beyond empirical validation, the paper provides theoretical analysis—framing source domain information leakage and the information-theoretic guarantees of the unlearning operation from an information-theoretic perspective.

## Limitations & Future Work

- Validation is currently limited to classification tasks; applicability to more complex visual tasks such as object detection and semantic segmentation remains to be explored.
- The quality of adversarial sample generation depends on the source model's discriminative strength—if the source model is weak on certain forget categories, the generated samples may not adequately cover the decision boundary.
- The unknown-forget-categories variant's detection accuracy is sensitive to the class distribution of the target domain; highly imbalanced distributions may lead to false positives.
- Experiments are primarily conducted on medium-scale datasets such as OfficeHome; large-scale (e.g., ImageNet-scale) domain adaptation unlearning requires further validation.
- The method assumes forget categories are independent, but some categories may share feature subspaces—unlearning one may inadvertently affect related categories.

## Related Work & Insights

- **vs SHOT (Liang et al. 2020)**: SHOT is a classic SFDA method that performs domain adaptation via information entropy minimization and pseudo-labeling, with no consideration of source domain information unlearning.
- **vs Machine Unlearning methods (e.g., SCRUB, Bad Teaching)**: Conventional MU methods assume a fixed data distribution; under domain shift, their unlearning effectiveness degrades significantly—unlearning operations may inadvertently remove useful target-domain knowledge as well.
- **vs Differential Privacy**: Differential privacy adds noise during training as a forward-protection mechanism; SCADA-UL provides backward protection—performing unlearning on an already-trained model.
- *Inspiration*: Analogous privacy leakage risks may exist in other model-sharing settings such as knowledge distillation and federated learning.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Identifying and formally defining the zero-shot leakage problem in SFDA (SCADA-UL) is a significant contribution; the method design (adversarial generation + rescaled labeling + minimax optimization) is natural yet effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers all 12 domain pairs, three variant settings, and multiple baselines, with comprehensive ablation and theoretical analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Problem motivation is clearly articulated; the logical chain from empirical observation to method design is complete.
- **Value**: ⭐⭐⭐⭐ Reveals a privacy blind spot in SFDA with direct implications for security-sensitive domains (medical and military imagery).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Do Vision-Language Models Leak What They Learn? Adaptive Token-Weighted Model Inversion Attacks](vlm_model_inversion_adaptive_token_weight.md)
- [\[ACL 2026\] CiPO: Counterfactual Unlearning for Large Reasoning Models through Iterative Preference Optimization](../../ACL2026/llm_safety/cipo_counterfactual_unlearning_for_large_reasoning_models_through_iterative_pref.md)
- [\[CVPR 2026\] Designing to Forget: Deep Semi-parametric Models for Unlearning](designing_to_forget_deep_semi-parametric_models_for_unlearning.md)
- [\[AAAI 2026\] TOFA: Training-Free One-Shot Federated Adaptation for Vision-Language Models](../../AAAI2026/llm_safety/tofa_training-free_one-shot_federated_adaptation_for_vision-language_models.md)
- [\[NeurIPS 2025\] Zero-Shot Robustness of Vision Language Models Via Confidence-Aware Weighting](../../NeurIPS2025/llm_safety/zero-shot_robustness_of_vision_language_models_via_confidence-aware_weighting.md)

</div>

<!-- RELATED:END -->

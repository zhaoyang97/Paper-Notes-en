---
title: >-
  [Paper Note] EgoPrivacy: What Your First-Person Camera Says About You?
description: >-
  [ICML2025][LLM Safety][egocentric vision] Introduces EgoPrivacy, the first large-scale first-person video privacy benchmark, defining three categories of privacy (demographic, individual, and situational) across seven tasks. It designs Retrieval-Augmented Attack (RAA), combining ego-to-exo retrieval and classification, to demonstrate that foundation models can infer the wearer's sensitive attributes (e.g., gender, race) with 70–80% accuracy in a zero-shot setting.
tags:
  - "ICML2025"
  - "LLM Safety"
  - "egocentric vision"
  - "privacy benchmark"
  - "demographic attack"
  - "retrieval-augmented attack"
  - "contrastive learning"
date: 2026-05-08
content_hash: fb8634c5ef9f8ba4
---

# EgoPrivacy: What Your First-Person Camera Says About You?

**Conference**: ICML2025  
**arXiv**: [2506.12258](https://arxiv.org/abs/2506.12258)  
**Code**: [GitHub](https://github.com/williamium3000/ego-privacy)  
**Area**: Privacy / AI Security  
**Keywords**: egocentric vision, privacy benchmark, demographic attack, retrieval-augmented attack, contrastive learning

## TL;DR

Introduces EgoPrivacy, the first large-scale first-person video privacy benchmark, defining three categories of privacy (demographic, individual, and situational) across seven tasks. It designs Retrieval-Augmented Attack (RAA), combining ego-to-exo retrieval and classification, to demonstrate that foundation models can infer the wearer's sensitive attributes (e.g., gender, race) with 70–80% accuracy in a zero-shot setting.

## Background & Motivation

- **Origin of Problem**: With the growing popularity of wearable cameras (AR glasses, GoPros), first-person videos are continuously collected for tasks such as activity recognition, behavioral analysis, and life-logging. Existing privacy research primarily focuses on third-party faces appearing in the frame, while the **privacy threats to the camera wearers themselves** have rarely been systematically studied.
- **Core Problem**: How much privacy about the wearer can be inferred solely from first-person videos? Can the wearer's gender, race, age, identity, scene, and time be reconstructed?
- **Limitations of Prior Work**: Existing first-person privacy datasets (FPSI, EVPR, IITMD) are extremely small in scale (6–32 individuals), cover only a single dimension of identity recognition, lack demographic annotations, and have no OOD test sets.
- **Goal**: To systematically define the attack surface of first-person video privacy and establish a comprehensive benchmark, quantifying the leakage of information by attackers with different capabilities and laying the foundation for future privacy defense.

## Method

### 1. Privacy Definitions and Task System

Wearer privacy is characterized into three main categories and seven tasks:

| Privacy Category | Task | Formulation | Evaluation Metric |
|---|---|---|---|
| Demographic Privacy | Gender / Race / Age Classification | Classification | Accuracy |
| Individual Privacy | ego-to-ego / ego-to-exo Identity Retrieval | Retrieval | HR@k |
| Situational Privacy | Scene Retrieval / Moment Retrieval | Retrieval | HR@k |

**Demographic privacy** is modeled as a classification problem:

$$\text{Acc}(\mathcal{D}; f) = \frac{1}{|\mathcal{D}|} \sum_{(\mathbf{x}, a) \in \mathcal{D}} \mathbb{1}[f(\mathbf{x}) = a]$$

**Individual / situational privacy** is modeled as a retrieval problem, measuring risk using Hit Rate@k ($HR@k$):

$$\text{HR@}k(\mathcal{D}; g) = \frac{1}{|\mathcal{D}|} \sum_{(\mathbf{x}, I) \in \mathcal{D}} \mathbb{1}[g^k(\mathbf{x}) \cap \mathcal{T}_I \neq \emptyset]$$

### 2. Threat Model (Attack Capability)

Four increasing levels of attack capability are defined:

- **Capability ⓪ (Zero-shot)**: The attacker has no training data and directly infers properties using foundation models in a zero-shot manner.
- **Capability ① (Fine-tuned)**: The attacker has access to a labeled training set to fine-tune models.
- **Capability ② (Retrieval-Augmented)**: The attacker possesses an ego-exo paired training set and an external third-person (exocentric) video pool.
- **Capability ③ (Identity-level)**: The attacker can judge whether two ego videos belong to the same identity.

### 3. Ego-Exo Joint Embedding

Supervised Contrastive Learning (SupCon) is adopted to learn a joint ego-exo embedding space:

$$L(g, g') = -\sum_{i=1}^{N} \frac{1}{|P(i)|} \sum_{k \in P(i)} \log \frac{\exp(\langle \mathbf{z}_i^E, \mathbf{z}_k^X \rangle / \tau)}{\sum_{j \in N(i)} \exp(\langle \mathbf{z}_i^E, \mathbf{z}_j^X \rangle / \tau)}$$

where $P(i)$ represents the set of positive pairs (defined according to the privacy type), $N(i)$ is the set of negative pairs, and $\tau$ is the temperature coefficient. Modifying the definition of $P(i)$ unifies individual privacy (where positive pairs are all exo videos of the same wearer) and situational privacy (where positive pairs are synchronously recorded exo clips).

### 4. Retrieval-Augmented Attack (RAA)

**Mechanism**: "Retrieve-then-predict" — leveraging ego-to-exo retrieval to compensate for the lack of facial/body visibility in first-person videos.

1. Given an ego query $\mathbf{x}^E$, an ego-exo retriever $g$ is used to retrieve the Top-$M$ most similar clips $\{\mathbf{x}_{1:M}^X\}$ from an external exo pool $\mathcal{D}^X$.
2. Privacy attributes are predicted independently using an ego classifier $f$ and an exo classifier $f'$ for each input.
3. Aggregation via voting yields the final output:

$$f^{\text{RAA}}(\mathbf{x}^E, \{\mathbf{x}_{1:M}^X\}) = \mathcal{A}\big(f(\mathbf{x}^E),\; f'(\mathbf{x}_1^X),\;\dots,\; f'(\mathbf{x}_M^X)\big)$$

The aggregation function $\mathcal{A}$ can be hard voting (majority voting) or soft voting (weighted pooling).

### 5. EgoPrivacy Benchmark Construction

- Built upon **Ego-Exo4D** (5,625 clips, 839 people, 131 scenes) and **Charades-Ego** (4,000 clips, 112 people).
- Demographic annotations were obtained via Amazon Mechanical Turk by labeling the visible wearers in exo videos. The label set includes: Gender {Female, Male}, Race {Asian, Black, White}, and Age {Young, Middle-aged, Senior}.
- Supports both ID (Ego-Exo4D train/test) and OOD (train=Ego-Exo4D, test=Charades-Ego) evaluations.

## Key Experimental Results

### Demographic Privacy Attack (OOD, Charades-Ego)

| Method | Capability | Gender | Race | Age |
|---|---|---|---|---|
| Random Chance | — | 50.00 | 33.33 | 33.33 |
| Prior (Majority Class) | — | 60.74 | 54.17 | 79.48 |
| CLIP H/14 zero-shot (ego) | ⓪ | 57.89 | 45.21 | 72.02 |
| CLIP H/14 fine-tuned (ego) | ① | 68.87 | 70.92 | 79.73 |
| CLIP H/14 + RAA | ①+② | 76.98 (+8.11) | 71.92 (+1.00) | 79.73 |
| CLIP H/14 zero-shot + RAA | ⓪+② | 67.35 (+9.46) | 60.98 (+15.77) | 76.23 (+4.21) |

**Key Findings**:
- Zero-shot foundation models infer gender/race/scenes with an accuracy of **70–80%**, far exceeding the random baseline.
- RAA improves the zero-shot attack accuracy on race by up to **+15.77%**, and gender by **+9.46%**.
- Fine-tuned ego attacks close in on exo attack performance, demonstrating that the "natural occlusion" in ego videos offers limited protection.

### Individual and Situational Privacy

- Ego-to-ego identity retrieval: Fine-tuned CLIP achieves a significantly higher HR@1 in ID evaluations compared to zero-shot, validating that hand gestures and environmental cues are sufficient to expose identity.
- Ego-to-exo identity retrieval: SupCon training achieves a substantial increase in HR@1, indicating that cross-view ego-exo identity association is indeed learnable.
- Scene/moment retrieval: Foundation models exhibit strong zero-shot scene matching capabilities, which are further improved via fine-tuning.

## Highlights & Insights

1. **First systematic ego privacy benchmark**: Refines wearer privacy into three categories and seven tasks, expanding from prior datasets (6–32 people) to 839 people and 131 scenes, filling a critical gap.
2. **Novel and practical RAA attack strategy**: Simulates real-world scenarios (where surveillance cameras and ego devices capture the same subject simultaneously), bridging both perspectives via ego-to-exo retrieval, significantly enhancing attack success rates without requiring direct face matching.
3. **High threat even in zero-shot settings**: Open-source foundation models can recover sensitive demographic attributes with no additional data, raising concerns for privacy regulations and device designs.
4. **Unified formulation**: Unifies the two primary retrieval tasks (individual/situational) by varying the definition of positive pairs in the SupCon loss, providing elegance and simplicity.

## Limitations & Future Work

1. **Coarse label taxonomy**: Gender is limited to Male/Female, race to Asian/Black/White (subjectively judged by annotators), and age to three brackets, leaving out substantial diversity.
2. **Dataset bias**: Ego-Exo4D predominantly features laboratory or activity-specific settings, while Charades-Ego is limited to home interiors, lacking high-frequency ego scenarios like outdoor environments, city streets, or driving.
3. **Focus on attack evaluation only**: Corresponding defensive methods or privacy-preserving strategies (e.g., differentially private representations, adversarial perturbations) are not proposed, limiting direct guidance for real-world deployment.
4. **Strong assumptions in RAA**: RAA requires the attacker to possess an exo video pool containing the target identity, whose practical accessibility remains to be further discussed.
5. **Static frame sampling**: Primarily utilizes frame-level features, not fully exploiting dynamic privacy cues such as temporal actions or gaits.

## Related Work & Insights

- Complements social media privacy benchmarks like VISPR and PIPA, extending research to egocentric perspectives.
- The "retrieval-augmented" concept of RAA is analogous to RAG in NLP, potentially inspiring future visual privacy attack and defense frameworks.
- Provides quantitative references for the privacy design of wearable device manufacturers (e.g., Meta Ray-Ban, Apple Vision Pro).

## Rating

- Novelty: ⭐⭐⭐⭐ — For systematically defining the attack surface of wearer privacy in ego videos for the first time, and the novelty of the RAA strategy.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Seven tasks × four threat levels × ID/OOD comprehensive testing, backed by extensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ — Clear problem definitions, unified formulation, and rich illustrations.
- Value: ⭐⭐⭐⭐ — Fills a critical gap in ego-privacy research, serving as a key reference for wearable privacy designs and regulatory standards.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Attention Smoothing Is All You Need For Unlearning](../../ICLR2026/llm_safety/attention_smoothing_is_all_you_need_for_unlearning.md)
- [\[ICML 2025\] Improving Your Model Ranking on Chatbot Arena by Vote Rigging](improving_your_model_ranking_on_chatbot_arena_by_vote_rigging.md)
- [\[ICML 2025\] Is Your Model Fairly Certain? Uncertainty-Aware Fairness Evaluation for LLMs](is_your_model_fairly_certain_uncertainty-aware_fairness_evaluation_for_llms.md)
- [\[ACL 2026\] Look Twice before You Leap: A Rational Framework for Localized Adversarial Anonymization](../../ACL2026/llm_safety/look_twice_before_you_leap_a_rational_framework_for_localized_adversarial_anonym.md)
- [\[ACL 2026\] Forget What Matters, Keep the Rest: Selective Unlearning of Informative Tokens](../../ACL2026/llm_safety/forget_what_matters_keep_the_rest_selective_unlearning_of_informative_tokens.md)

</div>

<!-- RELATED:END -->

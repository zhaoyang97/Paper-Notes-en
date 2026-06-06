---
title: >-
  [Paper Note] Experience with Single Domain Generalization in Real World Medical Imaging Deployments
description: >-
  [AAAI2026][Medical Imaging][Single domain generalization] This paper proposes the DL+EKE framework, which integrates domain-invariant expert knowledge with deep learning to address rare class single domain generalization…
tags:
  - "AAAI2026"
  - "Medical Imaging"
  - "Single domain generalization"
  - "rare class detection"
  - "expert knowledge integration"
  - "medical imaging deployment"
  - "diabetic retinopathy"
  - "epileptic focus localization"
  - "coronary artery disease"
date: 2026-05-08
content_hash: 4fd6c940fb8cdb8a
---

# Experience with Single Domain Generalization in Real World Medical Imaging Deployments

**Conference**: AAAI2026
**arXiv**: [2601.16359](https://arxiv.org/abs/2601.16359)  
**Authors**: Ayan Banerjee (ASU), Komandoor Srivathsan, Sandeep K.S. Gupta (ASU)
**Code**: Not released  
**Area**: Medical Imaging
**Keywords**: Single domain generalization, rare class detection, expert knowledge integration, medical imaging deployment, diabetic retinopathy, epileptic focus localization, coronary artery disease

## TL;DR

This paper proposes the DL+EKE framework, which integrates domain-invariant expert knowledge with deep learning to address rare class single domain generalization (SDG) in medical imaging. The approach significantly outperforms state-of-the-art SDG methods across three real-world deployment scenarios: diabetic retinopathy (DR) grading, resting-state fMRI seizure onset zone (SOZ) localization, and stress ECG-based coronary artery disease (CAD) detection.

## Background & Motivation

### Importance of Domain Generalization in Medical Imaging
Domain Generalization (DG) is a core capability for deployment-ready AI. In medical imaging, factors such as scanner hardware differences, acquisition protocol variations, and patient demographics cause severe domain shift across clinical centers. Multi-source domain generalization (MSDG) and domain adaptation (DA) require data from multiple centers, raising privacy and data acquisition challenges. Single domain generalization (SDG), which requires only a single source domain, is a more practical alternative—yet it is particularly challenging in rare class scenarios.

### Special Difficulties of Rare Class Detection
Clinically critical diagnostic tasks often correspond to rare classes, such as stage-5 diabetic retinopathy, seizure onset zones (SOZ) in rs-fMRI (comprising only 5–10% of independent components), and CAD-positive findings in stress ECGs. Rare classes exhibit four attributes: Discrimination, Scarcity, Significance, and Overlap. Due to extremely limited observed samples, purely data-driven DL methods cannot adequately estimate the rare class distribution. By the Cramér–Rao lower bound, a higher class entropy $\theta_r$ for a rare class implies greater parameter estimation error.

### Motivating Evidence from Real Deployment Failures
When deploying a ViT model at Mayo Clinic for stress ECG-based CAD detection, the model achieved PPV of 79% and NPV of 81.8% on 2010 training/test data, but PPV collapsed to 46% and NPV to 49% on 2025 blind test data. The root cause was a 2012 triage policy change: previously, only patients with S-T depression were referred for invasive coronary angiography (ICA); thereafter, referrals were made even without S-T depression. ECGs from CAD-positive patients in the new data contained expert-knowledge features such as inter-lead relationships that were absent in the old data, causing severe domain shift.

## Core Problem

How can reliable generalization to rare classes in medical imaging be achieved when only a single source domain is available? State-of-the-art SDG methods demonstrate insufficient performance on rare classes, motivating a new paradigm to address the inherent limitations of purely data-driven approaches in estimating rare class distributions.

## Method

### Rare Class Definition and Quantification
Class-wise entropy is used to quantify rarity. Given representation $x_i$ of observation $y_i$, the density function is defined as:

$$\lambda(x_i) = \frac{1}{|Q(x_i)|} \sum_{j=1}^{|Q(x_i)|} \frac{1}{\text{dist}(x_i, x_j)}$$

The class-averaged density is $\gamma(x_i) = \lambda(x_i) / \sum_{j=1}^{|c_r|} \lambda(x_j)$, and class entropy is:

$$\theta_r = \sum_{i=1}^{|c_r|} (-\gamma(x_i) \log_2 \gamma(x_i))$$

A class $c_r$ is deemed rare if $\theta_r^z > \theta_M^z + \sigma_\theta^z$ or $\theta_r^z < \theta_M^z - \sigma_\theta^z$.

### RareSaGe Algorithm Framework
RareSaGe (RARE class classification for Single domain Generalization) proceeds as follows:

1. **Rare class identification**: CLIP is used to extract class-agnostic feature embeddings; class entropy is computed for each class to identify rare classes satisfying Definition 2.
2. **Overlap class localization**: Cosine similarity between CLIP features of the rare class and each non-rare class is computed to identify the most similar "overlap class" $c_o$.
3. **DL machine orchestration**: DL techniques (ViT/CNN/LVM) distinguish overlap vs. non-overlap classes.
4. **Knowledge machine orchestration**: An expert-knowledge-based classifier (SVM + logical rules) distinguishes rare vs. non-rare classes.
5. **Label predictor**: DL and EKE are integrated—DL first classifies overlap/non-overlap; for non-overlap samples, the EKE label is adopted directly; for samples classified as overlap by DL but classified as rare by EKE with high confidence (> $t_c$ = 0.9), the rare class label overrides the DL prediction.

### Expert Knowledge for SOZ Detection
Two types of expert knowledge are utilized: (1) anatomical knowledge—brain region locations extracted via image processing; and (2) SOZ-specific knowledge—literature-based logical conjunction rules:

$$\kappa_{SOZ} = p_1 \wedge \neg p_s \wedge p_a \wedge [p_g \wedge (\neg p_w \vee (p_w \wedge p_v))]$$

where $p_1$ = single activation cluster, $p_s$ = sinusoidal domain sparsity, $p_a$ = activelet domain sparsity, $p_g$ = gray matter activation, $p_w$ = white matter overlap, $p_v$ = vascular region overlap.

### Expert Knowledge Integration for CAD Detection
The K1 configuration integrates the highest tier of expert knowledge: five expert-designated leads are selected (rather than all 12 leads), and only maximum MET level data is used. Ablation configurations K2–K4 validate the contribution of expert knowledge components.

## Key Experimental Results

### SOZ Cross-Trial Validation (Cross-Center SDG)

| Method | Accuracy | Precision | Sensitivity | F1 | Mean F1 | Ablation |
|--------|----------|-----------|-------------|----|---------|----------|
| Pre-trained ViT (A→B) | 64.5% | 86.9% | 71.4% | 78.4% | 77.2% | DL only |
| Pre-trained ViT (B→A) | 61.5% | 91.4% | 65.3% | 76.1% | — | DL only |
| Knowledge (A→B) | 83.8% | 89.6% | 92.8% | 91.2% | 78.9% | Knowledge only |
| Knowledge (B→A) | 50.0% | 89.6% | 53.0% | 66.6% | — | Knowledge only |
| **DL+EKE (A→B)** | **90.3%** | **90.3%** | **100%** | **94.9%** | **90.2%** | DL+EKE |
| **DL+EKE (B→A)** | **75.0%** | **92.8%** | **79.5%** | **85.6%** | — | DL+EKE |

### CAD Detection Results (Stress ECG)

| Metric | Method | Validation | Test (2010) | Blind Test (2025) |
|--------|--------|------------|-------------|-------------------|
| PPV | ViT | 80.4% | 79.0% | 46.0% |
| PPV | K1 (DL+EKE) | 91.2% | 91.2% | **75.0%** |
| NPV | ViT | 83.0% | 81.8% | 49.0% |
| NPV | K1 (DL+EKE) | 93.0% | 93.0% | **76.0%** |

K1 achieves PPV and NPV gains of 29% and 27% respectively over the pure ViT baseline on the blind test set, with a 5-fold cross-validation ROC AUC of 92.2 (±1.1).

## Highlights & Insights

- **Real-world deployment validation**: Unlike purely benchmark-driven academic work, this paper conducts genuine cross-center deployment validation at Mayo Clinic (CAD) and UNC (SOZ), offering high practical reference value.
- **Rigorous theoretical grounding**: The relationship between the Cramér–Rao lower bound and class entropy provides an information-theoretic explanation for why purely data-driven DL methods inevitably fail on rare classes.
- **General framework design**: The RareSaGe framework is generalizable—any application exhibiting rare class attributes can be instantiated within it, as demonstrated across three distinct domains: DR, SOZ, and CAD.
- **Clinical interaction feedback**: The paper documents real questions raised by Mayo Clinic cardiology experts following blind testing (LIME attention map interpretation, data quality assessment, sex-based disparity analysis), illustrating the authentic challenges of deploying AI in clinical settings.

## Limitations & Future Work

- **High cost of expert knowledge encoding**: Each application domain requires substantial manual effort to extract and encode domain expert knowledge, limiting scalability.
- **Sensitivity to knowledge override threshold $t_c$**: ROC analysis shows that the choice of the knowledge override threshold significantly affects performance, and no automated selection strategy is provided.
- **Evaluation limited to classification tasks**: Segmentation, detection, and other medical imaging task types are not addressed.
- **Difficulty anticipating cross-center variation**: The paper candidly acknowledges that many cross-center variations are difficult to anticipate in advance, requiring sustained collaboration between clinical and engineering teams.

## Related Work & Insights

- **State-of-the-art SDG methods** (EPVT, DiMix, ERM++, etc.): Performance is moderate on the DR benchmark; DL+EKE significantly outperforms these methods in both across-trial and aggregate-trial evaluations.
- **Knowledge-augmented DL** (Daniele & Serafini 2019): Symbolic constraints are imposed on outputs, but intra-class variability cannot be modeled; DL+EKE resolves knowledge ambiguity and conflict through its orchestration strategy.
- **CLIP-DR** (Yu et al. 2024, Li et al. 2022): Language-vision pretraining methods each focus on different aspects of DR grading; DL+EKE achieves a 4.5% F1 improvement in the aggregate trial setting.

## Personal Takeaways

The integration of expert knowledge with deep learning is an underappreciated direction. In rare class scenarios, the marginal return of additional data diminishes rapidly, whereas domain-invariant expert knowledge—such as anatomical rules and clinical diagnostic criteria—provides priors that data-driven methods cannot learn. This paradigm is transferable to any real-world deployment scenario characterized by severe class imbalance.

## Rating

- Novelty: ⭐⭐⭐⭐ — Integrating expert knowledge with DL for SDG rare class problems is a fresh perspective, though individual components are relatively standard.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Three real-world deployment scenarios plus the DR benchmark, with cross-center validation and blind testing; exceptionally comprehensive.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured, with thorough documentation of deployment experience and detailed appendices.
- Value: ⭐⭐⭐⭐ — Offers important guidance for real-world medical AI deployment and addresses a gap in SDG for rare classes.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Human Knowledge Integrated Multi-modal Learning for Single Source Domain Generalization](../../CVPR2026/medical_imaging/human_knowledge_integrated_multi-modal_learning_for_single_source_domain_general.md)
- [\[ICML 2026\] Evidential Reasoning Advances Interpretable Real-World Disease Screening](../../ICML2026/medical_imaging/evidential_reasoning_advances_interpretable_real-world_disease_screening.md)
- [\[NeurIPS 2025\] MIRA: Medical Time Series Foundation Model for Real-World Health Data](../../NeurIPS2025/medical_imaging/mira_medical_time_series_foundation_model_for_real-world_health_data.md)
- [\[AAAI 2026\] CountVid: Open-World Object Counting in Videos](open-world_object_counting_in_videos.md)
- [\[CVPR 2026\] From Adaptation to Generalization: Adaptive Visual Prompting for Medical Image Segmentation](../../CVPR2026/medical_imaging/apex_adaptive_visual_prompting.md)

</div>

<!-- RELATED:END -->

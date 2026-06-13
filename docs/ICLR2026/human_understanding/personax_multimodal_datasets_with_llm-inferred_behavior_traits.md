---
title: >-
  [Paper Note] PersonaX: Multimodal Datasets with LLM-Inferred Behavior Traits
description: >-
  [ICLR2026][Human Understanding][multimodal dataset] This work constructs the PersonaX multimodal dataset (comprising LLM-inferred Big Five behavior traits, facial embeddings…
tags:
  - "ICLR2026"
  - "Human Understanding"
  - "multimodal dataset"
  - "behavior traits"
  - "Big Five"
  - "causal representation learning"
  - "LLM"
  - "identifiability"
date: 2026-05-08
content_hash: 050390a81852b1ab
---

# PersonaX: Multimodal Datasets with LLM-Inferred Behavior Traits

**Conference**: ICLR2026
**arXiv**: [2509.11362](https://arxiv.org/abs/2509.11362)  
**Code**: [lokali/PersonaX](https://github.com/lokali/PersonaX)  
**Area**: Human Understanding
**Keywords**: multimodal dataset, behavior traits, Big Five, causal representation learning, LLM, identifiability

## TL;DR
This work constructs the PersonaX multimodal dataset (comprising LLM-inferred Big Five behavior traits, facial embeddings, and biographical metadata) and proposes a two-level analysis framework: structured independence testing combined with unstructured causal representation learning (with theoretical identifiability guarantees), revealing cross-modal causal structures.

## Background & Motivation
- Understanding human behavior traits is critical for human-computer interaction, computational social science, and personalized AI systems; however, existing datasets rarely provide behavioral descriptors alongside complementary modalities such as facial attributes and biographical information.
- Behavior traits differ from personality in psychology (i.e., internal dispositions): they are externally observable behavioral patterns inferable from public information at scale in an ethical manner.
- Advances in LLMs have made Big Five-based behavior trait assessment reliable under carefully designed prompts, yet systematic resources for cross-modal and causal analysis remain lacking.
- Existing multimodal datasets (e.g., YouTube-Vlogs, MuPTA, MDPE) generally lack explicit textual trait descriptions or cross-modal interpretive frameworks.

## Core Problem
1. How to construct a large-scale, multimodal, privacy-preserving behavior trait dataset?
2. What statistical dependencies exist between behavior traits and facial attributes or biographical features?
3. How to learn latent variables and their causal mechanisms from unstructured multimodal data with identifiability guarantees?

## Method

### Dataset Construction
PersonaX comprises two complementary datasets:

**AthlePersona**: Biographical information (name, date of birth, nationality), physical measurements (height, weight), and facial images of 4,181 male professional athletes are collected from the official websites of seven major sports leagues (NBA, NFL, NHL, ATP, PGA, Premier League, Bundesliga). Nationality is geocoded into latitude-longitude coordinates.

**CelebPersona**: Based on the CelebA dataset, the names of 9,444 public figures are linked to WikiData entities to retrieve biographical details. From CelebA's original 40 attributes, 10 that reflect stable appearance features (e.g., Big Nose, High Cheekbones) are retained, while attributes subject to short-term variation (e.g., Heavy Makeup) are removed.

Each record integrates three components:
1. Behavior trait textual descriptions and Big Five scores inferred by three high-performing LLMs
2. Facial images (released as 1,024-dimensional embeddings) with attribute annotations
3. Structured biographical metadata

### LLM Selection and Prompt Design
- Ten state-of-the-art LLMs are systematically evaluated across metrics including generation time, missing rate, hesitation rate, privacy compliance, output format, contextual consistency, and factual accuracy.
- ChatGPT-4o achieves the best overall performance (OS=0.96), followed closely by Gemini2.5-Pro and Llama-4.
- Experiments compare prompt variants including numeric/textual output and 3-point/5-point rating scales; the 3-point numeric scale yields the lowest variability.
- ChatGPT-4o-Latest, Gemini-2.5-Pro, and Llama-4-Maverick are ultimately selected for data generation.

### Privacy Protection
Raw images and texts are not released. Facial images are converted to 1,024-dimensional ImageBind embeddings, and texts to 3,584-dimensional gte-Qwen2 embeddings; both undergo additional invertible transformations for obfuscation. Categorical variables are converted to indices.

### Level I: Statistical Independence Testing on Structured Data
- Big Five trait scores are aggregated by median after removing "0" (insufficient information) entries.
- Five independence testing methods are applied: KCI, RCIT, HSIC (nonparametric) and Chi-square, G-square (for discrete variables).
- Dependencies are determined at a significance level of $p < 0.05$.

### Level II: Causal Representation Learning (CRL) on Unstructured Data

**Causal Model**: Let $\mathbf{x} = [\mathbf{x}_1, \dots, \mathbf{x}_M]$ denote observations across $M$ modalities, $\mathbf{z} = [\mathbf{z}_1, \dots, \mathbf{z}_M]$ the causally relevant latent variables, and $\mathbf{s}$ the cross-modal shared latent variables. The data generating process includes:
- Latent causal relationships: $z_{m,i} = g_{z_{m,i}}(\text{Pa}(z_{m,i}), \mathbf{s}, \epsilon_{m,i})$
- Generative functions: $\mathbf{x}_m = g_{\mathbf{x}_m}(\mathbf{z}_m, \boldsymbol{\eta}_m)$

**Identifiability Theory**: Under four mild assumptions, the identifiability of latent variables in the multimodal multi-measurement setting is proved — for the same observation $\mathbf{x}$, each estimated latent component $\hat{z}_{m,i}$ is equivalent to the ground-truth $z_{m,i}$ up to an invertible mapping.

**Network Training**: The loss function consists of three terms:
- Reconstruction loss $\mathcal{L}_{\text{Recon}}$: MSE reconstruction of each modality's observations
- Independence constraint $\mathcal{L}_{\text{Ind}}$: KL divergence aligning latent variable distributions to an isotropic Gaussian prior
- Sparsity regularization $\mathcal{L}_{\text{Sp}}$: L1 norm constraining a learnable adjacency matrix (implemented via normalizing flows)

Total loss: $\mathcal{L} = \alpha_{\text{Recon}} \mathcal{L}_{\text{Recon}} + \alpha_{\text{Ind}} \mathcal{L}_{\text{Ind}} + \alpha_{\text{Sp}} \mathcal{L}_{\text{Sp}}$

## Key Experimental Results

### Synthetic Experiments (Colored MNIST + Fashion MNIST)

| Method | R² | MCC |
|---|---|---|
| BetaVAE | Low | Low |
| MCL | Low | Low |
| MMCRL | 0.90 | 0.85 |
| **PersonaX (Ours)** | **0.96** | **0.92** |

### Independence Testing Findings
- **CelebPersona**: Gender and occupation exhibit strong dependencies with nearly all trait scores; facial features (e.g., pointed nose, high cheekbones) are significantly associated with trait scores.
- **AthlePersona**: Birth year and league affiliation are stronger sources of dependency; height and weight show consistent but moderate associations.
- Geographic variables (latitude and longitude) exhibit comparable moderate dependencies in both datasets.

### Causal Graph Analysis (AthlePersona)
The causal graph learned from real data reveals:
- A bidirectional relationship between shared factors $S_1$ (mindset) and $S_2$ (culture)
- Cross-modal causal links: confidence ($Z_{2,1}$) → facial expression ($Z_{1,4}$); emotional stability ($Z_{2,3}$) → grooming ($Z_{1,2}$)
- A sequential pathway among image latent variables: skin tone → attractiveness → facial expression

## Highlights & Insights
- **The first large-scale multimodal dataset unifying LLM-inferred behavior traits with facial embeddings and biographical metadata**, filling a gap in existing resources.
- **The two-level analysis framework is elegantly designed**: the structured level uses statistical testing to reveal dependencies, while the unstructured level employs CRL to learn causal mechanisms, with each level complementing the other.
- The proposed CRL method provides **new identifiability guarantees in the multimodal multi-measurement setting**, extending prior theory.
- The LLM selection process is systematic and rigorous, evaluating ten models across eight dimensions.
- Privacy protection measures are thorough: only embeddings with invertible transformations are released, with no raw data exposed.

## Limitations & Future Work
- **Population bias**: AthlePersona contains only male athletes, and CelebPersona skews toward affluent, high-profile individuals, limiting general representativeness.
- **Lack of temporal stability**: Behavior traits are dynamic, yet the data are inferred from static public information without longitudinal tracking.
- **Reliability of LLM inference**: Despite multi-model voting improving robustness, LLM-based behavior trait assessment remains inherently subjective.
- Future work may expand to additional data sources, include female athletes, and cover more diverse populations.
- Semantic interpretation of latent variables in the causal graph relies on post-hoc guidance from independence testing results and is not fully automated.

## Related Work & Insights

| Dataset | Modalities | Behavior Traits | Trait Framework | Causal Analysis |
|---|---|---|---|---|
| SALSA | Video + sensors | Indirect | None | None |
| YouTube-Vlogs | Video + audio | Impression ratings | Big Five | None |
| MuPTA | Video + audio + physiology | Yes | Big Five | None |
| MDPE | Multimodal | Personality + affect | Big Five | None |
| **PersonaX** | **Image emb. + text emb. + biography** | **LLM-inferred** | **Big Five** | **Yes (CRL + identifiability)** |

PersonaX distinguishes itself by: (1) being the largest in scale (9,444 + 4,181 subjects), (2) being the only dataset providing explicit LLM-inferred behavioral trait texts, and (3) being the only dataset incorporating causal representation learning analysis with theoretical guarantees.

This work demonstrates the feasibility and standardization of LLMs as tools for large-scale behavioral assessment, generalizable to other social science contexts. The multimodal multi-measurement CRL framework is applicable to any scenario with multi-view, multi-instance data (e.g., repeated scans in medical imaging). The privacy-preserving scheme (embeddings + invertible transformations) offers a practical paradigm for releasing sensitive data. The independence testing results reveal systematic differences in the information channels influencing behavior traits of celebrities versus athletes, with implications for personalized AI design.

## Rating
- Novelty: ⭐⭐⭐⭐ — The combination of a multimodal behavior trait dataset and new CRL theory is original.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Validated on both synthetic and real data with comprehensive independence testing.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured with a clearly articulated two-level analysis framework.
- Value: ⭐⭐⭐⭐ — Both the dataset and methodology offer long-term value to the multimodal causal inference community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] LaMoGen: Language to Motion Generation Through LLM-Guided Symbolic Inference](../../CVPR2026/human_understanding/lamogen_language_to_motion_generation_through_llm-guided_symbolic_inference.md)
- [\[ICML 2026\] Efficient, Validation-Free Intrinsic Quality Estimation for Large-Scale Face Recognition Datasets](../../ICML2026/human_understanding/efficient_validation-free_intrinsic_quality_estimation_for_large-scale_face_reco.md)
- [\[CVPR 2026\] FusionAgent: A Multimodal Agent with Dynamic Model Selection for Human Recognition](../../CVPR2026/human_understanding/fusionagent_a_multimodal_agent_with_dynamic_model_selection_for_human_recognitio.md)
- [\[CVPR 2026\] Team LEYA in 10th ABAW Competition: Multimodal Ambivalence/Hesitancy Recognition Approach](../../CVPR2026/human_understanding/team_leya_in_10th_abaw_competition_multimodal_ambi.md)
- [\[AAAI 2026\] PA-FAS: Towards Interpretable and Generalizable Multimodal Face Anti-Spoofing via Path-Augmented Reinforcement Learning](../../AAAI2026/human_understanding/pa-fas_towards_interpretable_and_generalizable_multimodal_face_anti-spoofing_via.md)

</div>

<!-- RELATED:END -->

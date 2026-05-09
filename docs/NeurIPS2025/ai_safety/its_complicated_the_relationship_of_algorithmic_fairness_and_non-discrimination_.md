---
title: >-
  [Paper Note] It's Complicated: The Relationship of Algorithmic Fairness and Non-Discrimination Provisions for High-Risk Systems in the EU AI Act
description: >-
  [NeurIPS 2025][AI Safety][EU AI Act] This paper systematically analyzes the complex relationship between the non-discrimination provisions for high-risk AI systems in the EU AI Act (AIA) and the field of algorithmic fairness in machine learning. It reveals critical gaps in the legal text concerning input-side bias detection, the absence of output-side protections, and standardization challenges, providing a foundational framework for interdisciplinary collaboration between computer science and law.
tags:
  - NeurIPS 2025
  - AI Safety
  - EU AI Act
  - High-Risk AI Systems
  - Algorithmic Fairness
  - Anti-Discrimination Law
  - Standardization
date: 2026-05-08
content_hash: f7cb249d01830954
---

# It's Complicated: The Relationship of Algorithmic Fairness and Non-Discrimination Provisions for High-Risk Systems in the EU AI Act

**Conference**: NeurIPS 2025
**arXiv**: [2501.12962](https://arxiv.org/abs/2501.12962)
**Code**: None
**Area**: AI Safety / Algorithmic Fairness / Legal Policy
**Keywords**: EU AI Act, High-Risk AI Systems, Algorithmic Fairness, Anti-Discrimination Law, Standardization

## TL;DR

This paper systematically analyzes the complex relationship between the non-discrimination provisions for high-risk AI systems in the EU AI Act (AIA) and the field of algorithmic fairness in machine learning. It reveals critical gaps in the legal text concerning input-side bias detection, the absence of output-side protections, and standardization challenges, providing a foundational framework for interdisciplinary collaboration between computer science and law.

## Background & Motivation

**Background**: The European Union adopted in 2024 the world's first comprehensive AI regulation—the AI Act (AIA)—establishing a risk-tiered regulatory framework that classifies AI systems into four levels: unacceptable risk, high risk, limited risk, and minimal risk. Concurrently, algorithmic fairness in machine learning has matured into an established research area, encompassing a rich set of fairness metrics (e.g., statistical parity, equalized odds) and bias mitigation methods across three stages: pre-processing, in-training, and post-processing.

**Limitations of Prior Work**: Although the AIA contains provisions related to non-discrimination, a significant gap exists between legal language and technical concepts. Legal scholars struggle to grasp the technical implementation of algorithmic fairness, while computer scientists find it difficult to parse the precise meaning of legal text. More critically, the AIA provides no explicit definition of "bias," and its provisions primarily address the input side of training data, with little direct regulation of discriminatory algorithmic outputs.

**Key Challenge**: The AIA is fundamentally a product safety regulation rather than an individual rights protection law. As a result, its non-discrimination provisions are oriented toward product compliance obligations for AI providers rather than toward safeguarding the rights of affected individuals. Furthermore, the mathematical incompatibility among different fairness metrics means that the AIA's choice of metric will directly affect compliance determinations.

**Goal**: (1) To provide an accessible introduction to EU non-discrimination law for computer-science-oriented researchers; (2) to systematically analyze the relationship between the AIA's non-discrimination provisions for high-risk systems and algorithmic fairness; (3) to examine the standardization process and the interaction between traditional non-discrimination law and the AIA.

**Key Insight**: From an interdisciplinary perspective, the paper conducts a provision-by-provision analysis of AIA articles related to non-discrimination (Articles 9, 10, 13, 15, etc.), mapping legal text onto technical concepts from algorithmic fairness.

**Core Idea**: The AIA's non-discrimination provisions focus primarily on input-side bias detection while neglecting the output side. As a product safety law, the AIA lacks an individual rights protection dimension and must therefore operate in conjunction with traditional non-discrimination law.

## Method

### Overall Architecture

The paper employs legal text analysis, beginning with a scan of the full AIA for terms related to "discrimination," "fundamental right," "fairness," and "bias" to locate core provisions. It then analyzes each provision for its technical implications and implementation challenges, before discussing the standardization process and interactions with existing law.

### Key Designs

1. **Mapping the EU Non-Discrimination Law Framework (Section 2.1)**:

    - **Function**: Establishes foundational legal concepts for technical researchers.
    - **Mechanism**: EU non-discrimination law distinguishes three types of discrimination—**direct discrimination** (differential treatment based on a protected attribute, requiring no intent), **indirect discrimination** (facially neutral rules that produce discriminatory effects, defensible via legitimate aims), and **intersectional discrimination** (discrimination arising from the intersection of multiple protected attributes, e.g., gender × age). Intersectional discrimination is particularly relevant to AI systems, as models use multidimensional inputs and discrimination may emerge only at the intersection of attributes.
    - **Design Motivation**: Technical researchers are generally unfamiliar with legal classifications of discrimination, which directly affects the interpretation of AIA provisions.

2. **In-Depth Analysis of Non-Discrimination Provisions for High-Risk Systems (Section 3.3)**:

    - **Function**: Maps AIA legal provisions onto the technical framework of algorithmic fairness.
    - **Mechanism**: Three core provisions are identified:
      - **Article 10(2)(f)**: Requires bias checks on training, validation, and testing data, but only on the input side, with "bias" left undefined.
      - **Article 10(2)(g)**: Requires bias detection, prevention, and mitigation measures—seemingly drawing on fairness metrics—but likewise applies only to input data.
      - **Article 15(4)**: The only provision addressing the output side, yet limited to feedback-loop scenarios in continuously learning systems.
    - **Design Motivation**: To expose the asymmetry between input-side and output-side protections in the AIA, which may give rise to systematic gaps in protection.

3. **Analysis of Standardization and Interaction with Traditional Law (Sections 4–5)**:

    - **Function**: Discusses critical pathways and potential issues for implementing the AIA.
    - **Mechanism**: The specific technical requirements of the AIA will be determined through the CEN/CENELEC standardization process, but standardization work must remain within the boundaries set by the legal text. If the law requires only input-side bias analysis without addressing the output side, standardization may not be able to exceed that scope. At the same time, while the AIA as *lex specialis* may take precedence over general law, traditional EU non-discrimination law (e.g., the Racial Equality Directive 2000/43/EC) and the Charter of Fundamental Rights (CFR) continue to apply in areas not covered by the AIA, such as output-side discrimination.
    - **Design Motivation**: To prevent the standardization process from generating a false sense of security, and to underscore the importance of multi-layered legal protection.

## Key Experimental Results

### Summary of Main Findings

| AIA Provision | Focus | Relation to Algorithmic Fairness | Key Gap |
|---|---|---|---|
| Article 9 AIA | Risk management system | Indirect (via fundamental rights) | Does not directly mention discrimination |
| Article 10(2)(f) | Training data bias checks | Direct | Input-side only; "bias" undefined |
| Article 10(2)(g) | Bias mitigation measures | Direct | Methods unspecified; standards pending |
| Article 15(4) | Output bias (feedback loops) | Direct | Limited to continuously learning systems |
| Article 10(5) | Permits processing of special personal data | Indirect | Restricted to high-risk regimes |

### Analysis of Legal Protection Coverage

| Protection Dimension | Covered by AIA? | Covered by Traditional Non-Discrimination Law? | Notes |
|---|---|---|---|
| Training data bias | ✓ (Art. 10) | Indirectly | Primary coverage area of the AIA |
| Discriminatory algorithmic output | △ (feedback loops only) | ✓ | Significant gap in the AIA |
| Post-deployment impact assessment | Public bodies only (Art. 27) | ✓ | Private entities not bound |
| Intersectional discrimination | Unclear | Contested | Not yet explicitly recognized by ECJ |

### Key Findings
- The term "bias" is never defined in the AIA, leaving substantial interpretive latitude and uncertainty for subsequent standardization work.
- As a product safety regulation, the AIA frames its non-discrimination protections as compliance obligations for providers rather than as rights held by individuals—a sharp contrast to the GDPR.
- The incompatibility of fairness metrics (Chouldechova's theorem; Kleinberg's impossibility results) implies that satisfying one metric may violate another, yet the AIA does not specify which metric to use.

## Highlights & Insights
- **The discovery of input–output protection asymmetry** is the paper's most central insight: Article 10 regulates only training data bias, while Article 15 covers only feedback-loop outputs, meaning that discriminatory outputs from non-continuously-learning systems are largely unregulated under the AIA framework. This finding has direct implications for AI compliance practice.
- **The invocation of the "fairness hacking" concept** highlights that, because providers may freely choose among fairness metrics, they can select the metric most favorable to themselves to claim compliance (d-hacking). This underscores the need for the standardization process to determine metric selection strategies upfront.
- **The application of the *lex specialis* framework** clearly delineates the relationship between the AIA and traditional non-discrimination law: the AIA takes precedence in areas it covers, while general law supplements it in areas it does not. This approach is transferable to the analysis of other emerging technology regulations.

## Limitations & Future Work
- The paper focuses on provisions for high-risk systems, with limited discussion of non-discrimination regulation for general-purpose AI (GPAI) models, even though LLMs fall within the GPAI category.
- The analysis is grounded primarily in legal text and lacks empirical case studies. As the AIA only began its phased implementation in 2025, how academic research translates into standardization practice remains to be seen.
- The paper does not engage deeply with the particular challenges of fairness metrics in the LLM era (e.g., measuring bias in generated text), offering only brief remarks on the topic.
- No concrete proposals are offered for how to technically fulfill the input data bias check required by Article 10(2)(f).

## Related Work & Insights
- **vs. Wachter et al. (2021)**: They proposed metrics linking algorithmic fairness to ECJ case law, but published before the AIA's adoption and therefore without analysis of AIA provisions.
- **vs. Weerts et al. (2023)**: The closest work in terms of computer science–law interaction, but without focus on the AIA.
- **vs. Bosoer et al. (2023)**: Also analyzed the AIA's non-discrimination provisions, but only on the basis of a draft version and without emphasizing interaction with computer science.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First systematic technical alignment analysis of AIA non-discrimination provisions with algorithmic fairness.
- **Experimental Thoroughness**: ⭐⭐⭐ Legal text analysis is meticulous, but lacks empirical validation and technical experiments.
- **Writing Quality**: ⭐⭐⭐⭐ Interdisciplinary writing is clear, terminology is precise, and the structure is logically coherent.
- **Value**: ⭐⭐⭐⭐ Highly relevant reference for AI regulatory compliance practice and the standardization process.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Unifying Proportional Fairness in Centroid and Non-Centroid Clustering](unifying_proportional_fairness_in_centroid_and_non-centroid_clustering.md)
- [\[NeurIPS 2025\] Understanding Challenges to the Interpretation of Disaggregated Evaluations of AI](understanding_challenges_to_the_interpretation_of_disaggregated_evaluations_of_a.md)
- [\[NeurIPS 2025\] Fairness under Competition](fairness_under_competition.md)
- [\[NeurIPS 2025\] Position: Bridge the Gaps between Machine Unlearning and AI Regulation](position_bridge_the_gaps_between_machine_unlearning_and_ai_regulation.md)
- [\[NeurIPS 2025\] Keep It Real: Challenges in Attacking Compression-Based Adversarial Purification](keep_it_real_challenges_in_attacking_compression-based_adversarial_purification.md)

</div>

<!-- RELATED:END -->

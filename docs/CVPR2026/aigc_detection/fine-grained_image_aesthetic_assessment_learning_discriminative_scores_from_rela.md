---
title: >-
  [Paper Note] Fine-grained Image Aesthetic Assessment: Learning Discriminative Scores from Relative Ranks
description: >-
  [CVPR 2026][AIGC Detection][Fine-grained Aesthetics] This paper introduces a new task of "fine-grained image aesthetic assessment," constructs the FGAesthetics benchmark containing 32,217 images across 10,028 series…
tags:
  - "CVPR 2026"
  - "AIGC Detection"
  - "Fine-grained Aesthetics"
  - "Relative Ranking"
  - "Difference-Preserving Tokenization"
  - "Rank Regression"
  - "FGAesthetics"
date: 2026-05-08
content_hash: 3841315341a4fb56
---

# Fine-grained Image Aesthetic Assessment: Learning Discriminative Scores from Relative Ranks

**Conference**: CVPR 2026
**arXiv**: [2603.03907](https://arxiv.org/abs/2603.03907)  
**Code**: [Project Page](https://yzc-ippl.github.io/FG-IAA/)  
**Area**: AIGC Detection
**Keywords**: Fine-grained Aesthetics, Relative Ranking, Difference-Preserving Tokenization, Rank Regression, FGAesthetics

## TL;DR

This paper introduces a new task of "fine-grained image aesthetic assessment," constructs the FGAesthetics benchmark containing 32,217 images across 10,028 series, and proposes FGAesQ: a model that learns discriminative aesthetic scores from relative rankings via Difference-Preserving Tokenization (DiffToken), Contrastive Text-Guided Alignment (CTAlign), and Ranking-Aware Regression (RankReg). The model achieves 0.779 pairwise accuracy on fine-grained scenes while maintaining a coarse-grained SRCC of 0.770.

## Background & Motivation

**Background**: Image Aesthetic Assessment (IAA) is widely applied in content recommendation, AI-generated image guidance, and intelligent photography. Existing datasets (AVA, TAD66K, etc.) evaluate coarse-grained aesthetics with significant inter-image variation, and deep models have achieved strong performance in this regime.

**Limitations of Prior Work**: Real-world applications often require selecting the best image from a series of semantically similar images with subtle aesthetic differences—e.g., selecting the best burst shot, choosing among AIGC-generated samples from the same prompt, or comparing different cropping strategies. Existing IAA models evaluate images independently based on absolute scores and fail to discriminate subtle differences. Two specific challenges arise: (1) **Semantic interference**—images within a series are highly semantically similar, impeding the extraction of fine-grained aesthetic differences, especially since most deep models are pretrained for semantic tasks; (2) **Subtle variation**—minor changes in color and composition require robust discriminative aesthetic representations.

**Key Challenge**: Existing fine-grained related datasets (SPS, Best Frame Selection) have not been fully released, limiting research progress. Current models achieve only 30%–50% series-level accuracy on FGAesthetics, far below their coarse-grained performance.

**Key Insight**: Rather than scoring images independently, the paper exploits **relative ranking relations** within series to learn discriminative scores—coarse-grained data establishes foundational aesthetic perception, while fine-grained data calibrates the regression space to distinguish subtle differences.

## Method

### Overall Architecture

FGAesQ uses ViT-B/16 (CLIP visual encoder) as its backbone and adopts a two-stage training strategy: first pretraining on the coarse-grained AVA dataset to establish basic aesthetic perception, then alternately joint-training on coarse-grained AVA and fine-grained FGAesthetics data. Fine-grained batches contain within-series paired images and ranking labels; coarse-grained batches contain independent images and absolute scores. Three key modules—DiffToken, CTAlign, and RankReg—address input representation, feature alignment, and score calibration, respectively.

### Key Designs

1. **Difference-Preserving Tokenization (DiffToken)**
    - **Function**: In fine-grained inputs, most regions within a series are similar; only a small number of differing regions determine the aesthetic ranking. DiffToken precisely localizes these "aesthetically decisive regions" and preserves their high-resolution details.
    - **Mechanism**: The target image $x$ and reference image $y_1$ are each divided into large-scale patches. The SSIM similarity between corresponding patches is computed as $s_{i,j} = \text{SSIM}(P_{i,j}^x, P_{i,j}^{y_1})$. Patches below the threshold $\tau = \text{percentile}(s, p)$ are identified as the difference region set $D$. Patches in $D$ are tokenized at the original ViT resolution to preserve detail, while the remaining patches are downscaled and randomly dropped to satisfy the token budget.
    - **Design Motivation**: LPIPS analysis reveals that within-series images exhibit low perceptual similarity at the $64\times64$ patch level, indicating that aesthetic differences are concentrated in local regions. Mixed-resolution tokenization directs model attention to key areas while retaining global compositional information.

2. **Contrastive Text-Guided Alignment (CTAlign)**
    - **Function**: Contrastive text descriptions guide the visual model to focus on fine-grained aesthetic differences, enhancing discriminative capability.
    - **Mechanism**: GPT-4o is used to generate contrastive reasoning descriptions $T_1: x \leftarrow y_1$ (using explicit contrastive vocabulary) for ranked image pairs. During training, the cosine distance between the visual embedding difference and the text embedding is minimized: $\mathcal{L}_{F\_align} = \cos(E_v(x) - E_v(y_1), E_t(T_1))$. At inference, only the image encoder is used.
    - **Design Motivation**: Text descriptions provide semantic anchors for human understanding of aesthetic differences—contrastive descriptions help visual representations learn to discriminate along aesthetically relevant directions.

3. **Ranking-Aware Regression (RankReg)**
    - **Function**: Ranking labels calibrate absolute score predictions, ensuring that the predicted score ordering is consistent with human-annotated aesthetic rankings.
    - **Mechanism**: After obtaining absolute scores via a regression head, the Bradley-Terry model is used to compute pairwise superiority probabilities $P_{(x \succ y_1)} = \frac{e^{Score_x}}{e^{Score_x} + e^{Score_{y_1}}}$. All pairwise probability distributions $\mathbf{P'}$ within a series are collected, and a ListMLE loss aligns predictions with ground-truth rankings.
    - **Design Motivation**: Direct regression of absolute scores provides insufficient discriminative power in fine-grained settings—ranking constraints force the model to learn score margins that correctly reflect subtle aesthetic differences.

### Dataset Construction: FGAesthetics

Data are drawn from three source types to ensure diversity: **Natural** (burst photos / video frame sequences), **AIGC** (multiple images generated from the same prompt), and **Cropping** (different cropping strategies applied to the same source image). A three-stage Metrics–MLLMs–Human filtering pipeline ensures that images within a series are visually similar yet distinguishable. Ranking labels are obtained via pairwise comparisons by 10 annotators, with ambiguous and indistinguishable samples filtered out. The final dataset comprises 32,217 images across 10,028 series with series lengths of 2–10.

### Loss & Training

The total loss uses alternating training:
$$\mathcal{L} = \delta \cdot (\lambda \mathcal{L}_{F\_align} + \mathcal{L}_{F\_RR}) + (1 - \delta) \cdot \mathcal{L}_{C\_EMD}$$
where $\delta$ is a binary alternating indicator and $\lambda = 10$. The coarse-grained branch uses EMD loss; the fine-grained branch uses CTAlign + RankReg losses. Momentum update coefficients are 0.615 and 0.8, respectively. Coarse-grained pretraining runs for 3 epochs; joint training runs for 7 epochs on an A800 GPU.

## Key Experimental Results

### Main Results: Comparison of IAA Methods on FGAesthetics

| Method | Params | Natural Pair Acc | Natural s-SRCC | AIGC Pair Acc | Cropping Pair Acc | Cropping s-SRCC |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| NIMA | 54.3M | 0.589 | 0.225 | 0.566 | 0.655 | 0.312 |
| MUSIQ | 78.6M | 0.607 | 0.233 | 0.535 | 0.731 | 0.495 |
| Charm | 85.7M | 0.672 | 0.404 | 0.616 | 0.707 | 0.432 |
| Q-Align | 8.20B | 0.711 | 0.496 | 0.646 | 0.738 | 0.487 |
| MUSIQ (FT) | 78.6M | 0.654 | 0.356 | 0.572 | 0.770 | 0.556 |
| Charm (FT) | 85.7M | 0.723 | 0.474 | 0.620 | 0.755 | 0.517 |
| **FGAesQ (w/o DiffToken)** | 86.3M | 0.773 | 0.664 | 0.688 | 0.764 | 0.537 |
| **FGAesQ (w DiffToken)** | **86.3M** | **0.779** | **0.729** | **0.709** | **0.774** | **0.590** |

### Ablation Study: Training Strategy and Module Contributions

| Configuration | Coarse SRCC | Coarse PLCC | Fine-grained Pair | Fine-grained Series |
|------|:---:|:---:|:---:|:---:|
| w/o Fine (coarse only) | 0.713 | 0.726 | 0.578 | 0.364 |
| w/o Coarse (fine only) | 0.031 | 0.050 | 0.565 | 0.299 |
| Coarse→Fine sequential training | 0.200 | 0.214 | 0.637 | 0.380 |
| w/o DiffToken | 0.751 | 0.760 | 0.666 | 0.423 |
| w/o CTAlign | 0.770 | 0.780 | 0.747 | 0.581 |
| w/o RankReg | 0.769 | 0.781 | 0.742 | 0.571 |
| **FGAesQ (full)** | **0.770** | **0.781** | **0.753** | **0.600** |

### Key Findings

- FGAesQ with only 86.3M parameters comprehensively outperforms Q-Align (8.2B), with Natural series-level SRCC of 0.729 vs. 0.496—a 47% improvement.
- Training with coarse-grained data only yields a fine-grained Series score of merely 0.364; training with fine-grained data only causes coarse-grained SRCC to collapse to 0.031—confirming the fundamentally distinct nature of the two granularities.
- DiffToken contributes the most (Series: 0.423→0.600), followed by RankReg and CTAlign.
- Fine-tuning existing models improves fine-grained performance but causes severe coarse-grained degradation (Charm SRCC: 0.777→0.470); FGAesQ maintains performance at both granularities through alternating training.
- FGAesQ also demonstrates advantages in cross-dataset generalization (ICAA17K, AADB, TAD66K), particularly achieving SRCC of 0.562 on AADB, surpassing VILA's 0.548.

## Highlights & Insights

- **New Task Definition**: Formally defines "fine-grained IAA," filling a gap in the IAA field regarding discriminability of subtle aesthetic differences.
- **Well-designed Dataset**: Three image source types ensure diversity; a three-stage Metrics–MLLMs–Human filtering pipeline combined with pairwise comparison annotation ensures quality.
- **Elegant DiffToken Design**: Mixed-resolution tokenization improves perception of key regions without increasing the total token count; even at inference without reference images, the model outperforms all existing methods.
- **Coarse–Fine Balance**: The alternating training strategy prevents catastrophic forgetting of coarse-grained performance caused by fine-tuning.

## Limitations & Future Work

- DiffToken relies on a reference image to identify difference regions; at inference, series context is required, and the method degrades to standard tokenization for standalone image evaluation.
- Contrastive texts are generated by GPT-4o, introducing additional cost and potential bias.
- Validation is limited to ViT-B/16; scalability to larger backbones remains to be verified.
- FGAesthetics series contain at most 10 images; ranking consistency evaluation for longer sequences is not addressed.

## Related Work & Insights

- **vs. Coarse-grained benchmarks (AVA/TAD66K, etc.)**: Independent scoring + absolute MOS labels vs. FGAesthetics series-based + ranking labels—fundamentally different evaluation paradigms.
- **vs. MLLM methods (Q-Align/UNIAA)**: Despite having 100× more parameters, their coarse-grained perceptual capability does not transfer to fine-grained discrimination—demonstrating that fine-grained judgment requires dedicated design.
- **Insights**: The mixed-resolution strategy of DiffToken is transferable to other visual tasks requiring attention to subtle differences (e.g., medical image comparison, defect detection); the joint ranking-plus-regression training paradigm is generalizable to quality assessment related fields.

## Rating

⭐⭐⭐⭐ (4/5)

Overall assessment: The paper defines a new task with genuine practical demand, constructs a rigorous dataset, and proposes a modular method with clear motivation for each component. DiffToken is a particularly elegant design. However, the overall method leans toward an engineering combination (DiffToken + CLIP + CTAlign + RankReg), with the primary innovations concentrated in the task definition and dataset contribution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Beyond the Final Actor: Modeling the Dual Roles of Creator and Editor for Fine-Grained LLM-Generated Text Detection](../../ACL2026/aigc_detection/beyond_the_final_actor_modeling_the_dual_roles_of_creator_and_editor_for_fine-gr.md)
- [\[AAAI 2026\] BAID: A Benchmark for Bias Assessment of AI Detectors](../../AAAI2026/aigc_detection/baid_a_benchmark_for_bias_assessment_of_ai_detectors.md)
- [\[ACL 2026\] From Scoring to Explanations: Evaluating SHAP and LLM Rationales for Rubric-based Teaching Quality Assessment](../../ACL2026/aigc_detection/from_scoring_to_explanations_evaluating_shap_and_llm_rationales_for_rubric-based.md)
- [\[ICML 2026\] Black-Box Detection of LLM-Generated Text Using Generalized Jensen-Shannon Divergence](../../ICML2026/aigc_detection/black-box_detection_of_llm-generated_text_using_generalized_jensen-shannon_diver.md)
- [\[ACL 2026\] When Personalization Tricks Detectors: The Feature-Inversion Trap in Machine-Generated Text Detection](../../ACL2026/aigc_detection/when_personalization_tricks_detectors_the_feature-inversion_trap_in_machine-gene.md)

</div>

<!-- RELATED:END -->

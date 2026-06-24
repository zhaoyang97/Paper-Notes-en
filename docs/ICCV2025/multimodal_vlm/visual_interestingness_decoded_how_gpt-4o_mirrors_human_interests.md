---
title: >-
  [Paper Note] Visual Interestingness Decoded: How GPT-4o Mirrors Human Interests
description: >-
  [ICCV2025][Multimodal VLM][Visual Interestingness] This work systematically investigates the extent to which large multimodal models such as GPT-4o understand the subjective visual concept of "visual interestingness." The study reveals a moderate positive correlation between GPT-4o and human judgments (73.8% pairwise image agreement rate). Furthermore, it proposes leveraging GPT-4o to automatically annotate image pairs to train a learning-to-rank model for predicting visual i…
tags:
  - "ICCV2025"
  - "Multimodal VLM"
  - "Visual Interestingness"
  - "GPT-4o"
  - "Large Multimodal Models"
  - "Human-AI Alignment"
  - "Learning to Rank"
date: 2026-05-08
content_hash: ae57452b8615ab84
---

# Visual Interestingness Decoded: How GPT-4o Mirrors Human Interests

**Conference**: ICCV2025  
**arXiv**: [2510.13316](https://arxiv.org/abs/2510.13316)  
**Code**: [https://github.com/fiabdu/Visual-Interestingness-Decoded](https://github.com/fiabdu/Visual-Interestingness-Decoded)  
**Area**: Multimodal VLMs / Visual Understanding  
**Keywords**: Visual Interestingness, GPT-4o, Large Multimodal Models, Human-AI Alignment, Learning to Rank

## TL;DR
This work systematically investigates the extent to which large multimodal models such as GPT-4o understand the subjective visual concept of "visual interestingness." The study reveals a moderate positive correlation between GPT-4o and human judgments (73.8% pairwise image agreement rate). Furthermore, it proposes leveraging GPT-4o to automatically annotate image pairs to train a learning-to-rank model for predicting visual interestingness, outperforming all existing methods.

## Background & Motivation

**Background**: Visual interestingness is a highly subjective concept—what kind of images can attract human attention? Since it was proposed by Berlyne in 1949, this question has been at the intersection of computer vision and psychology. Existing research mainly relies on manual annotation (highly expensive and limited in scale) or implicit social platform signals (e.g., Flickr favorites, which suffer from platform bias).

**Limitations of Prior Work**:
   - Direct crowd-sourced annotation (e.g., via AMT) is prohibitively expensive and difficult to scale.
   - Social media metrics (views, additions to favorites) reflect social interaction rather than pure visual interestingness.
   - There is a lack of scalable annotation methods to obtain large-scale interestingness labels.
   - While LMMs perform exceptionally well on objective visual tasks (classification, VQA), their capability to understand subjective concepts has not yet been systematically explored.

**Key Challenge**: Interestingness is subjective (varying from person to person), but there is also a phenomenon where certain images are "universally interesting." Can we leverage the large-scale human knowledge encoded in LMMs to automatically capture this "consensus interestingness"?

**Goal**: (1) To what extent do LMMs (especially GPT-4o) understand visual interestingness? (2) Where do the agreements and discrepancies lie between LMM annotations and human judgments? (3) Can the knowledge from LMMs be utilized to train a lightweight interestingness prediction model?

**Key Insight**: Utilize pairwise comparison instead of absolute scoring to evaluate interestingness, as human judgments on "which one is more interesting" are more reliable and discriminative than "whether it is interesting."

**Core Idea**: Replace human annotations with pairwise interestingness labels from GPT-4o, and distill a lightweight interestingness prediction model through a learning-to-rank framework.

## Method

### Overall Architecture
The study comprises four progressive experiments: (1) **Single-image interestingness evaluation**: humans and LMMs independently judge whether 1,000 images are interesting; (2) **Pairwise interestingness evaluation**: humans and GPT-4o judge which image is more interesting for 2,500 image pairs; (3) **Learning-to-Rank model training**: train a Siamese network to predict interestingness ranking using the annotated data; (4) **In-depth analysis**: interpret the agreement and disagreement between humans and GPT-4o through clustering.

### Key Designs

1. **Single-Image Interestingness Evaluation Experiment**:

    - **Function**: Establish a baseline—human and LMM judgments on absolute interestingness.
    - **Mechanism**: 1,000 images were uniformly sampled from the Flickr-User dataset. On AMT, 258 workers each provided 5 binary "interesting or not" judgments for each image. Simultaneously, GPT-4o, Llama 3.2, and DeepSeek-VL2 each performed 5 judgments. Consensus $|\mathcal{C}_x|$ is defined as the proportion of images where at least 4 out of 5 workers agreed.
    - **Key Findings**: Humans achieved a consensus rate of 91.9%, and GPT-4o achieved 93.9%, but **almost all images were judged as "interesting"** (99.9% for humans, 95.3% for GPT-4o). This indicates that absolute interestingness evaluation has virtually no discriminative power—both humans and models tend to "find something interesting" when asked.

2. **Pairwise Interestingness Evaluation Experiment**:

    - **Function**: Obtain discriminative interestingness annotations through relative comparison.
    - **Mechanism**: 2,500 image pairs were constructed, and 553 AMT workers made pairwise judgments. GPT-4o made the same pairwise judgments. A systematic bias in GPT-4o was discovered and addressed—in 36% of the pairs, GPT-4o consistently preferred the second image (position bias). By running the pairs in both normal and reversed orders, inconsistent pairs were filtered out, leaving 1,599 reliably annotated pairs.
    - **Key Findings**: Human consensus dropped to 56.3% in pairwise comparison (indicating that interestingness is indeed subjective). The overall agreement rate between GPT-4o and humans was 66.2%, which rose to 73.8% on image pairs with high human consensus.
    - **Design Motivation**: Pairwise comparisons capture differences in interestingness more effectively than absolute scoring.

3. **Learning-to-Rank Distillation Model**:

    - **Function**: Train a lightweight interestingness prediction model using annotated data.
    - **Mechanism**: A Siamese network architecture with shared weights is adopted. Given an input image pair $(I_0, I_1)$, features are extracted using CLIP, and fed into a shared linear layer (single neuron). The score difference is modeled via a sigmoid function and trained with binary cross-entropy loss. The score function is defined as $S(I_0, I_1) = \sigma(\mathbf{w}^\top \text{CLIP}(I_0) - \mathbf{w}^\top \text{CLIP}(I_1))$. After training, a single image can also be evaluated to output an interestingness score $S(I) = \sigma(\mathbf{w}^\top \text{CLIP}(I))$.
    - **Design Motivation**: The model is extremely lightweight (a single linear layer), and training data comes from automatic GPT-4o labeling (which can be scaled up easily). Since CLIP features already encode rich visual semantics, the linear layer only needs to learn the dimension of "interestingness."

4. **Human-AI Discrepancy Analysis**:

    - **Function**: Understand the differences in how humans and GPT-4o perceive interestingness through text embedding clustering.
    - **Mechanism**: The text of "reasons" provided by annotators was encoded using OpenAI's embedding model and hierarchically clustered to discover patterns of agreement and disagreement.
    - **Key Findings**: Humans and GPT-4o are highly aligned on "cute/emotional resonance" and "uniqueness." However, GPT-4o additionally favors "bright colors" and "dynamic scenes" (which humans do not necessarily find interesting).

### Loss & Training
Binary cross-entropy loss is used, with a 50/50 split for training/testing. The model is trained for 25 epochs with no overfitting. The average results over 50 different splits are reported.

## Key Experimental Results

### Main Results (Pairwise Interestingness Prediction Accuracy)

| Method | Agreement with Human $A^{(H,x)}$ | LtR Model Acc (Human GT) | Spearman $r_S$ (Human) |
|------|------------------------|-------------------|--------------------|
| GPT-4o Direct Labeling | **73.8%** | 73.4% | 0.59 |
| CuPL (zero-shot) | 60.3% | 61.5% | 0.34 |
| CI (Commonly Interesting) | 69.6% | 69.6% | 0.54 |
| Aesthetic (VILA) | 68.3% | 69.0% | 0.50 |
| Memorability | 35.5% | 34.7% | -0.42 |
| #Comments (Social Metric) | 68.0% | 66.6% | 0.46 |
| #Favorites (Social Metric) | 66.4% | 66.3% | 0.47 |
| Human baseline | - | **77.5%** | - |

### Ablation Study (Comparison of LtR Models Trained on Different Annotation Sources)

| Training Target Source | Test Acc (Human GT) | Test Acc (GPT-4o GT) | Explanation |
|-----------|-----------------|--------------------|----|
| Human | **77.5%** | 72.0% | Optimal human prediction |
| GPT-4o | 73.4% | **84.8%** | Optimal GPT-4o prediction |
| Aesthetic | 69.0% | 73.6% | Aesthetics $\neq$ Interestingness |
| CI | 69.6% | 69.1% | Correlated but different |
| Memorability | 34.7% | 38.3% | Memorability is negatively correlated with interestingness |

### Key Findings
- **Absolute evaluation is meaningless**: The single-image "is it interesting" question has almost no discriminative power—both humans and LMMs tend to find reasons why an image is interesting, leading to over 99% of images being labeled as interesting.
- **GPT-4o exhibits position bias**: GPT-4o consistently chooses the second image in 36% of the pairs, which is a noteworthy systematic error.
- **Memorability is negatively correlated with interestingness**: The Spearman correlation of the Memorability model is -0.42, indicating that easily remembered images are not necessarily interesting.
- **GPT-4o is insensitive to demographic prompting**: Testing persona prompts with different genders, ages, and regions yielded identical results for 116 image pairs.
- **GPT-4o additionally prefers vibrant colors and dynamic scenes**: Humans do not necessarily regard these as interesting, suggesting that the LMM's concept of "interestingness" is partially derived from visual saliency in its training data.

## Highlights & Insights
- **Proposed a valuable research paradigm**: Distilling knowledge from LMMs to substitute manual annotation. This idea is not limited to interestingness and can be generalized to any subjective visual attributes (aesthetics, emotion, style, etc.).
- **Insight on pairwise comparison**: Absolute ratings ("is it interesting") are almost ineffective, whereas pairwise comparisons ("which is more interesting") represent the correct approach to gathering subjective evaluations. This aligns with the classic 2AFC (two-alternative forced choice) methodology in psychophysics.
- **Discovery of GPT-4o's position bias**: A systematic second-image preference was found in 36% of the pairs, representing an important caveat for all studies utilizing LMMs for pairwise annotation.
- **Extremely simple yet effective model**: The final learning-to-rank model is a single-layer linear transformation over CLIP features. While extremely lightweight, it outperforms all specialized methods.

## Limitations & Future Work
- **Limited dataset scale**: The study is conducted with only 1,000 images and 2,500 image pairs, which limits the generalizability of the conclusions.
- **Limited to everyday images**: The focus is on everyday Flickr photos, leaving specialized domains such as artwork, scientific diagrams, and medical images unexplored.
- **Oversimplified handling of GPT-4o position bias**: Inconsistent pairs (36%) were simply discarded. A better approach might involve modeling and correcting the position bias.
- **Exclusion of Llama 3.2 and DeepSeek due to lack of multi-image input support**: Only GPT-4o was thoroughly evaluated, lacking a systematic comparison across multiple LMMs.
- **Simplistic architecture of the learning-to-rank model**: Utilizing only CLIP feature extraction followed by a linear layer; more complex architectures might yield further performance gains.

## Related Work & Insights
- **vs. CI (Commonly Interesting)**: CI defines "consensus interestingness" via Flickr user favorite counts, whereas this study uses GPT-4o pairwise annotations. The GPT-4o method yields a higher agreement rate with humans (73.8% vs. 69.6%) and is free from platform-specific biases.
- **vs. VILA (Aesthetic)**: Aesthetics and interestingness are highly correlated but not equivalent—highly aesthetic images are often interesting, but interesting images are not necessarily highly aesthetic (such as cluttered but narrative-driven scenes). In the experiments, the aesthetic model achieved a 68.3% agreement rate, falling short of GPT-4o's 73.8%.
- **vs. Memorability**: Memorability is negatively correlated with interestingness ($r = -0.42$). This counter-intuitive finding aligns with previous studies, indicating they represent distinct dimensions of visual attributes.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First systematic study of LMM understanding of subjective visual concepts, with an innovative research paradigm.
- **Experimental Thoroughness**: ⭐⭐⭐ Deep analysis from multiple angles, but limited by small dataset scale and reliance only on GPT-4o.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear structure, logical progression, with well-extracted key insights.
- **Value**: ⭐⭐⭐⭐ Pioneers research into subjective visual understanding via LMMs, offering a highly applicable distillation paradigm.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] A Frustratingly Simple Yet Highly Effective Attack Baseline: Over 90% Success Rate Against the Strong Black-box Models of GPT-4.5/4o/o1](../../NeurIPS2025/multimodal_vlm/a_frustratingly_simple_yet_highly_effective_attack_baseline.md)
- [\[ICLR 2026\] CitySeeker: How Do VLMs Explore Embodied Urban Navigation with Implicit Human Needs?](../../ICLR2026/multimodal_vlm/cityseeker_how_do_vlms_explore_embodied_urban_navigation_with_implicit_human_nee.md)
- [\[ACL 2025\] Evaluating Visual and Cultural Interpretation: The K-Viscuit Benchmark with Human-VLM Collaboration](../../ACL2025/multimodal_vlm/evaluating_visual_and_cultural_interpretation_the_k-viscuit_benchmark_with_human.md)
- [\[ICCV 2025\] HRScene: How Far Are VLMs from Effective High-Resolution Image Understanding?](hrscene_how_far_are_vlms_from_effective_high-resolution_image_understanding.md)
- [\[CVPR 2025\] RLAIF-V: Open-Source AI Feedback Leads to Super GPT-4V Trustworthiness](../../CVPR2025/multimodal_vlm/rlaif-v_open-source_ai_feedback_leads_to_super_gpt-4v_trustworthiness.md)

</div>

<!-- RELATED:END -->

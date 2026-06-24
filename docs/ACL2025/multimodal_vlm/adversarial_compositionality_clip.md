---
title: >-
  [Paper Note] Can LLMs Deceive CLIP? Benchmarking Adversarial Compositionality of Pre-trained Multimodal Representation via Text Updates
description: >-
  [ACL 2025][Multimodal VLM][Compositional Vulnerabilities] This paper proposes the MAC benchmark and a diversity-promoting self-training method. By leveraging LLMs to generate deceptive texts, it systematically exposes the compositional vulnerabilities of pre-trained multimodal representations like CLIP, significantly outperforming existing methods across image, video, and audio modalities.
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "Compositional Vulnerabilities"
  - "Adversarial Attacks"
  - "CLIP"
  - "Self-Training"
  - "Multimodal Representations"
date: 2026-05-08
content_hash: eee8d68be3b8a0e7
---

# Can LLMs Deceive CLIP? Benchmarking Adversarial Compositionality of Pre-trained Multimodal Representation via Text Updates

**Conference**: ACL 2025  
**arXiv**: [2505.22943](https://arxiv.org/abs/2505.22943)  
**Code**: [https://vision.snu.ac.kr/projects/mac](https://vision.snu.ac.kr/projects/mac)  
**Area**: Multimodal VLM  
**Keywords**: Compositional Vulnerabilities, Adversarial Attacks, CLIP, Self-Training, Multimodal Representations  

## TL;DR
This paper proposes the MAC benchmark and a diversity-promoting self-training method. By leveraging LLMs to generate deceptive texts, it systematically exposes the compositional vulnerabilities of pre-trained multimodal representations like CLIP, significantly outperforming existing methods across image, video, and audio modalities.

## Background & Motivation
**Background**: Pre-trained multimodal representations such as CLIP have become core components for downstream tasks like retrieval, generation, and reward modeling, directly impacting overall system quality.

**Limitations of Prior Work**: These representations exhibit severe compositional vulnerabilities—for instance, CLIP might assign a higher similarity score to "a bed is sitting on a baby" than to "a baby is sitting on a bed". Existing benchmarks (such as Winoground and SugarCrepe) are confined to specific modalities (images) and predefined types of text operations (replacements, swaps), failing to comprehensively expose these vulnerabilities.

**Key Challenge**: (a) Rule-based approaches (such as word swapping) generate unnatural and easily defensible negative samples; (b) human annotation is expensive and difficult to scale; (c) existing methods only evaluate attack success rate while overlooking the diversity of the attack sample set—monotonous attack patterns are easily defended and fail to reveal diverse vulnerabilities.

**Goal**: (a) Propose a modality-agnostic evaluation framework for compositional vulnerabilities; (b) simultaneously evaluate attack success rate and diversity; (c) achieve efficient vulnerability discovery using a small model (8B).

**Key Insight**: Formalize "whether LLMs can deceive CLIP" as an adversarial attack problem, define multidimensional evaluation criteria (cross-modal, unimodal, distance, auxiliary), and employ rejection sampling self-training paired with diversity-promoting selection to enable LLMs to generate more effective and diverse adversarial texts.

**Core Idea**: Generate deceptive texts using LLM self-training and systematically benchmark the compositional vulnerabilities of multimodal representations through a dual evaluation of multidimensional attack success rate and entropy-based diversity.

## Method

### Overall Architecture
Given a multimodal data pair $(t_i, x_i)$ (text + image/video/audio), the LLM generator $g$ produces an adversarial text $\tilde{t}_i$ that deceives the target representation $f$ (e.g., CLIP) into matching $\tilde{t}_i$ closer to $x_i$ than the original $t_i$. Then, the attack success is evaluated for each sample using a sample-wise four-dimensional criterion, and the diversity of the entire attack set is assessed via group-wise entropy.

### Key Designs

1. **MAC four-dimensional evaluation criteria (sample-wise)**:

    - **Function**: Defines strict multidimensional conditions for a successful attack.
    - **Mechanism**: A successful attack must simultaneously satisfy four criteria: (i) Cross-modal criterion: $d_\theta(y_{t_i}, y_{x_i}) < d_\theta(y_{\tilde{t}_i}, y_{x_i})$, which deceives the model into matching the adversarial text closer to the original modality; (ii) Unimodal criterion: an NLI model determines that $\tilde{t}_i$ and $t_i$ do not entail each other (not a simple paraphrase); (iii) Distance criterion: Levenshtein edit distance < half of the average token length (restricting the scale of modification); (iv) Auxiliary criterion: adhering to predefined rules (such as specified operation types and excluding shortcuts like negation). The total attack success rate is $R = \frac{1}{M_D}\sum_i (s_i^c \cdot s_i^u \cdot s_i^d \cdot s_i^a)$.
    - **Design Motivation**: No single criterion is sufficient to define an effective attack—focusing solely on cross-modality easily degrades into simple paraphrasing, while focusing solely on edit distance ignores semantic discrimination. The four-dimensional criteria formulate a comprehensive constraint on attack quality.

2. **Group-wise diversity evaluation**:

    - **Function**: Measures the variety of text transformation patterns used across the entire adversarial dataset.
    - **Mechanism**: For each sample pair $(t_i, \tilde{t}_i)$, attribute-augmented tokens are constructed (OP_POS_LEMMA format, e.g., I_NOUN_man for inserting the noun "man"). Then, the entropy $H = -\sum_j p_j \log p_j$ and distinct-1 $D_1$ of the token set are calculated. Higher $H$ indicates more diverse attack patterns.
    - **Design Motivation**: If attacks consistently employ the same vocabulary (e.g., always swapping man/woman), they are easily defended despite high success rates and fail to expose the diverse vulnerabilities of the representations.

3. **Diversity-Promoting Self-Training**:

    - **Function**: Trains a small LLM (Llama-3.1-8B) to automatically generate highly successful and diverse adversarial texts.
    - **Mechanism**: A three-step pipeline: (i) Use a base LLM to generate N=64 candidates for each training sample; (ii) Perform Gibbs-sampling-style iterative selection: for each sample, select the candidate from its successful attacks that maximizes the global entropy $H$ (Algorithm 1) over K iterations; (iii) Perform Rejection Sampling Fine-Tuning (RFT) using the selected diverse, successful samples, where the loss is standard autoregressive: $\mathcal{L} = -\frac{1}{M_{\hat{D}}}\sum_i\sum_j \log g(\tilde{t}_{i,j}|\tilde{t}_{i,<j}, \mathcal{I}, t_i; \Theta)$.
    - **Design Motivation**: Vanilla self-training only uses randomly chosen successful samples, which causes model trivialization (consistently generating similar patterns). Incorporating diversity optimization in the training data selection phase enables the model to learn a wider variety of attack patterns.

### Loss & Training
Uses the RFT loss (standard autoregressive cross-entropy). Training data is sourced from successful attack samples selected via diversity-promoting filtering after large-scale sampling with N=64. During inference, high performance is achieved with only N=4.

## Key Experimental Results

### Main Results
Comparison of attack performance across three modalities (N=4, Ours vs. best baseline):

| Modality/Dataset | Metric | Ours (Total ASR) | Best Baseline | Gain |
|-------------|------|-----------------|-------------|------|
| Image/COCO (CLIP) | Total ASR↑ | 42.10% | 23.33% (SeeTrue) | +18.77pp |
| Video/MSRVTT (LB) | Total ASR↑ | 45.60% | 36.90% (VFC) | +8.70pp |
| Audio/AudioCaps (LB) | Total ASR↑ | 52.87% | 5.76% (CompA) | +47.11pp |

Diversity comparison (N=4, Image/COCO):

| Method | H↑ | D1↑ |
|------|-----|------|
| Ours (Diversity-Promoted) | **7.747** | **0.129** |
| Self-Train (w/o diversity) | 7.507 | 0.120 |
| Zero-shot | 7.571 | 0.130 |
| SeeTrue | 7.168 | 0.124 |

### Ablation Study
Contribution of self-training components (Image/COCO, N=4):

| Configuration | Cross ASR | Total ASR | H |
|------|-----------|-----------|-----|
| Zero-shot | 37.29% | 19.19% | 7.571 |
| + Self-Train | 43.08% | 34.64% | 7.507 |
| + Large-N Distilled | 48.29% | 42.03% | 7.452 |
| + Diversity-Promoted (Full) | 47.93% | 42.10% | **7.747** |

### Key Findings
- **Self-training significantly improves attack success rate**: From 19.19% in zero-shot to 34.64% in self-training (+15.45pp), indicating that LLMs can substantially enhance their capability to discover vulnerabilities by learning from their own successful historical generations.
- **Good cross-model transferability**: Attacks trained on CLIP are also effective on SigLIP, NegCLIP, and BLIP (ASR 23-29%), indicating that compositional vulnerabilities are shared across different representation models.
- **Trade-off between ASR and diversity**: Vanilla self-training improves ASR but decreases diversity ($H$ drops from 7.571 to 7.507); diversity-promoting selection restores and further boosts diversity (7.747) with almost no loss in ASR.
- **Small models perform comparably to large models**: Llama-3.1-8B matches GPT-4o in attack effectiveness (even performing better under certain settings), demonstrating that vulnerability discovery does not rely on expensive large LLMs.
- **Audio modality is the most vulnerable**: Total ASR on Audio/AudioCaps reaches 52.87%, which is much higher than Image (42.10%) and Video (45.60%), indicating that compositional vulnerabilities are most severe in audio-language representations.

## Highlights & Insights
- **Modality-agnostic unified evaluation framework**: Expands compositional vulnerability assessment from vision-language to video and audio, utilizing a unified four-dimensional criterion combined with entropy-based diversity evaluation, which prior works failed to address.
- **Ingenious design of diversity-promoting selection**: Without altering the training loss, it introduces diversity optimization (utilizing Gibbs sampling to maximize global entropy) during the data selection phase. This simple yet highly effective approach resolves the trivialization issue of self-training, and the concept can be extended to any rejection sampling scenario.
- **Attribute-augmented token design**: The OP_POS_LEMMA encoding scheme structures text transformations into quantifiable, comparable tokens, providing a computable metric for diversity.

## Limitations & Future Work
- **Focus solely on text modifications without altering modal inputs**: The attack is performed only via text transformations, leaving joint attacks involving image/video/audio inputs unexplored.
- **Dependency on NLI models for evaluation**: The unimodal criterion relies on NLI models to determine entailment relations; thus, errors in the NLI model itself could affect the evaluation accuracy.
- **Insufficient analysis of defense strategies**: Although vulnerabilities are identified, how to exploit these findings to strengthen the model’s compositionality is not thoroughly explored.
- **Promising future directions**: Utilizing the discovered adversarial samples for contrastive learning data augmentation to remedy CLIP's compositional vulnerabilities.

## Related Work & Insights
- **vs. SugarCrepe**: SugarCrepe leverages ChatGPT to generate negative samples but is limited to images and neglects diversity. In contrast, this work provides a more stringent multidimensional evaluation and a diversity-promoting mechanism.
- **vs. RoCOCO**: RoCOCO uses rule-based methods to demonstrate the ASR-diversity trade-off across different word selection strategies, whereas this self-training method effectively breaks this trade-off.
- **vs. CompA**: CompA is restricted to audio. This paper unifies the evaluation of three modalities and dramatically outperforms CompA on the audio modality.

## Rating
- Novelty: ⭐⭐⭐⭐ First to apply unified adversarial attacks and diversity evaluations to benchmark multimodal compositional vulnerabilities.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Complete coverage of three modalities, multiple target models, cross-model transferability, and comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear problem formalization and rigorously defined evaluation criteria.
- Value: ⭐⭐⭐⭐ Exposes systemic vulnerabilities of core representations like CLIP, holding significant weight for VLM robustness and safety research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] MMSafeAware: Can't See the Forest for the Trees: Benchmarking Multimodal Safety Awareness for Multimodal LLMs](cant_see_the_forest_for_the.md)
- [\[ACL 2025\] MTVQA: Benchmarking Multilingual Text-Centric Visual Question Answering](mtvqa_benchmarking_multilingual_text-centric_visual_question_answering.md)
- [\[CVPR 2025\] Skip Tuning: Pre-trained Vision-Language Models are Effective and Efficient Adapters Themselves](../../CVPR2025/multimodal_vlm/skip_tuning_pre-trained_vision-language_models_are_effective_and_efficient_adapt.md)
- [\[ACL 2025\] Finding Needles in Images: Can Multi-modal LLMs Locate Fine Details?](finding_needles_in_images_can_multi-modal_llms_locate_fine_details.md)
- [\[AAAI 2026\] SafeR-CLIP: Mitigating NSFW Content in Vision-Language Models While Preserving Pre-Trained Knowledge](../../AAAI2026/multimodal_vlm/safer-clip_mitigating_nsfw_content_in_vision-language_models_while_preserving_pr.md)

</div>

<!-- RELATED:END -->

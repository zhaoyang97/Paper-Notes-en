---
title: >-
  [Paper Note] LPOI: Listwise Preference Optimization for Vision Language Models
description: >-
  [ACL 2025 (Main)][LLM Alignment][Listwise Preference Optimization] This paper proposes LPOI, the first object-aware listwise preference optimization method for VLMs. By identifying and occluding critical objects in images, LPOI interpolates between positive and negative samples to generate a sequence of progressive occlusions. This trains the model to rank according to object visibility, effectively reducing hallucinations without requiring extra annotation…
tags:
  - "ACL 2025 (Main)"
  - "LLM Alignment"
  - "Listwise Preference Optimization"
  - "Vision Language Models"
  - "Hallucination Mitigation"
  - "Object Occlusion"
  - "Image Interpolation"
date: 2026-05-08
content_hash: 4ff5d1e9f2e1a021
---

# LPOI: Listwise Preference Optimization for Vision Language Models

**Conference**: ACL 2025 (Main)  
**arXiv**: [2505.21061](https://arxiv.org/abs/2505.21061)  
**Code**: [GitHub](https://github.com/fatemehpesaran310/lpoi)  
**Area**: Alignment RLHF / Multimodal VLM  
**Keywords**: Listwise Preference Optimization, Vision Language Models, Hallucination Mitigation, Object Occlusion, Image Interpolation

## TL;DR

This paper proposes LPOI, the first object-aware listwise preference optimization method for VLMs. By identifying and occluding critical objects in images, LPOI interpolates between positive and negative samples to generate a sequence of progressive occlusions. This trains the model to rank according to object visibility, effectively reducing hallucinations without requiring extra annotation, outperforming existing preference optimization methods on MMHalBench, AMBER, and Object HalBench.

## Background & Motivation

**Background**: Aligning VLMs with human preferences is a key challenge. DPO and similar methods have been transferred to multimodal scenarios after succeeding in the textual domain, but simply replacing text preference data with multimodal data often yields sub-optimal performance or even exacerbates hallucinations. Methods like mDPO partially alleviate this issue by generating negative image samples through random cropping.

**Limitations of Prior Work**: (1) VLMs are prone to overfitting textual information during preference learning while ignoring image information, leading to object hallucinations; (2) existing negative image generation methods (random cropping or diffusion-based editing) either lose uncontrollable semantic information or incur extremely high computational costs; (3) listwise ranking has been proven superior to pairwise comparison in the textual domain, but remains unexplored in the visual domain due to the difficulty of constructing listwise samples.

**Key Challenge**: Pairwise preference data only captures binary "good vs. bad" signals, which prevents models from learning fine-grained distinctions like "excellent -> good -> fair -> poor". Such progressive understanding is critical for reducing hallucinations (i.e., models need to learn the difference between "partially visible" and "fully visible" objects).

**Goal**: To design an automatic method for constructing listwise image preference samples, enabling VLMs to reduce hallucinations more finely through listwise ranking learning.

**Key Insight**: The authors observe that hallucination fundamentally stems from the model describing objects that are absent in the image. If the model can learn the progressive relationship of "the more visible an object is, the more it should be mentioned", hallucinations can be effectively reduced.

**Core Idea**: Occluding key objects in the image and automatically generating an image list ordered by object visibility through interpolating occlusion ratios, followed by optimizing the model using a listwise preference loss.

## Method

### Overall Architecture

The input consists of preference datasets composed of image-question-answer triplets (containing chosen and rejected). On top of the standard DPO loss, LPOI adds a listwise loss: (1) using an object detection model to find key objects in the image; (2) occluding the target and validating that occlusion indeed leads to hallucination; (3) generating progressive occlusion image sequences through interpolating occlusion ratios; (4) training the model to rank according to visibility using a listwise loss.

### Key Designs

1. **Object-Aware Hard Negative Generation**:

    - **Function**: Generates high-quality hard negative image samples, causing originally correct answers to become hallucinations.
    - **Mechanism**: Uses Grounding-DINO-Tiny (172M parameters) for zero-shot object detection. Priority for target occlusion is selected as: objects in the first sentence of the chosen response $\rightarrow$ objects in the question $\rightarrow$ other objects in the response $\rightarrow$ randomly detected objects. The bounding box of the selected object is occluded and highlighted with a visual prompt (red circle) to guide the model's focus to the missing object. Hallucination occurrence after occlusion is validated using Idefics2-8B—if no hallucination is triggered, another object is tried.
    - **Design Motivation**: Unlike random cropping, object-aware occlusion retains the global semantic context of the image and only removes key objects, producing more target-oriented "hard" negative samples.

2. **Automatic Listwise Sample Construction**:

    - **Function**: Automatically generates progressively occluded image lists without requiring extra annotations.
    - **Mechanism**: Given a positive image $x_1$ (original) and a hard negative image $x_L$ (fully occluded), for the intermediate $k$-th image, progressively occlude $\frac{k-1}{L-1} \times 100\%$ of the bounding box starting from the image edges. This yields a sequence of $L$ images ordered from highest to lowest object visibility, where $x_1$ has the highest visibility (positive sample) and $x_L$ has the lowest (hard negative sample).
    - **Design Motivation**: Automatically generating fine-grained listwise preference data via continuous occlusion ratio interpolation avoids expensive manual labeling or diffusion model generation.

3. **Listwise Preference Loss**:

    - **Function**: Trains the model so that the likelihood of generating positive text increases monotonically with object visibility.
    - **Mechanism**: Define the listwise loss as $\mathcal{L}_{\text{Listwise}}(\theta) = -\log\left(\prod_{k=1}^{z} \frac{\exp(S_k)}{\sum_{j=k}^{z}\exp(S_j)}\right)$, where $S_k = \beta \log \frac{\pi_\theta(w|x_k, q)}{\pi_{\text{ref}}(w|x_k, q)}$ is the normalized log-likelihood of the model generating a positive response given the image $x_k$. Minimizing this loss forces $S_1 > S_2 > \cdots > S_L$, indicating that higher visibility of the target correlates with a higher probability of positive response generation. The total loss is formulated as $\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{DPO}} + \mathcal{L}_{\text{Anchor}} + \mathcal{L}_{\text{Listwise}}$.
    - **Design Motivation**: Listwise ranking captures relative relations among multiple samples, providing richer gradient signals than pairwise comparisons.

### Loss & Training

The total loss consists of three components: standard DPO loss (text preferences), Anchor loss (boosting the probability of correct answers on the original image), and listwise loss (ranking by image visibility). LoRA fine-tuning is used (rank=8, alpha=8). Idefics2-8B is trained for 3 epochs with a learning rate of 5e-6, and LLaVA-v1.5 is trained for 1 epoch with a learning rate of 1e-6. Training data consists of 10K preference samples sampled from Silkie and LLaVA-Instruct-150K.

## Key Experimental Results

### Main Results (LLaVA-v1.5-7B)

| Method | ObjHal CHAIRs↓ | ObjHal CHAIRi↓ | MMHal Score↑ | MMHal HalRate↓ | AMBER CHAIRs↓ |
|------|----------------|----------------|-------------|----------------|---------------|
| Base | 49.7 | 26.1 | 2.02 | 0.65 | 7.7 |
| + DPO | 42.3 | 23.2 | 2.00 | 0.69 | 6.7 |
| + mDPO | 30.7 | 16.0 | 2.26 | 0.56 | 5.3 |
| + **LPOI** | **24.3** | **14.6** | **2.38** | **0.53** | **4.8** |

### Ablation Study (Idefics2-8B, 5K data)

| Configuration | ObjHal CHAIRs↓ | MMHal Score↑ | HalRate↓ | AMBER CHAIRs↓ |
|------|----------------|-------------|----------|---------------|
| Full LPOI | **5.7** | **2.74** | **0.40** | **2.8** |
| w/o DPO loss | 7.7 | 2.56 | 0.44 | 3.3 |
| w/o DPO + Anchor | 6.0 | 2.50 | 0.45 | 3.5 |
| List size 3 | 7.3 | 2.86 | 0.36 | 2.9 |
| List size 4 | 6.7 | 2.86 | 0.36 | 2.5 |
| List size 5 | **5.3** | **2.88** | **0.36** | 2.6 |
| w/o visual prompting | 5.3 | 2.74 | 0.40 | 2.7 |
| w/ visual prompting | **5.0** | **2.91** | **0.35** | **2.6** |

### Key Findings

- LPOI decreases CHAIRs from 30.7 to 24.3 (-20.8%) on Object HalBench compared to mDPO, showing the most significant effect on LLaVA-v1.5-7B.
- Performance improves with larger list sizes (3 $\rightarrow$ 5), indicating that finer-grained ranking signals assist model learning.
- Visual prompting (marking the occluded area with a red circle) brings a significant boost, and attention map visualization confirms that the model indeed focuses more on the occluded region.
- Under the same GPU budget (20 hours), LPOI still outperforms DPO and mDPO, showing reasonable training efficiency.
- Performance drops when removing the DPO loss, indicating that textual preference signals (derived from rejected responses) remain indispensable.
- In human evaluation across 80 samples, three annotators consistently preferred the responses generated by LPOI.
- LPOI also achieves the best overall performance on HallusionBench (All Acc 49.78 vs. mDPO 48.45).

## Highlights & Insights

- **The "occlusion $\rightarrow$ interpolation $\rightarrow$ ranking" pipeline** is extremely elegant—automatically constructing high-quality listwise preference data with simple geometric operations (occluding bounding boxes and gradually changing the occlusion proportion), completely eliminating the need for extra annotations or expensive diffusion models. This core idea can be transferred to any multimodal task requiring listwise preference data.
- **Visual prompts (red circles)** enhance the model's attention to the occluded regions, which is clearly verified by saliency map visualizations. This simple trick reminds us that when constructing negative samples, we should not only modify the inputs but also guide the model to pay attention to where the modifications occur.
- **The verification module** ensures that occlusion indeed triggers hallucination, avoiding noise from "invalid occlusions"—although the model still outperforms the baseline without the verification module, having it yields better results.

## Limitations & Future Work

- Only focuses on the image-text domain, and listwise preference learning for other modalities like audio remains unexplored.
- Prompts are limited to English, leaving multilingual support to be investigated.
- The detection accuracy of Grounding-DINO directly affects the quality of negative samples—roughly 80% of critical objects can be correctly detected.
- The listwise loss incurs extra training overhead (when list size=5, the training time per epoch is approximately 3 times that of DPO).
- The method might fall short for attribute hallucinations (such as incorrect colors or sizes) that are difficult to occlude via bounding boxes.
- Future work could consider extending the method to mitigate temporal hallucinations in video understanding.

## Related Work & Insights

- **vs. DPO for VLMs**: Standard DPO only learns textual preferences and ignores image information, which makes it prone to overfitting textual patterns. LPOI forces the model to attend to visual information through image-level listwise ranking.
- **vs. mDPO**: mDPO uses random cropping to generate binary negative samples, which causes uncontrollable information loss. LPOI preserves context via object-aware occlusion and provides listwise (instead of binary) ranking signals.
- **vs. V-DPO**: V-DPO uses diffusion models to edit images and generate negative samples, which is computationally expensive. LPOI only requires object detection and occlusion, making it much more efficient.
- **vs. LiPO (Text Domain)**: LiPO applies listwise DPO to the textual domain. LPOI is the first to extend this to the image domain, resolving the hurdle of constructing image ranking data through occlusion interpolation.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First listwise preference optimization for VLMs; the occlusion + interpolation setup for listwise construction is simple and elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three models x three benchmarks, rich ablation studies (list size, loss components, visual prompting, etc.), and human evaluation.
- Writing Quality: ⭐⭐⭐⭐ Clear methodology flowchart, persuasive saliency map visualizations.
- Value: ⭐⭐⭐⭐⭐ ACL Main. It opens up a new direction of listwise ranking for VLM preference optimization and open-sources the code.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] S2H-DPO: Hardness-Aware Preference Optimization for Vision-Language Models](../../ACL2026/llm_alignment/s2h-dpo_hardness-aware_preference_optimization_for_vision-language_models.md)
- [\[ACL 2025\] HiddenDetect: Detecting Jailbreak Attacks against Large Vision-Language Models via Monitoring Hidden States](hiddendetect_detecting_jailbreak_attacks_against_multimodal_large_language_model.md)
- [\[ACL 2025\] AMoPO: Adaptive Multi-objective Preference Optimization without Reward Models and Reference Models](amopo_adaptive_multi-objective_preference_optimization_without_reward_models_and.md)
- [\[ICLR 2026\] Toward Universal and Transferable Jailbreak Attacks on Vision-Language Models (UltraBreak)](../../ICLR2026/llm_alignment/toward_universal_and_transferable_jailbreak_attacks_on_vision-language_models.md)
- [\[ACL 2025\] ASPO: Adaptive Sentence-Level Preference Optimization for Fine-Grained Multimodal Reasoning](aspo_adaptive_sentence-level_preference_optimization_for_fine-grained_multimodal.md)

</div>

<!-- RELATED:END -->

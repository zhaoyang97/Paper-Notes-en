---
title: >-
  [Paper Note] MAVias: Mitigate Any Visual Bias
description: >-
  [ICCV2025][Multimodal VLM][Visual Bias Mitigation] This paper proposes MAVias, an open-set visual bias mitigation framework that extracts visual attribute tags from images using a tagging foundation model, employs an LLM to filter out tags irrelevant to the target class as potential biases, encodes the identified biases via vision-language embeddings, and incorporates them into training to learn bias-invariant representations. MAVias substantially outperforms existing methods on CelebA, Waterbirds, UrbanCars, and ImageNet9.
tags:
  - ICCV2025
  - Multimodal VLM
  - Visual Bias Mitigation
  - Open-Set Bias
  - Foundation Models
  - Vision-Language Models
  - Fairness
date: 2026-05-08
content_hash: f8cf65a451fdb817
---

# MAVias: Mitigate Any Visual Bias

**Conference**: ICCV2025
**arXiv**: [2412.06632](https://arxiv.org/abs/2412.06632)
**Code**: [https://github.com/gsarridis/VB-Mitigator](https://github.com/gsarridis/VB-Mitigator) (VB-Mitigator library)
**Area**: Multimodal VLM / Bias Mitigation
**Keywords**: Visual Bias Mitigation, Open-Set Bias, Foundation Models, Vision-Language Models, Fairness

## TL;DR
This paper proposes MAVias, an open-set visual bias mitigation framework that extracts visual attribute tags from images using a tagging foundation model, employs an LLM to filter out tags irrelevant to the target class as potential biases, encodes the identified biases via vision-language embeddings, and incorporates them into training to learn bias-invariant representations. MAVias substantially outperforms existing methods on CelebA, Waterbirds, UrbanCars, and ImageNet9.

## Background & Motivation

**Background**: Deep learning models are prone to learning spurious correlations present in training data — for example, waterbirds consistently appearing against aquatic backgrounds, or blonde hair being predominantly associated with female subjects. Existing bias mitigation methods fall into two categories: Bias-Aware (BA) methods, which require annotated bias attributes, and Bias-Unaware (BU) methods, which derive pseudo-labels by training a bias proxy model.

**Limitations of Prior Work**:
   - BA methods rely on predefined, known bias labels and thus cannot scale to large-scale general-purpose datasets (e.g., ImageNet), where biases are diverse and unknown.
   - BU methods are effective only when bias is extremely salient (sufficient to train a proxy model) and cannot handle multi-attribute or unknown biases.
   - Neither category generalizes to open-set scenarios, where bias types are unknown in advance and their quantity is indeterminate.

**Key Challenge**: In real-world settings, bias operates at the instance level — each image may exhibit a distinct combination of task-irrelevant attributes — whereas existing methods are designed for dataset-level, single or few known biases.

**Goal**: To automatically discover and mitigate an arbitrary number and type of visual biases in images without any predefined bias specification.

**Key Insight**: The paper leverages the complementary capabilities of foundation models — an image tagging model, an LLM, and a vision-language model — to automatically extract visual attributes, assess their relevance to the target class, and encode irrelevant attributes as bias signals for training.

**Core Idea**: Foundation models are used to automatically discover instance-level open-set visual biases; the biases are encoded as vision-language embeddings and integrated into training via logit fusion to achieve bias-invariant learning.

## Method

### Overall Architecture
MAVias consists of two stages: (1) **Bias Modeling**: for each training image, descriptive tags are extracted → an LLM filters out irrelevant tags → a vision-language model encodes the resulting biases; (2) **Bias Mitigation Training**: the main model extracts image features and computes main logits; a projection layer maps bias embeddings into the same feature space to produce bias logits; the two are summed to form the final prediction, and gradient modulation causes the model to disregard bias features.

### Key Designs

1. **Language-driven Bias Modeling**:

    - **Function**: Automatically identifies visual attributes that are irrelevant to the target class for each training image.
    - **Mechanism**: A three-step pipeline — (a) the Recognize Anything Model (RAM, with a vocabulary of 4,000+ tags) is applied to extract a descriptive tag set $\mathcal{T}^{(i)}$ per image; (b) GPT-4o determines whether each tag is semantically related to the target class $y^{(i)}$, yielding an irrelevant subset $\mathcal{B}^{(i)} \subseteq \mathcal{T}^{(i)}$; (c) OpenCLIP encodes all irrelevant tags as a unified embedding $\mathbf{e}^{(i)} \in \mathbb{R}^d$ via the prompt "a photo of $t_1, t_2, ..., t_k$".
    - **Design Motivation**: (1) RAM covers 4,000+ visual concepts, satisfying open-set requirements; (2) LLMs possess commonsense reasoning to assess semantic relevance between tags and categories; (3) aggregating all irrelevant tags into a single embedding rather than processing each tag individually reduces computational overhead.

2. **Bias Mitigation Training**:

    - **Function**: Trains the main model to learn bias-invariant feature representations.
    - **Mechanism**: The main model $f_\theta$ produces feature $\mathbf{h}^{(i)}$ and main logits $\mathbf{z}_{\text{main}}^{(i)}$. A projection layer $g_\phi$ maps the bias embedding $\mathbf{e}^{(i)}$ into the main model's feature space, after which a classification head yields bias logits $\mathbf{z}_{\text{tag}}^{(i)}$. The final logits are $\mathbf{z}^{(i)} = \mathbf{z}_{\text{main}}^{(i)} + \mathbf{z}_{\text{tag}}^{(i)}$.
    - **Design Motivation**: For highly bias-aligned samples, $\mathbf{z}_{\text{tag}}$ is large, which reduces the relative contribution of $\mathbf{z}_{\text{main}}$ to the total logits and thereby diminishes the gradient updates for such samples — implicitly discouraging the model from relying on bias features.

3. **Logit Alignment Loss**:

    - **Function**: Balances the training of the main model and the projection layer to prevent either from dominating.
    - **Mechanism**: The overall loss is $\mathcal{L} = \mathcal{L}_{cls}(\mathbf{z}^{(i)}, y^{(i)}) + \alpha \cdot \mathcal{L}_{align}$, where the alignment term is $\mathcal{L}_{align} = \frac{1}{2} \| \|\mathbf{z}_{\text{main}}^{(i)}\| - \lambda \cdot \|\mathbf{z}_{\text{tag}}^{(i)}\| \|^2$.
    - **Design Motivation**: $\lambda \in (0,1)$ controls the relative magnitude of bias logits with respect to main logits; a smaller $\lambda$ is appropriate for stronger biases, producing smaller gradients for bias-aligned samples. $\alpha$ balances the classification and alignment losses.

### Loss & Training
SGD is used as the optimizer (Adam for CelebA), with a learning rate of 0.001 decayed by a factor of 10 every one-third of an epoch. Hyperparameters $(\alpha, \lambda)$ are tuned separately for each dataset. At inference, only the main model $f_\theta$ is used; the projection layer $g_\phi$ is discarded, incurring no additional inference overhead.

## Key Experimental Results

### Main Results (Open-Set Evaluation)

| Dataset | Metric | MAVias | JTT (2nd best) | LfF | Gain (vs. 2nd best) |
|--------|------|--------|-----------|-----|-------------|
| CelebA | WG Acc | **66.7%** | 31.5% | 14.7% | +35.2% |
| CelebA | Avg Acc | **81.4%** | 61.6% | 67.1% | +14.0% |
| Waterbirds | WG Acc | **75.4%** | 64.7% | 30.0% | +10.7% |
| Waterbirds | Avg Acc | **87.5%** | 85.2% | 72.7% | +2.3% |
| UrbanCars | WG Acc | **84.4%** | 69.0% | 34.6% | +15.4% |
| UrbanCars | Avg Acc | **89.3%** | 77.8% | 61.0% | +11.5% |
| ImageNet9 MIXED-NEXT | Acc | **88.26%** | 87.56% | 78.70% | +0.70% |
| ImageNet9 NO-FG | Acc | **53.02%** | 59.84% | 61.07% | −6.82% (↓ better) |
| ImageNet9 ONLY-BG-B | Acc | **21.83%** | 29.71% | 34.82% | −7.88% (↓ better) |

### Ablation Study (Bias Detection Effectiveness)

| Dataset | Top Detected Bias Tags | Consistent with Known Biases |
|--------|----------------------|-------------------|
| CelebA | man, woman, suit, tie, dress | ✓ Gender bias recovered + additional biases found |
| Waterbirds | background (water, bamboo, branch) | ✓ Background bias precisely captured |
| UrbanCars | path, forest, hydrant, park | ✓ Urban/rural background bias captured |
| ImageNet9 | 10 irrelevant tags per class | Newly discovered (color, texture, background) |

### Key Findings
- **Dominant advantage in open-set settings**: Existing BU methods (LfF, JTT, Debian, FLAC-B) perform poorly in multi-bias scenarios, while MAVias achieves substantial gains across all datasets.
- **Greatly reduced background dependency on ImageNet9**: On the ONLY-BG-B test set (background only), MAVias reduces accuracy from 35.18% (vanilla) to 21.83%, indicating the model no longer relies on background cues for prediction.
- **Bias discovery beyond predefined attributes**: On CelebA, MAVias not only recovers the known gender bias but also identifies novel bias sources such as clothing items (suit, tie).

## Highlights & Insights
- **Effective composition of foundation models**: RAM, GPT-4o, and OpenCLIP each serve a distinct role, forming a complete pipeline from visual feature extraction → semantic filtering → multimodal encoding. This "foundation model toolchain" paradigm is transferable to many tasks requiring open-set understanding.
- **Instance-level bias modeling**: Unlike conventional approaches that define bias at the dataset level (e.g., gender in CelebA), MAVias constructs an independent bias set for each image, enabling the handling of complex multi-attribute bias scenarios.
- **Zero inference overhead**: The bias projection layer is used only during training; only the main model is required at inference, adding neither computation nor parameters.

## Limitations & Future Work
- **Dependence on GPT-4o for tag filtering**: Tag relevance judgments rely on LLM commonsense reasoning, which is susceptible to errors. The effectiveness of alternative LLMs or open-source substitutes remains unexplored.
- **Limited RAM vocabulary**: Although 4,000+ tags provide broad coverage, fine-grained biases may still be missed.
- **Hyperparameter sensitivity**: $(\alpha, \lambda)$ require per-dataset tuning, increasing the barrier to adoption.
- **Not validated on large-scale generative tasks**: Evaluation is limited to classification; effectiveness in detection, segmentation, and generation tasks remains unknown.

## Related Work & Insights
- **vs. LfF/JTT**: These BU methods obtain pseudo-labels by training a bias proxy model and can only handle a single salient bias. MAVias leverages foundation models to directly discover multi-attribute biases without training a proxy model.
- **vs. FLAC**: FLAC requires indirect access to bias labels and remains constrained to predefined biases. MAVias is fully open-set and requires no prior knowledge of biases.
- **vs. OpenBias**: OpenBias performs open-set bias detection in text-to-image generation but relies solely on textual descriptions, lacking visual grounding. MAVias begins with image-level tag extraction and applies LLM filtering, yielding stronger visual grounding.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First open-set visual bias mitigation framework; creatively combines multiple foundation models.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on 4 datasets under both open-set and closed-set protocols; lacks validation across additional task types.
- Writing Quality: ⭐⭐⭐⭐ Problem motivation is clearly articulated; methodological intuition is well explained.
- Value: ⭐⭐⭐⭐⭐ Open-set bias mitigation is an important and underexplored direction with high practical value.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Omni-Mol: Multitask Molecular Model for Any-to-Any Modalities](../../NeurIPS2025/multimodal_vlm/omni-mol_multitask_molecular_model_for_any-to-any_modalities.md)
- [\[NeurIPS 2025\] MDReID: Modality-Decoupled Learning for Any-to-Any Multi-Modal Object Re-Identification](../../NeurIPS2025/multimodal_vlm/mdreid_modality-decoupled_learning_for_any-to-any_multi-modal_object_re-identifi.md)
- [\[ICCV 2025\] CVPT: Cross Visual Prompt Tuning](cvpt_cross_visual_prompt_tuning.md)
- [\[ICLR 2026\] Grasp Any Region: Towards Precise, Contextual Pixel Understanding for Multimodal LLMs](../../ICLR2026/multimodal_vlm/grasp_any_region_towards_precise_contextual_pixel_understanding_for_multimodal_l.md)
- [\[AAAI 2026\] SAGE: Spuriousness-Aware Guided Prompt Exploration for Mitigating Multimodal Bias](../../AAAI2026/multimodal_vlm/sage_spuriousness-aware_guided_prompt_exploration_for_mitigating_multimodal_bias.md)

<!-- RELATED:END -->

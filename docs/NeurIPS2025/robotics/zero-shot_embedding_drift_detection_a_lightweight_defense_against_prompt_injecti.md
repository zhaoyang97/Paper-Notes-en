---
title: >-
  [Paper Note] Zero-Shot Embedding Drift Detection: A Lightweight Defense Against Prompt Injections in LLMs
description: >-
  [NeurIPS 2025][Robotics][Prompt injection detection] This paper proposes ZEDD (Zero-shot Embedding Drift Detection), which detects prompt injection attacks by measuring semantic drift between benign and suspicious inputs…
tags:
  - "NeurIPS 2025"
  - "Robotics"
  - "Prompt injection detection"
  - "embedding drift"
  - "zero-shot detection"
  - "LLM security"
  - "lightweight defense"
date: 2026-05-08
content_hash: 4ad7bd2db356eb80
---

# Zero-Shot Embedding Drift Detection: A Lightweight Defense Against Prompt Injections in LLMs

**Conference**: NeurIPS 2025
**arXiv**: [2601.12359](https://arxiv.org/abs/2601.12359)
**Code**: [GitHub](https://github.com/AnirudhSekar/ZEDD)
**Area**: Robotics / AI Safety
**Keywords**: Prompt injection detection, embedding drift, zero-shot detection, LLM security, lightweight defense

## TL;DR

This paper proposes ZEDD (Zero-shot Embedding Drift Detection), which detects prompt injection attacks by measuring semantic drift between benign and suspicious inputs in the embedding space. It leverages GMM/KDE to automatically determine detection thresholds, achieving >93% detection accuracy with <3% false positive rate across multiple LLM architectures.

## Background & Motivation

Prompt injection attacks have emerged as a central security threat to LLM-based applications. Adversaries embed malicious instructions through indirect input channels (e.g., emails, user-generated content) to manipulate LLMs into bypassing alignment safeguards and producing harmful outputs. Existing defenses exhibit notable shortcomings:

**High computational overhead**: Detection schemes relying on auxiliary large models or rule-based filters introduce significant computational and latency costs, making them unsuitable for low-latency deployment scenarios.

**Underutilization of embedding-based approaches**: Although prior work has explored embedding-based classification for injection detection, these methods employ traditional optimizers such as logistic regression and XGBoost rather than fine-tuning the embedding space itself to produce optimized classifications.

**Lack of model-agnosticism and zero-shot capability**: Most methods are designed for specific models or attack types, require known attack samples or task-specific retraining, and struggle to generalize to unseen attack patterns.

The core mechanism of ZEDD is that adversarial prompts induce subtle yet quantifiable semantic drift in the embedding space, even when the surface text appears normal. Measuring this drift enables lightweight injection detection without access to model internals, prior knowledge of attacks, or task-specific retraining.

## Method

### Overall Architecture

The ZEDD pipeline consists of three stages: (1) embedding extraction using a fine-tuned encoder; (2) semantic drift computation via cosine similarity; and (3) drift distribution modeling with GMM/KDE to automatically determine detection thresholds. The encoder requires fine-tuning only once and can thereafter be applied zero-shot to new data.

### Key Designs

1. **Embedding Extraction and Fine-tuning**: For each paired clean/injected prompt, embeddings are extracted using multiple models — Sentence-BERT (All-MPNET-Base-V2), Llama 3 8B Instruct, Mistral 7B Instruct, and Qwen 2 7B Instruct. The fine-tuning stage optimizes the embedding space using "similar" (clean-clean pairs) and "dissimilar" (injected-clean pairs) labels, enabling better discrimination between injected and normal prompts. Crucially, fine-tuning uses only approximately 10% of training data (roughly 7% of total data), requiring only 15–18 minutes per model on an NVIDIA B200.

2. **Drift Score Computation**: For a paired prompt $(x, x')$, the embedding drift score is computed as:
    $$\text{Drift}(x, x') = 1 - \frac{f(x) \cdot f(x')}{\|f(x)\| \cdot \|f(x')\|}$$
   where $f(\cdot)$ denotes the embedding function of the fine-tuned encoder. A higher drift score indicates greater deviation of the candidate prompt from its clean counterpart, suggesting a higher likelihood of injection. This approach is more lightweight than direct classification, requiring only a single cosine similarity computation.

3. **Adaptive Threshold Determination via GMM/KDE**:

    - **GMM (primary)**: A two-component Gaussian Mixture Model is fitted to the drift score distribution, where the low-mean component corresponds to clean-clean pairs (small drift) and the high-mean component to injected-clean pairs (large drift). The optimal decision threshold is located at the intersection of the two weighted component densities: $f_{\text{clean}}(x) \cdot w_{\text{clean}} = f_{\text{injected}}(x) \cdot w_{\text{injected}}$.
    - **KDE (fallback)**: When GMM convergence is unstable, the method automatically falls back to kernel density estimation, distinguishing the two classes by identifying peaks and valleys in the distribution.
    - **Constrained optimization**: Binary search iteratively optimizes the threshold within a feasible range while satisfying an upper bound on the false positive rate (default 3%) and a target total flagging rate (default ~50%), ensuring threshold applicability across diverse embedding distributions.

### Loss & Training

The fine-tuning stage employs a binary classification loss: clean-clean pairs are labeled 1 (similar) and injected-clean pairs are labeled 0 (dissimilar). The training objective optimizes the embedding space to maximize separation between the cosine similarity distributions of the two pair types. Data is split 70% training and 30% testing, preserving the original proportion of each attack category. Only 10% of training data is used for fine-tuning to prevent overfitting while maintaining training efficiency.

## Key Experimental Results

### Main Results

**Detection performance across models on the LLMail-Inject dataset (51,603 test pairs)**

| Encoder Model | Accuracy | Precision | Recall | F1 | FPR |
|--------------|----------|-----------|--------|-----|-----|
| SBERT All-MPNET | 90.75% | 99.65% | 81.78% | 89.84% | 1.7% |
| Llama 3 8B | 95.32% | 95.85% | 94.75% | 95.30% | 5.5% |
| Mistral 7B | **95.55%** | **96.58%** | 94.45% | **95.50%** | 2.3% |
| Qwen 2 7B | 95.46% | 96.27% | **94.52%** | 95.38% | 2.2% |

**Detection rates per attack category**

| Model | Clean | Encoding Manipulation | Jailbreak | Prompt Obfuscation | System Leakage | Task Overwrite |
|-------|----|---|---|---|---|---|
| Llama 3 | 5.5% | 98.1% | 92.2% | 94.4% | 96.7% | 90.7% |
| Mistral 7B | 2.3% | 98.1% | 92.2% | 93.3% | 96.9% | 90.8% |
| Qwen 2 | 2.2% | **98.2%** | 90.8% | **94.2%** | **96.8%** | 90.3% |

### Ablation Study

**Effect of varying false positive rate upper bounds**

| FPR Bound | Model | Clean FPR | Encoding Manip. | Jailbreak | System Leakage |
|-----------|-------|-----------|-----------------|-----------|----------------|
| 3% (default) | Mistral 7B | 2.3% | 98.1% | 92.2% | 96.9% |
| 5% | Mistral 7B | 3.4% | 98.2% | 92.2% | 96.9% |
| 10% | Mistral 7B | 5.4% | 98.2% | 92.2% | 96.9% |

### Key Findings

- **LLM encoders consistently outperform SBERT**: Llama 3, Mistral, and Qwen 2 all achieve F1 scores above 95%, surpassing SBERT by 5–6 percentage points, indicating that large models capture subtle semantic variations more effectively.
- **Extremely low false positive rates**: The average FPR is only 2.93%, achieving high recall while generating almost no false alarms on legitimate emails.
- **Encoding manipulation and system leakage are most detectable** (>96%), while jailbreaks and task overwrites are slightly more challenging (~90–92%), likely because these attacks are semantically closer to normal requests.
- **Lightweight and efficient**: Total fine-tuning time for all four models on an NVIDIA B200 is under 60 minutes, and inference on approximately 51,000 test pairs requires only a few minutes.

## Highlights & Insights

- **Elegant simplicity**: The core insight — that injections alter semantics, and semantic change can be quantified via embedding distance — is intuitive yet effective.
- **GMM/KDE adaptive thresholding**: Eliminates manual parameter tuning by automatically learning decision boundaries from the data distribution, enhancing cross-scenario generalization.
- **Model-agnostic design**: The same pipeline seamlessly accommodates different encoders and delivers consistent performance across all tested models.

## Limitations & Future Work

- Detection quality is directly dependent on the encoder's embedding quality; for LLMs of different scales or architectures, the semantic structure of the embedding space may vary considerably.
- Validation is currently limited to the email-format LLMail-Inject dataset; generalization to other input formats such as chat, code, and documents remains unknown.
- Adversarial attacks that deliberately manipulate inputs at the embedding level may circumvent ZEDD's drift detection.
- The authors explicitly position ZEDD as a "lightweight first line of defense," acknowledging that it may be insufficient against all attacks when used in isolation.
- The method requires pre-constructed clean prompt pairs; automatically generating such pairs in practical deployment presents an engineering challenge.

## Related Work & Insights

- Compared to the embedding classifier approach of Ayub & Majumdar, ZEDD achieves better separation by fine-tuning the embedding space itself rather than applying traditional classifiers on top of it.
- In contrast to heavier detection methods that require auxiliary large models for inference, ZEDD's computational overhead is negligible.
- Insight: The concept of embedding drift can be extended to detect other forms of input manipulation, such as data poisoning and out-of-distribution detection.

## Rating

- **Novelty**: ⭐⭐⭐ The embedding drift idea is intuitively clear but not entirely novel; the GMM/KDE thresholding design is a clever contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Four encoder models × five attack categories, with confidence intervals and ablation studies, though evaluation is limited to a single dataset.
- **Writing Quality**: ⭐⭐⭐ Generally readable, but organization is somewhat loose and certain details (e.g., the fine-tuning objective function) lack clarity.
- **Value**: ⭐⭐⭐⭐ Provides a low-barrier, easily deployable practical defense layer for LLM security with clear engineering application value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Understanding Prompt Tuning and In-Context Learning via Meta-Learning](understanding_prompt_tuning_and_in-context_learning_via_meta-learning.md)
- [\[AAAI 2026\] PanoNav: Mapless Zero-Shot Object Navigation with Panoramic Scene Parsing and Dynamic Memory](../../AAAI2026/robotics/panonav_mapless_zero-shot_object_navigation_with_panoramic_scene_parsing_and_dyn.md)
- [\[NeurIPS 2025\] MIP against Agent: Malicious Image Patches Hijacking Multimodal OS Agents](mip_against_agent_malicious_image_patches_hijacking_multimod.md)
- [\[NeurIPS 2025\] SAFE: Multitask Failure Detection for Vision-Language-Action Models](safe_multitask_failure_detection_for_vision-language-action_models.md)
- [\[CVPR 2026\] FineCog-Nav: Integrating Fine-grained Cognitive Modules for Zero-shot Multimodal UAV Navigation](../../CVPR2026/robotics/finecog_nav_fine_grained_cognitive_modules_for_zero_shot_uav_navigation.md)

</div>

<!-- RELATED:END -->

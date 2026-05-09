---
title: >-
  [Paper Note] Controlling Multimodal LLMs via Reward-guided Decoding
description: >-
  [ICCV 2025][Multimodal VLM][reward-guided decoding] This paper proposes MRGD (Multimodal Reward-Guided Decoding), which trains a PaliGemma-based object hallucination reward model and an OWLv2-based object recall reward model. During MLLM inference, MRGD performs sentence-level beam search by scoring candidates with a linearly weighted combination of the two rewards. On CHAIR, it reduces LLaVA-1.5's CHAIRi from 15.05 to 4.53 (a 70% reduction) while enabling dynamic and controllable precision–recall trade-offs.
tags:
  - ICCV 2025
  - Multimodal VLM
  - reward-guided decoding
  - hallucination mitigation
  - visual grounding
  - controllable generation
  - multimodal reward model
date: 2026-05-08
content_hash: 284657b4f47a6cc4
---

# Controlling Multimodal LLMs via Reward-guided Decoding

**Conference**: ICCV 2025
**arXiv**: [2508.11616](https://arxiv.org/abs/2508.11616)
**Code**: None
**Area**: Multimodal VLM / Hallucination Mitigation / Inference-time Alignment
**Keywords**: reward-guided decoding, hallucination mitigation, visual grounding, controllable generation, multimodal reward model

## TL;DR
This paper proposes MRGD (Multimodal Reward-Guided Decoding), which trains a PaliGemma-based object hallucination reward model and an OWLv2-based object recall reward model. During MLLM inference, MRGD performs sentence-level beam search by scoring candidates with a linearly weighted combination of the two rewards. On CHAIR, it reduces LLaVA-1.5's CHAIRi from 15.05 to 4.53 (a 70% reduction) while enabling dynamic and controllable precision–recall trade-offs.

## Background & Motivation
**State of the Field**: Hallucination in MLLMs is primarily addressed through SFT, RLHF fine-tuning, and prompt engineering, but these approaches lack fine-grained controllability at inference time. Once SFT or RLHF training is complete, users cannot adjust model behavior during inference.

**Limitations of Prior Work**: User requirements are diverse—visually impaired users demand high precision (zero hallucinations), while users generating synthetic data need high recall (describing as many objects as possible). Existing methods do not allow users to dynamically trade off these two objectives at inference time. While reward-guided decoding has proven effective in the LLM domain, analogous work in multimodal settings is lacking.

**Root Cause**: There exists an inherent object precision–recall trade-off in MLLM generation—reducing hallucinations tends to make the model more conservative and prone to missing objects, whereas pursuing high recall easily introduces hallucinations. Users have no means to control this trade-off at inference time.

**Paper Goals**: (1) Construct multimodal reward models to evaluate visual grounding quality; (2) Enable dynamic and user-controllable precision–recall trade-offs via reward-guided decoding at inference time.

**Starting Point**: Two independent reward models are constructed separately—$r_{\text{hal}}$ (a hallucination detector trained from preference data) and $r_{\text{rec}}$ (a recall estimator based on an object detector)—and combined dynamically via a weighting parameter $w$.

**Core Idea**: Two complementary multimodal reward models evaluate precision and recall respectively; at inference time, an adjustable weight guides sentence-level decoding search in the MLLM.

## Method

### Overall Architecture
Given an image and a prompt, at each step the MLLM generates $k$ candidate sentences; each candidate is scored by the combined reward $s = w \cdot r_{\text{hal}} + (1-w) \cdot r_{\text{rec}}$, and the highest-scoring candidate is appended to the existing response. This process iterates until an EOS token is generated. Users control the precision–recall trade-off by adjusting $w$, and the computation–quality trade-off by adjusting $k$.

### Key Designs

1. **Object Hallucination Reward Model $r_{\text{hal}}$**:

    - **Function**: Evaluates the precision of object descriptions in generated captions (i.e., whether hallucinations are present).
    - **Mechanism**: Uses PaliGemma as the backbone with a linear regression head to produce a scalar output, trained under the Bradley-Terry preference model framework. The loss augments the standard preference loss with MSE regularization to constrain outputs to $[0,1]$: $\mathcal{L} = \mathcal{L}_{RM} + (r_\theta(x, y^+) - 1)^2 + r_\theta(x, y^-)^2$.
    - Training data: A mixture of four preference datasets (LLaVA-RLHF 9.4K + RLHF-V 5.7K + POVID 17K + SugarCrepe 7.5K), fine-tuned with LoRA for a single epoch.
    - **Design Motivation**: PaliGemma is preferred over CLIP because CLIP's text encoder has a context length of only 64 tokens, which is insufficient for long captions. PaliGemma requires only 9 minutes of training on 8×H100 GPUs, making it extremely lightweight.

2. **Object Recall Reward Model $r_{\text{rec}}$**:

    - **Function**: Estimates how many objects actually present in the image are covered by the generated caption.
    - **Mechanism**: Requires no training; instead, it assembles three off-the-shelf tools—OWLv2 (an open-vocabulary object detector) to detect reference objects in the image, NLTK POS tagger to extract objects mentioned in the caption, and Sentence-BERT to compute semantic similarity between object names. Recall = number of correctly matched reference objects / total reference objects.
    - **Design Motivation**: Detector outputs serve directly as pseudo ground truth, avoiding the overhead of training an additional model. Although the estimates are imperfect (detector recall is approximately 56%), experiments demonstrate that they suffice to effectively improve object coverage in captions.

3. **Sentence-level Search Strategy**:

    - **Function**: Performs reward evaluation and candidate selection once every $T$ sentences (default $T=1$) during generation.
    - **Mechanism**: At each step, $k$ candidate sentences are sampled from the MLLM, scored by the combined reward $s$, and the best candidate is selected. This approach is more than 6× more efficient than rejection sampling ($T=\infty$, selecting the best among fully generated responses).
    - **Design Motivation**: Evaluation at sentence boundaries rather than at the token level avoids noisy reward signals from incomplete sentences, balancing evaluation frequency with evaluation quality.

### Dual-axis Controllability
- **Precision–Recall Axis**: Continuously adjusted via $w \in [0,1]$. Setting $w=1$ maximizes hallucination reduction; $w=0$ maximizes recall.
- **Computation–Quality Axis**: Controlled via $k$ and $T$. Larger $k$ and smaller $T$ yield more thorough search at greater computational cost.

## Key Experimental Results

### Main Results (LLaVA-1.5 on COCO + AMBER)

| Method | CHAIRi↓ | CHAIRs↓ | Recall↑ | CHAIR (AMBER)↓ | Cov. (AMBER)↑ |
|--------|---------|---------|---------|----------------|--------------|
| Greedy | 15.05 | 48.94 | 81.30 | 7.6 | 49.3 |
| CGD | 9.48 | 37.48 | 80.11 | 5.1 | 48.3 |
| **MRGD w=1.0** | **4.53** | **18.19** | 76.04 | **3.4** | 52.4 |
| **MRGD w=0.5** | 5.34 | 22.54 | 78.63 | 4.4 | **60.8** |
| MRGD w=0.0 | 24.20 | 73.42 | **85.23** | 14.8 | 64.3 |

Cross-model generalization (Llama-3.2-Vision, already DPO-aligned):

| Configuration | CHAIRi↓ | CHAIRs↓ | Recall↑ |
|---------------|---------|---------|---------|
| Greedy | 5.82 | 20.52 | 71.45 |
| MRGD w=1.0 | **4.38** | **15.50** | 69.54 |
| MRGD w=0.5 | 4.76 | 16.75 | 71.47 |

### Ablation Study

| Configuration | CHAIRi↓ | Recall↑ | Notes |
|---------------|---------|---------|-------|
| MRGD (PaliGemma) | 5.34 | 78.63 | Default configuration |
| MRGD (PaliGemma-2) | 5.88 | 78.76 | Alternative backbone, comparable performance |
| MRGD +RLAIF-V | 7.83 | 77.54 | More preference data degrades performance |
| MRGD (DETR) | 5.37 | 82.04 | Alternative detector, stable performance |
| MRGD (SigLIP) | 7.19 | 73.71 | Replacing PaliGemma with SigLIP is inferior |

### Key Findings
- MRGD reduces CHAIRi by approximately 50% compared to CGD (4.53 vs. 9.48) while recall drops by only about 5%.
- An inherent precision–recall trade-off exists: as $w$ decreases from 1 to 0, CHAIRi rises from 4.53 to 24.20 while Recall increases from 76.04 to 85.23.
- Sentence-level evaluation ($T=1$) is more than 6× more efficient than rejection sampling ($T=\infty$): $k=5, T=1$ outperforms $k=30, T=\infty$.
- Adding RLAIF-V (83K additional preference data) degrades $r_{\text{hal}}$ quality, indicating that preference data quality matters more than quantity.
- The reward model is cross-model transferable—$r_{\text{hal}}$ trained on LLaVA-1.5 remains effective when directly applied to Llama-3.2-Vision and SmolVLM-2.
- MRGD further reduces hallucinations even on models that have already undergone DPO alignment.

## Highlights & Insights
- **The dual-reward decoupled design is practically effective**: Modeling precision and recall as two independent reward functions is an elegant and effective design. Users can dynamically control model behavior via $w$ as if turning a dial, offering far greater flexibility than the one-shot alignment of RLHF. This design philosophy generalizes to other dimensions (e.g., verbosity vs. conciseness).
- **Zero-training formulation of $r_{\text{rec}}$**: Assembling a training-free recall reward model from OWLv2 + POS tagger + Sentence-BERT is an elegant engineering choice that avoids the difficulty of collecting recall preference data.
- **Inference-time alignment is an important direction for MLLMs**: MRGD is complementary to VisVM—VisVM focuses on long-term value prediction via TD learning, while MRGD focuses on controllable trade-offs between dual objectives. Both demonstrate the effectiveness of inference-time compute scaling for VLMs.

## Limitations & Future Work
- Only object hallucination is addressed; attribute, quantity, and spatial relation hallucinations are not considered.
- $r_{\text{rec}}$ relies on the OWLv2 detector, whose recall is only approximately 56%, potentially missing many objects and underestimating actual recall.
- Validation is limited to captioning tasks; extension to discriminative tasks such as VQA is unexplored.
- The search strategy is relatively simple (best-of-$k$ per sentence); more sophisticated strategies such as beam search or MCTS have not been explored.
- At $k=30$, computational overhead is non-trivial; although batched generation reduces latency, it still requires generating 30× the amount of text.

## Related Work & Insights
- **vs. CGD**: CGD uses SigLIP as the reward signal, whereas MRGD employs $r_{\text{hal}}$ trained on preference data with PaliGemma. The latter is stronger due to (1) longer context capacity and (2) fine-tuning on preference data. MRGD achieves approximately 50% lower CHAIRi.
- **vs. VisVM**: Both perform inference-time search, but with different emphases—VisVM uses TD learning for long-term value prediction to reduce hallucinations, while MRGD uses dual-reward combination for controllable trade-offs. MRGD's advantage lies in controllability; VisVM's lies in self-training.
- **vs. VCD**: VCD reduces hallucinations via contrastive decoding (subtracting image-free probabilities), but on caption benchmarks it performs even worse than greedy decoding, suggesting that VCD is better suited to VQA than to open-ended description.

## Rating
- Novelty: ⭐⭐⭐⭐ First reward-guided decoding method for MLLMs; dual-reward controllable trade-off is a novel contribution
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Two benchmarks (CHAIR + AMBER), three MLLMs, extensive ablations (data mixture, backbone, detector, threshold), comprehensive precision–recall–computation triangle analysis
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure, precise problem formulation, well-designed ablations
- Value: ⭐⭐⭐⭐ Provides a flexible method for inference-time MLLM behavior control with practical value to the community

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] CaptionSmiths: Flexibly Controlling Language Pattern in Image Captioning](captionsmiths_flexibly_controlling_language_pattern_in_image_captioning.md)
- [\[ICCV 2025\] Multimodal LLMs as Customized Reward Models for Text-to-Image Generation](multimodal_llms_as_customized_reward_models_for_text-to-image_generation.md)
- [\[ICCV 2025\] Analyzing Finetuning Representation Shift for Multimodal LLMs Steering](analyzing_finetuning_representation_shift_for_multimodal_llms_steering.md)
- [\[ICCV 2025\] Enrich and Detect: Video Temporal Grounding with Multimodal LLMs](enrich_and_detect_video_temporal_grounding_with_multimodal_llms.md)
- [\[ICCV 2025\] G2D: Boosting Multimodal Learning with Gradient-Guided Distillation](g2d_boosting_multimodal_learning_with_gradient-guided_distillation.md)

<!-- RELATED:END -->

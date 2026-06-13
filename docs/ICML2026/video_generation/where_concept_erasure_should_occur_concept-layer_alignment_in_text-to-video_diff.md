---
title: >-
  [Paper Note] Where Concept Erasure Should Occur: Concept-Layer Alignment in Text-to-Video Diffusion Models
description: >-
  [ICML 2026][Video Generation][Text-to-Video Diffusion] This paper discovers that target concepts in text-to-video diffusion models are most separable only at specific depths. It proposes CLEAR…
tags:
  - "ICML 2026"
  - "Video Generation"
  - "Text-to-Video Diffusion"
  - "Concept Erasure"
  - "Layer Selection"
  - "Sparse Autoencoder"
  - "Safety Generation"
date: 2026-05-08
content_hash: 8abb69ac45e42ecd
---

# Where Concept Erasure Should Occur: Concept-Layer Alignment in Text-to-Video Diffusion Models

**Conference**: ICML 2026  
**arXiv**: [2605.25941](https://arxiv.org/abs/2605.25941)  
**Code**: No public code  
**Area**: Video Generation / Diffusion Model Safety  
**Keywords**: Text-to-Video Diffusion, Concept Erasure, Layer Selection, Sparse Autoencoder, Safety Generation  

## TL;DR
This paper discovers that target concepts in text-to-video diffusion models are most separable only at specific depths. It proposes CLEAR, which utilizes Gumbel-Softmax to learn "where to erase" and Sparse Autoencoders (SAE) to learn "which concept direction to erase," enabling more precise suppression of target concepts while preserving video quality without modifying diffusion model weights.

## Background & Motivation
**Background**: Large-scale text-to-video models generate high-quality, temporally consistent videos but may also generate copyrighted styles, celebrity identities, nudity, or other undesirable concepts. Existing concept erasure methods are generally divided into two categories: inference-time negative prompting or safety guidance, and training-time fine-tuning or closed-form weight modification.

**Limitations of Prior Work**: Many methods assume that concept erasure can be applied at fixed or arbitrary layers, or only focus on "which features to erase," rarely systematically studying "at which depth erasure should occur." However, semantic representations in Diffusion Transformers are non-uniform across depths, and different concepts become separable only at specific layers.

**Key Challenge**: Erasing at too shallow a depth precludes the target concept from being separated from background semantics, leading to weak intervention. Erasing too deeply risks the target concept being already mixed into the generative structure, where intervention causes semantic leakage or quality degradation. Safety generation requires precise suppression of target concepts without damaging non-target content and video quality.

**Goal**: The authors transform concept erasure from a fixed-layer heuristic into a learnable problem: automatically discovering the layer where target concepts and non-target semantics are most separable, then performing local intervention on feature-level directions only at that specific layer.

**Key Insight**: The paper proposes concept-layer topological alignment, which suggests that target concepts form more linearly separable and isolatable subspaces at certain model depths. This observation turns "layer selection" into an optimization objective based on concept-non-target separability.

**Core Idea**: CLEAR simultaneously learns the intervention layer and the concept direction. It uses Gumbel-Softmax for differentiable search of the layer index and Sparse Autoencoders (SAE) to decompose activations and construct specificity masks. During inference, it subtracts the target concept vector only at the selected layer.

## Method
The core of CLEAR is not to retrain the video diffusion model but to insert a learnable concept erasure module into the frozen T5 text encoder. During training, the system takes positive prompts of the target concept and negative prompts that are contextually similar but do not contain the concept, comparing activation separability across different layers. During inference, the model selects the most suitable layer, projects that layer's activations into the SAE sparse space to extract the target concept specificity direction, and subtracts this direction from the original activation.

### Overall Architecture
The input consists of the target concept, positive/negative prompt pairs, and a pre-trained T2V diffusion model. The training phase maintains a set of depth preference parameters $\alpha_1,\dots,\alpha_L$, representing the suitability of each layer for erasure. Simultaneously, an SAE is trained to decompose dense activations into a sparser, more semantic feature space. The optimization objective requires the SAE to reconstruct activations while ensuring the selected layer has high target concept specificity energy and low non-target shared energy.

After training, the layer distribution converges to $l^*=\arg\max_l\alpha_l$. During inference, the SAE is mounted only at this layer without updating diffusion model weights. Given a hidden state $h_{l^*}$, CLEAR encodes it to obtain sparse features, retains features related to the target concept, decodes them back to the model space to form $v_{tar}$, and performs $h'_{l^*}=h_{l^*}-\gamma v_{tar}$.

### Key Designs
1. **Differentiable Intervention Layer Search**:

	- **Function**: Automatically determines the model depth where concept erasure should occur instead of using fixed layers or manual search.
	- **Mechanism**: Maintains preference parameters $\boldsymbol{\alpha}$ for all candidate layers and uses Gumbel-Softmax to sample near-one-hot layer weights $p_k=\frac{\exp((\log\alpha_k+g_k)/\tau)}{\sum_j\exp((\log\alpha_j+g_j)/\tau)}$. As the temperature anneals, the soft distribution becomes a deterministic layer choice.
	- **Design Motivation**: Optimal layers vary by concept, and layer-by-layer enumeration is costly; differentiable search allows layer selection to be driven directly by the separability objective.

2. **SAE Concept-Specific Direction Decomposition**:

	- **Function**: Identifies target concept-related features within the selected layer while avoiding accidental deletion of non-target semantics.
	- **Mechanism**: The SAE encodes the activation $h$ into sparse features $f=\text{ReLU}(W_{enc}h+b_{enc})$. The authors use the feature activations of negative prompts to construct a shared mask $m_{shared}$, and define a specificity mask $m_{spec}=1-m_{shared}$, ensuring features less frequent in non-target contexts are treated as the target concept direction.
	- **Design Motivation**: Dense activations are often polysemous, and directly modifying certain dimensions can easily harm background semantics; sparse features combined with positive/negative prompt pairs help distinguish the "target concept" from "shared visual semantics."

3. **Separability-Driven Alternating Optimization and Inference-Time Subtraction**:

	- **Function**: Adapts "where to erase" and "how to erase" to each other while maintaining a lightweight inference process.
	- **Mechanism**: Training proceeds in two alternating steps. While fixing layer preferences, the SAE minimizes weighted reconstruction error and sparsity penalties. While fixing SAE parameters, layer preferences are updated using $\mathcal{L}_{con}=\log(1+S_{uni}/(S_{spe}+\epsilon))$, ensuring shared energy is relatively lower than concept-specific energy. During inference, $v_{tar}=W_{dec}(f\odot m_{spec})$ is used and subtracted from the hidden state.
	- **Design Motivation**: Optimizing reconstruction alone biases the model toward stable but entangled layers; adding the separability signal biases layer search toward positions where the target concept naturally separates.

### Loss & Training
The training objective consists of two parts. The first is the SAE reconstruction and sparsity loss, which encourages activations under the current layer preference weights to be reconstructed by sparse features. The second is CLEAR's contrastive separability loss, which penalizes shared/non-target energy when it is too strong relative to concept-specific energy. During optimization, layer preferences and SAE parameters are updated alternately, and the Gumbel-Softmax temperature is gradually annealed to obtain the optimal layer and SAE module for each independent concept.

An important detail is that CLEAR does not update the weights of the T2V backbone model; it only performs local modifications to the text encoder's hidden state during inference. Thus, it sits between pure prompt guidance and model fine-tuning: stronger than negative prompts, more local than weight modification, and easier to control for side effects.

## Key Experimental Results

### Main Results
The paper evaluates CLEAR on Wan2.2-5B and CogVideoX-2B T2V diffusion transformers. Target concepts include common objects, safety-sensitive concepts, celebrity identities, and artistic styles. The core metric is the Generative Rate of target concepts (lower is better), alongside Overall Consistency, Imaging Quality, Aesthetic Quality, and Motion Smoothness.

| Model / Object Concept | Metric | CLEAR | Prev. SOTA | Gain |
|--------|------|------|----------|------|
| Wan2.2-5B, 10-Object Avg. | Generative Rate ↓ | 12.8% | T2VUnlearning 24.5% / SAFREE 28.1% | Rate approx. halved |
| Wan2.2-5B, French horn | Generative Rate ↓ | 7.8% | T2VUnlearning 26.4% / SAFREE 28.6% | More thorough erasure |
| Wan2.2-5B, Image Quality | Imaging Quality ↑ | 0.7025 | Origin 0.6910 / T2VUnlearning 0.6652 | Fidelity maintained/improved |
| CogVideoX-2B, 10-Object Avg. | Generative Rate ↓ | 7.1% | T2VUnlearning 7.4% / SAFREE 14.6% | Lowest residual rate |
| CogVideoX-2B, Image Quality | Imaging Quality ↑ | 0.4683 | T2VUnlearning 0.3907 | Avoids quality collapse |

### Ablation Study
The paper extends beyond object concepts to examine harder sensitive concepts, identities, and styles. The following table highlights representative results for nudity and celebrity erasure.

| Configuration | Key Metrics | Description |
|------|---------|------|
| Wan2.2-5B, Nudity (Origin) | Generative Rate 67.3%，Imaging 0.6913 | Original model generates sensitive concepts with high probability |
| Wan2.2-5B, NegPrompt / SAFREE | Generative Rate 55.5% / 48.7% | Pure inference guidance struggles with deep sensitive concepts |
| Wan2.2-5B, T2VUnlearning | Generative Rate 18.6%，Imaging 0.6737 | Strong erasure but weaker than CLEAR |
| Wan2.2-5B, CLEAR | Generative Rate 11.1%，Imaging 0.6928 | Lowest residual with quality near the original model |
| Wan2.2-5B, Celebrity Avg. | CLEAR 0.0151 vs. T2VUnlearning 0.0215 | Lower identity similarity indicates local erasure capability |
| CogVideoX-2B, Nudity | CLEAR 14.63% vs. VideoEraser 19.22% | Maintains strong safety erasure across architectures |

### Key Findings
- Concept erasure is highly dependent on layer position. The paper demonstrates that the Generative Rate follows a V-shaped curve relative to the intervention layer; deviating from the optimal layer leads to semantic residuals or quality degradation.
- CLEAR outperforms T2VUnlearning on Wan2.2-5B with better image quality. On CogVideoX-2B, while T2VUnlearning reduces residuals, image quality drops significantly to 0.3907, indicating weight-level modifications cause more side effects in smaller models.
- For highly entangled concepts like nudity and celebrity identities, negative prompts and SAFREE leave high residuals, while CLEAR significantly lowers detection rates, proving "correct layer + sparse direction" is more reliable than simple prompt steering.

## Highlights & Insights
- The paper's most valuable observation is that "where a concept is separable" is inherently part of the erasure problem. While many safety methods treat intervention position as a hyperparameter, CLEAR elevates it to a primary optimization variable.
- The use of SAE is restrained: it is not used for generative feature visualization but serves controllable intervention, separating target-specific and shared semantic energy.
- Not modifying model weights during inference is a significant engineering advantage. For safety filtering, copyright style removal, and personalized deployment, local activation intervention is cheaper and easier to toggle by concept than re-fine-tuning the entire T2V model.

## Limitations & Future Work
- CLEAR requires training independent SAE and layer search modules for each concept. The paper acknowledges that different concepts have different geometric entanglement patterns, which leads to high concept-level maintenance costs.
- Experiments rely on pre-trained concept detectors to measure generative rates. False negatives, false positives, or insufficient coverage of video frames by the detector can affect erasure metrics.
- The current method primarily intervenes in text encoder layers; how temporal semantics are represented and leaked in the video diffusion backbone has not been deeply analyzed.
- Robustness against combination concepts, multilingual prompts, metaphorical prompts, or malicious prompt jailbreaks still needs further verification.

## Related Work & Insights
- **vs NegPrompt / safety guidance**: Negative prompting only changes the conditioning signal at the input layer and is easily bypassed by complex prompts; CLEAR intervenes in internal concept directions, providing stronger inhibition.
- **vs T2VUnlearning**: Weight-level unlearning can permanently change a model but may harm non-target quality; CLEAR avoids weight changes, reducing side effects through local layer alignment.
- **vs SAE-based text-to-image erasure**: Existing SAE methods focus mostly on "which features"; this work emphasizes the equal importance of "which layer," especially for T2V models with non-uniform deep transformer representations.
- **Transferable Insights**: Concept-layer alignment can be used for style editing, identity protection, and copyright content filtering, and could potentially transfer to text-to-image, audio generation, or safety control in multimodal large models.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Systematizing layer selection in concept erasure as differentiable optimization is a clear and effective perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers two video generation architectures and multiple concept classes, balancing erasure and quality; however, ablation details for layer search could be more complete.
- Writing Quality: ⭐⭐⭐⭐☆ Motivation and method pipeline are clear, and diagrams effectively illustrate the importance of layer position.
- Value: ⭐⭐⭐⭐☆ Highly practical for T2V safety generation and controllable concept editing, especially in deployment scenarios where model weights cannot be modified.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Composing Concepts from Images and Videos via Concept-prompt Binding](../../CVPR2026/video_generation/composing_concepts_from_images_and_videos_via_concept-prompt_binding.md)
- [\[ICML 2026\] Exploring Data-Free LoRA Transferability for Video Diffusion Models](exploring_data-free_lora_transferability_for_video_diffusion_models.md)
- [\[ICML 2026\] LocoT2V-Bench: Benchmarking Long-form and Complex Text-to-Video Generation](locot2v-bench_benchmarking_long-form_and_complex_text-to-video_generation.md)
- [\[ICML 2026\] World-R1: Reinforcing 3D Constraints for Text-to-Video Generation](world-r1_reinforcing_3d_constraints_for_text-to-video_generation.md)
- [\[ICML 2026\] T2AV-Compass: Towards Unified Evaluation for Text-to-Audio-Video Generation](t2av-compass_towards_unified_evaluation_for_text-to-audio-video_generation.md)

</div>

<!-- RELATED:END -->

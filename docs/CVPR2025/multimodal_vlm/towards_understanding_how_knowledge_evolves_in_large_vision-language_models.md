---
title: >-
  [Paper Note] Towards Understanding How Knowledge Evolves in Large Vision-Language Models
description: >-
  [CVPR 2025][Multimodal VLM][LVLM interpretability] This study presents the first systematic analysis of the multimodal knowledge evolution process within LVLMs. It reveals a "critical layer-mutation layer" dual-node pattern of knowledge evolution across three levels: **single-token probability**, **token probability distribution**, and **feature encoding**. The evolution process is categorized into three stages: **rapid evolution $\rightarrow$ stabilization $\rightarrow$ muta…
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "LVLM interpretability"
  - "Knowledge evolution"
  - "early exit"
  - "Hallucination analysis"
  - "Layer-wise analysis"
  - "Model compression"
date: 2026-05-08
content_hash: 7f8eb90bd6e2e35d
---

# Towards Understanding How Knowledge Evolves in Large Vision-Language Models

**Conference**: CVPR 2025  
**arXiv**: [2504.02862](https://arxiv.org/abs/2504.02862)  
**Code**: [https://github.com/XIAO4579/Vlm-interpretability](https://github.com/XIAO4579/Vlm-interpretability)  
**Area**: Multimodal VLMs  
**Keywords**: LVLM interpretability, Knowledge evolution, early exit, Hallucination analysis, Layer-wise analysis, Model compression

## TL;DR
This study presents the first systematic analysis of the multimodal knowledge evolution process within LVLMs. It reveals a "critical layer-mutation layer" dual-node pattern of knowledge evolution across three levels: **single-token probability**, **token probability distribution**, and **feature encoding**. The evolution process is categorized into three stages: **rapid evolution $\rightarrow$ stabilization $\rightarrow$ mutation**, and deep-layer mutations are shown to be closely associated with hallucinations.

## Background & Motivation
Large Vision-Language Models (LVLMs, e.g., the LLaVA series) have become foundational for many AI applications, yet their internal mechanisms remain a "black box." A core question is: **how are multimodal features progressively transformed into natural language across the Transformer layers of LVLMs?** Understanding this process is crucial for optimizing model efficiency and mitigating hallucination.

**Limitations of Prior Work**: Existing research on LLM interpretability (e.g., probing, attention analysis) primarily targets text-only models, with very limited studies on the cross-modal translation process of vision-language features in LVLMs. In particular, knowledge evolution patterns observed in LLMs cannot be directly generalized to LVLMs, as LVLMs must faithfully translate visual information into language descriptions.

**Key Challenge**: The language model component in an LVLM has significantly larger parameters and training data than the vision model, which can lead the language model to "dominate" the generation process. Once visual knowledge stabilizes in the middle layers, language priors in the deeper layers may inject image-unrelated knowledge, resulting in hallucinations.

**Key Insight**: Tracing backward from the output—first observing how token probabilities change across layers, then analyzing the inter-layer variations in probability distributions, and finally delving into the feature encoding space to construct a comprehensive view of knowledge evolution.

## Method

### Overall Architecture
The analysis is conducted on LLaVA-1.5 (with a 32-layer Vicuna-1.5 backbone). Utilizing the early exit technique, language heads are applied to the hidden features of each intermediate layer to observe changes across three levels: token probabilities, probability distributions, and feature encodings. This method requires no additional training, with all analyses conducted directly on the pre-trained model.

### Key Designs

1. **Token Probability Analysis**
    - Function: Reveal how the probability of an individual token changes at different depths of the network.
    - Mechanism: Use early exit to calculate the token probability $p_j(x_K^o|x_p) = \text{softmax}(\phi(H^j))_{x_K^o}$ at each intermediate layer $j$, tracking the probability changes of all predicted tokens from layer 0 to layer 32.
    - Key Findings:
        - **Critical Layer (approx. Layer 20)**: The probabilities of all tokens are close to zero in the shallow layers, but spike sharply around Layer 20, standing out from the vocabulary.
        - **Stable Tokens**: Punctuation and highly certain words (e.g., "image" directly copied from the input) stabilize in probability after the critical layer.
        - **Mutation Tokens**: Words carrying critical semantic information (e.g., nouns, verbs) undergo sudden probability shifts in deep layers, with intense competition among candidate tokens.
    - Design Motivation: To observe when knowledge "takes shape" from the most intuitive probability perspective.

2. **Distribution-level Analysis**
    - Function: Implicitly reveal the rate of knowledge variation via the Jensen-Shannon Divergence (JSD) of probability distributions between adjacent layers.
    - Mechanism: Calculate the probability distribution over the vocabulary for all tokens at each layer, and then compute the JSD between adjacent layers: $JSD(p_i \| p_j) = \frac{1}{2}(KLD(p_i\|A) + KLD(p_j\|A))$
    - Key Findings:
        - JSD is large in the shallow layers (indicating rapid knowledge variation), dramatically decreases and approaches zero after approximately Layer 18 (indicating knowledge stabilization).
        - Some tokens exhibit sudden jumps in JSD in the deep layers (secondary evolution of knowledge), aligning with the mutation layers.
        - This pattern divides knowledge evolution into three stages: **rapid evolution $\rightarrow$ stabilization $\rightarrow$ mutation**.
    - Design Motivation: Single-token probabilities struggle to capture global knowledge changes; distribution-level analysis provides a more comprehensive view.

3. **Feature Encoding Analysis**
    - Function: Visually observe the geometric trajectory of knowledge evolution in the feature space.
    - Mechanism: Project the 4096-dimensional feature vectors of each layer into 2D using t-SNE, and observe the feature changes of different tokens/different images across layers.
    - Key Findings:
        - **Single Image, Multiple Tokens**: Features of all tokens are tightly clustered in the initial layers and diffuse radially and linearly with depth, with features of each layer being adjacent to the previous layer (continuity).
        - **Multiple Images, Single Token (VQA)**: Features across different images form a "guitar-like" shape—shallow features cluster to form the "neck," and deep features diverge in different directions to form the "body."
        - The boundary between shallow and deep layers precisely corresponds to the critical layer.
    - Design Motivation: Feature-level analysis unveils the geometric nature of knowledge evolution—gradually transitioning from modality-agnostic general representations into token-specific representations.

### Skip Connection Validation Experiments
- skip.1 (skipping the stabilization phase between the critical layer and the mutation layer): The output is highly similar to the original, and even hallucinations are preserved $\rightarrow$ minimal knowledge changes occur during the stabilization phase.
- skip.2 (skipping only the mutation layers): Most semantics are preserved, and some hallucinations are corrected (e.g., "standing" $\rightarrow$ "playing") $\rightarrow$ the injection of external knowledge in mutation layers is a potential source of hallucinations.
- skip.3 (skipping from the critical layer to the final 5 layers): The output differs significantly from the original, with increased hallucinations $\rightarrow$ although slow, the stabilization phase still accumulates knowledge.

## Key Experimental Results

### Critical Layer Statistics

| Model | Total Layers | Critical Layer (Approx.) | Mutation Layer |
|------|-------|------------|--------|
| LLaVA-1.5-7B | 32 | ~18-20 | 26-30 (varies by token) |

### Hallucination Association Analysis
- All hallucinated tokens (e.g., "water" which should be "camera", "red" which should be "black", "dog" which should be "sheep") encounter probability reversals in the mutation layers.
- Correct tokens hold a probability dominance prior to mutation, but their probabilities plunge rapidly after the mutation layers, while the probabilities of hallucinated tokens spike quickly.

### Qualitative Results of Skip-Connection Experiments
- Semantic preservation of the output remains extremely high after skipping the stabilization phase (~10 layers) $\rightarrow$ supporting the feasibility of depth compression.
- Some hallucinations are fixed after skipping the mutation layers $\rightarrow$ supporting mutation layer intervention as a hallucination mitigation strategy.

### LVLM vs. LLM Comparison
- The same analysis performed on a text-only LLM (LLaMA-1.5) reveals **no** obvious hierarchical structures or mutation phenomena.
- The distribution variation patterns of functional and informational words differ from those in LVLMs $\rightarrow$ the three-stage knowledge evolution pattern is a unique characteristic of LVLMs.

## Highlights & Insights
- **Pioneering Nature**: This work is the first to completely unveil the evolutionary trajectory of multimodal knowledge from visual features to natural language in LVLMs.
- **Three-level Progressive Analysis**: Token probability $\rightarrow$ distribution JSD $\rightarrow$ feature t-SNE. This top-down hierarchical analysis enables mutual validation of the findings.
- **Practical Implications**:
    - Model Compression: Layers in the stabilization phase can be safely skipped $\rightarrow$ providing a theoretical basis for depth pruning.
    - Hallucination Mitigation: Mutation layers are where hallucinations are injected $\rightarrow$ targeted interventions in these layers can mitigate hallucinations.
    - Efficient Fine-tuning: Shallow features are highly similar across different images $\rightarrow$ fine-tuning only deeper parameters is sufficient for generalizing to new tasks.
- **Critical-Mutation Layer Dual-Node Model**: A elegant and powerful framework that characterizes the knowledge processing paradigm of LVLMs.

## Limitations & Future Work
- Validated only on LLaVA-1.5-7B (32 layers); the findings might differ for larger or newer models (e.g., LLaVA-NeXT, InternVL).
- The analysis is primarily observational and lacks causal proof (e.g., do mutation layers *cause* hallucinations, or are they merely correlated with them?).
- t-SNE visualization is sensitive to parameter choices and may not fully reflect the true structure of the high-dimensional space.
- Specific optimization methodologies (such as hallucination mitigation algorithms or compression algorithms built upon these findings) have not yet been proposed.

## Related Work & Insights
- DoLA (contrasting layer decoding for enhancing truthfulness) $\rightarrow$ The discovery of mutation layers in this study provides theoretical support for DoLA from an LVLM perspective.
- Knowledge Neurons $\rightarrow$ This work analyzes knowledge at the layer level rather than at the neuron granularity.
- Early exit technology $\rightarrow$ This work transforms it from an inference acceleration tool into an analytical tool.
- Insight: LVLMs do not simply "translate" visual features into language; instead, they undergo a complex knowledge evolution process where language priors can dominate in deep layers. This perspective offers profound implications for understanding and improving LVLMs.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to systematically reveal the three-stage knowledge evolution pattern and the critical-mutation layer dual-node structure in LVLMs.
- Experimental Thoroughness: ⭐⭐⭐⭐ The three-level progressive analyses validate each other, and skip-connection experiments are cleverly designed, though causal validation is lacking.
- Writing Quality: ⭐⭐⭐⭐ The analytical path is clear (probability $\rightarrow$ distribution $\rightarrow$ feature) with rich and intuitive illustrations.
- Value: ⭐⭐⭐⭐ Provides theoretical foundations for model compression, hallucination mitigation, and efficient fine-tuning, though concrete algorithms have not yet been implemented.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] VILA-M3: Enhancing Vision-Language Models with Medical Expert Knowledge](vila-m3_enhancing_vision-language_models_with_medical_expert_knowledge.md)
- [\[CVPR 2025\] EventGPT: Event Stream Understanding with Multimodal Large Language Models](eventgpt_event_stream_understanding_with_multimodal_large_language_models.md)
- [\[CVPR 2025\] Embodied Scene Understanding for Vision Language Models via MetaVQA](embodied_scene_understanding_for_vision_language_models_via_metavqa.md)
- [\[NeurIPS 2025\] FlySearch: Exploring how vision-language models explore](../../NeurIPS2025/multimodal_vlm/flysearch_exploring_how_vision-language_models_explore.md)
- [\[CVPR 2025\] VLsI: Verbalized Layers-to-Interactions from Large to Small Vision Language Models](vlsi_verbalized_layers-to-interactions_from_large_to_small_vision_language_model.md)

</div>

<!-- RELATED:END -->

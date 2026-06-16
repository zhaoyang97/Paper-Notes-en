---
title: >-
  [Paper Note] When Token Pruning is Worse than Random: Understanding Visual Token Information in VLLMs
description: >-
  [CVPR 2026][Multimodal VLM][Paper Note] This paper discovers that existing token pruning methods perform worse than random pruning in deep layers of VLLMs. It proposes a method to quantify visual token information based on output probability variations, revealing the "Information Horizon"—a critical layer where visual token information uniformly dissipates t
tags:
  - CVPR 2026
  - Multimodal VLM
date: 2026-05-08
content_hash: 28038af860d4149c
---
# When Token Pruning is Worse than Random: Understanding Visual Token Information in VLLMs

**Conference**: CVPR 2026  
**arXiv**: [2512.07580](https://arxiv.org/abs/2512.07580)  
**Code**: [https://github.com/YahongWang1/Information-Horizon](https://github.com/YahongWang1/Information-Horizon)  
**Area**: Multimodal VLM  
**Keywords**: token pruning, information horizon, visual token information, random pruning, VLM inference acceleration

## TL;DR
This paper discovers that existing token pruning methods perform worse than random pruning in deep layers of VLLMs. It proposes a method to quantify visual token information based on output probability variations, revealing the "Information Horizon"—a critical layer where visual token information uniformly dissipates to zero. This horizon is dynamically influenced by task visual complexity and model capability, and the study proves that integrating simple random pruning effectively enhances existing methods.

## Background & Motivation
**Background**: VLLMs encode images into a large number of visual tokens (576 for LLaVA-1.5, up to thousands for Qwen2.5-VL). Training-free token pruning is a mainstream acceleration scheme, categorized into importance-based methods (FastV/SparseVLM) and diversity-based methods (DivPrune/DART).

**Limitations of Prior Work**: A key observation reveals that in deep decoder layers (e.g., after layer 20), all existing pruning methods perform no better than, or even worse than, random pruning. This phenomenon consistently appears after layer 16-20 in LLaVA-1.5-7B and after layer 21 in Qwen2.5-VL-7B.

**Key Challenge**: Regardless of whether attention or similarity is used as the selection criterion, existing pruning methods cannot identify tokens more informative than random selection in deep layers—implying that visual token information has "dissipated" in deep layers.

**Goal**: (a) Why is deep pruning worse than random? (b) How does visual token information change across layers? (c) At which layer can all visual tokens be safely removed? (d) How can these findings be utilized to improve existing methods?

**Key Insight**: Define and quantify the information of individual visual tokens at specific layers and track their cross-layer evolution patterns.

**Core Idea**: Visual token information uniformly dissipates to zero in deep layers (the "Information Horizon"). Beyond this layer, pruning is equivalent to random selection. Integrating random pruning can improve existing methods.

## Method

### Overall Architecture
This paper does not propose a new pruning method but instead investigates the counter-intuitive phenomenon of why existing pruning methods fail against random pruning in deep decoder layers. The logical flow involves defining a measurable metric for "how much information a single visual token possesses at a certain layer," tracking how this information evolves across layers, identifying the "Information Horizon" where information dissipates to zero, and characterizing what determines this horizon. Finally, these findings are translated into a zero-cost improvement: switching to random pruning after the horizon.

### Key Designs

**1. Visual Token Information Definition: Quantifying contribution by "output probability drop"**

To explain "why deep pruning fails," one must first quantify the value of each visual token at each layer. This paper uses a causal leave-one-out measurement: at layer $i$, all visual tokens except the target token $\mathbf{V}_k$ are removed, and a forward pass is performed to obtain the output probability $p_k$ of the Ground Truth label. Then, all visual tokens are removed to obtain a text-only baseline probability $p_{text}$. The difference is defined as the information carried by that token:

$$I_i(\mathbf{V}_k) = p_k - p_{text}$$

Isolating the target token by excluding others prevents information mixing and ensures the metric truly reflects $\mathbf{V}_k$. This definition is not just an analysis tool—removing low-information tokens based on this metric actually improves model performance (MME increases by 27.8% on LLaVA-1.5-7B), indicating these tokens are active sources of interference rather than harmless noise.

**2. Information Horizon: The critical layer where visual token information dissipates to zero**

With layer-wise information quantified, its evolution can be tracked. A stable pattern emerges: in shallow layers, information varies significantly across tokens (high variance; some are critical, others useless). As depth increases, this variance is smoothed out until a specific layer where information for all tokens approaches zero—the "Information Horizon." Beyond this layer, removing all visual tokens barely affects the output. For LLaVA-1.5-7B, the MME horizon is around layer 16, while for TextVQA it is around layer 24. This explains why pruning fails in deep layers: since all tokens have near-zero information, criteria like attention or similarity lose their signal, and any "selection" degrades to random choice.

**3. Dynamics of the Information Horizon: Determined by task complexity and model capability**

The horizon is not a fixed layer number but fluctuates based on the scenario. First, task visual complexity: tasks like knowledge-based QA or hallucination detection, which rely on text knowledge, have shallower horizons (dissipating around layer 20 in Qwen2.5-VL). Conversely, visual-intensive tasks like OCR have deeper horizons (around layer 27). Second, the model's visual capability: the stronger Qwen2.5-VL has a deeper horizon than the weaker LLaVA-1.5, meaning it can utilize visual information in deeper layers. Together, these rules suggest that the "effective life" of visual tokens is longer when the task requires more visual processing and the model is more capable of it.

**4. Random Pruning Integration Strategy: Abandoning selection after the horizon**

Since deep information is uniform and no criterion holds an advantage, there is no need to calculate importance or similarity after the horizon. Random pruning can be applied directly at zero cost and integrated into any existing method. This "+Random" approach allows Qwen2.5-VL-7B to retain 96.9% of its original performance while pruning 50% of tokens (DivPrune+Random). Its value lies in pragmatism: converting the diagnostic of "information dissipation" into a deployable one-line fix.

## Key Experimental Results

### Main Results — Qwen2.5-VL-7B with 50% Pruning

| Method | MME | TextVQA | MMB | OCRBench | Avg. | Rel.(%) |
|------|-----|---------|------|---------|------|---------|
| Original | 2313 | 85.4 | 79.8 | 88.5 | 83.6 | 100.0 |
| DART | 2295 | 82.1 | 79.6 | 75.5 | 77.3 | 92.7 |
| DivPrune | 2291 | 83.1 | 79.4 | 84.1 | 80.7 | 96.7 |
| DART+Random | 2318 | 82.7 | 79.6 | 77.9 | 78.3 | 93.9 |
| **DivPrune+Random** | **2302** | **83.4** | **79.5** | **85.3** | **80.9** | **96.9** |

### Ablation Study — Validity of Information Quantification (LLaVA-1.5-7B)

| Operation | MME Change | TextVQA Change |
|------|---------|-----------|
| Remove 75% low-info tokens @ layer 10 | **+27.8%** | **+6.1%** |
| Remove 88% low-info tokens @ layer 10 | Better than original | Better than original |

### Key Findings
- **Information dissipation is universal**: Observed in both LLaVA-1.5 and Qwen2.5-VL, regardless of architecture.
- In shallow layers (1-7), pruning methods effectively retain high-info tokens, with diversity methods outperforming importance methods.
- In deep layers (after layer 14), all methods degrade to random levels because information variance approaches zero.
- Removing low-info tokens significantly boosts performance, proving they are sources of interference rather than neutral noise.
- The +Random improvement is most evident on OCRBench (DART: 75.5→77.9) because the OCR horizon is deeper, leaving usable information in deep layers.

## Highlights & Insights
- **Counter-intuitive "Pruning Worse than Random" Discovery**: Explained mechanistically through rigorous experiments and information quantification, elevating observations to actionable theory.
- **Information Horizon Concept**: Provides a concise and practical framework to understand the visual token lifecycle in VLLMs: Generation → Propagation → Dissipation.
- **Simple and Effective Strategy**: The +Random strategy improves existing methods with near-zero cost, offering high practical value.
- **Task-Model-Horizon Triangle**: Reveals the dynamic mechanism where visual complexity and model capability jointly determine the useful depth of visual tokens.

## Limitations & Future Work
- The information definition requires Ground Truth labels, making it unusable directly during inference.
- Information measurement requires extra forward passes (token-by-token removal), incurring high computational overhead.
- The precise location of the Information Horizon requires individual measurement for each task/model/sample, as a predictive model is currently lacking.
- Only LLaVA-1.5 and Qwen2.5-VL were tested.
- While effective, the +Random strategy lacks theoretical guarantees; it essentially represents "giving up fine selection" after info dissipation.

## Related Work & Insights
- **vs. FastV/SparseVLM/DART**: This paper does not propose a new method but explains why existing ones fail in deep layers and provides a simple patch from an information perspective.
- **vs. EmbedLens**: A complementary relationship—EmbedLens classifies tokens as sink/dead/alive from a representational structure perspective, while this paper discovers dissipation from an information quantification perspective. Both support the conclusion that visual token contribution is concentrated in shallow-to-middle layers.
- **Practical Application**: The +Random strategy can be directly superimposed on any existing method professionally.

## Rating
- Novelty: ⭐⭐⭐⭐ The Information Horizon concept is novel; the quantification method is direct and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive analysis across multiple models, benchmarks, and pruning methods.
- Writing Quality: ⭐⭐⭐⭐ Clear logical chain from observation → hypothesis → verification → application.
- Value: ⭐⭐⭐⭐ Provides critical theoretical understanding for token pruning research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] HAWK: Head Importance-Aware Visual Token Pruning in Multimodal Models](hawk_head_importance-aware_visual_token_pruning_in_multimodal_models.md)
- [\[CVPR 2026\] TransPrune: Token Transition Pruning for Efficient Large Vision-Language Model](transprune_token_transition_pruning_for_efficient_large_vision-language_model.md)
- [\[ICLR 2026\] Index-Preserving Lightweight Token Pruning for Efficient Document Understanding](../../ICLR2026/multimodal_vlm/index-preserving_lightweight_token_pruning_for_efficient_document_understanding_.md)
- [\[CVPR 2026\] DocPrune: Efficient Document Question Answering via Background, Question, and Comprehension-aware Token Pruning](docpruneefficient_document_question_answering_via_background_question_and_compre.md)
- [\[CVPR 2026\] VLM-Pruner: Buffering for Spatial Sparsity in an Efficient VLM Centrifugal Token Pruning Paradigm](vlm-pruner_buffering_for_spatial_sparsity_in_an_efficient_vlm_centrifugal_token_.md)

</div>

<!-- RELATED:END -->

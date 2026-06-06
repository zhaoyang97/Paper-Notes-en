---
title: >-
  [Paper Note] L2V-CoT: Cross-Modal Transfer of Chain-of-Thought Reasoning via Latent Intervention
description: >-
  [AAAI 2026][LLM Reasoning][CoT reasoning transfer] Through LAT analysis, this paper reveals that the low-frequency CoT directional representations of LLMs and VLMs share similar distributions. It proposes L2V-CoT: extrac…
tags:
  - "AAAI 2026"
  - "LLM Reasoning"
  - "CoT reasoning transfer"
  - "activation engineering"
  - "frequency-domain analysis"
  - "LLM to VLM"
  - "training-free"
date: 2026-05-08
content_hash: 72424ed14c755ae4
---

# L2V-CoT: Cross-Modal Transfer of Chain-of-Thought Reasoning via Latent Intervention

**Conference**: AAAI 2026
**arXiv**: [2511.17910](https://arxiv.org/abs/2511.17910)  
**Code**: None  
**Area**: Multimodal VLM / Reasoning Enhancement
**Keywords**: CoT reasoning transfer, activation engineering, frequency-domain analysis, LLM to VLM, training-free

## TL;DR
Through LAT analysis, this paper reveals that the low-frequency CoT directional representations of LLMs and VLMs share similar distributions. It proposes L2V-CoT: extract CoT directional representations from an LLM → apply low-pass filtering → frequency-domain resampling for dimension alignment → inject into VLM hidden layers. This training-free approach transfers LLM reasoning capabilities to VLMs, achieving an average improvement of 3.7% and a maximum gain of 8.6%.

## Background & Motivation

### Limitations of Prior Work

**Background**: CoT reasoning has significantly enhanced LLM capabilities, yet VLMs continue to struggle with multi-step reasoning tasks, primarily due to the scarcity of multimodal reasoning data. Existing transfer methods fall into three categories: (1) approaches such as Virgo fine-tune VLMs on textual CoT data, but incur high training costs; (2) model merging integrates LLM parameters into VLMs, but requires architectural compatibility (the LLM backbone of the VLM must match the source LLM); (3) activation engineering methods (e.g., RoT) activate reasoning neurons within the VLM itself, but are bounded by the VLM's inherent reasoning capacity. **Key Challenge**: How can the reasoning capabilities of a stronger LLM be transferred to a VLM across different architectures?

### Starting Point

**Goal**: Although LLMs and VLMs differ in architecture and even in hidden dimension size, do they share transferable reasoning representations? If so, how can cross-modal, cross-architecture reasoning transfer be achieved without any training?

## Method

### Overall Architecture
L2V-CoT consists of two steps: (1) extracting low-pass CoT pattern representations from an LLM (DeepSeek-R1-Distill-Qwen-32B); and (2) injecting these representations into intermediate VLM layers at inference time to implicitly enhance reasoning capability.

### Key Designs
1. **Key Findings from LAT Analysis**: Contrastive inputs ("Let's think step by step" vs. "Answer directly") are passed through LLMs/VLMs to obtain CoT/Non-CoT hidden states, and the directional representation is computed as $u = h(c) - h(d)$. Key observations: (a) CoT representations of VLMs and LLMs cluster in distinct regions of the latent space; (b) the dispersion of CoT directional representations in VLMs is substantially higher than in LLMs (1117.8 vs. 176.7), because representation drift induced by multimodal training concentrates in high-frequency components; (c) after applying low-pass filtering to VLM representations, dispersion drops to 197.7, approaching the LLM value of 176.7; (d) low-frequency components preserve CoT information — injecting low-frequency components activates reasoning, whereas injecting high-frequency components has no effect.

2. **Frequency-Domain Low-Pass Filtering + Resampling**: The CoT pattern representation $v(l_L)$ from the LLM (mean directional representation over 100 samples) undergoes FFT → low-pass filtering (retaining the top $k$ frequency components) → LMN frequency-domain resampling (aligning LLM dimensions to VLM dimensions) → IFFT → normalization. Crucially, resampling is performed in the frequency domain rather than via direct interpolation, which preserves more CoT information (ablations show that interpolation severely degrades performance).

3. **Latent Space Injection**: The low-pass CoT pattern representation is injected into intermediate VLM layers as: $\hat{h}_V = h_V + \alpha \cdot \hat{v}_{LPF}$, followed by norm normalization of the updated activations to preserve the scale of the original representation space. The injection coefficient $\alpha$ must be moderate — too small yields no effect, while too large disrupts the original semantics.

### Loss & Training
The method is entirely training-free. Only 100 CoT/Non-CoT sample pairs are needed to extract representations from the LLM (a one-time offline process); at inference time, a vector addition is performed at designated layers for each token.

## Key Experimental Results

| VLM | Method | MathVista-All | MathVerse | MMStar-All |
|-----|------|-------------|-----------|-----------|
| LLaVA-8B | Non-CoT | 35.2 | 20.9 | 22.9 |
| LLaVA-8B | Finetuned CoT | 39.9 | 24.1 | 25.8 |
| LLaVA-8B | **L2V-CoT** | **41.8** | **25.5** | **26.9** |
| QwenVL-7B | Non-CoT | 60.5 | 26.9 | 33.8 |
| QwenVL-7B | Finetuned CoT | 63.7 | 32.8 | 35.3 |
| QwenVL-7B | **L2V-CoT** | **64.2** | **35.5** | **35.9** |
| InternVL-8B | Non-CoT | 59.3 | 29.9 | 30.5 |
| InternVL-8B | **L2V-CoT** | **61.6** | **33.3** | **33.7** |

### Ablation Study
- Replacing LMN frequency-domain resampling with interpolation causes a sharp performance drop (LLaVA MathVista: 41.8 → 31.1), confirming the information-preservation advantage of the frequency-domain approach.
- Replacing the LLM representation with the VLM's own low-pass directional representation yields limited gains (36.3 vs. 41.8), confirming that the LLM possesses stronger reasoning capabilities.
- Stronger LLMs yield larger improvements: DeepSeek-R1 7B → 14B → 32B raises LLaVA MathVista from 38.6 → 38.9 → 41.8.
- Layer-wise injection analysis shows that middle layers are optimal; shallow layers interfere with perception, while deep layers leave insufficient remaining layers to process the injected information.
- The method is complementary to explicit reasoning methods (e.g., Mulberry/MCTS), enabling further gains when combined.

## Highlights & Insights
- **Profound Frequency-Domain Perspective**: The paper discovers that CoT representations of VLMs and LLMs are highly consistent in their low-frequency components, with high-frequency differences arising from noise introduced by multimodal training. This finding is itself a significant scientific contribution — suggesting that neural networks with different architectures may share similar internal representational structures at the level of abstract reasoning.
- **True Cross-Architecture Training-Free Transfer**: The method does not require VLMs and LLMs to share a backbone, elegantly resolving dimension mismatches via frequency-domain resampling. This means the strongest available LLMs (e.g., DeepSeek-R1) can be used to enhance the reasoning capabilities of arbitrary VLMs.
- **Surpassing Supervised Methods**: L2V-CoT (training-free) outperforms Finetuned CoT (supervised) on multiple benchmarks, suggesting that directly manipulating hidden representations may be more efficient than fine-tuning — because fine-tuning modifies both useful and irrelevant parameters, whereas activation injection targets only the critical reasoning representations.
- **Plug-and-Play and Complementary to Explicit Methods**: The method can be combined with explicit reasoning search approaches such as MCTS, achieving dual gains from implicit reasoning enhancement and explicit reasoning search.
- **Methodological Contribution of LAT Analysis**: The framework of using Linear Artificial Tomography to analyze cross-modal reasoning representations is generalizable to other capability transfer research.

## Limitations & Future Work
- Injection layer and injection strength require per-task tuning (Tables S.2/S.3 show different hyperparameters for different tasks), and an adaptive selection mechanism is lacking.
- Validation is limited to mathematical reasoning benchmarks; the method has not been tested on general VQA or broader reasoning tasks (e.g., spatial or causal reasoning).
- CoT samples are drawn from the STILL-2 dataset (mathematics, physics, chemistry, biology), and domain bias may affect generalization to other domains (e.g., legal or financial reasoning).
- Frequency-domain resampling assumes that CoT information is concentrated in low-frequency components — this holds for mathematical reasoning, but it is unclear whether it remains valid for tasks requiring fine-grained symbolic manipulation.
- The choice of low-pass filter cutoff frequency lacks theoretical grounding and currently relies on empirical tuning.
- The interpretability of hidden representation injection is limited — although the method is effective, it is difficult to explain precisely how the VLM's internal reasoning process changes after injection.
- It is unclear whether the method imposes requirements on VLM model scale; performance may differ for very small (<1B) or very large (>70B) VLMs.

## Related Work & Insights
- **vs. Virgo (Finetuned CoT)**: Virgo requires SFT on the VLM, incurring high training costs; L2V-CoT requires no training and achieves superior performance.
- **vs. Model Merging**: Model merging requires architectural compatibility, whereas L2V-CoT supports transfer across different architectures such as LLaMA and Qwen.
- **vs. RoT**: RoT extracts directional representations from CoT prompts and injects them into VLMs, but is bounded by the VLM's own reasoning capacity; L2V-CoT extracts representations from a stronger external LLM.
- **vs. MathNeuro**: MathNeuro activates reasoning neurons via pruning/scaling but neglects the coordination among neurons.

## Related Work & Insights
- "Low-frequency components encode core capabilities, while high-frequency components carry noise and modality-specific information" — this observation may apply to other cross-modal transfer scenarios.
- Frequency-domain resampling as a dimension alignment technique outperforms simple interpolation or linear projection, and merits broader exploration in other activation engineering tasks.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First work to use frequency-domain analysis to reveal cross-architecture consistency in LLM/VLM reasoning representations; the method is novel and elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 4 VLMs × 5 benchmarks with complete ablations and multi-scale LLM validation.
- Writing Quality: ⭐⭐⭐⭐ The logical flow from empirical analysis to method design is clear, and frequency-domain visualizations are highly convincing.
- Value: ⭐⭐⭐⭐ Provides a general cross-architecture reasoning transfer framework with practical significance for VLM reasoning enhancement; the frequency-domain analysis framework is broadly applicable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Efficient Thought Space Exploration Through Strategic Intervention](efficient_thought_space_exploration_through_strategic_intervention.md)
- [\[ACL 2026\] Towards Effective In-context Cross-domain Knowledge Transfer via Domain-invariant-neurons-based Retrieval](../../ACL2026/llm_reasoning/towards_effective_in-context_cross-domain_knowledge_transfer_via_domain-invarian.md)
- [\[AAAI 2026\] CMMCoT: Enhancing Complex Multi-Image Comprehension via Multi-Modal Chain-of-Thought and Memory Augmentation](cmmcot_enhancing_complex_multi-image_comprehension_via_multi.md)
- [\[CVPR 2026\] Rationale-Enhanced Decoding for Multi-modal Chain-of-Thought](../../CVPR2026/llm_reasoning/rationale-enhanced_decoding_for_multi-modal_chain-of-thought.md)
- [\[AAAI 2026\] ARCHE: A Novel Task to Evaluate LLMs on Latent Reasoning Chain Extraction](arche_a_novel_task_to_evaluate_llms_on_latent_reasoning_chai.md)

</div>

<!-- RELATED:END -->

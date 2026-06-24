---
title: >-
  [Paper Note] Bridging the Language Gaps in Large Language Models with Inference-Time Cross-Lingual Intervention
description: >-
  [ACL 2025][Multilingual & Machine Translation][Cross-Lingual Transfer] This paper proposes INCLINE (Inference-Time Cross-Lingual Intervention), a tuning-free inference-time framework. By learning an alignment matrix to transform internal representations of low-performance languages into the representation space of high-performance languages, it significantly boosts multilingual performance across 9 benchmarks and 5 LLMs.
tags:
  - "ACL 2025"
  - "Multilingual & Machine Translation"
  - "Cross-Lingual Transfer"
  - "Inference-Time Intervention"
  - "Multilingual LLM"
  - "Representation Alignment"
  - "Low-Resource Languages"
date: 2026-05-08
content_hash: 1760533225eba25d
---

# Bridging the Language Gaps in Large Language Models with Inference-Time Cross-Lingual Intervention

**Conference**: ACL 2025  
**arXiv**: [2410.12462](https://arxiv.org/abs/2410.12462)  
**Code**: [https://github.com/weixuan-wang123/INCLINE](https://github.com/weixuan-wang123/INCLINE)  
**Area**: Multilingual Translation  
**Keywords**: Cross-Lingual Transfer, Inference-Time Intervention, Multilingual LLM, Representation Alignment, Low-Resource Languages

## TL;DR
This paper proposes INCLINE (Inference-Time Cross-Lingual Intervention), a tuning-free inference-time framework. By learning an alignment matrix to transform internal representations of low-performance languages into the representation space of high-performance languages, it significantly boosts multilingual performance across 9 benchmarks and 5 LLMs.

## Background & Motivation

**Background**: Although multilingual LLMs are known for their cross-lingual capabilities, there is a significant performance gap among different languages—English typically far outperforms other languages, especially low-resource ones. Existing methods to narrow this gap mainly include: multilingual pre-training (e.g., XLM-R), multilingual instruction tuning, and language-specific adapters.

**Limitations of Prior Work**: (1) Pre-training and fine-tuning methods require massive computational resources, demanding a large investment for each new language; (2) Even with multilingual training, models with limited parameters still allocate insufficient representation capacity to low-resource languages; (3) Existing inference-time methods (e.g., prompt translation, few-shot demonstration translation) introduce translation error accumulation and rely heavily on the quality of translation systems.

**Key Challenge**: The performance gap of LLMs across different languages is essentially a gap in the representation space—high-performance languages (such as English) occupy "better" regions of the representation space (more aligned with knowledge and reasoning capabilities), while low-performance languages are squeezed into poorer regions. This disparity is rooted in the imbalance of training data.

**Goal**: To bridge the performance gap between languages through inference-time representation intervention, without modifying model parameters.

**Key Insight**: The authors hypothesize that an approximately linear mapping relationship exists between the high-level representation spaces of different languages. If this mapping can be found, the representations of low-performance languages can be "projected" into the representation spaces of high-performance languages, leveraging the "good representations" of high-resource languages like English to improve the performance of low-resource languages.

**Core Idea**: Learn cross-lingual alignment matrices using parallel sentence pairs, and linearly transform mid-layer representations of low-resource languages into the high-resource language space at inference time.

## Method

### Overall Architecture
The workflow of INCLINE consists of two stages: (1) Offline learning stage—collect parallel sentence pairs of the source language (low-performance) and the target language (high-performance, usually English), run them through the LLM to extract inner hidden state representations, and use least-squares optimization to learn a linear alignment matrix from source to target; (2) Inference-time intervention stage—when the LLM processes a source language input, the learned alignment matrix is applied to transform the hidden states at a specific layer before continuing the computation of subsequent layers.

### Key Designs

1. **Least-Squares-Based Alignment Matrix Learning**:

    - Function: Learn the representation space mapping from source language to target language
    - Mechanism: Given a set of parallel sentence pairs $\{(s_i, t_i)\}_{i=1}^N$, obtain the hidden state $h_l^s(s_i)$ of the source language sentence $s_i$ at the $l$-th layer and the hidden state $h_l^t(t_i)$ of the target language sentence $t_i$ at the $l$-th layer. Learn a linear transformation matrix $W^*$ such that $W^* = \arg\min_W \sum_i \|W \cdot h_l^s(s_i) - h_l^t(t_i)\|^2$, which is a standard least-squares problem with a closed-form solution $W^* = (H_s^T H_s)^{-1} H_s^T H_t$. The learning process is rapid and does not require backpropagation.
    - Design Motivation: The linear transformation hypothesis is a strong but reasonable simplification—prior cross-lingual representation studies (such as MUSE, VecMap) have shown that approximate linear mappings exist between word embedding spaces. Extending this finding to the hidden layers of LLMs is a natural progression. The least-squares method is computationally highly efficient, requiring only a few hundred parallel sentence pairs to be sufficient.

2. **Layer Selection Strategy**:

    - Function: Determine at which layer (or layers) of the LLM the intervention should be applied
    - Mechanism: Different layers capture different levels of linguistic information—lower layers contain more lexical/syntactic information, while higher layers contain more semantic/task information. The optimal intervention layer is determined by testing the intervention effect layer-by-layer on a validation set. Experiments reveal that mid-to-high layers (e.g., layers 18-24 in a 32-layer model) are typically the optimal choice—the representations of these layers are abstract enough to support cross-lingual alignment, but not too close to the output layer where the intervention impact could be excessively disruptive.
    - Design Motivation: Intervening at a too low layer disrupts the model's basic understanding of the source language; intervening at a too high layer limits the intervention's coverage, as subsequent computational steps are too few to fully exploit the aligned representations. The mid-to-high layers represent the best sweet spot balancing semantic abstraction and remaining computational capacity.

3. **Inference-Time Representation Intervention**:

    - Function: Transform source language representations into the target language space in real-time during inference
    - Mechanism: During inference, when the LLM processes a source language input and reaches the $l$-th layer, the hidden state $h_l^s$ is replaced with $\tilde{h}_l = W^* \cdot h_l^s$, after which the forward pass continues through subsequent layers. The intervention is seamless—requiring no modification of the model architecture or weights, and performing only a single matrix multiplication on the intermediate representations. Practically, this is achieved by intercepting and modifying the hidden states at specific layers via a hook mechanism.
    - Design Motivation: The greatest advantage of inference-time intervention is zero training cost—learning the alignment matrix takes only a few minutes, and the same matrix can be reused across all inputs in that language. This allows the method to scale rapidly to new languages.

### Loss & Training
The alignment matrix learning uses the least-squares loss, solved in closed-form without gradient optimization. The number of required parallel sentence pairs is in the order of hundreds to thousands.

## Key Experimental Results

### Main Results

| Benchmark Task | Metric | INCLINE | Direct Inference | Translate-then-Answer | Prompt Translation |
|---------|------|---------|---------|-----------|-----------|
| XNLI (Multilingual) | Acc | 72.5 | 65.8 | 70.1 | 68.4 |
| XQuAD (Multilingual) | F1 | 68.3 | 61.2 | 66.5 | 64.8 |
| MGSM (Multilingual Math) | Acc | 55.2 | 46.7 | 52.8 | 50.1 |
| X-COPA (Causal Reasoning) | Acc | 78.6 | 71.4 | 75.9 | 73.8 |

### Cross-Model Validation

| LLM | Avg. Multilingual Gain | Low-Resource Language Gain | Description |
|-----|--------------|---------------|------|
| LLaMA-2-7B | +6.2% | +9.5% | Weak base multilingual ability, large gain |
| LLaMA-2-13B | +5.1% | +8.3% | Significant gain still observed for larger model |
| mGPT | +4.8% | +7.1% | Multilingual pre-trained models also benefit |
| Mistral-7B | +5.5% | +8.8% | Effective across different architectures |
| LLaMA-3-8B | +3.9% | +6.7% | Stronger multilingual models have smaller but still significant gains |

### Key Findings
- INCLINE outperforms both direct inference and translation-based methods across all 5 LLMs and 9 benchmarks tested.
- Gains are most significant on low-resource languages (e.g., Swahili, Urdu) (+8-10%), while gains for high-resource languages (e.g., French, German) are smaller (+2-4%), which aligns with expectations.
- The mid-to-high layers (around 60-75% depth) are the optimal intervention locations, consistent with prior research findings on LLM hierarchical functionality.
- Robust alignment matrices can be learned from only a few hundred parallel sentence pairs, demonstrating extreme data efficiency.
- The linear alignment hypothesis holds for most language pairs and tasks, with slightly weaker results on language pairs with extreme morphological differences (e.g., English-Japanese).

## Highlights & Insights
- The concept of "inference-time intervention" is highly elegant—zero parameter modification, zero training costs, and a matrix learned once that can be reused for all inputs. This is a highly cost-efficient cross-lingual enhancement method, particularly ideal for resource-constrained deployment scenarios.
- The linear alignment feasibility of cross-lingual representation spaces is validated at the LLM level, representing an important generalization of earlier static word embedding alignment studies (e.g., MUSE).
- Learning alignment matrices requires only parallel sentence pairs rather than annotated data, heavily lowering the data acquisition threshold—pseudo-parallel pairs generated by machine translation can even be utilized.

## Limitations & Future Work
- The linear alignment hypothesis may not hold for language pairs with major morphological discrepancies (e.g., English-Japanese), prompting future exploration into non-linear mappings.
- The alignment matrix is static and cannot adapt to context variation—the same sentence may require different alignment depending on the context.
- Experiments are only validated on discriminative and short-form generation tasks; the effectiveness on long-form text generation (such as summarization or translation) remains to be confirmed.
- The intervention might potentially disrupt the model's source language comprehension in certain extreme cases—such as incorrectly mapping cultural concepts unique to the source language.
- Multi-layer joint intervention and adaptive intervention strength could be explored to further enhance performance.

## Related Work & Insights
- **vs Multilingual Fine-Tuning**: Fine-tuning requires substantial computational resources and can lead to catastrophic forgetting; INCLINE has zero training costs and does not modify parameters.
- **vs Translate-then-Answer**: Translation introduces error accumulation and increases inference latency; INCLINE operates directly in the representation space without translation errors.
- **vs Language Adapters (e.g., MAD-X)**: Adapters still require training and independent parameter maintenance for each language; INCLINE requires only a single alignment matrix.
- **vs MUSE/VecMap**: Classic word embedding alignment works focused on static representations, whereas INCLINE generalizes this concept to contextualized LLM representations.

## Rating
- Novelty: ⭐⭐⭐⭐ Inference-time cross-lingual representation intervention offers an elegant new perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely thorough, with validation across 5 LLMs, 9 benchmarks, and multiple languages.
- Writing Quality: ⭐⭐⭐⭐ The methodology is concise and clear.
- Value: ⭐⭐⭐⭐⭐ High practical value—zero-cost multilingual enhancement, open-source code.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Cross-Lingual Optimization for Language Transfer in Large Language Models](cross-lingual_optimization_for_language_transfer_in_large_language_models.md)
- [\[ACL 2025\] Cross-Lingual Pitfalls: Automatic Probing Cross-Lingual Weakness of Multilingual Large Language Models](crosslingual_pitfalls.md)
- [\[ACL 2025\] Middle-Layer Representation Alignment for Cross-Lingual Transfer in Fine-Tuned LLMs](mid_layer_crosslingual_alignment.md)
- [\[ACL 2025\] Semantic Aware Linear Transfer by Recycling Pre-trained Language Models for Cross-Lingual Transfer](semantic_aware_linear_transfer_by_recycling_pre-trained_language_models_for_cros.md)
- [\[ACL 2025\] Language Fusion for Parameter-Efficient Cross-lingual Transfer (FLARE)](flare_crosslingual_lora.md)

</div>

<!-- RELATED:END -->

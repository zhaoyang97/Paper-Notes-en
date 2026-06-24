---
title: >-
  [Paper Note] Vision-Language Models Create Cross-Modal Task Representations
description: >-
  [ICML 2025][Multimodal VLM][Task vectors] This paper discovers that autoregressive vision-language models (VLMs) compress conceptually equivalent inputs (whether text/image examples, instructions, or few-shot prompts) into a shared "task vector". It validates the existence and utility of such representational alignment through cross-modal patching experiments.
tags:
  - "ICML 2025"
  - "Multimodal VLM"
  - "Task vectors"
  - "cross-modal representations"
  - "VLM inner mechanisms"
  - "cross-modal transfer"
  - "in-context learning"
date: 2026-05-08
content_hash: 1bb5f9925c940d37
---

# Vision-Language Models Create Cross-Modal Task Representations

**Conference**: ICML 2025  
**arXiv**: [2410.22330](https://arxiv.org/abs/2410.22330)  
**Code**: [https://github.com/g-luo/vlm_cross_modal_reps](https://github.com/g-luo/vlm_cross_modal_reps)  
**Area**: Multimodal VLM  
**Keywords**: Task vectors, cross-modal representations, VLM inner mechanisms, cross-modal transfer, in-context learning

## TL;DR
This paper discovers that autoregressive vision-language models (VLMs) compress conceptually equivalent inputs (whether text/image examples, instructions, or few-shot prompts) into a shared "task vector". It validates the existence and utility of such representational alignment through cross-modal patching experiments.

## Background & Motivation

**Background**: Autoregressive VLMs (such as Idefics2, LLaVA, etc.) can handle various tasks within a single model—switching tasks by taking different in-context examples or instructions. The internal representational mechanisms of this flexibility remain unclear.

**Limitations of Prior Work**: 
   - Prior research on pure language models discovered the existence of "task vectors"—where hidden states at specific positions near the end of a sequence encode current task information.
   - However, whether similar cross-modal task vectors exist in VLMs is completely unknown.
   - **Failure of cross-modal few-shot prompting**: Using text examples to define a task followed by an image query results in extremely poor model performance, implying some bottleneck in VLM multimodal integration.

**Key Challenge**: While VLMs can handle text and image tasks separately, cross-modal few-shot prompting fails significantly. This indicates that there are issues in the **cross-modal transmission of the full prompt**. However, if a **compressed task representation** exists, is it possible for it to cross the modal gap?

**Goal**: Investigate whether VLMs internally form **modality-agnostic** shared task representations, and how these representations can be leveraged to fix cross-modal failures.

**Key Insight**: Drawing inspiration from the research paradigm of task vectors in LLMs, design **cross-modal patching experiments** to extract task vectors from one modality and inject them into the inference process of another modality.

**Core Idea**: Modality-agnostic task vectors exist in VLMs; a task vector from one modality can be directly used to drive correct generation in another modality.

## Method

### Overall Architecture
1. Select a specific token position (typically the end of the context sequence) in the VLM, where the middle-layer hidden states are designated as the **task vector**.
2. **Cross-Modal Patching**: Extract the hidden state $\mathbf{h}^{src}_l$ at this position from the forward pass of the **source modality** (e.g., text examples), and inject it into the same position during the forward pass of the **target modality** (e.g., image query).
3. The model then continues generation based on the patched hidden state, observing whether it can yield the correct task-specific output.

### Key Designs

1. **Cross-Modal Patching**:

    - Given task $T$, two modalities are defined: text examples $\{(x^{text}_i, y_i)\}$ and image examples $\{(x^{img}_i, y_i)\}$.
    - **Source run**: Use text examples as a few-shot prompt for a forward pass, and extract the hidden state $\mathbf{h}^{text}_l$ of the end token at layer $l$.
    - **Target run**: Use the image query $x^{img}_{query}$ (without any task context) for a forward pass. When reaching layer $l$, **replace** the hidden state at the corresponding position with $\mathbf{h}^{text}_l$.
    - The model continues inference and generates output from layer $l$ onward.
    - **Core formula**: $\hat{y} = \text{VLM}(\text{Patch}(x^{img}_{query}, \mathbf{h}^{text}_l, l))$
    - **Design Motivation**: If cross-modal patching succeeds (yielding correct task-specific answers), it demonstrates that task information from different modalities is indeed compressed into a unified representation space.

2. **Cross-Model Patching (LLM $\rightarrow$ VLM Transfer)**:

    - Many VLMs are fine-tuned from pretrained LLMs (e.g., LLaVA is based on Vicuna).
    - Extract the task vector $\mathbf{h}^{LLM}_l$ using text examples in the LLM.
    - Patch it into the corresponding position of the VLM when processing an image query.
    - **Design Motivation**: To verify whether the task vectors are preserved during the fine-tuning process. If so, it suggests that the VLM inherits the task representation capabilities of the LLM, and this representation is inherently "cross-modal".

3. **Instruction-based Task Vectors**:

    - In addition to using few-shot examples to define a task, natural language instructions (e.g., "Output the capital of this country") can also be used.
    - Input the instruction into the model and extract the hidden state at the same position as the task vector.
    - Furthermore, exemplar-based task vectors and instruction-based task vectors can be **combined** (via weighted average).
    - **Design Motivation**: Instructions are more concise than examples. If instructions can generate effective task vectors, the complexity of prompt design will be significantly reduced.

### Loss & Training
This paper is a **purely analytical/interpretability-focused work** and does not involve training. All experiments are conducted on pre-trained VLMs, including Idefics2-8B, LLaVA-1.5-7B, Qwen-VL, etc.

## Key Experimental Results

### Main Results

| Method | Country→Capital | Antonym | Translation | Object→Color | Average Accuracy |
|------|----------------|---------|-------------|-------------|-----------|
| Text Prompt (Text→Image) | 12.3% | 8.7% | 5.2% | 15.1% | 10.3% |
| Image Prompt (Image→Image) | **78.5%** | **72.3%** | **68.1%** | **82.4%** | 75.3% |
| Text Patch (Text→Image) | **76.2%** | **70.8%** | **65.4%** | **80.1%** | **73.1%** |
| Instruction Patch | 71.5% | 65.2% | 60.3% | 75.8% | 68.2% |
| LLM→VLM Patch | 68.3% | 62.1% | 58.7% | 72.5% | 65.4% |

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| Selection of Patching Layer (Shallow/Middle/Deep) | Middle layers are optimal (~73%) | Task vectors are most clearly formed in the middle layers |
| Number of Examples (1/2/4/8-shot) | Saturates at 4-shot | Few examples are sufficient to form stable task vectors |
| Model Scale (7B vs 13B) | Larger models perform slightly better (+3%) | Representation alignment is stronger in larger models |
| Task Vector Ensemble (Example + Instruction) | +4.2% vs. using examples only | The two information sources are complementary |
| Patching Single vs. Multiple Layers | Patching a single layer is effective | Task information is highly compressed at a specific position |

### Key Findings
1. **Cross-modal prompting fails, but patching succeeds**: Directly prompting an image query using text examples yields only ~10% accuracy, whereas patching achieves ~73%—close to the ~75% performance of in-modality prompting.
2. **Task vectors are modality-agnostic**: t-SNE visualizations show that task vectors of different modalities cluster by task rather than by modality.
3. **LLM task vectors transfer to VLMs**: Task vectors derived from the foundation LLM remain effective after patching into the fine-tuned VLM.
4. **Compression outperforms raw information**: A single task vector (one vector) outperforms the complete few-shot prompt under cross-modal scenarios—information compression effectively eliminates modal interference.

## Highlights & Insights
- **Most surprising discovery**: A compressed task vector performs better in cross-modal scenarios than a complete few-shot prompt. This implies that the failure of cross-modal full prompts is not due to insufficient information, but rather due to **formatting interference** between modalities.
- **Deep insight into VLM internal mechanisms**: VLMs do not simply process text and images separately and then concatenate them—they form truly unified semantic representations in the middle layers.
- **Practical value**: Task vector patching can serve as an efficient cross-modal adaptation tool without the need for re-prompting.

## Limitations & Future Work
- The experimental tasks are relatively simple (e.g., country capitals, antonyms), and more complex visual reasoning tasks have yet to be verified.
- The optimal patching layer needs to be manually selected, and no automated layer selection strategy has been proposed.
- The behavior of task vectors in dialogue/multi-turn interaction scenarios has not been thoroughly investigated.
- Only encoder-decoder and decoder-only architectures were covered; other VLM architecture types remain unexplored.

## Related Work & Insights
- Directly corresponds to the works of Hendel et al. (2023) and Todd et al. (2024), which discovered task vectors in LLMs.
- Provides practical implications for prompt engineering in VLMs: Poor performance in cross-modal few-shot scenarios? Try patching.
- Introduces a new paradigm for the study of representational alignment in VLMs.
- The success of cross-model patching implies that VLM fine-tuning mainly affects input processing rather than task representations.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Discovers cross-modal task vectors in VLMs for the first time; the finding that patching outperforms full prompt is highly insightful.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated across multiple models and tasks with t-SNE visualizations and detailed ablations, though task complexity remains somewhat low.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear storyline, with four progressive findings (Finding 1-4) well-organized.
- Value: ⭐⭐⭐⭐⭐ Significant contribution to understanding the internal working mechanisms of VLMs, offering both theoretical insight and practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Guiding Cross-Modal Representations with MLLM Priors via Preference Alignment](../../NeurIPS2025/multimodal_vlm/guiding_cross-modal_representations_with_mllm_priors_via_preference_alignment.md)
- [\[ICLR 2026\] FLARE: Fully Integration of Vision-Language Representations for Deep Cross-Modal Understanding](../../ICLR2026/multimodal_vlm/flare_fully_integration_of_vision-language_representations_for_deep_cross-modal_.md)
- [\[CVPR 2025\] Task Preference Optimization: Improving Multimodal Large Language Models with Vision Task Alignment](../../CVPR2025/multimodal_vlm/task_preference_optimization_improving_multimodal_large_language_models_with_vis.md)
- [\[ACL 2026\] Cross-Modal Taxonomic Generalization in (Vision-) Language Models](../../ACL2026/multimodal_vlm/cross-modal_taxonomic_generalization_in_vision-_language_models.md)
- [\[ICML 2025\] Core Knowledge Deficits in Multi-Modal Language Models](core_knowledge_deficits_in_multi-modal_language_models.md)

</div>

<!-- RELATED:END -->

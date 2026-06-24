---
title: >-
  [Paper Note] MG-MotionLLM: A Unified Framework for Motion Comprehension and Generation across Multiple Granularities
description: >-
  [CVPR 2025][LLM (Other)][Motion Comprehension] MG-MotionLLM proposes a unified multi-granularity motion-language model. Leveraging a Motion VQ-VAE + T5 language model architecture along with a carefully designed multi-granularity synergy pre-training scheme (comprising 28 tasks), it simultaneously supports coarse- and fine-grained motion comprehension and generation. While achieving state-of-the-art performance on classic tasks, it also enables novel applications such as fine…
tags:
  - "CVPR 2025"
  - "LLM (Other)"
  - "Motion Comprehension"
  - "Multi-Granularity"
  - "Large Language Models"
  - "Motion Generation"
  - "Fine-Grained Editing"
date: 2026-05-08
content_hash: 1622c1af0f76a832
---

# MG-MotionLLM: A Unified Framework for Motion Comprehension and Generation across Multiple Granularities

**Conference**: CVPR 2025  
**arXiv**: [2504.02478](https://arxiv.org/abs/2504.02478)  
**Code**: [https://github.com/CVI-SZU/MG-MotionLLM](https://github.com/CVI-SZU/MG-MotionLLM)  
**Area**: LLM/NLP  
**Keywords**: Motion Comprehension, Multi-Granularity, Large Language Models, Motion Generation, Fine-Grained Editing

## TL;DR
MG-MotionLLM proposes a unified multi-granularity motion-language model. Leveraging a Motion VQ-VAE + T5 language model architecture along with a carefully designed multi-granularity synergy pre-training scheme (comprising 28 tasks), it simultaneously supports coarse- and fine-grained motion comprehension and generation. While achieving state-of-the-art performance on classic tasks, it also enables novel applications such as fine-grained motion editing.

## Background & Motivation

1. **Background**: Existing motion-aware large language models (e.g., MotionGPT, MotionLLM) have demonstrated the potential of unified motion comprehension and generation. However, they mainly focus on coarse-grained motion-text modeling, where the text typically describes the overall semantics of the entire motion sequence using only a few words.

2. **Limitations of Prior Work**: Coarse-grained descriptions cannot handle fine-grained motion-related tasks, such as understanding and controlling the movement of specific body parts. Existing attempts to introduce detailed descriptions (e.g., SemanticBoost, MotionScript) only focus on fine-grained generation and fail to integrate fine-grained comprehension.

3. **Key Challenge**: Detailed descriptions contain an enormous amount of information (some exceeding 1000 tokens), making it highly difficult to establish direct correspondences with compact motion tokens (fewer than 50). The authors' experiments show that when direct instructions with both coarse and fine descriptions are used for generation, the Top-3 retrieval accuracy actually drops from 77.3% to 75.0%.

4. **Goal**: How to simultaneously support multi-granularity motion comprehension and generation within a unified model, allowing coarse- and fine-grained tasks to mutually enhance each other.

5. **Key Insight**: Focus on the fine-grained descriptions of short motion segments (snippets) to first establish local motion-detailed text relationships, and then extend them to the global scope. Auxiliary tasks (such as temporal grounding and snippet description) are designed to let tasks across different granularities promote each other.

6. **Core Idea**: Implement cooperative pre-training on 28 motion-related tasks of varying granularities, enabling coarse-grained tasks to assist fine-grained tasks in capturing semantics, while fine-grained tasks conversely improve the detail comprehension of coarse-grained tasks.

## Method

### Overall Architecture
MG-MotionLLM consists of two core components: (1) a Motion VQ-VAE, which encodes raw motion data into discrete motion tokens and decodes tokens back into motion sequences; and (2) a T5-based motion-aware language model, which unifies the modeling of motion and text tokens by extending the vocabulary. The input is an instruction sequence containing text and/or motion tokens at various granularities, and the output is the corresponding sequence of text or motion tokens.

### Key Designs

1. **Motion VQ-VAE (Motion Discretization)**:

    - **Function**: Encode a continuous $T$-frame motion sequence $\bm{M}$ into $T/l$ discrete tokens.
    - **Mechanism**: Use an encoder to map motions into a latent space, quantize them into discrete tokens via a codebook, and reconstruct the motion with a decoder. The reconstruction loss, embedding loss, and commitment loss ($\mathcal{L}_{\text{VQVAE}} = \|M - \hat{M}\|_2 + \|SG(Z) - \hat{Z}\|_2 + \beta\|Z - SG(\hat{Z})\|_2$) are used for training. The parameters are frozen after training.
    - **Design Motivation**: Discretization allows motion to be processed by LLMs like language vocabulary, achieving unified modeling of motion and language. Reusing the design of T2M-GPT ensures a fair comparison.

2. **Motion-Aware Language Model (Unified Vocabulary Design)**:

    - **Function**: Unify motion comprehension and generation into a single seq2seq problem.
    - **Mechanism**: Extend the text vocabulary of T5 $V_t$ to include the motion vocabulary $V_m$ (codebook indices) and special tokens $V_s$ (e.g., `<Motion Tokens>`, `<SEP>`, `<Motionless>`). All tasks are formatted as "input token sequence $\rightarrow$ output token sequence" and trained using cross-entropy loss $\mathcal{L}_{CE} = -\sum \log P(v^i_{out} | X_{in}, v^j_{out}, \theta)$.
    - **Design Motivation**: Employing T5's encoder-decoder architecture as a conditional generative model is naturally suited to unify multimodal tasks into text generation tasks.

3. **Granularity-Synergy Pre-training**:

    - **Function**: Effectively establish correspondences between motion and text at different granularities.
    - **Mechanism**: Design 28 tasks (12 classic coarse-grained + 16 newly proposed fine-grained) spanning various combinations of three types of information: textual descriptions, temporal information, and motion data. Key innovations include: (a) starting from short motion snippets and their corresponding detailed descriptions to establish local relationships before generalizing globally; (b) introducing a temporal grounding task (locating temporal boundaries of motion snippets based on detailed text); and (c) excluding motion generation tasks that only contain fine-grained descriptions (since different motions may share identical fine-grained descriptions, such as both "walking" and "running" involving alternating leg movements).
    - **Design Motivation**: Directly training on fine-grained tasks yields sub-optimal performance (too much information leading to a loss of global semantics). Coarse-grained tasks are needed to assist in semantic capture, while fine-grained tasks in turn enhance the coarse-grained tasks' understanding of details, creating a mutually beneficial relationship.

### Loss & Training
A two-stage training strategy is adopted:
- **Stage 1 (Granularity-Synergy Pre-training)**: Jointly train on 28 tasks with a learning rate of $2 \times 10^{-4}$, a batch size of 16, for 300K iterations.
- **Stage 2 (Task-Specific Instruction Tuning)**: Fine-tune on specific tasks with a learning rate of $10^{-4}$ for 300K iterations.
- Both stages leverage the AdamW optimizer and are conducted on a single A100 80G GPU.

## Key Experimental Results

### Main Results

**Text-to-Motion (HumanML3D)**:

| Method | Type | R-Top3↑ | FID↓ | MM-Dist↓ | Diversity↑ |
|------|------|---------|------|----------|------------|
| MoMask (CVPR'24) | Generation-Only | 0.807 | **0.045** | 2.958 | - |
| MotionGPT (NeurIPS'23) | Unified | 0.778 | 0.232 | 3.096 | 9.528 |
| **MG-MotionLLM** | Unified | **0.802** | 0.303 | **2.952** | **9.960** |

**Motion-to-Text (HumanML3D)**:

| Method | R-Top1↑ | R-Top3↑ | MM-Dist↓ | BertScore↑ |
|------|---------|---------|----------|------------|
| MotionGPT | 0.543 | 0.827 | 2.821 | 32.4 |
| **MG-MotionLLM** | **0.592** | **0.866** | **2.581** | **36.7** |

### Ablation Study

| Pre-training Granularity | Instruction Tuning | T2M Top3↑ | M2T Top1↑ | M2DT BertScore↑ |
|-----------|---------|-----------|-----------|-----------------|
| Coarse-only | ✗ | 0.725 | 0.431 | - |
| Fine-only | ✗ | - | - | 43.1 |
| Coarse + Fine | ✗ | 0.767 | 0.514 | 47.7 |
| **Coarse + Fine** | **✓** | **0.802** | **0.592** | **52.3** |
| No Pre-training | ✓ | 0.773 | 0.516 | 50.5 |

### Key Findings
- **Multi-granularity synergy pre-training yields significant gains**: Compared to using either granularity in isolation, joint coarse+fine-grained pre-training improves performance across all tasks, validating the key hypothesis that tasks at different granularities mutually promote each other.
- **Instruction tuning is necessary**: The number of samples allocated to each task during pre-training is limited (approx. 1/30), and task-specific fine-tuning can further unlock performance.
- **Model scale affects fine-grained tasks more significantly**: T5-Large improves by approximately 4 Bleu@4 points compared to T5-Small on the snippet level of Motion-to-Detailed Text, whereas the performance gap is smaller on coarse-grained tasks.

## Highlights & Insights
- **Mutually beneficial multi-granularity training paradigm**: Coarse-grained tasks assist fine-grained tasks in capturing global semantics (since fine-grained descriptions are too long and cause information overload), while fine-grained tasks help coarse-grained tasks understand local details. This "simple-to-complex" design philosophy is highly elegant and transferrable to other multi-scale comprehension tasks.
- **Unified framework enabling entirely new applications**: Motion editing (temporal, spatial, and spatio-temporal editing) via fine-grained descriptions is unprecedented. Users can complete the pipeline with a single model: generate an initial motion $\rightarrow$ obtain a fine-grained description $\rightarrow$ edit the description $\rightarrow$ regenerate. This interaction paradigm is highly inspiring.
- **Auxiliary task design**: The temporal grounding task (locating temporal boundaries of motion snippets based on text descriptions) is not only a meaningful standalone task but also serves as a bridge to establish correspondences between motion and long texts.

## Limitations & Future Work
- **Supports only single-person motion**: The current system only processes single-person motion sequences and cannot understand multi-person interaction scenarios.
- **Reliance on T2M-GPT's VQ-VAE**: The quality of the motion tokenizer directly determines the performance upper bound, but better tokenizers (such as RVQ) were not explored.
- **Insufficient evaluation of fine-grained description generation**: Motion-to-Detailed Text is mainly evaluated using automatic metrics, lacking human evaluation and real-world application verification.
- **Templatized descriptions in the FineMotion dataset** may limit the model's generalization capability to natural language fine-grained descriptions.
- Future research directions: Integrate RVQ with multi-layer codebooks to improve motion reconstruction quality; extend to multi-person interaction scenarios; explore joint training with the visual modality (video).

## Related Work & Insights
- **vs MotionGPT [Jiang et al., NeurIPS'23]**: MotionGPT also uses T5 to unify motion and language, but only handles coarse-grained tasks. MG-MotionLLM extends this to the fine-grained domain and comprehensively outperforms it on coarse-grained tasks as well (R-Top3 improvement of 2.4%).
- **vs FineMoGen [Zhang et al., NeurIPS'24]**: FineMoGen attempts fine-grained motion generation but designs a specialized network, focusing solely on generation without comprehension. MG-MotionLLM addresses both comprehension and generation using a unified model.
- **vs SemanticBoost/MotionScript**: These methods translate body part movements into predefined states to enhance generation, but they only focus on generation rather than comprehension and lack temporal information.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The idea of multi-granularity synergy training is novel and effective, although the overall framework (VQ-VAE + T5) is a combination of existing methods.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Ablation studies comprehensively validate the contribution of each component, but human evaluations and downstream application testing are lacking.
- **Writing Quality**: ⭐⭐⭐⭐ Clear logic, natural motivation derivation, and reasonable design of tables and figures.
- **Value**: ⭐⭐⭐⭐ Opens up research directions for multi-granularity motion comprehension and generation, with practical value in fine-grained motion editing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] SteerEval: How Controllable Are Large Language Models? A Unified Evaluation across Behavioral Granularities](../../ACL2026/llm_nlp/how_controllable_are_large_language_models_a_unified_evaluation_across_behaviora.md)
- [\[ACL 2025\] SSUF: A Semi-supervised Scalable Unified Framework for E-commerce Query Classification](../../ACL2025/llm_nlp/a_semi-supervised_scalable_unified_framework_for_e-commerce_query_classification.md)
- [\[ACL 2025\] One for All: Update Parameterized Knowledge Across Multiple Models with Once Edit](../../ACL2025/llm_nlp/one_for_all_update_parameterized_knowledge_across_multiple_models_with_once_edit.md)
- [\[ACL 2025\] BIPro: Zero-shot Chinese Poem Generation via Block Inverse Prompting Constrained Generation Framework](../../ACL2025/llm_nlp/bipro_zero-shot_chinese_poem_generation_via_block_inverse_prompting_constrained_.md)
- [\[ICCV 2025\] VIM: Versatile Interactive Motion-Language Model](../../ICCV2025/llm_nlp/vim_versatile_interactive_motion_language_model.md)

</div>

<!-- RELATED:END -->

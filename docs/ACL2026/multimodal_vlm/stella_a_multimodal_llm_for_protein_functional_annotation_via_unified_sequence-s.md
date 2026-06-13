---
title: >-
  [Paper Note] STELLA: A Multimodal LLM for Protein Functional Annotation via Unified Sequence-Structure Encoding
description: >-
  [ACL2026][Multimodal VLM][Protein Functional Annotation] STELLA integrates the unified sequence-structure protein representation of ESM3 into Llama-3.1-8B-Instruct. Through two-stage multimodal instruction tuning…
tags:
  - "ACL2026"
  - "Multimodal VLM"
  - "Protein Functional Annotation"
  - "Multimodal LLM"
  - "ESM3"
  - "Structure-Sequence Encoding"
  - "OPI-Struc"
date: 2026-05-08
content_hash: 5a436511c9e04ff5
---

# STELLA: A Multimodal LLM for Protein Functional Annotation via Unified Sequence-Structure Encoding

**Conference**: ACL2026 Findings  
**arXiv**: [2506.03800](https://arxiv.org/abs/2506.03800)  
**Code**: https://github.com/ocx-lab/STELLA  
**Area**: Multimodal VLM / Protein Functional Annotation  
**Keywords**: Protein Functional Annotation, Multimodal LLM, ESM3, Structure-Sequence Encoding, OPI-Struc

## TL;DR
STELLA integrates the unified sequence-structure protein representation of ESM3 into Llama-3.1-8B-Instruct. Through two-stage multimodal instruction tuning, it performs protein functional description and enzyme catalytic reaction prediction, refreshing multiple functional annotation metrics on the OPI-Struc benchmark series.

## Background & Motivation
**Background**: The core chain in protein research consists of sequence, structure, and function. In recent years, structure prediction and structural databases have expanded rapidly, yet a vast number of proteins still lack high-quality functional annotations. Protein language models (pLMs) can learn representations from sequence or structure, while LLMs excel at articulating knowledge in natural language.

**Limitations of Prior Work**: Many protein-text models employ separate sequence encoders, structure encoders, and additional fusion modules, resulting in complex architectures and unstable optimization. More importantly, many pLMs perform well only in latent representations or specific attribute predictions, lacking generative capabilities for natural language functional descriptions.

**Key Challenge**: Protein functional annotation requires a simultaneous understanding of structural geometry, sequence context, and textual knowledge. Pure pLMs are not proficient at generating detailed explanations, while pure LLMs lack protein structure inputs; multi-encoder fusion schemes can access multiple modalities, but at the cost of high system complexity and cross-modal alignment overhead.

**Goal**: The authors aim to construct a more concise protein multimodal LLM: utilizing a unified sequence-structure encoder for protein inputs, connected via a lightweight connector to a general instruction LLM. This allows the model to output reliable text or category judgments for both functional description and enzyme function prediction tasks.

**Key Insight**: The paper selects ESM3 as the unified protein encoder because it inherently tracks sequence, structure, and function-related tokens within the same representation space. Llama-3.1-8B-Instruct is then used to handle natural language generation and instruction following.

**Core Idea**: The "encoder + connector + LLM" paradigm from vision-language models is transferred to protein functional annotation. By using ESM3 for unified protein sequence-structure representation, the cost of multi-encoder fusion is reduced, and multimodal instruction tuning is supported through the OPI-Struc benchmark.

## Method

### Overall Architecture
This note summarizes STELLA from the perspective of the computational model and benchmark. The architecture of STELLA is similar to LLaVA: a protein encoder encodes sequence-structure information into high-dimensional representations, a linear modality connector maps these representations into the LLM's latent space, and Llama-3.1-8B-Instruct outputs functional descriptions or predictions based on natural language instructions.

Training employs a two-stage Multimodal Instruction Tuning. Stage 1 freezes ESM3 and the LLM while training only the connector for cross-modal representation alignment. Stage 2 continues to freeze ESM3 while training both the connector and the LLM, enabling the model to follow protein-related natural language instructions and complete task outputs. Due to the scarcity of high-quality protein-text instruction data, both stages utilize the same carefully constructed OPI-Struc data resource, though the optimization objectives and trainable modules differ.

### Key Designs
1. **ESM3 Unified Sequence-Structure Encoder**:

	- **Function**: Uses a single protein foundation model to carry both sequence and 3D structural information.
	- **Mechanism**: ESM3 organizes multiple protein modalities into a unified embedding space and introduces geometric attention in early transformer blocks, which facilitates capturing structural topology and local spatial relationships. STELLA directly utilizes `esm3_sm_open_v1` as the protein encoder.
	- **Design Motivation**: Compared to using separate sequence and structure encoders followed by complex fusion modules, a unified encoder reduces architectural fragmentation and allows the subsequent connector to focus solely on mapping protein representations to the text space.

2. **Lightweight Connector and LLM Generation Head**:

	- **Function**: Converts protein representations into language model input representations consumable by Llama.
	- **Mechanism**: The authors use a single-layer linear projection as the modality connector. The LLM utilizes Llama-3.1-8B-Instruct, responsible for parsing task instructions and outputting functional narratives, multiple-choice answers, or enzyme reaction names.
	- **Design Motivation**: Linear connectors have simple structures, stable training, and low computational costs. Delegating complex reasoning and linguistic expression to the LLM compensates for the difficulty traditional pLMs face in generating natural language functional explanations.

3. **OPI-Struc Multimodal Instruction Benchmark**:

	- **Function**: Provides unified training and evaluation resources for protein structure-to-text tasks.
	- **Mechanism**: OPI-Struc comprises Function and Enzyme domains. The Function domain includes free-text functional description tasks and multiple-choice functional recognition tasks; the Enzyme domain represents enzyme catalytic functions as standard name predictions. Evaluation includes FP_ft_eval, out-of-time FP_ft_eval_v2401, FP_mc_eval_1x, FP_mc_eval_4x, and EP_eval.
	- **Design Motivation**: Since free-text metrics can be affected by phrasing differences, the authors included MCQA and enzyme function accuracy to ensure the benchmark covers both generative quality and objective discriminative ability.

### Loss & Training
The training focus of the paper is multimodal instruction tuning rather than new loss functions. Stage 1 only updates the connector, while Stage 2 updates the connector and the LLM, with different learning rates set for different components. FP tasks use BLEU-4, BERTScore, and ROUGE-1/2/L to measure textual similarity and semantic consistency; FP-MCQA and EP tasks use Accuracy as the objective metric.

## Key Experimental Results

### Main Results
In the functional description hold-out evaluation, STELLA significantly outperforms the Foldseek retrieval baseline, Prot2Text, and ProteinChat.

| Method | BLEU-4 | BERTScore | ROUGE-1 | ROUGE-2 | ROUGE-L |
|------|--------|-----------|---------|---------|---------|
| Foldseek | 0.3627 | 0.8358 | 0.4799 | 0.4027 | 0.4586 |
| Prot2TextBASE | 0.3511 | 0.8430 | 0.5059 | 0.4271 | 0.4849 |
| Prot2TextLARGE | 0.3629 | 0.8520 | 0.5368 | 0.4560 | 0.5140 |
| ProteinChat | 0.1918 | 0.7970 | 0.3957 | 0.2799 | 0.3648 |
| STELLA (e3+e6) | 0.4300 | 0.8564 | 0.5423 | 0.4747 | 0.5257 |

In the structural degradation robustness evaluation, the ROUGE-L decline of STELLA was smaller than that of Prot2TextLARGE, indicating that the unified structural representation is more stable against incomplete inputs.

| Model | Complete ROUGE-L | Incomplete ROUGE-L | Performance Drop |
|------|------------------|--------------------|----------|
| Prot2TextLARGE | 0.5140 | 0.4438 | 13.7% |
| STELLA (e3+e3) | 0.5041 | 0.4805 | 4.7% |
| STELLA (e3+e6) | 0.5257 | 0.4915 | 4.1% |

In enzyme catalytic reaction prediction (EP_eval), STELLA refreshed the accuracy upper bound.

| Method | Accuracy |
|------|----------|
| UniRep | 72.90 |
| 3DCNN | 78.80 |
| IEConv | 87.20 |
| CDConv | 88.50 |
| GearNet-Multiview-Contrast | 87.50 |
| Sable | 88.50 |
| STELLA (e3+e3) | 88.06 |
| STELLA (e3+e6) | 88.85 |

### Ablation Study
Comparisons of encoders, LLM backbones, and training stages demonstrate that STELLA's performance stems from the combined effect of the unified protein encoder, a suitable LLM, and two-stage training.

| Variant | BLEU-4 | BERTScore | ROUGE-1 | ROUGE-2 | ROUGE-L | Observation |
|------|--------|-----------|---------|---------|---------|------|
| STELLA-ESM3-Llama-3.1-8B | 0.4024 | 0.8496 | 0.5218 | 0.4487 | 0.5041 | ESM3 + Llama-3.1 is the strongest overall |
| STELLA-ESM3-Llama-3-8B | 0.4020 | 0.8503 | 0.5138 | 0.4478 | 0.5001 | Close but slightly lower |
| STELLA-ESM3-Phi-3-mini | 0.3807 | 0.8435 | 0.4991 | 0.4273 | 0.4839 | Smaller model generation head is weaker |
| STELLA-Prot2Text-Llama-3.1-8B | 0.4009 | 0.8497 | 0.5284 | 0.4454 | 0.5031 | Prot2Text encoder is still strong, but below best ESM3 config |
| STELLA-SaProt-Llama-3-8B | 0.3588 | 0.8276 | 0.4685 | 0.3965 | 0.4523 | SaProt configuration is significantly weaker |

Two-stage training outperformed single-stage training, particularly with significant gains in ROUGE-L and BLEU-4.

| Strategy | Stage 1 epoch | Stage 2 epoch | BLEU-4 | BERTScore | ROUGE-1 | ROUGE-2 | ROUGE-L |
|------|---------------|---------------|--------|-----------|---------|---------|---------|
| Single-stage | - | e1 | 0.2233 | 0.7885 | 0.3530 | 0.2631 | 0.3350 |
| Single-stage | - | e3 | 0.3642 | 0.8363 | 0.4840 | 0.4073 | 0.4660 |
| Two-stage | e3 | e1 | 0.2653 | 0.8065 | 0.3938 | 0.3097 | 0.3770 |
| Two-stage | e3 | e3 | 0.4024 | 0.8496 | 0.5218 | 0.4487 | 0.5041 |

The out-of-time OOD functional description evaluation shows a significant drop in performance for all methods, indicating that recently annotated proteins remain a difficult scenario.

| Model | BLEU-4 | BERTScore | ROUGE-1 | ROUGE-2 | ROUGE-L |
|------|--------|-----------|---------|---------|---------|
| STELLA-ESM3-Llama-3.1-8B | 0.0489 | 0.7565 | 0.2210 | 0.1085 | 0.1867 |
| STELLA-Prot2Text-Llama-3.1-8B | 0.0425 | 0.7555 | 0.2454 | 0.1020 | 0.1919 |
| STELLA-Prot2Text-Mistral-7B | 0.0440 | 0.7685 | 0.2529 | 0.1046 | 0.1975 |
| ProteinChat | 0.0205 | 0.7413 | 0.2121 | 0.0855 | 0.1691 |

### Key Findings
- STELLA achieved a ROUGE-L of 0.5257 on the FP hold-out, which is higher than Prot2TextLARGE's 0.5140 and 14.6% higher than the Foldseek retrieval-based baseline.
- In FP-MCQA, STELLA achieved 80.56 and 76.18 accuracy under fixed and four-fold option permutations respectively, demonstrating that the model can perform discriminative functional selection in addition to text generation.
- In EP_eval, extending Stage 2 from e3 to e6 improved accuracy from 88.06 to 88.85, surpassing CDConv and Sable's 88.50.
- Higher structural homology density correlates with more accurate functional descriptions; ROUGE-L rises from 0.4323 for near-unique structures to 0.6691 for high-density structural clusters, identifying OOD structures as a core challenge.

## Highlights & Insights
- The primary highlight is the architectural simplification: using ESM3 to uniformly carry sequence and structure information, avoiding the complex fusion of multiple protein encoders.
- OPI-Struc places free-text, multiple-choice, and enzyme function prediction under a single instruction benchmark, making the evaluation of protein multimodal LLMs more systematic.
- The two-stage training design aligns with multimodal LLM empirical evidence: aligning representations first and then training instruction following reduces the risk of the LLM over-fitting to textual patterns prematurely.
- The OOD results in the paper are remarkably honest: while hold-out results are strong, the out-of-time evaluation shows a significant decline, suggesting that current protein functional annotation models are still far from generalizing to new knowledge based solely on existing structure-text pairs.

## Limitations & Future Work
- Current performance is limited by the granularity of structural tokenization and the difficulty of aligning high-dimensional geometric features with discrete text tokens.
- The model relies heavily on the general knowledge and reasoning of the LLM, which may lack sufficiently fine-grained domain judgment in highly specialized functional discovery scenarios.
- The significant drop in metrics on the OOD temporal benchmark indicates that recent functions, new structural motifs, and long-tail labels are still difficult to cover.
- Existing evaluation metrics are mostly derived from NLP, such as BLEU and ROUGE, which may not fully reflect the biological semantic correctness of functional annotations; MCQA and EP accuracy are supplementary but not yet comprehensive.
- Future valuable directions include high-resolution structural adapters, retrieval-augmented functional knowledge, and more rigorous multimodal protein benchmarks.

## Related Work & Insights
- **vs. Prot2Text**: Prot2Text mapped protein structures to functional text early on but relied on traditional encoder-decoder combinations; STELLA uses ESM3 + Llama instruction models to improve generative capacity and structural uniformity.
- **vs. ProteinChat / ProtChatGPT**: These methods emphasize conversational protein understanding but often rely on specific structural encoders or sequence inputs; STELLA emphasizes unified sequence-structure representation and systematic benchmarks.
- **vs. SaProt / ESM series pLMs**: pLMs excel at representation learning and attribute prediction; STELLA connects pLM representations to LLMs, resulting in stronger natural language output capabilities.
- **vs. Foldseek retrieval baseline**: Retrieval depends on structural similarity and existing annotation mapping; STELLA attempts to learn a more general mapping from sequence-structure-function to linguistic descriptions, outperforming retrieval baselines on hold-out sets.
- **Insight**: The key to biological multimodal LLMs may not necessarily be more complex fusion layers, but rather the selection of a sufficiently unified domain encoder combined with stable two-stage instruction tuning to connect it to a language model.

## Rating
- Novelty: ⭐⭐⭐⭐☆ The engineering path of connecting the ESM3 unified sequence-structure encoder to an LLM is clear and effective, representing a conceptual multimodal paradigm shift.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Evaluations across FP, MCQA, EP, robustness, OOD, and encoder/LLM/training stage comparisons are rich, though real functional semantic evaluation remains limited by metrics.
- Writing Quality: ⭐⭐⭐⭐☆ The architecture and benchmark descriptions are complete, and result tables are exhaustive; parts of the narrative emphasize application vision.
- Value: ⭐⭐⭐⭐☆ Holds high reference value for protein functional annotation and scientific multimodal LLMs, particularly regarding the unified encoder + two-stage training route.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] LIMSSR: LLM-Driven Sequence-to-Score Reasoning under Training-Time Incomplete Multimodal Observations](../../ICML2026/multimodal_vlm/limssr_llm-driven_sequence-to-score_reasoning_under_training-time_incomplete_mul.md)
- [\[ICCV 2025\] MUSE-VL: Modeling Unified VLM through Semantic Discrete Encoding](../../ICCV2025/multimodal_vlm/musevl_modeling_unified_vlm_through_semantic_discrete_encodi.md)
- [\[NeurIPS 2025\] STRUCTURE: With Limited Data for Multimodal Alignment, Let the Structure Guide You](../../NeurIPS2025/multimodal_vlm/with_limited_data_for_multimodal_alignment_let_the_structure_guide_you.md)
- [\[ICLR 2026\] Reasoning-Driven Multimodal LLM for Domain Generalization](../../ICLR2026/multimodal_vlm/reasoning-driven_multimodal_llm_for_domain_generalization.md)
- [\[CVPR 2026\] UNICBench: UNIfied Counting Benchmark for MLLM](../../CVPR2026/multimodal_vlm/unicbench_unified_counting_benchmark_for_mllm.md)

</div>

<!-- RELATED:END -->

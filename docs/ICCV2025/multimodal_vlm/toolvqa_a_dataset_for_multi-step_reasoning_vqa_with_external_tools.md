---
title: >-
  [Paper Note] ToolVQA: A Dataset for Multi-step Reasoning VQA with External Tools
description: >-
  [ICCV 2025][Multimodal VLM][Visual Question Answering] This paper proposes ToolVQA — a large-scale multimodal tool-augmented VQA dataset containing 23K samples. It is automatically constructed via the ToolEngine pipeline, which combines image-guided DFS with LCS-based example matching, to generate multi-step reasoning data in realistic scenarios. LLaVA-7B fine-tuned on this dataset surpasses GPT-3.5-Turbo on 5 OOD benchmarks.
tags:
  - ICCV 2025
  - Multimodal VLM
  - Visual Question Answering
  - Tool Use
  - Multi-step Reasoning
  - Dataset
  - Large Models
  - Tool Agent
date: 2026-05-08
content_hash: 8a717193bd5301fd
---

# ToolVQA: A Dataset for Multi-step Reasoning VQA with External Tools

**Conference**: ICCV 2025
**arXiv**: [2508.03284](https://arxiv.org/abs/2508.03284)
**Code**: [GitHub](https://github.com/Fugtemypt123/ToolVQA-release)
**Area**: Multimodal VLM
**Keywords**: Visual Question Answering, Tool Use, Multi-step Reasoning, Dataset, Large Models, Tool Agent

## TL;DR

This paper proposes ToolVQA — a large-scale multimodal tool-augmented VQA dataset containing 23K samples. It is automatically constructed via the ToolEngine pipeline, which combines image-guided DFS with LCS-based example matching, to generate multi-step reasoning data in realistic scenarios. LLaVA-7B fine-tuned on this dataset surpasses GPT-3.5-Turbo on 5 OOD benchmarks.

## Background & Motivation

Integrating external tools into large foundation models (LFMs) is a key direction toward building general-purpose AI assistants. However, existing tool-augmented VQA datasets suffer from three major gaps:

**Unrealistic scenarios**: Synthetic images or oversimplified PDF files are used, mismatching the complexity of real-world scenes.

**Overly simple queries**: Queries require only single-step reasoning or provide explicit tool-use hints (e.g., "use the Cheap YouTube API tool"), lacking implicit multi-step reasoning.

**High annotation cost**: Manually annotated datasets (e.g., GAIA with only 500 samples) are difficult to scale.

**Paper Goals**: To construct a large-scale, scalable dataset that simultaneously satisfies real-world scenarios and real-world queries, bridging the gap between synthetic data and authentic tool use.

Advantages over existing datasets (Table 1): ToolVQA is the only large-scale dataset that simultaneously supports multimodal input, real-world scenarios/queries, evaluable answers, and high reasoning complexity (2.38).

## Method

### ToolEngine Data Construction Pipeline

The pipeline comprises three core components (Fig. 3):

**1. Real-World Example Construction**

Ten users from diverse disciplines (mathematics, computer science, economics, Chinese, and arts) each recorded 15 common tool-use scenarios. After merging tools with similar functionality, the 10 most frequently used tools were selected, and the 150 initial scenarios were refined into 34 representative examples.

**2. Image-Guided DFS Search**

A depth-first search is performed over the tool graph to construct multi-step tool-use trajectories:

$$t_i = \mathcal{M}(choices=\mathcal{T}, image=\mathcal{I}, examples=\text{Ret}(\mathcal{E}, \mathcal{P}_{i-1}))$$
$$a_i = \mathcal{M}(tool=t_i, image=\mathcal{I}, examples=\text{Ret}(\mathcal{E}, \mathcal{P}_i))$$

The controller $\mathcal{M}$ is ChatGPT-4o-latest, which selects a tool and generates arguments at each step, involving real tool calls to extract image information.

**3. LCS Example Matching**

Dynamic example matching is performed based on the Longest Common Subsequence (LCS) algorithm. At step $i$, the LCS similarity between the current trajectory $\mathcal{P}_i$ and each element in the example set $\mathcal{P}^e$ is computed, and the Top-k most similar examples are retrieved:

$$\text{Ret}(\mathcal{E}, \mathcal{P}_i) = \text{TopK}_{\mathcal{P}^e \in \mathcal{E}}\{\text{LCS}(\mathcal{P}^e, \mathcal{P}_i)\}$$

Key advantage: Unlike fixed example matching, LCS allows dynamic switching of examples during DFS, integrating knowledge from different example types and significantly improving reasoning complexity and data quality.

### Tool Set Design

10 tools spanning 4 categories:
- **Perception**: ImageCaption, OCR, ObjectDetection, RegionDescription
- **Operation**: DrawBox, GoogleSearch
- **Logic**: Calculator, Plot, ItemCount
- **Creation**: TextToImage

### Training Objective

LLaVA-7B is fine-tuned with cross-entropy loss:

$$\mathcal{L} = \mathbb{E}_{\mathcal{E} \sim \mathcal{D}}\left[\frac{1}{n}\sum_{i=1}^{n} -\log p(t_i, a_i, r_i \mid \mathcal{I}, \mathcal{T}, \mathcal{Q}, \mathcal{P}_{i-1})\right]$$

Training configuration: batch_size=2, lr=2e-4, LoRA fine-tuning, 4000 epochs, 4×GTX3090.

## Key Experimental Results

### Main Results: ToolVQA Test Set (Table 4)

| Model | Setting | End-to-End Acc.↑ | Inst.↑ | Tool.↑ | Arg.↑ | Summ.↑ |
|------|------|-----------------|--------|--------|-------|--------|
| ChatGPT-4o-latest | VLM | 38.29 | - | - | - | - |
| ChatGPT-4o-latest | VLM+tool | 34.96 | 36.5 | 14.68 | 8.92 | 56.1 |
| GPT-3.5-Turbo | LLM+tool | 18.37 | 73.24 | 30.46 | 20.08 | 58.18 |
| LLaVA-7B (original) | VLM+tool | 1.17 | 16.39 | 9.43 | 0 | 0.01 |
| **Tuned LLaVA-7B** | **VLM+tool** | **18.80** | **86.62** | **61.61** | **39.34** | 30.91 |

Key findings:
- The fine-tuned 7B model achieves end-to-end accuracy comparable to the much larger closed-source GPT-3.5-Turbo.
- Instruction formatting and tool selection improve substantially (Inst. 86.62%, Tool. 61.61%), while argument prediction and answer summarization remain bottlenecks.
- GPT-4o with VLM+tool underperforms pure VLM (34.96 < 38.29), suggesting that tool-introduced noise can outweigh the benefits.

### OOD Generalization (Table 5)

| Model | TextVQA | TallyQA | InfoSeek | GTA | TEMPLAMA |
|------|---------|---------|----------|-----|----------|
| GPT-3.5-Turbo | 36.3 | 61 | 11.3 | 23.62 | 33.67 |
| LLaVA-7B | 41.2 | 60.1 | 5.2 | 12.12 | 3.06 |
| **Tuned LLaVA-7B** | **47** | **64.3** | **13.8** | **33.29** | 21.43 |

The fine-tuned model outperforms GPT-3.5-Turbo on 4 out of 5 OOD benchmarks, demonstrating strong generalization.

### Ablation Study on ToolEngine (Table 3)

| Method | Acc.↑ | Cor.↑ | Nec.↑ | R.C.↑ |
|------|-------|-------|-------|-------|
| **ToolEngine (full)** | **90.8** | **85.2** | **87.51** | **2.38** |
| w/o Example + LCS | 27.3 | 77.6 | 21.04 | 1.1 |
| w/o LCS | 41.6 | 81.4 | 54.26 | 1.61 |

LCS matching is critical to data quality: removing it drops accuracy from 90.8% to 41.6% and reasoning complexity from 2.38 to 1.61.

### Few-shot ICL Experiment (Table 6)

| Model | 0-shot | 1-shot | 5-shot | 10-shot |
|------|--------|--------|--------|---------|
| GPT-4o | 34.96 | 37.20 | 38.41 | 38.63 |
| Tuned LLaVA-7B | 18.80 | 19.41 | 21.13 | 20.69 |

The fine-tuned model still benefits from ICL (18.80→21.13), indicating that fine-tuning and ICL are complementary.

### Dataset Statistics

- Total samples: 23,655
- Total tool calls: 65,785
- Average reasoning trajectory length: 2.78 steps
- Average question length: 15.74 tokens
- Average answer length: 2.69 tokens (concise and evaluable)

## Highlights & Insights

1. **Elegance of LCS dynamic matching**: By progressively matching different examples, the DFS process can integrate diverse types of knowledge to produce more complex reasoning chains. Fixed example matching cannot adapt to varying scenarios — this is the key driver of data quality improvement.
2. **Easy to generate, hard to answer**: GPT-4o, which generates the questions, achieves less than 40% accuracy when answering its own questions, confirming that the ToolEngine pipeline successfully decouples single-step reasoning from multi-step end-to-end reasoning.
3. **Double-edged effect of tool use**: Multiple models perform worse under VLM+tool than pure VLM, indicating that tool-introduced noise can exceed the benefit. Fine-tuning effectively suppresses such noise.
4. **Performance bottleneck lies in argument prediction and answer summarization**: These subtasks require understanding newly returned tool outputs and extracting meaningful responses, areas where fine-tuning yields limited improvement.

## Limitations & Future Work

1. The tool set is relatively small (10 tools); although each tool has strong generalization capability, full coverage of all real-world scenarios is not achievable.
2. Data generation relies on GPT-4o, which incurs high cost and may introduce bias.
3. The 90.8% automatic generation accuracy implies approximately 10% noise in the training data.
4. Fine-tuning experiments are conducted only at the 7B scale; performance of larger models remains unknown.

## Related Work & Insights

- Compared to MM-Traj (Gao et al., 2025), ToolVQA uses real-scene images with verified answers.
- The LCS matching approach is generalizable to other data synthesis scenarios requiring dynamic example selection.
- The identified bottlenecks in tool use (argument prediction, answer summarization) point to future research on dynamic information processing in multi-turn dialogue.
- The design philosophy of combining highly generalizable tools (e.g., GoogleSearch) with task-specific tools is worth borrowing.

## Rating ⭐⭐⭐⭐

Novelty ★★★★☆: The ToolEngine pipeline and LCS example matching design are original.
Experimental Thoroughness ★★★★☆: Covers diverse models and OOD benchmarks with comprehensive analysis.
Writing Quality ★★★★☆: Data quality evaluation is thorough; error analysis is insightful.
Value ★★★★★: Code and dataset are publicly released, directly usable for tool-agent training and evaluation.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] MMAT-1M: A Large Reasoning Dataset for Multimodal Agent Tuning](mmat1m_a_large_reasoning_dataset_for_multimodal_agent_tuning.md)
- [\[ICCV 2025\] LLaVA-CoT: Let Vision Language Models Reason Step-by-Step](llava-cot_let_vision_language_models_reason_step-by-step.md)
- [\[NeurIPS 2025\] Can Multi-Modal LLMs Provide Live Step-by-Step Task Guidance?](../../NeurIPS2025/multimodal_vlm/can_multi-modal_llms_provide_live_step-by-step_task_guidance.md)
- [\[ICLR 2026\] FRIEDA: Benchmarking Multi-Step Cartographic Reasoning in Vision-Language Models](../../ICLR2026/multimodal_vlm/frieda_benchmarking_multi-step_cartographic_reasoning_in_vision-language_models.md)
- [\[ICCV 2025\] From Easy to Hard: The MIR Benchmark for Progressive Interleaved Multi-Image Reasoning](from_easy_to_hard_the_mir_benchmark_for_progressive_interleaved_multi-image_reas.md)

<!-- RELATED:END -->

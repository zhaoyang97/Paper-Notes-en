---
title: >-
  [Paper Note] Table2LaTeX-RL: High-Fidelity LaTeX Code Generation from Table Images via Reinforced Multimodal Language Models
description: >-
  [NeurIPS 2025][Code Intelligence][Table Recognition] This paper proposes VSGRPO — a dual-reward reinforcement learning strategy based on GRPO — that jointly optimizes a structure-level reward (TEDS-Structure) and a visua…
tags:
  - "NeurIPS 2025"
  - "Code Intelligence"
  - "Table Recognition"
  - "LaTeX Generation"
  - "GRPO Reinforcement Learning"
  - "Dual Reward Mechanism"
  - "Multimodal Large Language Models"
date: 2026-05-08
content_hash: dbd947ed343b028e
---

# Table2LaTeX-RL: High-Fidelity LaTeX Code Generation from Table Images via Reinforced Multimodal Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2509.17589](https://arxiv.org/abs/2509.17589)  
**Code**: [GitHub](https://github.com/newLLing/Table2LaTeX-RL)  
**Area**: Code Intelligence
**Keywords**: Table Recognition, LaTeX Generation, GRPO Reinforcement Learning, Dual Reward Mechanism, Multimodal Large Language Models

## TL;DR

This paper proposes VSGRPO — a dual-reward reinforcement learning strategy based on GRPO — that jointly optimizes a structure-level reward (TEDS-Structure) and a visual fidelity reward (CW-SSIM on rendered images). The fine-tuned MLLM (only 3B parameters) surpasses GPT-4o and models with 72B+ parameters on the table-image-to-LaTeX generation task, with particularly significant gains on complex tables.

## Background & Motivation

Tables are core components of scientific documents, and automatically generating compilable, high-quality LaTeX code from table images is critical for document digitization. However, existing work primarily focuses on generating HTML representations, lacking the structural expressiveness and typographic precision required by LaTeX. This task faces three core challenges:

**Difficulty in handling complex tables**: Large-scale, deeply nested structures (multi-row/multi-column merges) and semantically rich cell content (e.g., mathematical formulas) challenge both visual encoders and language decoders. The visual encoder must extract fine-grained visual and structural cues, while the language decoder must generate long, syntax-sensitive LaTeX sequences. Errors in either component can lead to hallucinated output or compilation failures.

**Inherent limitations of SFT**: Supervised fine-tuning employs teacher forcing, with token-level next-token prediction as the training signal. However, LaTeX has syntactic ambiguity — different syntactic forms can produce identical visual output. This creates a mismatch between training objectives and evaluation objectives, which is especially harmful for complex tables.

**Imperfect evaluation metrics**: TEDS is insensitive to fine-grained errors and faces adaptation issues between HTML and LaTeX; pixel-level metrics focus on local visual similarity but ignore global structural correctness. A hybrid evaluation strategy is therefore needed.

These challenges motivate the design of VSGRPO — by incorporating rendered visual feedback into the RL optimization loop, the framework directly optimizes final visual output quality, circumventing the LaTeX syntactic ambiguity problem.

## Method

### Overall Architecture

A three-stage pipeline: (1) large-scale data collection — crawling 1.2 million table–LaTeX pairs from arXiv; (2) MLLM supervised fine-tuning (SFT) — obtaining an initial table-to-LaTeX generation capability; (3) VSGRPO reinforcement fine-tuning — further improving performance on complex tables via the dual reward mechanism.

### Key Designs

1. **Large-Scale Table2LaTeX Dataset Construction**: LaTeX source files are crawled from arXiv papers spanning October 2017 to April 2023. Regular expressions are used to extract `tabular` environments; after removing citation commands, color settings, and other control sequences, **1,209,986** table–LaTeX pairs are obtained. Tables are divided into three complexity levels:

    - Simple: basic structure (94%)
    - Medium: containing 2+ `\multirow` or `\multicolumn` commands and 100–160 cells (3%)
    - Complex: more than 160 cells (3%)

   This stratification enables more fine-grained evaluation that accurately reflects model capability across different complexity levels.

2. **VSGRPO Dual-Reward Reinforcement Learning Strategy**: The core innovation lies in incorporating LaTeX rendering (a non-differentiable operation) into the RL optimization loop. For each table image input, the model samples a set of LaTeX outputs $\{o_1, ..., o_N\}$, and two rewards are computed for each:

    - **Visual Reward**: The generated LaTeX is compiled and rendered into an image, which is then compared against the ground-truth rendered image using CW-SSIM. A reward of 1 is assigned if the score exceeds a threshold (0.6), and 0 otherwise. CW-SSIM is specifically adapted for black-and-white table images: convert to grayscale → unify size → align rows and columns → 2×2 Haar wavelet decomposition into 4 sub-bands → compute SSIM independently on each sub-band → take the average.

    - **Structure Reward**: Both the generated and ground-truth LaTeX are converted to HTML, and TEDS-Structure is computed. A reward of 1 is assigned if the score exceeds a threshold (0.9), and 0 otherwise. TEDS-Structure measures table structure alignment via minimum tree edit distance.

   The optimization objective is based on the GRPO framework:
   $$J_{\text{RFT}}(\theta) = \mathbb{E}\left[\frac{1}{N}\sum_{i=1}^N \min\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}A_i, \text{clip}(\cdot, 1-\varepsilon, 1+\varepsilon)A_i\right) - \beta D_{KL}(\pi_\theta \| \pi_{ref})\right]$$

   where the advantage function is $A_i = \frac{r_i - \text{mean}(\{r_j\})}{\text{std}(\{r_j\})}$, $\varepsilon=0.2$, and $\beta=0.02$.

3. **Careful Training Strategy Design**:

    - VSGRPO is trained exclusively on **5,936 complex tables** (ground-truth LaTeX < 3,000 characters), balancing complexity and computational feasibility
    - SFT is a necessary prerequisite — performing RL directly without SFT yields drastically worse results (verified by ablation)
    - Outputs that fail to compile automatically receive a reward of 0

### Loss & Training

- **SFT stage**: Standard negative log-likelihood loss $\mathcal{L}_{\text{SFT}} = -\sum \log p_\theta(\mathbf{y}^{(i)}|\mathbf{x}^{(i)})$, full-parameter fine-tuning for one epoch
- **RFT stage**: PPO-style objective with KL regularization. InternVL2-1B uses the VLM-R1 framework (num\_gens=8); Qwen2.5-VL-3B uses the ms-swift framework (num\_gens=4)
- **Hybrid evaluation strategy**: A combined TEDS-Structure + CW-SSIM evaluation is adopted, where the former measures global structural correctness and the latter measures local visual fidelity

## Key Experimental Results

### Main Results — CW-SSIM and Compilation Rate

| Model | Simple CW-SSIM | Medium CW-SSIM | Complex CW-SSIM | Complex Compile Rate |
|------|:---:|:---:|:---:|:---:|
| Mathpix (commercial) | 0.6884 | 0.5647 | 0.4862 | 0.9889 |
| GPT-4o | 0.6792 | 0.5612 | 0.4747 | 0.9917 |
| Qwen2.5-VL-72B | 0.7077 | 0.6009 | 0.5112 | 0.9335 |
| Nougat (specialist) | 0.7401 | 0.5505 | 0.4699 | 0.3352 |
| **Qwen2.5-VL-3B-VSGRPO** | **0.8186** | **0.7236** | **0.6145** | **0.9917** |
| Gain over GPT-4o | +0.1394 | +0.1624 | +0.1398 | on par |

### Main Results — TEDS and TEDS-Structure

| Model | Complex TEDS | Complex TEDS-Struct | Notes |
|------|:---:|:---:|------|
| Mathpix | 0.7176 | 0.8100 | Commercial tool |
| GPT-4o | 0.5865 | 0.7745 | Severe degradation on complex tables |
| Qwen2.5-VL-72B | 0.7448 | 0.8334 | Best open-source 72B |
| Nougat | 0.0424 | 0.0527 | Near-complete collapse on complex tables |
| **Qwen2.5-VL-3B-VSGRPO** | **0.8673** | **0.9218** | **First model to surpass 0.9** |

### Ablation Study

| Configuration | Complex CW-SSIM | TEDS | TEDS-Struct | Notes |
|------|:---:|:---:|:---:|------|
| SFT only | 0.5806 | 0.8481 | 0.9047 | Baseline |
| + TEDS-Struct reward only | 0.5925 | 0.8608 | 0.9155 | Structure reward is effective |
| + CW-SSIM reward only | 0.6064 | 0.8607 | 0.9133 | Visual reward is effective |
| + Dual reward (VSGRPO) | **0.6145** | **0.8673** | **0.9218** | Complementary, best overall |
| VSGRPO w/o SFT | 0.4695 | 0.6884 | 0.8167 | Necessity of SFT pretraining |

### Key Findings

- **A 3B model comprehensively outperforms 72B+ models and commercial tools**: VSGRPO enables the small model to surpass Mathpix, GPT-4o, and 72B open-source models across all complexity levels, demonstrating that targeted RL strategies are more effective than pure scaling
- **The advantage grows with table complexity**: VSGRPO achieves the most significant gains on Complex tables (CW-SSIM +0.1398 vs. GPT-4o), reflecting the effectiveness of the "train on hard examples" strategy
- **Dual rewards are complementary**: The structure reward and visual reward have different emphases, and their combination yields the best results
- **SFT is a necessary foundation for RL**: Skipping SFT and applying RL directly leads to a comprehensive performance collapse

## Highlights & Insights

1. **Visual-in-the-loop RL** is the core innovation: rendering (a non-differentiable operation) is incorporated into the training loop via RL reward signals, circumventing the differentiability constraint
2. **"Train on hard examples" strategy**: Using only 5,936 complex tables for RL outperforms training on mixed or simple data, indicating that the RL stage should focus on the model's weakest areas
3. **Small model + precise RL > large model + general capability**: A 3B specialized model comprehensively outperforms a 72B general-purpose model, providing an important counterexample to the assumption that large models are always necessary
4. **Complexity-stratified evaluation** is a valuable contribution to evaluation methodology in this field

## Limitations & Future Work

- The VSGRPO training process requires rendering each LaTeX output to PDF and converting it to PNG for CW-SSIM computation, which constitutes a significant training bottleneck
- Due to GPU resource constraints, RL training is conducted on only 5,936 complex tables; larger training sets may yield further improvements
- Only the `tabular` environment is currently supported; other LaTeX table formats (e.g., `longtable`, `tabularx`) are not handled
- The reward design is binary (0/1); continuous rewards could provide more fine-grained optimization signals

## Related Work & Insights

- **Success of GRPO in mathematical reasoning**: This paper extends GRPO from text generation to a multimodal reward setting combining "text + rendering"
- **Implications for other code generation tasks**: Any task following the "generate code → compile/execute → evaluate output" paradigm can adopt the visual-in-the-loop RL approach, including HTML generation, SVG drawing, and Markdown typesetting
- **Comparison with Nougat**: Nougat generates LaTeX end-to-end but completely collapses on complex tables (TEDS of only 0.04), demonstrating that purely end-to-end training is insufficient for structurally complex outputs

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The dual-reward RL with rendering feedback is highly innovative and makes a methodological contribution to RL for visual-code generation tasks
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Multiple baselines (commercial/general/specialist), complexity-stratified evaluation, human evaluation, and thorough ablation studies
- **Writing Quality**: ⭐⭐⭐⭐ Problem analysis is thorough and method motivation is clearly articulated
- **Value**: ⭐⭐⭐⭐⭐ High practical value (directly serves scientific document digitization) and significant methodological value (a new paradigm of visual-in-the-loop RL)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] From Charts to Code: A Hierarchical Benchmark for Multimodal Models](../../ACL2026/code_intelligence/from_charts_to_code_a_hierarchical_benchmark_for_multimodal_models.md)
- [\[NeurIPS 2025\] Embedding Alignment in Code Generation for Audio](embedding_alignment_in_code_generation_for_audio.md)
- [\[NeurIPS 2025\] CodeCrash: Exposing LLM Fragility to Misleading Natural Language in Code Reasoning](codecrash_exposing_llm_fragility_to_misleading_natural_language_in_code_reasonin.md)
- [\[CVPR 2026\] GeoTikzBridge: Advancing Multimodal Code Generation for Geometric Perception and Reasoning](../../CVPR2026/code_intelligence/geotikzbridge_advancing_multimodal_code_generation_for_geometric_perception_and_.md)
- [\[ACL 2026\] Ro-SLM: Onboard Small Language Models for Robot Task Planning and Operation Code Generation](../../ACL2026/code_intelligence/ro-slm_onboard_small_language_models_for_robot_task_planning_and_operation_code_.md)

</div>

<!-- RELATED:END -->

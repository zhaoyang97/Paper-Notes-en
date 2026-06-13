---
title: >-
  [Paper Note] PRISM: Self-Pruning Intrinsic Selection Method for Training-Free Multimodal Data Selection
description: >-
  [ACL2026][Multimodal VLM][Multimodal Data Selection] PRISM identifies that the non-zero mean of MLLM visual features causes "Global Semantic Drift…
tags:
  - "ACL2026"
  - "Multimodal VLM"
  - "Multimodal Data Selection"
  - "Visual Instruction Tuning"
  - "Representation Anisotropy"
  - "Training-Free Selection"
  - "Redundancy Pruning"
date: 2026-05-08
content_hash: 5e675625850ad683
---

# PRISM: Self-Pruning Intrinsic Selection Method for Training-Free Multimodal Data Selection

**Conference**: ACL2026 Best Paper  
**arXiv**: [2502.12119](https://arxiv.org/abs/2502.12119)  
**Code**: Yes, but no clear URL provided for the cache  
**Area**: Multimodal VLM / Data Selection  
**Keywords**: Multimodal Data Selection, Visual Instruction Tuning, Representation Anisotropy, Training-Free Selection, Redundancy Pruning  

## TL;DR
PRISM identifies that the non-zero mean of MLLM visual features causes "Global Semantic Drift," which contaminates similarity-based data selection. By employing training-free mean re-centering and low-correlation sample filtering, it achieves 101.7% relative performance while retaining only approximately 30% of visual samples, reducing end-to-end GPU time by about 70%.

## Background & Motivation
**Background**: Multimodal Large Language Models (MLLMs) typically undergo large-scale image-text pre-training followed by visual instruction tuning. As data pools like LLaVA and VisionFlan expand, the number of instruction samples has grown significantly, but they contain many redundant, low-information, or noisy samples.

**Limitations of Prior Work**: Existing visual instruction data selection methods mostly rely on proxy models, external scorers, training loss, perplexity, gradients, or influence functions. These methods either require additional model inference or iterative training and gradient calculation, making the selection process itself expensive, sometimes even offsetting the efficiency gains from using less training data.

**Key Challenge**: Data selection is intended to save computational resources, but many selectors transfer the cost to the selection phase. The authors argue the root cause is that most methods directly use the raw geometry of MLLM visual features. These visual embeddings are not uniformly distributed around the origin but are pulled into a narrow cone by a strong global mean direction, causing cosine similarity to mistake shared background drift for semantic similarity.

**Goal**: Propose a multimodal instruction selection method that is proxy-free, training-free, and gradient-free. It aims to maintain or exceed full fine-tuning performance while truly reducing the total cost of selection and tuning.

**Key Insight**: The paper begins with a diagnosis of representation geometry, proving that visual features exhibit representation anisotropy and singular value concentration. It kemudian reformulates the data selection problem as "removing global drift first, then estimating redundancy based on the unique semantic components of the samples."

**Core Idea**: Use mean re-centering of the target MLLM's own visual features to restore a more reliable similarity geometry, then retain samples that have low correlation with the overall population and carry more unique information.

## Method
The key to PRISM lies not in training a stronger selector, but in making the existing visual representations within the MLLM usable again. The selection pipeline consists of four steps: single-pass feature extraction, global mean estimation, redundancy scoring, and percentile filtering. Thus, the cost is primarily a single forward pass and linear aggregation.

### Overall Architecture
Given a visual instruction dataset $D=\{d_1,\dots,d_N\}$, where each sample contains an image and a text instruction, PRISM first extracts visual features $F_i$ for each sample using the visual encoder, projector, and intermediate LLM layers of the target MLLM. A global image representation is obtained via average pooling.

Then, PRISM calculates the feature mean of the entire corpus: $\mu_F=\frac{1}{N}\sum_i F_i$. This mean represents the global drift shared by all visual samples rather than the unique semantics of an individual sample. For any two samples, PRISM no longer uses raw cosine similarity but computes the normalized inner product after centering: $(F_i-\mu_F)$ and $(F_j-\mu_F)$.

Next, the Redundancy Score for each sample is defined as its average correlation with all other centered samples in the corpus. Intuitively, if a sample is highly correlated with many others, it is more likely to be redundant or provide low marginal utility; if the correlation is low, it likely carries more unique visual semantics.

Finally, a subset of samples with the lowest redundancy scores is selected based on a budget $\tau$. In main experiments, PRISM uses a 30% budget for visual samples from LLaVA-665K while retaining text-only samples, resulting in PRISM-Instruct-250K.

### Key Designs
1. **Overall Selection Cost Evaluation Criterion**:
	- **Function**: Quantifies whether a data selection method is "truly efficient" using a metric that considers both performance and time.
	- **Mechanism**: OSC is defined as the product of the performance ratio and the time ratio:
      $$C=(P(D_{full})/P(D_{sub}))\times((T_{select}+T_{tune}(D_{sub}))/T_{tune}(D_{full}))$$
      A method provides a net efficiency gain only when $C<1$.
	- **Design Motivation**: Many selection methods only report subset performance while ignoring selection overhead. OSC forces comparison back to end-to-end costs, avoiding scenarios where "accurate selection is more expensive than full training."

2. **Visual Representation Anisotropy Diagnosis and Re-centering**:
	- **Function**: Corrects the global mean drift in raw visual embeddings to make similarity more reflective of unique sample semantics.
	- **Mechanism**: The authors decompose visual features into $x_i=\mu+\delta_i$, where $\mu$ is the shared global component and $\delta_i$ is the sample-specific semantic. When $\|\mu\|$ is much larger than $\|\delta_i\|$, raw cosine similarity approaches 1 and is dominated by $\mu$. PRISM replaces raw cosine with $\rho(F_i,F_j)=\frac{(F_i-\mu_F)^\top(F_j-\mu_F)}{\|F_i-\mu_F\|_2\|F_j-\mu_F\|_2}$.
	- **Design Motivation**: Rather than introducing an external scorer, it is better to first correct the most significant first-order geometric bias in the target model's own feature space. The authors specify that PRISM is not full whitening but a targeted correction for the corpus-level mean shift that disrupts redundancy estimation.

3. **Low-Correlation Redundancy Score Filtering**:
	- **Function**: Identifies semantically unique samples without constructing an $N\times N$ similarity matrix.
	- **Mechanism**: The redundancy score $R(d_i)=\frac{1}{N-1}\sum_{j\ne i}\rho(F_i,F_j)$ represents the average connection strength of a sample in the centered semantic graph. PRISM uses an exact aggregate implementation to compute this in $O(Nd)$ without materializing the full pairwise matrix, then keeps samples where $R(d_i)$ is below a percentile threshold.
	- **Design Motivation**: Low-correlation samples provide higher marginal information; the $O(Nd)$ aggregation avoids the prohibitive costs of k-center or exact greedy coverage at the scale of hundreds of thousands of samples.

### Loss & Training
PRISM itself has no training loss as it is a training-free selector. After selection, the authors perform one round of visual instruction tuning following official LLaVA-1.5 hyperparameters. The main setup uses LLaVA-665K and LLaVA-1.5-7B, comparing against Random, Length, EL2N, Perplexity, GraNd, TIVE, InstructionGPT-4, Self-Filter, COINCIDE, ICONS, and DataTailor. Additional experiments on VisionFlan-186K, different MLLM architectures, and text-only benchmarks verify generalization and knowledge retention.

## Key Experimental Results

### Main Results
On LLaVA-1.5-7B, PRISM achieves 101.7% integrated performance relative to full fine-tuning, outperforming both the full dataset and strong selection methods across multiple multimodal benchmarks.

| Method | SQA | SQA-I | VizWiz | POPE-P/R/A | MM-Vet | MMBench | MME-C | MMMU | Rel. |
|------|-----|-------|--------|------------|--------|---------|-------|------|------|
| Full-Finetune | 69.4 | 66.8 | 50.0 | 86.1 / 87.3 / 84.2 | 31.1 | 64.3 | 311.9 | 35.4 | 100% |
| TIVE | 72.2 | 70.6 | - | 85.6 / 85.6 / 85.6 | - | 63.2 | 322.1 | - | 100.6% |
| ICONS | - | 70.8 | - | 87.5 / 87.5 / 87.5 | - | 63.1 | - | - | 101.0% |
| PRISM | 71.3 | 69.1 | 50.1 | 87.7 / 88.7 / 85.5 | 32.0 | 65.2 | 330.0 | 34.7 | 101.7% |

On VisionFlan-186K, PRISM selects 57K samples (30% budget) and still exceeds the full-data aggregate, significantly outperforming random selection.

| Method | Samples | VizWiz | SQA-I | TextVQA | POPE | MME | MMBench | Rel. |
|------|--------|--------|-------|---------|------|-----|---------|------|
| Full Data | 186K | 41.7 | 60.8 | 50.4 | 83.4 | 1263.2 | 52.6 | 100.0 |
| Random | 57K | 38.8 | 56.5 | 46.9 | 83.1 | 1175.0 | 48.9 | 94.1 |
| PRISM | 57K | 42.3 | 61.1 | 50.8 | 84.1 | 1275.5 | 53.1 | 100.9 |

### Ablation Study
Core ablations verify three points: shallow visual features are best, low-correlation samples are best, and average pooling outperforms the last token.

| Configuration | SQA | SQA-I | VizWiz | POPE-P/R/A | MM-Vet | MMBench | MME-C | Rel. |
|------|-----|-------|--------|------------|--------|---------|-------|------|
| Deep Layer | 71.2 | 69.1 | 51.6 | 86.6 / 88.0 / 84.2 | 31.1 | 62.9 | 254.0 | 97.2% |
| Middle Layer | 70.9 | 69.1 | 47.7 | 86.5 / 87.8 / 84.2 | 31.9 | 65.0 | 276.0 | 97.9% |
| Shallow Layer | 71.3 | 69.1 | 50.1 | 87.7 / 88.7 / 85.5 | 32.0 | 65.2 | 330.0 | 100.0% |
| High Correlation | 70.6 | 68.0 | 48.1 | 85.8 / 87.6 / 83.9 | 30.7 | 64.0 | 275.3 | 96.3% |
| Low Correlation | 71.3 | 69.1 | 50.1 | 87.7 / 88.7 / 85.5 | 32.0 | 65.2 | 330.0 | 100.0% |
| Last Token | 69.9 | 67.3 | 49.4 | 87.4 / 88.3 / 85.0 | 31.6 | 62.6 | 272.0 | 97.4% |
| Avg Pooling | 71.3 | 69.1 | 50.1 | 87.7 / 88.7 / 85.5 | 32.0 | 65.2 | 330.0 | 100.0% |

Cross-model generalization shows that PRISM is not only applicable to LLaVA-1.5-7B but consistently slightly outperforms full fine-tuning across various LLM and vision encoder combinations.

| Model Combo | Full Rel. | PRISM Rel. | Representative Change |
|----------|-----------|-------------|------------|
| Phi2-3B | 100% | 100.1% | MME 1765.7 → 1790.5 |
| Vicuna-7B | 100% | 101.7% | MMBench 64.3 → 65.2 |
| Vicuna-13B | 100% | 100.4% | MME 1826.7 → 1846.0 |
| Qwen2.5-7B Base | 100% | 101.0% | SQA-I 76.7 → 78.9 |
| Qwen2.5-7B Instruct | 100% | 100.9% | MMBench 71.0 → 72.4 |
| Llama-3-8B | 100% | 100.8% | SQA-I 75.2 → 77.3 |

### Key Findings
- The main result of PRISM is not just parity with less data, but a slight improvement over full fine-tuning: 101.7% for LLaVA-665K and 100.9% for VisionFlan-186K.
- Low-correlation samples outperform high and medium-correlation samples, directly supporting the core hypothesis that low redundancy after re-centering yields higher training value.
- Shallow features outperform middle and deep layers, suggesting that the geometric structure for redundancy detection is cleaner in early visual-token representations, while deeper layers may mix in more task-specific or abstract artifacts.
- The final composition of PRISM-Instruct-250K (250,557 samples) includes LLaVA (53,591), VG (28,777), VQAv2 (27,567), OCRVQA (26,638), and Text-Only (40,688), indicating that the composition is naturally determined by a global redundancy threshold rather than hard-coded source ratios.
- Text-only retention also shows benefits: PRISM-7B achieved 101.9% and PRISM-13B achieved 130.6% on text benchmarks, suggesting that cleaner visual instruction data might reduce catastrophic forgetting.

## Highlights & Insights
- The brilliance of PRISM lies in converting the data selection "model scoring problem" into a "geometric calibration problem." If the similarity of raw embeddings is inherently flawed, even complex cheap distance selectors will be misled.
- The OSC metric is highly practical. It requires a selector to satisfy both performance fidelity and net efficiency gain, posing a more honest constraint for data selection research.
- Performing only corpus mean re-centering instead of full whitening is a pragmatic trade-off. Full whitening requires high-dimensional covariance estimation, regularization, and rank selection; PRISM sacrifices complete isotropization for stability, zero hyperparameters, and large-scale deployability.
- Visual-first selection is insightful. The appendix notes that text features are relatively closer to the center; joint multimodal feature selection only achieved 97.8% relative performance, lower than the 101.7% of visual-only PRISM. This suggests multimodal selection does not necessarily requiring direct concatenation of all modalities.

## Limitations & Future Work
- PRISM only targets semantic redundancy pruning based on feature correlation; it does not detect factual errors, ethical biases, harmful content, or annotation quality issues. Therefore, it is suitable as an efficiency selector but should not be treated as a complete data governance tool.
- The method primarily corrects first-order global mean drift rather than full whitening. If the main problem in certain data pools stems from more complex second-order covariance structures, simple re-centering may be insufficient.
- The tasks focus on vision-language instruction tuning. The authors mention future expansion to other modalities, but whether first-order drift is equally exploitable in speech, video, or robotic trajectories remains to be verified.
- PRISM relies on the intermediate visual representations of the target MLLM; deployment is restricted when the target model is inaccessible or visual token representations are unstable.
- The current objective is to retain low-redundancy samples, but it does not explicitly model task coverage, fairness, rare categories, or safety-critical samples. Future work could combine geometric redundancy scores with quality/safety filters.

## Related Work & Insights
- **vs Random / Length / Perplexity**: These methods are cheap but have weak semantic signals. PRISM is equally cheap but leverages the centered visual geometry of the target MLLM to estimate redundancy.
- **vs Proxy-Based Selection**: Methods like InstructionGPT-4, Self-Filter, and TIVE depend on external models or scorers, potentially introducing proxy bias and inference overhead. PRISM requires no external evaluator.
- **vs Training-Based Selection**: EL2N, GraNd, and ICONS use training dynamics or gradient signals, which provide strong information but at a high cost. PRISM replaces iterative training signals with a single-pass feature extraction.
- **vs Whitening / Top-PC Removal**: These geometric corrections are more thorough but require additional hyperparameters or high-dimensional covariance estimation. PRISM chooses the simplest first-order re-centering to balance performance and scalability.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Explaining data selection inefficiency via visual representation anisotropy is a clear and theoretically supported perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Solid analysis across main results, ablations, VisionFlan, cross-model generalization, language preservation, and efficiency.
- Writing Quality: ⭐⭐⭐⭐☆ The logic is complete and the appendix is comprehensive, though the density of formulas and figures in the main text is high.
- Value: ⭐⭐⭐⭐⭐ Highly practical for multimodal instruction data filtering, especially in scenarios requiring true reductions in end-to-end training costs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] iReasoner: Trajectory-Aware Intrinsic Reasoning Supervision for Self-Evolving Large Multimodal Models](ireasoner_trajectory-aware_intrinsic_reasoning_supervision_for_self-evolving_lar.md)
- [\[ICML 2026\] Toward Structural Multimodal Representations: Specialization, Selection, and Sparsification via Mixture-of-Experts](../../ICML2026/multimodal_vlm/toward_structural_multimodal_representations_specialization_selection_and_sparsi.md)
- [\[ICCV 2025\] Mastering Collaborative Multi-modal Data Selection: A Focus on Informativeness, Uniqueness, and Representativeness](../../ICCV2025/multimodal_vlm/mastering_collaborative_multi-modal_data_selection_a_focus_on_informativeness_un.md)
- [\[NeurIPS 2025\] CoIDO: Efficient Data Selection for Visual Instruction Tuning via Coupled Importance-Diversity Optimization](../../NeurIPS2025/multimodal_vlm/coido_efficient_data_selection_for_visual_instruction_tuning_via_coupled_importa.md)
- [\[ICML 2026\] CLIP Tricks You: Training-free Token Pruning for Efficient Pixel Grounding in Large Vision-Language Models](../../ICML2026/multimodal_vlm/clip_tricks_you_training-free_token_pruning_for_efficient_pixel_grounding_in_lar.md)

</div>

<!-- RELATED:END -->

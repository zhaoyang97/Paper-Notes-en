---
title: >-
  [Paper Note] VOYAGER: A Training Free Approach for Generating Diverse Datasets using LLMs
description: >-
  [ACL2026][LLM/NLP][Data diversity] Voyager is a training-free LLM data generation algorithm that maintains diverse anchors and explorers using Determinantal Point Processes (DPP) and iteratively rewrites prompts using te…
tags:
  - "ACL2026"
  - "LLM/NLP"
  - "Data diversity"
  - "synthetic data generation"
  - "DPP"
  - "textual gradients"
  - "LLM sampling"
date: 2026-05-08
content_hash: 241cce146282f607
---

# VOYAGER: A Training Free Approach for Generating Diverse Datasets using LLMs

**Conference**: ACL2026  
**arXiv**: [2512.12072](https://arxiv.org/abs/2512.12072)  
**Code**: None  
**Area**: llm_nlp  
**Keywords**: Data diversity, synthetic data generation, DPP, textual gradients, LLM sampling

## TL;DR
Voyager is a training-free LLM data generation algorithm that maintains diverse anchors and explorers using Determinantal Point Processes (DPP) and iteratively rewrites prompts using textual gradients. It significantly improves Vendi diversity in creative writing and reasoning data generation without compromising quality.

## Background & Motivation
**Background**: LLMs are frequently employed to generate synthetic training data, evaluation samples, user personas, and creative content. However, repeatedly calling the same prompt often leads to mode collapse: outputs exhibit surface-level variations but are highly concentrated in semantics and structure. Standard decoding methods like temperature sampling, top-p, and beam search increase local randomness at the token level but do not directly optimize the global diversity of the resulting dataset.

**Limitations of Prior Work**: Existing diversity control methods have various shortcomings. High-temperature sampling introduces noise without guaranteeing coverage of different semantic regions; instructions like "please be diverse" rely on model self-awareness and lack global constraints; appending historical samples to the prompt only prevents short-term repetition and fails once the window is exceeded; methods based on Reinforcement Learning (RL) or post-training are computationally expensive and typically require access to open-source model weights.

**Key Challenge**: The authentic requirement is to generate a globally diverse dataset rather than ensuring a single output appears random. Performing subset selection after generation requires pre-generating a massive candidate pool followed by DPP or other selection algorithms, which is costly in terms of both LLM calls and matrix computations. Voyager aims to actively explore diverse regions during the generation process itself.

**Goal**: The authors propose a training-free algorithm applicable even to closed-source LLMs. In each round, a batch of samples is generated; samples are accepted based on their marginal volume gain relative to an anchor set. Rejected samples are utilized as feedback, allowing the LLM to generate "textual gradients" that rewrite subsequent explorer prompts.

**Key Insight**: The paper defines diversity as the determinant (volume) of the similarity matrix and utilizes DPP for two selection processes: selecting representative anchor points and retaining mutually distinct explorer prompts.

**Core Idea**: Transforming "diverse data generation" from prompt engineering into an iterative volume maximization process. It uses DPP to provide global diversity pressure and textual gradients to provide transferable prompt search directions.

## Method

### Overall Architecture
Voyager takes a task prompt $p$, target dataset size $l$, marginal gain threshold $\tau$, explorer limit $b$, anchor set size $k$, maximum iterations $T$, and a similarity kernel $K_{Sim}$ as inputs. The algorithm maintains three sets: the accepted dataset $D$, the anchor set $\Phi$, and the explorer prompt set $E$. Initially, $D$ and $\Phi$ are empty, and $E$ contains only the original task prompt.

Inside each outer iteration, every explorer executes the Explore process. Explore first generates a batch of candidate samples $B$; it then calculates the marginal volume gain for each candidate sample when added to the anchor set. If the gain is not lower than $\tau$, the sample is added to the dataset and the candidate anchor pool; otherwise, it is moved to a rejected set. If rejected samples exist, Voyager invokes a LLM-judge to analyze the failures and current anchors to generate textual gradients, which are then applied to the current explorer to derive successor explorer prompts for the next round.

After a batch of explorers finishes, the algorithm samples $k$ points from the candidate anchor pool via DPP to keep the anchor set compact and diverse. It also samples $b$ prompts from the candidate explorer pool via DPP to ensure future exploration directions do not overlap. The process stops when the dataset reaches size $l$ or $T$ iterations are completed.

### Key Designs
1. **Representing dataset diversity via determinant / volume**:
	- **Function**: Converts "diversity" into an optimizable mathematical quantity.
	- **Mechanism**: A similarity matrix is constructed for a sample set; the probability of a subset $S$ in a DPP satisfies $P(S) \propto det(K_S)$. Matrix volume is small for highly similar samples and larger as samples span a broader space.
	- **Design Motivation**: Metrics like the Vendi Score are derived from the similarity matrix. The paper presents the approximation $V \approx n^2 D^{1/n}/C$, where $D$ is the determinant and $C$ is the trace, illustrating that maximizing the determinant correlates with increasing effective rank and diversity.

2. **Approximating global goals with a fixed-size anchor set**:
	- **Function**: Avoids computing high-order determinants on the full dataset.
	- **Mechanism**: Candidate samples only need to calculate marginal volume gain against the anchor set. The anchor set is maintained via k-DPP to remain representative. It functions as a small, diverse reservoir approximating the coverage of the generated dataset.
	- **Design Motivation**: Directly maximizing the determinant of the full dataset's similarity matrix is computationally infeasible. Using a fixed anchor set reduces the complexity from the full dataset size to $k$, allowing marginal gain calculations at $O(k^2)$ using cached inverse matrices.

3. **Updating explorer prompts via textual gradients**:
	- **Function**: Enables the algorithm to not only filter redundant samples but also learn new directions for exploration.
	- **Mechanism**: Rejected samples indicate that the current prompt produces outputs too close to existing anchors. Voyager uses an LLM-judge to analyze the prompt, rejected samples, and anchors to provide natural language suggestions (textual gradients) on how to generate "more distinct" samples. These suggestions are used by another LLM to refine the prompt.
	- **Design Motivation**: While DPP provides selection pressure, it does not generate new semantic directions. Textual gradients transform rejection feedback into prompt-level search signals, making the approach suitable for closed-source models without requiring weight updates.

### Loss & Training
Voyager involves no model training. Instead, it optimizes a proxy objective for set diversity during generation: maximizing the volume of the similarity matrix of the anchor set. The similarity kernel is a convex combination of an RBF embedding kernel (0.7 weight) and Jaccard lexical similarity (0.3 weight). Embeddings are generated via `text-embedding-3-small`, and the generator is `GPT-4o mini`. Default hyperparameters are $b=3, k=10, T=200$, batch size 10, and target size 500.

In terms of complexity, candidate filtering in Explore is $O(|B|k^2)$, anchor DPP pruning is $O(k_{max}^3)$, and explorer DPP pruning is $O(b_{max}^3)$. Compared to generating a massive pool and then performing subset selection on the entire set, Voyager's computation and LLM call volume are more controllable.

## Key Experimental Results

### Main Results
The paper evaluates Voyager against Default, Temp=2.0, Diverse, History, Hierarchical, and SubsetSelect across 4 creative writing and 2 reasoning tasks. Metrics include lexical Jaccard distance, cosine distance, Vendi Score, LLM-as-judge quality, and LLM call counts.

| Task | Default Vendi | Hierarchical Vendi | Voyager Vendi | Voyager LLM calls | Main Conclusion |
|------|---------------|--------------------|----------------|-------------------|-----------------|
| Sports sentence | 2.991 | 15.070 | 24.132 | 443 | Voyager significantly outperforms the strongest baseline with fewer calls than Hierarchical (550). |
| Political conversation | 4.589 | 8.450 | 15.035 | 426 | Highest semantic and lexical diversity; also achieves the highest quality. |
| Poem | 3.004 | 5.679 | 7.312 | 615 | Reaches highest diversity, with a quality of 24.505 exceeding all baselines. |
| Movie plot | 4.002 | 7.661 | 8.302 | 695 | Highest diversity, though requiring more calls than Hierarchical. |
| Grade-school math | 3.039 | 8.715 | 18.777 | 399 | Diversity gains in reasoning tasks are particularly significant. |
| Logic puzzle | 3.312 | 7.024 | 13.256 | 393 | Far exceeds all baselines in logical task diversity. |

Overall results: In creative tasks, Voyager achieves an average 2.96x Vendi improvement over Default and 0.43x over Hierarchical. In reasoning tasks, it achieves 4.12x over Default and 1.02x over Hierarchical. Quality scores remain stable, indicating that the algorithm does not trade usability for random noise.

### Ablation Study
Ablations investigate the impact of DPP explorer selection and textual gradients. Additional experiments include generation length control and downstream synthetic training.

| Experiment | Comparison Setup | Key Data | Conclusion |
|------|----------|----------|------|
| Explorer DPP Ablation | Voyager-RE (Random Explorer) vs Voyager | Sports Vendi 11.852 vs 14.282; LLM calls 361 vs 252 | Diverse explorers make search more efficient, increasing diversity while reducing calls. |
| Textual Gradients Ablation | Disabling prompt refinement | Significant increase in rejection rate and iterations | Filtering samples without updating prompts leads to stagnation in similar regions. |
| Length Control | Poem limited to ~150 tokens | Voyager Vendi 8.167 vs Hierarchical 6.646 vs Default 3.304 | Diversity gains are not merely due to longer outputs. |
| Human Eval (Sports) | Default vs Voyager | 2.16±0.55 vs 3.82±0.28; Fleiss kappa 0.41 | Humans perceive Voyager samples as significantly more distinct. |
| Human Eval (Math) | Default vs Voyager | 1.56±0.36 vs 3.72±0.33; Fleiss kappa 0.34 | Automated Vendi scores align generally with human perception. |
| GSM8K Synthetic Training | Default 1000 vs Voyager 1000 | Gemma 7B-IT: 35.7 vs 45.7 | Diverse synthetic data significantly improves downstream training. |
| Data Efficiency | Voyager 500 vs Default 1000 | Gemma 7B-IT: 42.8 vs 35.7 | Fewer but more diverse data can outperform a larger set of redundant data. |

### Key Findings
- While increasing temperature or adding "diverse" instructions improves upon the Default, the gains are limited and unstable. Hierarchical is a strong baseline but requires manual design for theme decomposition and involves high LLM costs.
- Voyager's advantage stems from two synchronized mechanisms: DPP prevents the dataset from collapsing into existing modes, and textual gradients transform failures into new exploration prompts.
- Gains on reasoning data are larger than on creative writing, suggesting the algorithm covers more problem structures rather than just varied styles.
- Downstream training results are critical: diversity metrics are not just "vanity" numbers; the improvement from 35.7 to 45.7 on GSM8K for Gemma 7B shows that the generated data is more valuable for training.
- The trade-off is the number of LLM calls. Voyager requires more calls than Default or Temp methods, but for most tasks, it offers a better cost-benefit ratio than Hierarchical.

## Highlights & Insights
- The primary highlight is elevating DPP from a "post-processing filter" to an "in-process generator controller." This proactive approach is more suitable for large-scale data generation.
- The use of textual gradients is highly practical. Rejected samples are not wasted but act as signals to tell the model *why* the current prompt is not yielding novelty, creating a closed-loop for exploration.
- The anchor set is an effective engineering compromise. It sacrifices global determinant precision for fixed memory and computational scalability.
- The algorithm does not treat quality and diversity as a single black-box reward; it uses an LLM-judge for explicit quality evaluation. Results indicate that Voyager primarily boosts diversity without significant detriment to quality.
- For synthetic data training, the insight is clear: It is more beneficial to use global diversity constraints to generate fewer, high-coverage samples than to batch-generate massive amounts of similar ones.

## Limitations & Future Work
- Voyager relies on strong instruction-following LLMs, particularly for extracting textual gradients. Weaver models may provide vague or ineffective feedback.
- The similarity kernel depends on embedding and lexical models. If the embedding does not capture critical task-specific nuances, the optimized "diversity" may not align with human requirements.
- While Vendi and human pairwise diversity scores improve, more granular analysis of *diversity types* (e.g., whether math diversity comes from numbers, types, or reasoning logic) is still needed.
- Currently, the method is restricted to text and does not address multimodal (image, audio, video) samples, where defining unified similarity kernels is more complex.
- LLM call volume remains higher than standard batch generation. For cost-sensitive applications, one must tune $\tau$, anchor size, and explorer beam width according to task value.

## Related Work & Insights
- **vs temperature / nucleus sampling**: Randomness at the decoding layer only expands single-instance variance; Voyager optimizes coverage at the dataset level.
- **vs prompt-based diversity control**: Prompt instructions like "generate diverse samples" lack quantifiable objectives and global constraints.
- **vs Hierarchical prompting**: Hierarchical methods rely on domain knowledge to decompose tasks; Voyager automatically discovers exploration directions via textual gradients.
- **vs SubsetSelect / DPP post-processing**: Post-processing requires a massive pre-generated pool; Voyager integrates DPP into the loop, using the anchor set to guide the LLM toward novelty.
- **Insights for Data Synthesis**: Voyager's diversity control can be integrated into instruction tuning, math generation, persona simulation, and evaluation set construction, particularly where reducing repetitive patterns is desired.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Combines DPP and textual gradients into a training-free generation control algorithm naturally and effectively.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers creative/reasoning tasks, human evaluations, ablations, and downstream training. Adding more open-source models and cost curves would enhance the work.
- Writing Quality: ⭐⭐⭐⭐☆ Clear algorithmic and theoretical motivation; experimental descriptions are comprehensive.
- Value: ⭐⭐⭐⭐⭐ Highly practical for synthetic data generation, especially for maximizing data coverage and training efficiency with closed-source LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SketchDeco: Training-Free Latent Composition for Precise Sketch Colourisation](../../CVPR2026/llm_nlp/sketchdeco_training-free_latent_composition_for_precise_sketch_colourisation.md)
- [\[NeurIPS 2025\] SubSpec: Speculate Deep and Accurate — Lossless and Training-Free Acceleration for Offloaded LLMs](../../NeurIPS2025/llm_nlp/speculate_deep_and_accurate_lossless_and_training-free_acceleration_for_offloade.md)
- [\[AAAI 2026\] Guess or Recall? Training CNNs to Classify and Localize Memorization in LLMs](../../AAAI2026/llm_nlp/guess_or_recall_training_cnns_to_classify_and_localize_memorization_in_llms.md)
- [\[ICML 2026\] SphericalDreamer: Generating Navigable Immersive 3D Worlds with Panorama Fusion](../../ICML2026/llm_nlp/sphericaldreamer_generating_navigable_immersive_3d_worlds_with_panorama_fusion.md)
- [\[ICLR 2026\] Evaluating Text Creativity across Diverse Domains: A Dataset and Large Language Model Evaluator](../../ICLR2026/llm_nlp/evaluating_text_creativity_across_diverse_domains_a_dataset_and_large_language_m.md)

</div>

<!-- RELATED:END -->

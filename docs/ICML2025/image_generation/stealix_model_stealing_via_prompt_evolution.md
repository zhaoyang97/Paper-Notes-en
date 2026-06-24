---
title: >-
  [Paper Note] Stealix: Model Stealing via Prompt Evolution
description: >-
  [ICML2025][Image Generation][Model Stealing] Stealix proposes the first model stealing approach that does not require human-designed prompts. It iteratively evolves prompts using a genetic algorithm, synthesizes target-class images with Stable Diffusion to query the victim model, and requires only 1 real image per class. Under tight query budgets, it outperforms existing methods that rely on class names or handcrafted prompts, improving accuracy by up to 22.2%.
tags:
  - "ICML2025"
  - "Image Generation"
  - "Model Stealing"
  - "Prompt Optimization"
  - "Genetic Algorithm"
  - "Diffusion Models"
  - "Contrastive Learning"
date: 2026-05-08
content_hash: 00a4a78663e69f77
---

# Stealix: Model Stealing via Prompt Evolution

**Conference**: ICML2025  
**arXiv**: [2506.05867](https://arxiv.org/abs/2506.05867)  
**Code**: [https://zhixiongzh.github.io/stealix/](https://zhixiongzh.github.io/stealix/) (Project Page)  
**Area**: Diffusion Models / AI Security  
**Keywords**: Model Stealing, Prompt Optimization, Genetic Algorithm, Diffusion Models, Contrastive Learning

## TL;DR

Stealix proposes the first model stealing approach that does not require human-designed prompts. It iteratively evolves prompts using a genetic algorithm, synthesizes target-class images with Stable Diffusion to query the victim model, and requires only 1 real image per class. Under tight query budgets, it outperforms existing methods that rely on class names or handcrafted prompts, improving accuracy by up to 22.2%.

## Background & Motivation

**Background**: Model stealing attacks enable adversaries to replicate the functionality of a black-box model by querying its API, thereby training a behaviorally similar surrogate model. Current methods fall into three categories: (1) querying via public datasets (e.g., Knockoff Nets), (2) training GANs from scratch to generate synthetic images, and (3) using pre-trained diffusion models (e.g., Stable Diffusion) to synthesize images via prompts. The third category is the most efficient as it avoids training a new generator and does not rely on online data.

**Limitations of Prior Work**: Although diffusion-based methods (e.g., ASPKD) are efficient, they heavily rely on human-designed prompts or known class names. When class names lack context or fail to accurately capture the features of the target data, these methods suffer from severe degradation in performance. Moreover, manual intervention hinders attack automation and scalability.

**Key Challenge**: In specialized domains (e.g., satellite image classification), adversaries attacking high-value models often lack the expertise to design precise prompts. Prior research assumes that adversaries possess class names or prompt-engineering capabilities, which oversimplifies the problem and underestimates the true threat of pre-trained generative models in model stealing. Figure 1 shows that when the query dataset mismatches the target distribution (e.g., using CIFAR-10 to steal a satellite image classifier), performance drops drastically.

**Goal**: Under a more realistic threat model where the adversary "does not know the class names" and "has no prompt design capability", how can highly efficient model stealing be automated? Specifically, the following sub-problems are addressed:
   - How to generate images matching the target distribution without prior knowledge?
   - How to automatically discover and optimize prompts that describe the target classes?
   - How to maximize surrogate model accuracy within a limited query budget?

**Key Insight**: The authors observe that prompt optimization is fundamentally a search problem—seeking high-quality prompts in the semantic space that generate images classified as the target class by the victim model. Genetic algorithms are naturally suited for such multi-objective, discrete search spaces. Combining this with contrastive learning from vision-language models allows the feedback of the victim model to be integrated into a closed-loop prompt optimization.

**Core Idea**: To evolve a population of prompts using a genetic algorithm, employing the classification consistency of synthesized images from the victim model (Prompt Consistency) as the fitness function to iteratively optimize prompt accuracy and diversity.

## Method

### Overall Architecture

The input to Stealix consists of only 1 real seed image per class and query API access to a black-box victim model. The output is a surrogate model with behavior similar to the victim model. The overall pipeline is divided into two main stages:

**Stage 1: Prompt Evolution (Core Innovation)**—Automatically generates multiple high-quality prompts describing each class through an iterative "optimization $\to$ evaluation $\to$ reproduction" three-step loop.

**Stage 2: Surrogate Model Training**—Synthesizes a large volume of images using Stable Diffusion driven by the optimized prompts, queries the victim model to acquire pseudo-labels, and trains the surrogate model.

Each iteration $t$ maintains a population $\mathcal{S}^t = \{(\mathbf{x}_c^s, \mathbf{x}_c^+, \mathbf{x}_c^-)_i^t\}_{i=1}^N$ containing $N$ image triplets, where $\mathbf{x}^s$ is the seed image, $\mathbf{x}^+$ is a positive sample (classified as the target class by the victim model), and $\mathbf{x}^-$ is a negative sample (classified as other classes).

### Key Designs

1. **Prompt Refinement**:

    - **Function**: Optimizes a randomly initialized discrete prompt $\mathbf{p}$ for each image triplet to capture target class features in the semantic space.
    - **Mechanism**: Leverages the text encoder $T$ and image encoder $I$ of a vision-language model (e.g., CLIP) to optimize the prompt using a contrastive learning loss. The key formula is:
    $$\mathcal{L} = -\log \frac{\exp(\text{sim}(T(\mathbf{p}), I(\mathbf{x}^s)) / \tau) + \exp(\text{sim}(T(\mathbf{p}), I(\mathbf{x}^+)) / \tau)}{\sum_{\mathbf{x} \in \{\mathbf{x}^s, \mathbf{x}^+, \mathbf{x}^-\}} \exp(\text{sim}(T(\mathbf{p}), I(\mathbf{x})) / \tau)}$$
      where $\text{sim}$ is the cosine similarity and $\tau$ is the temperature coefficient. This pulls the prompt's textual features closer to the seed image and positive samples, while pushing them away from negative samples.
    - **Design Motivation**: Unlike existing prompt optimization methods like Textual Inversion and PEZ, Stealix integrates the classification feedback of the victim model (partitioning of positive/negative samples) into the optimization objective. This aligns the prompt not only visually with the target but also in terms of "classification semantics". As shown in Figure 3, irrelevant features are filtered out via negative samples (e.g., removing "pool" features for the "bottle" class).

2. **Prompt Consistency (Fitness Function)**:

    - **Function**: Evaluates the quality of the optimized prompt, serving as the fitness function for the genetic algorithm.
    - **Mechanism**: Uses the optimized prompt to drive a generative model $G$ to synthesize a batch of images, which are then classified by the victim model $V$. Fitness is defined as:
    $$\text{PC}(\mathbf{p}, c) = \frac{1}{M} \sum_{j=1}^{M} \mathbb{1}[V(G(\mathbf{p})_j) = c]$$
      Namely, the proportion of the $M$ synthesized images classified as the target class $c$ by the victim model. A higher PC value indicates a more accurate prompt.
    - **Design Motivation**: Directly uses feedback from the victim model to evaluate prompts, creating a closed loop. Statistical analysis in the paper demonstrates that PC is highly correlated with the feature distance between synthetic images and real data. Meanwhile, the synthetic images generated during evaluation update the positive/negative sample sets based on classification results, accumulating more signals for the next optimization round.

3. **Prompt Reproduction (Genetic Evolution)**:

    - **Function**: Evolves the population of image triplets for the next generation using selection, crossover, and mutation operations of the genetic algorithm, based on the fitness scores.
    - **Mechanism**: Triplets with higher PC scores have a higher probability of being selected into the next generation. The crossover operation mixes the seeds, positive samples, and negative samples of different triplets to introduce diversity. The mutation operation randomly replaces some samples to avoid premature convergence.
    - **Design Motivation**: Genetic algorithms are naturally suited for non-differentiable, discrete search spaces. A prompt requires two black-box stages (the generative model and the victim model) for evaluation, preventing gradient backpropagation. Additionally, balancing accuracy and diversity is inherently achieved by the population-based mechanism of genetic algorithms.

4. **Surrogate Model Training**:

    - **Function**: Collects all synthetic images and victim model labels accumulated across all iterations to train the final surrogate model $A$.
    - **Mechanism**: Minimizes $\arg\min_{\theta_a} \mathbb{E}_{\mathbf{x} \sim G(\mathbf{p})}[\mathcal{L}_{CE}(V(\mathbf{x}), A(\mathbf{x}))]$.
    - **Design Motivation**: Following prompt evolution, the distribution of synthesized data is highly aligned with the victim model's training data, enabling the surrogate model to effectively learn the decision boundaries of the victim model.

### Loss & Training

- **Prompt Optimization Stage**: Contrastive learning loss (Eq. 3), optimizing discrete prompts via CLIP's image-text alignment space.
- **Surrogate Model Training Stage**: Standard cross-entropy loss, using the victim model's top-1 predictions as pseudo-labels.
- Requires only 1 seed image per class, a population size of $N$, and generates $M$ images per prompt for evaluation.
- The entire process is constrained by a query budget $B$ per class.

## Key Experimental Results

### Main Results

The paper compares Stealix with existing model stealing methods across multiple datasets. Stealix comprehensively outperforms baseline methods that use class names or handcrafted prompts, even without requiring class names:

| Method | Prior Knowledge Req. | Querying Mechanism | Low-budget Advantage | Applicability to Specialized Domains |
|------|------------|---------|-----------|-------------|
| Knockoff Nets | Requires public dataset of similar distribution | Directly query public images | Poor (collapses when dataset mismatches) | Poor |
| SD + Class Name Prompt | Requires class names | Generate via class name prompt | Moderate | Poor (class names not descriptive enough) |
| SD + Handcrafted Dense Prompt | Requires domain knowledge & prompt-engineering skills | Meticulously designed prompts | Better | Poor (requires expertise) |
| ASPKD | Requires class names + nearest-neighbor matching | Diffusion model + pseudo-labels | Better | Moderate |
| **Stealix** | **Only 1 image/class, no class names** | **Automated prompt evolution** | **Best (+22.2%)** | **Best** |

### Effectiveness Analysis of Prompt Consistency

| Experimental Setting | Correlation of PC vs. Feature Distance | Surrogate Model Accuracy Trend | Explanation |
|---------|-------------------|-----------------|------|
| Full Stealix (Complete Method) | Highly negatively correlated | Highest | Higher PC $\to$ closer feature distance $\to$ higher accuracy |
| w/o Prompt Refinement | — | Significant drop | Lacks contrastive optimization, leading to low-quality random prompts |
| w/o Prompt Consistency | — | Notable drop | Lacks the fitness function, causing blind search under the genetic algorithm |
| w/o Prompt Reproduction | — | Moderate drop | Lacks evolutionary mechanism, resulting in insufficient prompt diversity |
| DA-Fusion baseline | Low | Lower | Ignores victim model feedback, causing prompt misalignment with target |

### Key Findings

- **Prompt Consistency is a reliable surrogate metric**: Statistical analysis demonstrates that the PC metric is highly correlated with the feature distance between synthetic images and real data, enabling prompt quality evaluation without accessing real data.
- **Effective attack with only 1 seed image per class**: This extremely low-resource assumption is highly realistic, yet existing methods perform far worse than Stealix under this condition.
- **Most prominent advantage under low query budgets**: Achievements of up to +22.2% gain, because efficient prompts make every query more valuable, avoiding waste on numerous low-quality queries.
- **Pronounced advantage in specialized domains**: In specialized domains like satellite imagery, handcrafted prompts are extremely difficult to design. Stealix's automated search amplifies its advantage here.
- **Three components are indispensable**: Ablation studies show that Prompt Refinement, Prompt Consistency, and Prompt Reproduction each make significant contributions; removing any of them leads to a substantial performance drop.

## Highlights & Insights

- **First model stealing framework without prompt priors**: This is a major methodological breakthrough. Prior methods assumed adversaries know class names or can design prompts, severely underestimating the threat. Stealix demonstrates that effective prompts can be automatically discovered with just a single image, significantly lowering the barrier to model stealing.

- **Clever integration of genetic algorithms and contrastive learning**: Uses a genetic algorithm to handle search in the discrete prompt space while injecting vision-semantic alignment signals via CLIP's contrastive learning. The two complement each other—contrastive learning provides local optimization directions, while the genetic algorithm enables global exploration capability. This hybrid paradigm of "differentiable optimization + evolutionary search" holds broad reference value.

- **Prompt Consistency as a surrogate metric**: Employing the victim model's own classification consistency as a fitness function is both simple and effective. This paradigm of "attacking a target model using its own feedback" has universal significance in adversarial ML.

- **Closed-loop positive feedback design**: The positive and negative sample sets continuously expand throughout iterations. Early-stage exploration results steadily provide signals for subsequent optimization, establishing a self-improving positive loop. This is transferable to any scenario requiring interaction with black-box systems to progressively approach a target (e.g., red-teaming).

## Limitations & Future Work

- **Dependence on pre-trained model coverage**: Stealix relies on the generative capabilities of Stable Diffusion and the semantic understanding of CLIP. If the target domain lies far beyond the pre-training distribution of these models (e.g., rare medical imaging or industrial defect detection), the prompt evolution might fail to converge to an effective solution.

- **Room for query efficiency optimization**: Evaluating PC requires generating $M$ images and querying the victim model for each, accumulating a high volume of queries over generations. Future research could introduce Bayesian Optimization or train a lightweight surrogate model to reduce evaluation overhead.

- **Only validated on image classification**: The paper evaluates only on image classification, without extending to more complex vision tasks like object detection or semantic segmentation, or other modalities like NLP/multimodal settings.

- **Limited discussion on defense countermeasures**: The paper heavily focuses on attacks, lacking depth of analysis for defense perspectives (e.g., detecting anomalous query patterns, output perturbation, watermarking mechanisms).

- **Insufficient discussion on seed image sensitivity**: Does the representativeness of different seed images affect final performance? If the seed is an extreme or non-typical sample, the evolution might get stuck in local optima.

## Related Work & Insights

- **vs Knockoff Nets (Orekondy et al., 2019)**: Queries the victim model using public datasets; performance drops sharply when the dataset mismatches the target distribution. Stealix adapts to the target distribution by automatically evolving prompts, bypassing dataset selection constraints.

- **vs ASPKD (Hondru & Ionescu, 2023)**: Uses a diffusion model but relies on class name prompts + nearest-neighbor pseudo-labels. Stealix requires no class names, and generates more precise query images through iterative optimization, outperforming ASPKD under the same query budget.

- **vs DA-Fusion (Trabucco et al., 2024)**: A data augmentation method that uses Textual Inversion to generate visually similar images. Stealix extends this to model stealing. The key refinement is incorporating classification feedback from the victim model to guide prompt optimization, rather than simply relying on original class labels.

- **vs PEZ (Wen et al., 2024)**: Uses CLIP to optimize discrete prompts, but in a "task-agnostic" manner. Stealix introduces a "task-aware" contrastive learning objective incorporating victim model feedback, aligning optimization directions with the downstream model stealing goal.

- **Insight**: This paradigm of "genetic prompt evolution + black-box closed-loop feedback" is transferable to: (1) automated adversarial prompt search in red-teaming, (2) robustness evaluation of model watermarking, and (3) any scenario requiring black-box optimization in discrete spaces.

## Rating

- Novelty: ⭐⭐⭐⭐ The first model stealing method without prompt priors; the framework of using genetic algorithms to search for prompts is highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Rigorously validated with multi-dataset evaluations, ablation analyses, and PC metric statistical verification.
- Writing Quality: ⭐⭐⭐⭐ Well-defined threat model, clear and logical method description, and intuitive illustrations.
- Value: ⭐⭐⭐⭐ Reveals the underestimated risks of pre-trained generative models in model stealing, providing practical alerts to AI security defenders.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Emergence and Evolution of Interpretable Concepts in Diffusion Models](../../NeurIPS2025/image_generation/emergence_and_evolution_of_interpretable_concepts_in_diffusi.md)
- [\[ICML 2025\] DDIS: When Model Knowledge Meets Diffusion Model](when_model_knowledge_meets_diffusion_model_diffusion-assisted_data-free_image_synthesis.md)
- [\[CVPR 2025\] Minority-Focused Text-to-Image Generation via Prompt Optimization](../../CVPR2025/image_generation/minority-focused_text-to-image_generation_via_prompt_optimization.md)
- [\[CVPR 2025\] EvoTok: A Unified Image Tokenizer via Residual Latent Evolution for Visual Understanding and Generation](../../CVPR2025/image_generation/evotok_a_unified_image_tokenizer_via_residual_latent_evolution_for_visual_unders.md)
- [\[ICCV 2025\] LATINO-PRO: LAtent consisTency INverse sOlver with PRompt Optimization](../../ICCV2025/image_generation/latino-pro_latent_consistency_inverse_solver_with_prompt_optimization.md)

</div>

<!-- RELATED:END -->

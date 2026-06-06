---
title: >-
  [Paper Note] A Language-Guided Bayesian Optimization for Efficient LoRA Hyperparameter Search
description: >-
  [ICML2026][Model Compression][LoRA tuning] This paper represents LoRA hyperparameter configurations as text with domain explanations, allowing a frozen LLM, learnable tokens…
tags:
  - "ICML2026"
  - "Model Compression"
  - "LoRA tuning"
  - "Bayesian Optimization"
  - "LLM embedding"
  - "surrogate training"
  - "parameter-efficient fine-tuning"
date: 2026-05-08
content_hash: ca55a8f802a1922d
---

# A Language-Guided Bayesian Optimization for Efficient LoRA Hyperparameter Search

**Conference**: ICML2026  
**arXiv**: [2602.11171](https://arxiv.org/abs/2602.11171)  
**Code**: Project Page: https://baekseongeun.github.io/lora-bo/ (Code repository not found in cache)  
**Area**: Model Compression / LLM Efficiency / Parameter-Efficient Fine-Tuning  
**Keywords**: LoRA tuning, Bayesian Optimization, LLM embedding, surrogate training, parameter-efficient fine-tuning  

## TL;DR
This paper represents LoRA hyperparameter configurations as text with domain explanations, allowing a frozen LLM, learnable tokens, and a projection layer to jointly construct a continuous search space for BO. By utilizing a 10% data proxy for evaluation to reduce per-trial costs, the method significantly outperforms default LoRA configurations and conventional HPO methods within approximately 30 search iterations.

## Background & Motivation
**Background**: LoRA and its variants have become the most widely used parameter-efficient solutions for fine-tuning large models. In practice, the original model weights are frozen, and only low-rank adapters are trained. Adaptation capability, stability, and computational overhead are controlled through a few hyperparameters such as rank, scaling factor, learning rate, batch size, and dropout.

**Limitations of Prior Work**: While LoRA's advantage lies in its few trainable parameters, this does not imply that hyperparameter tuning is straightforward. The paper notes that combinations of rank-to-alpha ratios, learning rate, batch size, and dropout strongly impact final performance, with a full grid space exceeding 45,000 configurations. Exhaustive training is prohibitively expensive, and direct application of random search, Optuna, standard BO, or discrete space optimization fails to explicitly incorporate empirical LoRA principles into the search process.

**Key Challenge**: LoRA HPO faces two primary mismatches. First, most variables are discrete hyperparameters, whereas Gaussian Process (GP)-based BO prefers continuous, smooth, and structured input spaces. Second, while human experts possess empirical knowledge about LoRA tuning—such as the relationship between alpha and rank, the impact of large batches on generalization, and the role of dropout in stability—traditional BO typically only sees numerical encodings and lacks an understanding of these domain semantics.

**Goal**: The authors aim to transform LoRA tuning expertise into representations usable by BO, allowing the search to leverage LLM prior knowledge and find high-quality configurations within few real training iterations. Simultaneously, they aim to reduce per-evaluation costs so that a budget of approximately 30 trials can cover a large candidate pool.

**Key Insight**: The observation is that a hyperparameter configuration is not just a set of numbers but can also be a structured linguistic description. Pre-trained LLMs can encode relationships and roles in natural language. By including hyperparameter names, values, functions, and interrelationships in a prompt and mapping the LLM hidden states to a continuous space, discrete configurations can be transformed into continuous representations better suited for BO.

**Core Idea**: Use "verbalized LoRA domain knowledge + calibratable LLM embeddings" to replace standard numerical encoding, enabling Bayesian Optimization to select the next set of LoRA hyperparameters within a more semantically structured space.

## Method
The proposed method can be understood as a closed-loop tuner for LoRA. In each round, a set of LoRA hyperparameters is selected from a candidate pool, followed by proxy training and evaluation on a small data subset. These hyperparameters are then converted into a text template containing explanations, which is processed by a frozen LLM, learnable tokens, and a projection layer to obtain a continuous vector. Subsequently, these vectors and the corresponding evaluation scores are used to train a GP surrogate. Finally, an acquisition function scores all remaining candidate configurations to select the next configuration for evaluation.

### Overall Architecture
The input is a set of discrete candidate configurations $\mathcal{X}_{cand}$, each containing rank, scaling factor, batch size, learning rate, and dropout rate. The output is the optimal LoRA configuration found within the budget. In the $n$-th round, the framework takes configuration $x_n$, obtains performance $y_n$ via proxy training, and writes $x_n$ into an annotated text template $t_n$. The frozen LLM $\phi$ receives $t_n$ and learnable tokens $\psi$, while the projection layer $P(\cdot;\theta)$ maps the hidden state to BO features $z_n=P(\phi(t_n,\psi);\theta)$. The GP surrogate uses these $z$ and corresponding $y$ to maximize marginal log-likelihood, updating the GP kernel, projection layer, and learnable tokens. Every unevaluated configuration in the candidate pool is also encoded as $z_j$, and the acquisition function selects the next most promising point.

The key to this pipeline is not simply "letting the LLM generate hyperparameters," but rather tasking the LLM with constructing a continuous embedding space that incorporates the structural domain knowledge of LoRA, while BO handles the exploration/exploitation tradeoff. This preserves the sample efficiency of BO while avoiding the instability of pure prompt-agent-based searches.

### Key Designs
1. **Domain-Aware Text Template**:
    - **Function**: Rewrites each set of LoRA hyperparameters from discrete numerical encodings into structured text containing names, values, and functional explanations.
    - **Mechanism**: While standard templates only include `{name, value}`, this work adds `{explanation, name, value}`. The explanation text describes empirical knowledge, such as common relationships between rank and alpha, the effects of different batch sizes on training dynamics, and when dropout is beneficial. Thus, the LLM reads a description of "why this hyperparameter set is important" rather than isolated numbers.
    - **Design Motivation**: LoRA tuning relies heavily on manual experience, which is lost when feeding numbers directly into BO. By writing experience into the prompt, LLM embeddings can better organize configurations that are "numerically close but semantically different" or "numerically different but functionally similar."

2. **Learnable Tokens and Projection Layer for Embedding Space Calibration**:
    - **Function**: Transforms the general text representations from the frozen LLM into BO features more suitable for LoRA HPO.
    - **Mechanism**: The method appends learnable tokens $\psi$ to the prompt, takes the hidden state at the last token position, and passes it through a projection layer $P(\cdot;\theta)$ to obtain $z=P(\phi(t,\psi);\theta)$. LLM parameters remain frozen; only $\psi$, $\theta$, and the GP kernel parameters $\omega$ are trained to maximize the GP marginal log-likelihood.
    - **Design Motivation**: Original embeddings from a frozen LLM may not be ordered by "LoRA performance." The projection layer reshapes the geometry, while learnable tokens capture residual information difficult to express in prompts, making it easier for the surrogate to fit performance trends with few observation points.

3. **Proxy Training Evaluation to Reduce Per-Point Costs**:
    - **Function**: Uses training on a small data subset to approximate performance on a full dataset, reducing the cost of each function evaluation in BO.
    - **Mechanism**: Instead of full fine-tuning on a 100K training set each round, fine-tuning is performed on a randomly sampled 10K subset, and the result is fed back to BO as $y_n$. The authors further validated correlations between 1%, 5%, and 10% random subsets and TSDS sampling versus full training, ultimately adopting 10% random sampling.
    - **Design Motivation**: The bottleneck in HPO is often the training of each candidate rather than surrogate computation. If a 10% subset correlates highly with full results, the search budget can be allocated to more configurations rather than exhausting resources on a single full training run.

### Loss & Training
The BO surrogate utilizes a GP and models configuration performance through deep kernel learning after LLM embedding. The standard kernel $k(x,x'|\omega)$ is replaced by $k(g(x;\theta,\psi),g(x';\theta,\psi)|\omega,\theta,\psi)$. During training, the marginal log-likelihood $\mathcal{L}(\Phi)=\log p(y|X,\Phi)$ is maximized, where $\Phi=\{\omega,\theta,\psi\}$. In experiments, all HPO methods are limited to 30 iterations, with a candidate pool covering over 45,000 LoRA configurations. Tasks include mathematical reasoning, code generation, and dialogue, trained on MetaMathQA, CodeFeedback, and WizardLM-Evol-Instruct, and evaluated on GSM8K, MATH, HumanEval, MBPP, and MT-Bench.

## Key Experimental Results

### Main Results
Main results indicate that the proposed method improves various LoRA variants and works across different backbones. Representative results from the LoRA variant table are shown below, where gains are absolute improvements over the original paper/default recommended configurations.

| Variant | GSM8K Acc | MATH Acc | HumanEval Pass@1 | MBPP Pass@1 | MT-Bench |
| :--- | :--- | :--- | :--- | :--- | :--- |
| LoRA Default | 41.47 | 5.24 | 16.31 | 35.47 | 7.181 |
| LoRA + Ours | 62.93 (+21.46) | 12.88 (+7.64) | 30.49 (+14.18) | 42.59 (+7.12) | 7.350 (+0.169) |
| rsLoRA Default | 41.16 | 5.46 | 16.46 | 35.72 | 7.300 |
| rsLoRA + Ours | 58.15 (+16.99) | 10.76 (+5.30) | 29.87 (+13.41) | 42.06 (+6.34) | 7.662 (+0.362) |
| DoRA Default | 40.11 | 5.36 | 17.07 | 36.51 | 7.125 |
| DoRA + Ours | 57.01 (+16.90) | 10.78 (+5.42) | 30.58 (+13.51) | 42.33 (+5.82) | 7.475 (+0.350) |
| PiSSA Default | 52.46 | 7.34 | 22.56 | 40.48 | 7.200 |
| PiSSA + Ours | 60.88 (+8.42) | 12.06 (+4.72) | 31.71 (+9.15) | 41.53 (+1.05) | 7.475 (+0.275) |

Under the same budget of 30 iterations, the proposed method also outperforms common HPO baselines.

| Search Method | GSM8K Acc | MATH Acc | HumanEval Pass@1 | MBPP Pass@1 |
| :--- | :--- | :--- | :--- | :--- |
| Random | 59.14 | 10.51 | 23.17 | 36.77 |
| Optuna | 54.13 | 10.50 | 27.44 | 38.62 |
| BO | 57.32 | 11.42 | 20.12 | 35.19 |
| LBO | 59.51 | 11.88 | 26.83 | 37.83 |
| Ours | 62.93 | 12.88 | 30.49 | 42.59 |

### Ablation Study
Ablations confirm that all three components are effective, with domain-aware prompting being the most critical for explicit knowledge injection.

| Projection Layer | Domain-Aware Prompt | Learnable Token | GSM8K Acc | MATH Acc | Note |
| :--- | :--- | :--- | :--- | :--- | :--- |
| ✗ | ✗ | ✗ | 47.76 | 8.72 | Frozen LLM embeddings used directly for BO; poor search space discriminability |
| ✓ | ✗ | ✗ | 53.98 | 9.16 | Projection layer provides some calibration, but lacks LoRA semantics |
| ✓ | ✓ | ✗ | 61.41 | 12.46 | Performance significantly improves after explicitly adding tuning knowledge |
| ✓ | ✓ | ✓ | 62.93 | 12.88 | Learnable tokens further capture information hard to express in prompts |

Proxy training correlation suggests that a 10% random subset is sufficiently close to full training trends.

| Sampling Method | MATH Reasoning Corr. | Code Gen. Corr. | Conclusion |
| :--- | :--- | :--- | :--- |
| Random 1% | 0.7031 | 0.7429 | Captures trend, but noise is high |
| Random 5% | 0.8360 | 0.9282 | Already relatively stable |
| Random 10% | 0.8713 | 0.9427 | Adopted in this work; highest correlation in code tasks |
| TSDS 10% by test | 0.8754 | 0.9290 | Slightly higher for math, but lower for code than random 10% |
| TSDS 10% by train | 0.8649 | 0.9278 | Close to random 10% |

### Key Findings
- Domain-aware prompt is the largest contributor: adding the projection layer alone raises scores from 47.76/8.72 to 53.98/9.16, but adding prompts further boosts them to 61.41/12.46, proving that verbalizing LoRA knowledge changes the information available to BO.
- High-performance configurations found during search do not always follow traditional experience; for instance, alpha sometimes reaches 16 or 32 times the rank instead of the usual 2x. This suggests current LoRA tuning rules still have room for discovery.
- Proxy training is not merely a compute-saving trick. The correlation of the 10% random subset with full results (0.8713 for MATH, 0.9427 for Code) is sufficient for BO to select the next point.
- Hyperparameters cannot be easily transferred between models. Cross-model configuration experiments show that applying configurations found for one model series to another leads to significant performance drops, making automated per-model search more reliable than reused manual experience.

## Highlights & Insights
- The ingenuity of this paper lies in not asking the LLM to directly "guess" hyperparameters, but rather making the LLM the representation function for BO. Thus, the LLM provides semantic structure while BO handles the black-box optimization, creating a clear division of labor.
- The learnable token serves as a lightweight yet practical calibration point. While a prompt can state human-known rules, it cannot exhaust all residual information; allowing a token to update with the marginal likelihood acts as a latent variable that adapts to the current task.
- Validation of proxy evaluation is crucial. Many HPO papers assume subset training by default for efficiency, but this work explicitly compares sampling ratios and TSDS, showing that the 10% random subset is a well-founded choice.
- This approach is transferable to other discrete HPO problems, such as quantization configurations, distillation hyperparameters, RAG retrieval parameters, or inference-time decoding parameters. Any expert knowledge that can be verbalized can be used to set up structured prompts for BO search.

## Limitations & Future Work
- The paper primarily validates LoRA and a few variants and has yet to prove that the same verbalized BO representation generalizes to all PEFT methods or non-LLM tasks.
- The method depends on the embedding quality of the pre-trained LLM. Performance might decrease with weaker embedding models or if domain knowledge cannot be clearly expressed in prompts; although an embedding model ablation is provided, redistribution requires re-validation.
- Performance is strong under a 30-iteration budget, but the candidate pool remains a manually defined discrete range. Optimal points outside this range cannot be discovered regardless of representation quality.
- Proxy training assumes that subset performance remains highly correlated with full performance. This assumption might fail for small datasets, strong distribution shifts, or long-tail tasks, necessitating dynamic subset selection or multi-fidelity BO.
- Currently, the search target is primarily benchmark scores. Future work could include multi-objective constraints like training cost, VRAM usage, latency, and stability to bring LoRA configurations closer to real-world deployment.

## Related Work & Insights
- **vs. Traditional BO / Optuna / LBO**: These treat configurations as numerical or latent variables. This work extractions LoRA domain knowledge and LLM text understanding, making it easier to find superior configurations within the same 30-iteration budget.
- **vs. NOMAD-style LoRA HPO**: NOMAD also targets LoRA tuning, but this work achieves better results on GSM8K/MATH/HumanEval/MBPP within 24 hours compared to the 180 hours required by Tribes et al., owing to more efficient search space representation and proxy evaluation.
- **vs. Manual Tuning Experience**: Manual experience often provides fixed rules for rank, alpha, and batch size. This work finds that larger alpha/rank ratios are sometimes more effective, suggesting automated search can update empirical rules.
- **vs. LLM Agent-based Tuning**: Directly letting an LLM propose configurations is prone to prompt sensitivity and sampling instability. This work keeps the LLM as a trainable representation while optimization is handled by an acquisition function, effectively embedding LLM priors into a classic HPO framework.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Combines LLM representation, domain prompts, learnable tokens, and BO for LoRA HPO; the problem framing is highly practical.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers LoRA variants, multiple models, HPO baselines, component ablations, and proxy evaluation, though real-world industrial deployment dimensions could be expanded.
- Writing Quality: ⭐⭐⭐⭐☆ Clear methodology chain with strong tabular support; formulas and algorithms might be slightly challenging for non-BO readers.
- Value: ⭐⭐⭐⭐⭐ Extremely valuable for scenarios requiring frequent LLM fine-tuning; the core idea is transferable to other expert-knowledge-driven discrete search problems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Renormalization Group Guided Tensor Network Structure Search](../../AAAI2026/model_compression/renormalization_group_guided_tensor_network_structure_search.md)
- [\[NeurIPS 2025\] Learning to Better Search with Language Models via Guided Reinforced Self-Training](../../NeurIPS2025/model_compression/learning_to_better_search_with_language_models_via_guided_reinforced_self-traini.md)
- [\[ICML 2026\] Active Budget Allocation for Efficient Scaling Law Estimation via Surrogate-Guided Pruning](active_budget_allocation_for_efficient_scaling_law_estimation_via_surrogate-guid.md)
- [\[NeurIPS 2025\] Robust Federated Finetuning of LLMs via Alternating Optimization of LoRA](../../NeurIPS2025/model_compression/robust_federated_finetuning_of_llms_via_alternating_optimization_of_lora.md)
- [\[AAAI 2026\] Efficient Reasoning for Large Reasoning Language Models via Certainty-Guided Reflection Suppression](../../AAAI2026/model_compression/efficient_reasoning_for_large_reasoning_language_models_via_certainty-guided_ref.md)

</div>

<!-- RELATED:END -->

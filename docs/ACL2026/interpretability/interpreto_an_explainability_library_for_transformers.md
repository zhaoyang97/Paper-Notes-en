---
title: >-
  [Paper Note] Interpreto: An Explainability Library for Transformers
description: >-
  [ACL2026][Interpretability][Transformer Interpretability] Interpreto is an open-source Python interpretability library for HuggingFace language models that unifies token/word/sentence attribution and activation-level con…
tags:
  - "ACL2026"
  - "Interpretability"
  - "Transformer Interpretability"
  - "attribution"
  - "concept-based explanation"
  - "HuggingFace"
  - "mechanistic interpretability"
date: 2026-05-08
content_hash: 014d75038f7e9922
---

# Interpreto: An Explainability Library for Transformers

**Conference**: ACL2026  
**arXiv**: [2512.09730](https://arxiv.org/abs/2512.09730)  
**Code**: https://github.com/FOR-sight-ai/interpreto  
**Area**: Interpretability / Tool Library  
**Keywords**: Transformer Interpretability, attribution, concept-based explanation, HuggingFace, mechanistic interpretability  

## TL;DR
Interpreto is an open-source Python interpretability library for HuggingFace language models that unifies token/word/sentence attribution and activation-level concept explanations into a single API, providing demos, tutorials, metrics, and end-to-end concept explanation pipelines.

## Background & Motivation
**Background**: Transformer language models are widely used for classification and generation. Practical deployment requires explanation tools for debugging, bias analysis, security auditing, and documentation. Existing tools are generally divided into attribution libraries and mechanistic/concept interpretability libraries.

**Limitations of Prior Work**: Many libraries only cover a single family of explanations, one task type, or a single pipeline stage. For instance, some libraries excel at token attribution but do not support generative models; others can train SAEs or concept models but lack a complete workflow from activation extraction and concept learning to interpretation and contribution scoring. For general HuggingFace users, integrating these tools is costly.

**Key Challenge**: While interpretability research methods are increasing, practitioners need engineering tools that are installable, reproducible, comparable, and capable of running complete workflows. Tool fragmentation hinders the adoption of methods and complicates benchmarking different explanation results.

**Goal**: The authors aim to provide a unified library that allows users to explain both classification and generation models using the same interface, switching between attribution and concept-based explanations, while providing visualizations, metrics, tutorials, a demo gallery, and extensible interfaces for custom methods.

**Key Insight**: Interpreto is designed directly around the HuggingFace ecosystem. The `attributions` module provides common perturbation and gradient methods, while the `concepts` module wraps `nnsight` for model splitting and strings together activation extraction, concept learning, interpretation, and concept importance estimation.

**Core Idea**: Organize "collections of explanation methods" into executable engineering pipelines rather than just implementing isolated algorithms, specifically packaging the multiple stages of unsupervised concept discovery into a single library.

## Method
Interpreto's system consists of two main modules: `interpreto.attributions` and `interpreto.concepts`. The former explains the contribution of input features to predictions, while the latter learns higher-level concepts from the model's intermediate activations and analyzes how these concepts influence the output. The library covers both classification and generation tasks, providing notebook visualizations, a demo website, metrics, and minimal runnable snippets.

### Overall Architecture
The attribution pipeline typically involves three steps: instantiating the explainer with a HuggingFace model/tokenizer and the sample to be explained; computing the attribution by specifying a classification target or generated output token; and finally, visualizing the highlighted tokens, words, or sentences. In a classification example, LIME explains a BERT emotion classifier, showing "thrilled" driving the joy class; in a generation example, Occlusion explains a specific output token of Qwen3-0.6B and displays a slice of the input-output attribution matrix.

The concept pipeline consists of four steps. First, wrap the HuggingFace model using `ModelWithSplitPoints`, specify split points, and extract activations from a dataset. Second, learn the concept space using methods such as Semi-NMF, PCA, ICA, or SAE. Third, assign human-readable labels to concepts using top-k activating examples, tokens/ngrams, or LLM labels. Fourth, estimate the contribution of concepts to predictions via concept-to-output gradients or concept x gradients.

### Key Designs
1. **Unified attribution API covering classification and generation**:
	- Function: Explain `SequenceClassification` and `CausalLM` models using the same class of explainer.
	- Mechanism: Perturbation methods include KernelSHAP, LIME, Occlusion, and Sobol; gradient methods include GradientSHAP, Integrated Gradients, Saliency, SmoothGrad, SquareGrad, and VarGrad. It supports logits/softmax/log-softmax output spaces and token/word/sentence granularity.
	- Design Motivation: The readability of NLP attribution strongly depends on granularity and task objectives; a unified interface reduces the cost of switching between different libraries.

2. **End-to-end concept-based pipeline**:
	- Function: Learn concepts from model activations, interpret them, and estimate their importance.
	- Mechanism: The library uses `nnsight` for model splitting and implements concept learning using methods like overcomplete dictionary learning and sparse autoencoders (SAE), supporting neurons-as-concepts. Interpretation methods include top-k vocabulary tokens, MaxAct, and LLM-based labeling.
	- Design Motivation: Concept explanation is usually fragmented across multiple research tools; Interpreto's value lies in turning split, learn, interpret, and score into an executable workflow.

3. **Demos and runnable snippets bridging research and practice**:
	- Function: Allow users to explore pre-computed explanations on a web page before copying minimal code for local modification.
	- Mechanism: The demo website covers 3 classifiers and 3 generative models, allowing users to select tasks, models, datasets, explanation families, method subsets, and instances.
	- Design Motivation: The barrier to explanation tools is not just the API, but also understanding and reproducing results; a gallery + snippets lowers the barrier to trial.

### Loss & Training
As a system/tool paper, no new training loss is proposed. The library relies on the computation processes of existing explanation methods. Attribution methods are internally divided into perturbation, inference/gradient, and aggregation stages. Concept methods are executed according to activation extraction, concept model fitting, interpretation, and importance scoring. The environment supports Python 3.10 to 3.13, `torch >= 2.0`, `transformers >= 4.22`, and `nnsight >= 0.5.1`.

## Key Experimental Results

### Main Results
| Capability | Interpreto | Captum | Ferret | Inseq | SHAP |
|------|------------|--------|--------|-------|------|
| Sequence classification | ✓ | ✓ | ✓ | ✗ | ✓ |
| Text generation | ✓ | ✓ | ✗ | ✓ | ✓ |
| Faithfulness metrics | ✓ | ✓ | ✓ | ✗ | ✗ |
| Simple visualization | ✓ | ✗ | ✗ | ✗ | ✓ |
| Granularity control | ✓ | ✗ | ✗ | ✗ | ✗ |

In the comparison of attribution libraries, Interpreto is the only library in the table that simultaneously supports classification, generation, faithfulness metrics, simple visualization, and granularity control.

### Ablation Study
| Dimension | Interpreto Support | Details |
|------|--------------------|----------|
| Attribution methods | 10 types | 4 perturbation-based, 6 gradient-based |
| Attribution metrics | 2 types | Insertion, Deletion |
| Concept-learning options | 15 types | neurons, KMeans/PCA/SVD/ICA/NMF/Semi-NMF/Convex NMF, various SAEs |
| Concept interpretation | 3 categories | top-k tokens, top-k activating examples/words/n-grams, LLM labels |
| Concept metrics | 7 types | Including MSE, FID, sparsity, stability, ConSim, etc. |
| Tested architectures | 15+ | Albert, BART, BERT, DistilBERT, Electra, Roberta, T5, GPT2, GPT-Neo, GPT-J, CodeGen, Falcon, Llama3, Mistral, Starcoder, Qwen3 |

### Key Findings
- In the comparison of concept-based libraries, Interpreto simultaneously covers model splitting, concept learning, interpretation, contributions, metrics, pip package, and documentation; many existing libraries only cover one or two stages.
- The demo gallery covers 6 models: DistilBERT/IMDB, BERT/emotion, and RoBERTa/AG-News as classifiers, and GPT-2, Qwen3-0.6B, and Llama 3.1 8B as generative models.
- Regarding execution costs, attribution typically requires 10-100 forward passes or 5-20 gradient computations, taking seconds; the concept pipeline experiments take minutes on an RTX 3080, while larger SAEs may require hours.
- A generation concepts example using Qwen3-0.6B on 100 AG-News samples to extract activations, train Semi-NMF concepts, and label them with GPT-4.1-nano runs within 3 minutes on an RTX 3080 10GB.

## Highlights & Insights
- The core contribution lies not in "inventing a new explanation algorithm," but in transforming scattered algorithms into a unified set of executable interfaces. For interpretability research, such engineering integration significantly reduces the cost of reproduction and comparison.
- Interpreto’s handling of generation attribution is practical: each output token is a prediction target, and displaying the full matrix is difficult to read; allowing users to select an output token to view input contributions aligns better with analysis workflows.
- The end-to-end encapsulation of the concept pipeline is particularly valuable. Many practitioners want to use methods like SAE/NMF to view concepts but get stuck between model splitting, activation collection, label interpretation, and importance computation; Interpreto connects these steps.

## Limitations & Future Work
- The authors emphasize that there is no "single universal explanation method." Users still need to compare multiple methods and verify reliability using counterfactual checks, ablations, and targeted slices.
- The meaning of attribution scores depends on the method. The same highlighting result in LIME, Integrated Gradients, or Occlusion may represent different mechanisms and cannot be interpreted simply as a causal explanation.
- LLM-based concept labels are sensitive to prompts and may be over-generalized, repetitive, too granular, or non-actionable. An uninterpretable concept could result from the model itself, a failure in concept space learning, or a failure of the labeler, making root cause identification difficult.
- The library currently focuses on HuggingFace text language models and does not cover circuit-level MI, data attribution, or feature visualization. Plans include adding supervised concepts and more attribution methods/metrics, with long-term expansion to ViT and multimodal transformers.

## Related Work & Insights
- **vs Captum / SHAP / Ferret / Inseq**: These libraries have strengths but are often incomplete in task coverage, metrics, visualization, or granularity control; Interpreto’s advantage is unifying attribution for classification and generation.
- **vs TransformerLens / NNsight / SAELens / Neuronpedia**: These tools are more focused on the research stage or specific pipeline segments; Interpreto leverages their ideas or underlying capabilities for a more complete HuggingFace workflow.
- **vs Individual demo notebooks**: By releasing the demo website, documentation, tutorials, and pip package together, Interpreto is better suited as an entry point for practitioner debugging and teaching.

## Rating
- Novelty: ⭐⭐⭐☆☆ Limited algorithmic novelty, but system integration and concept pipeline encapsulation provide clear contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ As a demo/system paper, functional coverage, library comparisons, demos, run costs, and tested architecture descriptions are comprehensive; lacks user studies or large-scale usage statistics.
- Writing Quality: ⭐⭐⭐⭐☆ Clear structure, direct tabular comparisons, and specific code examples.
- Value: ⭐⭐⭐⭐☆ Highly practical for interpretability debugging in the HuggingFace ecosystem, especially for comparing attribution and concept explanations within the same project.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Cognitive Fatigue in Autoregressive Transformers: Formalization and Measurement](../../ICML2026/interpretability/cognitive_fatigue_in_autoregressive_transformers_formalization_and_measurement.md)
- [\[NeurIPS 2025\] nnterp: A Standardized Interface for Mechanistic Interpretability of Transformers](../../NeurIPS2025/interpretability/nnterp_a_standardized_interface_for_mechanistic_interpretability_of_transformers.md)
- [\[CVPR 2026\] Inside-Out: Measuring Generalization in Vision Transformers Through Inner Workings](../../CVPR2026/interpretability/inside-out_measuring_generalization_in_vision_transformers_through_inner_working.md)
- [\[ICLR 2026\] Implicit Statistical Inference in Transformers: Approximating Likelihood-Ratio Tests In-Context](../../ICLR2026/interpretability/implicit_statistical_inference_in_transformers_approximating_likelihood-ratio_te.md)
- [\[ICLR 2026\] How Do Transformers Learn to Associate Tokens: Gradient Leading Terms Bring Mechanistic Understanding](../../ICLR2026/interpretability/how_do_transformers_learn_to_associate_tokens_gradient_leading_terms_bring_mecha.md)

</div>

<!-- RELATED:END -->

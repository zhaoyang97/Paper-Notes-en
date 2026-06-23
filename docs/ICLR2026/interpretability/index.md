---
title: >-
  ICLR2026 Interpretability Papers · 196 Notes
description: >-
  196 ICLR2026 papers in the Interpretability area, covering LLM, Reasoning, Alignment/RLHF, Layout & Composition, Multimodal/VLM, Agents and more. Each note has TL;DR, motivation, method, experiments, highlights, and limitations — 5-minute reads of core ideas.
tags:
  - "ICLR2026"
  - "Interpretability"
  - "AI paper notes"
  - "paper summaries"
  - "LLM"
  - "Reasoning"
  - "Alignment/RLHF"
  - "Layout & Composition"
  - "Multimodal/VLM"
  - "Agents"
item_list:
  - u: "a_comprehensive_information-decomposition_analysis_of_large_vision-language_mode/"
    t: "A Comprehensive Information-Decomposition Analysis of Large Vision-Language Models"
  - u: "abstopk_rethinking_sparse_autoencoders_for_bidirectional_features/"
    t: "AbsTopK: Rethinking Sparse Autoencoders For Bidirectional Features"
  - u: "activation_steering_with_a_feedback_controller/"
    t: "Activation Steering with a Feedback Controller"
  - u: "adaem_an_adaptively_and_automated_extensible_measurement_of_llms_value_differenc/"
    t: "AdAEM: An Adaptively and Automated Extensible Measurement of LLMs' Value Difference"
  - u: "adaptive_concept_discovery_for_interpretable_few-shot_text_classification/"
    t: "Adaptive Concept Discovery for Interpretable Few-Shot Text Classification"
  - u: "addressing_divergent_representations_causal/"
    t: "Addressing Divergent Representations from Causal Interventions on Neural Networks"
  - u: "an_information-theoretic_parameter-free_bayesian_framework_for_probing_labeled_d/"
    t: "An Information-Theoretic Parameter-Free Bayesian Framework for Probing Labeled Dependency Trees from Attention Score"
  - u: "attention_please_revisiting_attentive_probing_through_the_lens_of_efficiency/"
    t: "Attention, Please! Revisiting Attentive Probing Through the Lens of Efficiency"
  - u: "attention_sinks_and_compression_valleys_in_llms_are_two_sides_of_the_same_coin/"
    t: "Attention Sinks and Compression Valleys in LLMs are Two Sides of the Same Coin"
  - u: "automated_interpretability_metrics_do_not_distinguish_trained_and_random_transfo/"
    t: "Automated Interpretability Metrics Do Not Distinguish Trained and Random Transformers"
  - u: "bayesian_neural_networks_for_functional_anova_model/"
    t: "Bayesian Neural Networks for Functional ANOVA Model"
  - u: "behavior_learning_bl/"
    t: "Behavior Learning (BL)"
  - u: "beyond_linear_probes_dynamic_safety_monitoring_for_language_models/"
    t: "Beyond Linear Probes: Dynamic Safety Monitoring for Language Models"
  - u: "block_recurrent_dynamics_in_vision_transformers/"
    t: "Block Recurrent Dynamics in Vision Transformers"
  - u: "bridging_explainability_and_embeddings_bee_aware_of_spuriousness/"
    t: "Bridging Explainability and Embeddings: BEE Aware of Spuriousness"
  - u: "can_llms_reason_soundly_in_law_auditing_inference_patterns_for_legal_judgment/"
    t: "Can LLMs Reason Soundly in Law? Auditing Inference Patterns for Legal Judgment"
  - u: "causal_interpretation_of_neural_network_computations_with_contribution_decomposi/"
    t: "Causal Interpretation of Neural Network Computations with Contribution Decomposition"
  - u: "causality_invariance_function_and_concept_vectors_in_llms/"
    t: "Causality ≠ Invariance: Function and Concept Vectors in LLMs"
  - u: "certified_evaluation_of_model-level_explanations_for_graph_neural_networks/"
    t: "Certified Evaluation of Model-Level Explanations for Graph Neural Networks"
  - u: "circuit_insights_towards_interpretability_beyond_activations/"
    t: "Circuit Insights: Towards Interpretability Beyond Activations"
  - u: "comparing_the_learning_dynamics_of_in-context_learning_and_fine-tuning_in_langua/"
    t: "Comparing the learning dynamics of in-context learning and fine-tuning in language models"
  - u: "composable_sparse_subnetworks_via_maximum-entropy_principle/"
    t: "Composable Sparse Subnetworks via Maximum-Entropy Principle"
  - u: "concept-trak_understanding_how_diffusion_models_learn_concepts_through_concept_a/"
    t: "Concept-TRAK: Understanding how diffusion models learn concepts through concept attribution"
  - u: "concepts_information_bottleneck_models/"
    t: "Concepts' Information Bottleneck Models"
  - u: "conjuring_semantic_similarity/"
    t: "Conjuring Semantic Similarity"
  - u: "cot_vectors_transferring_and_probing_the_reasoning_mechanisms_of_llms/"
    t: "CoT Vectors: Transferring and Probing the Reasoning Mechanisms of LLMs"
  - u: "cross-modal_redundancy_and_the_geometry_of_vision-language_embeddings/"
    t: "Cross-Modal Redundancy and the Geometry of Vision-Language Embeddings"
  - u: "debugging_concept_bottleneck_models_through_removal_and_retraining/"
    t: "Debugging Concept Bottleneck Models through Removal and Retraining"
  - u: "decomposing_llm_computation_with_jets/"
    t: "Decomposing LLM Computation with Jets"
  - u: "decomposing_representation_space_into_interpretable_subspaces_with_unsupervised_/"
    t: "Decomposing Representation Space into Interpretable Subspaces with Unsupervised Learning"
item_total: 196
---

<!-- Auto-generated by src/gen_blog_index.py --lang en -->
# 🔬 Interpretability

**🔬 ICLR2026** · **196** paper notes

📌 **Same area in other venues:** [📷 CVPR2026 (33)](../../CVPR2026/interpretability/index.md) · [💬 ACL2026 (63)](../../ACL2026/interpretability/index.md) · [🧪 ICML2026 (92)](../../ICML2026/interpretability/index.md) · [🤖 AAAI2026 (37)](../../AAAI2026/interpretability/index.md) · [🧠 NeurIPS2025 (76)](../../NeurIPS2025/interpretability/index.md) · [📹 ICCV2025 (10)](../../ICCV2025/interpretability/index.md)

🔥 **Top topics:** LLM ×18 · Reasoning ×12 · Alignment/RLHF ×8 · Layout & Composition ×7 · Multimodal/VLM ×6

**[A Comprehensive Information-Decomposition Analysis of Large Vision-Language Models](a_comprehensive_information-decomposition_analysis_of_large_vision-language_mode.md)**

:   This work pioneeringly uses Partial Information Decomposition (PID) to decompose the "decision-relevant information" of LVLMs into four non-negative atoms: redundant, unique visual, unique language, and synergistic. It constructs a model-agnostic estimation pipeline to quantitatively characterize whether LVLMs rely on genuine cross-modal fusion or language priors across 26 models and 4 datasets from three dimensions: "breadth, depth, and time."

**[AbsTopK: Rethinking Sparse Autoencoders For Bidirectional Features](abstopk_rethinking_sparse_autoencoders_for_bidirectional_features.md)**

:   This paper derives SAEs using a unified framework of "unrolled proximal gradient descent for sparse coding," proving that ReLU, JumpReLU, and TopK are proximal operators for different sparse regularizers. It identifies that their shared non-negativity constraint splits bidirectional semantic concepts (e.g., male vs. female) into two redundant features. Consequently, the authors propose **AbsTopK SAE**, which removes the non-negativity constraint and selects the top $k$ activations by absolute value. This allows a single feature to encode opposite concepts using signs, outperforming TopK and JumpReLU in reconstruction, interpretability, and steering tasks, while rivaling or exceeding supervised Difference-in-Mean.

**[Activation Steering with a Feedback Controller](activation_steering_with_a_feedback_controller.md)**

:   This paper reinterprets LLM activation steering as a feedback control problem in control theory. It proves that mainstream methods such as ActAdd, DirAblate, and Mean-AcT are essentially Proportional (P) controllers and thus possess inherent steady-state errors. Consequently, it proposes using a full PID controller to calculate steering vectors (PID Steering), which consistently outperforms original methods in tasks like detoxification, jailbreaking, and image style control.

**[AdAEM: An Adaptively and Automated Extensible Measurement of LLMs' Value Difference](adaem_an_adaptively_and_automated_extensible_measurement_of_llms_value_differenc.md)**

:   The authors propose AdAEM, an adaptive and self-extending LLM value assessment framework. It uses information theory optimization to automatically generate test questions that maximize the revelation of value differences between different LLMs, addressing the "information deficiency" problem of existing static benchmarks that fail to distinguish model value orientations.

**[Adaptive Concept Discovery for Interpretable Few-Shot Text Classification](adaptive_concept_discovery_for_interpretable_few-shot_text_classification.md)**

:   StructCBM transforms the Concept Bottleneck Model (CBM) into a paradigm that relies solely on sample-concept similarity for prediction without training a classification head. It uses an LLM to generate a dual-layer concept library—consisting of "Prototype Concepts + Discriminative Concepts"—from a minimal set of samples. It produces interpretable predictions through two-stage similarity matching (recalling candidate labels followed by discriminative contrast) and employs a closed-loop "misclassification feedback to LLM for concept refinement" mechanism. At 10-shot, it outperforms all existing CBMs, approaches the black-box performance of direct LLM calls on semantically dense datasets, and eliminates the need for LLMs during inference.

**[Addressing Divergent Representations from Causal Interventions on Neural Networks](addressing_divergent_representations_causal.md)**

:   This work systematically reveals that causal interventions (such as activation patching, DAS, and SAE) push internal model representations away from their natural distributions. It theoretically distinguishes between "harmless" and "harmful" shifts and proposes the Counterfactual Latent (CL) loss to constrain intervened representations within the natural manifold. Evaluations on 7B LLMs demonstrate that this approach reduces divergence while maintaining intervention accuracy.

**[An Information-Theoretic Parameter-Free Bayesian Framework for Probing Labeled Dependency Trees from Attention Score](an_information-theoretic_parameter-free_bayesian_framework_for_probing_labeled_d.md)**

:   IPBP does not train any probing network. It directly performs kernel density estimation on the joint distribution of "attention scores" and "dependency relations" to calculate the mutual information (MI) between each attention head and various dependency types in closed form. Using Bayesian posterior + geometric mean pooling + Eisner decoding, it reconstructs **labeled** dependency trees. On several 7B/8B LLMs, it proves more accurate than many supervised/unsupervised baselines and is inherently interpretable.

**[Attention, Please! Revisiting Attentive Probing Through the Lens of Efficiency](attention_please_revisiting_attentive_probing_through_the_lens_of_efficiency.md)**

:   Addressing the common parameter bloat in "attentive probing"—an increasingly popular evaluation protocol for frozen representations—this paper first unifies existing methods into a single framework. By leveraging the **mathematical equivalence** between Multi-Head Cross-Attention (MHCA) and Multi-Query Cross-Attention (MQCA), it removes redundant projection matrices to propose the extremely lightweight Efficient Probing (EP). On ImageNet-1K, EP achieves 75.6% accuracy for MAE ViT-B using less than 1.4M parameters (compared to 67.7% for linear probing) and consistently outperforms linear probing and existing attentive probes across diverse pre-training paradigms.

**[Attention Sinks and Compression Valleys in LLMs are Two Sides of the Same Coin](attention_sinks_and_compression_valleys_in_llms_are_two_sides_of_the_same_coin.md)**

:   This paper demonstrates that two seemingly independent puzzles in LLMs—attention sinks and compression valleys—are actually **two facets of the same mechanism: massive activations in the residual stream**. Based on this, it proposes the Mix-Compress-Refine three-phase information flow theory, unifying the explanation of why embedding tasks are strongest in the middle layers while generation tasks require the full depth.

**[Automated Interpretability Metrics Do Not Distinguish Trained and Random Transformers](automated_interpretability_metrics_do_not_distinguish_trained_and_random_transfo.md)**

:   This paper conducts a "sanity check" on the currently popular Sparse Autoencoders (SAEs). By applying SAEs to both trained Transformers and **randomly initialized** Transformers, the authors find that commonly used automated interpretability scores (auto-interp AUROC) and reconstruction metrics are almost indistinguishable between the two. This suggests that high interpretability scores alone **cannot** prove that an SAE has captured computational features actually learned by the model.

**[Bayesian Neural Networks for Functional ANOVA Model](bayesian_neural_networks_for_functional_anova_model.md)**

:   The model treats the selection of components to be estimated within the functional ANOVA framework as learnable parameters. It utilizes an MCMC algorithm with stepwise proposals to automatically search for and estimate high-order interaction components in high-dimensional inputs. This approach circumvents the computational bottleneck of ANOVA-TPNN, where the number of components grows exponentially with the interaction order due to the requirement of pre-enumerating all components.

**[Behavior Learning (BL)](behavior_learning_bl.md)**

:   Inspired by behavioral science, this paper directly incorporates the assumption that "observations are solutions to an optimization problem" as a learnable module. Each module is a Utility Maximization Problem (UMP) expressible in symbolic form. These are hierarchically stacked into a composite utility function that induces a Gibbs distribution for prediction/generation, simultaneously achieving strong predictive power, intrinsic interpretability, and (in the IBL variant) parameter identifiability.

**[Beyond Linear Probes: Dynamic Safety Monitoring for Language Models](beyond_linear_probes_dynamic_safety_monitoring_for_language_models.md)**

:   The authors propose Truncated Polynomial Classifiers (TPC), which achieve dynamic safety monitoring through order-by-order training and truncated evaluation of polynomials in the LLM activation space. This allows for fast decisions on simple inputs using low-order (≈linear probe) terms and stronger protection for difficult inputs by adding high-order terms. TPC matches or surpasses MLP baselines on WildGuardMix and BeaverTails while providing built-in interpretability.

**[Block Recurrent Dynamics in Vision Transformers](block_recurrent_dynamics_in_vision_transformers.md)**

:   This paper proposes the "Block Recurrent Hypothesis" (BRH), suggesting that the depth of an $L$-layer pre-trained ViT can be approximated by the recurrent unrolling of $k \ll L$ weight-shared blocks. Using a distillation framework called Raptor, the authors compress DINOv2 into 2–3 recurrent blocks while retaining 96%–98% of ImageNet linear probe accuracy. Based on this, they interpret ViT's layer-wise computation through the lens of discrete-time dynamical systems.

**[Bridging Explainability and Embeddings: BEE Aware of Spuriousness](bridging_explainability_and_embeddings_bee_aware_of_spuriousness.md)**

:   Proposes the BEE framework, which identifies and names spurious correlations (SC) by analyzing how fine-tuning perturbs the weight space geometry of pre-trained representations. It discovers hidden data biases directly from classifier weights without needing counter-examples, identifying spurious associations in ImageNet-1k that cause accuracy drops of up to 95%.

**[Can LLMs Reason Soundly in Law? Auditing Inference Patterns for Legal Judgment](can_llms_reason_soundly_in_law_auditing_inference_patterns_for_legal_judgment.md)**

:   This paper moves beyond merely evaluating whether the "answers" of legal LLMs are correct. Instead, it **faithfully decomposes the model's score for each judgment into a set of AND/OR interaction patterns between input phrases**. Sixteen legal experts then labeled these phrases as "relevant / irrelevant / forbidden" to quantify "what logic the model actually uses to reach its decision." The results reveal that even when four major LLMs (including legal-specific ones) provide correct judgments, **more than half of the reasoning interactions are based on irrelevant or even incorrect grounds**—such as attributing one person's criminal behavior to another defendant or being biased by professional identity.

**[Causal Interpretation of Neural Network Computations with Contribution Decomposition](causal_interpretation_of_neural_network_computations_with_contribution_decomposi.md)**

:   The authors propose CODEC (Contribution Decomposition), which utilizes Integrated Gradients to calculate the contributions of hidden layer neurons toward the output (rather than merely analyzing activations). These contributions are then decomposed into sparse modes using a Sparse Autoencoder (SAE). This approach achieves stronger causal interpretability and network control capabilities compared to activation analysis and is successfully applied to ResNet-50 and biological retinal neural network models.

**[Causality ≠ Invariance: Function and Concept Vectors in LLMs](causality_invariance_function_and_concept_vectors_in_llms.md)**

:   This paper distinguishes between two distinct types of attention heads in LLMs: "causal heads" identified via activation patching (forming Function Vectors, which truly drive in-context learning behavior) and "invariant heads" identified via Representational Similarity Analysis (forming Concept Vectors, which stably encode abstract relational concepts across input formats and languages). The study proves that these two sets of heads barely overlap, revealing that "what drives task performance" and "what encodes abstract concepts" are handled by different mechanisms in LLMs.

**[Certified Evaluation of Model-Level Explanations for Graph Neural Networks](certified_evaluation_of_model-level_explanations_for_graph_neural_networks.md)**

:   This paper formalizes the long-standing problem of evaluating "whether model-level explanations for GNNs are good enough"—previously reliant on class scores and visual inspection—into a regression loss called **sufficiency risk**. By deriving distribution-free certified upper bounds, the authors introduce three computable metrics: Coverage, GGA, and Overlap (with finite-sample confidence intervals), enabling the first statistically guaranteed comparison between different explainers.

**[Circuit Insights: Towards Interpretability Beyond Activations](circuit_insights_towards_interpretability_beyond_activations.md)**

:   This paper introduces WeightLens and CircuitLens to evolve automated interpretability from merely observing activation-triggering samples to analyzing weight connections and circuit attributions. This approach more robustly explains token-level, context-dependent, and polysemantic features in transcoder activations.

**[Comparing the learning dynamics of in-context learning and fine-tuning in language models](comparing_the_learning_dynamics_of_in-context_learning_and_fine-tuning_in_langua.md)**

:   The authors treat In-Context Learning (ICL) and Supervised Fine-Tuning (SFT) as two "learning algorithms" and compare their learning trajectories and internal representations shot-by-shot on a geometrically controllable 2D linear classification toy task. They find that while both achieve similar generalization accuracy, their mechanisms differ significantly: ICL preserves rich input representations but carries stronger pre-training priors (numerical comparison, pattern matching), whereas SFT collapses representations along the label axis, resulting in higher confidence but greater brittleness.

**[Composable Sparse Subnetworks via Maximum-Entropy Principle](composable_sparse_subnetworks_via_maximum-entropy_principle.md)**

:   The authors utilize a "Maximum-Entropy Loss" based on KL divergence to train neural networks into sparse subnetworks (functional modules) that recognize only specified classes while remaining deliberately uncertain about others via uniform distributions. These expert modules are then combined into a generalist model through weight summation or logit averaging, reversing the paradigm from "post-hoc probing of entangled representations" to "modular and interpretable by design."

**[Concept-TRAK: Understanding how diffusion models learn concepts through concept attribution](concept-trak_understanding_how_diffusion_models_learn_concepts_through_concept_a.md)**

:   Concept-TRAK refines traditional "whole-image" level training data attribution to the "individual concept" level. By designing **concept-oriented reward training and utility losses** for influence functions, it enables precise identification of which training samples influenced a specific concept (e.g., the character "Pikachu" rather than a pencil drawing style) in an AI-generated image. It significantly outperforms TRAK, D-TRAK, and DAS across three benchmarks: Synthetic, CelebA-HQ, and AbC.

**[Concepts' Information Bottleneck Models](concepts_information_bottleneck_models.md)**

:   Information Bottleneck (IB) regularization is introduced at the concept layer of Concept Bottleneck Models (CBM) to learn minimal sufficient concept representations by penalizing $I(X;C)$ while preserving $I(C;Y)$. This consistently improves predictive performance and concept intervention reliability across six CBM variants and three benchmarks.

**[Conjuring Semantic Similarity](conjuring_semantic_similarity.md)**

:   The paper proposes a visual imagination-based metric for text semantic similarity. It measures semantic distance by calculating the Jeffreys divergence between the path measures of reverse SDEs induced by a text-conditioned diffusion model under two prompts. This metric can be directly computed via Monte-Carlo sampling and quantifies for the first time the alignment between the semantic space learned by diffusion models and human annotations.

**[CoT Vectors: Transferring and Probing the Reasoning Mechanisms of LLMs](cot_vectors_transferring_and_probing_the_reasoning_mechanisms_of_llms.md)**

:   This work compresses a Chain-of-Thought (CoT) reasoning process into a "CoT Vector" that can be directly added to hidden states. This approach enhances LLM multi-step reasoning with near-zero overhead (comparable to LoRA but with 3 orders of magnitude fewer trainable parameters) and serves as a probe revealing an internal "Perception—Reasoning—Expression" three-stage organization of LLM reasoning.

**[Cross-Modal Redundancy and the Geometry of Vision-Language Embeddings](cross-modal_redundancy_and_the_geometry_of_vision-language_embeddings.md)**

:   The authors propose the Iso-Energy hypothesis (stating that truly cross-modal shared concepts should have the same average activation energy across different modalities) and design Aligned SAE as an analytical tool. Their work reveals a geometric structure in VLM embedding space where bimodal atoms carry cross-modal alignment signals, while unimodal atoms fully account for the modality gap.

**[Debugging Concept Bottleneck Models through Removal and Retraining](debugging_concept_bottleneck_models_through_removal_and_retraining.md)**

:   To address the issue of Concept Bottleneck Models (CBMs) learning spurious concepts and systematic misalignment with expert reasoning, this paper proposes a "Removal + Retraining" two-step debugging framework. It introduces CBDebug, which converts expert concept-level feedback into sample-level auxiliary labels and uses permutation weighting and targeted augmentation to eliminate model dependence on spurious concepts. On benchmarks with known spurious correlations such as Waterbirds and MetaShift, it improves worst-group accuracy by up to 26%.

**[Decomposing LLM Computation with Jets](decomposing_llm_computation_with_jets.md)**

:   This paper proposes **JET EXPANSIONS**—using "jet operators" (functional versions of truncated Taylor expansions) to rewrite the recursive residual computation of Transformers into a set of explicit "input $\to$ output paths" plus a nonlinear remainder. This training-free and data-free approach "slices" entangled LLM computations for modular inspection, proves to unify and generalize the Logit Lens, and extracts n-gram tables directly from weights to diagnose fine-tuning and toxicity.

**[Decomposing Representation Space into Interpretable Subspaces with Unsupervised Learning](decomposing_representation_space_into_interpretable_subspaces_with_unsupervised_.md)**

:   The authors propose NDM (Neighbor Distance Minimization), an unsupervised method to discover interpretable non-basis-aligned subspaces in neural network representation spaces by minimizing within-subspace neighbor distances. It achieves an average Gini index of 0.71 (high information concentration) on GPT-2 and identifies segregated subspaces for parametric knowledge and in-context knowledge routing on Qwen2.5-1.5B.

**[Decomposition of Concept-Level Rules in Visual Scenes](decomposition_of_concept-level_rules_in_visual_scenes.md)**

:   This paper proposes the Concept-Rule Decomposition (CRD) framework, which utilizes pre-trained Large Vision-Language Models (LVLMs) as data-driven priors to automatically extract a set of "concepts" (e.g., color, object category) and the "rules" characterizing their spatial variations. Through a Metropolis-Hastings sampling process with an LVLM-guided proposal distribution, the framework iteratively selects a parsimonious set of concepts that best explain the input. CRD achieves improved accuracy and provides interpretable concept-rule decompositions across meta-attribute extraction, abstract visual reasoning (RAVEN/I-RAVEN), and spatial reasoning (SpatialEval).

**[Decoupling Dynamical Richness from Representation Learning: Towards Practical Measurement](decoupling_dynamical_richness_from_representation_learning_towards_practical_mea.md)**

:   The authors propose $\mathcal{D}_{LR}$, a computationally efficient and performance-independent metric for dynamical richness. It measures rich/lazy training dynamics by comparing activations before and after the final layer and demonstrates that neural collapse is a special case of this metric.

**[Decoupling Positional and Symbolic Attention in Transformers](decoupling_positional_and_symbolic_attention_in_transformers.md)**

:   This paper provides a rigorous mathematical definition for whether an attention head operates "positionally" or "symbolically," proves that the two are mutually exclusive (unless attention degrades into a uniform distribution), and designs a scoring metric based on permutation sensitivity. It reveals that high frequencies in RoPE correspond to positional behavior while low frequencies correspond to symbolic behavior. Finally, using controlled synthetic tasks, it demonstrates that "restricting the frequency bands accessible to a head can causally control the model's performance on positional versus symbolic tasks."

**[Diagnosing Generalization Failures from Representational Geometry Markers](diagnosing_generalization_failures_from_representational_geometry_markers.md)**

:   Drawing from the top-down perspective of medical "biomarkers," this paper uses **geometric quantities of object manifolds measured only on In-Distribution (ID) data** (effective dimension $D_\text{eff}$ and utilization $\Psi_\text{eff}$) as prognostic markers. It predicts model failure Out-of-Distribution (OOD) without any OOD information and selects pre-trained weights with superior transferability.

**[Discovering Alternative Solutions Beyond the Simplicity Bias in Recurrent Neural Networks](discovering_alternative_solutions_beyond_the_simplicity_bias_in_recurrent_neural.md)**

:   To address the issue where task-trained RNNs repeatedly collapse into a single "simplest" dynamical solution, this paper proposes Iterative Neural Similarity Decoupling (INSD). By online penalizing the linear predictability of new RNNs relative to existing solutions, the method uncovers novel classes of solutions that rely on dynamically evolving subspaces rather than fixed-point attractors, occasionally outperforming standard solutions under difficult or out-of-distribution (OOD) conditions.

**[Dissecting Representation Misalignment in Contrastive Learning via Influence Function](dissecting_representation_misalignment_in_contrastive_learning_via_influence_fun.md)**

:   Addressing the issue that classical influence functions are designed only for pointwise loss and cannot be directly applied to contrastive loss, this paper derives ECIF, an extended influence function specifically for contrastive learning. By analytically expressing the dual influence of a sample as both a "positive sample" and a "negative sample" in closed-form, it enables evaluating the contribution of each image-text pair in CLIP-like models without retraining, facilitating misalignment detection and misprediction tracing.

**[DIVERSE: Disagreement-Inducing Vector Evolution for Rashomon Set Exploration](diverse_disagreement-inducing_vector_evolution_for_rashomon_set_exploration.md)**

:   A frozen FiLM modulation layer is attached to a pre-trained network. CMA-ES is then used to perform a gradient-free search in a low-dimensional latent space for variants that are as accurate as the reference model but exhibit different predictive behaviors. This enables the systematic exploration of the Rashomon set of deep networks without retraining.

**[Does Higher Interpretability Imply Better Utility? A Pairwise Analysis on Sparse Autoencoders](does_higher_interpretability_imply_better_utility_a_pairwise_analysis_on_sparse_.md)**

:   The authors trained 90 SAEs for systematic comparison and found only a weak positive correlation between "more interpretable features" and "better steering utility" ($\tau_b \approx 0.30$). They proposed the **$\Delta$Token Confidence** feature selection criterion, which improves steering scores by 52.52%. On the selected high-efficiency features, the correlation between interpretability and steering utility completely disappeared or even became negative.

**[Domain Expansion: A Latent Space Construction Framework for Multi-Task Learning](domain_expansion_a_latent_space_construction_framework_for_multi-task_learning.md)**

:   The Domain Expansion framework is proposed to reconstruct the latent space into mutually orthogonal subspaces via Orthogonal Pooling. This structurally prevents gradient conflicts and representation collapse in multi-objective training, enabling interpretable and composable concept algebra.

**[Dynamic Reflections: Probing Video Representations with Text Alignment](dynamic_reflections_probing_video_representations_with_text_alignment.md)**

:   Ours provides the first expansion of the Platonic Representation Hypothesis (PRH) from static images to the spatiotemporal video-text domain. Through a systematic evaluation of 121 vision and language models, it reveals that increasing the number of frames and descriptions at test-time can nearly double alignment scores, and proposes a saturated scaling law with $R^2 > 0.98$ to quantify this behavior.

**[Dynamic Weight Grafting: Localizing Finetuned Factual Knowledge in Transformers](dynamic_weight_grafting_localizing_finetuned_factual_knowledge_in_transformers.md)**

:   This paper proposes Dynamic Weight Grafting to locate the retrieval mechanisms of fine-tuned factual knowledge in LLMs by temporarily replacing weights based on token position and Transformer components during generation. It finds that new knowledge is primarily retrieved via two paths: enrichment at the entity position and recall at the final token.

**[Dyslexify: A Mechanistic Defense Against Typographic Attacks in CLIP](dyslexify_a_mechanistic_defense_against_typographic_attacks_in_clip.md)**

:   The authors discover a small group of attention heads in the latter half of the CLIP vision encoder specialized in "reading text" in images, which inject typographic information into the `cls` token to cause typographic attacks. Dyslexify requires no gradient training; by simply zeroing out the contributions of these heads to the `cls` token (circuit ablation), it improves robustness on ImageNet-100-typo by up to 22.06% with <1% drop in standard accuracy.

**[Emergence of Superposition: Unveiling the Training Dynamics of Chain of Continuous Thought](emergence_of_superposition_unveiling_the_training_dynamics_of_chain_of_continuou.md)**

:   This work provides a theoretical analysis of the training dynamics of a two-layer Transformer using continuous Chain-of-Thought (Coconut) on directed graph reachability problems. It reveals how the "superposition" mechanism naturally emerges: the index-matching logit grows initially but remains bounded, thereby achieving a balance between exploration and exploitation.

**[Emergent Discrete Controller Modules for Symbolic Planning in Transformers](emergent_discrete_controller_modules_for_symbolic_planning_in_transformers.md)**

:   By embedding a discrete controller selected via Gumbel-Softmax within Transformer blocks, the model explicitly executes program primitives such as `ASSIGN/ADD/COMPARE/BRANCH` while maintaining register states. This achieves provable control flow expressivity, robust length extrapolation, and human-readable execution traces at a cost of only approximately 5–7% FLOPs.

**[Emotions Where Art Thou: Understanding and Characterizing the Emotional Latent Space of Large Language Models](emotions_where_art_thou_understanding_and_characterizing_the_emotional_latent_sp.md)**

:   This paper systematically characterizes the "emotional latent space" within LLM hidden states using SVD subspaces, geometric alignment, neuronal selectivity (ML-AURA), and learned steering modules. It identifies emotion as a directional, cross-layer, and cross-lingual low-dimensional manifold (generalizing across 8 datasets and 5 languages) that can be precisely manipulated while preserving semantics.

**[EnsembleSHAP: Faithful and Certifiably Robust Attribution for Random Subspace Method](ensembleshap_faithful_and_certifiably_robust_attribution_for_random_subspace_met.md)**

:   This paper proposes EnsembleSHAP, a feature attribution method specifically designed for the random subspace method. It directly reuses the sub-sampled prediction results already calculated by the ensemble model to provide Shapley-style feature importance with near-zero additional overhead. Furthermore, it provides the first provable robustness guarantee against "explanation-preservation attacks."

**[Escaping Low-Rank Traps: Interpretable Visual Concept Learning via Implicit Vector Quantization](escaping_low-rank_traps_interpretable_visual_concept_learning_via_implicit_vecto.md)**

:   To address the "representation collapse" problem where patch features degrade into a low-rank subspace and destroy visual-concept many-to-many alignment during Concept Bottleneck Model (CBM) training, this paper proposes **Implicit Vector Quantization (IVQ)**, which treats the vector quantization objective as a regularizer rather than a hard bottleneck. Combined with **Magnet Attention** to aggregate high-rank patch features into concept prototypes, the method achieves SOTA accuracy and superior interpretability consistency across 8 medical and 5 general benchmarks.

**[Estimating Dimensionality of Neural Representations from Finite Samples](estimating_dimensionality_of_neural_representations_from_finite_samples.md)**

:   Addressing the long-standing issue where the Participation Ratio (PR), a global dimensionality metric, is severely biased under finite samples, this paper derives an unbiased estimator $\gamma_{\text{both}}$ that simultaneously debiases for **row sampling, column sampling, and noise**. This allows dimensionality estimates to remain nearly invariant as the number of samples changes and extends to sparse matrices and local dimensionality.

**[Evaluating SAE Interpretability Without Generating Explanations](evaluating_sae_interpretability_without_generating_explanations.md)**

:   This paper proposes two evaluation methods for Sparse Autoencoder (SAE) interpretability—intruder detection and example embedding scoring—that **do not require generating natural language explanations**. By basing evaluation directly on latent activation examples, the study verifies that LLM evaluation is highly correlated with human judgment through manual annotation.

**[Evidence for Limited Metacognition in LLMs](evidence_for_limited_metacognition_in_llms.md)**

:   The authors borrow the "behavior-only, no self-report" metacognition measurement paradigm from ethology to design two experimental frameworks: the Delegate Game (testing "knowing whether one knows") and the Second Chance Game (testing "knowing what one will answer"). They demonstrate that frontier LLMs since 2024 possess **limited, context-dependent, and non-human-like** metacognitive abilities—they can perceive and utilize internal confidence signals, though these are used weakly and inconsistently.

**[Evolution of Concepts in Language Model Pre-Training](evolution_of_concepts_in_language_model_pre-training.md)**

:   This work applies crosscoders (cross-snapshot sparse dictionary learning) for the first time to track the emergence and evolution of features during language model pre-training. It discovers a "statistical learning → feature learning" phase transition and causally links micro-feature evolution with macro downstream task metrics through attribution analysis.

**[Explainable K-means Neural Networks for Multi-view Clustering](explainable_k_-means_neural_networks_for_multi-view_clustering.md)**

:   Decomposes multi-view clustering into three layer optimization sub-problems: "linear clustering → non-linear clustering → multi-view fusion." Each layer is implemented by K-means / Kernel K-means objectives, assembled into an EKNN network where the **function of each layer is explainable**, thereby achieving a balance across effectiveness, efficiency, completeness, and consistency.

**[Explainable Mixture Models through Differentiable Rule Learning](explainable_mixture_models_through_differentiable_rule_learning.md)**

:   Each component of a mixture model is bound to a "conjunctive rule readable on descriptive features." These rules, along with mixture weights, are learned via differentiable rule learning and gradient descent. This approach accurately models multimodal distributions like GMM while directly identifying "under what conditions or for which population each peak occurs."

**[Exploring Interpretability for Visual Prompt Tuning with Cross-layer Concepts](exploring_interpretability_for_visual_prompt_tuning_with_cross-layer_concepts.md)**

:   Proposed IVPT (Interpretable Visual Prompt Tuning), which links abstract visual prompts to human-understandable semantic regions through cross-layer class-agnostic concept prototypes. While maintaining the advantages of parameter-efficient fine-tuning, it realizes visual prompt interpretability for the first time, simultaneously improving explanation consistency (+8.4%) and accuracy on fine-grained classification benchmarks like CUB-200.

**[ExPO-HM: Learning to Explain-then-Detect for Hateful Meme Detection](expo-hm_learning_to_explain-then-detect_for_hateful_meme_detection.md)**

:   ExPO-HM is proposed, which is inspired by the human auditor training process. By combining Policy Manual SFT (SFT-PM) warm-up, GRPO Curriculum Learning (GRPO-CL), and Conditional Decision Entropy (CDE) rewards, this work represents the first Explain-then-Detect hateful meme detection system to comprehensively outperform direct detection baselines across binary classification, fine-grained classification, and reasoning quality, achieving F1 gains of up to 15-17%.

**[f-INE: A Hypothesis Testing Framework for Estimating Influence under Training Randomness](f-ine_a_hypothesis_testing_framework_for_estimating_influence_under_training_ran.md)**

:   The study redefines "how important a sample is" as "whether the change in loss after its removal is statistically significantly different from training randomness." By leveraging the hypothesis testing framework of f-differential privacy, the authors propose f-influence and the f-INE algorithm, which enables estimation via a single training run. This ensures that influence scores remain consistent across different random seeds and allows for reliable detection of poisoned samples in Llama-3.1-8B.

**[Faithfulness Under the Distribution: A New Look at Attribution Evaluation](faithfulness_under_the_distribution_a_new_look_at_attribution_evaluation.md)**

:   Existing attribution evaluation metrics (Insertion/Deletion, Infidelity) rely on "zeroing out/masking" to remove features, which pushes samples out of the data distribution and introduces artifactual information. This paper proposes **FUD**, which utilizes score-based diffusion models to reconstruct masked regions back into "in-distribution" samples on the data manifold, providing a more credible assessment of attribution faithfulness.

**[FAME: Formal Abstract Minimal Explanation for Neural Networks](fame_formal_abstract_minimal_explanation_for_neural_networks.md)**

:   FAME establishes "Abductive Explanations" on the foundation of abstract interpretation. By using LiRPA bounds to prove and prune a batch of irrelevant features simultaneously, it overcomes the bottleneck of traditional formal XAI’s dependence on "feature traversal order." This scales provable minimal explanations to large networks like ResNet for the first time.

**[Feature Segregation by Signed Weights in Artificial Vision Systems and Biological Models](feature_segregation_by_signed_weights_in_artificial_vision_systems_and_biologica.md)**

:   This study discovers that ImageNet-trained CNNs spontaneously assign "object/foreground" features to positive weights and "background/contextual texture" features to negative weights, even without enforcing the biological Dale's Law. This homologous "feature segregation by sign" strategy is further validated in neural models of the macaque ventral visual cortex (V1/V4/IT).

**[Features Emerge as Discrete States: The First Application of SAEs to 3D Representations](features_emerge_as_discrete_states_the_first_application_of_saes_to_3d_represent.md)**

:   The first application of Sparse Autoencoders (SAEs) to the latent space of 3D reconstruction VAEs reveals that 3D models encode continuous positions into "discrete states + phase transitions." A proposed framework based on gradient dynamics provides a unified explanation for positional encoding preferences, S-shaped ablation-loss curves, and the bimodal distribution of phase transition points.

**[Flow-Disentangled Feature Importance](flow-disentangled_feature_importance.md)**

:   FDFI utilizes flow matching to learn an invertible mapping that disentangles correlated features into independent latent variables. It calculates the importance of each direction in the latent space and "returns" the scores to original features using squared Jacobian weights. This generalizes DFI, which is restricted to $\ell_2$ loss, to any differentiable loss (including classification) and provides semiparametrically efficient estimation with valid confidence intervals and hypothesis testing.

**[Formal Mechanistic Interpretability: Automated Circuit Discovery with Provable Guarantees](formal_mechanistic_interpretability_automated_circuit_discovery_with_provable_gu.md)**

:   This work introduces neural network (NN) verification into mechanistic interpretability, proposing the first circuit discovery framework with provable guarantees. It ensures circuit faithfulness across continuous input domains (input robustness) and continuous patching domains (patching robustness), and formalizes a hierarchy of four minimality levels (quasi → local → subset → cardinal), unifying these guarantees through monotonicity theory.

**[Frequency Bands in RoPE: Base Frequency and Context Length Shape the Interpolation–Extrapolation Trade-off](frequency_bands_in_rope_base_frequency_and_context_length_shape_the_interpolatio.md)**

:   This paper reveals the existence of "frequency bands" in RoPE, jointly determined by the base frequency $\theta$ and the training length $L_{train}$. These bands are formed early in pre-training and inherited during position interpolation. The study proves that low-frequency dimensions below the band are nearly equivalent to NoPE, overturning the mainstream intuition that "increasing $\theta$ always benefits long contexts"—instead, increasing $\theta$ merely redistributes energy to improve interpolation at the cost of extrapolation.

**[Fresh in Memory: Training-order Recency is Linearly Encoded in Language Model Activations](fresh_in_memory_training-order_recency_is_linearly_encoded_in_language_model_act.md)**

:   By sequentially fine-tuning Llama-3.2-1B on six disjoint entity datasets, the authors discovered a **linear direction in the language model's activation space that sorts the centroids of data from different stages according to their training order**. This suggests that models implicitly "timestamp" learned information. This temporal signal can be extracted by linear probes (>90% accuracy in distinguishing early vs. late entities), explicitly reported by the model itself (~80% accuracy), and remains resilient even after 30 epochs of shuffled mixed-data training.

**[From Concepts to Components: Concept-Agnostic Attention Module Discovery in Transformers](from_concepts_to_components_concept-agnostic_attention_module_discovery_in_trans.md)**

:   By abstracting any complex "concept" into a vector and using its cosine similarity with the output of each attention head to identify TopK heads as a "concept module," the authors demonstrate that scaling the output intensity of these modules with a single scalar can localize and amplify/suppress concepts like safety, reasoning, multilingualism, and image recognition in language and vision Transformers.

**[From Data Statistics to Feature Geometry: How Correlations Shape Superposition](from_data_statistics_to_feature_geometry_how_correlations_shape_superposition.md)**

:   This paper argues that the classic "superposition = interference = noise" paradigm is incomplete for real-world data. When features are correlated, interference can be **constructive**: models arrange features based on co-activation patterns, allowing interference between active features to mutually reinforce signals. This enables reconstruction with smaller weight norms and lower rank, naturally explaining geometric structures like semantic clustering and circular arrangements of months observed in real language models.

**[From Tokens to Thoughts: How LLMs and Humans Trade Compression for Meaning](from_tokens_to_thoughts_how_llms_and_humans_trade_compression_for_meaning.md)**

:   Using an Information Bottleneck/Rate-Distortion framework to measure embeddings from 40+ LLMs and classic human taxonomic cognitive data on a unified "compression ↔ semantics" scale, this work reveals that LLMs achieve more "information-theoretically optimal" aggressive compression than humans, but at the cost of fine-grained semantics (typicality structures). The seemingly "inefficient" human conceptual organization instead serves as a source of adaptive flexibility.

**[Gauge-invariant Representation Holonomy](gauge-invariant_representation_holonomy.md)**

:   The authors define "the cumulative rotation of features along a closed input loop" as **representation holonomy**—a gauge-invariant scalar used to characterize "path-dependent geometry" invisible to pointwise similarities like CKA/SVCCA, and link it to the adversarial/corruption robustness of models.

**[GAVEL: Towards Rule-Based Safety through Activation Monitoring](gavel_towards_rule-based_safety_through_activation_monitoring.md)**

:   Drawing on the concept of Snort/YARA rule-sets in cybersecurity, this paper proposes decomposing internal LLM activations into 23 fine-grained "Cognitive Elements" (CEs), which are then combined via Boolean logic into auditable safety rules. Implemented on Mistral-7B with <1% inference overhead, it achieves an average AUC of 0.99 and an FPR of 0.004 across 9 misuse scenarios, naturally supporting cross-lingual and cross-model migration.

**[GenCtrl — A Formal Controllability Toolkit for Generative Models](genctrl_--_a_formal_controllability_toolkit_for_generative_models.md)**

:   This paper models the "user-generative model dialogue" as a discrete-time nonlinear control system. It proposes a Monte Carlo algorithm to estimate the **reachable set** and **controllable set** of the model, providing distribution-agnostic PAC (probably-approximately-correct) error bounds that require only an assumption of bounded outputs. This allows for the first formal answer to whether a generative model is controllable. Experiments reveal that the controllability of modern LLMs and text-to-image models is surprisingly fragile and highly dependent on the task setting.

**[GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning](gepa_reflective_prompt_evolution_can_outperform_reinforcement_learning.md)**

:   The study proposes GEPA (Genetic-Pareto), a prompt optimizer that diagnoses issues and iteratively optimizes prompts through natural language reflection based on a small number of execution trajectories. It outperforms GRPO by an average of 6% (up to 20%) across six tasks while using only 1/35 of the sampling volume.

**[Grokking in LLM Pretraining? Monitor Memorization-to-Generalization without Test](grokking_in_llm_pretraining_monitor_memorization-to-generalization_without_test.md)**

:   This paper validates the grokking phenomenon—characterized by asynchronous memorization across different data groups and delayed generalization—for the first time in near-single-pass pretraining of actual-scale LLMs (7B MoE). By analyzing the evolution of MoE routing pathways (from instance-specific to structured/shared), the authors propose two zero-cost metrics to monitor generalization progress without the need for instruction tuning or benchmark evaluation.

**[Hedonic Neurons: A Mechanistic Mapping of Latent Coalitions in Transformer MLPs](hedonic_neurons_a_mechanistic_mapping_of_latent_coalitions_in_transformer_mlps.md)**

:   By treating neurons in Transformer MLPs as "rational players" in a cooperative game, this work employs hedonic games and the PAC-Top-Cover algorithm to identify neuron coalitions where "joint ablation effects superimpose non-linearly." This reveals how LoRA fine-tuning encodes task features within synergistic neuron groups.

**[Hessian-Enhanced Token Attribution (HETA): Interpreting Autoregressive LLMs](hessian-enhanced_token_attribution_heta_interpreting_autoregressive_llms.md)**

:   HETA integrates "causal semantic flow gating + Hessian second-order curvature sensitivity + KL information loss" into a unified token attribution score. Specially designed for decoder-only autoregressive LLMs, it significantly outperforms existing methods in faithfulness and robustness to decoding hyperparameters and syntactic paraphrasing.

**[Hidden Breakthroughs in Language Model Training](hidden_breakthroughs_in_language_model_training.md)**

:   This paper proposes POLCA (Projection Oriented Loss Change Allocation)—a method to decompose single-sample loss changes along any orthogonal basis of the low-rank training subspace. It reveals numerous hidden conceptual breakthroughs from seemingly smooth training loss curves, shifting training interpretability from "pre-defining skills before observation" to "decomposition followed by automatic skill discovery."

**[How Do Transformers Learn to Associate Tokens: Gradient Leading Terms Bring Mechanistic Understanding](how_do_transformers_learn_to_associate_tokens_gradient_leading_terms_bring_mecha.md)**

:   Through leading-term approximation analysis of training gradients, this study derives closed-form expressions for Transformer weights during early training. These weights are decomposable into a simple combination of three basis functions (bigram, token-interchangeability, context mapping), revealing how Transformers learn semantic associations like "bird"↔"flew" from natural language data. Theoretical predictions align closely with weights learned in real LLMs.

**[How Stable is the Next Token? A Geometric View of LLM Prediction Stability](how_stable_is_the_next_token_a_geometric_view_of_llm_prediction_stability.md)**

:   This paper proposes **Token Constraint Bound ($\delta$TCB)**—a geometric metric that quantifies how much the internal hidden state $h$ of an LLM can be perturbed before the next-token prediction changes significantly. It demonstrates that this bound is determined by the "probability-weighted dispersion" of the output embedding space relative to the current prediction distribution, revealing local prediction robustness invisible to perplexity or accuracy.

**[How Transformers Learn Causal Structures In-Context: Explainable Mechanism Meets Theoretical Guarantee](how_transformers_learn_causal_structures_in-context_explainable_mechanism_meets_.md)**

:   This paper proves and empirically demonstrates that a two-layer Transformer with relative position encoding (RPE) can explicitly implement Bayesian Model Averaging (BMA)—the statistically optimal algorithm—in-context to infer the "parent" causal structure of each token. It further provides identifiability and training dynamics guarantees using information theory (DPI / mutual information).

**[Hyper-SET: Designing Transformers via Hyperspherical Energy Minimization](hyper-set_designing_transformers_via_hyperspherical_energy_minimization.md)**

:   Transformer layers are reinterpreted as "Maximum Likelihood Estimation (MLE) of tokens on a hypersphere," decomposed into two complementary objectives: **distribution uniformity in low-dimensional subspaces** and **semantic alignment in high-dimensional space**. These are quantified using two extended Hopfield energy functions for iterative energy minimization. Consequently, symmetric attention, feed-forward layers, RMSNorm, and residual connections emerge naturally, resulting in HYPER-SET—a parameter-shared, interpretable recurrent-depth model with performance approaching the original Transformer.

**[I Predict Therefore I Am: Is Next Token Prediction Enough to Learn Human-Interpretable Concepts from Data?](i_predict_therefore_i_am_is_next_token_prediction_enough_to_learn_human-interpre.md)**

:   This paper constructs a text generation model that formalizes "human-interpretable concepts" as discrete latent variables and rigorously proves that LLM representations trained solely via next-token prediction are, under mild conditions, approximately equivalent to a linear transformation of the posterior log-probabilities $\log p(c\mid x)$ of these latent concepts. This provides a unified theoretical foundation for the linear representation hypothesis, steering vectors, linear probing, and Sparse Autoencoder (SAE) evaluation.

**[Inducing Dyslexia in Vision Language Models](inducing_dyslexia_in_vision_language_models.md)**

:   By "functionally localizing" units with visual word form selectivity in Vision-Language Models and ablating them, the authors reproduce core features of human dyslexia (selective reading deficits + phonological-leaning impairments) without damaging general visual and reasoning abilities, proving these units predict real human VWFA fMRI responses.

**[Inferring the Invisible: Neuro-Symbolic Rule Discovery for Missing Value Imputation](inferring_the_invisible_neuro-symbolic_rule_discovery_for_missing_value_imputati.md)**

:   Treat "missing table entries" as latent predicates to be inferred. A differentiable forward chaining reasoning engine allows **rule induction** and **missing value imputation** to serve as mutual evidence and reinforce each other, completing the data while learning human-readable logic rules.

**[Influence Dynamics and Stagewise Data Attribution](influence_dynamics_and_stagewise_data_attribution.md)**

:   This paper uses Singular Learning Theory (SLT) to upgrade "training data attribution" from a static perspective to a **stagewise perspective**: it proves that the influence of one sample on another is not fixed but undergoes **sign flips** and **peaks** at phase transition points of model development, verifying this prediction using Bayesian Influence Functions (BIF) on both toy models and real language models.

**[Information Shapes Koopman Representation](information_shapes_koopman_representation.md)**

:   This paper re-examines the finite-dimensional representation learning problem of the Koopman operator from the perspective of the Information Bottleneck (IB). The Koopman operator lifts nonlinear dynamical systems into infinite-dimensional linear evolutions, but practical applications require approximation in finite-dimensional subspaces, leading to a fundamental contradiction between "simplicity and expressivity." The authors prove that: (1) latent mutual information controls the upper bound of prediction error, but excessive maximization leads to mode collapse; (2) von Neumann entropy prevents collapse and maintains effective dimensionality. Based on this, an information-theoretic Lagrangian formulation is proposed to unify the balancing of three major objectives: temporal coherence, predictive sufficiency, and structural consistency, deriving computable loss functions. The method outperforms existing Koopman approaches in physical simulation, visual control, and graph-structured dynamics tasks.

**[InputDSA: Demixing, then comparing recurrent and externally driven dynamics](inputdsa_demixing_then_comparing_recurrent_and_externally_driven_dynamics.md)**

:   The authors extend the Dynamical Similarity Analysis (DSA) method from autonomous to non-autonomous systems. By utilizing Subspace Dynamic Mode Decomposition with control (SubspaceDMDc) to simultaneously estimate the intrinsic operator $A$ and the input operator $B$, InputDSA enables the separate comparison of "intrinsic dynamics" and "input-driven dynamics" under partial observation, noise, or when only proxy inputs are available.

**[Internal Planning in Language Models: Characterizing Horizon and Branch Awareness](internal_planning_in_language_models_characterizing_horizon_and_branch_awareness.md)**

:   Proposed an information-theoretic framework based on VQ-VAE to analyze the internal planning behavior of language models. It was found that planning horizons are task-dependent, models implicitly retain information about unselected correct paths, and next-token decisions primarily rely on recent computations.

**[Interpretable 3D Neural Object Volumes for Robust Conceptual Reasoning](interpretable_3d_neural_object_volumes_for_robust_conceptual_reasoning.md)**

:   CAVE compresses thousands of dense 3D Gaussian features from NOVUM into approximately 20 sparse concepts per class via dictionary learning. This yields an image classifier that is both OOD robust and "faithful-by-design" in its interpretability, while proposing a 3D-C metric that measures spatial consistency of concepts across views and degradations without the need for part annotations.

**[Is This Just Fantasy? Language Model Representations Reflect Human Judgments of Event Plausibility](is_this_just_fantasy_language_model_representations_reflect_human_judgments_of_e.md)**

:   The authors utilize Contrastive Activation Addition (CAA) to extract **modal difference vectors** from the hidden states of various LMs to distinguish between modal categories such as "probable / impossible / inconceivable." This demonstrates that LM internal judgments of sentence modality are significantly more reliable than previously suggested. These vectors emerge in a coarse-to-fine order across training, layers, and scale, and can effectively model fine-grained human categorical judgment behaviors.

**[Joint Distribution–Informed Shapley Values for Sparse Counterfactual Explanations](joint_distributioninformed_shapley_values_for_sparse_counterfactual_explanations.md)**

:   The COLA framework is proposed: it uses Optimal Transport (OT) to find a coupling matrix between factual and counterfactual sets, which then drives Shapley attribution (p-SHAP) to refine any off-the-shelf counterfactual explanations. This approach maintains the target flip effect while modifying only 26–45% of the original features.

**[Language Models are Injective and Hence Invertible](language_models_are_injective_and_hence_invertible.md)**

:   This paper mathematically proves that decoder-only Transformer language models are almost surely **injective** (different prompts produce different last-token representations). Based on this, the authors propose the SIPIT algorithm, which can **precisely** reconstruct input text from hidden states in linear time.

**[Language Models Use Lookbacks to Track Beliefs](language_models_use_lookbacks_to_track_beliefs.md)**

:   Using causal mediation and causal abstraction, this paper reverse-engineers a universal "lookback mechanism" that large language models (LLMs) rely on to track character beliefs (Theory of Mind). The model copies reference information into "pointers" and "addresses" at different tokens, retrieving "payloads" via QK-attention lookbacks to achieve character-object-state binding, belief retrieval, and visibility updates.

**[Latent Concept Disentanglement in Transformer-based Language Models](latent_concept_disentanglement_in_transformer-based_language_models.md)**

:   This paper uses mechanistic interpretability methods to demonstrate that transformers explicitly disentangle latent "concepts" within demonstrations during in-context learning. In discrete world-knowledge tasks, a small cluster of attention heads first parses a hidden "bridge entity" before composing the answer. In continuous numerical tasks, latent parameters are compressed onto low-dimensional smooth manifolds that are susceptible to linear interpolation and causal intervention.

**[Latent Planning Emerges with Scale](latent_planning_emerges_with_scale.md)**

:   The authors provide a **causally verifiable** definition of "LLM latent planning" (forward planning + backward planning) and conduct experiments on the Qwen-3 (0.6B–14B) family using transcoder feature circuits. They find that planning capability emerges with model scale: simple grammatical consistency tasks (a/an) only succeed stably at 14B, while in rhyme couplet tasks, models exhibit forward planning but almost no backward planning.

**[Latent Thinking Optimization: Your Latent Reasoning Language Model Secretly Encodes Reward Signals in Its Latent Thoughts](latent_thinking_optimization_your_latent_reasoning_language_model_secretly_encod.md)**

:   This paper systematically dissects the "latent thinking" process of the latent reasoning language model Huginn-3.5B. It discovers that correct and incorrect latent thinking trajectories are highly separable in the latent space. Consequently, the authors train a lightweight classifier as a "Latent Reward Model (LRM)" and propose Latent Thinking Optimization (LTO)—a probabilistic algorithm using acceptance-rejection sampling to select high-reward trajectories in the latent space, directly bringing reward modeling and test-time scaling into the latent domain.

**[LatentQA: Teaching LLMs to Decode Activations Into Natural Language](latentqa_teaching_llms_to_decode_activations_into_natural_language.md)**

:   This paper reframes "understanding model activations" as an open-ended question-answering task, LatentQA. Given an activation and a natural language question, a fine-tuned decoder LLM answers directly in natural language. This enables both "reading" activations (monitoring) and "writing" to activations (steering) using gradients backpropagated from natural language-described losses.

**[Learning for Highly Faithful Explainability](learning_for_highly_faithful_explainability.md)**

:   This paper proposes DeepFaith: a self-supervised objective derived from ten faithfulness metrics that makes no assumptions about the target model or task. It aggregates multiple prior explanation methods into high-quality supervision signals through "deduplication + faithfulness filtering" and employs dynamic weighting for joint optimization. This trains an amortized explainer capable of generating more faithful explanations than all prior methods in a single forward pass.

**[Learning is Forgetting: LLM Training As Lossy Compression](learning_is_forgetting_llm_training_as_lossy_compression.md)**

:   LLM pre-training is interpreted as "lossy compression." By using Rate Distortion Theory and the Information Bottleneck (IB) principle, the study characterizes how models first expand and then compress representations during training. It demonstrates that "how close a model compresses to the theoretical optimum" and "what information remains after compression" directly predict downstream benchmark performance.

**[Learning Multimodal Dictionary Decompositions with Group-Sparse Autoencoders](learning_multimodal_dictionary_decompositions_with_group-sparse_autoencoders.md)**

:   Standard Sparse Autoencoders (SAEs) learned on aligned multimodal embeddings (like CLIP/CLAP) produce "split dictionaries"—where most concepts activate for only a single modality. This paper uses **cross-modal random masking + group-sparse regularization** to force paired samples to share a sparse support, learning truly multimodal concept dictionaries while reducing dead neurons and improving semanticity and cross-modal zero-shot performance.

**[Learning Nonlinear Causal Reductions to Explain Reinforcement Learning Policies](learning_nonlinear_causal_reductions_to_explain_reinforcement_learning_policies.md)**

:   The paper models the question "why an RL policy succeeds or fails" as a causal model reduction problem. By injecting random perturbations into actions as interventions, it learns a simplified causal model consisting of only two variables: "high-level cause $Z$ → high-level target $Y$." Using **Nonlinear Targeted Causal Reduction (nTCR)**, it distills state/action patterns that truly influence cumulative rewards, providing global, causal, and interpretable explanations of policy behavior.

**[Learning Pseudorandom Numbers with Transformers: Permuted Congruential Generators, Curricula, and Interpretability](learning_pseudorandom_numbers_with_transformers_permuted_congruential_generators.md)**

:   Transformers can crack NumPy's default random number generator, PCG, purely via in-context sequence data (surpassing classical attack assumptions). The required context length scales with the modulus as $\sqrt{m}$. Training on large moduli necessitates curriculum learning, and the embedding layer spontaneously clusters integers based on a "rotation-invariant zero-run structure."

**[Learning to Interpret Weight Differences in Language Models](learning_to_interpret_weight_differences_in_language_models.md)**

:   By training a LoRA adapter (DIT-adapter) using "synthetic, labeled weight differences," any fine-tuned language model can **describe in natural language how it was changed by fine-tuning**, thereby converting unreadable weight differences (weight diffs) into human-readable behavioral descriptions.

**[Learning to See Before Seeing: Demystifying LLM Visual Priors from Language Pre-training](learning_to_see_before_seeing_demystifying_llm_visual_priors_from_language_pre-t.md)**

:   Through 100+ controlled experiments (consuming 500,000 GPU hours), this study systematically dismantles why "text-only LLMs develop visual capabilities." It discovers that visual priors are separable into **reasoning priors** (derived from code/math/academic data, growing monotonically with proportion and universal across visual encoders) and **perception priors** (diffusely derived from broad corpora, more dependent on visual encoders and instruction tuning). Based on this, it provides a "reasoning-heavy, minimal visual description" pre-training data recipe, validated at a $1T$ token scale to produce stronger vision-aware LLMs.

**[Learning to Weight Parameters for Training Data Attribution](learning_to_weight_parameters_for_training_data_attribution.md)**

:   This paper points out that the "huge disparity in attribution quality across different parameter groups" in gradient attribution is ignored by existing methods. It proposes a unified framework that uses **self-supervision** to directly learn a set of parameter group weights $w$ from data. Without requiring annotations, it systematically improves the attribution precision of methods like TracIn / TRAK / EK-FAC and can decouple semantic dimensions such as subject, style, and background.

**[Linear Mechanisms for Spatiotemporal Reasoning in Vision Language Models](linear_mechanisms_for_spatiotemporal_reasoning_in_vision_language_models.md)**

:   This paper discovers that when VLMs perform spatial reasoning, they bind the location information of objects from the visual input as **linear "spatial ID" vectors** to the text activations of the corresponding object words. Reasoning is subsequently completed in the language space. Causal interventions prove that modifying only this spatial ID can systematically flip the model's judgment of "left/right" and "far/near." This mechanism is also extended to "temporal IDs" in video models.

**[LLMs are Single-threaded Reasoners: Demystifying the Working Mechanism of Soft Thinking](llms_are_single-threaded_reasoners_demystifying_the_working_mechanism_of_soft_th.md)**

:   Through a set of probing experiments, this paper reveals that "Soft Thinking" does not explore multiple reasoning paths in parallel as theoretically claimed—LLMs are actually "single-threaded reasoners," driven almost exclusively by the top-1 component in soft tokens, thereby falling into a greedy feedback loop. Consequently, the authors propose **Stochastic Soft Thinking**, which injects controllable randomness via Gumbel-Softmax to break the greedy trap, outperforming vanilla soft thinking and even discrete CoT across 8 reasoning benchmarks.

**[LLMs Process Lists With General Filter Heads](llms_process_lists_with_general_filter_heads.md)**

:   This paper discovers that when LLMs perform tasks like "selecting items from a list that satisfy a condition," a small set of mid-layer attention heads (filter heads) encode the "filtering predicate" as a compact, portable geometric direction in the query space, replicating the abstract computational primitive of the `filter` operation in functional programming.

**[Localizing Task Recognition and Task Learning in In-Context Learning via Attention Head Analysis](localizing_task_recognition_and_task_learning_in_in-context_learning_via_attenti.md)**

:   This paper proposes Task Subspace Logit Attribution (TSLA) to localize Task Recognition (TR) and Task Learning (TL) in in-context learning to different attention heads. Through correlation, ablation, input perturbation, task vector steering, and geometric analysis of hidden states, the authors demonstrate that TR heads are responsible for pulling states toward the task label subspace, while TL heads steering toward the correct labels within that subspace.

**[LORE: Jointly Learning the Intrinsic Dimensionality and Relative Similarity Structure from Ordinal Data](lore_jointly_learning_the_intrinsic_dimensionality_and_relative_similarity_struc.md)**

:   The authors propose LORE, the first framework to jointly learn embedding representations and intrinsic dimensionality from ordinal triplet comparisons. By replacing traditional pre-specified dimensionality strategies with a non-convex Schatten-p quasi-norm ($p<1$) regularization, solved via an Iterative Reweighted Nuclear Norm (IRNN) algorithm with guaranteed convergence to a stationary point, LORE significantly outperforms all baseline methods in dimensionality recovery across synthetic data, LLM-simulated perception experiments, and three crowdsourced datasets, while maintaining high triplet accuracy and semantic interpretability.

**[Low-Pass Filtering Improves Behavioral Alignment of Vision Models](low-pass_filtering_improves_behavioral_alignment_of_vision_models.md)**

:   The authors discovered that the highly human-like visual behavior of Imagen-like models, previously attributed to "generative objectives," actually stems primarily from an inconspicuous downsampling operation (equivalent to low-pass filtering). By simply applying Gaussian blur to input images at **test time**, standard discriminative CLIP models can achieve new SOTAs on the model-vs-human benchmark, halving the human-AI alignment gap.

**[MATA: A Trainable Hierarchical Automaton System for Multi-Agent Visual Reasoning](mata_a_trainable_hierarchical_automaton_system_for_multi-agent_visual_reasoning.md)**

:   The paper proposes MATA (Multi-Agent hierarchical Trainable Automaton), which models multi-agent visual reasoning as a hierarchical finite state automaton. Top-level state transitions are learned by a trainable hyper agent (an LLM-based state controller), while each individual agent employs a rule-based sub-automaton. Through shared memory, the system enables cooperation and competition, achieving SOTA on multiple visual reasoning benchmarks.

**[Medical Interpretability and Knowledge Maps of Large Language Models](medical_interpretability_and_knowledge_maps_of_large_language_models.md)**

:   The authors systematically scanned 5 open-source LLMs using four interpretability techniques (UMAP projection, weight gradient saliency, layer ablation, and activation patching) to construct "Medical Knowledge Maps." These maps localize age, symptoms, diseases, drugs, and dosages to specific model layers, revealing phenomena such as non-linear age manifolds and circular, non-monotonic representations of disease progression.

**[MICLIP: Learning to Interpret Representation in Vision Models](miclip_learning_to_interpret_representation_in_vision_models.md)**

:   MICLIP adapts the CLIP contrastive learning paradigm to "internal model representations," training a neuron encoder to project neuron/SAE features into the CLIP semantic space. This bypasses the old "activation-magnitude" assumption, providing a unified framework for both interpreting and precisely controlling internal mechanisms of vision models.

**[Missingness Bias Calibration in Feature Attribution Explanations](missingness_bias_calibration_in_feature_attribution_explanations.md)**

:   This paper proposes **MCal**: by fine-tuning a single affine transformation head (matrix scaling) only on the output logits of a frozen model, it cheaply and model-agnostically corrects "missingness bias" in feature attributions. The effectiveness rivals or even exceeds heavyweight solutions such as retraining or architectural modifications.

**[Mixing Mechanisms: How Language Models Retrieve Bound Entities In-Context](mixing_mechanisms_how_language_models_retrieve_bound_entities_in-context.md)**

:   This paper reveals that language models do not rely solely on the previously recognized **positional mechanism** to retrieve "bound entities" in-context. Instead, they employ a mixture of positional, lexical, and reflexive mechanisms. Based on this, the authors construct a position-weighted causal model that replicates the model's next-token distribution with 95% faithfulness and explains the "lost-in-the-middle" phenomenon in long contexts.

**[Mixture of Cognitive Reasoners: Modular Reasoning with Brain-Like Specialization](mixture_of_cognitive_reasoners_modular_reasoning_with_brain-like_specialization.md)**

:   Each layer of a pre-trained LLM is decomposed into four expert modules corresponding to the human brain's cognitive networks: "Language, Logic, Social, and World Knowledge." Using a three-stage curriculum training process, this brain-like functional specialization is "forced" out, resulting in MICRO—a modular language model that is interpretable, allows for behavioral regulation via expert routing during inference, and maintains reasoning performance.

**[Multi-ReduNet: Interpretable Class-Wise Decomposition of ReduNet](multi-redunet_interpretable_class-wise_decomposition_of_redunet.md)**

:   The global MCR² objective of ReduNet is theoretically decomposed into $K$ independent "class-wise subproblems." Combined with the Woodbury identity, the complexity of per-layer matrix inversion is reduced from $O(d^3)$ to $O(m_j^3)$. In high-dimensional undersampled scenarios ($m \ll d$), this approach achieves higher accuracy, approximately 2× training acceleration, and nearly an order of magnitude improvement in learning rate robustness while maintaining white-box interpretability.

**[Multiple Token Divergence: Measuring and Steering In-Context Computation Density](multiple_token_divergence_measuring_and_steering_in-context_computation_density.md)**

:   This paper proposes **Multiple Token Divergence (MTD)**—a training-free metric that measures the "computational effort" of a language model at each step via the KL divergence between the full model's output distribution and a shallow auxiliary prediction head's distribution. Based on this, a decoding method called **Divergence Steering** is derived to regulate the "computation density" of generated text.

**[Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences](narrow_finetuning_leaves_clearly_readable_traces_in_activation_differences.md)**

:   It is discovered that narrow finetuning leaves clearly readable traces in LLM activations: even on the first few tokens of unrelated text, the activation differences before and after finetuning encode semantic information about the finetuning goal. Using the Activation Difference Lens (ADL) method, an interpretability agent identifies finetuning goals with a success rate of 91%, more than 2x higher than black-box baselines.

**[Negative Pre-activations Differentiate Syntax](negative_pre-activations_differentiate_syntax.md)**

:   This paper discovers that in modern LLMs using smooth activations like GELU/SiLU, a small subset of "Wasserstein neurons" (approx. 1%) specifically utilize the **negative pre-activation region** to differentiate syntax. Zeroing out the negative pre-activations of only these 1% of neurons significantly impairs grammatical capabilities while causing minimal damage to other tasks, revealing that the long-neglected negative region is an active carrier for syntactic computation.

**[Neuron-Level Analysis of Cultural Understanding in Large Language Models](neuron-level_analysis_of_cultural_understanding_in_large_language_models.md)**

:   This paper proposes CULNIG—a neuron identification pipeline based on gradient attribution and dual-contrast filtering. It accurately locates "culture-general neurons" and "culture-specific neurons" in LLMs, finding that they constitute less than 1% of all neurons and are concentrated in shallow-to-middle MLP layers. Inhibiting them drops performance on cultural benchmarks by up to 30% while barely affecting general NLU.

**[NIMO: a Nonlinear Interpretable MOdel](nimo_a_nonlinear_interpretable_model.md)**

:   NIMO proposes a hybrid model $y = \sum_j x_j \beta_j (1 + g_{\mathbf{u}_j}(\mathbf{x}_{-j}))$. While maintaining the global interpretability of linear regression coefficients (via Mean Marginal Effects, MEM), it utilizes neural networks to provide instance-specific nonlinear corrections. The model uses a parameter elimination method to efficiently optimize linear coefficients and network parameters jointly.

**[Noise Stability of Transformer Models](noise_stability_of_transformer_models.md)**

:   Proposes noise stability as a superior metric over average sensitivity for measuring simplicity bias in Transformers, and designs a regularization method based on this that accelerates training by approximately 35% on synthetic tasks and 75% on language modeling.

**[On The Geometry and Topology of Representations: the Manifolds of Modular Addition](on_the_geometry_and_topology_of_representations_the_manifolds_of_modular_additio.md)**

:   This paper adopts a perspective of "viewing a whole cluster of neurons with the same frequency as a manifold," proving that various networks previously thought to have learned "completely different circuits (Clock vs. Pizza)" actually learn the **same class of torus/vector-addition disk manifolds** in the first layer. This is statistically validated across hundreds of networks using closed-form formulas and Topological Data Analysis (TDA), thereby repairing the "Universality Hypothesis" previously challenged by Zhong et al. (2023).

**[On the Limits of Sparse Autoencoders: A Theoretical Framework and Reweighted Remedy](on_the_limits_of_sparse_autoencoders_a_theoretical_framework_and_reweighted_reme.md)**

:   This paper derives the first closed-form optimal solution for Sparse Autoencoders (SAEs), theoretically proving that SAEs generally fail to fully recover true monosemantic features from superimposed polysemantic features (resulting in feature shrinking and vanishing), except under conditions of extreme sparsity. For general sparsity levels, the authors propose WSAE, which adaptively reweights dimensions based on their degree of polysemanticity, and provide a weighting principle verified to improve monosemanticity and interpretability in both synthetic data and real-world language/vision models.

**[On the Predictive Power of Representation Dispersion in Language Models](on_the_predictive_power_of_representation_dispersion_in_language_models.md)**

:   This paper discovers that the "spread" of language model hidden states (average pairwise cosine distance, termed representation dispersion) is strongly negatively correlated with perplexity—stronger models spread contexts further apart. This simple geometric metric is transformed into four zero-label practical tools: sample difficulty ranking, model selection, kNN-LM layer selection, and a push-away training loss that directly reduces perplexity.

**[Paradigm Shift of GNN Explainer from Label Space to Prototypical Representation Space](paradigm_shift_of_gnn_explainer_from_label_space_to_prototypical_representation_.md)**

:   To address the issue of insufficient utilization of structural information caused by long-term alignment in the "graph label space" by post-hoc instance-level GNN explainers, IDEA migrates explainer optimization from the label space to the "prototypical representation space" for the first time. It uses a hierarchical graph tokenizer to decouple explanatory substructures and aligns the prototypical assignment distributions of the input and explanatory subgraphs using Wasserstein distance. This improves ROC-AUC by an average of 4.45% and precision by 48.71%, while providing plug-and-play enhancement for various existing explainers.

**[Partial Soft-Matching Distance for Neural Representational Comparison with Partial Unit Correspondence](partial_soft-matching_distance_for_neural_representational_comparison_with_parti.md)**

:   This paper generalizes "soft-matching distance" to **partial optimal transport**, allowing a portion of neurons to remain unmatched. This approach finds robust unit-level correspondences between neural populations containing noise or missing counterparts. It uses an L-curve heuristic to automatically select the optimal matching mass. Results in simulations, cross-subject fMRI alignment, and deep network neuron ranking show significant improvements over standard soft-matching that forces full correspondence.

**[Path Channels and Plan Extension Kernels: A Mechanistic Description of Planning in a Sokoban RNN](path_channels_and_plan_extension_kernels_a_mechanistic_description_of_planning_i.md)**

:   This paper reverse-engineers a Deep Recurrent Convolutional network (DRC) trained with model-free reinforcement learning to play Sokoban. It discovers that the internal "where to go" planning is directly stored in specific "path channels" of the hidden state. These plans are constructed via "plan extension kernels" that extend path segments forward from boxes and backward from targets. By utilizing negative activations for pruning and backtracking, and a winner-take-all mechanism to select a unique path, the network translates black-box planning behavior into a human-interpretable bidirectional search algorithm.

**[Patronus: Interpretable Diffusion Models with Prototypes](patronus_interpretable_diffusion_models_with_prototypes.md)**

:   Patronus grafts Prototype Proposal Networks (ProtoPNet) from the classification domain onto diffusion models: a patch-level prototype encoder encodes images into similarity vectors representing the activation degrees of each prototype. This vector is then used to condition the DDPM, making the diffusion generation process "inherently interpretable"—capable of clarifying which visual concepts are learned (what), where they appear in the frame (where), and at what moment during denoising they emerge (when). This is used to diagnose shortcut learning and hidden biases in training data.

**[PERSONA: Dynamic and Compositional Inference-Time Personality Control via Activation Vector Algebra](persona_dynamic_and_compositional_inference-time_personality_control_via_activat.md)**

:   The PERSONA framework is proposed to achieve training-free dynamic and compositional personality control by extracting approximately orthogonal personality vectors in the activation space and performing vector algebra operations (scaling, addition, subtraction). It achieves a score of 9.60 on PersonalityBench, nearly matching the SFT upper bound of 9.61.

**[Persona Features Control Emergent Misalignment](persona_features_control_emergent_misalignment.md)**

:   The authors perform "model diffing" on GPT-4o before and after fine-tuning using Sparse Autoencoders (SAEs). They find that a set of "misaligned persona" features (notably a "toxic persona" feature #10) is the primary internal cause of the "emergent misalignment" phenomenon—where fine-tuning on narrow-domain erroneous data leads to broad-domain misalignment. Based on this, they achieve misalignment prediction, steering-based suppression, and "re-alignment" using small amounts of benign data.

**[PolySHAP: Extending KernelSHAP with Interaction-Informed Polynomial Regression](polyshap_extending_kernelshap_with_interaction-informed_polynomial_regression.md)**

:   This paper proposes PolySHAP, which improves the estimation accuracy of Shapley values by extending the linear approximation of KernelSHAP to higher-order polynomial regression to capture non-linear feature interactions. It theoretically proves that paired sampling is equivalent to second-order PolySHAP, providing the first explanation for the superior performance of the paired sampling heuristic.

**[PoSh: Using Scene Graphs to Guide LLMs-as-a-Judge for Detailed Image Descriptions](posh_using_scene_graphs_to_guide_llms-as-a-judge_for_detailed_image_descriptions.md)**

:   The paper proposes PoSh, an evaluation metric that extracts scene graphs $G(d) = \langle O(d), E(d), K(d) \rangle$ from both generated and reference descriptions to serve as structured rubrics. These rubrics guide an open-source 14B LLM (Qwen3-14B) in performing QA-style fine-grained error localization. PoSh outperforms GPT-4o-as-Judge by +0.05 Spearman $\rho$ on the DOCENT artwork benchmark and CapArena while remaining fully reproducible.

**[Precise and Interpretable Editing of Code Knowledge in Large Language Models](precise_and_interpretable_editing_of_code_knowledge_in_large_language_models.md)**

:   This paper replaces an MLP layer in the Transformer with a sparse, monosemantic TransCoder module. By updating only a few neurons truly activated by the target error (TCPE), the method achieves precise editing while providing neuron-level explanations of "what was changed and why." The authors propose KECode, an editing benchmark for code translation based on functional equivalence, improving CodeLlama-7b's Java→D translation accuracy from 57.5% to 64.0%.

**[Priors in Time: Missing Inductive Biases for Language Model Interpretability](priors_in_time_missing_inductive_biases_for_language_model_interpretability.md)**

:   From a Bayesian perspective, this work reveals that standard Sparse Autoencoders (SAEs) imply a prior that "concepts are independent in time," which severely mismatches the highly non-stationary nature of LLM activations that accumulate dimensionality with context. The authors propose the Temporal SAE, which decomposes activations into a "predictable component" (from context) and a "novel component" (residual). By applying sparsity priors only to the novel component, it correctly parses garden-path sentences and identifies narrative event boundaries, successfully separating slow-varying and fast-varying information.

**[Probing Rotary Position Embeddings through Frequency Entropy](probing_rotary_position_embeddings_through_frequency_entropy.md)**

:   This paper proposes **Frequency Entropy (FE)**, a training-free diagnostic metric that applies Fourier analysis to the query norm signals of RoPE rotation pairs along the sequence to calculate Shannon entropy. By decoupling signals into "frequency band structures" and "periodic oscillations," it provides a unified explanation for previous contradictory findings regarding high vs. low-frequency dimensions and discovers that periodic dimensions are largely redundant and can be attenuated during inference with minimal performance loss.

**[Provably Explaining Neural Additive Models](provably_explaining_neural_additive_models.md)**

:   Specialized efficient explanation algorithms are designed for Neural Additive Models (NAMs). By requiring only logarithmic verification queries, these algorithms generate cardinally-minimal explanations that outperform existing general-purpose subset-minimal explanation algorithms in terms of both speed and explanation quality.

**[RADAR: Reasoning-Ability and Difficulty-Aware Routing for Reasoning LLMs](radar_reasoning-ability_and_difficulty-aware_routing_for_reasoning_llms.md)**

:   This paper proposes the Radar framework, which models the adaptive reasoning problem of Reasoning Large Language Models (RLMs) as a multi-objective optimization. By utilizing Item Response Theory (IRT) to jointly estimate interpretable query difficulty and model configuration capability parameters, Radar achieves lightweight and scalable query-level routing. It outperforms SOTA routing methods across 8 reasoning benchmarks with an added latency of only approximately 7ms.

**[REAL: Reading Out Transformer Activations for Precise Localization in Language Model Steering](real_reading_out_transformer_activations_for_precise_localization_in_language_mo.md)**

:   REAL trains a Vector Quantized Autoencoder (VQ-AE) for each attention head (or layer) of a Transformer to map highly entangled hidden activations into a separable discrete code space. It then uses the log-likelihood ratio of two autoregressive priors + AUC scoring to determine "how relevant this module is to the target behavior," thereby precisely selecting modules for intervention and adaptively adjusting steering strength based on relevance. This achieves an average improvement of 20% (up to 81.5%) in truthfulness steering compared to ITI.

**[Reforming the Mechanism: Editing Reasoning Patterns in LLMs with Circuit Reshaping](reforming_the_mechanism_editing_reasoning_patterns_in_llms_with_circuit_reshapin.md)**

:   This paper proposes a new paradigm called "Reasoning Editing"—modifying a specific reasoning pattern in LLMs without affecting other reasoning capabilities. It identifies the "Circuit-Interference Law" (the more neural circuits of two reasoning patterns overlap, the stronger the editing interference). Based on this, it introduces REdit: actively "reshaping circuits" via contrastive learning to decouple overlapping circuits before editing. This simultaneously improves Generality and Locality, consistently outperforming editing baselines like LoRA, ROME, and AlphaEdit on propositional logic tasks using Qwen2.5-3B.

**[Reinforcement Learning Fine-Tuning Enhances Activation Intensity and Diversity in the Internal Circuitry of LLMs](reinforcement_learning_fine-tuning_enhances_activation_intensity_and_diversity_i.md)**

:   The authors abstract LLM residual computation into a directed graph and use Edge Attribution Patching (EAP) to score the importance of all internal edges in a single forward-backward pass. By comparing edge weight distributions before and after RL fine-tuning, they find that online RL (PPO/GRPO) systematically **increases internal activation intensity and enhances activation diversity** (increased entropy, decreased kurtosis), whereas DPO shows almost no such changes—bridging the gap between "why RL post-training is stronger" and "how internal information pathways change."

**[Representational Alignment Across Model Layers and Brain Regions with Multi-Level Optimal Transport](representational_alignment_across_model_layers_and_brain_regions_with_multi-leve.md)**

:   This paper proposes Multi-Level Optimal Transport (MOT), a dual-layer optimal transport framework featuring "inner-layer neuron transport + outer-layer hierarchical transport." It upgrades representational alignment between two networks (or brain regions) from "layer-wise greedy matching" to "globally consistent soft coupling." This approach provides a single network-level alignment score, naturally handles depth inconsistencies, and spontaneously recovers hierarchical structures (e.g., early-to-early and deep-to-deep layer mappings).

**[Rethinking Layer Relevance in Large Language Models Beyond Cosine Similarity](rethinking_layer_relevance_in_large_language_models_beyond_cosine_similarity.md)**

:   This paper demonstrates both theoretically and experimentally that "cosine similarity" is not a reliable proxy for measuring the importance of Transformer layers—a layer can have extremely low cosine similarity yet be critical to model performance. The authors advocate for using the "actual drop in model accuracy after removing the layer" as a more faithful measure of layer relevance. This approach revises several interpretability conclusions previously based on cosine similarity and yields superior structured pruning results.

**[SAE as a Crystal Ball: Interpretable Features Predict Cross-domain Transferability of LLMs without Training](sae_as_a_crystal_ball_interpretable_features_predict_cross-domain_transferabilit.md)**

:   This paper proposes STS (SAE-based Transferability Score): it predicts which Sparse Autoencoder (SAE) dimensions will be modified by Supervised Fine-Tuning (SFT) using In-Context Learning (ICL) **without fine-tuning**, and then measures the relevance of these dimensions to various downstream domains to predict performance changes prior to training, achieving Pearson correlation coefficients generally exceeding 0.7.

**[SEED-SET: Scalable Evolving Experimental Design for System-level Ethical Testing](seed-set_scalable_evolving_experimental_design_for_system-level_ethical_testing.md)**

:   The SEED-SET framework is proposed to model the ethical evaluation of autonomous systems as a hierarchical Bayesian experimental design problem. By integrating objective metrics and subjective value judgments, it efficiently generates test cases with high ethical alignment under a limited budget.

**[Seeing but Not Believing: Probing the Disconnect Between Visual Attention and Answer Correctness in VLMs](seeing_but_not_believing_probing_the_disconnect_between_visual_attention_and_ans.md)**

:   This paper systematically analyzes the phenomenon of VLMs "seeing evidence but answering incorrectly" in VQA. It finds that deep-layer attention often successfully locates the correct visual evidence, but this information is not fully utilized during the generation stage. Accordingly, the authors propose Visual Evidence Augmentation (VEA), a training-free test-time visual evidence highlighting method, which consistently improves accuracy across various models including LLaVA, Qwen, Gemma, and InternVL on multiple evidence-based VQA tasks.

**[Semantic Regexes: Auto-Interpreting LLM Features with a Structured Language](semantic_regexes_auto-interpreting_llm_features_with_a_structured_language_of_re.md)**

:   This paper proposes **Semantic Regexes**, a structured language for automated description of LLM features. By combining primitives (symbol/lexeme/field) and modifiers (context/composition/quantification), it achieves feature descriptions that are as accurate as natural language but more concise, consistent, and analyzable.

**[Sequences of Logits Reveal the Low Rank Structure of Language Models](sequences_of_logits_reveal_the_low_rank_structure_of_language_models.md)**

:   This paper proposes the "extended logit matrix" as a model-agnostic object of study. It empirically finds that the logit matrices of modern autoregressive LLMs remain approximately low-rank on long sequence scales (with a singular value power law exponent $\alpha$ slightly greater than $1/2$). Based on this, the authors design the LINGEN program, which generates target continuations using linear combinations of irrelevant or nonsense histories. Finally, a provable learning theory equivalent to this low-rankness is provided via "time-varying ISAN."

**[Setting Up for Failure: Automatic Discovery of the Neural Mechanisms of Cognitive Errors](setting_up_for_failure_automatic_discovery_of_the_neural_mechanisms_of_cognitive.md)**

:   Instead of training RNNs to perform cognitive tasks "correctly," this paper trains them to "make human-like errors." By using a non-parametric generative model (BNS) to create synthetic behavioral data with swap errors and a diffusion model (DDPM) objective to treat the second delay period as a denoising process, the authors automatically discover neural dynamics underlying visual working memory. The resulting neural geometry aligns highly with recordings from the macaque lateral prefrontal cortex.

**[Signal in the Noise: Polysemantic Interference Transfers and Predicts Cross-Model Influence](signal_in_the_noise_polysemantic_interference_transfers_and_predicts_cross-model.md)**

:   This paper maps polysemantic interference structures in small language models using SAEs, discovering that certain features—semantically unrelated but mutually interfering in activation space—can consistently alter the next-token distribution of target semantics. Furthermore, these intervention signals transfer to larger instruction-tuned models, suggesting that polysemanticity is not mere random noise but likely contains latent structures shared across models.

**[Small Transformers Don't Need LayerNorm at Inference Time: Scaling LayerNorm Removal to GPT-2 XL and Implications for Mechanistic Interpretability](small_transformers_dont_need_layernorm_at_inference_time_scaling_layernorm_remov.md)**

:   By layer-wise fine-tuning, all LayerNorm layers in the GPT-2 family (up to the 1.5B parameter GPT-2 XL) are replaced with pure linear transformations. The validation loss increases by only $+0.03 \sim 0.1$ cross-entropy, proving that LN is non-essential during inference. Removing LN reduces Direct Logit Attribution error from 50% to 0%, enabling precise mechanistic interpretability analysis.

**[Sparling: End-to-End Spatial Concept Learning via Extremely Sparse Activations](sparling_end-to-end_spatial_concept_learning_via_extremely_sparse_activations.md)**

:   This paper proves a "Motif Identifiability Theorem"—asserting that as long as intermediate concepts are local, sparse, and sufficient/necessary for the output, they can be precisely recovered using only end-to-end supervision (without any intermediate concept labels). The authors introduce the SPARLING algorithm, which approximates this optimal solution using a "Spatial Sparsity Layer" that forces activations to $99\%+$ sparsity combined with an annealing-based adaptive sparsity schedule. It achieves $>90\%$ precision in locating intermediate spatial concepts across three synthetic domains.

**[Sparse Autoencoders Trained on the Same Data Learn Different Features](sparse_autoencoders_trained_on_the_same_data_learn_different_features.md)**

:   This is an analytical paper: the authors use the Hungarian algorithm to align multiple Sparse Autoencoders (SAEs) that differ only in their initialization random seeds while seeing identical data. They find that learned features only partially overlap (only 30% on Llama 3 8B). Furthermore, larger models/SAEs exhibit lower overlap, and TopK is more unstable than ReLU. This demonstrates that SAEs identify a "practical decomposition of the activation space" rather than a "unique and objective feature list" actually used by the model.

**[Sparse CLIP: Co-optimizing Interpretability and Performance in Contrastive Learning](sparse_clip_co-optimizing_interpretability_and_performance_in_contrastive_learni.md)**

:   This paper integrates "sparsity" directly into CLIP's contrastive pre-training (adding ReLU non-negative constraints to the final projection layer + significant dimensional expansion). This trains sparse CLIP representations that are interpretable, maintain accuracy, and naturally preserve cross-modal capabilities, thereby overthowing the common assumption that "interpretability must sacrifice precision."

**[Specialization after Generalization: Towards Understanding Test-Time Training in Foundation Models](specialization_after_generalization_towards_understanding_test-time_training_in_.md)**

:   Starting from the Linear Representation Hypothesis (LRH), this paper proposes a theoretical framework of "specialization after generalization" to systematically explain why TTT is effective in in-distribution scenarios for the first time. Foundation models experience concept superposition interference due to global underparameterization. TTT releases model capacity by temporarily forgetting irrelevant concepts and locally specializing on a few concepts related to the test task, providing theoretical guarantees that generalization is possible even when the feature space is exponentially smaller than the concept space.

**[Spilled Energy in Large Language Models](spilled_energy_in_large_language_models.md)**

:   Ours reinterprets the final softmax classifier of LLMs as an Energy-Based Model (EBM) and identifies a difference between two energy paths—termed "spilled energy"—that theoretically should be equal according to the probability chain rule but are instead read out at adjacent decoding steps. Ours proves that this **completely training-free difference, read directly from logits**, is strongly correlated with model errors. Across 9 benchmarks and multiple SOTA models, its cross-task generalization significantly outperforms probing classifiers that require task-specific training.

**[Structural Inference: Interpreting Small Language Models with Susceptibilities](structural_inference_interpreting_small_language_models_with_susceptibilities.md)**

:   This paper treats small language models as Bayesian statistical physics systems. By inducing model component responses through small perturbations in data distributions, the authors define susceptibility to characterize how attention heads express or suppress different data patterns. Using PCA on a 3M-parameter, two-layer attention-only Transformer, the method automatically isolates known structures such as word boundaries, induction circuits, and bracket matching.

**[Tackling the XAI Disagreement Problem with Adaptive Feature Grouping](tackling_the_xai_disagreement_problem_with_adaptive_feature_grouping.md)**

:   This paper argues that the core reason for the conflict between post-hoc explainers and faithfulness metrics is the presence of interaction terms between different feature groups. It proposes AGREED, which reduces disagreements between explanation methods by adaptively merging strongly interacting feature groups, leading to more consistent explanations across tabular data and image saliency maps.

**[Taming Polysemanticity in LLMs: Theory-Grounded Feature Recovery via Sparse Autoencoders](taming_polysemanticity_in_llms_theory-grounded_feature_recovery_via_sparse_autoe.md)**

:   This paper revisits Sparse Autoencoder (SAE) training from the perspective of "neuron activation frequency," identifying and proving the **neuron resonance** phenomenon—where monosemantic features are reliably learned only when the neuron's activation frequency $p$ falls within a "resonance band" around the feature occurrence frequency $f$. Based on this, the authors propose the **Group Bias Adaptation (GBA)** algorithm, providing the first SAE training method with theoretical recovery guarantees that scales to 2B-parameter LLMs.

**[Task Vectors, Learned Not Extracted: Performance Gains and Mechanistic Insights](task_vectors_learned_not_extracted_performance_gains_and_mechanistic_insights.md)**

:   Instead of "extracting" Task Vectors (TV) from model representations, this paper uses gradient descent to **directly train** an injected vector (Learned Task Vector, LTV). LTV outperforms extractive TV in classification and generation tasks and can be injected at arbitrary layers or positions. The study systematically deconstructs the mechanism of TV: lower layers operate primarily through the OV circuits of attention heads (where a few "key heads" are decisive), while higher layers propagate in a near-linear "rotation + scaling" manner.

**[Temporal Geometry of Deep Networks: Hyperbolic Representations of Training Dynamics for Intrinsic Explainability](temporal_geometry_of_deep_networks_hyperbolic_representations_of_training_dynami.md)**

:   This paper treats the entire training trajectory of an MLP as a sequence of "parameter graph" snapshots and uses a permutation-symmetry-preserving hyperbolic graph attention meta-network (GTH-GMN) to embed them into the Poincaré ball. This approach reconstructs the self-organizing geometry of the network during training within a negative curvature space, matching strong baselines in tasks such as INR classification, generalization prediction, and sine regression, while allowing interpretable signals to be read directly from the radial and angular structures of the embeddings.

**[Temporal Sparse Autoencoders: Leveraging the Sequential Nature of Language for Interpretability](temporal_sparse_autoencoders_leveraging_the_sequential_nature_of_language_for_in.md)**

:   Ours proposes Temporal SAEs (T-SAEs), which introduce a temporal contrastive loss to encourage high-level features to maintain consistent activation across adjacent tokens. This achieves disentanglement of semantic and syntactic features under self-supervised training without explicit semantic signals, restoring smoother and more coherent semantic concepts without sacrificing reconstruction quality.

**[Temporal Superposition and Feature Geometry of RNNs under Memory Demands](temporal_superposition_and_feature_geometry_of_rnns_under_memory_demands.md)**

:   This paper extends the concept of "feature superposition" from feedforward networks to the temporal dimension, proposing **temporal superposition**: by training linear/nonlinear RNNs on $k$-delay recall tasks, analytically decomposing the loss into four terms, identifying the ReLU-induced "interference-free space," and a phase transition between dense and sparse mechanisms, it mechanistically explains why and how RNNs choose different representational geometries under memory pressure.

**[The Achilles' Heel of LLMs: How Altering a Handful of Neurons Can Cripple Language Abilities](the_achilles_heel_of_llms_how_altering_a_handful_of_neurons_can_cripple_language.md)**

:   This paper proposes a "perturbation-based causal identification of critical neurons" method. Across 21 LLMs ranging from 0.5B to 72B, it is discovered that zeroing out only approximately 3 neurons can cause a 72B model with 110 billion neurons to collapse entirely (perplexity soaring by up to 20 orders of magnitude). These critical neurons are highly concentrated in the `down_proj` of outer MLP layers, and the collapse occurs as a "phase transition" rather than a gradual decline.

**[The Deleuzian Representation Hypothesis](the_deleuzian_representation_hypothesis.md)**

:   This paper proposes using "clustering of pairwise differences in activation values" to extract interpretable concepts from neural networks in an unsupervised manner, serving as a simple alternative to Sparse Autoencoders (SAEs). It models concepts as "differences" (echoing the Deleuzian philosophical view of "concepts as difference"), provides a theoretical basis through discriminant analysis, and enhances concept diversity by weighting clusters with the inverse skewness of activation distributions. The concept quality exceeds existing unsupervised SAE variants and approaches supervised LDA across 5 models, 3 modalities, and 20 tasks.

**[The Geometry of Reasoning: Flowing Logics in Representation Space](the_geometry_of_reasoning_flowing_logics_in_representation_space.md)**

:   This paper proposes a geometric framework that models the LLM reasoning process as "flows" (embedding trajectories) in representation space. Through controlled experiments decoupling logical structures from semantic content, it demonstrates that LLMs internalize logical invariants beyond surface forms and identifies potentially universal representation laws across model families.

**[The Potential of CoT for Reasoning: A Closer Look at Trace Dynamics](the_potential_of_cot_for_reasoning_a_closer_look_at_trace_dynamics.md)**

:   This paper proposes using "potential" to measure the conditional improvement of a given CoT prefix on final accuracy. Through trace analysis on mathematical, scientific QA, and coding tasks, it discovers that CoT effectiveness is often concentrated in a few reasoning insights, accompanied by relevant but harmful reasoning tangents, jumps difficult for humans to interpret, and lucky guesses.

**[The Price of Amortized inference in Sparse Autoencoders](the_price_of_amortized_inference_in_sparse_autoencoders.md)**

:   This paper argues that pathologies in SAEs, such as dead latents, dense latents, feature splitting, and feature absorption, are not isolated engineering issues but the result of the conflict between amortized inference via a shared encoder and instance-wise optimality. It proposes LocA-SAE, which performs local grouping based on angular variance to balance computational cost and monosemanticity.

**[The Shape of Adversarial Influence: Characterizing LLM Latent Spaces with Persistent Homology](the_shape_of_adversarial_influence_characterizing_llm_latent_spaces_with_persist.md)**

:   This paper utilizes persistent homology (PH) to transform LLM activation point clouds into cross-model comparable topological fingerprints. It discovers that indirect prompt injection and backdoor fine-tuning—two fundamentally different attack mechanisms—leave the same "topological compression" signature in the latent space: the representation collapses from "small-and-many, compact-and-diverse" to "large-and-few, sparse-and-dominant." This phenomenon is consistent across six models from 3.8B to 70B, emerges early, and is highly discriminative across layers.

**[The Tutor-Pupil Augmentation: Enhancing Learning and Interpretability via Input Corrections](the_tutor-pupil_augmentation_enhancing_learning_and_interpretability_via_input_c.md)**

:   This paper proposes the Tutor-Pupil augmentation framework: maintaining a fixed, interpretable "Pupil" model for the primary task, while training a flexible "Tutor" model to apply a minimal perturbation $\epsilon$ in the **input space** to "correct" samples the Pupil fails on. Since corrections occur at the input level and are constrained to be minimal, these corrections serve as a diagnostic map that reveals where and why the Pupil fails, simultaneously achieving performance gains and interpretability.

**[There Was Never a Bottleneck in Concept Bottleneck Models](there_was_never_a_bottleneck_in_concept_bottleneck_models.md)**

:   The paper points out that Concept Bottleneck Models (CBMs) do not actually possess a true "bottleneck"—the fact that a representation variable $z_j$ can predict concept $c_j$ does not mean it encodes *only* information about $c_j$. It proposes the Minimal Concept Bottleneck Model (MCBM), which uses information bottleneck regularization to constrain each $z_j$ to retain only information from its corresponding concept, achieving true decoupled representations and reliable concept interventions.

**[Thought Branches: Interpreting LLM Reasoning Requires Resampling](thought_branches_interpreting_llm_reasoning_requires_resampling.md)**

:   This paper argues that interpreting reasoning models requires analyzing the **distribution of all possible trajectories** generated under the same prompt rather than a single Chain-of-Thought (CoT). By **resampling subsequent text** starting from a specific sentence in a CoT, the authors measure its causal influence. They propose a suite of methods including counterfactual importance, resilience, Counterfactual++, and graft resampling to re-examine safety-related issues such as whether self-preservation truly drives model extortion, whether manual CoT edits truly manipulate reasoning, and how hidden information functions in unfaithful CoTs.

**[TimeSeg: An Information-Theoretic Segment-Wise Explainer for Time-Series Predictions](timeseg_an_information-theoretic_segment-wise_explainer_for_time-series_predicti.md)**

:   TimeSeg redefines "explaining black-box time-series models" as selecting a set of continuous sub-sequences that maximize the joint mutual information with the model's prediction. By using the chain rule, this intractable joint optimization is decomposed into a sequential decision-making problem solved via reinforcement learning. This allows for producing variable-length segment explanations with high alignment to ground truth and precise boundaries under strict black-box conditions (input-output access only).

**[To Sink or Not to Sink: Visual Information Pathways in Large Vision-Language Models](to_sink_or_not_to_sink_visual_information_pathways_in_large_vision-language_mode.md)**

:   This paper discovers that ViT sink tokens in Large Vision-Language Models (LVLMs) are not merely noise but propagate into the LLM, carrying coarse-grained high-level visual semantics. It proposes a training-free "sink-to-the-front" strategy and a trainable DIYSink framework to enable models to better utilize sink and non-sink visual information based on task requirements.

**[Token Alignment Heads: Unveiling Attention's Role in LLM Multilingual Translation](token_alignment_heads_unveiling_attentions_role_in_llm_multilingual_translation.md)**

:   The authors identify a specific class of attention heads in LLMs responsible for mapping source language tokens to target language tokens—token alignment heads (TAHs). They demonstrate that these heads are ubiquitous, highly sparse, cross-linguistically consistent, and play a direct causal role in translation. Based on these insights, they design a data scoring algorithm, TRater, which significantly enhances the model's translation capabilities using a minimal amount of critical data.

**[Tokenizing Single-Channel EEG with Time-Frequency Motif Learning](tokenizing_single-channel_eeg_with_time-frequency_motif_learning.md)**

:   Ours proposes TFM-Tokenizer, the first framework to learn a time-frequency motif vocabulary from single-channel EEG and encode it into discrete tokens. It consistently improves performance on tasks such as event classification and seizure detection and serves as a plug-and-play component to enhance existing EEG foundation models.

**[TokenSeek: Memory Efficient Fine Tuning via Instance-Aware Token Ditching](tokenseek_memory_efficient_fine_tuning_via_instance-aware_token_ditching.md)**

:   TokenSeek is proposed as a universal memory optimization plugin for Transformer fine-tuning. By combining contextual attention information and gradient signals for instance-level token importance evaluation, it retains only 10% of high-value tokens for gradient updates. This achieves up to 65.7% memory savings while maintaining or even exceeding the performance of full-token fine-tuning.

**[Toward Faithful Retrieval-Augmented Generation with Sparse Autoencoders](toward_faithful_retrieval-augmented_generation_with_sparse_autoencoders.md)**

:   RAGLens is proposed, which utilizes Sparse Autoencoders (SAE) to decouple RAG hallucination-specific features from internal LLM activations. By employing mutual information feature selection and a Generalized Additive Model (GAM), a lightweight interpretable hallucination detector is constructed. It outperforms existing methods across multiple benchmarks and supports token-level interpretable feedback and hallucination mitigation.

**[Towards Cognitively-Faithful Decision-Making Models to Improve AI Alignment](towards_cognitively-faithful_decision-making_models_to_improve_ai_alignment.md)**

:   Starting from a set of "weak axioms," the authors derive a class of **two-stage decision models** (first applying learnable editing rules to each feature, then using a fixed aggregation rule for dominance testing). This allows the learned preference models to maintain interpretability while faithfully reproducing the cognitive processes humans use in heuristics (such as thresholds and counting) for pairwise comparisons, achieving "comparable accuracy with superior interpretability" on moral judgment data for kidney allocation.

**[Towards Understanding Subliminal Learning: When and How Hidden Biases Transfer](towards_understanding_subliminal_learning_when_and_how_hidden_biases_transfer.md)**

:   Through controlled experiments and mechanistic analysis, this study reveals the nature of subliminal learning: hidden preferences of teacher models are transmitted to student models through a small number of "divergence tokens," with early layers being critical. Furthermore, the phenomenon is found to be fragile and can be suppressed by simple paraphrasing.

**[Towards Understanding the Nature of Attention with Low-Rank Sparse Decomposition](towards_understanding_the_nature_of_attention_with_low-rank_sparse_decomposition.md)**

:   This paper proposes Low-Rank Sparse Attention (Lorsa), which approximates the output of original Multi-Head Self-Attention (MHSA) using thousands of sparsely activated, single-dimensional output attention heads. This approach disentangles atomic attention units from "attention superposition," allowing for the independent and clean identification of induction heads, successor heads, attention sinks, and even novel sub-word level induction heads.

**[Tracking Equivalent Mechanistic Interpretations Across Neural Networks](tracking_equivalent_mechanistic_interpretations_across_neural_networks.md)**

:   This paper formalizes "whether two neural networks implement the same mechanistic interpretation" as an equivalence problem between interpretation implementation sets. It proposes a method that generates co-interpretive implementations via intervention and estimates Congruity using representation similarity. Experiments on synthetic Transformers, IOI circuits, and POS/next-token tasks demonstrate its ability to track mechanistic equivalence across models and tasks.

**[TreeGrad-Ranker: Feature Ranking via O(L)-Time Gradients for Decision Trees](treegrad-ranker_feature_ranking_via_ol-time_gradients_for_decision_trees.md)**

:   Addressing feature ranking for decision trees, the authors theoretically prove that "probabilistic values" such as Shapley or Banzhaf values are no better than random guessing when optimizing the joint goal corresponding to insertion/deletion metrics. They propose TreeGrad, which computes gradients on multilinear extensions in $O(L)$ time, and construct TreeGrad-Ranker to directly optimize the joint goal along with the numerically stable TreeGrad-Shap, significantly outperforming probabilistic value baselines.

**[Uncertainty as Feature Gaps: Epistemic Uncertainty Quantification of LLMs in Contextual Question-Answering](uncertainty_as_feature_gaps_epistemic_uncertainty_quantification_of_llms_in_cont.md)**

:   This paper derives the epistemic uncertainty of LLMs as the "feature gap between the current model's hidden states and an ideal model." In contextual QA (RAG) scenarios, this gap is approximated using three semantic features (context reliance, context comprehension, and honesty). By extracting feature directions from minimal labeled samples and ensembling them, the method improves PRR by approximately 13–16 points across multiple QA benchmarks with negligible inference overhead.

**[Uncovering Conceptual Blindspots in Generative Image Models Using Sparse Autoencoders](uncovering_conceptual_blindspots_in_generative_image_models_using_sparse_autoenc.md)**

:   This paper proposes a framework to systematically diagnose "conceptual blindspots" using Sparse Autoencoders (SAEs). By mapping both real and model-generated images onto 32,000 interpretable concepts learned by an RA-SAE, it introduces an energy delta metric $\delta(k)$ to quantify whether each concept is "suppressed" or "exaggerated" in the generative distribution. This transforms anecdotal generation failures (e.g., inability to draw bird feeders or incorrect finger counts) into quantifiable, comparable, and explorable structured analyses.

**[Understanding Cross-Layer Contributions to Mixture-of-Experts Routing in LLMs](understanding_cross-layer_contributions_to_mixture-of-experts_routing_in_llms.md)**

:   This paper proposes a lightweight **recursive decomposition** method to decompose the assignment scores given by MoE routers into contributions from "token embeddings + attention outputs of each layer + MoE outputs of each layer," and even down to individual attention heads or experts. By using score **variance** to measure influence, it reveals for the first time from a cross-layer perspective that MoE routing is not a local decision but is jointly shaped by entanglement effects among deep components.

**[Understanding Task Vectors in In-Context Learning: Emergence, Functionality, and Limitations](understanding_task_vectors_in_in-context_learning_emergence_functionality_and_li.md)**

:   This paper proposes the "Task Vectors as Representative Demonstrations" hypothesis—that an injected task vector is essentially a **single representative demonstration** distilled from multiple context examples. Through critical point analysis of linear attention models, the authors prove that task vectors naturally emerge in triplet prompt training and predict a fundamental limitation: they can only represent **rank-one mappings** and cannot solve general bijection tasks. Based on these insights, an enhanced multi-vector injection method is proposed.

**[Universal Properties of Activation Sparsity in Modern Large Language Models](universal_properties_of_activation_sparsity_in_modern_large_language_models.md)**

:   This paper provides a systematic study of activation sparsity in modern LLMs (GLU architecture + SiLU/GELU). It proposes a universal top-p sparsification framework and a "critical sparsity" metric, finding that activation sparsity increases monotonically with model scale. Input sparsification is identified as the most practical training-free acceleration scheme, and the authors demonstrate for the first time that diffusion-based LLMs also exhibit significant activation sparsity.

**[Watch the Weights: Unsupervised Monitoring and Control of Fine-tuned LLMs](watch_the_weights_unsupervised_monitoring_and_control_of_fine-tuned_llms.md)**

:   By focusing on weights rather than activations—specifically by performing SVD on the weight difference $\Delta W$ between a fine-tuned model and its base model—the authors find that top singular vectors precisely encode behaviors learned during fine-tuning. This allows for monitoring, steering, and even recovering these behaviors without any fine-tuning data, achieving up to 100% backdoor detection (FPR < 1%) and 95.42% detection for unlearned content.

**[What Do Large Language Models Know About Opinions?](what_do_large_language_models_know_about_opinions.md)**

:   Instead of observing LLM outputs, this paper examines internal activations and discovers that the "internal knowledge" of LLMs regarding group opinions far exceeds what they express (reducing KL divergence by 52–66%, achieving performance close to fine-tuning but ~300× cheaper). It identifies that this knowledge forms rapidly in middle layers and is "bottlenecked" by the final unembedding layer. Finally, it uses Sparse Autoencoders to trace this knowledge to attention head features that selectively encode group information, enabling causal steering of model outputs.

**[When Machine Learning Gets Personal: Evaluating Prediction and Explanation](when_machine_learning_gets_personal_evaluating_prediction_and_explanation.md)**

:   This paper proposes a unified framework to quantify the impact of model personalization on prediction accuracy and explanation quality. It proves that the two can be decoupled (prediction remains unchanged while explanation improves or degrades) and derives finite-sample lower bounds for hypothesis testing error probabilities based on dataset statistics, revealing that personalization effects are statistically untestable in many practical scenarios.

**[When Thinking Backfires: Mechanistic Insights into Reasoning-Induced Misalignment](when_thinking_backfires_mechanistic_insights_into_reason-induced_misalignment.md)**

:   This paper identifies and names "Reasoning-Induced Misalignment" (RIM)—a phenomenon where enhancing an LLM's reasoning capabilities (enabling CoT during inference or fine-tuning on mathematical problems) causes the model to become more compliant with malicious requests. Mechanistically, a class of "refusal attention heads" triggers refusal by reducing attention to CoT tokens, while training-time reasoning competes with safety for the same group of neurons, leading to the displacement of safety capabilities.

**[Why Low-Precision Transformer Training Fails: An Analysis on Flash Attention](why_low-precision_transformer_training_fails_an_analysis_on_flash_attention.md)**

:   To be added after further reading.

**[xRFM: Accurate, scalable, and interpretable feature learning models for tabular data](xrfm_accurate_scalable_and_interpretable_feature_learning_models_for_tabular_dat.md)**

:   xRFM embeds the AGOP-based Recursive Feature Machine into a supervised binary partition tree. This enables the model to learn local relevant features across different data subgroups while reducing training complexity to approximately $O(n\log n)$ and inference complexity to $O(\log n)$, achieving high competitiveness on TALENT regression, TabArena-Lite, and large-scale meta-test tabular benchmarks.

**[Your VAR Model is Secretly an Efficient and Explainable Generative Classifier](your_var_model_is_secretly_an_efficient_and_explainable_generative_classifier.md)**

:   By treating the computable likelihood of Visual Autoregressive (VAR) models directly as a generative classifier and employing a combination of "Likelihood Smoothing + Partial Scale Candidate Pruning + CCA Fine-tuning" to form A-VARC+, the method achieves accuracy comparable to DiT-based diffusion classifiers on ImageNet-100 (gap <1%) while reducing computation by 89×. It further provides visual interpretability via token-level mutual information and replay-free class-incremental learning capabilities.

**[ZeroTuning: Unlocking the Initial Token's Power to Enhance Large Language Models Without Training](zerotuning_unlocking_the_initial_tokens_power_to_enhance_large_language_models_w.md)**

:   ZeroTuning is proposed to enhance LLM performance across 15 datasets without training by applying head-specific scaling to the attention scores of the initial token (e.g., `<BOS>`), requiring only 4 lines of code modifications.

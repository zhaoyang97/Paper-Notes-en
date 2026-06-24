---
title: >-
  ACL2025 Pretraining Papers · 40 Notes
description: >-
  40 ACL2025 papers in the Pretraining area, covering LLM, Adversarial Robustness, Few-/Zero-Shot Learning, Speech & Audio and more. Each note has TL;DR, motivation, method, experiments, highlights, and limitations — 5-minute reads of core ideas.
tags:
  - "ACL2025"
  - "Pretraining"
  - "AI paper notes"
  - "paper summaries"
  - "LLM"
  - "Adversarial Robustness"
  - "Few-/Zero-Shot Learning"
  - "Speech & Audio"
item_list:
  - u: "adversarial_tokenization/"
    t: "Adversarial Tokenization"
  - u: "asynclm_efficient_and_adaptive_async_pre-training_of_language_models/"
    t: "AsyncLM: Efficient and Adaptive Async Pre-training of Language Models"
  - u: "autonomous_data_selection_with_zero-shot_generative_classifiers_for_mathematical/"
    t: "AutoDS: Autonomous Data Selection with Zero-shot Generative Classifiers for Mathematical Texts"
  - u: "between_circuits_chomsky/"
    t: "Between Circuits and Chomsky: Pre-pretraining on Formal Languages Imparts Linguistic Biases"
  - u: "byte_latent_transformer/"
    t: "Byte Latent Transformer: Patches Scale Better Than Tokens"
  - u: "chinese_grammatical_error_correction_with_pre-trained_models_and_linguistic_clue/"
    t: "Chinese Grammatical Error Correction With Pre-trained Models and Linguistic Clues"
  - u: "critiq_mining_data_quality_criteria_from_human_preferences/"
    t: "CritiQ: Mining Data Quality Criteria from Human Preferences"
  - u: "data-constrained_synthesis_of_training_data_for_de-identification/"
    t: "Data-Constrained Synthesis of Training Data for De-Identification"
  - u: "data_caricatures_on_the_representation_of_african_american_language_in_pretraini/"
    t: "Data Caricatures: On the Representation of African American Language in Pretraining Corpora"
  - u: "data_whisperer_data_selection/"
    t: "Data Whisperer: Efficient Data Selection for Task-Specific LLM Fine-Tuning via Few-Shot In-Context Learning"
  - u: "davir_data_selection_via_implicit_reward_for_large_language_models/"
    t: "DavIR: Data Selection via Implicit Reward for Large Language Models"
  - u: "diversity_explains_inference_scaling_laws_through_a_case_study_of_minimum_bayes_/"
    t: "Diversity Explains Inference Scaling Laws: Through a Case Study of Minimum Bayes Risk Decoding"
  - u: "dual_stage_curriculum_learning_sequence_labeling/"
    t: "An Effective Incorporating Heterogeneous Knowledge Curriculum Learning for Sequence Labeling"
  - u: "emergent_abilities_continued_pt/"
    t: "Emergent Abilities of Large Language Models under Continued Pretraining for Language Adaptation"
  - u: "fr_spec_speculative_sampling/"
    t: "FR-Spec: Accelerating Large-Vocabulary Language Models via Frequency-Ranked Speculative Sampling"
  - u: "how_do_llms_acquire_new_knowledge_a_knowledge_circuits_perspective_on_continual_/"
    t: "How Do LLMs Acquire New Knowledge? A Knowledge Circuits Perspective on Continual Pre-Training"
  - u: "improving_continual_pre-training_through_seamless_data_packing/"
    t: "Improving Continual Pre-training Through Seamless Data Packing"
  - u: "inconsistent_tokenizations_cause_language_models_to_be_perplexed_by_japanese_gra/"
    t: "Inconsistent Tokenizations Cause Language Models to be Perplexed by Japanese Grammar"
  - u: "incorporating_domain_knowledge_into_materials_tokenization/"
    t: "Incorporating Domain Knowledge into Materials Tokenization"
  - u: "inserter_speech_instruction/"
    t: "InSerter: Speech Instruction Following with Unsupervised Interleaved Pre-training"
  - u: "large_vocabulary_size_improves_large_language_models/"
    t: "Large Vocabulary Size Improves Large Language Models"
  - u: "leancode_understanding_models_better_for_code_simplification_of_pre-trained_larg/"
    t: "LEANCODE: Understanding Models Better for Code Simplification of Pre-trained Large Language Models"
  - u: "making_llms_better_many-to-many_speech-to-text_translators_with_curriculum_learn/"
    t: "Making LLMs Better Many-to-Many Speech-to-Text Translators with Curriculum Learning"
  - u: "metarater_a_multidimensional_data_selection_method/"
    t: "Meta-rater: A Multi-dimensional Data Selection Method for Pre-training Language Models"
  - u: "model_performance-guided_evaluation_data_selection_for_effective_prompt_optimiza/"
    t: "Model Performance-Guided Evaluation Data Selection for Effective Prompt Optimization"
  - u: "nemotron_cc_pretraining_data/"
    t: "Nemotron-CC: Transforming Common Crawl into a Refined Long-Horizon Pretraining Dataset"
  - u: "optimizing_pre-training_data_mixtures_with_mixtures_of_data_expert_models/"
    t: "Optimizing Pre-Training Data Mixtures with Mixtures of Data Expert Models"
  - u: "pre-training_curriculum_for_multi-token_prediction_in_language_models/"
    t: "Pre-Training Curriculum for Multi-Token Prediction in Language Models"
  - u: "retrofitting_large_language_models_with_dynamic_tokenization/"
    t: "Retrofitting Large Language Models with Dynamic Tokenization"
  - u: "scar_style_consistency_data_selection/"
    t: "SCAR: Data Selection via Style Consistency-Aware Response Ranking for Efficient Instruction-Tuning"
item_total: 40
---

<!-- Auto-generated by src/gen_blog_index.py --lang en -->
# 📚 Pretraining

**💬 ACL2025** · **40** paper notes

📌 **Same area in other venues:** [📷 CVPR2026 (5)](../../CVPR2026/llm_pretraining/index.md) · [🔬 ICLR2026 (79)](../../ICLR2026/llm_pretraining/index.md) · [💬 ACL2026 (12)](../../ACL2026/llm_pretraining/index.md) · [🧪 ICML2026 (27)](../../ICML2026/llm_pretraining/index.md) · [🤖 AAAI2026 (9)](../../AAAI2026/llm_pretraining/index.md) · [🧠 NeurIPS2025 (51)](../../NeurIPS2025/llm_pretraining/index.md)

🔥 **Top topics:** LLM ×8 · Adversarial Robustness ×2 · Few-/Zero-Shot Learning ×2 · Speech & Audio ×2

**[Adversarial Tokenization](adversarial_tokenization.md)**

:   This paper finds that while the BPE tokenizer in the LLM pipeline uses only a single unique word segmentation method, there are exponentially many valid segmentations for the same string. By adversarially selecting non-standard tokenization schemes, safety alignment can be bypassed without changing the original text, yielding an attack success rate comparable to existing SOTA text-level attack methods.

**[AsyncLM: Efficient and Adaptive Async Pre-training of Language Models](asynclm_efficient_and_adaptive_async_pre-training_of_language_models.md)**

:   This paper proposes AsyncLM, an efficient asynchronous pre-training framework that addresses the gradient staleness issue in asynchronous distributed training through adaptive gradient compensation and dynamic batch scheduling strategies, improving the throughput of large-scale language model pre-training by 1.4-1.8x while maintaining model quality comparable to synchronous training.

**[AutoDS: Autonomous Data Selection with Zero-shot Generative Classifiers for Mathematical Texts](autonomous_data_selection_with_zero-shot_generative_classifiers_for_mathematical.md)**

:   AutoDS is proposed, which uses the base language model itself as a zero-shot generative classifier to automatically evaluate mathematical text quality by calculating a continuous LM-Score from YES/NO token logits. It filters high-quality corpora for continual pre-training, achieving an approximately 2x token efficiency improvement on MATH, GSM8K, and BBH.

**[Between Circuits and Chomsky: Pre-pretraining on Formal Languages Imparts Linguistic Biases](between_circuits_chomsky.md)**

:   Proposes performing "pre-pretraining" on formal languages prior to natural language pre-training, demonstrating that formal languages with hierarchical dependency structures (such as k-Shuffle Dyck) provide effective inductive biases for Transformers, enabling a 1B-parameter model to achieve the same language modeling loss with 33% fewer tokens.

**[Byte Latent Transformer: Patches Scale Better Than Tokens](byte_latent_transformer.md)**

:   Proposes Byte Latent Transformer (BLT), a tokenizer-free byte-level LLM architecture that aggregates bytes into variable-length patches via entropy-based dynamic grouping. It matches the performance of token-based models at the 8B scale for the first time, while unlocking a new scaling dimension of improving inference efficiency by simultaneously scaling both patch and model sizes.

**[Chinese Grammatical Error Correction With Pre-trained Models and Linguistic Clues](chinese_grammatical_error_correction_with_pre-trained_models_and_linguistic_clue.md)**

:   This paper proposes a Chinese grammatical error correction method that integrates pre-trained language models with multi-level linguistic clues (pinyin, glyphs, and dependency syntax). By explicitly injecting linguistic prior knowledge, it enhances the correction model's ability to identify and amend Chinese-specific error types.

**[CritiQ: Mining Data Quality Criteria from Human Preferences](critiq_mining_data_quality_criteria_from_human_preferences.md)**

:   CritiQ proposes an automatic data quality criteria mining method based on agent collaboration. With only about 30 human preference annotation pairs, it can automatically discover interpretable data quality criteria and train a scorer for efficient data selection, significantly improving the downstream performance of Llama 3.1 in code, math, and logic domains.

**[Data-Constrained Synthesis of Training Data for De-Identification](data-constrained_synthesis_of_training_data_for_de-identification.md)**

:   This work systematically investigates how to generate synthetic clinical text using domain-adapted LLMs under data-constrained conditions and how to train NER models for Personal Identifiable Information (PII) detection via machine labeling. The study reveals that the quality of the machine labeler, rather than the scale of the generative model, is the key factor determining the utility of synthetic data.

**[Data Caricatures: On the Representation of African American Language in Pretraining Corpora](data_caricatures_on_the_representation_of_african_american_language_in_pretraini.md)**

:   Combining quantitative experiments, human judgment, and qualitative analysis, this work systematically evaluates the quantity and quality of African American Language (AAL) across 12 open-source pretraining corpora. It finds that AAL constitutes only 0.007%–0.18% of the documents (far below its population representation). In C4, 28.9% of AAL texts are judged inappropriate for LLM generation, and 24.5% reinforce harmful stereotypes. Furthermore, 13 out of 16 automated filters systematically favor retaining White Mainstream English (WME) over AAL.

**[Data Whisperer: Efficient Data Selection for Task-Specific LLM Fine-Tuning via Few-Shot In-Context Learning](data_whisperer_data_selection.md)**

:   Data Whisperer proposes a training-free few-shot ICL data selection method using attention weighting. It leverages the pre-trained model's own ICL capabilities and attention scores to evaluate training samples, outperforming full-data fine-tuning with only 10% of the data while operating 7-20 times faster than existing methods.

**[DavIR: Data Selection via Implicit Reward for Large Language Models](davir_data_selection_via_implicit_reward_for_large_language_models.md)**

:   The DavIR data selection method is proposed, which effectively eliminates the sequence-length dependency in the RHO objective through **reference model loss normalization** (rather than token-count normalization) of the loss difference between the base and reference models. This allows a model trained on only **6%** of the Alpaca dataset (3K/52K) to outperform one trained on the full dataset, while extending the normalization concept to DPO to yield DavIR-DPO, improving Zephyr's alignment performance on AlpacaEval by 8%.

**[Diversity Explains Inference Scaling Laws: Through a Case Study of Minimum Bayes Risk Decoding](diversity_explains_inference_scaling_laws_through_a_case_study_of_minimum_bayes_.md)**

:   Reinterprets MBR decoding from the theoretical perspective of bias-diversity decomposition: estimation error $MSE = Bias - Diversity$, indicating that increasing diversity (the diversity of pseudo-references) is the key to improving MBR performance. It further extends this to general inference methods through information theory, revealing that diversity is the theoretical root of the inference scaling law (improving performance by increasing samples yields diminishing returns), and empirically validates this on machine translation, summarization, and image captioning tasks.

**[An Effective Incorporating Heterogeneous Knowledge Curriculum Learning for Sequence Labeling](dual_stage_curriculum_learning_sequence_labeling.md)**

:   This paper proposes a Dual-stage Curriculum Learning (DCL) framework for sequence labeling. By employing a data-level and model-level two-stage training strategy from easy to difficult, combined with a Bayesian uncertainty-based token-level dynamic difficulty metric and a Root function training scheduler, the framework achieves the dual benefits of performance improvements and over 27% training acceleration across three tasks: CWS, POS, and NER.

**[Emergent Abilities of Large Language Models under Continued Pretraining for Language Adaptation](emergent_abilities_continued_pt.md)**

:   Reveals that mixing in English data during continued pre-training (CPT) for language adaptation is crucial for preserving the model's in-context learning (ICL) and downstream emergent abilities—despite having little impact on validation perplexity; furthermore, proposes curriculum learning and EMA weight averaging as effective alternatives.

**[FR-Spec: Accelerating Large-Vocabulary Language Models via Frequency-Ranked Speculative Sampling](fr_spec_speculative_sampling.md)**

:   This paper proposes the FR-Spec framework, which optimizes drafting candidate selection in speculative sampling by compressing the vocabulary space based on token frequency. This reduces the LM Head computation overhead by 75% and achieves an additional 1.12× speedup over EAGLE-2 while guaranteeing mathematical equivalence in the output distribution.

**[How Do LLMs Acquire New Knowledge? A Knowledge Circuits Perspective on Continual Pre-Training](how_do_llms_acquire_new_knowledge_a_knowledge_circuits_perspective_on_continual_.md)**

:   This work investigates the mechanisms of new knowledge acquisition during continual pre-training of LLMs from the perspective of knowledge circuit evolution. Across GPT-2, Llama, and Phi architectures, the authors find that: (1) new knowledge related to existing knowledge is easier to acquire; (2) knowledge circuits undergo a distinct phase transition of "formation $\rightarrow$ optimization"; (3) circuit evolution follows a deep-to-shallow pattern, where extraction functions are first established in mid-to-deep layers, followed by the enrichment of knowledge representations in shallow layers.

**[Improving Continual Pre-training Through Seamless Data Packing](improving_continual_pre-training_through_seamless_data_packing.md)**

:   This paper proposes Seamless Packing (SP), a data packing strategy for continual pre-training. Through a two-stage method combining sliding window processing for long texts and the First-Fit-Decreasing (FFD) algorithm for packing short texts, SP preserves context continuity and minimizes truncation/padding, outperforming baseline methods in 99% of the experimental settings.

**[Inconsistent Tokenizations Cause Language Models to be Perplexed by Japanese Grammar](inconsistent_tokenizations_cause_language_models_to_be_perplexed_by_japanese_gra.md)**

:   Reveals that inconsistent tokenization of the tokenizer is the root cause of LLMs failing to adhere to subtle grammatical rules such as the Japanese "first-person psych-predicate constraint"—when restricting test sentences to consistent tokenization, the perplexity gap of Llama 3 improves by 28 times.

**[Incorporating Domain Knowledge into Materials Tokenization](incorporating_domain_knowledge_into_materials_tokenization.md)**

:   This paper proposes MATTER, a domain-aware tokenization framework designed for materials science. By training a materials concept detector, MatDetector, and injecting its detection results into the token merge ranking, it prevents the fragmentation of domain terminology. It achieves average performance gains of 4% and 2% on generation and classification tasks, respectively.

**[InSerter: Speech Instruction Following with Unsupervised Interleaved Pre-training](inserter_speech_instruction.md)**

:   This paper proposes InSerter (Interleaved Speech-Text Pre-training), which utilizes Text-to-Speech (TTS) to synthesize large-scale text corpora into interleaved speech-text sequences for pre-training. This significantly boosts the speech instruction-following capability of SpeechLLMs. Additionally, the first comprehensive speech instruction-following benchmark, SpeechInstructBench, is constructed.

**[Large Vocabulary Size Improves Large Language Models](large_vocabulary_size_improves_large_language_models.md)**

:   Experiments demonstrate that larger subword vocabulary sizes consistently improve LLM performance on downstream tasks. This work also proposes a simple vocabulary replacement method (Swap & Insert) for switching to a more appropriate vocabulary in continual training scenarios.

**[LEANCODE: Understanding Models Better for Code Simplification of Pre-trained Large Language Models](leancode_understanding_models_better_for_code_simplification_of_pre-trained_larg.md)**

:   This paper proposes LeanCode, a context-aware attention-score-based code simplification method. By leveraging CLS attention (for classification tasks) and encoder-decoder attention (for generation tasks) to measure token importance, LeanCode outperforms SOTA methods DietCode/SlimCode by up to 60% and 29% in code search and code summarization tasks respectively while reducing inference time by up to 40.9%.

**[Making LLMs Better Many-to-Many Speech-to-Text Translators with Curriculum Learning](making_llms_better_many-to-many_speech-to-text_translators_with_curriculum_learn.md)**

:   Proposes LLM-SRT, which reformulates the Speech-to-Text Translation (S2TT) task as a joint Speech Recognition and Translation (SRT) task. Through a three-stage curriculum learning strategy (ASR→SMT→SRT), it effectively leverages the machine translation capabilities of LLMs to achieve state-of-the-art many-to-many speech translation performance across $15 \times 14$ language pairs in extremely low-resource scenarios (less than 10 hours of data per language).

**[Meta-rater: A Multi-dimensional Data Selection Method for Pre-training Language Models](metarater_a_multidimensional_data_selection_method.md)**

:   Proposes the Meta-rater multi-dimensional data selection framework, defining four quality dimensions under PRRC (Professionalism, Readability, Reasoning, and Cleanliness). By using a proxy model regression to learn the optimal weighted combination of multiple quality scores, it doubles the training convergence speed of a 1.3B model and improves downstream task performance by 3.23%.

**[Model Performance-Guided Evaluation Data Selection for Effective Prompt Optimization](model_performance-guided_evaluation_data_selection_for_effective_prompt_optimiza.md)**

:   Proposed IPOMP, a two-stage evaluation data selection method. The first stage selects diverse samples through semantic clustering and boundary analysis, and the second stage iteratively replaces redundant samples using real-time model performance during the prompt optimization process. It improves prompt optimization performance by 1.6%-3.1% and stability by 50%+ on BIG-bench and LIAR, with less than 1% extra overhead.

**[Nemotron-CC: Transforming Common Crawl into a Refined Long-Horizon Pretraining Dataset](nemotron_cc_pretraining_data.md)**

:   Nemotron-CC constructs a 6.3T token long-horizon pretraining dataset (consisting of 4.4T unique real tokens + 1.9T synthetic tokens) from Common Crawl. It implements three core strategies: classifier ensembling to increase high-quality token recall, synthetic data rewriting to expand the count of unique tokens, and removing heuristic filters for high-quality data. In a 15T token training scenario, it enables an 8B model to achieve an MMLU of 70.3, surpassing Llama 3.1 8B (65.3) trained on the same scale.

**[Optimizing Pre-Training Data Mixtures with Mixtures of Data Expert Models](optimizing_pre-training_data_mixtures_with_mixtures_of_data_expert_models.md)**

:   Proposes the Mixture of Data Experts (MDE) method, which independently trains expert models on each data domain and aggregates them via probability-level ensemble using mixture weights. This efficiently approximates language model loss under different data mixture ratios, significantly improving the search efficiency and prediction accuracy of pre-training data mixture proportions.

**[Pre-Training Curriculum for Multi-Token Prediction in Language Models](pre-training_curriculum_for_multi-token_prediction_in_language_models.md)**

:   To address the issue where small language models (SLMs) struggle to directly benefit from the multi-token prediction (MTP) objective, forward and reverse curriculum learning strategies are proposed. The forward curriculum (NTP→MTP) allows SLMs to improve generation quality while maintaining self-speculative decoding acceleration, whereas the reverse curriculum (MTP→NTP) achieves better NTP performance but loses the inference acceleration advantage.

**[Retrofitting Large Language Models with Dynamic Tokenization](retrofitting_large_language_models_with_dynamic_tokenization.md)**

:   This paper proposes retrofitting existing language models with dynamic tokenization. It dynamically determines token boundaries using a BPE-inspired subword merging algorithm, combined with a pre-trained embedding-prediction hypernetwork to calculate the embeddings of merged tokens on the fly. It achieves an average >20% reduction in sequence length with less than a 2% performance drop on encoder models, and up to a 17% sequence reduction on decoder models.

**[SCAR: Data Selection via Style Consistency-Aware Response Ranking for Efficient Instruction-Tuning](scar_style_consistency_data_selection.md)**

:   SCAR identifies "linguistic form" and "instructional surprisal" of responses as two key style factors influencing LLM instruction-tuning performance. It proposes a style-consistency-aware ranking method to automatically select high-quality training data, enabling the fine-tuned LLM to match or exceed the performance of training on the full dataset using only 0.7% of the original data.

**[Second Language (Arabic) Acquisition of LLMs via Progressive Vocabulary Expansion](second_language_arabic_acquisition_of_llms_via_progressive_vocabulary_expansion.md)**

:   Inspired by human second language acquisition, this paper proposes Progressive Vocabulary Expansion (PVE), a method that incrementally, exponentially introduces Arabic subwords into the LLaMA2 vocabulary across stages. This approach achieves efficient Arabic language adaptation while preserving the original English knowledge of the model, culminating in the AraLLaMA 7B/13B models.

**[Splintering Nonconcatenative Languages for Better Tokenization](splintering_nonconcatenative_languages_for_better_tokenization.md)**

:   This paper proposes **Splinter**—a "pre-tokenization" step added prior to BPE/UnigramLM. By iteratively deleting characters, it rearranges words from languages with "root-in-template" structures like Hebrew, Arabic, and Malay into linearly splittable sequences. This enables standard tokenizers to segment roots into contiguous tokens, leading to improvements in both intrinsic metrics and downstream Hebrew tasks.

**[Stealing Training Data from Large Language Models in Decentralized Training through Activation Inversion Attack](stealing_training_data_from_large_language_models_in_decentralized_training_thro.md)**

:   This paper proposes the Activation Inversion Attack (AIA), systematically revealing for the first time that malicious stages in decentralized training (pipeline parallelism) can efficiently reconstruct training data by intercepting intermediate activations. In a Bloom-7B1 fine-tuning scenario, AIA accurately recovers 62% of private emails and nearly 100% of birthday information.

**[Synthesizing Post-Training Data for LLMs through Multi-Agent Simulation](synthesizing_post-training_data_for_llms_through_multi-agent_simulation.md)**

:   This paper proposes the MATRIX multi-agent simulator and the MATRIX-Gen scenario-driven instruction generator to synthesize high-quality LLM post-training data by simulating real-world social scenarios. Llama-3-8B trained on only 20K synthesized data outperforms the official Meta Llama-3-8B-Instruct (trained on over 10M data) on AlpacaEval 2 and Arena-Hard.

**[TokAlign: Efficient Vocabulary Adaptation via Token Alignment](tokalign_vocab_adaptation.md)**

:   This paper proposes TokAlign, which learns a one-to-one mapping matrix between two vocabularies based on token co-occurrence information, efficiently replacing LLM vocabularies to realize cross-lingual knowledge transfer and cross-model token-level distillation.

**[Tokenization is Sensitive to Language Variation](tokenization_is_sensitive_to_language_variation.md)**

:   This paper systematically investigates the contrasting impacts of three key design choices of BPE tokenizers (fitting corpus, pre-tokenizer, and vocabulary size) on downstream performance across tasks requiring language variation robustness versus sensitivity. It proposes a task-aware tokenizer evaluation metric based on logistic regression, which significantly outperforms task-agnostic metrics such as Rényi efficiency.

**[Towards Effective and Efficient Continual Pre-training of Large Language Models](towards_effective_and_efficient_continual_pre-training_of_large_language_models.md)**

:   This work systematically investigates data strategies for the continual pre-training of Llama-3 (8B). By employing three primary strategies—topic-based data mixture, perplexity-based curriculum learning, and high-quality synthetic scientific QA data—the proposed approach significantly enhances Chinese capabilities (C-Eval +8.81) and scientific reasoning (MATH +12.00) using only 100B tokens, while effectively maintaining the original English capabilities.

**[Training Dynamics Underlying Language Model Scaling Laws: Loss Deceleration and Zero-Sum Learning](training_dynamics_underlying_language_model_scaling_laws_loss_deceleration_and_z.md)**

:   This work discovers the "loss deceleration" phenomenon in language model training—where the loss curves undergo a piecewise linear transition in log-log space. The root cause is identified as "zero-sum learning" (ZSL), where systematic opposition in per-token gradients leading to destructive interference offsets improvements in some tokens with deterioration in others. Scaling up mitigates ZSL by lowering the deceleration-onset loss $L_d$ and increasing the post-deceleration slope $r_d$, providing a directly actionable mechanism to bypass scaling law bottlenecks.

**[Unsupervised Morphological Tree Tokenizer](unsupervised_morphological_tree_tokenizer.md)**

:   TreeTok is proposed, an unsupervised neural morphological structure induction tokenizer. It learns character-level tree structures through a MorphOverriding mechanism and self-supervised objectives, performing tokenization via top-down vocabulary matching. It outperforms BPE/WordPiece on both morphological segmentation and language modeling tasks.

**[Velocitune: A Velocity-based Dynamic Domain Reweighting Method for Continual Pre-training](velocitune_a_velocity-based_dynamic_domain_reweighting_method_for_continual_pre-.md)**

:   The Velocitune framework is proposed to dynamically adjust sampling weights of different data domains during continual pre-training based on learning velocity. It prioritizes domains with slower learning progress and estimates target losses cost-effectively using scaling laws, significantly outperforming static mixing baselines on mathematical/code reasoning and system command generation tasks.

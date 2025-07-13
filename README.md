# Fine-Tuning Gemma for English-German Translation

This project demonstrates fine-tuning a large language model (LLM) for English-to-German translation, including evaluation, synthetic data generation, and performance comparison across multiple training strategies. The workflow is designed for efficient experimentation on a Google Colab T4 GPU environment.

## Features

- Uses a pre-trained general-purpose LLM (Gemma-2B) for translation tasks.
- Loads and preprocesses a benchmark English-German translation dataset.
- Implements prompt-based translation and evaluation with BLEU score (sacrebleu).
- Fine-tunes the model using PEFT/LoRA for resource efficiency.
- Generates synthetic translation data using a larger LLM for data augmentation.
- Compares model performance before and after fine-tuning, with and without synthetic data.
- Includes code for reproducible experiments and performance visualization.

## Dataset

- **Source:** [Darth-Vaderr/English-German](https://huggingface.co/datasets/Darth-Vaderr/English-German)
- **Size:** 1,500 sentence pairs (randomly sampled)
- **Splits:** 
  - Train: 64%
  - Validation: 16%
  - Test: 20%

## Model

- **Base Model:** `google/gemma-2b-it` (2B parameters, causal LM)
- **Fine-Tuning:** Parameter-efficient fine-tuning (PEFT/LoRA)
- **Tokenizer:** AutoTokenizer from Hugging Face Transformers

## Workflow

1. **Environment Setup:** Installs required libraries (`transformers`, `datasets`, `evaluate`, `sacrebleu`).
2. **Authentication:** Uses Hugging Face token for model access.
3. **Data Loading:** Loads and splits the English-German dataset.
4. **Prompt Engineering:** Formats translation prompts for both inference and training.
5. **Baseline Evaluation:** Evaluates the vanilla model on the test set using BLEU.
6. **Fine-Tuning:** Trains the model on the training set, validates on the validation set.
7. **Synthetic Data Generation:** (Planned) Uses a larger LLM to generate additional translation pairs for data augmentation.
8. **Performance Comparison:** Evaluates and compares BLEU scores across different training strategies.
9. **Visualization:** Plots model performance for easy comparison.

## Usage

1. **Run in Google Colab:**  
   Open `FineTuningEN_DE.ipynb` in Google Colab (T4 GPU recommended).

2. **Set Hugging Face Token:**  
   Save your Hugging Face token in Colab's `userdata` as `HF_TOKEN`.

3. **Install Dependencies:**  
   The notebook will automatically install required Python packages.

4. **Execute Cells:**  
   Follow the notebook cells sequentially for data loading, model evaluation, fine-tuning, and analysis.

## Evaluation

- **Metric:** BLEU score (via `sacrebleu`)
- **Baseline:** Evaluates the pre-trained model before fine-tuning.
- **Post-Tuning:** Evaluates after fine-tuning and after data augmentation.

## Requirements

- Python 3.8+
- Google Colab (T4 GPU, 15GB VRAM recommended)
- Hugging Face account and access token

## References

- [Gemma Model Card](https://huggingface.co/google/gemma-2b-it)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PEFT/LoRA](https://github.com/huggingface/peft)
- [SacreBLEU](https://github.com/mjpost/sacrebleu)

---

For questions or contributions, please open an issue or pull request.
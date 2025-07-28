# News Topic Classifier with Fine-tuned BERT

## Overview:

This project demonstrates the fine-tuning of a pre-trained BERT (Bidirectional Encoder Representations from Transformers) model for news topic classification. Utilizing the Hugging Face Transformers library, the model is trained on a subset of the AG News dataset and deployed with a simple, interactive Gradio interface.

## Objective of the Task:

The main objectives of this project were:

1.  **Develop a News Topic Classifier:** Build a machine learning model capable of categorizing news headlines into predefined topics (World, Sports, Business, Sci/Tech).
2.  **Utilize Transfer Learning:** Leverage the power of a large pre-trained language model (BERT) through fine-tuning on a specific task.
3.  **Create an Interactive Demo:** Provide a user-friendly interface to easily test the model's predictions.

## Methodology / Approach:

### 1. Data Acquisition and Preprocessing:

-   **Dataset:** The AG News Classification Dataset retrived from Hugging Face
-   **Data Subset:** To significantly reduce training time and manage memory, a smaller subset of the original dataset was sampled:
    -   **Training Set:** 10,000 samples (from 120,000)
    -   **Evaluation Set:** 1,000 samples (from 7,600)
-   **Tokenization:** The `bert-base-uncased` tokenizer from Hugging Face was used to convert text into numerical input IDs, along with attention masks and token type IDs. Padding and truncation were applied to handle variable sequence lengths.
-   **Dataset Preparation:** Columns were streamlined, the 'label' column was renamed to 'labels' (as required by Hugging Face `Trainer`), and the dataset format was set to PyTorch tensors.

### 2. Model Selection and Loading:

-   **Base Model:** `bert-base-uncased` was chosen for its balance of performance and size.
-   **Model Head:** `AutoModelForSequenceClassification` was used to load BERT with a pre-trained base and a randomly initialized classification head, suitable for fine-tuning on our 4 news categories.

### 3. Training Strategy:

**Framework:** Hugging Face `Trainer` API was utilized to manage the training loop, simplifying common tasks like optimization, evaluation, and checkpointing.
-   **Hyperparameters (Optimized for Speed & RAM):**
    -   `num_train_epochs`: **1** (to ensure the fastest possible training run).
    -   `per_device_train_batch_size`: **16** (chosen specifically to manage GPU VRAM consumption).
    -   `per_device_eval_batch_size`: **16**.
    -   `learning_rate`: `2e-5` (a standard effective learning rate for BERT fine-tuning).
    -   `weight_decay`: `0.01`.
    -   `evaluation_strategy`: `epoch` (evaluate after each epoch).
    -   `metric_for_best_model`: `f1` (F1-score was used to determine the best model).
-   **Metrics:** Accuracy and F1-score (weighted average for multi-class) were used for evaluation.

### 4. Tools and Environment:

-   **Primary Environment:** Kaggle Notebooks.
-   **Libraries:** `transformers`, `datasets`, `evaluate`, `gradio`, `torch`, `numpy`.

### 5. Deployment:

-   **Interactive Interface:** A simple web-based demo was created using `Gradio` to allow users to input news headlines and get real-time topic predictions. The `share=True` option was used to generate a public URL for access from remote notebooks.

## Key Results or Observations:

* **Efficient Fine-tuning:** By strategically selecting a smaller dataset (10,000 training samples) and a minimal number of epochs (1), training time was drastically reduced, making iterative development much faster.
* **VRAM Management:** The choice of `per_device_train_batch_size=16` successfully prevented CUDA Out of Memory errors in environments with moderate VRAM.
* **Kaggle T4 Performance:** The fine-tuning process on the Kaggle T4 GPU (with 10,000 samples, 1 epoch, batch size 16) completed very quickly (typically 5-15 minutes), demonstrating the power of dedicated GPU resources.
* **Functional Model:** Despite training for only 1 epoch on a small subset, the model successfully learned to distinguish between the four news categories, providing a strong baseline performance. Further epochs and more data would enhance accuracy.
* **Interactive Demo:** The Gradio interface successfully launched and provided an intuitive way to test the model's predictions, confirming the end-to-end functionality of the pipeline.

## How to Run This Project:

1.**Open in Kaggle/Colab:** Upload the `.ipynb` notebook file to a new Kaggle Notebook or Google Colab session.

2.  **Enable GPU:** In Kaggle Notebooks, go to "Settings" (right sidebar) and set "Accelerator" to "GPU". In Google Colab, go to `Runtime > Change runtime type` and select "GPU".

3.  **Run All Cells:** Execute all cells in the notebook from top to bottom. The notebook is designed to guide you through installations, data loading, model training, evaluation, and finally, launching the Gradio demo.

4.  **Interact with Gradio:** After the final cell executes, look for a public URL (e.g., `https://xxxxxx.gradio.live`) in the cell output. Click this URL to open the interactive classifier in a new tab.
   




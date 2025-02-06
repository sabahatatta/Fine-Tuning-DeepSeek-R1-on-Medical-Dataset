# Fine-Tuning DeepSeek R1 on Medical Dataset (**medical-o1-reasoning-SFT**)

This repository provides a comprehensive guide to fine-tuning **DeepSeek R1** efficiently using **Kaggle**, **Unsloth**, **Hugging Face**, and **Weights & Biases**. The goal is to fine-tune the model while optimizing memory usage and training speed, making it accessible even with limited computational resources.

---
## 🌟 Features
- **Kaggle’s Free GPUs**: Leverage Kaggle's high-performance GPUs for cost-effective training.
- **Unsloth Optimization**: Fine-tune models with improved memory efficiency and faster training speeds.
- **Weights & Biases Integration**: Track experiments, log metrics, and visualize performance seamlessly.
- **LoRA Adapters**: Reduce VRAM usage while maintaining model performance through parameter-efficient fine-tuning.
- **Hugging Face Integration**: Access pre-trained models and deploy fine-tuned models effortlessly.

---
## 🛠️ Prerequisites
- A **Kaggle** account with access to free GPUs.
- API keys for **Hugging Face** and **Weights & Biases** (stored securely in Kaggle Secrets).
- Basic knowledge of Python, PyTorch, and Hugging Face Transformers.

---
## ⚙️ Setup Instructions
### 1. Create a Kaggle Notebook
- Open a new notebook on Kaggle.
- Navigate to **Add-ons > Secrets** and add:
  - **Hugging Face API Key** (`HF_API_KEY`)
  - **Weights & Biases API Key** (`WANDB_API_KEY`)

### 2. Install Dependencies
Install the required libraries:
```bash
pip install unsloth transformers peft wandb
```

### 3. Authenticate with Hugging Face
Authenticate your Kaggle notebook with Hugging Face:
```python
from huggingface_hub import login
import os

hf_token = os.getenv("HF_API_KEY")  # Retrieve API key from Kaggle Secrets
login(hf_token)
```

### 4. Set Up Weights & Biases
Initialize Weights & Biases for experiment tracking:
```python
import wandb

wandb.login(key=os.getenv("WANDB_API_KEY"))
wandb.init(project="deepseek-r1-finetune")
```

### 5. Load the Model
Use **Unsloth** to load **DeepSeek R1** with optimized performance:
```python
from unsloth import FastLlama

model = FastLlama.from_pretrained("deepseek-ai/deepseek-r1-distill-8b")
```

### 6. Load the Tokenizer
Load the tokenizer for preprocessing:
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-r1-distill-8b")
```

---
## 🏗️ Fine-Tuning Process
The fine-tuning process involves the following steps:

### 1. **LoRA Adapters**
Reduce memory usage while preserving model performance using LoRA:
```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8, lora_alpha=32, lora_dropout=0.1, task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
```

### 2. **Dataset Preparation**
Prepare your dataset by tokenizing and formatting it correctly:
```python
def preprocess_data(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = raw_dataset.map(preprocess_data, batched=True)
```

### 3. **Training Loop**
Train the model using Hugging Face’s `Trainer` API:
```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_dir="./logs",
    report_to="wandb",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,  # Replace with your tokenized dataset
)

trainer.train()
```

### 4. **Experiment Tracking**
Weights & Biases logs loss curves, hyperparameters, and performance metrics for easy analysis.

---
## 📊 Results
- **Efficiency**: Fine-tuning is significantly faster and more memory-efficient than traditional methods.
- **Performance**: Loss curves and training metrics are logged in Weights & Biases for real-time monitoring.
- **Accuracy**: The fine-tuned model demonstrates improved accuracy on domain-specific datasets.

---
## 🚀 Deployment
Once fine-tuned, push the model and tokenizer to Hugging Face for inference:
```python
model.push_to_hub("your-hf-username/deepseek-r1-finetuned")
tokenizer.push_to_hub("your-hf-username/deepseek-r1-finetuned")
```

---
## 🌟 Next Steps
- **Deploy the Model**: Use the fine-tuned model for inference in production environments.
- **Experiment with Datasets**: Fine-tune on additional datasets for domain adaptation.
- **Optimize Hyperparameters**: Experiment with learning rates, batch sizes, and other parameters for better performance.
- **Explore Advanced Techniques**: Investigate other parameter-efficient fine-tuning methods like QLoRA or P-Tuning.

---
## 🤝 Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`feature/your-feature-name`).
3. Commit your changes and push to the branch.
4. Open a pull request with a detailed description of your enhancements.

---
## 📜 License
This project is open-source and available under the **MIT License**.

---
Feel free to reach out for feedback or suggestions to make this project even better! 😊

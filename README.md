# Chakra LLM

**Chakra** is a minimal GPT-like language model written from scratch in **PyTorch**.  
It combines both the **model architecture** and a **training pipeline**, making it easy to understand, modify, and extend.

Created by **Jagrit Verma** (2025).  

---

## 🔹 Features
-  GPT-style **Transformer architecture**  
-  **Causal masked multi-head attention**  
-  **Pre-Norm Transformer Blocks**  
-  **MLP with GELU**  
-  **Learned token + position embeddings**  
-  **Weight tying** (`lm_head <-> token_embed`)  
-  **Tokenizer integration** via [tiktoken](https://github.com/openai/tiktoken)  
-  **Training pipeline** (AdamW optimizer, LR scheduler, checkpointing)  
-  **Config system** with JSON (hyperparameters)  
-  **Perplexity calculation during training**  

---

## 🔹 Installation
Clone the repo and install dependencies:

```bash
git clone https://github.com/<your-username>/chakra-llm.git
cd chakra-llm
pip install -r requirements.txt

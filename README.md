
# 🌍 Multilingual Sentiment Analysis API

A **scalable REST API** for sentiment analysis in **100+ languages**, built with **FastAPI**.
Powered by **Hugging Face’s XLM-RoBERTa** and the **Google Generative AI API**, this project handles **long-form text (3500+ characters)** with **custom recursive chunking** and **aggregation logic** for accurate sentiment scoring.

---

## ✨ Features

* **🌍 Multilingual Support** – Works with over 100 languages.
* **⚡ High Performance** – FastAPI ensures low-latency and concurrency handling.
* **📄 Large Text Handling** – Custom chunking algorithm preserves meaning for long inputs.
* **🤖 Advanced NLP Models** – State-of-the-art XLM-RoBERTa integration.
* **🔗 AI Integration** – Optional Google Generative AI API for enhanced context.
* **🛠 Scalable Design** – Easily deployable to AWS, Docker, or any cloud platform.

---

## 🛠 Tech Stack

* **Language:** Python
* **Framework:** FastAPI
* **NLP Tools:** Hugging Face Transformers, NLTK, SpaCy
* **Models:** XLM-RoBERTa, Google Generative AI API
* **Utilities:** Git

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/IrfanaNizar/multilingual-sentiment-analysis-api.git

# Navigate into the project folder
cd multilingual-sentiment-analysis-api

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

Run the API:

```bash
python run.py
```

API will be live at:

```
http://127.0.0.1:8000
```

Example request (POST):

```json
{
  "text": "I love this product! The quality is amazing."
}
```

---

## 📌 Use Cases

* **Customer Feedback Analysis**
* **Social Media Sentiment Tracking**
* **Market Research Insights**
* **Content Moderation**

---

## 🏷 Keywords

`python`, `fastapi`, `sentiment-analysis`, `multilingual`, `huggingface`, `xlm-roberta`, `google-generative-ai`, `nlp`, `text-analysis`, `ai`, `machine-learning`, `backend`

---


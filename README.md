# Soybean Decision Support System (DSS)

## 🌱 Overview

The **Soybean Decision Support System (DSS)** is a **machine learning-powered** web application that predicts soybean **yield** and **protein content** based on key agricultural parameters. Farmers, researchers, and agronomists can use these insights to **optimize productivity** and **enhance crop quality**.

## ✨ Features

✅ **Interactive UI** – Select input presets or manually adjust parameters.  
✅ **Real-time Predictions** – Get instant soybean yield & protein estimates.  
✅ **Easy-to-Read Insights** – Understand predictions with helpful **tips & visualizations**.  
✅ **Multiple Presets** – Test different conditions (High/Low Yield & Protein).  
✅ **Robust Model** – Trained on real-world **agricultural data** for accuracy.  

## 📂 Project Structure

```
soybean-dss/
├── data/
│   └── soybean_data.csv        # Dataset used for training
├── model/
    ├── soybean_model.pkl       # Trained ML model
    ├── scaler.pkl  
    └── train_model.py              # Model training script        # Scaler for feature normalization
├── logos/
│   └── train_model.py.png  
├── app.py                      # Streamlit web application
├── requirements.txt            # Dependencies
└── LICENSE                     # License File
└── README.md                   # Project documentation
```

## 🛠 Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/ashishpatel8736/soybean-dss.git
cd soybean-dss
```

### 2️⃣ Create & Activate Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the App

```bash
streamlit run app.py
```

## 📊 Interpretation Guide

| Yield Prediction | Meaning |
|-----------------|---------|
| **High Yield** (> 6000 kg/ha) | Excellent productivity 🚀 |
| **Average Yield** (3000 - 6000 kg/ha) | Good, but may need adjustments 📊 |
| **Low Yield** (< 3000 kg/ha) | Requires significant improvement ❗ |

| Protein Content Prediction | Meaning |
|---------------------------|---------|
| **High Protein** (< 0.7) | Excellent nutritional quality 💪 |
| **Moderate Protein** (0.7 - 1.2) | Acceptable balance ⚖️ |
| **Low Protein** (> 1.2) | Needs improvement! Consider fertilizers 🌿 |

## 👤 Author  
**Ashish Patel**  
[![GitHub](https://github.com/ashishpatel8736/soybean-dss/blob/main/logos/icons8-github-50.png)](https://github.com/ashishpatel8736) | [![LinkedIn](https://img.icons8.com/ios-filled/50/0077b5/linkedin.png)](https://www.linkedin.com/in/ashishpatel8736)


## 📢 Contributing

💡 Found a bug? Have an idea? Feel free to open an issue or submit a pull request!  

## 📜 License

Distributed under the MIT License. See `LICENSE` for details.

---

🚀 *Happy Farming & Smart Decision-Making!* 🌾

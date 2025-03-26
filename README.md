# Drug Sentiment Analysis Dashboard

## 📌 Project Overview
This project is a **Streamlit-based dashboard** for analyzing drug reviews, performing **sentiment analysis**, and predicting future drug usage trends. It uses **NLP, data visualization, and time-series forecasting** to provide insights into drug effectiveness and user sentiments.

## 🚀 Features
- **Sentiment Analysis:** Uses **TextBlob** to analyze user reviews and determine sentiment polarity.
- **Interactive Visualizations:** Displays **histograms** and **word clouds** for selected drugs.
- **Time-Series Forecasting:** Predicts future drug usage trends using **Facebook Prophet**.
- **User-Friendly UI:** Built with **Streamlit** for easy interaction.

## 🛠️ Tech Stack
- **Programming Language:** Python
- **Libraries:** Streamlit, Pandas, Matplotlib, Seaborn, TextBlob, Scikit-Learn, Prophet, WordCloud
- **Machine Learning Models:** Sentiment Analysis & Time-Series Forecasting

## 📂 File Structure

## 📊 Dataset
The dataset used for this project is available on Kaggle:
[Drug Reviews Dataset - KUC Hackathon Winter 2018](https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018)
```
├── app.py                  # Main Streamlit app
├── requirements.txt        # Dependencies
├── drugsComTest_raw.csv    # Drug reviews dataset
```

## ⚡ Installation & Setup
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/kit-rak/drug-analysis.git
cd drug-analysis
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit App
```bash
streamlit run app.py
```

## 📊 How It Works
1. **Select drugs** from the sidebar dropdown.
2. **View sentiment analysis** histograms.
3. **Generate word clouds** for drug reviews.
4. **Predict future usage trends** with Prophet.

## 📬 Contact
- **LinkedIn:** [kit-rak](https://www.linkedin.com/in/kit-rak)
- **GitHub:** [kit-rak](https://github.com/kit-rak)

🚀 **Happy Analyzing!**

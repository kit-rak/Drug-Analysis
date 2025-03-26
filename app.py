import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from prophet import Prophet
from wordcloud import WordCloud

# Load Dataset
@st.cache_data
def load_data():
    # url = "https://www.kaggleusercontent.com/datasets/jessicali9530/kuc-hackathon-winter-2018/downloads/drugsComTrain_raw.tsv"
    data = pd.read_csv("drugsComTest_raw.csv")
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    return data

data = load_data()

# Sidebar for User Input
st.sidebar.title("Drug Sentiment Analysis")
selected_drugs = st.sidebar.multiselect(
    "Select Drugs for Analysis",
    options=data["drugName"].unique(),
    default=data["drugName"].unique()[:5]
)

# Filter data
filtered_data = data[data["drugName"].isin(selected_drugs)]

# Sentiment Analysis
def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

st.title("Drug Sentiment Analysis Dashboard")
st.write("Analyze user reviews about drugs and predict their future trends.")

# Sentiment Score Calculation
filtered_data['sentiment_score'] = filtered_data['review'].apply(analyze_sentiment)
st.subheader("Sentiment Analysis")
sentiment_fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(filtered_data, x="sentiment_score", hue="drugName", multiple="stack", ax=ax)
st.pyplot(sentiment_fig)

# Visualization: Word Clouds
st.subheader("Word Cloud of Reviews")
wordcloud_images = {}
for drug in selected_drugs:
    st.write(f"**{drug}**")
    reviews = " ".join(filtered_data[filtered_data["drugName"] == drug]["review"].tolist())
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(reviews)
    wordcloud_images[drug] = wordcloud
    st.image(wordcloud.to_array(), width=900)  # Display word clouds with consistent size

# Time-Series Prediction using Prophet
st.subheader("Drug Usage Prediction")
drug_usage = (
    filtered_data.groupby(['date', 'drugName']).size().reset_index(name='counts')
)
predictions = {}
for drug in selected_drugs:
    st.write(f"**{drug}**")
    drug_data = drug_usage[drug_usage['drugName'] == drug]
    if not drug_data.empty:
        df_prophet = drug_data.rename(columns={"date": "ds", "counts": "y"})
        model = Prophet()
        model.fit(df_prophet)
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)
        predictions[drug] = forecast
        fig = model.plot(forecast)
        st.pyplot(fig)

# # Prediction Results
# st.subheader("Predicted Usage Trends")
# for drug, forecast in predictions.items():
#     st.write(f"**{drug}**")
#     st.line_chart(forecast.set_index("ds")[["yhat"]])

📰 Fake News Detection Model

📌 Overview

This project is a Machine Learning-based Fake News Detection Model that classifies news articles as FAKE or REAL using Natural Language Processing (NLP) techniques. It is trained on TF-IDF vectorized text data and achieves 83% accuracy using a Naïve Bayes classifier.

🚀 Features
✅ Classifies news articles as FAKE or REAL
✅ Uses TF-IDF Vectorization for text representation
✅ Trained using Multinomial Naïve Bayes
✅ Automatically removes stopwords, punctuation, and URLs
✅ Achieves 83% accuracy

🛠️ Technologies Used
Python – (for model development)
Tweepy – (for real-time news collection)
Scikit-learn – (for model training & evaluation)
NLTK – (for text preprocessing)
Pandas & NumPy – (for data handling)
Matplotlib & Seaborn – (for visualization)
📂 Dataset Used
The model is trained on a labeled dataset containing real and fake news articles in a single CSV file.

Columns: title, text, label (FAKE/REAL)
Label Encoding: FAKE = 0, REAL = 1
📊 How It Works
1️⃣ Loads and cleans the dataset (removes stopwords, punctuation, and URLs)
2️⃣ Converts text into numerical format using TF-IDF Vectorization
3️⃣ Trains a Naïve Bayes classifier on the processed data
4️⃣ Predicts whether a news article is FAKE or REAL

⚙️ Installation & Usage
1️⃣ Install Required Dependencies
pip install pandas numpy scikit-learn nltk matplotlib seaborn

2️⃣ Run the Model
python fake_news_detection.py

3️⃣ Input a Custom News Article for Prediction
Modify the script to input a custom news article:
news_text = "Breaking: Scientists discover a new planet with water!"
news_text = clean_text(news_text)  # Apply the same preprocessing
news_tfidf = vectorizer.transform([news_text])  # Convert to TF-IDF
prediction = model.predict(news_tfidf)

print(f"Prediction: {'REAL' if prediction[0] == 1 else 'FAKE'}")

📈 Model Performance
Accuracy Achieved: 83% (Naïve Bayes)
Precision, Recall, F1-Score: Report available in model_evaluation.txt

📊 Visualizations Included
Confusion Matrix: Shows misclassification rate
WordCloud: Displays common words in Fake & Real news
Bar Graphs: Compare Real vs. Fake articles in the dataset
🔹 Future Enhancements
🔸 Improve accuracy using Logistic Regression or LSTM
🔸 Use Word2Vec/BERT embeddings instead of TF-IDF
🔸 Deploy as a web app using Flask or FastAPI
🔸 Integrate real-time news validation using Twitter API

📜 License
This project is open-source and free to use for educational purposes.

# Amazon Review Sentiment Analysis 🛒💬

This project performs sentiment analysis on Amazon product reviews using Natural Language Processing (NLP) techniques. It classifies customer feedback as either **positive (1)** or **negative (0)** using a Naive Bayes classifier.

---

## 📁 Files Included

- `TSA.ipynb`: Jupyter notebook containing the full analysis pipeline
- `amazon_reviews.csv`: Dataset of verified Amazon reviews and feedback labels

---

## 📊 Dataset

The dataset contains the following columns:

- **verified_reviews**: Customer review text
- **feedback**: Target label (1 = positive, 0 = negative)

---

## 🔧 Project Steps

1. **Text Preprocessing**  
   - Lowercasing, punctuation & digit removal  
   - Tokenization  
   - Stopword removal using `ENGLISH_STOP_WORDS`

2. **Handling Class Imbalance**  
   - Applied upsampling on the minority class using `sklearn.utils.resample`

3. **Model Building**  
   - Used `CountVectorizer` for feature extraction  
   - Trained a **Multinomial Naive Bayes** classifier  
   - Evaluated performance with classification metrics

4. **Model Evaluation**  
   - Accuracy: **95%**  
   - Balanced precision and recall for both classes

---

## 📈 Performance

| Metric     | Negative (0) | Positive (1) |
|------------|--------------|--------------|
| Precision  | 0.91         | 0.98         |
| Recall     | 0.98         | 0.91         |
| F1-Score   | 0.95         | 0.95         |

Overall **accuracy: 95%**

---

## 🚀 Future Improvements

- Use **TF-IDF** vectorization
- Experiment with other models: Logistic Regression, SVM, Random Forest
- Try deep learning models like LSTM or fine-tuned BERT
- Deploy as a simple web app using Flask/Streamlit

---

## 📌 Requirements

```bash
pandas
numpy
scikit-learn
matplotlib

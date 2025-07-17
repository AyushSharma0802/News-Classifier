📰 News Article Classifier

A Machine Learning project that classifies news articles into categories using Naive Bayes and Support Vector Machine (SVM). This is built using Python, Scikit-learn, and TF-IDF vectorization
 📂 Dataset
* Source: https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset
* Format: CSV with the following columns:

  * `Class Index` (category label)
  * `Title` (headline of the news article)
  * `Description` (summary of the article)

Combine `Title` and `Description` for better classification results.
Tech Stack

* Python 3
* Pandas, NumPy
* Scikit-learn
* TF-IDF Vectorizer

Project Steps

1. Data Loading using Pandas
2. Preprocessing: Combine `Title` and `Description` into one text column
3. Vectorization: TF-IDF vectorizer to convert text into numerical form
4. Train-Test Split: (80/20)
5. Model Training using:

     Multinomial Naive Bayes
     And
     Support Vector Machine (SVM)
7. Evaluation: Accuracy and classification report



 Results

| Model       | Accuracy |
| ----------- | -------- |
| Naive Bayes | 89%      |
| SVM         | 88%      |

 Sample Code Snippet

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Combine text
df['text'] = df['Title'] + " " + df['Description']
X = df['text']
y = df['Class Index']

# Vectorize
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_vec = vectorizer.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Naive Bayes
nb = MultinomialNB()
nb.fit(X_train, y_train)

# SVM
svm = LinearSVC()
svm.fit(X_train, y_train)
```


Author

Ayush Sharma


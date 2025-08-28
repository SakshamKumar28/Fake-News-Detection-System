import pandas as pd
import joblib
import os
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Read the data
df = pd.read_csv("FakeNewsNet.csv")

# 2. Using stratified shuffle split to create training set and testing set
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(df, df['real']):
    strat_train_set = df.iloc[train_index].copy()
    strat_test_set = df.iloc[test_index].copy()

# 3. Filling missing values in traing and testing set
strat_train_set.fillna({"news_url":"missing"}, inplace=True)
strat_train_set.fillna({"source_domain":"missing"}, inplace=True)

strat_test_set.fillna({"news_url":"missing"}, inplace=True)
strat_test_set.fillna({"source_domain":"missing"}, inplace=True)

# 4. Creating test and train dataframe for the model using onl title and real column
train = strat_train_set[['title', 'real']].reset_index(drop=True).copy()
test  = strat_test_set[['title', 'real']].reset_index(drop=True).copy()

# 4. Setting empty string in title column if there is any missing value
train['title'] = train['title'].fillna('')
test['title']  = test['title'].fillna('')

# 4. Removing Duplicate titles
before = len(train)
train = train.drop_duplicates(subset='title')
after = len(train)

# 5. Seperating features and labels
x_train, y_train = train['title'], train['real']
x_test, y_test = test['title'], test['real']

# 6. Making a pipeline

word_vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1,3),
    max_features=10000
)


char_vectorizer = TfidfVectorizer(
    analyzer='char_wb',
    ngram_range=(3,5),
    max_features=5000
)


pipeline = Pipeline([
    ('features', FeatureUnion([
        ('word', word_vectorizer),
        ('char', char_vectorizer)
    ])),
    ('clf', LinearSVC(class_weight="balanced"))

])

pipeline.fit(x_train, y_train)
# pred = pipeline.predict(x_test)


if not os.path.exists("fake_news_model.pkl"):
    joblib.dump(pipeline, "fake_news_model.pkl")
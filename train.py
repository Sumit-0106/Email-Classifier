from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Example data (replace with your real dataset)
X = ["free money now", "hello friend", "win cash prize"]
y = [1, 0, 1]  # 1 = spam, 0 = not spam

# vectorizer
tfidf = TfidfVectorizer()
X_vector = tfidf.fit_transform(X)

# model
model = MultinomialNB()
model.fit(X_vector, y)

# save both
pickle.dump(tfidf, open("vectorizer.pkl", "wb"))
pickle.dump(model, open("model.pkl", "wb"))

print("Training done ✅")
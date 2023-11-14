import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from tqdm import tqdm 

df=pd.read_csv('spam.csv',encoding='latin-1')
# df.dropna(inplace=True)
df.drop_duplicates(inplace= True)
df['label']=df['v1'].map({'ham':'ham','spam':'spam'})
x=df['v2']
y=df['label']
x_train,x_test ,y_train , y_test = train_test_split(x,y,test_size=0.2 ,random_state= 42)
tfidf_vectorizer =TfidfVectorizer()
x_train_tfidf =tfidf_vectorizer.fit_transform(x_train)
clf = MultinomialNB()
clf.fit(x_train_tfidf , y_train)
x_test_tfidf =tfidf_vectorizer.transform(x_test)
y_pred = clf.predict(x_test_tfidf)
accuracy = accuracy_score(y_test ,y_pred)
report = classification_report(y_test , y_pred , target_names =['Legitimate SMS ','Spam SMS'])
progress_bar =tqdm(total=100,position=0,leave=True)
for i in range (10,101,10):
    progress_bar.update(10)
    progress_bar.set_description(f'Progress :{i}%')

progress_bar.close()
print(f'Accuracy: {accuracy:.2f}')
print('Classification Report :')
print(report )
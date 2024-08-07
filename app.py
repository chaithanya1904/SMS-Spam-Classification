from flask import Flask,render_template,request
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
nltk.download('punkt')
ps = PorterStemmer()

def transform_text(text):
    
    text = text.lower()
    
    text = nltk.word_tokenize(text)
    
    temp=[]
    for word in text:
        if word.isalnum():
            temp.append(word)
            
    text=temp[:]
    temp.clear()
    
    for word in text:
         if word not in stopwords.words('english') and word not in string.punctuation:
                    temp.append(word)
                    
    text=temp[:]
    temp.clear()
    
    porter=PorterStemmer()
    for word in text:
        temp.append(porter.stem(word))
    
    return " ".join(temp) 

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('classifier.pkl','rb'))

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict",methods=['POST'])
def predict():
    msg = request.form['sms']
    msg = transform_text(msg)
    msg = tfidf.transform([msg])
    pred = model.predict(msg)
    if pred[0]==1:
        return render_template('predict.html',msg="Spam Message")
    else:
        return render_template('predict.html',msg="Not a Spam Message")
    
if __name__ == "__main__":
    app.run(debug=True)

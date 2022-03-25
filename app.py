import pickle
from flask import Flask,render_template,url_for,request

# load the pickled model
file_name='nlp_model.pkl'
cv=pickle.load(open('transform.pkl','rb'))
clf=pickle.load(open(file_name,'rb'))

app=Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        message=request.form['message']
        data=[message]
        vect=cv.transform(data).toarray()
        my_predction=clf.predict(vect)
    return render_template('result.html',prediction=my_predction)

if __name__=='__main__':
    app.run(debug=True)

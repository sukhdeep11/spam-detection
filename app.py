from flask import Flask, request, render_template
import pickle

tfid = pickle.load(open('models/vectorizer.pkl', 'rb'))
model_MNB = pickle.load(open('models/model.pkl', 'rb'))
print(tfid)

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    txt = request.form.get("data")
    transformed_txt = [txt]

    print(txt)
    print(transformed_txt)
    a = tfid.transform(transformed_txt)
    result = model_MNB.predict(a)

    print(result)
    print(result[0])
    type(result)

    if result[0] == 1:
        res = "It looks like a Spam Message"
    else:
        res = "It looks like Valid Message"

    return render_template('index.html', prediction_text=res)


if __name__ == "__main__":
    app.run()

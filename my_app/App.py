from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
classifierNB = pickle.load(open('modelos/classifierNB.sav', 'rb'))
vectorNB = pickle.load(open('modelos/vectorNB.sav', 'rb'))
classifierKNN = pickle.load(open('modelos/classifierKNN.sav', 'rb'))
vectorKNN = pickle.load(open('modelos/vectorKNN.sav', 'rb'))
classifierSVM = pickle.load(open('modelos/classifierSVM.sav', 'rb'))
vectorSVM = pickle.load(open('modelos/vectorSVM.sav', 'rb'))

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predecirNB", methods=['POST'])
def predecirNB():
    texto = request.form['TextoNB']
    vector = vectorNB.transform([texto])
    prediccion = classifierNB.predict(vector)
    if prediccion[0] == 0:
        resultado = "https://cdn-0.emojis.wiki/emoji-pics/google/face-without-mouth-google.png"
    elif prediccion[0] == 1:
        resultado = "https://images.emojiterra.com/google/noto-emoji/v2.034/512px/1f622.png"
    elif prediccion[0] == 2:
        resultado = "https://cdn-0.emojis.wiki/emoji-pics/google/neutral-face-google.png"
    else:
        resultado = "https://images.emojiterra.com/google/android-10/512px/1f60a.png"

    return render_template('index.html', prediccion_textoNB = resultado)


@app.route("/predecirkNN", methods=['POST'])
def predecirkNN():
    texto = request.form['TextokNN']
    vector = vectorKNN.transform([texto])
    prediccion = classifierKNN.predict(vector)
    if prediccion[0] == 0:
        resultado = "https://cdn-0.emojis.wiki/emoji-pics/google/face-without-mouth-google.png"
    elif prediccion[0] == 1:
        resultado = "https://images.emojiterra.com/google/noto-emoji/v2.034/512px/1f622.png"
    elif prediccion[0] == 2:
        resultado = "https://cdn-0.emojis.wiki/emoji-pics/google/neutral-face-google.png"
    else:
        resultado = "https://images.emojiterra.com/google/android-10/512px/1f60a.png"

    return render_template('index.html', prediccion_textokNN = resultado)


@app.route("/predecirSVM", methods=['POST'])
def predecirSVM():
    texto = request.form['TextoSVM']
    vector = vectorSVM.transform([texto])
    prediccion = classifierSVM.predict(vector)
    if prediccion[0] == 0:
        resultado = "https://cdn-0.emojis.wiki/emoji-pics/google/face-without-mouth-google.png"
    elif prediccion[0] == 1:
        resultado = "https://images.emojiterra.com/google/noto-emoji/v2.034/512px/1f622.png"
    elif prediccion[0] == 2:
        resultado = "https://cdn-0.emojis.wiki/emoji-pics/google/neutral-face-google.png"
    else:
        resultado = "https://images.emojiterra.com/google/android-10/512px/1f60a.png"

    return render_template('index.html', prediccion_textoSVM = resultado)

if __name__ == "__main__":
    app.run()

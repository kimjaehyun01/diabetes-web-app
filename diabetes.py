import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import os
from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap5
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

# Flask 앱 초기화
app = Flask(__name__)
app.config['SECRET_KEY'] = 'hard to guess string'
bootstrap5 = Bootstrap5(app)

# Flask-WTF 폼 클래스
class LabForm(FlaskForm):
    preg = StringField('# Pregnancies', validators=[DataRequired()])
    glucose = StringField('Glucose', validators=[DataRequired()])
    blood = StringField('Blood pressure', validators=[DataRequired()])
    skin = StringField('Skin thickness', validators=[DataRequired()])
    insulin = StringField('Insulin', validators=[DataRequired()])
    bmi = StringField('BMI', validators=[DataRequired()])
    dpf = StringField('DPF Score', validators=[DataRequired()])
    age = StringField('Age', validators=[DataRequired()])
    submit = SubmitField('Submit')

# index 페이지 (홈)
@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

# prediction 페이지
@app.route('/prediction', methods=['GET', 'POST'])
def lab():
    form = LabForm()
    if form.validate_on_submit():
        # 사용자 입력을 numpy 배열로 변환
        X_test = np.array([[float(form.preg.data),
                            float(form.glucose.data),
                            float(form.blood.data),
                            float(form.skin.data),
                            float(form.insulin.data),
                            float(form.bmi.data),
                            float(form.dpf.data),
                            float(form.age.data)]])
        print(X_test.shape)
        print(X_test)

        # 기존 csv 데이터로 스케일러 학습
        data = pd.read_csv('./diabetes.csv', sep=',')
        X = data.values[:, 0:8]
        y = data.values[:, 8]

        scaler = MinMaxScaler()
        scaler.fit(X)
        X_test = scaler.transform(X_test)

        # 모델 로드 및 예측
        model = keras.models.load_model('pima model.keras')
        prediction = model.predict(X_test)
        res = prediction[0][0]
        res = np.round(res, 2)
        res = float(np.round(res * 100))

        # 결과 페이지로 이동
        return render_template('result.html', res=res)

    # GET 요청 시 prediction 입력폼 보여주기
    return render_template('prediction.html', form=form)

# 앱 실행
if __name__ == '__main__':
    app.run()

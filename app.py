#Import thư viện
import pandas as pd
import numpy as np
import math
from datetime import date
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from flask import Flask, render_template, request


X = pd.read_csv("X.csv")
X

import pickle
# OLS = 'diabetes-prediction-model.pkl'
# model_OLS = pickle.load(open(OLS, 'rb'))

LNR = 'LN-prediction-model.pkl'
model_LNR = pickle.load(open(LNR, 'rb'))


#Hàm đổi dữ liệu năm
def fill_year(year):
  year_today = date.today().year
  if year <= year_today-2:
    return 'dưới 2 năm'
  elif year <= year_today - 4:
    return '2-4 năm'
  elif year <= year_today - 5:
    return '4-5 năm'
  else:
    return 'hơn 5 năm'


#Hàm đổi dữ liệu dung tích động cơ
def fill_engineSize(x):
  if x < 1.2:
    return 'Small'
  elif x < 1.5:
    return 'Medium'
  elif x < 2:
    return 'Large'
  else:
    return 'Very Large'


#Hàm đổi dữ liệu hãng xe
def fill_class(x):
  if x in ('merc','audi','bmw'):
    return "Luxury"
  if x in ('vw','skoda','focus'):
    return "Mid-range"
  else:
    return "Affordable"


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html', prediction=-1)


@app.route('/predict', methods=['POST'])
def predict():
        # Lấy giá trị đầu vào từ form
    year = float(request.form['year'])
    transmission = request.form['transmission']
    mileage = float(request.form['mileage'])
    fuelType = request.form['fuelType']
    tax = float(request.form['tax'])
    mpg = float(request.form['mpg'])
    classs = request.form['automaker']
    engineSize = float(request.form['engineSize'])

    year = fill_year(year)
    engineSize = fill_engineSize(engineSize)
    classs = fill_class(classs)
    mileage = round(math.log(mileage), 6)
    mpg = round(math.log(mpg), 6)
    data = np.array([year, transmission, fuelType, engineSize, classs, mileage, tax, mpg])

    # Tạo DataFrame từ mảng NumPy
    column_names = ['year', 'transmission', 'fuelType','engineSize','Class','ln_mileage', 'tax', 'ln_mpg' ]

    df = pd.DataFrame([data], columns=column_names)
    XX = pd.concat([X,df], ignore_index=True)

    # Tách ra các biến dạng số và dạng categorical
    numerical_features = ['ln_mileage', 'tax', 'ln_mpg']
    categorical_features = ['year', 'transmission', 'fuelType', 'engineSize', 'Class']

    num_transformer = Pipeline(steps = [
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline([
        ('encoder',OneHotEncoder())
    ])

    #Khai báo quá trình xử lý
    preprocessor = ColumnTransformer([
        ('num',num_transformer,numerical_features),
        ('cat',cat_transformer,categorical_features),
    ])

    transformed_data = preprocessor.fit_transform(XX)
    last_row_array = pd.DataFrame(transformed_data).iloc[-1].values.reshape(1, -1)
    my_prediction = model_LNR.predict(last_row_array)

    image_source = "/static/images/residuals-sklearn.png"


    # Trả về kết quả dự báo
    return render_template('result.html', prediction=my_prediction[0], image_source=image_source)


if __name__ == '__main__':
    app.run()



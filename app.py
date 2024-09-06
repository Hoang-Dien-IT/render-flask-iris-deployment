from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('random_forest_model_iris.pkl')

# Tạo danh sách các tên loài hoa và đường dẫn ảnh tương ứng
iris_classes = {
    0: {'name': 'Iris Setosa', 'image': 'static/iris_setosa.jpg'},
    1: {'name': 'Iris Versicolor', 'image': 'static/iris_versicolor.jpg'},
    2: {'name': 'Iris Virginica', 'image': 'static/iris_virginica.jpg'}
}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Lấy dữ liệu từ form
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])

    # Tạo mảng dữ liệu đầu vào
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Dự đoán
    prediction = model.predict(input_data)[0]

    # Lấy tên loài hoa và đường dẫn ảnh dựa trên dự đoán
    flower_name = iris_classes[prediction]['name']
    flower_image = iris_classes[prediction]['image']

    # Trả về kết quả
    return render_template('index.html', prediction=flower_name, image=flower_image)


if __name__ == '__main__':
    app.run(debug=True)

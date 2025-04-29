import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# 데이터 불러오기
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 정규화 (0~255 → 0~1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# 라벨을 one-hot 인코딩
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 모델 만들기
model = Sequential([
    Flatten(input_shape=(28, 28)),         # 28x28 이미지를 1차원으로 펼침
    Dense(128, activation='relu'),         # 은닉층
    Dense(10, activation='softmax')        # 출력층 (10개 클래스)
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 모델 학습
model.fit(x_train, y_train, epochs=3, validation_split=0.1)

# 테스트 정확도 출력
test_loss, test_acc = model.evaluate(x_test, y_test)
print("테스트 정확도:", test_acc)

from tensorflow.keras.preprocessing import image
import numpy as np

# 이미지 경로
img_path = 'example.png'

# 이미지 불러오기
img = image.load_img(img_path, target_size=(28, 28))  # 크기 맞춰서 읽기
img_array = image.img_to_array(img)                   # numpy 배열로 변환
img_array = img_array / 255.0                          # 정규화 (0~1)
img_array = np.expand_dims(img_array, axis=0)          # 배치 차원 추가

# 모델에 넣기
prediction = model.predict(img_array)
print("예측 결과:", prediction)


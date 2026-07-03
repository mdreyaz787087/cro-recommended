import numpy as np
import pickle

model = pickle.load(open("model.pkl", "rb"))
sc = pickle.load(open("scaler.pkl", "rb"))

crop_dict = {
    1:'rice', 2:'maize', 3:'jute', 4:'cotton', 5:'coconut',
    6:'papaya', 7:'orange', 8:'apple', 9:'muskmelon', 10:'watermelon',
    11:'grapes', 12:'mango', 13:'banana', 14:'pomegranate', 15:'lentil',
    16:'blackgram', 17:'mungbean', 18:'mothbeans', 19:'pigeonpeas',
    20:'kidneybeans', 21:'chickpea', 22:'coffee'
}

data = np.loadtxt("input.txt", delimiter=",")
data = data.reshape(1, -1)

scaled_data = sc.transform(data)

prediction = model.predict(scaled_data)[0]

print("Recommended Crop:", crop_dict[prediction])

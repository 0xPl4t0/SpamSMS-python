import pickle
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from sklearn import metrics
import numpy as np
import pandas as pd

with open('../dataset.pkl', 'rb') as input:
    dataset = pickle.load(input)
    X_train, X_test, y_train, y_test = dataset

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train.shape)
print(y_train)
alpha, epochs = 0.0035, 500
m, n = X_train.shape
print('m =', m)
print('n =', n)
print('Learning Rate =', alpha)
print('Number of Epochs =', epochs)

model = Sequential()
model.add(Dense(50, activation='relu', input_dim=n))

model.add(Dense(2, activation='softmax', ))
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train, epochs=100)

model.save_weights('weights.h5')
model.save('model.h5')

test_acc = model.evaluate(X_test, y_test)
train_acc = model.evaluate(X_train, y_train)

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

print('\n')

recall_test = metrics.recall_score(y_test, y_pred)
precision_test = metrics.precision_score(y_test, y_pred)

print('train_acc: ' + str(train_acc))
print('test_acc: ' + str(test_acc))
print('recall_test: ' + str(recall_test))
print('precision_test: ' + str(precision_test))

print('\n')

m_confusion_test = metrics.confusion_matrix(y_test, y_pred)
print(pd.DataFrame(data=m_confusion_test, columns=['Predicted ham', 'Predicted spam'],
                   index=['Actual ham', 'Actual spam']))

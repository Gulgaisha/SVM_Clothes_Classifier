from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras import utils
from tensorflow.keras.preprocessing import image
import numpy as np
from SVM_Class import SVM_Classifier
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    #У нас 60000 тренеровочных примеров, 10000 тестовых примеров, каждый объект картинка 28*28=784 пикселя
    # 10 видов/классов одежды
    classes = ['футболка', 'брюки', 'свитер', 'платье', 'пальто', 'туфли', 'рубашка', 'кроссовки', 'сумка', 'ботинки']
    #x_train = x_train[0:1000]
    x_train = x_train.reshape(60000, 784)
    x_train = x_train / 255
    #y_train = y_train[0:1000]
    y_train = utils.to_categorical(y_train, 10)
    y_train = np.where(y_train==0, -1, y_train)
    #x_test = x_test[0:200]
    x_test = x_test.reshape(10000, 784)
    x_test = x_test / 255
    #y_test = y_test[0:200]
    y_test = utils.to_categorical(y_test, 10)
    y_test = np.where(y_test==0, -1, y_test)

    svm = SVM_Classifier(etha=0.03, alpha=0.0001, epochs=300)
    Theta = []
    y_pred = []

    #тренеруем для каждого вида одежды отдельно, получаем для каждого вида свое Theta
    for i in range(0, 10):
        Theta.append(svm.learning(x_train, y_train[:,i]))
    #определяем вид одежды на 5 примерах
    for i in range(0, 10):
        y_pred.append(svm.prediction(x_test[0:5], Theta[i]))
        for j,x in enumerate(y_pred[i]):
            if x == 1:
                print("Вид одежды: {}, в {} примере" .format(classes[i], j))

    #график cost function
    plt.plot(svm.train_cost_func, linewidth=2, label='train_cost_func')
    plt.grid()
    plt.legend(prop={'size': 15})
    plt.show()

    #высчитываем accuracy
    y_prediction = []
    for i in range(0, 10):
        y_prediction.append(svm.prediction(x_train, Theta[i]))
        print(accuracy_score(y_train[:,i], y_prediction[i]))

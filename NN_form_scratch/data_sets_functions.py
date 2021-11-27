import mnist_loader
import keras.datasets.fashion_mnist as fashion_mnist
from tensorflow import keras








#y_train = keras.utils.to_categorical(y_train, num_classes)


def cifar_dataset():
    (x_train_cif, y_train_cif), (x_test_cif, y_test_cif) = keras.datasets.cifar10.load_data()
    num_classes = 10
    y_train_cif = keras.utils.to_categorical(y_train_cif, num_classes)

    x_train_cif = x_train_cif.reshape(50000, 32*32*3)
    y_tr_cif = []
    for i in range(len(y_train_cif)):
        y_tr_cif.append(y_train_cif[i])
    y_train_cif = y_tr_cif

    train_cif = []

    for i in range(len(y_train_cif)):
        train_cif.append( (x_train_cif[i].reshape(32*32*3,1), y_train_cif[i].reshape(10,1) ) )
        

    #y_test_cif = keras.utils.to_categorical(y_test_cif, num_classes)

    test_cif = []
    x_test_cif = x_test_cif.reshape(10000, 32*32*3)
    for i in range(len(y_test_cif)):
        test_cif.append( (x_test_cif[i].reshape(32*32*3, 1), y_test_cif[i][0]))
        
    return train_cif, test_cif



def mnist_fashion_dataset():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    assert x_train.shape == (60000, 28, 28)
    assert x_test.shape == (10000, 28, 28)
    assert y_train.shape == (60000,)
    assert y_test.shape == (10000,)





    x_train = x_train.reshape(60000, 784, 1)
    x = []
    y = []
    for i in range(len(x_train)):
        x.append(x_train[i][:])
    for i in range(len(x_test)):
        y.append(x_test[i][:].reshape(784,1))
    train_f = []
    test_f = []

    #num_classes = [0,1,2,3,4,5,6,7,8,9]
    num_classes = 10
    y_train = keras.utils.to_categorical(y_train, num_classes)

    for i in range(len(x_train)):
        train_f.append( (x[i]/256, y_train[i].reshape(10,1))  )
        
    for i in range(len(y_test)):
        test_f.append( (y[i]/256, y_test[i]) )
    """
    Label	Description
    0	T-shirt/top
    1	Trouser
    2	Pullover
    3	Dress
    4	Coat
    5	Sandal
    6	Shirt
    7	Sneaker
    8	Bag
    9	Ankle boot
    """
    return (train_f, test_f)





def mnist():
    training_data_ZIP, validation_data_ZIP, test_data_ZIP = mnist_loader.load_data_wrapper()
    training_data = list(training_data_ZIP)
    validation_data = list(validation_data_ZIP)
    test_data = list(test_data_ZIP)
    
    
    return (training_data, test_data)



















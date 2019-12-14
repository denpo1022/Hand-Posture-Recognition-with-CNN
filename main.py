import cv2
import numpy as np
import os
import tensorflow.keras
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
import matplotlib.pylab as plt

input_shape = (32, 32, 1)


def enumerate_files(dirs, path='Data/All_gray_1_32_32', n_poses=3, n_samples=20):
    filenames, targets = [], []
    for p in dirs:
        for n in range(n_poses):
            for j in range(3):
                dir_name = path + '/' + p + '/000' + str(n * 3 + j) + '/'
                for s in range(n_samples):
                    d = dir_name + '%04d/' % s
                    for f in os.listdir(d):
                        if f.endswith('jpg'):
                            filenames += [d + f]
                            targets.append(n)
    return filenames, targets


def read_images(files):
    imgs = []
    for f in files:
        img = cv2.imread(f, int(cv2.IMREAD_GRAYSCALE / 255.0))
        imgs.append(img)
    return imgs


def read_datasets(datasets):
    files, labels = enumerate_files(datasets)
    list_of_arrays = read_images(files)
    return np.array(list_of_arrays), np.array(labels)


def main():
    
    # prepare training and testing data
    train_sets = ['Set1', 'Set2', 'Set3']
    test_sets = ['Set4', 'Set5']
    trn_array, trn_labels = read_datasets(train_sets)
    tst_array, tst_labels = read_datasets(test_sets)

    # reshape the array in order to adapt the input shape of first layer
    trn_array = trn_array.reshape(trn_array.shape[0], 32, 32, 1)
    tst_array = tst_array.reshape(tst_array.shape[0], 32, 32, 1)

    # reshape the labels from (540,) into (540, 3)
    trn_labels = tensorflow.keras.utils.to_categorical(trn_labels, 3)
    tst_labels = tensorflow.keras.utils.to_categorical(tst_labels, 3)

    # declare the model layers using keras function with tensorflow backend
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(trn_array, trn_labels,
                        epochs=40,
                        batch_size=32,
                        verbose=1,
                        validation_data=(tst_array, tst_labels))

    # get and show values of test loss and accuracy
    score = model.evaluate(tst_array, tst_labels, verbose=0)
    print(model.summary())
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # declare a figure
    fig = plt.figure()
    fig.set_figheight(10)
    fig.set_figwidth(15)

    # summarize history for accuracy
    fig.add_subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # summarize history for loss
    fig.add_subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # plt.show()

    # fig.savefig('OutputFigure/result.png', bbox_inches='tight')
    # plt.close(fig)


if __name__ == '__main__':
    main()

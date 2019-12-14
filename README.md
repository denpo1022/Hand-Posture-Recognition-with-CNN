# Hand pose recognition with CNN

## Description

In this assignment, we have to design a Convolutional Neural Network to recognize the three diffrent hand postures.

![postures](https://i.imgur.com/vLPxvdE.jpg)

---

## Training environment

Using conda 4.7.12

`conda install -c anaconda keras-gpu`

---

## Description and Explanation of Code

### Training and testing data preparing

First I took the usage of functions from the attached pdf which refer three functions relating to data prepare: `enumerate_files(), read_images(), read_datasets(datasets)`. As soon I noticed the code here has some problems in `enumerate_files()`, so I did a little bit changing to make this function working again.

Here is the original function on pdf:

```python=0
def enumerate_files(dirs, path='All_gray_1_32_32', n_poses=3, n_samples=20):
    filenames, targets = [], []
    for p in dirs:
        for n in range(n_poses):
            for j in range(3):
                dir_name = path + p + '/000' + str(n * 3 + j) + '/'
                for s in range(n_samples):
                    d = dir_name + '%04d/' % s
                    for f in os.listdir(d):
                        if f.endswith('jpg'):
                            filenames += [d + f]
                            targets.append(n)
```

and the version after optimized:

```python=0
def enumerate_files(dirs, path='All_gray_1_32_32', n_poses=3, n_samples=20):
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
```

Two places had been changed:

First one

```python=5
dir_name = path + p + '/000' + str(n * 3 + j) + '/'
```

to

```python=5
dir_name = path + '/' + p + '/000' + str(n * 3 + j) + '/'
```

and second one:

```python=12
return filenames, targets
```

which there was no return values.

### Build a sequntial multi-layer model using CNN

One small trick for this model is adding a dopout layer. Due to the lack of training data, it may result to overfitting. Applying a dropout layer can efficiently reduce this situation.

```python=59
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
```

| Layer(type)                    | Output Shape       | Param # |
| ------------------------------ |:------------------ |:------- |
| conv2d (Conv2D)                | (None, 30, 30, 32) | 320     |
| conv2d_1 (Conv2D)              | (None, 28, 28, 32) | 9248    |
| max_pooling2d (MaxPooling2D)   | (None, 14, 14, 32) | 0       |
| conv2d_2 (Conv2D)              | (None, 12, 12, 64) | 18496   |
| conv2d_3 (Conv2D)              | (None, 10, 10, 64) | 36928   |
| max_pooling2d_1 (MaxPooling2D) | (None, 5, 5, 64)   | 0       |
| flatten (Flatten)              | (None, 1600)       | 0       |
| dense (Dense)                  | (None, 128)        | 204928  |
| dropout (Dropout)              | (None, 128)        | 0       |
| dense_1 (Dense)                | (None, 3)          | 387     |
Total params: 270,307
Trainable params: 270,307
Non-trainable params: 0

### Evalutate the training result and output the figure

```python=76
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
```

Here is the table with most recent training result:

| History | Loss                | Accuracy   |
| ------- | ------------------- |:---------- |
| 1       | 0.29477801006287335 | 0.94166666 |
| 2       | 0.26194892024828326 | 0.95555556 |
| 3       | 0.2294204789847653  | 0.9444444  |
| 4       | 0.20179064938339353 | 0.9527778  |
| 5       | 0.38640768849549606 | 0.94722223 |
| 6       | 0.38526384320575746 | 0.9527778  |
| 7       | 0.07503241179899002 | 0.9722222  |
| 8       | 0.1655165569934373  | 0.9583333  |
| 9       | 0.4368875512009254  | 0.9138889  |
| 10      | 0.3331649558411704  | 0.925      |

![result](https://i.imgur.com/JSuTeAm.png)

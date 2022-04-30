from operator import mod
import tensorflow as tf
import numpy as np
from keras.layers import Dense, SimpleRNN
from keras.models import Sequential


def run():
    log = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3,
           4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    time = 5

    wordBag = [0, 1, 2, 3, 4]

    encodedWordBag = tf.one_hot(wordBag, len(wordBag))
    encodedWordBag = tf.reshape(encodedWordBag, shape=[-1, len(wordBag)])

    print(encodedWordBag)

    trainData = [log[i:i+time] for i in range(0, len(log) - time)]
    trainData = np.array(trainData)
    trainData = trainData.repeat(1).reshape(
        trainData.shape[0], trainData.shape[1], 1
    )

    labelData = log[time:]
    encodedLabelData = list(
        map(lambda label: encodedWordBag[label], labelData))

    encodedLabelData = np.array(encodedLabelData)
    print(encodedLabelData)

    model = Sequential([
        SimpleRNN(70, input_shape=(trainData.shape[1:])),
        Dense(time, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    history = model.fit(trainData, encodedLabelData,
                        epochs=10, batch_size=10, verbose=2)

    predict = model.predict([log[-time:]])[0]

    sortedPredict = np.sort(predict)[::-1]

    top4Values = [np.where([predict] == sortedPredict[i])[1][0]
                  for i in range(0, 4)]

    print(top4Values)


if __name__ == '__main__':
    run()

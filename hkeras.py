from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D, GlobalAveragePooling2D
from keras.layers import Flatten, Dense, BatchNormalization, Dropout, Concatenate, Input, Activation
from keras.optimizers import Adam

def one_hot(labels, num_classes):
    m = len(labels)
    labels = np.squeeze(labels)
    Y = np.zeros((m, num_classes))
    Y[np.arange(m), labels] = 1
    return Y


def add_layers(model, cfg, batch_norm=True):
  for v in cfg:
    if v == 'M':
      model.add(MaxPooling2D(2, 2))
    elif v == 'A':
      model.add(AveragePooling2D(2, 2))
    else:
      model.add(Conv2D(v, (3, 3), padding='same', activation='relu'))
      if batch_norm:
        model.add(BatchNormalization())
  return model

net = Sequential()
net.add(ZeroPadding2D((2, 2), input_shape=(28, 28, 3)))
add_layers(net, [4 ,4, 'M', 8, 8, 'M', 16, 16, 'M', 32, 32, 'M', 64, 64, 'M'])
net.add(Reshape((64,)))
net.add(Dense(10, activation='softmax'))
net.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=['accuracy'])
net.fit(X, Y, epochs=5, batch_size=32)


def SqueezeNet(classes, inputs, data_format='channels_first'):
    def fire_block(filters):
        axis = 1
        if data_format=='channels_last':
          axis = -1
        def f(X):
            fire_squeeze = Conv2D(
                filters // 4, (1, 1),
                activation='relu', data_format=data_format)(X)
            fire_expand1 = Conv2D(
                filters, (1, 1),
                activation='relu', data_format=data_format)(fire_squeeze)
            fire_expand2 = Conv2D(
                filters, (1, 1),
                activation='relu', data_format=data_format)(fire_squeeze)
            merge = Concatenate(axis=-1)([fire_expand1, fire_expand2])
            return merge
        return f

    X = Input(shape=inputs)
    conv1 = Conv2D(
        32, (5, 5), strides=(1, 1), padding='same', 
        activation='relu', data_format=data_format)(X)
    maxpool1 = MaxPooling2D(
        (3, 3), strides=(2, 2), padding='same',
        data_format=data_format)(conv1)
    fire2 = fire_block(32)(maxpool1)
    fire3 = fire_block(32)(fire2)
    fire4 = fire_block(64)(fire3)
    maxpool4 = MaxPooling2D(
        (3, 3), strides=(2, 2), padding='same',
        data_format=data_format)(fire4)
    fire5 = fire_block(64)(maxpool4)
    fire6 = fire_block(64)(fire5)
    fire7 = fire_block(96)(fire6)
    maxpool7 = MaxPooling2D(
        (3, 3), strides=(2, 2),
        padding='same', data_format=data_format)(fire7)
    fire8 = fire_block(96)(maxpool7)
    fire9 = fire_block(128)(fire8)
    fire9_dropout = Dropout(0.5)(fire9)
    conv10 = Conv2D(classes, (1, 1), data_format=data_format)(fire9_dropout)
    global_avgpool10 = GlobalAveragePooling2D(data_format=data_format)(conv10)
    softmax = Activation('softmax')(global_avgpool10)

    return Model(inputs=X, outputs=softmax)

model = SqueezeNet(10, inputs=(1, 28, 28), data_format='channels_first')
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=['accuracy'])

model.fit(X, Y_hot, epochs=5, batch_size=32)




net = Sequential()
net.add(Conv2D(6, (5, 5), padding='same', activation='relu', input_shape=(28, 28, 1), data_format='channels_last'))
net.add(MaxPooling2D((2, 2)))
net.add(Conv2D(16, (5, 5), padding='valid', activation='relu'))
net.add(MaxPooling2D((2, 2)))
net.add(Conv2D(120, (5, 5), padding='valid', activation='relu'))
net.add(Flatten())
net.add(Dense(units=84, activation='relu', input_dim=120))
net.add(Dense(units=10, activation='softmax'))

net.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=['accuracy'])

net.fit(X, Y, epochs=5, batch_size=32)
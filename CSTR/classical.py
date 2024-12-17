
import tensorflow as tf
import data_seq

print("Libraries loaded!")
_from_save = True
_save_path = "classicalNN"
_training = True
model = tf.keras.Sequential([
    tf.keras.layers.Dense(14, activation="relu", input_shape=(7,)),
    tf.keras.layers.Dense(14, activation="sigmoid"),
    tf.keras.layers.Dense(4, activation="softmax"),
    ])

if _from_save:
    model.load_weights(_save_path)

loss_fn = tf.keras.losses.CategoricalCrossentropy()
model.compile(optimizer="adam", loss=loss_fn)
seq = data_seq.TrainDataSeq()

print("Model ready!")

if _training:
    for i in range(15):
        model.fit(seq)
    model.save_weights(_save_path)

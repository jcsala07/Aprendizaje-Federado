import tensorflow as tf

class build:
    @staticmethod
    def build_it():
        model = tf.keras.Sequential([
            # 1) Augmentación ligera
            tf.keras.layers.Input(shape=(28, 28, 1)),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),

            # 2) Bloque conv 1
            tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(),

            # 3) Bloque conv 2
            tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(),

            # 4) Bloque conv 3
            tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(),

            # 5) Reduce a vector y regulariza
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.3),

            # 6) Capa totalmente conectada
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),

            # 7) Salida
            tf.keras.layers.Dense(10, activation='softmax'),
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
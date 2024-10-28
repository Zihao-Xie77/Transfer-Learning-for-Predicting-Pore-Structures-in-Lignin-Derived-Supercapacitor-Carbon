import numpy as np
import pandas as pd
import random
import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Input, Lambda, Dropout
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import datetime
import os
import joblib


random_seed = 42
np.random.seed(random_seed)
tf.random.set_seed(random_seed)
random.seed(random_seed)
os.environ['PYTHONHASHSEED'] = str(random_seed)
os.environ['TF_DETERMINISTIC_OPS'] = '0'
os.environ['TF_CUDNN_DETERMINISTIC'] = '0'

session_config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
tf.compat.v1.set_random_seed(random_seed)
session = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_config)

file_path = r'Dataset\Original dataset.xlsx'
df = pd.read_excel(file_path)

for col in df.columns:
    if col != 'Label' and col != 'Agent':
        df[col] = pd.to_numeric(df[col], errors='coerce')

column_to_encode = 'Agent'
df_encoded = pd.get_dummies(df[column_to_encode], prefix=column_to_encode)
df = pd.concat([df, df_encoded], axis=1)
df.drop(columns=[column_to_encode], inplace=True)

column_to_encode = 'Label'
df_encoded = pd.get_dummies(df[column_to_encode], prefix=column_to_encode)
df = pd.concat([df, df_encoded], axis=1)
df.drop(columns=[column_to_encode], inplace=True)

num_features = df.shape[1] - 8
scaler = StandardScaler()
data = df.values.astype(np.float32)
data[:, :num_features] = scaler.fit_transform(data[:, :num_features])


def split_outputs(x, output_dim, num_classes, feature_indice1, feature_indice2):
    continuous_features = x[:, :output_dim - num_classes]
    categorical_features1 = tf.gather(x, feature_indice1, axis=1)
    categorical_features1 = tf.keras.activations.softmax(categorical_features1)

    categorical_features2 = tf.gather(x, feature_indice2, axis=1)
    categorical_features2 = tf.keras.activations.softmax(categorical_features2)

    return tf.concat([continuous_features, categorical_features1, categorical_features2], axis=-1)

def one_hot_encode(selected_features, feature_indices):
    selected_features = tf.gather(selected_features, feature_indices, axis=1)

    max_indices = tf.argmax(selected_features, axis=-1, output_type=tf.int32)

    one_hot_encoded = tf.cast(tf.equal(tf.expand_dims(tf.range(selected_features.shape[1], dtype=tf.int32), axis=0),
                                       tf.expand_dims(max_indices, axis=-1)), tf.float32)
    return one_hot_encoded

def normalize(x):
    mean, variance = tf.nn.moments(x, axes=[0])
    return (x - mean) / tf.sqrt(variance + 1e-8)

def generate_and_save_samples(generator, num_samples, noise_dim, save_path):
    noise = np.random.normal(-1, 1, (num_samples, noise_dim))
    generated_data = generator.predict(noise)

    feature_label = one_hot_encode(generated_data, [14, 15, 16, 17, 18, 19])  # 活化剂
    categorical_label = one_hot_encode(generated_data, [20, 21])
    feature_value = generated_data[:, :14]

    input_feature_label_np = feature_label.numpy()

    date = pd.read_excel(r"Dataset\Original LDPC dataset.xlsx")
    data_feature = date[['Agent_H3PO4', 'Agent_K2CO3', 'Agent_KOH', 'Agent_Na2CO3', 'Agent_NaOH',
                         'Agent_ZnCl2']]

    scaler = StandardScaler()
    scaler.fit(data_feature.to_numpy())
    input_feature_label_scaled = scaler.transform(input_feature_label_np)

    feature_label = tf.convert_to_tensor(input_feature_label_scaled, dtype=tf.float32)

    output_feature_value = tf.cast(feature_value, dtype=tf.float32)
    output_feature_label = tf.cast(feature_label, dtype=tf.float32)
    output_categorical_label = tf.cast(categorical_label, dtype=tf.float32)

    output_feature = tf.concat([output_feature_value, output_feature_label, output_categorical_label], axis=1)

    generated_df = pd.DataFrame(output_feature, columns=df.columns)

    print(generated_df.head())
    generated_df.to_csv(save_path, index=False)

def wasserstein_loss(y_true, y_pred):
    return -tf.reduce_mean(y_true * y_pred)

def gradient_penalty(discriminator, real_data, fake_data):
    batch_size = tf.shape(real_data)[0]
    alpha = tf.random.uniform([batch_size, 1], 0.0, 1.0)
    interpolated = alpha * real_data + (1 - alpha) * fake_data

    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        d_interpolated = discriminator(interpolated)

    grads = tape.gradient(d_interpolated, [interpolated])[0]
    grads = tf.reshape(grads, [grads.shape[0], -1])
    gp = tf.reduce_mean((tf.norm(grads, axis=1) - 1.0) ** 2)
    return gp

def build_generator(input_dim, output_dim, num_classes, feature_indices1, feature_indices2):
    model = Sequential()

    model.add(Dense(256, input_dim=input_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Dense(192))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Dense(64))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Dense(44))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Dense(output_dim))
    model.add(Lambda(normalize))

    model.add(Lambda(lambda x: split_outputs(x, output_dim, num_classes, feature_indices1, feature_indices2)))

    return model


def build_discriminator(input_dim):
    model = Sequential()
    model.add(Dense(22, input_dim=input_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))


    model.add(Dense(11))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    model.add(Dense(1))
    return model

def train_gan(generator, discriminator, data, XGBoost_SSA, XGBoost_TPV, epochs=1000, batch_size=200, gp_weight=10.0, penalty_weight=0.01):
    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))
    optimizer_d = Adam(0.00005, 0.5)
    optimizer_g = Adam(0.0001, 0.5)

    scalar_log_dir = r'logs\scalar' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_writer_scalar = tf.summary.create_file_writer(scalar_log_dir)

    for epoch in range(epochs):
        for _ in range(8):
            random.seed(random_seed + epoch)
            idx = np.random.randint(0, data.shape[0], batch_size)
            real_data = data[idx]

            noise = np.random.normal(0, 1, (batch_size, noise_dim))
            generated_data = generator.predict(noise)

            with tf.GradientTape() as tape:
                smooth_real = real * 0.9
                d_loss_real = tf.reduce_mean(discriminator(real_data))
                d_loss_fake = tf.reduce_mean(discriminator(generated_data))
                d_loss = tf.reduce_mean(d_loss_fake) - tf.reduce_mean(d_loss_real * smooth_real)

                gp = gradient_penalty(discriminator, real_data, generated_data)
                d_loss += gp_weight * gp

            d_grads = tape.gradient(d_loss, discriminator.trainable_variables)
            optimizer_d.apply_gradients(zip(d_grads, discriminator.trainable_variables))

        noise = np.random.normal(0, 1, (batch_size, noise_dim))
        with tf.GradientTape() as tape:
            generated_data = generator(noise)
            generated_data = tf.cast(generated_data, dtype=tf.float32)

            feature_label = one_hot_encode(generated_data, [14, 15, 16, 17, 18, 19])
            categorical_label = one_hot_encode(generated_data, [20, 21])
            feature_value = generated_data[:, :12]
            label_value1 = generated_data[:, 12]
            label_value2 = generated_data[:, 13]

            input_feature_label_np =feature_label.numpy()

            date_real = pd.read_csv(r"D:\python machine learning\GAN\Project1\Data\GBR_origin_1_test.csv")
            data_feature = date_real[['Agent_H3PO4', 'Agent_K2CO3', 'Agent_KOH', 'Agent_Na2CO3', 'Agent_NaOH',
                                      'Agent_ZnCl2']]
            scaler = StandardScaler()
            scaler.fit(data_feature.to_numpy())

            input_feature_label_scaled = scaler.transform(input_feature_label_np)

            feature_label = tf.convert_to_tensor(input_feature_label_scaled, dtype=tf.float32)

            condition_label = categorical_label[:, 1]
            condition_mask = tf.equal(condition_label, 1)
            condition_mask = tf.cast(condition_mask, tf.bool)

            if tf.reduce_any(condition_mask):
                input_feature_value = tf.boolean_mask(feature_value, condition_mask)
                input_feature_label = tf.boolean_mask(feature_label, condition_mask)

                input_label_value1 = tf.boolean_mask(label_value1, condition_mask)
                input_label_value2 = tf.boolean_mask(label_value2, condition_mask)

                input_feature_value = tf.cast(input_feature_value, dtype=tf.float32)
                input_feature_label = tf.cast(input_feature_label, dtype=tf.float32)
                input_feature = tf.concat([input_feature_value, input_feature_label], axis=1)

                if tf.shape(input_feature)[0] > 0:
                    XGB_prediction1 = XGBoost_SSA.predict(input_feature)
                    XGB_prediction2 = XGBoost_TPV.predict(input_feature)

                    XGB_loss1 = np.mean(np.square(XGB_prediction1 - input_label_value1))
                    XGB_loss2 = np.mean(np.square(XGB_prediction2 - input_label_value2))

                    alpha = 0.5
                    XGB_loss = alpha * XGB_loss1 + (1 - alpha) * XGB_loss2
                else:
                    XGB_loss = 0
            else:
                XGB_loss = penalty_weight * tf.reduce_sum(tf.cast(~condition_mask, tf.float32))

            g_loss = -tf.reduce_mean(discriminator(generated_data)) + XGB_loss

            noise_injected = tf.random.normal(tf.shape(generated_data), mean=0.0, stddev=0.2)
            generated_data += noise_injected

        g_grads = tape.gradient(g_loss, generator.trainable_variables)
        optimizer_g.apply_gradients(zip(g_grads, generator.trainable_variables))

        if epoch % 100 == 0:
            print(f"{epoch} [D loss: {d_loss.numpy()}] [G loss: {g_loss.numpy()}]")
        if epoch % 1 == 0:
            with summary_writer_scalar.as_default():
                tf.summary.scalar('d_loss', d_loss, step=epoch)
                tf.summary.scalar('g_loss', g_loss, step=epoch)
                tf.summary.scalar('XGB_loss_SSA', XGB_loss1, step=epoch)
                tf.summary.scalar('XGB_loss_TPV', XGB_loss2, step=epoch)

        if epoch == 1000:
            generate_and_save_samples(generator, 2000, noise_dim,
                                      r'Dataset\Generated dataset.csv')

        if np.isnan(d_loss).any() or np.isnan(g_loss).any():
            break
            print(f"NaN loss detected at epoch {epoch}")
        if np.abs(d_loss).any() > 1e5 or np.abs(g_loss).any() > 1e5:
            print(f"Exploding gradient detected at epoch {epoch}")
            break


if __name__ == '__main__':
    noise_dim = 256
    data_dim = data.shape[1]
    num_classes = 8
    generator = build_generator(noise_dim, data_dim, num_classes, [14, 15, 16, 17, 18, 19], [20, 21])
    discriminator = build_discriminator(data_dim)

    XGBoost_SSA = joblib.load('../XGBoost_SSA.joblib')
    XGBoost_TPV = joblib.load('../XGBoost_TPV.joblib')
    train_gan(generator, discriminator, data, XGBoost_SSA, XGBoost_TPV, epochs=1700, batch_size=30)

    generator.save('generator_model.keras')
    noise = np.random.normal(-1, 1, (4000, noise_dim))
    generated_data = generator.predict(noise)

    generated_data_to_inverse = generated_data[:, :num_features]
    generated_data_inversed = scaler.inverse_transform(generated_data_to_inverse)
    generated_data_combined = np.hstack((generated_data_inversed, generated_data[:, num_features:]))
    generated_df = pd.DataFrame(generated_data_combined, columns=df.columns)
    print(generated_df.head())

    generated_df.to_csv(r'Dataset\Generated dataset.csv', index=False)

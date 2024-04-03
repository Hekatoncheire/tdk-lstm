import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import layers, Model, Input, backend as be
import matplotlib.pyplot as plt

print("Hello, TDK")
#print(tf.__version__)

# Adatok beolvasása

df_CGM = pd.read_csv("./data/HDeviceCGM.txt", sep= '|')
print('CGM adatok beolvasva')

df_BGM = pd.read_csv("./data/HDeviceBGM.txt", sep='|')
print('BGM adatok beolvasva')

df_patients = pd.read_csv("./data/HPtRoster.txt", sep= '|')
print('Páciens adatok beolvasva')

# Dátummező kialakítása az adattáblák rendezéséhez

def preprocess (df, base_date = pd.Timestamp('2015-05-22')):
    df['DeviceDateCombined'] = pd.to_timedelta(df['DeviceDtTmDaysFromEnroll'], unit='D') + base_date
    df['DeviceDateCombined'] += pd.to_timedelta(df['DeviceTm'].astype(str))
    return df

# Rendezés előkészítése

df_CGM = preprocess(df_CGM)
df_BGM = preprocess(df_BGM)

# Rendezés (páciens azonosítója alapján)

def order_by (df, column1 = 'PtID', column2='DeviceDateCombined'):
    df = df.sort_values([column1, column2])
    return df

df_CGM = order_by(df_CGM)
df_BGM = order_by(df_BGM)

print('Rendezés kész!')
print(max(df_CGM['GlucoseValue']))

# Használt oszlopok kiválasztása
cgm_selectedColumns = df_CGM[['PtID', 'DeviceDateCombined', 'GlucoseValue']]

# Betegek szeparációja vizsgálati csoport alapján
df_cgmOnlyPatients = df_patients[df_patients['TrtGroup'] == 'CGM Only']['PtID'].unique()
df_cgmBgmPatients = df_patients[df_patients['TrtGroup'] == 'CGM+BGM']['PtID'].unique()

# Adatok normalizálása
scaler = MinMaxScaler()
cgm_selectedColumns['GlucoseValueNormalized'] = scaler.fit_transform(cgm_selectedColumns[['GlucoseValue']].values.reshape(-1,1))

# Idősorok előállítása betegenként külön-külön
def create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length):
        seq = data[i:(i + sequence_length)]
        sequences.append(seq)
    return np.array(sequences)

sequence_length = 6

patients_sequences = {}

for patient_id in cgm_selectedColumns['PtID'].unique():
    patient_data = cgm_selectedColumns[cgm_selectedColumns['PtID'] == patient_id]['GlucoseValueNormalized'].values
    # Adathalmaz csökkentése
    slice_index = int(len(patient_data)*0.1)
    patient_data_reduced = patient_data[:slice_index]
    
    patients_sequences[patient_id] = create_sequences(patient_data_reduced, sequence_length)

print('Sorok legenerálva!')

# Splittelés
train_test_data = {}

for patient_id, sequences in patients_sequences.items():
    sequences = np.array(sequences)

    X_train, X_test = train_test_split(sequences, test_size=0.2, random_state=42)

    train_test_data[patient_id] = {'X_train': X_train, 'X_test': X_test}

print('Splittelés kész!')
print(train_test_data[183])

# Modell építése
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
class ReconstructionLossLayer(layers.Layer):
    def call(self, inputs):
        encoder_inputs, vae_outputs = inputs
        reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.keras.losses.mean_squared_error(encoder_inputs, vae_outputs), axis=1))
        self.add_loss(reconstruction_loss)
        return vae_outputs

def lstm_vae_model(input_shape, latent_dim=2):
    # Encoder
    encoder_inputs = Input(shape=input_shape)
    x = layers.LSTM(64, return_sequences=True, activation='relu')(encoder_inputs)
    x = layers.LSTM(32, activation='relu')(x)
    z_mean = layers.Dense(latent_dim)(x)
    z_log_var = layers.Dense(latent_dim)(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    # Decoder
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = layers.RepeatVector(input_shape[0])(latent_inputs)
    x = layers.LSTM(32, return_sequences=True, activation='relu')(x)
    x = layers.LSTM(64, return_sequences=True, activation='relu')(x)
    decoder_outputs = layers.TimeDistributed(layers.Dense(input_shape[1]))(x)
    decoder = Model(latent_inputs, decoder_outputs, name="decoder")

    # VAE
    vae_outputs = decoder(encoder(encoder_inputs)[2])
    vae_outputs_with_loss = ReconstructionLossLayer()([encoder_inputs, vae_outputs])
    vae = Model(encoder_inputs, vae_outputs_with_loss, name='lstm_vae')

    vae.compile(optimizer = 'adam')  
    
    return vae

input_shape = (6, 1)  # Idősorok dimenziói
latent_dim = 2  

# Modell tanítása betegenként perszonalizálva
models_by_patient = {}
for patient_id in train_test_data.keys():
    print(f"Model tanítása {patient_id} azonosítójú beteg számára...")
    vae = lstm_vae_model(input_shape, latent_dim=latent_dim)
    
    X_train = train_test_data[patient_id]['X_train']
    X_test = train_test_data[patient_id]['X_test']
    
    vae.fit(X_train, X_train, epochs=12, batch_size=18, validation_split=0.2)
    
    models_by_patient[patient_id] = vae

    reconstructed_X_test = np.squeeze(vae.predict(X_test))
    reconstruction_errors = np.mean(np.power(X_test - reconstructed_X_test, 2), axis=-1)

    threshold = np.mean(reconstruction_errors) + 3*np.std(reconstruction_errors)

    anomalies = reconstruction_errors > threshold
    print(len(anomalies), len(X_test))

    plt.figure(figsize=(10, 6))

    # If X_test is originally 2D and flattened for plotting
    X_test_flattened = X_test.flatten()

    # Teszt halmaz visszaalakítása eredeti glükózértékekké
    X_test_flattened = scaler.inverse_transform(X_test_flattened.reshape(-1,1))

    # Generate indices for X_test_flattened
    indices = np.arange(len(X_test_flattened))

    # Assuming anomalies is a boolean array indicating anomalies at the sequence level
    # Repeat anomaly flags to match the flattened structure
    sequence_length = X_test.shape[1]  # Assuming X_test is 2D (n_sequences, sequence_length)
    anomalies_repeated = np.repeat(anomalies, sequence_length)

    # Ensure the repeated anomalies array matches the flattened data length
    assert len(anomalies_repeated) == len(X_test_flattened), "Anomalies array length must match the flattened data length."

    # Plot the original data
    plt.plot(indices, X_test_flattened, label='Original Data', color='blue', alpha=0.7)

    # Overlay anomalies
    plt.scatter(indices[anomalies_repeated], X_test_flattened[anomalies_repeated], color='red', label='Anomaly', alpha=0.5)

    plt.title('Detected Anomalies in Data')
    plt.xlabel('Data Point Index')
    plt.ylabel('Glucose Value')
    plt.legend()
    plt.show()


    be.clear_session()

print('Tanítás befejezve!')

# Anomáliák kiválasztása a rekonstrukciós hiba alapján
reconstructed_X_test = vae.predict(X_test)
reconstruction_errors = np.mean(np.power(X_test - reconstructed_X_test, 2), axis=-1)

threshold = np.mean(reconstruction_errors) + 2*np.std(reconstruction_errors)

anomalies = reconstruction_errors > threshold

# Plotting
plt.figure(figsize=(10, 6))

# Plot original data
plt.plot(X_test, label='Original Data')

# Overlay anomalies
plt.scatter(range(len(X_test)), X_test, c=anomalies, cmap='coolwarm', label='Anomaly')

plt.title('Detected Anomalies in Data')
plt.xlabel('Data Point Index')
plt.ylabel('Glucose Value')
plt.legend()
plt.show()

## táblák fontosabb jellemzői
#print(len(np.unique(df_BGM['PtID'])))
#print(max(df_CGM['DeviceDateCombined']))
#print(np.unique(df_CGM[df_CGM['PtID'] == 10]['DeviceDateCombined']))
#print(len(np.unique(df_CGM[df_CGM['PtID'] == 10]['DeviceDateCombined'])))
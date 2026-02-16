import lightkurve as lk
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from astropy.timeseries import LombScargle
from tqdm import tqdm
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os


SAMPLE_SIZE = 10000  
FEATURES_FILE = "tess_variable_star_features.csv" 
X_SEQ_FILE = "X_seq_cnn_input.npy"              
Y_SEQ_FILE = "y_seq_cnn_labels.npy"              



def clean_lc(tic_id):
    try:
        search = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS")
        lc_raw = search[0].download(flux_column='pdcsap_flux')
        lc_clean = lc_raw.remove_nans().remove_outliers(sigma=5)
        lc_flat = lc_clean.flatten(window_length=301) 
        return lc_flat
    except Exception:
        return None

def extract_features(lc):
    flux = lc.flux.value
    time = lc.time.value
    stats = {
        "mean_flux": np.mean(flux),
        "std_flux": np.std(flux),
        "skewness": skew(flux),
        "kurtosis": kurtosis(flux),
        "flux_range": np.ptp(flux),
    }
    try:
        frequency, power = LombScargle(time, flux).autopower()
        best_frequency = frequency[np.argmax(power)]
        stats["best_period"] = 1.0 / best_frequency
        stats["max_power"] = np.max(power)
    except Exception:
        stats["best_period"] = np.nan
        stats["max_power"] = np.nan
    return stats




df_balanced = pd.read_csv('final_star_list.csv')
df_sample = df_balanced.sample(n=SAMPLE_SIZE, random_state=42).copy()

print(f"Starting to process and save data for {len(df_sample)} stars...")


final_features = [] 
sequences = []      
labels_list = []   


for _, row in tqdm(df_sample.iterrows(), total=len(df_sample), desc="Processing & Saving"):
    tic_id = row['tic_id']
    label = row['label']
    
    lc_flat = clean_lc(tic_id)
    
    if lc_flat is not None:
        feats = extract_features(lc_flat)
        feats.update({"tic_id": tic_id, "label": label})
        final_features.append(feats)
        
        flux = lc_flat.flux.value
        flux_normalized = (flux - np.mean(flux)) / np.std(flux)
        
        sequences.append(flux_normalized)
        labels_list.append(label)


df_features = pd.DataFrame(final_features)
df_features.to_csv(FEATURES_FILE, index=False)
print(f"\n✅ Tabular features saved to: {FEATURES_FILE}")



from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_seq = le.fit_transform(labels_list)


max_len = max(len(seq) for seq in sequences)
X_seq = pad_sequences(
    sequences, 
    maxlen=max_len, 
    dtype="float32", 
    padding="post", 
    truncating="post"
)


X_seq = np.expand_dims(X_seq, axis=-1)


np.save(X_SEQ_FILE, X_seq)
np.save(Y_SEQ_FILE, y_seq)

print(f"Sequence data saved to: {X_SEQ_FILE} and {Y_SEQ_FILE}")
print(f"Final CNN Input Shape: {X_seq.shape}")

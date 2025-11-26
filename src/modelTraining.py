import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# --- Configuration (Ensure these paths are correct) ---
FEATURES_FILE = "../data/tess_variable_star_features.csv" 
X_SEQ_FILE = "../data/X_seq_cnn_input.npy"              
Y_SEQ_FILE = "../data/y_seq_cnn_labels.npy"              

print("1. Loading 10,000 Star Dataset...")

df_features = pd.read_csv(FEATURES_FILE).dropna()

X_seq = np.load(X_SEQ_FILE)
y_seq = np.load(Y_SEQ_FILE) 




lb = LabelEncoder()
df_features['label_encoded'] = lb.fit_transform(df_features['label'])
target_names = lb.classes_

features = [col for col in df_features.columns if col not in ['tic_id', 'label', 'label_encoded']]
X_rf = df_features[features]
y_rf = df_features['label_encoded']

X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
    X_rf, y_rf, test_size=0.3, random_state=42, stratify=y_rf
)


X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(
    X_seq, y_seq, test_size=0.3, random_state=42, stratify=y_seq
)

print(f"Data split complete. RF Train: {len(X_train_rf)}, CNN Train: {X_train_cnn.shape}")


print("\n2. Starting Random Forest Training...")
rf_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf_model.fit(X_train_rf, y_train_rf)
print("Random Forest Training Complete.")


y_pred_rf = rf_model.predict(X_test_rf)
print("\n--- Random Forest Classification Report (10k Stars) ---")
print(classification_report(y_test_rf, y_pred_rf, target_names=target_names))


importance = rf_model.feature_importances_
feature_names = X_rf.columns
plt.figure(figsize=(10, 6))
plt.bar(feature_names, importance)
plt.xticks(rotation=45, ha='right')
plt.title('Random Forest Feature Importance')
plt.tight_layout()
plt.savefig('random_forest_feature_importance.png')
plt.show()

#New Part

y_pred_rf = rf_model.predict(X_test_rf)


report_rf = classification_report(y_test_rf, y_pred_rf, target_names=target_names)


print("\n--- Random Forest Classification Report (10k Stars) ---")
print(report_rf)


RF_REPORT_FILE = "rf_classification_report.txt"
with open(RF_REPORT_FILE, 'w') as f:
    f.write(report_rf)
print(f"✅ Random Forest report saved to {RF_REPORT_FILE}")


plt.savefig('rf_feature_importance.png') 
plt.show()



print("\n3. Checking TensorFlow GPU access:", tf.config.list_physical_devices('GPU'))

input_timesteps = X_train_cnn.shape[1] 

cnn_model = Sequential([
    Conv1D(16, 5, activation="relu", input_shape=(input_timesteps, 1)),
    MaxPooling1D(2),
    
    Conv1D(64, 5, activation="relu"),
    MaxPooling1D(2),
    
    Flatten(),
    Dense(32, activation="relu"),
    Dropout(0.2),
    
    Dense(1, activation="sigmoid")
])

cnn_model.compile(
    optimizer="adam",
    loss="binary_crossentropy", 
    metrics=["accuracy"]
)


early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=5, 
    restore_best_weights=True 
)

print("\nStarting CNN Model Training (GPU)...")
history = cnn_model.fit(
    X_train_cnn,
    y_train_cnn,
    epochs=50, 
    batch_size=32, 
    validation_data=(X_test_cnn, y_test_cnn),
    callbacks=[early_stopping], 
    verbose=1
)
print("CNN Model Training Complete.")


y_pred_cnn_probs = cnn_model.predict(X_test_cnn)
y_pred_cnn = (y_pred_cnn_probs > 0.5).astype(int)

print("\n--- CNN Classification Report (10k Stars) ---")
print(classification_report(y_test_cnn, y_pred_cnn, target_names=target_names))

#New Part 2



y_pred_cnn_probs = cnn_model.predict(X_test_cnn)
y_pred_cnn = (y_pred_cnn_probs > 0.5).astype(int)


report_cnn = classification_report(y_test_cnn, y_pred_cnn, target_names=target_names)


print("\n--- CNN Classification Report (10k Stars) ---")
print(report_cnn)

CNN_REPORT_FILE = "cnn_classification_report.txt"
with open(CNN_REPORT_FILE, 'w') as f:
    f.write(report_cnn)
print(f"✅ CNN report saved to {CNN_REPORT_FILE}")


plt.savefig('cnn_accuracy_loss_history.png')
plt.show()




plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.legend()
plt.title("CNN Training Accuracy")

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("CNN Training Loss")
plt.show()


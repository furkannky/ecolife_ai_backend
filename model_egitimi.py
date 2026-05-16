import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Veri Setinin Yüklenmesi
# heart_disease_uci 2.csv dosyanız ile aynı klasörde olmalıdır
df = pd.read_csv('heart_disease_uci 2.csv')

# 2. Hedef Değişkenin Yeniden Düzenlenmesi (Binary Classification) [cite: 14]
# 0 -> Sağlıklı (0), 0'dan büyük tüm değerler -> Hasta (1) [cite: 15]
df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)

# Klinik anlamı olmayan id ve dataset gibi meta verilerin çıkarılması 
df = df.drop(columns=['id', 'dataset', 'num'])

# 3. Eksik Verilerin Doldurulması (Imputation) [cite: 19]
num_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
cat_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

# Sayısal alanları ortalama (Mean Imputation) ile doldurma 
for col in num_cols:
    df[col] = df[col].fillna(df[col].mean())

# Kategorik alanları en sık tekrar eden değer (Mode) ile doldurma
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# 4. Kategorik Verilerin Dönüştürülmesi (One-Hot Encoding) [cite: 17]
# Kukla Değişken Tuzağını (Dummy Variable Trap) önlemek için drop_first=True [cite: 18]
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Bağımsız değişkenler (X) ve Hedef Değişken (y) ayrımı
X = df_encoded.drop(columns=['target']).astype(np.float32)
y = df_encoded['target'].values

# 5. Veri Setinin Bölünmesi (%80 Eğitim, %20 Test) [cite: 26]
# Deneylerin tekrarlanabilir olması için random_state sabitlenmiştir [cite: 27]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Veri Ölçeklendirme (Standardization) [cite: 23]
# Tüm değişkenlerin ortalaması 0, varyansı 1 olacak şekilde ayarlanır [cite: 24]
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1D-CNN (1 Boyutlu Evrişimli Sinir Ağı) için veriyi 3 boyutlu matrise çevirme
# Şekil: [örnek_sayısı, özellik_sayısı, 1] şeklinde olmalıdır
X_train_cnn = np.expand_dims(X_train_scaled, axis=-1)
X_test_cnn = np.expand_dims(X_test_scaled, axis=-1)

# 7. 1D-CNN Derin Öğrenme Model Mimarisi
model = models.Sequential([
    layers.Input(shape=(X_train_cnn.shape[1], 1)),
    
    # 1. Evrişim ve Havuzlama (Pooling) Katmanı
    layers.Conv1D(filters=32, kernel_size=2, activation='relu'),
    layers.MaxPooling1D(pool_size=2),
    
    # 2. Evrişim Katmanı
    layers.Conv1D(filters=64, kernel_size=2, activation='relu'),
    layers.Flatten(),
    
    # Tam Bağlantılı (Dense) Katmanlar
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),  # Aşırı öğrenmeyi (overfitting) engellemek için sinirleri seyrekleştirme
    layers.Dense(1, activation='sigmoid')  # İkili sınıflandırma (0 veya 1) çıktısı
])

# Raporunuzda en kritik metrik olarak belirlediğiniz 'Recall' (Duyarlılık) metriği eklenmiştir 
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Recall(name='recall')]
)

# 8. Modelin Eğitilmesi
print("Yapay Sinir Ağı (1D-CNN) Modeli Eğitiliyor...")
history = model.fit(
    X_train_cnn, y_train,
    epochs=40,
    batch_size=32,
    validation_data=(X_test_cnn, y_test),
    verbose=1
)

# 9. MODELİN KAYDEDİLMESİ (Hocanızın raporda kesin istediği adım)
model.save('ecolife_heart_model.keras')
print("\n[BAŞARILI] Derin Öğrenme Modeli 'ecolife_heart_model.keras' adıyla kaydedildi!")

# Test seti üzerinde genel performans kontrolü
loss, accuracy, recall = model.evaluate(X_test_cnn, y_test, verbose=0)
print("-" * 40)
print(f"Test Doğruluğu (Accuracy): %{accuracy*100:.2f}")
print(f"Test Duyarlılığı (Recall): %{recall*100:.2f}")
print("-" * 40)
print("Sıra bir sonraki adımda! Backend (FastAPI) köprüsünü kurmaya hazır olduğunda haber ver.")
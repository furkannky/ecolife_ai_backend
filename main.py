from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = FastAPI(title="EcoLife Yapay Zeka Sağlık API")

# 1. Eğittiğimiz ve kaydettiğimiz derin öğrenme modelini yüklüyoruz
model = tf.keras.models.load_model('ecolife_heart_model.keras')

# 2. Veri ön işleme adımlarında kullandığımız scaler'ı (ölçekleyiciyi) tutarlılık için hazırlıyoruz
# Model eğitimindeki aynı mantıkla dummy bir fit işlemi yapıyoruz
df_sample = pd.read_csv('heart_disease_uci 2.csv')
df_sample['target'] = df_sample['num'].apply(lambda x: 1 if x > 0 else 0)
df_sample = df_sample.drop(columns=['id', 'dataset', 'num', 'target'])

num_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
cat_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

for col in num_cols:
    df_sample[col] = df_sample[col].fillna(df_sample[col].mean())
for col in cat_cols:
    df_sample[col] = df_sample[col].fillna(df_sample[col].mode()[0])

df_encoded_sample = pd.get_dummies(df_sample, columns=cat_cols, drop_first=True)
X_sample = df_encoded_sample.astype(np.float32)

scaler = StandardScaler()
scaler.fit(X_sample)

# 3. Flutter'dan gelecek verilerin yapısını (şemasını) tanımlıyoruz
class KullaniciSaglikVerisi(BaseModel):
    age: float
    sex: str       # 'Male' veya 'Female'
    cp: str        # 'typical angina', 'asymptomatic', 'non-anginal', 'atypical angina'
    trestbps: float
    chol: float
    fbs: str       # 'True' veya 'False'
    restecg: str   # 'normal', 'lv hypertrophy', 'st-t abnormality'
    thalch: float
    exang: str     # 'True' veya 'False'
    oldpeak: float
    slope: str     # 'flat', 'upsloping', 'downsloping'
    ca: float      # 0.0, 1.0, 2.0, 3.0
    thal: str      # 'normal', 'fixed defect', 'reversable defect'

@app.post("/tahmin")
def tahmin_et(input_data: KullaniciSaglikVerisi):
    try:
        # Gelen veriyi tek satırlık bir DataFrame'e dönüştürüyoruz
        data_dict = {k: [v] for k, v in input_data.dict().items()}
        input_df = pd.DataFrame(data_dict)
        
        # Eğitimdeki veri formatının aynısını oluşturmak için örnek şemayla birleştiriyoruz
        final_df = pd.DataFrame(columns=X_sample.columns)
        input_encoded = pd.get_dummies(input_df, columns=cat_cols, drop_first=True)
        
        for col in final_df.columns:
            if col in input_encoded.columns:
                final_df[col] = input_encoded[col]
            else:
                final_df[col] = 0 # Olmayan kategorileri 0 (False) yapıyoruz
                
        final_df = final_df.astype(np.float32)
        
        # Veriyi ölçeklendirip 1D-CNN formatına (3 boyut) sokuyoruz
        scaled_data = scaler.transform(final_df)
        cnn_input = np.expand_dims(scaled_data, axis=-1)
        
        # Modeli koşturuyoruz
        prediction = model.predict(cnn_input)
        risk_probability = float(prediction[0][0])
        
        # Sonucu mantıksal olarak dönüyoruz
        return {
            "durum": "Basarili",
            "kalp_hastaligi_riski": "Risk Var" if risk_probability > 0.5 else "Sağlıklı / Risk Düşük",
            "olasilik_yuzdesi": round(risk_probability * 100, 2)
        }
        
    except Exception as e:
        return {"durum": "Hata", "mesaj": str(e)}

@app.get("/")
def home():
    return {"mesaj": "EcoLife Yapay Zeka Sunucusu Aktif!"}
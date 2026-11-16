# Bu modül Random Forest (Rastgele Orman) algoritması ile motorun Kalan Ömrünü (RUL) tahmin eder
# ve her uçak için 0-100 arasında Sağlık Skoru, Yakıt Verimlilik Skoru ve Termal Stabilite Skoru üretir.

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# Proje ana klasörünü Python yoluna ekle
PROJE_KLASORU = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJE_KLASORU not in sys.path:
    sys.path.insert(0, PROJE_KLASORU)

from core.data_loader import (
    GORUNUR_ISIMLER,
    filo_son_durumu_getir,
    sabit_sutunlari_bul,
    veri_setini_yukle,
)

MODEL_KLASORU = os.path.join(PROJE_KLASORU, "data", "models")

# Literatürde motor henüz yepyeniyken sensörler değişmediği için üst RUL sınırı 130 kabul edilir.
MAKSIMUM_RUL = 130


def anlamli_sensorleri_sec(train_df):
    """
    Sabit kalan sensörleri ve uçuş ayarlarını çıkarıp,
    sadece motor bozuldukça değişen 15 anlamlı sensörü seçer.
    """
    sabitler = sabit_sutunlari_bul(train_df)
    cikarilacaklar = ["motor_id", "ucus_dongusu", "kalan_omur_rul", "irtifa_kft", "mach_hizi", "gaz_kolu_acisi"] + sabitler

    secilen_sensorler = []
    for sutun in train_df.columns:
        if sutun not in cikarilacaklar:
            secilen_sensorler.append(sutun)
    return secilen_sensorler


def referans_degerleri_hesapla(train_df, sensorler):
    """
    Sensörlerin 'Sağlıklı' (RUL >= 100) ve 'Kritik/Bozulmak Üzere' (RUL <= 15)
    olduğu zamanlardaki ortalama ve standart sapma değerlerini hesaplar.
    """
    saglikli_veri = train_df[train_df["kalan_omur_rul"] >= 100]
    kritik_veri = train_df[train_df["kalan_omur_rul"] <= 15]

    saglikli_istatistik = {}
    kritik_ortalamalar = {}

    for s in sensorler:
        std_degeri = float(saglikli_veri[s].std())
        if std_degeri < 0.000001:
            std_degeri = 1.0

        saglikli_istatistik[s] = {
            "mean": float(saglikli_veri[s].mean()),
            "std": std_degeri,
        }
        kritik_ortalamalar[s] = float(kritik_veri[s].mean())

    return saglikli_istatistik, kritik_ortalamalar


def sensor_puani_hesapla(deger, sensor_adi, saglikli_istatistik, kritik_ortalamalar):
    """
    Bir sensörün değerine bakarak 0 ile 100 arasında puan verir.
    Değer sağlıklı ortalamaya yakınsa 100'e yakın, kritik ortalamaya yakınsa düşük puan alır.
    """
    saglikli_ort = saglikli_istatistik[sensor_adi]["mean"]
    kritik_ort = kritik_ortalamalar[sensor_adi]
    fark = saglikli_ort - kritik_ort

    if abs(fark) < 0.000001:
        return 100.0

    oran = (float(deger) - kritik_ort) / fark
    puan = 15.0 + 85.0 * oran
    return float(np.clip(puan, 0.0, 100.0))


def modeli_egit_veya_yukle(senaryo="FD001", yeniden_egit=False):
    """
    Eğer model daha önce eğitilip kaydedildiyse hazır yükler;
    yoksa Random Forest modelini eğitir, test eder ve kaydeder.
    """
    train_df, test_df = veri_setini_yukle(senaryo)
    os.makedirs(MODEL_KLASORU, exist_ok=True)
    model_yolu = os.path.join(MODEL_KLASORU, f"{senaryo}_rul_modeli.joblib")

    # Hazır kayıtlı model varsa doğrudan oku
    if os.path.exists(model_yolu) and not yeniden_egit:
        model_paketi = joblib.load(model_yolu)
        return model_paketi, train_df, test_df

    # 1. Adım: Anlamlı sensörleri seç ve referans istatistikleri çıkar
    sensorler = anlamli_sensorleri_sec(train_df)
    saglikli_istatistik, kritik_ortalamalar = referans_degerleri_hesapla(train_df, sensorler)

    # 2. Adım: Giriş (X) ve Hedef (y - kalan ömür) verilerini hazırla
    X_train = train_df[sensorler]
    y_train = train_df["kalan_omur_rul"].clip(upper=MAKSIMUM_RUL)

    # 3. Adım: Random Forest modelini kur ve eğit
    rf_model = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)

    # 4. Adım: Test setindeki 100 uçağın son durumu üzerinde doğruluk testi yap
    filo_test = filo_son_durumu_getir(test_df)
    tahminler = np.clip(rf_model.predict(filo_test[sensorler]), 0.0, MAKSIMUM_RUL)
    gercek_rul = filo_test["kalan_omur_rul"].to_numpy()

    metrikler = {
        "mae": round(float(mean_absolute_error(gercek_rul, tahminler)), 2),
        "rmse": round(float(root_mean_squared_error(gercek_rul, tahminler)), 2),
        "sensor_sayisi": len(sensorler),
    }

    # 5. Adım: Her şeyi tek bir sözlük (dict) içinde kaydet
    model_paketi = {
        "model": rf_model,
        "sensorler": sensorler,
        "saglikli_istatistik": saglikli_istatistik,
        "kritik_ortalamalar": kritik_ortalamalar,
        "metrikler": metrikler,
        "maksimum_rul": MAKSIMUM_RUL,
    }
    joblib.dump(model_paketi, model_yolu)
    return model_paketi, train_df, test_df


def tek_ucak_skorla(sensor_sozlugu, model_paketi):
    """
    Tek bir uçağın sensör değerlerine bakarak Tahmini RUL, Sağlık, Yakıt ve Termal skorlarını hesaplar.
    """
    sensorler = model_paketi["sensorler"]
    rf_model = model_paketi["model"]
    saglikli_ist = model_paketi["saglikli_istatistik"]
    kritik_ort = model_paketi["kritik_ortalamalar"]
    maks_rul = model_paketi["maksimum_rul"]

    # Sensörleri tek satırlık tabloya çevirip RUL tahmini yap
    tek_satir_df = pd.DataFrame([sensor_sozlugu])[sensorler]
    tahmini_rul = float(np.clip(rf_model.predict(tek_satir_df)[0], 0.0, maks_rul))

    # 1) Sağlık Skoru (0-100): Tahmini kalan ömrün 130'a oranı
    saglik_skoru = round((tahmini_rul / maks_rul) * 100.0, 1)

    # 2) Yakıt Verimlilik Skoru (0-100): Yakıt akış/basınç oranı sensöründen
    yakit_skoru = round(
        sensor_puani_hesapla(sensor_sozlugu["yakit_akis_basinc_orani"], "yakit_akis_basinc_orani", saglikli_ist, kritik_ort),
        1,
    )

    # 3) Termal Stabilite Skoru (0-100): Sıcaklık ve basınç sensörlerinin ortalaması
    termal_sensorler = ["lpt_cikis_sicakligi_r", "hpc_cikis_sicakligi_r", "hpc_cikis_statik_basinci_psia"]
    termal_puanlar = [
        sensor_puani_hesapla(sensor_sozlugu[s], s, saglikli_ist, kritik_ort) for s in termal_sensorler
    ]
    termal_skoru = round(float(np.mean(termal_puanlar)), 1)

    return {
        "tahmini_rul": round(tahmini_rul, 1),
        "saglik_skoru": saglik_skoru,
        "yakit_verimlilik_skoru": yakit_skoru,
        "termal_stabilite_skoru": termal_skoru,
    }


def filoyu_skorla(filo_df, model_paketi):
    """
    Filodaki tüm uçaklar için RUL tahminini ve 3 ana kriter skorunu hesaplayıp tabloya ekler.
    """
    tablo = filo_df.copy()
    sensorler = model_paketi["sensorler"]
    rf_model = model_paketi["model"]
    saglikli_ist = model_paketi["saglikli_istatistik"]
    kritik_ort = model_paketi["kritik_ortalamalar"]
    maks_rul = model_paketi["maksimum_rul"]

    # Tüm uçaklar için tek seferde RUL tahmini
    tahminler = np.clip(rf_model.predict(tablo[sensorler]), 0.0, maks_rul)
    tablo["tahmini_rul"] = np.round(tahminler, 1)
    tablo["saglik_skoru"] = np.round((tahminler / maks_rul) * 100.0, 1)

    # Her uçak için yakıt ve termal skorları hesapla
    yakit_listesi = []
    termal_listesi = []
    for _, satir in tablo.iterrows():
        y_puan = sensor_puani_hesapla(satir["yakit_akis_basinc_orani"], "yakit_akis_basinc_orani", saglikli_ist, kritik_ort)
        t_puan = np.mean([
            sensor_puani_hesapla(satir["lpt_cikis_sicakligi_r"], "lpt_cikis_sicakligi_r", saglikli_ist, kritik_ort),
            sensor_puani_hesapla(satir["hpc_cikis_sicakligi_r"], "hpc_cikis_sicakligi_r", saglikli_ist, kritik_ort),
            sensor_puani_hesapla(satir["hpc_cikis_statik_basinci_psia"], "hpc_cikis_statik_basinci_psia", saglikli_ist, kritik_ort),
        ])
        yakit_listesi.append(round(y_puan, 1))
        termal_listesi.append(round(float(t_puan), 1))

    tablo["yakit_verimlilik_skoru"] = yakit_listesi
    tablo["termal_stabilite_skoru"] = termal_listesi
    return tablo


def en_onemli_sensorleri_getir(model_paketi, ilk_kac=10):
    """Random Forest modeline göre RUL tahminini en çok etkileyen sensörleri sıralar."""
    onemler = model_paketi["model"].feature_importances_
    sensorler = model_paketi["sensorler"]

    sonuc_df = pd.DataFrame({
        "sensor_kodu": sensorler,
        "sensor_adi": [GORUNUR_ISIMLER.get(s, s) for s in sensorler],
        "onem_yuzdesi": np.round(onemler * 100.0, 2),
    })
    return sonuc_df.sort_values("onem_yuzdesi", ascending=False).head(ilk_kac).reset_index(drop=True)


if __name__ == "__main__":
    paket, train_df, test_df = modeli_egit_veya_yukle("FD001", yeniden_egit=True)
    print("Model Doğruluk Metrikleri:", paket["metrikler"])
    print("\nEn Önemli 5 Sensör:")
    print(en_onemli_sensorleri_getir(paket, ilk_kac=5).to_string(index=False))

    filo = filoyu_skorla(filo_son_durumu_getir(test_df), paket)
    ozet_sutunlar = ["motor_id", "ucus_dongusu", "kalan_omur_rul", "tahmini_rul", "saglik_skoru", "yakit_verimlilik_skoru", "termal_stabilite_skoru"]
    print("\nİlk 5 Uçak Skorları:")
    print(filo[ozet_sutunlar].head(5).to_string(index=False))

# 4 FARKLI MAKİNE ÖĞRENMESİ MODELİ EĞİTİM VE KIYASLAMA MODÜLÜ
# Bu dosya aynı veri seti (varsayılan FD003) üzerinde 4 farklı algoritmayı eğitir:
#   1) Doğrusal Regresyon (LinearRegression)
#   2) Karar Ağacı (DecisionTreeRegressor)
#   3) Rastgele Orman (RandomForestRegressor)
#   4) Gradient Boosting (HistGradientBoostingRegressor)
# Her modelin MAE, RMSE ve R² başarı puanlarını kıyaslar ve istenen modeli KDS'ye bağlar.

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.tree import DecisionTreeRegressor

PROJE_KLASORU = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJE_KLASORU not in sys.path:
    sys.path.insert(0, PROJE_KLASORU)

from core.omur_tahmini import (
    MAKSIMUM_OMUR,
    MODEL_KLASORU,
    anlamli_sensorleri_sec,
    referans_degerleri_hesapla,
)
from core.veri_yukleyici import filo_son_durumu_getir, veri_setini_yukle

# Kıyaslanacak 4 makine öğrenmesi algoritmasının isimleri ve açıklamaları
ALGORITMA_BILGILERI = {
    "linear_regression": {
        "ad": "Doğrusal Regresyon (Linear Regression)",
        "kisa_ad": "Doğrusal Regresyon",
        "aciklama": "En basit temel model; sensörlerle kalan ömür arasında doğrusal ilişki kurar.",
    },
    "decision_tree": {
        "ad": "Karar Ağacı (Decision Tree)",
        "kisa_ad": "Karar Ağacı",
        "aciklama": "Tek bir karar ağacı ile sensör eşiklerine göre dallanarak tahmin yapar.",
    },
    "random_forest": {
        "ad": "Rastgele Orman (Random Forest)",
        "kisa_ad": "Rastgele Orman",
        "aciklama": "Yüzlerce karar ağacının ortalamasını alarak kararlı ve güçlü tahmin üretir.",
    },
    "gradient_boosting": {
        "ad": "Gradient Boosting (HistGradientBoosting)",
        "kisa_ad": "Gradient Boosting",
        "aciklama": "Önceki ağacın hatalarını düzelterek ilerleyen yüksek doğruluklu topluluk modeli.",
    },
}


def model_nesnesi_olustur(algoritma_kodu, agac_sayisi=150, derinlik=12):
    """Seçilen algoritma koduna ve hiperparametrelere göre boş scikit-learn modelini döndürür."""
    if algoritma_kodu == "linear_regression":
        return LinearRegression()
    elif algoritma_kodu == "decision_tree":
        return DecisionTreeRegressor(max_depth=int(derinlik), random_state=42)
    elif algoritma_kodu == "gradient_boosting":
        # Gradient Boosting ağaç derinliğini daha sığ (ör. 4-10) tutmak daha iyi genelleme sağlar
        gb_derinlik = min(int(derinlik), 10)
        return HistGradientBoostingRegressor(
            max_iter=int(agac_sayisi),
            max_depth=gb_derinlik,
            learning_rate=0.05,
            random_state=42,
        )
    else:
        return RandomForestRegressor(
            n_estimators=int(agac_sayisi),
            max_depth=int(derinlik),
            random_state=42,
            n_jobs=-1,
        )


def sensor_onemlerini_cikar(egitilmis_model, X_train, y_train, sensorler):
    """
    Ağaç modellerinden doğrudan öznitelik önemini alır;
    doğrudan desteklemeyen modellerde (ör. Gradient Boosting / Lineer) korelasyon tabanlı önem hesaplar.
    """
    if hasattr(egitilmis_model, "feature_importances_"):
        ham = np.array(egitilmis_model.feature_importances_, dtype=float)
    elif hasattr(egitilmis_model, "coef_"):
        # Katsayıları sensörlerin standart sapmasıyla çarparak ölçekten bağımsız etkiyi bul
        stdler = X_train[sensorler].std().to_numpy(dtype=float)
        ham = np.abs(np.array(egitilmis_model.coef_, dtype=float) * stdler)
    else:
        # Gradient Boosting gibi modellerde sensör-hedef mutlak korelasyonu ile temiz önem payı çıkar
        kor = X_train[sensorler].corrwith(y_train).abs().fillna(0.0).to_numpy(dtype=float)
        ham = kor

    toplam = float(np.sum(ham))
    if toplam <= 0.0:
        return {s: round(100.0 / len(sensorler), 2) for s in sensorler}
    return {s: round(float(v / toplam * 100.0), 2) for s, v in zip(sensorler, ham)}


def tek_model_egit(
    senaryo="FD003",
    algoritma_kodu="gradient_boosting",
    agac_sayisi=150,
    derinlik=10,
    maks_omur=MAKSIMUM_OMUR,
    kaydet=True,
    hazir_veriler=None,
):
    """
    Seçilen tek bir algoritmayı eğitir, 100 test uçağında ölçer ve standart model_paketi döndürür.
    """
    if hazir_veriler is not None:
        train_df, test_df = hazir_veriler
    else:
        train_df, test_df = veri_setini_yukle(senaryo)

    sensorler = anlamli_sensorleri_sec(train_df)
    saglikli_istatistik, kritik_ortalamalar = referans_degerleri_hesapla(train_df, sensorler)

    X_train = train_df[sensorler]
    y_train = train_df["kalan_ucus_omru"].clip(upper=int(maks_omur))

    filo_test = filo_son_durumu_getir(test_df)
    X_test = filo_test[sensorler]
    y_test = filo_test["kalan_ucus_omru"].to_numpy(dtype=float)

    basla = time.perf_counter()
    model = model_nesnesi_olustur(algoritma_kodu, agac_sayisi=agac_sayisi, derinlik=derinlik)
    model.fit(X_train, y_train)
    egitim_suresi = round(time.perf_counter() - basla, 2)

    tahminler = np.clip(model.predict(X_test), 0.0, float(maks_omur))

    bilgi = ALGORITMA_BILGILERI.get(algoritma_kodu, ALGORITMA_BILGILERI["random_forest"])
    metrikler = {
        "algoritma": bilgi["ad"],
        "kisa_ad": bilgi["kisa_ad"],
        "algoritma_kodu": algoritma_kodu,
        "aciklama": bilgi["aciklama"],
        "senaryo": senaryo,
        "mae": round(float(mean_absolute_error(y_test, tahminler)), 2),
        "rmse": round(float(root_mean_squared_error(y_test, tahminler)), 2),
        "r2": round(float(r2_score(y_test, tahminler)), 3),
        "egitim_suresi_sn": egitim_suresi,
        "sensor_sayisi": len(sensorler),
        "agac_sayisi": int(agac_sayisi),
        "derinlik": int(derinlik),
        "maks_omur": int(maks_omur),
    }

    # Gerçek vs Tahmin grafiği için 100 test uçağının karşılaştırma noktaları
    karsilastirma_noktalari = []
    for u_id, g_val, t_val in zip(filo_test["ucak_id"], y_test, tahminler):
        karsilastirma_noktalari.append({
            "ucak_id": int(u_id),
            "gercek_omur": round(float(g_val), 1),
            "tahmini_omur": round(float(t_val), 1),
            "hata": round(float(abs(g_val - t_val)), 1),
        })

    sensor_onemleri = sensor_onemlerini_cikar(model, X_train, y_train, sensorler)

    model_paketi = {
        "model": model,
        "senaryo": senaryo,
        "algoritma": bilgi["ad"],
        "algoritma_kodu": algoritma_kodu,
        "sensorler": sensorler,
        "saglikli_istatistik": saglikli_istatistik,
        "kritik_ortalamalar": kritik_ortalamalar,
        "sensor_onemleri": sensor_onemleri,
        "metrikler": metrikler,
        "maksimum_omur": int(maks_omur),
        "karsilastirma_noktalari": karsilastirma_noktalari,
    }

    if kaydet:
        os.makedirs(MODEL_KLASORU, exist_ok=True)
        model_yolu = os.path.join(MODEL_KLASORU, f"{senaryo}_omur_modeli.joblib")
        joblib.dump(model_paketi, model_yolu)

    return model_paketi


def dort_modeli_egit_ve_karsilastir(senaryo="FD003", en_iyiyi_kaydet=True):
    """
    Aynı veri setinde 4 farklı ML algoritmasını eğitip test eder,
    MAE / RMSE / R² skorlarına göre kıyaslar ve en düşük MAE'ye sahip modeli seçer.
    """
    train_df, test_df = veri_setini_yukle(senaryo)
    ayarlar = [
        ("linear_regression", 1, 1),
        ("decision_tree", 1, 10),
        ("random_forest", 150, 12),
        ("gradient_boosting", 200, 6),
    ]

    paketler = {}
    karsilastirma_listesi = []

    for kod, agac, derinlik in ayarlar:
        paket = tek_model_egit(
            senaryo=senaryo,
            algoritma_kodu=kod,
            agac_sayisi=agac,
            derinlik=derinlik,
            maks_omur=MAKSIMUM_OMUR,
            kaydet=False,
            hazir_veriler=(train_df, test_df),
        )
        paketler[kod] = paket
        karsilastirma_listesi.append(paket["metrikler"])

    # En düşük MAE (Ortalama Mutlak Hata) değerine sahip modeli "En İyi Model" seç
    karsilastirma_listesi.sort(key=lambda m: m["mae"])
    en_iyi_kod = karsilastirma_listesi[0]["algoritma_kodu"]
    for m in karsilastirma_listesi:
        m["en_iyi_mi"] = (m["algoritma_kodu"] == en_iyi_kod)

    en_iyi_paket = paketler[en_iyi_kod]
    en_iyi_paket["karsilastirma_tablosu"] = karsilastirma_listesi

    if en_iyiyi_kaydet:
        os.makedirs(MODEL_KLASORU, exist_ok=True)
        model_yolu = os.path.join(MODEL_KLASORU, f"{senaryo}_omur_modeli.joblib")
        joblib.dump(en_iyi_paket, model_yolu)

    return en_iyi_paket, karsilastirma_listesi, paketler


if __name__ == "__main__":
    en_iyi, tablo, _ = dort_modeli_egit_ve_karsilastir(senaryo="FD003", en_iyiyi_kaydet=True)
    print("\n================ FD003 — 4 ML MODELİ KARŞILAŞTIRMA SONUCU ================")
    df_tablo = pd.DataFrame(tablo)[["kisa_ad", "mae", "rmse", "r2", "egitim_suresi_sn", "en_iyi_mi"]]
    print(df_tablo.to_string(index=False))
    print(f"\nOtomatik Seçilen ve Kaydedilen En İyi Model: {en_iyi['algoritma']} (MAE={en_iyi['metrikler']['mae']}, R2={en_iyi['metrikler']['r2']})")

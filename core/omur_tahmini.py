# Bu dosya 3 rolün tamamını besleyen ortak yapay zeka motorudur.
# Aktif seçili makine öğrenmesi modeli ile uçağın Tahmini Kalan Uçuş Ömrünü hesaplar
# ve 0-100 arasında Motor Sağlığı, Yakıt Verimliliği ve Termal Denge puanlarını üretir.
# Varsayılan senaryo: FD003 (2 arıza tipi: Ana Hava Sıkıştırıcı + Ön Pervane / Fan).

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

PROJE_KLASORU = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJE_KLASORU not in sys.path:
    sys.path.insert(0, PROJE_KLASORU)

from core.veri_yukleyici import (
    GORUNUR_ISIMLER,
    filo_son_durumu_getir,
    sabit_sutunlari_bul,
    veri_setini_yukle,
)

MODEL_KLASORU = os.path.join(PROJE_KLASORU, "data", "models")

# Motor henüz yeniyken sensörler değişmediği için üst ömür sınırı 130 uçuş kabul edilir.
MAKSIMUM_OMUR = 130


def anlamli_sensorleri_sec(train_df):
    """
    Sabit kalan gereksiz sensörleri ve uçuş ayarlarını çıkarıp,
    motor yıprandıkça gerçekten değişen anlamlı sensörleri seçer (FD003'te 16 sensör).
    """
    sabitler = sabit_sutunlari_bul(train_df)
    cikarilacaklar = [
        "ucak_id",
        "ucus_sayisi",
        "kalan_ucus_omru",
        "irtifa",
        "mach_hizi",
        "gaz_kolu_acisi",
    ] + sabitler

    return [sutun for sutun in train_df.columns if sutun not in cikarilacaklar]


def referans_degerleri_hesapla(train_df, sensorler):
    """
    Sensörlerin 'Sağlıklı' (kalan ömür >= 100) dönemindeki ortalama ve standart sapmalarını bulur.
    """
    saglikli_veri = train_df[train_df["kalan_ucus_omru"] >= 100]
    kritik_veri = train_df[train_df["kalan_ucus_omru"] <= 15]

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


def sensor_puani_hesapla(deger, sensor_adi, saglikli_istatistik):
    """
    İki yönlü (Z-skoru tabanlı) sensör sağlık puanı (0-100):
    FD003'te 2 farklı arıza modu olduğu için sensör normalin çok altına da düşse,
    çok üstüne de çıksa bozulma işaretidir.
    |Z| <= 0.5 ise 100 puan; |Z| >= 4.5 ise 0 puan verir.
    """
    if sensor_adi not in saglikli_istatistik:
        return 50.0
    saglikli_ort = saglikli_istatistik[sensor_adi]["mean"]
    saglikli_std = saglikli_istatistik[sensor_adi]["std"]

    mutlak_z = abs((float(deger) - saglikli_ort) / saglikli_std)
    if mutlak_z <= 0.5:
        return 100.0

    puan = 100.0 - ((mutlak_z - 0.5) / 4.0) * 100.0
    return float(np.clip(puan, 0.0, 100.0))


def modeli_egit_veya_yukle(senaryo="FD003", yeniden_egit=False):
    """
    Kayıtlı aktif yapay zeka modeli varsa hızlıca yükler;
    yoksa varsayılan Random Forest modelini eğitir, test eder ve kaydeder.
    """
    train_df, test_df = veri_setini_yukle(senaryo)
    os.makedirs(MODEL_KLASORU, exist_ok=True)
    model_yolu = os.path.join(MODEL_KLASORU, f"{senaryo}_omur_modeli.joblib")

    if os.path.exists(model_yolu) and not yeniden_egit:
        model_paketi = joblib.load(model_yolu)
        return model_paketi, train_df, test_df

    sensorler = anlamli_sensorleri_sec(train_df)
    saglikli_istatistik, kritik_ortalamalar = referans_degerleri_hesapla(train_df, sensorler)

    X_train = train_df[sensorler]
    y_train = train_df["kalan_ucus_omru"].clip(upper=MAKSIMUM_OMUR)

    rf_model = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)

    filo_test = filo_son_durumu_getir(test_df)
    tahminler = np.clip(rf_model.predict(filo_test[sensorler]), 0.0, MAKSIMUM_OMUR)
    gercek_omur = filo_test["kalan_ucus_omru"].to_numpy()

    metrikler = {
        "algoritma": "Rastgele Orman (Random Forest)",
        "algoritma_kodu": "random_forest",
        "senaryo": senaryo,
        "mae": round(float(mean_absolute_error(gercek_omur, tahminler)), 2),
        "rmse": round(float(root_mean_squared_error(gercek_omur, tahminler)), 2),
        "r2": round(float(r2_score(gercek_omur, tahminler)), 3),
        "sensor_sayisi": len(sensorler),
    }

    sensor_onemleri = {
        s: round(float(imp * 100.0), 2)
        for s, imp in zip(sensorler, rf_model.feature_importances_)
    }

    model_paketi = {
        "model": rf_model,
        "senaryo": senaryo,
        "algoritma": metrikler["algoritma"],
        "algoritma_kodu": metrikler["algoritma_kodu"],
        "sensorler": sensorler,
        "saglikli_istatistik": saglikli_istatistik,
        "kritik_ortalamalar": kritik_ortalamalar,
        "sensor_onemleri": sensor_onemleri,
        "metrikler": metrikler,
        "maksimum_omur": MAKSIMUM_OMUR,
    }
    joblib.dump(model_paketi, model_yolu)
    return model_paketi, train_df, test_df


def tek_ucak_skorla(sensor_sozlugu, model_paketi):
    """
    Tek bir uçağın sensörlerine bakarak Tahmini Kalan Ömrü, Motor Sağlığını,
    Yakıt Verimliliğini ve Termal Denge skorlarını hesaplar.
    """
    sensorler = model_paketi["sensorler"]
    aktif_model = model_paketi["model"]
    saglikli_ist = model_paketi["saglikli_istatistik"]
    maks_omur = model_paketi["maksimum_omur"]

    tek_satir_df = pd.DataFrame([sensor_sozlugu])[sensorler]
    tahmini_omur = float(np.clip(aktif_model.predict(tek_satir_df)[0], 0.0, maks_omur))

    saglik_skoru = round((tahmini_omur / maks_omur) * 100.0, 1)
    yakit_skoru = round(
        sensor_puani_hesapla(sensor_sozlugu["yakit_verim_orani"], "yakit_verim_orani", saglikli_ist),
        1,
    )

    termal_sensorler = ["egzoz_turbin_sicakligi", "ana_sikistirici_sicakligi", "ana_sikistirici_basinci"]
    termal_puanlar = [
        sensor_puani_hesapla(sensor_sozlugu[s], s, saglikli_ist) for s in termal_sensorler
    ]
    termal_skoru = round(float(np.mean(termal_puanlar)), 1)

    return {
        "tahmini_kalan_omur": round(tahmini_omur, 1),
        "saglik_skoru": saglik_skoru,
        "yakit_verimlilik_skoru": yakit_skoru,
        "termal_denge_skoru": termal_skoru,
    }


def filoyu_skorla(filo_df, model_paketi):
    """
    Filodaki 100 uçağın tamamı için Tahmini Kalan Ömür ve 3 ana puanı hesaplayıp tabloya ekler.
    """
    tablo = filo_df.copy()
    sensorler = model_paketi["sensorler"]
    aktif_model = model_paketi["model"]
    saglikli_ist = model_paketi["saglikli_istatistik"]
    maks_omur = model_paketi["maksimum_omur"]

    tahminler = np.clip(aktif_model.predict(tablo[sensorler]), 0.0, maks_omur)
    tablo["tahmini_kalan_omur"] = np.round(tahminler, 1)
    tablo["saglik_skoru"] = np.round((tahminler / maks_omur) * 100.0, 1)

    yakit_listesi = []
    termal_listesi = []
    for _, satir in tablo.iterrows():
        y_puan = sensor_puani_hesapla(satir["yakit_verim_orani"], "yakit_verim_orani", saglikli_ist)
        t_puan = np.mean([
            sensor_puani_hesapla(satir["egzoz_turbin_sicakligi"], "egzoz_turbin_sicakligi", saglikli_ist),
            sensor_puani_hesapla(satir["ana_sikistirici_sicakligi"], "ana_sikistirici_sicakligi", saglikli_ist),
            sensor_puani_hesapla(satir["ana_sikistirici_basinci"], "ana_sikistirici_basinci", saglikli_ist),
        ])
        yakit_listesi.append(round(y_puan, 1))
        termal_listesi.append(round(float(t_puan), 1))

    tablo["yakit_verimlilik_skoru"] = yakit_listesi
    tablo["termal_denge_skoru"] = termal_listesi
    return tablo


def en_onemli_sensorleri_getir(model_paketi, ilk_kac=10):
    """Motor ömrünü en çok belirleyen sensörleri etki yüzdesine göre sıralar."""
    sensorler = model_paketi["sensorler"]
    onem_sozlugu = model_paketi.get("sensor_onemleri")

    if onem_sozlugu:
        onem_yuzdeleri = [float(onem_sozlugu.get(s, 0.0)) for s in sensorler]
    elif hasattr(model_paketi["model"], "feature_importances_"):
        onem_yuzdeleri = [round(float(x * 100.0), 2) for x in model_paketi["model"].feature_importances_]
    else:
        onem_yuzdeleri = [round(100.0 / len(sensorler), 2)] * len(sensorler)

    sonuc_df = pd.DataFrame({
        "sensor_kodu": sensorler,
        "sensor_adi": [GORUNUR_ISIMLER.get(s, s) for s in sensorler],
        "onem_yuzdesi": onem_yuzdeleri,
    })
    return sonuc_df.sort_values("onem_yuzdesi", ascending=False).head(ilk_kac).reset_index(drop=True)


if __name__ == "__main__":
    paket, train_df, test_df = modeli_egit_veya_yukle("FD003", yeniden_egit=True)
    print("Model Doğruluk Metrikleri (FD003):", paket["metrikler"])
    print("\nEn Önemli 5 Sensör (FD003):")
    print(en_onemli_sensorleri_getir(paket, ilk_kac=5).to_string(index=False))

    filo = filoyu_skorla(filo_son_durumu_getir(test_df), paket)
    ozet_sutunlar = ["ucak_id", "ucus_sayisi", "kalan_ucus_omru", "tahmini_kalan_omur", "saglik_skoru", "yakit_verimlilik_skoru", "termal_denge_skoru"]
    print("\nÖrnek Uçaklar (Sağlam #37, Orta #7, Kompresör Kritik #50, Fan Kritik #82):")
    print(filo[filo["ucak_id"].isin([37, 7, 50, 82])][ozet_sutunlar].to_string(index=False))

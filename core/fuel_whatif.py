# Bu modül Pilot ekranı için "What-If" (Ya Şöyle Olursa?) simülasyonunu yapar.
# Havadayken bir sensör arızalandığında yakıt tüketiminin ne kadar artacağını,
# kalan yakıtın hedefe yetip yetmeyeceğini ve pilotun ne yapması gerektiğini hesaplar.

import os
import sys
import numpy as np

PROJE_KLASORU = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJE_KLASORU not in sys.path:
    sys.path.insert(0, PROJE_KLASORU)

from core.fault_diagnosis import ariza_teshisi_yap
from core.rul_predictor import tek_ucak_skorla

# Uçuş moduna göre normal yakıt tüketimi (kg / km)
UCUS_MODU_TUKETIM = {
    "ekonomi": 2.70,   # Yüksek irtifa tasarruf uçuşu
    "standart": 3.20,  # Normal görev uçuşu
    "muharebe": 4.50,  # Yüksek hız / art yakıcı (afterburner) ağırlıklı uçuş
}

# Hazır "What-If" arıza senaryoları (sensörlere eklenecek standart sapma miktarları)
HAZIR_ARIZA_SENARYOLARI = {
    "normal": {},
    "hpc_kompresor_asinmasi": {
        "hpc_cikis_statik_basinci_psia": 3.2,
        "lpt_cikis_sicakligi_r": 2.8,
        "hpc_cikis_toplam_basinci_psia": -2.5,
    },
    "yakit_enjektor_verim_kaybi": {
        "yakit_akis_basinc_orani": -3.5,
        "lpt_cikis_sicakligi_r": 2.4,
    },
    "turbin_sogutma_kaybi": {
        "hpt_sogutma_havasi_lbms": -3.0,
        "lpt_sogutma_havasi_lbms": -3.0,
        "lpt_cikis_sicakligi_r": 3.5,
    },
    "agir_coklu_ariza": {
        "hpc_cikis_statik_basinci_psia": 4.0,
        "lpt_cikis_sicakligi_r": 3.8,
        "yakit_akis_basinc_orani": -4.0,
        "cekirdek_devri_rpm": 3.0,
    },
}


def whatif_simulasyonu_calistir(
    motor_id,
    mevcut_sensorler,
    model_paketi,
    mevcut_yakit_kg=3000.0,
    hedef_menzil_km=800.0,
    en_yakin_us_km=220.0,
    ucus_modu="standart",
    senaryo_adi="normal",
    ozel_sensor_degisimleri=None,
    ucus_dongusu=0,
):
    """
    1) Arıza öncesi durumu hesaplar.
    2) Seçilen arıza senaryosunu sensörlere uygular.
    3) Arıza sonrası yeni sağlık skorunu, yakıt tüketim artışını ve kalan menzili bulur.
    4) Pilota 'Göreve Devam', 'İrtifa/Hız Düşür' veya 'Acil Üsse Dön (RTB)' kararı verir.
    """
    saglikli_ist = model_paketi["saglikli_istatistik"]

    # 1. Adım: Arıza öncesi (mevcut) sağlık ve RUL değerlerini al
    onceki_durum = tek_ucak_skorla(mevcut_sensorler, model_paketi)

    # 2. Adım: Sensörlere yapay arıza / değişim uygula
    yeni_sensorler = dict(mevcut_sensorler)
    degisen_sensorler = []

    uygulanacak_sapmalar = HAZIR_ARIZA_SENARYOLARI.get(senaryo_adi, {})
    for sensor_adi, z_farki in uygulanacak_sapmalar.items():
        if sensor_adi in yeni_sensorler and sensor_adi in saglikli_ist:
            eski_deger = float(yeni_sensorler[sensor_adi])
            std_degeri = saglikli_ist[sensor_adi]["std"]
            yeni_deger = eski_deger + (z_farki * std_degeri)
            yeni_sensorler[sensor_adi] = yeni_deger
            degisen_sensorler.append({
                "sensor": sensor_adi,
                "onceki_deger": round(eski_deger, 2),
                "yeni_deger": round(yeni_deger, 2),
            })

    # Eğer kullanıcı arayüzden doğrudan yeni sensör sayısı girdiyse onu da uygula
    if ozel_sensor_degisimleri is not None:
        for sensor_adi, yeni_deger in ozel_sensor_degisimleri.items():
            if sensor_adi in yeni_sensorler:
                eski_deger = float(yeni_sensorler[sensor_adi])
                yeni_sensorler[sensor_adi] = float(yeni_deger)
                degisen_sensorler.append({
                    "sensor": sensor_adi,
                    "onceki_deger": round(eski_deger, 2),
                    "yeni_deger": round(float(yeni_deger), 2),
                })

    # 3. Adım: Arıza sonrası yeni RUL, Sağlık ve Teşhis raporunu hesapla
    yeni_durum = tek_ucak_skorla(yeni_sensorler, model_paketi)
    teshis = ariza_teshisi_yap(
        motor_id=motor_id,
        sensor_sozlugu=yeni_sensorler,
        tahmini_rul=yeni_durum["tahmini_rul"],
        saglik_skoru=yeni_durum["saglik_skoru"],
        saglikli_istatistik=saglikli_ist,
        ucus_dongusu=ucus_dongusu,
    )

    # 4. Adım: Yakıt tüketim artışı ve kalan menzil hesabı
    baz_tuketim = UCUS_MODU_TUKETIM.get(ucus_modu, 3.20)
    yakit_kaybi = (100.0 - yeni_durum["yakit_verimlilik_skoru"]) / 100.0
    termal_kayip = (100.0 - yeni_durum["termal_stabilite_skoru"]) / 100.0

    # Motor verimi düştükçe ve ısındıkça km başına yakıt tüketimi %0 ile %40 arası artar
    yakit_artisi_yuzde = float(np.clip((yakit_kaybi * 22.0) + (termal_kayip * 13.0), 0.0, 40.0))
    yeni_tuketim = baz_tuketim * (1.0 + yakit_artisi_yuzde / 100.0)

    # Maksimum menzil ve hedef / yedek üs için gereken (%10 güvenlik rezervli) yakıt
    maksimum_menzil_km = mevcut_yakit_kg / yeni_tuketim
    gerekli_yakit_hedef_kg = hedef_menzil_km * yeni_tuketim * 1.10
    gerekli_yakit_us_kg = en_yakin_us_km * yeni_tuketim * 1.10

    hedefe_yeterli_mi = mevcut_yakit_kg >= gerekli_yakit_hedef_kg
    usse_yeterli_mi = mevcut_yakit_kg >= gerekli_yakit_us_kg

    # 5. Adım: Pilot Karar Kuralları (if - elif - else)
    yeni_rul = yeni_durum["tahmini_rul"]
    yeni_saglik = yeni_durum["saglik_skoru"]
    kritik_sensor_sayisi = sum(1 for a in teshis["anomaliler"] if a["durum"] == "KRITIK")
    uyari_sensor_sayisi = sum(1 for a in teshis["anomaliler"] if a["durum"] == "UYARI")

    if (not hedefe_yeterli_mi) or yeni_rul < 25.0 or yeni_saglik < 40.0 or kritik_sensor_sayisi >= 3:
        risk_seviyesi = "KRITIK"
        pilot_karari = "ACIL_INIS_RTB (En Yakın Üsse Acil Dönüş)"
        if not hedefe_yeterli_mi:
            gerekce = (
                f"Yakıt tüketimi +%{yakit_artisi_yuzde:.1f} arttığı için maksimum menzil "
                f"({maksimum_menzil_km:.0f} km) görev hedefini ({hedef_menzil_km:.0f} km) karşılamıyor."
            )
        else:
            gerekce = f"Motor sağlığı kritik seviyeye (%{yeni_saglik:.1f}) düştü; görevi iptal edip en yakın üsse dönün."
    elif yeni_saglik < 82.0 or yakit_artisi_yuzde >= 4.0 or kritik_sensor_sayisi >= 1 or uyari_sensor_sayisi >= 2:
        risk_seviyesi = "ORTA"
        pilot_karari = "IRTIFA_HIZ_DUSUR (Kısıtlı Görev / Ekonomi Moduna Geç)"
        gerekce = (
            f"Sensör sapması nedeniyle yakıt tüketimi +%{yakit_artisi_yuzde:.1f} arttı, "
            f"sağlık %{yeni_saglik:.1f}'e geriledi. İrtifa/hız düşürerek ekonomi moduna geçin."
        )
    else:
        risk_seviyesi = "DUSUK"
        pilot_karari = "GOREVE_DEVAM (Normal Operasyon)"
        gerekce = f"Motor sağlığı (%{yeni_saglik:.1f}) ve menzil ({maksimum_menzil_km:.0f} km) görev için güvenli."

    return {
        "motor_id": motor_id,
        "senaryo_adi": senaryo_adi,
        "onceki_rul": onceki_durum["tahmini_rul"],
        "yeni_rul": yeni_rul,
        "onceki_saglik": onceki_durum["saglik_skoru"],
        "yeni_saglik": yeni_saglik,
        "baz_tuketim_kg_km": round(baz_tuketim, 2),
        "yeni_tuketim_kg_km": round(yeni_tuketim, 2),
        "yakit_artisi_yuzde": round(yakit_artisi_yuzde, 1),
        "mevcut_yakit_kg": round(mevcut_yakit_kg, 1),
        "gerekli_yakit_hedef_kg": round(gerekli_yakit_hedef_kg, 1),
        "gerekli_yakit_us_kg": round(gerekli_yakit_us_kg, 1),
        "maksimum_menzil_km": round(maksimum_menzil_km, 1),
        "hedefe_yeterli_mi": hedefe_yeterli_mi,
        "usse_yeterli_mi": usse_yeterli_mi,
        "risk_seviyesi": risk_seviyesi,
        "pilot_karari": pilot_karari,
        "karar_gerekcesi": gerekce,
        "teshis_raporu": teshis,
        "degisen_sensorler": degisen_sensorler,
    }


if __name__ == "__main__":
    from core.data_loader import filo_son_durumu_getir
    from core.rul_predictor import modeli_egit_veya_yukle

    paket, _, test_df = modeli_egit_veya_yukle("FD001")
    filo = filo_son_durumu_getir(test_df)
    satir_1 = filo[filo["motor_id"] == 1].iloc[0]
    sensorler_1 = {s: float(satir_1[s]) for s in paket["sensorler"]}

    for senaryo in ["normal", "hpc_kompresor_asinmasi", "agir_coklu_ariza"]:
        sonuc = whatif_simulasyonu_calistir(
            motor_id=1,
            mevcut_sensorler=sensorler_1,
            model_paketi=paket,
            senaryo_adi=senaryo,
            ucus_dongusu=int(satir_1["ucus_dongusu"]),
        )
        print(f"\n[{senaryo.upper()}] -> Sağlık: %{sonuc['onceki_saglik']} -> %{sonuc['yeni_saglik']} | Yakıt: +{sonuc['yakit_artisi_yuzde']}%")
        print(f"  Karar: {sonuc['pilot_karari']} | Gerekçe: {sonuc['karar_gerekcesi']}")

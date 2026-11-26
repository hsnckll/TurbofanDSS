# 3. ROL: PİLOT / UÇUŞ KONTROL YAKIT VE ARIZA SİMÜLASYONU ("YA ŞÖYLE OLURSA?")
# Bu dosya havadayken bir motor arızası yaşandığında yakıt tüketiminin ne kadar artacağını,
# mevcut yakıtın hedefe veya en yakın üsse yetip yetmeyeceğini ve pilotun kararını hesaplar.

import os
import sys
import numpy as np

PROJE_KLASORU = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJE_KLASORU not in sys.path:
    sys.path.insert(0, PROJE_KLASORU)

from core.bakim_muhendisi_teshis import ariza_teshisi_yap
from core.omur_tahmini import tek_ucak_skorla

# Uçuş moduna göre normal yakıt tüketimi (km başına kg yakıt)
UCUS_MODU_TUKETIM = {
    "ekonomi": 2.70,   # Tasarruflu sakin uçuş
    "standart": 3.20,  # Normal görev uçuşu
    "muharebe": 4.50,  # Yüksek hızlı tam güç uçuşu
}

# Hazır arıza senaryoları — yalnızca NASA FD003 veri setindeki 2 gerçek arıza modu (HPC ve Ön Pervane)
HAZIR_ARIZA_SENARYOLARI = {
    "normal": {},
    "sikistirici_asinmasi": {
        "ana_sikistirici_basinci": 3.2,
        "egzoz_turbin_sicakligi": 2.8,
        "ana_sikistirici_toplam_basinci": -2.0,
        "dis_ic_hava_orani": 2.8,
    },
    "on_pervane_fan_arizasi": {
        "on_pervane_hizi_rpm": 3.2,
        "dis_ic_hava_orani": -3.5,
        "ana_sikistirici_basinci": 3.5,
        "egzoz_turbin_sicakligi": 2.8,
        "yakit_verim_orani": 4.0,
    },
}


def pilot_simulasyonu_calistir(
    ucak_id,
    mevcut_sensorler,
    model_paketi,
    mevcut_yakit_kg=3000.0,
    hedef_menzil_km=800.0,
    en_yakin_us_km=220.0,
    ucus_modu="standart",
    senaryo_adi="normal",
    ozel_sensor_degisimleri=None,
    ucus_sayisi=0,
):
    """
    1) Arıza öncesi durumu hesaplar.
    2) Seçilen arıza senaryosunu sensörlere uygular.
    3) Arıza sonrası yeni motor sağlığını, yakıt artışını ve maksimum menzili bulur.
    4) Pilota 'GÖREVE DEVAM ET', 'HIZ VE İRTİFA DÜŞÜR' veya 'ACİL ÜSSE DÖN' kararı verir.
    """
    saglikli_ist = model_paketi["saglikli_istatistik"]

    # 1. Adım: Arıza öncesi (mevcut) sağlık ve kalan ömür
    onceki_durum = tek_ucak_skorla(mevcut_sensorler, model_paketi)

    # 2. Adım: Sensörlere seçilen arıza senaryosunu uygula
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

    # Kullanıcı arayüzden elle ek sensör değişimi verdiyse uygula
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

    # 3. Adım: Arıza sonrası yeni kalan ömür, sağlık ve teşhis raporu
    yeni_durum = tek_ucak_skorla(yeni_sensorler, model_paketi)
    teshis = ariza_teshisi_yap(
        ucak_id=ucak_id,
        sensor_sozlugu=yeni_sensorler,
        tahmini_kalan_omur=yeni_durum["tahmini_kalan_omur"],
        saglik_skoru=yeni_durum["saglik_skoru"],
        saglikli_istatistik=saglikli_ist,
        ucus_sayisi=ucus_sayisi,
    )

    # 4. Adım: Yakıt tüketim artışı ve kalan menzil hesabı
    baz_tuketim = UCUS_MODU_TUKETIM.get(ucus_modu, 3.20)
    yakit_kaybi = (100.0 - yeni_durum["yakit_verimlilik_skoru"]) / 100.0
    termal_kayip = (100.0 - yeni_durum["termal_denge_skoru"]) / 100.0

    # Motor verimi düştükçe ve ısındıkça yakıt tüketimi %0 ile %40 arası artar
    yakit_artisi_yuzde = float(np.clip((yakit_kaybi * 22.0) + (termal_kayip * 13.0), 0.0, 40.0))
    yeni_tuketim = baz_tuketim * (1.0 + yakit_artisi_yuzde / 100.0)

    # Maksimum menzil ve %10 güvenlik rezervli yakıt ihtiyacı
    maksimum_menzil_km = mevcut_yakit_kg / yeni_tuketim
    gerekli_yakit_hedef_kg = hedef_menzil_km * yeni_tuketim * 1.10
    gerekli_yakit_us_kg = en_yakin_us_km * yeni_tuketim * 1.10

    hedefe_yeterli_mi = mevcut_yakit_kg >= gerekli_yakit_hedef_kg
    usse_yeterli_mi = mevcut_yakit_kg >= gerekli_yakit_us_kg

    # 5. Adım: Pilot Karar Kuralları (Sade Türkçe)
    yeni_omur = yeni_durum["tahmini_kalan_omur"]
    yeni_saglik = yeni_durum["saglik_skoru"]
    kritik_sensor_sayisi = sum(1 for a in teshis["anomaliler"] if a["durum"] == "KRİTİK")
    uyari_sensor_sayisi = sum(1 for a in teshis["anomaliler"] if a["durum"] == "UYARI")

    if (not hedefe_yeterli_mi) or yeni_omur < 25.0 or yeni_saglik < 40.0 or kritik_sensor_sayisi >= 3:
        risk_seviyesi = "KRİTİK"
        pilot_karari = "ACİL ÜSSE DÖN (Görevi İptal Et)"
        if not hedefe_yeterli_mi:
            gerekce = (
                f"Yakıt tüketimi +%{yakit_artisi_yuzde:.1f} arttığı için maksimum menzil "
                f"({maksimum_menzil_km:.0f} km) görev hedefini ({hedef_menzil_km:.0f} km) karşılamıyor."
            )
        else:
            gerekce = (
                f"Motor sağlığı kritik seviyeye (%{yeni_saglik:.1f}) düştü; "
                f"görevi iptal edip en yakın üsse ({en_yakin_us_km:.0f} km) dönün."
            )
    elif yeni_saglik < 82.0 or yakit_artisi_yuzde >= 4.0 or kritik_sensor_sayisi >= 1 or uyari_sensor_sayisi >= 2:
        risk_seviyesi = "ORTA"
        pilot_karari = "HIZ VE İRTİFA DÜŞÜR (Tasarruf Modu)"
        gerekce = (
            f"Sensör sapması nedeniyle yakıt tüketimi +%{yakit_artisi_yuzde:.1f} arttı, "
            f"motor sağlığı %{yeni_saglik:.1f} seviyesine geriledi. Tasarruf moduna geçin."
        )
    else:
        risk_seviyesi = "DÜŞÜK"
        pilot_karari = "GÖREVE DEVAM ET (Normal Uçuş)"
        gerekce = (
            f"Motor sağlığı (%{yeni_saglik:.1f}) ve menzil ({maksimum_menzil_km:.0f} km) "
            f"görev hedefi ({hedef_menzil_km:.0f} km) için güvenli sınırlar içinde."
        )

    return {
        "ucak_id": ucak_id,
        "senaryo_adi": senaryo_adi,
        "onceki_omur": onceki_durum["tahmini_kalan_omur"],
        "yeni_omur": yeni_omur,
        "onceki_saglik": onceki_durum["saglik_skoru"],
        "yeni_saglik": yeni_saglik,
        "baz_tuketim_kg_km": round(baz_tuketim, 2),
        "yeni_tuketim_kg_km": round(yeni_tuketim, 2),
        "yakit_artisi_yuzde": round(yakit_artisi_yuzde, 1),
        "mevcut_yakit_kg": round(mevcut_yakit_kg, 1),
        "gerekli_yakit_hedef_kg": round(gerekli_yakit_hedef_kg, 1),
        "gerekli_yakit_us_kg": round(gerekli_yakit_us_kg, 1),
        "maksimum_menzil_km": round(maksimum_menzil_km, 1),
        "hedefe_yakit_yeterli_mi": hedefe_yeterli_mi,
        "usse_yakit_yeterli_mi": usse_yeterli_mi,
        "risk_seviyesi": risk_seviyesi,
        "pilot_karari": pilot_karari,
        "karar_gerekcesi": gerekce,
        "teshis_raporu": teshis,
        "degisen_sensorler": degisen_sensorler,
    }


if __name__ == "__main__":
    from core.veri_yukleyici import filo_son_durumu_getir
    from core.omur_tahmini import modeli_egit_veya_yukle

    paket, _, test_df = modeli_egit_veya_yukle("FD003")
    filo = filo_son_durumu_getir(test_df)
    satir_37 = filo[filo["ucak_id"] == 37].iloc[0]
    sensorler_37 = {s: float(satir_37[s]) for s in paket["sensorler"]}

    for senaryo in ["normal", "sikistirici_asinmasi", "on_pervane_fan_arizasi"]:
        sonuc = pilot_simulasyonu_calistir(
            ucak_id=37,
            mevcut_sensorler=sensorler_37,
            model_paketi=paket,
            senaryo_adi=senaryo,
            ucus_sayisi=int(satir_37["ucus_sayisi"]),
        )
        print(f"\n[FD003 - {senaryo.upper()}] -> Sağlık: %{sonuc['onceki_saglik']} -> %{sonuc['yeni_saglik']} | Yakıt: +{sonuc['yakit_artisi_yuzde']}%")
        print(f"  Karar: {sonuc['pilot_karari']} | Gerekçe: {sonuc['karar_gerekcesi']}")

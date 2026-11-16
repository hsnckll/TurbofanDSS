# Bu modül Bakım Mühendisi için uçağın sensörlerindeki sapmaları (Z-skoru ile) bulur,
# arızanın hangi alt sistemden kaynaklandığını (kök neden) teşhis eder ve bakım önerisi sunar.

import os
import sys
import numpy as np

PROJE_KLASORU = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJE_KLASORU not in sys.path:
    sys.path.insert(0, PROJE_KLASORU)

from core.data_loader import GORUNUR_ISIMLER


def sensor_sapmalarini_bul(sensor_sozlugu, saglikli_istatistik):
    """
    Her sensörün şu anki değerini sağlıklı ortalamayla kıyaslayıp Z-Skoru hesaplar:
    Z = (Şu Anki Değer - Sağlıklı Ortalama) / Sağlıklı Standart Sapma
    """
    sapma_listesi = []

    for sensor_adi, deger in sensor_sozlugu.items():
        if sensor_adi not in saglikli_istatistik:
            continue

        normal_ort = saglikli_istatistik[sensor_adi]["mean"]
        normal_std = saglikli_istatistik[sensor_adi]["std"]
        z_skoru = (float(deger) - normal_ort) / normal_std

        # Sapma büyüklüğüne göre durum etiketi ver
        mutlak_z = abs(z_skoru)
        if mutlak_z >= 3.0:
            durum = "KRITIK"
        elif mutlak_z >= 1.75:
            durum = "UYARI"
        else:
            durum = "NORMAL"

        sapma_listesi.append({
            "sensor": sensor_adi,
            "gorunur_ad": GORUNUR_ISIMLER.get(sensor_adi, sensor_adi),
            "deger": round(float(deger), 2),
            "normal_ortalama": round(normal_ort, 2),
            "z_skoru": round(z_skoru, 2),
            "durum": durum,
        })

    # En çok sapan sensör en üstte olacak şekilde sırala
    sapma_listesi.sort(key=lambda x: abs(x["z_skoru"]), reverse=True)
    return sapma_listesi


def ariza_teshisi_yap(motor_id, sensor_sozlugu, tahmini_rul, saglik_skoru, saglikli_istatistik, ucus_dongusu=0):
    """
    Sensör sapmalarına (Z-skorlarına) bakarak if-else uzman kurallarıyla
    arızanın kök nedenini ve yapılması gereken bakım işlemlerini belirler.
    """
    sapmalar = sensor_sapmalarini_bul(sensor_sozlugu, saglikli_istatistik)
    z_haritasi = {item["sensor"]: item["z_skoru"] for item in sapmalar}

    kok_nedenler = []
    onerilen_aksiyonlar = []

    # Kural 1: HPC (Yüksek Basınç Kompresörü) Aşınması Kontrolü
    hpc_basinc_z = z_haritasi.get("hpc_cikis_statik_basinci_psia", 0.0)
    lpt_sicaklik_z = z_haritasi.get("lpt_cikis_sicakligi_r", 0.0)
    hpc_toplam_basinc_z = z_haritasi.get("hpc_cikis_toplam_basinci_psia", 0.0)

    if hpc_basinc_z >= 2.0 or (lpt_sicaklik_z >= 2.0 and hpc_toplam_basinc_z <= -1.5):
        guven = min(98.0, 50.0 + abs(hpc_basinc_z) * 12.0)
        kok_nedenler.append({
            "alt_sistem": "HPC (Yüksek Basınç Kompresörü)",
            "olasi_ariza": "Kompresör kanatçık aşınması / basınç ve verim kaybı",
            "guven_yuzdesi": round(guven, 1),
            "oncelik": "YUKSEK",
        })
        onerilen_aksiyonlar.append("HPC kademelerine boroskop muayenesi yapın ve kanatçık boşluklarını ölçün.")

    # Kural 2: Yakıt Besleme ve Yanma Odası Kontrolü
    yakit_z = z_haritasi.get("yakit_akis_basinc_orani", 0.0)
    if yakit_z <= -2.0:
        guven = min(95.0, 50.0 + abs(yakit_z) * 10.0)
        kok_nedenler.append({
            "alt_sistem": "Yakıt Besleme & Yanma Odası",
            "olasi_ariza": "Yakıt enjektörü püskürtme bozukluğu veya yanma verim kaybı",
            "guven_yuzdesi": round(guven, 1),
            "oncelik": "YUKSEK" if abs(yakit_z) >= 3.0 else "ORTA",
        })
        onerilen_aksiyonlar.append("Yakıt enjektör nozullarını ve yakıt kontrol ünitesi (FCU) filtrelerini kontrol edin.")

    # Kural 3: Türbin Soğutma Havası Kaybı Kontrolü
    hpt_sogutma_z = z_haritasi.get("hpt_sogutma_havasi_lbms", 0.0)
    lpt_sogutma_z = z_haritasi.get("lpt_sogutma_havasi_lbms", 0.0)
    if (hpt_sogutma_z <= -1.8 or lpt_sogutma_z <= -1.8) and lpt_sicaklik_z >= 1.5:
        kok_nedenler.append({
            "alt_sistem": "Türbin Soğutma Sistemi",
            "olasi_ariza": "HPT/LPT soğutma havası akışında düşüş ve aşırı ısınma",
            "guven_yuzdesi": round(min(90.0, 45.0 + abs(lpt_sicaklik_z) * 10.0), 1),
            "oncelik": "YUKSEK",
        })
        onerilen_aksiyonlar.append("Soğutma havası (bleed) borularında tıkanıklık ve kaçak testi yapın.")

    # Kural 4: Çekirdek Şaft / Rulman Sürtünmesi Kontrolü
    cekirdek_devir_z = z_haritasi.get("cekirdek_devri_rpm", 0.0)
    if abs(cekirdek_devir_z) >= 2.2:
        kok_nedenler.append({
            "alt_sistem": "Çekirdek Şaft & Rulman Grubu",
            "olasi_ariza": "Rulman aşınması veya döner aksam mekanik sürtünmesi",
            "guven_yuzdesi": round(min(88.0, 40.0 + abs(cekirdek_devir_z) * 12.0), 1),
            "oncelik": "ORTA",
        })
        onerilen_aksiyonlar.append("Motor yağında metal çapak analizi ve titreşim ölçümü yapın.")

    # Hiçbir arıza kuralı tetiklenmediyse motor sağlıklıdır
    if len(kok_nedenler) == 0:
        kok_nedenler.append({
            "alt_sistem": "Tüm Alt Sistemler",
            "olasi_ariza": "Belirgin bir arıza belirtisi yok; tüm değerler normal",
            "guven_yuzdesi": 99.0,
            "oncelik": "DUSUK",
        })
        onerilen_aksiyonlar.append("Rutin hat bakımı ve uçuş öncesi görsel kontrol yeterlidir.")

    # Bakım Öncelik Puanı (0 = Çok Rahat, 100 = Acil Hangar)
    kritik_sayisi = sum(1 for s in sapmalar if s["durum"] == "KRITIK")
    uyari_sayisi = sum(1 for s in sapmalar if s["durum"] == "UYARI")

    ham_puan = (100.0 - saglik_skoru) * 0.7 + (kritik_sayisi * 8.0) + (uyari_sayisi * 2.0)
    if tahmini_rul < 20.0:
        ham_puan = max(ham_puan, 85.0)

    bakim_puani = round(float(np.clip(ham_puan, 0.0, 100.0)), 1)

    if bakim_puani >= 70.0 or tahmini_rul < 25.0:
        aciliyet = "ACIL_HANGAR (Uçuş Durduruldu)"
    elif bakim_puani >= 35.0:
        aciliyet = "PLANLI_BAKIM (Yakın İzleme)"
    else:
        aciliyet = "RUTIN (Uçuşa Elverişli)"

    return {
        "motor_id": motor_id,
        "ucus_dongusu": ucus_dongusu,
        "tahmini_rul": tahmini_rul,
        "saglik_skoru": saglik_skoru,
        "bakim_oncelik_puani": bakim_puani,
        "bakim_aciliyeti": aciliyet,
        "anomaliler": sapmalar,
        "kok_nedenler": kok_nedenler,
        "onerilen_aksiyonlar": onerilen_aksiyonlar,
    }


if __name__ == "__main__":
    from core.data_loader import filo_son_durumu_getir
    from core.rul_predictor import filoyu_skorla, modeli_egit_veya_yukle

    paket, _, test_df = modeli_egit_veya_yukle("FD001")
    filo = filoyu_skorla(filo_son_durumu_getir(test_df), paket)

    for m_id in [1, 3, 34]:
        satir = filo[filo["motor_id"] == m_id].iloc[0]
        sensorler = {s: float(satir[s]) for s in paket["sensorler"]}
        rapor = ariza_teshisi_yap(
            motor_id=m_id,
            sensor_sozlugu=sensorler,
            tahmini_rul=float(satir["tahmini_rul"]),
            saglik_skoru=float(satir["saglik_skoru"]),
            saglikli_istatistik=paket["saglikli_istatistik"],
            ucus_dongusu=int(satir["ucus_dongusu"]),
        )
        print(f"\n--- Uçak #{m_id} | Bakım Puanı: {rapor['bakim_oncelik_puani']} | {rapor['bakim_aciliyeti']} ---")
        for kn in rapor["kok_nedenler"]:
            print(f"  Kök Neden: {kn['alt_sistem']} -> {kn['olasi_ariza']} (%{kn['guven_yuzdesi']})")

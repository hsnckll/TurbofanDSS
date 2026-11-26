# 2. ROL: BAKIM MÜHENDİSİ ARIZA TEŞHİS MOTORU (FD003 ÇİFT ARIZA DESTEKLİ)
# Bu dosya Bakım Mühendisinin "Bu uçağın hangi sensörü normalden sapmış,
# sorun Ana Hava Sıkıştırıcıdan mı yoksa Ön Pervaneden (Fan) mi kaynaklanıyor?" sorularını cevaplar.

import os
import sys
import numpy as np

PROJE_KLASORU = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJE_KLASORU not in sys.path:
    sys.path.insert(0, PROJE_KLASORU)

from core.veri_yukleyici import GORUNUR_ISIMLER


def sensor_sapmalarini_bul(sensor_sozlugu, saglikli_istatistik):
    """
    Her sensörün şu anki değerini sağlıklı motor ortalamasıyla kıyaslayıp Z-Skoru hesaplar:
    Z = (Şu Anki Değer - Sağlıklı Ortalama) / Sağlıklı Standart Sapma
    """
    sapma_listesi = []

    for sensor_adi, deger in sensor_sozlugu.items():
        if sensor_adi not in saglikli_istatistik:
            continue

        normal_ort = saglikli_istatistik[sensor_adi]["mean"]
        normal_std = saglikli_istatistik[sensor_adi]["std"]
        z_skoru = (float(deger) - normal_ort) / normal_std

        mutlak_z = abs(z_skoru)
        if mutlak_z >= 3.0:
            durum = "KRİTİK"
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

    sapma_listesi.sort(key=lambda x: abs(x["z_skoru"]), reverse=True)
    return sapma_listesi


def ariza_teshisi_yap(ucak_id, sensor_sozlugu, tahmini_kalan_omur, saglik_skoru, saglikli_istatistik, ucus_sayisi=0):
    """
    Sensör sapmalarına bakarak NASA FD003'ün gerçek 2 ana kök arıza modunu
    (1: Ana Hava Sıkıştırıcı / HPC, 2: Ön Pervane / Fan) teşhis eder.
    Yakıt verimi, soğutma havası ve şaft devri gibi sensörlerdeki sapmalar ayrı
    bir arıza modu gibi değil, ana arızanın 'ikincil yan belirtileri' olarak raporlanır.
    """
    sapmalar = sensor_sapmalarini_bul(sensor_sozlugu, saglikli_istatistik)
    z_haritasi = {item["sensor"]: item["z_skoru"] for item in sapmalar}

    kok_nedenler = []
    yan_belirtiler = []
    onerilen_aksiyonlar = []

    ana_basinc_z = z_haritasi.get("ana_sikistirici_basinci", 0.0)
    egzoz_sicaklik_z = z_haritasi.get("egzoz_turbin_sicakligi", 0.0)
    toplam_basinc_z = z_haritasi.get("ana_sikistirici_toplam_basinci", 0.0)
    dis_ic_hava_z = z_haritasi.get("dis_ic_hava_orani", 0.0)
    motor_basinc_orani_z = z_haritasi.get("motor_basinc_orani", 0.0)
    pervane_hiz_z = z_haritasi.get("on_pervane_hizi_rpm", 0.0)
    yakit_z = z_haritasi.get("yakit_verim_orani", 0.0)
    on_sogutma_z = z_haritasi.get("on_turbin_sogutma_havasi", 0.0)
    arka_sogutma_z = z_haritasi.get("arka_turbin_sogutma_havasi", 0.0)
    saft_hiz_z = z_haritasi.get("motor_saft_hizi_rpm", 0.0)

    # Kural 1 (FD003 Ana Arıza Modu 2): Ön Pervane (Fan) Aşınması
    # FD003'te Fan arızasında motor basınç oranı (s10), yakıt oranı (s12) ve pervane hızı (s8) yükselir;
    # dış/iç hava oranı (s15) ise aşağı düşer.
    fan_baskin_imza = (
        (motor_basinc_orani_z >= 2.0 and (dis_ic_hava_z <= -1.2 or yakit_z >= 2.5 or toplam_basinc_z >= 2.5))
        or (dis_ic_hava_z <= -2.0 and (pervane_hiz_z >= 1.5 or yakit_z >= 2.0))
        or pervane_hiz_z >= 2.2
    )

    if fan_baskin_imza:
        fan_ileri = (
            dis_ic_hava_z <= -2.0
            or pervane_hiz_z >= 2.2
            or ana_basinc_z >= 2.2
            or saglik_skoru < 55.0
        )
        guven_olcu = max(abs(dis_ic_hava_z), abs(pervane_hiz_z), abs(yakit_z) * 0.5)
        guven = min(98.0, 58.0 + guven_olcu * 9.5)
        kok_nedenler.append({
            "alt_sistem": "Ön Pervane (Fan) ve Giriş Bölümü",
            "olasi_ariza": (
                "Ön pervane kanatçık aşınması, yabancı cisim darbesi veya dış/iç hava akışı dengesizliği"
                if fan_ileri
                else "Ön pervane (fan) bölgesinde erken aşınma belirtisi — basınç ve hava akışı dengesi kaymaya başlamış"
            ),
            "guven_yuzdesi": round(guven, 1),
            "oncelik": "YÜKSEK" if fan_ileri else "ORTA",
        })
        onerilen_aksiyonlar.append(
            "Ön pervane (fan) kanatçıklarını darbe/aşınma yönünden görsel inceleyin ve balans ayarı yapın."
        )

    # Kural 2 (FD003 Ana Arıza Modu 1): Ana Hava Sıkıştırıcı (HPC / Kompresör) Aşınması
    # HPC arızasında sıkıştırıcı basıncı (s11), egzoz ısısı (s4) ve dış/iç hava oranı (s15) yükselirken
    # toplam basınç (s7) ve yakıt verim oranı (s12) aşağı düşer.
    hpc_imza = (
        (not fan_baskin_imza and (
            ana_basinc_z >= 2.0
            or (egzoz_sicaklik_z >= 2.0 and (toplam_basinc_z <= -1.0 or dis_ic_hava_z >= 1.5))
        ))
        or (ana_basinc_z >= 2.5 and (dis_ic_hava_z >= 1.5 or toplam_basinc_z <= -1.2 or yakit_z <= -2.0))
    )

    if hpc_imza:
        guven = min(98.0, 52.0 + max(abs(ana_basinc_z), abs(dis_ic_hava_z)) * 10.5)
        kok_nedenler.append({
            "alt_sistem": "Ana Hava Sıkıştırıcı (Kompresör / HPC)",
            "olasi_ariza": "Ana hava sıkıştırıcı kanatçıklarda aşınma, iç kanal daralması ve basınç/sıcaklık artışı",
            "guven_yuzdesi": round(guven, 1),
            "oncelik": "YÜKSEK",
        })
        onerilen_aksiyonlar.append(
            "Ana hava sıkıştırıcı (HPC) kanatçıklarını kamera ile kontrol edin ve aşınma boşluklarını ölçün."
        )

    # İKİNCİL YAN BELİRTİLER (Ayrı bir kök arıza değil, ana arızanın diğer sensörlere yansımasıdır)
    if yakit_z <= -1.8 or yakit_z >= 2.5:
        yon = "yukarı fırladı" if yakit_z > 0 else "aşağı düştü"
        yan_belirtiler.append({
            "bolge": "Yakıt Verim Dengesi (s12)",
            "aciklama": f"Yakıt/basınç oranı normalden {yon} ({yakit_z:+.2f}σ) — ana arızanın yakıt verimine ikincil yansıması.",
        })

    if (on_sogutma_z <= -1.8 or arka_sogutma_z <= -1.8) or egzoz_sicaklik_z >= 2.2:
        yan_belirtiler.append({
            "bolge": "Türbin Isı & Soğutma Dengesi (s4, s20, s21)",
            "aciklama": "Motorun zorlanmasına bağlı olarak egzoz sıcaklığı yükseldi veya soğutma havası dengesi değişti.",
        })

    if abs(saft_hiz_z) >= 2.2:
        yan_belirtiler.append({
            "bolge": "Ana Şaft Devir Yükü (s9)",
            "aciklama": f"Motor şaft hızı normalden saptı ({saft_hiz_z:+.2f}σ) — döner parçalarda ikincil yük artışı.",
        })

    # Hiçbir kök arıza tetiklenmediyse motor sağlıklıdır
    if len(kok_nedenler) == 0:
        kok_nedenler.append({
            "alt_sistem": "Tüm Motor Sistemleri",
            "olasi_ariza": "Belirgin bir HPC veya Ön Pervane arızası yok; ana sensörler normal aralıkta",
            "guven_yuzdesi": 99.0,
            "oncelik": "DÜŞÜK",
        })
        onerilen_aksiyonlar.append("Rutin uçuş öncesi genel görsel kontrol yeterlidir.")

    # Bakım Öncelik Puanı (0 = Rahat, 100 = Acil Bakım)
    kritik_sayisi = sum(1 for s in sapmalar if s["durum"] == "KRİTİK")
    uyari_sayisi = sum(1 for s in sapmalar if s["durum"] == "UYARI")

    ham_puan = (100.0 - saglik_skoru) * 0.7 + (kritik_sayisi * 7.0) + (uyari_sayisi * 2.0)
    if tahmini_kalan_omur < 20.0:
        ham_puan = max(ham_puan, 85.0)

    bakim_puani = round(float(np.clip(ham_puan, 0.0, 100.0)), 1)

    if bakim_puani >= 70.0 or tahmini_kalan_omur < 25.0:
        aciliyet = "ACİL BAKIM (Uçuş Durduruldu)"
    elif bakim_puani >= 35.0:
        aciliyet = "PLANLI BAKIM (Yakın İzleme)"
    else:
        aciliyet = "RUTİN KONTROL (Uçabilir)"

    return {
        "ucak_id": ucak_id,
        "ucus_sayisi": ucus_sayisi,
        "tahmini_kalan_omur": tahmini_kalan_omur,
        "saglik_skoru": saglik_skoru,
        "bakim_oncelik_puani": bakim_puani,
        "bakim_aciliyeti": aciliyet,
        "anomaliler": sapmalar,
        "kok_nedenler": kok_nedenler,
        "yan_belirtiler": yan_belirtiler,
        "onerilen_aksiyonlar": onerilen_aksiyonlar,
    }


if __name__ == "__main__":
    from core.veri_yukleyici import filo_son_durumu_getir
    from core.omur_tahmini import filoyu_skorla, modeli_egit_veya_yukle

    paket, _, test_df = modeli_egit_veya_yukle("FD003")
    filo = filoyu_skorla(filo_son_durumu_getir(test_df), paket)

    # FD003'teki 4 örnek uçak: #37 Sağlam, #7 Orta, #50 Kompresör Arızası, #82 Ön Pervane (Fan) Arızası
    for u_id in [37, 7, 50, 82]:
        satir = filo[filo["ucak_id"] == u_id].iloc[0]
        sensorler = {s: float(satir[s]) for s in paket["sensorler"]}
        rapor = ariza_teshisi_yap(
            ucak_id=u_id,
            sensor_sozlugu=sensorler,
            tahmini_kalan_omur=float(satir["tahmini_kalan_omur"]),
            saglik_skoru=float(satir["saglik_skoru"]),
            saglikli_istatistik=paket["saglikli_istatistik"],
            ucus_sayisi=int(satir["ucus_sayisi"]),
        )
        print(f"\n--- FD003 Uçak #{u_id} (Gerçek Ömür: {int(satir['kalan_ucus_omru'])}) | Bakım Puanı: {rapor['bakim_oncelik_puani']} | {rapor['bakim_aciliyeti']} ---")
        for kn in rapor["kok_nedenler"]:
            print(f"  Arıza Sebebi: {kn['alt_sistem']} -> {kn['olasi_ariza']} (%{kn['guven_yuzdesi']})")

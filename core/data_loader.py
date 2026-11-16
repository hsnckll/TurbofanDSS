# Bu modül NASA C-MAPSS ham metin dosyalarını okur, sütunlara Türkçe anlamlı isimler verir,
# her satır için Kalan Faydalı Ömür (RUL) değerini hesaplar ve CSV olarak kaydeder.

import os
import pandas as pd

# Proje ana klasörü ve veri klasörlerinin yolları
PROJE_KLASORU = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HAM_VERI_KLASORU = os.path.join(PROJE_KLASORU, "data", "Nasa Turbofan-Ham Data")
ISLENMIS_VERI_KLASORU = os.path.join(PROJE_KLASORU, "data", "processed")

# NASA veri setindeki 26 sütunun anlamlı Türkçe karşılıkları (sırasıyla)
SUTUN_ISIMLERI = [
    "motor_id",                          # 1. Sütun: Motor (Uçak) numarası
    "ucus_dongusu",                      # 2. Sütun: Kaçıncı uçuşta olduğu
    "irtifa_kft",                        # 3. Sütun: Operasyonel Ayar 1 (İrtifa)
    "mach_hizi",                         # 4. Sütun: Operasyonel Ayar 2 (Mach hızı)
    "gaz_kolu_acisi",                    # 5. Sütun: Operasyonel Ayar 3 (Gaz kolu açısı)
    "fan_giris_sicakligi_r",             # s1: Fan giriş sıcaklığı
    "lpc_cikis_sicakligi_r",             # s2: LPC çıkış sıcaklığı
    "hpc_cikis_sicakligi_r",             # s3: HPC çıkış sıcaklığı
    "lpt_cikis_sicakligi_r",             # s4: LPT çıkış sıcaklığı
    "fan_giris_basinci_psia",            # s5: Fan giriş basıncı
    "bypass_basinci_psia",               # s6: Bypass kanal basıncı
    "hpc_cikis_toplam_basinci_psia",     # s7: HPC çıkış toplam basıncı
    "fan_devri_rpm",                     # s8: Fiziksel fan devri
    "cekirdek_devri_rpm",                # s9: Fiziksel çekirdek devri
    "motor_basinc_orani",                # s10: Motor basınç oranı (EPR)
    "hpc_cikis_statik_basinci_psia",     # s11: HPC çıkış statik basıncı (En önemli sensör)
    "yakit_akis_basinc_orani",           # s12: Yakıt akışının basınca oranı (phi)
    "duzeltilmis_fan_devri_rpm",         # s13: Düzeltilmiş fan devri
    "duzeltilmis_cekirdek_devri_rpm",    # s14: Düzeltilmiş çekirdek devri
    "bypass_orani",                      # s15: Bypass oranı
    "yakit_hava_orani",                  # s16: Yanma odası yakıt-hava oranı
    "bleed_entalpisi",                   # s17: Bleed entalpisi
    "istenen_fan_devri_rpm",             # s18: İstenen fan devri
    "istenen_duzeltilmis_fan_devri_rpm", # s19: İstenen düzeltilmiş fan devri
    "hpt_sogutma_havasi_lbms",           # s20: HPT türbin soğutma havası akışı
    "lpt_sogutma_havasi_lbms",           # s21: LPT türbin soğutma havası akışı
]

# Arayüzde (PyQt5 tablolarında ve grafiklerde) görünecek düzgün Türkçe başlıklar
GORUNUR_ISIMLER = {
    "motor_id": "Motor (Uçak) ID",
    "ucus_dongusu": "Uçuş Döngüsü",
    "lpc_cikis_sicakligi_r": "LPC Çıkış Sıcaklığı (°R)",
    "hpc_cikis_sicakligi_r": "HPC Çıkış Sıcaklığı (°R)",
    "lpt_cikis_sicakligi_r": "LPT Çıkış Sıcaklığı (°R)",
    "bypass_basinci_psia": "Bypass Kanal Basıncı (psia)",
    "hpc_cikis_toplam_basinci_psia": "HPC Çıkış Toplam Basıncı (psia)",
    "fan_devri_rpm": "Fan Devri (RPM)",
    "cekirdek_devri_rpm": "Çekirdek Devri (RPM)",
    "hpc_cikis_statik_basinci_psia": "HPC Çıkış Statik Basıncı (psia)",
    "yakit_akis_basinc_orani": "Yakıt Akışı / Basınç Oranı",
    "duzeltilmis_fan_devri_rpm": "Düzeltilmiş Fan Devri (RPM)",
    "duzeltilmis_cekirdek_devri_rpm": "Düzeltilmiş Çekirdek Devri (RPM)",
    "bypass_orani": "Bypass Oranı (BPR)",
    "bleed_entalpisi": "Bleed Entalpisi",
    "hpt_sogutma_havasi_lbms": "HPT Soğutma Havası Akışı (lbm/s)",
    "lpt_sogutma_havasi_lbms": "LPT Soğutma Havası Akışı (lbm/s)",
    "kalan_omur_rul": "Kalan Faydalı Ömür (RUL)",
}


def veri_setini_yukle(senaryo="FD001"):
    """
    Eğitim (train) ve Test (test) dosyalarını okur ve her satıra kalan ömrü (RUL) ekler.
    """
    # 1. Adım: Ham metin dosyalarını boşluklardan ayırarak oku ve sütun isimlerini ver
    train_yolu = os.path.join(HAM_VERI_KLASORU, f"train_{senaryo}.txt")
    test_yolu = os.path.join(HAM_VERI_KLASORU, f"test_{senaryo}.txt")
    rul_yolu = os.path.join(HAM_VERI_KLASORU, f"RUL_{senaryo}.txt")

    train_df = pd.read_csv(train_yolu, sep=r"\s+", header=None, names=SUTUN_ISIMLERI)
    test_df = pd.read_csv(test_yolu, sep=r"\s+", header=None, names=SUTUN_ISIMLERI)
    rul_df = pd.read_csv(rul_yolu, sep=r"\s+", header=None, names=["son_rul"])

    # 2. Adım: Eğitim verisinde motor bozulana kadar çalıştığı için:
    # Kalan Ömür (RUL) = Motorun ulaştığı son döngü - Şu anki döngü
    son_donguler_train = train_df.groupby("motor_id")["ucus_dongusu"].transform("max")
    train_df["kalan_omur_rul"] = son_donguler_train - train_df["ucus_dongusu"]

    # 3. Adım: Test verisinde RUL_FD001.txt dosyasındaki gerçek kalan ömrü her motora ekle
    rul_df["motor_id"] = range(1, len(rul_df) + 1)
    son_donguler_test = test_df.groupby("motor_id")["ucus_dongusu"].transform("max")
    test_df = test_df.merge(rul_df, on="motor_id", how="left")
    test_df["kalan_omur_rul"] = test_df["son_rul"] + (son_donguler_test - test_df["ucus_dongusu"])
    test_df = test_df.drop(columns=["son_rul"])

    return train_df, test_df


def sabit_sutunlari_bul(df):
    """
    Hiç değişmeyen (standart sapması 0 olan) sabit sensör sütunlarını bulur.
    Çünkü sabit sayılar makine öğrenmesi modeline hiçbir bilgi vermez.
    """
    sabitler = []
    korunacaklar = ["motor_id", "ucus_dongusu", "kalan_omur_rul"]
    for sutun in df.columns:
        if sutun not in korunacaklar:
            if df[sutun].std() < 0.000001:
                sabitler.append(sutun)
    return sabitler


def filo_son_durumu_getir(test_df):
    """
    Her motorun (uçağın) en son uçuş döngüsündeki satırını alır.
    Böylece 100 uçağın şu anki güncel durumunu gösteren tek bir filo tablosu elde ederiz.
    """
    son_indeksler = test_df.groupby("motor_id")["ucus_dongusu"].idxmax()
    filo_df = test_df.loc[son_indeksler].sort_values("motor_id").reset_index(drop=True)
    return filo_df


def csv_olarak_kaydet(senaryo="FD001"):
    """Okunan ve isimlendirilen tabloları incelemek için CSV olarak kaydeder."""
    os.makedirs(ISLENMIS_VERI_KLASORU, exist_ok=True)
    train_df, test_df = veri_setini_yukle(senaryo)
    filo_df = filo_son_durumu_getir(test_df)

    train_df.to_csv(os.path.join(ISLENMIS_VERI_KLASORU, f"{senaryo}_train_anlamli.csv"), index=False, encoding="utf-8-sig")
    test_df.to_csv(os.path.join(ISLENMIS_VERI_KLASORU, f"{senaryo}_test_anlamli.csv"), index=False, encoding="utf-8-sig")
    filo_df.to_csv(os.path.join(ISLENMIS_VERI_KLASORU, f"{senaryo}_filo_ozet.csv"), index=False, encoding="utf-8-sig")
    return train_df, test_df, filo_df


if __name__ == "__main__":
    train_df, test_df, filo_df = csv_olarak_kaydet("FD001")
    print("Eğitim verisi boyutu:", train_df.shape)
    print("Test verisi boyutu:", test_df.shape)
    print("Filo özet tablosu boyutu (100 uçak):", filo_df.shape)
    print("Sabit olduğu için elenecek sütunlar:", sabit_sutunlari_bul(train_df))

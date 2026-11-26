# Bu dosya NASA'nın ham motor verilerini okur, karmaşık teknik sütunlara sade Türkçe isimler verir,
# her uçak için Kalan Uçuş Ömrünü hesaplar ve inceleyebilmen için CSV olarak kaydeder.

import os
import pandas as pd

# Proje ana klasörü ve veri klasörlerinin yolları
PROJE_KLASORU = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HAM_VERI_KLASORU = os.path.join(PROJE_KLASORU, "data", "Nasa Turbofan-Ham Data")
ISLENMIS_VERI_KLASORU = os.path.join(PROJE_KLASORU, "data", "processed")

# NASA dosyasındaki 26 sütunun sade ve anlaşılır Türkçe isimleri (sırasıyla)
SUTUN_ISIMLERI = [
    "ucak_id",                         # 1. Sütun: Uçak / Motor numarası
    "ucus_sayisi",                     # 2. Sütun: Bugüne kadar yaptığı uçuş sayısı
    "irtifa",                          # 3. Sütun: Uçuş yüksekliği (irtifa)
    "mach_hizi",                       # 4. Sütun: Uçuş hızı (Mach)
    "gaz_kolu_acisi",                  # 5. Sütun: Pilot gaz kolu açısı
    "hava_giris_sicakligi",            # s1: Motor hava giriş sıcaklığı
    "on_sikistirici_sicakligi",        # s2: Ön hava sıkıştırıcı çıkış sıcaklığı
    "ana_sikistirici_sicakligi",       # s3: Ana hava sıkıştırıcı çıkış sıcaklığı
    "egzoz_turbin_sicakligi",          # s4: Arka türbin (egzoz) çıkış sıcaklığı
    "hava_giris_basinci",              # s5: Motor hava giriş basıncı
    "dis_kanal_basinci",               # s6: Dış hava kanalı basıncı
    "ana_sikistirici_toplam_basinci",  # s7: Ana hava sıkıştırıcı toplam basıncı
    "on_pervane_hizi_rpm",             # s8: Ön pervane (fan) dönüş hızı (RPM)
    "motor_saft_hizi_rpm",             # s9: Motor ana şaft dönüş hızı (RPM)
    "motor_basinc_orani",              # s10: Motor genel basınç oranı
    "ana_sikistirici_basinci",         # s11: Ana hava sıkıştırıcı statik basıncı (En önemli sensör)
    "yakit_verim_orani",               # s12: Yakıt akışının basınca oranı (Yakıt verimi)
    "duzeltilmis_pervane_hizi_rpm",    # s13: Düzeltilmiş ön pervane hızı (RPM)
    "duzeltilmis_saft_hizi_rpm",       # s14: Düzeltilmiş motor ana şaft hızı (RPM)
    "dis_ic_hava_orani",               # s15: Dış / iç hava akışı oranı
    "yakit_hava_karisim_orani",        # s16: Yanma odası yakıt-hava karışım oranı
    "sogutma_havasi_isi_yuku",         # s17: Motor soğutma havası ısı yükü
    "istenen_pervane_hizi_rpm",        # s18: İstenen ön pervane hızı (RPM)
    "istenen_duzeltilmis_pervane_hizi",# s19: İstenen düzeltilmiş pervane hızı
    "on_turbin_sogutma_havasi",        # s20: Ön türbin soğutma havası akışı
    "arka_turbin_sogutma_havasi",      # s21: Arka türbin soğutma havası akışı
]

# Ekrandaki tablolarda ve grafiklerde görünecek temiz Türkçe başlıklar
GORUNUR_ISIMLER = {
    "ucak_id": "Uçak No",
    "ucus_sayisi": "Yapılan Uçuş Sayısı",
    "on_sikistirici_sicakligi": "Ön Sıkıştırıcı Sıcaklığı",
    "ana_sikistirici_sicakligi": "Ana Sıkıştırıcı Sıcaklığı",
    "egzoz_turbin_sicakligi": "Arka Türbin (Egzoz) Sıcaklığı",
    "dis_kanal_basinci": "Dış Hava Kanalı Basıncı",
    "ana_sikistirici_toplam_basinci": "Ana Hava Sıkıştırıcı Toplam Basıncı",
    "on_pervane_hizi_rpm": "Ön Pervane (Fan) Hızı (RPM)",
    "motor_saft_hizi_rpm": "Motor Ana Şaft Hızı (RPM)",
    "motor_basinc_orani": "Motor Basınç Oranı",
    "ana_sikistirici_basinci": "Ana Hava Sıkıştırıcı Basıncı",
    "yakit_verim_orani": "Yakıt Akış / Basınç Oranı",
    "duzeltilmis_pervane_hizi_rpm": "Düzeltilmiş Ön Pervane Hızı (RPM)",
    "duzeltilmis_saft_hizi_rpm": "Düzeltilmiş Ana Şaft Hızı (RPM)",
    "dis_ic_hava_orani": "Dış / İç Hava Akışı Oranı",
    "sogutma_havasi_isi_yuku": "Soğutma Havası Isı Yükü",
    "on_turbin_sogutma_havasi": "Ön Türbin Soğutma Havası",
    "arka_turbin_sogutma_havasi": "Arka Türbin Soğutma Havası",
    "kalan_ucus_omru": "Kalan Uçuş Ömrü",
}


def veri_setini_yukle(senaryo="FD003"):
    """
    Eğitim (train) ve Test (test) dosyalarını okur ve her satıra Kalan Uçuş Ömrünü ekler.
    Varsayılan olarak 2 arıza modu içeren (Ana Sıkıştırıcı + Ön Pervane) FD003 kullanılır.
    """
    train_yolu = os.path.join(HAM_VERI_KLASORU, f"train_{senaryo}.txt")
    test_yolu = os.path.join(HAM_VERI_KLASORU, f"test_{senaryo}.txt")
    rul_yolu = os.path.join(HAM_VERI_KLASORU, f"RUL_{senaryo}.txt")

    # 1. Adım: Ham metin dosyalarını oku ve Türkçe sütun isimlerini ver
    train_df = pd.read_csv(train_yolu, sep=r"\s+", header=None, names=SUTUN_ISIMLERI)
    test_df = pd.read_csv(test_yolu, sep=r"\s+", header=None, names=SUTUN_ISIMLERI)
    rul_df = pd.read_csv(rul_yolu, sep=r"\s+", header=None, names=["son_kalan_omur"])

    # 2. Adım: Eğitim verisinde motor bozulana kadar uçtuğu için:
    # Kalan Uçuş Ömrü = Motorun ulaştığı en son uçuş sayısı - Şu anki uçuş sayısı
    son_ucus_train = train_df.groupby("ucak_id")["ucus_sayisi"].transform("max")
    train_df["kalan_ucus_omru"] = son_ucus_train - train_df["ucus_sayisi"]

    # 3. Adım: Test verisine NASA'nın gerçek kalan ömür (RUL) cevap anahtarını ekle
    rul_df["ucak_id"] = range(1, len(rul_df) + 1)
    son_ucus_test = test_df.groupby("ucak_id")["ucus_sayisi"].transform("max")
    test_df = test_df.merge(rul_df, on="ucak_id", how="left")
    test_df["kalan_ucus_omru"] = test_df["son_kalan_omur"] + (son_ucus_test - test_df["ucus_sayisi"])
    test_df = test_df.drop(columns=["son_kalan_omur"])

    return train_df, test_df


def sabit_sutunlari_bul(df):
    """
    Baştan sona hiç değişmeyen (standart sapması 0 olan) sensörleri bulur.
    Hiç değişmeyen sayılar yapay zekaya bilgi vermediği için elenir.
    """
    sabitler = []
    korunacaklar = ["ucak_id", "ucus_sayisi", "kalan_ucus_omru"]
    for sutun in df.columns:
        if sutun not in korunacaklar:
            if df[sutun].std() < 0.000001:
                sabitler.append(sutun)
    return sabitler


def filo_son_durumu_getir(test_df):
    """
    Her uçağın en son uçuş kaydını alır.
    Böylece 100 uçağın şu anki güncel durumunu gösteren tek bir filo tablosu çıkarırız.
    """
    son_indeksler = test_df.groupby("ucak_id")["ucus_sayisi"].idxmax()
    filo_df = test_df.loc[son_indeksler].sort_values("ucak_id").reset_index(drop=True)
    return filo_df


def csv_olarak_kaydet(senaryo="FD003"):
    """Okunan ve sade Türkçe etiketlenen tabloları CSV olarak kaydeder."""
    os.makedirs(ISLENMIS_VERI_KLASORU, exist_ok=True)
    train_df, test_df = veri_setini_yukle(senaryo)
    filo_df = filo_son_durumu_getir(test_df)

    train_df.to_csv(os.path.join(ISLENMIS_VERI_KLASORU, f"{senaryo}_train_anlamli.csv"), index=False, encoding="utf-8-sig")
    test_df.to_csv(os.path.join(ISLENMIS_VERI_KLASORU, f"{senaryo}_test_anlamli.csv"), index=False, encoding="utf-8-sig")
    filo_df.to_csv(os.path.join(ISLENMIS_VERI_KLASORU, f"{senaryo}_filo_ozet.csv"), index=False, encoding="utf-8-sig")
    return train_df, test_df, filo_df


if __name__ == "__main__":
    train_df, test_df, filo_df = csv_olarak_kaydet("FD003")
    print("Senaryo: FD003 (Çift Arıza: Ana Sıkıştırıcı + Ön Pervane)")
    print("Eğitim verisi boyutu:", train_df.shape)
    print("Test verisi boyutu:", test_df.shape)
    print("Filo özet tablosu boyutu (100 uçak):", filo_df.shape)
    print("Sabit olduğu için elenecek sütunlar:", sabit_sutunlari_bul(train_df))

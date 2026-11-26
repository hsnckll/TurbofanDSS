# 1. ROL: FİLO KOMUTANI KARAR MOTORU (AHP + TOPSIS)
# Bu dosya Filo Komutanının "100 uçaktan hangisini bu göreve göndereyim?" sorusunu çözer:
# 1) AHP ile 4 kriterin ağırlıklarını ve Tutarlılık Oranını (CR < 0.10) hesaplar.
# 2) TOPSIS ile uçakları 1'den 100'e kadar sıralar ve GÖREVE UYGUN / KISITLI GÖREV / UÇAMAZ kararı verir.

import os
import sys
import numpy as np
import pandas as pd

PROJE_KLASORU = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJE_KLASORU not in sys.path:
    sys.path.insert(0, PROJE_KLASORU)

# Saaty'nin standart Rastgele İndeks (RI) tablosu (Matris boyutuna göre)
RI_TABLOSU = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24}

# Uçak seçiminde kullandığımız 4 ana kriter
KRITERLER = [
    "saglik_skoru",            # Kriter 1: Motor Sağlığı (Fayda - Yüksek olması iyi)
    "yakit_verimlilik_skoru",  # Kriter 2: Yakıt Verimliliği (Fayda - Yüksek olması iyi)
    "termal_denge_skoru",      # Kriter 3: Sıcaklık/Basınç Dengesi (Fayda - Yüksek olması iyi)
    "ucus_sayisi",             # Kriter 4: Yapılan Uçuş Sayısı / Yaş (Maliyet - Düşük olması iyi)
]

# Hangi kriterin Fayda (True), hangisinin Maliyet (False) olduğunu belirtir
FAYDA_MI = {
    "saglik_skoru": True,
    "yakit_verimlilik_skoru": True,
    "termal_denge_skoru": True,
    "ucus_sayisi": False,  # Maliyet kriteri (daha az uçmuş genç uçak tercih edilir)
}


def gorev_matrisi_getir(gorev_tipi="dengeli"):
    """
    Seçilen görev tipine göre 4x4 AHP ikili karşılaştırma matrisini döndürür.
    Sıra: [Motor Sağlığı, Yakıt Verimliliği, Termal Denge, Yapılan Uçuş Sayısı]
    """
    if gorev_tipi == "kritik_gorev":
        # Yüksek riskli / muharebe görevi: Motor sağlığı ve termal dayanıklılık öncelikli
        return np.array([
            [1.0,   3.0, 2.0,   5.0],
            [1/3,   1.0, 1/2,   2.0],
            [1/2,   2.0, 1.0,   3.0],
            [1/5,   1/2, 1/3,   1.0],
        ])
    elif gorev_tipi == "uzun_menzil":
        # Uzun menzil görevi: Yakıt verimliliği en yüksek önceliğe sahip
        return np.array([
            [1.0, 1/3, 2.0, 3.0],
            [3.0, 1.0, 4.0, 5.0],
            [1/2, 1/4, 1.0, 2.0],
            [1/3, 1/5, 1/2, 1.0],
        ])
    else:
        # Standart / Dengeli görev
        return np.array([
            [1.0, 2.0, 2.0, 3.0],
            [1/2, 1.0, 1.0, 2.0],
            [1/2, 1.0, 1.0, 2.0],
            [1/3, 1/2, 1/2, 1.0],
        ])


def ahp_agirlik_hesapla(ikili_matris):
    """
    AHP Adımları:
    1) Her sütunu kendi toplamına böl (Normalizasyon).
    2) Her satırın ortalamasını alarak Kriter Ağırlıklarını bul.
    3) Tutarlılık Oranını (CR = CI / RI) hesapla ve CR < 0.10 şartını kontrol et.
    """
    n = len(ikili_matris)

    # 1. Adım: Sütun toplamlarına böl
    sutun_toplamlari = np.sum(ikili_matris, axis=0)
    normalize_matris = ikili_matris / sutun_toplamlari

    # 2. Adım: Satır ortalamalarını al (Ağırlık vektörü)
    agirliklar = np.mean(normalize_matris, axis=1)

    # 3. Adım: En büyük özdeğer (lambda_max) ve Tutarlılık İndeksi (CI)
    agirlikli_toplam = np.dot(ikili_matris, agirliklar)
    lambda_max = float(np.mean(agirlikli_toplam / agirliklar))
    ci = (lambda_max - n) / (n - 1)

    # 4. Adım: Tutarlılık Oranı (CR = CI / RI)
    ri = RI_TABLOSU.get(n, 0.90)
    cr = round(max(0.0, ci / ri), 4)
    tutarli_mi = cr < 0.10

    return np.round(agirliklar, 4), cr, tutarli_mi


def topsis_sirala(filo_df, agirliklar):
    """
    TOPSIS Adımları:
    1) Karar matrisini vektör normalizasyonu ile normalize et.
    2) AHP ağırlıkları ile çarp.
    3) En iyi (İdeal) ve En kötü (Ters İdeal) uçak değerlerini bul.
    4) Her uçağın en iyiye ve en kötüye uzaklığını hesapla.
    5) TOPSIS skoruna göre 1'den 100'e sırala ve görev kararı ata.
    """
    matris = filo_df[KRITERLER].to_numpy(dtype=float)

    # 1. Adım: Vektör Normalizasyonu
    payda = np.sqrt(np.sum(matris ** 2, axis=0))
    payda[payda == 0] = 0.000001
    normalize_matris = matris / payda

    # 2. Adım: Ağırlıklı Normalize Matris
    agirlikli_matris = normalize_matris * agirliklar

    # 3. Adım: En İyi (A+) ve En Kötü (A-) Noktalar
    ideal_iyi = np.zeros(len(KRITERLER))
    ideal_kotu = np.zeros(len(KRITERLER))

    for j, kriter_adi in enumerate(KRITERLER):
        sutun = agirlikli_matris[:, j]
        if FAYDA_MI[kriter_adi]:
            ideal_iyi[j] = np.max(sutun)
            ideal_kotu[j] = np.min(sutun)
        else:
            ideal_iyi[j] = np.min(sutun)
            ideal_kotu[j] = np.max(sutun)

    # 4. Adım: Öklid Uzaklıkları
    uzaklik_iyi = np.sqrt(np.sum((agirlikli_matris - ideal_iyi) ** 2, axis=1))
    uzaklik_kotu = np.sqrt(np.sum((agirlikli_matris - ideal_kotu) ** 2, axis=1))

    # 5. Adım: TOPSIS Yakınlık Skoru (0 ile 1 arasında)
    topsis_skoru = uzaklik_kotu / (uzaklik_iyi + uzaklik_kotu)

    sonuc_df = filo_df.copy()
    sonuc_df["topsis_skor"] = np.round(topsis_skoru, 4)

    sonuc_df = sonuc_df.sort_values("topsis_skor", ascending=False).reset_index(drop=True)
    sonuc_df["siralama"] = range(1, len(sonuc_df) + 1)

    # 6. Adım: Sade Türkçe Karar Etiketleri
    kararlar = []
    for _, satir in sonuc_df.iterrows():
        kalan_omur = float(satir["tahmini_kalan_omur"])
        saglik = float(satir["saglik_skoru"])
        skor = float(satir["topsis_skor"])

        # Güvenlik Kuralı: Kalan ömrü 25 uçuştan az veya sağlığı %30'dan düşükse kesinlikle uçamaz
        if kalan_omur < 25.0 or saglik < 30.0 or skor < 0.30:
            kararlar.append("UÇAMAZ (BAKIMA AL)")
        elif skor < 0.55 or saglik < 60.0:
            kararlar.append("KISITLI GÖREV (Kısa Menzil)")
        else:
            kararlar.append("GÖREVE UYGUN")

    sonuc_df["karar"] = kararlar
    return sonuc_df


def filo_karar_analizi_yap(skorlu_filo_df, gorev_tipi="dengeli", ozel_matris=None):
    """AHP ve TOPSIS'i birleştirip Filo Komutanı ekranı için hazır sonuçları döndürür."""
    matris = ozel_matris if ozel_matris is not None else gorev_matrisi_getir(gorev_tipi)
    agirliklar, cr, tutarli_mi = ahp_agirlik_hesapla(matris)
    agirlik_sozlugu = {k: float(w) for k, w in zip(KRITERLER, agirliklar)}
    sirali_filo = topsis_sirala(skorlu_filo_df, agirliklar)
    return sirali_filo, agirlik_sozlugu, cr, tutarli_mi


if __name__ == "__main__":
    from core.veri_yukleyici import filo_son_durumu_getir
    from core.omur_tahmini import filoyu_skorla, modeli_egit_veya_yukle

    paket, _, test_df = modeli_egit_veya_yukle("FD003")
    skorlu_filo = filoyu_skorla(filo_son_durumu_getir(test_df), paket)

    sirali_tablo, agirliklar, cr, tutarli = filo_karar_analizi_yap(skorlu_filo, "dengeli")
    print("AHP Ağırlıkları:", agirliklar)
    print(f"Tutarlılık Oranı (CR): {cr} -> Tutarlı mı: {tutarli}")
    sutunlar = ["siralama", "ucak_id", "topsis_skor", "karar", "tahmini_kalan_omur", "saglik_skoru"]
    print("\nEn Uygun İlk 3 Uçak:")
    print(sirali_tablo[sutunlar].head(3).to_string(index=False))
    print("\nEn Riskli Son 3 Uçak:")
    print(sirali_tablo[sutunlar].tail(3).to_string(index=False))

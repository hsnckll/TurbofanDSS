# TurbofanDSS — Hafif Python Web Sunucusu ve JSON API
# Standart Python kütüphanesi (ThreadingHTTPServer) ile ek sunucu paketi gerektirmeden çalışır.

import json
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROJE_KLASORU = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJE_KLASORU not in sys.path:
    sys.path.insert(0, PROJE_KLASORU)

from core.bakim_muhendisi_teshis import ariza_teshisi_yap
from core.model_egitimi import ALGORITMA_BILGILERI, dort_modeli_egit_ve_karsilastir, tek_model_egit
from core.omur_tahmini import (
    en_onemli_sensorleri_getir,
    filoyu_skorla,
    modeli_egit_veya_yukle,
    tek_ucak_skorla,
)
from core.pilot_yakit_simulasyonu import pilot_simulasyonu_calistir
from core.veri_yukleyici import GORUNUR_ISIMLER, filo_son_durumu_getir

WEB_KLASORU = os.path.dirname(os.path.abspath(__file__))
HTML_YOLU = os.path.join(WEB_KLASORU, "index.html")

DURUM_KILIDI = threading.Lock()
DURUM = {
    "model_paketi": None,
    "train_df": None,
    "test_df": None,
    "filo_ham": None,
    "bootstrap": None,
}


def _durumu_hazirla(model_paketi):
    """Aktif yapay zeka modeli ile 100 test uçağını ön-skorlar ve arayüz önbelleğini oluşturur."""
    filo_ham = DURUM["filo_ham"]
    skorlu = filoyu_skorla(filo_ham, model_paketi)
    sensorler = model_paketi["sensorler"]
    saglikli_ist = model_paketi["saglikli_istatistik"]

    filo_listesi = []
    for _, s in skorlu.iterrows():
        uid = int(s["ucak_id"])
        s_dict = {k: round(float(s[k]), 4) for k in sensorler}
        rapor = ariza_teshisi_yap(
            ucak_id=uid,
            sensor_sozlugu=s_dict,
            tahmini_kalan_omur=float(s["tahmini_kalan_omur"]),
            saglik_skoru=float(s["saglik_skoru"]),
            saglikli_istatistik=saglikli_ist,
            ucus_sayisi=int(s["ucus_sayisi"]),
        )
        kok0 = rapor["kok_nedenler"][0]["alt_sistem"] if rapor["kok_nedenler"] else "Sağlıklı"
        if "Pervane" in kok0 or "Fan" in kok0:
            ariza_tipi, ariza_etiketi = "fan", "Ön Pervane (Fan)"
        elif "Sıkıştırıcı" in kok0 or "HPC" in kok0:
            ariza_tipi, ariza_etiketi = "hpc", "Ana Sıkıştırıcı (HPC)"
        else:
            ariza_tipi, ariza_etiketi = "saglikli", "Sağlıklı"

        filo_listesi.append({
            "ucak_id": uid,
            "ucus_sayisi": int(s["ucus_sayisi"]),
            "kalan_ucus_omru": int(s["kalan_ucus_omru"]),
            "tahmini_kalan_omur": float(s["tahmini_kalan_omur"]),
            "saglik_skoru": float(s["saglik_skoru"]),
            "yakit_verimlilik_skoru": float(s["yakit_verimlilik_skoru"]),
            "termal_denge_skoru": float(s["termal_denge_skoru"]),
            "ariza_tipi": ariza_tipi,
            "ariza_etiketi": ariza_etiketi,
            "bakim_oncelik_puani": rapor["bakim_oncelik_puani"],
            "bakim_aciliyeti": rapor["bakim_aciliyeti"],
            "teshis": rapor,
            "sensorler": s_dict,
        })

    onem_df = en_onemli_sensorleri_getir(model_paketi, ilk_kac=16)
    sensor_onemleri = [
        {
            "sensor_kodu": str(r["sensor_kodu"]),
            "sensor_adi": str(r["sensor_adi"]),
            "onem_yuzdesi": round(float(r["onem_yuzdesi"]), 2),
        }
        for _, r in onem_df.iterrows()
    ]

    m = model_paketi.get("metrikler", {})
    karsilastirma = model_paketi.get("karsilastirma_tablosu") or [m]
    noktalar = model_paketi.get("karsilastirma_noktalari") or [
        {
            "ucak_id": f["ucak_id"],
            "gercek_omur": f["kalan_ucus_omru"],
            "tahmini_omur": f["tahmini_kalan_omur"],
            "hata": round(abs(f["kalan_ucus_omru"] - f["tahmini_kalan_omur"]), 1),
        }
        for f in filo_listesi
    ]

    DURUM["model_paketi"] = model_paketi
    DURUM["bootstrap"] = {
        "model": {
            "algoritma": model_paketi.get("algoritma"),
            "algoritma_kodu": model_paketi.get("algoritma_kodu"),
            "senaryo": model_paketi.get("senaryo", "FD003"),
            "metrikler": m,
            "karsilastirma_tablosu": karsilastirma,
            "karsilastirma_noktalari": noktalar,
            "sensor_onemleri": sensor_onemleri,
            "sensorler": sensorler,
            "saglikli_istatistik": saglikli_ist,
            "gorunur_isimler": {k: GORUNUR_ISIMLER.get(k, k) for k in sensorler},
            "algoritma_bilgileri": ALGORITMA_BILGILERI,
        },
        "filo": filo_listesi,
    }


def verileri_baslat():
    """NASA FD003 veri setini ve kayıtlı aktif modeli yükleyip 100 test uçağını hazırlar."""
    print("[TurbofanDSS] NASA FD003 verileri ve aktif yapay zeka modeli yükleniyor...", flush=True)
    paket, train_df, test_df = modeli_egit_veya_yukle("FD003", yeniden_egit=False)
    DURUM["train_df"] = train_df
    DURUM["test_df"] = test_df
    DURUM["filo_ham"] = filo_son_durumu_getir(test_df)
    _durumu_hazirla(paket)
    print(f"[TurbofanDSS] Hazır! Aktif Model: {paket['algoritma']} | 100 test uçağı yüklendi.", flush=True)


class ApiHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _json_gonder(self, veri, durum=200):
        govde = json.dumps(veri, ensure_ascii=False).encode("utf-8")
        self.send_response(durum)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(govde)))
        self.end_headers()
        self.wfile.write(govde)

    def _govde_oku(self):
        uzunluk = int(self.headers.get("Content-Length", "0"))
        if uzunluk <= 0:
            return {}
        return json.loads(self.rfile.read(uzunluk).decode("utf-8"))

    def do_GET(self):
        yol = self.path.split("?")[0]
        if yol in ("/", "/index.html"):
            with open(HTML_YOLU, "rb") as f:
                icerik = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(icerik)))
            self.end_headers()
            self.wfile.write(icerik)
            return

        if yol == "/api/bootstrap":
            with DURUM_KILIDI:
                self._json_gonder(DURUM["bootstrap"])
            return

        self.send_error(404, "Not Found")

    def do_POST(self):
        yol = self.path.split("?")[0]
        try:
            veri = self._govde_oku()

            if yol == "/api/bakim/teshis":
                uid = int(veri.get("ucak_id", 50))
                sensorler = veri.get("sensorler", {})
                ucus = int(veri.get("ucus_sayisi", 0))
                with DURUM_KILIDI:
                    mp = DURUM["model_paketi"]
                skor = tek_ucak_skorla(sensorler, mp)
                rapor = ariza_teshisi_yap(
                    ucak_id=uid,
                    sensor_sozlugu=sensorler,
                    tahmini_kalan_omur=skor["tahmini_kalan_omur"],
                    saglik_skoru=skor["saglik_skoru"],
                    saglikli_istatistik=mp["saglikli_istatistik"],
                    ucus_sayisi=ucus,
                )
                self._json_gonder({"skor": skor, "teshis": rapor})
                return

            if yol == "/api/pilot/simulasyon":
                uid = int(veri.get("ucak_id", 37))
                with DURUM_KILIDI:
                    mp = DURUM["model_paketi"]
                    filo_kayit = next((x for x in DURUM["bootstrap"]["filo"] if x["ucak_id"] == uid), None)
                if not filo_kayit:
                    self._json_gonder({"hata": "Uçak bulunamadı"}, 404)
                    return

                ozel = veri.get("ozel_sensor_degisimleri") or None
                sonuc = pilot_simulasyonu_calistir(
                    ucak_id=uid,
                    mevcut_sensorler=filo_kayit["sensorler"],
                    model_paketi=mp,
                    mevcut_yakit_kg=float(veri.get("mevcut_yakit_kg", 3000)),
                    hedef_menzil_km=float(veri.get("hedef_menzil_km", 800)),
                    en_yakin_us_km=float(veri.get("en_yakin_us_km", 220)),
                    ucus_modu=str(veri.get("ucus_modu", "standart")),
                    senaryo_adi=str(veri.get("senaryo_adi", "normal")),
                    ozel_sensor_degisimleri=ozel,
                    ucus_sayisi=filo_kayit["ucus_sayisi"],
                )
                self._json_gonder(sonuc)
                return

            if yol == "/api/model/egit":
                mod = veri.get("mod", "tekli")
                kod = veri.get("algoritma_kodu", "gradient_boosting")
                with DURUM_KILIDI:
                    hazir = (DURUM["train_df"], DURUM["test_df"])
                    mevcut_tablo = list(DURUM["bootstrap"]["model"]["karsilastirma_tablosu"])

                if mod == "dortlu":
                    en_iyi, _, _ = dort_modeli_egit_ve_karsilastir(senaryo="FD003", en_iyiyi_kaydet=True)
                    yeni_paket = en_iyi
                else:
                    ayar = {
                        "linear_regression": (1, 1),
                        "decision_tree": (1, 10),
                        "random_forest": (150, 12),
                        "gradient_boosting": (200, 6),
                    }.get(kod, (150, 12))
                    yeni_paket = tek_model_egit(
                        senaryo="FD003",
                        algoritma_kodu=kod,
                        agac_sayisi=ayar[0],
                        derinlik=ayar[1],
                        kaydet=True,
                        hazir_veriler=hazir,
                    )
                    guncel = False
                    for i, row in enumerate(mevcut_tablo):
                        if row.get("algoritma_kodu") == kod:
                            mevcut_tablo[i] = dict(yeni_paket["metrikler"])
                            guncel = True
                    if not guncel:
                        mevcut_tablo.append(dict(yeni_paket["metrikler"]))
                    yeni_paket["karsilastirma_tablosu"] = mevcut_tablo

                with DURUM_KILIDI:
                    _durumu_hazirla(yeni_paket)
                    yanit = DURUM["bootstrap"]
                self._json_gonder(yanit)
                return

            self.send_error(404, "Endpoint not found")
        except Exception as exc:
            self._json_gonder({"hata": str(exc)}, 500)


def sunucuyu_baslat(port=8787, tarayici_ac=True):
    verileri_baslat()
    sunucu = ThreadingHTTPServer(("127.0.0.1", int(port)), ApiHandler)
    adres = f"http://127.0.0.1:{port}"
    print(f"[TurbofanDSS] Web arayüzü çalışıyor: {adres}", flush=True)
    print("[TurbofanDSS] Kapatmak için Ctrl+C tuşlarına basabilirsiniz.", flush=True)
    if tarayici_ac:
        threading.Timer(0.7, lambda: webbrowser.open(adres)).start()
    try:
        sunucu.serve_forever()
    except KeyboardInterrupt:
        print("\n[TurbofanDSS] Sunucu kapatıldı.")
    finally:
        sunucu.server_close()

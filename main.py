# TurbofanDSS — Savaş Uçağı Motoru Karar Destek Sistemi (Ana Giriş Noktası)
# Çalıştırmak için: python main.py

import argparse
import os
import sys

PROJE_KLASORU = os.path.dirname(os.path.abspath(__file__))
if PROJE_KLASORU not in sys.path:
    sys.path.insert(0, PROJE_KLASORU)

from web.sunucu import sunucuyu_baslat


def main():
    parser = argparse.ArgumentParser(description="TurbofanDSS — Savaş Uçağı Motoru Karar Destek Sistemi")
    parser.add_argument("--port", type=int, default=8787, help="Yerel web sunucusu portu (varsayılan: 8787)")
    parser.add_argument("--no-browser", action="store_true", help="Tarayıcıyı otomatik açma")
    args = parser.parse_args()

    sunucuyu_baslat(port=args.port, tarayici_ac=not args.no_browser)


if __name__ == "__main__":
    main()

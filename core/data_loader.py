"""Data loading, meaningful column labeling, and CSV export for NASA C-MAPSS."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "Nasa Turbofan-Ham Data"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

RAW_COLUMNS: List[str] = (
    ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
)

# Meaningful snake_case column names mapped from raw NASA C-MAPSS indices.
RAW_TO_MEANINGFUL: Dict[str, str] = {
    "unit": "motor_id",
    "cycle": "ucus_dongusu",
    "op1": "irtifa_kft",
    "op2": "mach_hizi",
    "op3": "gaz_kolu_acisi",
    "s1": "fan_giris_sicakligi_r",
    "s2": "lpc_cikis_sicakligi_r",
    "s3": "hpc_cikis_sicakligi_r",
    "s4": "lpt_cikis_sicakligi_r",
    "s5": "fan_giris_basinci_psia",
    "s6": "bypass_basinci_psia",
    "s7": "hpc_cikis_toplam_basinci_psia",
    "s8": "fan_devri_rpm",
    "s9": "cekirdek_devri_rpm",
    "s10": "motor_basinc_orani",
    "s11": "hpc_cikis_statik_basinci_psia",
    "s12": "yakit_akis_basinc_orani",
    "s13": "duzeltilmis_fan_devri_rpm",
    "s14": "duzeltilmis_cekirdek_devri_rpm",
    "s15": "bypass_orani",
    "s16": "yakit_hava_orani",
    "s17": "bleed_entalpisi",
    "s18": "istenen_fan_devri_rpm",
    "s19": "istenen_duzeltilmis_fan_devri_rpm",
    "s20": "hpt_sogutma_havasi_lbms",
    "s21": "lpt_sogutma_havasi_lbms",
}

# Human-readable Turkish labels for PyQt5 tables, charts, and documentation.
DISPLAY_LABELS: Dict[str, str] = {
    "motor_id": "Motor (Uçak) ID",
    "ucus_dongusu": "Uçuş Döngüsü",
    "irtifa_kft": "İrtifa (bin ft)",
    "mach_hizi": "Mach Hızı",
    "gaz_kolu_acisi": "Gaz Kolu Açısı (TRA)",
    "fan_giris_sicakligi_r": "Fan Giriş Sıcaklığı (°R)",
    "lpc_cikis_sicakligi_r": "LPC Çıkış Sıcaklığı (°R)",
    "hpc_cikis_sicakligi_r": "HPC Çıkış Sıcaklığı (°R)",
    "lpt_cikis_sicakligi_r": "LPT Çıkış Sıcaklığı (°R)",
    "fan_giris_basinci_psia": "Fan Giriş Basıncı (psia)",
    "bypass_basinci_psia": "Bypass Kanal Basıncı (psia)",
    "hpc_cikis_toplam_basinci_psia": "HPC Çıkış Toplam Basıncı (psia)",
    "fan_devri_rpm": "Fan Devri (RPM)",
    "cekirdek_devri_rpm": "Çekirdek Devri (RPM)",
    "motor_basinc_orani": "Motor Basınç Oranı (EPR)",
    "hpc_cikis_statik_basinci_psia": "HPC Çıkış Statik Basıncı (psia)",
    "yakit_akis_basinc_orani": "Yakıt Akışı / Basınç Oranı",
    "duzeltilmis_fan_devri_rpm": "Düzeltilmiş Fan Devri (RPM)",
    "duzeltilmis_cekirdek_devri_rpm": "Düzeltilmiş Çekirdek Devri (RPM)",
    "bypass_orani": "Bypass Oranı (BPR)",
    "yakit_hava_orani": "Yanma Odası Yakıt-Hava Oranı",
    "bleed_entalpisi": "Bleed Entalpisi",
    "istenen_fan_devri_rpm": "İstenen Fan Devri (RPM)",
    "istenen_duzeltilmis_fan_devri_rpm": "İstenen Düzeltilmiş Fan Devri (RPM)",
    "hpt_sogutma_havasi_lbms": "HPT Soğutma Havası Akışı (lbm/s)",
    "lpt_sogutma_havasi_lbms": "LPT Soğutma Havası Akışı (lbm/s)",
    "kalan_omur_rul": "Kalan Faydalı Ömür (RUL - Döngü)",
}

NON_FEATURE_COLUMNS = {"motor_id", "ucus_dongusu", "kalan_omur_rul"}


def _read_raw_table(file_path: Path) -> pd.DataFrame:
    """Read a whitespace-delimited C-MAPSS text file with meaningful column names."""
    if not file_path.exists():
        raise FileNotFoundError(f"Veri dosyası bulunamadı: {file_path}")
    df = pd.read_csv(file_path, sep=r"\s+", header=None, names=RAW_COLUMNS)
    return df.rename(columns=RAW_TO_MEANINGFUL)


def _attach_train_rul(train_df: pd.DataFrame) -> pd.DataFrame:
    """Compute Remaining Useful Life (RUL) for run-to-failure training trajectories."""
    df = train_df.copy()
    max_cycles = df.groupby("motor_id")["ucus_dongusu"].transform("max")
    df["kalan_omur_rul"] = max_cycles - df["ucus_dongusu"]
    return df


def _attach_test_rul(test_df: pd.DataFrame, rul_path: Path) -> pd.DataFrame:
    """Attach true RUL values to every cycle of the truncated test trajectories."""
    if not rul_path.exists():
        raise FileNotFoundError(f"RUL dosyası bulunamadı: {rul_path}")
    rul_df = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["end_rul"])
    rul_df["motor_id"] = range(1, len(rul_df) + 1)

    df = test_df.copy()
    max_cycles = df.groupby("motor_id")["ucus_dongusu"].transform("max")
    df = df.merge(rul_df, on="motor_id", how="left")
    df["kalan_omur_rul"] = df["end_rul"] + (max_cycles - df["ucus_dongusu"])
    return df.drop(columns=["end_rul"])


def find_constant_columns(df: pd.DataFrame, tol: float = 1e-9) -> List[str]:
    """Return feature columns whose standard deviation is effectively zero."""
    return [
        col
        for col in df.columns
        if col not in NON_FEATURE_COLUMNS and float(df[col].std()) < tol
    ]


def load_dataset(
    subset: str = "FD001",
    drop_constant: bool = False,
    raw_dir: Path = RAW_DATA_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test DataFrames for a C-MAPSS subset with labeled columns and RUL."""
    train_df = _attach_train_rul(_read_raw_table(raw_dir / f"train_{subset}.txt"))
    test_df = _attach_test_rul(
        _read_raw_table(raw_dir / f"test_{subset}.txt"),
        raw_dir / f"RUL_{subset}.txt",
    )

    if drop_constant:
        const_cols = find_constant_columns(train_df)
        train_df = train_df.drop(columns=const_cols)
        test_df = test_df.drop(columns=const_cols)

    return train_df, test_df


def get_fleet_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """Return the latest cycle row for each engine as a current-state fleet table."""
    idx = df.groupby("motor_id")["ucus_dongusu"].idxmax()
    return df.loc[idx].sort_values("motor_id").reset_index(drop=True)


def export_meaningful_csvs(
    subset: str = "FD001",
    output_dir: Path = PROCESSED_DATA_DIR,
    drop_constant: bool = False,
) -> Dict[str, Path]:
    """Export labeled train, test, and fleet-snapshot CSV files for inspection."""
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df, test_df = load_dataset(subset=subset, drop_constant=drop_constant)
    fleet_df = get_fleet_snapshot(test_df)

    paths = {
        "train": output_dir / f"{subset}_train_anlamli.csv",
        "test": output_dir / f"{subset}_test_anlamli.csv",
        "fleet": output_dir / f"{subset}_filo_ozet.csv",
    }
    train_df.to_csv(paths["train"], index=False, encoding="utf-8-sig")
    test_df.to_csv(paths["test"], index=False, encoding="utf-8-sig")
    fleet_df.to_csv(paths["fleet"], index=False, encoding="utf-8-sig")
    return paths


if __name__ == "__main__":
    exported = export_meaningful_csvs(subset="FD001", drop_constant=False)
    for key, path in exported.items():
        df_preview = pd.read_csv(path)
        print(f"[{key}] -> {path} | shape={df_preview.shape}")

from __future__ import annotations

from pathlib import Path
import re

from astropy.io import fits
from dataclasses import dataclass


@dataclass(frozen=True)
class TessProductMeta:
    tic_id: str
    sector: str

    @property
    def label(self) -> str:
        return f"TIC_{self.tic_id}_{self.sector}"


def find_fits(input_path: Path) -> list[Path]:
    """Return FITS files from a single file path or a directory tree."""
    input_path = input_path.expanduser().resolve()
    if input_path.is_file():
        if input_path.suffix.lower() != ".fits":
            raise ValueError(f"Input file is not a .fits file: {input_path}")
        return [input_path]
    if input_path.is_dir():
        return sorted(input_path.rglob("*.fits"))
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def fits_label(fits_path: Path) -> str:
    """Create a compact label from a TESS product filename."""
    meta = parse_tess_product_meta(fits_path)
    if meta:
        return f"TIC {meta.tic_id} | {meta.sector}"
    return fits_path.stem


def parse_tess_product_meta(fits_path: Path) -> TessProductMeta | None:
    """Parse TIC and sector identifiers from common TESS product filenames."""
    match = re.search(r"-(s\d{4})-(\d{16})-", fits_path.name.lower())
    if not match:
        return None
    sector = match.group(1).upper()
    tic_id = str(int(match.group(2)))
    return TessProductMeta(tic_id=tic_id, sector=sector)


def validate_fits(path: Path) -> None:
    """Raise if a FITS file cannot be opened and fully materialized."""
    with fits.open(path, memmap=False) as hdul:
        for hdu in hdul:
            data = getattr(hdu, "data", None)
            if data is not None:
                _ = data.shape

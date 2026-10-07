# -*- coding: utf-8 -*-
"""Testlar uchun sintetik geofizik loyiha generatori (rasterlar + musbat nuqtalar + AOI)."""
from __future__ import annotations

import os

import numpy as np
import rasterio
from rasterio.transform import from_origin, xy as rio_xy
import geopandas as gpd
from scipy.ndimage import gaussian_filter
from shapely.geometry import Point, box

X0, Y0 = 11_500_000.0, 4_600_000.0   # EPSG:28411 hududidagi koordinatalar


def make_synthetic_project(root, size=120, n_layers=6, n_pos=22, resolution=100.0,
                           categorical=False, with_crs=True, nodata_patch=True, seed=0,
                           signal_layers=(0, 1)):
    """
    root ichida tiff/, pts/, aoi/ papkalarini yaratadi.
    Musbat nuqtalar signal (signal_layers yig'indisi) yuqori bo'lgan joylardan olinadi.
    categorical=True bo'lsa, qo'shimcha butun sonli 'geology_cat.tif' (1..4) qatlam qo'shiladi.
    Qaytaradi: dict(tiff=..., points=..., aoi=..., size, resolution, transform, layer_names, n_pos).
    """
    rng = np.random.default_rng(seed)
    for d in ("tiff", "pts", "aoi"):
        os.makedirs(os.path.join(root, d), exist_ok=True)
    H = W = int(size)
    transform = from_origin(X0, Y0 + H * resolution, resolution, resolution)
    sigma = max(3.0, size / 20.0)

    fields = []
    names = []
    for i in range(n_layers):
        f = gaussian_filter(rng.normal(size=(H, W)), sigma=sigma)
        f = (f - f.mean()) / f.std()
        fields.append(f)
        names.append(f"layer{i + 1}")
        arr = f.astype("float32")
        if nodata_patch and i == 2:
            arr[:6, :6] = -9999
        _write_tif(os.path.join(root, "tiff", f"layer{i + 1}.tif"), arr, transform, with_crs, nodata=-9999)

    if categorical:
        g = gaussian_filter(rng.normal(size=(H, W)), sigma=sigma * 1.5)
        qs = np.quantile(g, [0.25, 0.5, 0.75])
        cat = (1 + np.digitize(g, qs)).astype("float32")
        _write_tif(os.path.join(root, "tiff", "geology_cat.tif"), cat, transform, with_crs, nodata=-9999)
        names.append("geology_cat")

    score = sum(fields[i] for i in signal_layers)
    rows, cols = np.where(score > np.percentile(score, 92))
    sel = rng.choice(len(rows), size=min(n_pos, len(rows)), replace=False)
    pts = []
    for r, c in zip(rows[sel], cols[sel]):
        x, y = rio_xy(transform, int(r), int(c))
        pts.append(Point(x, y))
    gpd.GeoDataFrame(geometry=pts, crs="EPSG:28411").to_file(os.path.join(root, "pts", "pts.shp"))

    margin = 5 * resolution
    aoi = box(X0 + margin, Y0 + margin, X0 + W * resolution - margin, Y0 + H * resolution - margin)
    gpd.GeoDataFrame(geometry=[aoi], crs="EPSG:28411").to_file(os.path.join(root, "aoi", "aoi.shp"))
    return {
        "tiff": os.path.join(root, "tiff"), "points": os.path.join(root, "pts"),
        "aoi": os.path.join(root, "aoi"), "size": size, "resolution": resolution,
        "transform": transform, "layer_names": names, "n_pos": len(pts),
    }


def _write_tif(path, arr, transform, with_crs, nodata):
    kw = dict(driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1, dtype="float32",
              transform=transform, nodata=nodata)
    if with_crs:
        kw["crs"] = "EPSG:28411"
    with rasterio.open(path, "w", **kw) as dst:
        dst.write(arr, 1)

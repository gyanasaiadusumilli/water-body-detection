"""Download training scenes straight from Earth Engine as GeoTIFFs (no Google Drive export).

Each scene is a 1024 x 1024 pixel cloud-masked monthly composite with 7 bands
(blue, green, red, nir, swir1, swir2, label), written to data/raw/<sensor>/<area>.tif.
Sentinel-2 scenes are 10 m pixels (about 10 km across); Landsat-8 scenes are 30 m (about 31 km).

Usage:
    python gee/fetch_scenes.py --project YOUR_GEE_PROJECT
    python gee/fetch_scenes.py --project YOUR_GEE_PROJECT --sensor sentinel2 --areas hyderabad chilika
"""
import argparse
import math
import sys
from pathlib import Path

import ee

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_scenes import landsat8, sentinel2, water_label  # noqa: E402

# name: (longitude, latitude, year, month). Dry-season months keep cloud cover low.
# JRC monthly water labels run to the end of 2021.
AREAS = {
    "hyderabad":   (78.47, 17.42, 2021, 2),   # city lakes and reservoirs
    "kolleru":     (81.25, 16.65, 2021, 2),   # shallow lake surrounded by aquaculture ponds
    "chilika":     (85.35, 19.70, 2021, 2),   # coastal lagoon
    "varanasi":    (83.00, 25.30, 2021, 3),   # Ganges river and floodplain
    "okeechobee":  (-80.85, 26.95, 2021, 3),  # large lake and wetlands
    "ijsselmeer":  (5.35, 52.55, 2021, 4),    # lake, polders and canals
    "hudson":      (-73.95, 41.05, 2021, 9),  # river and reservoirs near New York
    "victoria":    (32.65, 0.25, 2021, 7),    # lake shoreline near Kampala
}
SIZE = 1024


def utm_epsg(lon, lat):
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def fetch(sensor, name, lon, lat, year, month, max_cloud, out_dir):
    epsg = utm_epsg(lon, lat)
    center = ee.Geometry.Point([lon, lat]).transform(f"EPSG:{epsg}", 1).coordinates().getInfo()
    start = ee.Date.fromYMD(year, month, 1)
    end = start.advance(1, "month")
    fn = sentinel2 if sensor == "sentinel2" else landsat8
    scale = 10 if sensor == "sentinel2" else 30
    half = SIZE * scale / 2
    x0, y1 = math.floor(center[0] - half), math.ceil(center[1] + half)
    aoi = ee.Geometry.Rectangle([x0, y1 - SIZE * scale, x0 + SIZE * scale, y1], f"EPSG:{epsg}", False)

    img, _, n = fn(aoi, start, end, max_cloud)
    n = n.getInfo()
    if n == 0:
        print(f"  {sensor:9s} {name:11s} skipped: no clear scenes in {year}-{month:02d}")
        return False
    lbl = water_label(year, month)
    lbl = lbl.where(img.select("blue").mask().Not(), 255).unmask(255)
    stack = img.unmask(0).addBands(lbl).toFloat()

    data = ee.data.computePixels({
        "expression": stack,
        "fileFormat": "GEO_TIFF",
        "grid": {
            "dimensions": {"width": SIZE, "height": SIZE},
            "affineTransform": {"scaleX": scale, "shearX": 0, "translateX": x0,
                                "shearY": 0, "scaleY": -scale, "translateY": y1},
            "crsCode": f"EPSG:{epsg}",
        },
    })
    path = Path(out_dir) / sensor / f"{name}_{year}{month:02d}.tif"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    print(f"  {sensor:9s} {name:11s} {n:2d} images composited -> {path} ({len(data) / 1e6:.0f} MB)")
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True, help="Google Cloud project registered for Earth Engine")
    p.add_argument("--sensor", choices=["sentinel2", "landsat8", "both"], default="both")
    p.add_argument("--areas", nargs="*", default=list(AREAS), choices=list(AREAS))
    p.add_argument("--max-cloud", type=float, default=60, help="Scene-level filter; clouds are also masked per pixel")
    p.add_argument("--out-dir", default="data/raw")
    args = p.parse_args()

    ee.Initialize(project=args.project)
    sensors = ["sentinel2", "landsat8"] if args.sensor == "both" else [args.sensor]
    ok = 0
    for sensor in sensors:
        for name in args.areas:
            try:
                ok += fetch(sensor, name, *AREAS[name], args.max_cloud, args.out_dir)
            except Exception as e:  # one bad area should not stop the rest
                print(f"  {sensor:9s} {name:11s} failed: {e}")
    print(f"Downloaded {ok} scenes.")


if __name__ == "__main__":
    main()

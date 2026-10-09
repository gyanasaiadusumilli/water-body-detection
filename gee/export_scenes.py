"""Export cloud-masked Sentinel-2 and Landsat-8 composites with water labels from Google Earth Engine.

Each export is one GeoTIFF with 7 bands:
    blue, green, red, nir, swir1, swir2   (surface reflectance, 0-1)
    label                                 (1 = water, 0 = land, 255 = no data / ignore)

Labels come from the JRC Global Surface Water Monthly History for the same month as the
imagery, so the label and image describe the same point in time.

Setup (one time):
    pip install earthengine-api
    earthengine authenticate
Usage:
    python gee/export_scenes.py --project YOUR_GEE_PROJECT --name hyderabad \
        --bbox 78.25 17.25 78.60 17.55 --year 2021 --month 3 --sensor both

Exports land in Google Drive under /water_body_data. Download them into
data/raw/sentinel2/ and data/raw/landsat8/.
"""
import argparse

import ee

BANDS = ["blue", "green", "red", "nir", "swir1", "swir2"]


def sentinel2(aoi, start, end, max_cloud):
    def mask_clouds(img):
        scl = img.select("SCL")
        # SCL classes: 1 saturated, 3 cloud shadow, 8/9 cloud, 10 cirrus
        ok = scl.neq(1).And(scl.neq(3)).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
        return img.updateMask(ok)

    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
           .filterBounds(aoi).filterDate(start, end)
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
           .map(mask_clouds))
    img = col.median().select(["B2", "B3", "B4", "B8", "B11", "B12"], BANDS).divide(10000)
    return img, 10, col.size()


def landsat8(aoi, start, end, max_cloud):
    def mask_and_scale(img):
        qa = img.select("QA_PIXEL")
        # Bits 1-4: dilated cloud, cirrus, cloud, cloud shadow
        ok = qa.bitwiseAnd(0b11110).eq(0)
        sr = img.select(["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"], BANDS)
        return sr.multiply(0.0000275).add(-0.2).updateMask(ok)

    col = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
           .filterBounds(aoi).filterDate(start, end)
           .filter(ee.Filter.lt("CLOUD_COVER", max_cloud))
           .map(mask_and_scale))
    return col.median(), 30, col.size()


def water_label(year, month):
    gsw = (ee.ImageCollection("JRC/GSW1_4/MonthlyHistory")
           .filter(ee.Filter.eq("year", year))
           .filter(ee.Filter.eq("month", month))
           .first()
           .select("water"))
    # water: 0 = no data, 1 = not water, 2 = water
    return gsw.remap([0, 1, 2], [255, 0, 1]).rename("label")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True, help="Your Google Cloud project registered for Earth Engine")
    p.add_argument("--name", required=True, help="Short name for this area, used in file names")
    p.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("W", "S", "E", "N"))
    p.add_argument("--year", type=int, default=2021, help="JRC monthly labels cover 1984-2021")
    p.add_argument("--month", type=int, default=3)
    p.add_argument("--sensor", choices=["sentinel2", "landsat8", "both"], default="both")
    p.add_argument("--max-cloud", type=float, default=30)
    p.add_argument("--folder", default="water_body_data")
    args = p.parse_args()

    ee.Initialize(project=args.project)
    aoi = ee.Geometry.Rectangle(args.bbox)
    start = ee.Date.fromYMD(args.year, args.month, 1)
    end = start.advance(1, "month")
    label = water_label(args.year, args.month)

    sensors = ["sentinel2", "landsat8"] if args.sensor == "both" else [args.sensor]
    for sensor in sensors:
        fn = sentinel2 if sensor == "sentinel2" else landsat8
        img, scale, n_scenes = fn(aoi, start, end, args.max_cloud)
        print(f"{sensor}: {n_scenes.getInfo()} scenes in {args.year}-{args.month:02d}")
        # Pixels masked as cloud in the imagery become "ignore" in the label
        lbl = label.where(img.select("blue").mask().Not(), 255).unmask(255)
        stack = img.unmask(0).addBands(lbl).toFloat().clip(aoi)
        desc = f"{sensor}_{args.name}_{args.year}{args.month:02d}"
        task = ee.batch.Export.image.toDrive(
            image=stack, description=desc, folder=args.folder, fileNamePrefix=desc,
            region=aoi, scale=scale, maxPixels=1e10, fileFormat="GeoTIFF")
        task.start()
        print(f"Started export {desc} (scale {scale} m). Track it at https://code.earthengine.google.com/tasks")


if __name__ == "__main__":
    main()

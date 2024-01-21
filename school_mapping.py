import json
import os
import requests
from PIL import Image

INFER_PIPELINE_URL = "http://localhost:8552"


"""
Setup the folder structure with the required data for school mapping inference.
Inputs:
- folder: the directory path of the folder to save the required files
- img: the satellite imagery
- bounds: a dictionary containing the boundary coordinates. 
    It must have the following keys: lon_min, lon_max, lat_min, lat_max
"""


def setup_folder(folder, img, bounds):
    img_folder = os.path.join(folder, 'imgs')
    os.makedirs(img_folder, exist_ok=True)

    geo_folder = os.path.join(folder, 'geojsons')
    os.makedirs(geo_folder, exist_ok=True)

    # save the image
    Image.fromarray(img, 'RGB').save(os.path.join(img_folder, 'Bounds_stitched_img.png'))

    # save the geojson
    geo_js = create_geojson(**bounds)
    with open(os.path.join(geo_folder, 'Bounds_stitched_img.geojson'), 'w') as f:
        f.write(geo_js)


def create_geojson(lon_min, lon_max, lat_min, lat_max):
    geo_str = f"""{{
    "type": "FeatureCollection",
    "crs": {{"type": "name", "properties": {{"name": "urn:ogc:def:crs:EPSG::4326"}}}},
    "features": [
    {{"type": "Feature", 
      "properties": {{"top": {lat_max}, 
                      "left": {lon_min}, 
                      "bottom": {lat_min}, 
                      "right": {lon_max}, "id": 0 }}, """

    geo_str += f"""
      "geometry": {{"type": "Polygon", 
                    "coordinates": [ [ [ {lon_min}, {lat_min} ],
                                       [ {lon_max}, {lat_min} ], 
                                       [ {lon_max}, {lat_max} ], 
                                       [ {lon_min}, {lat_max} ], 
                                       [ {lon_min}, {lat_min} ]
                                         ] ] }} }}
    ]
    }}
    """

    return geo_str


def run_school_mapping(folder, img, bounds):
    print('Running school mapping...')

    setup_folder(folder, img, bounds)

    # run pipeline
    response = requests.post(url=INFER_PIPELINE_URL,
                             json={'folder': os.path.abspath(folder)})

    response = response.json()
    print(response)

    # processes the response
    results = {}
    if 'results' in response:
        with open(response['results'], 'r') as f:
            results = json.load(f)

    print('School mapping done!')

    return results

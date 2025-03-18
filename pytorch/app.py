from flask import Flask, request
from flask_cors import CORS, cross_origin
import logging
import json

import math
import mapbox as md

import os
import uuid

from sat2graph import run_sat2graph
from school_mapping import run_school_mapping
from time import time
from datetime import datetime

app = Flask(__name__)
app.secret_key = "123"

app.config['CORS_HEADERS'] = 'Content-Type'
app.config['CORS_RESOURCES'] = {r"/*": {"origins": "*"}}
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 300
cors = CORS(app)

tracker = 0  # a variable to keep a count of healthcheck api calls
UI_PASS = 'test123' # password for the UI


@app.route('/', methods=['GET', 'POST'])
@cross_origin()
def index():
    try:
        if request.get_json():
            t0 = time()
            data = request.get_json()

            cache_folder = get_temp_folder()
            lat, lon = data["lat"], data["lon"]

            sat_img, bounds = download_tile(lat, lon, cache_folder)

            # Run sat2graph
            t1 = time()
            lines, points = run_sat2graph(sat_img, data)

            # Run school mapping
            t2 = time()
            schools = run_school_mapping(folder=cache_folder, img=sat_img, bounds=bounds)
            t3 = time()

            return_str = json.dumps({"graph": [lines, points], "success": "true", "schools": schools})

            t4 = time()
            print('output:', return_str)
            print('Total time:', t4 - t0)
            print('- img download time:', t1 - t0)
            print('- sat2graph time:', t2 - t1)
            print('- school mapping time:', t3 - t2)
        else:
            return_str = json.dumps({"success": "false"})

    except Exception as e:
        logging.error(f'[exception] : {e}')
        return_str = json.dumps({"success": "false"})

    return return_str


@app.after_request
def add_header(response):
    response.cache_control.max_age = 300
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response


@app.route('/healthcheck', methods=['GET', 'POST'])
@cross_origin()
def health_check():
    global tracker

    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    tracker = tracker + 1

    return_str = json.dumps({'text': 'Hello ASM!',
                             'counter': tracker,
                             'current-time': now})
    return return_str


@app.route('/login', methods=['POST'])
@cross_origin()
def login():
    try:
        data = request.get_json()
        if data:
            password = data['p']
            if password == UI_PASS:
                return '{"success": 1}'
    except Exception as e:
        print(e)
        return '{"success": 0}'
    return '{"success": 0}'


def download_tile(lat, lon, cache_folder='mapbox_cache/'):
    lat_end = lat + (500 + 102) / 111111.0
    lon_end = lon + (500 + 102) / 111111.0 / math.cos(math.radians(lat_end))

    lat_st = lat - 102 / 111111.0
    lon_st = lon - 102 / 111111.0 / math.cos(math.radians(lat_end))

    if abs(lat) < 33:
        zoom = 17
    else:
        zoom = 16

    # download imagery
    img, _ = md.GetMapInRect(lat_st, lon_st, lat_end, lon_end, folder=cache_folder, zoom=zoom)

    # save the bounding coordinates
    bounds = {'lon_min': lon_st, 'lon_max': lon_end,
              'lat_min': lat_st, 'lat_max': lat_end}

    return img, bounds


def get_temp_folder():
    unique_filename = str(uuid.uuid4().hex)
    os.makedirs(unique_filename, exist_ok=True)
    unique_filename = unique_filename + '/'
    return unique_filename


if __name__ == '__main__':
    app.run(debug=False, port=8002, host="0.0.0.0", threaded=True)
    logging.info('[server is running] : 8002')
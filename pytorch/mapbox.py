import math
import os
from PIL import Image
import requests
from shutil import move
import numpy as np
from time import sleep


def lonlat2mapboxTile(lonlat, zoom):
    n = np.exp2(zoom)
    x = int((lonlat[0] + 180)/360*n)
    y = int((1 - math.log(math.tan(lonlat[1] * math.pi / 180) + (1 / math.cos(lonlat[1] * math.pi / 180))) / math.pi) / 2 * n)
    return [x, y]


def lonlat2TilePos(lonlat, zoom):
    n = np.exp2(zoom)
    ix = int((lonlat[0] + 180) / 360 * n)
    iy = int((1 - math.log(math.tan(lonlat[1] * math.pi / 180) + (1 / math.cos(lonlat[1] * math.pi / 180))) / math.pi) / 2 * n)

    x = ((lonlat[0] + 180) / 360 * n)
    y = ((1 - math.log(math.tan(lonlat[1] * math.pi / 180) + (1 / math.cos(lonlat[1] * math.pi / 180))) / math.pi) / 2 * n)

    x = int((x - ix) * 512)
    y = int((y - iy) * 512)

    return x, y


def downloadMapBox(zoom, p, outputname, folder):
    url = f"https://c.tiles.mapbox.com/v4/mapbox.satellite/{zoom}/{p[0]}/{p[1]}@2x.jpg?access_token=pk.eyJ1IjoibWZhdGVoIiwiYSI6ImNsYjR1Y2V2MjAwOHYzeHRueGlzYWl5ZmEifQ.sxZE4O12cUQdrx_kK_1bHg"

    filename = f"{p[1]}@2x.jpg?access_token=pk.eyJ1IjoibWZhdGVoIiwiYSI6ImNsYjR1Y2V2MjAwOHYzeHRueGlzYWl5ZmEifQ.sxZE4O12cUQdrx_kK_1bHg"
    filename = os.path.join(folder, filename)

    # Set the initial retry timeout to 10 seconds
    retry_timeout = 10

    while not os.path.isfile(filename):
        # Use requests library to download the tile image with a timeout of 30 seconds
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error downloading tile: {e}")
            sleep(retry_timeout)
            retry_timeout += 10
            if retry_timeout > 60:
                retry_timeout = 60
            print("Retry, timeout is ", retry_timeout)
            continue

        # Save the downloaded image to the specified output name
        with open(outputname, 'wb') as f:
            f.write(response.content)

        # Move the downloaded image to the specified folder
        move(outputname, filename)

    return True


def GetMapInRect(min_lat, min_lon, max_lat, max_lon, folder="mapbox_cache/", zoom=19):
    mapbox1 = lonlat2mapboxTile([min_lon, min_lat], zoom)
    mapbox2 = lonlat2mapboxTile([max_lon, max_lat], zoom)

    ok = True

    print(mapbox1, mapbox2)
    print((mapbox2[0] - mapbox1[0]) * (mapbox1[1] - mapbox2[1]))

    dimx = (mapbox2[0] - mapbox1[0] + 1) * 512  # lon
    dimy = (mapbox1[1] - mapbox2[1] + 1) * 512  # lat

    img = np.zeros((dimy, dimx, 3), dtype=np.uint8)

    for i in range(mapbox2[0] - mapbox1[0] + 1):
        if not ok:
            break

        for j in range(mapbox1[1] - mapbox2[1] + 1):
            filename = f"{folder}/{zoom}_{i + mapbox1[0]}_{j + mapbox2[1]}.jpg"

            succ = downloadMapBox(zoom, [i + mapbox1[0], j + mapbox2[1]], filename, folder)

            if succ:
                sub_img = Image.open(filename)
                sub_img = np.array(sub_img).astype(np.uint8)
                img[j * 512:(j + 1) * 512, i * 512:(i + 1) * 512, :] = sub_img
            else:
                ok = False
                break

    x1, y1 = lonlat2TilePos([min_lon, max_lat], zoom)
    x2, y2 = lonlat2TilePos([max_lon, min_lat], zoom)

    x2 = x2 + dimx - 512
    y2 = y2 + dimy - 512

    img = img[y1:y2, x1:x2]

    return img, ok

import math
import json

def load_config(path="config.json"):
    with open(path, 'r') as f:
        return json.load(f)
    
def sun_vector(azimuth_deg, elevation_deg):
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    sx = math.cos(el) * math.sin(az)
    sy = math.cos(el) * math.cos(az)
    sz = math.sin(el)
    v = [sx, sy, sz]
    norm = math.sqrt(sum([x**2 for x in v]))
    return [x / norm for x in v]


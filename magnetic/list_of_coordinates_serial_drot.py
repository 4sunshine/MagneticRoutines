from sunpy.net import Fido, attrs as a
from astropy.time import Time
from sunpy.map import Map
from typing import Union
from astropy.coordinates import SkyCoord
from sunpy.coordinates.frames import HeliographicCarrington, Helioprojective
from sunpy.physics.differential_rotation import solar_rotate_coordinate
import astropy.units as u
import os


def xy_arcsec_sdo(x: Union[int, float], y: Union[int, float], obs_time: Time):
    Tx = x * u.arcsec
    Ty = y * u.arcsec
    return SkyCoord(Tx, Ty, obstime=obs_time, observer="SDO", frame="helioprojective")


def carrington_lon_lat_2sky(x_lon: Union[int, float], y_lat: Union[int, float], obs_time: Time):
    lon = float(x_lon) * u.deg
    lat = float(y_lat) * u.deg
    return SkyCoord(lon=lon, lat=lat, frame=HeliographicCarrington, obstime=obs_time, observer='earth')

def carrington_sky_2lonlat(sky_coord_carr: SkyCoord):
    lon = float(sky_coord_carr.lon / u.deg)
    lat = float(sky_coord_carr.lat / u.deg)
    return round(lon, 3), round(lat, 3)

def carrington_sky_2helioprojective(sky_coord_carr: SkyCoord):
    hpc_coord = sky_coord_carr.transform_to(Helioprojective(observer="earth"))
    return hpc_coord

def helioprojective_sky_2xy(sky_coord_hp: SkyCoord):
    x = float(sky_coord_hp.Tx / u.arcsec)
    y = float(sky_coord_hp.Ty / u.arcsec)
    return round(x, 3), round(y, 3)

event_time = Time("2017-09-06T09:10:00")

# SDO Coodrinates maybe incorrect
# event_x, event_y = 501, -233
# event_coord = xy_arcsec_sdo(501, -233, event_time)

# From magnetic/get_coord_at_time.py
lon, lat = 115.929197, -8.
event_coord = carrington_lon_lat_2sky(lon, lat, event_time)

start_time = Time("2017-09-03T09:00:00")
end_time = Time("2017-09-06T12:00:00")

jsoc_email = os.environ["JSOC_EMAIL"]

result = Fido.search(
    a.Time(start_time, end_time),
    a.Sample(720 * u.s),
    a.jsoc.Series("hmi.B_720s"),
    a.jsoc.Notify(jsoc_email),
    a.jsoc.Segment("field") & a.jsoc.Segment("inclination") & a.jsoc.Segment("azimuth"),
)

print("-- Columns:")
print(result.all_colnames)

print("-- Search Result:")
print(result)

x = result.show("T_OBS", "SOURCE", "WCSNAME")

lines = ["T_OBS,CRLON,CRLAT,X,Y"]

for table in x:
    for t_obs, in table.iterrows("T_OBS"):
        t_obs = t_obs.replace(".", "-").replace("_TAI", "").replace("_", "T")
        if t_obs != "MISSING":
            cur_time = Time(t_obs)
            cur_coord = solar_rotate_coordinate(event_coord, time=cur_time)
            lon, lat = carrington_sky_2lonlat(cur_coord)

            cur_coord_helioproj = carrington_sky_2helioprojective(cur_coord)
            x, y = helioprojective_sky_2xy(cur_coord_helioproj)

            # This stands for Helioprojective
            # cur_line = ";".join(
            #     [
            #         t_obs.replace("T", " "),
            #         str(cur_coord.Tx).replace('arcsec', ''),
            #         str(cur_coord.Ty).replace('arcsec', ''),
            #     ])

            cur_line = ",".join(
                [
                    t_obs.replace("T", " "),
                    str(lon),
                    str(lat),
                    str(x),
                    str(y),
                ])

            lines.append(cur_line)

lines = "\n".join(lines)

with open(f"./hgc_coords_t_lon_lat_4EVENT_{event_time}.csv".replace(":", "."), "w") as fw:
    fw.write(lines)

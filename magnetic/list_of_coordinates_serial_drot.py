from sunpy.net import Fido, attrs as a
from astropy.time import Time
from sunpy.map import Map
from typing import Union
from astropy.coordinates import SkyCoord
from sunpy.physics.differential_rotation import solar_rotate_coordinate
import astropy.units as u
import os


def xy_arcsec_sdo(x: Union[int, float], y: Union[int, float], obs_time: Time):
    Tx = x * u.arcsec
    Ty = y * u.arcsec
    return SkyCoord(Tx, Ty, obstime=obs_time, observer="earth", frame="helioprojective")


event_time = Time("2017-09-06T09:10:00")
event_coord = xy_arcsec_sdo(501, -233, event_time)


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

lines = []

for table in x:
    for t_obs, in table.iterrows("T_OBS"):
        t_obs = t_obs.replace(".", "-").replace("_TAI", "").replace("_", "T")
        if t_obs != "MISSING":
            cur_time = Time(t_obs)
            cur_coord = solar_rotate_coordinate(event_coord, time=cur_time)
            cur_line = ";".join(
                [
                    t_obs.replace("T", " "),
                    str(cur_coord.Tx).replace('arcsec', ''),
                    str(cur_coord.Ty).replace('arcsec', ''),
                ])

            lines.append(cur_line)
            
lines = "\n".join(lines)

with open(f"./coordinates_list_{event_time}.txt", "w") as fw:
    fw.write(lines)

from sunpy.coordinates.frames import Helioprojective
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u
from astropy.time import Time
from sunpy.physics.differential_rotation import solar_rotate_coordinate

# observer_location = EarthLocation.of_site("XRT")
# print(EarthLocation.get_site_names())

# https://arxiv.org/pdf/2310.19617
# https://xrt.cfa.harvard.edu/flare_catalog/2017.html?search=160610

observer = "earth"

observation_time = Time("2017-09-06T09:10:00")

target_time = Time("2017-09-03T09:00:00")

Tx = 501 * u.arcsec
Ty = -233 * u.arcsec

c = SkyCoord(Tx, Ty, obstime=observation_time, observer=observer, frame="helioprojective")
z = solar_rotate_coordinate(c, time=target_time)

def show_helioprojective_coord(coord: SkyCoord, name: str):
    print(f"{name}\nx: {coord.Tx} y: {coord.Ty}\n")

show_helioprojective_coord(c, f"initial_at_{observation_time}")
show_helioprojective_coord(z, f"target_at_{target_time}")


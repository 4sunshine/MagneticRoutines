from sunpy.coordinates.frames import Helioprojective, HeliographicCarrington
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u
from astropy.time import Time
from sunpy.net import Fido, attrs as a
from sunpy.physics.differential_rotation import solar_rotate_coordinate

from astropy.coordinates import solar_system_ephemeris
solar_system_ephemeris.set("jpl")
from sunpy.coordinates import get_horizons_coord

# observer_location = EarthLocation.of_site("XRT")
# print(EarthLocation.get_site_names())

# https://arxiv.org/pdf/2310.19617
# https://xrt.cfa.harvard.edu/flare_catalog/2017.html?search=160610

tr = a.Time('2017-09-06T08:30:00', '2017-09-06T09:10:00')
SDO_TIME = "2017-09-06"
print(a.hek)

res = Fido.search(
    tr,
    a.hek.OBS.Instrument == 'AIA',
    a.hek.FRM.Name == 'SSW Latest Events',
    a.hek.EventType('FL'),
)

print(res['hek'].columns)

for hgc_x, hgc_y, event_peaktime, event_description in res['hek'].iterrows('hgc_x', 'hgc_y', 'event_peaktime', 'event_description'):
    print(f"********\nEvent:\n{event_description}\n---------")
    print("hgc_xy (Heliographic Carrington):", hgc_x, hgc_y, "\nTime:", event_peaktime)
    
    # Conversions
    peak_time = Time(event_peaktime)

    lon = float(hgc_x) * u.deg
    lat = float(hgc_y) * u.deg

    hgc_coord = SkyCoord(lon=lon, lat=lat, frame=HeliographicCarrington, obstime=peak_time, observer='earth')
    
    print(f"HGC sunpy:\n{hgc_coord}")

    sdo = get_horizons_coord(body='SDO', id_type=None, time=SDO_TIME)

    aia_coord = hgc_coord.transform_to(Helioprojective(observer=sdo, obstime=peak_time))

    # Better ephemeris aquiring needed
    # print("AIA:", aia_coord)

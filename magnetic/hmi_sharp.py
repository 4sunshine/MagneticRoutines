import os
import astropy.units as u
from sunpy.net.jsoc.jsoc import drms
from sunpy.net import Fido
from sunpy.net import attrs as a

jsoc_email = os.environ["JSOC_EMAIL"]

client = drms.Client()

# Search for HARPNUM
NOAA_ARS = '12673'
EVENT_TIME = '2017.09.06_09:00:00_TAI'

# Search for Data
START_TIME = '2017-09-03 09:00:00'
END_TIME = '2017-09-03 12:00:00'  # TODO: Change Later
target_dir = 'D:\\datasets\\fido\\test'

HARPNUM = None

harp = client.query(f'hmi.sharp_720s[][{EVENT_TIME}]', key = ['HARPNUM','T_REC','NOAA_ARS'])
print(harp)

HARPNUM = 7115

if HARPNUM is not None:
    target_dir = os.path.join(target_dir, f"HARPNUM_{HARPNUM}")
    os.makedirs(target_dir, exist_ok=True)
    query = Fido.search(
        a.Time(START_TIME, END_TIME),
        a.Sample(12*u.minute),  # 720s cadence
        a.jsoc.Series("hmi.sharp_cea_720s"),  # CEA coordinates
        a.jsoc.PrimeKey("HARPNUM", HARPNUM),  # HARP number
        a.jsoc.Notify(jsoc_email),
        a.jsoc.Segment("Bp"),
        a.jsoc.Segment("Bt"),
        a.jsoc.Segment("Br")
    )  # Magnetic field components

    files = Fido.fetch(query, path=target_dir + "/{file}")

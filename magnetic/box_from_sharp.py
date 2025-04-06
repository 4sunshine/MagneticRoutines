import os
import sys
import numpy as np
import astropy.units as u
from collections import defaultdict
from sunpy.map import Map
from pydave4vm import do_dave4vm

# Br Normal to the solar surface (positive outward from Sun's center)
# Bt Along the solar meridian (North-South direction)
# Bp Along the solar parallel (East-West direction)

def calculate_dx_dy(meta):
    assert meta["CDELT1"] == meta["CDELT2"]
    cdelt1_deg = float(meta["CDELT1"])
    rsun_km = (float(meta["RSUN_REF"]) / 1.e3)
    dx = rsun_km * cdelt1_deg * np.pi / 180.
    meta['DX'] = dx
    meta['DX_UNIT'] = 'km'


def extract_components(components):
    bt = Map(components["Bt"])
    by = -np.array(bt.data)
    bp = Map(components["Bp"])
    bx = np.array(bp.data)
    br = Map(components["Br"])
    bz = np.array(br.data)
    meta_ = bt.meta

    data = np.stack([bx, by, bz], axis=0)

    keys = ["HARPNUM", "DATE-OBS", "BUNIT", "NOAA_ARS", "CDELT1", "CDELT2", "RSUN_OBS", "RSUN_REF"]

    meta = {k: meta_.get(k, None) for k in keys}

    calculate_dx_dy(meta)

    return data, meta


def load_files(sharp_data_dir):

    HARPNUM = None
    data_type = None

    time_files = defaultdict(dict)

    for f in sorted(os.listdir(sharp_data_dir)):

        if not os.path.isfile(os.path.join(sharp_data_dir, f)):
            continue

        parts = os.path.basename(f).split('.')
        HARPNUM_ = parts[2]

        if HARPNUM is None:
            HARPNUM = HARPNUM_
        else:
            assert HARPNUM_ == HARPNUM

        t_rec = parts[3]
        if data_type is None:
            data_type = ".".join(parts[:2])
        component = parts[4]

        time_files[t_rec][component] = os.path.join(sharp_data_dir, f)

    np_dir = os.path.join(sharp_data_dir, "np_data")
    os.makedirs(np_dir, exist_ok=True)

    t_recs = [t_rec for t_rec in time_files]
    num_times = len(t_recs)

    for i, (t_rec, components) in enumerate(time_files.items()):

        data, meta = extract_components(components)

        data_next, meta_next = extract_components(time_files[t_recs[i + 1]])

        dt = 720. # test

        magvm, vel4vm = do_dave4vm.do_dave4vm(
            dt,
            data_next[0], data[0],
            data_next[1], data[1],
            data_next[2], data[2],
            meta["DX"], meta["DX"], 21
        )

        assert vel4vm["solved"] == True

        vx = vel4vm["U0"]
        vy = vel4vm["V0"]
        vz = vel4vm["W0"]

        V = np.stack([vx, vy, vz], axis=0)

        result = {
            "B": data,
            "V": V,
            "meta": meta,
        }

        np.savez(os.path.join(np_dir, f"{data_type}.{HARPNUM}.{t_rec}.npz"), **result)
        
        if i == (num_times - 2):
            break


if __name__ == "__main__":
    load_files(sys.argv[1])

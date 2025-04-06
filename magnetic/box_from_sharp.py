import os
import sys
import numpy as np
from collections import defaultdict
from sunpy.map import Map


# Br Normal to the solar surface (positive outward from Sun's center)
# Bt Along the solar meridian (North-South direction)
# Bp Along the solar parallel (East-West direction)


def load_files(sharp_data_dir):

    HARPNUM = None
    data_type = None

    time_files = defaultdict(dict)

    for f in os.listdir(sharp_data_dir):

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

    for t_rec, components in time_files.items():
        bt = Map(components["Bt"])
        by = -np.array(bt.data)
        bp = Map(components["Bp"])
        bx = np.array(bp.data)
        br = Map(components["Br"])
        bz = np.array(br.data)
        meta_ = bt.meta

        data = np.stack([bx, by, bz], axis=0)

        meta = {
            "HARPNUM": meta_["HARPNUM"],
            "DATE-OBS": meta_["DATE-OBS"].replace("T", " "),
            "UNITS": meta_["BUNIT"],
        }

        result = {
            "data": data,
            "meta": meta,
        }

        np.savez(os.path.join(np_dir, f"{data_type}.{HARPNUM}.{t_rec}.npz"), **result)


if __name__ == "__main__":
    load_files(sys.argv[1])

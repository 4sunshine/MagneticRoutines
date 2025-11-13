import os
import sys
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from sunpy.visualization import wcsaxes_compat
from datetime import datetime
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
    meta['ARCSEC2CM'] = meta["RSUN_REF"] * 1e2 / meta["RSUN_OBS"]


def extract_components(components):
    bt = Map(components["Bt"])
    by = -np.array(bt.data)
    bp = Map(components["Bp"])
    bx = np.array(bp.data)
    br = Map(components["Br"])
    bz = np.array(br.data)
    meta_ = bt.meta

    try:
        wcs = bt.wcs
        assert meta_["CUNIT1"] == meta_["CUNIT2"] == "degree"
        center = wcs.pixel_to_world(float(meta_["CRPIX1"]) * u.pix, float(meta_["CRPIX2"]) * u.pix)
        center = center.transform_to("helioprojective")
        center = (float(center.Tx / u.arcsec), float(center.Ty / u.arcsec))
    except Exception as e:
        print("!!! WCS exception\n", e)
        raise ValueError

    data = np.stack([bx, by, bz], axis=0)

    keys = ["HARPNUM", "DATE-OBS", "BUNIT", "NOAA_ARS", "CDELT1", "CDELT2", "RSUN_OBS", "RSUN_REF"]

    meta = {k: meta_.get(k, None) for k in keys}

    # needed for potential field

    meta["ARCSEC_XC"] = center[0]
    meta["ARCSEC_YC"] = center[1]

    calculate_dx_dy(meta)

    return data, meta, meta_


def load_files(sharp_data_dir):

    HARPNUM = None
    data_type = None

    time_files = defaultdict(dict)

    fits_files = [
        f for f in os.listdir(sharp_data_dir)
        if (os.path.isfile(os.path.join(sharp_data_dir, f)) and f.endswith(".fits"))
    ]

    for f in sorted(fits_files):

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

    plot_dir = os.path.join(sharp_data_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    t_recs = [t_rec for t_rec in time_files]
    num_times = len(t_recs)

    saved_files = []

    for i, (t_rec, components) in enumerate(time_files.items()):

        # boundary condition in time
        if i == (num_times - 1):
            break

        save_basename = f"{data_type}.{HARPNUM}.{t_rec}"
        np_out_path = os.path.join(np_dir, f"{save_basename}.npz")
        plot_path = os.path.join(plot_dir, f"{save_basename}.png")

        if os.path.exists(np_out_path) and os.path.exists(plot_path):
            continue

        data, meta, meta_orig = extract_components(components)

        data_next, meta_next, meta_next_orig = extract_components(time_files[t_recs[i + 1]])

        time_start = datetime.fromisoformat(meta["DATE-OBS"])
        time_next = datetime.fromisoformat(meta_next["DATE-OBS"])

        dt = (time_next - time_start).total_seconds()

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

        b_min = np.min(data, axis=(1,2))
        b_max = np.max(data, axis=(1,2))

        print(f"Bmin:{b_min}\nBmax:{b_max}")

        v_min = np.min(V, axis=(1,2))
        v_max = np.max(V, axis=(1,2))

        print(f"Vmin:{v_min}\nVmax:{v_max}")


        np.savez(np_out_path, **result)
        saved_files.append(np_out_path)

        fig, axes = plt.subplots(3, 2, figsize=(12, 15),
                                 subplot_kw={'projection': Map(data[0], meta_orig).wcs})
        fig.suptitle(f"HARP {HARPNUM} at {meta['DATE-OBS']}", fontsize=16)

        b_names = ['Bx', 'By', 'Bz']
        v_names = ['Vx', 'Vy', 'Vz']

        # Plot magnetic field components
        for i in range(3):
            b_map = Map(data[i], meta_orig)
            im = b_map.plot(axes=axes[i, 0], title=b_names[i], 
                           vmin=-2000, 
                           vmax=2000,
                           cmap='RdBu_r')
            axes[i, 0].set_xlabel('')
            axes[i, 0].set_ylabel('')
            wcsaxes_compat.default_wcs_grid(axes[i, 0])
            plt.colorbar(im, ax=axes[i, 0], label=f"{meta.get('BUNIT', 'Gauss')}")
            
        # Plot velocity components
        for i in range(3):
            # Create new meta for velocity to change BUNIT
            v_meta = meta_orig.copy()
            v_meta['BUNIT'] = 'km/s'
            v_map = Map(V[i], v_meta)

            # Use same scale for all velocity components for better comparison
            vmax = np.nanmax(np.abs(V[i]))
            im = v_map.plot(axes=axes[i, 1], title=v_names[i],
                           vmin=-1.5, vmax=1.5,
                           cmap='PuOr_r')
            axes[i, 1].set_xlabel('')
            axes[i, 1].set_ylabel('')
            wcsaxes_compat.default_wcs_grid(axes[i, 1])
            plt.colorbar(im, ax=axes[i, 1], label='km/s')

        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()


    with open(os.path.join(sharp_data_dir, "BV_npz_files.txt"), "w") as fw:
        fw.write("\n".join(saved_files))


if __name__ == "__main__":
    load_files(sys.argv[1])

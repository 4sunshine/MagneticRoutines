import os
from sunpy.net import Fido, attrs as a
from astropy import units as u
# Time range
jsoc_email = os.environ["JSOC_EMAIL"]

timerange = a.Time('2017-09-03 09:00:00', '2017-09-06 12:00:00')
out_path = './downloads'

fido_result_path = os.path.join(out_path, "fido.txt")
os.makedirs(out_path, exist_ok=True)

if not os.path.exists(fido_result_path):
# Query using Fido
    result = Fido.search(
        timerange,
        a.jsoc.Series('hmi.B_720s'),
        a.jsoc.Notify(jsoc_email),  # Your email for notification
        a.jsoc.Segment('inclination') & a.jsoc.Segment('azimuth') & a.jsoc.Segment('disambig') & a.jsoc.Segment('field'),
        a.Sample(720 * u.s)  # 720s cadence
    )

    result = str(result)

    with open(fido_result_path, "w") as fw:
        fw.write(result)

else:
    with open(fido_result_path, "r") as fr:
        result = fr.read()

print(result)

def get_times_list(text: str, data_dir: str, prefix='hmi.b_720s'):
    times = []
    pairs = []

    def check_if_data_correct(timestamp: str):
        data_types = ['field.fits', 'azimuth.fits', 'disambig.fits', 'inclination.fits']
        return all(
            os.path.exists(
                os.path.join(data_dir, ".".join([prefix, timestamp, d_type]))
            ) for d_type in data_types
        )

    prev_time = None
    for l in text.splitlines():
        if ("SDO/HMI" in l) and ("MISSING" not in l):
            parts = l.split()
            time = parts[0]
            time = time.replace(".", "").replace(":", "")
            if not check_if_data_correct(time):
                continue
            times.append(time)
            if prev_time is not None:
                pairs.append(" ".join([prev_time, time]))
            prev_time = time

    times = "\n".join(times)
    pairs = "\n".join(pairs)

    return times, pairs

times, pairs = get_times_list(result, out_path)

with open(os.path.join(out_path, "timestamps.txt"), "w") as fw:
    fw.write(times)

with open(os.path.join(out_path, "times_paired.txt"), "w") as fw:
    fw.write(pairs)

# fetch the data:
# files = Fido.fetch(result, path=out_path + '/{file}')

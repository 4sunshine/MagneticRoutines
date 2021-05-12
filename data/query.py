from sunpy.net import attrs as a
from sunpy.net import Fido

tstart = '2013/04/12 19:00:56'

tend = '2013/04/12 20:30:29'

event_type = 'FL'

result = Fido.search(a.Time(tstart,tend), a.hek.EventType(event_type))

[print(r) for r in result]
print(result.keys())

print(result['hek']['event_coord1'])
print(result['hek']['event_coord2'])
print(result['hek']['event_coord3'])
print(result['hek']['event_title'])
print(result['hek'].keys())

from paraview.simple import *

renderView1 = GetActiveViewOrCreate('RenderView')

all_slices = [(k[0], s) for k, s in GetSources().items() if s.__class__.__name__ == "SliceWithPlane"]
all_proj = []
for name, plane_source in all_slices:
    plane_index = name.replace("SliceWithPlane", "")
    assert plane_index.isnumeric()
    proj_name = f"ProgrammableFilter{plane_index}"
    proj = FindSource(proj_name)
    if proj is None:
        proj = ProgrammableFilter(registrationName=proj_name, Input=plane_source)
    all_proj.append(proj)
    all_proj[-1].Script = f"""field = inputs[0]

def get_normal_proj(normal, key):
    field_n = normal[0] * field.PointData[key][..., 0] + \
              normal[1] * field.PointData[key][..., 1] + \
              normal[2] * field.PointData[key][..., 2]
    return field_n

def get_abs_tangent_proj(normal_proj, key):
    return (mag(field.PointData[key])**2 - normal_proj**2)**0.5

n_x = {plane_source.PlaneType.Normal[0]}
n_y = {plane_source.PlaneType.Normal[1]}
n_z = {plane_source.PlaneType.Normal[2]}

n = [n_x, n_y, n_z]
j_n = get_normal_proj(n, "j")
B_n = get_normal_proj(n, "B")

j_t = get_abs_tangent_proj(j_n, "j")
B_t = get_abs_tangent_proj(B_n, "B")

output.PointData.append(j_n, "j_n")
output.PointData.append(j_t, "j_t")
output.PointData.append(B_n, "B_n")
output.PointData.append(B_t, "B_t")
"""

    projDisplay = Show(all_proj[-1], renderView1, 'UniformGridRepresentation')

renderView1.Update()

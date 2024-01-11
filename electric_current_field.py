field = inputs[0]
# dx, km
dx = 400.0
# speed of light
c = 299792.458
# j = B * c / 4 / Pi / dx
j_multiplier = c/4/math.pi/dx
# J calculation
curl_data = j_multiplier*curl(field.PointData["B nlfffe"])
output.PointData.append(curl_data, "j")
output.PointData.append(field.PointData["B nlfffe"], "B")

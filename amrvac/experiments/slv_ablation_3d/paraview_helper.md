# Physical Units Conversion in ParaView

This note documents the **Calculator filter expressions** used to convert **dimensionless MHD variables** into **physical units**.

The normalization assumes:

* Velocity unit = **Alfvén speed**
* Magnetic field unit from pressure balance
* Current density derived consistently from MHD units

---

## Velocity → km/s

**Physical meaning:**
Converts normalized velocity components `(v1, v2, v3)` to **km/s**.

**Conversion factor:**
[
v_0 = 116.45\ \mathrm{km/s}
]

**Calculator filter expression:**

```text
116.45 * (iHat*v1 + jHat*v2 + kHat*v3)
```

**Output:** Vector velocity in **km/s**

---

## Current Density → A/km²

**Physical meaning:**
Converts normalized current density components `(j1, j2, j3)` to **A/km²**.

**Conversion factor:**
[
J_0 = 15.89619\ \mathrm{A/km^2}
]

**Calculator filter expression:**

```text
15.89619 * (iHat*j1 + jHat*j2 + kHat*j3)
```

**Output:** Vector current density in **A/km²**

---

## Magnetic Field → Gauss

**Physical meaning:**
Converts normalized magnetic field components `(b1, b2, b3)` to **Gauss**.

**Conversion factor:**
[
B_0 = 1.99757357615242\ \mathrm{G}
]

**Calculator filter expression:**

```text
1.99757357615242 * (iHat*b1 + jHat*b2 + kHat*b3)
```

**Output:** Vector magnetic field in **Gauss**

## Simulation Time Annotation

This note documents the **Annotate Time** filter configuration used to display
physical simulation time in **hours**.

### Annotate Time → Format

```text
t = {time:.2f} h
```

---

## Notes

* These expressions assume **cell-centered vector components** (`v1`, `j1`, `b1`, etc.).
* Units are consistent with **Alfvénic normalization**:
  [
  v_0 = \frac{B_0}{\sqrt{\mu_0 \rho_0}}
  ]
* Suitable for **solar coronal MHD**, flux ropes, and rotating footpoint simulations.

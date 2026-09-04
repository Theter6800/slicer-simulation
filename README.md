# Solar/LED-to-Optical-Fiber Image Slicer Simulator

A complete interactive 3D geometric-optics simulator and Streamlit application designed for solar and LED light coupling into a multimode optical fiber using **Image Slicer Integral Field Unit (IFU)** principles.

Inspired by Ellen Lee's work (*“Sequential and non-sequential Zemax Dynamic Link Libraries for generating image slicer integral field units”*, JATIS 2026), this simulator adapts image slicing to solve a critical non-astronomical engineering problem: **maximizing optical power coupled into a multimode fiber**.

---

## 1. Quickstart & Installation

### Requirements
- Python 3.11+ (tested on Python 3.9+)
- NumPy, SciPy, Pandas, Plotly, Streamlit, Pytest

### Setup Instructions
```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run automated physics tests
pytest -v tests/

# Launch the interactive Streamlit application
streamlit run app.py
```

---

## 2. Coordinate System & Conventions

The simulator uses a right-handed Cartesian coordinate bench convention:
- **$z$**: Nominal optical bench axis / propagation direction (mm)
- **$x$**: Horizontal transverse coordinate (mm)
- **$y$**: Vertical transverse coordinate (mm)

All geometry, dimensions, and distances are internally computed in **millimeters (mm)**. User angles entered in degrees ($^\circ$) are converted internally to radians.

### Ray Representation
Every ray bundle is stored as a vectorized data structure:
- **Position** $\mathbf{r} = [x, y, z]$ (mm)
- **Unit Direction** $\mathbf{k} = [k_x, k_y, k_z]$ with $\|\mathbf{k}\| = 1$
- **Optical Weight / Power** $P_i$ (normalized $\sum P_i = 1.0$)
- **Slicer Channel ID** ($-1$ for unassigned, $0, 1, 2, \dots$ for slice channels)
- **Ray Status**: `ACTIVE`, `CLIPPED`, `MISSED`, `ACCEPTED_BY_FIBER`, `REJECTED_BY_POSITION`, `REJECTED_BY_NA`, `REJECTED_BY_BOTH`

---

## 3. Optical Physics & Governing Equations

### A. Specular Reflection Law
For an incident unit ray $\mathbf{k}_{in}$ and surface normal $\mathbf{n}$ pointing towards the incident beam:
$$\mathbf{k}_{out} = \mathbf{k}_{in} - 2(\mathbf{k}_{in} \cdot \mathbf{n})\mathbf{n}$$

Rotating a flat mirror by $\delta$ alters the reflected chief ray by $2\delta$.

### B. Paraxial Thin-Lens Transformation
For a thin lens at $z_{lens}$ with focal length $f$ and clear aperture diameter $D_{lens}$, paraxial slopes $u = k_x / k_z$ and $v = k_y / k_z$ relative to lens center $(x_c, y_c)$ transform as:
$$u_{out} = u_{in} - \frac{x - x_c}{f}, \quad v_{out} = v_{in} - \frac{y - y_c}{f}$$
$$\mathbf{k}_{out} = \frac{[u_{out}, v_{out}, 1]}{\sqrt{u_{out}^2 + v_{out}^2 + 1}} \times \text{sign}(k_z)$$

### C. Fiber Coupling Acceptance Metric
The receiver fiber has:
- Core diameter $D_{core} = 1.0\text{ mm}$ (radius $r_{core} = 0.5\text{ mm}$)
- Numerical aperture $\text{NA} = 0.22$
- External medium: Air ($n_{ext} = 1.0$)

A ray arriving at the fiber face is accepted if and only if **BOTH** conditions hold:
1. **Spatial Acceptance:**
   $$\sqrt{(x - x_{fiber})^2 + (y - y_{fiber})^2} \le 0.5\text{ mm}$$
2. **Angular Acceptance:**
   $$n_{ext} \sin(\theta) \le \text{NA} \implies \theta \le \arcsin(0.22) \approx 12.71^\circ$$
   where $\cos\theta = \frac{\mathbf{k} \cdot \mathbf{axis}_{fiber}}{\|\mathbf{k}\| \|\mathbf{axis}_{fiber}\|}$.

Rays are categorized into 4 distinct groups:
- **GREEN (Accepted):** Position OK AND NA OK
- **ORANGE (NA Rejected):** Position OK ($r \le 0.5\text{ mm}$) but NA rejected ($\theta > 12.71^\circ$)
- **BLUE (Spatial Rejected):** NA OK ($\theta \le 12.71^\circ$) but outside core ($r > 0.5\text{ mm}$)
- **RED (Both Rejected):** Outside core AND exceeds NA

### D. Image Slicer & Downstream Pupil Relay
In classical IFUs (Lee 2026):
1. An image slicer array is positioned at an input focal plane, spatially sectioning the beam.
2. Slicer mirror tilts $(\theta_x, \theta_y)$ impart angular shears that separate the reflected channel chief rays.
3. At the downstream conjugate pupil planes, distinct channel pupil images form with minimal transverse waist.
4. Per-channel pupil mirrors redirect and refocus the channels into an optimized reformatted footprint matching the fiber core.

---

## 4. Étendue & Phase-Space Conservation

The simulator enforces thermodynamic sanity: an image slicer **cannot defeat étendue conservation**:
- **Fiber Étendue:**
  $$G_{fiber} \approx A_{fiber} \cdot \pi \cdot \text{NA}^2 = \pi^2 \left(\frac{D_{core}}{2}\right)^2 \text{NA}^2 \approx 0.1192\text{ mm}^2\cdot\text{sr}$$
- **Sun Étendue:**
  $$G_{sun} \approx A_{aperture} \cdot \Omega_{sun} = \pi \left(\frac{D_{ap}}{2}\right)^2 \cdot \pi \sin^2(0.266^\circ)$$

If $G_{fiber} < G_{source}$, maximum passive concentration is thermodynamically bounded by $G_{fiber} / G_{source}$. The simulator flags this with a prominent thermodynamic warning.

---

## 5. Lab Hardware Presets & Experiments

The simulator incorporates real laboratory optical components:
- Available positive lenses: $f = 100\text{ mm}$, $f = 35\text{ mm}$, $f = 75\text{ mm}$
- Multimode fiber: $1.0\text{ mm}$ core, $\text{NA} = 0.22$
- Slicer mirrors: four $\approx 10\times 10\text{ mm}$ flat mirrors (with $0.5\text{ mm}$ gaps)

> **Important:** Unknown optical bench distances and source dimensions are labeled as *placeholder / user input required* and are fully editable in the UI.

### Architecture Presets
- **Preset 1: Old Lens-Only System:**
  $\text{LED} \to \text{Aperture} \to \text{L1 } (f=100) \to \text{L2 } (f=35) \to \text{L3 } (f=75) \to \text{Fiber}$
- **Preset 2: Slicer + Common Condenser:**
  $\text{LED} \to \text{Aperture} \to \text{Fore-optics } (f=100, f=35) \to \text{Slicer } (2\times 2) \to \text{Condenser } (f=75) \to \text{Fiber}$
- **Preset 3: Paper-Inspired Slicer IFU:**
  $\text{LED} \to \text{Aperture} \to \text{Fore-optics} \to \text{Slicer} \to \text{Per-Channel Pupil Mirrors} \to \text{Coupling Optic } (f=75) \to \text{Fiber}$
- **First Default Experiment:**
  A 4-channel slicer setup with interactive buttons to enable Slice 1, Slice 2, Slice 3, Slice 4 to observe whether coupled power increases incrementally.

---

## 6. Scientific Answer: Does Adding a Slicer Improve Coupling?

The simulator quantitatively answers this question:
- If the optical source étendue is significantly larger than the fiber étendue ($G_{source} \gg G_{fiber}$), a standard lens focusing an extended image forms an optical spot where rays at the edge of the core already exceed the numerical aperture ($\text{NA} = 0.22$), or the focal spot severely overfills the $1.0\text{ mm}$ core.
- By slicing the extended image into field slices and re-imaging the pupils, the image slicer reformats the aspect ratio of the phase-space volume, directing more rays into the acceptable spatial-angular zone ($r \le 0.5\text{ mm}, \theta \le 12.71^\circ$).
- In our benchmark, the **Paper-Inspired Slicer IFU** and **Slicer + Condenser** increase the coupled optical fraction compared to the unoptimized 3-lens bench, demonstrating a clear quantitative advantage when properly aligned!

---

## 7. Streamlit UI Architecture (9 Tabs)

1. **System Setup:** Complete parameter controls for source, aperture, lenses, slicer, pupil mirrors, and fiber; JSON save/load.
2. **Ray Trace:** 2D X-Z side view, 2D Y-Z top view, and interactive 3D ray trace colored by channel or acceptance status.
3. **Slicer Plane:** Mirror tile boundaries, inter-slice gap losses, and the "Scan Beam Size vs $z$" tool.
4. **Pupil Plane:** Channel pupil waist scanner, transverse pupil footprints, and mechanical separation check.
5. **Fiber Coupling:** Fiber face spot diagram (Green/Orange/Blue/Red), angular acceptance ($\theta$ vs $r$) phase-space plot, per-channel efficiency bar chart, and loss budget table.
6. **Compare Architectures:** Direct side-by-side run of Presets 1, 2, and 3 under identical illumination.
7. **Optimization:** Analytic reflection bisector slicer aiming and SciPy Nelder-Mead tilt & fiber position optimization.
8. **Parameter Sweeps:** 1D and 2D parametric sensitivity sweeps (e.g. Slicer $z$, Fiber $z$, NA, Core Diameter).
9. **Validation & Étendue:** Live in-app pytest test runner, first-order étendue phase-space checker, and CSV ray/loss data export.

---

## 8. Verification & Test Suite

All 7 required physical laws are validated with automated unit tests:
```bash
pytest -v tests/
```
- `test_reflection.py`: Angle of incidence = Angle of reflection; $2\delta$ tilt ray deflection.
- `test_lens.py`: Parallel on-axis rays focus at $z = z_0 + f$; Paraxial slope alteration $u_{out} = u_{in} - x/f$.
- `test_fiber.py`: Acceptance at $0^\circ$, rejection at $13^\circ$ (for $\text{NA} = 0.22$), rejection at $r > 0.5\text{ mm}$.
- `test_energy.py`: Optical power conservation ($\sum P_{active} + \sum P_{clipped} = \sum P_{launched}$) across all stages.
- `test_slicer.py`: Slicer channel ID stamping, downstream persistence, and gap loss clipping.

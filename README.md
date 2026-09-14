# Digital Fly 数字果蝇

English · [中文](README.zh-CN.md)

A real whole-CNS fly connectome driving a real fly body:
**sensing → spiking network → descending commands → muscles → physics → feedback**.

![split screen](docs/images/web_split.jpg)

Left: the fly body in MuJoCo. Right: all 185,348 neurons at the same instant,
white dots are the ones spiking.

---

## Where the data comes from

| Source | What it is |
|---|---|
| **MaleCNS v1.0** (HHMI Janelia FlyEM · Cambridge · MRC LMB · Google Research, *Cell* 2026) | Complete male CNS wiring diagram: 176,422 neurons, ~125M synapses, central brain + optic lobes + VNC with the neck connective intact. CC-BY 4.0 |
| **flybody** (Google DeepMind × HHMI Janelia, *Nature* 643, 2025) | Anatomically detailed MuJoCo fly body — six legs, wings, proboscis |
| **NeuroMechFly v2 / flygym** (NeLy-EPFL) | Measured fly step kinematics |

The built network: **185,348 nodes, 26,006,173 synaptic edges**. Synapse **sign comes
from the presynaptic neuron's predicted transmitter**: ACh +1; GABA / glutamate /
histamine −1; monoamines +1 by default (`--modulator-sign 0` for the control).

---

## Running it

```bash
bash setup.sh && conda activate digitalfly
python cli.py download          # three feather files, ~1.2 GB
python cli.py build             # build the signed sparse matrix
python cli.py calibrate         # calibrate synaptic strength (required, see below)
python cli.py web --port 8080   # open http://localhost:8080
```

---

## How the brain runs

Leaky integrate-and-fire, parameters from Shiu et al. (*Nature* 634, 2024):

```
rest −52 mV · threshold −45 mV · τ_m 20 ms · refractory 2.2 ms · dt 0.5 ms
```

**Synaptic strength has to be calibrated.** The literature's 0.275 mV unitary EPSP
applied to this connectome gives a network-wide seizure (48 Hz). `calibrate`
bisects for the non-self-sustaining point and lands on **0.0341 mV**. Every
experiment runs at that operating point.

**Sensory input is rate-coded**, not sustained current. LIF has no adaptation, so a
constant current saturates the network (pinned at 11 Hz regardless of strength).
Injecting a spike with probability `p = rate × dt` per step makes input strength
actually graded.

Backends auto-degrade: `torch-cuda` → `torch-cpu` → `scipy`.

---

## The one limit that shapes everything

> **At the calibrated operating point this reconstruction reliably transmits about
> one synapse.**

Not a tuning problem — a structural fact, measured repeatedly. It dictates every
design decision below:

| Gets through (one synapse) | Does not (two or more) |
|---|---|
| retina → medulla (85.8% linearly decodable) | medulla → VPN → mushroom body |
| ALPN → Kenyon cells (22,586 connections) | retina → medulla → LC (**LC4+LC12, 624 cells, not one spikes**) |
| taste GRN → proboscis motor neuron MN9 | bilateral odour → descending asymmetry |

So wherever something has to be read out, the read-out layer is forced to sit one
synapse from the input.

---

## Behaviours

```bash
python cli.py behave --mode walk   --seconds 4
python cli.py behave --mode forage --seconds 6
python cli.py behave --mode flight --seconds 3
```

**Walking.** Descending neuron (DN) population rate is read out as two commands,
speed and turn, which drive a gait generator built from measured kinematics. DNs do
not drive joints directly — tried it, the fly collapses; the connectome gives
neuron identity, not its quantitative mapping onto MuJoCo actuators. Measured
**1.31 body lengths/s**.

**Foraging.** Odour concentration is injected per antenna, the fly approaches the
sugar, and on contact extends its proboscis — **that last part is computed by the
connectome**: sugar GRNs drive MN9 at 16.7 Hz, bitter at 0.0 Hz.

**Flight.** Free flight tumbles — 150 parameter sets searched, all of them. So it's
**tethered flight** (the tether rod is visible in frame), showing wing kinematics
and the steering read-out. Body pitch 47.5°.

![walk](docs/images/walk.jpg) ![flight](docs/images/flight.jpg)

---

## Rubik's cube

![cube](docs/images/cube_legs.jpg)

The fly walks up to the cube, stops facing it, and turns the face with a **front
leg (T1)**; the face rotates with the stroke. The division of labour:

| Part | Decided by |
|---|---|
| **When to turn** | the connectome — DN population rate integrates to a threshold; a busier network turns faster |
| **Which face** | an IDA\* solver (corner orientation 3⁷ / edge orientation 2¹¹ / corner permutation 8! pruning tables) |
| **The leg stroke** | kinematic animation. Leg pose and face angle share one phase; it is not friction-driven contact |

Six DN subgroups also vote on the face. Their agreement with the solver is shown
live: **~17%, indistinguishable from chance**. The connectome sets the rhythm but
does not encode which face — a negative result, reported as such.

---

## Character recognition

The hardest part of the project; thrown away twice. The final design:

### The decision happens inside the connectome

```
image ──► 892 hexagonal columns ──► L1 (ON) / L2 (OFF), 1,785 real lamina neurons
                                      │  ★ 12,710 plastic synapses — already in the connectome
                                      ▼
                            Dm6 / Dm19 / Dm17 / Dm15 / Dm1
                            665 wide-field amacrine cells, split into 10 groups
                                      ▼
                              whichever group fires most is the answer
```

**The classifier's parameters are entries in `W` itself.** No second set of weights,
no backpropagation.

![retina mapping](docs/images/retina.png)

*Each pair: original digit, then the same digit reconstructed from what the 892
hexagonal columns sample — the mapping is faithful.*

### Why these neurons

The decision layer uses wide-field Dm amacrine cells. Two measured reasons:

| | Dm17 | Dm19 | Dm6 | Tm1 (first attempt) |
|---|---|---|---|---|
| columns pooled per cell | **204** | **130** | **44.5** | **0.6** |
| firing rate | 90.9 Hz | 113.8 Hz | 48.5 Hz | 2.2 Hz |
| modulation across characters | 20% | 13% | 19% | 8% |

Tm1, a single-column cell, averages 0.6 synapses from L1/L2 per cell — most of them
have no plastic input at all. Worse, splitting them into 10 groups gives each group
one tenth of the visual field, so the groups compare different parts of the image.
**A decision about the whole image needs neurons that pool the whole field.**

Biologically the fly's actual visual decision neurons are LCs (lobula columnar,
Wu et al. *eLife* 2016). Unusable here: **LC4 + LC12, 624 cells, not one of them
spikes** at the calibrated operating point — they sit two synapses away.

### How it learns

Same rule as the mushroom body: **teaching-signal-gated bidirectional plasticity**
(Hige et al. *Neuron* 2015; Cohn et al. *Cell* 2015; Aso & Rubin *eLife* 2016).

```
present 600 ms ──► group rates ──► argmax gives the answer
                                     │
              wrong ──► potentiate active synapses onto the correct group
                        depress active synapses onto the wrongly-winning group
              right ──► slow recovery
```

Four constraints. Drop any one and it stops learning; each was forced by measurement:

1. **Must be bidirectional.** Depression-only locks weights into [0, w0], so the
   correct class can never exceed baseline and the system only contracts — 4-class
   stalled at 42.5% with total synaptic strength down to 0.87 and still falling.
2. **Presynaptic activity must be baseline-normalised.** 80% of a handwritten image
   is black, so the OFF channel is fully active for every character; without
   normalisation the "currently active" gate selects nothing and depression
   degenerates into undifferentiated global decay.
3. **Synaptic scaling is required** (Turrigiano 1998). Unconstrained, potentiation
   wins and total strength climbs to 1.378, letting global drift dominate the
   decision. Rescaling each group back to its original total preserves the learned
   relative weights.
4. **Long enough gaze.** At 240 ms the signal-to-noise ratio is 1.45 — noise as
   large as signal. At 600 ms it is 4.05. Past 1500 ms there is no further gain.

### The result, stated plainly

| | Accuracy | Chance |
|---|---|---|
| **Linearly decodable** ceiling of medulla activity | **85.8%** (pixel baseline 86.9%, shuffled 9.6%) | 10% |
| **In-brain decision**, 4 classes (one epoch) | 14.6% → **52.1%** | 25% |
| **In-brain decision**, 10 classes (14 epochs) | 14.2% → **36.7%**, 45% on train | 10% |

The 10-class 36.7% is not uniform — **some digits are learned, others not at all**
(12 test images each): 0 scores 10/12, 1 and 4 score 8/12; 2, 5 and 7 score 0/12.
Zero becomes an attractor and 7 is read as 9 almost every time. That is what a
capacity-limited decision layer looks like: 665 neurons split ten ways is 66 cells
and ~1,270 plastic synapses per class.

**The information really is in the brain** — the same medulla activity reads out at
85.8% linearly, 99% of the pixel baseline. The in-brain rule reaches 36.7%. Those 49
points are the price of using the fly's own synapses and the fly's own learning rule,
reported as measured.

Going further is limited by decision-layer capacity, not by the rule. The right cells
are the LCs (624 cells, strongly convergent) — and none of them spike at the
calibrated operating point. Same one-synapse wall; more epochs do not get around it.

What the three iterations cost:

| Version | 4-class | 10-class | What was wrong |
|---|---|---|---|
| Depression-only + Tm1 single-column | 42.5% | — | weights locked in [0, w0], monotone contraction; 0.6 plastic inputs per decision neuron |
| Bidirectional + wide-field Dm | **52.1%** | 13–23%, oscillating | potentiation ran away, total strength up to 1.378 |
| Bidirectional + Dm + synaptic scaling | — | **36.7%** | strength pinned at 1.000, rises steadily to plateau |

The web console lets you draw with the mouse, hit recognise, and watch which neurons
light up during recognition.

![brain during recognition](docs/images/recognize_brain.jpg)

---

## Mushroom body associative learning

The one place where learning happens **entirely inside the connectome** and works
solidly: 61,210 real KC→MBON synapses are modified.

```
odour A + dopamine (the shock) × 10 trials
  → paired odour A in the punished compartment  −39.0%
  → unpaired odour B                             +7.8%
  → 5.0× specificity, with a valence flip
```

Two model additions, both flagged: **sparse coding is imposed explicitly** (APL has
only 2 neurons, which cannot suppress 640k recurrent KC→KC excitatory connections —
measured, any input lights all 4,064 KCs to 145 Hz); and **sparsification is relative
to each KC's own baseline** (without it, two odours' KC codes overlap 73%, higher
than the 25% input overlap; with it, 0%).

Vision cannot use this route: visual input to KCs is 1,616 connections, 1/35th of the
olfactory pathway; even amplified 25× digit separability stays at 1.02.

---

## Three reference projects

The same author (nftechie) built three things on the same MaleCNS v1.0 connectome.
All three are useful controls.

**[doomfly](https://github.com/nftechie/doomfly)** — the connectome plays Doom. The
closest comparison:

| | doomfly | this project |
|---|---|---|
| Visual input | R1–R6 photoreceptors (3,335) + R8 (811) | L1/L2 driven directly (1,785) |
| Plastic synapses | 4,184 KC→MBON11, PPL101-gated | 61,210 KC→MBON (olfactory) / 12,710 L1/L2→Dm (visual) |
| Weight bounds | 0.1–2× original, η=0.001 | 0–3×, η=0.1 |
| Learning outcome | visual, conditioning and survival gates **all failed** | 10-class 36.7% vs 10% chance |

Two independent corroborations:

* **KC→MBON11 edge count is 4,184 in both.** Two separately written importers with
  different retention policies land on the same number for the same named pathway —
  stronger evidence than either project's own self-check.
* Their review found that "incoming synaptic signals were retained during refractory
  periods, unlike the stated reference equations in Brian2" and fixed it. This project
  independently hit and fixed the same bug (a refractory off-by-one between the scipy
  and torch backends).

Adopted from it: the conditioning experiment's punished compartment is now the real
named **MBON11** rather than an arbitrary half of all MBONs, taught by the two named
PPL101 cells. Measured here, PPL101→MBON11 carries weight 2,311 while the runner-up
MBON30 gets 101 (23× less); the control compartment is MBON09, which has comparable
KC input (4,682 vs 4,184 edges) but receives essentially no PPL101.

**[flm](https://github.com/nftechie/flm)** — the connectome as a frozen reservoir
feeding a language model. Reproduced below.

**Others**: [fly-brain-olympiad](https://github.com/TomkeMonke/fly-brain-olympiad)
(brain never trained; linear readout of every neuron),
[classi-fly](https://github.com/bhodgens/classi-fly) (mushroom body as a fixed random
projection plus linear readout; the author reports a negative result), and
[fly-brain-bench](https://github.com/RaphaelSR/fly-brain-bench) (APL-style divisive
inhibition restores sparseness but pattern overlap only falls to ~70%; this project's
per-KC baseline normalisation takes it to 0%).

---

## FLM: wiring the connectome to a language model

```bash
python cli.py flm-download          # fetch the frozen LFM2.5-1.2B (via hf-mirror)
python cli.py sim --exp flm         # train the adapter + two controls
python cli.py flm-chat --prompt "What does a fly see?"
```

Token embeddings drive the whole graph through `x = tanh(W·(0.6x + 0.4·in))`; the
graph's state produces a **bounded correction** (max ±0.5) to the language model's
logits. Only the adapter trains; the connectome and the language model stay frozen.

**This layer is not a spiking simulation** — abstract numerical state, no transmitter
signs, no dopamine, no biological time.

The controls are the whole point; the adapter alone has 3.19M parameters:

| | Test NLL | Gain vs baseline | Parameters |
|---|---|---|---|
| Language model baseline | 7.4623 | — | — |
| **fly** (connectome reservoir) | 6.9068 | +0.5555 | 3,194,928 |
| **shuffled** (edges permuted, degrees kept) | 6.9087 | +0.5536 | 3,194,928 |
| **direct** (no reservoir) | **6.8262** | **+0.6361** | 3,194,928 |

**Fly vs shuffled differs by 0.0019 — no difference; direct input beats fly by 0.0806.**
The reservoir does not help; it destroys information. This matches the original
author's finding: **no evidence that fly anatomy helps language modelling.** All the
language ability comes from the pretrained model.

---

## Web console

```bash
python cli.py web --host 0.0.0.0 --port 8080
```

- **Free mouse-drag 3D view**: drag to rotate, wheel to zoom, double-click to reset
- Switch walk / forage / tethered flight / cube live
- Inject stimuli, ablate neuron groups, take over speed and turn, move the sugar
- Whole-brain activity map, spike raster, firing-rate time series
- Mouse handwriting + recognition + feedback

**Three threads, one job each**: the brain integrates at full speed and emits
speed/turn commands; physics advances by real elapsed time; rendering runs at its own
cadence.

| | Before | After |
|---|---|---|
| Frame rate | 3 FPS | **30.2 FPS** |
| Worst frame gap | — | 44 ms |
| Jitter (sd) | 20 ms | **8 ms** |
| Body | 0.29× real time | **0.95×** |
| Brain | 0.14× | 0.32× |

Three threads plus two details; miss any one and it still stutters:

1. **Physics and rendering cannot share a thread.** 51 ms of physics + 12.5 ms of
   rendering per iteration caps you at 16 FPS and slows the physics too.
2. **MJPEG must be producer-driven.** Polling `sim.frame` every 50 ms is unsynchronised
   with the producer — frames get duplicated and dropped. Average FPS looks fine; it
   looks jerky.
3. **Physics advances 20 ms at a time.** The physics thread holds the body lock for
   that whole span and the renderer waits on it. At a 60 ms cap one hold is 51 ms and
   the worst frame gap measured 136 ms; at 20 ms it drops to 44 ms.

The cost, stated: in interactive mode brain time runs ~3× slower than body time, and
the physics timestep is widened from 0.1 ms to 0.4 ms (measured stable — 1.31 vs 1.07
body lengths/s; at 0.8 ms contacts fail). For strict 1:1, render offline with `behave`.

### Deploy as a service

```bash
sudo cp deploy/digitalfly.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now digitalfly
```

Starts at boot, restarts on crash, logs to `/var/log/digitalfly.log`.

---

## Commands

```bash
python cli.py download                 # fetch the connectome
python cli.py build                    # build the network
python cli.py calibrate                # calibrate synaptic strength
python cli.py doctor                   # self-check + cross-check vs neuprint public API
python cli.py sim --exp sugar_pe       # sugar → proboscis extension (positive)
python cli.py sim --exp ablation       # ablation reveals feed-forward inhibition
python cli.py sim --exp steering       # odour laterality (negative)
python cli.py sim --exp cube_decode    # next cube move (negative)
python cli.py sim --exp vision         # visual decoding (positive)
python cli.py sim --exp conditioning   # olfactory learning (MBON11 / PPL101)
python cli.py sim --exp flm            # connectome + language model, with controls
python cli.py flm-download             # fetch the frozen language model
python cli.py flm-chat                 # talk to the fly
python cli.py train-vision             # train the handwriting read-out
python cli.py behave --mode walk|forage|flight
python cli.py web --port 8080
pytest tests/                          # 74 tests
```

---

## Layout

```
digitalfly/
  connectome.py    feather → signed CSR sparse graph
  brain.py         LIF engine (torch-cuda / torch-cpu / scipy)
  calibrate.py     synaptic strength calibration
  neurons.py       query by type / class / ROI / side
  body.py          flybody MuJoCo wrapper + EGL offscreen render + free camera
  bridge.py        DN → commands; proprioception / vision / taste → injected current
  locomotion.py    measured-kinematics gait, wingbeat
  vision.py        image presentation in hexagonal ommatidial coordinates
  visual_brain.py  in-brain visual decision + bidirectional plasticity + scaling
  handwriting.py   MNIST training + linear decodability ceiling
  plasticity.py    mushroom body: dopamine-gated KC→MBON plasticity
  flm.py           connectome as frozen reservoir + language-model adapter
  cube.py          cube state + IDA* solver
  cube_scene.py    26 mocap cubies
  cube_hands.py    front-leg cube manipulation
  brainview.py     whole-brain 3D point cloud (real soma coordinates)
  experiments/     sugar_pe · ablation · steering · cube_decode
                   vision_decode · conditioning · flm_train
  web/             Flask + SSE + MJPEG console
```

---

## What is data and what is engineering

**Data-driven**: the wiring itself, synapse signs (from transmitter predictions),
neuron identity and regions, LIF dynamics, the location and count of plastic
synapses, hexagonal column coordinates.

**Engineering approximations, all flagged in code**: the DN → MuJoCo actuator mapping
(the connectome gives identity, not quantitative mechanics); mushroom body sparse
coding (doing APL's job for it); which decision group means which character
(arbitrary — as in the real fly, where MBON valence is set by which DAN teaches it);
the front-leg cube animation; bilateral odour comparison for foraging turns.

## Citations

- Janelia FlyEM et al., MaleCNS v1.0 connectome, *Cell* (2026)
- Vaxenburg et al., flybody, *Nature* 643 (2025)
- Shiu et al., whole-brain connectome LIF simulation, *Nature* 634 (2024)
- Hige et al., *Neuron* (2015); Cohn et al., *Cell* (2015); Aso & Rubin, *eLife* (2016)
- Wu et al., LC neurons and behaviour, *eLife* (2016)
- Turrigiano et al., synaptic scaling, *Nature* (1998)

# Fixture provenance

All fixtures are from Wikimedia Commons and depict **public figures** (see ETHICS.md,
demo-subject policy). Downloaded 2026-09-01 by `spikes/spike_face.py`.

| file | subject | source (Wikimedia Commons) | note |
|---|---|---|---|
| `A1_pichai_hanoi2015.jpg` | (audience, not Pichai) | Meet Google CEO Sundar Pichai @Hanoi, Vietnam (23291242774) | **Negative control.** Detector finds 2 faces, both blurred out-of-focus audience members at ~1% frame area. Retained deliberately as the quality-gate test case. |
| `A2_pichai_warsaw2022.jpg` | Morawiecki (f0) + Pichai (f1) | Mateusz Morawiecki spotkał się z CEO Google Sundar Pichai w KPRM (2022.03.29) 01 | Two-face group photo; defeats a "largest face" heuristic. |
| `A3_pichai_warsaw_crop.jpg` | Pichai | ...(2022.03.29) 01 (cropped) | Crop of A2 -> expected near-duplicate (cos 0.9766 vs A2 f1). |
| `B1_nadella_2017.jpg` | Nadella | MS-Exec-Nadella-Satya-2017-08-31-22-2 | Same-subject pair with B2. |
| `B2_nadella_smiling.jpg` | Nadella | Satya smiling-print | Genuinely different photograph from B1 (cos 0.7208). |

# 4DGS Holograms

Playback-only WebGPU / WebXR demo for **[Faltwerk](https://github.com/edrethardo/Faltwerk)** (CurveStream-GS): decoder, four P44 clips, and the MR player (carousel, world-space colour desk, planar floor shadow, hand occluders).

This repo is a convenience slice that shows how to isolate playback outside the training and encoding pipeline. The codec, format docs, and research code live in the main repository:

**https://github.com/edrethardo/Faltwerk**

## Authors

- **James Kane** ([jameskane05](https://github.com/jameskane05))
- **Aaron Nowak** ([edrethardo](https://github.com/edrethardo))

## Live

- Desktop: https://jameskane05.github.io/4dgs-holograms/
- Quest 3 (MR profile): https://jameskane05.github.io/4dgs-holograms/?quest=1

Quest Browser 146+, enable **WebXR experiments** in `chrome://flags`, then hard-reload. Tap **MR**, place the hologram, double-pinch to show/hide the colour desk.

## Clips

| Scene | Take |
| --- | --- |
| `scene_angry_j.json` | `james/angry_j_p44` |
| `scene_female.json` | `james/female_p44` |
| `scene_male_1.json` | `james/male_1_p44` |
| `scene_male_2.json` | `james/male_2_p44` |

## Local preview

```bash
npx --yes serve -p 8790 .
# open http://127.0.0.1:8790/
```

HTTPS is required for WebXR on a headset (GitHub Pages already provides it).

## Refresh from the codec checkout

From `4dgs_codec` after `npm run build` in `web/`:

- copy `web/dist/{main,worker}.js`
- keep this repo’s trimmed `index.html` (landing + credits; player-only HUD)
- copy `web/hands/*.glb` → `hands/`
- copy the four `*_p44` section folders and the four `scene_*.json` + `scenes.json` (UTF-8 **without** BOM)

# Foresight visuals

One place for interactive diagrams and charts.

## Browse

From the foresight machine (server on port **8765**):

- Index: http://10.125.145.129:8765/
- Or open any `viz/**/*.html` in a browser

Serve locally:

```bash
cd viz && python3 -m http.server 8765 --bind 0.0.0.0
```

## Layout

Each diagram gets its own dated folder:

```
viz/<type>-<slug>-<YYYYMMDD-HHMMSS>/
  candidate.json   # Archify source (when used)
  *.html           # open this
```

`.archify` is a symlink to `viz/`, so Archify’s default output path still lands here.

## Learn vs Archify

- **Archify** HTML under dated folders: checked diagrams (click, zoom, route). Mostly explore, not edit.
- **Pace Lab** (`learn/pace-lab.html`): editable teaching layer — insert/delete/reset values at your pace. Notes in `learn/PACE_LAB.md`. Candidate for a later standalone GitHub repo.

## Add a new visual

1. Put the new folder under `viz/`.
2. Add a link on `viz/index.html`.

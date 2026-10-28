---
doc_type: incident_report
equipment_model: HX-500
plant_id: DET
title: RCA-2026-014 Repeated E-42 Seal Failures on Detroit HX-500 Presses
---

# RCA-2026-014 — Repeated E-42 Seal Failures on Detroit HX-500 Presses

**Date of report:** 2026-04-06
**Plant:** Detroit Stamping (DET)
**Equipment:** HX-500 presses DET-L1-M1 and DET-L2-M1

## Summary

During March 2026 both Detroit HX-500 presses raised repeated E-42 (pressure decay during
hold) alarms, causing more than 40 hours of unplanned downtime on body-panel lines. Several
tickets show the gland nut being tightened as a temporary fix because seal kits (P-1001) were
out of stock.

## Five Whys

1. Why did the presses lose pressure? — The ram seals were leaking at the gland.
2. Why were the seals leaking? — The seals were worn and the ram rod on DET-L1-M1 had scoring.
3. Why was the rod scored? — The gland nut was repeatedly over-tightened as a temporary fix.
4. Why was tightening used instead of seal replacement? — No seal kits were in stock.
5. Why were no seal kits in stock? — The seal kit was not on the critical spares list and
   the reorder point was too low for two presses.

## Root Cause

Inadequate critical-spare stocking of HX-500 seal kits combined with a non-compliant
temporary repair practice.

## Corrective Actions

| # | Action | Owner | Due | Status |
|---|--------|-------|-----|--------|
| 1 | Add P-1001 to the critical spares list (POL-MRO-004) | Supply Chain | 2026-04-30 | Done |
| 2 | Raise the Detroit reorder point for P-1001 to 4 kits | Maintenance Planner | 2026-04-30 | Done |
| 3 | Re-chrome the DET-L1-M1 ram rod | Maintenance | 2026-05-15 | Done |
| 4 | Retrain technicians: gland tightening is temporary only (max 24 h) | Maintenance Manager | 2026-05-31 | Done |
| 5 | Add the weekly hold test to the PM schedule for both presses | Maintenance Planner | 2026-04-15 | Done |

## Lessons Learned

Stock-outs of critical spares turn a 4-hour planned repair into recurring unplanned downtime.
The plant should check P-1001 stock levels against the reorder point every week.

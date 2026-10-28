---
doc_type: sop
title: SOP-PM-010 Preventive Maintenance Program
---

# SOP-PM-010 — Preventive Maintenance Program

## Objectives

- Planned maintenance (reason code PM) should make up at least 30% of total downtime minutes.
  A lower share indicates a reactive, breakdown-driven operation.
- Mean time to repair (MTTR) target for high-priority tickets: under 4 hours.
- High-priority tickets must be closed or escalated within 24 hours of opening.

## Downtime Reason Codes

| Code | Meaning |
|------|---------|
| MECH | Unplanned mechanical failure |
| ELEC | Unplanned electrical or controls failure |
| PM | Planned preventive maintenance |
| CHG | Product changeover |
| MAT | Waiting for material |
| QUAL | Quality hold |

MECH and ELEC are **unplanned** downtime. PM, CHG, MAT and QUAL are planned or
operational downtime.

## Ticket Priority Rules

- **High**: any HX-500 press fault (safety-critical hydraulic system), or any fault expected
  to stop a line for more than 3 hours.
- **Medium**: faults with a workaround that keeps the line running at reduced rate.
- **Low**: cosmetic or convenience issues.

## Responsibilities

The maintenance planner schedules PM work orders weekly. Technicians record error code,
root cause and part used in every ticket before closing it.

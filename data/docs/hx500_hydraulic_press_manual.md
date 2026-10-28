---
doc_type: equipment_manual
equipment_model: HX-500
title: HX-500 Hydraulic Press — Operation and Maintenance Manual (Rev. F)
---

# HX-500 Hydraulic Press — Operation and Maintenance Manual

## 1. Specifications

- Rated force: 500 tonnes
- Nominal working pressure: 210 bar (3,045 psi)
- Maximum relief valve setting: 240 bar
- Hydraulic oil: ISO VG 46, reservoir capacity 900 L
- Normal oil operating temperature: 40–55 °C
- Acceptable pressure decay during a 60-second hold test: no more than 2 bar/min

## 2. Alarm and Error Codes

| Code | Meaning | Likely cause | First corrective step |
|------|---------|--------------|-----------------------|
| E-17 | Oil overtemperature (above 65 °C) | Clogged oil cooler, failed cooler fan, low oil level | Stop the cycle, verify oil level, clean the heat exchanger fins |
| E-41 | Low system pressure at start of stroke | Drifting pressure transducer (part P-1003), pump wear, relief valve stuck open | Compare transducer reading against the reference gauge; replace transducer if off by more than 3 bar |
| E-42 | Pressure decay during hold | Worn ram/cylinder seal leaking at the gland (seal kit P-1001), internal leakage across the piston | Perform the hold test in section 4.2. If decay exceeds 2 bar/min, lock out the press (see SOP-LOTO-001) and replace the seal kit |

E-42 must never be cleared by tightening the gland nut as a permanent fix. Over-tightening
scores the ram rod and shortens the life of the replacement seals. Tightening is permitted
only as a temporary measure for up to 24 hours while a seal kit is sourced, and must be
logged in the maintenance ticket.

## 3. Preventive Maintenance

| Interval | Task |
|----------|------|
| Daily | Check oil level and temperature; inspect for external leaks at the cylinder gland |
| Weekly | Run the pressure hold test (section 4.2) and record decay rate |
| Every 2,000 operating hours | Replace return-line filter; sample oil for particle count (target ISO 4406 18/16/13) |
| Every 8,000 operating hours or 24 months | Replace ram seal kit P-1001 regardless of condition |

## 4. Procedures

### 4.1 Lockout before any hydraulic work

Hydraulic presses store energy in accumulators and in the raised ram. Always follow
SOP-LOTO-001, bleed the accumulator to 0 bar, and lower the ram onto blocks before
opening any hydraulic line.

### 4.2 Pressure hold test

1. Bring the press to 210 bar with the ram on the test block.
2. Close the directional valve and start a 60-second timer.
3. Record pressure at 0 s and at 60 s.
4. Decay above 2 bar/min indicates seal leakage (E-42 condition).

### 4.3 Seal kit replacement (P-1001)

1. Apply lockout per SOP-LOTO-001 and lower the ram onto blocks.
2. Remove the gland retainer bolts and extract the gland.
3. Inspect the ram rod for scoring. Any scoring deeper than 0.05 mm requires rod
   re-chroming before new seals are installed.
4. Install new seals from kit P-1001, lubricating with clean hydraulic oil.
5. Torque gland bolts to 180 Nm in a cross pattern.
6. Remove lockout, run the pressure hold test, and record the result in the ticket.

Estimated time: 3–4 hours with two technicians.

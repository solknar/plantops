---
doc_type: equipment_manual
equipment_model: CNC-7
title: CNC-7 Vertical Machining Center — Service Manual
---

# CNC-7 Vertical Machining Center — Service Manual

## 1. Specifications

- Spindle: 12,000 rpm, 22 kW, grease-packed angular contact bearings
- Automatic tool changer (ATC): 30 pockets, side-mount arm
- Coolant: water-soluble, concentration 6–8%, tank 400 L
- Spindle chiller set point: 22 °C

## 2. Alarm Codes

| Code | Meaning | Likely cause | First corrective step |
|------|---------|--------------|-----------------------|
| A-17 | Spindle overtemperature | Blocked chiller filter, failing spindle bearings (part P-2001), excessive duty at high rpm | Check the chiller filter and coolant flow first. Only if the alarm persists with a clean filter, check bearings for noise and vibration (above 4.5 mm/s RMS) |
| A-22 | Tool changer fault | Worn gripper fingers (part P-2003), misaligned ATC home position | Inspect gripper, re-teach ATC positions |
| A-30 | Coolant low / flow fault | Low tank level, cavitating coolant pump (part P-2002), clogged strainer | Check level and concentration, clean strainer, then inspect pump |

About 60% of A-17 alarms in the field are caused by the chiller filter and are resolved
without replacing bearings. Replacing a spindle bearing set (P-2001) takes a full shift and
requires a run-in procedure, so rule out the chiller first.

## 3. Preventive Maintenance

| Interval | Task |
|----------|------|
| Daily | Check coolant level and concentration with refractometer |
| Weekly | Clean chiller filter; inspect ATC gripper fingers |
| Monthly | Vibration measurement at spindle housing; record in CMMS |
| Every 10,000 spindle hours | Replace spindle bearing set P-2001 |

## 4. Spindle Bearing Replacement (P-2001)

1. Lock out per SOP-LOTO-001.
2. Remove spindle cartridge using the lifting fixture.
3. Press out old bearings, clean housing, install new bearing set with matched preload.
4. Run the 45-minute stepped warm-up: 2,000 → 6,000 → 10,000 → 12,000 rpm.
5. Verify vibration below 2.0 mm/s RMS before releasing to production.

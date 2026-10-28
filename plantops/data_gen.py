"""Deterministic synthetic data for a fictional manufacturer, "Northwind Fabrication".

The structured tables are written as CSVs (the customer's "export from the ERP/CMMS"),
and they are deliberately consistent with the hand-written documents in data/docs:
error codes, part numbers, supplier thresholds and the incidents described in the
RCA reports all show up in the numbers.
"""

from __future__ import annotations

import csv
import random
from datetime import date, datetime, timedelta
from pathlib import Path

AS_OF = date(2026, 9, 30)
START = date(2026, 1, 1)
SEED = 7

PLANTS = [
    ("DET", "Detroit Stamping", "Detroit", "MI"),
    ("HOU", "Houston Assembly", "Houston", "TX"),
    ("AUS", "Austin Precision", "Austin", "TX"),
]

MODELS = {
    "HX-500": ("hydraulic_press", ["E-41", "E-42", "E-17"]),
    "CNC-7": ("machining_center", ["A-17", "A-22", "A-30"]),
    "RW-3": ("robotic_welder", ["W-05", "W-12", "W-20"]),
    "CV-200": ("conveyor", ["C-03", "C-09"]),
}
LINE_LAYOUT = ["HX-500", "CNC-7", "CNC-7", "RW-3", "CV-200"]

PARTS = [
    ("P-1001", "HX-500 Seal Kit", "HX-500", 1850.00, "S-01"),
    ("P-1002", "HX-500 Hydraulic Pump", "HX-500", 12400.00, "S-01"),
    ("P-1003", "HX-500 Pressure Transducer", "HX-500", 640.00, "S-05"),
    ("P-2001", "CNC-7 Spindle Bearing Set", "CNC-7", 3200.00, "S-02"),
    ("P-2002", "CNC-7 Coolant Pump", "CNC-7", 980.00, "S-05"),
    ("P-2003", "CNC-7 Tool Changer Gripper", "CNC-7", 450.00, "S-02"),
    ("P-3001", "RW-3 Wire Feed Roller Kit", "RW-3", 310.00, "S-03"),
    ("P-3002", "RW-3 Contact Tips (box of 50)", "RW-3", 95.00, "S-03"),
    ("P-3003", "RW-3 Torch Cable", "RW-3", 1150.00, "S-03"),
    ("P-4001", "CV-200 Drive Belt", "CV-200", 520.00, "S-04"),
    ("P-4002", "CV-200 Gear Motor", "CV-200", 2750.00, "S-04"),
]

# supplier_id, name, country, probability a PO arrives late
SUPPLIERS = [
    ("S-01", "Midwest Hydraulics", "US", 0.03),
    ("S-02", "Precision Spindle GmbH", "DE", 0.10),
    ("S-03", "WeldTech Supply", "US", 0.14),
    ("S-04", "ConveyCo", "MX", 0.04),
    ("S-05", "Lone Star Industrial", "US", 0.07),
]

# Error-code -> (reason_code, part usually replaced, technician note templates)
FAULTS = {
    "E-41": ("MECH", "P-1003", [
        "Low system pressure alarm E-41. Found transducer reading drifting {d} bar low. Replaced pressure transducer, recalibrated.",
        "E-41 on startup. Hydraulic oil level OK. Transducer connector corroded; cleaned and reseated, monitored 2 cycles.",
    ]),
    "E-42": ("MECH", "P-1001", [
        "E-42 pressure decay during hold. Ram seal leaking at gland. Replaced seal kit, pressure test passed at 210 bar.",
        "Repeat E-42 on press. Oil weeping from cylinder seal. Installed new seal kit; recommend checking rod for scoring.",
        "E-42 pressure decay {d} bar/min. Temporary fix: tightened gland nut. Seal kit not in stock, ordered.",
    ]),
    "E-17": ("MECH", None, [
        "E-17 oil overtemperature. Cooler fins clogged with dust; cleaned heat exchanger, temp back to 48C.",
    ]),
    "A-17": ("MECH", "P-2001", [
        "Spindle overtemp A-17 at {d} rpm. Bearing noise present. Replaced spindle bearing set and ran warm-up cycle.",
        "A-17 alarm. Spindle chiller filter blocked. Replaced filter, alarm cleared, no bearing replacement needed.",
    ]),
    "A-22": ("ELEC", "P-2003", [
        "A-22 tool changer fault. Gripper finger worn, dropped tool T{d}. Replaced gripper, re-taught ATC positions.",
    ]),
    "A-30": ("MECH", "P-2002", [
        "A-30 coolant low/flow fault. Coolant pump cavitating. Replaced coolant pump and topped off concentrate to 7%.",
    ]),
    "W-05": ("MECH", "P-3001", [
        "W-05 wire feed jam. Feed rollers worn and liner kinked. Replaced roller kit and liner, wire speed verified.",
        "W-05 recurring on cell. Spool brake tension too high. Adjusted brake, replaced feed rollers as precaution.",
    ]),
    "W-12": ("ELEC", "P-3002", [
        "W-12 arc instability. Contact tip worn and spatter buildup. Replaced contact tip, cleaned nozzle.",
    ]),
    "W-20": ("ELEC", "P-3003", [
        "W-20 collision detect. Torch cable damaged after fixture crash. Replaced torch cable, re-ran TCP calibration.",
    ]),
    "C-03": ("MECH", "P-4001", [
        "C-03 belt slip. Belt glazed and stretched. Replaced drive belt, re-tensioned to spec.",
    ]),
    "C-09": ("ELEC", "P-4002", [
        "C-09 motor overload trip. Gear motor drawing {d}A over nameplate. Replaced gear motor.",
    ]),
}

FIRST_NAMES = ["Ana", "Ben", "Chen", "Dana", "Eli", "Fatima", "Gus", "Hana", "Ivan", "Jo", "Kai", "Lena"]
LAST_NAMES = ["Ortiz", "Brooks", "Wei", "Kowalski", "Nguyen", "Haddad", "Larsen", "Sato", "Petrov", "Reyes", "Okafor", "Fischer"]


def _ts(d: date, rng: random.Random) -> datetime:
    return datetime(d.year, d.month, d.day, rng.randint(0, 23), rng.choice([0, 15, 30, 45]))


def _days(rng: random.Random) -> date:
    return START + timedelta(days=rng.randint(0, (AS_OF - START).days))


def generate(out_dir: Path) -> dict[str, int]:
    """Write all CSVs into out_dir and return row counts per table."""
    rng = random.Random(SEED)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables: dict[str, list[dict]] = {}

    tables["plants"] = [
        {"plant_id": p, "name": n, "city": c, "state": s} for p, n, c, s in PLANTS
    ]

    lines, machines = [], []
    for plant_id, *_ in PLANTS:
        for i in (1, 2):
            line_id = f"{plant_id}-L{i}"
            family = {"DET": "body panels", "HOU": "chassis frames", "AUS": "drivetrain housings"}[plant_id]
            lines.append({"line_id": line_id, "plant_id": plant_id, "name": f"{plant_id} Line {i}", "product_family": family})
            for j, model in enumerate(LINE_LAYOUT, start=1):
                machines.append({
                    "machine_id": f"{line_id}-M{j}",
                    "line_id": line_id,
                    "plant_id": plant_id,
                    "model": model,
                    "machine_type": MODELS[model][0],
                    "install_date": date(rng.randint(2014, 2023), rng.randint(1, 12), 1).isoformat(),
                    "status": "running",
                })
    tables["production_lines"] = lines
    tables["machines"] = machines

    techs = []
    for i in range(12):
        plant_id = PLANTS[i % 3][0]
        first, last = FIRST_NAMES[i], LAST_NAMES[i]
        techs.append({
            "technician_id": f"T-{100 + i}",
            "name": f"{first} {last}",
            "plant_id": plant_id,
            "email": f"{first.lower()}.{last.lower()}@northwind-fab.example",
            "phone": f"+1-555-01{i:02d}",
            "certification": rng.choice(["hydraulics", "CNC", "welding", "electrical"]),
        })
    tables["technicians"] = techs

    tables["suppliers"] = [{"supplier_id": s, "name": n, "country": c} for s, n, c, _ in SUPPLIERS]
    tables["parts"] = [
        {"part_id": p, "name": n, "compatible_model": m, "unit_cost_usd": f"{c:.2f}", "supplier_id": s}
        for p, n, m, c, s in PARTS
    ]

    # Downtime + maintenance tickets. Detroit presses get extra E-42 (see the March RCA),
    # Houston welders get extra W-05 (see the June RCA).
    downtime, tickets = [], []
    ev = tk = 0
    for m in machines:
        codes = MODELS[m["model"]][1]
        n_faults = rng.randint(4, 9)
        weights = [1.0] * len(codes)
        if m["plant_id"] == "DET" and m["model"] == "HX-500":
            n_faults += 8
            weights[codes.index("E-42")] = 4.0
        if m["plant_id"] == "HOU" and m["model"] == "RW-3":
            n_faults += 6
            weights[codes.index("W-05")] = 4.0
        for _ in range(n_faults):
            code = rng.choices(codes, weights=weights)[0]
            reason, part, notes = FAULTS[code]
            d = _days(rng)
            if code == "E-42" and m["plant_id"] == "DET" and rng.random() < 0.5:
                d = date(2026, 3, rng.randint(1, 28))
            start = _ts(d, rng)
            duration = rng.randint(30, 240) if reason == "MECH" else rng.randint(15, 120)
            ev += 1
            downtime.append({
                "event_id": f"DT-{ev:05d}", "machine_id": m["machine_id"], "plant_id": m["plant_id"],
                "start_ts": start.isoformat(sep=" "), "duration_min": duration,
                "reason_code": reason, "error_code": code,
            })
            tk += 1
            opened = start + timedelta(minutes=rng.randint(2, 20))
            is_open = (AS_OF - d).days < 14 and rng.random() < 0.6
            closed = "" if is_open else (opened + timedelta(minutes=duration + rng.randint(10, 90))).isoformat(sep=" ")
            priority = "high" if (m["model"] == "HX-500" or duration > 180) else rng.choice(["medium", "low"])
            plant_techs = [t for t in techs if t["plant_id"] == m["plant_id"]]
            tickets.append({
                "ticket_id": f"MT-{tk:05d}", "machine_id": m["machine_id"], "plant_id": m["plant_id"],
                "opened_at": opened.isoformat(sep=" "), "closed_at": closed,
                "priority": priority, "status": "open" if is_open else "closed",
                "technician_id": rng.choice(plant_techs)["technician_id"],
                "error_code": code, "part_used": part if (part and not is_open) else "",
                "summary": f"{code} on {m['model']} {m['machine_id']}",
                "notes": rng.choice(notes).format(d=rng.randint(3, 25)),
            })
        # Planned / non-fault downtime
        for _ in range(rng.randint(3, 6)):
            ev += 1
            d = _days(rng)
            downtime.append({
                "event_id": f"DT-{ev:05d}", "machine_id": m["machine_id"], "plant_id": m["plant_id"],
                "start_ts": _ts(d, rng).isoformat(sep=" "), "duration_min": rng.randint(30, 180),
                "reason_code": rng.choice(["PM", "CHG", "MAT", "QUAL"]), "error_code": "",
            })
    downtime.sort(key=lambda r: r["start_ts"])
    tables["downtime_events"] = downtime
    tables["maintenance_tickets"] = tickets

    late_p = {s: p for s, _, _, p in SUPPLIERS}
    pos = []
    for i in range(1, 161):
        part = rng.choice(PARTS)
        supplier = part[4]
        ordered = _days(rng)
        promised = ordered + timedelta(days=rng.randint(7, 21))
        if promised > AS_OF:
            received = ""
        else:
            slip = rng.randint(2, 12) if rng.random() < late_p[supplier] * 1.5 else -rng.randint(0, 3)
            received = (promised + timedelta(days=slip)).isoformat()
        pos.append({
            "po_id": f"PO-{i:04d}", "supplier_id": supplier, "part_id": part[0],
            "plant_id": rng.choice(PLANTS)[0], "qty": rng.randint(1, 20),
            "ordered_at": ordered.isoformat(), "promised_date": promised.isoformat(), "received_date": received,
        })
    tables["purchase_orders"] = pos

    inventory = []
    for plant_id, *_ in PLANTS:
        for p in PARTS:
            reorder = 4 if p[0] in ("P-1001", "P-2001") else rng.randint(2, 10)
            on_hand = rng.randint(0, 25)
            if (plant_id, p[0]) in {("DET", "P-1001"), ("HOU", "P-3001"), ("HOU", "P-3002"), ("AUS", "P-2001")}:
                on_hand = rng.randint(0, reorder - 1)
            inventory.append({"plant_id": plant_id, "part_id": p[0], "on_hand": on_hand, "reorder_point": reorder})
    tables["inventory"] = inventory

    inspections = []
    i = 0
    for line in lines:
        week = START
        while week <= AS_OF:
            i += 1
            units = rng.randint(800, 1500)
            rate = 0.012 if line["plant_id"] != "HOU" else 0.019
            inspections.append({
                "inspection_id": f"QI-{i:05d}", "line_id": line["line_id"], "plant_id": line["plant_id"],
                "inspection_date": week.isoformat(), "units_inspected": units,
                "defects": max(0, int(rng.gauss(units * rate, 3))),
                "top_defect_type": rng.choice(["weld porosity", "dimensional", "surface scratch", "burr", "misalignment"]),
            })
            week += timedelta(days=7)
    tables["quality_inspections"] = inspections

    counts = {}
    for name, rows in tables.items():
        with open(out_dir / f"{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        counts[name] = len(rows)
    return counts


if __name__ == "__main__":
    from plantops.config import get_settings

    print(generate(get_settings().structured_dir))

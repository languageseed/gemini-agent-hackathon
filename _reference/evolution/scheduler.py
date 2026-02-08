#!/usr/bin/env python3
"""
Evolver Scheduler Service

Manages scheduled Evolver runs with a web UI for configuration.
"""

import json
import os
import subprocess
import threading
import time
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Configuration paths
CONFIG_DIR = Path(os.getenv("EVOLVER_CONFIG_DIR", "/app/config"))
SCHEDULE_FILE = CONFIG_DIR / "schedule.json"
PLATFORM_ROOT = Path(os.getenv("PLATFORM_ROOT", "/workspace"))
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", "/reports"))

# Scheduler state
scheduler_thread: Optional[threading.Thread] = None
scheduler_running = False


def load_schedule() -> Dict:
    """Load schedule configuration"""
    if SCHEDULE_FILE.exists():
        with open(SCHEDULE_FILE, "r") as f:
            return json.load(f)
    return {"schedules": [], "default_timezone": "UTC"}


def save_schedule(config: Dict):
    """Save schedule configuration"""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(SCHEDULE_FILE, "w") as f:
        json.dump(config, f, indent=2)


def day_name_to_number(day_name: str) -> int:
    """Convert day name to weekday number (0=Monday, 6=Sunday)"""
    days = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }
    return days.get(day_name.lower(), 0)


def parse_time(time_str: str) -> dt_time:
    """Parse time string (HH:MM) to time object"""
    hour, minute = map(int, time_str.split(":"))
    return dt_time(hour, minute)


def should_run_now(schedule: Dict) -> bool:
    """Check if a schedule should run now"""
    if not schedule.get("enabled", True):
        return False

    timezone_str = schedule.get("timezone", "UTC")
    tz = ZoneInfo(timezone_str)
    now = datetime.now(tz)

    # Check day of week
    days = schedule.get("days", [])
    day_names = [d.lower() for d in days]
    current_day = now.strftime("%A").lower()

    if current_day not in day_names:
        return False

    # Check time (within 1 minute window)
    schedule_time = parse_time(schedule.get("time", "00:00"))
    current_time = now.time()

    # Check if we're within 1 minute of the scheduled time
    time_diff = abs(
        (current_time.hour * 60 + current_time.minute)
        - (schedule_time.hour * 60 + schedule_time.minute)
    )

    return time_diff <= 1


def run_evolver(schedule: Dict):
    """Run Evolver for a scheduled product"""
    product = schedule.get("product")
    model = schedule.get("model", "gpt-5.2-codex")
    timeout = schedule.get("timeout_sec", 3600)

    if not product:
        print(f"ERROR: Schedule {schedule.get('id')} has no product")
        return

    print(f"[SCHEDULER] Running Evolver for {product} (model: {model})")

    # Build docker compose command
    cmd = [
        "docker",
        "compose",
        "-f",
        str(PLATFORM_ROOT / "apps/evolver/docker-compose.yml"),
        "run",
        "--rm",
        "-e",
        f"EVOLVER_MODEL={model}",
        "-e",
        f"EVOLVER_TIMEOUT_SEC={timeout}",
        "-e",
        f"PLATFORM_MOUNT={PLATFORM_ROOT}",
        "-v",
        f"{PLATFORM_ROOT}:{PLATFORM_ROOT}:ro",
        "-v",
        f"{REPORTS_DIR}:{REPORTS_DIR}",
        "evolver",
        product,
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=str(PLATFORM_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout + 300,  # Add 5 min buffer
        )
        if result.returncode == 0:
            print(f"[SCHEDULER] Successfully completed review for {product}")
        else:
            print(f"[SCHEDULER] Error running Evolver for {product}: {result.stderr}")
    except subprocess.TimeoutExpired:
        print(f"[SCHEDULER] Timeout running Evolver for {product}")
    except Exception as e:
        print(f"[SCHEDULER] Exception running Evolver for {product}: {e}")


def scheduler_loop():
    """Main scheduler loop - checks schedules every minute"""
    global scheduler_running
    last_runs = {}  # Track last run time per schedule to avoid duplicates

    while scheduler_running:
        try:
            config = load_schedule()
            schedules = config.get("schedules", [])

            for schedule in schedules:
                schedule_id = schedule.get("id")
                if not schedule_id:
                    continue

                # Check if we should run
                if should_run_now(schedule):
                    # Avoid running multiple times in the same minute
                    now_key = f"{schedule_id}_{datetime.now().strftime('%Y-%m-%d %H:%M')}"
                    if now_key not in last_runs:
                        last_runs[now_key] = True
                        # Run in a separate thread to avoid blocking
                        threading.Thread(
                            target=run_evolver, args=(schedule,), daemon=True
                        ).start()
                        # Clean old entries (keep last 100)
                        if len(last_runs) > 100:
                            last_runs.clear()

        except Exception as e:
            print(f"[SCHEDULER] Error in scheduler loop: {e}")

        time.sleep(60)  # Check every minute


# API Routes
@app.route("/api/schedules", methods=["GET"])
def get_schedules():
    """Get all schedules"""
    config = load_schedule()
    return jsonify(config)


@app.route("/api/schedules", methods=["POST"])
def save_schedules():
    """Save schedules"""
    try:
        data = request.json
        save_schedule(data)
        return jsonify({"success": True, "message": "Schedules saved"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/schedules/<schedule_id>/run", methods=["POST"])
def run_schedule_now(schedule_id: str):
    """Manually trigger a schedule"""
    config = load_schedule()
    schedules = config.get("schedules", [])

    schedule = next((s for s in schedules if s.get("id") == schedule_id), None)
    if not schedule:
        return jsonify({"success": False, "error": "Schedule not found"}), 404

    threading.Thread(target=run_evolver, args=(schedule,), daemon=True).start()
    return jsonify({"success": True, "message": f"Running schedule {schedule_id}"})


@app.route("/api/status", methods=["GET"])
def get_status():
    """Get scheduler status"""
    return jsonify(
        {
            "running": scheduler_running,
            "schedules": len(load_schedule().get("schedules", [])),
            "enabled_schedules": len(
                [
                    s
                    for s in load_schedule().get("schedules", [])
                    if s.get("enabled", True)
                ]
            ),
        }
    )


@app.route("/", methods=["GET"])
def index():
    """Serve the web UI"""
    ui_path = Path(__file__).parent.parent / "ui" / "index.html"
    if ui_path.exists():
        return send_from_directory(ui_path.parent, "index.html")
    return "<h1>Evolver Scheduler</h1><p>UI not found. Check /api/schedules for API.</p>", 404


if __name__ == "__main__":
    # Start scheduler thread
    scheduler_running = True
    scheduler_thread = threading.Thread(target=scheduler_loop, daemon=True)
    scheduler_thread.start()

    # Run Flask app
    port = int(os.getenv("SCHEDULER_PORT", "8080"))
    app.run(host="0.0.0.0", port=port, debug=False)

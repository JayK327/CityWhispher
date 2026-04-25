#!/usr/bin/env python3
"""
Test script to fetch real POIs from Overpass API.
Run with: python scripts/test_overpass.py
"""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.module1_ingestion.overpass import fetch_overpass_pois

# Real coordinates for famous landmarks
LOCATIONS = {
    "Paris (Eiffel Tower)": (48.8584, 2.2945),
    "London (Big Ben)": (51.5007, -0.1246),
    "Rome (Colosseum)": (41.8902, 12.4922),
    "Tokyo (Tokyo Tower)": (35.6586, 139.7454),
    "New York (Statue of Liberty)": (40.6892, -74.0445),
}


async def test_location(name: str, lat: float, lon: float, radius_m: int = 500):
    """Fetch and display POIs for a location."""
    print(f"\n{'='*70}")
    print(f" {name}")
    print(f"   Coordinates: ({lat:.4f}, {lon:.4f})")
    print(f"   Radius: {radius_m}m")
    print(f"{'='*70}")

    try:
        pois = await fetch_overpass_pois(lat, lon, radius_m)

        if not pois:
            print(" No POIs found")
            return

        print(f" Found {len(pois)} POIs:\n")

        for i, poi in enumerate(pois[:10], 1):  # Show first 10
            tags = poi.tags
            category = tags.get("tourism") or tags.get("historic") or tags.get("amenity") or "unknown"
            opening_hours = tags.get("opening_hours", "N/A")
            description = tags.get("description", "—")[:50]

            print(f"{i}. {poi.name}")
            print(f"   Category: {category}")
            print(f"   Opening hours: {opening_hours}")
            print(f"   Description: {description}")
            print()

    except Exception as e:
        print(f" Error: {e}")


async def main():
    """Run tests for all locations."""
    print("\n  Overpass API Test — Fetching real POIs from worldwide landmarks\n")

    for name, (lat, lon) in LOCATIONS.items():
        await test_location(name, lat, lon)

    print(f"\n{'='*70}")
    print("Test completed!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    asyncio.run(main())

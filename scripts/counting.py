import os
import shutil
from collections import Counter

# otkud zelimo brojati fileove
SOURCE_DIR = r"./data/raw/Final_Veh"
DEST_DIR = r"./data/sorted_by_vehicle"

# tipovi vozila koje zelimo countati
VEHICLE_TYPES = [
    "BIC",
    "MC",
    "CAR",
    "VAN",
    "BUS",
    "TRUCK",
    "TRAM"
]

# izlazni folderi
for vehicle in VEHICLE_TYPES:
    os.makedirs(os.path.join(DEST_DIR, vehicle), exist_ok=True)

# neprepoznati fileovi
UNKNOWN_DIR = os.path.join(DEST_DIR, "UNKNOWN")
os.makedirs(UNKNOWN_DIR, exist_ok=True)

counter = Counter()

# prodjemo sve fileove u source folderu
for root, dirs, files in os.walk(SOURCE_DIR):

    for file_name in files:

        if not file_name.lower().endswith(".wav"):
            continue

        upper_name = file_name.upper()

        found_vehicle = None

        for vehicle in VEHICLE_TYPES:
            if vehicle in upper_name:
                found_vehicle = vehicle
                break

        source_file = os.path.join(root, file_name)

        if found_vehicle:
            destination_file = os.path.join(
                DEST_DIR,
                found_vehicle,
                file_name
            )

            counter[found_vehicle] += 1

        else:
            # ako nije prepoznato vozilo
            destination_file = os.path.join(
                UNKNOWN_DIR,
                file_name
            )

            counter["UNKNOWN"] += 1

        # kopiranje filea
        shutil.copy2(source_file, destination_file)

print("\nGOTOVO! -----\n")

print("Broj fileova po tipu vozila:\n")

for vehicle, count in sorted(counter.items()):
    print(f"{vehicle}: {count}")

print("\nUkupno fileova:", sum(counter.values()))

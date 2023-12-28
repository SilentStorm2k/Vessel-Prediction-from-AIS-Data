# Vessel Prediction from Automatic Identification System (AIS) Data

## Overview

This project predicts vessel movements from AIS data, even when MMSIs (Maritime Mobile Service Identity numbers) are unavailable. It aims to track individual vessels over time by assigning position reports to their corresponding vessels with reasonable accuracy. 

## Main Program

- **Name:** `predictVessel.py` 
**Functions:**

- **`predictWithK(num_vessels)`:** Tracks vessel movements using AIS data when the number of vessels is known.
- **`predictWithoutK()`:** Tracks vessel movements using AIS data without prior knowledge of the number of vessels.


## Data Description

**Format:** Each row represents a single vessel observation at a specific time point.

**Columns:**

1. **Object ID (OBJECT_ID):** Unique identifier for each position report.
2. **Vessel ID (VID):** Anonymized MMSI number, unique within a dataset but not across datasets.
3. **Timestamp (SEQUENCE_DTTM):** Time of reporting in UTC format (hh:mm:ss). Date information is excluded.
4. **Latitude (LAT):** Vessel position in decimal degrees.
5. **Longitude (LON):** Vessel position in decimal degrees.
6. **Speed over Ground (SPEED_OVER_GROUND):** Vessel speed in tenths of a knot (nautical mile per hour), up to 1022 (102.2 knots or higher).
7. **Course over Ground (COURSE_OVER_GROUND):** Angle of vessel movement in tenths of a degree (0-3599). Reported even when speed is 0 due to currents and other factors.

## Algorithm

**DBSCAN (Density-Based Spatial Clustering of Applications with Noise):**

- Unsupervised learning algorithm for finding high-density clusters.
- Selected hyperparameters:
    - Epsilon (eps): 0.7
    - Min_samples: 5


<sup>For further tests done and my thought process, read through my [report].</sup>

[report]: https://www.mediafire.com/file/zxhev8uscdarj0u/annotated-Case+Study+2+-+Report.pdf/file 
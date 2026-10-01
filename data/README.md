# Dataset Documentation: IPL Ball-by-Ball Data (2008–2017)

## Overview
This directory contains the historical Indian Premier League (IPL) cricket match dataset used by the project: [`ipl.csv`](./ipl.csv).

## Dataset Specifications
- **File Name**: `ipl.csv`
- **File Size**: ~9.43 MB
- **Total Rows**: 76,014 delivery records
- **Total Columns**: 15 features
- **Unique Matches**: 617 matches
- **Temporal Coverage**: 2008-04-18 (Season 1 opening match) to 2017-05-21 (Season 10 final)

## Feature Schema
| Column | Type | Description |
| :--- | :--- | :--- |
| `mid` | Integer | Unique identifier for each match |
| `date` | String | Match date in `DD-MM-YYYY` format |
| `venue` | String | Stadium / match venue |
| `bat_team` | String | Batting team name |
| `bowl_team` | String | Bowling team name |
| `batsman` | String | Striker batsman name |
| `bowler` | String | Bowler name |
| `runs` | Integer | Cumulative runs scored up to the current delivery |
| `wickets` | Integer | Cumulative wickets lost up to the current delivery |
| `overs` | Float | Completed overs up to the current delivery (e.g., 5.1 overs) |
| `runs_last_5` | Integer | Runs scored in the preceding 5 overs |
| `wickets_last_5` | Integer | Wickets lost in the preceding 5 overs |
| `striker` | Integer | Runs scored by current striker |
| `non-striker` | Integer | Runs scored by non-striker |
| `total` | Integer | **Prediction Target**: Total final innings score |

## Provenance & Attribution
- **Upstream Source**: Kaggle – [IPL Dataset Season 2008 to 2017 (yuvrajdagur/ipl-dataset-season-2008-to-2017)](https://www.kaggle.com/datasets/yuvrajdagur/ipl-dataset-season-2008-to-2017)
- **Data Heritage**: The dataset represents historical match scorecard telemetry from the first 10 seasons of the Indian Premier League (2008–2017).
- **Upstream License**: Refer to the upstream Kaggle dataset page for the specific license terms established by the dataset publisher.
- **Redistribution Terms**: The dataset is included in this repository strictly in the form utilized for the academic college project. The MIT software license covering this repository's source code does not assert copyright or grant license terms over this third-party sports dataset. Users redistributing or utilizing this data independently should consult upstream terms.

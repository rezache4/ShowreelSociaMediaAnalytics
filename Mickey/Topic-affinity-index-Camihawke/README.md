# Topic-affinity-index-Camihawke

This repository contains the files used to compute YouTube topic affinity against CamiHawke with the severe taxonomy.

## Contents

- Channel Excel files: one workbook per YouTube channel, with macro-topic coverage in the `Macro Topics` sheet.
- `CamiHawke_macro_topics_comparable_v3.xlsx`: the reference workbook used to build the CamiHawke YouTube topic vector from the `YouTube` sheet only.
- `youtube_affinity_index_vs_camihawke.ipynb`: the Python notebook that reads the Excel files, builds comparable vectors, and computes cosine similarity.
- `youtube_affinity_index_vs_camihawke.csv`: the final ranking of YouTube channels ordered by affinity index in descending order.

## From Excel Files to Affinity Index

1. Read the `Macro Topics` sheet for each YouTube channel and the `YouTube` sheet for CamiHawke.
2. Build one vector per channel using the 27 final severe macro-topics. Missing topics are set to `0`.
3. Normalize each channel vector so the topic weights sum to `100%` at channel level.
4. Build the CamiHawke reference vector with the same 27-topic structure.
5. Compute cosine similarity between each channel vector and the CamiHawke vector.
6. Sort channels by similarity score from highest to lowest and export the result to the CSV ranking.

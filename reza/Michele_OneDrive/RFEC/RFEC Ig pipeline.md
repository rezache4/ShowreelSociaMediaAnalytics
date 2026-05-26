## IG RFEC pipeline

Clean comments (remove camihawke's, filter top-level comments, pick users that commented at least 3 times per era).
Compute RFEC variables using the definitions available on the canva.
- build_ig_comments_rfe_macro.py 

Find thresholds with trial and error looking at the distribution and Codex suggestions (they have to be both reasonable from a behavioral pov and readable in terms of clusters). Once thresholds have been set, plot again the distributions.
- plot_ig_rfe_distributions.py
- ig_comments_rfe_behavioral_thresholds.csv

Then classify users per era according to the chosen thresholds
- classify_ig_comments_rfe_macro.py 

Use k-medoids clustering (manhattan distance). Try different k and choose the best silhouette score
- cluster_ig_comments_kmedoids.py 
The clusters' names were produced by codex and are included in:
- ig_clusters_semantics.py

Then plot bar chart, pie charts and radar chart for the resulting clusters
- plot_ig_comments_kmedoids_era_shares.py
- plot_ig_comments_kmedoids_cluster_radar.py
 
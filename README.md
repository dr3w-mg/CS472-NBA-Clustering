# CS472-NBA-Clustering

## Overview
Code for clustering NBA players into archetypes with data from the nba_api package. We also built team cluster profiles and recommended NBA players to teams based on team weaknesses and player strengths.

## API
We pulled player metrics from three nba_api datasets; traditional stats (points, rebounds, assists, etc.), advanced stats (usage rate, TS percentage, offensive/defensive rating, etc.), and hustle stats (contested shots, charges drawn, screen assists, etc.)

## Preprocessing
After building our three player tables, we filtered them to only contain rows for players who played at least 30 games and averaged 15 minutes per game in the 2024-25 regular season. Then we joined all three to end up with one table with 322 rows.

## Player Archetype Modeling
Our feature set (columns that define playstyle) were built using the column data from the nba_api package. For example, our scoring and shot profile feature included points, shot attempts, made shots, 3pt rate, and 2pt/3pt shot ratio. We ended with 26 features. We then standardized all 26 to assure that large-scale states like points don't dominate small-scale stats like shooting percentage, and each feature contributes to the model fairly. To select k, our number of clusters, we tested values 3 to 10 using BIC and AIC to balance fit and complexity. We landed on k = 9, so 9 clusters. To actually cluster players into archetypes, we used a Gaussian-Mixture Model. The thought process here was that it's hard to assign NBA players to just one cluster because there's a lot of overlap and players play differently given their current roster, team needs, or how early/deep in the season they are. With A GMM, we were able to **soft assign** players to clusters to better understand their role in the league.

## Team Need Modeling
To appropriately match players to team needs, we built team cluster profiles. These give percentages for how many players each team has in a cluster. We also build a league average cluster distribution to compare the team cluster profiles too. By comparing, we created a team needs table and ranked players by how well their cluster probability matched a team's need.

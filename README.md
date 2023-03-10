# Predicting 2023 Women's March Madness

### Data Analysis for Draft King's
#### Ash Heinke


## Overview

As an analyst for Draft Kings, I am exploring the growing markets of interest in women's sports and women betting in sports, to see how we can not just be a part of this ngrowing market, but how we can also add value.


## Business Problem

Draft Kings currently does not offer betting on any women’s sports so there is some major opportunity to be some of the first in the markt who do so.
Women who are betting on sports is growing, but they are still drastically underrepresented, as only about 20% of fans who bet on sports are women. 
This is not to say that these women only want to bet on women's sports, but that the representation over all is lacking, and especially in such a growing market.

GOAL : 
What is more watched than March Madness??! 
So we are going to use Machine Learning to attemot to predict the outcome of Women’s March Madness, 2023, as a mean's of entry into the women's sports space.
Could potentially be a test as well for a feature to be added to our VIP services.
<br>


# Exploratory Data Analysis

## Data Understanding

We initially started with over 30 different datasets, all pulled from Kaggle, but then narrowed it down to these few which we thought were most important:
- WRegularSeasonDetailedResults.csv
- WNCAATourneyDetailedResults.csv
- WNCAATourneySeeds.csv
- WTeams.csv

The first two datasets cover an abundance of stats for every game played since the 2009-2010 (deeper descriptions below) from which we have determined our target feature, whether a team wins or loses a game, which we added to our dataframe and is dented by 'Win1_Lose0'. The seed data was added to our dataframe as a reference of which teams were invited to March Madness every year, as well as a likelihood predictor of whether that team would make it to the Final Four (in the past few years of womens data, at least three of the teams who made it to the Final Four were 1 seeds).

Since Selection Sunday has yet to happen, it will be this Sunday March 12th, the overall goal is to see how accurately we can predict every possible potential matchup, that is to be predicted by the winning probability of the first team listed in each row. There are a total of 361 women's college basketball teams in Division I, so we should end up with a total of 64,980 predictions.


## Importing Datasets

### WNCAA Regular Season Detailed Results Dataset

This data includes game-by-game stats at a team level (free throws attempted, defensive rebounds, turnovers, etc.) for all regular season, conference tournament, and NCAA® tournament games since the 2009-10 season.

For each season, the file includes all games played from DayNum 0 through 132. The "Regular Season" games are simply defined to be all games played on DayNum = 132 or earlier (DayNum = 132 is Selection Sunday, and there are always a few conference tournament finals actually played early in the day on Selection Sunday itself). Thus a game played on or before Selection Sunday will show up here whether it was a pre-season tournament, a non-conference game, a regular conference game, a conference tournament game, etc.

Each of the columns is described below, noting that a "W" or "L" refers to the winning or losing team:

- 'Season': this is the year in which the final tournament occurs. For example, during the 2016 season, there were regular season games played between November 2015 and March 2016, and all of those games will show up with a Season of 2016.
- 'DayNum': this integer always ranges from 0 to 132, and tells you what day the game was played on. It represents an offset from the "DayZero" date. For example, the first game was DayNum = 11, which means it occured 11 days into the season from day zero.
- 'WTeamID': this identifies the id number of the team that won the game, whether the game was won by the home team or visiting team, or if it was a neutral-site game.
- 'WScore': this identifies the number of points scored by the winning team.
- 'LTeamID': this identifies the id number of the team that lost the game.
- 'LScore': this identifies the number of points scored by the losing team. Thus you can be confident that WScore will be greater than LScore for all games listed.
- 'WLoc': this identifies the "location" of the winning team. If the winning team was the home team, this value will be "H". If the winning team was the visiting (or "away") team, this value will be "A". If it was played on a neutral court, then this value will be "N". 
- 'NumOT': this indicates the number of overtime periods in the game, an integer 0 or higher.

- 'WFGM': field goals made (by the winning team)
- 'WFGA': field goals attempted (by the winning team)
- 'WFGM3': three pointers made (by the winning team)
- 'WFGA3': three pointers attempted (by the winning team)
- 'WFTM': free throws made (by the winning team)
- 'WFTA': free throws attempted (by the winning team)
- 'WOR': offensive rebounds (pulled by the winning team)
- 'WDR': defensive rebounds (pulled by the winning team)
- 'WAst': assists (by the winning team)
- 'WTO': turnovers committed (by the winning team)
- 'WStl': steals (accomplished by the winning team)
- 'WBlk': blocks (accomplished by the winning team)
- 'WPF': personal fouls committed (by the winning team)

*The same set of stats from the perspective of the losing team: LFGM is the number of field goals made by the losing team, and so on up to LPF.

Note: by convention, "field goals made" (either WFGM or LFGM) refers to the total number of field goals made by a team, a combination of both two-point field goals and three-point field goals, however, "three point field goals made" (either WFGM3 or LFGM3) is just the three-point fields goals made. So to calculate two-point field goals, you have to subtract one from the other (e.g., WFGM - WFGM3). The total number of points scored is 2 * FGM + FGM3 + FTM.


There were some interesting findings in this first dataset!
- The lowest score ever in a season since 2010 for a losing team was 11 points? In comparison a team has won with only 30 points...
- There have been a game/games with 5 overtimes!
- It is interesting that certain stats could have a minimum of 0 (no assists, three point attempts, etc??) This could mean that we have some missing data that we will need to deal with...
We're going to be utilizing various calculations, especially the mean, for a lot of these statistics, so this may help resolve some of the data that seems funny.

### WNCAA Team Names Dataset

These files identify the different college teams present in the dataset. Each school is uniquely identified by a 4 digit id number, starting with a 3. There are 361 teams currently in Women's Division-I. Each year, some teams might start being Division-I programs, and others might stop being Division-I programs. So there will be some teams listed in the data only for historical seasons and not for the current season.

- 'TeamID': a 4 digit id number, uniquely identifying each NCAA® women's team. A school's TeamID does not change from one year to the next, so for instance Uconn's TeamID is 3163 for all seasons. These ID's do not denote anything in particular to the game of basketball, they were created by the owner of the dataset.
- 'TeamName': a compact spelling of the team's college name, 16 characters or fewer. There are no commas or double-quotes in the team names, but you will see some characters that are not letters or spaces, e.g., Texas A&M, St Mary's CA, TAM C. Christi, and Bethune-Cookman.

### WNCAA Tourney Detailed Results Dataset

This dataset provides team-level, game-by-game NCAA® tournament results for all seasons of historical data, starting with the 2010 season. Note that this tournament game data also includes the play-in games for the years that had them, so each season you will see between 63 and 67 games listed, depending on how many play-in games there were.

Even with the somewhat varied structure of the women's tournament schedule, you can generally tell what round a game was, depending on the DayNum. In general the schedule will be:

- DayNum = 134 or 135 (Tue/Wed): play-in games to get the tournament field down to the final 64 teams
- DayNum = 136 or 137 (Thu/Fri): Round 1, to bring the tournament field from 64 teams to 32 teams
- DayNum = 138 or 139 (Sat/Sun): Round 2, to bring the tournament field from 32 teams to 16 teams
- DayNum = 143 or 144 (Thu/Fri): Round 3, the "Sweet Sixteen", to bring the tournament field from 16 teams to 8 teams
- DayNum = 145 or 146 (Sat/Sun): Round 4, known as the "Elite Eight" or "regional finals", to bring the tournament field from 8 teams to 4 teams
- DayNum = 152 (Sat): Round 5, known as the "Final Four" or "national semifinals", to bring the tournament field from 4 teams to 2 teams
- DayNum = 154 (Mon): Round 6, known as the "national final" or "national championship", to bring the tournament field from 2 teams to 1 champion team

### WNCAA Tourney Seed Dataset
 
This file identifies the seeds for all teams in the NCAA® tournament, for all seasons of historical data. Thus, there are between 64-68 rows for each year, depending on whether there were any play-in games and how many there were. In recent years the structure has settled at 68 total teams, with four "play-in" games leading to the final field of 64 teams entering Round 1 on Thursday/Friday of the first week (by definition, that is DayNum = 136/137 each season). We will not know the seeds of the respective tournament teams, or even exactly which 68 teams it will be, until Selection Sunday on March 12, 2023 (DayNum = 132).

- 'Season': the year that the tournament was played in
- 'Seed': this is a 3/4-character identifier of the seed, where the first character is either W, X, Y, or Z (identifying the region the team was in) and the next two digits (either 01, 02, ..., 15, or 16) tell you the seed within the region. For play-in teams, there is a fourth character (a or b) to further distinguish the seeds, since teams that face each other in the play-in games will have seeds with the same first three characters. The "a" and "b" are assigned based on which Team ID is lower numerically. As an example of the format of the seed, the first record in the file is seed W01 from 1998, which means we are looking at the #1 seed in the W region.
- 'TeamID': this identifies the id number of the team, as specified in the teams file


Once we have pulled in all of our necessary dataframes, we merged the necessary pieces together to create the main dataframe that we will run through our model, and then we calculated some new features to help train our models. This includes the below:
- ScoreGap: the difference between the winning and losing score
- Total Season Wins: the total number of games a team win in a season
- Total Season Losses: the total number of games a team lost in a season
- Win Margin: when a team won, by how many points did they win
- Loss Margin: when a team lost, by how many points did they lose
- GapAvg: based on a teams wins and losses, their average score gap
- True Shooting Percentage: a teams shooting percentage based on regular shots, as well as three's and freethrows. Also accounts for all types of shots missed.

After adding these new features, to get the rest of our stats features into a preferable format for our model, I grouped by season and the winning team id's to get the means of the rest of these stats. These means were then pulled into the main dataframe for every team. Then I created a game id column so that once we split each of the rows, the correspnding games could be easily referenced.

From here I split the main dataframe into winners and losers, with each of there respective stats. The goal is to put this all back together to create a final dataframe that will have two instances of every game, with the winning and lossing teams plus their stats swapped, as well as renamed, so that in that the model isn't simply learning that the first team listed is always the one that wins. We want a bit more randomness that it can learn from. As well, at this point I am adding in a Win/Lose column, where a '1' denotes a win and a '0' a loss.

Finally we made sure all data types were numerical, and then removed any 2023 data from our final dataframe, as we will be using the 2023 data for our prediction.

Our final dataframe ends up with over 130,000 rows, so nthese are the models I ran:
- XGBoost
- Decision Trees
- Decision Trees Pipeline with Cross Validation

With earlier dataframe iterations, I was getting an 88% from my XGBoost, but now we are getting 99% across the board.
- accuracy score: 0.9997733111682031, indicates the proportion of correctly predicted outcomes out of all predictions made by the model.
- precision score: 0.9997746563509352, indicates the proportion of true positive predictions out of all positive predictions made by the model.
- recall score: 0.9997746563509352, indicates the proportion of true positive predictions out of all actual positive outcomes in the test dataset.
- F1 score: 0.9997746563509352, is the harmonic mean of precision and recall and represents the balance between precision and recall of the model.

In summary, the XGBoost model achieved good performance with high accuracy, precision, recall, and F1 scores on the test dataset, indicating that it can effectively predict the outcomes of NCAA Women's Basketball games.

We pulled the classification report and features importance, interestingly enough, after all the extra calculations that we did, the most important features were:
- Winning team score
- Losing team score
- Score Gap

Finally, I created the 2023 dataframe to be run through our predictio model. However, after running the model it unfortunately seems that we keep getting the same result for every matchup, so either we do have to many redundant datafeatures in our final dataset or we need to try out some different code to get our desired result.

Ran a couple of other models, but they were giving us the same result, a train of 1.0 and validation of about 0.99.

Our confusion matrix also looks good, as it seems to be performing quite well! It looks like most of our instances were correctly classified.


## Conclusion

In conclusion, it seemed that we were able to make a pretty accurate model with great train and test scores, but I will need to dig deeper into the prediction of unseen data to really determine how accurate our model truly is.

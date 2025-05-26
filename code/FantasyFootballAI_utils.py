# Importing packages
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from groq import Groq
import datetime
import time
from tqdm import tqdm
from matplotlib import pyplot as plt
import re
import requests
from bs4 import BeautifulSoup
import nfl_data_py as nfl


import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email import encoders



# Defining Winner/Loser of the week
def weekly_winner_loser(df, week, regular_season_max_week=18):
    """
    Takes in the Scores DataFrame and returns the DataFrame
    with winner and loser columns added for games up to the specified week.

    Parameters:
        df (DataFrame): Input DataFrame with columns ['week', 'away_team', 'home_team', 'away_score', 'home_score']
        week (int): Week number to filter scores up to (inclusive)
        regular_season_max_week (int): Maximum week number for the regular season (default is 18)

    Returns:
        DataFrame: Original DataFrame filtered by week with 'winner' and 'loser' columns appended
    """
    
    # Subsetting to the latest week in the season
    scores = df[(df.week <= week) & (df.week <= regular_season_max_week)]

    # Calculating which team won which game
    scores['winner'] = np.where(scores.away_score > scores.home_score, scores.away_team,
                            np.where(scores.away_score < scores.home_score, scores.home_team, 
                                    scores.away_team + '_' + scores.home_team + '_TIE'))

    scores['loser'] = np.where(scores.away_score < scores.home_score, scores.away_team,
                            np.where(scores.away_score > scores.home_score, scores.home_team, 
                                    scores.away_team + '_' + scores.home_team + '_TIE'))
    
    return scores

# Defining total wins by team
def season_wins_func(df):
    """
    Returns a DataFrame with total wins and losses for each team in the season.

    Parameters:
        df (DataFrame): Input DataFrame with columns ['winner', 'loser', 'week'].

    Returns:
        DataFrame: DataFrame with columns ['team', 'wins', 'losses'].
    """

    # Dataframe of all teams
    teams = ['BAL', 'GB', 'PIT', 'ARI', 'TEN', 'NE', 'HOU', 'JAX', 'CAR', 'MIN', 'LV', 'DEN', 'DAL', 'WAS', 'LA', 'NYJ',
             'KC', 'PHI', 'ATL', 'BUF', 'CHI', 'CIN', 'IND', 'MIA', 'NO', 'NYG', 'LAC', 'SEA', 'CLE', 'TB', 'DET', 'SF']
    
    # Compute wins and losses
    wins = df['winner'].value_counts().reindex(teams, fill_value=0)
    losses = df['loser'].value_counts().reindex(teams, fill_value=0)
    
    # Combine results into a single DataFrame
    season_wins = pd.DataFrame({
        'team': teams,
        'wins': wins.values,
        'losses': losses.values
    })

    return season_wins


# Defining function for playoff points if there are any
def playoff_points(df, season_wins):
    """
    Takes scores DataFrame and season wins DataFrame, and computes playoff points for each team.

    Points system:
    - Wildcard Round appearance: 3 points
    - Divisional Round appearance: 3 points
    - Conference Round appearance: 5 points
    - Super Bowl appearance: 5 points
    - Division winner: 1 point

    Returns:
        Updated season_wins DataFrame with playoff points and total_points columns.
    """
    # Initialize points columns
    season_wins['wildcard'] = 0
    season_wins['divisional'] = 0
    season_wins['conference'] = 0
    season_wins['superbowl'] = 0
    season_wins['division_winner'] = 0

    # Wildcard round (week 19)
    wildcard_teams = set(df.loc[df.week == 19, ['away_team', 'home_team']].stack())
    
    # If season is at least wildcard week, find top teams not already included
    if df.week.max() >= 19:
        additional_teams = (
            season_wins.loc[~season_wins.team.isin(wildcard_teams)]
                        .nlargest(2, 'wins')['team']
        )
        wildcard_teams.update(additional_teams)

    season_wins.loc[season_wins.team.isin(wildcard_teams), 'wildcard'] = 3

    # Division Winners: Home teams week 19 plus home teams week 20 not playing in wildcard
    home_teams_week19 = set(df.loc[df.week == 19, 'home_team'])
    home_teams_week20 = set(df.loc[df.week == 20, 'home_team'])
    wildcard_participants = set(df.loc[df.week == 19, ['away_team', 'home_team']].stack())

    division_winners = home_teams_week19.union(
        team for team in home_teams_week20 if team not in wildcard_participants
    )
    
    season_wins.loc[season_wins.team.isin(division_winners), 'division_winner'] = 1

    # Points per playoff round dictionary
    playoff_rounds = {
        'divisional': (20, 3),
        'conference': (21, 5),
        'superbowl': (22, 5)
    }

    for round_name, (week, points) in playoff_rounds.items():
        teams = set(df.loc[df.week == week, ['away_team', 'home_team']].stack())
        season_wins.loc[season_wins.team.isin(teams), round_name] = points

    # Calculate total points
    points_cols = ['wins', 'wildcard', 'divisional', 'conference', 'superbowl', 'division_winner']
    season_wins['total_points'] = season_wins[points_cols].sum(axis=1)

    return season_wins


# Defining function to get total points from Bill and Mom over the course of the season
def total_points_func(fantasy_team_df, season_wins):
    """
    Takes in the scores dataframe and returns the total points for each team
    """

    # Caluclating Fantasy Team Points
    season_wins2 = season_wins.merge(fantasy_team_df, how='inner', on='team')
    fantasy_wins = season_wins2.groupby('Player')['total_points'].sum().reset_index().sort_values(['total_points', 'Player'], ascending=[False, True])
    
    return fantasy_wins


# Defining the function to get key games (upsets and head-to-heads)
def key_games_func(df, week, fantasy_team_df):
    """
    Takes in the scores dataframe and returns the key games for the week
    Based on upsets and head-to-head matchups
    """
    # Looking at scores from just this week
    scores_week = df[df.week == week].drop(columns = ['referee', 'away_spread_odds', 'result', 'total',
                                                            'home_spread_odds', 'total_line', 'under_odds', 'over_odds'])
    # Getting the biggest Underdog
    scores_week['moneyline_spread'] = abs(scores_week.away_moneyline) + abs(scores_week.home_moneyline)

    # Checking if either Squad has one of the teams
    scores_week['mom_team'] = np.where((scores_week.away_team.isin(fantasy_team_df[fantasy_team_df.Player == 'Mom'].team)) | 
                                    (scores_week.home_team.isin(fantasy_team_df[fantasy_team_df.Player == 'Mom'].team)), 1, 0)
    scores_week['bill_team'] = np.where((scores_week.away_team.isin(fantasy_team_df[fantasy_team_df.Player == 'Bill'].team)) | 
                                    (scores_week.home_team.isin(fantasy_team_df[fantasy_team_df.Player == 'Bill'].team)), 1, 0)

    # Flag for if Bill, Mom, or Unpicked Won
    scores_week['mom_win'] = np.where( (scores_week.away_score > scores_week.home_score) & 
                                    (scores_week.away_team.isin(fantasy_team_df[fantasy_team_df.Player == 'Mom'].team)) | 
                                    (scores_week.home_score > scores_week.away_score) & 
                                    (scores_week.home_team.isin(fantasy_team_df[fantasy_team_df.Player == 'Mom'].team)), 1 , 0)
    scores_week['bill_win'] = np.where( (scores_week.away_score > scores_week.home_score) & 
                                    (scores_week.away_team.isin(fantasy_team_df[fantasy_team_df.Player == 'Bill'].team)) | 
                                    (scores_week.home_score > scores_week.away_score) & 
                                    (scores_week.home_team.isin(fantasy_team_df[fantasy_team_df.Player == 'Bill'].team)), 1 , 0)
    scores_week['unpicked_win'] = np.where( (scores_week.away_score > scores_week.home_score) & 
                                    (scores_week.away_team.isin(fantasy_team_df[fantasy_team_df.Player == 'Unpicked'].team)) | 
                                    (scores_week.home_score > scores_week.away_score) & 
                                    (scores_week.home_team.isin(fantasy_team_df[fantasy_team_df.Player == 'Unpicked'].team)), 1 , 0)

    # Flag for Upset:
    # If away team > home team AND away team moneyline > home moneyline AND away moneyline > 150 AND home moneyline < -150
    # And then vice versa for home team > away team
    scores_week['upset'] = np.where( ((scores_week.away_score > scores_week.home_score) & 
                                    (scores_week.away_moneyline > 150) & (scores_week.home_moneyline < -150))  | 
                                ((scores_week.home_score > scores_week.away_score) & 
                                (scores_week.home_moneyline > 150) & (scores_week.away_moneyline < -150)), 1, 0)

    # Sort by Biggest
    scores_week = scores_week.sort_values('moneyline_spread', ascending=False)

    # Getting Bill vs Mom head to head matchups:
    head_to_heads = scores_week[(scores_week.mom_team == 1) & (scores_week.bill_team == 1)]

    # Biggest Upsets
    upsets = scores_week[((scores_week.mom_team == 1) | (scores_week.bill_team == 1)) & (scores_week.upset == 1)]


    return scores_week, head_to_heads, upsets


# Defining the cumulative win graphic function
def cum_win_func(weekly_wins):
    """
    Takes in the scores dataframe and returns a cumulative win graphic
    """
 
    # Getting weekly Wins by team
    weekly_wins2 = weekly_wins.groupby(['Player', 'week'])[['home_team']].count().reset_index()
    weekly_wins2 = weekly_wins2.sort_values(['Player', 'week'], ascending=[True, True])
    weekly_wins2.columns = ['Player', 'week', 'wins']

    # Cumulative Wins
    weekly_wins2['cumsum'] = weekly_wins2.groupby(['Player'])['wins'].transform(pd.Series.cumsum)

    # Wins per team per week
    weekly_wins2['number_of_teams'] = np.where(weekly_wins2.Player == 'Unpicked', 12, 10)
    weekly_wins2['Cum_Wins_Per_Team'] = weekly_wins2['cumsum'] / weekly_wins2['number_of_teams']

    # Plotting
    plt.figure(figsize=(15, 10))

    # Define custom colors for teams
    color_mapping = {
        'Bill': 'purple',
        'Mom': 'orange',
        'Unpicked': 'green'
    }

    for team, group in weekly_wins2.groupby('Player'):
        plt.plot(group['week'], group['Cum_Wins_Per_Team'], label=team, color=color_mapping[team], linewidth=2.5)

        # Add text labels at the end of each line
        plt.text(
            group['week'].iloc[-1] + .1,                # x-coordinate: last week
            group['Cum_Wins_Per_Team'].iloc[-1],  # y-coordinate: last cumulative wins
            f"{group['Cum_Wins_Per_Team'].iloc[-1]:.2f}",  # Text to display
            color=color_mapping[team],            # Color matching the line
            fontsize=12,                          # Adjust font size
            va='center',                          # Vertical alignment
            ha='left'                             # Horizontal alignment
        )


    # Customize the plot
    plt.title('Cumulative Wins Per Team by Week', fontsize=16)
    plt.xlabel('Week', fontsize=12)
    plt.ylabel('Cumulative Wins', fontsize=12)
    plt.legend(title='Team', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(range(1, weekly_wins2.week.max()+1))


    # # Show the plot
    weekly_wins_filename = f'Weekly_Wins_Week_{weekly_wins2.week.max()}.png'
    # plt.savefig(fname=weekly_wins_filename)

    return plt, weekly_wins_filename


def top_games_func(scores_week, important_games):
    """
    Returns the top games of the week based on custom algorithm.
    Also returns the rest of the games
    """

    # Sort Order to focus on the top 3 games: Games that involve Bill + Mom, Then Upsets, then Final Score
    important_games['sort_order'] = important_games['mom_team'] * 100 +\
                                    important_games['bill_team']*100 +\
                                    important_games['upset'] * 50 +\
                                    important_games['div_game'] * 25 +\
                                    important_games['home_score'] + important_games['away_score']
    important_games = important_games.sort_values('sort_order', ascending=False)
    top_games = important_games.iloc[:3, :].reset_index().drop(columns='index')

    # Rest of the games (for quick recap)
    rest_of_the_games = scores_week[(~scores_week.away_team.isin(top_games.away_team)) & 
                                    (~scores_week.home_team.isin(top_games.home_team))
                                    & ((scores_week.mom_team == 1) | (scores_week.bill_team == 1))].reset_index().drop(columns='index')

    return top_games, rest_of_the_games


# Defining all the string creations for each game and player stat
# Returning the final score for a game
def final_score(game, fantasy_team_df2, team_longname_dict):
    home_team = game['home_team']
    away_team = game['away_team']

    home_points = round(game['home_score'], 0)
    away_points = round(game['away_score'], 0)

    # Who won?
    if home_points > away_points:
        
        winner = home_team
        winner_points = home_points
        loser = away_team
        loser_points = away_points
    
    elif home_points < away_points:
        
        winner = away_team
        winner_points = away_points
        loser = home_team
        loser_points = home_points

    
    else:
        winner = 'Tie'

    winner_owner = fantasy_team_df2[fantasy_team_df2.team == winner]['Player'].values[0]
    loser_owner = fantasy_team_df2[fantasy_team_df2.team == loser]['Player'].values[0]

    if winner != 'Tie':
    
        final_score_string = f"""\
{winner_owner}'s {team_longname_dict[winner]} defeated {loser_owner}'s {team_longname_dict[loser]} by a score of \
{winner_points} to {loser_points}. 
"""
    else:
        # If there's a tie
        final_score_string = f"""The {team_longname_dict[home_team]} and The {team_longname_dict[away_team]} tied with a score of \
{home_points} to {away_points}"""

    #Adding an Upset Flag
    upset_flag = int(game['upset'])
    if upset_flag == 1:
        point_spread = game['spread_line']
        if point_spread < 0:
            # AWAY TEAM FAVORED
            final_score_string += f'This was an upset, as The {team_longname_dict[away_team]} were favored by {-point_spread} points'
        else:
            # HOME TEAM FAVORED
            final_score_string += f'This was an upset, as The {team_longname_dict[home_team]} were favored by {point_spread} points'
    
    return final_score_string

# Defining QB Stats for a game
def qb_stats(game, weekly_player, team_longname_dict, home=True):
    home_team = game['home_team']
    away_team = game['away_team']

    home_qb_stats = weekly_player[(weekly_player.recent_team == home_team) & 
                    (weekly_player.position == 'QB')].sort_values('passing_yards', ascending=False).reset_index().\
                    drop(columns='index').iloc[0, :]
    away_qb_stats = weekly_player[(weekly_player.recent_team == away_team) & 
                    (weekly_player.position == 'QB')].sort_values('passing_yards', ascending=False).reset_index().\
                    drop(columns='index').iloc[0, :]

    # Names
    home_qb_name = home_qb_stats['player_display_name']
    away_qb_name = away_qb_stats['player_display_name']

    # Passing Yards
    home_pass_yards = home_qb_stats['passing_yards']
    away_pass_yards = away_qb_stats['passing_yards']

    # Completion Percentage
    home_comp_per = round(home_qb_stats['completions'] / home_qb_stats['attempts'] * 100, 2)
    away_comp_per = round(away_qb_stats['completions'] / away_qb_stats['attempts'] * 100, 2)

    # Yards Per Attempt
    home_yrds_per_att = round(home_pass_yards / home_qb_stats['attempts'], 2)
    away_yrds_per_att = round(away_pass_yards / away_qb_stats['attempts'], 2)
    
    # TDs
    home_pass_tds = home_qb_stats['passing_tds']
    away_pass_tds = away_qb_stats['passing_tds']

    # INTs
    home_ints = home_qb_stats['interceptions']
    away_ints = away_qb_stats['interceptions']

    # Sacks
    home_sacks = home_qb_stats['sacks']
    away_sacks = away_qb_stats['sacks']

    # Fumbles
    home_fumbles = home_qb_stats['sack_fumbles_lost'] + home_qb_stats['rushing_fumbles_lost'] + home_qb_stats['receiving_fumbles_lost']
    away_fumbles = away_qb_stats['sack_fumbles_lost'] + away_qb_stats['rushing_fumbles_lost'] + away_qb_stats['receiving_fumbles_lost']

    # Rush Yards
    home_rush_yds = home_qb_stats['rushing_yards']
    away_rush_yds = away_qb_stats['rushing_yards']

    # Rush Attempts
    home_carries = home_qb_stats['carries']
    away_carries = away_qb_stats['carries']

    # Rush TDs
    home_rush_tds = home_qb_stats['rushing_tds']
    away_rush_tds = away_qb_stats['rushing_tds']

    # Receptions
    home_receptions = home_qb_stats['receptions']
    away_receptions = away_qb_stats['receptions']

    # Targets
    home_targets = home_qb_stats['targets']
    away_targets = away_qb_stats['targets']

    # Receiving Yards
    home_rec_yds = home_qb_stats['receiving_yards']
    away_rec_yds = away_qb_stats['receiving_yards']

    # Receiving TDs
    home_rec_tds = home_qb_stats['receiving_tds']
    away_rec_tds = away_qb_stats['receiving_tds']

    home_qb_stats_string = f"""

The {team_longname_dict[home_team]} Quarterback, {home_qb_name}, threw for {home_pass_yards} yards, with {home_pass_tds} touchdowns \
and {home_ints} interceptions, at {home_yrds_per_att} yards per attempt and a completion percentage of {home_comp_per}%. """
    
    away_qb_stats_string = f"""

The {team_longname_dict[away_team]} Quarterback, {away_qb_name}, threw for {away_pass_yards} yards, with {away_pass_tds} touchdowns \
and {away_ints} interceptions, at {away_yrds_per_att} yards per attempt and a completion percentage of {away_comp_per}%. """

    # If significant rushing
    if (home_rush_yds > 20) | (home_rush_tds > 0):
        home_rush_adder = f"""\
He also added {home_rush_yds} yards rushing on {home_carries} attempts, \
with {home_rush_tds} rushing touchdowns."""

        # Adding the string
        home_qb_stats_string += home_rush_adder
    
    if (away_rush_yds > 20) | (away_rush_tds > 0):
        away_rush_adder = f"""\
He also added {away_rush_yds} yards rushing on {away_carries} attempts, \
with {away_rush_tds} rushing touchdowns."""

        # Adding the string
        away_qb_stats_string += away_rush_adder


    # If significant receiving
    if home_receptions > 0:
        home_rec_adder = f"""\
He also added {home_rec_yds} yards receiving on {home_receptions} catches, \
with {home_rec_tds} receiving touchdowns."""

        # Adding the string
        home_qb_stats_string += home_rec_adder

    if away_receptions > 0:
        away_rec_adder = f"""\
He also added {away_rec_yds} yards receiving on {away_receptions} catches, \
with {away_rec_tds} receiving touchdowns."""

        # Adding the string
        away_qb_stats_string += away_rec_adder
    
    if home:
        return home_qb_stats_string
    else:
        return away_qb_stats_string
    
# Defining other player stats
# Need to get leading receiver, and leading rusher and all with TDs
def player_stats(game, weekly_player, team_longname_dict, home=True):
    if home:
        team = game['home_team']
    else:
        team = game['away_team']

    team_stats = weekly_player[(weekly_player.recent_team == team)]

    # Get leading rusher
    lead_rush_name = team_stats\
                    .sort_values('rushing_yards', ascending=False).reset_index().iloc[0,:]['player_display_name']
    
    lead_rush_yds = int(team_stats\
                    .sort_values('rushing_yards', ascending=False).reset_index().iloc[0,:]['rushing_yards'])
    
    lead_rush_carries = int(team_stats\
                    .sort_values('rushing_yards', ascending=False).reset_index().iloc[0,:]['carries'])
    
    lead_rush_tds = int(team_stats\
                    .sort_values('rushing_yards', ascending=False).reset_index().iloc[0,:]['rushing_tds'])
    

    # Get leading Receiver
    lead_rec_name = team_stats\
                    .sort_values('receiving_yards', ascending=False).reset_index().iloc[0,:]['player_display_name']
    
    lead_rec_yds = int(team_stats\
                    .sort_values('receiving_yards', ascending=False).reset_index().iloc[0,:]['receiving_yards'])
    
    lead_rec_receptions = int(team_stats\
                    .sort_values('receiving_yards', ascending=False).reset_index().iloc[0,:]['receptions'])
    
    lead_rec_tds = int(team_stats\
                    .sort_values('receiving_yards', ascending=False).reset_index().iloc[0,:]['receiving_tds'])

    
    # Get all others who eclipsed 100 rush yards (WHO AREN'T THE ABOVE)
    other_100_rush_yards = team_stats[(team_stats.rushing_yards >= 100) & (team_stats.player_display_name != lead_rush_name)]
    
    big_rusher_names = other_100_rush_yards['player_display_name'].values
    big_rusher_yds = other_100_rush_yards['rushing_yards'].values
    big_rusher_carries = other_100_rush_yards['carries'].values
    big_rusher_tds = other_100_rush_yards['rushing_tds'].values

    rush_add_string = ''
    if len(big_rusher_names) > 0:
        rush_add_string = 'Additionally, '
        for i in range(len(big_rusher_names)):
            rush_add_string += f"""{big_rusher_names[i]} added {big_rusher_yds[i]} yards on {big_rusher_carries[i]} \
carries with {big_rusher_tds[i]} touchdowns. """

    # Get all others who eclipsed 100 rec yards (WHO AREN'T THE ABOVE)
    other_100_rec_yards = team_stats[(team_stats.receiving_yards >= 100) & (team_stats.player_display_name != lead_rec_name)]
    
    big_receiver_names = other_100_rec_yards['player_display_name'].values
    big_receiver_yds = other_100_rec_yards['receiving_yards'].values
    big_receiver_receptions = other_100_rec_yards['receptions'].values
    big_receiver_tds = other_100_rec_yards['receiving_tds'].values

    rec_add_string = ''
    if len(big_receiver_names) > 0:
        rec_add_string = 'Additionally, '
        for i in range(len(big_receiver_names)):
            rec_add_string += f"""{big_receiver_names[i]} added {big_receiver_yds[i]} yards on {big_receiver_receptions[i]} \
receptions with {big_receiver_tds[i]} touchdowns. """

    
    # Get all others who scored TDs (WHO AREN'T THE ABOVE)
    rush_tuddys = team_stats[(team_stats.rushing_tds > 0) & (team_stats.player_display_name != lead_rush_name)]
    rec_tuddys = team_stats[(team_stats.receiving_tds > 0) & (team_stats.player_display_name != lead_rec_name)]
    
    other_rush_td_names = rush_tuddys['player_display_name'].values
    other_rush_tds = rush_tuddys['rushing_tds'].values
    other_rec_td_names = rec_tuddys['player_display_name'].values
    other_rec_tds = rec_tuddys['receiving_tds'].values

    rush_td_add_string = ''
    if len(other_rush_td_names) > 0:
        rush_td_add_string = 'Additionally, '
        for i in range(len(other_rush_td_names)):
            rush_td_add_string += f"""{other_rush_td_names[i]} had {other_rush_tds[i]} rushing touchdowns for The {team_longname_dict[team]}.
"""

    rec_td_add_string = ''
    if len(other_rec_td_names) > 0:
        rec_td_add_string = 'Additionally, '
        for i in range(len(other_rec_td_names)):
            rec_td_add_string += f"""{other_rec_td_names[i]} had {other_rec_tds[i]} receiving touchdowns for The {team_longname_dict[team]}.
"""


    final_string = f"""
The leading rusher for {team_longname_dict[team]} was {lead_rush_name}, who carried {lead_rush_carries} times for {lead_rush_yds} \
yards ({round(lead_rush_yds / lead_rush_carries, 2)} yards per attempt) and {lead_rush_tds} touchdowns.

The leading receiver for {team_longname_dict[team]} was {lead_rec_name}, who had {lead_rec_receptions} receptions for {lead_rec_yds} \
yards and {lead_rec_tds} touchdowns.
"""

    final_string = final_string + rush_add_string + rec_add_string + rush_td_add_string + rec_td_add_string

    return final_string


# Combining all the strings for a game together to create a game summary
def final_compiler(game, fantasy_team_df2, team_longname_dict, weekly_player):

    final_final_string = final_score(game, fantasy_team_df2, team_longname_dict) + qb_stats(game, weekly_player, team_longname_dict, True) + qb_stats(game, weekly_player, team_longname_dict, False) +\
                        player_stats(game, weekly_player, team_longname_dict, True) + player_stats(game, weekly_player, team_longname_dict, False)
    return final_final_string

# Defining Function to get the rest of the games' summaries
def rest_of_the_games_func(rest_of_the_games, fantasy_team_df2, team_longname_dict):
    rest_of_games_recap = ''

    for i in range(len(rest_of_the_games)-1):
        my_game = rest_of_the_games.iloc[i]

        rest_of_games_recap += '\n' + final_score(my_game, fantasy_team_df2, team_longname_dict) + '\n'
    
    return(rest_of_games_recap)


# Defining Email formatting function
def format_email_content(content, font_size=24):
    # Split the content at the first <br><br> (or newline) to identify the first line
    parts = content.split("<br><br>", 1)
    
    # Apply the font size only to the first part (first line)
    if len(parts) > 1:
        first_line = f'<p style="font-size: {font_size}px;">{parts[0]}</p>'
        rest_of_content = parts[1]
        return first_line + "<br><br>" + rest_of_content
    else:
        # If no <br><br> found, return the whole content with font size
        return f'<p style="font-size: {font_size}px;">{content}</p>'
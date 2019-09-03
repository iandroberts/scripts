import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import os
import time
import pickle
import sys

'''
argument 1 : path to player name text file
argument 2 (optional) : path to player position preferences pickle file
argument 3 (optional) : number of iterations to run
'''

###########

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def make_frames(player_names, player_mf, inning_list, position_names, fill_order, positions):
    '''Function to generate random lineup dataframe, assigning a player to each
       position for each inning.
    
       Input: array of player names, array of player genders, array of inning names,
              array of position names

       Output: dataframe assigning a player id to each position for each inning,
               dataframe of genders for each position assignment
    '''
    # Initiate dataframe to fill
    frame_players = pd.DataFrame(-1, index=position_names, columns=inning_list)

    # Loop through innings and positions
    for i in range(inning_list.size):
        players_on_field = []
        positions_already_filled = []

        names_f = player_names[player_mf == 1]
        names_f_rand = np.random.choice(names_f, size=3, replace=False)
        sizes_f = np.array([positions['{}'.format(names_f_rand[0])].size, positions['{}'.format(names_f_rand[1])].size, positions['{}'.format(names_f_rand[2])].size])
        names_f_rand = names_f_rand[np.argsort(sizes_f)]

        for play in names_f_rand:
            these_pos = positions['{}'.format(play)][positions['{}'.format(play)]>=0]
            mask_pos = np.isin(these_pos, np.asarray(positions_already_filled))
            rand_pos = np.random.choice(these_pos[~mask_pos], size=1, replace=False)[0]
            frame_players.iloc[rand_pos,i] = play
            positions_already_filled.append(rand_pos)
            players_on_field.append(play)

        mask_fill = np.isin(fill_order, np.asarray(positions_already_filled))
        
        for j in fill_order[~mask_fill]: # Fill positions in specified order
            players_for_this_position = []
            for play in player_names:
                ind1 = np.where(positions['{}'.format(play)] == j)[0]
                if ind1.size > 0: # Find players eligible for current position
                    players_for_this_position.append(play)

            mask_on_field = np.isin(np.asarray(players_for_this_position), np.asarray(players_on_field)) # Make sure players aren't already on field
            rand_player = np.random.choice(np.asarray(players_for_this_position)[~mask_on_field], size=1, replace=False)[0] # Select player randomly
            frame_players.iloc[j,i] = rand_player # Add to dataframe
            players_on_field.append(rand_player)

    return frame_players
    

def likelihood(frame_players, player_names, inning_list, positions):
    '''Function to evaluate the 'likelihood' of the randomly generated
       lineup being optimal.  The lineup that will be selected is the
       one which gives the highest 'likelihood'.  Likelihood is determined
       by the following factors:

       - Are three females on field at all times?
       - Are all players playing roughly an equal number of innings?
       - Are players playing equal innings in infield and outfield?
    '''
    # Define factors used as penalties
    inning_cts_factor = 100.
    outfield_infield_factor = 5.
    bench_spacing_factor = 25.

    # Initiate arrays for later
    player_cts = np.zeros(len(player_names))
    infield_cts = np.zeros(len(player_names))
    outfield_cts = np.zeros(len(player_names))
    bench_spacing = np.zeros(len(player_names))

    like = 0 # Initiate likelihood

    infield = frame_players.iloc[0:5,:]
    outfield = frame_players.iloc[5:,:]

    for i in range(len(players)):
        innings_on_field = np.sort(np.where(frame_players == player_names[i])[1]+1)
        diffs = np.diff(innings_on_field, append=9)
        bench_spacing[i] = np.var(diffs)
        
        player_cts[i] = innings_on_field.size
        infield_cts[i] = np.where(infield == player_names[i])[1].size
        outfield_cts[i] = np.where(outfield == player_names[i])[1].size

    cts_var = np.var(player_cts)
    like -= inning_cts_factor * np.exp(cts_var) # Likelihood penalty for uneven innings played

    inf_frac = np.round(infield_cts / (infield_cts + outfield_cts), 2)
    like -= np.sum(np.abs(0.5 - inf_frac)) * outfield_infield_factor # Likelihood penalty for uneven IF/OF split

    #like -= np.sum(bench_spacing) * bench_spacing_factor # Likelihood penalty for uneven bench/on-field spacing

    return like, player_cts, inf_frac

#############
'''
0 -> C
1 -> 1B
2 -> 2B
3 -> 3B
4 -> SS
5 -> LF
6 -> CF 
7 -> RF
8 -> SL
9 -> SR
'''

fill_order = np.array([4,1,0,6,7,3,8,9,5,2])

#############

players = np.genfromtxt(sys.argv[1], dtype=str, delimiter=',', usecols=(0,),
                        unpack=True)

if len(sys.argv) > 2:
    positions = load_obj(sys.argv[2])
else:
    print("")
    positions_file = input("Path to player preferences file? ")
    positions = load_obj(positions_file)
    print("")

if len(sys.argv) > 3:
    Nit = int(sys.argv[3])
else:
    Nit = 500

no_of_players = players.size

player_mf = np.zeros(no_of_players)
for i in range(len(players)):
    player_mf[i] = positions['{}'.format(players[i])][-1]

player_mf[player_mf==-2] = 1
player_mf[player_mf==-1] = 0

inning_list = np.array(['inn1','inn2','inn3','inn4','inn5','inn6','inn7','inn8','inn9'])
position_names = np.array(['C','1B','2B','3B','SS','LF','CF','RF','SL','SR'])

like = -np.inf

for it in trange(Nit, desc='Iterating to find optimal lineup'):
    frame_ids = make_frames(players, player_mf, inning_list, position_names, fill_order, positions)
    like_new, counts, out_in = likelihood(frame_ids, players, inning_list, positions)

    if like_new > like:
        final_lineup = frame_ids
        final_counts = counts
        final_out_in = out_in
        like = like_new

time.sleep(2)

print("")
print("### Lineup ###")
print("")
print(final_lineup)

time.sleep(2)

print("")
print("### Constraint results ###")
print("")
print(pd.DataFrame({'No. of innings': final_counts, 'Fraction of IF innings': final_out_in}, index=players))
print("")

time.sleep(2)

print("")
print("Likelihood: ", like)
print("")

#time.sleep(2)

#print("### Creating latex file ###")
#print("")

#with open('lineup_table.tex','w') as tf:
    #tf.write(final_lineup.to_latex())

#os.system("pdflatex lineup.tex")


"""
Created on Thu May  7 23:45:09 2020

@author: loren
"""
import numpy.random as rnd
import numpy as np
import pandas as pd
import sys
import seaborn as sns
import matplotlib.pyplot as plt
pd.options.display.float_format = '{:.2%}'.format

def check_length(*args):
        args = [[x] if type(x) == int else x for x in args] 
        max_length = max([len(x) for x in args])
        if any([len(x) not in [1, max_length] for x in args]):
            print ('please use consistent lengths for the input of length higher than 1!') 
            print([len(x) for x in args])
            sys.exit()

        else:
            args = [x * max_length if len(x) == 1 else x for x in args] 
        return args #np.unique(args, axis = 1)
    
def play_monty_hall(num_total_doors = 3, num_winning_doors = 1, num_doors_to_play = 1, num_doors_to_open = 1, num_doors_to_switch = 1, trials = 10000):
    """ (int, int, int, int, int, int) -> (int, int, int, int, int, int)
    Simulates playing monty_hall. Returns win rate if switch and if not_switchs. 
    """
    #accumulator for win/lose
    win_switch, win_no_switch = 0, 0, 
    #calculate some parameters
    num_doors_to_keep = num_doors_to_play - num_doors_to_switch
    
    #check consistency of parameters
    check_list = list() #list to check consistency of initial set up
    check_list.append(["num_doors_to_play + num_doors_to_open + num_doors_to_switch <= num_total_doors", num_doors_to_play + num_doors_to_open + num_doors_to_switch <= num_total_doors])
    check_list.append(["num_doors_to_play + num_doors_to_open + num_winning_doors <= num_total_doors", num_doors_to_play + num_doors_to_open + num_winning_doors <= num_total_doors])
    check_list.append(["num_doors_to_switch <= num_doors_to_play", num_doors_to_switch <= num_doors_to_play])
    
    if any(x[1] == False for x in check_list):
        print('please ensure that the initial parameters are consistent: ')
        print(pd.DataFrame(check_list, columns = ['VALUE', 'CONDITION']).to_string(index = False))
        sys.exit()
    
    for i in range(trials):
        choices = np.arange(num_total_doors)
        winning_doors = rnd.choice(choices, size = num_winning_doors, replace = False) #set which doors win
        initial_choices = rnd.choice(choices, size = num_doors_to_play, replace = False) #player select first choice
        open_choices = np.delete(choices, np.union1d(winning_doors,initial_choices)) #remove winning doors and first choice
        doors_to_open = rnd.choice(open_choices, size = num_doors_to_open, replace = False) #choose doors to open from previous line
        second_choices = np.delete(choices, np.union1d(doors_to_open,initial_choices)) #remove winnin doors which were initial choice
        
        #update choices - switch
        updated_choices = rnd.choice(second_choices, size = num_doors_to_switch, replace = False)
        if num_doors_to_keep > 0 :
            choices_to_keep = rnd.choice(initial_choices, size = num_doors_to_keep, replace = False)
            updated_choices = np.union1d(updated_choices,choices_to_keep) 
        
        #accumulate points
        if any(i in winning_doors for i in updated_choices):
             win_switch += 1
        
        #use initial choices - no switch
        if any(i in winning_doors for i in initial_choices):
             win_no_switch += 1
    
    return win_switch/trials, win_no_switch/trials, num_total_doors, num_winning_doors, num_doors_to_open, num_doors_to_play, num_doors_to_switch

def simulate_monty_hall(num_total_doors = [3], 
                        num_winning_doors = [1], 
                        num_doors_to_play = [1], 
                        num_doors_to_open = [1], 
                        num_doors_to_switch = [1], 
                        trials = [10000]):

    num_total_doors, num_winning_doors, num_doors_to_play, num_doors_to_open, num_doors_to_switch, trials = check_length(num_total_doors, num_winning_doors, num_doors_to_play, num_doors_to_open, num_doors_to_switch, trials)
    
    #simuation
    data_collection = []
    for total_doors, winning_doors, doors_to_play, doors_to_open, doors_to_switch, trials in zip(num_total_doors, num_winning_doors, num_doors_to_play, num_doors_to_open, num_doors_to_switch, trials):
        data_collection.append(play_monty_hall(
                           num_total_doors = total_doors, 
                           num_winning_doors = winning_doors, 
                           num_doors_to_play = doors_to_play, 
                           num_doors_to_open = doors_to_open, 
                           num_doors_to_switch = doors_to_switch, 
                           trials = trials))
    return pd.DataFrame(data_collection, 
                        columns = ['switch', 'no_switch', 'total doors', 'winning doors', 'doors opened', 'doors chosen', 'doors switched'])

def compute_monty_hall(n = [3], w = [1], o = [1]):
    
    n, w, o = check_length(n, w, o)

    return pd.DataFrame(
                    np.array([np.multiply(np.divide(w,n),np.divide(np.subtract(n,1),np.subtract(n,o)-1)), #win rate by switching
                    np.divide(w,n), #win rate no switch
                    n, w, o]).transpose() #parameters
                    , columns = ['switch', 'no_switch', 'total doors', 'winning doors', 'doors opened']).astype({'switch': 'float64', 'no_switch': 'float64', 'total doors': 'int32', 'winning doors': 'int32', 'doors opened': 'int32'}).round(4)



def print_line_chart_monty_hall(win_rates, rng = None, varying_param = 'parameters', x_switch = 0, x_no_switch = 0, y_switch = 0, y_no_switch = 0, open_labels = False, size = (16,8)):
    if rng == None:
        rng = range(0, win_rates.shape[0])
    fig, ax = plt.subplots(figsize = size)
    plt.xticks(ticks = win_rates.index, labels = rng)
    plt.xlim(win_rates.index[0] - 1, win_rates.index[-1] + 1)
    plt.title('win rates with different number of {}'.format(varying_param))
    sns.lineplot(data = win_rates.loc[:,['switch', 'no_switch']], palette = ['darkgreen', 'darkred'], dashes = False, marker = 'o')
    plt.ylim(0, 1)
    
    if open_labels == True:
        for i in range(0, win_rates.shape[0]):
                plt.text(x = i + x_switch, 
                         y = win_rates['switch'].iloc[i] + y_switch,  
                         s = "{:.1%}".format(win_rates['switch'].iloc[i]),
                         color = 'darkgreen')
                plt.text(x = i + x_no_switch, 
                         y = win_rates['no_switch'].iloc[i] + y_no_switch,  
                         s = "{:.1%}".format(win_rates['no_switch'].iloc[i]),
                         color = 'darkred')

def main():
    pass
    
if __name__ == '__main__':     
    main()

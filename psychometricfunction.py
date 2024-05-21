import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, splrep, splev

plt.style.use(('seaborn-v0_8'))

"""HELPER FUNCTIONS"""
def get_y_axis(x, data):
    """ 
    calculates and gathers accuracy percentages per azimuth.
        args:
            x (List): Azimuth angles measured
            data (pandas df): Pandas Dataframe of original data
            num_resp (int): total number of responses 
        Returns:
            List of y values (Accuracies in percentage form)
    """
    
    y_axis = [] 

    numresp = get_num_resp(data, 'walker.azimuth')

    for item in x: # for each azimuth
            vals = data.loc[data['walker.azimuth']==item, 'response.responseScore'].sum() # gets response score
            # print(item, 'scored', vals) 
            total_correct = vals 
    
            # print(item, 'results in total', total_correct, 'correct responses') # "" add them together to get total responses
            total = numresp[item] #total number possible responses
            # print('The max number of possible correct was', total)
            accuracy = (int(total_correct)/total)
            y_axis.append(accuracy*100)
            print('Your accuracy for azimuth', item, 'was', accuracy*100, '%') # response accuracy 
            print()
    return y_axis
        # plt.plot(x, y_congruent, 'o--')

def get_num_resp(data, column):
    """ 
    gathers total # of responses per azimuth. Used to help calculate accuracies. 
        args:
            data (pandas df): Pandas Dataframe of original data
            column (str): Column name to retrieve desired data from, in this case 'walker.azimuth' to retrieve 
            total number of responses for each azimuth.
        returns:
            The number of responses per desired azimuth as a series
    """
    return data[column].value_counts()


def calc_smooth_amt(x):
    """ 
    calculates ideal amount of smoothing to apply to spline interpolation
    args:
        x (list): a list of length = to total number of data points
    returns:
        int representing ideal amount of smoothing based on total number of data points available
    """
    m = len(x)
    return round(m-np.sqrt(2*m))

def spline_interp(x, y, sample_num, calc_smooth=True):
    """
    performs spline interpolation on data
    args:
        x (list): azimuth
        y (list): accuracy percentages 
        sample_num (int): number of data points to generate with interpolation
        calc_smooth (boolean): whether to calculate ideal smoothing amount by default. Otherwise, enter desired smoothing amount.
    returns:
        Interpolated new x and y values
    """
    if calc_smooth==True:
        smooth_amt = calc_smooth_amt(x)
    else:
        smooth_amt=int(input('Input desired smoothing amount (int)'))
    spline_rep = splrep(x, y, s=smooth_amt) #cubic spline rep by default

    x_new = np.linspace(-11.5, 11.5, sample_num)
    y_new = splev(x_new, spline_rep)

    return (x_new, y_new)

def poly_interp(x, y, sample_num):
    """
    performs quadratic interpolation
    args:
        x (list): azimuth
        y (list): accuracy percentages
        sample_num (int): number of data points to generate with interpolation
    returns:
        Interpolated new x and y values
    """
    x_new = np.linspace(-11.5, 11.5, sample_num)

    new_x, new_y = add_moving_averages(x, y)

    f = interp1d(new_x, new_y, kind='quadratic', assume_sorted = True)

    y_interp = [f(val) for val in x_new]
    y_interp = [f(val) for val in x_new]

    return (x_new, y_interp) #to plot vals separately

def extract_min(condition, interp):
    """
    Extracts minimum based on y value interpolation
    args:
        file (str): filename
        interp (tuple): tuple resulting from interpolation functions, thus a tuple representing (new x values, new y values)
    returns:
        tuple holding point in which y is the global maximum (x, y)
    """

    print(f'Condition {condition}')

    min_i = np.argmin(interp[1])
    min_x = interp[0][min_i]
    min_y = interp[1][min_i]

    print('Minimum x:', min_x)
    print('Minimum y:', min_y)
    print()
    return (min_x, min_y)

        
def add_moving_averages(x, y):
    """
    Calculates data points using averages between angles to account for large gap of missing data
    args:
        x (List): azimuth
        y (List): accuracy percentages
    returns:
        Tuple holding revised list of x and y values. Revised meaning the addition of predicted poinst at angles between existing measured angle data points. 
    """
    assert len(x)==len(y), print('This data is not valid')
    """return (new x, new y)"""

    new_x = []
    new_y= []  #y arra yto get our vals from
    window_size = 2 #since we have like no data

    i = 0

    moving_averages = [] 

    while i < len(y) - window_size + 1:
        window = y[i : i+window_size]

        #average of window
        window_average = (sum(window) / window_size)

        moving_averages.append(window_average) #avg and index collecte

        i+=1
    
    #now create new y and x vals
    for i in range(len(y)-1):
        current_item = y[i]
        add_item = moving_averages.pop(0) #get first item 

        x_new = (x[i] + x[i+1]) / 2

        new_x.append(x[i])
        new_y.append(current_item)
        if (x_new == 0.0):
            continue

        new_y.append(add_item)
        new_x.append(round(x_new, 1))
    
    new_x.append(x[-1])
    new_y.append(y[-1])
    assert len(new_x) == len(new_y), print('Something went wrong')
    return (new_x, new_y)


def collect_conditions(files):
    """
    args:
        files (list of str): list of all filenames to be processed. Assumes all csv files are sorted by condition
    """
    unorganized_files = files[:] #copy of filelist names
    organized_files = []

    for i in range (1,7): #for conditions 1-6
        this_condition = []
        # print(unorganized_files)

        for file in unorganized_files: # now look through all files
            data = pd.read_csv(file+'.csv') #turn into df
            if i in data['trial.condition'].values:#if this file is corresponding condition
                this_condition.append(file) #append it to our list
                unorganized_files.remove(file) #then pop from unorganized_files to prevent looking at it in next iteration
                continue
        else:
            organized_files.append(this_condition)
    return organized_files #should now be organized as [[condition 1 files], [condition 2 files],...[condition 6 files]]

def concat_conditions(organized_files):
    """
    concatenate dataframes based on condition so we can flatten list to one dimension
    args:
        organized_files (list of str): list of list of filenames organized by condition
    returns:
        list of dataframes in order of condition 1-6
    """
    new_file_list = []
    i = 1

    for filenames in organized_files:
        if len(filenames)==0: #no data for this condition
            continue
        if len(filenames)==1:
            new_file_list.append(pd.read_csv(filenames[0]+'.csv'))
        else:
            df_list = [pd.read_csv(file+'.csv') for file in filenames]
            concatenated = pd.concat(df_list)
            concatenated.name = 'Condition ' + str(i)
            new_file_list.append(concatenated)
        i+=1

    return new_file_list

def calc_av(x, adata, vdata):
    """
    Calculates average percentage of answers correct for both audio-only condition 6 and visual-only condition 1 (assuming corresponding conditions are input files)
    args:
        afile (str): audio file to be formatted
        vfile (str): visual file to be formatted
        numresp (int): number of responses per azimuth
    returns:
        List with new y-axis values representing the averages
    """

    vfile_percentages = get_y_axis(x, vdata)
    afile_percentages = get_y_axis(x, adata)

    mean_percentages = (np.array(vfile_percentages) + np.array(afile_percentages))/2
    return mean_percentages

def get_condition_num(data):
    return data['trial.condition'].values[0]

def calc_av_multiple(filenames, patient_names, save_figure=False):
    """
    Calculates average percentage of answers correct for audio-only condition 6 and visual-only condition 1 
    across multiple samples. 
    args: 
        x (list of floats): list of floats that represent the azimuth angles
        filenames (list of list): list of AV filenames per patient for analysis (eg [['condition6_patient1', 'condition1_patient1'], ['condition1_patient2', 'condition6_patient2']])
    """
    x = [-45, -22.5, -11.2, -5.6, -2.8, 2.8, 5.6, 11.2, 22.5, 45]
    overall_average = np.zeros(shape=len(x))
    
    for sample in filenames:
        audio_data = sample[1]
        visual_data = sample[0]

        average = calc_av(x, audio_data, visual_data)

        overall_average += average

    patients = ''
    for name in patient_names:
        patients += name + ', '
    patients = patients.strip(', ')

    overall_average = overall_average/len(filenames)
    print(overall_average)
    
    plt.ylim(top=100)
    plt.ylim(bottom=0)
    plt.title('Average AV results across ' + patients, fontsize=10)
    plt.ylabel("Percentage of Correct Responses", fontsize=10)
    plt.xlabel("Azimuth", fontsize=10)

    plt.plot(x, overall_average, 'o--', linewidth=2.5, label='Average Results')
    
    plt.xticks(x, x)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)

    if save_figure:
        file_name = input(f"You are currently observing the average AV results of {patients}\nPlease enter desired filename for figure")
        plt.savefig(file_name)

    plt.show()

def flatten_data(filenames_unorganized):
    filenames_organized = collect_conditions(filenames_unorganized)
    # print(filenames_organized)
    #now calculate averages of each list inside filenames_organized so we can flatten to one list
    filenames = concat_conditions(filenames_organized)
    return filenames


"""END OF HELPER FUNCTIONS"""

"""TO DISPLAY MULTIPLE TRIALS"""

"""CHANGE FILE NAME TO DESIRED DATA VISUALIZATION WITHOUT .CSV"""

def all_results(filenames_unorganized, sample_num=50, audio_file=False, audio_format=False, save_figure=False, interpolate = True, y_axis_100=False):
    """
    Displays data formatted in scatterplot with spline and polynomial interpolated lines. 
    assumes:
        -input list is in numerical order (Condition 1 data, condition 2 data,..., condition 6 data)
        -input list is a list of str with filenames without the '.csv'
        -Columns for analysis are labelled accordingly: 'walker.azimuth', 'response.responseScore', 'trial.congruent' (if congruency considered, eg. not conditions 1 or 6), 'sound.direction' (if audio_format=False)
    args:
        filenames (list): list of .csv data files to visualize and analyze
        sample_num (int): Number of data points wanted to generate with interpolation. Default = 50
        audio_file (boolean): =True if audio file present within list of files
        audio_format (boolean): =True if audio file format works with format of other data, in other words, if the audio file has an azimuth column. Otherwise, formats audio only condition in bar graph.
        save_figure (boolean): =True if want to save figure to device
        interpolate (boolean): =True if want to perform and plot interpolations. 
        y_axis_100 (boolean): =True if want y-axis range [0,100]. Otherwise, [0,150]
    """
    plt.ylim(.5,1)

    #first flatten all datafiles into 6- so only one file per condition and in numerical order
   
    #now calculate averages of each list inside filenames_organized so we can flatten to one list
    filenames = flatten_data(filenames_unorganized)
    
    for data in filenames: #for each dataframe
        condition = get_condition_num(data)
        if condition == 6 and audio_format==False:
            continue

        plt.title(f'Condition {condition} results', fontsize=10)
        plt.ylabel("Percentage of Correct Responses", fontsize=10)
        plt.xlabel("Azimuth", fontsize=10)
        congruency_considered = "trial.congruent" in data.columns 

        """X AXIS VALUES - AZIMUTH"""
        if congruency_considered:
            y_congruent = []
            y_incongruent = []
        
        else:
            y_axis = []

        x = [] #x values for plotting
        
        azimuth = data['walker.azimuth'].unique()
        for item in azimuth:
            if item not in x:
                x.append(item)
        x = sorted(x) #sort by ascension

        """Y AXIS VALUES - RESPONSE ACCURACY"""
        if y_axis_100:
            plt.ylim(top=100)
            plt.ylim(bottom=0)
        else:
            plt.ylim(top=150)
            plt.ylim(bottom=0)
        
        if congruency_considered: 
            congruent_vals = data.loc[data['trial.congruent'] == 1] #only look at congruent trials to collect this list first
            incongruent_vals = data.loc[data['trial.congruent'] == 0]

            y_congruent = get_y_axis(x, congruent_vals)
            y_incongruent = get_y_axis(x, incongruent_vals)

            if interpolate:
                plt.plot(x, y_congruent, 'o--', alpha=.5, label='congruent')
                plt.plot(x, y_incongruent, 'o--', alpha=.5, label='incongruent')

            #FOR POLYNOMIAL INTERPOLATION
                interp_congruent = poly_interp(x, y_congruent, sample_num)
                interp_incongruent = poly_interp(x, y_incongruent, sample_num)

                plt.plot(interp_congruent[0], interp_congruent[1], '-', label='congruent polynomial interpolation')
                plt.plot(interp_incongruent[0], interp_incongruent[1], '-', label='incongruent polynomial interpolation')

                print('Polynomial Interpolation Minimums (Congruent):')
                extract_min(condition, interp_congruent)

                print('Polynomial Interpolation Minimums (Incongruent):')
                extract_min(condition, interp_incongruent)


                # #FOR SPLINE INTERPOLATION
                interp_congruent = spline_interp(x, y_congruent, sample_num)
                interp_incongruent = spline_interp(x, y_incongruent, sample_num)
                plt.plot(interp_congruent[0], interp_congruent[1], '-', label='congruent spline interpolation')
                plt.plot(interp_incongruent[0], interp_incongruent[1], '-', label='incongruent spline interpolation')

                print('B-Spline Interpolation Minimums (Congruent):')
                extract_min(condition, interp_congruent)

                print('B-Spline Interpolation Minimums (Incongruent):')
                extract_min(condition, interp_incongruent)
            
            else:
                plt.plot(x, y_congruent, 'o--', linewidth=2.5, label='congruent')
                plt.plot(x, y_incongruent, 'o--', linewidth=2.5, label='incongruent')
            
        else:
            y_axis = get_y_axis(x, data)

            if interpolate:
            #FOR POLYNOMIAL INTERPOLATION
                plt.plot(x, y_axis, 'o--', alpha=.5, label='results')

                interp = poly_interp(x, y_axis, sample_num)
                plt.plot(interp[0], interp[1], label='polynomial interpolation')

                print('Polynomial Interpolation Minimums:')
                extract_min(condition, interp)

                #FOR SPLINE INTERPOLATION
                interp = spline_interp(x, y_axis, sample_num)
                plt.plot(interp[0], interp[1], '-', label='congruent spline interpolation')

                print('B-Spline Interpolation Minimums:')
                extract_min(condition, interp)
            else:
                plt.plot(x, y_axis, 'o--', linewidth=2.5, label='results')

        plt.legend()
        plt.xticks(x, x)
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)

        if save_figure:
            file_name = input(f"You are currently observing condition {condition}\nPlease enter desired filename for figure")
            plt.savefig(file_name)

        plt.show()
        condition +=1
    
    if audio_file:
        if not audio_format:
            audio_plots(filenames[-1], save_figure)
        else:
            avg_y = calc_av(x, filenames[0], filenames[-1])

            if interpolate:
            #FOR POLYNOMIAL INTERPOLATION
                plt.plot(x, avg_y, 'o--', alpha=.5, label='Average AV results')
                interp = poly_interp(x, avg_y, sample_num)
                plt.plot(interp[0], interp[1], label='polynomial interpolation')

                print('Polynomial Interpolation Minimums:')
                extract_min(condition, interp)

                #FOR SPLINE INTERPOLATION
                interp = spline_interp(x, avg_y, sample_num)
                plt.plot(interp[0], interp[1], '-', label='congruent spline interpolation')

                print('B-Spline Interpolation Minimums:')
                extract_min(condition, interp)
            else:
                
                plt.plot(x, avg_y, 'o--', linewidth=2.5, label='Average AV results')

            plt.title('Average AV results', fontsize=10)

            plt.ylabel("Percentage of Correct Responses", fontsize=10)
            plt.xlabel("Azimuth", fontsize=10)
            plt.legend()
            plt.xticks(x, x)
            plt.xticks(fontsize=6)
            plt.yticks(fontsize=6)

            if save_figure:
                file_name = input(f"You are currently observing the average AV results\nPlease enter desired filename for figure")
                plt.savefig(file_name)

            plt.show()
    
            

def audio_plots(data, save_figure):
    """
    Displays audio data formatted in bar graph format. (Direction vs. Percentage Answered Correct)
    args:
        file (str): audio file to be formatted
        save_figure (boolean): =True if want to save figure to device
    """
    plt.title('Condition 6', fontsize=10)
    plt.ylabel("Percentage of Correct Responses", fontsize=10)
    plt.xlabel("Audio Direction", fontsize=10)

    y = [1,2,3,4,5,6,7,8,9,10]

    plt.ylim(top=1.0)
    plt.ylim(bottom=0)
    
    #for percent
    x = [1,2] #audio direction
    y_axis = []
    num_resp = get_num_resp(data, 'sound.direction')
    for item in x: # for direction
            # y = [] #just reset y axis each direction 
            vals = data.loc[data['sound.direction']==item, 'response.responseScore'].sum() # gets response score
            # print(item, 'scored', vals) 
            total_correct = vals#+neg_vals
            # print(item, 'results in total', total_correct, 'correct responses') # "" add them together to get total responses
            total = num_resp[item] #total number possible responses
            # print('The max number of possible correct was', total)
            accuracy = (total_correct/total)
            y_axis.append(accuracy)
            # print(y_axis)
            print('Your accuracy for direction', item, 'was', accuracy*100, '%') # response accuracy 

    plt.bar(['L', 'R'], y_axis) #then plot
    plt.xticks((0,1))
    plt.yticks(fontsize=6)

    if save_figure:
        file_name = input("You are currently observing Condition 6\nPlease enter desired filename for figure")
        plt.savefig(file_name)

    plt.show()

    
"""PLOTTING DATA"""

"""Examples commented below"""
# #Plotting data without audio file and 50 sample interpolation
# #Good for plotting select files, single files, etc.
# filenames = ['Sample_Condition1Data']
# all_results(filenames, audio_file=False)

# #plotting data with audio file but file is not formatted with azimuth and 100 sample interpolation
# filenames = ['Sample_Condition1Data', 'Sample_Condition2Data', 'Sample_Condition3Data', 'Sample_Condition4Data','Sample_Condition5Data', 'Sample_Condition6Data']
# all_results(filenames, sample_num=100, audio_file=True)

# #plotting data with audio file but file is formatted with azimuth and 60 sample interpolation
# filenames = ['Sample_Condition1Data', 'Sample_Condition2Data', 'Sample_Condition3Data', 'Sample_Condition4Data','Sample_Condition5Data', 'Sample_Condition6Data']
# all_results(filenames, audio_file=True, audio_format=True)

# #plotting data with audio file but file is formatted with azimuth and 100 sample interpolation
# filenames = ['Sample_Condition1Data', 'Sample_Condition2Data', 'Sample_Condition3Data', 'Sample_Condition4Data','Sample_Condition5Data', 'Sample_Condition6Data']
# all_results(filenames, sample_num=100, audio_file=True, audio_format=True)

# #plotting data with formatted audio file but don't want to perform interpolations
# filenames = ['Sample_Condition1Data', 'Sample_Condition2Data', 'Sample_Condition3Data', 'Sample_Condition4Data','Sample_Condition5Data', 'Sample_Condition6Data']
# all_results(filenames, sample_num=100, audio_file=True, audio_format=True, interpolate=False)

# #plotting data with formatted audio file but don't want to perform interpolations and want to see results on scale from 0% correct to 100% correct
# filenames = ['Sample_Condition1Data', 'Sample_Condition2Data', 'Sample_Condition3Data', 'Sample_Condition4Data','Sample_Condition5Data', 'Sample_Condition6Data']
# all_results(filenames, sample_num=100, audio_file=True, audio_format=True, interpolate=False, y_axis_100=True)



"""my data plots"""

# scramble_cc = ['1V_S1200_CC', '2AV_C_S_S1200_CC', '3AV_C_OS_S1200_CC', '4AV_I_S_S1200_CC', '5AV_I_OS_S1200_CC', '6A_S1200_CC']
# inversion_sl = ['1V_I1200_SL', '2AV_C_S_I1200_SL', '3AV_C_OS_I1200_SL', '4AV_I_S_I1200_SL', '5AV_I_OS_I1200_SL', '6A_I1200_SL']
# scramble_sl = ['1V_S1200_SL', '2AV_C_S_S1200_SL', '3AV_C_OS_S1200_SL', '4AV_I_S_S1200_SL', '5AV_I_OS_S1200_SL', '6A_S1200_SL']
# inversion_llm =['2AV_C_S_I1200_LLM','3AV_C_OS_I1200_LLM', '4AV_I_S_I1200_LLM', '5AV_I_OS_I1200_LLM', '6A_I1200_LLM']
# inversion_sl = ['1V_I1200_SL', '2AV_C_S_I1200_SL', '3AV_C_OS_I1200_SL', '4AV_I_S_I1200_SL', '5AV_I_OS_I1200_SL', '6A_I1200_SL']
control_anshra = ['Anshra_Control_blur_1_run1', 'Anshra_Control_blur_1_run3', 'Anshra_Control_blur_2_run1', 'Anshra_Control_blur_2_run3', 'Anshra_Control_blur_3_run2', 'Anshra_Control_blur_3_run4', 'Anshra_Control_blur_4_run1', 'Anshra_Control_blur_4_run3', 'Anshra_Control_blur_5_run2', 'Anshra_Control_blur_5_run4', 'Anshra_Control_blur_6_run5']
control_kushagra = ['Kushagra_control_blur_1_run1', 'Kushagra_control_blur_1_run3', 'Kushagra_control_blur_2_run1', 'Kushagra_control_blur_2_run3', 'Kushagra_control_blur_3_run2', 'Kushagra_control_blur_3_run4', 'Kushagra_control_blur_4_run1', 'Kushagra_control_blur_4_run3', 'Kushagra_control_blur_5_run2', 'Kushagra_control_blur_5_run4', 'Kushagra_control_blur_6_run5']
# control_jl = ['1V_JL', '2AV_JL', '3A_JL']
# control_sl = ['1V_SL', '2AV_SL', '3A_SL']
# control_cc = ['1V_CC', '2AV_CC', '3A_CC']
control_arushi = ['Arushi_C1', 'Arushi_C2', 'Arushi_C3', 'Arushi_C4', 'Arushi_C5', 'Arushi_C6']
control_ritesh = ['ritesh_C!', 'ritesh_C2', 'ritesh_C3', 'ritesh_C4', 'ritesh_C5', 'ritesh_C6']
control_uday = ['uday_C1', 'uday_C2', 'uday_C3', 'uday_C4', 'uday_C5', 'uday_C6']
control_wasiya = ['wasiya_C1', 'wasiya_C2', 'wasiya_C3', 'wasiya_C4', 'wasiya_C5', 'wasiya_C6']
# filenames14


# s_patient = ['1V_Followup3_BM', '2AV_Followup3_BM', '3AV_Followup3_BM', '4AV_Followup3_BM', '5AV_Followup3_BM', '6A_Followup3_BM']
# z_patient = ['Z_Condition1', 'Z_Condition2', 'Z_Condition3', 'Z_Condition4', 'Z_Condition5', 'Z_Condition6',]
# k_patient = ['K_Condition1_run1', 'K_Condition1_run3', 'K_Condition2_run1', 'K_Condition2_run3', 'K_Condition3_run2', 'K_Condition3_run4', 'K_Condition4_run1', 'K_Condition4_run3', 'K_Condition5_run2', 'K_Condition5_run4', 'K_Condition6_run5']
control_alshifa = ['Alshifa_Control_blur_1_run1', 'Alshifa_Control_blur_1_run3', 'Alshifa_Control_blur_2_run1', 'Alshifa_Control_blur_2_run3', 'Alshifa_Control_blur_3_run2', 'Alshifa_Control_blur_3_run4', 'Alshifa_Control_blur_4_run1', 'Alshifa_Control_blur_4_run3', 'Alshifa_Control_blur_5_run2', 'Alshifa_Control_blur_5_run4', 'Alshifa_Control_blur_6_run5',]


# all_results(control_wasiya, audio_file=True, audio_format=True, interpolate=False, y_axis_100=True, save_figure=False)

p1 = flatten_data(control_anshra) 
p2 = flatten_data(control_kushagra) 
p3 = flatten_data(control_arushi) 
p4 = flatten_data(control_ritesh) 
p5 = flatten_data(control_uday) 
p6 = flatten_data(control_wasiya) 
p7 = flatten_data(control_alshifa) 

names = ['Anshra', 'Kushagra', 'Arushi', 'Ritesh', 'Uday', 'Wasiya', 'Alshifa']

calc_av_multiple([[p1[0], p1[-1]], [p2[0], p2[-1]], [p3[0], p3[-1]], [p4[0], p4[-1]], [p5[0], p5[-1]], [p6[0], p6[-1]], [p7[0], p7[-1]]], names, save_figure=True)
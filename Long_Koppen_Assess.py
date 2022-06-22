import pandas as pd
import numpy as np
import glob
from tqdm import tqdm

def koppen(P, T, lat, z = 0, Peel = True):
    data = pd.DataFrame(data = (P, T), index = ['P', 'T']).transpose()
    T = np.array(T)
    P = np.array(P)
    
    MAP = np.round(np.sum(P), 4)
    MAT = np.round(np.mean(T), 4)
    Thot = np.round(np.max(T), 4)
    Tcold = np.round(np.min(T), 4)
    Tmonth10 = np.round(sum(map(lambda x : x >= 10, T)), 4)
    Pdry = np.round(np.min(P), 4)

    if not Peel:
        seqT = np.array((T[0], T[1], T[2], T[3], T[4], T[5], T[6], T[7], T[8], T[9], T[10], T[11], T[0], T[1], T[2], T[3], T[4]))
        seqP = np.array((P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7], P[8], P[9], P[10], P[11], P[0], P[1], P[2], P[3], P[4]))
        data['seqT'] = np.convolve(seqT, np.ones(6, dtype=np.int), mode='valid')
        data['seqP'] = np.convolve(seqP, np.ones(6, dtype=np.int), mode='valid')
        
        s_st = data['seqT'].idxmax(axis = 1)
        s_f = s_st + 6
        w_st = s_f
        w_f = w_st + 6
        
        if s_f > 12:
            s_f = s_f - 12
        if w_f > 12:
            w_f = w_f - 12
        if s_st > 12:
            s_st = s_st - 12
        if w_st > 12:
            w_st = w_st - 12
        
        drymonth = data['P'].idxmin(axis = 1)
        season = []
        
        if s_st < s_f and w_st < w_f:
            Psdry = data['P'][(data.index >= s_st) & (data.index < s_f)].min()
            Pwdry = (data['P'][(data.index >= w_st) & (data.index < w_f)].min())
            Pswet = data['P'][(data.index >= s_st) & (data.index < s_f)].max()
            Pwwet = (data['P'][(data.index >= w_st) & (data.index < w_f)].max())
            if s_st < drymonth and s_f > drymonth:
                season.append('summer')
            else:
                season.append('winter')
        
        if s_st < s_f and w_st > w_f:
            Psdry = data['P'][(data.index >= s_st) & (data.index < s_f)].min()
            Pwdry = (data['P'][(data.index >= w_st)].min(), data['P'][(data.index < w_f)].min())
            Pwdry = np.min(Pwdry)
            Pswet = data['P'][(data.index >= s_st) & (data.index < s_f)].max()
            Pwwet = (data['P'][(data.index >= w_st)].max(), data['P'][(data.index < w_f)].max())
            Pwwet = np.max(Pwwet)
            if s_st < drymonth and s_f > drymonth:
                season.append('summer')
            else:
                season.append('winter')
        
        if s_st > s_f and w_st < w_f:
            Psdry = (data['P'][(data.index >= s_st)].min(), data['P'][(data.index < s_f)].min())
            Psdry = np.min(Psdry)
            Pwdry = (data['P'][(data.index >= w_st) & (data.index < w_f)].min())
            Pswet = (data['P'][(data.index >= s_st)].max(), data['P'][(data.index < s_f)].max())
            Pswet = np.max(Pswet)
            Pwwet = (data['P'][(data.index >= w_st) & (data.index < w_f)].max())
        
            if w_st < drymonth and w_f > drymonth:
                season.append('winter')
            else:
                season.append('summer')    
        
        if s_st < w_st:
            s_f = s_f - 1
            w_f = w_f - 1
        else:
            w_f = w_f - 1
            s_f = s_f - 1
    
        if w_st == 12:
            w_st = 0
            
        if 0.7 * MAP <= data['seqP'].loc[w_st]:
            Pthresh = 2 * MAT
        elif 0.7 * MAP <= data['seqP'].loc[s_st]:
            Pthresh = 2 * MAT + 28
        else:
            Pthresh = 2 * MAT + 14
    
    else:
        if lat > 0:
            # N
            winter = np.concatenate((P[0:3], P[9:12]))
            summer = P[3:9]
            Psdry = np.min(summer)
            Pwdry = np.min(winter)
            Pswet = np.max(summer)
            Pwwet = np.max(winter)
        else:
            #S
            winter = P[3:9]
            summer = np.concatenate((P[0:3], P[9:12]))
            Psdry = np.min(summer)
            Pwdry = np.min(winter)
            Pswet = np.max(summer)
            Pwwet = np.max(winter)
    
    if P[3:9].sum()/(MAP + 0.1) > 0.3 and  P[3:9].sum()/(MAP + 0.1) < 0.7:
        p1 = 2 * MAT + 14
    else:
        p1 = 0
    if lat < 0 or p1 != 0:
        p2 = 0
    else:
        if P[3:9].sum()/(MAP + 0.1) > 0.7:
            p2 = 2 * MAT + 28
        else:
            p2 = 2 * MAT
    if lat > 0 or p1 != 0:
        p3 = 0
    else:
        if P[3:9].sum()/(MAP + 0.1) > 0.7:
            p3 = 2 * MAT
        else:
            p3 = 2 * MAT + 28
            
    Pthresh = p1 + p2 + p3

    main_class = []
    if z >= 2300:
        if Thot >= 0:
            main_class.append('HT - Tundra')
        else:
            main_class.append('HF - Frost or Ice Cap')
    elif MAP < (10 * Pthresh):
        main_class.append('B')
    elif len(main_class) == 0 and not Tcold < 18:
        main_class.append('A')
    elif len(main_class) == 0 and Thot >= 10 and Tcold < 18 and Tcold > 0:
        main_class.append('C')
    elif len(main_class) == 0 and Thot >= 10 and Tcold <= 0:
        main_class.append('D')
    elif len(main_class) == 0 and Thot < 10:
        main_class.append('E')
    
    second_class = []
    if main_class[0] == 'A':
        if Pdry >= 60:
            second_class.append('f')
        else:
            if Pdry >= 100 - MAP / 25:
                second_class.append('m')
            elif Psdry < 100 - MAP / 25:
                second_class.append('s')
            elif Pwdry < 100 - MAP / 25:
                second_class.append('w')

    elif main_class[0] == 'B':
        if MAP < 5 * Pthresh:
            second_class.append('W')
        else:
            second_class.append('S')
        if MAT >= 18:
            second_class.append('h')
        else:
            second_class.append('k')

    elif main_class[0] == 'C':
        if Psdry < 40 and Psdry < Pwwet / 3:
            second_class.append('s')
        elif Pwdry < Pswet / 10:
            second_class.append('w')
        else:
            second_class.append('f')
        if Thot >= 22:
            second_class.append('a')
        elif Tmonth10 >= 4:
            second_class.append('b')
        elif 1 <= Tmonth10 and Tmonth10 < 4:
            second_class.append('c')
    
    if main_class[0] == 'D':
        if Psdry < 40 and Psdry < Pwwet / 3:
            second_class.append('s')
        elif Pwdry < Pswet / 10:
            second_class.append('w')
        else:
            second_class.append('f')
        if Thot >= 22:
            second_class.append('a')
        elif Tmonth10 >= 4:
            second_class.append('b')
        elif Tcold < -38:
            second_class.append('d')
        else:
            second_class.append('c')

    if main_class[0] == 'E':
        if Thot > 0:
            second_class.append('T')
        else:
            second_class.append('F')
    main_class = ''.join(main_class)
    second_class = ''.join(second_class)
    classification = main_class + second_class
    return classification

path_T = r'C:\Users\DTouloumidis\OneDrive - imetb\Desktop\Koppen\ECA_blend_tg'
path_P = r'C:\Users\DTouloumidis\OneDrive - imetb\Desktop\Koppen\ECA_blend_rr'

stations_T = r"C:\Users\DTouloumidis\OneDrive - imetb\Desktop\Koppen\ECA_blend_tg\stations.txt"
stations_P = r"C:\Users\DTouloumidis\OneDrive - imetb\Desktop\Koppen\ECA_blend_rr\stations.txt"

def split_years(dt):
    dt['year'] = dt.index.year
    out = [dt[dt['year'] == y] for y in dt['year'].unique()]
    for i in range(len(out)):
        if not len(out[i]) == 12:
            out[i] = out[i].copy()
            out[i]['P'] = np.nan
            out[i]['TG'] = np.nan
    return out

# Read files
all_P_files = []
for name in glob.glob('{}\*[0-9].*'.format(path_P)): # Read all files with any number in name
    all_P_files.append(name)

all_T_files = []
for name in glob.glob('{}\*[0-9].*'.format(path_T)): # Read all files with any number in name
    all_T_files.append(name)

T_stations = []
for i in all_T_files:
    T_stations.append(i[75:-4]) # Isolate the unique number of the temperature stations
    
P_stations = []
for i in all_P_files:
    P_stations.append(i[75:-4]) # Isolate the unique number of the precipitation stations

# Find the common stations with data for both precipitation and temperature and read the characteristics from the total stations
common_stations = [i for i in T_stations if i in P_stations] # Get all the stations with values for both precipitation and temperature
stations = pd.read_csv(stations_T, sep=",", skiprows = 17) # Read the characteristics of the temperature stations
stations.columns = ['STAID', 'STANAME', 'CN', 'LAT', 'LON', 'HGHT'] # Rename the columns to conventional names

# Convert the coordinates of the temperature stations from degrees to decimal degrees
phi = np.zeros(len(stations['LAT'])) # Define zero arrows to convert latitude to decimal
labda = np.zeros(len(stations['LON'])) # Define zero arrows to convert longitute to decimal
for i in range(len(stations)):
    phi[i] = (int(stations['LAT'].iloc[i][1:3]) + int(stations['LAT'].iloc[i][4:6])/60 + int(stations['LAT'].iloc[i][7:9])/3600)
    labda[i] = (int(stations['LON'].iloc[i][1:4]) + int(stations['LON'].iloc[i][5:7])/60 + int(stations['LON'].iloc[i][8:10])/3600)
    if not stations['LAT'].iloc[i][0:1] == '+':
        phi[i] = - phi[i]
    if not stations['LON'].iloc[i][0:1] == '+':
        labda[i] = - labda[i]
stations['LAT'] = phi
stations['LON'] = labda

# Create a dataframe with the numerical id of the stations and keep only the common stations
num_stations = np.zeros(len(common_stations))
for i in range(len(common_stations)):
    num_stations[i] = int(common_stations[i]) # Convert string to integer
num_stations = pd.DataFrame(num_stations, columns = ['STAID'])
final_stations = pd.merge(num_stations, stations,on='STAID', how='left') # Keep the common stations

# Read temperature and precipitation data
data = []
# for i in tqdm(range(len(common_stations))):
for i in tqdm(range(100)):
    t_pr = pd.read_csv('{}\\TG_STAID{}.txt'.format(path_T, common_stations[i]), names = ['STAID', 'SOUID', 'DATE', 'TG', 'QUALITY'], skiprows = 21, parse_dates = ['DATE'])
    t_pr.index = t_pr['DATE']
    t_pr['TG'] = t_pr['TG']/10 # convert temperature to the right units
    t_pr = t_pr[t_pr['QUALITY'] == 0] # keep only the consinstent observations
    t_pr = t_pr.dropna() # drop nan days
    count_t = t_pr['STAID'].resample(rule = 'M').count() # calculate days per month used to calculate the monthly
    t_pr = t_pr.resample(rule = 'M').median() # resample all the values with median
    t_pr['count'] = count_t # assign the count days per month
    t_pr = t_pr[t_pr['count'] > 20] # remove months with less than 20 days

    p_pr = pd.read_csv('{}\\RR_STAID{}.txt'.format(path_P, common_stations[i]), names = ['STAID', 'SOUID', 'DATE', 'P', 'QUALITY'], skiprows = 21, parse_dates = ['DATE'])
    staid = p_pr['STAID'].iloc[0]
    p_pr.index = p_pr['DATE']
    p_pr['P'] = p_pr['P']/10 # convert precipitation to the right units
    p_pr = p_pr[p_pr['QUALITY'] == 0]# keep only the consinstent observations
    p_pr = p_pr.dropna() # drop nan days
    count_p = p_pr['STAID'].resample(rule = 'M').count() # calculate days per month used to calculate the monthly
    p_pr = p_pr.resample(rule = 'M').sum() # resample all the values with sum
    p_pr['count'] = count_p # assign the count days per month
    p_pr['STAID'] = staid # assign the station id to each stations dataframe cause sum 
    p_pr = p_pr[p_pr['count'] > 20] # remove months with less than 20 days

    conc = p_pr.merge(t_pr, left_index=True, right_index=True)[['STAID_x', 'P', 'TG']] # merge the precipitation and the temperature dataframes by index. It keeps only the common dates
    conc.columns = ['STAID', 'P', 'TG'] # rename columns
    conc = conc.loc[(conc.index > '1900-01-01') & (conc.index < '2021-01-01')] # filter the data of each station to specific years
    if len(conc) >= 12 * 20 and conc.index.year.max() > 2000: # keep only VALID stations with 20 years observations with the maximum year over 2000
        conc['LAT'] = np.ones(len(conc)) * final_stations['LAT'][final_stations['STAID'] == conc['STAID'][0]].values # assign the latitute of each VALID station
        conc['LON'] = np.ones(len(conc)) * final_stations['LON'][final_stations['STAID'] == conc['STAID'][0]].values # assign the longitude of each VALID station
        conc['HGHT'] = np.ones(len(conc)) * final_stations['HGHT'][final_stations['STAID'] == conc['STAID'][0]].values # assign the height of each VALID station
        conc['CN'] = str(final_stations['CN'][final_stations['STAID'] == conc['STAID'][0]].values) # assign the country of each VALID station
        data.append(conc)

data_m = []
datas = []
for i in range(len(data)):
    datas.append(split_years(data[i])) # for each station, split into years and keep only years with 12 months
    data_m.append(data[i].groupby(data[i].index.month).median()) # calculate the all time median of each station

station_res = pd.DataFrame()
for i in tqdm(range(len(datas))):
    kop_m = koppen(data_m[i]['P'], data_m[i]['TG'], data_m[i]['LAT'].iloc[0], z = 0, Peel = True) # koppen with the all time median
    for j in range(len(datas[i])):
        if len(datas[i][j]['P']) == 12:
            kop = koppen(datas[i][j]['P'], datas[i][j]['TG'], datas[i][j]['LAT'][0], z = 0, Peel = True) # koppen of each month
        else:
            kop = np.nan
        staid = datas[i][j]['STAID'][0]
        year = datas[i][j].index.year[0]
        helper = pd.DataFrame(data = [year, kop, kop_m], columns = [staid], index = ['YEAR', 'KOPPEN', 'KOPPEN_M']).T
        station_res = pd.concat([station_res, helper])
station_res['STAID'] = station_res.index

results = []
for i in range(len(np.unique(station_res['STAID']))):
    most_freq = station_res[station_res['STAID'] == np.unique(station_res['STAID'])[i]]
    results.append([most_freq['STAID'].iloc[0], most_freq['KOPPEN'].value_counts().idxmax(), most_freq['KOPPEN_M'].iloc[0]])

results = pd.DataFrame(results, columns = ['STAID', 'Freq Kop', 'Median Kop'])
results.index = results['STAID']
results = results.drop('STAID', axis = 1)
results['Check_Koppen'] = np.where(results['Freq Kop'] == results['Median Kop'], True, False)
results['Check_Koppen_class'] = np.where(results['Freq Kop'].astype(str).str[0] == results['Median Kop'].astype(str).str[0], True, False)

ranges = []
for i in range(len(datas)):
    ranges.append([datas[i][0].index.year.min(), datas[i][-1].index.year.max(), len(datas[i])])
    

ranges = pd.DataFrame(ranges, columns = ['START DATE', 'EMD DATE', 'YEARS'])
ranges = final_stations.merge(ranges, left_index=True, right_index=True)

results = pd.merge(ranges, results, on='STAID')
results.index = results['STAID']
results = results.drop('STAID', axis = 1)

print(' \n')
print('Class (1st letter):  Same Koppen class can be met at {:.2%} ({}) of the total {} instances!'.format(len(results['Check_Koppen_class'][results['Check_Koppen_class'] == True])/len(results['Check_Koppen_class']), len(results['Check_Koppen_class'][results['Check_Koppen_class'] == True]), len(results['Check_Koppen_class'])))
print('Total:               Same Koppen class can be met at {:.2%} ({}) of the total {} instances!'.format(len(results['Check_Koppen'][results['Check_Koppen'] == True])/len(results['Check_Koppen']), len(results['Check_Koppen'][results['Check_Koppen'] == True]), len(results['Check_Koppen'])))

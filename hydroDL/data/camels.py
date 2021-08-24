# read camels dataset
import os
import pandas as pd
import numpy as np
import datetime as dt
from hydroDL import utils, pathCamels
from pandas.api.types import is_numeric_dtype, is_string_dtype
import time
import json
from . import Dataframe

# module variable
tRange = [19800101, 20200101]
tRangeobs = [19800101, 20200101]    #[19801001, 20161001] #  streamflow observations
tLst = utils.time.tRange2Array(tRange)
tLstobs = utils.time.tRange2Array(tRangeobs)
nt = len(tLst)
ntobs = len(tLstobs)

###############  for a paper with two upstream node with temp data  ###############
#forcingLst = ['Q3Tw', 'Q7Tw', 'Q3Q', 'Q5Q', 'Q7Q',
 #      'Q9Q', 'Btamean', 'V', 'Pressure', 'SkyCov', 'SR.sum', 'Precip',
#       ]
###################################################################################

###############   for stream_temp Module  ####################
##forcingLst = ['basin_ccov', 'basin_humid', 'basin_rain',
  ##     'basin_tave_air', 'basin_gwflow', 'basin_potet', 'basin_sroff',
    ##   'basin_ssflow', 'basin_swrad', 'basin_tave_gw',
      ## 'basin_tave_ss', 'network_width','outlet_width', 'outlet_outflow', 'gw_tau', 'ss_tau']  #, 'gw_tau', 'ss_tau' 'obs_discharge'

##attrLstSel = ['hru_elev', 'hru_slope', 'network_elev',
  ##     'outlet_elev', 'network_length', 'network_slope', 'outlet_slope',
    ##   'basin_area']
################################################################

##############   Water Temperature for CONUS scale  ##########
# forcingLst = ['dayl(s)', 'prcp(mm/day)', 'srad(W/m2)', 'tmax(C)',
    #    'tmin(C)', 'vp(Pa)', '00060_Mean']   #, 'pred_discharge' , '00060_Mean' ,'combine_discharge', 'combine_discharge' 'swe(mm)' ,'outlet_outflow',, 'pred_discharge', , '00060_Mean'
forcingLst = ['PRCP (Daymet)'] #'streamflow'] #, '80154_mean']


# attrLstSel = [  'NDAMS_2009',	'DDENS_2009',	'STOR_NID_2009',	
#                 'STOR_NOR_2009',	'MAJ_NDAMS_2009',	'MAJ_DDENS_2009',	
#                 'DEVNLCD06',	'FORESTNLCD06',	'PLANTNLCD06',	'WATERNLCD06',	
#                 'SNOWICENLCD06',	'DEVOPENNLCD06',	'DEVLOWNLCD06',	'DEVMEDNLCD06',	
#                 'DEVHINLCD06',	'BARRENNLCD06',	'DECIDNLCD06',	'EVERGRNLCD06',	
#                 'MIXEDFORNLCD06',	'SHRUBNLCD06',	'GRASSNLCD06',	'PASTURENLCD06',
#                 'CROPSNLCD06',	'WOODYWETNLCD06',	'MAINS800_FOREST',
#                 'MAINS800_PLANT',	'RIP100_DEV',	'RIP100_FOREST',	
#                 'RIP100_PLANT',	'RIP800_DEV',	'RIP800_FOREST',	'RIP800_PLANT',	
#                 'PERMAVE',	'BDAVE',	'OMAVE',	'WTDEPAVE',	
#                 'ROCKDEPAVE',	'NO4AVE',	'NO200AVE',	'NO10AVE',	
#                 'CLAYAVE',	'SILTAVE',	'SANDAVE',	'KFACT_UP',	'RFACT',
#                 'BASIN_BOUNDARY_CONFIDENCE','PPTAVG_BASIN',	'DRAIN_SQKM',	
#                 'HYDRO_DISTURB_INDX']  ##Farshid_like

# attr below are the new selection
attrLstSel = ['PPTAVG_BASIN',
	            'DDENS_2009',	'STOR_NID_2009',	'MAJ_DDENS_2009',
                'DEVNLCD06',	'FORESTNLCD06',	'PLANTNLCD06',
                'HGA',	'HGB',	'HGC',	'HGD',	'PERMAVE',	'NO4AVE',	'NO200AVE',
                'NO10AVE',	'CLAYAVE',	'SILTAVE',	'SANDAVE',	'KFACT_UP',	
                'RFACT',	'ELEV_MEAN_M_BASIN',	'SLOPE_PCT',	'ASPECT_DEGREES',	
                'DRAIN_SQKM',	'HYDRO_DISTURB_INDX']
# # attrLstSel = ['DRAIN_SQKM',

#        'STREAMS_KM_SQ_KM',
#        'STOR_NID_2009', 'FORESTNLCD06', 'PLANTNLCD06',
#        'SLOPE_PCT', 'RAW_DIS_NEAREST_MAJ_DAM',

#       'PERDUN', 'RAW_DIS_NEAREST_DAM', 'RAW_AVG_DIS_ALL_MAJ_DAMS',
#     'T_MIN_BASIN', 'T_MINSTD_BASIN', 'RH_BASIN', 'RAW_AVG_DIS_ALLDAMS', 'PPTAVG_BASIN',
#      'HIRES_LENTIC_PCT','T_AVG_BASIN', 'T_MAX_BASIN','T_MAXSTD_BASIN', 'NDAMS_2009', 'ELEV_MEAN_M_BASIN'] #, 'MAJ_NDAMS_2009',
# # attrLstSel = ['DRAIN_SQKM', 'PPTAVG_BASIN', 'T_AVG_BASIN', 'T_MAX_BASIN',
#        'T_MAXSTD_BASIN', 'T_MIN_BASIN', 'T_MINSTD_BASIN', 'RH_BASIN',
#        'STREAMS_KM_SQ_KM', 'PERDUN', 'HIRES_LENTIC_PCT', 'NDAMS_2009',
#        'STOR_NID_2009', 'FORESTNLCD06', 'PLANTNLCD06', 'ELEV_MEAN_M_BASIN',
#        'SLOPE_PCT', 'RAW_DIS_NEAREST_DAM', 'RAW_AVG_DIS_ALLDAMS',
#        'RAW_DIS_NEAREST_MAJ_DAM', 'RAW_AVG_DIS_ALL_MAJ_DAMS',
#        'MAJ_NDAMS_2009', 'POWER_NUM_PTS', 'POWER_SUM_MW', 'lat', 'lon',
#        'HYDRO_DISTURB_INDX', 'BFI_AVE', 'FRAGUN_BASIN', 'DEVNLCD06',
#        'PERMAVE', 'RFACT', 'BARRENNLCD06', 'DECIDNLCD06', 'EVERGRNLCD06',
#        'MIXEDFORNLCD06', 'SHRUBNLCD06', 'GRASSNLCD06', 'WOODYWETNLCD06',
#        'EMERGWETNLCD06', 'GEOL_REEDBUSH_DOM_PCT',
#        'STRAHLER_MAX', 'MAINSTEM_SINUOUSITY', 'REACHCODE', 'ARTIFPATH_PCT',
#        'ARTIFPATH_MAINSTEM_PCT', 'PERHOR', 'TOPWET', 'CONTACT', 'CANALS_PCT',
#        'RAW_AVG_DIS_ALLCANALS', 'NPDES_MAJ_DENS', 'RAW_AVG_DIS_ALL_MAJ_NPDES',
#        'FRESHW_WITHDRAWAL', 'PCT_IRRIG_AG', 'ROADS_KM_SQ_KM',
#        'PADCAT1_PCT_BASIN', 'PADCAT2_PCT_BASIN']

########################################################################
############# Streamflow prediction for CONUS scale  ##########################
# attrLstSel = ['ELEV_MEAN_M_BASIN', 'SLOPE_PCT', 'DRAIN_SQKM',
#       'HYDRO_DISTURB_INDX', 'STREAMS_KM_SQ_KM', 'BFI_AVE', 'NDAMS_2009',
#       'STOR_NID_2009', 'RAW_DIS_NEAREST_DAM', 'FRAGUN_BASIN', 'DEVNLCD06',
#       'FORESTNLCD06', 'PLANTNLCD06', 'PERMAVE', 'RFACT',
#       'PPTAVG_BASIN', 'BARRENNLCD06', 'DECIDNLCD06', 'EVERGRNLCD06',
#       'MIXEDFORNLCD06', 'SHRUBNLCD06', 'GRASSNLCD06', 'WOODYWETNLCD06',
#       'EMERGWETNLCD06', 'GEOL_REEDBUSH_DOM_PCT',
#        'STRAHLER_MAX', 'MAINSTEM_SINUOUSITY', 'REACHCODE', 'ARTIFPATH_PCT',
#       'ARTIFPATH_MAINSTEM_PCT', 'HIRES_LENTIC_PCT', 'PERDUN', 'PERHOR',
#       'TOPWET', 'CONTACT', 'CANALS_PCT', 'RAW_AVG_DIS_ALLCANALS',
#        'NPDES_MAJ_DENS', 'RAW_AVG_DIS_ALL_MAJ_NPDES',
#       'RAW_AVG_DIS_ALL_MAJ_DAMS', 'FRESHW_WITHDRAWAL', 'PCT_IRRIG_AG',
#       'POWER_NUM_PTS', 'POWER_SUM_MW', 'ROADS_KM_SQ_KM', 'PADCAT1_PCT_BASIN',
#       'PADCAT2_PCT_BASIN']   # 'GEOL_REEDBUSH_SITE', , 'AWCAVE'
##############################################################################
def readGageInfo(dirDB):
    gageFile = os.path.join(dirDB, 'basin_timeseries_v1p2_metForcing_obsFlow',
                            'basin_dataset_public_v1p2', 'basin_metadata',
                            'gauge_information.txt')

    data = pd.read_csv(gageFile, sep='\t', header=None, skiprows=1)
    # header gives some troubles. Skip and hardcode
    fieldLst = ['huc', 'id', 'name', 'lat', 'lon', 'area']
    out = dict()
    for s in fieldLst:
        if s is 'name':
            out[s] = data[fieldLst.index(s)].values.tolist()
        else:
            out[s] = data[fieldLst.index(s)].values
    return out

def readUsgsGage(usgsId, Target, *, readQc=False):
    ##ind = np.argwhere(gageDict['id'] == usgsId)[0][0]
    ##huc = gageDict['huc'][ind]
    ##usgsFile = os.path.join(dirDB, 'basin_timeseries_v1p2_metForcing_obsFlow',
          ##                  'basin_dataset_public_v1p2', 'usgs_streamflow',
        ##                    str(huc).zfill(2),
      ##                      '%08d_streamflow_qc.txt' % (usgsId))
    ##dataTemp = pd.read_csv(usgsFile, sep=r'\s+', header=None)
    ##obs = dataTemp[4].values

    obs = forcing_data.loc[forcing_data['sta_id']==usgsId, Target].to_numpy()

    ##obs[obs < 0] = np.nan
    # if readQc is True:
    #     qcDict = {'A': 1, 'A:e': 2, 'M': 3}
    #     qc = np.array([qcDict[x] for x in dataTemp[5]])
    # if len(obs) != ntobs:
    #   ##  out = np.full([ntobs], np.nan)
    #     ##dfDate = dataTemp[[1, 2, 3]]
    #     ##dfDate.columns = ['year', 'month', 'day']
    #     ##date = pd.to_datetime(dfDate).values.astype('datetime64[D]')
    #     if 'datetime' in forcing_data.columns:
    #         date = forcing_data.loc[forcing_data['sta_id']==usgsId, 'datetime']
    #     elif 'date' in forcing_data.columns:
    #         date = forcing_data.loc[forcing_data['sta_id']==usgsId, 'date']
    #     [C, ind1, ind2] = np.intersect1d(date, tLstobs, return_indices=True)
    #     out[ind2] = obs
    #     if readQc is True:
    #         outQc = np.full([ntobs], np.nan)
    #         outQc[ind2] = qc
    # else:
    #     out = obs
    #     if readQc is True:
    #         outQc = qc
    #
    # if readQc is True:
    #     return out, outQc
    # else:
    #     return out
    return obs


def readUsgs(usgsIdLst: object) -> object:
    t0 = time.time()
    y = np.empty([len(usgsIdLst), ntobs])
    for k in range(len(usgsIdLst)):
        # print(usgsIdLst[k])
        dataObs = readUsgsGage(usgsIdLst[k], Target)
        y[k, :] = dataObs.flatten()
    print("read ssc", time.time() - t0)
    return y


def readForcingGage(usgsId, varLst=forcingLst, *, dataset='nldas'):
    forc_variables = forcing_data.loc[forcing_data['sta_id']==usgsId]
    nf = len(varLst)
    out = np.zeros([nt, nf])
    for k in range(nf):
        out[:, k] = forc_variables[varLst[k]].values
    return out


def readForcing(usgsIdLst, varLst):
    t0 = time.time()

    x = np.zeros([len(usgsIdLst), nt, len(varLst)])   #previous version is np.empty

    for k in range(len(usgsIdLst)):
        data = readForcingGage(usgsIdLst[k], varLst)
        x[k, :, :] = data
    print("read forcing data", time.time() - t0)
    #changing nan to zero
    #x = np.nan_to_num(x)
    return x


def readAttrAll(*, saveDict=False):
    dataFolder = os.path.join(dirDB, 'camels_attributes_v2.0',
                              'camels_attributes_v2.0')
    fDict = dict()  # factorize dict
    varDict = dict()
    varLst = list()
    outLst = list()
    keyLst = ['topo', 'clim', 'hydro', 'vege', 'soil', 'geol']

    for key in keyLst:
        dataFile = os.path.join(dataFolder, 'camels_' + key + '.txt')
        dataTemp = pd.read_csv(dataFile, sep=';')
        varLstTemp = list(dataTemp.columns[1:])
        varDict[key] = varLstTemp
        varLst.extend(varLstTemp)
        k = 0
        nGage = len(gageDict['id'])
        outTemp = np.full([nGage, len(varLstTemp)], np.nan)
        for field in varLstTemp:
            if is_string_dtype(dataTemp[field]):
                value, ref = pd.factorize(dataTemp[field], sort=True)
                outTemp[:, k] = value
                fDict[field] = ref.tolist()
            elif is_numeric_dtype(dataTemp[field]):
                outTemp[:, k] = dataTemp[field].values
            k = k + 1
        outLst.append(outTemp)
    out = np.concatenate(outLst, 1)
    if saveDict is True:
        fileName = os.path.join(dataFolder, 'dictFactorize.json')
        with open(fileName, 'w') as fp:
            json.dump(fDict, fp, indent=4)
        fileName = os.path.join(dataFolder, 'dictAttribute.json')
        with open(fileName, 'w') as fp:
            json.dump(varDict, fp, indent=4)
    return out, varLst


def readAttr(usgsIdLst, varLst):
    attrAll, varLstAll = readAttrAll()
    indVar = list()
    for var in varLst:
        indVar.append(varLstAll.index(var))
    idLstAll = gageDict['id']
    C, indGrid, ind2 = np.intersect1d(idLstAll, usgsIdLst, return_indices=True)
    temp = attrAll[indGrid, :]
    out = temp[:, indVar]
    return out


def calStat(x):
    a = x.flatten()
    bb = a[~np.isnan(a)]  # kick out Nan
    b = bb[bb != (-999999)]
    p10 = np.percentile(b, 10).astype(float)
    p90 = np.percentile(b, 90).astype(float)
    mean = np.mean(b).astype(float)
    std = np.std(b).astype(float)
    if std < 0.001:
        std = 1
    return [p10, p90, mean, std]


def calSed(x):
    a = x.flatten()
    bb = a[~np.isnan(a)]  # kick out Nan
    b = bb[bb != (-999999)]
    b = np.log10(b)  # do some tranformation to change gamma characteristics
    p10 = np.percentile(b, 10).astype(float)
    p90 = np.percentile(b, 90).astype(float)
    mean = np.mean(b).astype(float)
    std = np.std(b).astype(float)
    if std < 0.001:
        std = 1
    return [p10, p90, mean, std]

def calStatgamma(x):
    a = x.flatten()
    b = np.log10(np.sqrt(a)+0.1)
    p10 = np.percentile(b, 10).astype(float)
    p90 = np.percentile(b, 90).astype(float)
    mean = np.mean(b).astype(float)
    std = np.std(b).astype(float)
    if std < 0.001:
        std = 1
    return [p10, p90, mean, std]

def calStatbasinnorm(x):  # for daily streamflow normalized by basin area and precipitation
   ## basinarea = readAttr(gageDict['id'], ['area_gages2'])
    x[x<0]=0

    # x[x = -99999] = 0
    #np.where(x==(-999999), 0, x)
    basinarea = attr_data['DRAIN_SQKM']
   ## meanprep = readAttr(gageDict['id'], ['p_mean'])
    meanprep = attr_data['PPTAVG_BASIN'] #  anual average precipitation
    # meanprep = readAttr(gageDict['id'], ['q_mean'])
    temparea = np.tile(basinarea, ( x.shape[1], 1)).transpose()
    tempprep = np.tile(meanprep, ( x.shape[1],1)).transpose()
    flowua = (x * 0.0283168 * 3600 * 24) / ((temparea * (10 ** 6)) * (tempprep * 10 ** (-2))/365) # unit (m^3/day)/(m^3/day)
    a = flowua.flatten()
    b = a[~np.isnan(a)] # kick out Nan
    b = np.log10(np.sqrt(b)+0.1) # do some transformation to change gamma characteristics plus 0.1 for 0 values
    #p10 = np.percentile(b, 10).astype(float)
    #p90 = np.percentile(b, 90).astype(float)


    p10 = np.nanpercentile(b, 10).astype(float)
    p90 = np.nanpercentile(b, 90).astype(float)
    mean = np.nanmean(b).astype(float)
    std = np.nanstd(b).astype(float)
    if std < 0.001:
        std = 1
    return [p10, p90, mean, std]


def calStatAll():
    statDict = dict()

    idLst = forcing_data['sta_id'].unique()
    y = readUsgs(idLst)
    if Target == ['80154_mean']:
        statDict['80154_mean'] = calSed(y)    #calStatbasinnorm(y), calSed(y)
    elif Target == ['streamflow'] or ['PRCP (Daymet)']:
        statDict['streamflow'] = calStatgamma(y)
    else:
        statDict[Target] = calStat(y)
    # forcing
    # x = readUsgs(idLst, forcingLst)
    x = readForcing(idLst, forcingLst)
    for k in range(len(forcingLst)):
        var = forcingLst[k]
        if var=='APCP':
            statDict[var] = calStatgamma(x[:, :, k])
        elif var=='PRCP (Daymet)':
            statDict[var] = calStatgamma(x[:, :, k])
        elif var=='streamflow':
            statDict[var] = calStatbasinnorm(x[:, :, k])
        else:
            statDict[var] = calStat(x[:, :, k])
    # const attribute
    ##attrData, attrLst = readAttrAll()
    attrLst = attrLstSel
    attrData = np.empty([len(idLst), len(attrLst)])
    for i, ii in enumerate(attrLst):
        attrData[:,i] = attr_data[ii].values
    for k in range(len(attrLst)):
        var = attrLst[k]
        statDict[var] = calStat(attrData[:, k])
    statFile = os.path.join(dirDB, 'Statistics_basinnorm.json')
    with open(statFile, 'w') as fp:
        json.dump(statDict, fp, indent=4)


def transNorm(x, varLst, *, toNorm):
    if type(varLst) is str:
        varLst = [varLst]
    out = np.zeros(x.shape)

    for k in range(len(varLst)):
        var = varLst[k]

        stat = statDict[var]
        if toNorm is True:
            if len(x.shape) == 3:
                if var == 'streamflow' or var == 'PRCP (Daymet)':
                    x[:, :, k] = np.log10(np.sqrt(x[:, :, k] + 0.1))
                elif var == '80154_mean':
                    x[:, :, k] = np.log10(np.sqrt(x[:, :, k]))
                #simple standardization
                out[:, :, k] = (x[:, :, k] - stat[2]) / stat[3]

            elif len(x.shape) == 2:
                if var == 'PRCP (Daymet)' or var == 'streamflow':
                    x[:, k] = np.log10(np.sqrt(x[:, k] + 0.1))
                if var == '80154_mean':
                    x[:, k] = np.log10(np.sqrt(x[:, k]))
                out[:, k] = (x[:, k] - stat[2]) / stat[3]
        else: #denormalization
            if len(x.shape) == 3:
                out[:, :, k] = x[:, :, k] * stat[3] + stat[2]
                if var == 'streamflow' or var == 'PRCP (Daymet)':
                    out[:, :, k] = (np.power(10, out[:, :, k]) - 0.1) ** 2
                if var == '80154_mean':
                    out[:, :, k] = (np.power(10, out[:, :, k])) ** 2


            elif len(x.shape) == 2:
                out[:, k] = x[:, k] * stat[3] + stat[2]
                if var == 'streamflow' or 'PRCP (Daymet)':
                    out[:, k] = (np.power(10, out[:, k]) - 0.1) ** 2
                if var == '80154_mean':
                    out[:, k] = (np.power(10, out[:, k])) ** 2


    return out

def basinNorm(x, gageid, toNorm):
    # for regional training, gageid should be numpyarray
    #if type(gageid) is str:
       # if gageid == 'All':
         #   gageid = gageDict['id']
    nd = len(x.shape)
    meanprep = attr_data['PPTAVG_BASIN']
    basinarea = attr_data['DRAIN_SQKM']

   # basinarea = readAttr(gageid, ['area_gages2'])
  #  meanprep = readAttr(gageid, ['p_mean'])
   # # meanprep = readAttr(gageid, ['q_mean'])  #this line was ponded from the beginning
    if nd == 3 and x.shape[2] == 1:
        x = x[:,:,0] # unsqueeze the original 3 dimension matrix
    temparea = np.tile(basinarea, ( x.shape[1], 1)).transpose()
    tempprep = np.tile(meanprep, (x.shape[1], 1)).transpose()
    if toNorm is True:
        flow = (x * 0.0283168 * 3600 * 24) / ((temparea * (10 ** 6)) * (tempprep * 10 ** (-2))/365) # (m^3/day)/(m^3/day)
    else:

        flow = x * ((temparea * (10 ** 6)) * (tempprep * 10 ** (-2))/365)/(0.0283168 * 3600 * 24)
    if nd == 3:
        flow = np.expand_dims(flow, axis=2)
    return flow

def createSubsetAll(opt, **kw):
    if opt is 'all':
        idLst = gageDict['id']
        subsetFile = os.path.join(dirDB, 'Subset', 'all.csv')
        np.savetxt(subsetFile, idLst, delimiter=',', fmt='%d')

# Define and initialize module variables
if os.path.isdir(pathCamels['DB']):
    dirDB = pathCamels['DB']
    gageDict = readGageInfo(dirDB)
    statFile = os.path.join(dirDB, 'Statistics_basinnorm.json')
    if not os.path.isfile(statFile):
        calStatAll()
    with open(statFile, 'r') as fp:
        statDict = json.load(fp)
else:
    dirDB = None
    gageDict = None
    statDict = None

def initcamels(forcing, attribute, target, rootDB = pathCamels['DB']):
    # reinitialize module variable
    global dirDB, gageDict, statDict, forcing_data, attr_data, Target
    dirDB = rootDB
    forcing_data = forcing
    attr_data = attribute
    Target = target
   # gageDict = readGageInfo(dirDB)
    statFile = os.path.join(dirDB, 'Statistics_basinnorm.json')
    if not os.path.isfile(statFile):
        calStatAll()
    with open(statFile, 'r') as fp:
        statDict = json.load(fp)


class DataframeCamels(Dataframe):
    #what's this?
    def __init__(self, *, subset='All', tRange):
        self.subset = subset        
        #if subset == 'All':  # change to read subset later
#            self.usgsId = gageDict['id']
 #           crd = np.zeros([len(self.usgsId), 2])
  #          crd[:, 0] = gageDict['lat']
   #         crd[:, 1] = gageDict['lon']
    #        self.crd = crd
     #   elif type(subset) is list:
      #      self.usgsId = np.array(subset)
       #     crd = np.zeros([len(self.usgsId), 2])
        #    C, ind1, ind2 = np.intersect1d(self.usgsId, gageDict['id'], return_indices=True)
         #   crd[:, 0] = gageDict['lat'][ind2]
          #  crd[:, 1] = gageDict['lon'][ind2]
           # self.crd = crd
      #  else:
        #    raise Exception('The format of subset is not correct!')
        self.time = utils.time.tRange2Array(tRange)        

    def getGeo(self):
        return self.crd

    def getT(self):
        return self.time

    def getDataObs(self, Target, forcing_path, attr_path, *, doNorm=False, rmNan=False, basinnorm = False):

        df_obs = pd.DataFrame()
        inputfiles = os.path.join(forcing_path)   #obs_18basins     forcing_350days_T_S_GAGESII
        dfMain = pd.read_csv(inputfiles)
        inputfiles = os.path.join(attr_path)    #attr_18basins   attr_350days_T_S_GAGESII
        dfC = pd.read_csv(inputfiles)
        nNodes = len(dfC['STAID'])
        id_order_dfMain = dfMain['sta_id'].unique()
        seg_id = pd.DataFrame()
        dfC1 = pd.DataFrame()
        for i, ii in enumerate(id_order_dfMain):  # to have the same order of seg_id_nat in both dfMain & dfC
            A = dfC.loc[dfC['STAID'] == ii]
            dfC1 = dfC1.append(A, ignore_index=True)
        dfC = dfC1
        seg_id['STAID'] = dfC['STAID']
        df_obs[Target] = dfMain[Target]    #

        y = np.empty([nNodes, ntobs])
        for i in range(nNodes):
           
            a = ntobs * i
            b = ntobs * (i + 1)
            data = df_obs.iloc[a:b]
            kk = dfMain.columns.get_loc('sta_id')
            id = dfMain.iloc[a:a + 1, kk]
            val_mask = seg_id == id[a]
            k = val_mask.index[val_mask['STAID'] == True][0]
            y[k, :] = data.iloc[:, 0]



        data = y



        #data = readUsgs(self.usgsId)
     #   if basinnorm is True:
        #    for k in range(len(varLst)):
           #     var = varLst[k]
                #    stat = statDict[var]

     #       data = basinNorm(data, gageid=self.usgsId, toNorm=True)
        data = np.expand_dims(data, axis=2)
        C, ind1, ind2 = np.intersect1d(self.time, tLstobs, return_indices=True)
        data = data[:, ind2, :]  # What is this line?
        if doNorm is True:
            data = transNorm(data, Target, toNorm=True)
        if rmNan is True:
            data[np.where(np.isnan(data))] = 0
            # data[np.where(np.isnan(data))] = -99
        return data

    def getDataTs(self,forcing_path, attr_path, out, *, varLst=forcingLst, doNorm=True, rmNan=True):
        if type(varLst) is str:
            varLst = [varLst]
        # read ts forcing
        #rootDatabase = os.path.join(os.path.sep, absRoot, 'scratch', 'SNTemp')
        inputfiles = os.path.join(forcing_path)   #   forcing_350days_T_S_GAGESII
        dfMain = pd.read_csv(inputfiles)
        ############ I'm just curious the following line
        dfMain[dfMain['streamflow']<0] = 0  # try to kick -99999 and some of negative values
        inputfiles = os.path.join(attr_path)       #   attr_350days_T_S_GAGESII
        dfC = pd.read_csv(inputfiles)
        nNodes = len(dfC['STAID'])
        x = np.empty([nNodes, ntobs, len(forcingLst)])
        id_order_dfMain = dfMain['sta_id'].unique()
        seg_id = pd.DataFrame() #what's this? same as the uniques?
        dfC1 = pd.DataFrame()
        #selecting attribute by stations
        for i, ii in enumerate(id_order_dfMain):  # to have the same order of seg_id_nat in both dfMain & dfC
            A = dfC.loc[dfC['STAID'] == ii]
            dfC1 = dfC1.append(A, ignore_index=True)
        dfC = dfC1
        seg_id['site_no'] = dfC['STAID']
        forcing = pd.DataFrame()
        for i , ii in enumerate(forcingLst):
            forcing[ii] = dfMain[ii]
        for i in range(nNodes):
                
            a = ntobs * i
            b = ntobs * (i + 1)
            data = forcing.iloc[a:b, :]
            kk = dfMain.columns.get_loc('sta_id')
            id = dfMain.iloc[a:a+1, kk]
            #val_mask = seg_id == id[a]
            #k = val_mask.index[val_mask['site_no'] == True][0]

            x[i, :, :] = data

            #print(x.size)
            #changing some nan to zero
            #x = np.nan_to_num(x)
        data = x # readForcing(self.usgsId, varLst) # data:[gage*day*variable]
        C, ind1, ind2 = np.intersect1d(self.time, tLst, return_indices=True) #C: training period, ind1: index of training days, ind2: index of testing? days
        data = data[:, ind2, :]
        if os.path.isdir(out):
            pass
        else:
            os.makedirs(out)
        np.save(os.path.join(out, 'x.npy'), data) #x.npy = forcing data
        # Apply a normalization
        if doNorm is True:
            data = transNorm(data, varLst, toNorm=True)
        if rmNan is True:
            data[np.where(np.isnan(data))] = 0
        return data

    def getDataConst(self, forcing_path, attr_path, *, varLst=attrLstSel, doNorm=True, rmNan=True):
        if type(varLst) is str:
            varLst = [varLst]
        inputfiles = os.path.join(forcing_path)  # obs_18basins  forcing_350days_T_S_GAGESII
        dfMain = pd.read_csv(inputfiles)
        inputfiles = os.path.join(attr_path)   #attr_18basins   attr_350days_T_S_GAGESII
        dfC = pd.read_csv(inputfiles)
        nNodes = len(dfC['STAID'])
        x = np.empty([nNodes, ntobs, len(forcingLst)])
        id_order_dfMain = dfMain['sta_id'].unique()
        seg_id = pd.DataFrame()
        dfC1 = pd.DataFrame()
        for i, ii in enumerate(id_order_dfMain):  # to have the same order of seg_id_nat in both dfMain & dfC
            A = dfC.loc[dfC['STAID'] == ii]
            dfC1 = dfC1.append(A, ignore_index=True)
        dfC = dfC
        c = np.empty([nNodes, len(varLst)])
        df_constant = pd.DataFrame()
        for i, ii in enumerate(varLst):
            df_constant[ii] = dfC[ii]
            c[:, i] = df_constant.iloc[:, i]
       # data = readAttr(self.usgsId, varLst)
        data = c
        if doNorm is True:
            data = transNorm(data, varLst, toNorm=True)
        if rmNan is True:
            data[np.where(np.isnan(data))] = 0
        return data

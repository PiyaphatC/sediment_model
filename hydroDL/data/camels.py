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
tRange = [19800101, 20191231]
tRangeobs = [19800101, 20191231]    #[19801001, 20161001] #  streamflow observations
tLst = utils.time.tRange2Array(tRange)
tLstobs = utils.time.tRange2Array(tRangeobs)
nt = len(tLst)
ntobs = len(tLstobs)


forcingLst = ['prcp(mm/day)', 'new_streamflow','srad(W/m2)','swe(mm)',	'tmax(C)',	'tmin(C)',	'vp(Pa)']#, 'srad(W/m2)','swe(mm)',	'tmax(C)',	'tmin(C)',	'vp(Pa)' ] #, 'streamflow','80154_mean']
# attr below are the new selection
attrLstSel = ['PPTAVG_BASIN',
	            'DDENS_2009',	'STOR_NID_2009',
                'MAJ_DDENS_2009',
                'DEVNLCD06',		'PLANTNLCD06',    'FORESTNLCD06',
                'HGA',	'HGB',	'HGC',	'HGD',
                'PERMAVE',	'NO4AVE',	'NO200AVE',
                'NO10AVE',	'CLAYAVE',	'SILTAVE',	'SANDAVE',
                'KFACT_UP',
                'RFACT',	'ELEV_MEAN_M_BASIN',	'SLOPE_PCT',	'ASPECT_DEGREES',	
                'DRAIN_SQKM',	'HYDRO_DISTURB_INDX']

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

def readUsgsGage(usgsId, Target, num, *, readQc=False):
    obsx = forcing_data.loc[(forcing_data['sta_id']==usgsId, Target)].reset_index(drop = True)
    obs = obsx.loc[0:ntobs-1]
    # print(obs.shape)
    return obs

def readUsgs(usgsIdLst: object) -> object:
    t0 = time.time()
    y = np.zeros([len(usgsIdLst), ntobs])
    for k in range(len(usgsIdLst)):
        dataObs = readUsgsGage(usgsIdLst[k], Target, k)
        y[k, :] = dataObs.values.reshape(-1)
    print("read ssc", time.time() - t0)
    return y

def readForcingGage(usgsId, varLst=forcingLst, *, dataset='nldas'):
    forx = forcing_data.loc[forcing_data['sta_id']==usgsId].reset_index(drop = True)
    forc_variables = forx.loc[0:ntobs-1]
    nf = len(varLst)
    out = np.zeros([nt, nf])
    for k in range(nf):
        out[:, k] = forc_variables[varLst[k]].values
    return out


def readForcing(usgsIdLst, varLst):
    t0 = time.time()

    x = np.empty([len(usgsIdLst), nt, len(varLst)])   #previous version is np.empty

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
    b = np.log10(np.sqrt(b)+ 0.1)  # do a transformation
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
    else:
        statDict[Target] = calStat(y)
    # forcing
    x = readForcing(idLst, forcingLst)
    for k in range(len(forcingLst)):
        var = forcingLst[k]
        if var=='APCP':
            statDict[var] = calStat(x[:, :, k])
        elif var=='prcp(mm/day)':
            statDict[var] = calStat(x[:, :, k])
        elif var=='new_streamflow':
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
    out = np.empty(x.shape)

    for k in range(len(varLst)):
        var = varLst[k]

        stat = statDict[var]
        if toNorm is True:
            if len(x.shape) == 3:
                if var == 'new_streamflow' or var == 'prcp(mm/day)':
                    x[:, :, k] = np.log10(np.sqrt(x[:, :, k]) + 0.1)
                elif var == '80154_mean':
                    x[:, :, k] = np.log10(np.sqrt(x[:, :, k]) + 0.1)
                #simple standardization
                out[:, :, k] = (x[:, :, k] - stat[2]) / stat[3]

            elif len(x.shape) == 2:
                if var == 'prcp(mm/day)' or var == 'new_streamflow':
                    x[:, k] = np.log10(np.sqrt(x[:, k]) + 0.1)
                if var == '80154_mean':
                    x[:, k] = np.log10(np.sqrt(x[:, k]) + 0.1)
                out[:, k] = (x[:, k] - stat[2]) / stat[3]
        else: #denormalization
            if len(x.shape) == 3:
                out[:, :, k] = x[:, :, k] * stat[3] + stat[2]
                if var == 'new_streamflow' or var == 'prcp(mm/day)':
                    out[:, :, k] = (np.power(10, out[:, :, k]) - 0.1) ** 2
                if var == '80154_mean':
                    out[:, :, k] = (np.power(10, out[:, :, k]) - 0.1) ** 2


            elif len(x.shape) == 2:
                out[:, k] = x[:, k] * stat[3] + stat[2]
                if var == 'streamflow' or 'prcp(mm/day)':
                    out[:, k] = (np.power(10, out[:, k]) - 0.1) ** 2
                if var == '80154_mean':
                    out[:, k] = (np.power(10, out[:, k])) ** 2


    return out

def basinNorm(x, gageid, toNorm):

    nd = len(x.shape)
    meanprep = attr_data['PPTAVG_BASIN']
    basinarea = attr_data['DRAIN_SQKM']

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
        # if subset == 'All':  # change to read subset later
        #    self.usgsId = gageDict['id']
        #    crd = np.zeros([len(self.usgsId), 2])
        #    crd[:, 0] = gageDict['lat']
        #    crd[:, 1] = gageDict['lon']
        #    self.crd = crd
        # elif type(subset) is list:
        #    self.usgsId = np.array(subset)
        #    crd = np.zeros([len(self.usgsId), 2])
        #    C, ind1, ind2 = np.intersect1d(self.usgsId, gageDict['id'], return_indices=True)
        #    crd[:, 0] = gageDict['lat'][ind2]
        #    crd[:, 1] = gageDict['lon'][ind2]
        #    self.crd = crd
        # else:
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
        dfMain['80154_mean'].mask(dfMain['80154_mean'] >= 10000.0 , np.nan, inplace=True)
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
        usgsIdLst = seg_id.to_numpy()
        df_obs[Target] = dfMain[Target]    #

        # y = np.empty([nNodes, ntobs])
        # for i in range(nNodes):

        y = np.zeros([len(usgsIdLst), ntobs])
        for k in range(len(usgsIdLst)):
            dataObs = readUsgsGage(usgsIdLst[k].item(), Target, k)
            y[k, :] = dataObs.values.reshape(-1)


            # a = ntobs * i
            # b = ntobs * (i + 1)
            # data = df_obs.iloc[a:b]
            # kk = dfMain.columns.get_loc('sta_id')
            # id = dfMain.iloc[a:a + 1, kk]
            # val_mask = seg_id == id[a]
            # k = val_mask.index[val_mask['STAID'] == True][0]
            # y[k, :] = data.iloc[:, 0]

        # hellooo

        data = y
        #select only period we want such as training period or testing period
        data = np.expand_dims(data, axis=2) #change from (basin, tTraining) to (basin, tTrain, 1)
        #tLstobs = range of observed data
        C, ind1, ind2 = np.intersect1d(self.time, tLstobs, return_indices=True)
        data = data[:, ind2, :]  # select only the period we want
        if doNorm is True:
            data = transNorm(data, Target, toNorm=True)
        if doNorm is False:
            data = transNorm(data, Target, toNorm=False)
        if rmNan is True:
            data[np.where(np.isnan(data))] = 0
            # data[np.where(np.isnan(data))] = -99
        return data

    def getDataForc(self,forcing_path, attr_path, out, *, varLst=forcingLst, doNorm=True, rmNan=True): #forcing datasets
        if type(varLst) is str:
            varLst = [varLst]
        # read ts forcing
        inputfiles = os.path.join(forcing_path)
        dfMain = pd.read_csv(inputfiles)
        #dfMain['streamflow'].mask(dfMain['streamflow'] < 0.0, 0.0, inplace=True)
        inputfiles = os.path.join(attr_path)
        dfC = pd.read_csv(inputfiles)
        nNodes = len(dfC['STAID'])
        x = np.empty([nNodes, ntobs, len(forcingLst)])
        id_order_dfMain = dfMain['sta_id'].unique()
        seg_id = pd.DataFrame()
        dfC1 = pd.DataFrame()
        #selecting attribute by stations
        for i, ii in enumerate(id_order_dfMain):  # to have the same order of seg_id_nat in both dfMain & dfC
            A = dfC.loc[dfC['STAID'] == ii]
            dfC1 = dfC1.append(A, ignore_index=True)
        dfC = dfC1
        seg_id['site_no'] = dfC['STAID']
        usgsIdLst = seg_id.to_numpy()
        forcing = pd.DataFrame()
        for i , ii in enumerate(forcingLst):
            forcing[ii] = dfMain[ii]

        x = np.empty([len(usgsIdLst), nt, len(varLst)])
        for k in range(len(usgsIdLst)):
            data = readForcingGage(usgsIdLst[k].item(), varLst)
            x[k, :, :] = data
            # a = ntobs * i
            # b = ntobs * (i + 1)
            # data = forcing.iloc[a:b, :]




            #print(x.size)
            #changing some nan to zero
            #x = np.nan_to_num(x)
        data = x
        #select only training and testing period we want
        C, ind1, ind2 = np.intersect1d(self.time, tLst, return_indices=True) #C: training period, ind1: index of training days, ind2: index of testing? days
        data = data[:, ind2, :]
        if os.path.isdir(out):
            pass
        else:
            os.makedirs(out)
        np.save(os.path.join(out, 'forcing_train.npy'), data)
        # Apply a normalization
        if doNorm is True:
            data = transNorm(data, varLst, toNorm=True)
        if doNorm is False:
            data = transNorm(data, varLst, toNorm=False)
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
        dfC = dfC1
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

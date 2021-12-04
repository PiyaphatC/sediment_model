import numpy as np
import scipy.stats

keyLst = ['Bias', 'RMSE', 'ubRMSE', 'Corr']


def statError(pred, target):
    ngrid, nt = pred.shape  # I changed it? Park changes this from "pred.shape" to "len(pred"
    #ngrid: number of basin


    #############################################
    # Bias
    Bias = np.nanmean(pred - target, axis=1)
    # RMSE

   ## RMSE = np.sqrt(np.nanmean((pred - target)**2, axis=1))
    # ubRMSE
   ## predMean = np.tile(np.nanmean(pred, axis=1), (nt, 1)).transpose()
   ## targetMean = np.tile(np.nanmean(target, axis=1), (nt, 1)).transpose()
   ## predAnom = pred - predMean.

  ##  targetAnom = target - targetMean
  ##  ubRMSE = np.sqrt(np.nanmean((predAnom - targetAnom)**2, axis=1))
    #defining RNSE & ubRMSE & flat metrics
    RMSE = np.full(ngrid, np.nan)
    ubRMSE = np.full(ngrid, np.nan)
    predflat = []
    targetflat = []
    # rho R2 NSE
    Corr = np.full(ngrid, np.nan)
    R2 = np.full(ngrid, np.nan)
    NSE = np.full(ngrid, np.nan)

    PBiaslow = np.full(ngrid, np.nan)
    PBiashigh = np.full(ngrid, np.nan)
    PBias = np.full(ngrid, np.nan)
    for k in range(0, ngrid):
        x = pred[k, :]   #predicted SSC
        y = target[k, :] #observed SSC
        ind = np.where(np.logical_and(~np.isnan(x), ~np.isnan(y)))[0] #what's this line? Park
        if ind.shape[0] > 0:
            xx = x[ind]
            yy = y[ind]
            # RMSE by Farshid
            RMSE[k] = np.sqrt(np.nanmean((xx - yy) ** 2))
            predMean = np.tile(np.nanmean(xx), (len(xx))).transpose()
            targetMean = np.tile(np.nanmean(yy), (len(yy))).transpose()
            predAnom = xx - predMean
            targetAnom = yy - targetMean
            ubRMSE[k] = np.sqrt(np.nanmean((predAnom - targetAnom) ** 2))
            predflat = np.append(predflat, xx)
            targetflat = np.append(targetflat, yy)


            # percent bias
            PBias[k] = np.sum(xx - yy) / np.sum(yy) * 100

            # FHV the peak flows bias 10%
            # FLV the low flows bias bottom 30%, log space
            pred_sort = np.sort(xx)
            target_sort = np.sort(yy)
            indexlow = round(0.3 * len(pred_sort))
            indexhigh = round(0.9 * len(pred_sort))
            lowpred = pred_sort[:indexlow]
            highpred = pred_sort[indexhigh:]
            lowtarget = target_sort[:indexlow]
            hightarget = target_sort[indexhigh:]
            PBiaslow[k] = np.sum(lowpred - lowtarget) / np.sum(lowtarget) * 100
            PBiashigh[k] = np.sum(highpred - hightarget) / np.sum(hightarget) * 100

            if ind.shape[0] > 1:
                # Theoretically at least two points for correlation
                yymean = yy.mean()
                SST = np.sum((yy - yymean) ** 2)
                SSReg = np.sum((xx - yymean) ** 2)
                SSRes = np.sum((xx - yy) ** 2)
                NSE[k] = 1 - (SSRes / SST)
                Corr[k] = scipy.stats.pearsonr(xx, yy)[0]
                xxmean = xx.mean()
                R2[k] = ((np.sum((yy-yymean)*(xx-xxmean))) / (((np.sum((yy-yymean)**2)) ** 0.5)*(np.sum((xx - xxmean)**2)) ** 0.5))**2





    ### use flatted pred and target to have one value for Bias, RMSE, ubRMSE
    predflat = predflat.flatten()
    targetflat = targetflat.flatten()
    Biasflat = np.nanmean(predflat-targetflat)
    absBiasflat = np.nanmean(abs(predflat-targetflat))
    RMSEflat = np.sqrt(np.nanmean((predflat-targetflat)**2))
    ubRMSEflat=np.sqrt(((RMSEflat)**2)-((Biasflat)**2))
    ind = np.where(np.logical_and(~np.isnan(predflat), ~np.isnan(targetflat)))[0]
    if ind.shape[0] > 0:
        xx = predflat[ind]
        yy = targetflat[ind]
        corrflat = scipy.stats.pearsonr(xx, yy)[0]

    NSEflat = 1 - ((np.nansum((predflat-targetflat)**2))/(np.nansum((targetflat - np.nanmean(targetflat))**2)))

    outDict = dict(Bias=Bias, RMSE=RMSE, ubRMSE=ubRMSE, Corr=Corr, R2=R2, NSE=NSE,
                   FLV=PBiaslow, FHV=PBiashigh, PBias=PBias, Biasflat=Biasflat,
                   absBiasflat=absBiasflat, RMSEflat=RMSEflat, ubRMSEflat=ubRMSEflat,
                   corrflat=corrflat, NSEflat=NSEflat)  #
    return outDict



def statError_res(pred, target, pred_res, target_res):
    ngrid, nt = pred.shape  # I changed it



    #############################################
    # Bias
    Bias = np.nanmean(pred - target, axis=1)
    # RMSE
   # RMSE = np.sqrt(np.nanmean((pred - target)**2, axis=1))
    
   # predMean = np.tile(np.nanmean(pred, axis=1), (nt, 1)).transpose()
   # targetMean = np.tile(np.nanmean(target, axis=1), (nt, 1)).transpose()
   # predAnom = pred - predMean
  #  targetAnom = target - targetMean
  #  ubRMSE = np.sqrt(np.nanmean((predAnom - targetAnom)**2, axis=1))
    #defining RNSE & ubRMSE & flat metrics
    RMSE = np.full(ngrid, np.nan)
    ubRMSE = np.full(ngrid, np.nan)
    predflat = []
    targetflat = []
    # rho R2 NSE
    Corr = np.full(ngrid, np.nan)
    Corr_res = np.full(ngrid, np.nan)
    R2 = np.full(ngrid, np.nan)
    R2_res = np.full(ngrid, np.nan)
    NSE = np.full(ngrid, np.nan)
    NSE_res = np.full(ngrid, np.nan)

    PBiaslow = np.full(ngrid, np.nan)
    PBiashigh = np.full(ngrid, np.nan)
    PBias = np.full(ngrid, np.nan)
    PBias_res = np.full(ngrid, np.nan)
    for k in range(0, ngrid):
        x = pred[k, :]
        x_res = pred_res[k, :]
        y = target[k, :]
        y_res = target_res[k, :]
        ind = np.where(np.logical_and(~np.isnan(x), ~np.isnan(y)))[0]
        ind_res = np.where(np.logical_and(~np.isnan(x_res), ~np.isnan(y_res)))[0]
        if ind.shape[0] > 0:
            xx = x[ind]
            yy = y[ind]
            # RMSE by Farshid
            RMSE[k] = np.sqrt(np.nanmean((xx - yy) ** 2))
            predMean = np.tile(np.nanmean(xx), (len(xx))).transpose()
            targetMean = np.tile(np.nanmean(yy), (len(yy))).transpose()
            predAnom = xx - predMean
            targetAnom = yy - targetMean
            ubRMSE[k] = np.sqrt(np.nanmean((predAnom - targetAnom) ** 2))
            predflat = np.append(predflat, xx)
            targetflat = np.append(targetflat, yy)


            # percent bias
            PBias[k] = np.sum(xx - yy) / np.sum(yy) * 100

            # # FHV the peak flows bias 10%
            # # FLV the low flows bias bottom 30%, log space
            # pred_sort = np.sort(xx)
            # target_sort = np.sort(yy)
            # indexlow = round(0.3 * len(pred_sort))
            # indexhigh = round(0.9 * len(pred_sort))
            # lowpred = pred_sort[:indexlow]
            # highpred = pred_sort[indexhigh:]
            # lowtarget = target_sort[:indexlow]
            # hightarget = target_sort[indexhigh:]
            # PBiaslow[k] = np.sum(lowpred - lowtarget) / np.sum(lowtarget) * 100
            # PBiashigh[k] = np.sum(highpred - hightarget) / np.sum(hightarget) * 100

            if ind.shape[0] > 1:
                # Theoretically at least two points for correlation
                # xx = predicted value
                # yy = observed value
                Corr[k] = scipy.stats.pearsonr(xx, yy)[0]
                yymean = yy.mean()
                SST = np.sum((yy-yymean)**2)
                SSReg = np.sum((xx-yymean)**2)
                SSRes = np.sum((yy-xx)**2)
                NSE[k] = 1-SSRes/SST
                xxmean = xx.mean()
                R2[k] = ((np.sum((yy-yymean)*(xx-xxmean))) / (((np.sum((yy-yymean)**2)) ** 0.5)*(np.sum((xx - xxmean)**2)) ** 0.5))**2



        if ind_res.shape[0] > 0:
            xx_res = x_res[ind_res]
            yy_res = y_res[ind_res]
            # percent bias_res
            PBias_res[k] = np.sum(xx_res - yy_res) / np.sum(yy_res) * 100
            if ind_res.shape[0] > 1:
                # Theoretically at least two points for correlation
                Corr_res[k] = scipy.stats.pearsonr(xx_res, yy_res)[0]
                yymean_res = yy_res.mean()
                SST_res = np.sum((yy_res-yymean_res)**2)
                SSReg_res = np.sum((xx_res-yymean_res)**2)
                SSRes_res = np.sum((yy_res-xx_res)**2)
                NSE_res[k] = 1-(SSRes_res/SST_res)
                xxmean_res = xx_res.mean()
                R2_res[k] = ((np.sum((yy_res-yymean_res)*(xx_res-xxmean_res))) / (((np.sum((yy_res-yymean_res)**2)) ** 0.5)*(np.sum((xx_res - xxmean_res)**2)) ** 0.5))**2


    ### use flatted pred and target to have one value for Bias, RMSE, ubRMSE
    predflat = predflat.flatten()
    targetflat = targetflat.flatten()
    Biasflat = np.nanmean(predflat-targetflat)
    absBiasflat = np.nanmean(abs(predflat-targetflat))
    RMSEflat = np.sqrt(np.nanmean((predflat-targetflat)**2))
    ubRMSEflat=np.sqrt(((RMSEflat)**2)-((Biasflat)**2))
    ind = np.where(np.logical_and(~np.isnan(predflat), ~np.isnan(targetflat)))[0]
    if ind.shape[0] > 0:
        xx = predflat[ind]
        yy = targetflat[ind]
        corrflat = scipy.stats.pearsonr(xx, yy)[0]

    NSEflat = 1 - ((np.nansum((predflat-targetflat)**2))/(np.nansum((targetflat - np.nanmean(targetflat))**2)))

    outDict = dict(Bias=Bias, RMSE=RMSE, ubRMSE=ubRMSE, Corr=Corr, R2=R2, NSE=NSE, R2_res=R2_res,
                   FLV=PBiaslow, FHV=PBiashigh, PBias=PBias, Biasflat=Biasflat, NSE_res=NSE_res,
                   absBiasflat=absBiasflat, RMSEflat=RMSEflat, ubRMSEflat=ubRMSEflat,
                   corrflat=corrflat, NSEflat=NSEflat, PBias_res=PBias_res, Corr_res=Corr_res)  #
    return outDict
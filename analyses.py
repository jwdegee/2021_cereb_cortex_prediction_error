import os
import numpy as np
import scipy as sp
import scipy.stats as stats
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from joblib import Parallel, delayed
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import AnovaRM
import pingouin as pg

from IPython import embed as shell

# from hddm_tools import ddm_utils
# from jwtools import myfuncs

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
sns.set(style='ticks', font='Arial', font_scale=1, rc={
    'axes.linewidth': 0.25, 
    'axes.labelsize': 7, 
    'axes.titlesize': 7, 
    'xtick.labelsize': 6, 
    'ytick.labelsize': 6, 
    'legend.fontsize': 6, 
    'xtick.major.width': 0.25, 
    'ytick.major.width': 0.25,
    'text.color': 'Black',
    'axes.labelcolor':'Black',
    'xtick.color':'Black',
    'ytick.color':'Black',} )
sns.plotting_context()

def baseline_correlation(df, x, y):

    fig = plt.figure(figsize=(1.5*6, 1.5*6))
    plt_nr = 1
    rs = np.zeros(len(df['subject'].unique()))
    ps = np.zeros(len(df['subject'].unique()))
    for s, d in df.groupby(['subj_idx']):
        ax = fig.add_subplot(6,6,plt_nr)
        sns.regplot(x=x, y=y, scatter=True, fit_reg=True, data=d, ax=ax, scatter_kws={'alpha':0.1, 's':2, 'rasterized':True})
        # plt.xlim(df['pupil_0'].quantile(0.05), df['pupil_0'].quantile(0.95))
        # plt.ylim(df['pupil_resp_1s'].quantile(0.05), df['pupil_resp_1s'].quantile(0.95))
        r,p = sp.stats.pearsonr(d[x], d[y])
        rs[plt_nr-1] = r
        ps[plt_nr-1] = p
        plt.title('r = {}, p = {}'.format(round(np.mean(r),3), round(p, 3)))
        plt_nr += 1
    sns.despine(offset=3, trim=True)
    plt.tight_layout()

    fig2 = plt.figure(figsize=(2,2))
    ax = fig2.add_subplot(1,1,1) 
    sns.swarmplot(x=None, y=rs[ps<=0.05], color='g')
    sns.swarmplot(x=None, y=rs[ps>0.05])
    sns.pointplot(y=rs, ci=66)
    plt.title('r={}({}), p={}'.format(round(np.mean(rs),3), round(sp.stats.sem(rs),3), round(sp.stats.wilcoxon(rs, np.zeros(len(rs)))[1],3)))
    plt.xlabel('coefs')
    sns.despine(offset=3, trim=True)
    plt.tight_layout()

    # fig2 = plt.figure(figsize=(2,2))
    # ax = fig2.add_subplot(111)   
    # sns.regplot(x=x, y=y, units='subj_idx', scatter=True, fit_reg=False, x_bins=20, data=df, ax=ax)

    return fig, fig2

def plot_within_subjects_correlations(df, measure, bins=5):
    
    fig = plt.figure(figsize=(4,10))
    plt_nr = 1
    for i in range(5):

        if i == 0:
            d = df.reset_index()
        if i == 1:
            d = df.loc[(df['correct']==0) & (df['confidence']==1),:].reset_index()
        if i == 2:
            d = df.loc[(df['correct']==0) & (df['confidence']==0),:].reset_index()
        if i == 3:
            d = df.loc[(df['correct']==1) & (df['confidence']==0),:].reset_index()
        if i == 4:
            d = df.loc[(df['correct']==1) & (df['confidence']==1),:].reset_index()

        ax = fig.add_subplot(5,2,plt_nr)
        d['bins'] = d.groupby(['subject'])[measure].apply(pd.qcut, bins, labels=False)
        means = d.groupby(['subject', 'bins']).mean().groupby('bins').mean()
        sems = d.groupby(['subject', 'bins']).mean().groupby('bins').sem()
        ax.errorbar(x=means[measure], y=means['slow_wave'], xerr=sems[measure], yerr=sems['slow_wave'], fmt='none', elinewidth=0.5, markeredgewidth=0.5, ecolor='k', capsize=0, zorder=1)
        ax.plot(means[measure], means['slow_wave'], 'o', mfc='lightgrey', color='blue', zorder=2)
        cor = d.groupby(['subject'])[[measure,'slow_wave']].corr().reset_index().iloc[1::2]
        ax.set_title('r = {}, p = {}'.format(round(cor[measure].mean(),3), round(sp.stats.wilcoxon(cor[measure],np.zeros(cor.shape[0]))[1],3)))
        ax.set_ylabel('slow_wave')
        ax.set_xlabel('Pupil')
        ax.set_xlim(-8,12)
        ax.set_ylim(1,6.5)

        plt_nr += 1

        ax = fig.add_subplot(5,2,plt_nr)
        means = d.groupby('subj_idx').mean()
        sems = d.groupby('subj_idx').sem()
        ax.errorbar(x=means[measure], y=means['slow_wave'], xerr=sems[measure], yerr=sems['slow_wave'], fmt='none', elinewidth=0.5, markeredgewidth=0.5, ecolor='k', capsize=0, zorder=1)
        ax.plot(means[measure], means['slow_wave'], 'o', mfc='lightgrey', color='blue', zorder=2)
        r, p = sp.stats.pearsonr(means[measure], means['slow_wave'])
        if p < 0.05:
            
            (m,b) = sp.polyfit(means[measure], means['slow_wave'],1)
            x_line = np.linspace(ax.axis()[0], ax.axis()[1], 100)
            regression_line = sp.polyval([m,b],x_line)
            ax.plot(x_line, regression_line, color='k')

        ax.set_title('r = {}, p = {}'.format(round(r,3), round(p,3)))
        ax.set_ylabel('slow_wave')
        ax.set_xlabel('Pupil')
        ax.set_xlim(-2,6)
        ax.set_ylim(0,12)

        plt_nr += 1

    sns.despine(offset=3, trim=True)
    plt.tight_layout()
    return fig

def plot_across_subjects_correlations(df, x_measure, y_measure, nboot=5000):

    nr_subjs = len(df['subj_idx'].unique())

    # main and interaction effects:
    try:
        d = df.groupby(['subj_idx', 'correct', 'confidence']).mean()[[x_measure, y_measure]]
        print('succes')
    except:
        d = df.groupby(['subj_idx', 'correct', 'confidence']).mean()[[y_measure]]
    print(d)
    main = ((d.loc[(np.arange(nr_subjs),0,0)].reset_index() + d.loc[(np.arange(nr_subjs),0,1)].reset_index()) - 
                    (d.loc[(np.arange(nr_subjs),1,0)].reset_index() + d.loc[(np.arange(nr_subjs),1,1)].reset_index()))
    interaction = ((d.loc[(np.arange(nr_subjs),0,1)].reset_index() - d.loc[(np.arange(nr_subjs),1,1)].reset_index()) - 
                    (d.loc[(np.arange(nr_subjs),0,0)].reset_index() - d.loc[(np.arange(nr_subjs),1,0)].reset_index()))

    if x_measure == 'confidence':
        x_data = np.array(df.groupby(['subj_idx'])['confidence'].mean())
    elif x_measure == 'correct':
        x_data = np.array(df.groupby(['subj_idx'])['correct'].mean())
    elif x_measure == 'meta_d':
        x_data = np.array(df.groupby(['subj_idx']).apply(sdt))

    # bootstrap confidence intervals:
    main_sem = []
    interaction_sem = []
    for i in range(nboot):
        try:
            dd = df.groupby(['subj_idx']).apply(lambda x: x.sample(frac=1, axis=0, replace=True)).reset_index(drop=True).groupby(['subj_idx', 'correct', 'confidence']).mean()[[x_measure, y_measure]]
        except:
            dd = df.groupby(['subj_idx']).apply(lambda x: x.sample(frac=1, axis=0, replace=True)).reset_index(drop=True).groupby(['subj_idx', 'correct', 'confidence']).mean()[[y_measure]]

        m = ((dd.loc[(np.arange(nr_subjs),0,0)].reset_index() + dd.loc[(np.arange(nr_subjs),0,1)].reset_index()) - 
                        (dd.loc[(np.arange(nr_subjs),1,0)].reset_index() + dd.loc[(np.arange(nr_subjs),1,1)].reset_index()))
        m['subj_idx'] = np.arange(len(m))
        main_sem.append(m)
        
        m = ((dd.loc[(np.arange(nr_subjs),0,1)].reset_index() - dd.loc[(np.arange(nr_subjs),1,1)].reset_index()) - 
                        (dd.loc[(np.arange(nr_subjs),0,0)].reset_index() - dd.loc[(np.arange(nr_subjs),1,0)].reset_index()))
        m['subj_idx'] = np.arange(len(m))
        interaction_sem.append(m)
    
    main_sem = (pd.concat(main_sem).groupby(['subj_idx']).quantile(0.8) - pd.concat(main_sem).groupby(['subj_idx']).quantile(0.2)) / 2
    interaction_sem = (pd.concat(interaction_sem).groupby(['subj_idx']).quantile(0.8) - pd.concat(interaction_sem).groupby(['subj_idx']).quantile(0.2)) / 2

    fig = plt.figure(figsize=(4,2))
    plt_nr = 1
    for data, data_sem, title in zip([main, interaction], [main_sem, interaction_sem], ['main', 'interaction']):
        ax = fig.add_subplot(1,2,plt_nr)
        
        if 'pupil' in x_measure:
            corr = pg.corr(x=data[x_measure], y=data[y_measure])
            plt.errorbar(data[x_measure], data[y_measure], xerr=data_sem[x_measure], yerr=data_sem[y_measure], color='lightgrey', elinewidth=0.5, markeredgewidth=0.25, ecolor='k', markeredgecolor='k', fmt='o')
            sns.regplot(x=data[x_measure], y=data[y_measure], ax=ax, scatter=False)
            plt.axvline(0, lw=0.5, color='k')
        else:
            corr = pg.corr(x=x_data, y=data[y_measure])
            plt.errorbar(x_data, data[y_measure], yerr=data_sem[y_measure], color='lightgrey', elinewidth=0.5, markeredgewidth=0.25, ecolor='k', markeredgecolor='k', fmt='o')
            sns.regplot(x=x_data, y=data[y_measure], ax=ax, scatter=False)
            plt.xlabel(x_measure)
            plt.ylabel(y_measure)
        plt.axhline(0, lw=0.5, color='k')
        ax.set_title('r = {}, p = {}, BF10 = {}'.format(round(float(corr['r']),3), round(float(corr['p-val']),3), round(float(corr['BF10']),3)))
        plt_nr += 1
    sns.despine(offset=3, trim=True)
    plt.tight_layout()
    return fig
    
def erp_timecourses(df, epoch, cluster_correction=True):

    epoch['subj_idx'] = np.array(df['subj_idx'])
    epoch['correct'] = np.array(df['correct'])
    epoch['confidence'] = np.array(df['confidence'])
    epoch = epoch.set_index(['subj_idx', 'confidence', 'correct'])

    # xlim:
    epoch = epoch.loc[:,(epoch.columns>=-0.2)&(epoch.columns<=1)]

    # # downsample:
    # epoch = epoch.iloc[:,::10]

    return plot_timecourses(epoch=epoch, ylim=(-2,8), span=(0.5,0.8), cluster_correction=cluster_correction)

def pupil_timecourses(df, epoch, cluster_correction=True):
    
    epoch['subj_idx'] = np.array(df['subj_idx'])
    epoch['correct'] = np.array(df['correct'])
    epoch['confidence'] = np.array(df['confidence'])
    epoch = epoch.set_index(['subj_idx', 'confidence', 'correct'])

    # xlim:
    epoch = epoch.loc[:,(epoch.columns>=-0.5)&(epoch.columns<=2.5)]

    return plot_timecourses(epoch=epoch, ylim=(-2,4), span=(0.5,1.5), cluster_correction=cluster_correction)

def plot_timecourses(epoch, ylim=(-2,4), span=(0.5,1.5), cluster_correction=True):

    # plot:
    fig = plt.figure(figsize=(2,2))
    x = np.array(epoch.columns, dtype=float)
    means = np.array(epoch.groupby(['correct', 'confidence']).mean())
    sems = np.array(epoch.groupby(['correct', 'confidence']).sem())
    ax = fig.add_subplot(1,1,1)
    for i, color, ls in zip([0,1,2,3], [sns.color_palette("muted")[1], sns.color_palette("muted")[1], sns.color_palette("muted")[0], sns.color_palette("muted")[0]], ['--','-','--','-']):
        plt.fill_between(x, means[i,:]-sems[i,:], means[i,:]+sems[i,:], color=color, alpha=0.1)
        plt.plot(x, means[i,:], color=color, ls=ls)
    plt.axvline(0, lw=0.5, color='k')
    plt.axhline(0, lw=0.5, color='k')
    plt.axvspan(span[0], span[1], color='k', alpha=0.1)
    if span == (0.5,0.8):
        plt.axvspan(0.35, 0.45, color='k', alpha=0.1)

    if cluster_correction:

        from mne.stats import permutation_cluster_1samp_test
        nr_subjs = len(epoch.index.get_level_values('subj_idx').unique())
        d = epoch.groupby(['subj_idx', 'correct', 'confidence']).mean()
        main = ((d.loc[(np.arange(nr_subjs),0,0)].reset_index() + d.loc[(np.arange(nr_subjs),0,1)].reset_index()) - 
                        (d.loc[(np.arange(nr_subjs),1,0)].reset_index() + d.loc[(np.arange(nr_subjs),1,1)].reset_index())).drop(['subj_idx', 'correct', 'confidence'], axis=1)
        interaction = ((d.loc[(np.arange(nr_subjs),0,1)].reset_index() - d.loc[(np.arange(nr_subjs),1,1)].reset_index()) - 
                        (d.loc[(np.arange(nr_subjs),0,0)].reset_index() - d.loc[(np.arange(nr_subjs),1,0)].reset_index())).drop(['subj_idx', 'correct', 'confidence'], axis=1)
        
        for data, loc, color in zip([main,interaction], [-1.75, -1.5,], ['g', 'orange']):
        
            threshold = None
            T_obs, clusters, cluster_p_values, H0 = \
                permutation_cluster_1samp_test(np.array(data), n_permutations=5000,
                                        threshold=threshold, tail=0, n_jobs=32, out_type='mask')
            
            times = epoch.columns
            for i_c, c in enumerate(clusters):
                c = c[0]
                if cluster_p_values[i_c] <= 0.05:
                    ax.hlines(loc, times[c.start], times[c.stop - 1], color=color, alpha=1, linewidth=2.5)

    else:

        # add ANOVA:
        epoch2 = epoch.reset_index()
        anova_table = []
        for c in epoch.columns:
            print(c)
            anova_table.append(rm_anova(epoch2, c, ['correct', 'confidence']))
        anova_table = np.stack(anova_table)
        for i, loc, color in zip([0,1,2], [-1.75, -1.5, -1.25], ['g', 'b', 'orange']):
            pvals = anova_table[:,i,-1]
            sig_indices = np.array(pvals<0.05, dtype=int)
            sig_indices[0] = 0
            sig_indices[-1] = 0
            s_bar = zip(np.where(np.diff(sig_indices)==1)[0]+1, np.where(np.diff(sig_indices)==-1)[0])
            for sig in s_bar:
                ax.hlines(loc, x[int(sig[0])]-(np.diff(x)[0] / 2.0), x[int(sig[1])]+(np.diff(x)[0] / 2.0), color=color, alpha=1, linewidth=2.5)

    # plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel('Time from feedback (s)')
    plt.ylabel('ERP')
    sns.despine(offset=3, trim=True)
    plt.tight_layout()

    return fig

def mixed_linear_modeling(df, y):
    
    df['y'] = df[y]
    df['rt'] = np.log10(df['rt'])
    for s, d in df.groupby('subj_idx'):
        # df.loc[d.index, 'y'] = (df.loc[d.index, 'y'] - df.loc[d.index, 'y'].mean()) / df.loc[d.index, 'y'].std() 
        df.loc[d.index, 'rt'] = (df.loc[d.index, 'rt'] - df.loc[d.index, 'rt'].mean()) / df.loc[d.index, 'rt'].std()

    res1 = smf.mixedlm(formula="y ~ 1 + correct + rt + (correct*rt)", 
                        re_formula="1",
                        groups=df["subj_idx"],
                        data=df).fit(reml=False)
    res2 = smf.mixedlm(formula="y ~ 1 + correct + rt + (correct*rt)", 
                        re_formula="1 + correct",
                        groups=df["subj_idx"],
                        data=df).fit(reml=False)
    res3 = smf.mixedlm(formula="y ~ 1 + correct + rt + (correct*rt)", 
                        re_formula="1 + rt",
                        groups=df["subj_idx"],
                        data=df).fit(reml=False)
    res4 = smf.mixedlm(formula="y ~ 1 + correct + rt + (correct*rt)", 
                        re_formula="1 + (correct*rt)",
                        groups=df["subj_idx"],
                        data=df).fit(reml=False)
    
    print(res2.aic - res1.aic)
    print(res3.aic - res1.aic)
    print(res4.aic - res1.aic)

    re_formula = "1"
    if res2.aic - res1.aic < -10:
        print('random correct!')
        re_formula += " + correct" 
    if res3.aic - res1.aic < -10:
        print('random rt!')
        re_formula += " + rt" 
    if res4.aic - res1.aic < -10:
        print('random interaction!')
        re_formula += " + (correct*rt)" 

    res = smf.mixedlm(formula="{} ~ 1 + correct + rt + (correct*rt)".format(y), 
                    re_formula=re_formula,
                    groups=df["subj_idx"],
                    data=df).fit(reml=True)
    
    return res

def sdt(df):
    """
    Computes d' and criterion
    """
    
    import numpy as np
    import scipy as sp
    
    target = (np.array(df['correct'])==1)
    choice = (np.array(df['confidence'])==1)
    hit = target&choice
    fa = ~target&choice
    
    hit_rate = (np.sum(hit)) / (float(np.sum(target)))
    fa_rate = (np.sum(fa)) / (float(np.sum(~target)))
    
    if hit_rate > 0.999:
        hit_rate = 0.999
    elif hit_rate < 0.001:
        hit_rate = 0.001
    if fa_rate > 0.999:
        fa_rate = 0.999
    elif fa_rate < 0.001:
        fa_rate = 0.001
        
    hit_rate_z = stats.norm.isf(1-hit_rate)
    fa_rate_z = stats.norm.isf(1-fa_rate)
    
    d = hit_rate_z - fa_rate_z
    c = -(hit_rate_z + fa_rate_z) / 2.0
    
    return d

def load_eeg_data(directory):

    epochs = pd.read_csv(os.path.join(directory, 'epoch_e_feed.csv'), header=None).iloc[:,6:]
    epochs.columns = np.cumsum(np.repeat(1/512, 615))-0.2-(1/512)
    
    df_ = pd.read_csv(os.path.join(directory, 'frn.csv'))
    df = pd.read_csv(os.path.join(directory, 'JW_all.csv'))
    df['FRN'] = df_['slow_wave']
    
    # remove:
    ind = (df['masking'] != 3)
    epochs = epochs.loc[ind,:].reset_index(drop=True)
    df = df.loc[ind,:].reset_index(drop=True)
    
    # rename:
    df.rename(columns={ 
                        'masking': 'conscious_feedback',
                        'correctness': 'correct',
                        'visibility': 'visib_confidence',
                        'RT': 'rt'}, inplace=True)
    df['subject'] = df['subject'].astype(int)
    df['correct'] = (df['correct']==1).astype(int)
    df['confidence'] = (df['confidence']==1).astype(int)
    df['conscious_feedback'] = (df['conscious_feedback']==1).astype(int)
    df['visib_confidence'] = (df['visib_confidence']==1).astype(int)

    return df, epochs

def match_pupil_eeg(df_pupil, df_eeg):

    # match:
    indices = []
    for s in np.unique(df_eeg.subject):
        
        print(s)

        d = df_pupil.loc[df_pupil['subject']==s,:]
        d2 = df_eeg.loc[df_eeg['subject']==s,:]
        # print(d.shape)

        eeg_row = 0
        keep = []
        # for pupil_row in range(df.shape[0]):
        for pupil_row in range(d.shape[0]):
            # print(pupil_row)
            
            # if pupil_row == 1019:
            #     print(eeg_row)
            
            try:
                match_columns = ['subject', 'correct', 'confidence', 'conscious_feedback', 'visib_confidence']
                eeg = np.array(d2.iloc[eeg_row][match_columns], dtype=int)
                pupil = np.array(d.iloc[pupil_row][match_columns], dtype=int)
                rt_eeg = float(d2.iloc[eeg_row][['rt',]])
                rt_pupil = float(d.iloc[pupil_row][['rt',]]) * 1000
                if ((eeg == pupil).sum()==len(match_columns)) & (abs(rt_eeg-rt_pupil)<50):
                # if ((eeg == pupil).sum()==len(match_columns)):  
                # if ((eeg == pupil).sum()==4):
                    # print('match!')
                    eeg_row += 1
                    keep.append(pupil_row)
            except IndexError:
                continue
        
        inds = np.zeros(d.shape[0])
        inds[keep] = 1

        indices.append(inds)
    
    indices = np.concatenate(indices)

    return indices

def rm_anova(df, dv, within):
    aovrm = AnovaRM(df, dv, 'subj_idx', within=within, aggregate_func='mean')
    res = aovrm.fit()
    print(res)
    return np.array(res.anova_table)

def correlation_plot(X, Y, ax=False, dots=True, line=True, stat='pearson', rasterized=False, color='black', markersize=6, markeredgewidth=0.5, xlim=None, ylim=None):
    
    if not ax:
        fig = plt.figure(figsize=(3,3))
        ax = fig.add_subplot(111)
        despine = True
    else:
        despine = False
        
    slope, intercept, r_value, p_value, std_err = stats.linregress(X,Y)
    
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    
    if stat == 'spearmanr':
        r_value, p_value = stats.spearmanr(X,Y)
    
    (m,b) = sp.polyfit(X,Y,1)
    if dots:
        ax.plot(X, Y, 'o', color=color, marker='o', markersize=markersize, markeredgecolor='w', markeredgewidth=markeredgewidth, rasterized=rasterized) #s=20, zorder=2, linewidths=2)
    x_line = np.linspace(ax.axis()[0], ax.axis()[1], 100)
    regression_line = sp.polyval([m,b],x_line)
    if line:
        if p_value < 0.05:
            ax.plot(x_line, regression_line, color='black', zorder=3, lw=1.5)
        # ax.text(ax.axis()[0] + ((ax.axis()[1] - ax.axis()[0]) / 10.0), ax.axis()[3] - ((ax.axis()[3]-ax.axis()[2]) / 5.0), 'r = ' + str(round(r_value, 3)) + '\np = ' + str(round(p_value, 4)), size=6, color='k')
    ax.text(0.5, 0.9,'r={}, p={}'.format(round(r_value, 3), round(p_value, 3)), size=5, color='r', horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
    if despine:
        sns.despine(offset=3, trim=True)
        plt.tight_layout()

    return ax, r_value

def sequential_effects(df, ind, m, measure, bins=5, reverse=False):

    fig = plt.figure(figsize=(4,2))
    plt_nr = 1

    rs_ = []

    for reverse in [1,0]:

        ind_0 = np.concatenate((np.diff(df['trial'])==1, np.array([False]))) & np.array(ind)
        ind_1 = np.concatenate((np.array([False]), ind_0[:-1],))
        if ind_0[-2]:
            ind_1[-1] = True
        print(np.where(ind_0))
        print(np.where(ind_1))

        if reverse:
            dummy = ind_0.copy()
            ind_0 = ind_1.copy()
            ind_1 = dummy.copy()

        df['bins_0'] = np.NaN
        df['bins_1'] = np.NaN
        df.loc[ind_0, 'bins_0'] = np.array(df.loc[ind_0,:].groupby(['subject', 'session'])[m].apply(pd.qcut, bins, labels=False))
        df.loc[ind_1, 'bins_1'] = np.array(df.loc[ind_0, 'bins_0'])

        d_0 = df.loc[ind_0,:]
        means_0 = d_0.groupby(['subject', 'bins_0']).mean().groupby('bins_0').mean()
        sems_0 = d_0.groupby(['subject', 'bins_0']).mean().groupby('bins_0').sem()
        d_1 = df.loc[ind_1,:]
        means_1 = d_1.groupby(['subject', 'bins_1']).mean().groupby('bins_1').mean()
        sems_1 = d_1.groupby(['subject', 'bins_1']).mean().groupby('bins_1').sem()

        ax = fig.add_subplot(1,3,plt_nr)
        ax.errorbar(x=means_0[m], y=means_1[measure], xerr=sems_0[m], yerr=sems_1[measure], fmt='none', elinewidth=0.5, markeredgewidth=0.5, ecolor='k', capsize=0, zorder=1)
        ax.plot(means_0[m], means_1[measure], 'o', mfc='lightgrey', color='blue', zorder=2)
        rs = []
        for s in df['subject'].unique():
            d0 = d_0.groupby(['subject', 'bins_0']).mean().reset_index()
            d1 = d_1.groupby(['subject', 'bins_1']).mean().reset_index()
            r, p = sp.stats.pearsonr(d0.loc[d0['subject']==s, m], d1.loc[d1['subject']==s, measure])
            rs.append(r)

        p = sp.stats.wilcoxon(rs, np.zeros(len(rs)))[1]
        if p < 0.05:
            sns.regplot(means_0[m], means_1[measure], ax=ax)
        if reverse:
            ax.set_xlabel('{} (t=0)'.format(m))
            ax.set_ylabel('{} (t=-1)'.format(measure))
        else:
            ax.set_xlabel('{} (t=0)'.format(m))
            ax.set_ylabel('{} (t=1)'.format(measure))
        ax.set_title('r = {}, p = {}'.format(round(np.mean(rs),3), round(p,3)))
        rs_.append(np.array(rs))
        plt_nr += 1

    ax = fig.add_subplot(1,3,plt_nr)
    plt.bar(x=[0,1,2], height=[rs_[0].mean(), rs_[1].mean(), rs_[1].mean()-rs_[0].mean()],
            yerr=[sp.stats.sem(rs_[0]), sp.stats.sem(rs_[1]), sp.stats.sem(rs_[1]-rs_[0])])
    ps = [sp.stats.wilcoxon(rs_[0], np.zeros(len(rs)))[1],
            sp.stats.wilcoxon(rs_[1], np.zeros(len(rs)))[1],
            sp.stats.wilcoxon(rs_[1]-rs_[0], np.zeros(len(rs)))[1],
            ]
    print(ps)
    for x, r, p in zip([0.25,0.5,0.75], [rs_[0].mean(), rs_[1].mean(), rs_[1].mean()-rs_[0].mean()], ps):
        ax.text(x=x, y=0.95, s='r={}, p={}'.format(round(r,3), round(p,3)), transform=ax.transAxes, size=6, rotation=45)
    sns.despine(offset=3, trim=True)
    plt.tight_layout()

    return df, fig


# variables:
project_dir = '~/repos/2021_cereb_cortex_prediction_error/'
data_dir = os.path.join(project_dir, 'data')
fig_dir = os.path.join(project_dir, 'figs')

exp_names = [
    '2AFC',
    ]
rt_cutoffs = [
    (0, 5),
    ]

pupil_cutoffs = [
    [(0.5,1.5),],
    ]

nrs_bins = [
    5,
    ]

# for analyse_exp in [0,1,2,3,4,5]:
for analyse_exp in [0]:

    exp_name, rt_cutoff, pupil_cutoff, nr_bin = exp_names[analyse_exp], rt_cutoffs[analyse_exp], pupil_cutoffs[analyse_exp], nrs_bins[analyse_exp]

    # load data:
    df = pd.read_csv(os.path.join(data_dir, 'df_meta_{}_prep.csv'.format(exp_name)))
    epoch_p_stim = pd.read_hdf(os.path.join(data_dir, 'epoch_p_stim_{}_prep.hdf'.format(exp_name)))
    epoch_b_stim = pd.read_hdf(os.path.join(data_dir, 'epoch_b_stim_{}_prep.hdf'.format(exp_name)))
    epoch_p_feed = pd.read_hdf(os.path.join(data_dir, 'epoch_p_feed_{}_prep.hdf'.format(exp_name)))
    epoch_b_feed = pd.read_hdf(os.path.join(data_dir, 'epoch_b_feed_{}_prep.hdf'.format(exp_name)))
    epoch_eeg_feed = pd.read_hdf(os.path.join(data_dir, 'epoch_eeg_feed_{}_prep.hdf'.format(exp_name)))

    # timepoints:
    x_stim = np.array(epoch_p_stim.columns, dtype=float)
    # x_resp = np.array(epoch_p_resp.columns, dtype=float)
    x_feed = np.array(epoch_p_feed.columns, dtype=float)
    x_eeg_feed = np.array(epoch_eeg_feed.columns, dtype=float)

    # add blinks:
    df['blink_stim'] = (np.array(epoch_b_stim.loc[:,(x_stim>=0)&(x_stim<=0.5)].sum(axis=1)) >= 1).astype(bool)
    df['blink_feed'] = (np.array(epoch_b_feed.loc[:,(x_feed>=0.5)&(x_feed<=1.5)].sum(axis=1)) >= 1).astype(bool)
    df['blink'] = df['blink_stim'] | df['blink_feed']

    print()
    print(df.groupby(['subject'])['blink'].mean()*100)

    fig = plt.figure(figsize=(2,2))
    sns.swarmplot(y=df.groupby(['subject'])['blink'].mean()*100)
    plt.axhline(50, lw=1, color='r')
    plt.ylabel('Blink on % of trials')
    sns.despine(offset=3, trim=True)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'blinks.pdf'))

    # add pupil values:
    df['pupil_0'] = np.array(epoch_p_stim.loc[:,(x_stim>-0.5)&(x_stim<0)].mean(axis=1))
    # df['pupil_0'] = np.array(epoch_p_feed.loc[:,(x_feed>-0.25)&(x_feed<0)].mean(axis=1))
    df['pupil_1'] = np.array(epoch_p_feed.loc[:,(x_feed>pupil_cutoff[0][0])&(x_feed<pupil_cutoff[0][1])].mean(axis=1)) - np.array(epoch_p_feed.loc[:,(x_feed>-0.25)&(x_feed<0)].mean(axis=1))
    df['pupil_stim'] = np.array(epoch_p_stim.loc[:,(x_stim>0.5)&(x_stim<1.5)].mean(axis=1)) - np.array(epoch_p_stim.loc[:,(x_stim>-0.25)&(x_stim<0)].mean(axis=1))
    df['P3'] = np.array(epoch_eeg_feed.loc[:,(x_eeg_feed>=0.5)&(x_eeg_feed<=0.6)].mean(axis=1))

    # omissions step 1:
    omissions = (
        np.zeros(df.shape[0], dtype=bool)
        | np.array(df['rt']>5)
        | np.array(df['blink']>0)
        | np.array( (df['conscious_feedback']==1) & (df['visib_confidence']==0) )
        | np.array( (df['conscious_feedback']==0) & (df['visib_confidence']==1) ) 
        )
    df = df.loc[~omissions,:].reset_index(drop=True)
    epoch_p_stim = epoch_p_stim.loc[~omissions,:].reset_index(drop=True)
    epoch_p_feed = epoch_p_feed.loc[~omissions,:].reset_index(drop=True)
    epoch_b_stim = epoch_b_stim.loc[~omissions,:].reset_index(drop=True)
    epoch_b_feed = epoch_b_feed.loc[~omissions,:].reset_index(drop=True)
    epoch_eeg_feed = epoch_eeg_feed.loc[~omissions,:].reset_index(drop=True)

    print(df.loc[(df['conscious_feedback']==1)&(df['correct']==0)&(df['confidence']==1)].groupby(['subject'])['blink'].count())
    print(df.loc[(df['conscious_feedback']==0)&(df['correct']==0)&(df['confidence']==1)].groupby(['subject'])['blink'].count())

    # subj_idx:
    df['subj_idx'] = np.concatenate(np.array([np.repeat(i, sum(df['subject'] == df['subject'].unique()[i])) for i in range(len(df['subject'].unique()))]))

    # add RT bins:
    df['rt2'] = df['rt']*-1
    # df['rt_bin'] = df.groupby(['subj_idx'])['rt2'].apply(pd.qcut, 5, labels=False)
    df['rt_bin'] = df.groupby(['subj_idx', 'session', 'conscious_feedback'])['rt2'].apply(pd.qcut, 5, labels=False)
    df['rt_bin2'] = df.groupby(['subj_idx', 'session', 'conscious_feedback'])['rt2'].apply(pd.qcut, 10, labels=False)
    
    # within subjects correlations:
    for y in ['pupil_1', 'pupil_stim']:
        fig, fig2 = baseline_correlation(df, x='pupil_0', y=y)
        fig.savefig(os.path.join(fig_dir, 'baseline_correlations_within_{}.pdf'.format(y)))
        fig2.savefig(os.path.join(fig_dir, 'baseline_correlations_within_group_{}.pdf'.format(y)))
    for c, d in df.groupby('conscious_feedback'):
        for y in ['pupil_1', 'pupil_stim']:
            fig, fig2 = baseline_correlation(d, x='pupil_0', y=y)
            fig.savefig(os.path.join(fig_dir, 'baseline_correlations_within_{}_{}.pdf'.format(y,c)))
            fig2.savefig(os.path.join(fig_dir, 'baseline_correlations_within_group_{}_{}.pdf'.format(y,c)))

    # correct:
    df['pupil_1c'] = 0
    for (subj, ses, cons), d in df.groupby(['subj_idx', 'session', 'conscious_feedback']):
        Y = df.loc[d.index, 'pupil_1']
        X = df.loc[d.index, 'pupil_0']
        model = sm.OLS(Y,sm.add_constant(X))
        results = model.fit()
        print(results.rsquared)
        df.loc[d.index, 'pupil_1c'] = np.array(results.resid) + Y.mean()

    # baseline:
    baselines = np.atleast_2d(epoch_p_stim.loc[:,(x_stim>-0.5)&(x_stim<0)].mean(axis=1)).T
    epoch_p_stim = epoch_p_stim - baselines
    baselines_feed = np.atleast_2d(epoch_p_feed.loc[:,(x_feed>-0.5)&(x_feed<0)].mean(axis=1)).T
    epoch_p_feed = epoch_p_feed - baselines_feed

    # ANOVA's
    aovrm = AnovaRM(df, 'slow_wave', 'subj_idx', within=['correct', 'confidence', 'conscious_feedback'], aggregate_func='mean')
    print(aovrm.fit().summary())

    aovrm = AnovaRM(df, 'pupil_1', 'subj_idx', within=['correct', 'confidence', 'conscious_feedback'], aggregate_func='mean')
    print(aovrm.fit().summary())

    aovrm = AnovaRM(df, 'pupil_1c', 'subj_idx', within=['correct', 'confidence', 'conscious_feedback'], aggregate_func='mean')
    print(aovrm.fit().summary())

    aovrm = AnovaRM(df, 'slow_wave', 'subj_idx', within=['correct', 'rt_bin', 'conscious_feedback'], aggregate_func='mean')
    print(aovrm.fit().summary())

    aovrm = AnovaRM(df, 'pupil_1', 'subj_idx', within=['correct', 'rt_bin', 'conscious_feedback'], aggregate_func='mean')
    print(aovrm.fit().summary())

    aovrm = AnovaRM(df, 'pupil_1c', 'subj_idx', within=['correct', 'rt_bin', 'conscious_feedback'], aggregate_func='mean')
    print(aovrm.fit().summary())

    # aovrm = AnovaRM(d, 'response', 'subj_idx', within=['correct', 'confidence'], aggregate_func='mean')
    # print(aovrm.fit().summary())

    aovrm = AnovaRM(df, 'slow_wave', 'subj_idx', within=['correct', 'confidence', 'visib_confidence'], aggregate_func='mean')
    print(aovrm.fit().summary())

    aovrm = AnovaRM(df, 'pupil_1', 'subj_idx', within=['correct', 'confidence', 'visib_confidence'], aggregate_func='mean')
    print(aovrm.fit().summary())

    fig = plt.figure()
    plt.hist(df.rt, bins=100)
    fig.savefig(os.path.join(fig_dir, 'rt_hist.pdf'))

    # timecourses:
    for c in [0,1]:
        fig = erp_timecourses(df.loc[df['conscious_feedback']==c,:], epoch_eeg_feed.loc[df['conscious_feedback']==c,:], cluster_correction=False)
        fig.savefig(os.path.join(fig_dir, 'ERP_timecourses_{}.pdf'.format(c)))

        fig = pupil_timecourses(df.loc[df['conscious_feedback']==c,:], epoch_p_feed.loc[df['conscious_feedback']==c,:], cluster_correction=False)
        fig.savefig(os.path.join(fig_dir, 'pupil_timecourses_{}.pdf'.format(c)))

    # RT analyses:
    # ------------
    
    fig = plt.figure(figsize=(2,2))
    ax = fig.add_subplot(111)
    sns.pointplot(x='rt_bin2',  y='correct', units='subject', data=df, ci=66, color='green', alpha=0.5, linewidth=0.5, saturation=0.8, ax=ax)
    ax = ax.twinx()
    sns.pointplot(x='rt_bin2',  y='confidence', units='subject', data=df, ci=66, alpha=0.5, linewidth=0.5, saturation=0.8, ax=ax)
    # ax.set_ylabel(measure)
    # ax.set_xticklabels([0,1])
    ax.set_xlabel('RT (s)')
    # sns.despine(offset=3, trim=True)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'rt_bin.pdf'))

    # interaction plots:
    for measure, ylim in zip(['P3', 'FRN', 'slow_wave', 'pupil_1'], [(1,8), (0,4), (2,6), (1,2.75), (1,2.75)]):
        for c in [0,1]:
            fig = plt.figure(figsize=(2,2))
            ax = fig.add_subplot(111)
            sns.pointplot(x='rt_bin',  y=measure, hue='correct', units='subject', hue_order=[1,0], palette=sns.color_palette("muted"), data=df.loc[df['conscious_feedback']==c,:], ci=66, alpha=0.5, linewidth=0.5, saturation=0.8, ax=ax)
            plt.ylim(ylim)
            ax.set_ylabel(measure)
            
            # ax.set_xticklabels([0,1])
            ax.set_xlabel('RT (s)')
            ax.get_legend().remove()
            sns.despine(offset=3, trim=True)
            plt.tight_layout()

            # # ANOVA:
            # aovrm = AnovaRM(df.loc[df['conscious_feedback']==c,:], measure, 'subj_idx', within=['correct', 'rt_bin'], aggregate_func='mean')
            # res = aovrm.fit().anova_table
            # ax.text(x=0, y=0.9, s='F({},{}) = {}, p = {}'.format(int(res['Num DF']['correct']), int(res['Den DF']['correct']), round(res['F Value']['correct'],2), round(res['Pr > F']['correct'],3)),
            #         size=6, transform=ax.transAxes)
            # ax.text(x=0, y=0.8, s='F({},{}) = {}, p = {}'.format(int(res['Num DF']['rt_bin']), int(res['Den DF']['rt_bin']), round(res['F Value']['rt_bin'],2), round(res['Pr > F']['rt_bin'],3)),
            #         size=6, transform=ax.transAxes)
            # ax.text(x=0, y=0.7, s='F({},{}) = {}, p = {}'.format(int(res['Num DF']['correct:rt_bin']), int(res['Den DF']['correct:rt_bin']), round(res['F Value']['correct:rt_bin'],2), round(res['Pr > F']['correct:rt_bin'],3)),
            #         size=6, transform=ax.transAxes)

            # MIXED LM:
            res = mixed_linear_modeling(df.loc[df['conscious_feedback']==c], measure)
            ax.text(x=0, y=0.9, s='p = {}'.format(round(res.pvalues['correct'],3)),
                    size=6, transform=ax.transAxes)
            ax.text(x=0, y=0.8, s='p = {}'.format(round(res.pvalues['rt'],3)),
                    size=6, transform=ax.transAxes)
            ax.text(x=0, y=0.7, s='p = {}'.format(round(res.pvalues['correct:rt'],3)),
                    size=6, transform=ax.transAxes)

            # ints = []
            # for s, d in df.loc[df['conscious_feedback']==c].groupby(['subj_idx']):
            #     a = smf.ols(formula="{} ~ 1 + correct + rt + (correct*rt)".format(measure), data=d).fit()
            #     ints.append(a.params['correct:rt'])

            fig.savefig(os.path.join(fig_dir, 'rt_bin2_{}_{}.pdf'.format(measure, int(c))))

    # Confidence analyses:
    # --------------------
    d = df.groupby(['subject', 'confidence']).mean().reset_index()
    for y in ['rt', 'correct', 'response',]:
        fig = plt.figure(figsize=(1.5,2))
        ax = fig.add_subplot(111)
        sns.barplot(x='confidence',  y=y, units='subject', data=df, ci=None, alpha=1, palette=sns.color_palette("Blues", n_colors=2), ax=ax)
        sns.stripplot(x='confidence', y=y, data=d, jitter=False, size=2, edgecolor='black', linewidth=0.25, alpha=1, palette=['grey', 'grey'], ax=ax)
        values = np.vstack((d.loc[d['confidence'] == 0, y], d.loc[d['confidence'] == 1, y]))
        ax.plot(np.array([0, 1]), values, color='black', lw=0.5, alpha=0.5)
        # ax.set_title('p = {}'.format(round(myfuncs.permutationTest(values[0,:], values[1,:], paired=True)[1],3)))
        ax.set_title('p = {}'.format(round(sp.stats.ttest_rel(values[0,:], values[1,:])[1],3)))
        ax.set_ylabel(y)
        ax.set_xticklabels([0,1])
        ax.set_xlabel('confidence')
        sns.despine(offset=3, trim=True)
        plt.tight_layout()
        fig.savefig(os.path.join(fig_dir, 'confidence_{}.pdf'.format(y)))

    fig = plt.figure(figsize=(1.5,2))
    ax = fig.add_subplot(111)
    d = df.groupby(['subj_idx', 'correct', 'confidence']).count().reset_index()
    y = 'response'
    for subj in d['subj_idx'].unique():
        d.loc[d['subj_idx']==subj, y] = d.loc[d['subj_idx']==subj, y] / d.loc[d['subj_idx']==subj, y].sum() * 100
    sns.pointplot(x='confidence',  y=y, hue='correct', units='subject', data=d, palette=['red', 'green'], ci=66, alpha=0.5, linewidth=0.5, saturation=0.8, ax=ax)
    ax.set_ylabel(y)
    ax.set_xticklabels([0,1])
    ax.set_xlabel('confidence')
    sns.despine(offset=3, trim=True)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'confidence2_{}.pdf'.format('counts')))

    aovrm = AnovaRM(df, 'ones', 'subject', within=['correct', 'confidence'], aggregate_func='sum')
    res = aovrm.fit().anova_table

    # interaction plots:
    for measure, ylim in zip(['P3', 'FRN', 'slow_wave', 'pupil_1', 'pupil_1c'], [(1,8), (0,4), (2,6), (1,2.75), (1,2.75),]):
        for c in [True, False]:
            fig = plt.figure(figsize=(2,2))
            ax = fig.add_subplot(111)
            sns.pointplot(x='confidence',  y=measure, hue='correct', units='subject', hue_order=[1,0], palette=sns.color_palette("muted"), data=df.loc[df['conscious_feedback']==c,:], ci=66, alpha=0.5, linewidth=0.5, saturation=0.8, ax=ax)
            aovrm = AnovaRM(df.loc[df['conscious_feedback']==c,:], measure, 'subject', within=['correct', 'confidence'], aggregate_func='mean')
            res = aovrm.fit().anova_table
            ax.text(x=0, y=0.9, s='F({},{}) = {}, p = {}'.format(int(res['Num DF']['correct']), int(res['Den DF']['correct']), round(res['F Value']['correct'],2), round(res['Pr > F']['correct'],3)),
                    size=6, transform=ax.transAxes)
            ax.text(x=0, y=0.8, s='F({},{}) = {}, p = {}'.format(int(res['Num DF']['confidence']), int(res['Den DF']['confidence']), round(res['F Value']['confidence'],2), round(res['Pr > F']['confidence'],3)),
                    size=6, transform=ax.transAxes)
            ax.text(x=0, y=0.7, s='F({},{}) = {}, p = {}'.format(int(res['Num DF']['correct:confidence']), int(res['Den DF']['correct:confidence']), round(res['F Value']['correct:confidence'],2), round(res['Pr > F']['correct:confidence'],3)),
                    size=6, transform=ax.transAxes)
            ax.get_legend().remove()
            plt.ylim(ylim)
            sns.despine(offset=3, trim=True)
            plt.tight_layout()
            fig.savefig(os.path.join(fig_dir, 'scalars_{}_{}.pdf'.format(measure, int(c))))

    # within and across subjects correlations:
    for c, d in df.groupby('conscious_feedback'):
        fig, fig2 = baseline_correlation(d, x='pupil_1', y='slow_wave')
        fig.savefig(os.path.join(fig_dir, 'correlations_within_{}.pdf'.format(c)))
        fig2.savefig(os.path.join(fig_dir, 'correlations_within_group_{}.pdf'.format(c)))

        fig, fig2 = baseline_correlation(d, x='pupil_1c', y='slow_wave')
        fig.savefig(os.path.join(fig_dir, 'correlations_c_within_{}.pdf'.format(c)))
        fig2.savefig(os.path.join(fig_dir, 'correlations_c_within_group_{}.pdf'.format(c)))

        for x_measure, y_measure in zip(['pupil_1', 'confidence', 'correct', 'meta_d', 'confidence', 'correct', 'meta_d'], 
                                        ['slow_wave',    'slow_wave',       'slow_wave',    'slow_wave',   'pupil_1',    'pupil_1', 'pupil_1']):
            fig = plot_across_subjects_correlations(df=df.loc[df['conscious_feedback']==c,:], x_measure=x_measure, y_measure=y_measure, nboot=5000)
            fig.savefig(os.path.join(fig_dir, 'correlations_{}_{}_{}.pdf'.format(x_measure, y_measure, c)))

    # sequential effects:
    n_bins = 5
    n_bins_ddm = 5
    df['repeat'] = np.concatenate((np.array([False]), np.array(df.iloc[0:-1]['response']) == np.array(df.iloc[1:]['response']))).astype(int)
    for x in ['pupil_1', 'slow_wave']:
        for y in ['correct', 'rt', 'confidence', 'response', 'repeat']:
            for c in [0, 1]:

                ind = (df['conscious_feedback'] == c)
                df_, fig = sequential_effects(df, ind, x, y, bins=n_bins)
                fig.savefig(os.path.join(fig_dir, 'sequential_effects_{}_{}_{}.pdf'.format(c, x, y)))
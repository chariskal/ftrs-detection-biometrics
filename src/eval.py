import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import metrics
import os
import pandas as pd                 # package for data analysis with fast and flexible data structures
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler('./logs/evaluation.log')
handler.setLevel(logging.DEBUG)

f_format = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
handler.setFormatter(f_format)
logger.addHandler(handler)

class Eval_class:
    """
    Class Description: for evaluation of a biometrics system
    """

    def __init__(self, no_subjects, numbins = 300, fig_dir='./figs/'):

        self.max_val = 1                 # max threshold
        self.min_val = 0                 # min threshold
        self.nbins = numbins             # number of bins for getting distribution
        #self.bins = np.linspace(self.min_val, self.max_val, self.nbins)
        #self.bin_centers = 0.5*(self.bins[1:] + self.bins[:-1])
        self.fig_directory = fig_dir
        self.num_subjects = no_subjects
        self.rank_array = np.zeros(self.num_subjects)     # each index corresponds to the rank

    def plot_distribution(self, scores, labels):
        """
        Description: Plots 2 distributions of genuine and imposter scores and saves a figure as 'pdf.png'
        Args:
            scores (list nd.array float):   Similarity scores
            labels (list of strings):       Corresponding labels for each array i.e. left index, right index
        """
        plt.figure()
        bins2 = np.linspace(self.min_val, self.max_val, self.nbins)
        bin_centers = 0.5*(bins2[1:] + bins2[:-1])                  # for x axis plot
        for score,l in zip(scores,labels):
            hist1, _ = np.histogram(score, bins=bins2, density=True)
            pdf = hist1/hist1.sum()                                 # normalize to get probability distr
            plt.plot(bin_centers, pdf, label=l)
        plt.title('pdf')
        plt.xlabel('scores')
        plt.ylabel('probabilty')
        plt.legend()
        plt.savefig(self.fig_directory + 'pdf.png')
        logger.info("Figure 'pdf.png' saved! ")

    def plot_ROC(self, y_true, scores, labels):
        """
        Description: Calculates and plots ROC curve according to 'y_true' and probability scores 'prob_scores'. Saves plot as 'roc.png'

        Args:
            y_true (list of nd.array of int):   Actual truth labels for the scores to plot
            scores (list of nd.array of float): List of probability scores to plot
            labels (list of strings):           Corresponding labels for each array i.e. left index, right index
        """
        plt.figure()
        for true,score,l in zip(y_true,scores,labels):
            fpr, tpr, thresholds = metrics.roc_curve(true, score, pos_label=1)       # use built-in function of sklearn
            plt.plot(fpr,tpr,'o-',label=l)
        identity_line = np.linspace(0, 1,50)
        plt.plot(identity_line,identity_line, linestyle='dashed', color='black', alpha=0.8, label="No Skill")
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC')
        plt.legend()
        self.thres = thresholds
        plt.savefig(self.fig_directory + 'roc.png')
        logger.info("Figure 'roc.png' saved!")

    def calc_auc(self, y_true, scores):
        """
        Description: Calculates AUC on ROC curve according to 'y_true' and probability scores 'scores'.

        Args:
            y_true (list of nd.array of int):   Actual truth labels for the scores to plot
            scores (list of nd.array of float): List of probability scores to plot
            labels (list of strings):           Corresponding labels for each array i.e. left index, right index
        Returns:
            auc_list (list of float):   list of calculated Areas under Curve
        """
        auc_list = []
        for true,score in zip(y_true,scores):
            fpr, tpr, _ = metrics.roc_curve(true, score, pos_label=1)       # use built-in function of sklearn
            auc = metrics.auc(fpr, tpr)
            auc_list.append(auc)
        return auc_list

    def plot_errvth(self, y_true, scores, labels):
        """
        Description: Calculates and plots error rates v thresholds for all arrays in y_true and probability scores

        Args:
            y_true (list of nd.array of int):           Actual truth labels for the scores to plot
            scores (list of nd.array of float):         List of probability scores to plot
            labels (list of strings):                   Corresponding labels for each array i.e. left index, right index
        """
        plt.figure()
        for true,score,l in zip(y_true,scores,labels):
            fpr, tpr, thresholds = metrics.roc_curve(true, score, pos_label=1)       # use built-in function of sklearn
            frr = np.ones(len(tpr))-tpr
            # thr_axis = np.arange(0, 1, 1/len(tpr)).tolist()
            th = (np.sort(thresholds))
            th[np.argmax(th)]=1
            plt.plot(th,np.flip(fpr), label="FPR " + l)
            plt.plot(th,np.flip(frr), label="FRR " + l)

        plt.xlabel('threshold')
        plt.ylabel('error rate')
        plt.legend()
        # self.thres = thresholds
        plt.title('Error rates depending on threshold')
        plt.savefig(self.fig_directory + 'err_th.png')
        logger.info("Figure 'err_th.png' saved!")

    def plot_det(self, y_true, scores, labels):
        """
        Description: Calculates and plots DET curve for all arrays in y_true and probability scores

        Args:
            y_true (list of nd.array of int):           Actual truth labels for the scores to plot
            scores (list of nd.array of float):         List of probability scores to plot
            labels (list of strings):                   Corresponding labels for each array i.e. left index, right index
        """
        plt.figure()
        for true,score,l in zip(y_true,scores,labels):
            fpr, frr, _ = metrics.det_curve(true, score)
            plt.plot(fpr, frr, label=l)
        plt.xlabel('FPR')
        plt.ylabel('FRR')
        plt.title('DET curve')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.grid()
        plt.savefig(self.fig_directory + 'det.png')
        logger.info("Figure 'det.png' saved!")

    def plot_f1_acc(self, y_true, scores, labels):
        """
        Description: Calculates and plots F1-score and accuracy for all arrays in y_true and probability scores

        Args:
            y_true (list of nd.array of int):           Actual truth labels for the scores to plot
            scores (list of nd.array of float):         List of probability scores to plot
            labels (list of strings):                   Corresponding labels for each array i.e. left index, right index
        Returns:
            f1_th_max (list of floats):                 List containing at which threshold the f1 score is max for each array
            acc_th_max (list of floats):                List containing at which threshold the acc is max for each array
        """
        plt.figure()
        f1_th_max = []
        acc_th_max = []
        for true,score,l in zip(y_true,scores,labels):
            f1_list = []
            acc_list = []
            #_,_, th_roc = metrics.roc_curve(true, score)       # use built-in function of sklearn
            #new_list = np.flip(th_roc)
            xaxis = np.linspace(0,1,100)
            for thr in xaxis:           # calculate scores every few thresholds to reduce cost of calculations
                f1 = metrics.f1_score(true.astype(bool), score>thr, pos_label=1)
                acc = metrics.accuracy_score(true.astype(bool), score>thr, normalize=True, sample_weight=None)
                f1_list.append(f1)
                acc_list.append(acc)

            xaxis = np.linspace(0,1,len(f1_list))
            plt.plot(xaxis,f1_list, label='f1 '+l)
            plt.plot(xaxis,acc_list, label='acc '+l)
            f1_th_max.append(xaxis[np.argmax(f1_list)])               # find where f1 is maximum and get threshold there
            acc_th_max.append(xaxis[np.argmax(acc_list)])                   # find where accuracy is maximum and get the threshold value there
        plt.xlabel('threshold')
        plt.ylabel('score')
        plt.legend()
        plt.title('Accuracy and F1-score depending on threshold')
        plt.savefig(self.fig_directory + 'f1_acc.png')
        logger.info("Figure 'f1_acc.png' saved!")

        return f1_th_max, acc_th_max

    def plot_eer(self, y_true, scores, labels):
        """
        Description: Calculates and plots Equal Error rate on FPR v FRR curve for all arrays in y_true and probability scores

        Args:
            y_true (list of nd.array of int):           Actual truth labels for the scores to plot
            scores (list of nd.array of float):         List of probability scores to plot
            labels (list of strings):                   Corresponding labels for each array i.e. left index, right index
        Returns:
            eer_pos_list (list of int):                 List with the index with EER for each array
        """
        plt.figure()
        eer_pos_list = []            # list to store positions of eer
        for true,score,l in zip(y_true,scores,labels):
            fpr, tpr, th_roc = metrics.roc_curve(true, score)                # use built-in function of sklearn
            xaxis = th_roc[::-1]
            fpr = fpr[::-1]
            tpr = tpr[::-1]
            frr = np.ones(len(tpr))-tpr
            plt.plot(fpr, frr,'o-', label=l)
            eer_pos = np.nanargmin(np.absolute((frr - fpr)))            # find index where |FRR - FPR| is min
            eer_pos_list.append(eer_pos)
            logger.info(f'EER threshold pos for {l} is {xaxis[eer_pos]}')
            plt.plot(fpr[eer_pos],frr[eer_pos],'o', label='EER '+l)             # since it is on the y=x line, use index for both x and y

        plt.xlabel('FPR')
        plt.ylabel('FRR')
        plt.title('FPR vs FRR')

        identity_line = np.linspace(0, 1,len(tpr))
        plt.plot(identity_line,identity_line, linestyle='dashed', color='black', alpha=0.8, label="No Skill")
        plt.ylim(-0.02, 1.02)
        plt.xlim(-0.02, 1.02)
        plt.legend()

        plt.savefig(self.fig_directory + 'eer.png')
        logger.info("Figure 'eer.png' saved!")
        return eer_pos_list

    def plot_prc(self, y_true, scores, labels):
        """
        Description: Calculates and plots Precision-Recall curve for all arrays in y_true and probability scores

        Args:
            y_true (list of nd.array of int):           Actual truth labels for the scores to plot
            scores (list of nd.array of float):         List of probability scores to plot
            labels (list of strings):                   Corresponding labels for each array i.e. left index, right index
        Returns:
            auprc_list (list of floats):                List with the AUC for the prec-recall curve for each array
            ap_list (list of floats):                   List with the Average Precision score for each array
        """

        plt.figure()
        auprc_list = []         # list to store the Area under Prec-Rec curve
        ap_list = []           # list to store average precision score
        for true,score,l in zip(y_true,scores,labels):
            precision, rec, _ = metrics.precision_recall_curve(true, score)
            auprc = np.abs(np.trapz(rec,precision, dx=1.0, axis=0))          # use np.trapz to calculate the area
            aps = metrics.average_precision_score(true, score)
            plt.plot(rec, precision, marker='.', label=l)
            auprc_list.append(auprc)
            ap_list.append(aps)

        no_skill = len(true[true==1]) / len(true)         # random classifier
        plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.savefig(self.fig_directory + 'prec_rec.png')
        logger.info("Figure 'prec_rec.png' saved!")

        return auprc_list, ap_list

    def plot_cmc(self, y_true, scores, labels):
        """
        Description: Calculates and plots CMC for all arrays in y_true and probability scores

        Args:
            y_true (list of nd.array of int):           Actual truth labels for the scores to plot
            scores (list of nd.array of float):         List of probability scores to plot
            labels (list of strings):                   Corresponding labels for each array i.e. left index, right index
        Returns:
            rank1_list (list of float):                 List with the rank-1 rate for each array
        """
        plt.figure()
        rank1_list  = []
        for true,score,l in zip(y_true,scores,labels):
            self.rank_array = np.zeros(self.num_subjects)     # each index corresponds to the rank
            rank_arr_tmp = self.rank_array
            # Calculate for left index first
            y_new = true.reshape(self.num_subjects,int(true.shape[0]/self.num_subjects))
            y_new = y_new.transpose()               # change format
            # print(y_new, true)
            scores_new = score.reshape(self.num_subjects,int(score.shape[0]/self.num_subjects))       # get li scores
            scores_new = scores_new.transpose()     # do the same

            for j in range(0,y_new.shape[0]):
                tmp1 = scores_new[j][:]                         # get first row
                tmp = np.unique(tmp1)                           # get unique values
                true_sim = tmp1[y_new[j][:].astype(bool)][0]    # find value of genuine similarity
                sorted_array = np.sort(tmp)                     # sort array
                sorted_desc = sorted_array[::-1]                # reverse for descending order
                r = np.where(sorted_desc==true_sim)[0][0]       # calculate rank as to where the true sim value exists
                # print('rank', r)
                rank_arr_tmp[r] = rank_arr_tmp[r]+1         # keep track of the rank in sum array

            R_t = np.cumsum(rank_arr_tmp)           # Rank_t ID rate, also known as TPIR
            R_t = R_t/R_t.max()*100.0               # make it percentage
            plt.plot(R_t[0:80], label=l)            # plot it
            rank_1 = rank_arr_tmp[0]/rank_arr_tmp.sum()
            rank1_list.append(rank_1)
        plt.xlabel('Rank t')
        plt.ylabel('Recognition rate (%)')
        plt.title('Cumulative Match Characteristic (CMC) curve')
        plt.legend()
        plt.savefig(self.fig_directory + 'cmc.png')
        logger.info("Figure 'cmc.png' saved!")

        return rank1_list

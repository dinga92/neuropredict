#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from sklearn import datasets
from sklearn.model_selection import train_test_split
from scipy.stats import mannwhitneyu
import random
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score, brier_score_loss
import numpy as np

from statsmodels.stats.proportion import binom_test, proportion_confint

from scipy.stats import norm



def calculate_accuracy_statistics(y_true, y_pred_class, 
                                  confint_alpha = 0.05, confint_method='normal'):
    """
    calculate accuracy (proportion of samples predicted correctly) and its
    respective p-values and confidence intervals using binomial proportions    
    """    
    num_predictions = len(y_true)         
    acc = accuracy_score(y_true, y_pred_class)
    number_of_successes = num_predictions*acc    
    p_value = binom_test(number_of_successes, 
                          num_predictions, 
                          0.5, 
                          alternative='larger')
    conf_int = proportion_confint(number_of_successes,
                                  num_predictions,
                                  alpha=confint_alpha,
                                  method=confint_method)    
    return([acc, p_value, conf_int[0], conf_int[1]])
      

def calculate_auroc_statistics(y_true, y_pred, confint_alpha=0.05):
    """ 
    calculate AUROC and it's p-values and CI 
    """
    #TODO: small sample test
    #TODO: check when it crashes
    #TODO: confidence intervals
    
    predictions_group0 = y_pred[y_true==0, 1]
    predictions_group1 = y_pred[y_true==1, 1]
    try:
        pval_auc = mannwhitneyu(predictions_group0, 
                                predictions_group1, 
                                alternative='less')[1]
    except:
        pval_auc = 1        
    auroc = roc_auc_score(y_true, y_pred[:,1])
    
    auroc_ci = calculate_auroc_confint(auroc, len(predictions_group0),
                                       len(predictions_group1), confint_alpha)
    
    return([auroc, pval_auc, auroc_ci[0], auroc_ci[1]])
    

def calculate_auroc_se(auroc, n1, n2):
    """
    calculate SE for AUROC used to compute confidence intervals according to
    Hanley & McNeil 1982 RADIOLOGY and https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/PASS/Confidence_Intervals_for_the_Area_Under_an_ROC_Curve.pdf
    """
    # TODO: check other ways to get these like Delong, bootstrap as in pROC or Hamadicharef
    
    q1 = auroc / (2-auroc)
    q2 = (2*(auroc**2)) / (1+auroc)
    se_auroc = np.sqrt((auroc*(1-auroc) + (n1-1)*(q1-auroc**2) +
                       (n2-1)*(q2-auroc**2)) / (n1*n2))
    return(se_auroc)

    
def calculate_auroc_confint(auroc, n1, n2, alpha=0.05):
    auroc_se = calculate_auroc_se(auroc, n1, n2)
    return(norm.interval(1-alpha, auroc, auroc_se))
    
    
def calculate_logscore_statistics(y_true, y_pred, n_permutations=199):
    """ 
    calculate logscore and it's p-values using permutation test
    """
    # TODO: check which column is prediction and if bigger==better
    # TODO: confidence intervals
    logscore = log_loss(y_test, y_pred)
    p_value = permutation_test(y_true,
                               y_pred,
                               log_loss, 
                               n_permutations, 
                               'bigger')
    return([logscore, p_value, 0, 0])

    
def calculate_brierscore_statistics(y_true, y_pred, n_permutations=199):
    """ 
    calculate logscore and it's p-values using permutation test
    """
    # TODO: check what is better, greater/smaller and which column of predictions to use
    # TODO: confidence intervals
    
    brierscore = brier_score_loss(y_test, y_pred)
    p_value = permutation_test(y_true,
                               y_pred,
                               brier_score_loss, 
                               n_permutations, 
                               'smaller')
    return([brierscore, p_value, 0, 0])
    
    
def permutation_test(y_true, y_pred, eval_function, n_permutations, better='smaller'):    
    """ 
    calculate p-values for any performance measure using permutation test
    """
    # TODO: accelerated perm-test
    # TODO: approximate tail p-values    
    null_results = []
    for i in range(n_permutations):
        y_shuffled = y_true.copy()
        random.shuffle(y_shuffled)
        null_results.append(eval_function(y_shuffled, y_pred))        
    real_result = eval_function(y_true, y_pred)
    
    if better == 'smaller':
        real_result_rank = sum(real_result < null_results) + 1
    elif better == 'bigger':
        real_result_rank = sum(real_result > null_results) + 1
    else:
        assert()        
    pval = real_result_rank / (len(null_results) + 1)
    return(pval)

    
def validate_out_of_sample_predictions(y_true, y_pred_proba):
    """
    Evaluate performance of out of sample predictions. Calculate multiple
    performance measures and their CI and p-values. CI and p-values are valid 
    only for for out of sample predictions, they are not valid for within sample
    or predictions or predictions obtained by cross-validation. 
    """
    predicted_class_labels = y_pred_proba[:,1] > 0.5
    accuracy_statistics = (calculate_accuracy_statistics(y_test, predicted_class_labels))
    auroc_statistics = (calculate_auroc_statistics(y_test, y_pred_proba))
    logscore_statistics = (calculate_logscore_statistics(y_test, y_pred_proba))
    brierscore_statistics = (calculate_brierscore_statistics(y_test, y_pred_proba[:,0]))
    return([accuracy_statistics, auroc_statistics, logscore_statistics, brierscore_statistics])

    

dataset_size = 50
X, y = datasets.make_classification(n_samples=dataset_size, 
                                    n_features=5,
                                    n_informative=2,
                                    flip_y=0,
                                    class_sep=0.5)
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.5,
                                                    stratify=y)
fit = LogisticRegression(C=1, penalty='l1').fit(X_train, y_train)
predicted_probabilities = fit.predict_proba(X_test)

results = validate_out_of_sample_predictions(y_test, predicted_probabilities)
print(np.array(results))


#    [[0.84  0.    0.696 0.984]
#     [0.865 0.001 0.715 1.016]
#     [0.53  0.005 0.    0.   ]
#     [0.383 0.005 0.    0.   ]] 

    [[0.76  0.007 0.593 0.927]
     [0.821 0.004 0.649 0.992]
     [0.532 0.01  0.    0.   ]
     [0.559 0.005 0.    0.   ]]

    
    
    

#pvals_results = []
#
#for i in range(1):
#    dataset_size = 100
#    X, y = datasets.make_classification(n_samples=dataset_size, 
#                                        n_features=5,
#                                        n_informative=2,
#                                        flip_y=0,
#                                        class_sep=0.2)
#    
#    X_train, X_test, y_train, y_test = train_test_split(X, 
#                                                        y, 
#                                                        test_size=0.5,
#                                                        stratify=y); sum(y_test)
#    
#    fit = LogisticRegression(C=1, penalty='l1').fit(X_train, y_train)
#    predicted_probabilities = fit.predict_proba(X_test)
#    
#    results_measures, results_measures_pvals = validate_out_of_sample_predictions(y_test, predicted_probabilities)
#    pvals_results.append(results_measures_pvals)
#    print(results_measures)
                

#for measure, pval in zip(results_measures, results_measures_pvals):
#    print(round(measure, 2), round(pval, 3))

#signifficant_results = np.array(pvals_results) < 0.05
#print(np.sum(signifficant_results, 0))
#   
#
#sum(datasets.make_classification(n_samples=dataset_size, 
#                                        n_features=5,
#                                        n_informative=2,
#                                        flip_y=0,
#                                        class_sep=0.2)[1])

# run everything for different n
# plot power vs n





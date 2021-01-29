import os
import os.path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.metrics import roc_auc_score
import torch

from penalties import compute_rpl_loss



def abridged_auc_softmax(min_hbr, hbr, f1):
    abridged_hbr_beg = 0
    for i in range(0, len(hbr)):
        if hbr[i] >= min_hbr:
            abridged_hbr_beg = i
            break
    
    abridged_hbr_f1_auc = sklearn.metrics.auc(hbr[abridged_hbr_beg:], f1[abridged_hbr_beg:])
    print("Abridged w/uncert AUC with min_hbr=" + str(min_hbr) + ": " + str(abridged_hbr_f1_auc/(1-min_hbr)))
    
    axes = plt.gca()
    axes.set_xlim([min_hbr,1.0])
    axes.set_ylim([0.0,0.8])
    plt.plot(hbr, f1)
    plt.show()
    
    return hbr[abridged_hbr_beg:], f1[abridged_hbr_beg:]


def calc_auroc(seen_conf_dict, unseen_conf_dict, key):
    """ Computes standard AUROC given predictions. 
    
    Args:
    seen_conf_dict: [dict] the max probability assigned to any class for each datapoint, across seen inputs
    unseen_conf_dict: [dict] the max probability assigned to any class for eacah datapoint, across novel inputs
    key: [str] whether to retrieve the unnormalized 'dist' value or the normalized 'prob' value
    """
    # Interpret scores as probability for open cls
    raw_seenset_openscores = []
    for leaf_name in seen_conf_dict.keys():
        raw_seenset_openscores = raw_seenset_openscores + [record[key] for record in seen_conf_dict[leaf_name]]
    raw_seenset_openscores = np.array(raw_seenset_openscores)
    seenset_openscores = -raw_seenset_openscores
    seenset_true = np.zeros((len(seenset_openscores),))
    raw_unseenset_openscores = []
    for leaf_name in unseen_conf_dict.keys():
        raw_unseenset_openscores = raw_unseenset_openscores + [record[key] for record in unseen_conf_dict[leaf_name]]
    raw_unseenset_openscores = np.array(raw_unseenset_openscores)
    unseenset_openscores = -raw_unseenset_openscores
    unseenset_true = np.ones((len(unseenset_openscores),))
    all_openscores = np.concatenate((seenset_openscores, unseenset_openscores), axis=None)
    all_true = np.concatenate((seenset_true, unseenset_true), axis=None)
    auc_score = roc_auc_score(all_true, all_openscores)
    return auc_score


def collect_rpl_max(model, dataset_type, loader, folder_to_name, gamma, cifar=False, idx_to_label=None):
    """ Care about 1) identity of max known class 2) distance to reciprocal points of this class. """
    with torch.no_grad():
        confidence_dict = defaultdict(list)
        for i, data in enumerate(loader, 0):   
            # get the inputs & combine positive and negatives together
            img = data['image']
            img = img.cuda()
            if cifar:
                label_idx = data['label']
            else:
                folder_names = data['folder_name']

            outputs = model.forward(img)   

            logits, dist_to_rp = compute_rpl_logits(model, outputs, gamma)
            max_distances, max_indices = torch.max(logits, 1)
            probs = torch.softmax(logits, dim=1)
            max_probs, max_indices = torch.max(probs, 1)
            
            for j in range(0, img.shape[0]):
                if cifar:
                    correct_leaf = idx_to_label[label_idx[j].item()]
                else:
                    correct_leaf = folder_to_name['n' + folder_names[j]]
                predicted_leaf_idx = max_indices[j].item()
                dist = max_distances[j].item()
                prob = max_probs[j].item()
                confidence_dict[correct_leaf].append({'idx': predicted_leaf_idx, 'dist': dist, 'prob': prob})

        return confidence_dict


def evaluate_val(model, criterion, val_loader, gamma, lamb, divide, logger):
    with torch.no_grad():
        running_loss = 0.0
        normal_correct = 0.
        used_correct = 0.
        normal_total = 0.0
        used_total = 0.0 
        normal_running_loss = 0.
        used_running_loss = 0.
        val_rpl_loss = 0.
        
        logger.info("beginning validation")

        for i, data in enumerate(val_loader, 0):
            
            # get the inputs & combine positive and negatives together
            img = data['image']
            img = img.cuda()
            
            labels = data['label']
            labels = labels.cuda()

            outputs = model.forward(img)
   
            # Compute RPL loss
            loss, open_loss, closed_loss, logits = compute_rpl_loss(model, outputs, labels, criterion, lamb, gamma, divide == 'TRUE')
            val_rpl_loss += loss.item()

            probs = torch.softmax(logits, dim=1)
            max_probs, max_indices = torch.max(probs, 1)
            
            used_correct += torch.sum(max_indices == labels).item()
            used_total += probs.shape[0]
            used_running_loss += loss.item()

        used_val_acc = used_correct/(used_total)
        logger.info("Used Validation Accuracy is : " + str(used_val_acc))
        logger.info("Used Average validation loss is: " + str(used_running_loss/used_total))
        logger.info("finished validation")
    
        return used_running_loss, used_val_acc
    
    
def seenval_baseline_thresh(confidence_dict, thresh, cifar=False, label_to_idx=None, folder_to_idx=None, name_to_folder=None, save_path=None, value_key='prob'):
    """ Given a softmax classifier, evaluate performance on seen validation set using thresholds on softmax probabilities. """
    softmax_correct = [0.0 for i in range(0, len(thresh))]
    per_cls_softmax_correct = [{leaf_name: 0.0 for leaf_name in confidence_dict.keys()} for i in range(0, len(thresh))]
    mistake_log = [{leaf_name: {} for leaf_name in confidence_dict.keys()} for i in range(0, len(thresh))]
    semantic_precision = np.zeros((len(thresh), ))
    semantic_tot = np.zeros((len(thresh), ))
    num_correctly_predict_seen = np.zeros((len(thresh), ))
    num_predict_seen = np.zeros((len(thresh), ))
    total = 0.0
    # fix class
    for leaf_name in confidence_dict.keys():
        
        max_indices = torch.tensor([record['idx'] for record in confidence_dict[leaf_name]]).long()
        max_probs = torch.tensor([record[value_key] for record in confidence_dict[leaf_name]])
            
        if cifar:
            labels = torch.full(max_indices.size(), label_to_idx[leaf_name], dtype=torch.long)
        else:
            labels = torch.full(max_indices.size(), folder_to_idx[name_to_folder[leaf_name]], dtype=torch.long)
        
        total += max_indices.shape[0]
        
        for i in range(0, len(thresh)):
            proceed = max_probs.cpu() > torch.full(max_probs.size(), thresh[i])
            num_predict_seen[i] += torch.sum(proceed).item()
            correct_cls = max_indices.cpu() == labels
            correct_vec = correct_cls & proceed
            num_corr = torch.sum(correct_vec).item()
            softmax_correct[i] += num_corr
            per_cls_softmax_correct[i][leaf_name] = num_corr/len(confidence_dict[leaf_name])
    
    total = float(sum([len(confidence_dict[leaf_name]) for leaf_name in confidence_dict.keys()]))
    success_rate = [softmax_correct[i]/total for i in range(0, len(softmax_correct))]
    
    info = {'success_rate': success_rate, 'per_cls_softmax_correct': per_cls_softmax_correct, 'mistake_log': mistake_log, 
            'semantic_precision': semantic_precision, 'num_correctly_predict_seen': softmax_correct, 'num_predict_seen': num_predict_seen, 'total': total}
    
    if save_path is not None:
        
        if os.path.exists(save_path):
            raise ValueError(save_path + ' already exists.')
        
        with open(save_path, 'wb') as f:
            pickle.dump(info, f)
 
    return info


def summarize(seenval_report, unseenval_report, thresh, verbose=True):
    """ Produce miscellaneous metrics based on predictions. """
    seen_precision = []
    hbr = []
    whole_prec = []
    recall = []
    f1 = []
    f1_hbr_avg = []
    mod_f1 = []
    harm_mod_f1 = []
    unique_hbr = []
    unique_f1 = []
    unique_seen_recall = []
    prev_hbr = -100.0
    best_f1 = -10.0
    best_seen_recall = -10.0
    fpr = []
    unique_fpr = []
    fpr_assoc_hbr = []
    prev_fpr = -100.0
    best_fpr_assoc_hbr = -10.0
    for i in range(0, len(thresh)):

        curr_seen_prec = seenval_report['num_correctly_predict_seen'][i]/seenval_report['num_predict_seen'][i]
        seen_precision.append(curr_seen_prec)
        
        curr_hbr = (unseenval_report['total'] - unseenval_report['num_predict_seen'][i])/unseenval_report['total']
        hbr.append(curr_hbr)
        
        curr_whole_prec = (curr_seen_prec + curr_hbr)/2.0
        curr_harm_whole_prec = (2.0 * ((curr_seen_prec * curr_hbr)/(curr_seen_prec + curr_hbr)))
        whole_prec.append(curr_whole_prec)

        curr_recall = seenval_report['num_correctly_predict_seen'][i]/seenval_report['total']
        recall.append(curr_recall)
        curr_f1 = 2.0 * ((curr_seen_prec * curr_recall)/(curr_seen_prec + curr_recall))
        f1.append(curr_f1)
        
        # if we are on a streak of > 1 of the same value, then update best associated f1 and maintain streak to next hbr
        if curr_hbr == prev_hbr:
            if curr_f1 > best_f1:
                best_f1 = curr_f1
            if curr_recall > best_seen_recall:
                best_seen_recall = curr_recall
        # otherwise streak is broken with prev_hbr, then record prev_hbr/best_associated_f1 and reset trackers for new hbr
        else:
            # edge case where covering first possible hbr, then don't add anything
            if prev_hbr >= 0:
                unique_hbr.append(prev_hbr)
                unique_f1.append(best_f1)
                unique_seen_recall.append(best_seen_recall)
            prev_hbr = curr_hbr
            best_f1 = curr_f1
            best_seen_recall = curr_recall
            
        # edge case: last threshold, must close down. if streak, end with best value. else no streak, put in current value.
        if i == len(thresh) - 1:
            unique_hbr.append(prev_hbr)
            unique_f1.append(best_f1)
            unique_seen_recall.append(best_seen_recall)
        
        f1_hbr_avg.append((curr_f1 + curr_hbr)/2.0)
        mod_f1.append(2.0 * ((curr_whole_prec * curr_recall)/(curr_whole_prec + curr_recall)))
        harm_mod_f1.append(2.0 * ((curr_harm_whole_prec * curr_recall)/(curr_harm_whole_prec + curr_recall)))
        
        # calculate fpr
        curr_fpr = (seenval_report['total'] - seenval_report['num_predict_seen'][i])/seenval_report['total']
        fpr.append(curr_fpr)
            
        if curr_fpr == prev_fpr:
            if curr_hbr > best_fpr_assoc_hbr:
                best_fpr_assoc_hbr = curr_hbr
        else:
            if prev_fpr >= 0:
                unique_fpr.append(prev_fpr)
                fpr_assoc_hbr.append(best_fpr_assoc_hbr)
            prev_fpr = curr_fpr
            best_fpr_assoc_hbr = curr_hbr
            
        if i == len(thresh) - 1:
            unique_fpr.append(prev_fpr)
            fpr_assoc_hbr.append(best_fpr_assoc_hbr)
    
    
    # clean nan values so auc calc works
    clean_unique_hbr = []
    clean_unique_f1 = []
    for i in range(0, len(unique_hbr)):
        if not np.isnan(unique_f1[i]):
            clean_unique_hbr.append(unique_hbr[i])
            clean_unique_f1.append(unique_f1[i])
    
    hbr_recall_auc = sklearn.metrics.auc(unique_hbr, unique_seen_recall)
    
    lit_auc = sklearn.metrics.auc(unique_fpr, fpr_assoc_hbr)
    
    if verbose:
        print("AUC of Open-Recall vs. Closed-Recall curve: " + str(hbr_recall_auc))
        print("AUROC from literature: " + str(lit_auc))
        print("U-OSR/U-CSR: ")
        for i in range(0, len(unique_hbr)):
            print(str(unique_hbr[i]) + '/' + str(unique_seen_recall[i]))
        
#     print("\nFull Details --- SeenP/R/ModF1/ModHarmF1/FPR/HBR/F1/F1_HBR_AVG: ")
#     for i in range(0, len(thresh)):
#         print("At thresh " + str(thresh[i]) + ': ' + str(round(seen_precision[i], 3)) + '/' + str(round(recall[i], 3)) + '/' + str(round(mod_f1[i], 3)) + '/' + str(round(harm_mod_f1[i], 3)) + '/' + str(round(fpr[i], 3)) + '/' +  str(round(hbr[i], 3)) + '/' + str(round(f1[i], 3)) + '/' + str(round(f1_hbr_avg[i], 3)))

    return {'lit_AUROC': lit_auc, 'OSR_CSR_AUC': hbr_recall_auc, 'cleaned_open_recall': clean_unique_hbr, 'cleaned_f1': clean_unique_f1, 'unique_open_recall': unique_hbr, 'unique_closed_recall': unique_seen_recall, 'unique_fpr': unique_fpr, 'fpr_assoc_hbr': fpr_assoc_hbr}


def unseenval_baseline_thresh(confidence_dict, thresh, save_path=None, value_key='prob'):
    """ Given a softmax classifier, evaluate performance on unseen validation set using thresholds on softmax probabilities. """
    softmax_correct = [0.0 for i in range(0, len(thresh))]
    per_cls_softmax_correct = [{leaf_name: 0.0 for leaf_name in confidence_dict.keys()} for i in range(0, len(thresh))]
    mistake_log = [{leaf_name: {} for leaf_name in confidence_dict.keys()} for i in range(0, len(thresh))]
    semantic_precision = np.zeros((len(thresh), ))
    num_predict_seen = np.zeros((len(thresh), ))
    # fix class
    for leaf_name in confidence_dict.keys():
        
        max_indices = torch.tensor([record['idx'] for record in confidence_dict[leaf_name]]).long()
        max_probs = torch.tensor([record[value_key] for record in confidence_dict[leaf_name]])
        
        for i in range(0, len(thresh)):
            proceed = max_probs.cpu() > torch.full(max_probs.size(), thresh[i])
            num_predict_seen[i] += torch.sum(proceed).item()
            correct_vec = ~proceed
            num_corr = torch.sum(correct_vec).item()
            softmax_correct[i] += num_corr
            per_cls_softmax_correct[i][leaf_name] = num_corr/len(confidence_dict[leaf_name])
            
    total = float(sum([len(confidence_dict[leaf_name]) for leaf_name in confidence_dict.keys()]))
    success_rate = [softmax_correct[i]/total for i in range(0, len(softmax_correct))]
    
    info = {'success_rate': success_rate, 'per_cls_softmax_correct': per_cls_softmax_correct, 'mistake_log': mistake_log, 
            'semantic_precision': semantic_precision, 'num_predict_seen': num_predict_seen, 'total': total}
    
    if save_path is not None:
        
        if os.path.exists(save_path):
            raise ValueError(save_path + ' already exists.')
        
        with open(save_path, 'wb') as f:
            pickle.dump(info, f)
 
    return info

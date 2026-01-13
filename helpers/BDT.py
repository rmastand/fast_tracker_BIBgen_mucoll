import numpy as np
import xgboost as xgb
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import cupy as cp


"""

def custom_predict(x):
    return np.clip(x, a_min = 0, a_max = 1e5)


def MLE_Loss(preds, dtrain):
    
    labels = dtrain.get_label()
    weights = dtrain.get_weight()
    
    preds = custom_predict(preds)
    grad = 1.0 - ((labels * (1.0 + preds)) / preds)
    hess = labels / preds**2

    return grad*weights, hess*weights


def MLE_metric(preds, dtrain):
    
    labels = dtrain.get_label()
    weights = dtrain.get_weight()
    preds = custom_predict(preds)

    loss = -(labels*np.log(preds + 1e-5) + (1.0 - labels)*(1.0 - preds) )

    return "MLE_loss", np.sum(loss*weights)



"""

def custom_predict(x):
    
    return 1.0 / (1.0+np.exp(-x))


def MLE_Loss(preds, dtrain):
    
    labels = dtrain.get_label()
    weights = dtrain.get_weight()
    
    preds = custom_predict(preds)
    grad = preds - labels
    hess = preds * (1.0 - preds)
    return grad*weights, hess*weights



def MLE_metric(preds, dtrain):
    
    labels = dtrain.get_label()
    weights = dtrain.get_weight()
    preds = custom_predict(preds)
    
    logloss = -(labels*np.log(preds) + (1.0-labels)*(np.log(1.0-preds)))
    return "MLE_loss", np.mean(weights*logloss)





def run_BDT_bump_hunt(flow_samples_SR, data_samples_SR, data_samples_SB, num_folds, hyperparams_dict, 
                      alt_test_sets_data={}, visualize=True, pdf=None, take_ensemble_avg=False, device=None):
    
    """
    Classifier is trained only on SR data, to distinguish flow SR samples from SR data
    
    Classifier is evaluated on test data from SR AND SB
    
    Note that alt test sets are NOT split into folds, since we aren't training on them. We do get diff scores for each fold
    """
    
    test_data_splits  = {i:0 for i in range(num_folds)}
    scores_splits = {i:0 for i in range(num_folds)}
    
    # split the alternative test sets
    alt_scores_splits = {}
    alt_data_splits = {}
    for alt_id in alt_test_sets_data.keys():
        alt_scores_splits[alt_id] = {i:0 for i in range(num_folds)}
        # generate a nfold split for the alt test data
        loc_alt_data_split = np.array_split(shuffle(alt_test_sets_data[alt_id]), num_folds) 
        alt_data_splits[alt_id] = {i:loc_alt_data_split[i] for i in range(num_folds)}
        
    # shuffle anything with SB data to mix the low and high masses before splitting 
    flow_samples_SR = shuffle(flow_samples_SR)
    data_samples_SR = shuffle(data_samples_SR)
    data_samples_SB = shuffle(data_samples_SB)
    
    flow_SR_splits = np.array_split(flow_samples_SR, num_folds)
    data_SR_splits = np.array_split(data_samples_SR, num_folds)
    data_SB_splits = np.array_split(data_samples_SB, num_folds) 
    
    for i_fold in range(num_folds):
            
        print(f"Fold {i_fold}:")
        
        """
        ASSEMBLE THE TRAIN / VAL / TEST DATA
        """
        
        # Assemble the train / test data
        training_data, training_labels = [], []
        validation_data, validation_labels = [], []
        testing_data = []

        for ii in range(num_folds):
            
            # test set comprised of SR and SB data
            if ii == i_fold:
                testing_data.append(data_SR_splits[ii])
                testing_data.append(data_SB_splits[ii])
                
            # validation set: flow SR samples, data SR sampkes
            elif ((ii+1)%num_folds) == i_fold:
                validation_data.append(flow_SR_splits[ii])
                validation_labels.append(np.zeros((flow_SR_splits[ii].shape[0],1)))
                validation_data.append(data_SR_splits[ii])
                validation_labels.append(np.ones((data_SR_splits[ii].shape[0],1)))
                
            else:
                training_data.append(flow_SR_splits[ii])
                training_labels.append(np.zeros((flow_SR_splits[ii].shape[0],1)))
                training_data.append(data_SR_splits[ii])
                training_labels.append(np.ones((data_SR_splits[ii].shape[0],1)))
                
        X_train_fold = np.concatenate(training_data)
        Y_train_fold = np.concatenate(training_labels)
        X_val_fold = np.concatenate(validation_data)
        Y_val_fold = np.concatenate(validation_labels)
        
        X_test_fold = np.concatenate(testing_data)
        
        # record the local fold data
        test_data_splits[i_fold] = X_test_fold
     
        """
        SORT THE WEIGHTS OUT
        """
        
        # First do the weights for the regular BC (non-decorr)
        #class_weight = {0: 1, 1: sum(Y_train_fold==0)[0]/sum(Y_train_fold==1)[0]}
        class_weight = {0: sum(Y_train_fold==1)[0]/sum(Y_train_fold==0)[0], 1: 1}
        class_weights_train = class_weight[0]*(1.0-Y_train_fold)+class_weight[1]*Y_train_fold
        class_weights_val = class_weight[0]*(1.0-Y_val_fold)+class_weight[1]*Y_val_fold
        
        """
        COMBINE W/ DECORRELATED TRAINING
        """
        # we only want to train on the non-mass features

        X_train_fold = X_train_fold[:,:-1]
        X_val_fold = X_val_fold[:,:-1]

        w_train_fold = class_weights_train
        w_val_fold = class_weights_val

        # shuffle for good measure
        X_train_fold, Y_train_fold, w_train_fold = shuffle(X_train_fold, Y_train_fold, w_train_fold)
        X_val_fold, Y_val_fold, w_val_fold = shuffle(X_val_fold, Y_val_fold, w_val_fold)
        
        X_test_fold = X_test_fold[:,:-1]
        
        
        print(f"X train shape: {X_train_fold.shape}, Y train shape: {Y_train_fold.shape}, w train shape: {w_train_fold.shape}.")
        print(f"X val shape: {X_val_fold.shape}, Y val shape: {Y_val_fold.shape}, w val shape: {w_val_fold.shape}.")
        print(f"X test shape: {X_test_fold.shape}." )

        if hyperparams_dict["sampling_method"] == "gradient_based":
            X_train_fold = cp.array(X_train_fold)
            Y_train_fold = cp.array(Y_train_fold)
            w_train_fold = cp.array(w_train_fold)
            X_val_fold = cp.array(X_val_fold)
            Y_val_fold = cp.array(Y_val_fold)
            w_val_fold = cp.array(w_val_fold)
            X_test_fold = cp.array(X_test_fold)
            dd = device
        else:
            dd = "cpu"
        
        """
        INITIALIZE SCORE OBJECTS
        """
        
        scores_fold = np.empty((X_test_fold.shape[0], hyperparams_dict["n_ensemble"]))
        alt_scores_fold = {}
        for alt_id in alt_test_sets_data.keys():
            alt_scores_fold[alt_id] = np.empty((alt_data_splits[alt_id][i_fold].shape[0],  hyperparams_dict["n_ensemble"]))
            
            
   
        """
        TRAIN ENSEMBLE OF TREES
        """
    
        if visualize:
            plt.figure()
        
        for i_tree in range( hyperparams_dict["n_ensemble"]):
            
            if i_tree % 10 == 0:
                print("   Network number:", i_tree)
            random_seed = i_fold* hyperparams_dict["n_ensemble"] + i_tree + 1
                
                
            eval_set = [(X_train_fold, Y_train_fold), (X_val_fold, Y_val_fold)]
            

            bst_i = xgb.XGBClassifier(
                n_estimators=hyperparams_dict["n_estimators"],
                max_depth=hyperparams_dict["max_depth"], 
                learning_rate=hyperparams_dict["learning_rate"], 
                min_split_loss=hyperparams_dict["min_split_loss"], 
                max_delta_step=hyperparams_dict["max_delta_step"],
                subsample=hyperparams_dict["subsample"], 
                early_stopping_rounds=hyperparams_dict["early_stopping_rounds"],
                sampling_method=hyperparams_dict["sampling_method"], 
                tree_method="hist",
                objective="binary:logistic",
                device=dd,
                random_state = random_seed,
                eval_metric="logloss")

            bst_i.fit(X_train_fold, Y_train_fold, sample_weight=w_train_fold, 
                      eval_set=eval_set, sample_weight_eval_set = [w_train_fold, w_val_fold],
                      verbose=False)
            results_f = bst_i.evals_result()
            losses = results_f["validation_0"]["logloss"]
            losses_val = results_f["validation_1"]["logloss"]
            best_epoch = bst_i.best_iteration


            # get scores
            scores_fold[:,i_tree] = bst_i.predict_proba(X_test_fold, iteration_range=(0,bst_i.best_iteration))[:,1]
            for alt_id in alt_test_sets_data.keys():
                if hyperparams_dict["sampling_method"] == "gradient_based":
                    tmp = cp.array(alt_data_splits[alt_id][i_fold][:,:-1])
                else:
                    tmp = alt_data_splits[alt_id][i_fold][:,:-1]
                    
                alt_scores_fold[alt_id][:,i_tree] = bst_i.predict_proba(tmp, iteration_range=(0,bst_i.best_iteration))[:,1]
            
            
            if visualize:
                
                plt.plot(losses, label = f"{i_tree}", color = f"C{i_tree}")
                plt.plot(losses_val, color = f"C{i_tree}", linestyle = "dashed")
                plt.axvline(best_epoch, color = f"C{i_tree}")
                
        
        if visualize:
            if pdf is not None:
                plt.xlabel("Epochs")
                plt.ylabel("Loss")
                
                plt.title(f"Fold {i_fold}")
                plt.legend()
                pdf.savefig()
            else:
                plt.show()
            
        """
        AVERAGE OVER ENSEMBLE
        """     
        if take_ensemble_avg:
            scores_splits[i_fold] = np.mean(scores_fold, axis = 1)
            for alt_id in alt_test_sets_data.keys():
                alt_scores_splits[alt_id][i_fold] = np.mean(alt_scores_fold[alt_id], axis = 1)
        else:
            scores_splits[i_fold] = scores_fold
            for alt_id in alt_test_sets_data.keys():
                alt_scores_splits[alt_id][i_fold] = alt_scores_fold[alt_id]
        """
        if visualize:
            plt.figure()
            plt.hist2d(test_data_splits[i_fold][:,-1], np.mean(scores_fold, axis = 1), bins = 40, cmap = "hot", density = True)
            plt.xlabel("M (rescaled)")
            plt.ylabel("score")
            #plt.axvline(SR_min_rescaled, color = "red")
            #plt.axvline(SR_max_rescaled, color = "red")
            plt.colorbar()
            plt.colorbar()
            plt.show()
        """
        """
        
        plt.figure()
        plt.hist2d(alt_data_splits["FPR_validation"][i_fold][:,-1], np.mean(alt_scores_splits["FPR_validation"][i_fold], axis = 1), bins = 40, cmap = "hot", density = True)
        plt.xlabel("M (rescaled)")
        plt.ylabel("score")
        plt.axvline(SR_min_rescaled, color = "red")
        plt.axvline(SR_max_rescaled, color = "red")
        plt.title("FPR_validation")
        plt.colorbar()
        plt.show()
        
        plt.figure()
        plt.hist2d(alt_data_splits["samesign"][i_fold][:,-1], np.mean(alt_scores_splits["samesign"][i_fold], axis = 1), bins = 40, cmap = "hot", density = True)
        plt.xlabel("M (rescaled)")
        plt.ylabel("score")
        #plt.axvline(SR_min_rescaled, color = "red")
        #plt.axvline(SR_max_rescaled, color = "red")
        plt.title("samesign")
        plt.colorbar()
        plt.show()
        print()
        """
        
    return test_data_splits, scores_splits, alt_data_splits, alt_scores_splits




def run_BDT_bump_hunt_custom(flow_samples_SR, data_samples_SR, data_samples_SB, num_folds, hyperparams_dict, 
                      alt_test_sets_data={}, visualize=True, pdf=None, take_ensemble_avg=False):
    
    """
    Classifier is trained only on SR data, to distinguish flow SR samples from SR data
    
    Classifier is evaluated on test data from SR AND SB
    
    Note that alt test sets are NOT split into folds, since we aren't training on them. We do get diff scores for each fold
    """
    
    test_data_splits  = {i:0 for i in range(num_folds)}
    scores_splits = {i:0 for i in range(num_folds)}
    
    # split the alternative test sets
    alt_scores_splits = {}
    alt_data_splits = {}
    for alt_id in alt_test_sets_data.keys():
        alt_scores_splits[alt_id] = {i:0 for i in range(num_folds)}
        # generate a nfold split for the alt test data
        loc_alt_data_split = np.array_split(shuffle(alt_test_sets_data[alt_id]), num_folds) 
        alt_data_splits[alt_id] = {i:loc_alt_data_split[i] for i in range(num_folds)}
        
    # shuffle anything with SB data to mix the low and high masses before splitting 
    flow_samples_SR = shuffle(flow_samples_SR)
    data_samples_SR = shuffle(data_samples_SR)
    data_samples_SB = shuffle(data_samples_SB)
    
    flow_SR_splits = np.array_split(flow_samples_SR, num_folds)
    data_SR_splits = np.array_split(data_samples_SR, num_folds)
    data_SB_splits = np.array_split(data_samples_SB, num_folds) 
    
    for i_fold in range(num_folds):
            
        print(f"Fold {i_fold}:")
        
        """
        ASSEMBLE THE TRAIN / VAL / TEST DATA
        """
        
        # Assemble the train / test data
        training_data, training_labels = [], []
        validation_data, validation_labels = [], []
        testing_data = []

        for ii in range(num_folds):
            
            # test set comprised of SR and SB data
            if ii == i_fold:
                testing_data.append(data_SR_splits[ii])
                testing_data.append(data_SB_splits[ii])
                
            # validation set: flow SR samples, data SR sampkes
            elif ((ii+1)%num_folds) == i_fold:
                validation_data.append(flow_SR_splits[ii])
                validation_labels.append(np.zeros((flow_SR_splits[ii].shape[0],1)))
                validation_data.append(data_SR_splits[ii])
                validation_labels.append(np.ones((data_SR_splits[ii].shape[0],1)))
                
            else:
                training_data.append(flow_SR_splits[ii])
                training_labels.append(np.zeros((flow_SR_splits[ii].shape[0],1)))
                training_data.append(data_SR_splits[ii])
                training_labels.append(np.ones((data_SR_splits[ii].shape[0],1)))
                
        X_train_fold = np.concatenate(training_data)
        Y_train_fold = np.concatenate(training_labels)
        X_val_fold = np.concatenate(validation_data)
        Y_val_fold = np.concatenate(validation_labels)
        
        X_test_fold = np.concatenate(testing_data)
        
        # record the local fold data
        test_data_splits[i_fold] = X_test_fold
     
        """
        SORT THE WEIGHTS OUT
        """
        
        # First do the weights for the regular BC (non-decorr)
        #class_weight = {0: 1, 1: sum(Y_train_fold==0)[0]/sum(Y_train_fold==1)[0]}
        class_weight = {0: sum(Y_train_fold==1)[0]/sum(Y_train_fold==0)[0], 1: 1}
        class_weights_train = class_weight[0]*(1.0-Y_train_fold)+class_weight[1]*Y_train_fold
        class_weights_val = class_weight[0]*(1.0-Y_val_fold)+class_weight[1]*Y_val_fold
        
        """
        COMBINE W/ DECORRELATED TRAINING
        """
        # we only want to train on the non-mass features

        X_train_fold = X_train_fold[:,:-1]
        X_val_fold = X_val_fold[:,:-1]

        w_train_fold = class_weights_train
        w_val_fold = class_weights_val

        # shuffle for good measure
        X_train_fold, Y_train_fold, w_train_fold = shuffle(X_train_fold, Y_train_fold, w_train_fold)
        X_val_fold, Y_val_fold, w_val_fold = shuffle(X_val_fold, Y_val_fold, w_val_fold)
        
        X_test_fold = X_test_fold[:,:-1]
        
        
        print(f"X train shape: {X_train_fold.shape}, Y train shape: {Y_train_fold.shape}, w train shape: {w_train_fold.shape}.")
        print(f"X val shape: {X_val_fold.shape}, Y val shape: {Y_val_fold.shape}, w val shape: {w_val_fold.shape}.")
        print(f"X test shape: {X_test_fold.shape}." )
        
        """
        INITIALIZE SCORE OBJECTS
        """
        
        scores_fold = np.empty((X_test_fold.shape[0], hyperparams_dict["n_ensemble"]))
        alt_scores_fold = {}
        for alt_id in alt_test_sets_data.keys():
            alt_scores_fold[alt_id] = np.empty((alt_data_splits[alt_id][i_fold].shape[0],  hyperparams_dict["n_ensemble"]))
            
   
        """
        TRAIN ENSEMBLE OF TREES
        """
    
        
        plt.figure()
        
        for i_tree in range( hyperparams_dict["n_ensemble"]):
            
            if i_tree % 10 == 0:
                print("   Network number:", i_tree)
            random_seed = i_fold* hyperparams_dict["n_ensemble"] + i_tree + 1

            loc_train_data = xgb.DMatrix(X_train_fold, Y_train_fold, weight = w_train_fold)
            loc_val_data = xgb.DMatrix(X_val_fold, Y_val_fold, weight = w_val_fold)
            
            results_f = {}
            
            bst_i = xgb.train(
                dtrain = loc_train_data,
                obj = MLE_Loss,
                custom_metric = MLE_metric,
                evals=[(loc_train_data, 'dtrain'), (loc_val_data, 'dval')],
                early_stopping_rounds = hyperparams_dict["early_stopping_rounds"],
                num_boost_round = hyperparams_dict["n_estimators"],
                params = {'tree_method': 'exact',
                          "max_depth": hyperparams_dict["max_depth"], 
                            "learning_rate": hyperparams_dict["learning_rate"], 
                             "subsample": hyperparams_dict["subsample"], 
                                                      "random_state": random_seed},
            evals_result = results_f,
                verbose_eval = False,
                maximize = False

                )

        
            losses = results_f["dtrain"]["MLE_loss"]
            losses_val = results_f["dval"]["MLE_loss"]
            best_epoch = bst_i.best_iteration


            # get scores
            loc_test_data = xgb.DMatrix(X_test_fold)
            tmp = bst_i.predict(loc_test_data, iteration_range=(0,bst_i.best_iteration))
            scores_fold[:,i_tree] = custom_predict(tmp)
            for alt_id in alt_test_sets_data.keys():
                data_matrix =  xgb.DMatrix(alt_data_splits[alt_id][i_fold][:,:-1])
                tmp = bst_i.predict(data_matrix, iteration_range=(0,bst_i.best_iteration))
                alt_scores_fold[alt_id][:,i_tree] = custom_predict(tmp)
                
            
            
                
            plt.plot(losses, label = f"{i_tree}", color = f"C{i_tree}")
            plt.plot(losses_val, color = f"C{i_tree}", linestyle = "dashed")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.axvline(best_epoch, color = f"C{i_tree}")
        plt.title(f"Fold {i_fold}")
    
        if pdf is not None:
            pdf.savefig()
        else:
            plt.show()
        
        """
        AVERAGE OVER ENSEMBLE
        """     
        if take_ensemble_avg:
            scores_splits[i_fold] = np.mean(scores_fold, axis = 1)
            for alt_id in alt_test_sets_data.keys():
                alt_scores_splits[alt_id][i_fold] = np.mean(alt_scores_fold[alt_id], axis = 1)
        else:
            scores_splits[i_fold] = scores_fold
            for alt_id in alt_test_sets_data.keys():
                alt_scores_splits[alt_id][i_fold] = alt_scores_fold[alt_id]
     
    return test_data_splits, scores_splits, alt_data_splits, alt_scores_splits





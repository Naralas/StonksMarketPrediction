import lime
from lime import lime_tabular, submodular_pick
import matplotlib.pyplot as plt
import matplotlib


def explain_model(predict_func, X_train, y_train, data_columns, class_names=[], n_exps=2, mode='classification', recurrent=False):
    if not recurrent:
        explainer = lime_tabular.LimeTabularExplainer(X_train, training_labels=y_train,
                                                    discretize_continuous = True,
                                                    feature_names=data_columns, class_names=class_names,                                                
                                                    verbose=False, mode=mode)
    else:
        explainer = lime_tabular.RecurrentTabularExplainer(X_train, training_labels=y_train,
                                                    discretize_continuous = True,
                                                    feature_names=data_columns, class_names=class_names,                                                
                                                    verbose=False, mode=mode)
        
    sp_obj = submodular_pick.SubmodularPick(explainer, X_train, predict_func, sample_size=n_exps, num_features=5, num_exps_desired=1)
    return sp_obj.explanations

        

def plot_explanation(exp, class_names=[], mode='classification'):
    label = list(exp.as_map().keys())[0]
    fig = exp.as_pyplot_figure(label=label)
    ax = plt.gca()
    ax.set_ylabel('Feature representation', fontsize=16)
    ax.set_xlabel('Weight', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    if mode is 'classification':
        plt.suptitle("Explanation probabilities for classification")
        ax.set_title(f"Class : {class_names[label]}", fontsize=24)
    else:
        plt.suptitle(f"Explanation weights for regression")
        ax.set_title(f"Model prediction : {exp.predicted_value:.2f}, LIME local prediction : {exp.local_pred[0]:.2f}, intercept : {exp.intercept[0]:.2f}", fontsize=24)
    return ax
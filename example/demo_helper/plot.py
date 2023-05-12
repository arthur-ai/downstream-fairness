import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# from eval import get_abs_ci

# resultdf is the output of `get_eval`

def plot_result(ax, resultdf, metrics=['positivity_rate_differences', 'tpr_differences', 'eqodds_differences'], legend_map = None, title=None, size=None):

  if "trial" not in resultdf.columns:
    print("Expected multiple trials! Plotting for the single trial:")

  legend_map = {} if legend_map is None else legend_map
  for metric in metrics:
    if len(legend_map.keys()) < len(metrics):
      legend_map = {}
      legend_map[metric] = metric
    sns.lineplot(x=resultdf.thresholds, y=np.abs(resultdf[metric]), label=legend_map[metric], ax=ax)

  plt.legend(fontsize=int(size*0.7))
  plt.ylim((0,0.5))
  # hacky but works
  # ax.set(xlabel = "Threshold")
  ax.set_xlabel("Threshold", fontsize=int(size*0.7))
  ax.set(ylabel = "")
  if title:
    if size:
      plt.title(title, size=size)  
    else:
      plt.title(title)

def plot_metrics_comparison(axes, resultdf, groups, metrics):

  for i, m in enumerate(metrics):
    for g in groups:
      sns.lineplot(x=resultdf.thresholds, y=resultdf[m+'_'+g], ax=axes[i], label = g)
    axes[i].legend()
    axes[i].set_xlabel('Threshold')
    axes[i].set_ylabel(m)
    axes[i].set_title(m + ' by group')

def plot_ROC_curve(resultdf):
  
  ax = sns.lineplot(x=resultdf.fpr_A, y=resultdf.tpr_A, label="ROC group A")
  ax = sns.lineplot(x=resultdf.fpr_B, y=resultdf.tpr_B, label="ROC group B")

  ax.set(xlabel="FPR")
  ax.set(ylabel="TPR")

  plt.legend()

def plot_result_summary(resultdf, filename=None): 
  plt.figure(figsize=(25, 10))
  plt.subplot(2,3,1)
  plot_ROC_curve(resultdf)
  plt.subplot(2,3,2)
  plot_result(resultdf)
  plt.subplot(2,3,3)
  plot_result(resultdf, ['acc_A', 'acc_B', 'acc_overall'])
  plt.subplot(2,3,4)
  plot_result(resultdf, ['selection_A', 'selection_B'])
  plt.subplot(2,3,5)
  plot_result(resultdf, ['tpr_A', 'tpr_B'])
  plt.subplot(2,3,6)
  plot_result(resultdf, ['fpr_A', 'fpr_B'])

  if filename:
    plt.savefig(filename + '.png')

def lambda_aucs_abs(wts, filename=None): # FIG 2 
  sns.lineplot(data=wts, x="adjust_weight", y=[0]*1010, label="zero area between curves", linestyle='dashed', color='darkgray')
  sns.lineplot(data=wts, x="adjust_weight", y="abs_pos", label="area between selection rate curves", color='indianred')
  sns.lineplot(data=wts, x="adjust_weight", y="abs_tpr", label="area between TPR curves", color='mediumseagreen')
  ax = sns.lineplot(data=wts, x="adjust_weight", y="abs_fpr", label="area between FPR curves", color='slateblue')

  plt.rcParams["axes.labelsize"] = 13
  plt.rcParams["axes.titlesize"] = 17
  ax.set(xlabel="adjustment weight $\\lambda$")
  ax.set(ylabel="Area between curves (absolute)")
  ax.set(title="Absolute area between curves over $\\lambda$")

  if filename:
    plt.savefig(filename + '_auc_abs.png')

def lambda_aucs_signed(wts, filename=None): # FIG 2 
  sns.lineplot(data=wts, x="adjust_weight", y=[0]*1010, label="zero area between curves", linestyle='dashed', color='darkgray')
  sns.lineplot(data=wts, x="adjust_weight", y='positivity_rate_differences', label="area between selection rate curves", color='indianred')
  sns.lineplot(data=wts, x="adjust_weight", y="tpr_differences", label="area between TPR curves", color='mediumseagreen')
  ax = sns.lineplot(data=wts, x="adjust_weight", y="fpr_differences", label="area between FPR curves", color='slateblue')

  plt.rcParams["axes.labelsize"] = 13
  plt.rcParams["axes.titlesize"] = 17
  ax.set(xlabel="adjustment weight $\\lambda$")
  ax.set(ylabel="Area between curves (signed)")
  ax.set(title="Signed area between curves over $\\lambda$")

  if filename:
    plt.savefig(filename + '_auc_signed.png')


def lambda_acc(wts, filename=None):
  sns.lineplot(data=wts, x="adjust_weight", y="acc_overall", label="overall")
  sns.lineplot(data=wts, x="adjust_weight", y="acc_A", label="A")
  ax = sns.lineplot(data=wts, x="adjust_weight", y="acc_B", label="B")

  plt.rcParams["axes.labelsize"] = 13
  plt.rcParams["axes.titlesize"] = 17
  ax.set(xlabel="adjustment weight $\\lambda$")
  ax.set(ylabel="Accuracy")
  ax.set(title="Average overall accuracy over $\\lambda$")

  plt.savefig(filename + '_acc.png')

def lambda_auc_full(wts, title=None, hide_key=True):
  if not hide_key:
    sns.lineplot(data=wts, x="adjust_weight", y=[0]*1010, label="zero area between curves", linestyle='dashed', color='darkgray')
    sns.lineplot(data=wts, x="adjust_weight", y="abs_pos", label="area between selection rate curves", color='indianred')
    sns.lineplot(data=wts, x="adjust_weight", y="abs_tpr", label="area between TPR curves", color='mediumseagreen')
    sns.lineplot(x=wts.adjust_weight, y=wts.abs_tpr + wts.abs_fpr, label="area between TPR & FPR curves", color='cadetblue')
    ax = sns.lineplot(data=wts, x="adjust_weight", y="abs_fpr", label="area between FPR curves", color='slateblue')

  else: # this is horrible but i gave up on fighting matplotlib
    sns.lineplot(data=wts, x="adjust_weight", y=[0]*1010, linestyle='dashed', color='darkgray')
    sns.lineplot(data=wts, x="adjust_weight", y="abs_pos", color='indianred')
    sns.lineplot(data=wts, x="adjust_weight", y="abs_tpr", color='mediumseagreen')
    sns.lineplot(x=wts.adjust_weight, y=wts.abs_tpr + wts.abs_fpr, color='cadetblue')
    ax = sns.lineplot(data=wts, x="adjust_weight", y="abs_fpr", color='slateblue')


  tpr_ci = get_abs_ci(wts, 'abs_tpr')
  tpr = np.mean([wts.iloc[101*i + int(np.mean(tpr_ci)*100)]['abs_tpr'] for i in range(10)])
  plt.plot(np.mean(tpr_ci), tpr, 'd', color='seagreen')
  sns.lineplot(x=tpr_ci, y=tpr, color='seagreen', linestyle='dashed')

  fpr_ci = get_abs_ci(wts, 'abs_fpr')
  fpr = np.mean([wts.iloc[101*i + int(np.mean(fpr_ci)*100)]['abs_fpr'] for i in range(10)])
  plt.plot(np.mean(fpr_ci), fpr, 'd', color='darkslateblue')
  sns.lineplot(x=fpr_ci, y=fpr, color='darkslateblue', linestyle='dashed')

  eo_ci = get_abs_ci(wts, ['abs_fpr', 'abs_tpr'])
  eo = np.mean([wts.iloc[101*i + int(np.mean(eo_ci)*100)]['abs_fpr'] for i in range(10)])
  eo += np.mean([wts.iloc[101*i + int(np.mean(eo_ci)*100)]['abs_tpr'] for i in range(10)])
  plt.plot(np.mean(eo_ci), eo, 'd', color='darkcyan')
  sns.lineplot(x=eo_ci, y=eo, color='darkcyan', linestyle='dashed')

  plt.rcParams["axes.labelsize"] = 13
  plt.rcParams["axes.titlesize"] = 17
  ax.set(xlabel="adjustment weight $\\lambda$")
  ax.set(ylabel="Area between curves (absolute)")
  if title is None:
    ax.set(title="Absolute area between curves over $\\lambda$")
  else:
    ax.set(title=title)

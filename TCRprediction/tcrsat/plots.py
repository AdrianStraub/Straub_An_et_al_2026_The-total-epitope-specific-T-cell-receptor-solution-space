import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import pandas as pd


def plot_confusion_matrix(cm, title):
	group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
	group_counts = ["{0: 0.0f}".format(value) for value in cm.flatten()]
	group_percentages = ["{0:.2%}".format(value) for value in cm.flatten() / cm.sum()]
	labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
	labels = np.asarray(labels).reshape(2, 2)
	sns.heatmap(cm, annot=labels, fmt='', cmap='Blues')
	plt.xlabel('True Label')
	plt.ylabel('Predicted Label')
	plt.title(title)


def plot_confusion_matrix_wrapper(y_hat, y, optimal_threshold_auc, optimal_threshold_prc, mode='undefined split', experiment=None, current_epoch=-1):
	plt.figure(figsize=(11, 5))

	plt.subplot(1, 2, 1)
	cm = metrics.confusion_matrix(y_hat > optimal_threshold_auc, y.int())
	plot_confusion_matrix(cm, f'AUC Thresh {optimal_threshold_auc:.4f}')

	plt.subplot(1, 2, 2)
	cm = metrics.confusion_matrix(y_hat > optimal_threshold_prc, y.int())
	plot_confusion_matrix(cm, f'F1 Thresh {optimal_threshold_auc:.4f}')

	plt.suptitle(f'{mode} epoch={current_epoch}')
	if experiment is not None:
		plt.close()
		experiment.log_figure(figure_name=f'{mode}_cm', step=current_epoch)
	else:
		plt.show()


def plot_roc_prc_curves(fpr, tpr, precision, recall, y_hat, y, mode='undefined split', experiment=None, current_epoch=-1):
	fig, [ax_roc, ax_prc] = plt.subplots(1, 2, figsize=(11, 5))

	# AUC Curve
	roc_auc = metrics.auc(fpr, tpr)
	display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
	display.plot(ax=ax_roc)

	# Precision-Recall Curve
	aps = metrics.average_precision_score(y, y_hat)
	disp = metrics.PrecisionRecallDisplay(precision=precision, recall=recall, average_precision=aps)
	disp.plot(ax=ax_prc)

	plt.suptitle(f'{mode} e={current_epoch}')
	if experiment is not None:
		experiment.log_figure(figure_name=f'{mode}_curve', step=current_epoch)
		plt.close()
	else:
		plt.show()

def plot_predicted_scores(y_hat, y, optimal_threshold_auc, optimal_threshold_prc, mode='undefined split', experiment=None, current_epoch=-1):
	df = pd.DataFrame({'y_hat': y_hat.cpu(), 'true_label': y.cpu()})
	lower, upper = df['y_hat'].quantile(0.05), df['y_hat'].quantile(0.95)
	df_no_outliers = df[(df['y_hat'] > lower) & (df['y_hat'] < upper)]

	plt.figure(figsize=(20, 4))

	plt.subplot(1, 4, 1)
	sns.histplot(df, x='y_hat', hue='true_label', bins=200, stat='percent', common_norm=False)
	plt.axvline(optimal_threshold_auc, c='r')
	plt.axvline(optimal_threshold_prc, c='g')
	plt.title('With outliers')

	plt.subplot(1, 4, 2)
	sns.histplot(df_no_outliers, x='y_hat', hue='true_label', bins=200, stat='percent', common_norm=False)
	plt.axvline(optimal_threshold_auc, c='r')
	plt.axvline(optimal_threshold_prc, c='g')
	plt.title('Without outliers 0.05 - 0.95 quantile')

	plt.subplot(1, 4, 3)
	sns.stripplot(df, y='y_hat', x='true_label', hue='true_label', size=2)
	sns.violinplot(df, y='y_hat', x='true_label', color="0.8")
	plt.axhline(optimal_threshold_auc, c='r')
	plt.axhline(optimal_threshold_prc, c='g')
	plt.title('With outliers')

	plt.subplot(1, 4, 4)
	sns.stripplot(df_no_outliers, y='y_hat', x='true_label', hue='true_label', size=2)
	# sns.violinplot(df_no_outliers, y='y_hat', x='true_label', color="0.8")
	plt.axhline(optimal_threshold_auc, c='r')
	plt.axhline(optimal_threshold_prc, c='g')
	plt.title('With outliers 0.05 - 0.95 quantile')

	plt.suptitle(f'{mode} e={current_epoch}')
	if experiment is not None:
		experiment.log_figure(figure_name=f'{mode}_pred_scores', step=current_epoch)
		plt.close()
	else:
		plt.show()

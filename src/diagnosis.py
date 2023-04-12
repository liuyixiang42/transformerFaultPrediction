import numpy as np
from sklearn.metrics import ndcg_score


# Hit@p%指标是一种用于评估多标签分类模型性能的指标。该指标主要用于评估模型在预测结果中覆盖多少正样本。
# 百分比值p是Hit@p%指标中的一个参数，用于控制选取得分最高的样本数占正样本数的比例。
# 具体来说，对于一个样本，它的得分越高，则它被正确预测的可能性就越大。因此，选取得分最高的样本作为预测结果，可以提高预测的准确率。但是，如果只选取得分最高的样本，可能会出现预测结果不全面的情况，即预测结果只包括一部分正样本，而另外一部分正样本被漏掉了。
# 因此，需要在保证准确率的前提下，尽可能地覆盖更多的正样本。
#
# 百分比值p的作用就是控制选取得分最高的样本数占正样本数的比例，从而在保证准确率的前提下，尽可能地覆盖更多的正样本。
# 如果设置p为100%，则会选取得分最高的样本作为预测结果，此时预测结果的准确率最高，但是可能会漏掉部分正样本。如果设置p为150%，则会选取得分最高的1.5倍样本作为预测结果，此时预测结果的准确率可能会稍微降低一些，但是能够更全面地覆盖更多的正样本。
# 因此，百分比值p的具体取值需要根据具体的场景和需求来进行选择。

# ascore是样本对每一种类别的置信值
# labels是样本对每一种类别是正样本还是负样本（[0,1]）
def hit_att(ascore, labels, ps = [100, 50]):
	res = {}
	for p in ps:
		hit_score = []
		for i in range(ascore.shape[0]):
			a, l = ascore[i], labels[i]
			a, l = np.argsort(a).tolist()[::-1], set(np.where(l == 1)[0])
			if l:
				size = round(p * len(l) / 100)
				a_p = set(a[:size])
				intersect = a_p.intersection(l)
				hit = len(intersect) / len(l)
				hit_score.append(hit)
		res[f'Hit@{p}%'] = np.mean(hit_score)
	return res

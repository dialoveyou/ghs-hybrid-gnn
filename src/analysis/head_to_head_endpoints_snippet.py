# ============================================================
# Table 5용 6개 overlapping endpoint 클래스별 지표 추출
# 노트북 9.2절(AUC/Balanced Accuracy 계산) 이후에 실행하세요.
# test_blend_proba, pred_stacked, test_y_np, blend_thresholds, CLASS_NAMES 가
# 이미 메모리에 있어야 합니다 (섹션 3.4, 3.6, 9.2 실행 후).
# ============================================================
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score

# Fuadah et al.과 겹치는 6개 클래스
OVERLAP_CLASSES = [
    "09_Acute_Toxicity_Oral",
    "10_Acute_Toxicity_Dermal",
    "11_Acute_Toxicity_Inhalation",
    "13_Skin_Corrosion_Irritation",
    "14_Eye_Damage_Irritation",
    "15_Sensitization",
]

print(f"{'Class':<32}{'AUC':>8}{'F1':>8}{'BalAcc':>8}")
results = {}
for cls_name in OVERLAP_CLASSES:
    c = CLASS_NAMES.index(cls_name)
    y_true = test_y_np[:, c]
    y_proba = test_blend_proba[:, c]
    y_pred = pred_stacked[:, c]

    auc = roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else float("nan")
    f1 = f1_score(y_true, y_pred, zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    results[cls_name] = {"AUC": auc, "F1": f1, "BalAcc": bal_acc}
    print(f"{cls_name:<32}{auc:>8.3f}{f1:>8.3f}{bal_acc:>8.3f}")

mean_auc = np.mean([v["AUC"] for v in results.values()])
mean_f1 = np.mean([v["F1"] for v in results.values()])
mean_ba = np.mean([v["BalAcc"] for v in results.values()])
print(f"\n{'Mean (6 endpoints)':<32}{mean_auc:>8.3f}{mean_f1:>8.3f}{mean_ba:>8.3f}")

import pandas as pd
pd.DataFrame(results).T

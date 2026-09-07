# ============================================================
# Table 5용 6개 overlapping endpoint AUC 추출 (전체 코드)
# ------------------------------------------------------------
# 전제조건: 노트북에서 9.2절(AUC/Balanced Accuracy 계산, §섹션 "9.2 ROC-AUC 및
# Balanced Accuracy")까지 이미 실행되어 있어야 합니다.
# 즉 test_blend_proba, test_y_np, CLASS_NAMES 변수가 커널 메모리에 남아있어야 해요.
#
# Colab 세션이 끊겼다면(런타임 재시작 등) 이 셀만 실행해서는 안 되고,
# 노트북을 처음부터 다시 실행해서 9.2절까지 도달한 뒤에 이 셀을 이어서 실행하세요.
# ============================================================

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score

# --- 안전장치: 필요한 변수가 실제로 있는지 확인 ---
required_vars = ["test_blend_proba", "test_y_np", "CLASS_NAMES", "pred_stacked"]
missing = [v for v in required_vars if v not in dir()]
if missing:
    raise NameError(
        f"다음 변수가 메모리에 없습니다: {missing}\n"
        "노트북을 처음부터 실행해서 9.2절(ROC-AUC / Balanced Accuracy)까지 도달한 뒤 "
        "이 셀을 실행하세요."
    )

# --- Fuadah et al.(2024, ACS Omega)과 겹치는 6개 GHS 클래스 ---
OVERLAP_CLASSES = [
    "09_Acute_Toxicity_Oral",
    "10_Acute_Toxicity_Dermal",
    "11_Acute_Toxicity_Inhalation",
    "13_Skin_Corrosion_Irritation",
    "14_Eye_Damage_Irritation",
    "15_Sensitization",
]

# Fuadah et al. Table 5 (consensus model) 보고값 - 논문 Table 5 좌측 컬럼과 비교용
FUADAH_REFERENCE = {
    "09_Acute_Toxicity_Oral":       {"AUC": 0.87, "F1": 0.82, "CCR": 0.78},
    "10_Acute_Toxicity_Dermal":     {"AUC": 0.84, "F1": 0.79, "CCR": 0.78},
    "11_Acute_Toxicity_Inhalation": {"AUC": 0.83, "F1": 0.76, "CCR": 0.79},
    "13_Skin_Corrosion_Irritation": {"AUC": 0.90, "F1": 0.84, "CCR": 0.86},
    "14_Eye_Damage_Irritation":     {"AUC": 0.81, "F1": 0.75, "CCR": 0.74},
    "15_Sensitization":             {"AUC": 0.78, "F1": 0.71, "CCR": 0.72},
}

print(f"{'Class':<32}{'AUC':>8}{'F1':>8}{'BalAcc':>8}   (Fuadah AUC/F1/CCR)")
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
    ref = FUADAH_REFERENCE[cls_name]
    print(f"{cls_name:<32}{auc:>8.3f}{f1:>8.3f}{bal_acc:>8.3f}   "
          f"({ref['AUC']:.2f}/{ref['F1']:.2f}/{ref['CCR']:.2f})")

mean_auc = np.mean([v["AUC"] for v in results.values()])
mean_f1 = np.mean([v["F1"] for v in results.values()])
mean_ba = np.mean([v["BalAcc"] for v in results.values()])
print(f"\n{'Mean (6 endpoints)':<32}{mean_auc:>8.3f}{mean_f1:>8.3f}{mean_ba:>8.3f}")

# --- Table 5에 바로 붙여넣기 좋은 형태로 출력 ---
print("\n--- Table 5용 정리표 ---")
df = pd.DataFrame(results).T
df.columns = ["This work AUC", "This work F1", "This work Bal.Acc."]
df.index.name = "Class"
print(df.round(3).to_string())

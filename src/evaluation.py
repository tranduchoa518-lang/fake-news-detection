
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns


# Đánh giá mô hình
def evaluate_model(y_true, y_pred, model_name="Model"):

    acc = accuracy_score(y_true, y_pred)

    prec = precision_score(y_true, y_pred)

    rec = recall_score(y_true, y_pred)

    f1 = f1_score(y_true, y_pred)

    print(f"=== Kết quả: {model_name} ===")

    print(f"Accuracy:  {acc:.4f}")

    print(f"Precision: {prec:.4f}")

    print(f"Recall:    {rec:.4f}")

    print(f"F1-Score:  {f1:.4f}")

    print("\nMa trận nhầm lẫn:")

    print(confusion_matrix(y_true, y_pred))

    print("\nBáo cáo phân loại:")

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["Tin giả", "Tin thật"]
        )
    )

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }


# Vẽ ma trận nhầm lẫn
def plot_confusion_matrix(
    y_true,
    y_pred,
    model_name="Model",
    save_path=None
):

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=["Tin giả", "Tin thật"],
        yticklabels=["Tin giả", "Tin thật"]
    )

    plt.xlabel("Dự đoán")

    plt.ylabel("Thực tế")

    plt.title(f"Ma trận nhầm lẫn - {model_name}")

    if save_path:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches='tight'
        )

    plt.show()


# So sánh hiệu suất các mô hình
def plot_comparison(results_dict, save_path=None):

    models = list(results_dict.keys())

    metrics = {
        'Accuracy': [
            results_dict[m]['accuracy']
            for m in models
        ],

        'Precision': [
            results_dict[m]['precision']
            for m in models
        ],

        'Recall': [
            results_dict[m]['recall']
            for m in models
        ],

        'F1-Score': [
            results_dict[m]['f1']
            for m in models
        ]
    }

    x = np.arange(len(models))

    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (metric, values) in enumerate(metrics.items()):

        offset = width * (i - 1.5)

        rects = ax.bar(
            x + offset,
            values,
            width,
            label=metric
        )

        ax.bar_label(rects, fmt='%.3f', padding=3)

    ax.set_ylabel('Điểm số')

    ax.set_title('So sánh hiệu suất các mô hình')

    ax.set_xticks(x + width / 2)

    ax.set_xticklabels(models, rotation=15)

    ax.legend(loc='lower right', ncol=4)

    ax.set_ylim(0.8, 1.0)

    if save_path:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches='tight'
        )

    plt.show()

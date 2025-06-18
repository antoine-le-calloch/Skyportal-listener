import base64
import os
import matplotlib.pyplot as plt

def plot_probs(probs_dict, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    classes = list(probs_dict.keys())
    probs = list(probs_dict.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(classes, probs, color='skyblue')
    plt.ylabel("Probability")
    plt.title("Classification probabilities")
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)

    for bar, prob in zip(bars, probs):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, f"{prob:.2%}",
                 ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def store_result(obj_id, spectra_id, ml_result, log_path):
    plot_probs(ml_result, save_path=f"ml_results/{obj_id}_{spectra_id}_ML_probs.png")

    best_result = max(ml_result, key=ml_result.get)
    best_score = ml_result[best_result]
    with open(log_path, "a") as log_file:
        log_file.write(f"Object ID: {obj_id}\n")
        if spectra_id:
            log_file.write(f"Spectra ID: {spectra_id}\n")
        log_file.write(f"Best classification: {best_result}\n")
        log_file.write(f"Score: {best_score:.4f}\n")
        log_file.write("-" * 40 + "\n")


def post_result(client, obj_id, ml_result, attach_path=None):
    best_result = max(ml_result, key=ml_result.get)
    data = {
        "text": "Machine Learning Classification using spectra:\n\n"
                f"Best result: '{best_result}' with probability {ml_result[best_result]:.2%}\n\n"
    }

    if attach_path:
        plot_probs(ml_result, attach_path)
        with open(attach_path, "rb") as img_file:
            at_str = base64.b64encode(img_file.read()).decode('utf-8')

        data['attachment'] = {'body': at_str, 'name': os.path.basename(attach_path)}

    status, data = client.api('POST', f"/api/sources/{obj_id}/comments", data=data)

    if status != 200:
        raise ValueError(f"Error posting comment: {data}")
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


def store_result(client, obj_id, spectra_id, ml_result, log_path):
    plot_probs(ml_result, save_path=f"ml_results/{obj_id}_{spectra_id}_ML_probs.png")

    status, data = client.api('GET', f"/api/sources/{obj_id}")
    if status != 200:
        raise ValueError(f"Error fetching source: {data}")
    if not data.get('data'):
        raise ValueError(f"Source {obj_id} not found")

    skyportal_classifications = ""
    for c in data['data'].get('classifications', []):
        skyportal_classifications += f"{c['classification']} (prob={c['probability']:.3%}) - "

    tns_name = data['data'].get('tns_name', 'N/A')

    best_result = max(ml_result, key=ml_result.get)
    best_score = ml_result[best_result]
    with open(log_path, "a") as log_file:
        log_file.write(f"Object ID: {obj_id}\n")
        if spectra_id:
            log_file.write(f"Spectra ID: {spectra_id}\n")
        log_file.write(f"TNS name: {tns_name}\n")
        log_file.write(f"SkyPortal classifications: {skyportal_classifications}\n")
        log_file.write(f"Apple-cider classification: {best_result} (prob={best_score:.3%})\n")
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


def process_result(client, obj_id, spectra_id, ml_result, publish_to_skyportal):
    if ml_result and publish_to_skyportal:
        post_result(client, obj_id, ml_result, attach_path=f"ml_results/{obj_id}_{spectra_id}_ML_probs.png")
    else:
        store_result(client, obj_id, spectra_id, ml_result, log_path='ml_results.log')
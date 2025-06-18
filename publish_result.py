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


def post_comment(client, obj_id, ml_resultat, attach_path=None):
    data = {
        "text": "Machine Learning Classification using spectra",
    }

    if attach_path:
        plot_probs(ml_resultat, attach_path)
        with open(attach_path, "rb") as img_file:
            at_str = base64.b64encode(img_file.read()).decode('utf-8')

        data['attachment'] = {'body': at_str, 'name': os.path.basename(attach_path)}

    status, data = client.api('POST', f"/api/sources/{obj_id}/comments", data=data)

    if status != 200:
        print(f"Error posting comment: {data}")
    else:
        print(f"Comment posted successfully for object {obj_id}")
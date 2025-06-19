import re
from collections import defaultdict
import matplotlib.pyplot as plt

def extract_best_skyportal_class(text):
    if "duplicate" in text.lower():
        return None, []

    matches = re.findall(r'-?([A-Za-z0-9]+)[^ ]*\s+\(prob=(\d+\.\d+)%\)', text)
    if not matches:
        return None, []
    matches = [("Tidal Disruption Event" if "Event" in cls else cls, float(prob)) for cls, prob in matches]
    best_class = max(matches, key=lambda x: x[1])
    return best_class[0], matches

def parse_log_file(filepath):
    with open(filepath, "r") as f:
        content = f.read()

    entries = content.strip().split('----------------------------------------')
    data = []

    for entry in entries:
        if not entry.strip():
            continue

        object_id = re.search(r'Object ID:\s+(\S+)', entry)
        spectra_id = re.search(r'Spectra ID:\s+(\d+)', entry)
        skyportal_block = re.search(r'SkyPortal classifications:\s+(.+?)Apple-cider classification:', entry, re.DOTALL)
        apple_class = re.search(r'Apple-cider classification:\s+(\w+)', entry)

        if object_id and spectra_id and skyportal_block and apple_class:
            best_sky_class, _ = extract_best_skyportal_class(skyportal_block.group(1))
            if best_sky_class is None:
                continue

            data.append({
                "object_id": object_id.group(1),
                "spectra_id": int(spectra_id.group(1)),
                "skyportal_class": best_sky_class,
                "apple_class": apple_class.group(1)
            })
    return data

def compute_class_accuracy(data):
    class_stats = defaultdict(lambda: {"match": 0, "total": 0, "apple_classes": set()})
    for entry in data:
        sky = entry["skyportal_class"]
        apple = entry["apple_class"]

        class_stats[sky]["total"] += 1
        if sky == apple:
            class_stats[sky]["match"] += 1
        class_stats[sky]["apple_classes"].add(apple)
    return class_stats

def plot_class_comparison(class_stats):
    sky_classes = sorted(class_stats.keys())
    match_rates = [
        100 * class_stats[c]["match"] / class_stats[c]["total"]
        for c in sky_classes
    ]

    apple_labels = [
        ", ".join(sorted(class_stats[c]["apple_classes"]))
        for c in sky_classes
    ]

    sample_counts = [class_stats[c]["total"] for c in sky_classes]

    # Format top-axis labels like: "Ia (45)"
    skyportal_labels = [f"{cls} ({count})" for cls, count in zip(sky_classes, sample_counts)]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    bars = ax1.bar(sky_classes, match_rates, color="cornflowerblue")
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("Apple-cider Accuracy (%)")
    ax1.set_title("Apple-cider Accuracy per SkyPortal Class")

    # Top X axis (SkyPortal classes + sample counts)
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(range(len(sky_classes)))
    ax2.set_xticklabels(skyportal_labels, rotation=45, ha="left")
    ax2.set_xlabel("SkyPortal classifications (n = sample count)")

    # Bottom X labels = Apple-cider classes observed
    ax1.set_xticks(range(len(sky_classes)))
    ax1.set_xticklabels(apple_labels, rotation=45, ha="right")
    ax1.set_xlabel("Apple-cider classifications (per SkyPortal classification)")

    # Add percentage above each bar
    for bar, acc in zip(bars, match_rates):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                 f"{acc:.1f}%", ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    log_path = "ml_results.log"
    data = parse_log_file(log_path)
    class_stats = compute_class_accuracy(data)
    plot_class_comparison(class_stats)

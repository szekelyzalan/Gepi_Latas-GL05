import os
import matplotlib.pyplot as plt
from ultralytics import YOLO

def evaluate_and_compare_models(models_dict, dataset_yaml):
    """
    Evaluates multiple YOLO models on the test dataset, extracts mAP and inference speed,
    and generates comparison bar charts.
    
    Args:
        models_dict (dict): A dictionary where keys are model names and values are paths to .pt files.
        dataset_yaml (str): Path to the dataset.yaml file.
    """
    
    # Dictionaries to store the extracted metrics for each model
    map_scores = {}
    inference_speeds = {}

    # Iterate through the provided models
    for model_name, model_path in models_dict.items():
        print(f"\n{'='*50}")
        print(f"🚀 Starting evaluation for: {model_name}")
        print(f"{'='*50}")

        # Check if the weight file actually exists to prevent crashes
        if not os.path.exists(model_path):
            print(f"⚠️ WARNING: The file '{model_path}' was not found. Skipping {model_name}.")
            continue

        # Load the trained model weights
        model = YOLO(model_path)

        # Run validation on the test split
        # split="test" ensures we are evaluating on unseen data
        metrics = model.val(data=dataset_yaml, split="test")

        # Extract mAP (mean Average Precision @ IoU 0.5:0.95)
        # This is the primary metric for object detection accuracy
        current_map = metrics.box.map
        map_scores[model_name] = current_map
        
        # Extract inference speed (milliseconds per image)
        # This is crucial for your future "Live test" in the car
        current_speed = metrics.speed['inference']
        inference_speeds[model_name] = current_speed

        print(f"✅ {model_name} evaluated. mAP: {current_map:.4f}, Speed: {current_speed:.2f} ms/img")

    # If no models were successfully evaluated, exit the function
    if not map_scores:
        print("❌ No models were evaluated. Please check your file paths.")
        return

    # ==========================================
    # PLOTTING THE RESULTS
    # ==========================================
    
    model_names = list(map_scores.keys())
    map_values = list(map_scores.values())
    speed_values = list(inference_speeds.values())

    # Create a figure with 2 subplots (1 row, 2 columns)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Plot 1: mAP Comparison (Accuracy) ---
    # Higher is better
    bars1 = ax1.bar(model_names, map_values, color='#4C72B0', edgecolor='black')
    ax1.set_title('Model Accuracy Comparison (mAP 50-95)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('mAP Score (0.0 to 1.0)', fontsize=12)
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # Add text labels on top of the bars for exact values
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.3f}", ha='center', va='bottom', fontweight='bold')

    # --- Plot 2: Inference Speed Comparison ---
    # Lower is better (faster)
    bars2 = ax2.bar(model_names, speed_values, color='#DD8452', edgecolor='black')
    ax2.set_title('Inference Speed Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Milliseconds (ms) / Image', fontsize=12)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    # Add text labels on top of the bars
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f"{yval:.1f} ms", ha='center', va='bottom', fontweight='bold')

    # Adjust layout and display the charts
    plt.tight_layout()
    
    # Save the figure to the project folder
    plt.savefig('model_comparison_results.png', dpi=300)
    print("\n📊 Charts generated and saved as 'model_comparison_results.png'.")
    
    # Show the interactive plot window
    plt.show()

# ==========================================
# EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    
    # Add your trained models here. 
    # You can add as many as you want, the charts will adjust automatically.
    my_models = {
        "YOLOv8 Base (30 Epochs)": "/Users/gergo/Egyetem/MSC/1. félév/Gépi látás/best.pt",
        #"YOLOv8 Augmented": "runs/detect/train2/weights/best.pt",
        # "YOLOv8 Nano": "runs/detect/train3/weights/best.pt"  <-- Uncomment to add a 3rd model
    }

    # Path to your dataset configuration file
    dataset_file = "yolo_dataset/data.yaml"  # Update this if your yaml is in a different folder!

    # Run the evaluation pipeline
    evaluate_and_compare_models(my_models, dataset_file)
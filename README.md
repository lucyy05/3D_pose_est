# 🚀 3D Pose Estimation with Domain Adaptation
This project implements a 3D pose estimation pipeline, designed for effective training on synthetic data and generalization to real-world images (Sim-to-Real).

---
(For more details: https://www.notion.so/AIR-Lab-CV-2715bca3a2fb80bc9fa0fe026fc5f45d)

## 🛠️ Project Setup
1.  **Create Virtual Environment (Optional but Recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use: .\venv\Scripts\activate
    ```

2.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Execution:** Once dependencies are installed, you can run the main script.

    ```bash
    python test_posenet.py
    ```

-----

## 💻 Code Flow

The pipeline executes the following high-level steps in sequence during training, managed by the `main()` function:

```text
Raw Images + 3D Labels (.txt)
        |
        v
EnhancedPoseDataset (+ Augmentations)
        |
        v
DataLoader (Batches)
        |
        v
InceptionV3PoseNet (Encoder + Regressor)
        |
        v
Predicted 3D Coordinates
        |
        +--> Pose Loss (MSELoss)
        |
        +--> [Optional: Domain Adaptation Loss
        |       (Synthetic vs Real alignment)]
        |
        v
Backpropagation (L_total = L_pose - 0.1 * L_domain)
        |
        +--> Gradient Clipping (Stabilizes training)
        |
        v
NAdam Optimizer Step (Updates model weights)
        |
        v
Validation (Median Distance Error)
        |
        v
Save Best Model (Based on Validation Error)
        |
        v
Visualize 3D Path
```

-----

## ✅ Training Results

The results log below show the model training for 20 epochs. The performance is tracked using **Pose Loss** (training loss), **Domain Loss** (adversarial loss), and **Val Error** (median distance error on the validation set, measured in meters).

| Metric | Target Value | Best Achieved Value |
| :--- | :--- | :--- |
| **Best Val Error** | **<1m** | $\mathbf{0.0537m}$ |


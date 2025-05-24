## <span style="color:#2E86C1">Understanding the `main()` Function in The U-Net Segmentation Pipeline</span>

This is a REDME file that explains the structure and output of the `main()` function used in this tomato segmentation project. The function loads data, trains a U-Net model, evaluates it, and saves results.

---

## <span style="color:#117A65">1. Device Setup</span>

```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

* Chooses **GPU** if available; otherwise uses **CPU**.
* Helps ensure the training runs on the fastest device available.

---

## <span style="color:#117A65">2. Data Transformations</span>

```python
transform = get_transforms()
```

* Loads image preprocessing transformations (resizing, normalization, etc.).
* Applied to both input images and masks to ensure consistency.

---

## <span style="color:#117A65">3. Loading and Combining Datasets</span>

### <span style="color:#148F77">Training Data</span>

```python
train_dataset1 = TomatoDataset(...)
train_dataset2 = TomatoDataset(...)
```

* Loads training images and masks from `Train` and `Train2` folders.

```python
full_train_dataset = ConcatDataset([...])
```

* Combines both training datasets.

### <span style="color:#148F77">Testing Data</span>

```python
test_dataset1 = TomatoDataset(...)
test_dataset2 = TomatoDataset(...)
test_dataset = ConcatDataset([...])
```

* Loads test data similarly, from `Test` and `Test2`.

---

## <span style="color:#117A65">4. Train/Validation Split</span>

```python
train_dataset, val_dataset = random_split(full_train_dataset, [80%, 20%])
```

* Randomly splits training data into training and validation sets (80/20).

---

## <span style="color:#117A65">5. Data Loaders</span>

```python
DataLoader(...)
```

* Wraps datasets with batch and shuffle logic.

| Loader         | Purpose          |
| -------------- | ---------------- |
| `train_loader` | Training         |
| `val_loader`   | Validation       |
| `test_loader`  | Final Evaluation |

---

## <span style="color:#117A65">6. Sample Visualization</span>

```python
plt.savefig("results/sample_data.png")
```

* Visualizes a sample input image and its mask.
* <span style="color:#C0392B">Saved at:</span> `results/sample_data.png`

---

## <span style="color:#117A65">7. Model Initialization</span>

```python
model = UNet(in_channels=3, out_channels=1)
```

* Builds a U-Net for **binary segmentation** (1-channel output mask).
* Sends model to GPU or CPU using `.to(device)`.

---

## <span style="color:#117A65">8. Loss Function and Optimizer</span>

```python
criterion = nn.BCELoss()
optimizer = optim.Adam(...)
```

* Loss: Binary Cross Entropy.
* Optimizer: Adam with learning rate 0.001.

---

## <span style="color:#117A65">9. Training the Model</span>

```python
trained_model, history = train_model(...)
```

* Trains the model for `num_epochs = 20`.
* Stores training loss/metrics in a `history` object.

---

## <span style="color:#117A65">10. Training History Visualization</span>

```python
plot_training_history(history)
```

* Plots training/validation loss curves.
* Likely saved to `results/` folder (e.g., `results/loss_curve.png`).

---

## <span style="color:#117A65">11. Evaluation on Test Data</span>

```python
test_results = evaluate_model(...)
```

* Evaluates the trained model on the test set.
* Returns test loss and/or segmentation metrics (IoU, Dice, etc.).

---

## <span style="color:#117A65">12. Saving the Model</span>

```python
torch.save(..., "results/tomato_unet_model.pth")
```

* Saves model weights to disk.

* <span style="color:#C0392B">Saved at:</span> `results/tomato_unet_model.pth`

---

## <span style="color:#117A65">13. Inference on a New Image</span>

```python
segment_new_image(...)
```

* Loads and segments one test image from the `Test` folder.
* Saves result to disk (usually inside `results/`, check the function body).

---

## <span style="color:#2C3E50"> Outputs Directory: `results/`</span>

| File                              | Description                             |
| --------------------------------- | --------------------------------------- |
| `sample_data.png`                 | Visual of a training image and its mask |
| `tomato_unet_model.pth`           | Trained model weights                   |
| `loss_curve.png` (or similar)     | Plot of training/validation loss        |
| `segmented_test.png` (or similar) | Predicted segmentation on a test image  |

---

## <span style="color:#2C3E50"> Summary</span>

The `main()` function:

* Prepares the data and transforms.
* Loads and visualizes training/test data.
* Initializes and trains a U-Net segmentation model.
* Evaluates and saves the model.
* Applies the model on a new image for inference.



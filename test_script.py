import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from tqdm import tqdm

torch.manual_seed(42)
np.random.seed(42)

#model architecture
class GlaucomaResNet(nn.Module):
    def __init__(self, pretrained=False):
        super(GlaucomaResNet, self).__init__()
        self.resnet = models.resnet50(pretrained=pretrained)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # Remove the final layer

        #new classifier
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.resnet(x)
        x = self.classifier(x)
        return x

class TestDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image

def main():
    try:
        from google.colab import drive
        in_colab = True
        # Mount Google Drive
        drive.mount('/content/drive')
        model_path = '/content/drive/MyDrive/glaucoma_model.pt'
        augmented_dir = '/content/drive/MyDrive/Augmented_Data'
        augmentation_info_path = os.path.join(augmented_dir, 'augmentation_info.csv')
        results_path = '/content/drive/MyDrive/prediction_results.csv'
    except:
        in_colab = False
        model_path = 'glaucoma_model.pt'
        augmented_dir = 'Augmented_Data'
        augmentation_info_path = os.path.join(augmented_dir, 'augmentation_info.csv')
        results_path = 'prediction_results.csv'

    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        return

    if not os.path.exists(augmentation_info_path):
        print(f"Error: Augmentation info file {augmentation_info_path} not found!")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model from {model_path}...")
    model = GlaucomaResNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    print(f"Loading augmentation info from {augmentation_info_path}...")
    augmentation_df = pd.read_csv(augmentation_info_path)

    #data for predictions
    print(f"Found {len(augmentation_df)} augmented images to predict on")

    #image transformation for evaluation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    #storing predictions
    predictions = []

    #process images in batches
    image_paths = [os.path.join(augmented_dir, filename) for filename in augmentation_df['filename']]
    test_dataset = TestDataset(image_paths, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    print("Processing augmented images and making predictions...")

    batch_idx = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            images = batch.to(device)

            outputs = model(images)
            pred_probs = outputs.cpu().numpy().flatten()

            start_idx = batch_idx * test_loader.batch_size
            end_idx = min(start_idx + test_loader.batch_size, len(augmentation_df))

            #sttore predictions for this batch
            for i, idx in enumerate(range(start_idx, end_idx)):
                filename = augmentation_df.iloc[idx]['filename']
                true_label = augmentation_df.iloc[idx]['label']
                pred_prob = pred_probs[i]
                pred_class = 1 if pred_prob >= 0.5 else 0

                predictions.append({
                    'filename': filename,
                    'true_label': true_label,
                    'predicted_label': pred_class,
                    'prediction_probability': float(pred_prob),
                    'correct': pred_class == true_label
                })

            batch_idx += 1

    predictions_df = pd.DataFrame(predictions)

    predictions_df.to_csv(results_path, index=False)
    print(f"Predictions saved to {results_path}")

    true_labels = predictions_df['true_label'].values
    pred_labels = predictions_df['predicted_label'].values
    pred_probs = predictions_df['prediction_probability'].values

    accuracy = (pred_labels == true_labels).mean()
    print(f"\nOverall Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels,
                               target_names=['Normal', 'Glaucoma']))

    cm = confusion_matrix(true_labels, pred_labels)
    print("\nConfusion Matrix:")
    print(cm)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Normal', 'Glaucoma'], rotation=45)
    plt.yticks(tick_marks, ['Normal', 'Glaucoma'])

    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if in_colab:
        plt.savefig('/content/drive/MyDrive/confusion_matrix.png')
    else:
        plt.savefig('confusion_matrix.png')

    fpr, tpr, _ = roc_curve(true_labels, pred_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    if in_colab:
        plt.savefig('/content/drive/MyDrive/roc_curve.png')
    else:
        plt.savefig('roc_curve.png')

    print(f"\nROC AUC: {roc_auc:.4f}")
    print("\nPlots saved: confusion_matrix.png and roc_curve.png")

if __name__ == "__main__":
    main()

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

class GlaucomaModel(nn.Module):
    def __init__(self, num_classes=1):
        super(GlaucomaModel, self).__init__()
        self.resnet = models.resnet50(pretrained=False)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  
        self.custom_layers = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.resnet(x)
        x = self.custom_layers(x)
        return x

class GlaucomaTestDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        self.images_dir = os.path.join(data_dir, "images")
        self.image_files = sorted(os.listdir(self.images_dir))
        self.labels = [1 if 'g' in img_file.lower() else 0 for img_file in self.image_files]
        self.image_paths = [os.path.join(self.images_dir, img_file) for img_file in self.image_files]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, self.image_paths[idx]

def parse_args():
    parser = argparse.ArgumentParser(description='Test the trained glaucoma detection model')
    parser.add_argument('--model', type=str, default='glaucoma_model.pt', help='Path to the trained model file')
    parser.add_argument('--test_dir', type=str, default='./refuge2/test', help='Path to the test directory')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--img_height', type=int, default=224, help='Image height for model input')
    parser.add_argument('--img_width', type=int, default=224, help='Image width for model input')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for prediction')
    return parser.parse_args()

def evaluate_model(model, dataloader, device):
    """
    Evaluate model performance on the test set
    """
    model.eval()
    all_labels = []
    all_predictions = []
    all_probabilities = []
    image_paths = []
    
    with torch.no_grad():
        for images, labels, paths in dataloader:
            images = images.to(device)

            outputs = model(images).squeeze()
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities > 0.5).float()

            all_labels.extend(labels.numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            image_paths.extend(paths)

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)
    fpr, tpr, _ = roc_curve(all_labels, all_probabilities)
    roc_auc = auc(fpr, tpr)
    cm = confusion_matrix(all_labels, all_predictions)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': roc_auc,
        'confusion_matrix': cm,
        'roc_curve': (fpr, tpr)
    }
    
    return metrics, all_labels, all_predictions, all_probabilities, image_paths

def save_results(metrics, output_dir):
    """
    Save metrics and plots to the output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC'],
        'Value': [
            metrics['accuracy'], 
            metrics['precision'], 
            metrics['recall'], 
            metrics['f1_score'], 
            metrics['auc']
        ]
    })
    metrics_df.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)
    print(f"Metrics saved to {os.path.join(output_dir, 'metrics.csv')}")

    plt.figure(figsize=(8, 6))
    cm = metrics['confusion_matrix']
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Normal', 'Glaucoma'])
    plt.yticks(tick_marks, ['Normal', 'Glaucoma'])

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), 
                    horizontalalignment='center', 
                    color='white' if cm[i, j] > thresh else 'black')
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    confusion_matrix_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(confusion_matrix_path)
    plt.close()
    print(f"Confusion matrix saved to {confusion_matrix_path}")

    plt.figure(figsize=(8, 6))
    fpr, tpr = metrics['roc_curve']
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {metrics["auc"]:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    roc_curve_path = os.path.join(output_dir, 'roc_curve.png')
    plt.savefig(roc_curve_path)
    plt.close()
    print(f"ROC curve saved to {roc_curve_path}")

def visualize_predictions(images_data, labels, predictions, probabilities, output_dir, num_samples=10):
    """
    Visualize some test examples with their predictions
    """
    os.makedirs(output_dir, exist_ok=True)

    if len(images_data) > num_samples:
        indices = np.random.choice(len(images_data), num_samples, replace=False)
    else:
        indices = range(len(images_data))
    
    plt.figure(figsize=(20, 10))
    for i, idx in enumerate(indices):
        img_path = images_data[idx]
        true_label = labels[idx]
        pred = predictions[idx]
        prob = probabilities[idx]

        img = Image.open(img_path)
        plt.subplot(2, 5, i + 1)
        plt.imshow(img)

        pred_label = "Glaucoma" if pred == 1 else "Normal"
        true_label_str = "Glaucoma" if true_label == 1 else "Normal"

        color = 'green' if pred == true_label else 'red'
        
        plt.title(f"Pred: {pred_label} ({prob:.2f})\nTrue: {true_label_str}", color=color)
        plt.axis('off')
    
    plt.tight_layout()
    sample_path = os.path.join(output_dir, 'prediction_samples.png')
    plt.savefig(sample_path)
    plt.close()
    print(f"Prediction samples saved to {sample_path}")

    results_df = pd.DataFrame({
        'Image': [os.path.basename(img_path) for img_path in images_data],
        'True Label': ['Glaucoma' if label == 1 else 'Normal' for label in labels],
        'Predicted Label': ['Glaucoma' if pred == 1 else 'Normal' for pred in predictions],
        'Probability': probabilities,
        'Correct': [pred == label for pred, label in zip(predictions, labels)]
    })
    results_path = os.path.join(output_dir, 'prediction_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Detailed prediction results saved to {results_path}")

def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((args.img_height, args.img_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print(f"Loading test data from {args.test_dir}")
    test_dataset = GlaucomaTestDataset(args.test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Loaded {len(test_dataset)} test samples")

    print(f"Loading model from {args.model}")
    model = GlaucomaModel().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    print(f"Model loaded successfully")

    print("Evaluating model performance")
    metrics, labels, predictions, probabilities, image_paths = evaluate_model(model, test_loader, device)
    print("\nTest Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")

    print(f"\nSaving results to {args.output_dir}")
    save_results(metrics, args.output_dir)
    print("Visualizing predictions")
    visualize_predictions(image_paths, labels, predictions, probabilities, args.output_dir)
    
    print("\nTesting complete!")
    print(f"All results have been saved to: {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main()

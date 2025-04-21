import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
from tqdm import tqdm


torch.manual_seed(42)
np.random.seed(42)

MODEL_PATH = '/content/drive/MyDrive/glaucoma_model.pt' 
TEST_DIR = '/content/drive/MyDrive/refuge2/test'  
RESULTS_DIR = '/content/drive/MyDrive/test_results'  
BATCH_SIZE = 16
IMAGE_SIZE = (224, 224)

print(f"Checking if test directory exists: {os.path.exists(TEST_DIR)}")
images_dir = os.path.join(TEST_DIR, 'images')
masks_dir = os.path.join(TEST_DIR, 'mask')
print(f"Checking if images directory exists: {os.path.exists(images_dir)}")
print(f"Checking if mask directory exists: {os.path.exists(masks_dir)}")


if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)
    print(f"Created directory: {RESULTS_DIR}")

class GlaucomaResNet(nn.Module):
    def __init__(self, pretrained=False):
        super(GlaucomaResNet, self).__init__()
        self.resnet = models.resnet50(pretrained=pretrained)
        in_features = self.resnet.fc.in_features
        
        self.resnet.fc = nn.Identity()  #remove the final layer
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


class GlaucomaDataset(Dataset):
    def __init__(self, images, labels, filenames, transform=None):
        self.images = images
        self.labels = labels
        self.filenames = filenames
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        filename = self.filenames[idx]

        if isinstance(image, np.ndarray):
            image = Image.fromarray((image * 255).astype(np.uint8))

        if self.transform:
            image = self.transform(image)

        return image, label, filename

def is_glaucoma(image_path, masks_dir):
    filename = os.path.basename(image_path)

    if filename.startswith('g'):
        return True  # Glaucoma image
    elif filename.startswith('n'):
        return False  # Normal image

    mask_files = [f for f in os.listdir(masks_dir) if filename in f]
    if not mask_files:
        print(f"Warning: No mask found for {filename}. Assuming unlabeled or normal.")
        return False  

    mask_path = os.path.join(masks_dir, mask_files[0])
    mask = Image.open(mask_path)
    mask_array = np.array(mask)

    return np.max(mask_array) > 0


def load_test_dataset(test_dir):
    images_dir = os.path.join(test_dir, 'images')
    masks_dir = os.path.join(test_dir, 'mask')
    
 
    if not os.path.exists(images_dir):
        print(f"Error: Images directory not found: {images_dir}")
        if os.path.exists(test_dir):
            print(f"Contents of test directory ({test_dir}):")
            for item in os.listdir(test_dir):
                print(f"  - {item}")
        return [], [], []
    
    if not os.path.exists(masks_dir):
        print(f"Warning: Mask directory not found: {masks_dir}")
        print("Will use filename patterns for labeling.")
    else:
        print(f"Using masks from {masks_dir}")
    
    print(f"Loading images from {images_dir}")
    
    images = []
    labels = []
    filenames = []
    
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    print(f"Found {len(image_files)} image files")
    
    if image_files:
        print("Sample image filenames:")
        for f in image_files[:5]:
            print(f"  - {f}")
    
    print(f"Loading test images")
    for filename in tqdm(image_files):
        img_path = os.path.join(images_dir, filename)
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize(IMAGE_SIZE)

            if os.path.exists(masks_dir):
                is_glaucoma_image = is_glaucoma(img_path, masks_dir)
            else:
               
                is_glaucoma_image = filename.startswith('g') or 'glaucoma' in filename.lower()
            
            img_array = np.array(img) / 255.0

            images.append(img_array)
            labels.append(1 if is_glaucoma_image else 0)
            filenames.append(filename)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    return np.array(images), np.array(labels), filenames

def test_model(model, test_loader):
    model.eval()
    all_preds = []
    all_probs = []
    all_true = []
    all_filenames = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    with torch.no_grad():
        for inputs, labels, filenames in tqdm(test_loader, desc="Testing model"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            probs = outputs.cpu().numpy().flatten()  #flatten to ensure 1D
            preds = (probs > 0.5).astype(int)
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_true.extend(labels.numpy())
            all_filenames.extend(filenames)
    
    return np.array(all_probs), np.array(all_preds), np.array(all_true), all_filenames

def plot_roc_curve(y_true, y_probs):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(RESULTS_DIR, 'roc_curve.png'))
    plt.show()
    
    return roc_auc

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Glaucoma'],
                yticklabels=['Normal', 'Glaucoma'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'))
    plt.show()
    
    return cm

def save_misclassified_examples(X_test, y_true, y_pred, filenames, max_examples=5):
    misclassified_indices = np.where(y_true != y_pred)[0]
    
    if len(misclassified_indices) == 0:
        print("No misclassified examples found!")
        return
    
    num_examples = min(max_examples, len(misclassified_indices))
    
    plt.figure(figsize=(15, 5 * num_examples // 3 + 5))
    for i in range(num_examples):
        idx = misclassified_indices[i]
        plt.subplot(num_examples // 3 + 1, 3, i + 1)
        plt.imshow(X_test[idx])
        plt.title(f"File: {filenames[idx]}\nTrue: {'Glaucoma' if y_true[idx] == 1 else 'Normal'}\n"
                  f"Pred: {'Glaucoma' if y_pred[idx] == 1 else 'Normal'}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'misclassified_examples.png'))
    plt.show()

def main():
    print("Loading test dataset...")
    X_test, y_test, test_filenames = load_test_dataset(TEST_DIR)
    
    if len(X_test) == 0:
        print("No test images found. Please check your test directory.")
        return
    
    print(f"Loaded {len(X_test)} test images")
    print(f"Glaucoma images: {np.sum(y_test == 1)}")
    print(f"Normal images: {np.sum(y_test == 0)}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    test_dataset = GlaucomaDataset(
        images=X_test,
        labels=y_test,
        filenames=test_filenames,
        transform=transform
    )
    
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GlaucomaResNet(pretrained=False)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    model = model.to(device)
    
    #test the model
    print("testing the model")
    all_probs, all_preds, all_true, all_filenames = test_model(model, test_loader)
    
    accuracy = accuracy_score(all_true, all_preds)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(all_true, all_preds, target_names=['Normal', 'Glaucoma']))
    
    print("\nPlotting ROC curve")
    roc_auc = plot_roc_curve(all_true, all_probs)
    print(f"AUC: {roc_auc:.4f}")
    
    print("\nPlotting confusion matrix")
    cm = plot_confusion_matrix(all_true, all_preds)
    
    print("\nShowing some misclassified examples")
    save_misclassified_examples(X_test, all_true, all_preds, test_filenames)
    
    if len(all_probs) == len(all_filenames):
        results_df = pd.DataFrame({
            'Filename': all_filenames,
            'True_Label': all_true,
            'Predicted_Label': all_preds,
            'Glaucoma_Probability': all_probs
        })
        
        results_path = os.path.join(RESULTS_DIR, 'test_results.csv')
        results_df.to_csv(results_path, index=False)
        print(f"\nTest results saved to {results_path}")
    else:
        print(f"Cannot create results CSV: length mismatch between filenames ({len(all_filenames)}) and probabilities ({len(all_probs)})")
    
    #summary statistics
    print("\n===== TEST SUMMARY =====")
    print(f"Total test images: {len(all_true)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {roc_auc:.4f}")
    print(f"True Positives: {cm[1][1]}")
    print(f"False Positives: {cm[0][1]}")
    print(f"True Negatives: {cm[0][0]}")
    print(f"False Negatives: {cm[1][0]}")
    
    sensitivity = cm[1][1] / (cm[1][1] + cm[1][0]) if (cm[1][1] + cm[1][0]) > 0 else 0
    specificity = cm[0][0] / (cm[0][0] + cm[0][1]) if (cm[0][0] + cm[0][1]) > 0 else 0
    
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")

if __name__ == "__main__":
    main()

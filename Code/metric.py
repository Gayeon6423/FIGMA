import os
import json
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from lpips import LPIPS
import clip
from transformers import CLIPProcessor, CLIPModel
# from tifa import TIFA
from torchmetrics.image.fid import FrechetInceptionDistance

# LPIPS
def load_image_lpips(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)

def calculate_lpips(imgs1, imgs2):
    lpips_model = LPIPS(net='alex').cuda()
    with torch.no_grad():
        lpips_values = lpips_model(imgs1.cuda(), imgs2.cuda())
    return lpips_values.mean().item()

def average_lpips(answer_folder, figma_image_folder):
    lpips_scores = []
    for filename in os.listdir(answer_folder):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            answer_img_path = os.path.join(answer_folder, filename)
            stable_img_path = os.path.join(figma_image_folder, filename)

            if os.path.exists(stable_img_path):
                img1 = load_image_lpips(answer_img_path)
                img2 = load_image_lpips(stable_img_path)
                lpips_score = calculate_lpips(img1, img2)
                lpips_scores.append(lpips_score)

    return sum(lpips_scores) / len(lpips_scores) if lpips_scores else 0

# FID
def load_image_fid(image_path):
    image = default_loader(image_path)  # PIL 이미지 로드
    transform = transforms.Compose([
        transforms.Resize((299, 299)),  # Inception 모델 입력 크기
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)  # 배치 차원 추가

def calculate_fid(answer_folder, figma_image_folder):
    fid = FrechetInceptionDistance(normalize=True).cuda()

    imgs1, imgs2 = [], []

    for filename in os.listdir(answer_folder):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            answer_img_path = os.path.join(answer_folder, filename)
            stable_img_path = os.path.join(figma_image_folder, filename)

            if os.path.exists(stable_img_path):
                img1 = load_image_fid(answer_img_path).cuda()
                img2 = load_image_fid(stable_img_path).cuda()
                imgs1.append(img1)
                imgs2.append(img2)

    if not imgs1 or not imgs2:
        return float('inf')  # 이미지가 없으면 FID 점수를 무한대로 반환

    imgs1 = torch.cat(imgs1, dim=0)  # 배치로 결합
    imgs2 = torch.cat(imgs2, dim=0)

    fid.update(imgs1, real=True)
    fid.update(imgs2, real=False)

    return fid.compute().item()


# CLIP SIMILARITY
def load_image_clip_similarity(image_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, preprocess = clip.load("ViT-B/32", device=device)    
    image = Image.open(image_path).convert("RGB")
    return preprocess(image).unsqueeze(0).to(device)

def calculate_clip_similarity(img1, img2):
    with torch.no_grad():
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model, preprocess = clip.load("ViT-B/32", device=device)  
        image_features1 = model.encode_image(img1)
        image_features2 = model.encode_image(img2)

        image_features1 /= image_features1.norm(dim=-1, keepdim=True)
        image_features2 /= image_features2.norm(dim=-1, keepdim=True)

        similarity = (image_features1 @ image_features2.T).item()
    return similarity

def average_clip_similarity(answer_folder, figma_image_folder):
    similarities = []
    for filename in os.listdir(answer_folder):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            answer_img_path = os.path.join(answer_folder, filename)
            stable_img_path = os.path.join(figma_image_folder, filename)

            if os.path.exists(stable_img_path):
                img1 = load_image_clip_similarity(answer_img_path)
                img2 = load_image_clip_similarity(stable_img_path)
                similarity = calculate_clip_similarity(img1, img2)
                similarities.append(similarity)

    return sum(similarities) / len(similarities) if similarities else 0

# CLIP SCORE
def calculate_clip_score(image_path, text):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        similarity = torch.nn.functional.cosine_similarity(outputs.image_embeds, outputs.text_embeds)

    return similarity.item()

# TIFA 
# tifa_model = TIFA()
def calculate_tifa_score(image_path, prompt):
    try:
        image = Image.open(image_path).convert("RGB")
        score = tifa_model.score(image, prompt)
        return score
    except Exception as e:
        print(f"Error calculating TIFA score for {image_path}: {e}")
        return None

def process_tifa_scores(prompt_folder, figma_image_folder):
    tifa_scores = []
    for filename in os.listdir(prompt_folder):
        if filename.endswith('.json'):
            prompt_path = os.path.join(prompt_folder, filename)
            image_path = os.path.join(figma_image_folder, filename.replace('.json', '.jpg'))

            if os.path.exists(image_path):
                try:
                    with open(prompt_path, 'r') as f:
                        prompt_data = json.load(f)
                        prompt = prompt_data.get("summary", "")

                    if prompt:
                        tifa_score = calculate_tifa_score(image_path, prompt)
                        if tifa_score is not None:
                            tifa_scores.append(tifa_score)
                            print(f"File: {filename}, TIFA Score: {tifa_score}")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

    if tifa_scores:
        average_score = sum(tifa_scores) / len(tifa_scores)
        print(f"Average TIFA Score: {average_score}")
    else:
        print("No valid TIFA scores calculated")



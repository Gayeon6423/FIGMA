import os, random, warnings
import numpy as np
from PIL import Image
from tqdm import tqdm
from tqdm.auto import tqdm
import torch
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
from torchmetrics.image.fid import FrechetInceptionDistance
from lpips import LPIPS
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from functools import partial
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from scipy import linalg
warnings.filterwarnings('ignore')

# FID
def calculate_fid(answer_image_folder,answer_image_file,generate_image_folder,generate_image_file):
    from torchvision import transforms as T
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fid_metric = FrechetInceptionDistance(feature=64).to(device)
    fid_metric.reset()
    transform = T.PILToTensor()
    
    for filename in tqdm(answer_image_file):    
        real_image = Image.open(os.path.join(answer_image_folder, filename))
        generate_image = Image.open(os.path.join(generate_image_folder, filename))
        
        real_image_tensor = transform(real_image).unsqueeze(0).to(device)
        generate_image_tensor = transform(generate_image).unsqueeze(0).to(device)
        
        fid_metric.update(real_image_tensor, real=True)
        fid_metric.update(generate_image_tensor, real=False)
    
    fid_score = fid_metric.compute().item()
    return fid_score

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

# CLIP SIMILARITY
def load_image_clip_similarity(image_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    return inputs["pixel_values"].to(device)

def calculate_clip_similarity(img1, img2):
    with torch.no_grad():
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        image_features1 = model.get_image_features(pixel_values=img1)
        image_features2 = model.get_image_features(pixel_values=img2)

        image_features1 /= image_features1.norm(dim=-1, keepdim=True)
        image_features2 /= image_features2.norm(dim=-1, keepdim=True)

        similarity = (image_features1 @ image_features2.T).item()
    return similarity

def average_clip_similarity(answer_folder, generate_image_folder):
    similarities = []
    for filename in tqdm(os.listdir(answer_folder)):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            answer_img_path = os.path.join(answer_folder, filename)
            generate_img_path = os.path.join(generate_image_folder, filename)

            if os.path.exists(generate_img_path):
                img1 = load_image_clip_similarity(answer_img_path)
                img2 = load_image_clip_similarity(generate_img_path)
                similarity = calculate_clip_similarity(img1, img2)
                similarities.append(similarity)
    return sum(similarities) / len(similarities) if similarities else 0

# CLIP SCORE
def calculate_clip_score(image_path, text):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=[text], images=image, truncation=True,return_tensors="pt", padding=True) # truncation=Ture : limit token size 77

    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        similarity = torch.nn.functional.cosine_similarity(outputs.image_embeds, outputs.text_embeds)

    return similarity.item()


# Precision, Recall
INCEPTION_V3_URL = "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/classify_image_graph_def.pb"
INCEPTION_V3_PATH = "classify_image_graph_def.pb"
FID_POOL_NAME = "pool_3:0"
FID_SPATIAL_NAME = "mixed_6/conv:0"

# --- Inception 모델 관련 함수 ---
def _download_inception_model():
    if os.path.exists(INCEPTION_V3_PATH):
        return
    print("downloading InceptionV3 model...")
    import requests
    with requests.get(INCEPTION_V3_URL, stream=True) as r:
        r.raise_for_status()
        tmp_path = INCEPTION_V3_PATH + ".tmp"
        with open(tmp_path, "wb") as f:
            for chunk in tqdm(r.iter_content(chunk_size=8192)):
                f.write(chunk)
        os.rename(tmp_path, INCEPTION_V3_PATH)

def _create_feature_graph(input_batch):
    _download_inception_model()
    prefix = f"{random.randrange(2**32)}_{random.randrange(2**32)}"
    with open(INCEPTION_V3_PATH, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    pool3, spatial = tf.import_graph_def(
        graph_def,
        input_map={f"ExpandDims:0": input_batch},
        return_elements=[FID_POOL_NAME, FID_SPATIAL_NAME],
        name=prefix,
    )
    _update_shapes(pool3)
    spatial = spatial[..., :7]
    return pool3, spatial

def _create_softmax_graph(input_batch):
    _download_inception_model()
    prefix = f"{random.randrange(2**32)}_{random.randrange(2**32)}"
    with open(INCEPTION_V3_PATH, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    (matmul,) = tf.import_graph_def(
        graph_def, return_elements=[f"softmax/logits/MatMul"], name=prefix
    )
    w = matmul.inputs[1]
    logits = tf.matmul(input_batch, w)
    return tf.nn.softmax(logits)

def _update_shapes(pool3):
    for op in pool3.graph.get_operations():
        for o in op.outputs:
            shape = o.get_shape()
            if shape._dims is not None:
                new_shape = [None if (s == 1 and j == 0) else s for j, s in enumerate(shape)]
                o.__dict__["_shape_val"] = tf.TensorShape(new_shape)
    return pool3

# --- Pairwise 거리 계산 ---
def _batch_pairwise_distances(U, V):
    with tf.variable_scope("pairwise_dist_block"):
        norm_u = tf.reduce_sum(tf.square(U), 1)
        norm_v = tf.reduce_sum(tf.square(V), 1)
        norm_u = tf.reshape(norm_u, [-1, 1])
        norm_v = tf.reshape(norm_v, [1, -1])
        D = tf.maximum(norm_u - 2 * tf.matmul(U, V, False, True) + norm_v, 0.0)
    return D

def _numpy_partition(arr, kth, **kwargs):
    num_workers = min(cpu_count(), len(arr))
    chunk_size = len(arr) // num_workers
    extra = len(arr) % num_workers
    start_idx = 0
    batches = []
    for i in range(num_workers):
        size = chunk_size + (1 if i < extra else 0)
        batches.append(arr[start_idx: start_idx + size])
        start_idx += size
    with ThreadPool(num_workers) as pool:
        return list(pool.map(partial(np.partition, kth=kth, **kwargs), batches))

# --- FID Statics ---
class FIDStatistics:
    def __init__(self, mu: np.ndarray, sigma: np.ndarray):
        self.mu = mu
        self.sigma = sigma

    def frechet_distance(self, other, eps=1e-6):
        mu1, sigma1 = np.atleast_1d(self.mu), np.atleast_2d(self.sigma)
        mu2, sigma2 = np.atleast_1d(other.mu), np.atleast_2d(other.sigma)
        assert mu1.shape == mu2.shape, f"Different shapes: {mu1.shape}, {mu2.shape}"
        assert sigma1.shape == sigma2.shape, f"Different dimensions: {sigma1.shape}, {sigma2.shape}"
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            warnings.warn("fid calculation produces singular product; adding eps")
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1+offset).dot(sigma2+offset))
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                raise ValueError("Imaginary component in sqrt")
            covmean = covmean.real
        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)

# --- Evaluator Class  ---
class Evaluator:
    def __init__(self, session, batch_size=64, softmax_batch_size=512):
        self.sess = session
        self.batch_size = batch_size
        self.softmax_batch_size = softmax_batch_size
        self.manifold_estimator = ManifoldEstimator(session)
        with self.sess.graph.as_default():
            self.image_input = tf.placeholder(tf.float32, shape=[None, None, None, 3])
            self.softmax_input = tf.placeholder(tf.float32, shape=[None, 2048])
            self.pool_features, self.spatial_features = _create_feature_graph(self.image_input)
            self.softmax = _create_softmax_graph(self.softmax_input)

    def warmup(self):
        self.compute_activations(np.zeros([1, 8, 64, 64, 3]))

    def read_activations(self, npz_path: str) -> tuple:
        with np.load(npz_path) as obj:
            arr = obj["arr_0"]
            batches = [arr[i:i+self.batch_size] for i in range(0, arr.shape[0], self.batch_size)]
            return self.compute_activations(batches)

    def compute_activations(self, batches: iter) -> tuple:
        preds, spatial_preds = [], []
        for batch in tqdm(batches):
            batch = batch.astype(np.float32)
            pred, spatial_pred = self.sess.run(
                [self.pool_features, self.spatial_features],
                {self.image_input: batch}
            )
            preds.append(pred.reshape([pred.shape[0], -1]))
            spatial_preds.append(spatial_pred.reshape([spatial_pred.shape[0], -1]))
        return np.concatenate(preds, axis=0), np.concatenate(spatial_preds, axis=0)

    def compute_statistics(self, activations: np.ndarray) -> FIDStatistics:
        mu = np.mean(activations, axis=0)
        sigma = np.cov(activations, rowvar=False)
        return FIDStatistics(mu, sigma)

    def compute_inception_score(self, activations: np.ndarray, split_size: int = 5000) -> float:
        softmax_out = []
        for i in range(0, len(activations), self.softmax_batch_size):
            acts = activations[i:i+self.softmax_batch_size]
            softmax_out.append(self.sess.run(self.softmax, feed_dict={self.softmax_input: acts}))
        preds = np.concatenate(softmax_out, axis=0)
        scores = []
        for i in range(0, len(preds), split_size):
            part = preds[i:i+split_size]
            kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, axis=0), axis=0)))
            scores.append(np.exp(np.mean(np.sum(kl, axis=1))))
        return float(np.mean(scores))

    def compute_prec_recall(self, activations_ref: np.ndarray, activations_sample: np.ndarray) -> tuple:
        radii_1 = self.manifold_estimator.manifold_radii(activations_ref)
        radii_2 = self.manifold_estimator.manifold_radii(activations_sample)
        pr = self.manifold_estimator.evaluate_pr(activations_ref, radii_1, activations_sample, radii_2)
        return float(pr[0][0]), float(pr[1][0])

# --- Manifold Estimator & DistanceBlock ---
class ManifoldEstimator:
    def __init__(self, session, row_batch_size=10000, col_batch_size=10000, nhood_sizes=(3,), clamp_to_percentile=None, eps=1e-5):
        self.distance_block = DistanceBlock(session)
        self.row_batch_size = row_batch_size
        self.col_batch_size = col_batch_size
        self.nhood_sizes = nhood_sizes
        self.num_nhoods = len(nhood_sizes)
        self.clamp_to_percentile = clamp_to_percentile
        self.eps = eps

    def manifold_radii(self, features: np.ndarray) -> np.ndarray:
        num_images = len(features)
        radii = np.zeros([num_images, self.num_nhoods], dtype=np.float32)
        distance_batch = np.zeros([self.row_batch_size, num_images], dtype=np.float32)
        seq = np.arange(max(self.nhood_sizes) + 1, dtype=np.int32)
        for begin1 in range(0, num_images, self.row_batch_size):
            end1 = min(begin1 + self.row_batch_size, num_images)
            row_batch = features[begin1:end1]
            for begin2 in range(0, num_images, self.col_batch_size):
                end2 = min(begin2 + self.col_batch_size, num_images)
                col_batch = features[begin2:end2]
                distance_batch[0:end1-begin1, begin2:end2] = self.distance_block.pairwise_distances(row_batch, col_batch)
            radii[begin1:end1, :] = np.concatenate(
                [x[:, self.nhood_sizes] for x in _numpy_partition(distance_batch[0:end1-begin1, :], seq, axis=1)],
                axis=0,
            )
        if self.clamp_to_percentile is not None:
            max_distances = np.percentile(radii, self.clamp_to_percentile, axis=0)
            radii[radii > max_distances] = 0
        return radii

    def evaluate_pr(self, features_1: np.ndarray, radii_1: np.ndarray, features_2: np.ndarray, radii_2: np.ndarray) -> tuple:
        features_1_status = np.zeros([len(features_1), radii_2.shape[1]], dtype=bool)
        features_2_status = np.zeros([len(features_2), radii_1.shape[1]], dtype=bool)
        for begin_1 in range(0, len(features_1), self.row_batch_size):
            end_1 = begin_1 + self.row_batch_size
            batch_1 = features_1[begin_1:end_1]
            for begin_2 in range(0, len(features_2), self.col_batch_size):
                end_2 = begin_2 + self.col_batch_size
                batch_2 = features_2[begin_2:end_2]
                batch_1_in, batch_2_in = self.distance_block.less_thans(
                    batch_1, radii_1[begin_1:end_1], batch_2, radii_2[begin_2:end_2]
                )
                features_1_status[begin_1:end_1] |= batch_1_in
                features_2_status[begin_2:end_2] |= batch_2_in
        return (np.mean(features_2_status.astype(np.float64), axis=0),
                np.mean(features_1_status.astype(np.float64), axis=0))

class DistanceBlock:
    def __init__(self, session):
        self.session = session
        with session.graph.as_default():
            self._features_batch1 = tf.placeholder(tf.float32, shape=[None, None])
            self._features_batch2 = tf.placeholder(tf.float32, shape=[None, None])
            distance_block_16 = _batch_pairwise_distances(
                tf.cast(self._features_batch1, tf.float16),
                tf.cast(self._features_batch2, tf.float16),
            )
            self.distance_block = tf.cond(
                tf.reduce_all(tf.math.is_finite(distance_block_16)),
                lambda: tf.cast(distance_block_16, tf.float32),
                lambda: _batch_pairwise_distances(self._features_batch1, self._features_batch2),
            )
            self._radii1 = tf.placeholder(tf.float32, shape=[None, None])
            self._radii2 = tf.placeholder(tf.float32, shape=[None, None])
            dist32 = tf.cast(self.distance_block, tf.float32)[..., None]
            self._batch_1_in = tf.reduce_any(dist32 <= self._radii2, axis=1)
            self._batch_2_in = tf.reduce_any(dist32 <= self._radii1[:, None], axis=0)

    def pairwise_distances(self, U, V):
        return self.session.run(
            self.distance_block,
            feed_dict={self._features_batch1: U, self._features_batch2: V},
        )

    def less_thans(self, batch_1, radii_1, batch_2, radii_2):
        return self.session.run(
            [self._batch_1_in, self._batch_2_in],
            feed_dict={
                self._features_batch1: batch_1,
                self._features_batch2: batch_2,
                self._radii1: radii_1,
                self._radii2: radii_2,
            },
        )
        
def load_and_preprocess_image(img_path, target_size=(299, 299)):
    img = Image.open(img_path).convert('RGB')
    img = img.resize(target_size)
    return np.array(img).astype(np.float32)

def batch_generator(file_list, folder, batch_size=64, target_size=(299, 299)):
    images = []
    for f in file_list:
        img_path = os.path.join(folder, f)
        img = load_and_preprocess_image(img_path, target_size)
        images.append(img)
        if len(images) == batch_size:
            yield np.stack(images, axis=0)
            images = []
    if images:
        yield np.stack(images, axis=0)
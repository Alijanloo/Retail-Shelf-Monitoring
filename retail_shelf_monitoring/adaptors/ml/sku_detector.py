import glob
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import faiss
import numpy as np
from tqdm import tqdm

from ...frameworks.logging_config import get_logger
from ...usecases.interfaces.inference_model import InferenceModel

logger = get_logger(__name__)


class SKUDetector:
    """
    SKU detection using MobileNet embeddings and FAISS similarity search.

    This class handles both indexing (building the SKU database from reference images)
    and inference (identifying SKUs from cropped detections).
    """

    def __init__(
        self,
        inference_model: InferenceModel,
        index_path: Optional[str] = None,
        top_k: int = 1,
        use_gpu: bool = True,
        gpu_id: int = 0,
    ):
        """
        Initialize SKU detector with MobileNet embedding model.

        Args:
            inference_model: InferenceModel implementation for embedding extraction
            index_path: Path to FAISS index file. If None, must call build_index first
            top_k: Number of top similar SKUs to return
            use_gpu: Whether to use GPU for FAISS index (requires faiss-gpu)
            gpu_id: GPU device ID to use (default: 0)
        """
        self.inference_model = inference_model
        self.index_path = Path(index_path) if index_path else None
        self.top_k = top_k
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id

        # Initialize GPU resources if using GPU
        self.gpu_resources = None
        if self.use_gpu:
            try:
                self.gpu_resources = faiss.StandardGpuResources()
                logger.info(f"Initialized GPU resources for device {gpu_id}")
            except Exception as e:
                logger.warning(f"Failed to initialize GPU resources: {e}")
                logger.warning("Falling back to CPU")
                self.use_gpu = False

        # Get input shape from the inference model
        input_shape = self.inference_model.input_shape
        if len(input_shape) == 3:  # (C, H, W)
            self.input_height = input_shape[1]
            self.input_width = input_shape[2]
        elif len(input_shape) == 4:  # (B, C, H, W) - should not happen with property
            self.input_height = input_shape[2]
            self.input_width = input_shape[3]
        else:
            raise ValueError(f"Unexpected input shape: {input_shape}")

        logger.info("SKUDetector initialized successfully")
        logger.info(f"Input shape: {input_shape}")

        self.index: Optional[faiss.Index] = None
        self.sku_labels: Optional[np.ndarray] = None
        self.image_paths: Optional[list] = None

        if self.index_path and self.index_path.exists():
            self.load_index(str(self.index_path))

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for MobileNet inference.

        Applies ImageNet normalization
        as per standard PyTorch transforms.

        Args:
            image: Input image in BGR format (from cropped detection)

        Returns:
            Preprocessed image tensor ready for inference
        """
        resized = cv2.resize(
            image, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR
        )

        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        tensor = rgb_image.astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        tensor = (tensor - mean) / std

        tensor = tensor.transpose(2, 0, 1)
        tensor = np.expand_dims(tensor, axis=0)

        return tensor

    def extract_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        Extract embedding vector from image using MobileNet.

        Args:
            image: Input image in BGR format

        Returns:
            Normalized embedding vector (L2 normalized)
        """
        tensor = self.preprocess_image(image)

        embedding = self.inference_model.infer(tensor)

        # Ensure embedding is 2D for normalization
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        # L2 normalize the embedding
        embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)

        embedding_flat = embedding.flatten()

        return embedding_flat

    def build_index(
        self, data_dir: str, index_output_path: Optional[str] = None
    ) -> Tuple[int, int]:
        """
        Build FAISS index from SKU reference images.

        Walks through directory structure where each subdirectory
        represents a SKU class:
        data_dir/
            0/
                image1.jpg
                image2.jpg
            1/
                image1.jpg
            ...

        Args:
            data_dir: Root directory containing SKU class folders
                index_output_path: Where to save the FAISS index. If None,
                uses self.index_path

        Returns:
            Tuple of (number of images indexed, number of unique SKU classes)
        """
        root_pattern = str(Path(data_dir) / "**" / "*.jpg")
        self.image_paths = sorted(glob.glob(root_pattern, recursive=True))

        if not self.image_paths:
            raise ValueError(f"No .jpg images found in {data_dir}")

        logger.info(f"Found {len(self.image_paths)} images for indexing")

        # embeddings = []
        labels = []
        images = []

        for img_path in tqdm(self.image_paths):
            try:
                image = cv2.imread(img_path)
                if image is None:
                    logger.warning(f"Could not read image: {img_path}")
                    continue

                images.append(image)

                label = int(Path(img_path).parent.name)
                labels.append(label)
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                continue

        embeddings = self.inference_model.batch_infer(images, batch_size=128)

        if not embeddings:
            raise ValueError("No valid embeddings extracted from images")

        embeddings_matrix = np.vstack(embeddings).astype("float32")
        self.sku_labels = np.array(labels, dtype=np.int32)

        embedding_dim = embeddings_matrix.shape[1]

        # Create GPU or CPU index based on configuration
        if self.use_gpu and self.gpu_resources is not None:
            try:
                # Create GPU index
                cpu_index = faiss.IndexFlatL2(embedding_dim)
                self.index = faiss.index_cpu_to_gpu(
                    self.gpu_resources, self.gpu_id, cpu_index
                )
                logger.info(f"Created GPU index on device {self.gpu_id}")
            except Exception as e:
                logger.warning(f"Failed to create GPU index: {e}")
                logger.warning("Falling back to CPU index")
                self.index = faiss.IndexFlatL2(embedding_dim)
        else:
            # Create CPU index
            self.index = faiss.IndexFlatL2(embedding_dim)
            logger.info("Created CPU index")

        self.index.add(embeddings_matrix)

        logger.info(
            f"Built FAISS index with {self.index.ntotal} vectors, "
            f"dimension {embedding_dim}"
        )

        output_path = index_output_path or self.index_path
        if output_path:
            self.save_index(str(output_path))

        unique_classes = len(np.unique(self.sku_labels))
        return len(embeddings), unique_classes

    def save_index(self, index_path: str):
        """
        Save FAISS index and metadata to disk.

        Args:
            index_path: Path to save the index (will also save labels as .npy)
        """
        if self.index is None:
            raise ValueError("No index to save. Call build_index first.")

        index_file = Path(index_path)
        index_file.parent.mkdir(parents=True, exist_ok=True)

        # Convert GPU index to CPU for saving
        if self.use_gpu and hasattr(self.index, "index"):
            # GPU index - convert to CPU for saving
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, str(index_file))
            logger.info(f"Saved GPU index (converted to CPU) to {index_file}")
        else:
            # CPU index - save directly
            faiss.write_index(self.index, str(index_file))
            logger.info(f"Saved CPU index to {index_file}")

        labels_file = index_file.with_suffix(".labels.npy")
        np.save(str(labels_file), self.sku_labels)
        logger.info(f"Saved labels to {labels_file}")

        if self.image_paths:
            paths_file = index_file.with_suffix(".paths.npy")
            np.save(str(paths_file), np.array(self.image_paths, dtype=object))
            logger.info(f"Saved image paths to {paths_file}")

    def load_index(self, index_path: str):
        """
        Load FAISS index and metadata from disk.

        Args:
            index_path: Path to the FAISS index file
        """
        index_file = Path(index_path)
        if not index_file.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")

        # Load CPU index from disk
        cpu_index = faiss.read_index(str(index_file))

        # Convert to GPU if requested and available
        if self.use_gpu and self.gpu_resources is not None:
            try:
                self.index = faiss.index_cpu_to_gpu(
                    self.gpu_resources, self.gpu_id, cpu_index
                )
                logger.info(
                    f"Loaded FAISS index to GPU {self.gpu_id} "
                    f"({self.index.ntotal} vectors)"
                )
            except Exception as e:
                logger.warning(f"Failed to load index to GPU: {e}")
                logger.warning("Using CPU index")
                self.index = cpu_index
        else:
            self.index = cpu_index
            logger.info(
                f"Loaded FAISS index from {index_file} ({self.index.ntotal} vectors)"
            )

        labels_file = index_file.with_suffix(".labels.npy")
        if labels_file.exists():
            self.sku_labels = np.load(str(labels_file))
            logger.info(f"Loaded {len(self.sku_labels)} labels")
        else:
            logger.warning(f"Labels file not found: {labels_file}")

        paths_file = index_file.with_suffix(".paths.npy")
        if paths_file.exists():
            self.image_paths = np.load(str(paths_file), allow_pickle=True).tolist()
            logger.info(f"Loaded {len(self.image_paths)} image paths")

        self.index_path = index_file

    def identify_sku(
        self, cropped_image: np.ndarray
    ) -> Tuple[int, float, Optional[list]]:
        """
        Identify SKU from cropped detection image.

        Args:
            cropped_image: Cropped object image in BGR format

        Returns:
            Tuple of (predicted SKU label, confidence score, top-k results)
            top-k results is list of tuples: [(sku_label, distance), ...]
        """
        if self.index is None:
            raise ValueError("No index loaded. Call load_index or build_index first.")

        embedding = self.extract_embedding(cropped_image)
        embedding_query = embedding.reshape(1, -1)

        distances, indices = self.index.search(embedding_query, self.top_k)

        top_results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.sku_labels):
                sku_label = int(self.sku_labels[idx])
                top_results.append((sku_label, float(dist)))

        if not top_results:
            logger.warning("No valid results from FAISS search")
            return -1, 0.0, []

        best_sku, best_distance = top_results[0]

        confidence = 1.0 / (1.0 + best_distance)

        return best_sku, confidence, top_results

    def get_sku_id(self, cropped_image: np.ndarray) -> str:
        """
        Simplified interface: Get SKU ID string from cropped image.

        Args:
            cropped_image: Cropped object image in BGR format

        Returns:
            SKU ID as string (e.g., "sku_123" or "unknown_sku")
        """
        try:
            sku_label, confidence, _ = self.identify_sku(cropped_image)
            if sku_label >= 0:
                return f"sku_{sku_label}"
            return "unknown_sku"
        except Exception as e:
            logger.error(f"Error identifying SKU: {e}")
            return "error_sku"

    def batch_identify_skus(self, cropped_images: List[np.ndarray]) -> List[str]:
        """
        Identify SKUs for a batch of cropped images using batch inference and
        GPU batch search.

        Args:
            cropped_images: List of cropped detection images in BGR format

        Returns:
            List of SKU ID strings corresponding to input images
        """
        if not cropped_images:
            return []

        if self.index is None:
            logger.warning("No index loaded for batch identification")
            return ["no_index"] * len(cropped_images)

        try:
            preprocessed_images = []
            for img in cropped_images:
                preprocessed = self.preprocess_image(img)
                if preprocessed.ndim == 4 and preprocessed.shape[0] == 1:
                    preprocessed = preprocessed.squeeze(0)
                preprocessed_images.append(preprocessed)

            embeddings = self.inference_model.batch_infer(
                preprocessed_images, batch_size=len(preprocessed_images)
            )

            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)

            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

            distances, indices = self.index.search(embeddings.astype(np.float32), 1)

            sku_ids = []
            for i in range(len(embeddings)):
                if (
                    len(indices[i]) > 0
                    and indices[i][0] >= 0
                    and indices[i][0] < len(self.sku_labels)
                ):
                    sku_label = int(self.sku_labels[indices[i][0]])
                    sku_ids.append(f"sku_{sku_label}")
                else:
                    sku_ids.append("unknown_sku")

            logger.info(
                f"Batch identified {len(sku_ids)} SKUs using "
                f"{'GPU' if self.use_gpu else 'CPU'} index"
            )
            return sku_ids

        except Exception as e:
            logger.error(f"Error in batch SKU identification: {e}")
            return ["error_sku"] * len(cropped_images)

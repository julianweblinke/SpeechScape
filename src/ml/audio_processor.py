import os
import torch
import numpy as np
from torch import nn
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForPreTraining
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2GumbelVectorQuantizer
import soundfile as sf
from utils import get_logger
import torchaudio
from huggingface_hub import snapshot_download

# Create module-specific logger
logger = get_logger(__name__)  # Will be 'src.ml.ml_pipeline'

class CustomQuantizer(Wav2Vec2GumbelVectorQuantizer):
    """
    Custom quantizer that extends Wav2Vec2GumbelVectorQuantizer to return codebook indices.
    
    This class modifies the forward pass to return the actual codebook indices used
    for quantization, which are needed for downstream analysis.
    """
    
    def __init__(self, config):
        """
        Initialize the custom quantizer with Wav2Vec2 configuration.
        
        Args:
            config: Wav2Vec2 configuration object containing quantization parameters.
        """
        super().__init__(config)  

    def forward(self, hidden_states, mask_time_indices=None):
        """
        Forward pass through the quantizer with codebook index extraction.
        
        Args:
            hidden_states (torch.Tensor): Input features of shape [batch_size, sequence_length, hidden_size]
            mask_time_indices (torch.Tensor, optional): Mask for time indices. Defaults to None.
        
        Returns:
            tuple: (codevectors, perplexity, codevector_idx_reshaped)
                - codevectors (torch.Tensor): Quantized vectors [batch_size, sequence_length, hidden_size]
                - perplexity (torch.Tensor): Perplexity of the quantization
                - codevector_idx_reshaped (torch.Tensor): Codebook indices [batch_size, sequence_length, num_groups]
        """
        batch_size, sequence_length, hidden_size = hidden_states.shape  # [1, T, 512]

        # Project to codevector dimension
        hidden_states = self.weight_proj(hidden_states)  # [1, T, 640]
        hidden_states = hidden_states.view(batch_size * sequence_length * self.num_groups, -1)  # [1*T*2, 320]

        # Compute hard codevector distribution (one-hot)
        codevector_idx = hidden_states.argmax(dim=-1)  # [1*T*2] t1_group1, t1_group2, t2_group1, t2_group2, ...
        codevector_probs = hidden_states.new_zeros(hidden_states.shape).scatter_(
            -1, codevector_idx.view(-1, 1), 1.0
        )  # [1*T*2, 320]
        codevector_probs = codevector_probs.view(batch_size * sequence_length, self.num_groups, -1)  # [T, 2, 320]

        perplexity = self._compute_perplexity(codevector_probs, mask_time_indices)

        codevector_probs = codevector_probs.view(batch_size * sequence_length, -1)  # [T, 640]

        # Use probabilities to retrieve codevectors
        codevectors_per_group = codevector_probs.unsqueeze(-1) * self.codevectors  # [T, 640, 384]
        codevectors = codevectors_per_group.view(
            batch_size * sequence_length, self.num_groups, self.num_vars, -1
        )  # [T, 2, 320, 384]
        codevectors = codevectors.sum(-2).view(batch_size, sequence_length, -1)  # [1, T, 768]

        codevector_idx_reshaped = codevector_idx.view(batch_size, sequence_length, self.num_groups) # [1, T, 2]
        # [
        #   [t1_g1, t1_g2],
        #   [t2_g1, t2_g2],
        #   [t3_g1, t3_g2]
        # ]

        return codevectors, perplexity, codevector_idx_reshaped

class AudioProcessor:
    """
    Processes audio files through Wav2Vec2 model to extract quantized codebook representations.
    
    This class handles the complete audio processing pipeline from raw audio files
    to discrete codebook indices that represent speech features.
    """
    
    def __init__(self, model_name: str = "facebook/wav2vec2-large-xlsr-53", 
                 sample_rate: int = 16000, cache_dir: str = None, 
                 skip_model_loading: bool = False):
        """
        Initialize the AudioProcessor with model configuration.
        
        Args:
            model_name (str): HuggingFace model identifier for Wav2Vec2 model.
            sample_rate (int): Target sample rate for audio processing.
            cache_dir (str): Directory to cache downloaded model files.
            skip_model_loading (bool): If True, delay model loading until initialize_model() is called.
        """
        logger.debug(f"... (AudioProcessor.__init__): construct audio_processor...")
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.cache_dir = cache_dir
        self.model = None
        self.feature_extractor = None
        
        if not skip_model_loading:
            self.initialize_model()
    
    def initialize_model(self):
        """
        Initialize the Wav2Vec2 model and feature extractor.
        
        Downloads model files if not cached, loads the pre-trained model,
        and sets up the feature extractor for audio preprocessing.
        """
        logger.debug(f"... (AudioProcessor.initialize_model): load model {self.model_name}...")
        self.load_model(self.model_name)
        self.model = Wav2Vec2ForPreTraining.from_pretrained(self.model_name, cache_dir=self.cache_dir, local_files_only=True)
        self.model.eval()
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name, cache_dir=self.cache_dir)

    def load_model(self, model_name: str):
        """
        Download model files from HuggingFace Hub if not already cached.
        
        Args:
            model_name (str): HuggingFace model identifier to download.
        """
        logger.debug(f"... (AudioProcessor.load_model): download model files")
        snapshot_download(repo_id=model_name,
                          cache_dir=self.cache_dir,
                          allow_patterns=["*.bin", "*.json"],
                          ignore_patterns=["*.safetensors"])

    def process(self, audio_file: str) -> tuple[torch.Tensor, dict]:
        """
        Process an audio file through the complete Wav2Vec2 pipeline.
        
        Args:
            audio_file (str): Path to the audio file to process.
        
        Returns:
            tuple: (codebook_indices_combined, file_info)
                - codebook_indices_combined (torch.Tensor): Combined codebook indices (0-102399)
                - file_info (dict): Dictionary containing:
                    - path (str): Original file path
                    - categoryA_id (str): Category A identifier extracted from filename
                    - categoryB_id (str): Category B identifier extracted from filename
                    - duration (float): Audio duration in seconds
        """
        # Ensure model is initialized
        if self.model is None or self.feature_extractor is None:
            raise RuntimeError("... (AudioProcessor.process): model not initialized. call initialize_model() first.")
            
        # Load with soundfile (original implementation)
        categoryA_id = self._extract_categoryA_id(audio_file)
        categoryB_id = self._extract_categoryB_id(audio_file)
        audio_input, sr = sf.read(audio_file)
        
        # Resample if needed (using torchaudio for better compatibility)
        logger.debug(f"... (AudioProcessor.process): read and resample if necessary...")
        if sr != self.sample_rate:
            audio_input = torchaudio.functional.resample(
                torch.from_numpy(audio_input).float(),
                orig_freq=sr,
                new_freq=self.sample_rate
            ).numpy()

        # Process through feature extractor
        logger.debug(f"... (AudioProcessor.process): extract features...")
        input_values = self.feature_extractor(
            audio_input,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True
        ).input_values
        #print(f"np.shape(input_values)={np.shape(input_values)}")

        with torch.no_grad():
            logger.debug(f"... (AudioProcessor.process): process features with wav2vec2...")
            outputs = self.model.wav2vec2(input_values, return_dict=False)
        #print(f"np.shape(outputs[0])={np.shape(outputs[0])}")
        #print(f"np.shape(outputs[1])={np.shape(outputs[1])}")

        logger.debug(f"... (AudioProcessor.process): get transformer features...")
        transformer_features = self.model.project_hid(outputs[0])
        #print(f"np.shape(transformer_features)={np.shape(transformer_features)}")

        extract_features = self.model.dropout_features(outputs[1])
        #print(f"np.shape(extract_features)={np.shape(extract_features)}")

        # RANDOM INIT:
        #quantizer = CustomQuantizer(self.model.config)
        #quantized_features, codevector_perplexity, codebook_indices = quantizer(extract_features, mask_time_indices=None)
        #print(codevector_perplexity, codebook_indices)

        # CUSTOM QUANTIZER:
        logger.debug(f"... (AudioProcessor.process): get quantized features...")
        quantizer = CustomQuantizer(self.model.config)
        quantizer.load_state_dict(self.model.quantizer.state_dict())
        quantized_features, codevector_perplexity, codebook_indices = quantizer(extract_features, mask_time_indices=None)
        # EQUIVALENT BUT CODEBOOK_INDICES ARE NOT RETURNED:
        # quantizer = self.model.quantizer
        # quantized_features, codevector_perplexity = quantizer(extract_features, mask_time_indices=None)
        # print(codevector_perplexity)
        # print(codevector_perplexity, codebook_indices)
        # print(codebook_indices.shape) # [1, T, 2]

        # CREATE 320²=102400 indices
        logger.debug(f"... (AudioProcessor.process): combine codebook entries...")
        codebook_indices_combined = (
            codebook_indices[0,:,0] * self.model.config.num_codevectors_per_group 
            + codebook_indices[0,:,1]
         ) # (0-319) × 320 +  (0-319)

        file_info = {
            "path": audio_file,
            "categoryA_id": categoryA_id,
            "categoryB_id": categoryB_id,
            "duration": len(audio_input) / sr
        }
        
        return codebook_indices_combined, file_info
    
    def _extract_categoryB_id(self, file_path: str) -> str:
        """
        Extract Category B identifier from audio filename.
        
        Args:
            file_path (str): Path to the audio file.
        
        Returns:
            str: Category B identifier (last part of filename before extension).
        """
        filename = os.path.basename(file_path)
        parts = filename.split(".")[0].split("_")
        return parts[-1]

    def _extract_categoryA_id(self, file_path: str) -> str:
        """
        Extract Category A identifier from audio filename.
        
        Args:
            file_path (str): Path to the audio file.
        
        Returns:
            str: Category A identifier (second to last part of filename before extension).
        """
        filename = os.path.basename(file_path)
        parts = filename.split(".")[0].split("_")
        return parts[-2]

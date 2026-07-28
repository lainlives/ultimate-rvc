#!/usr/bin/env python3
"""
Ultimate RVC Training Script

This is a static Python script with hardcoded variables for training voice conversion models.
It performs the complete training pipeline:
1. Preprocess/slice audio dataset
2. Extract features (pitch and embeddings)
3. Train the voice model

Usage:
    - Set the configuration variables in the CONFIGURATION section below
    - Ensure your audio files are in the specified dataset directory
    - Run: python train_voice_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from multiprocessing import cpu_count

# Add the src directory to the path to import ultimate_rvc modules
SRC_DIR = Path(__file__).parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# =============================================================================
# CONFIGURATION - EDIT THESE VARIABLES FOR YOUR TRAINING RUN
# =============================================================================

# --- Dataset Configuration ---
# Name for your voice model (will be used for model files and directories)
MODEL_NAME = "my_voice_model"

# Path to directory containing your raw audio files (WAV, MP3, FLAC, OGG, M4A, AAC)
# Each file should contain clean speech/vocals of the voice you want to clone
DATASET_PATH = "/path/to/your/audio/dataset"

# --- Preprocessing Configuration ---
# Target sample rate for training (16000, 32000, 40000, 44100, 48000, 96000, 192000)
SAMPLE_RATE = 40000  # Recommended: 40000 or 48000

# Audio normalization method: "none", "pre", or "post"
NORMALIZATION_MODE = "post"

# Whether to apply high-pass filter to remove low-frequency sounds
FILTER_AUDIO = True

# Whether to apply noise reduction cleaning
CLEAN_AUDIO = False

# Strength of noise reduction (0.0 to 1.0), only used if CLEAN_AUDIO=True
CLEAN_STRENGTH = 0.7

# Method for splitting audio files: "Skip", "Simple", or "Automatic"
# - "Skip": Don't split (use if already split into chunks)
# - "Simple": Split into fixed-length chunks
# - "Automatic": Detect silence and split around it
SPLIT_METHOD = "Automatic"

# Length of each audio chunk in seconds (used with "Simple" split method)
CHUNK_LENGTH = 3.0

# Overlap between chunks in seconds (used with "Simple" split method)
OVERLAP_LENGTH = 0.3

# Number of CPU cores to use for preprocessing (default: all available)
PREPROCESS_CPU_CORES = cpu_count()

# --- Feature Extraction Configuration ---
# F0 (pitch) extraction method: "rmvpe", "hpa-rmvpe", "crepe", "crepe-tiny", "fcpe"
# Recommended: "rmvpe" for best quality
F0_METHOD = "rmvpe"

# Embedder model for extracting audio embeddings
# Options: "contentvec", "spin", "spin-v2", "wavlm-plus-ft", "wavlm-base-plus",
#          "Deep_WavLM_ft", "wavLM-SPIN", "KLM-Vocal-X1",
#          "chinese-hubert-base", "japanese-hubert-base", "korean-hubert-base", "custom"
EMBEDDER_MODEL = "contentvec"

# Path to custom embedder model (only used if EMBEDDER_MODEL="custom")
CUSTOM_EMBEDDER_MODEL = None

# Number of mute/silent files to include in training (helps model handle silence)
INCLUDE_MUTES = 2

# Number of CPU cores to use for feature extraction
EXTRACT_CPU_CORES = cpu_count()

# Hardware acceleration: "Automatic", "CPU", or "GPU"
HARDWARE_ACCELERATION = "Automatic"

# GPU IDs to use (only used if HARDWARE_ACCELERATION="GPU")
# Example: {0} for first GPU, {0, 1} for first two GPUs
GPU_IDS = None  # Auto-select first available GPU

# --- Training Configuration ---
# Number of training epochs (more epochs = better quality but risk of overtraining)
# Typical range: 100-1000 depending on dataset size
NUM_EPOCHS = 500

# Batch size (should fit in your GPU VRAM)
# Lower values (4) = better quality but slower
# Higher values (8-16) = faster but may reduce quality
BATCH_SIZE = 8

# Whether to detect overtraining automatically
DETECT_OVERTRAINING = False

# Maximum epochs without improvement before stopping (if DETECT_OVERTRAINING=True)
OVERTRAINING_THRESHOLD = 50

# Vocoder for audio synthesis: "HiFi-GAN", "MRF HiFi-GAN", or "RefineGAN"
# RefineGAN provides highest quality but requires more resources
VOCODER = "HiFi-GAN"

# Index algorithm for voice conversion: "Auto", "Faiss", or "KMeans"
# KMeans is better for large datasets
INDEX_ALGORITHM = "Auto"

# Pretrained model type: "None", "Default", or "Custom"
# "Default" uses pretrained models for better results
# "None" trains from scratch
PRETRAINED_TYPE = "Default"

# Name of custom pretrained model (only used if PRETRAINED_TYPE="Custom")
CUSTOM_PRETRAINED = None

# Epoch interval for saving checkpoints and weights
SAVE_INTERVAL = 10

# Whether to save all checkpoints (not just the latest)
SAVE_ALL_CHECKPOINTS = False

# Whether to save all weight versions (not just the best)
SAVE_ALL_WEIGHTS = False

# Whether to clear existing training data before starting
CLEAR_SAVED_DATA = False

# Whether to upload model after training (for use in Ultimate RVC app)
UPLOAD_MODEL = False

# Name for uploaded model (only used if UPLOAD_MODEL=True)
UPLOAD_NAME = None

# Precision type: "fp32", "fp16", or "bf16"
# fp16/bf16 can speed up training and reduce VRAM usage on supported hardware
PRECISION = "fp32"

# Whether to preload entire dataset into GPU memory (requires lots of VRAM)
PRELOAD_DATASET = False

# Whether to reduce VRAM usage via activation checkpointing (slower but uses less VRAM)
REDUCE_MEMORY_USAGE = False

# =============================================================================
# END OF CONFIGURATION - DO NOT EDIT BELOW THIS LINE UNLESS YOU KNOW WHAT YOU'RE DOING
# =============================================================================


def convert_to_enum(value: str, enum_class) -> any:
    """Convert a string value to the corresponding enum member."""
    try:
        return enum_class(value)
    except ValueError:
        valid_values = [e.value for e in enum_class]
        raise ValueError(
            f"Invalid value '{value}'. Valid options are: {valid_values}"
        ) from None


def run_training_pipeline():
    """Execute the complete training pipeline."""
    from ultimate_rvc.typing_extra import (
        AudioNormalizationMode,
        AudioSplitMethod,
        DeviceType,
        EmbedderModel,
        F0Method,
        IndexAlgorithm,
        PrecisionType,
        PretrainedType,
        TrainingSampleRate,
        Vocoder,
    )
    from ultimate_rvc.core.train.prepare import preprocess_dataset
    from ultimate_rvc.core.train.extract import extract_features
    from ultimate_rvc.core.train.train import run_training

    print("=" * 80)
    print("ULTIMATE RVC TRAINING PIPELINE")
    print("=" * 80)
    print(f"\nModel Name: {MODEL_NAME}")
    print(f"Dataset Path: {DATASET_PATH}")
    print(f"Sample Rate: {SAMPLE_RATE}")
    print(f"F0 Method: {F0_METHOD}")
    print(f"Embedder Model: {EMBEDDER_MODEL}")
    print(f"Training Epochs: {NUM_EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Hardware Acceleration: {HARDWARE_ACCELERATION}")
    print(f"Precision: {PRECISION}")
    print("\n" + "=" * 80)

    # Validate dataset path exists
    dataset_path = Path(DATASET_PATH)
    if not dataset_path.exists():
        print(f"\n❌ ERROR: Dataset path does not exist: {DATASET_PATH}")
        print("Please update DATASET_PATH in the configuration section.")
        sys.exit(1)

    if not dataset_path.is_dir():
        print(f"\n❌ ERROR: Dataset path is not a directory: {DATASET_PATH}")
        print("Please ensure DATASET_PATH points to a directory containing audio files.")
        sys.exit(1)

    # Check if directory has audio files
    audio_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
    audio_files = [f for f in dataset_path.iterdir() if f.suffix.lower() in audio_extensions]
    if not audio_files:
        print(f"\n❌ ERROR: No audio files found in: {DATASET_PATH}")
        print(f"Supported formats: {', '.join(audio_extensions)}")
        sys.exit(1)

    print(f"\n✓ Found {len(audio_files)} audio files in dataset")

    # Convert string configurations to enum types
    sample_rate = TrainingSampleRate(SAMPLE_RATE)
    normalization_mode = convert_to_enum(NORMALIZATION_MODE, AudioNormalizationMode)
    split_method = convert_to_enum(SPLIT_METHOD, AudioSplitMethod)
    f0_method = convert_to_enum(F0_METHOD, F0Method)
    embedder_model = convert_to_enum(EMBEDDER_MODEL, EmbedderModel)
    device_type = convert_to_enum(HARDWARE_ACCELERATION, DeviceType)
    vocoder = convert_to_enum(VOCODER, Vocoder)
    index_algorithm = convert_to_enum(INDEX_ALGORITHM, IndexAlgorithm)
    pretrained_type = convert_to_enum(PRETRAINED_TYPE, PretrainedType)
    precision = convert_to_enum(PRECISION, PrecisionType)

    gpu_ids_set = set(GPU_IDS) if GPU_IDS else None

    # Step 1: Preprocess Dataset
    print("\n" + "=" * 80)
    print("STEP 1: PREPROCESSING DATASET")
    print("=" * 80)
    print("\nThis step will:")
    print("  - Copy audio files to training directory")
    print("  - Resample audio to target sample rate")
    print("  - Apply normalization and filtering")
    print("  - Split audio into chunks")
    print("\nStarting preprocessing...\n")

    try:
        preprocess_dataset(
            model_name=MODEL_NAME,
            dataset=dataset_path,
            sample_rate=sample_rate,
            normalization_mode=normalization_mode,
            filter_audio=FILTER_AUDIO,
            clean_audio=CLEAN_AUDIO,
            clean_strength=CLEAN_STRENGTH,
            split_method=split_method,
            chunk_len=CHUNK_LENGTH,
            overlap_len=OVERLAP_LENGTH,
            cpu_cores=PREPROCESS_CPU_CORES,
        )
        print("\n✓ Dataset preprocessing completed successfully!")
    except Exception as e:
        print(f"\n❌ ERROR during preprocessing: {e}")
        print("Please check the error message above and fix any issues.")
        sys.exit(1)

    # Step 2: Extract Features
    print("\n" + "=" * 80)
    print("STEP 2: EXTRACTING FEATURES")
    print("=" * 80)
    print("\nThis step will:")
    print("  - Extract pitch (F0) features using", F0_METHOD)
    print("  - Extract audio embeddings using", EMBEDDER_MODEL)
    print("  - Generate training file lists")
    print("\nStarting feature extraction...\n")

    try:
        extract_features(
            model_name=MODEL_NAME,
            f0_method=f0_method,
            embedder_model=embedder_model,
            custom_embedder_model=CUSTOM_EMBEDDER_MODEL,
            include_mutes=INCLUDE_MUTES,
            cpu_cores=EXTRACT_CPU_CORES,
            hardware_acceleration=device_type,
            gpu_ids=gpu_ids_set,
        )
        print("\n✓ Feature extraction completed successfully!")
    except Exception as e:
        print(f"\n❌ ERROR during feature extraction: {e}")
        print("Please check the error message above and fix any issues.")
        sys.exit(1)

    # Step 3: Train Model
    print("\n" + "=" * 80)
    print("STEP 3: TRAINING VOICE MODEL")
    print("=" * 80)
    print("\nThis step will:")
    print(f"  - Train for {NUM_EPOCHS} epochs")
    print(f"  - Use batch size of {BATCH_SIZE}")
    print(f"  - Save checkpoints every {SAVE_INTERVAL} epochs")
    print(f"  - Use {VOCODER} vocoder")
    print("\nStarting training...\n")

    try:
        result = run_training(
            model_name=MODEL_NAME,
            num_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            detect_overtraining=DETECT_OVERTRAINING,
            overtraining_threshold=OVERTRAINING_THRESHOLD,
            vocoder=vocoder,
            index_algorithm=index_algorithm,
            pretrained_type=pretrained_type,
            custom_pretrained=CUSTOM_PRETRAINED,
            save_interval=SAVE_INTERVAL,
            save_all_checkpoints=SAVE_ALL_CHECKPOINTS,
            save_all_weights=SAVE_ALL_WEIGHTS,
            clear_saved_data=CLEAR_SAVED_DATA,
            upload_model=UPLOAD_MODEL,
            upload_name=UPLOAD_NAME,
            hardware_acceleration=device_type,
            gpu_ids=gpu_ids_set,
            precision=precision,
            preload_dataset=PRELOAD_DATASET,
            reduce_memory_usage=REDUCE_MEMORY_USAGE,
        )

        if result:
            print("\n✓ Training completed successfully!")
            print(f"\nTrained model files:")
            for file_path in result:
                print(f"  - {file_path}")
        else:
            print("\n⚠ Training completed but no model files were generated.")
            print("This may indicate an issue during training.")

    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user.")
        print("You can resume training later or check partial results in the model directory.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERROR during training: {e}")
        print("Please check the error message above and fix any issues.")
        sys.exit(1)

    # Final Summary
    print("\n" + "=" * 80)
    print("TRAINING PIPELINE COMPLETED")
    print("=" * 80)
    print(f"\nModel Name: {MODEL_NAME}")
    print("All steps completed successfully!")
    print("\nYour trained voice model is ready to use.")
    if UPLOAD_MODEL:
        print(f"The model has been uploaded as: {UPLOAD_NAME}")
    else:
        print(f"Model files are located in the training models directory.")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    run_training_pipeline()

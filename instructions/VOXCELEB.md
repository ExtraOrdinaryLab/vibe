# VoxCeleb Dataset Setup

This repository contains instructions and scripts to set up the VoxCeleb1 and VoxCeleb2 datasets for audio processing tasks.

## Dataset Preparation

1. Download Datasets

 - VoxCeleb1 : Download `vox1_dev_wav_partaa` to `vox1_dev_wav_partad` and `vox1_test_wav.zip`, and place them under `/path/to/voxceleb1`.
 - VoxCeleb2 : Download `vox2_dev_aac_partaa` to `vox2_dev_aac_partah` and `vox2_test_aac.zip`, and place them under `/path/to/voxceleb2`.

2. Extract VoxCeleb1

```bash
cd /path/to/voxceleb1
cat vox1_dev_wav* > vox1_dev_wav.zip
unzip vox1_dev_wav.zip -d dev
unzip vox1_test_wav.zip -d test
```

3. Extract VoxCeleb2

```bash
cd /path/to/voxceleb2
cat vox2_dev_aac* > vox2_dev_aac.zip
unzip vox2_dev_aac.zip -d dev
unzip vox2_test_aac.zip -d test
```

4. Convert VoxCeleb2 Audio to WAV Format

VoxCeleb2 files are in AAC format. Use the provided script to convert them to WAV:

```python
python convert_voxceleb2.py --voxceleb2_root /path/to/voxceleb2 --max_workers 16
```

Expected Folder Structure

```
/path/to/voxceleb1
├── dev
│   └── wav
└── test
    └── wav

/path/to/voxceleb2
├── dev
│   └── wav
└── test
    └── wav
```

Ensure the final structure matches the above for compatibility with most audio processing pipelines.
# Vision Train 📰🖼️

An end-to-end machine learning pipeline that scrapes images from news websites, organizes them using AI, and trains a CNN for automatic classification.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16+-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Features

- 🕷️ **Automated Web Scraping** - Collect images from multiple news sources
- 🤖 **CLIP AI Organization** - 85-90% accuracy using OpenAI's vision-language model
- 🧠 **Custom CNN** - 27M parameter model with 4 convolutional blocks
- 📊 **10 Categories** - Politics, Sports, Technology, Business, Entertainment, Health, Science, World, Environment, Education
- 🎨 **Data Augmentation** - Automatic image enhancement for better training
- 📈 **Visualization** - Training plots and prediction confidence scores

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/wombus23/news-image-classifier.git
cd news-image-classifier

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Install PyTorch and CLIP
pip install torch torchvision
pip install ftfy regex
pip install git+https://github.com/openai/CLIP.git
```

### Usage

```bash
# 1. Scrape images
python scripts/scrape_images.py --max-images 200

# 2. Organize with CLIP AI
python scripts/organize_with_clip.py --force

# 3. Train model
python src/models/train.py

# 4. Make predictions
python scripts/predict.py --image test.jpg --visualize
```

## 📁 Project Structure

```
news-image-classifier/
├── config/config.yaml          # Configuration
├── src/
│   ├── scraper/               # Web scraping
│   ├── preprocessing/         # Image processing
│   └── models/                # CNN architecture
├── scripts/
│   ├── scrape_images.py      # Scraper
│   ├── organize_with_clip.py # CLIP organizer
│   └── predict.py            # Predictions
└── data/
    ├── raw/                   # Scraped images
    ├── processed/             # Organized images
    └── models/                # Trained models
```

## 🏗️ Model Architecture

```
Input (224×224×3)
    ↓
4× Conv Blocks (32→64→128→256 filters)
    ↓
Flatten → Dense(512) → Dense(256)
    ↓
Output (10 classes)
```

**Stats:** 27M parameters | ~103 MB | 75-85% validation accuracy

## ⚙️ Configuration

Edit `config/config.yaml`:

```yaml
scraping:
  urls:
    - "https://www.bbc.com/news"
    - "https://www.cnn.com"
  max_images_per_url: 100

model:
  num_classes: 10
  architecture: "custom"

training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001
```

## 📊 Performance

| Method | Accuracy | Speed |
|--------|----------|-------|
| CLIP Organization | 85-90% | 2s/image |
| Model Training | 75-85% | - |
| Inference | - | 50-100ms |

**Recommended:** 100+ images per category for best results

## 🛠️ Advanced Usage

### Custom Categories

```yaml
# config/config.yaml
classes:
  - "your_category_1"
  - "your_category_2"
```

### Transfer Learning

```yaml
model:
  architecture: "vgg16"  # or "resnet50"
```

### Monitor Training

```bash
tensorboard --logdir=data/models/logs
```

## 📝 Example Output

```bash
$ python scripts/predict.py --image sports.jpg

Top Class: sports
Confidence: 87.45%

Top 3 Predictions:
1. sports: 87.45%
2. entertainment: 8.23%
3. politics: 2.31%
```

## 🤝 Contributing

Contributions welcome! Fork the repo, create a feature branch, and submit a pull request.

## ⚠️ Legal Notice

**Web Scraping:** Review `robots.txt` and Terms of Service. Respect rate limits. Educational use only.

## 📄 License

MIT License - See [LICENSE](LICENSE) file

## 🙏 Acknowledgments

- OpenAI CLIP for image organization
- TensorFlow for deep learning framework
- Open source community

## 📮 Contact

**Repository:** [github.com/wombus23/Vision-Train](https://github.com/wombus23/Vision-Train)

---

⭐ Star this repo if you find it helpful!

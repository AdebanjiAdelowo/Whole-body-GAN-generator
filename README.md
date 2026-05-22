# Whole-Body GAN Generator

Full-body human image generation and editing using **StyleGAN-Human** and **InsetGAN**, with a **FastAPI** inference server and a **Swift iOS** client app. Built during a Machine Learning internship at [Fellowship.AI](https://fellowship.ai) (May–Aug 2022).

---

## How It Works

The pipeline has three stages:

```
User photo (iOS app)
       │
       ▼
  ReStyle encoder          ← encodes the face photo into StyleGAN-FFHQ latent space
       │
       ▼
  InsetGAN joint optimiser ← composites the face GAN (FFHQ) into the body GAN (StyleGAN-Human)
       │                      running iterative optimisation to make them seamless
       ▼
  Generated full-body image
       │
       ▼
  FastAPI server (Colab)   ← serves results back to the iOS app via ngrok tunnel
       │
       ▼
  Swift iOS app            ← displays result, uses Firebase Storage for image transfer
```

---

## Repository Layout

```
Whole-body-GAN-generator/
├── server/
│   ├── Adebanji_User_Whole_Body_Generation.ipynb          # Main generation + FastAPI server
│   ├── Copy_of_Optimised_User_Whole_Body_Generation.ipynb # Optimised variant
│   ├── User_Whole_Body_Generation.ipynb                   # Original pipeline notebook
│   ├── User_Whole_Body_Generation_Using_PTI.ipynb         # PTI-based personalisation
│   └── Deploying_Style_Human_Inference_in_Google_Colab_environment.ipynb  # Deployment
├── models/
│   └── style_human.ipynb    # StyleGAN-Human setup, generation, editing, style-mixing
├── user_interface/
│   ├── UI/                  # Swift iOS UI source
│   └── Whole-Body GAN Demo/ # Xcode workspace (Firebase + CocoaPods)
├── User Whole Body Generation/
│   ├── StyleGAN-Human/      # Pretrained model directory
│   ├── restyle/             # ReStyle encoder directory
│   └── *.jpg                # Sample input images
└── requirements/            # Python dependencies
```

---

## ML Stack

| Component | Role |
|---|---|
| **StyleGAN-Human** (StyleGAN2, 1024×1024) | Full-body image generator — produces photorealistic human images |
| **StyleGAN2-FFHQ** (1024×1024) | Face generator — provides the source face latent code |
| **ReStyle** (pSp encoder) | Encodes a real user photo into the FFHQ GAN latent space |
| **InsetGAN** | Joint optimiser that composites the face GAN output into the body GAN seamlessly |
| **PTI** (Pivotal Tuning Inversion) | Fine-tunes the generator on a specific user photo for higher fidelity |
| **FastAPI + ColabCode + ngrok** | Serves inference from a Colab GPU runtime over a public URL |
| **Firebase Storage** | Transfers images between the iOS app and the Colab server |

---

## Capabilities

Beyond user-photo generation, the StyleGAN-Human model supports:

- **Attribute editing** — change clothing length (upper / bottom) via latent space directions
- **Style mixing** — blend body styles from different generated identities
- **Unconditional generation** — generate random full-body human images from seeds

---

## Notebooks

### `models/style_human.ipynb`
Standalone StyleGAN-Human notebook. Steps:
1. Clone StyleGAN-Human and install dependencies (lpips, ninja)
2. Download pretrained `stylegan2_1024.pkl` from Google Drive
3. Generate images (`generate.py --seeds --trunc`)
4. Edit attributes (`edit.py --attr_name upper_length / bottom_length`)
5. Style mixing (`style_mixing.py`)
6. InsetGAN joint optimisation (`insetgan.py --face_seed --body_seed --joint_steps`)

### `server/Adebanji_User_Whole_Body_Generation.ipynb`
End-to-end pipeline with a FastAPI server:
1. Clone StyleGAN-Human and ReStyle, install dependencies
2. Download StyleGAN-Human and FFHQ pretrained models
3. Load ReStyle pSp encoder and encode user photo into face latent code
4. Run InsetGAN joint optimisation to composite face into full-body image
5. Serve results through a FastAPI endpoint via ngrok tunnel

### `server/Deploying_Style_Human_Inference_in_Google_Colab_environment.ipynb`
Deployment-focused notebook. Exposes these API endpoints:
- `GET /` — health check
- `POST /generate` — generate images from seeds
- `POST /edit` — attribute editing
- `POST /style_mix` — style mixing
- `POST /insetgan` — InsetGAN joint optimisation

---

## iOS App Setup (CocoaPods + Firebase)

**Requirements:** Xcode, CocoaPods

```bash
brew install cocoapods
cd user_interface/Whole-Body\ GAN\ Demo
pod init
```

Edit `Podfile`:

```ruby
platform :ios, '15.4'

target 'Whole-Body GAN Demo' do
  use_frameworks!
  pod 'Firebase/Core'
  pod 'Firebase/Storage'
end
```

Then:

```bash
pod install
open Whole-Body\ GAN\ Demo.xcworkspace
```

The app uses **Firebase Storage** to upload the user photo and retrieve the generated image from the Colab FastAPI server.

---

## Python Requirements

```bash
pip install torch torchvision lpips fastapi colabcode
```

Full list in `requirements/`.

---

## References

- [StyleGAN-Human](https://github.com/stylegan-human/StyleGAN-Human) — Fu et al. (2022)
- [ReStyle](https://github.com/yuval-alaluf/restyle-encoder) — Alaluf et al. (2021)
- [InsetGAN](https://github.com/stylegan-human/StyleGAN-Human) — joint face-body optimisation
- [PTI](https://github.com/danielroich/PTI) — Roich et al. (2022)
- [Fellowship.AI](https://fellowship.ai)

---

## License

MIT

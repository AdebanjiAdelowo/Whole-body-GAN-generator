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
       │                      by iterative optimisation that matches appearance across the join
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
│   ├── StyleGAN-Human/      # Vendored StyleGAN-Human source (code only; pretrained weights are downloaded at runtime)
│   ├── restyle/             # Vendored ReStyle encoder source (includes its own LICENSE and per-dependency licenses/)
│   └── *.jpg                # Sample input images
├── data_cleaning/           # Background blurring/replacement notebooks used to prep sample images
├── docs/                    # Longer-form technical documentation (InsetGAN write-up, reading guide)
└── requirements/            # Firebase config for the iOS app (GoogleService-Info.plist)
```

`StyleGAN-Human/` and `restyle/` are third-party repositories included in full, not thin wrappers; the code authored for this project is the FastAPI server notebooks, the iOS client, and the glue that connects them. Only the vendored `restyle/` copy ships its own `LICENSE` file (plus a `licenses/` directory covering its own dependencies); the vendored `StyleGAN-Human/` copy in this repo does not include a `LICENSE` file, so consult the upstream [StyleGAN-Human](https://github.com/stylegan-human/StyleGAN-Human) repository for its licensing terms.

---

## ML Stack

| Component | Role |
|---|---|
| **StyleGAN-Human** (StyleGAN2, 1024×1024) | Full-body image generator, produces photorealistic human images |
| **StyleGAN2-FFHQ** (1024×1024) | Face generator, provides the source face latent code |
| **ReStyle** (pSp encoder) | Encodes a real user photo into the FFHQ GAN latent space |
| **InsetGAN** | Joint optimiser that composites the face GAN output into the body GAN, matching appearance across the join |
| **PTI** (Pivotal Tuning Inversion) | Fine-tunes the generator on a specific user photo for higher fidelity |
| **FastAPI + ColabCode + ngrok** | Serves inference from a Colab GPU runtime over a public URL |
| **Firebase Storage** | Transfers images between the iOS app and the Colab server |

---

## Capabilities

Beyond user-photo generation, the StyleGAN-Human model supports:

- **Attribute editing**: change clothing length (upper / bottom) via latent space directions
- **Style mixing**: blend body styles from different generated identities
- **Unconditional generation**: generate random full-body human images from seeds

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
Deployment-focused notebook. Exposes these API endpoints (all `GET`, with query-string parameters such as `seed`, `trunc`, `row_seeds`, `face_seed`):
- `GET /`: health check
- `GET /generate_single_image`, `GET /generate_image`: generate images from seeds
- `GET /upper_length_Edit`, `GET /bottom_length_Edit`: attribute editing
- `GET /Style_Mixing_EndPoint`: style mixing
- `GET /Joint_Optimisation_Endpoint`: InsetGAN joint optimisation

### `server/User_Whole_Body_Generation.ipynb`
Original joint-optimisation server. Runs its own `FastAPI()` app with CORS enabled:
- `GET /joint_Optimisation`: run joint optimisation from query params (`body_seed`, `joint_steps`, `trunc`), returns an MP4
- `POST /joint_Optimisation_upload`: upload a face photo (multipart `file`) plus the same query params, returns an MP4

### `server/Copy_of_Optimised_User_Whole_Body_Generation.ipynb`
Optimised variant. Runs its own separate `FastAPI()` app with CORS enabled:
- `GET /join-optimisation-url-endpoint`: run joint optimisation from an `image_url` plus `body_seed`, `joint_steps`, `trunc`, returns an MP4
- `POST /join-optimisation-upload-endpoint`: same, via multipart file upload
- `GET /generate_single_image`: generate a single image from `seed`/`trunc`

The same notebook also defines `GET /html` (a Jinja2 status page) and `GET /video` (streams the last generated MP4) earlier on, but registers them on an `app` instance that is discarded when a later cell runs `app = FastAPI()` again to attach CORS middleware; run top to bottom, only the three routes above are actually reachable.

Each of the four server notebooks above instantiates its own independent FastAPI app; they are four separate server implementations with four different endpoint sets, not a single shared API split across two notebooks.

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

The notebooks were run in Google Colab and install dependencies inline. Core packages:

```bash
pip install torch torchvision lpips fastapi colabcode
```

No pinned `requirements.txt` is included; the `requirements/` directory contains
only the Firebase configuration file (`GoogleService-Info.plist`) used by the iOS app.

---

## References

- [StyleGAN-Human](https://github.com/stylegan-human/StyleGAN-Human) (Fu et al., ECCV 2022)
- [ReStyle](https://github.com/yuval-alaluf/restyle-encoder) (Alaluf et al., ICCV 2021)
- [InsetGAN](https://github.com/afruehstueck/insetGAN) (Frühstück et al., CVPR 2022): joint face-body optimisation; the `insetgan.py` script used here is vendored inside the `StyleGAN-Human` repository
- [PTI](https://github.com/danielroich/PTI) (Roich et al., ACM TOG 2022)
- [Fellowship.AI](https://fellowship.ai)


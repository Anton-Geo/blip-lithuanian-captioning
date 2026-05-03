from src.image_utils import load_image_from_url
from src.inference import generate_caption
from src.model_loader import load_blip_model


def main():
    image_urls = [
        "https://sobakovod.club/uploads/posts/2021-12/1639914627_1-sobakovod-club-p-sobaki-angliiskii-buldog-s-chelovekom-1.jpg",
        "https://live.staticflickr.com/3783/9166044338_21f49698c8.jpg",
        "https://images.delfi.lt/media-api-image-cropper/v1/8710c150-7b91-11ed-8d0e-3313c2a18a43.jpg?w=1200&h=800&fx=0.5&fy=0.25",
    ]

    processor, model, device = load_blip_model()

    print(f"Device: {device}")

    for url in image_urls:
        image = load_image_from_url(url)

        caption = generate_caption(
            image=image,
            processor=processor,
            model=model,
            device=device,
        )

        print("\nURL:", url)
        print("Caption:", caption)


if __name__ == "__main__":
    main()
